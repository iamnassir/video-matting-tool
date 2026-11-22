"""SAM 2 + RVM hybrid: the identity of the subject from SAM 2, the soft matte from RVM.

Each model covers the other's weakness.  Robust Video Matting predicts a true alpha with soft hair
and motion-blur edges, but mattes *every* person in the frame and cannot be told whom to keep.
SAM 2 follows exactly the selected subject through occlusions, but only produces a coarse mask.
The hybrid runs both on every frame and keeps RVM's alpha only where SAM 2 says the subject is::

    gate  = SAM 2 mask, grown by a margin and softened      (hair just outside the mask survives)
    alpha = RVM alpha * gate
    alpha = max(alpha, shrunken SAM 2 mask)                  (optional: parts RVM dropped, e.g. a
                                                              hand-held object, are restored)

Unlike MatAnyone this combination carries no non-commercial restriction (SAM 2 is Apache-2.0; RVM
is GPL-3.0, see its licence notes).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import torch

from ...core.errors import ModelLoadError
from ...utils import torch_ops
from ..base import (
    LicenseInfo,
    MatteOutput,
    MattingModel,
    ModelCapabilities,
    ModelInfo,
    ModelVariant,
    SequenceInfo,
)
from ..registry import register_model

log = logging.getLogger(__name__)


def _variant(variant_id: str, label: str, sam2: str, rvm: str, text: str, vram: int) -> ModelVariant:
    return ModelVariant(id=variant_id, label=label, weights=(), description=text, vram_hint_mb=vram,
                        options={"sam2_variant": sam2, "rvm_variant": rvm})


@register_model
class Sam2RvmAdapter(MattingModel):
    info = ModelInfo(
        id="sam2_rvm",
        name="SAM 2 + RVM (hybrid)",
        description=(
            "Follows the one person you select (SAM 2) and gives them a true soft alpha matte (RVM): other people "
            "in the shot are excluded, hair and motion blur stay soft. The best commercial-friendly choice for people."
        ),
        license=LicenseInfo(
            name="GPL-3.0 (RVM) + Apache-2.0 (SAM 2)",
            spdx="GPL-3.0-only AND Apache-2.0",
            url="https://github.com/PeterL1n/RobustVideoMatting/blob/master/LICENSE",
            commercial_use="conditional",
            notes=(
                "The stricter component governs: RVM is GPL-3.0 and its weights carry no separate licence "
                "statement (processing your own footage is permitted; distributing the model file or a product "
                "that bundles it falls under the GPL). SAM 2 is Apache-2.0."
            ),
        ),
        capabilities=ModelCapabilities(
            alpha_type="alpha",
            native_prompts=frozenset({"points", "box", "mask"}),
            needs_initial_mask=False,
            subject_domain="human",
            temporal="memory",
            max_batch_size=1,
            supports_fp16=True,
            devices=frozenset({"cuda", "cpu"}),
            supports_reverse=True,
            tracks_single_subject=True,
        ),
        variants=(
            _variant("fast", "SAM 2.1 tiny + RVM MobileNetV3", "tiny", "mobilenetv3", "Fastest combination.", 1600),
            _variant("balanced", "SAM 2.1 small + RVM MobileNetV3 (recommended)", "small", "mobilenetv3",
                     "Stable tracking with the fast matting network.", 1800),
            _variant("quality", "SAM 2.1 small + RVM ResNet50", "small", "resnet50",
                     "Slightly more accurate matte edges.", 2000),
        ),
        default_variant="balanced",
        quality_variants={"fast": "fast", "balanced": "balanced", "high": "quality", "max": "quality"},
        limitations=(
            "The soft matte comes from RVM, which is trained on people only: for other subjects use 'sam2'.",
            "Where RVM misses part of the selected subject (a hand-held object, an unusual pose) the result falls "
            "back to the SAM 2 mask there, whose edges are harder (option fill_from_mask=false switches this off).",
            "Two networks run on every frame: slower and more memory-hungry than either model alone.",
            "Needs a prompt on the reference frame like SAM 2 (box, clicks, mask or automatic detection).",
        ),
        requires=("sam2", "hydra"),
        install_hint="Install the SAM 2 package with:  python scripts/install_models.py sam2",
        homepage="https://github.com/facebookresearch/sam2",
        citation="Ravi et al., SAM 2 (ICLR 2025); Lin et al., Robust High-Resolution Video Matting (WACV 2022).",
        depends_on=("sam2", "rvm"),
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sam2: Optional[MattingModel] = None
        self._rvm: Optional[MattingModel] = None
        #: pixels at 1080p by which the SAM 2 mask is grown before it gates the RVM alpha
        self._margin = float(self.options.get("gate_margin", 18.0))
        #: pixels at 1080p by which the SAM 2 mask is shrunk before it may fill holes of the RVM alpha
        self._shrink = float(self.options.get("fill_shrink", 14.0))
        self._fill = bool(self.options.get("fill_from_mask", True))

    # ------------------------------------------------------------------------- life cycle
    def load(self) -> None:
        if self.factory is None:
            raise ModelLoadError(
                "The hybrid adapter must be created through the model registry (it builds its two component models).",
                hint="Use app.models.registry.create_model('sam2_rvm', ...).",
            )
        passthrough = {k: v for k, v in self.options.items() if k in ("downsample_ratio", "internal_side", "jit_optimize")}
        self._sam2 = self.factory("sam2", self.variant.options["sam2_variant"], {})
        self._rvm = self.factory("rvm", self.variant.options["rvm_variant"], passthrough)
        self._sam2.load()
        try:
            self._rvm.load()
        except BaseException:
            self._sam2.unload()
            self._sam2 = None
            raise
        self.loaded = True

    def unload(self) -> None:
        for model in (self._sam2, self._rvm):
            if model is not None:
                try:
                    model.unload()
                except Exception as exc:  # pragma: no cover - best effort
                    log.debug("unloading a component failed: %s", exc)
        self._sam2 = self._rvm = None
        super().unload()

    def start_sequence(self, seq: SequenceInfo) -> None:
        if self._sam2 is None or self._rvm is None:
            raise ModelLoadError("Hybrid adapter used before load().")
        self._sam2.start_sequence(seq)             # validates the prompt
        self._rvm.start_sequence(seq)              # ignores the subject: it mattes every person

    def end_sequence(self) -> None:
        for model in (self._sam2, self._rvm):
            if model is not None:
                model.end_sequence()

    # ------------------------------------------------------------------------- inference
    def process(self, frames: torch.Tensor, frame_indices: Sequence[int]) -> MatteOutput:
        if self._sam2 is None or self._rvm is None:
            raise ModelLoadError("Hybrid adapter used before load().")
        # Keep the two temporal states in step even when one network fails (for example out of
        # memory): RVM runs first and its state is put back if SAM 2 then fails; SAM 2 leaves its
        # own state untouched when it fails.  A retry of the same frame is therefore exact.
        snapshot = self._rvm.snapshot_state()
        matte = self._rvm.process(frames, frame_indices)
        try:
            track = self._sam2.process(frames, frame_indices)
        except BaseException:
            self._rvm.restore_state(snapshot)
            raise
        height, width = int(frames.shape[-2]), int(frames.shape[-1])
        scale = min(height, width) / 1080.0
        mask = (track.alpha > 0.5).to(torch.float32)
        margin = max(2.0, self._margin * scale)
        gate = torch_ops.gaussian_blur(torch_ops.max_filter(mask, margin), max(1.0, margin / 3.0)).clamp_(0.0, 1.0)
        # Exactly one well inside the mask (so RVM's alpha passes unchanged) and exactly zero far
        # outside it (so nothing of another person can leak through a blur tail).
        gate = torch.where(mask > 0.5, torch.ones_like(gate), gate)
        gate = gate * (torch_ops.max_filter(mask, 2.0 * margin) > 0.0)
        alpha = matte.alpha * gate
        if self._fill:
            shrink = max(2.0, self._shrink * scale)
            interior = torch_ops.gaussian_blur(torch_ops.min_filter(mask, shrink), max(1.0, shrink / 4.0))
            alpha = torch.maximum(alpha, interior.clamp_(0.0, 1.0))
        return MatteOutput(alpha=alpha.clamp_(0.0, 1.0), presence=track.presence, is_mask=False)

    def reduce_memory(self) -> bool:
        return bool(self._rvm is not None and self._rvm.reduce_memory())

    def runtime_summary(self) -> Dict[str, Any]:
        summary = super().runtime_summary()
        summary["options"].pop("quality", None)
        summary["gate_margin_px_at_1080p"] = self._margin
        summary["fill_from_mask"] = self._fill
        if self._sam2 is not None:
            summary["sam2"] = self._sam2.runtime_summary()
        if self._rvm is not None:
            summary["rvm"] = self._rvm.runtime_summary()
        return summary
