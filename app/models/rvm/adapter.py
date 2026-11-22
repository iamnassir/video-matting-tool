"""Robust Video Matting (RVM) adapter.

RVM (Lin et al., WACV 2022) is a recurrent human-matting network: it predicts a true soft alpha
matte per frame and carries four ConvGRU states from frame to frame for temporal consistency.

The model is loaded from the OFFICIAL TorchScript files published on the project's GitHub release
(no model source code is vendored in this repository).  Facts this adapter relies on were verified
against those files:

* ``forward(src, r1=None, r2=None, r3=None, r4=None, downsample_ratio: float = 1.0,
  segmentation_pass: bool = False) -> [fgr, pha, r1, r2, r3, r4]``
* ``src`` is ``[B,3,H,W]`` (or ``[B,T,3,H,W]``) RGB in 0..1 in the model's dtype; ImageNet
  normalisation happens inside the model; any frame size is accepted.
* The exported file crashes on a 5-D input when the states are ``None`` (it builds the initial
  state with ``zeros_like`` of the 5-D input).  The first frame of a sequence is therefore run as
  a 4-D call and later frames as one 5-D chunk with tensor states.
* The fp16 release files are exactly the fp32 weights cast to half, so only the fp32 file is
  downloaded and ``.half()`` is applied on CUDA.
* Device / dtype must be set before ``torch.jit.freeze``.
* While the adapter is loaded, TorchScript graph optimisation is switched off
  (``torch.jit.optimized_execution(False)``): the profiling executor is extremely slow for this graph
  on CUDA (see ``__init__`` and ``load``).
* The states are only valid for one (H, W, downsample_ratio); they are reset when any of them
  changes.

Limitations (see ``info.limitations``): people only, no subject selection (every person in frame is
matted; the pipeline's component tracker restricts the result to the primary / selected person).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..base import (
    LicenseInfo,
    MatteOutput,
    MattingModel,
    ModelCapabilities,
    ModelInfo,
    ModelVariant,
    SequenceInfo,
    WeightFile,
)
from ..registry import register_model
from ...core.errors import ModelLoadError

log = logging.getLogger(__name__)

_RELEASE = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0"

#: Long side (pixels) of the network's internal working resolution per quality preset.  The
#: official tool uses 512 for every input ("auto"); the authors recommend going up to about 768
#: for full-body shots where the person is small in the frame.
_INTERNAL_SIDE: Dict[str, int] = {"fast": 384, "balanced": 512, "high": 512, "max": 640}
_MIN_INTERNAL_SIDE = 256


@register_model
class RVMAdapter(MattingModel):
    info = ModelInfo(
        id="rvm",
        name="Robust Video Matting",
        description=(
            "Fast recurrent human matting with true soft alpha (hair, motion blur). Fully automatic: "
            "no subject prompt needed. Best speed/quality trade-off for people."
        ),
        license=LicenseInfo(
            name="GPL-3.0",
            spdx="GPL-3.0-only",
            url="https://github.com/PeterL1n/RobustVideoMatting/blob/master/LICENSE",
            commercial_use="conditional",
            notes=(
                "The code is GPL-3.0 and the pretrained weights carry no separate licence statement. "
                "Processing your own footage with the model is permitted; distributing the model file, "
                "or a product that bundles it, falls under the GPL copyleft terms. The official "
                "TorchScript file embeds the model source. Seek legal advice before commercial distribution."
            ),
        ),
        capabilities=ModelCapabilities(
            alpha_type="alpha",
            native_prompts=frozenset(),
            needs_initial_mask=False,
            subject_domain="human",
            temporal="recurrent",
            max_batch_size=None,
            supports_fp16=True,
            devices=frozenset({"cuda", "cpu", "mps"}),
            supports_reverse=True,
            tracks_single_subject=False,
        ),
        variants=(
            ModelVariant(
                id="mobilenetv3",
                label="MobileNetV3 (fast, recommended)",
                description="3.7 M parameters. The authors' recommended variant.",
                vram_hint_mb=1200,
                weights=(
                    WeightFile(
                        filename="rvm_mobilenetv3_fp32.torchscript",
                        url=f"{_RELEASE}/rvm_mobilenetv3_fp32.torchscript",
                        size_bytes=15_501_891,
                        sha256="f01e0c9338b9a6a31b881ea6d4360d70c1e549701b3792e14c9ed88d6196c5a1",
                        description="Official TorchScript export, fp32",
                        source="GitHub release v1.0.0 of PeterL1n/RobustVideoMatting (official)",
                    ),
                ),
            ),
            ModelVariant(
                id="resnet50",
                label="ResNet50 (slower, slightly more accurate)",
                description="26.9 M parameters. Small accuracy gain at roughly 70% of the speed.",
                vram_hint_mb=2200,
                weights=(
                    WeightFile(
                        filename="rvm_resnet50_fp32.torchscript",
                        url=f"{_RELEASE}/rvm_resnet50_fp32.torchscript",
                        size_bytes=108_063_684,
                        sha256="072adec3c75a1af773ec35d8b595612f8e3472b7a4b8a210684432b493376852",
                        description="Official TorchScript export, fp32",
                        source="GitHub release v1.0.0 of PeterL1n/RobustVideoMatting (official)",
                    ),
                ),
            ),
        ),
        default_variant="mobilenetv3",
        quality_variants={"fast": "mobilenetv3", "balanced": "mobilenetv3", "high": "resnet50", "max": "resnet50"},
        limitations=(
            "Trained on people only: other subjects (animals, products, vehicles) are not matted reliably.",
            "No subject selection inside the model: every person in the frame is matted. The pipeline restricts "
            "the matte to the primary / selected person by connected-component tracking, which cannot separate "
            "people who touch or overlap. Use the 'sam2_rvm' or 'matanyone' model for those shots.",
            "Quality is lower during roughly the first 15 frames while the recurrent state warms up.",
            "Works best with a clearly visible person; small, distant or heavily occluded people and very fast "
            "motion can flicker or lose limbs.",
            "The internal working resolution (downsample ratio) matters: too low erodes thin limbs, too high can "
            "confuse the network. Override with model option 'downsample_ratio' (e.g. 0.4 for full-body 1080p).",
        ),
        requires=(),
        install_hint="No extra package needed; only the weights file (python main.py models download rvm).",
        homepage="https://github.com/PeterL1n/RobustVideoMatting",
        citation="Lin, Yang, Saleemi, Sengupta: Robust High-Resolution Video Matting with Temporal Guidance. WACV 2022.",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model: Optional[torch.jit.ScriptModule] = None
        self._dtype = torch.float32
        self._rec: List[Optional[torch.Tensor]] = [None, None, None, None]
        self._key: Optional[Tuple[int, int, float]] = None   # (H, W, ratio) the states belong to
        quality = str(self.options.get("quality", "balanced"))
        self._internal_side = int(self.options.get("internal_side") or _INTERNAL_SIDE.get(quality, 512))
        ratio = self.options.get("downsample_ratio")
        self._fixed_ratio: Optional[float] = float(ratio) if ratio not in (None, "", "auto") else None
        if self._fixed_ratio is not None and not (0.0 < self._fixed_ratio <= 1.0):
            from ...core.errors import ConfigError

            raise ConfigError("model option 'downsample_ratio' must be in (0, 1].")
        # The TorchScript profiling executor re-specialises (and on CUDA re-fuses) the graph for
        # every new input signature, which was measured at 2.6 fps for 1080p on an RTX 3060 versus
        # 100-127 fps with graph optimisation switched off.  Off is therefore the default.
        self._jit_optimize = bool(self.options.get("jit_optimize", False))
        self._jit_guard: Optional[Any] = None
        self._warm: Optional[Tuple[int, int, float]] = None

    # ------------------------------------------------------------------------- life cycle
    def load(self) -> None:
        path = self.weight_path(self.variant.weights[0])
        self._dtype = torch.float16 if (self.precision == "fp16" and self.device.type == "cuda") else torch.float32
        try:
            model = torch.jit.load(str(path), map_location="cpu").eval()
            # Device and dtype must be final before freezing: freezing inlines the weights as
            # constants, after which .to() / .half() silently do nothing.
            model = model.to(device=self.device, dtype=self._dtype)
            self._model = torch.jit.freeze(model)
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load the RVM weights from {path}: {exc}",
                hint="Verify the file with 'python main.py models verify rvm' or download it again.",
            ) from exc
        # cuDNN auto-tuning (cudnn.benchmark) is deliberately left alone: measured on an RTX 3060 it
        # does not speed this network up (127 vs 130 fps at 1080p) but costs 1-2 s for every new
        # batch shape, which dominates short clips.
        if not self._jit_optimize and self._jit_guard is None:
            # Process-wide switch, held for as long as this adapter is loaded.  It must not be
            # toggled around every call: each toggle discards the cached execution plan and the
            # plan is rebuilt on the next call (measured: 6 fps instead of 100+).
            self._jit_guard = torch.jit.optimized_execution(False)
            self._jit_guard.__enter__()
        self._reset_state()
        self.loaded = True
        log.info("RVM %s loaded on %s (%s)", self.variant.id, self.device, "fp16" if self._dtype == torch.float16 else "fp32")

    def unload(self) -> None:
        self._model = None
        self._reset_state()
        if self._jit_guard is not None:
            try:
                self._jit_guard.__exit__(None, None, None)
            finally:
                self._jit_guard = None
        super().unload()

    def _reset_state(self) -> None:
        self._rec = [None, None, None, None]
        self._key = None

    def start_sequence(self, seq: SequenceInfo) -> None:
        self._reset_state()
        if self._model is None:
            raise ModelLoadError("RVM adapter used before load().")
        self._warm_up(seq.height, seq.width)

    def end_sequence(self) -> None:
        self._reset_state()

    # ------------------------------------------------------------------------- inference
    def _ratio(self, height: int, width: int) -> float:
        if self._fixed_ratio is not None:
            return self._fixed_ratio
        return float(min(self._internal_side / float(max(height, width)), 1.0))

    def _call(self, src: torch.Tensor, rec: Sequence[Optional[torch.Tensor]], ratio: float):
        assert self._model is not None
        return self._model(src, rec[0], rec[1], rec[2], rec[3], ratio)

    def _warm_up(self, height: int, width: int) -> None:
        """Run one blank frame so CUDA / cuDNN initialisation does not land on the first real batch."""

        ratio = self._ratio(height, width)
        key = (height, width, ratio)
        if self._warm == key:
            return
        try:
            with torch.inference_mode():
                dummy = torch.zeros((1, 3, height, width), device=self.device, dtype=self._dtype)
                out = self._call(dummy, [None, None, None, None], ratio)
                del dummy, out
            self._warm = key
        except Exception as exc:  # warm-up is an optimisation; real errors surface in process()
            log.debug("RVM warm-up skipped: %s", exc)

    def process(self, frames: torch.Tensor, frame_indices: Sequence[int]) -> MatteOutput:
        if self._model is None:
            raise ModelLoadError("RVM adapter used before load().")
        batch, _, height, width = frames.shape
        ratio = self._ratio(height, width)
        key = (height, width, ratio)
        rec: List[Optional[torch.Tensor]] = list(self._rec) if key == self._key else [None, None, None, None]
        # Plain NCHW, contiguous: a permuted (channels-last strided) input makes cuDNN run this
        # network several times slower, most of all the depthwise convolutions of MobileNetV3.
        src = frames.to(self._dtype).contiguous(memory_format=torch.contiguous_format)
        mattes: List[torch.Tensor] = []
        with torch.inference_mode():
            start = 0
            if rec[0] is None:
                # First frame of a sequence: 4-D call (the official export cannot initialise its
                # states from a 5-D input).
                out = self._call(src[0:1], rec, ratio)
                mattes.append(out[1])
                rec = list(out[2:6])
                start = 1
            if batch > start:
                out = self._call(src[start:].unsqueeze(0), rec, ratio)     # [1, T, 3, H, W]
                mattes.append(out[1][0])                                    # [T, 1, H, W]
                rec = list(out[2:6])
            alpha = torch.cat(mattes, dim=0).float().clamp_(0.0, 1.0)
        # Commit the new temporal state only after the whole batch succeeded, so a failed call
        # (for example out of memory) can be retried with a smaller batch from the same state.
        self._rec = rec
        self._key = key
        return MatteOutput(alpha=alpha, presence=None, is_mask=False)

    def snapshot_state(self) -> Any:
        """The recurrent state as an opaque token (cheap: the state tensors are replaced, never
        modified in place).  Lets a composite adapter undo a step when its other network fails."""

        return (list(self._rec), self._key)

    def restore_state(self, snapshot: Any) -> None:
        self._rec, self._key = list(snapshot[0]), snapshot[1]

    def reduce_memory(self) -> bool:
        """Lower the internal working resolution by a quarter (down to 256 px on the long side)."""

        if self._fixed_ratio is not None:
            new_ratio = self._fixed_ratio * 0.75
            if new_ratio * 1.0 < 0.05:
                return False
            self._fixed_ratio = new_ratio
        else:
            new_side = int(self._internal_side * 0.75)
            if new_side < _MIN_INTERNAL_SIDE:
                return False
            self._internal_side = new_side
        self._reset_state()   # the states were computed for the old internal size
        self._warm = None
        log.warning("RVM: internal working resolution reduced to save memory.")
        return True

    def runtime_summary(self) -> Dict[str, Any]:
        summary = super().runtime_summary()
        summary.update(
            {
                "internal_side": self._internal_side if self._fixed_ratio is None else None,
                "downsample_ratio": self._key[2] if self._key else self._fixed_ratio,
                "dtype": "float16" if self._dtype == torch.float16 else "float32",
            }
        )
        summary["options"].pop("quality", None)
        return summary
