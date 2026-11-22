"""MatAnyone / MatAnyone 2 adapter: true video matting of one person, started from a mask.

MatAnyone (Yang et al., CVPR 2025; MatAnyone 2, CVPR 2026) propagates a soft alpha matte through a
video from a segmentation mask of the target on the first frame.  It combines what the other models
offer separately - it follows *one assigned* person and predicts genuine soft edges (hair, motion
blur) - but its licence forbids commercial use, which the tool enforces.

How the adapter drives it (the official streaming API, ``InferenceCore.step``):

* first frame of a sequence: the mask is memorised, then the frame is predicted ``n_warmup + 1``
  times so the temporal memory settles (the official warm-up procedure);
* every further frame: one ``step``.

The network's memory use grows with the square of the frame size, so frames are analysed at a
reduced *internal* resolution chosen from the GPU memory that is really free; the matte is returned
at full size.  If a step fails (out of memory) the temporal memory is rebuilt on the next call from
the last matte the adapter produced, at a lower internal resolution if the pipeline asks for it.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ...core.errors import ModelError, ModelLoadError, SubjectError
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

log = logging.getLogger(__name__)

#: Internal short sides the adapter steps through when memory is short.
_LADDER: Tuple[int, ...] = (1080, 864, 720, 576, 480, 432, 384, 288)
_FLOOR = _LADDER[-1]


def prepare_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binarise the subject mask and close small gaps exactly like the official script does
    (dilate, then erode, with an elliptical kernel); returns float32 with values 0 / 255."""

    binary = (np.asarray(mask) > 0.5).astype(np.uint8) * 255
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(radius), int(radius)))
        binary = cv2.erode(cv2.dilate(binary, kernel, iterations=1), kernel, iterations=1)
    return binary.astype(np.float32)


def internal_size(height: int, width: int, short_side: int) -> Tuple[int, int]:
    """Frame size used inside the network: the short side reduced to *short_side* (never enlarged)."""

    shortest = min(height, width)
    if short_side <= 0 or shortest <= short_side:
        return height, width
    return max(16, int(round(height * short_side / shortest))), max(16, int(round(width * short_side / shortest)))


@register_model
class MatAnyoneAdapter(MattingModel):
    info = ModelInfo(
        id="matanyone",
        name="MatAnyone",
        description=(
            "Target-assigned video matting: follows the one person given by a mask on the reference frame and "
            "predicts a true soft alpha (hair, motion blur). Highest matte quality for people; non-commercial licence."
        ),
        license=LicenseInfo(
            name="NTU S-Lab License 1.0",
            url="https://github.com/pq-yang/MatAnyone/blob/main/LICENSE",
            commercial_use="prohibited",
            notes="Code and weights may be used for research and other non-commercial purposes only. "
            "Commercial use needs written permission from the authors.",
        ),
        capabilities=ModelCapabilities(
            alpha_type="alpha",
            native_prompts=frozenset({"mask"}),
            needs_initial_mask=True,
            subject_domain="human",
            temporal="memory",
            max_batch_size=1,
            supports_fp16=True,
            devices=frozenset({"cuda", "cpu"}),
            supports_reverse=True,
            tracks_single_subject=True,
        ),
        variants=(
            ModelVariant(
                id="matanyone2", label="MatAnyone 2 (recommended)", vram_hint_mb=2500,
                description="Weights of MatAnyone 2 (CVPR 2026); same network, better robustness.",
                weights=(WeightFile(
                    filename="matanyone2.pth",
                    url="https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth",
                    size_bytes=141_429_115,
                    sha256="5e9821e4087231427376b437c85bb6e072b41e582314f06fd524f75bc4af5914",
                    source="GitHub release v1.0.0 of pq-yang/MatAnyone2 (official)",
                ),),
            ),
            ModelVariant(
                id="matanyone", label="MatAnyone (original)", vram_hint_mb=2500,
                description="Weights of the original MatAnyone (CVPR 2025).",
                weights=(WeightFile(
                    filename="matanyone.pth",
                    url="https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth",
                    size_bytes=141_429_992,
                    sha256="dd26b991d020ed5eb4be50996f97354c45cfdfc0f59958e8983ac6a198f4809d",
                    source="GitHub release v1.0.0 of pq-yang/MatAnyone (official)",
                ),),
            ),
        ),
        default_variant="matanyone2",
        quality_variants={"fast": "matanyone2", "balanced": "matanyone2", "high": "matanyone2", "max": "matanyone2"},
        limitations=(
            "Non-commercial licence (NTU S-Lab License 1.0): refused when commercial use is switched on.",
            "Needs a mask of the subject on the reference frame. Boxes, clicks, text and automatic selection are "
            "turned into that mask by the helper models (SAM 2 / the detector), which must be installed for them.",
            "Trained on people: other subjects are tracked but their matte quality is not assured.",
            "Memory grows with the square of the frame size, so frames are analysed at a reduced internal "
            "resolution chosen from the free GPU memory; fine detail is softer when that is below the source size.",
            "One subject per run; no recovery after a cut to a different shot.",
        ),
        requires=("matanyone", "omegaconf"),
        install_hint="Install the package with:  python scripts/install_models.py matanyone --accept-license",
        homepage="https://github.com/pq-yang/MatAnyone",
        citation="Yang et al.: MatAnyone: Stable Video Matting with Consistent Memory Propagation. CVPR 2025.",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._network: Any = None
        self._core: Any = None
        self._amp = False
        self._warmup = max(0, int(self.options.get("n_warmup", 10)))
        self._mem_every = max(1, int(self.options.get("mem_every", 5)))
        self._max_mem = max(2, int(self.options.get("max_mem_frames", 5)))
        forced = self.options.get("internal_short_side")
        self._forced_side: Optional[int] = int(forced) if forced not in (None, "", "auto") else None
        self._side = 0                                   # current internal short side
        self._size = (0, 0)                              # (height, width) of the sequence
        self._seed: Optional[np.ndarray] = None          # prepared mask (0 / 255) for the next initialisation
        self._last_alpha: Optional[torch.Tensor] = None  # last matte produced (for recovery)
        self._radius = 0

    # ------------------------------------------------------------------------- life cycle
    def load(self) -> None:
        from .loader import build_network

        path = self.weight_path(self.variant.weights[0])
        self._network = build_network(path, self.device)           # raises ModelLoadError with details
        self._amp = self.precision == "fp16" and self.device.type == "cuda"
        self.loaded = True
        log.info("MatAnyone (%s) loaded on %s (%s)", self.variant.id, self.device, "fp16" if self._amp else "fp32")

    def unload(self) -> None:
        self._drop_core()
        self._network = None
        self._seed = None
        self._last_alpha = None
        super().unload()

    def _drop_core(self) -> None:
        if self._core is not None:
            try:
                self._core.image_feature_store._store.clear()
            except Exception:  # pragma: no cover - attribute of a third-party class
                pass
        self._core = None

    def _free_vram_mb(self) -> Optional[float]:
        if self.device.type != "cuda":
            return None
        try:
            from ...core.hardware import cuda_index, gpu_stats

            stats = gpu_stats(cuda_index(self.device))
            if stats["gpu_memory_total_mb"] and stats["gpu_memory_used_mb"] is not None:
                return float(stats["gpu_memory_total_mb"] - stats["gpu_memory_used_mb"])
            free, _ = torch.cuda.mem_get_info(self.device)
            return free / (1024.0 * 1024.0)
        except Exception:
            return None

    def _choose_side(self, height: int, width: int, hint: Optional[int]) -> int:
        """Internal short side: forced by option, else from the free GPU memory (measured on an
        RTX 3060 in half precision: about 0.6 GB at 432, 1.1 GB at 576, 1.7 GB at 720)."""

        shortest = min(height, width)
        if self._forced_side is not None:
            return max(_FLOOR, min(self._forced_side, shortest))
        if self.device.type != "cuda":
            side = 432                                       # the CPU path is slow: keep it small
        else:
            free = self._free_vram_mb()
            if free is None or free >= 9000:
                side = 1080
            elif free >= 4500:
                side = 720
            elif free >= 2600:
                side = 576
            elif free >= 1600:
                side = 480
            else:
                side = 432
        if hint:
            side = min(side, max(_FLOOR, int(hint * shortest / max(height, width))))
        return max(_FLOOR, min(side, shortest))

    def start_sequence(self, seq: SequenceInfo) -> None:
        if self._network is None:
            raise ModelLoadError("MatAnyone adapter used before load().")
        subject = seq.subject
        if subject is None or subject.mask is None:
            raise SubjectError(
                "MatAnyone needs a mask of the subject on the reference frame, but none was provided.",
                hint="Give a mask image (--subject mask:<file>), or install SAM 2 and the detector so that boxes, "
                "clicks and automatic selection can be turned into a mask "
                "(python scripts/install_models.py sam2; python main.py models download detector).",
            )
        height, width = int(seq.height), int(seq.width)
        mask = np.asarray(subject.mask)
        if mask.shape != (height, width):
            raise SubjectError(f"The subject mask is {mask.shape[1]}x{mask.shape[0]} but the frames are {width}x{height}.")
        # Kernel size of the official script, scaled with the resolution (15 at 1080p, 10 at 720p).
        self._radius = max(2, int(round(min(height, width) / 72.0)))
        seed = prepare_mask(mask, self._radius)
        if not seed.any():
            raise SubjectError("The subject mask is empty.", hint="Select the subject on a frame where it is visible.")
        self._drop_core()
        self._size = (height, width)
        self._seed = seed
        self._last_alpha = None
        self._side = self._choose_side(height, width, seq.inference_max_side)

    def end_sequence(self) -> None:
        self._drop_core()
        self._seed = None
        self._last_alpha = None

    # ------------------------------------------------------------------------- inference
    def _autocast(self) -> Any:
        return torch.autocast("cuda", dtype=torch.float16) if self._amp else contextlib.nullcontext()

    def process(self, frames: torch.Tensor, frame_indices: Sequence[int]) -> MatteOutput:
        from .loader import bound_device, create_inference_core

        if self._network is None or self._seed is None:
            raise ModelError("MatAnyone adapter: process() called before start_sequence().")
        if frames.shape[0] != 1:
            raise ModelError("MatAnyone processes one frame per call (max_batch_size is 1).")
        height, width = int(frames.shape[-2]), int(frames.shape[-1])
        if (height, width) != self._size:
            raise ModelError(f"MatAnyone adapter: frame size changed within a sequence ({width}x{height}).")
        inner = internal_size(height, width, self._side)
        image = frames.to(torch.float32)
        if inner != (height, width):
            image = F.interpolate(image, size=inner, mode="area")
        image = image[0].clamp(0.0, 1.0).contiguous()
        try:
            with bound_device(self.device), self._autocast():
                if self._core is None:
                    # (Re)build the temporal memory from the seed mask on THIS frame.
                    core = create_inference_core(self._network, self.device, mem_every=self._mem_every,
                                                 max_mem_frames=self._max_mem)
                    seed = torch.from_numpy(self._seed).to(self.device)
                    if inner != (height, width):
                        seed = F.interpolate(seed[None, None], size=inner, mode="nearest")[0, 0]
                    core.step(image, seed, objects=[1])
                    probability = None
                    for _ in range(self._warmup + 1):
                        probability = core.step(image, first_frame_pred=True)
                    self._core = core
                else:
                    probability = self._core.step(image)
                # The returned tensor shares storage with the network's own state: copy before use.
                alpha = self._core.output_prob_to_mask(probability).to(torch.float32).clone()
        except BaseException:
            # The memory may be half-updated: rebuild it on the next call from the last good matte.
            self._drop_core()
            if self._last_alpha is not None:
                recovered = (self._last_alpha[0, 0] > 0.5).cpu().numpy()
                if recovered.any():
                    self._seed = prepare_mask(recovered, self._radius)
            raise
        alpha = alpha[None, None]
        if alpha.shape[-2:] != (height, width):
            alpha = F.interpolate(alpha, size=(height, width), mode="bilinear", align_corners=False)
        alpha = alpha.clamp_(0.0, 1.0)
        self._last_alpha = alpha
        return MatteOutput(alpha=alpha, presence=None, is_mask=False)

    def reduce_memory(self) -> bool:
        """Step the internal resolution down one notch and rebuild the memory from the last matte."""

        lower = [side for side in _LADDER if side < self._side]
        if not lower:
            return False
        self._side = lower[0]
        self._drop_core()
        if self._last_alpha is not None:
            recovered = (self._last_alpha[0, 0] > 0.5).cpu().numpy()
            if recovered.any():
                self._seed = prepare_mask(recovered, self._radius)
        log.warning("MatAnyone: internal resolution reduced to a short side of %d px to save memory.", self._side)
        return True

    def runtime_summary(self) -> Dict[str, Any]:
        summary = super().runtime_summary()
        summary["options"].pop("quality", None)
        summary.update({"internal_short_side": self._side or None, "n_warmup": self._warmup,
                        "mem_every": self._mem_every, "max_mem_frames": self._max_mem,
                        "autocast": "float16" if self._amp else "off"})
        return summary
