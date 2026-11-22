"""SAM 2.1 adapter: promptable video segmentation that follows ONE selected subject.

SAM 2 (Meta, Apache-2.0) segments whatever the user points at - a person, an animal, a product -
and tracks it with a memory of earlier frames, through occlusions and re-entries.  It produces a
*mask*, not a matte: fine detail such as hair comes from this tool's refinement stage (guided
filter), or from the ``sam2_rvm`` hybrid which borrows a true alpha for people.

How the adapter drives the predictor: the pipeline *pushes* frames one at a time, while SAM 2's
predictor *pulls* frames by index from its state.  The bridge is a one-slot frame provider
(:class:`~app.models.sam2.streaming.PushFrameProvider`) plus the predictor's own propagation
generator, advanced by exactly one frame per :meth:`process` call.  Old tracking outputs are pruned
after every frame, so memory use does not grow with the length of the video.

Retry safety: if a step fails (for example out of GPU memory) the generator is discarded and
recreated at the same frame on the next call.  SAM 2 stores a frame's output only after the frame
was computed completely, so the tracking memory is exactly what it was before the failed call.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch

from ...core.errors import ModelError, ModelLoadError, SubjectError
from ...core.types import ResolvedSubject
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

_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
_SOURCE = "Meta AI public files (official SAM 2.1 checkpoints)"
_OBJECT_ID = 1


def _variant(variant_id: str, label: str, filename: str, size: int, sha256: str, vram: int, text: str) -> ModelVariant:
    return ModelVariant(
        id=variant_id, label=label, description=text, vram_hint_mb=vram,
        weights=(WeightFile(filename=filename, url=f"{_BASE_URL}/{filename}", size_bytes=size, sha256=sha256,
                            description="Official SAM 2.1 checkpoint", source=_SOURCE),),
    )


@register_model
class Sam2Adapter(MattingModel):
    info = ModelInfo(
        id="sam2",
        name="SAM 2.1 (Segment Anything Model 2)",
        description=(
            "Promptable video segmentation: follows the one subject you select (box, clicks, mask or automatic "
            "detection) through occlusions, for people and any other object. Produces masks that the refinement "
            "stage turns into soft mattes."
        ),
        license=LicenseInfo(
            name="Apache-2.0",
            spdx="Apache-2.0",
            url="https://github.com/facebookresearch/sam2/blob/main/LICENSE",
            commercial_use="allowed",
            notes="Code and checkpoints are released under Apache-2.0.",
        ),
        capabilities=ModelCapabilities(
            alpha_type="mask",
            native_prompts=frozenset({"points", "box", "mask"}),
            needs_initial_mask=False,
            subject_domain="any",
            temporal="memory",
            max_batch_size=1,
            supports_fp16=True,
            devices=frozenset({"cuda", "cpu"}),
            supports_reverse=True,
            tracks_single_subject=True,
        ),
        variants=(
            _variant("tiny", "Hiera-Tiny (fastest)", "sam2.1_hiera_tiny.pt", 156_008_466,
                     "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69", 1100,
                     "39 M parameters. Fastest; slightly less stable on small or fast subjects."),
            _variant("small", "Hiera-Small (recommended)", "sam2.1_hiera_small.pt", 184_416_285,
                     "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38", 1200,
                     "46 M parameters. Good balance of speed and tracking stability."),
            _variant("base_plus", "Hiera-Base+ (more accurate)", "sam2.1_hiera_base_plus.pt", 323_606_802,
                     "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5", 1700,
                     "81 M parameters. More accurate masks; slower."),
            _variant("large", "Hiera-Large (most accurate)", "sam2.1_hiera_large.pt", 898_083_611,
                     "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318", 3200,
                     "224 M parameters. Most accurate; needs the most memory and time."),
        ),
        default_variant="small",
        quality_variants={"fast": "tiny", "balanced": "small", "high": "small", "max": "base_plus"},
        limitations=(
            "Produces segmentation masks, not alpha mattes: hair, fur and motion blur are approximated by the "
            "refinement stage (guided filter). For people, the 'sam2_rvm' hybrid or 'matanyone' give true soft edges.",
            "Masks are predicted at 256 x 256 internally and upsampled: very thin structures (fingers, straps, "
            "antennas) can be lost or merged.",
            "Needs a prompt on the reference frame: a box, clicks, a mask or automatic detection (which needs the "
            "'detector' helper model).",
            "Tracks one subject per run. Very long occlusions or a cut to a different shot can lose the subject; "
            "frames where it is not visible are reported.",
            "Every frame is squashed to 1024 x 1024 for analysis, so very wide or very tall pictures lose "
            "resolution along their long side.",
        ),
        requires=("sam2", "hydra"),
        install_hint="Install the SAM 2 package with:  python scripts/install_models.py sam2",
        homepage="https://github.com/facebookresearch/sam2",
        citation="Ravi et al.: SAM 2: Segment Anything in Images and Videos. ICLR 2025.",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._predictor: Any = None
        self._provider: Any = None
        self._state: Optional[Dict[str, Any]] = None
        self._generator: Optional[Iterator[Any]] = None
        self._subject: Optional[ResolvedSubject] = None
        self._reverse = False
        self._prompted = False
        self._last_index: Optional[int] = None
        self._size = (0, 0)                                    # (height, width) of the sequence
        self._autocast_dtype: Optional[torch.dtype] = None
        self._prune_margin = max(0, int(self.options.get("prune_margin", 1)))
        self._peak_entries = 0

    # ------------------------------------------------------------------------- life cycle
    def load(self) -> None:
        from .streaming import build_predictor

        path = self.weight_path(self.variant.weights[0])
        try:
            self._predictor = build_predictor(self.variant.id, path, self.device)
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load SAM 2.1 ({self.variant.id}) from {path}: {exc}",
                hint="Verify the file with 'python main.py models verify sam2' and that the SAM 2 package is "
                "installed (python scripts/install_models.py sam2).",
            ) from exc
        if self.precision == "fp16" and self.device.type == "cuda":
            # Meta runs SAM 2 under bfloat16 autocast; float16 is the fallback for GPUs without it.
            self._autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self._autocast_dtype = None
        self.loaded = True
        log.info("SAM 2.1 %s loaded on %s (autocast %s)", self.variant.id, self.device,
                 str(self._autocast_dtype).replace("torch.", "") if self._autocast_dtype else "off")

    def unload(self) -> None:
        self._drop_sequence()
        self._predictor = None
        super().unload()

    def _amp(self) -> Any:
        if self._autocast_dtype is not None:
            return torch.autocast("cuda", dtype=self._autocast_dtype)
        return contextlib.nullcontext()

    def _drop_generator(self) -> None:
        if self._generator is not None:
            try:
                self._generator.close()
            except Exception:  # pragma: no cover - closing must never mask the real error
                pass
            self._generator = None

    def _drop_sequence(self) -> None:
        self._drop_generator()
        self._state = None
        if self._provider is not None:
            self._provider.clear()
        self._provider = None
        self._prompted = False
        self._last_index = None

    def _new_state(self) -> None:
        from .streaming import PushFrameProvider, init_streaming_state

        height, width = self._size
        self._provider = PushFrameProvider(self._predictor.image_size, self.device)
        self._state = init_streaming_state(self._predictor, self._provider, height, width)
        self._generator = None
        self._prompted = False
        self._last_index = None

    def start_sequence(self, seq: SequenceInfo) -> None:
        if self._predictor is None:
            raise ModelLoadError("SAM 2 adapter used before load().")
        subject = seq.subject
        if subject is None or not (subject.has_mask or subject.box is not None or subject.points):
            raise SubjectError(
                "SAM 2 needs to be told which subject to follow, but no box, clicks or mask were provided.",
                hint="Select the subject (--subject box:x1,y1,x2,y2 / points:x,y / mask:<image>), or install the "
                "'detector' helper model for automatic selection (python main.py models download detector).",
            )
        if subject.points and not any(label == 1 for _, _, label in subject.points) and subject.box is None \
                and not subject.has_mask:
            raise SubjectError("The subject selection contains only negative clicks; add a click on the subject.")
        self._drop_sequence()
        self._subject = subject
        self._reverse = bool(seq.reverse)
        self._size = (int(seq.height), int(seq.width))
        self._new_state()

    def end_sequence(self) -> None:
        self._drop_sequence()
        self._subject = None

    # ------------------------------------------------------------------------- inference
    def _add_prompt(self, index: int) -> torch.Tensor:
        """Give the subject to SAM 2 on the reference frame; returns its mask logits there."""

        subject = self._subject
        assert subject is not None and self._state is not None
        height, width = self._size
        if subject.box is not None or subject.points:
            points = labels = None
            if subject.points:
                points = np.asarray([[x, y] for x, y, _ in subject.points], dtype=np.float32)
                labels = np.asarray([label for _, _, label in subject.points], dtype=np.int32)
            box = None if subject.box is None else np.asarray(subject.box, dtype=np.float32)
            _, _, logits = self._predictor.add_new_points_or_box(
                self._state, frame_idx=index, obj_id=_OBJECT_ID, points=points, labels=labels, box=box,
            )
            return logits
        mask = np.asarray(subject.mask)
        if mask.shape != (height, width):
            raise SubjectError(
                f"The subject mask is {mask.shape[1]}x{mask.shape[0]} but the frames are {width}x{height}."
            )
        binary = mask > 0.5
        if not binary.any():
            raise SubjectError("The subject mask is empty.", hint="Select the subject on a frame where it is visible.")
        _, _, logits = self._predictor.add_new_mask(self._state, frame_idx=index, obj_id=_OBJECT_ID, mask=binary)
        return logits

    def _start_generator(self, index: int) -> None:
        self._generator = self._predictor.propagate_in_video(
            self._state, start_frame_idx=index, max_frame_num_to_track=None, reverse=self._reverse,
        )

    def process(self, frames: torch.Tensor, frame_indices: Sequence[int]) -> MatteOutput:
        from .streaming import frame_output, prune_tracking_state, state_entries

        if self._predictor is None or self._state is None:
            raise ModelError("SAM 2 adapter: process() called before start_sequence().")
        if frames.shape[0] != 1 or len(frame_indices) != 1:
            raise ModelError("SAM 2 processes one frame per call (max_batch_size is 1).")
        index = int(frame_indices[0])
        height, width = int(frames.shape[-2]), int(frames.shape[-1])
        if (height, width) != self._size:
            raise ModelError(f"SAM 2 adapter: frame size changed within a sequence ({width}x{height}).")
        if self._last_index is not None:
            expected = self._last_index + (-1 if self._reverse else 1)
            if index != expected:
                raise ModelError(
                    f"SAM 2 adapter: frames must be consecutive (expected frame {expected}, got {index})."
                )
        first = not self._prompted
        prompt_logits: Optional[torch.Tensor] = None
        try:
            self._provider.push(index, frames[0])
            with self._amp():
                if first:
                    prompt_logits = self._add_prompt(index)
                if self._generator is None:
                    self._start_generator(index)
                try:
                    produced_index, _, logits = next(self._generator)
                except StopIteration:
                    # Reverse tracking that starts on frame 0 has nothing to propagate: the mask of
                    # the prompted frame itself is the result.
                    if not first or prompt_logits is None:
                        raise ModelError("SAM 2 stopped tracking unexpectedly.") from None
                    produced_index, logits = index, prompt_logits
                    self._generator = None
        except BaseException:
            # Leave everything as it was before this call (see the module docstring).
            self._drop_generator()
            if first:
                self._new_state()                 # a half-registered prompt is simply forgotten
            elif self._provider is not None:
                self._provider.discard(index)
            raise
        if int(produced_index) != index:
            self._drop_generator()
            raise ModelError(f"SAM 2 adapter: internal frame order mismatch ({produced_index} != {index}).")
        self._prompted = True
        self._last_index = index

        output = frame_output(self._state, index)
        presence: Optional[List[float]] = None
        if output is not None and output.get("object_score_logits") is not None:
            score = output["object_score_logits"].float().reshape(-1)[0]
            presence = [float(torch.sigmoid(score))]
        alpha = torch.sigmoid(logits[:1].float()).clamp_(0.0, 1.0)             # [1, 1, H, W]
        if alpha.shape[-2:] != (height, width):                                # defensive: always full size
            alpha = torch.nn.functional.interpolate(alpha, size=(height, width), mode="bilinear", align_corners=False)
        self._peak_entries = max(self._peak_entries, state_entries(self._state)["non_cond_entries"])
        prune_tracking_state(self._predictor, self._state, index, self._reverse, self._prune_margin)
        return MatteOutput(alpha=alpha, presence=presence, is_mask=True)

    def reduce_memory(self) -> bool:
        """SAM 2 has no cheaper configuration to fall back to at run time (its working size is fixed);
        the pipeline then reports the shortage or continues on the CPU. Use a smaller variant instead."""

        return False

    def runtime_summary(self) -> Dict[str, Any]:
        from .streaming import CONFIGS, memory_window

        summary = super().runtime_summary()
        summary["options"].pop("quality", None)
        summary.update(
            {
                "config": CONFIGS.get(self.variant.id),
                "autocast": str(self._autocast_dtype).replace("torch.", "") if self._autocast_dtype else "off",
                "working_size": int(self._predictor.image_size) if self._predictor is not None else None,
                "memory_window_frames": memory_window(self._predictor) if self._predictor is not None else None,
                "peak_tracking_entries": self._peak_entries,
            }
        )
        return summary
