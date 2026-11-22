"""Bounded-memory streaming on top of Meta's SAM 2 video predictor.

The stock ``SAM2VideoPredictor`` loads every frame of a clip into one tensor (12.6 MB per frame at
its 1024 x 1024 working size) and keeps every per-frame output forever, so memory grows with the
length of the video.  This module runs the same predictor over arbitrarily long videos with constant
memory, without patching upstream code:

* frames are handed to the predictor through :class:`PushFrameProvider`, an object that stands in
  for the predictor's ``images`` tensor and only ever holds the frames currently being worked on;
* after every frame, tracking outputs that can never be read again are deleted
  (:func:`prune_tracking_state`).

Only two details of the upstream implementation (facebookresearch/sam2, commit 2b90b9f) are relied
on, both validated against the stock predictor (identical logits on forward, reverse, multi-prompt
and chunked runs):

1. the layout of ``inference_state`` created by ``SAM2VideoPredictor.init_state`` and the single
   place where frames are read (``inference_state["images"][frame_idx]``);
2. which earlier outputs ``SAM2Base._prepare_memory_conditioned_features`` reads: all conditioning
   frames, the ``num_maskmem - 1`` spatial memories nearest to the current frame and the object
   pointers of the last ``max_obj_ptrs_in_encoder - 1`` frames.

Everything else goes through the public predictor API.  The adapter must therefore stay pinned to
that commit (see ``scripts/install_models.py``).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

#: ImageNet statistics used by SAM 2's own frame loader.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

#: "Number of frames" told to the predictor: the real count is irrelevant for streaming, it only
#: bounds the propagation range.
UNBOUNDED_FRAMES = 1 << 30

#: Hydra config names inside the installed ``sam2`` package, per variant.
CONFIGS: Dict[str, str] = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


class FrameNotAvailable(IndexError):
    """The predictor asked for a frame that has not been pushed (or was already dropped)."""


def build_predictor(variant_id: str, checkpoint_path: Any, device: Any) -> Any:
    """Build the SAM 2.1 video predictor of one variant from a LOCAL checkpoint (no downloads).

    The returned object also serves single images: ``SAM2ImagePredictor(predictor)`` shares its
    weights (see ``app.tracking.backends``).
    """

    from sam2.build_sam import build_sam2_video_predictor
    import sam2.sam2_video_predictor as video_module

    if variant_id not in CONFIGS:
        raise KeyError(f"unknown SAM 2 variant {variant_id!r}")
    predictor = build_sam2_video_predictor(CONFIGS[variant_id], str(Path(checkpoint_path)), device=str(device))
    # Hole filling needs SAM 2's optional compiled CUDA extension, which is deliberately not built;
    # without it every frame would only produce a failed import and a warning.  Small holes are
    # filled by this tool's own matte refinement instead.
    predictor.fill_hole_area = 0
    # propagate_in_video wraps its loop in a tqdm progress bar; the tool reports progress itself.
    video_module.tqdm = lambda iterable, *args, **kwargs: iterable
    predictor.eval()
    return predictor


class PushFrameProvider:
    """Stands in for the predictor's ``images`` tensor: ``images[frame_idx]`` returns the
    normalised ``(3, S, S)`` tensor of a frame that was pushed, and nothing else is kept."""

    def __init__(self, image_size: int, device: Any, keep: int = 2) -> None:
        self.image_size = int(image_size)
        self.device = torch.device(device)
        self.keep = max(1, int(keep))
        self._mean = torch.tensor(_MEAN, dtype=torch.float32, device=self.device)[:, None, None]
        self._std = torch.tensor(_STD, dtype=torch.float32, device=self.device)[:, None, None]
        self._frames: "OrderedDict[int, torch.Tensor]" = OrderedDict()

    def __len__(self) -> int:
        return UNBOUNDED_FRAMES

    def push(self, frame_idx: int, rgb: torch.Tensor) -> None:
        """Store frame *frame_idx*; ``rgb`` is ``[3, H, W]`` float R'G'B' in 0..1 on the device."""

        size = self.image_size
        image = rgb.to(self.device, torch.float32).unsqueeze(0)
        if image.shape[-2:] != (size, size):
            # SAM 2 squashes every frame to a square; antialiasing matters when shrinking HD frames.
            image = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        image = (image[0].clamp(0.0, 1.0) - self._mean) / self._std
        self._frames[int(frame_idx)] = image
        self._frames.move_to_end(int(frame_idx))
        while len(self._frames) > self.keep:
            self._frames.popitem(last=False)

    def discard(self, frame_idx: int) -> None:
        self._frames.pop(int(frame_idx), None)

    def clear(self) -> None:
        self._frames.clear()

    def __getitem__(self, frame_idx: Any) -> torch.Tensor:
        try:
            return self._frames[int(frame_idx)]
        except KeyError:
            raise FrameNotAvailable(int(frame_idx)) from None


def init_streaming_state(predictor: Any, images: Any, video_height: int, video_width: int) -> Dict[str, Any]:
    """The same keys as ``SAM2VideoPredictor.init_state`` creates, without loading any frame."""

    device = predictor.device
    state: Dict[str, Any] = {
        "images": images,
        "num_frames": UNBOUNDED_FRAMES,
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
        "video_height": int(video_height),
        "video_width": int(video_width),
        "device": device,
        "storage_device": device,
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "cached_features": {},
        "constants": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "frames_tracked_per_obj": {},
    }
    return state


def memory_window(predictor: Any) -> int:
    """How many frames behind the current one may still be read while tracking the next frame.

    Spatial memories reach back ``(num_maskmem - 2) * stride`` frames, object pointers
    ``max_obj_ptrs_in_encoder - 2`` frames; the larger of the two is the window.
    """

    stride = max(1, int(predictor.memory_temporal_stride_for_eval))
    memory_span = max(0, (int(predictor.num_maskmem) - 2) * stride)
    pointer_span = max(0, int(predictor.max_obj_ptrs_in_encoder) - 2) if predictor.use_obj_ptrs_in_encoder else 0
    return max(memory_span, pointer_span)


def prune_tracking_state(predictor: Any, state: Dict[str, Any], frame_idx: int, reverse: bool, margin: int = 1) -> int:
    """Delete non-conditioning outputs that tracking in the current direction can never read again.

    Conditioning-frame outputs (the prompted frames) are never touched.  Returns the number of
    entries removed.
    """

    keep = memory_window(predictor) + int(margin)
    removed = 0
    for obj_idx, outputs in state["output_dict_per_obj"].items():
        non_cond = outputs["non_cond_frame_outputs"]
        cond = outputs["cond_frame_outputs"]
        tracked = state["frames_tracked_per_obj"][obj_idx]
        if reverse:
            dead = [t for t in non_cond if t > frame_idx + keep]
            dead_tracked = [t for t in tracked if t > frame_idx + keep and t not in cond]
        else:
            dead = [t for t in non_cond if t < frame_idx - keep]
            dead_tracked = [t for t in tracked if t < frame_idx - keep and t not in cond]
        for t in dead:
            del non_cond[t]
        for t in dead_tracked:
            del tracked[t]
        removed += len(dead)
    return removed


def state_entries(state: Dict[str, Any]) -> Dict[str, int]:
    """Sizes of everything in the tracking state that could grow with the video length."""

    non_cond = cond = tracked = 0
    for obj_idx, outputs in state["output_dict_per_obj"].items():
        non_cond = max(non_cond, len(outputs["non_cond_frame_outputs"]))
        cond = max(cond, len(outputs["cond_frame_outputs"]))
        tracked = max(tracked, len(state["frames_tracked_per_obj"][obj_idx]))
    return {
        "non_cond_entries": non_cond,
        "cond_entries": cond,
        "frames_tracked": tracked,
        "cached_feature_frames": len(state["cached_features"]),
    }


def frame_output(state: Dict[str, Any], frame_idx: int, obj_idx: int = 0) -> Optional[Dict[str, Any]]:
    """The stored tracking output of one frame (conditioning or not), or None."""

    outputs = state["output_dict_per_obj"].get(obj_idx)
    if outputs is None:
        return None
    found = outputs["cond_frame_outputs"].get(frame_idx)
    if found is None:
        found = outputs["non_cond_frame_outputs"].get(frame_idx)
    return found
