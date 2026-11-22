"""The configurable matte refiner: raw model matte in, clean and temporally stable matte out.

Everything is driven by :class:`app.core.config.RefinementParams`.  The order of operations is

1. **mask to alpha** - guided filter that turns a hard segmentation mask into a soft matte aligned
   with the image edges (``mask_to_alpha``; automatic for mask models).  The guided result is used
   as far as the picture explains the mask; where the picture has no edge the mask is only
   anti-aliased instead of being blurred into a wide halo (:func:`app.refinement.ops.mask_to_alpha`);
2. **clean-up** - remove small detached components, fill small enclosed holes;
3. **spatial operators** - erode, dilate, Gaussian smoothing, edge feather; with ``preserve_hair``
   their result is blended back to the untouched matte wherever the detail map finds thin
   structure, so strands are not wiped out;
4. **temporal filter** - motion-gated recursive smoothing (:mod:`app.refinement.temporal`);
5. **levels** - black / white point, relaxed inside moving motion blur with
   ``preserve_motion_blur`` so wide semi-transparent regions are not hardened;
6. **clamp** to 0..1.

Each step is skipped when its parameter is zero / false.  Every step keeps fully opaque and fully
transparent plateaus exact, so after the levels the matte is exactly 1.0 in the subject's core and
exactly 0.0 in the background: the compositor relies on exact equality for its bit-exact source
copy and its constant background (invariants I4 / I5 of ``docs/ARCHITECTURE.md``).

Frames are processed one at a time even when a batch is passed.  That bounds the working memory to
one frame's temporaries whatever the batch size (important at 4K on a GPU that also holds the
model), and it makes a batch call exactly equivalent to frame-by-frame calls.

The refiner also measures what it did: flicker of the raw and of the refined matte on the static
parts of the picture, sudden area jumps, and how many components / holes the clean-up touched.
"""

from __future__ import annotations

import logging
import math
from array import array
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..core.config import RefinementParams
from ..core.errors import ConfigError
from ..utils import torch_ops
from . import ops
from .ops import RefinementError
from .temporal import TemporalFilter

log = logging.getLogger(__name__)

__all__ = ["MatteRefiner"]

_MASK_TO_ALPHA_MODES = ("auto", "guided_filter", "none")


class MatteRefiner:
    """Applies the refinement chain to consecutive frames and keeps the temporal state.

    Parameters
    ----------
    params        : resolved refinement settings (pixel quantities are for a 1080-line frame).
    width, height : frame size in display orientation.
    alpha_is_mask : the model delivers a hard segmentation mask instead of a soft matte; with
                    ``mask_to_alpha == "auto"`` this switches the guided filter on.
    """

    #: Size (1080-line pixels) of the structure the hair protection treats as "thin".
    DETAIL_RADIUS = 2.5
    #: Sigma (1080-line pixels) of the anti-aliasing a hard mask gets where the picture offers no edge.
    MASK_SOFTEN = 0.75
    #: Black point / distance of the white point from 1 that remain inside moving motion blur.
    RELAXED_LEVEL = 0.004
    #: Motion-map value up to which a pixel counts as static for the flicker metric.
    STATIC_MOTION = 0.25
    #: Relative change of the matte area between two frames that is reported as a jump.
    AREA_JUMP = 0.25
    #: Matte area (fraction of the frame) below which area changes are too noisy to judge.
    MIN_JUMP_AREA = 5e-4
    #: A jump only counts when less than this share of the changed matte lies in moving picture areas.
    MAX_JUMP_MOTION = 0.3

    def __init__(
        self,
        params: RefinementParams,
        width: int,
        height: int,
        *,
        alpha_is_mask: bool = False,
    ) -> None:
        if int(width) < 1 or int(height) < 1:
            raise ConfigError(f"Invalid frame size {width}x{height} for the matte refiner.")
        if params.mask_to_alpha not in _MASK_TO_ALPHA_MODES:
            raise ConfigError(
                f"Unknown mask_to_alpha mode {params.mask_to_alpha!r}.",
                hint=f"Use one of: {', '.join(_MASK_TO_ALPHA_MODES)}.",
            )
        if not (0.0 <= params.black_point < params.white_point <= 1.0):
            raise ConfigError(
                f"Invalid levels: black_point={params.black_point}, white_point={params.white_point}.",
                hint="Use 0 <= black_point < white_point <= 1.",
            )
        self.params = params
        self.width = int(width)
        self.height = int(height)
        self.alpha_is_mask = bool(alpha_is_mask)
        enabled = bool(params.enabled)

        def px(value: float) -> float:
            return max(0.0, params.px(value, self.width, self.height))

        frame_area = float(self.width * self.height)
        self._use_guided = enabled and (
            params.mask_to_alpha == "guided_filter" or (params.mask_to_alpha == "auto" and self.alpha_is_mask)
        )
        self._guided_radius = px(params.guided_radius)
        # Fallback softening of a hard mask where the picture cannot guide the matte: plain
        # anti-aliasing, at least 0.6 real pixels whatever the resolution (a staircase is a staircase).
        self._mask_soften = max(0.6, px(self.MASK_SOFTEN)) if self.alpha_is_mask else 0.0
        self._min_component_px = params.min_artifact_area * frame_area if (enabled and params.remove_small_artifacts) else 0.0
        self._max_hole_px = params.max_hole_area * frame_area if (enabled and params.fill_holes) else 0.0
        self._erode = px(params.erode) if enabled else 0.0
        self._dilate = px(params.dilate) if enabled else 0.0
        self._smooth_sigma = px(params.gaussian_smoothing) if enabled else 0.0
        self._feather = px(params.edge_feather) if enabled else 0.0
        self._spatial_active = (self._erode + self._dilate + self._smooth_sigma + self._feather) > 0.0
        # How far the spatial operators can carry a matte sample (their combined support).
        self._reach = (
            int(math.ceil(self._erode))
            + int(math.ceil(self._dilate))
            + ops.smooth_support(self._smooth_sigma)
            + ops.smooth_support(self._feather / 2.0)
        )
        self._detail_radius = max(1.0, px(self.DETAIL_RADIUS))
        # The protected zone must cover everything the operators could smear a strand into,
        # otherwise a dilated strand would leave a detached ghost outline around itself.
        self._detail_spread = math.ceil(self._detail_radius) + self._reach
        self._crop_margin = 2 + max(
            self._reach,
            ops.detail_map_support(self._detail_radius, self._detail_spread) if params.preserve_hair else 0,
        )

        self._temporal = TemporalFilter(
            self.width,
            self.height,
            params.temporal_smoothing if enabled else 0.0,
            motion_aware=params.motion_aware,
            motion_compensation=params.motion_compensation,
            preserve_motion_blur=bool(params.preserve_motion_blur) and enabled,
        )

        self._last_index: Optional[int] = None
        self._prev_low: Optional[torch.Tensor] = None     # [1, 2, h, w]: refined and raw matte of the last frame
        self._frames = 0
        self._components_removed = 0
        self._holes_filled = 0
        self._area_jumps = 0
        self._resets = 0
        # One value per frame transition, 8 bytes each: negligible even for a film-length render.
        self._flicker = array("d")
        self._raw_flicker = array("d")

    # ------------------------------------------------------------------ public API
    def reset(self) -> None:
        """Start a new sequence: forget the temporal state.  Accumulated statistics are kept."""

        self._temporal.reset()
        self._last_index = None
        self._prev_low = None

    def refine(self, alpha: torch.Tensor, rgb: torch.Tensor, frame_indices: Sequence[int]) -> torch.Tensor:
        """Refine a batch of consecutive frames.

        ``alpha`` ``[B, 1, H, W]`` raw matte in 0..1, ``rgb`` ``[B, 3, H, W]`` the frames in display
        orientation (guide for the mask-to-alpha filter and for motion analysis), ``frame_indices``
        the ascending frame numbers.  Whenever an index is not the successor of the previous one
        (a new sequence, a seek, a gap) the temporal state is reset automatically, so the filter
        never blends across a discontinuity.

        Returns a new float32 tensor ``[B, 1, H, W]`` in 0..1 on the same device; the inputs are not
        modified.  With ``params.enabled == False`` the matte is returned unchanged apart from the
        clamp to 0..1 (non-finite samples are replaced, they would be meaningless downstream).
        """

        self._check(alpha, rgb, frame_indices)
        batch = alpha.shape[0]
        if batch == 0:
            return alpha.to(torch.float32).clamp(0.0, 1.0)
        outputs: List[torch.Tensor] = []
        for i, index in enumerate(frame_indices):
            index = int(index)
            if self._last_index is not None and index != self._last_index + 1:
                log.debug("matte refiner: frame %d does not follow %d, temporal state reset", index, self._last_index)
                self.reset()
                self._resets += 1
            self._last_index = index
            outputs.append(self._refine_frame(alpha[i : i + 1], rgb[i : i + 1]))
        return outputs[0] if batch == 1 else torch.cat(outputs, dim=0)

    def statistics(self) -> Dict[str, Any]:
        """What the refiner did so far (JSON-serialisable, for the processing report).

        ``flicker_*`` describe the refined output, ``raw_flicker_*`` the matte as it came from the
        model: for every pair of consecutive frames the mean absolute matte change over the pixels
        whose picture did not move (0 when there are none), summarised as mean / 95th percentile /
        maximum over all pairs.  ``area_jumps`` counts frame transitions where the matte area changed
        by more than 25 % although the picture was static where the matte changed.
        """

        def summary(values: Sequence[float], prefix: str) -> Dict[str, float]:
            if not len(values):
                return {f"{prefix}_mean": 0.0, f"{prefix}_p95": 0.0, f"{prefix}_max": 0.0}
            arr = np.asarray(values, dtype=np.float64)
            return {
                f"{prefix}_mean": float(arr.mean()),
                f"{prefix}_p95": float(np.percentile(arr, 95.0)),
                f"{prefix}_max": float(arr.max()),
            }

        stats: Dict[str, Any] = {
            "enabled": bool(self.params.enabled),
            "frames": self._frames,
            "frames_processed": self._frames,
            "components_removed": self._components_removed,
            "holes_filled": self._holes_filled,
        }
        stats.update(summary(self._flicker, "flicker"))
        stats.update(summary(self._raw_flicker, "raw_flicker"))
        stats["area_jumps"] = self._area_jumps
        stats["transitions"] = len(self._flicker)
        stats["temporal_resets"] = self._resets
        return stats

    # ------------------------------------------------------------------ one frame
    def _check(self, alpha: torch.Tensor, rgb: torch.Tensor, frame_indices: Sequence[int]) -> None:
        if not isinstance(alpha, torch.Tensor) or alpha.dim() != 4 or alpha.shape[1] != 1:
            got = tuple(alpha.shape) if isinstance(alpha, torch.Tensor) else type(alpha).__name__
            raise RefinementError(f"refine() expects alpha of shape [B, 1, H, W], got {got}.")
        if tuple(alpha.shape[-2:]) != (self.height, self.width):
            raise RefinementError(
                f"The matte is {alpha.shape[-1]}x{alpha.shape[-2]} but the refiner was created for "
                f"{self.width}x{self.height}.",
                hint="Create the MatteRefiner with the display-oriented frame size of the video.",
            )
        expected_rgb = (alpha.shape[0], 3, self.height, self.width)
        if not isinstance(rgb, torch.Tensor) or tuple(rgb.shape) != expected_rgb:
            got = tuple(rgb.shape) if isinstance(rgb, torch.Tensor) else type(rgb).__name__
            raise RefinementError(f"refine() expects rgb of shape {expected_rgb}, got {got}.")
        if rgb.device != alpha.device:
            raise RefinementError(f"alpha is on {alpha.device} but rgb is on {rgb.device}.")
        if len(frame_indices) != alpha.shape[0]:
            raise RefinementError(
                f"{len(frame_indices)} frame indices were given for a batch of {alpha.shape[0]} frames."
            )

    def _refine_frame(self, alpha: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """Run the chain on one frame (``[1, 1, H, W]`` / ``[1, 3, H, W]``)."""

        # A fresh, sanitised copy: the chain below never writes into caller memory, and a NaN from
        # the model can neither poison the recursive filter nor reach the compositor.
        raw = torch.nan_to_num(alpha.to(torch.float32), nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        guide = rgb if rgb.dtype == torch.float32 else rgb.to(torch.float32)

        if not self.params.enabled:
            out = raw
            self._temporal.apply(out, guide)           # strength 0: only keeps the motion analysis going
        else:
            a = raw
            if self._use_guided:
                a = ops.mask_to_alpha(
                    guide, a, self._guided_radius, self.params.guided_eps, soften_px=self._mask_soften
                )
            if self._min_component_px > 0.0:
                a, removed = ops.remove_small_components(a, self._min_component_px)
                self._components_removed += removed
            if self._max_hole_px > 0.0:
                a, filled = ops.fill_small_holes(a, self._max_hole_px)
                self._holes_filled += filled
            if self._spatial_active:
                a = self._spatial_operators(a)
            a = self._temporal.apply(a, guide)
            a = self._apply_levels(a)
            out = a.clamp(0.0, 1.0)

        self._update_statistics(raw, out)
        return out

    # ------------------------------------------------------------------ step 3
    def _active_box(self, a: torch.Tensor) -> Optional[Tuple[int, int, int, int]]:
        """Bounding box ``(y0, y1, x0, x1)`` of everything that is not flat, grown by the crop margin.

        Outside the box of all pixels that differ from a neighbour the matte is constant, and every
        spatial operator maps a constant neighbourhood to itself.  Running them on this crop is
        therefore identical to running them on the whole frame (replicate padding at the crop border
        reproduces the constant that really is there) - and usually several times cheaper.
        """

        dx = a[..., :, 1:] != a[..., :, :-1]
        dy = a[..., 1:, :] != a[..., :-1, :]
        cols = dy.any(dim=2).flatten()
        rows = dx.any(dim=3).flatten()
        col_pairs = dx.any(dim=2).flatten()
        row_pairs = dy.any(dim=3).flatten()
        cols = cols.clone()
        rows = rows.clone()
        cols[:-1] |= col_pairs
        cols[1:] |= col_pairs
        rows[:-1] |= row_pairs
        rows[1:] |= row_pairs
        xs = torch.nonzero(cols).flatten()
        ys = torch.nonzero(rows).flatten()
        if xs.numel() == 0 or ys.numel() == 0:
            return None
        bounds = torch.stack([ys[0], ys[-1], xs[0], xs[-1]]).tolist()
        m = self._crop_margin
        return (
            max(0, int(bounds[0]) - m),
            min(self.height, int(bounds[1]) + 1 + m),
            max(0, int(bounds[2]) - m),
            min(self.width, int(bounds[3]) + 1 + m),
        )

    def _spatial_operators(self, a: torch.Tensor) -> torch.Tensor:
        box = self._active_box(a)
        if box is None:
            return a                                   # a constant matte: nothing any operator could change
        y0, y1, x0, x1 = box
        original = a[..., y0:y1, x0:x1]
        x = original
        if self._erode > 0.0:
            x = ops.erode(x, self._erode)
        if self._dilate > 0.0:
            x = ops.dilate(x, self._dilate)
        if self._smooth_sigma > 0.0:
            x = ops.smooth(x, self._smooth_sigma)
        if self._feather > 0.0:
            x = ops.feather(x, self._feather)
        if self.params.preserve_hair:
            protect = ops.detail_map(original, self._detail_radius, self._detail_spread)
            # lerp is exact at both ends: untouched where nothing is protected, the original matte
            # bit for bit where the protection is complete.
            x = torch.lerp(x, original, protect)
        if (y1 - y0, x1 - x0) == (self.height, self.width):
            return x
        out = a.clone()
        out[..., y0:y1, x0:x1] = x
        return out

    # ------------------------------------------------------------------ step 5
    def _apply_levels(self, a: torch.Tensor) -> torch.Tensor:
        black, white = float(self.params.black_point), float(self.params.white_point)
        if black <= 0.0 and white >= 1.0:
            return a
        blur_zone = self._temporal.motion_blur_map if self.params.preserve_motion_blur else None
        if blur_zone is None or not bool((blur_zone > 0.0).any()):
            return ops.levels(a, black, white)
        # Inside moving motion blur the levels fade out to (almost) the identity: there the wide
        # semi-transparent ramp IS the content, and crushing it would harden the blur.  One
        # quantisation step of an 8-bit matte is still cut at both ends, so the last trace of haze
        # next to the blur becomes exactly 0 and a nearly opaque core exactly 1.  Everywhere else -
        # including faint haze in a moving background - the configured levels apply in full.
        relax = torch_ops.resize(blur_zone, (self.height, self.width), mode="bilinear").clamp_(0.0, 1.0)
        relaxed_black = min(black, self.RELAXED_LEVEL)
        relaxed_white = max(white, 1.0 - self.RELAXED_LEVEL)
        return ops.levels(a, black + (relaxed_black - black) * relax, white + (relaxed_white - white) * relax)

    # ------------------------------------------------------------------ statistics
    def _update_statistics(self, raw: torch.Tensor, out: torch.Tensor) -> None:
        self._frames += 1
        low = torch_ops.resize(torch.cat([out, raw], dim=1), self._temporal.analysis_size, mode="area")
        prev, self._prev_low = self._prev_low, low
        motion = self._temporal.motion_map
        if prev is None or motion is None:
            return                                     # first frame of a sequence: nothing to compare with
        if prev.device != low.device:
            prev = prev.to(low.device)
        change = (low - prev).abs()                                            # [1, 2, h, w]
        static = (motion <= self.STATIC_MOTION).to(low.dtype)                  # [1, 1, h, w]
        static_count = static.sum()
        flicker = (change * static).sum(dim=(0, 2, 3)) / static_count.clamp_min(1.0)
        refined_change = change[:, 0:1]
        values = torch.stack(
            [
                flicker[0],
                flicker[1],
                low[:, 0].mean(),
                prev[:, 0].mean(),
                refined_change.sum(),
                (refined_change * motion).sum(),
            ]
        ).tolist()                                                             # one device round trip
        refined_flicker, raw_flicker, area, prev_area, changed, changed_moving = (float(v) for v in values)
        self._flicker.append(refined_flicker)
        self._raw_flicker.append(raw_flicker)

        larger = max(area, prev_area)
        if larger >= self.MIN_JUMP_AREA and abs(area - prev_area) > self.AREA_JUMP * larger:
            # "While the image motion is low": judged where the matte changed, not over the whole
            # frame - a subject walking into a still picture is motion, not a jump.
            if changed <= 0.0 or changed_moving / changed < self.MAX_JUMP_MOTION:
                self._area_jumps += 1
