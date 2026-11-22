"""Motion-aware temporal smoothing of the matte.

Matting models flicker: the matte of a perfectly still edge changes a little from frame to frame.
A recursive filter removes that, but a plain exponential average also drags a trail behind every
moving edge and washes out motion blur.  :class:`TemporalFilter` therefore blends per pixel::

    out_t = alpha_t + w * (prev_out - alpha_t)          w = strength * g,   g in 0..1

where the gate ``g`` is the product of

* an **image-motion gate** - the local difference between the current and the previous picture,
  measured on a small blurred luma: where the picture moves the matte is supposed to move too, so
  nothing is blended there (no trails, no lag);
* a **matte-change gate** - a change of the matte far above flicker amplitude is real content
  appearing or disappearing and passes unfiltered (no ghosts), small fluctuations are smoothed;
* a **motion-blur gate** - wide semi-transparent regions inside moving areas (motion blur, flying
  hair) are never blended, even where the two other gates would allow a little.

With motion compensation the previous output is first warped along dense optical flow (DIS, on the
small luma, on the CPU), so smoothing follows the motion; the image-motion gate is then evaluated
on what the warp could not explain (residual and forward/backward inconsistency).

Exactness.  The blend is written as ``alpha + w * (prev - alpha)``: where the previous output and
the current matte are equal the result is that value bit for bit, whatever ``w`` is.  A constant
matte in a static scene therefore never drifts, and stable 0.0 / 1.0 pixels stay exactly 0.0 / 1.0
(the compositor depends on it).
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..core.errors import ConfigError
from ..utils import torch_ops
from .ops import RefinementError

log = logging.getLogger(__name__)

__all__ = ["TemporalFilter"]


class TemporalFilter:
    """Recursive, motion-gated temporal filter for one frame at a time (frames must arrive in order).

    Parameters
    ----------
    width, height : frame size in pixels (display orientation).
    strength      : 0 = off (exact pass-through) .. below 1; the weight of the previous output where
                    every gate is open.
    motion_aware  : close the gate where the picture moves.
    motion_compensation : warp the previous output along optical flow before blending (slower).
    preserve_motion_blur : never blend wide semi-transparent regions that are in motion.

    The filter also publishes two small maps of its last call for the refiner and the statistics:
    :attr:`motion_map` and :attr:`motion_blur_map`.
    """

    #: Long side of the analysis picture (motion, optical flow).  A 1080p frame is analysed at 1/4 scale.
    ANALYSIS_SIDE = 480
    #: Luma difference (fraction of full scale) up to which a pixel is static: sensor / codec noise.
    MOTION_LOW = 0.012
    #: Luma difference from which a pixel is fully "moving".
    MOTION_HIGH = 0.05
    #: Matte change up to which the full smoothing weight applies (flicker amplitude).
    CHANGE_LOW = 0.2
    #: Matte change from which nothing is blended (content appeared / disappeared).
    CHANGE_HIGH = 0.4
    #: Matte range that counts as semi-transparent for the motion-blur gate.
    SEMI_LOW, SEMI_HIGH = 0.05, 0.95
    #: Optical flow (analysis pixels) below which a frame / pixel is treated as not moving at all.
    STATIC_FLOW = 0.1

    def __init__(
        self,
        width: int,
        height: int,
        strength: float,
        *,
        motion_aware: bool = True,
        motion_compensation: bool = False,
        preserve_motion_blur: bool = True,
    ) -> None:
        if int(width) < 1 or int(height) < 1:
            raise ConfigError(f"Invalid frame size {width}x{height} for the temporal filter.")
        strength = float(strength)
        if not math.isfinite(strength) or strength < 0.0 or strength >= 1.0:
            raise ConfigError(
                f"Temporal smoothing strength must be in 0 <= strength < 1, got {strength!r}.",
                hint="1.0 would freeze the matte on the first frame; the configuration allows up to 0.95.",
            )
        self.width = int(width)
        self.height = int(height)
        self.strength = strength
        self.motion_aware = bool(motion_aware)
        self.motion_compensation = bool(motion_compensation)
        self.preserve_motion_blur = bool(preserve_motion_blur)

        factor = max(1, min(8, int(round(max(self.width, self.height) / float(self.ANALYSIS_SIDE)))))
        self._low_w = max(1, int(round(self.width / factor)))
        self._low_h = max(1, int(round(self.height / factor)))

        self._dis: Optional["cv2.DISOpticalFlow"] = None
        self._np_grid: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._prev_out: Optional[torch.Tensor] = None
        self._prev_luma: Optional[torch.Tensor] = None
        self._prev_semi: Optional[torch.Tensor] = None
        self._motion: Optional[torch.Tensor] = None
        self._motion_blur: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ public state
    @property
    def analysis_size(self) -> Tuple[int, int]:
        """``(height, width)`` of the analysis picture and of the published maps."""

        return self._low_h, self._low_w

    @property
    def motion_map(self) -> Optional[torch.Tensor]:
        """Image motion of the last call: ``[1, 1, h, w]``, 0 = static .. 1 = moving.

        ``None`` when the last frame had no predecessor (first frame, or first frame after
        :meth:`reset`): there is no motion evidence then, which is different from "static".
        """

        return self._motion

    @property
    def motion_blur_map(self) -> Optional[torch.Tensor]:
        """Wide semi-transparent matte inside moving picture areas: ``[1, 1, h, w]`` in 0..1.

        ``None`` under the same condition as :attr:`motion_map` and when ``preserve_motion_blur``
        is off.  The refiner relaxes its levels where this map is high.
        """

        return self._motion_blur

    def reset(self) -> None:
        """Forget the previous frame.  The next frame passes through unchanged."""

        self._prev_out = None
        self._prev_luma = None
        self._prev_semi = None
        self._motion = None
        self._motion_blur = None

    # ------------------------------------------------------------------ main entry
    def apply(self, alpha: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """Filter ONE frame: ``alpha`` ``[1, 1, H, W]``, ``rgb`` ``[1, 3, H, W]`` -> ``[1, 1, H, W]``.

        The first frame after a reset and every frame with ``strength == 0`` are returned as they
        are (the very same tensor); the analysis maps are still updated so the refiner can use
        them.  The filter keeps private copies of what it needs, so the caller may do whatever it
        likes with both the input and the result afterwards.
        """

        a = self._check(alpha, rgb)
        luma = self._analysis_luma(rgb)
        prev_luma, self._prev_luma = self._prev_luma, luma
        if prev_luma is not None and prev_luma.device != luma.device:
            # The pipeline moved to another device (e.g. CPU fallback after CUDA ran out of memory).
            prev_luma = prev_luma.to(luma.device)
            if self._prev_out is not None:
                self._prev_out = self._prev_out.to(a.device)
            if self._prev_semi is not None:
                self._prev_semi = self._prev_semi.to(a.device)

        semi = self._semi_fraction(a) if self.preserve_motion_blur else None
        prev_semi, self._prev_semi = self._prev_semi, semi

        if prev_luma is None:
            self._motion = None
            self._motion_blur = None
            self._prev_out = a.clone() if self.strength > 0.0 else None
            return a

        motion = self._motion_between(luma, prev_luma)
        self._motion = motion
        blur_zone: Optional[torch.Tensor] = None
        if semi is not None:
            wide = semi if prev_semi is None else torch.maximum(semi, prev_semi)
            wide = ((torch_ops.box_blur(wide, 1) - 0.35) / 0.3).clamp_(0.0, 1.0)
            # Motion anywhere in the 5x5 neighbourhood counts, and counts early: blur reaches further
            # than the band in which the picture changed (soft dilation with a gain).
            nearby_motion = (torch_ops.box_blur(motion, 2) * 4.0).clamp_(0.0, 1.0)
            blur_zone = wide * nearby_motion
        self._motion_blur = blur_zone

        prev = self._prev_out
        if self.strength <= 0.0 or prev is None:
            self._prev_out = a.clone() if self.strength > 0.0 else None
            return a

        gate: Optional[torch.Tensor] = None             # analysis resolution, None = fully open
        flow: Optional[torch.Tensor] = None
        if self.motion_compensation:
            flow, gate = self._estimate_flow(luma, prev_luma, motion)
        elif self.motion_aware:
            gate = 1.0 - motion
        if blur_zone is not None:
            gate = (1.0 - blur_zone) if gate is None else gate * (1.0 - blur_zone)

        size = (self.height, self.width)
        if flow is not None:
            assert gate is not None
            full = torch_ops.resize(torch.cat([flow, gate], dim=1), size, mode="bilinear")
            # A pixel whose matte did not change at all is stable, whatever the flow field claims for
            # it: it keeps its value.  Without this an imperfect flow next to a moving edge could
            # carry a trace of matte into pixels that are, and always were, exactly 0 (or 1).
            changed = prev != a
            prev = self._warp(prev, full[:, 0], full[:, 1])
            gate_full: Optional[torch.Tensor] = full[:, 2:3] * changed
        else:
            gate_full = None if gate is None else torch_ops.resize(gate, size, mode="bilinear")

        # A non-finite sample (e.g. a NaN from half-precision inference) must not live on in the
        # recursion: where the difference is not a number nothing is blended, so the state heals with
        # the next finite frame.
        diff = torch.nan_to_num(prev - a, nan=0.0, posinf=0.0, neginf=0.0)
        scale = self.strength / (self.CHANGE_HIGH - self.CHANGE_LOW)
        weight = ((self.CHANGE_HIGH - diff.abs()) * scale).clamp_(0.0, self.strength)
        if gate_full is not None:
            weight = weight * gate_full
        out = a + weight * diff
        self._prev_out = out.clone()
        return out

    # ------------------------------------------------------------------ analysis
    def _check(self, alpha: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        expected_a = (1, 1, self.height, self.width)
        expected_rgb = (1, 3, self.height, self.width)
        if not isinstance(alpha, torch.Tensor) or tuple(alpha.shape) != expected_a:
            got = tuple(alpha.shape) if isinstance(alpha, torch.Tensor) else type(alpha).__name__
            raise RefinementError(
                f"TemporalFilter.apply expects alpha of shape {expected_a}, got {got}.",
                hint="The filter is recursive: pass exactly one frame per call, in presentation order.",
            )
        if not isinstance(rgb, torch.Tensor) or tuple(rgb.shape) != expected_rgb:
            got = tuple(rgb.shape) if isinstance(rgb, torch.Tensor) else type(rgb).__name__
            raise RefinementError(f"TemporalFilter.apply expects rgb of shape {expected_rgb}, got {got}.")
        if rgb.device != alpha.device:
            raise RefinementError(f"alpha is on {alpha.device} but rgb is on {rgb.device}.")
        return alpha if alpha.dtype == torch.float32 else alpha.to(torch.float32)

    def _analysis_luma(self, rgb: torch.Tensor) -> torch.Tensor:
        """Small, slightly blurred luma: averages away noise and makes the gates spatially smooth."""

        low = torch_ops.resize(rgb.to(torch.float32), (self._low_h, self._low_w), mode="area")
        return torch_ops.gaussian_blur(torch_ops.luma(low), 1.0)

    def _motion_from_difference(self, difference: torch.Tensor) -> torch.Tensor:
        """Normalise an absolute luma difference to 0 (static) .. 1 (moving)."""

        return ((difference - self.MOTION_LOW) / (self.MOTION_HIGH - self.MOTION_LOW)).clamp_(0.0, 1.0)

    def _motion_between(self, luma: torch.Tensor, prev_luma: torch.Tensor) -> torch.Tensor:
        motion = self._motion_from_difference((luma - prev_luma).abs())
        # Grow by one analysis pixel and soften: the matte edge of a moving object is wider than the
        # band in which the luma actually changed.
        return torch_ops.gaussian_blur(torch_ops.max_filter(motion, 1), 1.0).clamp_(0.0, 1.0)

    def _semi_fraction(self, alpha: torch.Tensor) -> torch.Tensor:
        """Fraction of semi-transparent pixels per analysis cell.

        Averaged over 3x3 cells this separates wide semi-transparent regions (motion blur, hair:
        close to 1) from the ordinary anti-aliased outline of a subject (a thin band: about 0.25).
        """

        semi = ((alpha > self.SEMI_LOW) & (alpha < self.SEMI_HIGH)).to(torch.float32)
        return torch_ops.resize(semi, (self._low_h, self._low_w), mode="area")

    # ------------------------------------------------------------------ motion compensation
    def _flow_estimator(self) -> "cv2.DISOpticalFlow":
        if self._dis is None:
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            if min(self._low_h, self._low_w) >= 64:
                # The preset stops two pyramid levels above full resolution, which biases the flow by
                # about 10 %; one level finer brings the error down to a few hundredths of a pixel.
                dis.setFinestScale(1)
            self._dis = dis
        return self._dis

    def _estimate_flow(
        self, luma: torch.Tensor, prev_luma: torch.Tensor, motion: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Backward flow (current -> previous) and its reliability, both at analysis resolution.

        Returns ``(flow [1, 2, h, w] in analysis pixels or None, gate [1, 1, h, w])``.  ``None``
        means "do not warp": the scene is static or the flow could not be computed, and the gate is
        then the ordinary image-motion gate.
        """

        fallback_gate = 1.0 - motion
        cur = luma[0, 0].detach().cpu().numpy()
        prev = prev_luma[0, 0].detach().cpu().numpy()
        cur_u8 = np.clip(cur * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        prev_u8 = np.clip(prev * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        try:
            estimator = self._flow_estimator()
            backward = estimator.calc(cur_u8, prev_u8, None)       # prev(x + backward(x)) ~ cur(x)
            forward = estimator.calc(prev_u8, cur_u8, None)        # cur(x + forward(x)) ~ prev(x)
        except cv2.error as exc:                                   # e.g. a frame too small for the pyramid
            log.debug("optical flow failed (%s); frame is filtered without motion compensation", exc)
            return None, fallback_gate
        if not (np.isfinite(backward).all() and np.isfinite(forward).all()):
            return None, fallback_gate
        magnitude = np.sqrt(backward[..., 0] ** 2 + backward[..., 1] ** 2)
        if float(magnitude.max()) < self.STATIC_FLOW:
            return None, fallback_gate

        if self._np_grid is None:
            xs, ys = np.meshgrid(
                np.arange(self._low_w, dtype=np.float32), np.arange(self._low_h, dtype=np.float32)
            )
            self._np_grid = (xs, ys)
        xs, ys = self._np_grid
        map_x = xs + backward[..., 0]
        map_y = ys + backward[..., 1]
        inside = (map_x >= -0.5) & (map_x <= self._low_w - 0.5) & (map_y >= -0.5) & (map_y <= self._low_h - 0.5)

        # Forward/backward consistency: following the backward flow and then the forward flow must
        # lead back to the start.  It fails at occlusions and wherever the flow is a guess.
        forward_there = cv2.remap(forward, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        round_trip = backward + forward_there
        error = np.sqrt(round_trip[..., 0] ** 2 + round_trip[..., 1] ** 2)
        tolerance = 0.5 + 0.1 * magnitude
        consistent = np.clip(2.0 - error / tolerance, 0.0, 1.0)

        # Residual: what the picture still differs by after the warp.
        prev_warped = cv2.remap(prev, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        residual = np.abs(cur - prev_warped)
        explained = 1.0 - np.clip((residual - self.MOTION_LOW) / (self.MOTION_HIGH - self.MOTION_LOW), 0.0, 1.0)

        reliability = (consistent * explained * inside).astype(np.float32)
        reliability = cv2.GaussianBlur(cv2.erode(reliability, np.ones((3, 3), np.uint8)), (0, 0), 1.0)

        device = luma.device
        flow_t = torch.from_numpy(np.ascontiguousarray(backward.transpose(2, 0, 1))).to(device).unsqueeze(0)
        gate_t = torch.from_numpy(reliability).to(device).clamp_(0.0, 1.0)[None, None]
        return flow_t, gate_t

    def _warp(self, prev: torch.Tensor, flow_x: torch.Tensor, flow_y: torch.Tensor) -> torch.Tensor:
        """Sample the previous output at ``x + flow`` (flow in analysis pixels, ``[1, H, W]``)."""

        device, dtype = prev.device, prev.dtype
        height, width = self.height, self.width
        # Pixel centres in grid_sample's normalised coordinates (align_corners=False).
        base_x = ((torch.arange(width, device=device, dtype=dtype) * 2.0 + 1.0) / width - 1.0).view(1, 1, width)
        base_y = ((torch.arange(height, device=device, dtype=dtype) * 2.0 + 1.0) / height - 1.0).view(1, height, 1)
        grid = torch.stack(
            (base_x + flow_x * (2.0 / self._low_w), base_y + flow_y * (2.0 / self._low_h)), dim=-1
        )
        sampled = F.grid_sample(prev, grid, mode="bilinear", padding_mode="border", align_corners=False)
        # Bilinear weights sum to one only up to rounding: restore exact opacity (zeros are exact).
        sampled = torch.where(sampled > 1.0 - 1e-6, torch.ones((), dtype=dtype, device=device), sampled)
        # Pixels that do not move keep the previous value untouched instead of a resampled copy, so
        # static parts of the frame cannot blur or drift through repeated interpolation.
        moving = (flow_x * flow_x + flow_y * flow_y) > self.STATIC_FLOW ** 2
        return torch.where(moving.unsqueeze(1), sampled, prev)
