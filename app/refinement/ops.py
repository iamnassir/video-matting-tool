"""Matte refinement operators: pure functions on ``[B, 1, H, W]`` float32 alpha tensors.

Design rules shared by every operator in this module
----------------------------------------------------
* **Any device.**  Tensors are created on the input's device; the only CPU work is the connected
  component labelling of the clean-up operators, which runs on a *reduced* binary map so that a
  small array is transferred instead of the frame.
* **Inputs are never modified.**  An operator that has nothing to do returns its input tensor
  itself (the convention of :mod:`app.utils.torch_ops`), so callers must not write into results.
* **Exact plateaus.**  The compositor copies source pixels bit-exactly where the matte is exactly
  1.0 and writes one constant colour where it is exactly 0.0 (invariants I4 / I5 of
  ``docs/ARCHITECTURE.md``).  Floating point filters do not keep these values by themselves (the
  weights of a Gaussian do not sum to exactly one), therefore every operator here restores the
  exact value wherever its whole support is fully opaque or fully transparent.
* **Conservative.**  Nothing here invents matte: the clean-up operators change a region only when it
  is a complete, small connected component / enclosed hole, and :func:`detail_map` exists so that the
  spatial operators can leave hair and other thin structure alone.

Pixel quantities are real pixels of the given tensor; scaling from the 1080-line reference used by
``RefinementParams`` is the caller's job (``RefinementParams.px``).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..core.errors import ConfigError, VmtError
from ..utils import torch_ops

__all__ = [
    "RefinementError",
    "levels",
    "erode",
    "dilate",
    "feather",
    "smooth",
    "guided_filter",
    "mask_to_alpha",
    "remove_small_components",
    "fill_small_holes",
    "detail_map",
    "detail_map_support",
    "smooth_support",
]

#: Scalar or per-pixel (broadcastable to the matte) level.
Level = Union[float, torch.Tensor]

#: Smallest regularisation the float32 covariance solve of the guided filter can resolve.
_MIN_GUIDED_EPS = 1e-6
#: Share of the mask's local variance the picture must explain before the guided result is trusted
#: at all / trusted completely (see mask_to_alpha).  Pure chance explains a few percent.
_CONFIDENCE_LOW, _CONFIDENCE_HIGH = 0.15, 0.40


class RefinementError(VmtError):
    """A refinement operator was called with data that violates its contract (shape, size, type).

    ``app.core.errors`` has no class for internal contract violations between engine modules, so the
    refinement package defines this one; it is a :class:`VmtError` and therefore carries a code, a
    message and a hint like every other error of the application.
    """

    code = "refinement_error"
    default_hint = "This is a programming error in the caller, not a problem with the video."


# --------------------------------------------------------------------------- validation helpers


def _as_alpha(alpha: torch.Tensor, name: str = "alpha") -> torch.Tensor:
    """Validate a ``[B, 1, H, W]`` matte and return it as float32 (the input itself when possible)."""

    if not isinstance(alpha, torch.Tensor):
        raise RefinementError(f"{name} must be a torch.Tensor, got {type(alpha).__name__}.")
    if alpha.dim() != 4 or alpha.shape[1] != 1:
        raise RefinementError(
            f"{name} must have shape [B, 1, H, W], got {tuple(alpha.shape)}.",
            hint="Add the batch / channel dimensions (e.g. matte[None, None]) before calling the refinement operators.",
        )
    if alpha.shape[-1] < 1 or alpha.shape[-2] < 1:
        raise RefinementError(f"{name} has an empty spatial size {tuple(alpha.shape)}.")
    return alpha if alpha.dtype == torch.float32 else alpha.to(torch.float32)


def _check_radius(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0.0:
        raise ConfigError(f"{name} must be a finite number >= 0, got {value!r}.")
    return v


# --------------------------------------------------------------------------- levels


def levels(alpha: torch.Tensor, black_point: Level, white_point: Level) -> torch.Tensor:
    """Remap ``alpha`` so that ``black_point`` becomes 0 and ``white_point`` becomes 1.

    ``clamp((alpha - black) / (white - black), 0, 1)``.  Values ``<= black`` come out as exactly 0.0
    and values ``>= white`` as exactly 1.0: numerator and denominator are computed in the same
    float32 arithmetic and rounding is monotonic, so ``alpha >= white`` implies a quotient ``>= 1``
    (and ``alpha <= black`` a quotient ``<= 0``) before the clamp.  The compositor relies on this.

    ``black_point`` / ``white_point`` may be floats or tensors broadcastable to ``alpha`` (per-pixel
    levels, used to relax the levels inside motion blur).
    """

    a = _as_alpha(alpha)
    per_pixel = isinstance(black_point, torch.Tensor) or isinstance(white_point, torch.Tensor)
    if not per_pixel:
        bp, wp = float(black_point), float(white_point)
        if not (math.isfinite(bp) and math.isfinite(wp)) or bp < 0.0 or wp > 1.0 or wp <= bp:
            raise ConfigError(
                f"Invalid levels: black_point={black_point!r}, white_point={white_point!r}.",
                hint="Use 0 <= black_point < white_point <= 1 (e.g. 0.015 and 0.985).",
            )
        if bp == 0.0 and wp == 1.0:
            return a.clamp(0.0, 1.0)
    black = torch.as_tensor(black_point, dtype=a.dtype, device=a.device)
    white = torch.as_tensor(white_point, dtype=a.dtype, device=a.device)
    span = white - black
    if per_pixel:
        # A degenerate per-pixel pair must not produce inf / nan; it degrades to a hard threshold.
        span = span.clamp_min(1e-6)
    return ((a - black) / span).clamp_(0.0, 1.0)


# --------------------------------------------------------------------------- morphology


def erode(alpha: torch.Tensor, radius_px: float) -> torch.Tensor:
    """Grey-scale erosion (shrinks the matte by ``radius_px``; fractional radii are interpolated)."""

    a = _as_alpha(alpha)
    r = _check_radius(radius_px, "erode radius")
    return torch_ops.min_filter(a, r) if r > 0.0 else a


def dilate(alpha: torch.Tensor, radius_px: float) -> torch.Tensor:
    """Grey-scale dilation (grows the matte by ``radius_px``; fractional radii are interpolated)."""

    a = _as_alpha(alpha)
    r = _check_radius(radius_px, "dilate radius")
    return torch_ops.max_filter(a, r) if r > 0.0 else a


# --------------------------------------------------------------------------- smoothing


def smooth(alpha: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Gaussian smoothing of the matte with standard deviation ``sigma_px``.

    Flat regions are unchanged.  Fully transparent neighbourhoods stay exactly 0 because a sum of
    zeros is zero; fully opaque neighbourhoods are detected explicitly (no sample below one inside
    the kernel support) and set to exactly 1, because the float32 kernel weights do not sum to
    exactly one and the blur alone would return 0.99999994 there.
    """

    a = _as_alpha(alpha)
    sigma = _check_radius(sigma_px, "smoothing sigma")
    if sigma <= 0.0:
        return a
    blurred = torch_ops.gaussian_blur(a, sigma).clamp_(0.0, 1.0)
    radius = smooth_support(sigma)                    # exactly the support of the blur above
    below_one = (a < 1.0).to(a.dtype)
    opaque = torch_ops.box_blur(below_one, radius) == 0.0
    return torch.where(opaque, torch.ones((), dtype=a.dtype, device=a.device), blurred)


def feather(alpha: torch.Tensor, radius_px: float) -> torch.Tensor:
    """Soften the matte edge symmetrically over about ``radius_px`` on each side.

    Implemented as a Gaussian with ``sigma = radius / 2``: 95 % of the transition of a hard edge then
    lies within ``+- radius`` of the original edge position, and the 50 % contour does not move
    (unlike erode / dilate, feathering neither shrinks nor grows the subject).
    """

    r = _check_radius(radius_px, "feather radius")
    if r <= 0.0:
        return _as_alpha(alpha)
    return smooth(alpha, r / 2.0)


# --------------------------------------------------------------------------- guided filter


def _guided(
    guide_rgb: torch.Tensor,
    src: torch.Tensor,
    radius_px: float,
    eps: float,
    subsample: Optional[int],
    with_confidence: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fast colour guided filter.  Returns ``(q, confidence)``; see :func:`guided_filter`.

    ``confidence`` (only when requested, ``[B, 1, H, W]`` in 0..1) says how much of the local variation
    of ``src`` the picture explains: the coefficient of determination of the window regressions,
    taken from the best window that contains the pixel.
    """

    p = _as_alpha(src, "src")
    if not isinstance(guide_rgb, torch.Tensor) or guide_rgb.dim() != 4 or guide_rgb.shape[1] != 3:
        shape = tuple(guide_rgb.shape) if isinstance(guide_rgb, torch.Tensor) else type(guide_rgb).__name__
        raise RefinementError(f"guide_rgb must have shape [B, 3, H, W], got {shape}.")
    if guide_rgb.shape[0] != p.shape[0] or guide_rgb.shape[-2:] != p.shape[-2:]:
        raise RefinementError(
            f"guide_rgb {tuple(guide_rgb.shape)} and src {tuple(p.shape)} must have the same batch and frame size."
        )
    if guide_rgb.device != p.device:
        raise RefinementError(f"guide_rgb is on {guide_rgb.device} but src is on {p.device}.")
    radius = _check_radius(radius_px, "guided filter radius")
    eps_f = float(eps)
    if not math.isfinite(eps_f) or eps_f <= 0.0:
        raise ConfigError(f"guided filter eps must be > 0, got {eps!r}.", hint="Typical values are 1e-4 .. 1e-2.")
    if subsample is not None and int(subsample) < 1:
        raise ConfigError(f"guided filter subsample must be >= 1, got {subsample!r}.")
    if radius <= 0.0 or p.shape[0] == 0:
        q = p.clamp(0.0, 1.0)
        return q, (torch.zeros_like(q) if with_confidence else None)
    eps_f = max(eps_f, _MIN_GUIDED_EPS)
    guide = guide_rgb if guide_rgb.dtype == torch.float32 else guide_rgb.to(torch.float32)

    height, width = p.shape[-2:]
    if subsample is None:
        # As coarse as possible while the reduced window keeps a radius of 2 (up to a factor of 4);
        # beyond that only as far as needed to keep the reduced radius at 8 or less, which bounds the
        # cost of very large radii.
        step = max(1, min(4, int(radius // 2)), int(math.ceil(radius / 8.0)))
    else:
        step = int(subsample)
    step = max(1, min(step, height, width))
    low_h, low_w = max(1, int(round(height / step))), max(1, int(round(width / step)))
    low_radius = max(1, int(round(radius / step)))

    if step > 1:
        guide_low = torch_ops.resize(guide, (low_h, low_w), mode="area")
        p_low = torch_ops.resize(p, (low_h, low_w), mode="area")
    else:
        guide_low, p_low = guide, p

    # All first and second order moments in one box filter call.
    r_, g_, b_ = guide_low[:, 0:1], guide_low[:, 1:2], guide_low[:, 2:3]
    planes = [guide_low, p_low, guide_low * p_low, r_ * r_, r_ * g_, r_ * b_, g_ * g_, g_ * b_, b_ * b_]
    if with_confidence:
        planes.append(p_low * p_low)
    moments = torch_ops.box_blur(torch.cat(planes, dim=1), low_radius)
    mean_i, mean_p = moments[:, 0:3], moments[:, 3:4]
    cov_ip = moments[:, 4:7] - mean_i * mean_p
    mr, mg, mb = mean_i[:, 0:1], mean_i[:, 1:2], mean_i[:, 2:3]
    var_rr = moments[:, 7:8] - mr * mr + eps_f
    var_rg = moments[:, 8:9] - mr * mg
    var_rb = moments[:, 9:10] - mr * mb
    var_gg = moments[:, 10:11] - mg * mg + eps_f
    var_gb = moments[:, 11:12] - mg * mb
    var_bb = moments[:, 12:13] - mb * mb + eps_f

    # Closed-form solve of the symmetric 3x3 system through its cofactors: far cheaper than a
    # batched LU on every pixel and identical on CPU and GPU.
    inv_rr = var_gg * var_bb - var_gb * var_gb
    inv_rg = var_gb * var_rb - var_rg * var_bb
    inv_rb = var_rg * var_gb - var_gg * var_rb
    inv_gg = var_rr * var_bb - var_rb * var_rb
    inv_gb = var_rb * var_rg - var_rr * var_gb
    inv_bb = var_rr * var_gg - var_rg * var_rg
    # eps > 0 makes the matrix positive definite; the floor only guards float32 rounding.
    det = (var_rr * inv_rr + var_rg * inv_rg + var_rb * inv_rb).clamp_min(eps_f ** 3 * 1e-3)
    cr, cg, cb = cov_ip[:, 0:1], cov_ip[:, 1:2], cov_ip[:, 2:3]
    a = torch.cat(
        [
            inv_rr * cr + inv_rg * cg + inv_rb * cb,
            inv_rg * cr + inv_gg * cg + inv_gb * cb,
            inv_rb * cr + inv_gb * cg + inv_bb * cb,
        ],
        dim=1,
    ) / det
    b = mean_p - (a * mean_i).sum(dim=1, keepdim=True)

    coeff = torch_ops.box_blur(torch.cat([a, b], dim=1), low_radius)
    if step > 1:
        coeff = torch_ops.resize(coeff, (height, width), mode="bilinear")
    q = ((coeff[:, 0:3] * guide).sum(dim=1, keepdim=True) + coeff[:, 3:4]).clamp_(0.0, 1.0)

    # Exact plateaus.  A pixel's output depends on src within two cascaded box windows plus the
    # bilinear up-sampling footprint; where no sample in that support is below one (above zero)
    # the result is the constant itself.
    flags = torch.cat([p < 1.0, p > 0.0], dim=1).to(p.dtype)
    if step > 1:
        flags = torch_ops.resize(flags, (low_h, low_w), mode="area")       # > 0 iff any pixel of the cell is flagged
    near = torch_ops.box_blur(flags, 2 * low_radius + 2) > 0.0
    if step > 1:
        near = F.interpolate(near.to(p.dtype), size=(height, width), mode="nearest") > 0.5
    one = torch.ones((), dtype=p.dtype, device=p.device)
    zero = torch.zeros((), dtype=p.dtype, device=p.device)
    q = torch.where(near[:, 0:1], q, one)
    q = torch.where(near[:, 1:2], q, zero)
    if not with_confidence:
        return q, None

    # Coefficient of determination of every window: the share of the variance of src that the affine
    # colour model reproduces (a . cov(I, p) / var(p)).  A pixel takes the best of the windows it lies
    # in, so an image edge a few pixels beside the mask edge still counts.
    var_p = (moments[:, 13:14] - mean_p * mean_p).clamp_min(0.0)
    explained = (a * cov_ip).sum(dim=1, keepdim=True).clamp_min(0.0)
    r_squared = (explained / (var_p + 1e-6)).clamp_(0.0, 1.0)
    confidence = ((r_squared - _CONFIDENCE_LOW) / (_CONFIDENCE_HIGH - _CONFIDENCE_LOW)).clamp_(0.0, 1.0)
    confidence = torch_ops.box_blur(torch_ops.max_filter(confidence, low_radius), 1)
    if step > 1:
        confidence = torch_ops.resize(confidence, (height, width), mode="bilinear")
    return q, confidence.clamp_(0.0, 1.0)


def guided_filter(
    guide_rgb: torch.Tensor,
    src: torch.Tensor,
    radius_px: float,
    eps: float,
    subsample: Optional[int] = None,
) -> torch.Tensor:
    """Edge-aware filtering of ``src`` (matte or hard mask) guided by the colour image.

    Fast guided filter (He, Sun, Tang: "Guided Image Filtering", 2013; He, Sun: "Fast Guided
    Filter", 2015) with a colour guide: inside every window the output is modelled as an affine
    function of the guide colour, ``q = a . I + b``, where ``a`` solves the 3x3 system
    ``(cov(I) + eps * Id) a = cov(I, p)``.  The coefficients are computed on a copy reduced by
    ``subsample`` and up-sampled, the final product uses the full resolution guide, so the matte
    edge snaps to the image edge with full resolution accuracy.

    Parameters
    ----------
    guide_rgb : ``[B, 3, H, W]`` R'G'B' in 0..1.
    src       : ``[B, 1, H, W]`` matte / mask in 0..1.
    radius_px : window radius in pixels of the full resolution frame.
    eps       : regularisation (squared colour contrast below which the image counts as flat).
    subsample : reduction factor for the coefficient computation.  ``None`` chooses it automatically:
                the largest factor up to 4 that keeps the reduced window radius at 2 or more, and a
                larger one for very large radii so that the reduced radius never exceeds 8.

    Guarantees: values are clamped to 0..1, and wherever ``src`` is exactly 1 (exactly 0) over the
    whole support of the filter the output is exactly 1.0 (0.0) - mathematically the filter returns
    the constant there, but only up to rounding, and the compositor needs the exact value.

    Note the well-known property of the filter: where the picture shows no edge, ``src`` is simply
    box-filtered twice, i.e. a hard edge becomes a ramp about ``4 * radius`` wide.
    :func:`mask_to_alpha` builds on this function and avoids that.
    """

    return _guided(guide_rgb, src, radius_px, eps, subsample, with_confidence=False)[0]


def mask_to_alpha(
    guide_rgb: torch.Tensor,
    mask: torch.Tensor,
    radius_px: float,
    eps: float,
    subsample: Optional[int] = None,
    *,
    soften_px: float = 0.0,
) -> torch.Tensor:
    """Turn a segmentation mask into a soft, image-aligned matte without blurring what it cannot align.

    The guided filter aligns the matte with the picture where the picture has an edge near the mask
    edge.  Where it has none (dark hair in front of a dark wall, a shadowed shoulder) the guided
    filter degenerates into a wide blur of the mask - a halo tens of pixels wide at 1080p.  This
    function therefore trusts the guided result only as far as the picture explains the mask
    (the local coefficient of determination of the filter's own colour model) and otherwise keeps
    the mask, merely anti-aliased with a small Gaussian of sigma ``soften_px``::

        base  = smooth(mask, soften_px)
        alpha = base + confidence * (guided_filter(...) - base)

    ``soften_px = 0`` keeps the source itself as the fallback, which is what an already soft matte
    needs.  Plateaus stay exact exactly as in :func:`guided_filter`.
    """

    q, confidence = _guided(guide_rgb, mask, radius_px, eps, subsample, with_confidence=True)
    assert confidence is not None
    base = _as_alpha(mask, "mask").clamp(0.0, 1.0)
    if _check_radius(soften_px, "mask softening") > 0.0:
        base = smooth(base, soften_px)
    # lerp is exact at both ends: the fallback where the picture says nothing, the guided result
    # (with its exact plateaus) where it explains the mask completely.
    return torch.lerp(base, q, confidence).clamp_(0.0, 1.0)


# --------------------------------------------------------------------------- connected components


def _reduction_block(height: int, width: int, max_side: int) -> int:
    """Integer block size that brings the long side of the frame to ``max_side`` or less."""

    if max_side < 1:
        raise ConfigError(f"max_side must be >= 1, got {max_side!r}.")
    return max(1, int(math.ceil(max(height, width) / float(max_side))))


def _block_counts(mask: torch.Tensor, block: int) -> torch.Tensor:
    """Number of set pixels in every ``block x block`` cell: ``[B, C, ceil(H/b), ceil(W/b)]``.

    One pooling pass yields both things the clean-up needs: ``count > 0`` is the max-pooled binary
    map (a single pixel keeps its cell alive, so tiny specks and tiny holes stay visible after the
    reduction) and the count itself gives component areas in exact full-resolution pixels.
    """

    x = mask.to(torch.float32)
    if block <= 1:
        return x
    h, w = x.shape[-2:]
    pad_h, pad_w = (-h) % block, (-w) % block
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))           # zeros: the padding never adds to a count
    return torch.round(F.avg_pool2d(x, block) * float(block * block))


def _counts_to_numpy(counts: torch.Tensor, block: int) -> np.ndarray:
    dtype = torch.uint8 if block * block <= 255 else torch.int32
    return counts.to(dtype).cpu().numpy()


def _block_upsample(mask: torch.Tensor, block: int, height: int, width: int) -> torch.Tensor:
    """Inverse of the block reduction for a decision map: every cell is replicated ``block`` times."""

    if block > 1:
        mask = mask.repeat_interleave(block, dim=-2).repeat_interleave(block, dim=-1)
    return mask[..., :height, :width]


def _fringe_kernel(block: int) -> np.ndarray:
    """Structuring element that extends a decision by about two full-resolution pixels."""

    margin = 2 if block == 1 else 1
    return np.ones((2 * margin + 1, 2 * margin + 1), dtype=np.uint8)


#: Above this many candidate regions the per-region sums are replaced by one histogram pass.
_MAX_REGION_LOOP = 512


def _region_sums(labels: np.ndarray, stats: np.ndarray, candidates: np.ndarray, *weights: np.ndarray) -> List[np.ndarray]:
    """Sum every ``weights`` map over each candidate label; entries of other labels are 0.

    Equivalent to ``np.bincount(labels.ravel(), weights=w.ravel())`` restricted to the candidate
    labels, but it only reads the bounding box of each candidate (known from the labelling
    statistics) instead of the whole map: a few microseconds for the handful of small regions a
    frame normally has, against milliseconds for the histogram.  The sums are integers, so both
    ways give exactly the same numbers.
    """

    n_labels = stats.shape[0]
    sums = [np.zeros(n_labels, dtype=np.float64) for _ in weights]
    if candidates.size == 0:
        return sums
    if candidates.size > _MAX_REGION_LOOP:
        flat = labels.ravel()
        keep = np.zeros(n_labels, dtype=bool)
        keep[candidates] = True
        for out, weight in zip(sums, weights):
            out[:] = np.bincount(flat, weights=weight.ravel().astype(np.float64), minlength=n_labels)
            out[~keep] = 0.0
        return sums
    for label in candidates.tolist():
        left, top = int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP])
        right, bottom = left + int(stats[label, cv2.CC_STAT_WIDTH]), top + int(stats[label, cv2.CC_STAT_HEIGHT])
        inside = labels[top:bottom, left:right] == label
        for out, weight in zip(sums, weights):
            out[label] = float(weight[top:bottom, left:right][inside].sum(dtype=np.int64))
    return sums


def remove_small_components(
    alpha: torch.Tensor,
    min_area_px: float,
    threshold: float = 0.05,
    *,
    max_side: int = 960,
) -> Tuple[torch.Tensor, int]:
    """Zero every connected component of the matte whose area is below ``min_area_px``.

    A component is a connected region (8-neighbourhood) of ``alpha > threshold``.  The threshold is
    deliberately low so that the soft fringe of a speck belongs to the speck and disappears with
    it; the faint ring that remains below the threshold right next to a removed component is
    cleared as well (it cannot belong to anything else: cells of kept components are never
    touched).  The area is the number of full-resolution pixels above the threshold.

    The largest component of a frame is never removed, however small it is, and kept components
    are returned bit-identical (the matte is multiplied by a 0 / 1 map).

    Labelling runs on the CPU with OpenCV on a block-reduced map whose long side is at most
    ``max_side``; two components closer together than one block are treated as one, which errs on
    the side of keeping.

    Returns ``(matte, number of removed components)``.
    """

    a = _as_alpha(alpha)
    min_area = float(min_area_px)
    if not math.isfinite(min_area) or min_area <= 0.0 or a.shape[0] == 0:
        return a, 0
    batch, _, height, width = a.shape
    block = _reduction_block(height, width, max_side)
    counts = _counts_to_numpy(_block_counts(a > float(threshold), block), block)[:, 0]
    kernel = _fringe_kernel(block)

    remove: Optional[np.ndarray] = None
    removed = 0
    for i in range(batch):
        cell_counts = counts[i]
        support = (cell_counts > 0).astype(np.uint8)
        if not support.any():
            continue
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8, ltype=cv2.CV_32S)
        if n_labels <= 2:                                   # background plus at most one component
            continue
        # Every cell of a component holds at least one pixel, so a component with at least
        # ``min_area`` cells cannot be small: exact pixel areas are only needed for the others.
        cells = stats[:, cv2.CC_STAT_AREA]
        candidates = np.flatnonzero(cells < min_area)
        candidates = candidates[candidates != 0]            # label 0 is the background of the map
        if candidates.size == 0:
            continue
        (areas,) = _region_sums(labels, stats, candidates, cell_counts)
        small = np.zeros(n_labels, dtype=bool)
        small[candidates] = areas[candidates] < min_area
        if candidates.size == n_labels - 1:
            # No component is large by its cell count: the largest one (by pixels) is never removed.
            # (When a large one exists, it is the largest or smaller than the largest, and neither
            # of them is "small", so there is nothing to protect.)
            small[1 + int(np.argmax(areas[1:]))] = False
        if not small.any():
            continue
        gone = small[labels]
        kept_support = support.astype(bool) & ~gone
        fringe = cv2.dilate(gone.astype(np.uint8), kernel).astype(bool) & ~kept_support
        if remove is None:
            remove = np.zeros(counts.shape, dtype=bool)
        remove[i] = fringe
        removed += int(small.sum())

    if remove is None:
        return a, 0
    keep = torch.from_numpy(~remove).to(device=a.device, dtype=a.dtype).unsqueeze(1)
    return a * _block_upsample(keep, block, height, width), removed


def fill_small_holes(
    alpha: torch.Tensor,
    max_area_px: float,
    threshold: float = 0.5,
    *,
    rim_threshold: float = 0.9,
    max_side: int = 960,
) -> Tuple[torch.Tensor, int]:
    """Fill small holes that are completely enclosed by solid matte with 1.

    A hole is a connected region (4-neighbourhood, the topological dual of the 8-neighbourhood used
    for the subject) of ``alpha < rim_threshold`` that

    * does not touch the frame border (a bay that is open to the border is background, not a hole),
    * contains at least one pixel below ``threshold`` (a real hole, not a slightly soft patch),
    * and covers at most ``max_area_px`` full-resolution pixels.

    Deriving the region from the high ``rim_threshold`` has two purposes.  The soft fringe of the
    hole is part of the region and is filled with it, so no semi-transparent ring is left behind.
    And a hole only qualifies when a closed rim of nearly opaque matte surrounds it: gaps between
    hair strands or inside motion blur are bounded by thin or semi-transparent matte, so they are
    never "enclosed" and are left alone.

    Labelling runs on the CPU with OpenCV on a block-reduced map (long side at most ``max_side``)
    in which a cell is background as soon as one of its pixels is (min-pooling of the subject), so
    a foreground wall thinner than a block does not count as enclosing.

    Returns ``(matte, number of filled holes)``; everything outside the filled regions is returned
    bit-identical.
    """

    a = _as_alpha(alpha)
    max_area = float(max_area_px)
    if not math.isfinite(max_area) or max_area <= 0.0 or a.shape[0] == 0:
        return a, 0
    batch, _, height, width = a.shape
    rim = max(float(rim_threshold), float(threshold))
    block = _reduction_block(height, width, max_side)
    counts = _counts_to_numpy(_block_counts(torch.cat([a < rim, a < float(threshold)], dim=1), block), block)
    kernel = _fringe_kernel(block)
    low_h, low_w = counts.shape[-2:]

    fill: Optional[np.ndarray] = None
    filled = 0
    for i in range(batch):
        basin_counts, core_counts = counts[i, 0], counts[i, 1]
        if not core_counts.any():
            continue
        background = (basin_counts > 0).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=4, ltype=cv2.CV_32S)
        if n_labels <= 1:
            continue
        left, top = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
        right, bottom = left + stats[:, cv2.CC_STAT_WIDTH], top + stats[:, cv2.CC_STAT_HEIGHT]
        enclosed = ~((left == 0) | (top == 0) | (right == low_w) | (bottom == low_h))
        enclosed[0] = False                                 # label 0 is the solid matte, not a region
        if not enclosed.any():
            continue                                        # the usual frame: only the outer background
        # Every cell of a region holds at least one background pixel, so a region with more cells
        # than ``max_area`` cannot qualify: exact pixel areas are only needed for the others.
        candidates = np.flatnonzero(enclosed & (stats[:, cv2.CC_STAT_AREA] <= max_area))
        if candidates.size == 0:
            continue
        basin_area, core_area = _region_sums(labels, stats, candidates, basin_counts, core_counts)
        hole = np.zeros(n_labels, dtype=bool)
        hole[candidates] = (core_area[candidates] > 0) & (basin_area[candidates] <= max_area)
        if not hole.any():
            continue
        region = hole[labels]
        # Cells next to a hole that are not background themselves hold only values >= rim: raising
        # them to one removes the last faint ring without touching any other region.
        ring = cv2.dilate(region.astype(np.uint8), kernel).astype(bool) & ~background.astype(bool)
        if fill is None:
            fill = np.zeros((batch, low_h, low_w), dtype=bool)
        fill[i] = region | ring
        filled += int(hole.sum())

    if fill is None:
        return a, 0
    mask = _block_upsample(torch.from_numpy(fill).to(a.device).unsqueeze(1), block, height, width)
    return torch.where(mask, torch.ones((), dtype=a.dtype, device=a.device), a), filled


# --------------------------------------------------------------------------- detail


def _detail_geometry(radius_px: float, spread_px: Optional[float]) -> Tuple[int, float]:
    """``(integer top-hat radius, sigma of the spreading Gaussian)`` for :func:`detail_map`."""

    radius = max(1, int(math.ceil(_check_radius(radius_px, "detail radius"))))
    spread = float(radius) if spread_px is None else max(_check_radius(spread_px, "detail spread"), 1.0)
    # A one pixel wide line of full response, blurred with sigma and multiplied by the gain used in
    # detail_map, stays >= 1 up to a distance of 1.48 sigma: that distance is the requested spread.
    return radius, spread / 1.48


def detail_map_support(radius_px: float = 2.0, spread_px: Optional[float] = None) -> int:
    """Distance in pixels beyond which a matte sample cannot influence :func:`detail_map`.

    Callers that run the operators on a crop need it to choose a margin that makes the cropped
    result identical to the full-frame one.
    """

    radius, sigma = _detail_geometry(radius_px, spread_px)
    return 2 * radius + (torch_ops.gaussian_kernel1d(sigma).numel() - 1) // 2


def smooth_support(sigma_px: float) -> int:
    """Distance in pixels beyond which a matte sample cannot influence :func:`smooth` (0 when off)."""

    sigma = _check_radius(sigma_px, "smoothing sigma")
    if sigma <= 0.0:
        return 0
    return (torch_ops.gaussian_kernel1d(sigma).numel() - 1) // 2


def detail_map(alpha: torch.Tensor, radius_px: float = 2.0, spread_px: Optional[float] = None) -> torch.Tensor:
    """Map in 0..1 that is high on fine matte structure (hair strands, thin gaps, thin tips).

    The spatial operators (erode, dilate, smoothing, feather) cannot tell a hair strand from the
    outline of a shoulder; this map can, so the refiner blends their result back to the original
    matte where it is high.

    Method: morphological top-hats.  ``alpha - opening(alpha)`` is what an opening with a
    ``(2r+1)`` square removes, i.e. bright structure thinner than the square (strands);
    ``closing(alpha) - alpha`` is the same for thin dark gaps between strands.  Both are exactly zero
    in flat regions and along straight or gently curved edges of large shapes, whatever their
    softness, because a half-plane is unchanged by an opening - this is what distinguishes the
    measure from plain high-frequency energy, which is just as large on every hard edge.  The
    response is normalised (a strand of 25 % opacity already counts fully, matte noise below 10 %
    does not count at all) and then spread with a Gaussian so that the map is saturated within
    ``spread_px`` of the structure and fades to zero at about twice that distance.

    ``radius_px`` is rounded up to an integer: only then are opening and closing true morphological
    filters (anti-extensive / extensive), which the zero response on plain edges depends on.
    ``spread_px`` defaults to that radius; the refiner passes the reach of its operators so the
    protected zone covers everything they could smear a strand into.
    """

    a = _as_alpha(alpha)
    radius, sigma = _detail_geometry(radius_px, spread_px)

    # Opening and closing are two filters in a row.  The frame is extended once by their combined
    # reach, so both see the same (replicated) surroundings; padding each step separately is not a
    # morphological opening at the frame border and would report an edge that merely leaves the
    # frame as detail.
    pad = 2 * radius
    height, width = a.shape[-2:]
    extended = F.pad(a, (pad, pad, pad, pad), mode="replicate")
    opened = torch_ops.max_filter(torch_ops.min_filter(extended, radius), radius)
    closed = torch_ops.min_filter(torch_ops.max_filter(extended, radius), radius)
    tophat = torch.maximum(extended - opened, closed - extended)[..., pad : pad + height, pad : pad + width]
    detail = ((tophat - 0.10) / 0.15).clamp_(0.0, 1.0)

    # Gaussian weighted local energy.  The gain makes a one pixel wide line of full response reach
    # 3 at its centre, so the map saturates (>= 1) within 1.48 sigma (= the spread) of the line.
    gain = 3.0 * sigma * math.sqrt(2.0 * math.pi)
    return (torch_ops.gaussian_blur(detail, sigma) * gain).clamp_(0.0, 1.0)
