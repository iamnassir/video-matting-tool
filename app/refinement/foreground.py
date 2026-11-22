"""Foreground colour estimation for the semi-transparent band of a matte ("decontamination").

Why
---
A pixel on the edge of the subject is a mixture ``I = a*F + (1-a)*B`` of the subject's colour ``F``
and the *old* background ``B``.  Compositing ``I`` over a new background keeps a share of the old
one - the familiar halo around hair and soft edges.  The compositor therefore needs an estimate of
``F`` for every pixel with ``0 < a < 1``.

Method
------
Blur-Fusion (Forte and Pitie, "Approximate Fast Foreground Colour Estimation", ICIP 2021).  One
stage takes the image ``I``, current estimates ``F`` and ``B``, the matte ``a`` and a box-blur
radius ``r``::

    Fb = blur(F * a)     / (blur(a)     + eps)        local mean colour of the foreground
    Bb = blur(B * (1-a)) / (blur(1 - a) + eps)        local mean colour of the background
    F' = clamp(Fb + a * (I - a*Fb - (1-a)*Bb), 0, 1)  correct the mean so the mixture explains I

Two stages are run, as in the paper: stage 1 starts from ``F = B = I`` with a large radius (it
gathers colour from well inside the subject); stage 2 starts from stage 1's ``F`` and ``Bb`` with a
small radius and restores local colour detail.

Guarantees the compositor relies on
-----------------------------------
* Where ``alpha >= 1`` the result is the input, bit for bit: opaque pixels are never touched.
* ``strength`` blends between the input (0) and the estimate (1); ``strength == 0`` returns the
  input tensor itself.
* Every frame of a batch is processed independently, so a batch gives exactly what the frames give
  one by one - a resumed run (different batch boundaries) renders the same pixels.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from ..utils.torch_ops import box_blur, fit_size, resize

log = logging.getLogger(__name__)

__all__ = ["estimate_foreground"]

#: Regulariser of the two normalised blurs (the value used by the reference implementation).
_EPS = 1e-5

#: Frames are processed in sub-batches of about this many pixels.  The estimate needs several
#: full-resolution temporaries with up to seven channels; bounding the sub-batch keeps the transient
#: memory at a few hundred megabytes for HD material whatever batch size the pipeline uses.  The
#: result does not depend on the split because frames never interact.
_CHUNK_PIXELS = 1 << 21


def _fuse(
    image: torch.Tensor,
    alpha: torch.Tensor,
    inv_alpha: torch.Tensor,
    fg_mean: torch.Tensor,
    bg_mean: torch.Tensor,
) -> torch.Tensor:
    """``clamp(Fb + a * (I - a*Fb - (1-a)*Bb), 0, 1)`` - the fusion step shared by both stages.

    Written with plain products and sums on purpose: PyTorch's fused kernels (``addcmul``,
    ``add(alpha=)``, ``lerp``) may round differently in their vectorised body and their scalar
    tail, which would make a frame's result depend on its position in the batch.
    """

    residual = image - alpha * fg_mean
    residual.sub_(inv_alpha * bg_mean)
    return residual.mul_(alpha).add_(fg_mean).clamp_(0.0, 1.0)


def _blur_fusion(
    image: torch.Tensor,
    alpha: torch.Tensor,
    radius_work: int,
    radius_refine: int,
    work_size: Tuple[int, int],
) -> torch.Tensor:
    """Two-stage Blur-Fusion for one sub-batch. ``work_size`` is ``(height, width)`` of stage 1."""

    full_size = (image.shape[-2], image.shape[-1])
    reduced = work_size != full_size
    inv_alpha = 1.0 - alpha

    # ---- stage 1: F = B = I, large radius, on the reduced working copy -------------------------
    # The products are formed at full resolution and *then* reduced: averaging I*a keeps thin
    # structures (hair) free of background colour, whereas multiplying the averaged I by the
    # averaged a would mix the background into them.  blur is linear, so blur(I*(1-a)) and
    # blur(1-a) follow from blur(I), blur(I*a) and blur(a): one seven-channel blur does it all.
    weighted = image * alpha
    if reduced:
        stack = torch.cat(
            [
                resize(image, work_size, mode="bilinear", antialias=True),
                resize(weighted, work_size, mode="bilinear", antialias=True),
                resize(alpha, work_size, mode="bilinear", antialias=True),
            ],
            dim=1,
        )
    else:
        stack = torch.cat([image, weighted, alpha], dim=1)
    del weighted
    blurred = box_blur(stack, radius_work)
    del stack
    blur_image, blur_weighted, blur_alpha = blurred[:, 0:3], blurred[:, 3:6], blurred[:, 6:7]
    fg_mean = blur_weighted / (blur_alpha + _EPS)
    # Cancellation can leave the two differences a hair below zero where the window holds no
    # background at all; clamp them so the quotient stays a colour.
    bg_mean = (blur_image - blur_weighted).clamp_(min=0.0) / ((1.0 - blur_alpha).clamp_(min=0.0) + _EPS)
    del blurred
    means = torch.cat([fg_mean, bg_mean], dim=1)
    if reduced:
        means = resize(means, full_size, mode="bilinear")
    fg_mean, bg_mean = means[:, 0:3], means[:, 3:6]
    estimate = _fuse(image, alpha, inv_alpha, fg_mean, bg_mean)
    if radius_refine <= 0:
        return estimate

    # ---- stage 2: F from stage 1, B = stage 1's blurred background, small radius, full size ----
    stack = torch.cat([estimate * alpha, bg_mean * inv_alpha, alpha], dim=1)
    del means, fg_mean, bg_mean, estimate
    blurred = box_blur(stack, radius_refine)
    del stack
    blur_alpha = blurred[:, 6:7]
    fg_mean = blurred[:, 0:3] / (blur_alpha + _EPS)
    bg_mean = blurred[:, 3:6] / ((1.0 - blur_alpha).clamp_(min=0.0) + _EPS)
    return _fuse(image, alpha, inv_alpha, fg_mean, bg_mean)


@torch.no_grad()
def estimate_foreground(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    *,
    strength: float = 1.0,
    radius: float = 90.0,
    refine_radius: float = 6.0,
    max_working_side: Optional[int] = 640,
) -> torch.Tensor:
    """Estimate the un-mixed foreground colour of every pixel (two-stage Blur-Fusion).

    Parameters
    ----------
    rgb:
        ``float32 [B, 3, H, W]`` non-linear R'G'B' in 0..1 - the observed, mixed colours.
    alpha:
        ``float32 [B, 1, H, W]`` matte in 0..1.
    strength:
        0 returns ``rgb`` unchanged (the same tensor); 1 returns the full estimate; values in
        between blend linearly.  Values above 1 are treated as 1.
    radius, refine_radius:
        Box-blur radii of stage 1 and stage 2 in pixels **at the actual frame size** (the caller
        scales its 1080p-referenced settings).  A radius ``r`` means a ``(2r + 1)``-wide window, so
        the kernel *sizes* 90 and 6 of the reference implementation correspond to radii of about 45
        and 3.  The stage 1 radius should exceed the width of the widest soft edge, otherwise the
        window never reaches pure foreground.  ``refine_radius <= 0`` skips stage 2.
    max_working_side:
        Stage 1 only needs low frequencies, so its blurs run on a copy whose long side is at most
        this many pixels (the radius is scaled along) and the blurred means are up-sampled.  Stage 2
        always runs at full resolution.  ``None`` or 0 disables the reduction.

    Returns
    -------
    ``float32 [B, 3, H, W]`` in 0..1.  Exactly ``rgb`` wherever ``alpha >= 1``.  Outside the subject
    (``alpha == 0``) the result is the surrounding foreground colour within the refine radius and
    fades to black beyond it; those values carry no information and are only meaningful to callers
    that multiply by alpha.
    """

    if rgb.dim() != 4 or rgb.shape[1] != 3:
        raise ValueError(f"estimate_foreground: rgb must be [B, 3, H, W], got shape {tuple(rgb.shape)}")
    if alpha.dim() != 4 or alpha.shape[1] != 1 or alpha.shape[0] != rgb.shape[0] or alpha.shape[-2:] != rgb.shape[-2:]:
        raise ValueError(
            f"estimate_foreground: alpha must be [B, 1, H, W] matching rgb {tuple(rgb.shape)}, "
            f"got shape {tuple(alpha.shape)}"
        )
    strength = float(strength)
    if not strength > 0.0:              # also true for NaN: never blend with an undefined weight
        return rgb
    strength = min(strength, 1.0)
    batch, _, height, width = rgb.shape
    if batch == 0 or height == 0 or width == 0:
        return rgb

    image = rgb if rgb.dtype == torch.float32 else rgb.to(torch.float32)
    # A private, sanitised copy of the matte: the input is never modified, and an undefined sample
    # (NaN from a failed inference) is treated as "no subject" instead of poisoning its whole blur
    # window.
    matte = torch.nan_to_num(alpha.to(device=image.device, dtype=torch.float32), nan=0.0).clamp_(0.0, 1.0)

    side = int(max_working_side) if max_working_side else 0
    work_w, work_h = fit_size(width, height, side if side > 0 else None)
    scale = min(work_w / float(width), work_h / float(height))
    radius_work = min(max(1, int(round(float(radius) * scale))), max(work_w, work_h))
    radius_refine = 0
    if refine_radius is not None and float(refine_radius) > 0.0:
        radius_refine = min(max(1, int(round(float(refine_radius)))), max(width, height))

    step = max(1, _CHUNK_PIXELS // (height * width))
    if step >= batch:
        estimate = _blur_fusion(image, matte, radius_work, radius_refine, (work_h, work_w))
    else:
        estimate = torch.empty_like(image)
        for start in range(0, batch, step):
            part = slice(start, start + step)
            estimate[part] = _blur_fusion(image[part], matte[part], radius_work, radius_refine, (work_h, work_w))

    if strength < 1.0:
        estimate = estimate.sub_(image).mul_(strength).add_(image)      # image + strength * (estimate - image)
    # At a == 1 the fusion formula reduces to I mathematically but not bit for bit; opaque pixels
    # must be the observed pixels exactly.
    return torch.where(matte >= 1.0, image, estimate)
