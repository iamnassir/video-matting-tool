"""Small, dependency-free tensor operators shared by refinement and compositing.

All functions accept ``[B, C, H, W]`` float tensors, run on any device, never modify their input
and preserve constant regions exactly up to floating point rounding (a constant image stays the
same constant after every filter here, which matters for the uniform-background invariant).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _pad_replicate(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    """Replicate-pad ``(left, right, top, bottom)``; works for pads larger than the image."""

    left, right, top, bottom = pad
    if left == right == top == bottom == 0:
        return x
    h, w = x.shape[-2:]
    # F.pad(mode="replicate") handles any pad size, but needs a 4D/5D float tensor.
    return F.pad(x, (left, right, top, bottom), mode="replicate") if (h > 0 and w > 0) else x


def gaussian_kernel1d(sigma: float, device=None, dtype=torch.float32, truncate: float = 3.0) -> torch.Tensor:
    """Normalised 1-D Gaussian kernel with radius ``ceil(truncate * sigma)``."""

    radius = max(1, int(math.ceil(truncate * float(sigma))))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=torch.float64)
    k = torch.exp(-(xs * xs) / (2.0 * float(sigma) * float(sigma)))
    k = k / k.sum()
    return k.to(dtype)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur with replicate borders. ``sigma <= 0`` returns the input."""

    if sigma is None or sigma <= 0.0:
        return x
    k = gaussian_kernel1d(sigma, device=x.device, dtype=x.dtype)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    y = F.conv2d(_pad_replicate(x, (r, r, 0, 0)), kx, groups=c)
    y = F.conv2d(_pad_replicate(y, (0, 0, r, r)), ky, groups=c)
    return y


def box_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Separable mean filter over a ``(2r+1) x (2r+1)`` window with replicate borders."""

    r = int(radius)
    if r <= 0:
        return x
    k = 2 * r + 1
    y = F.avg_pool2d(_pad_replicate(x, (r, r, 0, 0)), kernel_size=(1, k), stride=1)
    y = F.avg_pool2d(_pad_replicate(y, (0, 0, r, r)), kernel_size=(k, 1), stride=1)
    return y


def _max_filter_int(x: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return x
    k = 2 * r + 1
    y = F.max_pool2d(_pad_replicate(x, (r, r, 0, 0)), kernel_size=(1, k), stride=1)
    y = F.max_pool2d(_pad_replicate(y, (0, 0, r, r)), kernel_size=(k, 1), stride=1)
    return y


def max_filter(x: torch.Tensor, radius: float) -> torch.Tensor:
    """Grey-scale dilation with a square structuring element; fractional radii are interpolated."""

    if radius is None or radius <= 0.0:
        return x
    lo = int(math.floor(radius))
    frac = float(radius) - lo
    a = _max_filter_int(x, lo)
    if frac < 1e-6:
        return a
    b = _max_filter_int(x, lo + 1)
    return torch.lerp(a, b, frac)


def min_filter(x: torch.Tensor, radius: float) -> torch.Tensor:
    """Grey-scale erosion with a square structuring element; fractional radii are interpolated."""

    if radius is None or radius <= 0.0:
        return x
    return -max_filter(-x, radius)


def resize(
    x: torch.Tensor,
    size: Tuple[int, int],
    mode: str = "bilinear",
    antialias: Optional[bool] = None,
) -> torch.Tensor:
    """Resize to ``(height, width)``.

    ``mode``: ``nearest`` | ``bilinear`` | ``bicubic`` | ``area``.  For bilinear / bicubic
    downscaling ``antialias`` defaults to True so thin structures are averaged, not skipped.
    """

    h, w = int(size[0]), int(size[1])
    if x.shape[-2] == h and x.shape[-1] == w:
        return x
    if mode in ("nearest", "area"):
        return F.interpolate(x, size=(h, w), mode=mode)
    if antialias is None:
        antialias = h < x.shape[-2] or w < x.shape[-1]
    return F.interpolate(x, size=(h, w), mode=mode, align_corners=False, antialias=antialias)


def fit_size(width: int, height: int, max_side: Optional[int], multiple: int = 1) -> Tuple[int, int]:
    """Largest ``(width, height)`` with the same aspect whose long side is <= max_side.

    Never upscales.  The result is rounded down to a multiple of *multiple* (minimum one multiple).
    """

    if not max_side or max(width, height) <= max_side:
        w, h = width, height
    else:
        scale = float(max_side) / float(max(width, height))
        w, h = int(round(width * scale)), int(round(height * scale))
    if multiple > 1:
        w = max(multiple, (w // multiple) * multiple)
        h = max(multiple, (h // multiple) * multiple)
    return max(1, w), max(1, h)


def luma(rgb: torch.Tensor) -> torch.Tensor:
    """BT.709 luma of ``[B, 3, H, W]`` R'G'B' -> ``[B, 1, H, W]``."""

    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def rotate_to_display(x: torch.Tensor, rotation: int) -> torch.Tensor:
    """Turn a coded-orientation ``[..., H, W]`` tensor into display orientation.

    *rotation* is the clockwise angle (0 / 90 / 180 / 270) a player applies to the coded picture.
    """

    k = (int(rotation) // 90) % 4
    if k == 0:
        return x
    # torch.rot90 with positive k rotates counter-clockwise, so clockwise needs -k.
    return torch.rot90(x, k=-k, dims=(-2, -1))


def rotate_to_coded(x: torch.Tensor, rotation: int) -> torch.Tensor:
    """Inverse of :func:`rotate_to_display`."""

    k = (int(rotation) // 90) % 4
    if k == 0:
        return x
    return torch.rot90(x, k=k, dims=(-2, -1))
