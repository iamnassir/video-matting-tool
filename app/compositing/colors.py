"""Colour conversion between native working planes and non-linear R'G'B'.

Why this module exists
----------------------
Frames travel through the pipeline as *native planes* (``app.core.pixfmt.PixFmt``): planar code
values with the source's chroma subsampling and bit depth.  On the inference device every plane is a
``float32 [B, 1, h, w]`` tensor that holds **integer code values** (exact in float32 up to 2^24).
R'G'B' is derived from them only where it is really needed - as model input, for previews and for
the semi-transparent edge band of the composite - so the pixels of the opaque core never go through
a colour-matrix round trip (invariant I4).

Three kinds of operations live here:

* :class:`ColorConverter` - matrix conversions planes <-> R'G'B' and the exact integer code values
  of a constant colour (the background constant of invariant I5).
* :func:`convert_planes` - matrix-free conversions between formats of the same family (bit depth and
  chroma subsampling only), used to restore the opaque core in an output whose format differs from
  the source.
* :func:`to_integer_planes`, :func:`planes_from_numpy`, :func:`planes_to_numpy` - the boundary
  between float code values on the device and the integer arrays of the decoder / encoder.

Conventions
-----------
* Plane order follows ``PixFmt``: yuv ``(Y, Cb, Cr[, A])``, rgb ``(G, B, R[, A])``, gray ``(Y,)``.
* Chroma is treated as *centre sited*: chroma sample ``j`` covers the luma samples ``2j`` and
  ``2j + 1`` (clipped to the frame for odd sizes).  Up-sampling is bilinear with
  ``align_corners=False``; down-sampling is the exact box average over that footprint.
* Rounding to integers is always "half away from zero" - never banker's rounding - so what an
  exact ``x.5`` becomes does not depend on the parity of ``x``.
* Nothing here modifies its inputs; outputs are created on the device of the inputs.

Determinism
-----------
A frame must render to the same bits whatever it is batched with (a resumed run has other batch
boundaries than the interrupted one).  PyTorch's fused element-wise kernels (``add(..., alpha=)``,
``lerp``, ``addcmul`` ...) are hand-vectorised on the CPU.  Where their vector body and their
scalar tail round differently - observed for ``add(alpha=)`` and scalar-weight ``lerp`` with
PyTorch 2.5 on x86-64: one path contracts ``a + k*b`` into a fused multiply-add, the other does
not - the result depends on where an element sits in the buffer, i.e. on the batch.  The arithmetic
here, in the compositor and in the foreground estimator therefore uses only plain ``mul`` / ``add`` /
``sub`` / ``div`` calls: each rounds exactly once per element, identically everywhere.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..core.errors import ConfigError
from ..core.pixfmt import PixFmt, ResolvedColor, quantization

log = logging.getLogger(__name__)

__all__ = [
    "ColorConverter",
    "convert_planes",
    "downsample_chroma",
    "planes_from_numpy",
    "planes_to_numpy",
    "to_integer_planes",
    "upsample_chroma",
]


# --------------------------------------------------------------------------- small helpers


def _as_float(plane: torch.Tensor) -> torch.Tensor:
    """Return *plane* as float32 (the very same tensor when it already is float32)."""

    return plane if plane.dtype == torch.float32 else plane.to(torch.float32)


def _round_half_up_(x: torch.Tensor) -> torch.Tensor:
    """Round a tensor **this module owns** to integers, halves upwards; *x* is used as scratch.

    ``floor(x + 0.5)`` is not used because the addition itself rounds: the largest float below 0.5
    would become 1.  The fractional part ``x - floor(x)`` is exact in floating point, so comparing
    it with 0.5 gives the mathematically correct result.  After clamping at zero this is identical
    to rounding half away from zero.
    """

    whole = torch.floor(x)
    x.sub_(whole).ge_(0.5)          # x now holds 1.0 where the fraction is >= 0.5
    return whole.add_(x)


def _round_code(value: float, max_value: int) -> int:
    """Round a float64 code value half away from zero and clamp it to ``0..max_value``.

    The value is first rounded to nine decimals so that a mathematically exact ``x.5`` that arrives
    as ``x.4999999999`` because of binary floating point still rounds away from zero.
    """

    v = round(float(value), 9)
    n = math.floor(abs(v) + 0.5)
    if v < 0:
        n = -n
    return int(min(max(n, 0), max_value))


def _check_planes(planes: Sequence[torch.Tensor], fmt: PixFmt, what: str) -> Tuple[int, int, int]:
    """Validate the colour planes of *fmt* and return ``(batch, height, width)`` of the frame."""

    count = fmt.color_planes
    if len(planes) < count:
        raise ValueError(f"{what}: {fmt.name} needs {count} colour plane(s), got {len(planes)}")
    first = planes[0]
    if first.dim() != 4 or first.shape[1] != 1:
        raise ValueError(f"{what}: planes must be [B, 1, h, w] tensors, got shape {tuple(first.shape)}")
    batch, _, height, width = first.shape
    for index in range(count):
        plane = planes[index]
        expected = (batch, 1) + fmt.plane_size(index, width, height)
        if tuple(plane.shape) != expected:
            raise ValueError(
                f"{what}: plane {index} of {fmt.name} has shape {tuple(plane.shape)}, expected {expected} "
                f"for a {width}x{height} frame"
            )
    return batch, height, width


def upsample_chroma(chroma: torch.Tensor, chroma_w: int, chroma_h: int, height: int, width: int) -> torch.Tensor:
    """Bilinear x2 up-sampling of ``[B, C, h, w]`` chroma along the subsampled axes.

    The plane is enlarged by exactly two (so the sample grid stays aligned with the luma grid) and
    then cropped to ``height x width``, which is how odd frame sizes are handled: their last chroma
    sample covers a single luma sample.  An axis that is not subsampled is left untouched bit for
    bit.
    """

    if not (chroma_w or chroma_h):
        return chroma
    h, w = chroma.shape[-2:]
    up = F.interpolate(chroma, size=(h << chroma_h, w << chroma_w), mode="bilinear", align_corners=False)
    if up.shape[-2] != height or up.shape[-1] != width:
        up = up[..., :height, :width]
    return up


def downsample_chroma(chroma: torch.Tensor, chroma_w: int, chroma_h: int) -> torch.Tensor:
    """Exact box average of ``[B, C, H, W]`` over each chroma sample's footprint.

    ``ceil_mode`` keeps the partial footprints of odd sizes and ``count_include_pad=False`` divides
    them by the number of samples that really exist, so the last row / column is a true average and
    a constant plane stays the same constant.
    """

    if not (chroma_w or chroma_h):
        return chroma
    kernel = (1 << chroma_h, 1 << chroma_w)
    return F.avg_pool2d(chroma, kernel_size=kernel, stride=kernel, ceil_mode=True, count_include_pad=False)


def _weighted_downsample(chroma: torch.Tensor, weight: torch.Tensor, chroma_w: int, chroma_h: int) -> torch.Tensor:
    """Weighted box average ``sum(w * c) / sum(w)`` of ``[B, C, H, W]`` over each chroma footprint.

    Footprints whose weights are all zero fall back to the plain average (their value is undefined
    by the weights and must still be finite).  With equal weights this is the plain box average.
    """

    total = downsample_chroma(weight, chroma_w, chroma_h)
    weighted = downsample_chroma(chroma * weight, chroma_w, chroma_h)
    defined = total > 0
    quotient = weighted / torch.where(defined, total, torch.ones_like(total))
    return torch.where(defined, quotient, downsample_chroma(chroma, chroma_w, chroma_h))


# --------------------------------------------------------------------------- matrix conversions


class ColorConverter:
    """Converts between the working planes of one stream and non-linear R'G'B'.

    ``fmt`` is the working format of the *source*; ``color`` the resolved matrix (``kr`` / ``kb``)
    and range.  The same matrix is used in both directions, so whatever interpretation was chosen
    for the source is also the one that produces the edge band and the background constant of the
    output - source and output therefore look alike in every player, whatever that player assumes.
    """

    def __init__(self, fmt: PixFmt, color: ResolvedColor) -> None:
        kr, kb = float(color.kr), float(color.kb)
        kg = 1.0 - kr - kb
        if not (0.0 < kr < 1.0 and 0.0 < kb < 1.0 and kg > 0.0):
            raise ConfigError(
                f"Invalid colour matrix coefficients kr={kr}, kb={kb} for matrix {color.matrix!r}.",
                hint="Use a matrix known to app.core.pixfmt.MATRIX_COEFFICIENTS (bt709, smpte170m, bt2020nc ...).",
            )
        self.fmt = fmt
        self.color = color
        self.kr, self.kg, self.kb = kr, kg, kb

    # -- range ---------------------------------------------------------------------------
    def full_range_for(self, fmt: PixFmt) -> bool:
        """Range (``True`` = full) used for planes of *fmt* by this converter.

        * Same kind of samples as the source (Y'CbCr / gray on both sides, or RGB on both sides):
          the source's range, so code values keep their meaning and the output can carry the
          source's range tag verbatim.
        * RGB planes derived from a Y'CbCr / gray source: full range.  Limited-range RGB is ignored
          by practically every decoder, so emitting it would wash the picture out.
        * Y'CbCr / gray planes derived from an RGB source: limited ("video") range, the meaning
          every player assumes for untagged Y'CbCr.

        Callers that tag an output whose family differs from the source must use this value rather
        than the source's tag.
        """

        src_rgb = self.fmt.family == "rgb"
        dst_rgb = fmt.family == "rgb"
        if src_rgb == dst_rgb:
            return bool(self.color.full_range)
        return dst_rgb

    # -- planes -> R'G'B' -----------------------------------------------------------------
    @torch.no_grad()
    def planes_to_rgb(self, planes: Sequence[torch.Tensor]) -> torch.Tensor:
        """Native planes of ``self.fmt`` -> ``float32 [B, 3, H, W]`` R'G'B' in 0..1 (clamped).

        ``planes`` are code values in ``PixFmt`` plane order, each ``[B, 1, h, w]`` of any numeric
        dtype; an extra alpha plane is ignored.  Chroma is brought to luma resolution with bilinear
        interpolation.  The result is only ever used as model input, for previews and inside the
        edge band - never for the opaque core of the output.
        """

        fmt = self.fmt
        batch, height, width = _check_planes(planes, fmt, "planes_to_rgb")
        luma_offset, luma_scale, chroma_offset, chroma_scale = quantization(fmt, self.full_range_for(fmt))

        if fmt.family == "rgb":
            green, blue, red = (_as_float(planes[i]) for i in range(3))
            rgb = torch.cat([red, green, blue], dim=1)
            if luma_offset:
                rgb.sub_(luma_offset)
            return rgb.div_(luma_scale).clamp_(0.0, 1.0)

        luma = _as_float(planes[0])
        if fmt.family == "gray":
            grey = (luma - luma_offset).div_(luma_scale).clamp_(0.0, 1.0)
            return grey.repeat(1, 3, 1, 1)

        chroma = torch.cat([_as_float(planes[1]), _as_float(planes[2])], dim=1)
        chroma = upsample_chroma(chroma, fmt.chroma_w, fmt.chroma_h, height, width)
        # Centre the chroma codes first: the subtraction is exact (the values are integers or, after
        # interpolation, multiples of 1/16), which keeps the following products small and the
        # float32 result accurate to a fraction of a code value even for 16-bit sources.
        # ``chroma`` is a tensor created above, so it may be modified in place.
        chroma.sub_(chroma_offset)
        cb, cr = chroma[:, 0:1], chroma[:, 1:2]
        y = (luma - luma_offset).div_(luma_scale)

        r_from_cr = 2.0 * (1.0 - self.kr) / chroma_scale
        b_from_cb = 2.0 * (1.0 - self.kb) / chroma_scale
        g_from_cb = -self.kb * 2.0 * (1.0 - self.kb) / (self.kg * chroma_scale)
        g_from_cr = -self.kr * 2.0 * (1.0 - self.kr) / (self.kg * chroma_scale)

        # Plain multiplications and additions only (see "Determinism" in the module docstring).
        rgb = torch.empty((batch, 3, height, width), dtype=torch.float32, device=luma.device)
        red, green, blue = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        torch.mul(cr, r_from_cr, out=red)
        red.add_(y)
        torch.mul(cb, g_from_cb, out=green)
        green.add_(cr * g_from_cr).add_(y)
        torch.mul(cb, b_from_cb, out=blue)
        blue.add_(y)
        return rgb.clamp_(0.0, 1.0)

    # -- R'G'B' -> planes -----------------------------------------------------------------
    @torch.no_grad()
    def rgb_to_planes(
        self,
        rgb: torch.Tensor,
        fmt: Optional[PixFmt] = None,
        *,
        chroma_weight: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """``float32 [B, 3, H, W]`` R'G'B' -> **unrounded** float32 code values per colour plane.

        The planes are those of *fmt* (default: the source format) with this converter's matrix,
        the range given by :meth:`full_range_for` and *fmt*'s bit depth.  Chroma is produced at
        4:4:4 and reduced by exact box averaging over each chroma sample's footprint, which is
        correct for odd sizes too.  No alpha plane is emitted.  Rounding is left to
        :func:`to_integer_planes` so that later steps (overrides, scaling) work on full precision.

        ``chroma_weight`` (``[B, 1, H, W]``, non-negative) turns the box average into a weighted
        one.  It exists for straight-alpha pictures: weighting by alpha keeps the colour of
        invisible pixels out of a chroma sample they share with visible ones.  Luma and
        non-subsampled chroma are not affected.
        """

        fmt = fmt or self.fmt
        if rgb.dim() != 4 or rgb.shape[1] != 3:
            raise ValueError(f"rgb_to_planes: rgb must be [B, 3, H, W], got shape {tuple(rgb.shape)}")
        if chroma_weight is not None and (
            chroma_weight.dim() != 4 or chroma_weight.shape[1] != 1
            or chroma_weight.shape[0] != rgb.shape[0] or chroma_weight.shape[-2:] != rgb.shape[-2:]
        ):
            raise ValueError(
                f"rgb_to_planes: chroma_weight must be [B, 1, H, W] matching rgb {tuple(rgb.shape)}, "
                f"got shape {tuple(chroma_weight.shape)}"
            )
        rgb = _as_float(rgb)
        luma_offset, luma_scale, chroma_offset, chroma_scale = quantization(fmt, self.full_range_for(fmt))

        if fmt.family == "rgb":
            scaled = rgb * luma_scale
            if luma_offset:
                scaled.add_(luma_offset)
            return [scaled[:, 1:2], scaled[:, 2:3], scaled[:, 0:1]]

        red, green, blue = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        y = red * self.kr
        y.add_(green * self.kg).add_(blue * self.kb)
        luma = y * luma_scale
        if luma_offset:
            luma.add_(luma_offset)
        if fmt.family == "gray":
            return [luma]

        batch, _, height, width = rgb.shape
        chroma = torch.empty((batch, 2, height, width), dtype=torch.float32, device=rgb.device)
        torch.sub(blue, y, out=chroma[:, 0:1])
        torch.sub(red, y, out=chroma[:, 1:2])
        chroma[:, 0:1].mul_(chroma_scale / (2.0 * (1.0 - self.kb)))
        chroma[:, 1:2].mul_(chroma_scale / (2.0 * (1.0 - self.kr)))
        chroma.add_(chroma_offset)
        if chroma_weight is not None and fmt.is_subsampled:
            chroma = _weighted_downsample(chroma, _as_float(chroma_weight), fmt.chroma_w, fmt.chroma_h)
        else:
            chroma = downsample_chroma(chroma, fmt.chroma_w, fmt.chroma_h)
        return [luma, chroma[:, 0:1], chroma[:, 1:2]]

    # -- constants ------------------------------------------------------------------------
    def color_code_values(self, rgb8: Tuple[int, int, int], fmt: Optional[PixFmt] = None) -> List[int]:
        """Exact integer code value per colour plane of *fmt* for a constant 8-bit R'G'B' colour.

        Computed in float64, rounded half away from zero and clamped to the code range.  These are
        the values every sample of a plane takes where the matte is zero (invariant I5), so they are
        computed once, in one place, instead of being whatever a float32 frame conversion happens to
        round to.
        """

        fmt = fmt or self.fmt
        try:
            red, green, blue = (float(v) for v in rgb8)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"A colour must be three 8-bit R'G'B' values, got {rgb8!r}.") from exc
        if not all(0.0 <= v <= 255.0 for v in (red, green, blue)):
            raise ConfigError(f"Colour components must lie in 0..255, got {rgb8!r}.")
        red, green, blue = red / 255.0, green / 255.0, blue / 255.0
        luma_offset, luma_scale, chroma_offset, chroma_scale = quantization(fmt, self.full_range_for(fmt))
        if fmt.family == "rgb":
            values = [luma_offset + luma_scale * c for c in (green, blue, red)]
        else:
            y = self.kr * red + self.kg * green + self.kb * blue
            values = [luma_offset + luma_scale * y]
            if fmt.family == "yuv":
                values.append(chroma_offset + chroma_scale * (blue - y) / (2.0 * (1.0 - self.kb)))
                values.append(chroma_offset + chroma_scale * (red - y) / (2.0 * (1.0 - self.kr)))
        return [_round_code(v, fmt.max_value) for v in values]


# --------------------------------------------------------------------------- plane-domain conversion

_PLANE_CONVERSIONS = {("yuv", "yuv"), ("rgb", "rgb"), ("gray", "gray"), ("gray", "yuv"), ("yuv", "gray")}


def _convert_depth(x: torch.Tensor, src: PixFmt, dst: PixFmt, full_range: bool) -> torch.Tensor:
    """Re-quantise code values from ``src.bit_depth`` to ``dst.bit_depth`` (returns a new tensor).

    Limited range: the nominal levels of every bit depth are the 8-bit levels shifted left, so the
    conversion is a multiplication / division by a power of two - exact when the depth increases.
    Full range: the peak moves from ``max_src`` to ``max_dst``.  A decrease cannot be exact; it is
    rounded half away from zero and clamped here so the caller receives legal code values.
    """

    if src.bit_depth == dst.bit_depth:
        return x
    if full_range:
        y = x * (float(dst.max_value) / float(src.max_value))
    elif dst.bit_depth > src.bit_depth:
        return x * float(1 << (dst.bit_depth - src.bit_depth))
    else:
        y = x / float(1 << (src.bit_depth - dst.bit_depth))
    if dst.bit_depth < src.bit_depth:
        y = _round_half_up_(y).clamp_(0.0, float(dst.max_value))
    return y


@torch.no_grad()
def convert_planes(planes: Sequence[torch.Tensor], src: PixFmt, dst: PixFmt, full_range: bool) -> List[torch.Tensor]:
    """Plane-domain conversion **without a colour matrix** between formats of the same family.

    Supported: yuv -> yuv, rgb -> rgb, gray -> gray, gray -> yuv (neutral chroma is added) and
    yuv -> gray (chroma is dropped).  Only bit depth and chroma subsampling change:

    * bit depth - limited range multiplies / divides by ``2 ** delta_bits`` (exact on increase,
      rounded on decrease); full range scales by ``max_dst / max_src``;
    * chroma - resampled only where the subsampling differs: bilinear up, box average down.  It is
      resampled *before* a depth reduction so the result is rounded once.

    Returns float32 code values for the colour planes of *dst* (an alpha plane of the source is
    ignored).  When the two formats have the same layout (alpha aside) the input values come back
    unchanged - for float32 inputs the very same tensors - which is what makes the opaque core of
    a same-format output bit-exact.  Do not modify the returned tensors in place.
    """

    batch, height, width = _check_planes(planes, src, "convert_planes")
    count = src.color_planes
    same_layout = (src.family, src.bit_depth, src.chroma_w, src.chroma_h) == (
        dst.family, dst.bit_depth, dst.chroma_w, dst.chroma_h
    )
    if same_layout:
        return [_as_float(planes[i]) for i in range(count)]
    if (src.family, dst.family) not in _PLANE_CONVERSIONS:
        raise ConfigError(
            f"Cannot convert {src.name} planes to {dst.name} without a colour matrix.",
            hint="Conversions between RGB and Y'CbCr / gray go through ColorConverter.planes_to_rgb / rgb_to_planes.",
        )

    if src.family == "rgb":
        return [_convert_depth(_as_float(planes[i]), src, dst, full_range) for i in range(3)]

    luma = _convert_depth(_as_float(planes[0]), src, dst, full_range)
    if dst.family == "gray":
        return [luma]

    if src.family == "gray":
        neutral = quantization(dst, full_range)[2]
        chroma_h, chroma_w = dst.plane_size(1, width, height)
        cb = torch.full((batch, 1, chroma_h, chroma_w), neutral, dtype=torch.float32, device=luma.device)
        return [luma, cb, cb.clone()]

    if (src.chroma_w, src.chroma_h) == (dst.chroma_w, dst.chroma_h):
        return [
            luma,
            _convert_depth(_as_float(planes[1]), src, dst, full_range),
            _convert_depth(_as_float(planes[2]), src, dst, full_range),
        ]

    chroma = torch.cat([_as_float(planes[1]), _as_float(planes[2])], dim=1)
    up_w = int(bool(src.chroma_w and not dst.chroma_w))
    up_h = int(bool(src.chroma_h and not dst.chroma_h))
    if up_w or up_h:
        chroma = upsample_chroma(
            chroma, up_w, up_h,
            height if up_h else chroma.shape[-2],
            width if up_w else chroma.shape[-1],
        )
    down_w = int(bool(dst.chroma_w and not src.chroma_w))
    down_h = int(bool(dst.chroma_h and not src.chroma_h))
    chroma = downsample_chroma(chroma, down_w, down_h)
    chroma = _convert_depth(chroma, src, dst, full_range)
    return [luma, chroma[:, 0:1], chroma[:, 1:2]]


# --------------------------------------------------------------------------- integer boundary


def _integer_dtype(fmt: PixFmt) -> torch.dtype:
    """Smallest signed-safe torch dtype for the code values of *fmt*.

    torch has no fully supported unsigned 16-bit type, so 9..15 bit formats use ``int16`` (their
    values fit) and 16-bit formats use ``int32``.
    """

    if fmt.bit_depth <= 8:
        return torch.uint8
    if fmt.bit_depth <= 15:
        return torch.int16
    return torch.int32


@torch.no_grad()
def to_integer_planes(planes: Sequence[torch.Tensor], fmt: PixFmt) -> List[torch.Tensor]:
    """Round float code values half away from zero, clamp to ``[0, fmt.max_value]`` and cast.

    The result dtype is ``uint8`` for 8-bit formats, ``int16`` for 9..15 bit and ``int32`` for
    16 bit.  Integer-valued inputs (the opaque core, the background constant) pass through exactly.
    """

    dtype = _integer_dtype(fmt)
    max_value = float(fmt.max_value)
    out: List[torch.Tensor] = []
    for plane in planes:
        if not plane.is_floating_point():
            out.append(plane.clamp(0, fmt.max_value).to(dtype))
            continue
        x = _as_float(plane)
        whole = torch.floor(x)
        fraction = torch.sub(x, whole).ge_(0.5)
        out.append(whole.add_(fraction).clamp_(0.0, max_value).to(dtype))
    return out


def planes_from_numpy(arrays: Sequence[np.ndarray], device) -> List[torch.Tensor]:
    """Decoder arrays ``[B, h, w]`` (``uint8`` or little-endian ``uint16``) -> float32 ``[B, 1, h, w]`` on *device*.

    The integer data is moved to the device first and widened there, so the host-to-device copy is
    one or two bytes per sample instead of four.  16-bit data travels as its ``int16`` bit pattern
    (torch's unsigned 16-bit support is incomplete, notably on CUDA) and the wrapped values are
    corrected on the device, so the full 0..65535 range survives.  The arrays are never written to
    and the result never aliases them.
    """

    device = torch.device(device) if device is not None else torch.device("cpu")
    out: List[torch.Tensor] = []
    for index, array in enumerate(arrays):
        arr = np.asarray(array)
        if arr.ndim != 3:
            raise ValueError(f"planes_from_numpy: plane {index} must be [B, h, w], got shape {arr.shape}")
        if not arr.flags.writeable:
            arr = arr.copy()            # torch refuses to wrap read-only memory without a warning
        if arr.dtype == np.uint8:
            plane = torch.from_numpy(np.ascontiguousarray(arr)).to(device).to(torch.float32)
        elif arr.dtype.kind == "u" and arr.dtype.itemsize == 2:
            if not arr.dtype.isnative:
                arr = arr.astype(arr.dtype.newbyteorder("="))
            bits = torch.from_numpy(np.ascontiguousarray(arr).view(np.int16)).to(device)
            plane = bits.to(torch.float32)
            plane.add_((plane < 0).to(torch.float32).mul_(65536.0))     # undo the int16 wrap-around
        elif arr.dtype.kind in "iuf":
            # Not a decoder layout, but harmless to accept: astype always copies, so no aliasing.
            plane = torch.from_numpy(arr.astype(np.float32)).to(device)
        else:
            raise ValueError(f"planes_from_numpy: plane {index} has unsupported dtype {arr.dtype}")
        out.append(plane.unsqueeze(1))
    return out


@torch.no_grad()
def planes_to_numpy(planes: Sequence[torch.Tensor], fmt: PixFmt) -> List[np.ndarray]:
    """Integer planes (see :func:`to_integer_planes`) -> C-contiguous numpy ``[B, h, w]`` of ``fmt.dtype``.

    Float planes are accepted and rounded first.  For CPU tensors the result may share memory with
    the given integer plane (no copy is made when the bit patterns already match).
    """

    out: List[np.ndarray] = []
    for index, plane in enumerate(planes):
        if plane.is_floating_point():
            plane = to_integer_planes([plane], fmt)[0]
        if plane.dim() == 4:
            if plane.shape[1] != 1:
                raise ValueError(f"planes_to_numpy: plane {index} must be [B, 1, h, w], got shape {tuple(plane.shape)}")
            plane = plane[:, 0]
        elif plane.dim() != 3:
            raise ValueError(f"planes_to_numpy: plane {index} must be [B, 1, h, w], got shape {tuple(plane.shape)}")
        arr = plane.detach().cpu().numpy()
        if arr.dtype == np.int16 and fmt.dtype.itemsize == 2:
            arr = arr.view(np.uint16)   # values are 0..32767: the bit pattern is already correct
        out.append(np.ascontiguousarray(arr.astype(fmt.dtype, copy=False)))
    return out
