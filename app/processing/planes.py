"""Lossless bridge between PyAV frames and numpy planes.

The pipeline keeps frames as *native planes* (see ``app.core.pixfmt``): no YUV<->RGB conversion, no
range conversion, no resampling.  This module maps every decoder pixel format to a planar
little-endian **working format** and moves sample values between ``av.VideoFrame`` objects and
numpy arrays without changing them.

Why not PyAV's own conversion helpers (verified on PyAV 14.4 and 17.1):

* ``VideoFrame.reformat(format="yuv420p")`` on a ``yuvj420p`` frame *rescales* full range to limited
  range - it is not a relabel;
* ``VideoFrame.to_ndarray()`` does not support several planar formats and had a plane mix-up for
  ``yuv420p10le``;
* decoder plane buffers are padded (``line_size`` > width) and are the decoder's reference frames, so
  they must be cropped and copied, never written to.

Mapping classes (``WorkingFormat.method``):

=============  =================================================  ======
method         examples                                           exact
=============  =================================================  ======
direct         yuv420p, yuv422p10le, yuv444p, gbrp, gray16le       yes
relabel        yuvj420p -> yuv420p (+ full-range hint)             yes
byteswap       yuv420p10be -> yuv420p10le                          yes
semiplanar     nv12 / nv21 / nv16 / p010le -> yuv420p[10le] ...    yes
packed_yuv     yuyv422 / uyvy422 / yvyu422 -> yuv422p              yes
packed_rgb     rgb24 / bgra / rgb48le ... -> gbrp / gbrap          yes
chroma_repeat  yuv410p / yuv411p -> yuv420p / yuv422p              values kept, chroma replicated
swscale        pal8, rgb565, bayer ... -> gbrp                     no (converted by libswscale)
=============  =================================================  ======
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import av
import numpy as np

from ..core.errors import UnsupportedFormatError
from ..core.pixfmt import PixFmt, ResolvedColor, make_pixfmt, parse_working_pixfmt

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkingFormat:
    """How a decoder pixel format is brought into the planar working representation."""

    source_pix_fmt: str
    fmt: PixFmt
    method: str
    exact: bool = True
    full_range_hint: Optional[bool] = None    # True for yuvj* (the format name implies full range)
    note: str = ""


# ----------------------------------------------------------------------------- format tables

_FULL_RANGE_ALIASES: Dict[str, str] = {
    "yuvj420p": "yuv420p", "yuvj422p": "yuv422p", "yuvj444p": "yuv444p", "yuvj440p": "yuv440p",
}

#: semi-planar: (working format, index of U in the interleaved plane, index of V, right shift)
_SEMI_PLANAR: Dict[str, Tuple[str, int, int, int]] = {
    "nv12": ("yuv420p", 0, 1, 0),
    "nv21": ("yuv420p", 1, 0, 0),
    "nv16": ("yuv422p", 0, 1, 0),
    "nv24": ("yuv444p", 0, 1, 0),
    "nv42": ("yuv444p", 1, 0, 0),
    "p010le": ("yuv420p10le", 0, 1, 6),     # 10 significant bits stored in the HIGH bits of 16
    "p012le": ("yuv420p12le", 0, 1, 4),
    "p016le": ("yuv420p16le", 0, 1, 0),
    "p210le": ("yuv422p10le", 0, 1, 6),
    "p216le": ("yuv422p16le", 0, 1, 0),
    "p410le": ("yuv444p10le", 0, 1, 6),
    "p416le": ("yuv444p16le", 0, 1, 0),
}

#: packed 4:2:2: byte offsets of (Y0, U, Y1, V) inside each 4-byte group
_PACKED_YUV422: Dict[str, Tuple[int, int, int, int]] = {
    "yuyv422": (0, 1, 2, 3),
    "uyvy422": (1, 0, 3, 2),
    "yvyu422": (0, 3, 2, 1),
}

#: packed RGB: (bytes per sample, channels per pixel, index of R, G, B, index of A or None)
_PACKED_RGB: Dict[str, Tuple[int, int, int, int, int, Optional[int]]] = {
    "rgb24": (1, 3, 0, 1, 2, None),
    "bgr24": (1, 3, 2, 1, 0, None),
    "rgba": (1, 4, 0, 1, 2, 3),
    "bgra": (1, 4, 2, 1, 0, 3),
    "argb": (1, 4, 1, 2, 3, 0),
    "abgr": (1, 4, 3, 2, 1, 0),
    "rgb0": (1, 4, 0, 1, 2, None),
    "bgr0": (1, 4, 2, 1, 0, None),
    "0rgb": (1, 4, 1, 2, 3, None),
    "0bgr": (1, 4, 3, 2, 1, None),
    "rgb48le": (2, 3, 0, 1, 2, None),
    "bgr48le": (2, 3, 2, 1, 0, None),
    "rgba64le": (2, 4, 0, 1, 2, 3),
    "bgra64le": (2, 4, 2, 1, 0, 3),
}

#: heavily subsampled chroma -> nearest working layout (horizontal sample replication)
_CHROMA_REPEAT: Dict[str, Tuple[str, int]] = {
    "yuv410p": ("yuv420p", 2),     # 4x4 chroma -> 2x2 after repeating each sample twice in both directions
    "yuv411p": ("yuv422p", 2),     # 4x1 chroma -> 2x1 after repeating each sample twice horizontally
    "yuvj411p": ("yuv422p", 2),
}


def _descriptor(name: str) -> "av.VideoFormat":
    try:
        return av.VideoFormat(name, 16, 16)
    except Exception as exc:
        raise UnsupportedFormatError(f"Unknown pixel format '{name}'.") from exc


def _is_working(name: str) -> Optional[PixFmt]:
    try:
        return parse_working_pixfmt(name)
    except ValueError:
        return None


@lru_cache(maxsize=256)
def working_format_for(pix_fmt: str) -> WorkingFormat:
    """Describe how frames of *pix_fmt* are represented in the pipeline."""

    name = pix_fmt.strip().lower()
    fmt = _is_working(name)
    if fmt is not None:
        return WorkingFormat(name, fmt, "direct")
    if name in _FULL_RANGE_ALIASES:
        return WorkingFormat(name, parse_working_pixfmt(_FULL_RANGE_ALIASES[name]), "relabel", True, True,
                             "full-range variant of the same layout")
    if name in _SEMI_PLANAR:
        return WorkingFormat(name, parse_working_pixfmt(_SEMI_PLANAR[name][0]), "semiplanar", True, None,
                             "interleaved chroma separated into two planes")
    if name in _PACKED_YUV422:
        return WorkingFormat(name, make_pixfmt("yuv", 8, 1, 0), "packed_yuv", True, None, "packed 4:2:2 unpacked")
    if name in _PACKED_RGB:
        bps, _, _, _, _, a = _PACKED_RGB[name]
        return WorkingFormat(name, make_pixfmt("rgb", 8 if bps == 1 else 16, 0, 0, a is not None), "packed_rgb",
                             True, None, "packed RGB split into planes")
    if name in _CHROMA_REPEAT:
        target, _ = _CHROMA_REPEAT[name]
        return WorkingFormat(
            name, parse_working_pixfmt(target), "chroma_repeat", False,
            True if name.startswith("yuvj") else None,
            "chroma samples replicated to the nearest supported subsampling",
        )
    if name.endswith("be"):
        le = _is_working(name[:-2] + "le")
        if le is not None:
            return WorkingFormat(name, le, "byteswap", True, None, "big-endian samples byte-swapped")
    if name in ("ya8", "gray8a"):
        return WorkingFormat(name, make_pixfmt("gray", 8), "gray_alpha", True, None, "alpha channel of the source dropped")
    if name == "ya16le":
        return WorkingFormat(name, make_pixfmt("gray", 16), "gray_alpha", True, None, "alpha channel of the source dropped")

    # Anything else (palette, bit-packed RGB, Bayer, ...) goes through libswscale once, on input.
    desc = _descriptor(name)
    bits = max((c.bits for c in desc.components), default=8)
    has_alpha = any(getattr(c, "is_alpha", False) for c in desc.components) and not desc.has_palette
    depth = 8 if bits <= 8 else 16
    if desc.is_rgb or desc.has_palette or name.startswith("bayer") or name.startswith("mono"):
        target = make_pixfmt("rgb", depth, 0, 0, has_alpha)
    elif len([c for c in desc.components if not getattr(c, "is_alpha", False)]) == 1:
        target = make_pixfmt("gray", depth)
    else:
        target = make_pixfmt("yuv", depth, 0, 0, has_alpha)
    return WorkingFormat(name, target, "swscale", False, None, "converted by libswscale (no exact planar equivalent)")


# ----------------------------------------------------------------------------- raw plane access


def _plane_view(plane: "av.video.plane.VideoPlane", bytes_per_sample: int, samples_per_pixel: int = 1) -> np.ndarray:
    """Read-only-by-convention view of the visible part of a plane (alignment padding removed)."""

    line = plane.line_size
    if line <= 0:
        raise UnsupportedFormatError("Vertically flipped frames (negative line size) are not supported.")
    dtype = np.uint8 if bytes_per_sample == 1 else np.dtype("<u2")
    arr = np.frombuffer(plane, dtype=dtype).reshape(plane.height, line // bytes_per_sample)
    return arr[:, : plane.width * samples_per_pixel]


def _own(arr: np.ndarray, copy: bool) -> np.ndarray:
    return np.ascontiguousarray(arr).copy() if copy and arr.flags.c_contiguous else np.ascontiguousarray(arr)


def frame_to_planes(frame: "av.VideoFrame", working: Optional[WorkingFormat] = None, copy: bool = True) -> List[np.ndarray]:
    """Planes of *frame* in its working format (``working.fmt`` plane order), without value changes.

    The result always owns its memory when ``copy`` is true (the default): decoder buffers are reused
    for later frames and must not be kept.  ``copy=False`` returns views where the layout allows it;
    they are only valid while *frame* is alive.
    """

    name = frame.format.name
    wf = working if working is not None and working.source_pix_fmt == name else working_format_for(name)
    fmt = wf.fmt
    bps = fmt.bytes_per_sample
    method = wf.method

    if method in ("direct", "relabel"):
        return [_own(_plane_view(frame.planes[i], bps), copy) for i in range(fmt.num_planes)]

    if method == "byteswap":
        # The buffer holds big-endian samples but was read as little-endian: swapping the bytes of
        # each sample yields the true values.
        return [np.ascontiguousarray(_plane_view(frame.planes[i], bps).byteswap()) for i in range(fmt.num_planes)]

    if method == "semiplanar":
        _, iu, iv, shift = _SEMI_PLANAR[name]
        luma = _plane_view(frame.planes[0], bps)
        chroma = _plane_view(frame.planes[1], bps, 2).reshape(frame.planes[1].height, -1, 2)
        y, u, v = luma, chroma[..., iu], chroma[..., iv]
        if shift:
            y, u, v = y >> shift, u >> shift, v >> shift
        return [np.ascontiguousarray(y) if shift else _own(y, copy), np.ascontiguousarray(u), np.ascontiguousarray(v)]

    if method == "packed_yuv":
        oy0, ou, oy1, ov = _PACKED_YUV422[name]
        packed = _plane_view(frame.planes[0], 1, 2)                 # width * 2 bytes per row
        width = frame.width
        y = np.empty((frame.height, width), dtype=np.uint8)
        y[:, 0::2] = packed[:, oy0::4][:, : (width + 1) // 2]
        y[:, 1::2] = packed[:, oy1::4][:, : width // 2]
        return [y, np.ascontiguousarray(packed[:, ou::4]), np.ascontiguousarray(packed[:, ov::4])]

    if method == "packed_rgb":
        sample_bytes, channels, ir, ig, ib, ia = _PACKED_RGB[name]
        packed = _plane_view(frame.planes[0], sample_bytes, channels).reshape(frame.height, frame.width, channels)
        planes = [np.ascontiguousarray(packed[..., ig]), np.ascontiguousarray(packed[..., ib]),
                  np.ascontiguousarray(packed[..., ir])]
        if ia is not None:
            planes.append(np.ascontiguousarray(packed[..., ia]))
        return planes

    if method == "chroma_repeat":
        _, factor = _CHROMA_REPEAT[name]
        y = _own(_plane_view(frame.planes[0], 1), copy)
        out = [y]
        for i in (1, 2):
            c = _plane_view(frame.planes[i], 1)
            c = np.repeat(c, factor, axis=1)
            if name == "yuv410p":
                c = np.repeat(c, factor, axis=0)
            h, w = fmt.plane_size(i, frame.width, frame.height)
            out.append(np.ascontiguousarray(c[:h, :w]))
        return out

    if method == "gray_alpha":
        packed = _plane_view(frame.planes[0], bps, 2).reshape(frame.height, frame.width, 2)
        return [np.ascontiguousarray(packed[..., 0])]

    # swscale fallback: one conversion on input; the result is then handled like a direct format
    converted = frame.reformat(format=fmt.name)
    return [_own(_plane_view(converted.planes[i], bps), True) for i in range(fmt.num_planes)]


def planes_to_frame(
    planes: Sequence[np.ndarray],
    fmt: PixFmt,
    pts: Optional[int] = None,
    time_base: Optional[Fraction] = None,
) -> "av.VideoFrame":
    """Build an ``av.VideoFrame`` in the working format *fmt* from numpy planes (strict, no conversion).

    Shapes and dtypes must match the format exactly; nothing is clipped, converted or resampled.
    Frames allocated by PyAV pad each row to an alignment boundary, so the samples are written through
    a view of the plane buffer that honours ``line_size``.
    """

    if len(planes) != fmt.num_planes:
        raise ValueError(f"{fmt.name} needs {fmt.num_planes} planes, got {len(planes)}")
    height, width = planes[0].shape[:2]
    frame = av.VideoFrame(width, height, fmt.name)
    bps = fmt.bytes_per_sample
    dtype = fmt.dtype
    for i, src in enumerate(planes):
        want = fmt.plane_size(i, width, height)
        arr = np.asarray(src)
        if arr.dtype != dtype:
            raise ValueError(f"plane {i} of {fmt.name} must be {dtype}, got {arr.dtype}")
        if arr.shape != want:
            raise ValueError(f"plane {i} of {fmt.name} must have shape {want}, got {arr.shape}")
        plane = frame.planes[i]
        dst = np.frombuffer(plane, dtype=dtype).reshape(plane.height, plane.line_size // bps)
        dst[:, : plane.width] = arr
    if pts is not None:
        frame.pts = int(pts)
    if time_base is not None:
        frame.time_base = time_base
    return frame


def to_semiplanar(planes: Sequence[np.ndarray], fmt: PixFmt, pipe_pix_fmt: str) -> List[np.ndarray]:
    """Repack planar YUV into a semi-planar layout (``nv12``, ``p010le``, ...) for encoders that only
    accept that (NVENC 10-bit).  Sample values are unchanged apart from the documented bit position."""

    spec = _SEMI_PLANAR.get(pipe_pix_fmt)
    if spec is None or fmt.family != "yuv":
        raise ValueError(f"cannot repack {fmt.name} as {pipe_pix_fmt}")
    target, iu, iv, shift = spec
    if parse_working_pixfmt(target).name != make_pixfmt("yuv", fmt.bit_depth, fmt.chroma_w, fmt.chroma_h).name:
        raise ValueError(f"{pipe_pix_fmt} does not match the layout of {fmt.name}")
    y, u, v = planes[0], planes[1], planes[2]
    chroma = np.empty(u.shape + (2,), dtype=u.dtype)
    chroma[..., iu] = u
    chroma[..., iv] = v
    if shift:
        return [np.ascontiguousarray(y << shift), np.ascontiguousarray(chroma << shift)]
    return [np.ascontiguousarray(y), chroma]


def semiplanar_to_frame(
    planes: Sequence[np.ndarray], pipe_pix_fmt: str, pts: Optional[int] = None, time_base: Optional[Fraction] = None
) -> "av.VideoFrame":
    """Build a frame in a semi-planar format from the output of :func:`to_semiplanar`."""

    y, chroma = planes
    height, width = y.shape
    frame = av.VideoFrame(width, height, pipe_pix_fmt)
    bps = y.dtype.itemsize
    dtype = np.uint8 if bps == 1 else np.dtype("<u2")
    p0, p1 = frame.planes[0], frame.planes[1]
    np.frombuffer(p0, dtype=dtype).reshape(p0.height, p0.line_size // bps)[:, :width] = y
    flat = chroma.reshape(chroma.shape[0], -1)
    np.frombuffer(p1, dtype=dtype).reshape(p1.height, p1.line_size // bps)[:, : flat.shape[1]] = flat
    if pts is not None:
        frame.pts = int(pts)
    if time_base is not None:
        frame.time_base = time_base
    return frame


def planes_to_rgb8(planes: Sequence[np.ndarray], fmt: PixFmt, color: "ResolvedColor") -> np.ndarray:
    """Convert working-format planes to an 8-bit R'G'B' image ``uint8 [H, W, 3]`` on the CPU.

    For thumbnails, subject selection and previews only - never for pixels that go to the output.
    Uses the resolved matrix / range (``app.core.pixfmt.resolve_color``); chroma is upsampled
    bilinearly.
    """

    import cv2

    from ..core.pixfmt import quantization

    luma_off, luma_scale, chroma_off, chroma_scale = quantization(fmt, color.full_range)
    height, width = planes[0].shape[:2]
    if fmt.family == "gray":
        y = (planes[0].astype(np.float32) - luma_off) / luma_scale
        rgb = np.repeat(y[..., None], 3, axis=2)
    elif fmt.family == "rgb":
        scale = float(fmt.max_value) if color.full_range else luma_scale
        offset = 0.0 if color.full_range else luma_off
        g, b, r = ((planes[i].astype(np.float32) - offset) / scale for i in range(3))
        rgb = np.stack([r, g, b], axis=2)
    else:
        y = (planes[0].astype(np.float32) - luma_off) / luma_scale

        def chroma(plane: np.ndarray) -> np.ndarray:
            c = (plane.astype(np.float32) - chroma_off) / chroma_scale
            if c.shape != (height, width):
                c = cv2.resize(c, (width, height), interpolation=cv2.INTER_LINEAR)
            return c

        cb, cr = chroma(planes[1]), chroma(planes[2])
        kr, kb = color.kr, color.kb
        kg = 1.0 - kr - kb
        r = y + 2.0 * (1.0 - kr) * cr
        b = y + 2.0 * (1.0 - kb) * cb
        g = (y - kr * r - kb * b) / kg
        rgb = np.stack([r, g, b], axis=2)
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def blank_planes(fmt: PixFmt, width: int, height: int, values: Sequence[int]) -> List[np.ndarray]:
    """Constant planes (one code value per plane) - used for padding and tests."""

    return [
        np.full(fmt.plane_size(i, width, height), int(values[i]), dtype=fmt.dtype) for i in range(fmt.num_planes)
    ]
