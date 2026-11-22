"""Pixel-format and colour descriptions used across decoding, compositing and encoding.

Design rule (pixel preservation): frames travel through the pipeline as *native planes* in a
"working format" - a planar, little-endian layout with the same chroma subsampling and bit depth as
the source (``yuv420p``, ``yuv422p10le``, ``gbrp`` ...).  No YUV<->RGB matrix conversion is ever
applied to the pixels that end up in the output where the matte is fully opaque; RGB is derived only
as model input, for previews, and for the semi-transparent edge band.

This module is intentionally free of PyAV / torch imports so it can be used anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- pixel formats


@dataclass(frozen=True)
class PixFmt:
    """A planar working pixel format.

    Plane order:
      * family ``yuv``  -> (Y, Cb, Cr[, A])
      * family ``rgb``  -> (G, B, R[, A])      (FFmpeg ``gbrp`` layout)
      * family ``gray`` -> (Y,)
    """

    name: str            # FFmpeg / PyAV pixel format name
    family: str          # "yuv" | "rgb" | "gray"
    bit_depth: int       # 8, 9, 10, 12, 14 or 16 significant bits per component
    chroma_w: int = 0    # log2 horizontal chroma subsampling (0 or 1)
    chroma_h: int = 0    # log2 vertical chroma subsampling (0 or 1)
    has_alpha: bool = False

    # -- derived -------------------------------------------------------------------------
    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.uint8) if self.bit_depth <= 8 else np.dtype("<u2")

    @property
    def bytes_per_sample(self) -> int:
        return 1 if self.bit_depth <= 8 else 2

    @property
    def max_value(self) -> int:
        return (1 << self.bit_depth) - 1

    @property
    def num_planes(self) -> int:
        base = 1 if self.family == "gray" else 3
        return base + (1 if self.has_alpha else 0)

    @property
    def color_planes(self) -> int:
        return 1 if self.family == "gray" else 3

    @property
    def is_subsampled(self) -> bool:
        return bool(self.chroma_w or self.chroma_h)

    def is_chroma_plane(self, index: int) -> bool:
        return self.family == "yuv" and index in (1, 2)

    def plane_size(self, index: int, width: int, height: int) -> Tuple[int, int]:
        """Return ``(plane_height, plane_width)`` of plane *index* for a ``width x height`` frame."""

        if self.is_chroma_plane(index):
            return (
                (height + (1 << self.chroma_h) - 1) >> self.chroma_h,
                (width + (1 << self.chroma_w) - 1) >> self.chroma_w,
            )
        return (height, width)

    def frame_bytes(self, width: int, height: int) -> int:
        total = 0
        for i in range(self.num_planes):
            h, w = self.plane_size(i, width, height)
            total += h * w * self.bytes_per_sample
        return total

    def with_alpha(self, has_alpha: bool = True) -> "PixFmt":
        return make_pixfmt(self.family, self.bit_depth, self.chroma_w, self.chroma_h, has_alpha)

    def with_depth(self, bit_depth: int) -> "PixFmt":
        return make_pixfmt(self.family, bit_depth, self.chroma_w, self.chroma_h, self.has_alpha)

    def with_chroma(self, chroma_w: int, chroma_h: int) -> "PixFmt":
        return make_pixfmt(self.family, self.bit_depth, chroma_w, chroma_h, self.has_alpha)


_SUBSAMPLING_NAMES: Dict[Tuple[int, int], str] = {(1, 1): "420", (1, 0): "422", (0, 0): "444", (0, 1): "440"}


def pixfmt_name(family: str, bit_depth: int, chroma_w: int = 0, chroma_h: int = 0, has_alpha: bool = False) -> str:
    """Build the FFmpeg name of a planar little-endian format."""

    suffix = "" if bit_depth <= 8 else f"{bit_depth}le"
    if family == "yuv":
        try:
            sub = _SUBSAMPLING_NAMES[(chroma_w, chroma_h)]
        except KeyError as exc:  # pragma: no cover - guarded by make_pixfmt
            raise ValueError(f"unsupported chroma subsampling {(chroma_w, chroma_h)}") from exc
        return f"yuv{'a' if has_alpha else ''}{sub}p{suffix}"
    if family == "rgb":
        return f"gbr{'a' if has_alpha else ''}p{suffix}"
    if family == "gray":
        if has_alpha:
            raise ValueError("gray + alpha is not a supported working format")
        return f"gray{suffix}"
    raise ValueError(f"unknown pixel format family: {family!r}")


def make_pixfmt(family: str, bit_depth: int, chroma_w: int = 0, chroma_h: int = 0, has_alpha: bool = False) -> PixFmt:
    if bit_depth not in (8, 9, 10, 12, 14, 16):
        raise ValueError(f"unsupported bit depth: {bit_depth}")
    if family != "yuv":
        chroma_w = chroma_h = 0
    if chroma_w not in (0, 1) or chroma_h not in (0, 1):
        raise ValueError(f"unsupported chroma subsampling {(chroma_w, chroma_h)}")
    return PixFmt(
        name=pixfmt_name(family, bit_depth, chroma_w, chroma_h, has_alpha),
        family=family,
        bit_depth=bit_depth,
        chroma_w=chroma_w,
        chroma_h=chroma_h,
        has_alpha=has_alpha,
    )


def parse_working_pixfmt(name: str) -> PixFmt:
    """Parse the name of a *working* format (planar little-endian) back into a :class:`PixFmt`.

    Accepts e.g. ``yuv420p``, ``yuv422p10le``, ``yuva444p10le``, ``gbrp``, ``gbrap12le``, ``gray16le``.
    Raises ``ValueError`` for anything else (packed, big-endian, ``yuvj*`` ...); those must first be
    mapped with ``app.processing.planes.working_format_for``.
    """

    n = name.strip().lower()
    depth = 8
    body = n
    if n.endswith("le"):
        digits = ""
        i = len(n) - 3
        while i >= 0 and n[i].isdigit():
            digits = n[i] + digits
            i -= 1
        if digits:
            depth = int(digits)
            body = n[: i + 1]
    for (cw, ch), sub in _SUBSAMPLING_NAMES.items():
        if body == f"yuv{sub}p":
            return make_pixfmt("yuv", depth, cw, ch, False)
        if body == f"yuva{sub}p":
            return make_pixfmt("yuv", depth, cw, ch, True)
    if body == "gbrp":
        return make_pixfmt("rgb", depth, 0, 0, False)
    if body == "gbrap":
        return make_pixfmt("rgb", depth, 0, 0, True)
    if body == "gray":
        return make_pixfmt("gray", depth, 0, 0, False)
    raise ValueError(f"not a planar little-endian working format: {name!r}")


# --------------------------------------------------------------------------- colour description


@dataclass(frozen=True)
class ColorInfo:
    """Colour tags exactly as reported for the source stream (FFmpeg names, ``unknown`` if absent)."""

    range: str = "unknown"            # "tv" (limited) | "pc" (full) | "unknown"
    space: str = "unknown"            # matrix coefficients: bt709, smpte170m, bt470bg, bt2020nc, ...
    primaries: str = "unknown"        # bt709, bt2020, smpte432, ...
    transfer: str = "unknown"         # bt709, smpte2084 (PQ), arib-std-b67 (HLG), iec61966-2-1 ...
    chroma_location: str = "unspecified"

    def to_dict(self) -> Dict[str, str]:
        return {
            "range": self.range,
            "space": self.space,
            "primaries": self.primaries,
            "transfer": self.transfer,
            "chroma_location": self.chroma_location,
        }

    @property
    def is_hdr(self) -> bool:
        return self.transfer in ("smpte2084", "arib-std-b67")


#: Luma coefficients (Kr, Kb) per matrix name.  Kg = 1 - Kr - Kb.
MATRIX_COEFFICIENTS: Dict[str, Tuple[float, float]] = {
    "bt709": (0.2126, 0.0722),
    "bt470bg": (0.299, 0.114),
    "smpte170m": (0.299, 0.114),
    "bt601": (0.299, 0.114),
    "smpte240m": (0.212, 0.087),
    "fcc": (0.30, 0.11),
    "bt2020nc": (0.2627, 0.0593),
    "bt2020c": (0.2627, 0.0593),   # constant-luminance treated as non-constant (documented approximation)
    "bt2020": (0.2627, 0.0593),
}

_UNSPECIFIED = ("unknown", "unspecified", "reserved", "", "none", "2")


@dataclass(frozen=True)
class ResolvedColor:
    """The colour interpretation actually used for conversions.

    ``matrix`` / ``full_range`` are what the pipeline uses to convert between the working planes and
    R'G'B' (for model input, previews, edge blending and the background colour).  ``assumed`` lists
    which of them were guessed because the source did not carry the tag; the output keeps the
    source's tags verbatim (unknown stays unknown) so every player treats source and output alike.
    """

    matrix: str
    kr: float
    kb: float
    full_range: bool
    assumed: Tuple[str, ...] = ()

    @property
    def kg(self) -> float:
        return 1.0 - self.kr - self.kb


def resolve_color(
    color: ColorInfo,
    fmt: PixFmt,
    width: int,
    height: int,
    full_range_hint: Optional[bool] = None,
) -> ResolvedColor:
    """Decide matrix and range for a stream, applying the standard player heuristics for missing tags.

    * matrix: the tagged one if we know its coefficients; otherwise BT.2020 when the primaries say
      so, else BT.709 for HD (height >= 720 or width >= 1280) and BT.601 for SD.
    * range: the tagged one; otherwise the ``yuvj*`` hint from the decoder; otherwise limited for
      YUV / gray and full for RGB.
    """

    assumed = []
    space = (color.space or "unknown").lower()
    if space in MATRIX_COEFFICIENTS:
        matrix = space
    else:
        if (color.primaries or "").lower() == "bt2020":
            matrix = "bt2020nc"
        elif height >= 720 or width >= 1280:
            matrix = "bt709"
        else:
            matrix = "smpte170m"
        if fmt.family == "yuv":
            assumed.append("matrix")
    rng = (color.range or "unknown").lower()
    if rng in ("pc", "jpeg", "full"):
        full = True
    elif rng in ("tv", "mpeg", "limited"):
        full = False
    else:
        if full_range_hint is not None:
            full = bool(full_range_hint)
        else:
            full = fmt.family == "rgb"
        if fmt.family != "rgb" and full_range_hint is None:
            assumed.append("range")
    if fmt.family == "rgb":
        full = True if rng in _UNSPECIFIED else full
    kr, kb = MATRIX_COEFFICIENTS[matrix]
    return ResolvedColor(matrix=matrix, kr=kr, kb=kb, full_range=full, assumed=tuple(assumed))


def quantization(fmt: PixFmt, full_range: bool) -> Tuple[float, float, float, float]:
    """Return ``(luma_offset, luma_scale, chroma_offset, chroma_scale)`` in code values.

    A normalised luma ``y`` in 0..1 maps to ``luma_offset + luma_scale * y`` and a normalised chroma
    ``c`` in -0.5..0.5 maps to ``chroma_offset + chroma_scale * c``.
    For RGB / gray-as-RGB planes use the luma pair.
    """

    shift = fmt.bit_depth - 8
    if full_range:
        maxv = float((1 << fmt.bit_depth) - 1)
        return 0.0, maxv, float(1 << (fmt.bit_depth - 1)), maxv
    return float(16 << shift), float(219 << shift), float(128 << shift), float(224 << shift)
