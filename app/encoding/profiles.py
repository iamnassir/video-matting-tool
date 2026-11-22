"""Codec profiles: turn "codec + container + source properties" into a complete FFmpeg encode plan.

An :class:`EncodePlan` answers three questions for one deliverable:

1. **Which pixel format must the compositor deliver?**  The source's working format whenever the
   encoder accepts it natively - then opaque pixels reach the encoder bit-exact (invariant I4).
   Otherwise the nearest format that loses nothing (more bits, finer chroma), otherwise the closest
   lossy one, always with a user-facing warning.  The table of formats an encoder accepts is read
   from the installed FFmpeg (``ffmpeg -h encoder=NAME``), not from memory: the encode command
   forbids every implicit conversion (``-pix_fmt +``), so a format the encoder does not take is a
   hard FFmpeg error and the plan must always be acceptable.
2. **How is the encoder configured?**  Rate control and speed from the quality preset, unless
   ``EncodingConfig`` overrides them.
3. **Which options keep timestamps and colour exact?**  See :meth:`EncodePlan.command`.

Validated facts behind the command line (FFmpeg 8.1, re-checked with PyAV 17.1):

* ``-fps_mode passthrough -enc_time_base demux`` - without them FFmpeg re-times variable frame
  rate video to ``1 / frame rate`` and drops or duplicates frames.
* ``-copyts`` keeps a non-zero start; MP4 / MOV additionally need ``-movie_timescale`` (the start
  offset is stored in movie time units) and get ``-video_track_timescale`` so the track keeps the
  source's time scale.  Matroska / WebM always round to 1 ms.
* The NUT pipe carries no colour information.  Tags are therefore declared as **input** options
  (they only label the frames).  As *output* options ``-color_primaries`` / ``-color_trc`` are
  ignored and ``-color_range`` / ``-colorspace`` convert pixels when they differ from the input;
  repeating the two with the *same* values is required for encoders that advertise limited range
  only (FFV1, ProRes) and harmless elsewhere.
* ``-pix_fmt +`` disables all automatic filter-graph conversions.

Sample aspect ratio is not part of this API: plans assume square pixels (the muxer restores a
non-square container aspect ratio for outputs that keep the source size).
"""

from __future__ import annotations

import logging
import math
import re
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.config import (
    CODECS,
    CONTAINER_EXTENSIONS,
    QUALITY_PRESETS,
    EncodingConfig,
    QualityPreset,
    VideoCodec,
)
from ..core.errors import ConfigError, EncoderUnavailableError, VmtError
from ..core.pixfmt import ColorInfo, PixFmt, ResolvedColor, make_pixfmt
from ..processing.planes import working_format_for
from .ffmpeg import FfmpegTools, ffmpeg_query

log = logging.getLogger(__name__)

__all__ = [
    "EncodePlan",
    "MUXER_NAMES",
    "PACKED_RGB8",
    "PURPOSES",
    "encoder_pix_fmts",
    "parse_bitrate",
    "pipe_layout",
    "plan_encoding",
]

PURPOSES: Tuple[str, ...] = ("main", "alpha_matte", "prores_4444", "green_screen", "preview")

#: FFmpeg muxer per container.  Given explicitly (``-f``) so the output file may have any name.
MUXER_NAMES: Dict[str, str] = {"mp4": "mp4", "mov": "mov", "mkv": "matroska", "webm": "webm"}

#: Largest MP4 / MOV time scale (a signed 32-bit field).
_MAX_TIMESCALE = (1 << 31) - 1

#: Packed 8-bit RGB layouts the NUT pipe can be fed with: (bytes per pixel, index of R, G, B, A).
#: FFV1 has no planar 8-bit RGB input (``gbrp`` only exists from 9 bits up) but accepts ``bgr0`` /
#: ``bgra``; interleaving the planes moves no sample value, so the round trip stays bit-exact.
PACKED_RGB8: Dict[str, Tuple[int, int, int, int, Optional[int]]] = {
    "bgr0": (4, 2, 1, 0, None),
    "bgra": (4, 2, 1, 0, 3),
    "rgb0": (4, 0, 1, 2, None),
    "rgba": (4, 0, 1, 2, 3),
}

_SUBSAMPLING = {(1, 1): "4:2:0", (1, 0): "4:2:2", (0, 0): "4:4:4", (0, 1): "4:4:0"}

# --------------------------------------------------------------------------- encoder formats

#: Used only if ``ffmpeg -h encoder=NAME`` cannot be parsed (a warning is logged).  Deliberately
#: small: formats every build of these encoders has accepted for years.
_FALLBACK_PIX_FMTS: Dict[str, Tuple[str, ...]] = {
    "libx264": ("yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le", "yuv444p10le", "gray"),
    "libx265": ("yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le", "yuv444p10le", "gray"),
    "h264_nvenc": ("yuv420p", "nv12", "yuv444p"),
    "hevc_nvenc": ("yuv420p", "nv12", "p010le", "yuv444p"),
    "prores_ks": ("yuv422p10le", "yuv444p10le", "yuva444p10le"),
    "ffv1": ("yuv420p", "yuva420p", "yuv422p", "yuva422p", "yuv444p", "yuva444p", "yuv420p10le", "yuv422p10le",
             "yuv444p10le", "yuva420p10le", "yuva422p10le", "yuva444p10le", "yuv420p16le", "yuv422p16le",
             "yuv444p16le", "gray", "gray16le", "bgr0", "bgra", "gbrp10le", "gbrp16le", "gbrap16le"),
    "libvpx-vp9": ("yuv420p", "yuva420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le", "yuv444p10le"),
}

_PIX_FMT_CACHE: Dict[Tuple[str, str], Tuple[str, ...]] = {}
_PIX_FMT_LOCK = threading.Lock()
_PIX_FMT_LINE = re.compile(r"^\s*Supported pixel formats:\s*(.+?)\s*$", re.MULTILINE)


def encoder_pix_fmts(tools: FfmpegTools, encoder: str) -> Tuple[str, ...]:
    """Pixel formats *encoder* accepts natively, as listed by the installed FFmpeg (cached)."""

    key = (tools.ffmpeg, encoder)
    with _PIX_FMT_LOCK:
        cached = _PIX_FMT_CACHE.get(key)
    if cached is not None:
        return cached
    names: Tuple[str, ...] = ()
    try:
        text = ffmpeg_query(tools, ["-h", f"encoder={encoder}"])
        match = _PIX_FMT_LINE.search(text)
        if match:
            names = tuple(dict.fromkeys(match.group(1).split()))
    except VmtError as exc:
        log.warning("Could not query the pixel formats of %s: %s", encoder, exc.message)
    if not names:
        names = _FALLBACK_PIX_FMTS.get(encoder, ())
        log.warning("FFmpeg did not list the pixel formats of %s; using the built-in list %s.", encoder, list(names))
    with _PIX_FMT_LOCK:
        _PIX_FMT_CACHE[key] = names
    return names


@dataclass(frozen=True)
class _Candidate:
    """A way to feed an encoder: the working format to composite in and the layout written to the pipe."""

    fmt: PixFmt
    pipe: str
    rank: int        # 0 planar (fed as is), 1 semi-planar repack, 2 packed repack


def pipe_layout(pipe_pix_fmt: str) -> Optional[Tuple[PixFmt, str]]:
    """How frames of *pipe_pix_fmt* are produced from planar working planes.

    Returns ``(working format, kind)`` with kind ``"planar"`` (the planes are written as they are),
    ``"semiplanar"`` (``planes.to_semiplanar``: nv12, p010le ...) or ``"packed"`` (8-bit RGB
    interleaving, see :data:`PACKED_RGB8`); ``None`` for layouts this tool cannot produce exactly.
    """

    name = pipe_pix_fmt.strip().lower()
    if name in PACKED_RGB8:
        return make_pixfmt("rgb", 8, 0, 0, PACKED_RGB8[name][4] is not None), "packed"
    try:
        working = working_format_for(name)
    except VmtError:
        return None
    if working.method == "direct":
        return working.fmt, "planar"
    if working.method == "semiplanar" and working.exact:
        return working.fmt, "semiplanar"
    return None


_RANKS = {"planar": 0, "semiplanar": 1, "packed": 2}


def _candidates(tools: FfmpegTools, encoder: str) -> List[_Candidate]:
    out: List[_Candidate] = []
    for name in encoder_pix_fmts(tools, encoder):
        layout = pipe_layout(name)
        if layout is not None:
            out.append(_Candidate(layout[0], name.strip().lower(), _RANKS[layout[1]]))
    return out


def _allowed_by_default(codec: VideoCodec, fmt: PixFmt) -> bool:
    """Formats a codec is given without an explicit ``encoding.pix_fmt`` request.

    Narrower than what the encoder lists where the wider set is unsafe by default:

    * NVENC accepts RGB and 4:2:2 / 16-bit layouts, but converts RGB inside the GPU with an
      uncontrolled matrix, supports 4:2:2 only on the newest cards and cannot encode 10-bit H.264
      on most of them - an unsupported choice fails only when the encoder starts.
    * The two ProRes flavours are defined by their chroma layout.
    """

    chroma = (fmt.chroma_w, fmt.chroma_h)
    if codec is VideoCodec.H264:
        return fmt.family in ("yuv", "gray")
    if codec is VideoCodec.H264_NVENC:
        return fmt.family == "yuv" and fmt.bit_depth == 8 and chroma in ((1, 1), (0, 0))
    if codec is VideoCodec.HEVC_NVENC:
        return fmt.family == "yuv" and (
            (fmt.bit_depth == 8 and chroma in ((1, 1), (0, 0))) or (fmt.bit_depth == 10 and chroma == (1, 1))
        )
    if codec is VideoCodec.PRORES_422_HQ:
        return fmt.family == "yuv" and chroma == (1, 0)
    if codec is VideoCodec.PRORES_4444:
        return fmt.family == "yuv" and chroma == (0, 0)
    if codec is VideoCodec.VP9:
        return fmt.family in ("yuv", "rgb")
    return True      # H265 (yuv / gray / rgb), FFV1 (everything it lists)


def _distance(src: PixFmt, dst: PixFmt) -> Tuple[int, int, int, int, int, int]:
    """Ordering key: how far *dst* is from *src* (alpha ignored); smaller is better.

    ``(family change, bits lost, chroma resolution lost, different family, chroma gained, bits
    gained)``.  Losing bit depth ranks worse than losing chroma resolution (banding versus the
    subsampling every delivery format uses anyway); among lossless choices more bits rank better
    than finer chroma because a depth increase is an exact shift while chroma has to be
    interpolated.  Grey to Y'CbCr only adds neutral chroma and therefore counts as lossless.
    """

    same = src.family == dst.family
    grey_to_yuv = src.family == "gray" and dst.family == "yuv"
    family_change = 0 if (same or grey_to_yuv) else 1
    depth_loss = max(0, src.bit_depth - dst.bit_depth)
    depth_gain = max(0, dst.bit_depth - src.bit_depth)
    chroma_loss = chroma_gain = 0
    if src.family == "yuv" and dst.family == "yuv":
        chroma_loss = max(0, dst.chroma_w - src.chroma_w) + max(0, dst.chroma_h - src.chroma_h)
        chroma_gain = max(0, src.chroma_w - dst.chroma_w) + max(0, src.chroma_h - dst.chroma_h)
    elif grey_to_yuv:
        chroma_gain = 2 - (dst.chroma_w + dst.chroma_h)        # constant chroma: prefer the compact 4:2:0
    elif src.family == "rgb" and dst.family == "yuv":
        chroma_loss = dst.chroma_w + dst.chroma_h              # RGB is full resolution in every channel
    elif src.family == "yuv" and dst.family == "gray":
        chroma_loss = 3                                        # all colour gone
    return (family_change, depth_loss, chroma_loss, 0 if same else 1, chroma_gain, depth_gain)


def _same_layout(a: PixFmt, b: PixFmt) -> bool:
    return (a.family, a.bit_depth, a.chroma_w, a.chroma_h) == (b.family, b.bit_depth, b.chroma_w, b.chroma_h)


def _is_lossless(src: PixFmt, dst: PixFmt) -> bool:
    family_change, depth_loss, chroma_loss = _distance(src, dst)[:3]
    return family_change == 0 and depth_loss == 0 and chroma_loss == 0


def _pick(reference: PixFmt, pool: Sequence[_Candidate]) -> _Candidate:
    return min(pool, key=lambda c: _distance(reference, c.fmt) + (c.rank,))


def _describe_loss(src: PixFmt, dst: PixFmt) -> str:
    parts: List[str] = []
    if src.family == "rgb" and dst.family != "rgb":
        parts.append("RGB is converted to Y'CbCr")
    elif src.family != "rgb" and dst.family == "rgb":
        parts.append("Y'CbCr is converted to RGB")
    elif src.family == "yuv" and dst.family == "gray":
        parts.append("the colour information is dropped")
    if dst.bit_depth < src.bit_depth:
        parts.append(f"the bit depth is reduced from {src.bit_depth} to {dst.bit_depth} bits")
    if src.family == "yuv" and dst.family == "yuv" and (dst.chroma_w > src.chroma_w or dst.chroma_h > src.chroma_h):
        parts.append(
            f"the chroma resolution is reduced from {_SUBSAMPLING[(src.chroma_w, src.chroma_h)]} "
            f"to {_SUBSAMPLING[(dst.chroma_w, dst.chroma_h)]}"
        )
    elif src.family == "rgb" and dst.family == "yuv" and dst.is_subsampled:
        parts.append(f"the chroma resolution is reduced to {_SUBSAMPLING[(dst.chroma_w, dst.chroma_h)]}")
    return ", ".join(parts) or "the sample layout changes"


# --------------------------------------------------------------------------- colour tags

_UNKNOWN = ("", "unknown", "unspecified", "reserved", "none", "n/a", "2")

_RANGES = {"tv": "tv", "mpeg": "tv", "limited": "tv", "pc": "pc", "jpeg": "pc", "full": "pc"}

#: ffprobe / libavutil names (and this project's aliases) -> the constant the ffmpeg CLI accepts.
#: The oldest spelling of each constant is used so the command also works with older releases.
_SPACES = {
    "gbr": "rgb", "rgb": "rgb", "bt709": "bt709", "fcc": "fcc", "bt470bg": "bt470bg", "smpte170m": "smpte170m",
    "smpte240m": "smpte240m", "ycgco": "ycgco", "ycocg": "ycgco", "bt2020nc": "bt2020nc", "bt2020_ncl": "bt2020nc",
    "bt2020c": "bt2020c", "bt2020_cl": "bt2020c", "smpte2085": "smpte2085",
    "chroma-derived-nc": "chroma-derived-nc", "chroma-derived-c": "chroma-derived-c", "ictcp": "ictcp",
    "bt601": "smpte170m", "bt2020": "bt2020nc",
}
_PRIMARIES = {
    "bt709": "bt709", "bt470m": "bt470m", "bt470bg": "bt470bg", "smpte170m": "smpte170m", "smpte240m": "smpte240m",
    "film": "film", "bt2020": "bt2020", "smpte428": "smpte428", "smpte428_1": "smpte428", "smpte431": "smpte431",
    "smpte432": "smpte432", "jedec-p22": "jedec-p22", "ebu3213": "jedec-p22",
}
_TRANSFERS = {
    "bt709": "bt709", "gamma22": "gamma22", "bt470m": "gamma22", "gamma28": "gamma28", "bt470bg": "gamma28",
    "smpte170m": "smpte170m", "smpte240m": "smpte240m", "linear": "linear", "log100": "log100", "log": "log100",
    "log316": "log316", "log_sqrt": "log316", "iec61966-2-4": "iec61966-2-4", "iec61966_2_4": "iec61966-2-4",
    "bt1361e": "bt1361e", "bt1361": "bt1361e", "iec61966-2-1": "iec61966-2-1", "iec61966_2_1": "iec61966-2-1",
    "srgb": "iec61966-2-1", "bt2020-10": "bt2020-10", "bt2020_10bit": "bt2020-10", "bt2020-12": "bt2020-12",
    "bt2020_12bit": "bt2020-12", "smpte2084": "smpte2084", "pq": "smpte2084", "smpte428": "smpte428",
    "smpte428_1": "smpte428", "arib-std-b67": "arib-std-b67", "hlg": "arib-std-b67",
}
_CHROMA_LOCATIONS = ("left", "center", "topleft", "top", "bottomleft", "bottom")

#: Order of the tag options on the command line (name of the FFmpeg option without the dash).
_TAG_ORDER = ("color_range", "colorspace", "color_primaries", "color_trc", "chroma_sample_location")
#: The two tags that take part in filter-graph negotiation and are repeated as output options.
_NEGOTIATED_TAGS = ("color_range", "colorspace")


def _lookup(table: Dict[str, str], value: Optional[str], what: str, unknown: List[str]) -> Optional[str]:
    text = (value or "").strip().lower()
    if text in _UNKNOWN:
        return None
    mapped = table.get(text)
    if mapped is None:
        unknown.append(f"{what} '{value}'")
    return mapped


def _color_tags(
    purpose: str,
    fmt: PixFmt,
    source_fmt: PixFmt,
    color: ColorInfo,
    resolved: ResolvedColor,
) -> Tuple[Dict[str, str], List[str]]:
    """The tags to declare for frames of *fmt*, and the source tags FFmpeg has no name for.

    Rules: source tags are carried verbatim and unknown stays unknown - except the range, which is
    declared whenever leaving it out would misdescribe the samples (full-range data, and every grey
    or RGB layout, for which FFmpeg otherwise assumes full range).  Where the samples are *not* the
    source's kind any more the tags describe what the compositor produced instead:

    * matte: limited-range Y'CbCr / full-range grey, BT.709;
    * RGB source rendered to Y'CbCr: limited range with the matrix used for the conversion;
    * Y'CbCr source rendered to RGB: full range, identity matrix.
    """

    unknown: List[str] = []
    tags: Dict[str, str] = {}
    if purpose == "alpha_matte":
        tags["color_range"] = "tv" if fmt.family == "yuv" else "pc"
        tags["colorspace"] = "rgb" if fmt.family == "rgb" else "bt709"
        tags["color_primaries"] = "bt709"
        tags["color_trc"] = "bt709"
        return tags, unknown

    source_is_rgb = source_fmt.family == "rgb"
    target_is_rgb = fmt.family == "rgb"
    declared_range = _RANGES.get((color.range or "").strip().lower())
    if source_is_rgb == target_is_rgb:
        if declared_range is not None:
            tags["color_range"] = declared_range
        elif fmt.family in ("gray", "rgb"):
            tags["color_range"] = "pc" if resolved.full_range else "tv"
        elif resolved.full_range:
            tags["color_range"] = "pc"
        space = _lookup(_SPACES, color.space, "matrix", unknown)
        if target_is_rgb and space not in (None, "rgb"):
            space = None               # a Y'CbCr matrix tag on RGB samples would be a contradiction
        if space is not None:
            tags["colorspace"] = space
    elif target_is_rgb:
        tags["color_range"] = "pc"
        tags["colorspace"] = "rgb"
    else:
        tags["color_range"] = "tv"
        matrix = _SPACES.get((resolved.matrix or "").lower())
        if matrix is not None and matrix != "rgb":
            tags["colorspace"] = matrix
    primaries = _lookup(_PRIMARIES, color.primaries, "primaries", unknown)
    if primaries is not None:
        tags["color_primaries"] = primaries
    transfer = _lookup(_TRANSFERS, color.transfer, "transfer", unknown)
    if transfer is not None:
        tags["color_trc"] = transfer
    location = (color.chroma_location or "").strip().lower()
    if (
        location in _CHROMA_LOCATIONS
        and fmt.family == "yuv" and source_fmt.family == "yuv" and fmt.is_subsampled
        and (fmt.chroma_w, fmt.chroma_h) == (source_fmt.chroma_w, source_fmt.chroma_h)
    ):
        # Only meaningful while the chroma grid is the source's own; resampled chroma has the
        # siting of the compositor's box filter and stays untagged.
        tags["chroma_sample_location"] = location
    return tags, unknown


# --------------------------------------------------------------------------- rate control tables

#: Per encoder effort 0..3 (QualityPresetInfo.encoder_effort): (speed preset, constant-quality value).
_X264_LEVELS = (("veryfast", 20), ("fast", 18), ("medium", 16), ("slow", 14))
_X265_LEVELS = (("faster", 24), ("fast", 21), ("medium", 19), ("slow", 17))
_NVENC_LEVELS = (("p3", 24), ("p4", 21), ("p5", 19), ("p7", 17))
#: (CRF, -cpu-used) for libvpx-vp9 with ``-deadline good``.
_VP9_LEVELS = ((36, 5), (31, 4), (24, 3), (18, 2))

_X26X_PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow", "placebo")
_NVENC_PRESETS = ("p1", "p2", "p3", "p4", "p5", "p6", "p7", "slow", "medium", "fast")
_VP9_DEADLINES = ("good", "best", "realtime")

#: Largest constant-quality value per encoder family (x264 allows up to 63 only for high bit depth).
_MAX_QUALITY = {"x264": 51, "x265": 51, "nvenc": 51, "vp9": 63}

_PREVIEW_CRF = 26

#: Bits per pixel and frame at the reference quality, deliberately on the generous side (noisy
#: hand-held footage), used by :meth:`EncodePlan.estimated_bytes`: (reference quality, bpp, step).
#: The estimate doubles every ``step`` quality points below the reference.
_BPP_MODEL = {
    "x264": (18.0, 0.30, 6.0),
    "x265": (21.0, 0.20, 6.0),
    "nvenc_h264": (21.0, 0.34, 6.0),
    "nvenc_hevc": (21.0, 0.26, 6.0),
    "vp9": (31.0, 0.22, 10.0),
}
#: Nominal ProRes data rates in bits per pixel and frame (220 / 330 Mbit/s at 1920x1080, 29.97 fps).
_PRORES_BPP = {"hq": 3.54, "4444": 5.31, "4444_alpha": 6.9}


def parse_bitrate(text: str) -> int:
    """``"20M"`` / ``"8000k"`` / ``"5000000"`` -> bits per second.

    FFmpeg reads a lower-case ``m`` suffix as *milli*; converting here makes the value unambiguous.
    """

    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*", str(text))
    if match is None:
        raise ConfigError(f"Invalid bit rate {text!r}.", hint="Use a value such as 8000k, 20M or 5000000.")
    factor = {"": 1, "k": 1_000, "m": 1_000_000}[match.group(2).lower()]
    value = int(round(float(match.group(1)) * factor))
    if value <= 0:
        raise ConfigError(f"Invalid bit rate {text!r}: it must be greater than zero.")
    return value


# --------------------------------------------------------------------------- the plan


@dataclass
class EncodePlan:
    """Everything needed to encode one deliverable through the NUT pipe (see module docstring)."""

    purpose: str
    codec: VideoCodec
    container: str
    extension: str                    # ".mp4" ...
    encoder: str                      # ffmpeg encoder name
    fmt: PixFmt                       # planar working format the compositor must deliver for this target
    pipe_pix_fmt: str                 # what is written into the NUT pipe: fmt.name, or a repack of it
    width: int
    height: int
    time_base: Fraction
    frame_rate: Optional[Fraction]
    input_args: List[str]             # options before "-f nut -i pipe:0" (colour tags as INPUT options)
    video_args: List[str]             # encoder, rate control, preset, profile, tags
    output_args: List[str]            # passthrough timing, the "-pix_fmt +" guard, repeated tags, time scales
    lossless: bool                    # the encoder reproduces the planes it is given bit-exactly
    color_tags: Dict[str, str]
    warnings: List[str]               # user-facing notes (format changes, limited player support, ...)
    description: str
    #: parameters of the size estimate (``kind``: quality | bitrate | prores | ffv1)
    estimate: Dict[str, Any] = field(default_factory=dict)

    def estimated_bytes(
        self,
        duration_seconds: float,
        frames: int,
        raw_frame_bytes: int,
        source_bit_rate: Optional[int],
    ) -> float:
        """A deliberately generous but sane size estimate for the encoded video (bytes).

        Used to refuse a job before it starts when the disk clearly cannot hold the result, so it
        errs on the high side without being absurd:

        * constant-quality codecs: bits per pixel for the chosen quality, steered between half and
          twice that figure by the source's own bit rate (a clean source rarely grows past 1.5x);
        * target bit rate: the requested rate plus 10 %;
        * ProRes: its nominal data rate scaled by the pixel count;
        * FFV1: three quarters of the raw frame size;
        * mattes and the preview are small.

        ``raw_frame_bytes`` (raw size of one source-format frame) is only used when the plan's own
        frame size is unknown.  The result never decreases with *duration_seconds* or *frames*.
        """

        seconds = max(0.0, float(duration_seconds or 0.0))
        count = max(0, int(frames or 0))
        rate = float(self.frame_rate) if self.frame_rate else 30.0
        if count == 0 and seconds > 0.0:
            count = int(math.ceil(seconds * rate))
        if seconds == 0.0 and count > 0:
            seconds = count / rate
        pixels = max(1, int(self.width) * int(self.height))
        info = self.estimate or {}
        kind = info.get("kind", "quality")
        if kind == "bitrate":
            bits = float(info.get("bps", 8_000_000)) * 1.1 * seconds
        elif kind == "prores":
            bits = float(info.get("bpp", _PRORES_BPP["hq"])) * 1.1 * pixels * count
        elif kind == "ffv1":
            try:
                raw = self.fmt.frame_bytes(int(self.width), int(self.height))
            except Exception:       # pragma: no cover - defensive: fall back to the caller's figure
                raw = 0
            raw = raw or max(0, int(raw_frame_bytes or 0))
            bits = 0.75 * raw * 8.0 * count
        else:
            bits = float(info.get("bpp", 0.3)) * pixels * count
            if source_bit_rate and source_bit_rate > 0 and info.get("follow_source", False):
                bits = min(max(0.5 * bits, 1.5 * float(source_bit_rate) * seconds), 2.0 * bits)
        container_overhead = 64.0 * count + 65536.0
        return bits / 8.0 + container_overhead

    def command(self, tools: FfmpegTools, output: Path) -> List[str]:
        """The complete ffmpeg argv for the NUT-pipe encode into *output* (video only).

        ``-copyts`` is a global option: the absolute timestamps of the pipe are kept, including a
        start that is not zero.  The muxer is named explicitly so *output* may have any extension.
        """

        return [
            tools.ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "warning",
            "-copyts",
            *self.input_args,
            "-f", "nut", "-i", "pipe:0",
            "-map", "0:v:0", "-an", "-sn", "-dn",
            *self.video_args,
            *self.output_args,
            "-f", MUXER_NAMES[self.container], str(output),
        ]


_ALTERNATIVES: Dict[VideoCodec, Tuple[VideoCodec, ...]] = {
    VideoCodec.H264_NVENC: (VideoCodec.H264, VideoCodec.H265),
    VideoCodec.HEVC_NVENC: (VideoCodec.H265, VideoCodec.H264),
    VideoCodec.H265: (VideoCodec.H264, VideoCodec.HEVC_NVENC),
    VideoCodec.H264: (VideoCodec.H264_NVENC, VideoCodec.H265),
    VideoCodec.PRORES_422_HQ: (VideoCodec.FFV1, VideoCodec.H264),
    VideoCodec.PRORES_4444: (VideoCodec.FFV1, VideoCodec.VP9),
    VideoCodec.FFV1: (VideoCodec.PRORES_4444, VideoCodec.H264),
    VideoCodec.VP9: (VideoCodec.H264, VideoCodec.H265),
}


def _require_encoder(tools: FfmpegTools, codec: VideoCodec) -> str:
    encoder = CODECS[codec].ffmpeg_encoder
    if tools.has_encoder(encoder):
        return encoder
    options = _ALTERNATIVES.get(codec, ())
    available = [c for c in options if tools.has_encoder(CODECS[c].ffmpeg_encoder)]
    alternative = (available or list(options) or [VideoCodec.H264])[0]
    raise EncoderUnavailableError(
        f"The FFmpeg build in use ({tools.version}) has no '{encoder}' encoder, which codec '{codec.value}' needs.",
        hint=f"Use codec '{alternative.value}' instead of '{codec.value}', or install an FFmpeg build that includes "
        f"{encoder}" + (" (NVENC also needs an NVIDIA GPU with a current driver)." if "nvenc" in encoder else "."),
        details={"codec": codec.value, "encoder": encoder, "alternative": alternative.value,
                 "available": sorted(c.value for c in VideoCodec if tools.has_encoder(CODECS[c].ffmpeg_encoder))},
    )


def _resolve_override(name: str, candidates: Sequence[_Candidate], encoder: str, need_alpha: bool) -> _Candidate:
    """Find the candidate for an explicit ``encoding.pix_fmt`` (a working format or a pipe layout)."""

    wanted = name.strip().lower()
    matches = [c for c in candidates if c.pipe == wanted]
    if not matches:
        layout = pipe_layout(wanted)
        target: Optional[PixFmt] = layout[0] if layout is not None else None
        if target is None:
            try:
                working = working_format_for(wanted)
                target = working.fmt if working.exact else None      # e.g. yuvj420p -> yuv420p
            except VmtError:
                target = None
        if target is not None:
            matches = sorted((c for c in candidates if c.fmt.name == target.name), key=lambda c: c.rank)
    usable = sorted({c.fmt.name for c in candidates if c.fmt.has_alpha == need_alpha})
    if not matches:
        raise ConfigError(
            f"Pixel format '{name}' cannot be used with the {encoder} encoder.",
            hint="Formats this encoder accepts: " + ", ".join(usable) + ". Leave pix_fmt unset to keep the source format.",
            details={"pix_fmt": name, "encoder": encoder, "supported": usable},
        )
    chosen = matches[0]
    if need_alpha and not chosen.fmt.has_alpha:
        raise ConfigError(
            f"Pixel format '{name}' has no alpha channel, but a transparent output was requested.",
            hint="Formats with alpha for this encoder: " + (", ".join(usable) or "none") + ".",
            details={"pix_fmt": name, "encoder": encoder, "supported": usable},
        )
    return chosen


def _quality_bpp(model: str, quality: float, fmt: PixFmt) -> float:
    reference, bpp, step = _BPP_MODEL[model]
    value = bpp * (2.0 ** ((reference - float(quality)) / step))
    if fmt.family == "yuv" and not fmt.is_subsampled or fmt.family == "rgb":
        value *= 1.5
    elif fmt.family == "yuv" and (fmt.chroma_w, fmt.chroma_h) != (1, 1):
        value *= 1.25
    if fmt.has_alpha:
        value *= 1.3
    return value


def plan_encoding(
    *,
    purpose: str,
    codec: VideoCodec,
    container: str,
    with_alpha: bool,
    source_fmt: PixFmt,
    color: ColorInfo,
    resolved_color: ResolvedColor,
    width: int,
    height: int,
    frame_rate: Optional[Fraction],
    time_base: Fraction,
    is_vfr: bool,
    encoding: EncodingConfig,
    quality: QualityPreset,
    tools: FfmpegTools,
) -> EncodePlan:
    """Plan the encode of one deliverable.

    ``purpose``:

    * ``"main"`` / ``"green_screen"`` - the composited picture.  ``encoding.pix_fmt``, ``crf``,
      ``bitrate`` / ``max_bitrate``, ``preset`` and ``extra_args`` apply (an unusable value is a
      :class:`ConfigError` for the main output and is ignored for the green-screen copy).
    * ``"alpha_matte"`` - a grey matte video: 4:2:0 at 8 or 10 bits for H.264 / H.265 / VP9,
      ``yuv422p10le`` for ProRes 422 HQ, a grey format for FFV1; quality from the preset only.
    * ``"prores_4444"`` - the transparent ProRes 4444 export.
    * ``"preview"`` - always libx264 ``yuv420p``, fastest preset, CRF 26, a keyframe every second.

    ``width`` / ``height`` are the coded size of the frames that will be written, ``time_base`` the
    source stream time base the timestamps are expressed in, ``frame_rate`` a hint for the encoder
    (it never affects timestamps).  ``is_vfr`` only shapes a warning text.

    Raises :class:`ConfigError` for impossible requests and :class:`EncoderUnavailableError` when
    the FFmpeg build lacks the encoder.
    """

    if purpose not in PURPOSES:
        raise ConfigError(f"Unknown encode purpose {purpose!r}.", hint=f"Use one of: {', '.join(PURPOSES)}.")
    try:
        codec = VideoCodec(codec)
    except ValueError as exc:
        raise ConfigError(f"Unknown video codec {codec!r}.") from exc
    if purpose == "preview":
        codec, with_alpha = VideoCodec.H264, False
    info = CODECS[codec]
    if container not in MUXER_NAMES:
        raise ConfigError(f"Unknown output container {container!r}.", hint="Use mp4, mov, mkv or webm.")
    if container not in info.containers:
        raise ConfigError(
            f"Codec {codec.value} cannot be stored in a .{container} file.",
            hint=f"Use one of: {', '.join('.' + c for c in info.containers)} or pick another codec.",
        )
    if with_alpha and not info.supports_alpha:
        raise ConfigError(
            f"Codec {codec.value} has no alpha channel; transparent output needs prores_4444, ffv1 or vp9."
        )
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ConfigError(f"Invalid frame size {width}x{height}.")
    time_base = Fraction(time_base)
    if time_base <= 0:
        raise ConfigError(f"Invalid time base {time_base}.")
    rate = Fraction(frame_rate) if frame_rate else None
    if rate is not None and rate <= 0:
        rate = None

    encoder = _require_encoder(tools, codec)
    candidates = _candidates(tools, encoder)
    if not candidates:
        raise EncoderUnavailableError(
            f"The '{encoder}' encoder of this FFmpeg build lists no pixel format this tool can deliver.",
            hint="Install a standard FFmpeg build.",
            details={"encoder": encoder, "pix_fmts": list(encoder_pix_fmts(tools, encoder))},
        )
    effort = min(3, max(0, int(QUALITY_PRESETS[QualityPreset(quality)].encoder_effort)))
    user_settings = purpose in ("main", "green_screen")       # do EncodingConfig overrides apply?
    strict = purpose == "main"                                 # is an unusable override an error?
    warnings: List[str] = []
    source_colour = source_fmt.with_alpha(False)

    # ---- pixel format -----------------------------------------------------------------
    by_alpha = [c for c in candidates if c.fmt.has_alpha == bool(with_alpha)]
    pool = [c for c in by_alpha if _allowed_by_default(codec, c.fmt)]
    if codec is VideoCodec.VP9 and with_alpha:
        # 8-bit 4:2:0 is the only transparent VP9 layout browsers play.
        pool = [c for c in pool if c.fmt.name == "yuva420p"]
    if not pool:
        pool = by_alpha
    if not pool:
        raise ConfigError(
            f"The {encoder} encoder of this FFmpeg build has no pixel format "
            f"{'with' if with_alpha else 'without'} an alpha channel that this tool can deliver.",
            details={"encoder": encoder, "pix_fmts": list(encoder_pix_fmts(tools, encoder))},
        )

    reference = source_colour
    chosen: Optional[_Candidate] = None
    if purpose == "preview":
        preview = [c for c in candidates if c.fmt.name == "yuv420p"]
        if not preview:
            raise EncoderUnavailableError(
                f"The '{encoder}' encoder of this FFmpeg build does not accept yuv420p.",
                details={"encoder": encoder},
            )
        chosen = min(preview, key=lambda c: c.rank)
        reference = chosen.fmt
    elif purpose == "alpha_matte":
        greys = [c for c in pool if c.fmt.family == "gray"]
        if codec is VideoCodec.FFV1 and greys:
            pool = greys
            reference = make_pixfmt("gray", source_fmt.bit_depth)
        else:
            reference = make_pixfmt("yuv", 8 if source_fmt.bit_depth <= 8 else 10, 1, 1)
            colour = [c for c in pool if c.fmt.family == "yuv"]
            pool = colour or pool
        chosen = _pick(reference, pool)
    else:
        if user_settings and encoding.pix_fmt:
            try:
                chosen = _resolve_override(encoding.pix_fmt, candidates, encoder, bool(with_alpha))
            except ConfigError:
                if strict:
                    raise
                chosen = None       # the green-screen copy falls back to the automatic choice
        if chosen is None:
            chosen = _pick(reference, pool)
        if not _same_layout(reference, chosen.fmt):
            if not _is_lossless(reference, chosen.fmt):
                warnings.append(
                    f"{info.label}: the output pixel format is {chosen.fmt.name} instead of the source's "
                    f"{source_colour.name} ({_describe_loss(reference, chosen.fmt)})."
                    + (" Requested with encoding.pix_fmt." if user_settings and encoding.pix_fmt else "")
                )
            elif purpose == "main":
                warnings.append(
                    f"{info.label}: the output pixel format is {chosen.fmt.name} because the encoder does not "
                    f"accept the source's {source_colour.name}. No picture information is lost, but the samples "
                    "are no longer bit-identical to the source."
                )
    fmt = chosen.fmt

    # ---- encoder settings -------------------------------------------------------------
    video_args: List[str] = ["-c:v", encoder]
    estimate: Dict[str, Any] = {"kind": "quality", "follow_source": purpose in ("main", "green_screen")}
    preset_request = (encoding.preset or "").strip() if user_settings else ""
    crf_request = encoding.crf if user_settings else None
    bitrate_bps = parse_bitrate(encoding.bitrate) if (user_settings and encoding.bitrate) else None
    maxrate_bps = parse_bitrate(encoding.max_bitrate) if (user_settings and encoding.max_bitrate) else None
    if bitrate_bps is not None and maxrate_bps is not None and maxrate_bps < bitrate_bps:
        if strict:
            raise ConfigError(
                f"max_bitrate ({encoding.max_bitrate}) is lower than bitrate ({encoding.bitrate}).",
                hint="Raise max_bitrate or leave it unset.",
            )
        maxrate_bps = bitrate_bps

    def bad_setting(message: str, hint: str) -> None:
        if strict:
            raise ConfigError(message, hint=hint)

    def rate_limit_args() -> List[str]:
        if maxrate_bps is None:
            return []
        return ["-maxrate", str(maxrate_bps), "-bufsize", str(2 * maxrate_bps)]

    def quality_value(family: str, default: int) -> int:
        if crf_request is None:
            return default
        limit = 63 if (family == "x264" and fmt.bit_depth > 8) else _MAX_QUALITY[family]
        low = 1 if family == "nvenc" else 0             # NVENC reads -cq 0 as "automatic"
        if not low <= int(crf_request) <= limit:
            bad_setting(
                f"crf {crf_request} is outside the range {low}..{limit} of the {encoder} encoder.",
                f"Use a value between {low} and {limit}, or leave crf unset to follow the quality preset.",
            )
        return min(limit, max(low, int(crf_request)))

    summary = ""
    if purpose == "preview":
        fps = float(rate) if rate is not None else 30.0
        gop = int(min(300, max(1, round(fps))))
        video_args += ["-preset", "ultrafast", "-crf", str(_PREVIEW_CRF), "-g", str(gop)]
        estimate.update(bpp=_quality_bpp("x264", _PREVIEW_CRF, fmt) * 1.4)     # ultrafast spends more bits
        summary = f"CRF {_PREVIEW_CRF}, preset ultrafast"
    elif codec in (VideoCodec.H264, VideoCodec.H265):
        family = "x264" if codec is VideoCodec.H264 else "x265"
        preset, default_quality = (_X264_LEVELS if family == "x264" else _X265_LEVELS)[effort]
        if preset_request:
            if preset_request.lower() in _X26X_PRESETS:
                preset = preset_request.lower()
            else:
                bad_setting(
                    f"Preset '{preset_request}' is not valid for {encoder}.",
                    "Use one of: " + ", ".join(_X26X_PRESETS) + ".",
                )
        video_args += ["-preset", preset]
        if bitrate_bps is not None:
            video_args += ["-b:v", str(bitrate_bps)] + rate_limit_args()
            estimate.update(kind="bitrate", bps=max(bitrate_bps, maxrate_bps or 0))
            summary = f"{bitrate_bps / 1e6:g} Mbit/s, preset {preset}"
        else:
            value = quality_value(family, default_quality)
            video_args += ["-crf", str(value)] + rate_limit_args()
            estimate.update(bpp=_quality_bpp(family, value, fmt))
            summary = f"CRF {value}, preset {preset}"
        if family == "x265":
            video_args += ["-x265-params", "log-level=error"]       # x265 otherwise floods stderr with statistics
            if container in ("mp4", "mov"):
                video_args += ["-tag:v", "hvc1"]                    # the tag Apple players require for HEVC
    elif codec in (VideoCodec.H264_NVENC, VideoCodec.HEVC_NVENC):
        preset, default_quality = _NVENC_LEVELS[effort]
        if preset_request:
            if preset_request.lower() in _NVENC_PRESETS:
                preset = preset_request.lower()
            else:
                bad_setting(
                    f"Preset '{preset_request}' is not valid for {encoder}.",
                    "Use p1 (fastest) .. p7 (best quality).",
                )
        video_args += ["-preset", preset, "-tune", "hq", "-rc", "vbr"]
        model = "nvenc_h264" if codec is VideoCodec.H264_NVENC else "nvenc_hevc"
        if bitrate_bps is not None:
            video_args += ["-b:v", str(bitrate_bps)] + rate_limit_args()
            estimate.update(kind="bitrate", bps=max(bitrate_bps, maxrate_bps or 0))
            summary = f"{bitrate_bps / 1e6:g} Mbit/s, preset {preset}"
        else:
            value = quality_value("nvenc", default_quality)
            video_args += ["-cq", str(value), "-b:v", "0"] + rate_limit_args()
            estimate.update(bpp=_quality_bpp(model, value, fmt))
            summary = f"CQ {value}, preset {preset}"
        # B-frames: exact timestamps were verified for h264_nvenc with two B-frames; hevc_nvenc
        # B-frames are unverified (and unsupported on older GPUs), so they stay off.
        video_args += ["-bf", "2" if codec is VideoCodec.H264_NVENC else "0"]
        if codec is VideoCodec.HEVC_NVENC and container in ("mp4", "mov"):
            video_args += ["-tag:v", "hvc1"]
    elif codec in (VideoCodec.PRORES_422_HQ, VideoCodec.PRORES_4444):
        if codec is VideoCodec.PRORES_422_HQ:
            video_args += ["-profile:v", "3", "-vendor", "apl0"]
            estimate.update(kind="prores", bpp=_PRORES_BPP["hq"])
            summary = "422 HQ"
        else:
            video_args += ["-profile:v", "4", "-vendor", "apl0"]
            if fmt.has_alpha:
                video_args += ["-alpha_bits", "16"]
            estimate.update(kind="prores", bpp=_PRORES_BPP["4444_alpha" if fmt.has_alpha else "4444"])
            summary = "4444" + (" with alpha" if fmt.has_alpha else "")
    elif codec is VideoCodec.FFV1:
        # Level 3 = multi-threaded slices with CRCs; -g 1 = intra only (every frame decodable alone).
        pixels = width * height
        slices = 4 if pixels < 320 * 240 * 3 else 9 if pixels < 1280 * 720 else 16 if pixels < 3840 * 2160 else 24
        context = 1 if effort >= 2 else 0
        coder = 1 if (fmt.bit_depth > 8 or effort >= 2) else 0     # Golomb-Rice is 8-bit only
        video_args += ["-level", "3", "-g", "1", "-slices", str(slices), "-slicecrc", "1",
                       "-coder", str(coder), "-context", str(context)]
        estimate.update(kind="ffv1")
        summary = "lossless"
    elif codec is VideoCodec.VP9:
        default_quality, speed = _VP9_LEVELS[effort]
        deadline = "good"
        if preset_request:
            text = preset_request.lower()
            if text in _VP9_DEADLINES:
                deadline = text
            elif text.isdigit() and 0 <= int(text) <= 8:
                speed = int(text)
            else:
                bad_setting(
                    f"Preset '{preset_request}' is not valid for {encoder}.",
                    "Use a speed 0 (slowest) .. 8, or one of: " + ", ".join(_VP9_DEADLINES) + ".",
                )
        if bitrate_bps is not None:
            video_args += ["-b:v", str(bitrate_bps)] + rate_limit_args()
            estimate.update(kind="bitrate", bps=max(bitrate_bps, maxrate_bps or 0))
            summary = f"{bitrate_bps / 1e6:g} Mbit/s, speed {speed}"
        else:
            value = quality_value("vp9", default_quality)
            video_args += ["-crf", str(value), "-b:v", "0"] + rate_limit_args()      # -b:v 0 = pure constant quality
            estimate.update(bpp=_quality_bpp("vp9", value, fmt))
            summary = f"CRF {value}, speed {speed}"
        video_args += ["-deadline", deadline, "-cpu-used", str(speed), "-row-mt", "1"]
        if fmt.has_alpha:
            video_args += ["-auto-alt-ref", "0"]         # alternate reference frames and the alpha layer do not mix
    if purpose == "alpha_matte" and estimate.get("kind") == "quality":
        estimate["bpp"] = float(estimate.get("bpp", 0.3)) * 0.25             # two-tone pictures compress very well
    if strict and info.rate_control == "none":
        ignored = [n for n, v in (("crf", encoding.crf), ("bitrate", encoding.bitrate), ("preset", encoding.preset)) if v]
        if ignored:
            warnings.append(f"{info.label} has no rate control: the setting(s) {', '.join(ignored)} are ignored.")

    # ---- colour -----------------------------------------------------------------------
    tags, unknown_tags = _color_tags(purpose, fmt, source_fmt, color, resolved_color)
    input_args: List[str] = []
    for key in _TAG_ORDER:
        if key in tags:
            input_args += [f"-{key}", tags[key]]
    if unknown_tags and purpose == "main":
        warnings.append(
            "The source's colour tag(s) " + ", ".join(unknown_tags) + " have no FFmpeg command-line name and are "
            "not written to the output."
        )

    # ---- timestamps, guard, repeated tags, time scales --------------------------------
    output_args: List[str] = ["-fps_mode", "passthrough", "-enc_time_base", "demux", "-pix_fmt", "+"]
    for key in _NEGOTIATED_TAGS:
        if key in tags:
            output_args += [f"-{key}", tags[key]]
    if container in ("mp4", "mov"):
        denominator = time_base.denominator
        if denominator <= _MAX_TIMESCALE:
            output_args += ["-video_track_timescale", str(denominator), "-movie_timescale", str(denominator)]
        elif purpose != "preview":
            warnings.append(
                f"The source time base {time_base} is finer than MP4 / MOV can express; frame times may be "
                "rounded by a fraction of a microsecond."
            )
    elif purpose in ("main", "green_screen") and (Fraction(1, 1000) / time_base).denominator != 1:
        warnings.append(
            f"{'Matroska' if container == 'mkv' else 'WebM'} stores timestamps in milliseconds: frame times are "
            f"rounded by up to 0.5 ms (the source counts in {time_base} s"
            + (", variable frame rate" if is_vfr else "") + "). Use MP4 or MOV for exact timestamps."
        )
    if user_settings and encoding.extra_args:
        output_args += [str(a) for a in encoding.extra_args]

    # ---- notes on compatibility -------------------------------------------------------
    if purpose == "main":
        chroma = (fmt.chroma_w, fmt.chroma_h)
        if codec in (VideoCodec.H264, VideoCodec.H264_NVENC) and fmt.bit_depth > 8:
            warnings.append(
                f"{fmt.bit_depth}-bit H.264 is not played by most hardware decoders and browsers; "
                "H.265 is the usual choice for high bit depth."
            )
        if codec in (VideoCodec.H264, VideoCodec.H265, VideoCodec.H264_NVENC, VideoCodec.HEVC_NVENC, VideoCodec.VP9):
            if fmt.family == "yuv" and chroma != (1, 1):
                warnings.append(
                    f"{info.label} with {_SUBSAMPLING[chroma]} chroma has limited player support (hardware "
                    "decoders and browsers generally expect 4:2:0); set encoding.pix_fmt to a 4:2:0 format for "
                    "maximum compatibility."
                )
            elif fmt.family in ("gray", "rgb"):
                warnings.append(
                    f"{info.label} with {'grey-scale' if fmt.family == 'gray' else 'RGB'} samples ({fmt.name}) has "
                    "limited player support."
                )
        if codec is VideoCodec.VP9 and fmt.bit_depth > 8 and chroma == (1, 1) and fmt.family == "yuv":
            warnings.append("High bit depth VP9 (profile 2) is not played by every browser.")
        if codec is VideoCodec.VP9 and fmt.has_alpha:
            warnings.append(
                "VP9 transparency is shown by browsers; many desktop players and FFmpeg's built-in VP9 decoder "
                "ignore the alpha layer."
            )
    if (
        purpose in ("main", "green_screen", "prores_4444")
        and codec in (VideoCodec.PRORES_422_HQ, VideoCodec.PRORES_4444)
        and tags.get("color_range") == "pc"
    ):
        warnings.append(
            "The source is full range but ProRes files cannot signal that: the samples are stored unchanged and "
            "applications will interpret them as video range (deeper blacks, clipped highlights). Use H.264, "
            "H.265, FFV1 or VP9 to keep a full-range flag."
        )

    description = f"{info.label} [{encoder}] {fmt.name}" + (f", {summary}" if summary else "") + f" in .{container}"
    return EncodePlan(
        purpose=purpose,
        codec=codec,
        container=container,
        extension=CONTAINER_EXTENSIONS[container],
        encoder=encoder,
        fmt=fmt,
        pipe_pix_fmt=chosen.pipe,
        width=width,
        height=height,
        time_base=time_base,
        frame_rate=rate,
        input_args=input_args,
        video_args=video_args,
        output_args=output_args,
        lossless=bool(info.lossless),
        color_tags=dict(tags),
        warnings=warnings,
        description=description,
        estimate=estimate,
    )
