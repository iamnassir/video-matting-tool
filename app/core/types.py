"""Core data types shared by all modules (metadata, timing, frames, subjects).

Conventions
-----------
* **Coded orientation**: decoded frames are in the orientation stored in the bitstream.  Rotation
  metadata is preserved on the output; pixels are never physically rotated.
* **Display orientation**: what a viewer sees after applying ``rotation``.  Subject prompts (clicks,
  boxes, masks) and everything handed to a matting model are in display orientation; the pipeline
  rotates model inputs / mattes with lossless 90-degree turns.
* **Timestamps** are integers in the video stream ``time_base``; they are never converted to floats
  on the way to the encoder.  The pipeline never assumes a constant frame rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pixfmt import ColorInfo, PixFmt

# --------------------------------------------------------------------------- helpers


def frac_to_str(value: Optional[Fraction]) -> Optional[str]:
    if value is None:
        return None
    return f"{value.numerator}/{value.denominator}"


def frac_to_float(value: Optional[Fraction]) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def parse_fraction(value: Any) -> Optional[Fraction]:
    """Parse ffprobe style rationals (``"30000/1001"``, ``"0/0"``, ``"29.97"``, numbers)."""

    if value is None:
        return None
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int,)):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(value).limit_denominator(1_000_000)
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        if "/" in text or ":" in text:
            num, den = text.replace(":", "/").split("/", 1)
            num_i, den_i = int(num), int(den)
            if den_i == 0:
                return None
            return Fraction(num_i, den_i)
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


# --------------------------------------------------------------------------- metadata


@dataclass(frozen=True)
class AudioStreamInfo:
    index: int                              # absolute stream index in the container
    codec: str
    codec_long: str = ""
    profile: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    channel_layout: Optional[str] = None
    sample_fmt: Optional[str] = None
    bit_rate: Optional[int] = None
    time_base: Optional[Fraction] = None
    start_time: Optional[Fraction] = None   # seconds
    duration: Optional[Fraction] = None     # seconds
    nb_frames: Optional[int] = None
    language: Optional[str] = None
    title: Optional[str] = None
    is_default: bool = False
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "codec": self.codec,
            "codec_long": self.codec_long,
            "profile": self.profile,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "channel_layout": self.channel_layout,
            "sample_fmt": self.sample_fmt,
            "bit_rate": self.bit_rate,
            "time_base": frac_to_str(self.time_base),
            "start_time": frac_to_float(self.start_time),
            "duration": frac_to_float(self.duration),
            "nb_frames": self.nb_frames,
            "language": self.language,
            "title": self.title,
            "is_default": self.is_default,
            "tags": dict(self.tags),
        }


@dataclass(frozen=True)
class VideoStreamInfo:
    index: int                               # absolute stream index in the container
    codec: str
    width: int                               # coded picture width (what the decoder returns)
    height: int                              # coded picture height
    pix_fmt: str                             # source pixel format name as reported by FFmpeg
    time_base: Fraction
    codec_long: str = ""
    profile: Optional[str] = None
    codec_tag: Optional[str] = None
    bit_depth: int = 8
    has_alpha: bool = False
    color: ColorInfo = field(default_factory=ColorInfo)
    field_order: str = "progressive"         # progressive | tt | bb | tb | bt | unknown
    rotation: int = 0                        # clockwise degrees a player applies for display: 0/90/180/270
    sample_aspect_ratio: Optional[Fraction] = None
    display_aspect_ratio: Optional[Fraction] = None
    start_pts: Optional[int] = None          # in time_base units
    start_time: Optional[Fraction] = None    # seconds
    duration: Optional[Fraction] = None      # seconds
    nb_frames: Optional[int] = None          # as declared by the container; may be absent or wrong
    avg_frame_rate: Optional[Fraction] = None
    r_frame_rate: Optional[Fraction] = None
    bit_rate: Optional[int] = None
    hdr: Dict[str, Any] = field(default_factory=dict)   # mastering display, content light level, Dolby Vision...
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def display_width(self) -> int:
        """Width after applying rotation (pixel aspect ratio is not applied)."""

        return self.height if self.rotation in (90, 270) else self.width

    @property
    def display_height(self) -> int:
        return self.width if self.rotation in (90, 270) else self.height

    @property
    def is_interlaced(self) -> bool:
        return self.field_order not in ("progressive", "unknown")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "codec": self.codec,
            "codec_long": self.codec_long,
            "profile": self.profile,
            "codec_tag": self.codec_tag,
            "width": self.width,
            "height": self.height,
            "display_width": self.display_width,
            "display_height": self.display_height,
            "pix_fmt": self.pix_fmt,
            "bit_depth": self.bit_depth,
            "has_alpha": self.has_alpha,
            "color": self.color.to_dict(),
            "field_order": self.field_order,
            "rotation": self.rotation,
            "sample_aspect_ratio": frac_to_str(self.sample_aspect_ratio),
            "display_aspect_ratio": frac_to_str(self.display_aspect_ratio),
            "time_base": frac_to_str(self.time_base),
            "start_pts": self.start_pts,
            "start_time": frac_to_float(self.start_time),
            "duration": frac_to_float(self.duration),
            "nb_frames": self.nb_frames,
            "avg_frame_rate": frac_to_str(self.avg_frame_rate),
            "avg_frame_rate_float": frac_to_float(self.avg_frame_rate),
            "r_frame_rate": frac_to_str(self.r_frame_rate),
            "bit_rate": self.bit_rate,
            "hdr": dict(self.hdr),
            "tags": dict(self.tags),
        }


@dataclass
class FrameTimeline:
    """Exact presentation timeline of the video stream (one entry per frame, presentation order).

    Built from a packet-level scan (no decoding).  This is the single source of truth for the frame
    count, for VFR detection and for timestamp verification of the output.
    """

    time_base: Fraction
    pts: np.ndarray                       # int64, ascending, in time_base units
    durations: np.ndarray                 # int64, packet durations (0 where unknown)
    keyframes: np.ndarray                 # bool, True where the packet is a keyframe
    synthesized: bool = False             # True if the container had no usable PTS (derived from frame rate)

    def __post_init__(self) -> None:
        self.pts = np.asarray(self.pts, dtype=np.int64)
        self.durations = np.asarray(self.durations, dtype=np.int64)
        self.keyframes = np.asarray(self.keyframes, dtype=bool)

    def __len__(self) -> int:
        return int(self.pts.shape[0])

    def time(self, index: int) -> Fraction:
        """Exact presentation time of frame *index* in seconds."""

        return Fraction(int(self.pts[index])) * self.time_base

    def seconds(self) -> np.ndarray:
        """Presentation times in seconds as float64 (for reporting / plotting only)."""

        return self.pts.astype(np.float64) * float(self.time_base)

    def deltas(self) -> np.ndarray:
        """Differences between consecutive PTS in time_base units (length ``n - 1``)."""

        if len(self) < 2:
            return np.zeros(0, dtype=np.int64)
        return np.diff(self.pts)

    def frame_duration(self, index: int) -> int:
        """Duration of frame *index* in time_base units (next PTS delta, else packet duration, else median)."""

        n = len(self)
        if index + 1 < n:
            return int(self.pts[index + 1] - self.pts[index])
        if n and self.durations[index] > 0:
            return int(self.durations[index])
        d = self.deltas()
        return int(np.median(d)) if d.size else 0

    @property
    def start(self) -> Optional[Fraction]:
        return self.time(0) if len(self) else None

    @property
    def duration(self) -> Fraction:
        """Total presentation duration in seconds (first PTS to end of last frame)."""

        n = len(self)
        if n == 0:
            return Fraction(0)
        span = int(self.pts[-1] - self.pts[0]) + self.frame_duration(n - 1)
        return Fraction(span) * self.time_base

    @property
    def average_rate(self) -> Optional[Fraction]:
        """Mean frame rate over the whole stream (frames per second)."""

        n = len(self)
        if n < 2:
            return None
        span = int(self.pts[-1] - self.pts[0])
        if span <= 0:
            return None
        return Fraction(n - 1, 1) / (Fraction(span) * self.time_base)

    def cfr_fit_error(self) -> int:
        """Largest deviation (in ticks) of any PTS from the best constant-frame-rate fit.

        A constant-rate stream whose timestamps were merely rounded to a coarse time base
        (e.g. 29.97 fps stored in Matroska's 1 ms ticks) deviates by at most one tick.
        """

        n = len(self)
        if n < 3:
            return 0
        span = int(self.pts[-1] - self.pts[0])
        idx = np.arange(n, dtype=np.float64)
        ideal = float(self.pts[0]) + idx * (span / float(n - 1))
        return int(np.ceil(np.max(np.abs(self.pts.astype(np.float64) - ideal)) - 1e-9))

    @property
    def is_vfr(self) -> bool:
        """True when the timeline is not a (possibly tick-rounded) constant frame rate."""

        return self.cfr_fit_error() > 1

    def vfr_stats(self) -> Dict[str, Any]:
        d = self.deltas()
        tb = float(self.time_base)
        if d.size == 0:
            return {"is_vfr": False, "frames": len(self)}
        return {
            "is_vfr": self.is_vfr,
            "frames": len(self),
            "min_frame_duration": float(d.min()) * tb,
            "max_frame_duration": float(d.max()) * tb,
            "mean_frame_duration": float(d.mean()) * tb,
            "distinct_durations": int(np.unique(d).size),
            "cfr_fit_error_ticks": self.cfr_fit_error(),
            "timestamps_synthesized": self.synthesized,
        }

    def index_of_pts(self, pts: int) -> int:
        """Index of the frame with exactly this PTS, or the nearest one."""

        i = int(np.searchsorted(self.pts, pts))
        if i >= len(self):
            return len(self) - 1
        if i > 0 and abs(int(self.pts[i - 1]) - pts) <= abs(int(self.pts[i]) - pts):
            return i - 1
        return i

    def preceding_keyframe(self, index: int) -> int:
        """Index of the last keyframe at or before *index* (0 if none is flagged)."""

        ks = np.flatnonzero(self.keyframes[: index + 1])
        return int(ks[-1]) if ks.size else 0


@dataclass(frozen=True)
class VideoMetadata:
    """Everything known about an input (or output) file."""

    path: str
    size_bytes: int
    format_name: str                        # FFmpeg demuxer name(s), e.g. "mov,mp4,m4a,3gp,3g2,mj2"
    container: str                          # normalised: mp4 | mov | mkv | webm | avi | other
    video: VideoStreamInfo
    frame_count: int                        # exact number of video frames (packet scan)
    is_vfr: bool
    format_long_name: str = ""
    duration: Optional[Fraction] = None     # container duration in seconds
    start_time: Optional[Fraction] = None   # container start time in seconds
    bit_rate: Optional[int] = None
    audio: Tuple[AudioStreamInfo, ...] = ()
    subtitle_streams: int = 0
    data_streams: int = 0
    extra_video_streams: int = 0            # attached pictures, secondary video tracks
    frame_rate: Optional[Fraction] = None   # nominal rate: mean over the exact timeline
    video_duration: Optional[Fraction] = None   # exact duration of the video timeline in seconds
    vfr: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()

    @property
    def filename(self) -> str:
        return self.path.replace("\\", "/").rsplit("/", 1)[-1]

    @property
    def has_audio(self) -> bool:
        return len(self.audio) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "format_name": self.format_name,
            "format_long_name": self.format_long_name,
            "container": self.container,
            "duration": frac_to_float(self.duration),
            "start_time": frac_to_float(self.start_time),
            "bit_rate": self.bit_rate,
            "video": self.video.to_dict(),
            "audio": [a.to_dict() for a in self.audio],
            "has_audio": self.has_audio,
            "subtitle_streams": self.subtitle_streams,
            "data_streams": self.data_streams,
            "extra_video_streams": self.extra_video_streams,
            "frame_count": self.frame_count,
            "is_vfr": self.is_vfr,
            "frame_rate": frac_to_str(self.frame_rate),
            "frame_rate_float": frac_to_float(self.frame_rate),
            "video_duration": frac_to_float(self.video_duration),
            "vfr": dict(self.vfr),
            "tags": dict(self.tags),
            "warnings": list(self.warnings),
        }


@dataclass
class ProbeResult:
    """Result of probing a file: static metadata plus the exact frame timeline."""

    metadata: VideoMetadata
    timeline: FrameTimeline


# --------------------------------------------------------------------------- frames


@dataclass
class DecodedFrame:
    """One decoded video frame as native planes (no colour conversion applied)."""

    index: int                    # 0-based presentation-order index
    pts: int                      # presentation timestamp in stream time_base units
    duration: int                 # frame duration in time_base units (0 if unknown)
    key_frame: bool
    planes: List[np.ndarray]      # C-contiguous arrays, shapes per PixFmt.plane_size, dtype PixFmt.dtype
    fmt: PixFmt

    @property
    def height(self) -> int:
        return int(self.planes[0].shape[0])

    @property
    def width(self) -> int:
        return int(self.planes[0].shape[1])


# --------------------------------------------------------------------------- subject selection


@dataclass
class ResolvedSubject:
    """A subject prompt resolved against a concrete reference frame (display orientation, pixels).

    Produced by ``app.tracking.subject.SubjectSelector``.  ``frame_index`` is the reference frame the
    coordinates refer to; model sequences always start at that frame.
    """

    frame_index: int = 0
    mask: Optional[np.ndarray] = None               # HxW float32 in 0..1 (display orientation) or None
    box: Optional[Tuple[float, float, float, float]] = None   # x1, y1, x2, y2 in pixels
    points: List[Tuple[float, float, int]] = field(default_factory=list)   # (x, y, label) label 1=positive 0=negative
    source: str = "none"                            # e.g. "auto:maskrcnn", "sam2:box", "user:mask", "text:owlv2+sam2"
    score: Optional[float] = None
    label: Optional[str] = None                     # detected class name, if any
    text: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def has_mask(self) -> bool:
        return self.mask is not None

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "frame_index": self.frame_index,
            "source": self.source,
            "score": self.score,
            "label": self.label,
            "box": [float(v) for v in self.box] if self.box else None,
            "points": [[float(x), float(y), int(l)] for x, y, l in self.points],
            "text": self.text,
            "has_mask": self.has_mask,
            "notes": list(self.notes),
        }
        if self.mask is not None:
            m = self.mask
            out["mask_area_fraction"] = float((m > 0.5).mean())
        return out


def box_from_mask(mask: np.ndarray, threshold: float = 0.5) -> Optional[Tuple[float, float, float, float]]:
    """Tight ``(x1, y1, x2, y2)`` box around ``mask > threshold`` or None if the mask is empty."""

    ys, xs = np.nonzero(mask > threshold)
    if ys.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def clamp_box(
    box: Sequence[float], width: int, height: int
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    return (x1, y1, x2, y2)
