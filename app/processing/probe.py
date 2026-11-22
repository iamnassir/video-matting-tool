"""Video input analysis: static metadata (one ``ffprobe`` call) plus the exact frame timeline.

``probe(path)`` returns a :class:`~app.core.types.ProbeResult`:

* ``metadata`` - everything ffprobe reports about the container and its streams, normalised into
  :class:`~app.core.types.VideoMetadata`;
* ``timeline`` - one entry per frame **that the decoder will actually deliver**, in presentation
  order, as integer timestamps in the video stream's time base.  The timeline is the single source of
  truth for the frame count, variable-frame-rate detection and the timestamps of the output.

How the timeline is built (validated on PyAV 17.1 / FFmpeg 8.1, see ``tests/test_probe.py``)
----------------------------------------------------------------------------------------------

1. **Packet scan** (no decoding): demux the video stream with PyAV and collect PTS, duration and
   flags of every packet.  The presented frames are the packets that

   * are not flagged *discard* (MP4 edit lists hide packets that are decoded but never shown),
   * carry a PTS, and
   * are not *leading pictures* in front of the first key frame (open-GOP streams cut at a key frame:
     the decoder never outputs them),

   sorted by PTS (B-frames are stored in decoding order).

2. **Head verification**: the first few frames are decoded and compared with the prediction.  This
   costs a few milliseconds and catches everything the packet flags cannot express (packets without
   PTS that do or do not produce frames, decoders that drop more leading pictures than expected).

3. **Decode scan** (slow path): when the packet scan cannot be trusted - packets without PTS (AVI,
   raw streams), duplicate PTS, timestamps that contradict the decoding order, or a failed head
   verification - the whole stream is decoded once and the timestamps are taken from the decoded
   frames with :class:`TimestampFixer`, which reproduces what the ``ffmpeg`` CLI itself does with
   missing timestamps.  A wrong timeline is never returned for the sake of speed.

``app.processing.reader.VideoReader`` uses the same container options, the same decode loop
(:func:`iter_decoded_frames`) and the same :class:`TimestampFixer`, so what it yields is exactly this
timeline.

Pitfalls handled here (found by experiment, see the research notes):

* PyAV sets ``AVFMT_FLAG_GENPTS`` on every container; libavformat then *invents* presentation
  timestamps (non-monotonic ones for H.264 in AVI).  :func:`open_container` clears the flag so that
  packets and frames carry exactly what ``ffprobe`` / ``ffmpeg`` see.
* ``stream.frames`` / ``nb_frames`` is the container's claim (650 for a clip that presents 600).
* ffprobe reports the display-matrix rotation counter-clockwise; ``VideoStreamInfo.rotation`` is the
  clockwise angle a player applies.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import struct
import subprocess
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import av
import av.error
import numpy as np

from ..core.errors import (
    CorruptedVideoError,
    FfmpegFailedError,
    FfmpegNotFoundError,
    InvalidPathError,
    NoVideoStreamError,
    UnsupportedFormatError,
    VmtError,
)
from ..core.pixfmt import ColorInfo
from ..core.types import (
    AudioStreamInfo,
    FrameTimeline,
    ProbeResult,
    VideoMetadata,
    VideoStreamInfo,
    parse_fraction,
)

from ..utils.avlog import silence_pyav_logging

log = logging.getLogger(__name__)

__all__ = [
    "probe",
    "TimelineScan",
    "VideoProbeResult",
    "TimestampFixer",
    "open_container",
    "select_video_stream",
    "iter_decoded_frames",
    "describe_av_error",
]

#: libavformat flag that PyAV switches on and that must be off for ffmpeg-equivalent timestamps.
AVFMT_FLAG_GENPTS = 0x0001
#: Demuxer flag of formats without timestamps (raw elementary streams).
AVFMT_NOTIMESTAMPS = 0x0080

#: Frames decoded to verify the packet scan against the decoder.
HEAD_FRAMES = 8
#: Give up the verification (and decode everything instead) when this many packets yield too few frames.
HEAD_MAX_PACKETS = 600
#: Packets without PTS are only tolerated by the fast path this close to the start of the stream,
#: where the head verification can prove that the decoder drops them.
HEAD_NOPTS_LIMIT = 32
#: Packets fed beyond the last packet without PTS so that the decoder delay (frame threads plus
#: reorder buffer) cannot hide a frame that such a packet produced.
HEAD_NOPTS_MARGIN = 48
#: A frame is displayed at most this many positions away from its decoding position in any real
#: codec (H.264 / HEVC allow 16); larger shifts mean the timestamps contradict the decoding order.
MAX_REORDER_DISPLACEMENT = 64
#: Denominators above this are reduced with ``limit_denominator`` so that rates stay readable.
READABLE_DENOMINATOR = 100_000

FFPROBE_TIMEOUT_SECONDS = 120.0
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_NOPTS = -(1 << 62)          # sentinel inside int64 arrays for "no timestamp"
#: What PyAV raises for I/O and bitstream problems (``InvalidDataError`` etc. derive from ``FFmpegError``).
_AV_ERRORS = (av.error.FFmpegError, OSError)


# =========================================================================== result types


@dataclass(frozen=True)
class TimelineScan:
    """How the frame timeline was obtained (diagnostics; also shown by ``vmt probe --json`` callers)."""

    method: str = "packets"            # "packets" (fast path) | "decode" (slow path) | "skipped"
    packets: int = 0                   # video packets found by the demuxer
    discarded: int = 0                 # packets hidden by an edit list (decoded but never shown)
    leading_dropped: int = 0           # leading pictures in front of the first key frame
    without_pts: int = 0               # shown packets that carry no presentation timestamp
    synthesized_timestamps: int = 0    # frames whose timestamp did not come from a container PTS
    adjusted_timestamps: int = 0       # frames whose timestamp was moved to keep the timeline increasing
    decode_errors: int = 0             # decoder errors seen while scanning (slow path / head check)
    damaged_packets: int = 0           # packets flagged as damaged by the demuxer (left out)
    reason: str = ""                   # why the packet scan was not trusted (method == "decode")
    seconds: float = 0.0               # wall-clock time of the timeline scan

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class VideoProbeResult(ProbeResult):
    """:class:`~app.core.types.ProbeResult` plus a description of how the timeline was built."""

    scan: TimelineScan = field(default_factory=TimelineScan)


# =========================================================================== PyAV helpers (shared with the reader)


def describe_av_error(exc: BaseException) -> str:
    """Short, human-readable text of a PyAV / OS error (without the numeric FFmpeg error code)."""

    text = getattr(exc, "strerror", None) or str(exc) or type(exc).__name__
    entry = getattr(exc, "log", None)
    try:
        if entry and entry[2] and str(entry[2]).strip() not in text:
            text = f"{text} ({str(entry[2]).strip()})"
    except (IndexError, TypeError):
        pass
    return str(text)


def open_container(path: Any) -> "av.container.InputContainer":
    """Open *path* for reading the way every part of this package must.

    * ``AVFMT_FLAG_GENPTS`` is cleared: PyAV sets it, the ``ffmpeg`` CLI does not, and with it
      libavformat fabricates presentation timestamps for packets that have none (H.264 in AVI then
      gets non-monotonic frame timestamps).  Probe and reader must both see the real ones.
    * metadata is decoded leniently: a tag that is not valid UTF-8 must not make the file unreadable.

    Raises the original PyAV exception; callers convert it to the appropriate ``VmtError``.
    """

    silence_pyav_logging()      # a log callback that needs the GIL deadlocks decoder teardown
    container = av.open(os.fspath(path), mode="r", metadata_errors="replace")
    try:
        container.flags = container.flags & ~AVFMT_FLAG_GENPTS
    except Exception:
        container.close()
        raise
    return container


def select_video_stream(container: "av.container.InputContainer", index: int) -> "av.video.stream.VideoStream":
    """Return the stream with absolute index *index*, making sure it is a video stream."""

    try:
        stream = container.streams[index]
    except (IndexError, TypeError) as exc:
        raise CorruptedVideoError(
            f"Stream #{index} does not exist in the file any more (it was replaced or truncated after probing)."
        ) from exc
    if stream.type != "video":
        raise CorruptedVideoError(f"Stream #{index} is not a video stream ({stream.type}).")
    return stream


def _discard_other_streams(container: "av.container.InputContainer", keep_index: int) -> None:
    """Tell the demuxer to skip every stream except *keep_index* (MP4 then never reads their data)."""

    for stream in container.streams:
        if stream.index == keep_index:
            continue
        try:
            stream.discard = type(stream.discard).all
        except Exception:  # optimisation only - never fatal
            return


def iter_decoded_frames(
    container: "av.container.InputContainer",
    stream: "av.video.stream.VideoStream",
    errors: Optional[List[str]] = None,
) -> Iterator["av.VideoFrame"]:
    """Demux and decode *stream* from the current position to its end, in output order.

    This is the one decode loop used by the timeline scan and by the reader, so both see the same
    frames:

    * empty packets are skipped (an empty packet would put the decoder into draining mode in the
      middle of the stream; the ``ffmpeg`` CLI skips them too) and the decoder is flushed exactly once
      at the end;
    * packets the demuxer flags as damaged (typically the packet a truncated file ends in) are not
      decoded: their frame is not part of the timeline either, and with a multi-threaded decoder the
      error they cause would surface while draining and cost the intact frames still in the decoder;
    * a decoder error for one packet does not end the stream: its text is appended to *errors* and
      decoding continues with the next packet (the consumer notices the missing frame by its
      timestamp);
    * a demuxer error ends the stream (after flushing what the decoder still holds).
    """

    def note(exc: BaseException, where: str) -> None:
        if errors is not None:
            if len(errors) >= 64:
                del errors[1:33]          # keep the first and the most recent messages
            errors.append(f"{where}: {describe_av_error(exc)}")

    codec = stream.codec_context
    try:
        for packet in container.demux(stream):
            if packet.size == 0 or packet.is_corrupt:
                continue
            try:
                frames = packet.decode()
            except av.error.FFmpegError as exc:
                note(exc, "decoder")
                continue
            for frame in frames:
                yield frame
    except av.error.FFmpegError as exc:
        note(exc, "demuxer")
    try:
        frames = codec.decode(None)
    except av.error.FFmpegError as exc:
        note(exc, "decoder")
        frames = []
    for frame in frames:
        yield frame


class TimestampFixer:
    """Turns the timestamps of decoded frames into a strictly increasing presentation timeline.

    For frames that carry a PTS the result is that PTS.  Otherwise the logic of the ``ffmpeg`` CLI is
    reproduced, so the timeline equals what ``ffmpeg -i`` itself would use:

    * ``libavcodec/decode.c  guess_correct_pts()``      best-effort timestamp from (pts, pkt_dts)
    * ``fftools/ffmpeg_dec.c video_frame_process()``    extrapolation when there is no timestamp
    * ``fftools/ffmpeg_dec.c video_duration_estimate()``
    * ``fftools/ffmpeg_dec.c packet_decode()``          formats without timestamps: ignore them all

    (validated against ``ffmpeg -copyts -vf showinfo`` for MPEG-4 and H.264 in AVI and for raw
    elementary streams).  On top of that a timestamp that is not larger than its predecessor is moved
    to one tick after it: the encoder, the NUT pipe and every muxer need strictly increasing values.

    The object is stateful (it must see every frame from the start of the stream, in output order);
    that is why files whose timeline needed it are never seeked into.
    """

    _MIN = -(1 << 63)

    def __init__(
        self,
        time_base: Fraction,
        codec_rate: Optional[Fraction] = None,
        average_rate: Optional[Fraction] = None,
        ts_unreliable: bool = False,
    ) -> None:
        self.time_base = Fraction(time_base)
        self.codec_rate = Fraction(codec_rate) if codec_rate else None
        self.average_rate = Fraction(average_rate) if average_rate else None
        self.ts_unreliable = bool(ts_unreliable)
        self._last_pts_seen = self._MIN
        self._last_dts_seen = self._MIN
        self._faulty_pts = 0
        self._faulty_dts = 0
        self._last_out: Optional[int] = None
        self._last_duration = 0
        self.frames = 0
        self.synthesized = 0      # timestamps that are not the frame's own PTS
        self.adjusted = 0         # of those: moved because they were not increasing

    @classmethod
    def for_stream(cls, container: "av.container.InputContainer", stream: "av.video.stream.VideoStream") -> "TimestampFixer":
        """Build the fixer from the stream's own hints (identical in the probe and in the reader)."""

        def rate(value: Any) -> Optional[Fraction]:
            try:
                value = Fraction(value) if value else None
            except (TypeError, ValueError, ZeroDivisionError):
                return None
            return value if value and value > 0 else None

        codec_rate = None
        try:
            codec_rate = rate(stream.codec_context.framerate)
        except Exception:
            codec_rate = None
        average = rate(getattr(stream, "guessed_rate", None)) or rate(getattr(stream, "average_rate", None))
        unreliable = bool(int(getattr(container.format, "flags", 0) or 0) & AVFMT_NOTIMESTAMPS)
        return cls(Fraction(stream.time_base), codec_rate, average, unreliable)

    # -- ffmpeg logic --------------------------------------------------------------------
    def _best_effort(self, pts: Optional[int], dts: Optional[int]) -> Optional[int]:
        if dts is not None:
            self._faulty_dts += dts <= self._last_dts_seen
            self._last_dts_seen = dts
        elif pts is not None:
            self._last_dts_seen = pts
        if pts is not None:
            self._faulty_pts += pts <= self._last_pts_seen
            self._last_pts_seen = pts
        elif dts is not None:
            self._last_pts_seen = dts
        if (self._faulty_pts <= self._faulty_dts or dts is None) and pts is not None:
            return pts
        return dts

    def _ticks(self, rate: Optional[Fraction]) -> int:
        if not rate:
            return 0
        return int(round(Fraction(1) / rate / self.time_base))

    def _duration_estimate(self, out: int, previous: Optional[int], duration: int) -> int:
        diff = out - previous if previous is not None else -1
        duration_unreliable = duration == 1 and diff > 2 * duration
        if duration > 0 and not self.ts_unreliable and not duration_unreliable:
            return duration
        codec_duration = self._ticks(self.codec_rate)
        if codec_duration > 0 and self.ts_unreliable:
            return codec_duration
        if diff > 0:
            return diff
        if duration > 0:
            return duration
        if codec_duration > 0:
            return codec_duration
        average = self._ticks(self.average_rate)
        if average > 0:
            return average
        return max(self._last_duration, 1)

    # -- public --------------------------------------------------------------------------
    def __call__(self, pts: Optional[int], dts: Optional[int], duration: Optional[int]) -> int:
        """Timestamp (stream time base ticks) of the next decoded frame."""

        raw = pts
        if self.ts_unreliable:
            pts = dts = None
        out = self._best_effort(pts, dts)
        if out is None:
            out = 0 if self._last_out is None else self._last_out + max(self._last_duration, 1)
        previous = self._last_out
        if previous is not None and out <= previous:
            out = previous + 1
            self.adjusted += 1
        if raw is None or out != raw:
            self.synthesized += 1
        self._last_duration = self._duration_estimate(out, previous, int(duration or 0))
        self._last_out = out
        self.frames += 1
        return out


# =========================================================================== ffprobe


def _find_ffprobe(explicit: Optional[str]) -> str:
    """Resolve the ffprobe executable: explicit path, ``VMT_FFPROBE``, then ``PATH``."""

    candidate = explicit or os.environ.get("VMT_FFPROBE") or None
    if candidate:
        resolved = shutil.which(candidate)
        if resolved is None and Path(candidate).is_file():
            resolved = str(candidate)
        if resolved is None:
            raise FfmpegNotFoundError(
                f"ffprobe was not found at '{candidate}'.", details={"ffprobe": str(candidate)}
            )
        return resolved
    resolved = shutil.which("ffprobe")
    if resolved is None:
        raise FfmpegNotFoundError("ffprobe was not found on PATH.")
    return resolved


def _tail(text: str, lines: int = 20, limit: int = 4000) -> str:
    kept = [ln for ln in text.replace("\r\n", "\n").split("\n") if ln.strip()][-lines:]
    return "\n".join(kept)[-limit:]


_MEDIA_SIGNATURES: Tuple[Tuple[int, bytes], ...] = (
    (4, b"ftyp"), (4, b"moov"), (4, b"mdat"), (4, b"free"), (4, b"wide"), (4, b"skip"),   # ISO BMFF / QuickTime
    (4, b"styp"), (4, b"moof"), (4, b"sidx"), (4, b"uuid"), (4, b"pnot"), (4, b"junk"),
    (0, b"\x1a\x45\xdf\xa3"),                                                              # Matroska / WebM
    (0, b"FLV"), (0, b"OggS"), (0, b"\x00\x00\x01\xba"), (0, b"\x30\x26\xb2\x75\x8e\x66\xcf\x11"),
)


def _looks_like_media_container(path: Path) -> bool:
    """True if the first bytes carry the signature of a known video container.

    Used only to word the error when FFmpeg cannot read a file: a recognisable container that cannot
    be opened is *damaged*, anything else is simply *not a video*.
    """

    try:
        with open(path, "rb") as fh:
            head = fh.read(4096)
    except OSError:
        return False
    for offset, magic in _MEDIA_SIGNATURES:
        if head[offset: offset + len(magic)] == magic:
            return True
    if head[:4] == b"RIFF" and head[8:12] in (b"AVI ", b"AVIX"):
        return True
    if len(head) >= 377 and head[0] == 0x47 and head[188] == 0x47 and head[376] == 0x47:      # MPEG-TS
        return True
    return False


#: Extensions of the container families whose signature :func:`_looks_like_media_container` knows.
#: FFmpeg picks a demuxer by file extension when the content matches nothing, so for these a demuxer
#: error message alone does not mean the file is a (damaged) video.
_SIGNATURE_EXTENSIONS = frozenset({
    ".mp4", ".m4v", ".mov", ".qt", ".3gp", ".3g2", ".mkv", ".webm", ".mka", ".avi", ".flv", ".ogv", ".ogg",
    ".ts", ".m2ts", ".mts", ".mpg", ".mpeg", ".vob", ".wmv", ".asf",
})

_NOT_A_VIDEO_HINT = (
    "Choose a video file: MP4, MOV, MKV, AVI, WebM or another format FFmpeg can read. "
    "Renaming a file does not change what it contains."
)


def _looks_like_text(path: Path) -> bool:
    """True if the beginning of the file is plain text (notes, subtitles, a renamed document)."""

    try:
        with open(path, "rb") as fh:
            head = fh.read(4096)
    except OSError:
        return False
    if not head or b"\x00" in head:
        return False
    printable = sum(1 for byte in head if byte in (9, 10, 13) or 32 <= byte < 127 or byte >= 128)
    if printable < 0.97 * len(head):
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        return exc.start >= len(head) - 4          # only a character cut at the end of the sample
    return True


def _unreadable(path: Path, stderr: str, returncode: Optional[int]) -> VmtError:
    """Choose between "damaged" and "not a video" for a file ffprobe could not open."""

    tail = _tail(stderr)
    last = tail.splitlines()[-1] if tail else "no error text"
    details = {"path": str(path), "stderr_tail": tail, "returncode": returncode}
    lowered = stderr.lower()
    recognised = _looks_like_media_container(path)
    is_text = not recognised and _looks_like_text(path)
    if not recognised and not is_text and path.suffix.lower() not in _SIGNATURE_EXTENSIONS:
        # "[mxf @ 0x..] ..." style lines mean a demuxer accepted the content and then failed.
        recognised = any(ln.lstrip().startswith("[") and " @ " in ln.split("]", 1)[0] for ln in stderr.splitlines())
    if is_text:
        return UnsupportedFormatError(
            f"'{path.name}' is a text file, not a video (ffprobe: {last}).", hint=_NOT_A_VIDEO_HINT, details=details
        )
    if recognised:
        if "moov atom not found" in lowered:
            return CorruptedVideoError(
                f"'{path.name}' is an MP4/MOV file without its index (moov atom): the recording was not "
                "finalised or the file is truncated, so it cannot be read.",
                hint="Copy or download the file again. A recording that was interrupted can sometimes be "
                "repaired with a tool such as 'untrunc' and a healthy file from the same camera.",
                details=details,
            )
        return CorruptedVideoError(
            f"'{path.name}' looks like a video file but FFmpeg cannot read it (ffprobe: {last}).",
            details=details,
        )
    return UnsupportedFormatError(
        f"'{path.name}' is not a video file FFmpeg can read (ffprobe: {last}).", hint=_NOT_A_VIDEO_HINT, details=details
    )


def _run_ffprobe(executable: str, path: Path) -> Dict[str, Any]:
    """One ffprobe call that returns the format and every stream as JSON."""

    command = [
        executable, "-v", "error", "-hide_banner", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=FFPROBE_TIMEOUT_SECONDS, creationflags=_NO_WINDOW,
        )
    except FileNotFoundError as exc:
        raise FfmpegNotFoundError(f"ffprobe could not be started ('{executable}').", details={"ffprobe": executable}) from exc
    except subprocess.TimeoutExpired as exc:
        stderr = (exc.stderr or b"").decode("utf-8", "replace")
        raise FfmpegFailedError(
            f"ffprobe did not finish analysing '{path.name}' within {FFPROBE_TIMEOUT_SECONDS:.0f} s.",
            hint="The file may be on a very slow or disconnected drive, or it is damaged.",
            details={"path": str(path), "stderr_tail": _tail(stderr), "command": command},
        ) from exc
    except OSError as exc:
        raise FfmpegFailedError(
            f"ffprobe could not be started ('{executable}'): {exc}",
            hint="Check that ffprobe is installed correctly and executable.",
            details={"ffprobe": executable, "stderr_tail": ""},
        ) from exc
    stdout = result.stdout.decode("utf-8", "replace")
    stderr = result.stderr.decode("utf-8", "replace")
    data: Dict[str, Any] = {}
    try:
        parsed = json.loads(stdout) if stdout.strip() else {}
        if isinstance(parsed, dict):
            data = parsed
    except ValueError:
        if result.returncode == 0:
            raise FfmpegFailedError(
                f"ffprobe returned output that could not be parsed for '{path.name}'.",
                details={"path": str(path), "stderr_tail": _tail(stderr), "stdout_tail": stdout[-500:]},
            ) from None
    if result.returncode != 0 or not data.get("streams"):
        raise _unreadable(path, stderr, result.returncode)
    if stderr.strip():
        log.debug("ffprobe messages for %s: %s", path, _tail(stderr, 5))
    return data


# =========================================================================== ffprobe field parsing


def _to_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return int(number) if math.isfinite(number) else None


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text.upper() != "N/A" else None


def _tags(entry: Dict[str, Any]) -> Dict[str, str]:
    raw = entry.get("tags")
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def _tag(tags: Dict[str, str], *names: str) -> Optional[str]:
    lowered = {k.lower(): v for k, v in tags.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _seconds(value: Any) -> Optional[Fraction]:
    """Exact value of a decimal seconds string such as ``"25.003988"``."""

    text = _clean(value)
    if text is None:
        return None
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def _clock_duration(value: Optional[str]) -> Optional[Fraction]:
    """Matroska keeps per-stream durations only as a tag: ``DURATION=00:00:03.033000000``."""

    if not value:
        return None
    try:
        hours, minutes, seconds = value.strip().split(":")
        return Fraction(int(hours)) * 3600 + Fraction(int(minutes)) * 60 + Fraction(seconds)
    except (ValueError, ZeroDivisionError):
        return None


def _rational(value: Any) -> Optional[Fraction]:
    """ffprobe rational (``30000/1001``, ``16:9``); ``0/0`` and zero mean "unknown"."""

    parsed = parse_fraction(value)
    return parsed if parsed is not None and parsed != 0 else None


def _stream_times(entry: Dict[str, Any], tags: Dict[str, str]) -> Tuple[Optional[Fraction], Optional[int], Optional[Fraction], Optional[Fraction]]:
    """``(time_base, start_pts, start_time, duration)`` of a stream, exact where the container allows."""

    time_base = _rational(entry.get("time_base"))
    start_pts = _to_int(entry.get("start_pts"))
    start_time = Fraction(start_pts) * time_base if start_pts is not None and time_base else _seconds(entry.get("start_time"))
    duration_ts = _to_int(entry.get("duration_ts"))
    if duration_ts is not None and duration_ts > 0 and time_base:
        duration: Optional[Fraction] = Fraction(duration_ts) * time_base
    else:
        duration = _seconds(entry.get("duration")) or _clock_duration(_tag(tags, "DURATION", "DURATION-eng"))
    return time_base, start_pts, start_time, duration


def _color_tag(value: Any) -> str:
    text = _clean(value)
    if text is None or text.lower() in ("unknown", "unspecified"):
        return "unknown"
    return text.lower()


def _pixel_format_facts(pix_fmt: str, bits_per_raw_sample: Optional[int]) -> Tuple[int, bool]:
    """``(bit_depth, has_alpha)`` of a pixel format, from FFmpeg's own descriptor.

    ``bits_per_raw_sample`` is missing for VP9, ProRes and sometimes HEVC, so the descriptor wins.
    """

    try:
        descriptor = av.VideoFormat(pix_fmt, 16, 16)
    except Exception:
        return (bits_per_raw_sample if bits_per_raw_sample and bits_per_raw_sample > 0 else 8), False
    colour = [c.bits for c in descriptor.components if not getattr(c, "is_alpha", False)]
    alpha = any(getattr(c, "is_alpha", False) for c in descriptor.components)
    depth = max(colour) if colour else (bits_per_raw_sample or 8)
    return int(depth), bool(alpha)


def _side_data(entry: Dict[str, Any], kind: str) -> Optional[Dict[str, Any]]:
    for item in entry.get("side_data_list") or []:
        if isinstance(item, dict) and str(item.get("side_data_type", "")).lower() == kind.lower():
            return item
    return None


def _display_matrix(item: Dict[str, Any]) -> Optional[List[int]]:
    text = item.get("displaymatrix")
    if not isinstance(text, str):
        return None
    try:
        values = [int(token) for token in text.split() if not token.endswith(":")]
    except ValueError:
        return None
    return values if len(values) == 9 else None


def _matrix_rotation_ccw(matrix: Sequence[int]) -> Optional[float]:
    """``av_display_rotation_get``: counter-clockwise degrees of a 3x3 display matrix (16.16 fixed point)."""

    a, b, c, d = (float(matrix[i]) / 65536.0 for i in (0, 1, 3, 4))
    scale0, scale1 = math.hypot(a, c), math.hypot(b, d)
    if scale0 == 0.0 or scale1 == 0.0:
        return None
    return -math.degrees(math.atan2(b / scale1, a / scale0))


def _rotation(entry: Dict[str, Any], tags: Dict[str, str]) -> Tuple[int, List[str]]:
    """Clockwise display rotation (0 / 90 / 180 / 270) and the warnings it gives rise to.

    ffprobe (like ``av_display_rotation_get``) reports the angle counter-clockwise: an iPhone portrait
    clip has ``rotation=-90`` and players turn the coded picture 90 degrees clockwise.  The sign was
    verified against ``ffmpeg``'s auto-rotated output (see ``tests/test_reader.py``).
    """

    warnings: List[str] = []
    item = _side_data(entry, "Display Matrix")
    clockwise: Optional[float] = None
    flipped = False
    if item is not None:
        matrix = _display_matrix(item)
        ccw: Optional[float] = None
        try:
            ccw = float(item["rotation"]) if item.get("rotation") is not None else None
        except (TypeError, ValueError):
            ccw = None
        if ccw is None or not math.isfinite(ccw):
            ccw = _matrix_rotation_ccw(matrix) if matrix else None
        if ccw is not None and math.isfinite(ccw):
            clockwise = (-ccw) % 360.0
        if matrix:
            flipped = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) < 0
    else:
        legacy = _tag(tags, "rotate")
        if legacy is not None:
            try:
                clockwise = float(legacy) % 360.0          # the legacy tag already is the clockwise angle
            except ValueError:
                clockwise = None
    if clockwise is None:
        return 0, warnings
    snapped = int(round(clockwise / 90.0)) % 4 * 90
    deviation = abs(((clockwise - snapped) + 180.0) % 360.0 - 180.0)
    if deviation > 0.5:
        warnings.append(
            f"The display matrix rotates the picture by {clockwise:.1f} degrees, which is not a multiple of 90; "
            f"it is treated as {snapped} degrees for previews and matting. The output keeps the original matrix."
        )
    if flipped:
        warnings.append(
            f"The display matrix of the video mirrors the picture in addition to rotating it by {snapped} degrees. "
            "Previews and subject selection show the picture rotated but not mirrored; the output keeps the "
            "original matrix, so players display it exactly like the source."
        )
    return snapped, warnings


_HDR_SIDE_DATA: Dict[str, str] = {
    "mastering display metadata": "mastering_display",
    "content light level metadata": "content_light_level",
    "dovi configuration record": "dolby_vision",
    "ambient viewing environment": "ambient_viewing_environment",
}


def _hdr_from_stream(entry: Dict[str, Any]) -> Dict[str, Any]:
    """HDR side data that the container declares for the stream (ffprobe ``side_data_list``)."""

    out: Dict[str, Any] = {}
    for item in entry.get("side_data_list") or []:
        if not isinstance(item, dict):
            continue
        key = _HDR_SIDE_DATA.get(str(item.get("side_data_type", "")).lower())
        if key is None:
            continue
        out[key] = {k: v for k, v in item.items() if k != "side_data_type" and isinstance(v, (str, int, float, bool))}
    return out


def _hdr_from_frame(frame: "av.VideoFrame") -> Dict[str, Any]:
    """Static HDR metadata carried in the bitstream (SEI) of the first frame.

    Encoders such as x265 store the mastering display colour volume and the content light level only
    there, so ffprobe's stream section does not show them.  The two structures have a fixed layout
    (``AVMasteringDisplayMetadata``: ten ``AVRational`` plus two flags, ``AVContentLightMetadata``:
    two unsigned integers); field names follow ffprobe.
    """

    out: Dict[str, Any] = {}
    try:
        entries = list(frame.side_data)
    except Exception:
        return out
    for entry in entries:
        name = str(getattr(entry, "type", "")).upper()
        try:
            raw = bytes(entry)
        except Exception:
            continue
        if name.endswith("MASTERING_DISPLAY_METADATA") and len(raw) >= 88:
            v = struct.unpack_from("<22i", raw)
            ratio = lambda i: f"{v[2 * i]}/{v[2 * i + 1]}"   # noqa: E731
            data: Dict[str, Any] = {"source": "bitstream"}
            if v[20]:
                data.update(
                    red_x=ratio(0), red_y=ratio(1), green_x=ratio(2), green_y=ratio(3),
                    blue_x=ratio(4), blue_y=ratio(5), white_point_x=ratio(6), white_point_y=ratio(7),
                )
            if v[21]:
                data.update(min_luminance=ratio(8), max_luminance=ratio(9))
            if len(data) > 1:
                out["mastering_display"] = data
        elif name.endswith("CONTENT_LIGHT_LEVEL") and len(raw) >= 8:
            max_content, max_average = struct.unpack_from("<2I", raw)
            out["content_light_level"] = {"max_content": int(max_content), "max_average": int(max_average), "source": "bitstream"}
        elif name.endswith("DOVI_METADATA") or name.endswith("DOVI_RPU_BUFFER"):
            out.setdefault("dolby_vision_rpu", True)
        elif "DYNAMIC_HDR_PLUS" in name or "DYNAMIC_HDR10_PLUS" in name:
            out.setdefault("hdr10_plus", True)
    return out


_WEBM_VIDEO = {"vp8", "vp9", "av1"}
_WEBM_AUDIO = {"opus", "vorbis"}


def _normalise_container(format_name: str, path: Path, format_tags: Dict[str, str], streams: Sequence[Dict[str, Any]]) -> str:
    """Map FFmpeg's demuxer name (plus the file extension) to ``mp4 | mov | mkv | webm | avi | other``."""

    names = {part.strip().lower() for part in (format_name or "").split(",") if part.strip()}
    ext = path.suffix.lower()
    if names & {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}:
        if ext in (".mov", ".qt"):
            return "mov"
        if ext in (".mp4", ".m4v", ".m4a", ".3gp", ".3g2", ".f4v"):
            return "mp4"
        brand = (_tag(format_tags, "major_brand") or "").strip().lower()
        return "mov" if brand == "qt" else "mp4"
    if names & {"matroska", "webm"}:
        if ext == ".webm":
            return "webm"
        if ext in (".mkv", ".mk3d", ".mka", ".mks"):
            return "mkv"
        codecs = [(s.get("codec_type"), str(s.get("codec_name", "")).lower()) for s in streams]
        video_ok = all(name in _WEBM_VIDEO for kind, name in codecs if kind == "video")
        audio_ok = all(name in _WEBM_AUDIO for kind, name in codecs if kind == "audio")
        other = any(kind not in ("video", "audio") for kind, _ in codecs)
        return "webm" if video_ok and audio_ok and not other else "mkv"
    if "avi" in names:
        return "avi"
    return "other"


def _audio_stream(entry: Dict[str, Any]) -> AudioStreamInfo:
    tags = _tags(entry)
    time_base, _start_pts, start_time, duration = _stream_times(entry, tags)
    disposition = entry.get("disposition") if isinstance(entry.get("disposition"), dict) else {}
    return AudioStreamInfo(
        index=int(entry.get("index", 0)),
        codec=_clean(entry.get("codec_name")) or "unknown",
        codec_long=_clean(entry.get("codec_long_name")) or "",
        profile=_clean(entry.get("profile")),
        sample_rate=_to_int(entry.get("sample_rate")),
        channels=_to_int(entry.get("channels")),
        channel_layout=_clean(entry.get("channel_layout")),
        sample_fmt=_clean(entry.get("sample_fmt")),
        bit_rate=_to_int(entry.get("bit_rate")) or _to_int(_tag(tags, "BPS", "BPS-eng")),
        time_base=time_base,
        start_time=start_time,
        duration=duration,
        nb_frames=_to_int(entry.get("nb_frames")) or _to_int(_tag(tags, "NUMBER_OF_FRAMES", "NUMBER_OF_FRAMES-eng")),
        language=_tag(tags, "language"),
        title=_tag(tags, "title", "name"),
        is_default=bool(_to_int(disposition.get("default"))),
        tags=tags,
    )


# =========================================================================== timeline scan


@dataclass
class _PacketTable:
    """Raw result of the demux pass (arrays in demux order)."""

    time_base: Fraction
    pts: np.ndarray           # int64, _NOPTS where the packet has no PTS
    duration: np.ndarray      # int64
    keyframe: np.ndarray      # bool
    discard: np.ndarray       # bool
    corrupt: np.ndarray       # bool: flagged as damaged by the demuxer (not decoded, not presented)
    ts_unreliable: bool
    demux_error: Optional[str] = None
    #: frames the decoder may hold back for reordering (0 = the stream has no B-frames; None = unknown)
    reorder_depth: Optional[int] = None

    @property
    def count(self) -> int:
        return int(self.pts.shape[0])


@dataclass
class _Head:
    """Result of decoding the first frames of the stream."""

    pts: List[Optional[int]] = field(default_factory=list)
    complete: bool = False                 # the whole stream was decoded
    packets_fed: int = 0
    errors: List[str] = field(default_factory=list)
    pix_fmt: Optional[str] = None
    width: int = 0
    height: int = 0
    hdr: Dict[str, Any] = field(default_factory=dict)


def _scan_packets(path: Path, stream_index: int) -> _PacketTable:
    """Demux the whole video stream without decoding and return the packet timestamps and flags."""

    pts: List[int] = []
    duration: List[int] = []
    keyframe: List[bool] = []
    discard: List[bool] = []
    corrupt: List[bool] = []
    demux_error: Optional[str] = None
    try:
        container = open_container(path)
    except _AV_ERRORS as exc:
        raise CorruptedVideoError(
            f"The video stream of '{path.name}' cannot be opened for reading: {describe_av_error(exc)}.",
            details={"path": str(path), "error": describe_av_error(exc)},
        ) from exc
    try:
        stream = select_video_stream(container, stream_index)
        _discard_other_streams(container, stream_index)
        time_base = Fraction(stream.time_base)
        unreliable = bool(int(getattr(container.format, "flags", 0) or 0) & AVFMT_NOTIMESTAMPS)
        try:
            reorder_depth = int(stream.codec_context.has_b_frames)
        except Exception:
            reorder_depth = None
        try:
            for packet in container.demux(stream):
                if packet.size == 0:
                    continue                      # PyAV's end-of-stream marker, or an empty AVI chunk
                value = packet.pts
                pts.append(_NOPTS if value is None else value)
                duration.append(packet.duration or 0)
                keyframe.append(packet.is_keyframe)
                discard.append(packet.is_discard)
                corrupt.append(packet.is_corrupt)
        except av.error.FFmpegError as exc:
            demux_error = describe_av_error(exc)
            log.warning("Demuxing %s stopped after %d video packets: %s", path, len(pts), demux_error)
    finally:
        container.close()
    return _PacketTable(
        time_base=time_base,
        pts=np.asarray(pts, dtype=np.int64),
        duration=np.asarray(duration, dtype=np.int64),
        keyframe=np.asarray(keyframe, dtype=bool),
        discard=np.asarray(discard, dtype=bool),
        corrupt=np.asarray(corrupt, dtype=bool),
        ts_unreliable=unreliable,
        demux_error=demux_error,
        reorder_depth=reorder_depth,
    )


def _decode_head(path: Path, stream_index: int, min_packets: int = 0) -> _Head:
    """Decode the first :data:`HEAD_FRAMES` frames (and at least *min_packets* packets)."""

    head = _Head()
    try:
        container = open_container(path)
    except _AV_ERRORS as exc:
        head.errors.append(describe_av_error(exc))
        return head
    try:
        stream = select_video_stream(container, stream_index)
        _discard_other_streams(container, stream_index)
        stream.thread_type = "AUTO"
        codec = stream.codec_context

        def take(frames: Sequence["av.VideoFrame"]) -> None:
            for frame in frames:
                if not head.pts:
                    head.pix_fmt = frame.format.name
                    head.width, head.height = int(frame.width), int(frame.height)
                    head.hdr = _hdr_from_frame(frame)
                head.pts.append(frame.pts)

        finished = True
        try:
            for packet in container.demux(stream):
                if packet.size == 0 or packet.is_corrupt:
                    continue
                head.packets_fed += 1
                try:
                    take(packet.decode())
                except av.error.FFmpegError as exc:
                    head.errors.append(describe_av_error(exc))
                enough = len(head.pts) >= HEAD_FRAMES and head.packets_fed >= min_packets
                if enough or head.packets_fed >= max(HEAD_MAX_PACKETS, min_packets):
                    finished = False
                    break
        except av.error.FFmpegError as exc:
            head.errors.append(describe_av_error(exc))
        if finished:
            try:
                take(codec.decode(None))
            except av.error.FFmpegError as exc:
                head.errors.append(describe_av_error(exc))
            head.complete = True
    except VmtError as exc:
        head.errors.append(exc.message)
    except _AV_ERRORS as exc:
        head.errors.append(describe_av_error(exc))
    finally:
        container.close()
    return head


def _decode_timeline(path: Path, stream_index: int) -> Tuple[List[int], List[int], List[bool], TimestampFixer, List[str], _Head]:
    """Slow path: decode the whole stream and take the timeline from the decoded frames."""

    pts: List[int] = []
    durations: List[int] = []
    keyframes: List[bool] = []
    errors: List[str] = []
    head = _Head()
    try:
        container = open_container(path)
    except _AV_ERRORS as exc:
        raise CorruptedVideoError(
            f"The video stream of '{path.name}' cannot be opened for decoding: {describe_av_error(exc)}.",
            details={"path": str(path), "error": describe_av_error(exc)},
        ) from exc
    try:
        stream = select_video_stream(container, stream_index)
        _discard_other_streams(container, stream_index)
        stream.thread_type = "AUTO"
        fixer = TimestampFixer.for_stream(container, stream)
        try:
            for frame in iter_decoded_frames(container, stream, errors):
                if not pts:
                    head.pix_fmt = frame.format.name
                    head.width, head.height = int(frame.width), int(frame.height)
                    head.hdr = _hdr_from_frame(frame)
                pts.append(fixer(frame.pts, frame.dts, frame.duration))
                durations.append(int(frame.duration or 0))
                keyframes.append(bool(frame.key_frame))
        except _AV_ERRORS as exc:          # e.g. the decoder could not be opened at all
            errors.append(describe_av_error(exc))
    finally:
        container.close()
    head.complete = True
    return pts, durations, keyframes, fixer, errors, head


def _build_timeline(path: Path, stream_index: int) -> Tuple[FrameTimeline, TimelineScan, _Head, List[str]]:
    """Frame timeline of the video stream, the description of how it was found, and warnings."""

    started = time.perf_counter()
    warnings: List[str] = []
    table = _scan_packets(path, stream_index)
    has_pts = table.pts != _NOPTS
    damaged = int(np.count_nonzero(table.corrupt))
    shown = ~table.discard & ~table.corrupt
    if damaged:
        warnings.append(
            f"{damaged} video packet(s) are damaged (the file is probably truncated or was not copied completely); "
            "their frames are left out."
        )
    discarded = int(np.count_nonzero(table.discard))
    without_pts = int(np.count_nonzero(shown & ~has_pts))
    keep = shown & has_pts

    # Leading pictures: packets displayed before the first key frame of the stream (in demux order).
    # After a cut at an open-GOP key frame they reference pictures that no longer exist and the
    # decoder does not output them.  The first key frame may itself be hidden by an edit list.
    leading = 0
    key_positions = np.flatnonzero(table.keyframe & has_pts)
    if key_positions.size:
        first_key_pts = int(table.pts[key_positions[0]])
        is_leading = keep & (table.pts < first_key_pts)
        leading = int(np.count_nonzero(is_leading))
        keep &= ~is_leading

    positions = np.flatnonzero(keep)
    kept_pts = table.pts[positions]
    order = np.argsort(kept_pts, kind="stable")
    sorted_pts = kept_pts[order]

    reason = ""
    min_packets = 0
    if table.count == 0:
        reason = "the demuxer returned no video packets"
    elif table.ts_unreliable:
        reason = "the container format carries no timestamps"
    elif positions.size == 0:
        reason = "no video packet carries a presentation timestamp"
    elif without_pts:
        last_without = int(np.flatnonzero(shown & ~has_pts)[-1])
        if last_without >= HEAD_NOPTS_LIMIT:
            reason = f"{without_pts} video packets carry no presentation timestamp"
        else:
            min_packets = last_without + 1 + HEAD_NOPTS_MARGIN
    if not reason and sorted_pts.size > 1 and bool(np.any(np.diff(sorted_pts) <= 0)):
        reason = "several packets share the same presentation timestamp"
    if not reason and order.size and int(np.max(np.abs(order - np.arange(order.size)))) > MAX_REORDER_DISPLACEMENT:
        reason = "the presentation timestamps contradict the decoding order"
    if not reason and table.reorder_depth == 0 and kept_pts.size > 1 and bool(np.any(np.diff(kept_pts) <= 0)):
        # Without B-frames the decoder delivers the frames in stored order, so a timestamp that runs
        # backwards is a glitch of the file, not a reordering: sorting would swap two pictures' times.
        reason = "packet timestamps run backwards although the stream has no frame reordering"

    head = _Head()
    if not reason:
        head = _decode_head(path, stream_index, min_packets=min_packets)
        got = head.pts
        predicted = sorted_pts[: len(got)].tolist()
        if not got:
            reason = "the first frames could not be decoded" + (f" ({head.errors[0]})" if head.errors else "")
        elif any(value is None for value in got):
            reason = "decoded frames carry no presentation timestamp"
        elif got != predicted or (head.complete and len(got) != sorted_pts.size):
            reason = "the first decoded frames do not match the packet timestamps"
        elif min_packets and not head.complete and head.packets_fed < min_packets:
            reason = "packets without a presentation timestamp could not be verified"

    if not reason:
        timeline = FrameTimeline(
            time_base=table.time_base,
            pts=sorted_pts,
            durations=table.duration[positions][order],
            keyframes=table.keyframe[positions][order],
            synthesized=False,
        )
        scan = TimelineScan(
            method="packets", packets=table.count, discarded=discarded, leading_dropped=leading,
            without_pts=without_pts, decode_errors=len(head.errors), seconds=time.perf_counter() - started,
            damaged_packets=damaged,
        )
        if table.demux_error:
            warnings.append(
                f"The container could not be read to its end ({table.demux_error}); "
                f"the video is processed up to frame {len(timeline) - 1}."
            )
        return timeline, scan, head, warnings

    # ---- slow path -----------------------------------------------------------------------
    log.info("Timeline of %s: packet scan not usable (%s); decoding the whole stream instead.", path.name, reason)
    pts, durations, keyframes, fixer, errors, head = _decode_timeline(path, stream_index)
    if not pts:
        message = errors[0] if errors else (table.demux_error or reason)
        raise CorruptedVideoError(
            f"No video frame of '{path.name}' could be decoded ({message}).",
            details={"path": str(path), "packets": table.count, "errors": errors[:10], "reason": reason},
        )
    timeline = FrameTimeline(
        time_base=table.time_base,
        pts=np.asarray(pts, dtype=np.int64),
        durations=np.asarray(durations, dtype=np.int64),
        keyframes=np.asarray(keyframes, dtype=bool),
        synthesized=fixer.synthesized > 0,
    )
    scan = TimelineScan(
        method="decode", packets=table.count, discarded=discarded, leading_dropped=leading,
        without_pts=without_pts, synthesized_timestamps=fixer.synthesized, adjusted_timestamps=fixer.adjusted,
        decode_errors=len(errors), reason=reason, seconds=time.perf_counter() - started,
    )
    invented = fixer.synthesized - fixer.adjusted
    if invented > 0:
        warnings.append(
            f"{invented} of {len(pts)} video frames have no presentation timestamp in the container; their "
            "timestamps were derived the way FFmpeg does (from decoding timestamps and the frame rate)."
        )
    if fixer.adjusted > 0:
        warnings.append(
            f"{fixer.adjusted} video frame timestamp(s) were duplicated or out of order and were moved by "
            "the smallest possible amount to keep the timeline strictly increasing."
        )
    if errors:
        warnings.append(
            f"FFmpeg reported {len(errors)} error(s) while the video was scanned (first: {errors[0]}); "
            "frames that could not be decoded are not part of the timeline."
        )
    return timeline, scan, head, warnings


# =========================================================================== public entry point


def _check_path(path: Any) -> Path:
    try:
        p = Path(os.fspath(path))
    except TypeError as exc:
        raise InvalidPathError(f"Not a file path: {path!r}") from exc
    try:
        if not p.exists():
            raise InvalidPathError(f"Input file not found: {p}", details={"path": str(p)})
        if not p.is_file():
            raise InvalidPathError(f"Input path is not a file: {p}", details={"path": str(p)})
        size = p.stat().st_size
    except OSError as exc:
        raise InvalidPathError(f"Input file cannot be accessed: {p} ({exc})", details={"path": str(p)}) from exc
    if size == 0:
        raise CorruptedVideoError(
            f"'{p.name}' is empty (0 bytes).",
            hint="The copy or download did not complete; fetch the file again.",
            details={"path": str(p)},
        )
    return Path(os.path.abspath(p))


def _nominal_rate(timeline: FrameTimeline, candidates: Sequence[Optional[Fraction]]) -> Optional[Fraction]:
    """The frame rate the container declares, if the timeline is consistent with it.

    Matroska and WebM round every timestamp to a millisecond, so the mean over such a timeline
    (29.99 fps for a 30 fps clip) is not the frame rate.  A declared rate is accepted when the span of
    the timeline differs from what that rate predicts by no more than one tick of the time base.
    """

    count = len(timeline)
    for rate in candidates:
        if rate is None or rate <= 0:
            continue
        if count < 2:
            return rate
        span = Fraction(int(timeline.pts[-1]) - int(timeline.pts[0])) * timeline.time_base
        if abs(span - Fraction(count - 1) / rate) <= timeline.time_base:
            return rate
    return None


def _readable_rate(rate: Optional[Fraction]) -> Optional[Fraction]:
    if rate is None or rate <= 0:
        return None
    return rate.limit_denominator(READABLE_DENOMINATOR) if rate.denominator > READABLE_DENOMINATOR else rate


def probe(path: Any, *, ffprobe: Optional[str] = None, scan_timeline: bool = True) -> ProbeResult:
    """Analyse a video file.

    Parameters
    ----------
    path:
        The input file (``str`` or path-like; spaces and non-ASCII characters are fine).
    ffprobe:
        The ffprobe executable.  Default: ``VMT_FFPROBE`` or the one found on ``PATH``.
    scan_timeline:
        Build the exact frame timeline (default).  With ``False`` only the ffprobe metadata is
        gathered: the returned timeline is empty and ``frame_count`` / ``frame_rate`` /
        ``video_duration`` are the container's own (unverified) claims.

    Returns a :class:`VideoProbeResult` (a :class:`~app.core.types.ProbeResult` with a ``scan``
    attribute).  Invariants when the timeline was scanned::

        metadata.frame_count == len(timeline)
        metadata.is_vfr == timeline.is_vfr
        metadata.frame_rate == the declared rate when the timeline agrees with it, else
                               timeline.average_rate (denominator limited for readability)
        metadata.video_duration == timeline.duration

    Raises
    ------
    InvalidPathError        the path does not exist or is not a file
    UnsupportedFormatError  FFmpeg does not recognise the file as media
    NoVideoStreamError      the file has no video stream (audio only, cover art only)
    CorruptedVideoError     the file is empty, truncated or damaged, or contains no decodable frame
    FfmpegNotFoundError     ffprobe is not installed
    FfmpegFailedError       ffprobe could not be run or did not answer
    """

    p = _check_path(path)
    executable = _find_ffprobe(ffprobe)
    data = _run_ffprobe(executable, p)
    streams = [s for s in data.get("streams") or [] if isinstance(s, dict)]
    fmt = data.get("format") if isinstance(data.get("format"), dict) else {}
    format_tags = _tags(fmt)

    def attached(entry: Dict[str, Any]) -> bool:
        disposition = entry.get("disposition")
        return bool(isinstance(disposition, dict) and _to_int(disposition.get("attached_pic")))

    video_entries = [s for s in streams if s.get("codec_type") == "video"]
    primary = next((s for s in video_entries if not attached(s)), None)
    audio_entries = [s for s in streams if s.get("codec_type") == "audio"]
    if primary is None:
        if video_entries:
            what = "only an attached picture (cover art), no video stream"
        elif audio_entries:
            what = "audio only, no video stream"
        else:
            what = "no video stream"
        raise NoVideoStreamError(
            f"'{p.name}' contains {what}.",
            hint="Choose a file that contains a video track.",
            details={"path": str(p), "stream_types": [str(s.get("codec_type")) for s in streams]},
        )

    if (_clean(primary.get("codec_name")) or "").lower() == "ansi" or "tty" in (_clean(fmt.get("format_name")) or "").split(","):
        raise UnsupportedFormatError(
            f"'{p.name}' is a text file, not a video (FFmpeg would only render its characters as a picture).",
            hint=_NOT_A_VIDEO_HINT, details={"path": str(p), "stderr_tail": ""},
        )

    index = int(primary.get("index", 0))
    stream_tags = _tags(primary)
    width, height = _to_int(primary.get("width")) or 0, _to_int(primary.get("height")) or 0
    pix_fmt = _clean(primary.get("pix_fmt"))
    time_base, start_pts, start_time, stream_duration = _stream_times(primary, stream_tags)
    warnings: List[str] = []

    # ---- timeline ------------------------------------------------------------------------
    scan = TimelineScan(method="skipped")
    head = _Head()
    if scan_timeline:
        try:
            timeline, scan, head, scan_warnings = _build_timeline(p, index)
        except _AV_ERRORS as exc:          # safety net: PyAV errors never leave this module
            raise CorruptedVideoError(
                f"The video stream of '{p.name}' cannot be read: {describe_av_error(exc)}.",
                details={"path": str(p), "error": describe_av_error(exc)},
            ) from exc
        if len(timeline) == 0:
            raise CorruptedVideoError(
                f"The video stream of '{p.name}' contains no frames.", details={"path": str(p)}
            )
        if time_base is not None and time_base != timeline.time_base:
            log.warning("ffprobe and libavformat disagree about the time base of %s (%s vs %s); using %s.",
                        p.name, time_base, timeline.time_base, timeline.time_base)
        time_base = timeline.time_base
        if start_pts is None:
            # Known exactly from the timeline even where ffprobe cannot tell (very short Matroska files).
            start_pts = int(timeline.pts[0])
            start_time = Fraction(start_pts) * time_base
    else:
        if time_base is None:
            raise CorruptedVideoError(
                f"The video stream of '{p.name}' has no time base; the file is probably damaged.",
                details={"path": str(p)},
            )
        timeline = FrameTimeline(time_base=time_base, pts=np.zeros(0, np.int64), durations=np.zeros(0, np.int64),
                                 keyframes=np.zeros(0, bool))
        scan_warnings = []

    # What the decoder really delivers wins over the container's description of the stream.
    if head.pix_fmt:
        if (pix_fmt, width, height) != (head.pix_fmt, head.width, head.height) and pix_fmt is not None:
            log.info("%s: the decoder delivers %dx%d %s (container: %dx%d %s).", p.name, head.width, head.height,
                     head.pix_fmt, width, height, pix_fmt)
        pix_fmt, width, height = head.pix_fmt, head.width, head.height
    if not pix_fmt or width <= 0 or height <= 0:
        raise CorruptedVideoError(
            f"The size or pixel format of the video stream of '{p.name}' could not be determined; "
            "the file is probably truncated or damaged.",
            details={"path": str(p), "width": width, "height": height, "pix_fmt": pix_fmt},
        )

    # ---- video stream ----------------------------------------------------------------------
    rotation, rotation_warnings = _rotation(primary, stream_tags)
    warnings.extend(rotation_warnings)
    bit_depth, has_alpha = _pixel_format_facts(pix_fmt, _to_int(primary.get("bits_per_raw_sample")))
    hdr = _hdr_from_stream(primary)
    for key, value in head.hdr.items():
        hdr.setdefault(key, value)
    codec_tag = _clean(primary.get("codec_tag_string"))
    if codec_tag is not None and (codec_tag.startswith("[0]") or set(codec_tag) <= {"0", "[", "]"}):
        codec_tag = None
    nb_frames = _to_int(primary.get("nb_frames")) or _to_int(_tag(stream_tags, "NUMBER_OF_FRAMES", "NUMBER_OF_FRAMES-eng"))
    video = VideoStreamInfo(
        index=index,
        codec=_clean(primary.get("codec_name")) or "unknown",
        width=width,
        height=height,
        pix_fmt=pix_fmt,
        time_base=time_base,
        codec_long=_clean(primary.get("codec_long_name")) or "",
        profile=_clean(primary.get("profile")),
        codec_tag=codec_tag,
        bit_depth=bit_depth,
        has_alpha=has_alpha,
        color=ColorInfo(
            range=_color_tag(primary.get("color_range")),
            space=_color_tag(primary.get("color_space")),
            primaries=_color_tag(primary.get("color_primaries")),
            transfer=_color_tag(primary.get("color_transfer")),
            chroma_location=(_clean(primary.get("chroma_location")) or "unspecified").lower(),
        ),
        field_order=(_clean(primary.get("field_order")) or "unknown").lower(),
        rotation=rotation,
        sample_aspect_ratio=_rational(primary.get("sample_aspect_ratio")),
        display_aspect_ratio=_rational(primary.get("display_aspect_ratio")),
        start_pts=start_pts,
        start_time=start_time,
        duration=stream_duration,
        nb_frames=nb_frames,
        avg_frame_rate=_rational(primary.get("avg_frame_rate")),
        r_frame_rate=_rational(primary.get("r_frame_rate")),
        bit_rate=_to_int(primary.get("bit_rate")) or _to_int(_tag(stream_tags, "BPS", "BPS-eng")),
        hdr=hdr,
        tags=stream_tags,
    )

    # ---- timing --------------------------------------------------------------------------
    if scan_timeline:
        frame_count = len(timeline)
        is_vfr = bool(timeline.is_vfr)
        declared = None if is_vfr else _nominal_rate(timeline, (video.avg_frame_rate, video.r_frame_rate))
        frame_rate = declared or _readable_rate(timeline.average_rate) or video.avg_frame_rate or video.r_frame_rate
        video_duration: Optional[Fraction] = timeline.duration
        vfr_stats = timeline.vfr_stats()
        if nb_frames is not None and nb_frames != frame_count:
            hidden = f" ({scan.discarded} hidden by an edit list)" if scan.discarded else ""
            # Frames that are listed but neither readable nor hidden: the data ends before the index
            # does.  (When a packet was cut in half, the damaged-packet warning already says so.)
            cut_short = nb_frames > frame_count + scan.discarded and not scan.damaged_packets
            warnings.append(
                f"The container declares {nb_frames} video frames but {frame_count} are presented{hidden}; "
                "the presented frames are processed."
                + (" The file is probably truncated or was not copied completely." if cut_short else "")
            )
        warnings.extend(scan_warnings)
        if stream_duration is not None and video_duration is not None and frame_count > 1:
            # MP4 declares the length of the track, Matroska (DURATION tag) the time at which it ends:
            # the file is only suspicious if the frames fall short of both readings.
            end_time = timeline.time(frame_count - 1) + Fraction(timeline.frame_duration(frame_count - 1)) * time_base
            missing = min(stream_duration - video_duration, stream_duration - end_time)
            mean = video_duration / frame_count
            if missing > max(Fraction(1, 2), 3 * mean) and missing > stream_duration / 50:
                warnings.append(
                    f"The video stream is declared to last {float(stream_duration):.3f} s but its frames end after "
                    f"{float(video_duration):.3f} s; the file is probably truncated or damaged."
                )
    else:
        frame_rate = video.avg_frame_rate or video.r_frame_rate
        video_duration = stream_duration or _seconds(fmt.get("duration"))
        frame_count = nb_frames or 0
        if not frame_count and frame_rate and video_duration:
            frame_count = int(round(video_duration * frame_rate))
        is_vfr = False
        vfr_stats = {"is_vfr": False, "frames": frame_count, "scanned": False}

    real_extra = [s for s in video_entries if s is not primary and not attached(s)]
    if real_extra:
        warnings.append(
            f"The file has {len(real_extra) + 1} video streams; only the first one (stream #{index}) is processed."
        )

    format_name = _clean(fmt.get("format_name")) or ""
    try:
        size_bytes = p.stat().st_size
    except OSError:
        size_bytes = _to_int(fmt.get("size")) or 0
    metadata = VideoMetadata(
        path=str(p),
        size_bytes=int(size_bytes),
        format_name=format_name,
        container=_normalise_container(format_name, p, format_tags, streams),
        video=video,
        frame_count=int(frame_count),
        is_vfr=is_vfr,
        format_long_name=_clean(fmt.get("format_long_name")) or "",
        duration=_seconds(fmt.get("duration")),
        start_time=_seconds(fmt.get("start_time")),
        bit_rate=_to_int(fmt.get("bit_rate")),
        audio=tuple(_audio_stream(s) for s in audio_entries),
        subtitle_streams=sum(1 for s in streams if s.get("codec_type") == "subtitle"),
        data_streams=sum(1 for s in streams if s.get("codec_type") not in ("video", "audio", "subtitle")),
        extra_video_streams=len(video_entries) - 1,
        frame_rate=frame_rate,
        video_duration=video_duration,
        vfr=vfr_stats,
        tags=format_tags,
        warnings=tuple(dict.fromkeys(warnings)),
    )
    log.debug(
        "probe %s: %s %dx%d %s, %d frames (%s, %.3f s scan), vfr=%s",
        p.name, video.codec, width, height, pix_fmt, frame_count, scan.method, scan.seconds, is_vfr,
    )
    return VideoProbeResult(metadata=metadata, timeline=timeline, scan=scan)
