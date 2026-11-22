"""Frame-accurate decoding of the input video into native planes.

``VideoReader`` turns a probed file (``app.processing.probe.probe``) into a stream of
:class:`~app.core.types.DecodedFrame` objects whose indices, timestamps and count are *exactly* those
of the probe's :class:`~app.core.types.FrameTimeline`::

    [f.pts for f in reader.frames()] == list(reader.timeline.pts)

Design
------

* **Timestamps are looked up, never computed.**  A decoded frame is identified by its presentation
  timestamp, which is searched in the timeline.  That makes every access path (sequential, after a
  seek, in reverse) agree on the index of a frame and detects frames the decoder dropped, wherever
  they are.  Files whose timeline had to be synthesised (AVI, raw streams: no usable PTS) are decoded
  from the first packet through :class:`~app.processing.probe.TimestampFixer`, exactly like the probe
  did, and are never seeked into.
* **Correctness before speed.**  A seek is only an optimisation and is verified: the frames between
  the landing point and the requested frame must arrive without gaps and without decoder error flags.
  Otherwise the read falls back to the previous key frame and finally to a decode from the very
  beginning, which is the ground truth: only what *that* pass reports is treated as damage of the
  file.  What it found is remembered, so a damaged spot costs the long way round only once.
* **Each iteration owns its container.**  ``frames()`` / ``frames_reverse()`` can run on a background
  thread while ``read_frame()`` / ``frame_rgb()`` serve the user interface from another one.
  ``close()`` may be called from any thread at any time; iterators that are still alive raise
  :class:`ReaderClosedError` on their next step instead of touching a closed decoder.
* **No pixel is converted.**  Planes come from ``app.processing.planes.frame_to_planes`` (owned
  copies in the working format); ``frame.reformat`` / ``to_ndarray`` are never used and decoder
  buffers are never written to.

Decode errors
-------------

A frame *cannot be decoded* when the decoder does not deliver it (its timestamp never shows up) or the
stream ends before it.  With ``tolerate_errors=False`` this raises
:class:`~app.core.errors.CorruptedVideoError` naming the frame and its time; so does a frame the decoder
delivers but flags as damaged (the same condition makes ``ffmpeg -xerror`` stop).  With
``tolerate_errors=True`` missing frames are skipped and listed in ``skipped_frames``; flagged frames
are delivered (they show concealment artifacts, like in any player) and listed in ``corrupt_frames``;
``warnings`` describes both.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import av
import av.error
import numpy as np

from ..core.errors import CorruptedVideoError, InputError, InvalidPathError, UnsupportedFormatError
from ..core.pixfmt import PixFmt, ResolvedColor, resolve_color
from ..core.types import DecodedFrame, FrameTimeline, ProbeResult, VideoMetadata
from .planes import WorkingFormat, frame_to_planes, planes_to_rgb8, working_format_for
from .probe import (
    TimestampFixer,
    describe_av_error,
    iter_decoded_frames,
    open_container,
    select_video_stream,
)

log = logging.getLogger(__name__)

__all__ = ["VideoReader", "FrameIndexError", "ReaderClosedError", "ReverseReadError"]

#: Default memory budget for the frames ``frames_reverse`` holds at any time (bytes of planes).
DEFAULT_REVERSE_BUDGET_BYTES = 512 * 1024 * 1024
#: ``frames_reverse`` refuses a file only when reading it backwards is both hugely redundant (more
#: than this many frames decoded per frame delivered) ...
REVERSE_MAX_WORK_FACTOR = 30.0
#: ... and enormous in absolute terms (decoded pixels; 1.5e12 is roughly 12 hours of 1080p30 video).
REVERSE_MAX_WORK_PIXELS = 1.5e12
#: Seek attempts that turned out to be the seek's fault before seeking is switched off for a file.
SEEK_STRIKE_LIMIT = 3
#: ``read_frame`` keeps decoding forward instead of seeking when the wanted frame is this close
#: behind the key frame a seek would land on.
CURSOR_SLACK_FRAMES = 8
#: Frames just before a randomly accessed frame that are cached as well (cheap, and it makes stepping
#: backwards through a long GOP fast).
READ_BEHIND_FRAMES = 3
#: At most this many damaged regions are described individually in ``warnings``.
MAX_LISTED_REGIONS = 12

_RETRY = object()       # the current entry point is not trustworthy: fall back to a safer one
_CONTINUE = object()    # nothing to deliver yet (pre-roll or an unusable frame)
_END = object()         # the requested range is exhausted


# =========================================================================== errors


class FrameIndexError(InputError, IndexError):
    """A frame index outside ``0 .. frame_count - 1`` was requested."""

    code = "frame_out_of_range"


class ReaderClosedError(InputError):
    """The reader was closed while (or before) frames were being read."""

    code = "reader_closed"


class ReverseReadError(InputError):
    """The file cannot reasonably be read backwards (no usable seek points and far too long)."""

    code = "reverse_read_unsupported"


def _compact(indices: List[int], limit: int = 12) -> str:
    """``[1, 2, 3, 7]`` -> ``"1-3, 7"`` (at most *limit* ranges)."""

    if not indices:
        return ""
    ordered = sorted(set(int(i) for i in indices))
    ranges: List[str] = []
    first = previous = ordered[0]
    for value in ordered[1:] + [None]:  # type: ignore[list-item]
        if value is not None and value == previous + 1:
            previous = value
            continue
        ranges.append(str(first) if first == previous else f"{first}-{previous}")
        if value is not None:
            first = previous = value
    if len(ranges) > limit:
        return ", ".join(ranges[:limit]) + f", ... ({len(ordered)} in total)"
    return ", ".join(ranges)


# =========================================================================== decode session


class _Session:
    """One open container with its decoder.  Not thread-safe: every use happens under ``lock``."""

    def __init__(self, reader: "VideoReader") -> None:
        self.reader = reader
        self.lock = threading.Lock()
        self.closed = False
        self.errors: List[str] = []
        self.frames: Optional[Iterator["av.VideoFrame"]] = None
        self.fixer: Optional[TimestampFixer] = None
        try:
            self.container = open_container(reader.path)
        except av.error.FileNotFoundError as exc:
            raise InvalidPathError(
                f"Input file not found: {reader.path} (it was moved or deleted after it was analysed).",
                details={"path": str(reader.path)},
            ) from exc
        except (av.error.FFmpegError, OSError) as exc:
            raise CorruptedVideoError(
                f"'{reader.path.name}' cannot be opened for decoding: {describe_av_error(exc)}.",
                details={"path": str(reader.path), "error": describe_av_error(exc)},
            ) from exc
        try:
            self.stream = select_video_stream(self.container, reader.metadata.video.index)
            for other in self.container.streams:
                if other.index != self.stream.index:
                    try:
                        other.discard = type(other.discard).all      # never read audio / data packets
                    except Exception:
                        break
            self.stream.thread_type = "AUTO"                          # frame + slice threads (6-8x faster)
            if reader.threads > 0:
                self.stream.codec_context.thread_count = reader.threads
        except BaseException:
            self.container.close()
            raise
        reader._register(self)

    # -- positioning (call with ``lock`` held) -------------------------------------------
    def start(self, use_fixer: bool) -> None:
        """Decode from the first packet (only valid on a freshly opened session)."""

        self.fixer = TimestampFixer.for_stream(self.container, self.stream) if use_fixer else None
        self.frames = iter_decoded_frames(self.container, self.stream, self.errors)

    def seek(self, timestamp: int) -> bool:
        """Position at the key frame at or before *timestamp*; False if the demuxer refuses."""

        self.frames = None
        self.fixer = None
        try:
            self.container.seek(int(timestamp), stream=self.stream, backward=True, any_frame=False)
        except (av.error.FFmpegError, OSError) as exc:
            log.debug("seek to %s failed in %s: %s", timestamp, self.reader.path.name, describe_av_error(exc))
            return False
        self.frames = iter_decoded_frames(self.container, self.stream, self.errors)
        return True

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.frames = None
            try:
                self.container.close()
            except Exception as exc:  # pragma: no cover - closing must never fail the caller
                log.debug("closing the container of %s failed: %s", self.reader.path.name, exc)
        self.reader._unregister(self)


# =========================================================================== frame stream


class _FrameStream:
    """Delivers the frames ``start .. stop - 1`` in order, choosing and verifying the entry point.

    Entry points, from fast to safe: seek to the key frame at or before the next wanted frame, seek to
    the key frame before that one, decode from the first packet.  A seek-based pass is abandoned as
    soon as anything is unexpected (see the module docstring); the pass from the first packet is
    authoritative.
    """

    def __init__(self, reader: "VideoReader", start: int, stop: int, shared: Optional[_Session] = None) -> None:
        self.reader = reader
        self.expected = int(start)          # index of the next frame to deliver
        self.stop = int(stop)
        self.shared = shared                # long-lived session lent by the caller (used for seeks only)
        self.session: Optional[_Session] = None
        self.owned: Optional[_Session] = None
        self.level = 0                      # 0 = nearest key frame, 1 = key frame before, 2 = first packet
        self.seeking = False                # the current pass started with a seek
        self.previous: Optional[int] = None  # index of the last frame the current pass produced
        self.unsettled = 0                  # fall-backs whose cause (seek or damage) is not known yet
        self.finished = False

    # -- entry points ----------------------------------------------------------------------
    def _begin(self) -> None:
        reader = self.reader
        while True:
            target: Optional[int] = None
            if self.level < 2 and reader._can_seek():
                key = reader._key_at_or_before(self.expected)
                if self.level == 0:
                    target = key if key > 0 else (0 if self.shared is not None else None)
                else:
                    before = reader._key_at_or_before(key - 1) if key > 0 else -1
                    target = before if before > 0 else (0 if self.shared is not None and key > 0 else None)
            if target is None:
                self.level = 2
                self._release_owned()
                session = _Session(reader)
                self.owned = session
                with session.lock:
                    session.start(reader._use_fixer)
                self.session, self.seeking, self.previous = session, False, -1
                return
            session = self.shared
            if session is None:
                if self.owned is None:
                    self.owned = _Session(reader)
                session = self.owned
            with session.lock:
                if session.closed:
                    raise reader._closed_error()
                positioned = session.seek(int(reader._pts[target]))
            if positioned:
                self.session, self.seeking, self.previous = session, True, None
                return
            self.level += 1

    def _release_owned(self) -> None:
        if self.owned is not None:
            self.owned.close()
            self.owned = None

    def _fall_back(self) -> None:
        if self.seeking:
            self.unsettled += 1
        self.level = min(2, self.level + 1)
        self.session = None
        log.debug("%s: re-reading from a safer entry point (level %d) for frame %d",
                  self.reader.path.name, self.level, self.expected)

    # -- one decoder output ----------------------------------------------------------------
    def _advance(self, session: _Session) -> Any:
        reader = self.reader
        frames = session.frames
        frame = next(frames, None) if frames is not None else None
        seeking = self.seeking

        if frame is None:                                    # the stream is exhausted
            if self.expected >= self.stop:
                return _END
            if seeking:
                if self.previous is None or not reader._known_missing(self.expected, self.stop):
                    return _RETRY
            else:
                reader._confirm_missing((self.previous or 0) + 1 if self.previous is not None and self.previous >= 0 else 0,
                                        reader.frame_count)
                self.unsettled = 0
            first, self.expected = self.expected, self.stop
            reader._report_missing(first, self.stop, True, session.errors)
            return _END

        index = reader._index_of(session, frame)
        if index < 0:
            if seeking:
                return _RETRY
            reader._note_unknown_frame(frame)
            return _CONTINUE
        if self.previous is not None and index <= self.previous:
            return _CONTINUE              # late or repeated output after damage: it has no place any more

        corrupt = bool(frame.is_corrupt)
        if seeking:
            if self.previous is None:
                if index > self.expected and not reader._known_missing(self.expected, min(index, self.stop)):
                    return _RETRY         # the seek landed behind the wanted frame
            elif index > self.previous + 1 and not reader._known_missing(self.previous + 1, index):
                return _RETRY             # frames are missing between the landing point and here
            if corrupt and not reader._known_corrupt(index):
                return _RETRY             # may be an artifact of the entry point, not of the file
        else:
            if index > self.previous + 1:
                reader._confirm_missing(self.previous + 1, index)
            if corrupt:
                reader._confirm_corrupt(index)
        self.previous = index

        if index < self.expected:
            return _CONTINUE              # pre-roll between the entry point and the wanted frame
        reported = False
        if index > self.expected:
            first, self.expected = self.expected, min(index, self.stop)
            self.unsettled = 0
            reported = True
            reader._report_missing(first, self.expected, False, session.errors)
        if index >= self.stop:
            return _END
        reader._check_format(frame, index)
        if corrupt:
            self.unsettled = 0
            reported = True
            reader._report_corrupt(index, session.errors)
        planes = frame_to_planes(frame, reader.working)
        self.expected = index + 1
        if self.unsettled and not seeking:
            if not reported:
                reader._blame_seek()      # the same frame decodes cleanly from the start: the seek was at fault
            self.unsettled = 0
        return reader._decoded(index, planes)

    # -- public ----------------------------------------------------------------------------
    def next_frame(self) -> Optional[DecodedFrame]:
        """The next frame of the range, or None when the range is exhausted."""

        reader = self.reader
        while not self.finished:
            if self.expected >= self.stop:
                break
            if reader._closed:
                raise reader._closed_error()
            if self.session is None:
                self._begin()
            session = self.session
            assert session is not None
            with session.lock:
                if session.closed:
                    raise reader._closed_error()
                outcome = self._advance(session)
            if outcome is _RETRY:
                self._fall_back()
            elif outcome is _END:
                break
            elif outcome is not _CONTINUE:
                return outcome
        self.close()
        return None

    def close(self) -> None:
        self.finished = True
        self.session = None
        self._release_owned()


# =========================================================================== reader


class VideoReader:
    """Random and sequential access to the frames of a probed video file.

    Parameters
    ----------
    path:
        The file that was probed.
    probe_result:
        Result of :func:`app.processing.probe.probe` for *path* (with the timeline scanned).
    threads:
        Decoder threads per open decoder; 0 lets FFmpeg choose (one per CPU core, at most 16).
    tolerate_errors:
        Skip frames that cannot be decoded instead of raising (see the module docstring).
    cache_frames / cache_bytes:
        Size of the cache behind ``read_frame`` / ``frame_rgb`` (whichever limit is hit first; one
        frame is always kept).

    Attributes: ``path``, ``metadata``, ``timeline``, ``working`` (how the source pixel format maps
    to planes), ``fmt`` (the planar working format of the planes), ``width`` / ``height`` (coded
    size), ``rotation`` (clockwise display rotation), ``color`` (resolved matrix / range),
    ``frame_count``, ``skipped_frames``, ``corrupt_frames``, ``warnings``.
    """

    def __init__(
        self,
        path: Any,
        probe_result: ProbeResult,
        *,
        threads: int = 0,
        tolerate_errors: bool = False,
        cache_frames: int = 8,
        cache_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.path = Path(path)
        self.metadata: VideoMetadata = probe_result.metadata
        self.timeline: FrameTimeline = probe_result.timeline
        self.scan = getattr(probe_result, "scan", None)
        video = self.metadata.video
        if not self.path.is_file():
            raise InvalidPathError(f"Input file not found: {self.path}", details={"path": str(self.path)})
        if len(self.timeline) == 0:
            raise InputError(
                f"The probe result of '{self.path.name}' has no frame timeline.",
                hint="Probe the file with scan_timeline=True (the default) before creating a VideoReader.",
            )
        self.working: WorkingFormat = working_format_for(video.pix_fmt)
        self.fmt: PixFmt = self.working.fmt
        self.width: int = int(video.width)
        self.height: int = int(video.height)
        self.rotation: int = int(video.rotation)
        self.color: ResolvedColor = resolve_color(
            video.color, self.fmt, self.width, self.height, self.working.full_range_hint
        )
        self.frame_count: int = len(self.timeline)
        self.threads = max(0, int(threads))
        self.tolerate_errors = bool(tolerate_errors)
        self.skipped_frames: List[int] = []     # frames that could not be decoded (tolerate_errors)
        self.corrupt_frames: List[int] = []     # frames delivered although the decoder flagged them
        self.warnings: List[str] = []

        self._pts = np.ascontiguousarray(self.timeline.pts, dtype=np.int64)
        durations = np.zeros(self.frame_count, dtype=np.int64)
        if self.frame_count > 1:
            durations[:-1] = np.diff(self._pts)
        durations[-1] = self.timeline.frame_duration(self.frame_count - 1)
        self._durations = durations
        self._keyframes = np.asarray(self.timeline.keyframes, dtype=bool)
        self._key_indices = np.flatnonzero(self._keyframes)
        self._time_base = self.timeline.time_base
        self._frame_bytes = max(1, self.fmt.frame_bytes(self.width, self.height))
        # A synthesised timeline exists only as the result of decoding from the first packet.
        self._use_fixer = bool(self.timeline.synthesized)
        self._seekable = not self._use_fixer
        self._seek_strikes = 0
        self._missing = np.zeros(self.frame_count, dtype=bool)        # confirmed by a pass from the start
        self._damaged = np.zeros(self.frame_count, dtype=bool)
        self._regions: List[Tuple[int, int, bool, str]] = []          # reported (first, stop, ended, error)
        self._unknown_frames = 0

        self._state_lock = threading.Lock()
        self._sessions: List[_Session] = []
        self._closed = False

        self._random_lock = threading.Lock()
        self._cache: "OrderedDict[int, DecodedFrame]" = OrderedDict()
        self._cache_frames = max(1, int(cache_frames))
        self._cache_bytes = max(0, int(cache_bytes))
        self._cursor: Optional[_FrameStream] = None
        self._random_session: Optional[_Session] = None

    # ------------------------------------------------------------------------------ context manager
    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------------------ session bookkeeping
    def _closed_error(self) -> ReaderClosedError:
        return ReaderClosedError(
            f"The video reader for '{self.path.name}' was closed; no more frames can be read.",
            details={"path": str(self.path)},
        )

    def _register(self, session: _Session) -> None:
        with self._state_lock:
            if self._closed:
                closed = True
            else:
                closed = False
                self._sessions.append(session)
        if closed:
            session.closed = True
            session.container.close()
            raise self._closed_error()

    def _unregister(self, session: _Session) -> None:
        with self._state_lock:
            try:
                self._sessions.remove(session)
            except ValueError:
                pass

    def _require_open(self) -> None:
        if self._closed:
            raise self._closed_error()

    # ------------------------------------------------------------------------------ timeline helpers
    def _time(self, index: int) -> float:
        return float(self.timeline.time(index))

    def _key_at_or_before(self, index: int) -> int:
        """Index of the last flagged key frame at or before *index*; -1 if there is none."""

        if index < 0 or self._key_indices.size == 0:
            return -1
        position = int(np.searchsorted(self._key_indices, index, side="right")) - 1
        return int(self._key_indices[position]) if position >= 0 else -1

    def _can_seek(self) -> bool:
        return self._seekable

    def _index_of(self, session: _Session, frame: "av.VideoFrame") -> int:
        """Timeline index of a decoded frame, or -1 if its timestamp is not part of the timeline."""

        if session.fixer is not None:
            timestamp: Optional[int] = session.fixer(frame.pts, frame.dts, frame.duration)
        else:
            timestamp = frame.pts
        if timestamp is None:
            return -1
        position = int(np.searchsorted(self._pts, timestamp))
        if position < self.frame_count and int(self._pts[position]) == timestamp:
            return position
        return -1

    def _decoded(self, index: int, planes: List[np.ndarray]) -> DecodedFrame:
        return DecodedFrame(
            index=index,
            pts=int(self._pts[index]),
            duration=int(self._durations[index]),
            key_frame=bool(self._keyframes[index]),
            planes=planes,
            fmt=self.fmt,
        )

    def _check_format(self, frame: "av.VideoFrame", index: int) -> None:
        video = self.metadata.video
        if frame.width == self.width and frame.height == self.height and frame.format.name == video.pix_fmt:
            return
        raise UnsupportedFormatError(
            f"The video changes its picture format at frame {index} ({self._time(index):.3f} s): "
            f"{self.width}x{self.height} {video.pix_fmt} becomes {frame.width}x{frame.height} {frame.format.name}. "
            "Streams that change resolution or pixel format on the way are not supported.",
            hint="Re-encode the file to a constant resolution and pixel format first, for example with "
            "'ffmpeg -i in -c:v libx264 -crf 12 -c:a copy out.mp4'.",
            details={"frame_index": index, "path": str(self.path)},
        )

    # ------------------------------------------------------------------------------ damage bookkeeping
    def _known_missing(self, first: int, stop: int) -> bool:
        return stop > first and bool(self._missing[first:stop].all())

    def _known_corrupt(self, index: int) -> bool:
        return bool(self._damaged[index])

    def _confirm_missing(self, first: int, stop: int) -> None:
        if stop > first:
            self._missing[max(0, first):stop] = True

    def _confirm_corrupt(self, index: int) -> None:
        self._damaged[index] = True

    def _blame_seek(self) -> None:
        with self._state_lock:
            self._seek_strikes += 1
            if self._seek_strikes >= SEEK_STRIKE_LIMIT and self._seekable:
                self._seekable = False
                log.info("Seeking in %s is unreliable; frames are decoded from the beginning from now on.",
                         self.path.name)

    def _note_unknown_frame(self, frame: "av.VideoFrame") -> None:
        with self._state_lock:
            self._unknown_frames += 1
            if self._unknown_frames == 1:
                log.warning("%s: the decoder delivered a frame (pts %s) that is not part of the probed timeline; "
                            "it is ignored.", self.path.name, frame.pts)
            self._refresh_warnings()

    def _report_missing(self, first: int, stop: int, ended: bool, errors: List[str]) -> None:
        """Frames ``first .. stop - 1`` of a requested range cannot be decoded: skip them or raise."""

        if stop <= first:
            return
        error = errors[-1] if errors else ""
        count = stop - first
        if not self.tolerate_errors:
            if ended:
                message = (
                    f"The video stream of '{self.path.name}' ends early: frame {first} (at {self._time(first):.3f} s)"
                    + (f" and the {count - 1} frame(s) after it" if count > 1 else "")
                    + " could not be read. The file is probably truncated."
                )
            else:
                message = (
                    f"Frame {first} (at {self._time(first):.3f} s) of '{self.path.name}' could not be decoded"
                    + (f" ({count} consecutive frames, {first}-{stop - 1}, are affected)" if count > 1 else "")
                    + "."
                )
            if error:
                message += f" FFmpeg: {error}."
            raise CorruptedVideoError(
                message,
                details={
                    "path": str(self.path), "frame_index": int(first), "frame_count": int(count),
                    "time_seconds": self._time(first), "pts": int(self._pts[first]), "decoder_error": error,
                },
            )
        with self._state_lock:
            known = set(self.skipped_frames)
            new = [i for i in range(first, stop) if i not in known]
            if not new:
                return
            self.skipped_frames.extend(new)
            self.skipped_frames.sort()
            self._regions.append((first, stop, ended, error))
            self._refresh_warnings()
        log.warning("%s: frame(s) %s could not be decoded and are skipped%s.", self.path.name,
                    _compact(list(range(first, stop))), f" ({error})" if error else "")

    def _report_corrupt(self, index: int, errors: List[str]) -> None:
        """A frame of a requested range was decoded with errors: deliver it with a warning or raise."""

        error = errors[-1] if errors else ""
        if not self.tolerate_errors:
            raise CorruptedVideoError(
                f"Frame {index} (at {self._time(index):.3f} s) of '{self.path.name}' is damaged: the decoder "
                "reported errors in it and could only conceal them." + (f" FFmpeg: {error}." if error else ""),
                details={
                    "path": str(self.path), "frame_index": int(index), "time_seconds": self._time(index),
                    "pts": int(self._pts[index]), "decoder_error": error, "concealed": True,
                },
            )
        with self._state_lock:
            if index in self.corrupt_frames:
                return
            self.corrupt_frames.append(index)
            self.corrupt_frames.sort()
            self._refresh_warnings()

    def _refresh_warnings(self) -> None:
        """Rebuild ``warnings`` (in place) from what is known; call with ``_state_lock`` held."""

        texts: List[str] = []
        regions = sorted(set(self._regions))
        for first, stop, ended, error in regions[:MAX_LISTED_REGIONS]:
            where = (f"Frame {first} ({self._time(first):.3f} s)" if stop - first == 1 else
                     f"Frames {first}-{stop - 1} ({self._time(first):.3f} s to {self._time(stop - 1):.3f} s)")
            why = "the video stream ends early (truncated file)" if ended else "could not be decoded"
            verb = "is" if stop - first == 1 else "are"
            text = f"{where}: {why}; {'it' if stop - first == 1 else 'they'} {verb} skipped"
            texts.append(text + (f" (FFmpeg: {error})." if error else "."))
        if len(regions) > MAX_LISTED_REGIONS:
            texts.append(f"{len(regions) - MAX_LISTED_REGIONS} more damaged region(s) are not listed individually; "
                         f"{len(self.skipped_frames)} frames were skipped in total.")
        if self.skipped_frames:
            texts.append(
                "Frames that follow a skipped frame may show decoding artifacts until the next key frame."
            )
        if self.corrupt_frames:
            texts.append(
                f"{len(self.corrupt_frames)} frame(s) were decoded with errors and may show artifacts "
                f"(frames {_compact(self.corrupt_frames)})."
            )
        if self._unknown_frames:
            texts.append(
                f"{self._unknown_frames} decoded frame(s) carried timestamps that are not part of the analysed "
                "timeline and were ignored."
            )
        self.warnings[:] = texts

    # ------------------------------------------------------------------------------ range checks
    def _check_index(self, index: Any) -> int:
        try:
            value = int(index)
        except (TypeError, ValueError) as exc:
            raise FrameIndexError(f"Not a frame index: {index!r}") from exc
        if value < 0 or value >= self.frame_count:
            raise FrameIndexError(
                f"Frame {value} does not exist: '{self.path.name}' has {self.frame_count} frames "
                f"(0 to {self.frame_count - 1}).",
                details={"frame_index": value, "frame_count": self.frame_count},
            )
        return value

    def _check_range(self, start: Any, stop: Any) -> Tuple[int, int]:
        try:
            first = int(start)
            last = self.frame_count if stop is None else int(stop)
        except (TypeError, ValueError) as exc:
            raise FrameIndexError(f"Not a frame range: {start!r} .. {stop!r}") from exc
        if first < 0 or last < 0:
            raise FrameIndexError(
                f"Frame ranges cannot be negative (got {first} .. {last}).",
                details={"start": first, "stop": last, "frame_count": self.frame_count},
            )
        last = min(last, self.frame_count)
        return min(first, last), last

    # ------------------------------------------------------------------------------ sequential access
    def frames(self, start: int = 0, stop: Optional[int] = None) -> Iterator[DecodedFrame]:
        """Iterate the frames ``start .. stop - 1`` in presentation order.

        ``stop=None`` (or anything beyond the end) means "to the last frame".  The iterator opens its
        own container on first use and releases it when it is exhausted, closed or garbage collected;
        it may be consumed on any thread.  For ``start > 0`` the decoder is positioned on the key frame
        at or before *start* and the frames in between are decoded and dropped; if that is not
        possible or not reliable for the file, decoding starts at the first frame.
        """

        first, last = self._check_range(start, stop)
        self._require_open()
        return self._iterate(first, last)

    def _iterate(self, start: int, stop: int) -> Iterator[DecodedFrame]:
        stream = _FrameStream(self, start, stop)
        try:
            while True:
                frame = stream.next_frame()
                if frame is None:
                    return
                yield frame
        finally:
            stream.close()

    # ------------------------------------------------------------------------------ reverse access
    def frames_reverse(
        self, start: int, stop: int = 0, *, memory_budget_bytes: int = DEFAULT_REVERSE_BUDGET_BYTES
    ) -> Iterator[DecodedFrame]:
        """Iterate the frames ``start - 1, start - 2, ..., stop`` (descending), same content as ``frames()``.

        Video can only be decoded forwards, so the range is cut at key frames and each piece (GOP) is
        decoded forwards and handed out backwards.  At most ``memory_budget_bytes`` of planes are held:
        a GOP that does not fit is read in several forward passes, each keeping one window of frames
        (the last window first).  Files that cannot be seeked are read in passes from their first
        frame.

        Raises :class:`ReverseReadError` (immediately, not on first use) if that would mean decoding
        an absurd multiple of the video, and :class:`FrameIndexError` for an invalid range.
        """

        try:
            first, last = int(start), int(stop)
        except (TypeError, ValueError) as exc:
            raise FrameIndexError(f"Not a frame range: {start!r} .. {stop!r}") from exc
        if not 0 <= last <= first <= self.frame_count:
            raise FrameIndexError(
                f"Invalid reverse range {first} .. {last}: it must satisfy 0 <= stop <= start <= {self.frame_count}.",
                details={"start": first, "stop": last, "frame_count": self.frame_count},
            )
        self._require_open()
        windows = self._plan_reverse(first, last, int(memory_budget_bytes))
        return self._iterate_reverse(windows)

    def _plan_reverse(self, start: int, stop: int, budget: int) -> List[Tuple[int, int]]:
        """Windows ``(first, stop)`` in delivery order; each is decoded forwards from its key frame."""

        window = max(1, int(budget) // self._frame_bytes)
        seekable = self._can_seek()
        windows: List[Tuple[int, int]] = []
        work = 0
        upper = start
        while upper > stop:
            key = max(0, self._key_at_or_before(upper - 1)) if seekable else 0
            lower = max(key, stop)
            high = upper
            while high > lower:
                low = max(lower, high - window)
                windows.append((low, high))
                work += high - key                       # this pass decodes key .. high - 1
                high = low
            upper = lower
        count = start - stop
        if count and work > REVERSE_MAX_WORK_FACTOR * count and work * self.width * self.height > REVERSE_MAX_WORK_PIXELS:
            reason = ("its frames can only be reached by decoding from the first frame" if not seekable else
                      "it has too few key frames")
            raise ReverseReadError(
                f"'{self.path.name}' cannot be read backwards from frame {start - 1} to frame {stop}: {reason}, "
                f"so about {work:,} frames would have to be decoded to deliver {count:,}.",
                hint="Choose a reference frame closer to the start of the video, or re-encode the file with "
                "regular key frames (for example 'ffmpeg -i in -c:v libx264 -g 60 -crf 12 -c:a copy out.mp4').",
                details={"start": start, "stop": stop, "decode_work_frames": int(work), "window_frames": int(window)},
            )
        return windows

    def _iterate_reverse(self, windows: List[Tuple[int, int]]) -> Iterator[DecodedFrame]:
        shared: Optional[_Session] = None
        buffer: List[DecodedFrame] = []
        try:
            for low, high in windows:
                if shared is None and self._can_seek():
                    shared = _Session(self)                  # one container for all the seeks of this pass
                stream = _FrameStream(self, low, high, shared)
                try:
                    while True:
                        frame = stream.next_frame()
                        if frame is None:
                            break
                        buffer.append(frame)
                finally:
                    stream.close()
                while buffer:
                    yield buffer.pop()
        finally:
            buffer.clear()
            if shared is not None:
                shared.close()

    # ------------------------------------------------------------------------------ random access
    def _cache_put(self, frame: DecodedFrame) -> None:
        self._cache[frame.index] = frame
        self._cache.move_to_end(frame.index)
        while len(self._cache) > 1 and (
            len(self._cache) > self._cache_frames or len(self._cache) * self._frame_bytes > self._cache_bytes
        ):
            self._cache.popitem(last=False)

    def _drop_cursor(self) -> None:
        if self._cursor is not None:
            self._cursor.close()
            self._cursor = None

    def _frame(self, index: int) -> DecodedFrame:
        """The cached frame object for *index* (shared: callers must not modify its planes)."""

        index = self._check_index(index)
        with self._random_lock:
            self._require_open()
            cached = self._cache.get(index)
            if cached is not None:
                self._cache.move_to_end(index)
                return cached
            seekable = self._can_seek()
            key = max(0, self._key_at_or_before(index)) if seekable else 0
            cursor = self._cursor
            # Keep decoding forward when the wanted frame is ahead of the decoder and a seek would not
            # skip anything worth mentioning: stepping through a video must not re-decode its GOP.
            reuse = (
                cursor is not None and not cursor.finished
                and cursor.expected <= index and cursor.expected >= key - CURSOR_SLACK_FRAMES
            )
            if not reuse:
                self._drop_cursor()
                if seekable and (self._random_session is None or self._random_session.closed):
                    self._random_session = _Session(self)
                first = max(key, index - READ_BEHIND_FRAMES)
                cursor = _FrameStream(self, first, self.frame_count, self._random_session if seekable else None)
                self._cursor = cursor
            assert cursor is not None
            found: Optional[DecodedFrame] = None
            try:
                while True:
                    frame = cursor.next_frame()
                    if frame is None:
                        break
                    if frame.index >= index - READ_BEHIND_FRAMES:
                        self._cache_put(frame)
                    if frame.index >= index:
                        found = frame if frame.index == index else None
                        break
            except BaseException:
                self._drop_cursor()
                raise
            if cursor.finished:
                self._cursor = None
            if found is None:
                raise CorruptedVideoError(
                    f"Frame {index} (at {self._time(index):.3f} s) of '{self.path.name}' could not be decoded.",
                    details={"path": str(self.path), "frame_index": index, "time_seconds": self._time(index),
                             "pts": int(self._pts[index])},
                )
            return found

    def read_frame(self, index: int) -> DecodedFrame:
        """Decode the single frame *index* (random access).

        Recently used frames are cached and the decoder keeps its position, so stepping forward frame
        by frame decodes each frame once.  Safe to call from several threads.  The returned frame owns
        its planes.  A frame that cannot be decoded raises ``CorruptedVideoError`` (also with
        ``tolerate_errors``: there is nothing to return in its place).
        """

        frame = self._frame(index)
        return DecodedFrame(
            index=frame.index, pts=frame.pts, duration=frame.duration, key_frame=frame.key_frame,
            planes=[plane.copy() for plane in frame.planes], fmt=frame.fmt,
        )

    def frame_rgb(self, index: int, max_side: Optional[int] = None) -> np.ndarray:
        """Frame *index* as 8-bit R'G'B' ``uint8 [H, W, 3]`` in **display orientation**.

        For the user interface and subject selection only (never for output pixels): the planes are
        converted with the resolved matrix / range, turned by ``rotation`` (clockwise, like a player
        does) and, if ``max_side`` is given, reduced so that the longer side is at most ``max_side``.
        """

        frame = self._frame(index)
        rgb = planes_to_rgb8(frame.planes, self.fmt, self.color)
        turns = (self.rotation // 90) % 4
        if turns:
            rgb = np.rot90(rgb, k=-turns)          # np.rot90 turns counter-clockwise for positive k
        rgb = np.ascontiguousarray(rgb)
        if max_side is not None:
            limit = int(max_side)
            if limit < 1:
                raise FrameIndexError(f"max_side must be a positive number of pixels, not {max_side!r}.")
            height, width = rgb.shape[:2]
            longest = max(height, width)
            if longest > limit:
                import cv2

                scale = limit / float(longest)
                size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
                rgb = np.ascontiguousarray(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA))
        return rgb

    # ------------------------------------------------------------------------------ lifetime
    def close(self) -> None:
        """Release every container (idempotent, callable from any thread).

        Iterators that are still alive stop with :class:`ReaderClosedError` on their next step.
        """

        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            sessions = list(self._sessions)
        for session in sessions:
            session.close()
        with self._random_lock:
            self._cursor = None
            self._random_session = None
            self._cache.clear()
