"""Automatic verification of a finished output against its source.

After every run the delivered file is read back and compared with what was promised: the same
resolution, orientation, frame count and frame times, the audio untouched and in sync, a perfectly
uniform background of the requested colour, the subject's pixels preserved, no black or missing
frames.  Each test becomes a :class:`~app.core.report.QualityCheck` with a stable id, and the
input-versus-output property table of the report is assembled here too.

Design rules:

* A failed check is a *result*, never an exception; only cancellation raises.
* A check that cannot be evaluated is reported as ``skipped`` with the reason - never as ``pass``.
* ``critical`` is set when the file cannot be trusted at all (unreadable, undecodable, truncated):
  the pipeline then discards it instead of delivering it.

Two container facts shape the comparisons (both measured on real files):

* Matroska / WebM store timestamps in milliseconds, so frame times there may differ from the source
  by up to half a millisecond.
* A container that cannot hide encoder priming (AAC copied from MP4 into Matroska) starts *all*
  streams some milliseconds later.  Frame times are therefore compared as a timeline *shape*
  (relative to the first frame) and audio sync is measured between the first audio packet and the
  first video frame, which is unaffected by such a common shift.
"""

from __future__ import annotations

import logging
import math
import re
import subprocess
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.config import VideoCodec
from ..core.errors import JobCancelledError, VmtError
from ..core.events import CancelToken
from ..core.pixfmt import PixFmt, ResolvedColor, make_pixfmt
from ..core.report import CheckStatus, ComparisonRow, QualityCheck
from ..core.types import FrameTimeline, VideoMetadata
from ..encoding.ffmpeg import CREATE_NO_WINDOW, FfmpegTools, _kill, _StderrCollector
from ..processing.probe import probe
from ..processing.reader import VideoReader

log = logging.getLogger(__name__)

__all__ = ["FrameSample", "VerifyInput", "VerifyResult", "verify_output"]

PASS, WARN, FAIL, SKIPPED = CheckStatus.PASS, CheckStatus.WARN, CheckStatus.FAIL, CheckStatus.SKIPPED

#: Every check, in report order: id -> title.
CHECKS: Dict[str, str] = {
    "integrity": "File integrity",
    "decodable": "Full decode",
    "resolution": "Resolution",
    "orientation": "Orientation",
    "pixel_format": "Pixel format",
    "color_tags": "Colour tags",
    "frame_count": "Frame count",
    "timestamps": "Frame timestamps",
    "frame_rate": "Frame rate",
    "duration": "Duration",
    "audio_streams": "Audio streams",
    "av_sync": "Audio / video sync",
    "background_color": "Background colour",
    "alpha_channel": "Alpha channel",
    "foreground": "Foreground preservation",
    "black_frames": "Black frames",
    "matte_flicker": "Matte stability",
    "subject_tracking": "Subject tracking",
}

_ERROR_LINE = re.compile(r"\[(error|fatal|panic)\]", re.IGNORECASE)
_BLACK_LINE = re.compile(r"black_start:\s*([-+0-9.eE]+)(?:\s+black_end:\s*([-+0-9.eE]+))?")
_PROGRESS_FRAME = re.compile(r"^frame=\s*(\d+)")


@dataclass
class FrameSample:
    """What one output frame must look like, captured while it was rendered."""

    index: int                        # position of the frame in the OUTPUT (0-based)
    pts: int                          # source PTS of that frame (source time base)
    source_planes: List[np.ndarray]   # the source frame's planes in the source working format
    background_mask: np.ndarray       # bool [H, W], coded orientation: matte exactly 0, away from edges
    core_mask: np.ndarray             # bool [H, W]: matte exactly 1, away from edges


@dataclass
class VerifyInput:
    source: VideoMetadata
    source_pts: np.ndarray            # int64 PTS of every frame that was written, in order
    source_time_base: Fraction
    output_path: Path
    codec: VideoCodec
    container: str
    src_fmt: PixFmt
    out_fmt: PixFmt
    resolved_color: ResolvedColor
    width: int
    height: int
    resized: bool
    background: Optional[Tuple[int, int, int]]     # None = transparent output
    lossless: bool
    audio_mode: str                                # "copy" | "transcode" | "none"
    expect_audio_streams: int
    truncated: bool                                # only the first part of the source was processed
    samples: List[FrameSample] = field(default_factory=list)
    refine_stats: Dict[str, Any] = field(default_factory=dict)
    tracking_report: Dict[str, Any] = field(default_factory=dict)
    #: Seconds the muxer deliberately added to every timestamp (minus a start offset common to all
    #: source streams); informational - frame times are compared as a shape.
    time_shift: Fraction = Fraction(0)


@dataclass
class VerifyResult:
    checks: List[QualityCheck]
    comparison: List[ComparisonRow]
    output: Optional[VideoMetadata]
    critical: bool


# =========================================================================== small helpers


def _ms(seconds: Any) -> float:
    return round(float(seconds) * 1000.0, 4)


def _effective_range(tag: str, assumed_full: bool) -> str:
    tag = (tag or "unknown").lower()
    if tag in ("pc", "jpeg", "full"):
        return "pc"
    if tag in ("tv", "mpeg", "limited"):
        return "tv"
    return "pc" if assumed_full else "tv"


def _color_only(fmt: PixFmt) -> PixFmt:
    return make_pixfmt(fmt.family, fmt.bit_depth, fmt.chroma_w, fmt.chroma_h, False)


def _same_layout(a: PixFmt, b: PixFmt) -> bool:
    return (a.family, a.bit_depth, a.chroma_w, a.chroma_h) == (b.family, b.bit_depth, b.chroma_w, b.chroma_h)


def _plane_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Reduce a luma-resolution mask to a plane's resolution: a plane sample counts only if every
    picture sample it covers is set (so chroma samples that straddle an edge are excluded)."""

    if mask.shape == tuple(shape):
        return mask
    height, width = shape
    fy = max(1, int(round(mask.shape[0] / height)))
    fx = max(1, int(round(mask.shape[1] / width)))
    padded = np.zeros((height * fy, width * fx), dtype=bool)
    h = min(mask.shape[0], padded.shape[0])
    w = min(mask.shape[1], padded.shape[1])
    padded[:h, :w] = mask[:h, :w]
    return padded.reshape(height, fy, width, fx).all(axis=(1, 3))


def _timeline_from(pts: np.ndarray, time_base: Fraction) -> FrameTimeline:
    n = len(pts)
    return FrameTimeline(time_base, np.asarray(pts, dtype=np.int64), np.zeros(n, np.int64), np.zeros(n, bool))


def _time_step(container: Optional[str], time_base: Fraction) -> Fraction:
    """The step in which the output container really counts time.

    Normally that is the time base of the video stream.  Matroska and WebM store timestamps in
    milliseconds whatever time base a demuxer reports for the stream (FFmpeg versions differ in
    what they report), so their step is never finer than one millisecond.
    """

    step = Fraction(time_base)
    if container in ("mkv", "webm"):
        return max(step, Fraction(1, 1000))
    return step


def _duration_tolerance(reference: FrameTimeline, src_tb: Fraction, out_tb: Fraction) -> Fraction:
    """How far the video duration of the output may differ from that of the source.

    The time from the first to the last frame is carried exactly, but how long the *last* frame is
    shown is not stored per frame in the intermediate stream: the encoder estimates it.  With a
    variable frame rate that estimate can be off by up to the longest frame duration of the clip.
    """

    deltas = reference.deltas()
    longest = int(deltas.max()) if deltas.size else 1
    return Fraction(max(1, longest)) * src_tb + out_tb


def _run_capture(command: Sequence[str], timeout: float = 600.0) -> Tuple[int, str, str]:
    try:
        result = subprocess.run(
            list(command), stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout,
            creationflags=CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return -1, "", str(exc)
    return result.returncode, result.stdout.decode("utf-8", "replace"), result.stderr.decode("utf-8", "replace")


def _audio_packet_hash(tools: FfmpegTools, path: str, index: int) -> Optional[str]:
    """SHA-256 over the packets of one audio stream (stream copy: no decoding)."""

    code, out, _ = _run_capture(
        [tools.ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-i", str(path), "-map", f"0:a:{index}",
         "-c", "copy", "-f", "streamhash", "-hash", "sha256", "-"]
    )
    match = re.search(r"sha256=([0-9a-fA-F]{64})", out or "", re.IGNORECASE)
    return match.group(1).lower() if code == 0 and match else None


def _audio_packet_count(tools: FfmpegTools, path: str, index: int) -> Optional[int]:
    code, out, _ = _run_capture(
        [tools.ffprobe, "-v", "error", "-select_streams", f"a:{index}", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)]
    )
    try:
        return int(out.strip().split(",")[0]) if code == 0 else None
    except (ValueError, IndexError):
        return None


def _audio_head(tools: FfmpegTools, path: str, index: int, count: int) -> List[Tuple[float, int]]:
    """``(presentation time, size)`` of the first *count* packets of one audio stream."""

    code, out, _ = _run_capture(
        [tools.ffprobe, "-v", "error", "-select_streams", f"a:{index}", "-read_intervals", f"%+#{int(count)}",
         "-show_entries", "packet=pts_time,size", "-of", "csv=p=0", str(path)], timeout=120.0,
    )
    packets: List[Tuple[float, int]] = []
    if code != 0:
        return packets
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        try:
            packets.append((float(parts[0]), int(parts[1])))
        except (ValueError, IndexError):
            continue
    return packets


def _matched_audio_start(
    source_head: List[Tuple[float, int]], output_head: List[Tuple[float, int]]
) -> Optional[Tuple[float, float]]:
    """Times of the same audio packet in the source and in the output, or None if no match is found.

    With stream copy the output starts with the packets of the source; a container that cannot hide
    lead-in audio may have dropped a few at the head, so the first output packets are located in the
    source by their sizes.
    """

    if not source_head or not output_head:
        return None
    probe_sizes = [size for _, size in output_head[:4]]
    for start in range(0, len(source_head) - len(probe_sizes) + 1):
        if [size for _, size in source_head[start : start + len(probe_sizes)]] == probe_sizes:
            return source_head[start][0], output_head[0][0]
    return None


def _decode_pass(
    tools: FfmpegTools,
    path: Path,
    total_frames: int,
    detect_black: bool,
    cancel: Optional[CancelToken],
    progress: Optional[Callable[[float], None]],
) -> Tuple[int, List[str], List[Tuple[float, Optional[float]]]]:
    """Decode the whole video stream once, strictly.

    Returns ``(return code, error lines, black intervals)``.  Messages are requested with their level
    prefixed (``-loglevel level+info``) so that genuine decode errors can be told apart from the
    informational output of the black-frame detector that shares the pass.
    """

    command = [
        tools.ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "level+info" if detect_black else "level+error",
        "-xerror", "-err_detect", "explode", "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
    ]
    if detect_black:
        # A frame counts as black when 98 % of its pixels are in the darkest 10 % of the luma range.
        command += ["-vf", "blackdetect=d=0:pic_th=0.98:pix_th=0.10"]
    command += ["-progress", "pipe:1", "-f", "null", "-"]
    try:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=CREATE_NO_WINDOW)
    except OSError as exc:
        return -1, [f"ffmpeg could not be started: {exc}"], []
    collector = _StderrCollector(proc.stderr, name="vmt-verify-stderr")
    collector.start()
    latest = {"frame": 0}

    def read_progress() -> None:
        try:
            for raw in iter(proc.stdout.readline, b""):
                match = _PROGRESS_FRAME.match(raw.decode("ascii", "replace"))
                if match:
                    latest["frame"] = int(match.group(1))
        except (OSError, ValueError):
            pass

    reader = threading.Thread(target=read_progress, name="vmt-verify-progress", daemon=True)
    reader.start()
    try:
        while True:
            try:
                proc.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                pass
            if cancel is not None and cancel.cancelled:
                _kill(proc)
                cancel.raise_if_cancelled()
            if progress is not None and total_frames > 0:
                progress(min(1.0, latest["frame"] / float(total_frames)))
    except BaseException:
        if proc.poll() is None:
            _kill(proc)
        raise
    finally:
        if proc.poll() is not None:
            collector.join(timeout=10)
            reader.join(timeout=5)
            for stream in (proc.stdout, proc.stderr):
                try:
                    stream.close()
                except OSError:
                    pass
    text = collector.text()
    errors = [line.strip() for line in text.splitlines() if _ERROR_LINE.search(line)]
    black: List[Tuple[float, Optional[float]]] = []
    for line in text.splitlines():
        if "black_start" not in line:
            continue
        match = _BLACK_LINE.search(line)
        if match:
            try:
                black.append((float(match.group(1)), float(match.group(2)) if match.group(2) else None))
            except ValueError:
                pass
    return proc.returncode, errors, black


# =========================================================================== verification


def verify_output(
    data: VerifyInput,
    tools: FfmpegTools,
    *,
    cancel: Optional[CancelToken] = None,
    progress: Optional[Callable[[float], None]] = None,
) -> VerifyResult:
    """Run every check on ``data.output_path`` and build the comparison table."""

    results: Dict[str, QualityCheck] = {}

    def put(check_id: str, status: CheckStatus, detail: str, expected: Any = None, measured: Any = None) -> None:
        results[check_id] = QualityCheck(id=check_id, name=CHECKS[check_id], status=status, detail=detail,
                                         expected=expected, measured=measured)

    def tick(fraction: float) -> None:
        if cancel is not None:
            cancel.raise_if_cancelled()
        if progress is not None:
            try:
                progress(max(0.0, min(1.0, float(fraction))))
            except Exception:  # a progress callback must never break verification
                pass

    def guarded(check_id: str, function: Callable[[], None]) -> None:
        """Run one check; an unexpected error makes it 'skipped' (visible), never 'pass'."""

        try:
            function()
        except JobCancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("Verification check '%s' could not be evaluated", check_id)
            put(check_id, SKIPPED, f"Could not be evaluated ({type(exc).__name__}: {exc}).")

    def finish(output: Optional[VideoMetadata], comparison: List[ComparisonRow], critical: bool) -> VerifyResult:
        ordered: List[QualityCheck] = []
        for check_id, title in CHECKS.items():
            if check_id in results:
                ordered.append(results[check_id])
            elif critical:
                ordered.append(QualityCheck(id=check_id, name=title, status=SKIPPED,
                                            detail="Not evaluated because the file is unusable."))
        return VerifyResult(checks=ordered, comparison=comparison, output=output, critical=critical)

    path = Path(data.output_path)
    source, sv = data.source, data.source.video
    src_tb = Fraction(data.source_time_base)
    src_pts = np.asarray(data.source_pts, dtype=np.int64)
    expected_frames = int(src_pts.shape[0])
    tick(0.0)

    # ------------------------------------------------------------------ integrity (critical)
    try:
        size = path.stat().st_size
    except OSError:
        size = -1
    if size <= 0:
        put("integrity", FAIL, "The output file is missing or empty.", measured=size)
        return finish(None, [], True)
    try:
        out = probe(path, ffprobe=tools.ffprobe)
    except VmtError as exc:
        put("integrity", FAIL, f"The output file cannot be read: {exc.message}")
        return finish(None, [], True)
    om, ov, timeline = out.metadata, out.metadata.video, out.timeline
    put("integrity", PASS, f"Readable {om.container} file with a {ov.codec} video stream.", measured=size)

    # ------------------------------------------------------------------ full decode (critical) + black frames
    detect_black = data.background is not None and tuple(data.background) != (0, 0, 0)
    returncode, errors, black = _decode_pass(
        tools, path, max(1, len(timeline)), detect_black, cancel,
        (lambda fraction: tick(0.75 * fraction)) if progress is not None else None,
    )
    critical = False
    if returncode != 0 or errors:
        reason = errors[0] if errors else f"FFmpeg exit code {returncode}"
        put("decodable", FAIL, f"Decoding the output reported errors: {reason[:300]}",
            measured={"returncode": returncode, "errors": errors[:5]})
        critical = True
    else:
        put("decodable", PASS, "Every frame decodes without errors.")
    if not detect_black:
        put("black_frames", SKIPPED, "Not applicable: the background is black or transparent.")
    elif critical:
        put("black_frames", SKIPPED, "Not evaluated because decoding failed.")
    elif black:
        spans = ", ".join(f"{start:.3f} s" + (f"-{end:.3f} s" if end is not None else "") for start, end in black[:6])
        put("black_frames", FAIL, f"{len(black)} black passage(s) found at {spans}.", expected=0, measured=len(black))
    else:
        put("black_frames", PASS, "No black frames.", expected=0, measured=0)
    tick(0.75)

    # ------------------------------------------------------------------ structure
    def check_resolution() -> None:
        want, got = [data.width, data.height], [ov.width, ov.height]
        if want == got:
            note = " (scaled on request)" if data.resized else ", identical to the source"
            put("resolution", PASS, f"{got[0]}x{got[1]}{note}.", want, got)
        else:
            put("resolution", FAIL, f"The output is {got[0]}x{got[1]} but {want[0]}x{want[1]} was expected.", want, got)

    def check_orientation() -> None:
        if ov.rotation == sv.rotation:
            text = "No rotation metadata, like the source." if not sv.rotation else \
                f"Rotation metadata preserved ({sv.rotation} degrees; displayed {ov.display_width}x{ov.display_height})."
            put("orientation", PASS, text, sv.rotation, ov.rotation)
        else:
            put("orientation", FAIL, f"Rotation metadata is {ov.rotation} degrees in the output but "
                f"{sv.rotation} degrees in the source.", sv.rotation, ov.rotation)

    def decoded_format() -> Optional[PixFmt]:
        from ..processing.planes import working_format_for

        try:
            return working_format_for(ov.pix_fmt).fmt
        except VmtError:
            return None

    def check_pixel_format() -> None:
        want, got = data.out_fmt, decoded_format()
        if got is None:
            put("pixel_format", WARN, f"The output pixel format {ov.pix_fmt} is not recognised.", want.name, ov.pix_fmt)
            return
        same_shape = (got.family, got.chroma_w, got.chroma_h) == (want.family, want.chroma_w, want.chroma_h)
        alpha_ok = got.has_alpha == want.has_alpha
        vp9_alpha = want.has_alpha and not got.has_alpha and data.codec is VideoCodec.VP9
        if vp9_alpha:
            flag = str(ov.tags.get("alpha_mode", ov.tags.get("ALPHA_MODE", "")))
            alpha_ok = flag == "1"
        # ProRes decoders report 12-bit for 4444 material although 10 bits were encoded.
        depth_ok = got.bit_depth == want.bit_depth or (data.codec in (VideoCodec.PRORES_4444, VideoCodec.PRORES_422_HQ)
                                                       and got.bit_depth >= want.bit_depth)
        note = ""
        if not _same_layout(_color_only(want), _color_only(data.src_fmt)):
            note = f" (the source is {data.src_fmt.name}; this codec needs {want.name})"
        if same_shape and alpha_ok and depth_ok:
            extra = " The alpha channel is stored as a separate VP9 layer." if vp9_alpha else ""
            put("pixel_format", PASS, f"{ov.pix_fmt}, {ov.bit_depth}-bit{note}.{extra}", want.name, ov.pix_fmt)
        else:
            put("pixel_format", FAIL, f"The output is {ov.pix_fmt} but {want.name} was planned.", want.name, ov.pix_fmt)

    def check_color_tags() -> None:
        sc, oc = sv.color, ov.color
        assumed_full = data.resolved_color.full_range
        src_range, out_range = _effective_range(sc.range, assumed_full), _effective_range(oc.range, False)
        if data.out_fmt.family == "rgb":
            out_range = _effective_range(oc.range, True)
        problems: List[str] = []
        for name, a, b in (("matrix", sc.space, oc.space), ("primaries", sc.primaries, oc.primaries),
                           ("transfer", sc.transfer, oc.transfer)):
            if a != b and not (a in ("unknown", "unspecified") and b in ("unknown", "unspecified")):
                problems.append(f"{name}: source {a}, output {b}")
        measured = {"range": oc.range, "matrix": oc.space, "primaries": oc.primaries, "transfer": oc.transfer}
        expected = {"range": sc.range, "matrix": sc.space, "primaries": sc.primaries, "transfer": sc.transfer}
        if src_range != out_range:
            put("color_tags", FAIL, f"Colour range differs: the source is {'full' if src_range == 'pc' else 'limited'} "
                f"range but the output is flagged {'full' if out_range == 'pc' else 'limited'}: levels would be "
                "displayed wrongly.", expected, measured)
        elif problems:
            put("color_tags", WARN, "Colour tags differ from the source (" + "; ".join(problems) + ").", expected, measured)
        else:
            rng = "full" if out_range == "pc" else "limited"
            put("color_tags", PASS, f"{rng} range, matrix {oc.space}, primaries {oc.primaries}, transfer {oc.transfer} "
                "- as in the source.", expected, measured)

    out_times: List[Fraction] = []
    src_times: List[Fraction] = []

    def check_frame_count() -> None:
        nonlocal critical
        got = len(timeline)
        if got == expected_frames:
            put("frame_count", PASS, f"{got} frames, one per processed source frame.", expected_frames, got)
        else:
            put("frame_count", FAIL, f"The output has {got} frames but {expected_frames} were written.",
                expected_frames, got)
            if got < expected_frames:
                critical = True               # frames are missing: the file is truncated

    def check_timestamps() -> None:
        nonlocal out_times, src_times
        if len(timeline) != expected_frames or expected_frames == 0:
            put("timestamps", SKIPPED, "Not compared because the frame counts differ.")
            return
        out_tb = Fraction(ov.time_base)
        out_step = _time_step(om.container, out_tb)
        out_times = [Fraction(int(p)) * out_tb for p in timeline.pts]
        src_times = [Fraction(int(p)) * src_tb for p in src_pts]
        offset = out_times[0] - src_times[0]
        tolerance = max(out_step, src_tb) / 2
        worst, worst_index = Fraction(0), 0
        for index, (a, b) in enumerate(zip(out_times, src_times)):
            deviation = abs(a - b - offset)
            if deviation > worst:
                worst, worst_index = deviation, index
        measured = {"max_deviation_ms": _ms(worst), "frame": worst_index, "start_offset_ms": _ms(offset),
                    "tolerance_ms": _ms(tolerance)}
        shifted = ""
        if abs(offset - data.time_shift) > tolerance:
            shifted = (f" The whole timeline starts {_ms(offset - data.time_shift)} ms later than in the source "
                       "(this container cannot store the audio's negative priming timestamps; sync is unaffected).")
        elif data.time_shift != 0:
            shifted = f" A start offset of {_ms(-data.time_shift)} ms common to all source streams was removed."
        if worst <= tolerance:
            exact = "exactly" if worst == 0 else f"within {_ms(worst)} ms (the container counts in {_ms(out_step)} ms steps)"
            put("timestamps", PASS, f"All {expected_frames} frame times match the source {exact}.{shifted}",
                {"max_deviation_ms": 0.0}, measured)
        else:
            put("timestamps", FAIL, f"Frame times deviate from the source by up to {_ms(worst)} ms "
                f"(frame {worst_index}); allowed {_ms(tolerance)} ms.{shifted}", {"max_deviation_ms": 0.0}, measured)

    def check_frame_rate() -> None:
        if expected_frames < 2 or len(timeline) < 2:
            put("frame_rate", SKIPPED, "Not enough frames to measure a frame rate.")
            return
        reference = _timeline_from(src_pts, src_tb)
        want, got = reference.average_rate, timeline.average_rate
        if want is None or got is None:
            put("frame_rate", SKIPPED, "The frame rate could not be measured.")
            return
        span = max(Fraction(1, 1000), reference.duration)
        allowed = max(1e-6, float((_time_step(om.container, Fraction(ov.time_base)) + src_tb) / span))
        relative = abs(float(got) - float(want)) / float(want)
        mode_want, mode_got = reference.is_vfr, timeline.is_vfr
        measured = {"fps": round(float(got), 6), "variable": mode_got}
        expected = {"fps": round(float(want), 6), "variable": mode_want}
        if relative > allowed:
            put("frame_rate", FAIL, f"Average frame rate is {float(got):.4f} fps but the source has {float(want):.4f} fps.",
                expected, measured)
        elif mode_want != mode_got:
            put("frame_rate", FAIL, "The source has a " + ("variable" if mode_want else "constant") +
                " frame rate but the output a " + ("variable" if mode_got else "constant") + " one.", expected, measured)
        else:
            put("frame_rate", PASS, f"{float(got):.4f} fps, " + ("variable frame rate preserved." if mode_got
                else "constant frame rate."), expected, measured)

    def check_duration() -> None:
        if expected_frames == 0 or len(timeline) == 0:
            put("duration", SKIPPED, "No frames to measure.")
            return
        reference = _timeline_from(src_pts, src_tb)
        want, got = reference.duration, timeline.duration
        allowed = _duration_tolerance(reference, src_tb, _time_step(om.container, Fraction(ov.time_base)))
        measured, expected = round(float(got), 6), round(float(want), 6)
        if abs(got - want) <= allowed:
            put("duration", PASS, f"{float(got):.3f} s of video, as in the " +
                ("processed part of the source." if data.truncated else "source."), expected, measured)
        else:
            put("duration", FAIL, f"The video lasts {float(got):.3f} s but {float(want):.3f} s were expected.",
                expected, measured)

    # ------------------------------------------------------------------ audio
    hashes_equal: List[bool] = []

    def check_audio() -> None:
        want, got = data.expect_audio_streams, len(om.audio)
        if want == 0 and got == 0:
            text = "No audio, as requested." if source.audio else "The source has no audio; none was written."
            put("audio_streams", PASS, text, 0, 0)
            return
        if want != got:
            put("audio_streams", FAIL, f"The output has {got} audio stream(s) but {want} were expected.", want, got)
            return
        if data.audio_mode != "copy":
            pairs = ", ".join(f"{s.codec} -> {o.codec}" for s, o in zip(source.audio, om.audio))
            put("audio_streams", WARN, f"The audio was re-encoded ({pairs}) because the container cannot hold the "
                "original codec or a codec was forced.", want, got)
            return
        problems: List[str] = []
        notes: List[str] = []
        for index, (s, o) in enumerate(zip(source.audio, om.audio)):
            if (s.codec, s.sample_rate, s.channels) != (o.codec, o.sample_rate, o.channels):
                problems.append(f"stream {index + 1}: {s.codec} {s.sample_rate} Hz {s.channels} ch became "
                                f"{o.codec} {o.sample_rate} Hz {o.channels} ch")
                hashes_equal.append(False)
                continue
            if data.truncated:
                hashes_equal.append(False)
                continue
            a, b = _audio_packet_hash(tools, source.path, index), _audio_packet_hash(tools, str(path), index)
            if a is None or b is None:
                notes.append(f"stream {index + 1}: packets could not be hashed")
                hashes_equal.append(False)
            elif a == b:
                hashes_equal.append(True)
            else:
                hashes_equal.append(False)
                ca, cb = _audio_packet_count(tools, source.path, index), _audio_packet_count(tools, str(path), index)
                if ca is not None and cb is not None and abs(ca - cb) <= 3:
                    notes.append(f"stream {index + 1}: {abs(ca - cb) or 'some'} boundary packet(s) differ "
                                 f"({ca} packets in the source, {cb} in the output)")
                else:
                    problems.append(f"stream {index + 1}: the audio packets differ from the source ({ca} vs {cb} packets)")
        codecs = ", ".join(f"{o.codec} {o.sample_rate} Hz {o.channels} ch" for o in om.audio)
        if problems:
            put("audio_streams", FAIL, "Stream copy was expected but " + "; ".join(problems) + ".", want, got)
        elif notes:
            put("audio_streams", WARN, f"{got} stream(s) copied ({codecs}), but " + "; ".join(notes) + ".", want, got)
        elif data.truncated:
            put("audio_streams", PASS, f"{got} stream(s) stream-copied ({codecs}); cut to the processed part.", want, got)
        else:
            put("audio_streams", PASS, f"{got} stream(s) stream-copied bit-identically ({codecs}).", want, got)

    def check_sync() -> None:
        if not om.audio or not source.audio or len(om.audio) != len(source.audio):
            put("av_sync", SKIPPED, "No audio to compare.")
            return
        if not len(timeline) or expected_frames == 0:
            put("av_sync", SKIPPED, "No video frames to compare with.")
            return
        src_video0 = float(Fraction(int(src_pts[0])) * src_tb)
        out_video0 = float(Fraction(int(timeline.pts[0])) * Fraction(ov.time_base))
        worst, details, skipped = 0.0, [], 0
        limit = 2.0
        for index, (s, o) in enumerate(zip(source.audio, om.audio)):
            if data.audio_mode == "copy" and s.codec == o.codec:
                # The same packets exist in both files: compare where one of them sits relative to
                # the first video frame.  Unaffected by a start shift common to all streams.
                matched = _matched_audio_start(_audio_head(tools, source.path, index, 96),
                                               _audio_head(tools, str(path), index, 4))
                if matched is None:
                    skipped += 1
                    continue
                change = (matched[1] - out_video0) - (matched[0] - src_video0)
            else:
                # Re-encoded audio: packets do not correspond, compare the stream start times
                # (the priming of an encoder may be exposed, hence the wider limit).
                if s.start_time is None or o.start_time is None:
                    skipped += 1
                    continue
                limit = max(limit, 25.0)
                src_video_start = float(sv.start_time) if sv.start_time is not None else src_video0
                out_video_start = float(ov.start_time) if ov.start_time is not None else out_video0
                change = (float(o.start_time) - out_video_start) - (float(s.start_time) - src_video_start)
            worst = max(worst, abs(change) * 1000.0)
            details.append(round(change * 1000.0, 3))
        if not details:
            put("av_sync", SKIPPED, "Audio start times could not be read.")
            return
        duration_note = ""
        duration_problem = False
        if not data.truncated:
            for s, o in zip(source.audio, om.audio):
                if s.duration is not None and o.duration is not None and abs(float(s.duration) - float(o.duration)) > 0.05:
                    duration_problem = True
                    duration_note = (f" Audio duration changed from {float(s.duration):.3f} s to "
                                     f"{float(o.duration):.3f} s.")
        measured = {"offset_change_ms": details}
        if worst > limit:
            put("av_sync", FAIL, f"The audio moved by {worst:.1f} ms relative to the video.{duration_note}",
                {"max_change_ms": limit}, measured)
        elif duration_problem:
            put("av_sync", WARN, f"Audio / video offset unchanged (within {worst:.2f} ms).{duration_note}",
                {"max_change_ms": limit}, measured)
        else:
            put("av_sync", PASS, f"Audio / video offset unchanged (within {worst:.2f} ms).",
                {"max_change_ms": limit}, measured)

    # ------------------------------------------------------------------ pixels on the sampled frames
    def check_pixels() -> None:
        transparent = data.background is None
        samples = [s for s in data.samples if 0 <= s.index < len(timeline)]
        if data.resized:
            reason = "Not evaluated because the output was scaled on request."
            put("alpha_channel" if transparent else "background_color", SKIPPED, reason)
            put("foreground", SKIPPED, reason)
            return
        if not samples:
            reason = "No frames were sampled for this check."
            put("alpha_channel" if transparent else "background_color", SKIPPED, reason)
            put("foreground", SKIPPED, reason)
            return
        from ..compositing.colors import ColorConverter, convert_planes, planes_from_numpy, planes_to_numpy, to_integer_planes
        import torch

        reader = VideoReader(path, out)
        try:
            dec = reader.fmt
            colour_fmt = _color_only(dec)
            if dec.family != data.out_fmt.family:
                reason = f"Decoded format {dec.name} cannot be compared with the planned {data.out_fmt.name}."
                put("alpha_channel" if transparent else "background_color", SKIPPED, reason)
                put("foreground", SKIPPED, reason)
                return
            unit = float(1 << max(0, dec.bit_depth - 8))
            converter = ColorConverter(_color_only(data.src_fmt), data.resolved_color)
            expected = None if transparent else converter.color_code_values(tuple(data.background), colour_fmt)
            planes_n = colour_fmt.color_planes
            # accumulators: background
            bg_count = [0] * planes_n
            bg_sum = [0.0] * planes_n
            bg_max = [0.0] * planes_n
            bg_p999 = [0.0] * planes_n
            bg_lo = [math.inf] * planes_n
            bg_hi = [-math.inf] * planes_n
            # accumulators: foreground
            comparable = dec.family == data.src_fmt.family or (data.src_fmt.family == "gray" and dec.family == "yuv")
            fg_luma: List[float] = []
            fg_chroma: List[float] = []
            fg_exact = True
            fg_used = 0
            alpha_bad = 0.0
            alpha_used = 0
            peak = float(dec.max_value)
            for number, sample in enumerate(samples):
                tick(0.78 + 0.2 * number / max(1, len(samples)))
                frame = reader.read_frame(sample.index)
                planes = frame.planes
                # ---- background / alpha
                if not transparent:
                    for p in range(planes_n):
                        mask = _plane_mask(sample.background_mask, planes[p].shape)
                        values = planes[p][mask]
                        if values.size == 0:
                            continue
                        deviation = np.abs(values.astype(np.int64) - int(expected[p]))
                        bg_count[p] += int(values.size)
                        bg_sum[p] += float(values.astype(np.float64).sum())
                        bg_max[p] = max(bg_max[p], float(deviation.max()))
                        bg_p999[p] = max(bg_p999[p], float(np.percentile(deviation, 99.9)))
                        bg_lo[p] = min(bg_lo[p], float(values.min()))
                        bg_hi[p] = max(bg_hi[p], float(values.max()))
                elif dec.has_alpha and len(planes) > planes_n:
                    alpha = planes[planes_n].astype(np.float64) / peak
                    outside = alpha[_plane_mask(sample.background_mask, alpha.shape)]
                    inside = alpha[_plane_mask(sample.core_mask, alpha.shape)]
                    if outside.size:
                        alpha_bad = max(alpha_bad, float(outside.max()))
                        alpha_used += 1
                    if inside.size:
                        alpha_bad = max(alpha_bad, float(1.0 - inside.min()))
                        alpha_used += 1
                # ---- foreground
                if comparable and bool(sample.core_mask.any()):
                    src_planes = [np.asarray(pl) for pl in sample.source_planes[: data.src_fmt.color_planes]]
                    if not _same_layout(_color_only(data.src_fmt), colour_fmt):
                        tensors = planes_from_numpy([pl[None] for pl in src_planes], torch.device("cpu"))
                        full = converter.full_range_for(colour_fmt) if hasattr(converter, "full_range_for") \
                            else data.resolved_color.full_range
                        converted = to_integer_planes(convert_planes(tensors, _color_only(data.src_fmt), colour_fmt, full),
                                                      colour_fmt)
                        src_planes = [a[0] for a in planes_to_numpy(converted, colour_fmt)]
                    errors: List[float] = []
                    for p in range(planes_n):
                        mask = _plane_mask(sample.core_mask, planes[p].shape)
                        if not mask.any():
                            errors.append(float("nan"))
                            continue
                        diff = planes[p][mask].astype(np.float64) - src_planes[p][mask].astype(np.float64)
                        mse = float(np.mean(diff * diff))
                        errors.append(mse)
                        if mse != 0.0:
                            fg_exact = False
                    if errors and not math.isnan(errors[0]):
                        fg_used += 1
                        fg_luma.append(errors[0])
                        chroma = [e for e in errors[1:] if not math.isnan(e)]
                        if chroma:
                            fg_chroma.append(sum(chroma) / len(chroma))
        finally:
            reader.close()

        def psnr(mse: float) -> float:
            return float("inf") if mse <= 0.0 else 10.0 * math.log10(peak * peak / mse)

        # ---- verdict: background colour
        if transparent:
            if alpha_used == 0:
                reason = "The alpha channel cannot be read back with the standard decoder." if not dec.has_alpha \
                    else "No sampled frame had clear background or core regions."
                put("alpha_channel", SKIPPED, reason)
            elif alpha_bad <= (0.0 if data.lossless else 0.02):
                put("alpha_channel", PASS, "Alpha is fully transparent outside the subject and fully opaque inside "
                    f"(largest deviation {alpha_bad * 100:.2f} %).", 0.0, round(alpha_bad, 5))
            else:
                put("alpha_channel", FAIL, f"Alpha deviates by up to {alpha_bad * 100:.1f} % from fully transparent / "
                    "fully opaque in regions that must be exact.", 0.0, round(alpha_bad, 5))
        elif sum(bg_count) == 0:
            put("background_color", SKIPPED, "No sampled frame had a background region away from the subject.")
        else:
            means = [bg_sum[p] / bg_count[p] if bg_count[p] else float("nan") for p in range(planes_n)]
            mean_dev = max(abs(means[p] - expected[p]) for p in range(planes_n) if bg_count[p])
            max_dev = max(bg_max)
            p999 = max(bg_p999)
            spread = max((bg_hi[p] - bg_lo[p]) for p in range(planes_n) if bg_count[p])
            with torch.no_grad():
                full = converter.full_range_for(colour_fmt) if hasattr(converter, "full_range_for") \
                    else data.resolved_color.full_range
                back = ColorConverter(colour_fmt, ResolvedColor(data.resolved_color.matrix, data.resolved_color.kr,
                                                                data.resolved_color.kb, full))
                rgb = back.planes_to_rgb([torch.tensor([[[[means[p] if bg_count[p] else float(expected[p])]]]])
                                          for p in range(planes_n)])
                decoded_rgb = [int(round(float(v) * 255.0)) for v in rgb.flatten().tolist()][:3]
            wanted = "#%02X%02X%02X" % tuple(data.background)
            seen = "#%02X%02X%02X" % tuple(max(0, min(255, v)) for v in decoded_rgb)
            measured = {"decoded_rgb": seen, "max_deviation": max_dev, "mean_deviation": round(mean_dev, 4),
                        "uniformity_spread": spread, "code_values": [round(m, 3) for m in means]}
            target = {"rgb": wanted, "code_values": [int(v) for v in expected]}
            if data.lossless:
                if max_dev == 0:
                    put("background_color", PASS, f"Every background sample equals {wanted} exactly "
                        f"(code values {', '.join(str(int(v)) for v in expected)}); perfectly uniform.", target, measured)
                else:
                    put("background_color", FAIL, f"Background samples deviate by up to {max_dev:.0f} code values from "
                        f"{wanted} although the codec is lossless.", target, measured)
            elif mean_dev > 2.0 * unit or p999 > 6.0 * unit:
                put("background_color", FAIL, f"The background decodes to {seen} instead of {wanted} "
                    f"(mean deviation {mean_dev / unit:.2f}, 99.9 % of samples within {p999 / unit:.1f} 8-bit steps).",
                    target, measured)
            elif mean_dev > 1.0 * unit or p999 > 2.0 * unit:
                put("background_color", WARN, f"The background decodes to {seen}; it is not perfectly uniform after "
                    f"compression (99.9 % of samples within {p999 / unit:.1f} 8-bit steps of {wanted}). "
                    "Use a higher quality or a lossless codec for a cleaner key.", target, measured)
            else:
                put("background_color", PASS, f"The background decodes to {seen}: uniform {wanted} within "
                    f"{max(p999 / unit, 0.0):.1f} 8-bit steps (lossy codec).", target, measured)

        # ---- verdict: foreground
        if not comparable:
            put("foreground", SKIPPED, f"Source format {data.src_fmt.name} and output format {dec.name} are not "
                "directly comparable.")
        elif fg_used == 0:
            put("foreground", SKIPPED, "No sampled frame had a fully opaque subject region away from its edges.")
        else:
            worst_luma = min(psnr(m) for m in fg_luma)
            worst_chroma = min((psnr(m) for m in fg_chroma), default=float("inf"))
            measured = {"luma_psnr_db": None if math.isinf(worst_luma) else round(worst_luma, 2),
                        "chroma_psnr_db": None if math.isinf(worst_chroma) else round(worst_chroma, 2),
                        "frames": fg_used, "bit_exact": fg_exact}
            if fg_exact:
                put("foreground", PASS, f"The subject's pixels are bit-identical to the source on all {fg_used} "
                    "sampled frames.", {"bit_exact": True} if data.lossless else {"luma_psnr_db": ">= 35"}, measured)
            elif data.lossless:
                put("foreground", FAIL, "The subject's pixels differ from the source although the codec is lossless "
                    f"(luma PSNR {worst_luma:.1f} dB).", {"bit_exact": True}, measured)
            elif worst_luma >= 35.0:
                put("foreground", PASS, f"The subject matches the source: luma PSNR {worst_luma:.1f} dB, chroma "
                    f"{worst_chroma:.1f} dB (differences are compression only).", {"luma_psnr_db": ">= 35"}, measured)
            elif worst_luma >= 28.0:
                put("foreground", WARN, f"The subject differs visibly from the source (luma PSNR {worst_luma:.1f} dB): "
                    "the encoder quality is low for this footage.", {"luma_psnr_db": ">= 35"}, measured)
            else:
                put("foreground", FAIL, f"The subject does not match the source (luma PSNR {worst_luma:.1f} dB): "
                    "levels or colours were changed.", {"luma_psnr_db": ">= 35"}, measured)

    # ------------------------------------------------------------------ matte statistics
    def check_flicker() -> None:
        stats = data.refine_stats or {}
        transitions = int(stats.get("transitions", 0) or 0)
        if not stats or transitions == 0:
            put("matte_flicker", SKIPPED, "No matte statistics are available (single frame or refinement inactive).")
            return
        p95, raw_p95 = float(stats.get("flicker_p95", 0.0)), float(stats.get("raw_flicker_p95", 0.0))
        jumps = int(stats.get("area_jumps", 0) or 0)
        measured = {"flicker_p95": round(p95, 5), "raw_flicker_p95": round(raw_p95, 5), "area_jumps": jumps,
                    "flicker_mean": round(float(stats.get("flicker_mean", 0.0)), 5)}
        improvement = f" (the model's raw matte: {raw_p95 * 100:.2f} %)" if raw_p95 > 0 else ""
        jump_limit = max(1, int(0.005 * transitions))
        if p95 > 0.008 or jumps > jump_limit:
            put("matte_flicker", WARN, f"The matte is unstable in static areas: 95 % of frame-to-frame changes are below "
                f"{p95 * 100:.2f} % of the picture{improvement}; {jumps} sudden area jump(s). Raise temporal smoothing "
                "or try another model.", {"flicker_p95": "<= 0.008"}, measured)
        else:
            put("matte_flicker", PASS, f"Stable matte: frame-to-frame change in static areas stays below "
                f"{p95 * 100:.2f} %{improvement}; {jumps} area jump(s).", {"flicker_p95": "<= 0.008"}, measured)

    def check_tracking() -> None:
        report = data.tracking_report or {}
        frames = int(report.get("frames", 0) or 0)
        if not report or frames == 0:
            put("subject_tracking", SKIPPED, "No tracking statistics are available.")
            return
        empty = int(report.get("empty_frames", 0) or 0)
        absent = report.get("absent_segments") or []
        switches = report.get("suspected_switches") or []
        measured = {"frames": frames, "empty_frames": empty, "absent_segments": absent[:20],
                    "suspected_switches": switches[:20], "coverage_mean": round(float(report.get("coverage_mean", 0.0)), 4)}
        if empty >= frames:
            put("subject_tracking", FAIL, "The matte is empty in every frame: no subject was found. Select the subject "
                "manually or use a model that fits the footage.", None, measured)
        elif switches:
            put("subject_tracking", WARN, f"The matte jumps to a different place at frame(s) "
                f"{', '.join(str(i) for i in switches[:8])}: the subject may have been confused with another one.",
                None, measured)
        elif absent:
            spans = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in absent[:6])
            put("subject_tracking", WARN, f"No subject is visible in {empty} of {frames} frames (frames {spans}).",
                None, measured)
        else:
            put("subject_tracking", PASS, f"The subject is present in all {frames} frames "
                f"(average coverage {float(report.get('coverage_mean', 0.0)) * 100:.1f} % of the picture).", None, measured)

    guarded("resolution", check_resolution)
    guarded("orientation", check_orientation)
    guarded("pixel_format", check_pixel_format)
    guarded("color_tags", check_color_tags)
    guarded("frame_count", check_frame_count)
    guarded("timestamps", check_timestamps)
    guarded("frame_rate", check_frame_rate)
    guarded("duration", check_duration)
    tick(0.76)
    guarded("audio_streams", check_audio)
    guarded("av_sync", check_sync)
    tick(0.78)
    if critical:
        for check_id in ("background_color", "foreground"):
            put(check_id, SKIPPED, "Not evaluated because the file is unusable.")
    else:
        try:
            check_pixels()
        except JobCancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("Pixel verification could not be evaluated")
            for check_id in ("alpha_channel" if data.background is None else "background_color", "foreground"):
                if check_id not in results:
                    put(check_id, SKIPPED, f"Could not be evaluated ({type(exc).__name__}: {exc}).")
    guarded("matte_flicker", check_flicker)
    guarded("subject_tracking", check_tracking)
    comparison = _comparison(data, om, timeline, src_pts, src_tb)
    tick(1.0)
    return finish(om, comparison, critical)


# =========================================================================== comparison table


def _comparison(
    data: VerifyInput, om: VideoMetadata, timeline: FrameTimeline, src_pts: np.ndarray, src_tb: Fraction
) -> List[ComparisonRow]:
    """The input-versus-output property table of the report."""

    source, sv, ov = data.source, data.source.video, om.video
    rows: List[ComparisonRow] = []

    def row(name: str, a: Any, b: Any, match: Optional[bool] = None, note: str = "") -> None:
        rows.append(ComparisonRow(property=name, input=a, output=b, match=match, note=note))

    partial = "partial render" if data.truncated else ""
    row("Container", source.container, om.container, None)
    row("Video codec", sv.codec, ov.codec, None)
    row("Resolution", f"{sv.width}x{sv.height}", f"{ov.width}x{ov.height}",
        (sv.width, sv.height) == (ov.width, ov.height), "scaled on request" if data.resized else "")
    row("Display orientation", f"{sv.rotation} deg ({sv.display_width}x{sv.display_height})",
        f"{ov.rotation} deg ({ov.display_width}x{ov.display_height})", sv.rotation == ov.rotation)
    same_pix = sv.pix_fmt.replace("yuvj", "yuv") == ov.pix_fmt.replace("yuvj", "yuv")
    row("Pixel format", sv.pix_fmt, ov.pix_fmt, True if same_pix else None, "" if same_pix else "required by the output codec")
    row("Bit depth", sv.bit_depth, ov.bit_depth, True if sv.bit_depth == ov.bit_depth else None)
    reference = _timeline_from(src_pts, src_tb) if len(src_pts) else None
    want_rate = reference.average_rate if reference is not None else None
    got_rate = timeline.average_rate if len(timeline) else None
    rate_match = None
    if want_rate and got_rate:
        rate_match = abs(float(want_rate) - float(got_rate)) / float(want_rate) < 1e-3
    row("Frame rate", f"{float(want_rate):.4f} fps" if want_rate else "-", f"{float(got_rate):.4f} fps" if got_rate else "-",
        rate_match)
    if reference is not None and len(timeline):
        row("Frame rate mode", "variable" if reference.is_vfr else "constant",
            "variable" if timeline.is_vfr else "constant", reference.is_vfr == timeline.is_vfr)
    row("Frame count", source.frame_count, len(timeline), None if data.truncated else source.frame_count == len(timeline),
        partial)
    src_duration = float(source.video_duration or 0)
    out_duration = float(om.video_duration or 0)
    slack = (float(_duration_tolerance(reference, src_tb, _time_step(om.container, Fraction(ov.time_base))))
             if reference is not None else 0.05)
    row("Duration", f"{src_duration:.3f} s", f"{out_duration:.3f} s",
        None if data.truncated else abs(src_duration - out_duration) <= slack + 0.002, partial)
    row("Start time", f"{float(sv.start_time or 0):.4f} s", f"{float(ov.start_time or 0):.4f} s", None)
    row("Time base", f"{sv.time_base.numerator}/{sv.time_base.denominator}",
        f"{ov.time_base.numerator}/{ov.time_base.denominator}", None)
    assumed = data.resolved_color.full_range
    row("Colour range", sv.color.range, ov.color.range,
        _effective_range(sv.color.range, assumed) == _effective_range(ov.color.range, data.out_fmt.family == "rgb"))
    for label, a, b in (("Colour matrix", sv.color.space, ov.color.space), ("Colour primaries", sv.color.primaries, ov.color.primaries),
                        ("Transfer function", sv.color.transfer, ov.color.transfer)):
        row(label, a, b, a == b)
    sa = source.audio[0] if source.audio else None
    oa = om.audio[0] if om.audio else None
    expect_audio = data.expect_audio_streams > 0
    row("Audio codec", sa.codec if sa else "-", oa.codec if oa else "-",
        (sa.codec == oa.codec) if (sa and oa) else (not expect_audio and oa is None),
        "" if data.audio_mode != "transcode" else "re-encoded")
    row("Audio channels", sa.channels if sa else "-", oa.channels if oa else "-",
        (sa.channels == oa.channels) if (sa and oa) else (not expect_audio and oa is None))
    row("Audio sample rate", f"{sa.sample_rate} Hz" if sa else "-", f"{oa.sample_rate} Hz" if oa else "-",
        (sa.sample_rate == oa.sample_rate) if (sa and oa) else (not expect_audio and oa is None))
    if sa or oa:
        a_dur = float(sa.duration) if sa and sa.duration is not None else None
        b_dur = float(oa.duration) if oa and oa.duration is not None else None
        row("Audio duration", f"{a_dur:.3f} s" if a_dur is not None else "-", f"{b_dur:.3f} s" if b_dur is not None else "-",
            None if (data.truncated or a_dur is None or b_dur is None) else abs(a_dur - b_dur) <= 0.05, partial)

        def offset(meta: VideoMetadata) -> Optional[float]:
            stream = meta.audio[0] if meta.audio else None
            if stream is None or stream.start_time is None or meta.video.start_time is None:
                return None
            return float(stream.start_time - meta.video.start_time)

        a_off, b_off = offset(source), offset(om)
        row("Audio start relative to video", f"{a_off * 1000:.1f} ms" if a_off is not None else "-",
            f"{b_off * 1000:.1f} ms" if b_off is not None else "-", None,
            "see the audio / video sync check for the packet-level comparison")
    row("Bit rate", f"{(source.bit_rate or 0) / 1e6:.2f} Mb/s" if source.bit_rate else "-",
        f"{(om.bit_rate or 0) / 1e6:.2f} Mb/s" if om.bit_rate else "-", None)
    row("File size", f"{source.size_bytes / 1e6:.2f} MB", f"{om.size_bytes / 1e6:.2f} MB", None)
    return rows
