"""Timestamp-exact video encoder: raw frames with their exact PTS piped into the FFmpeg CLI.

Why this design: the output must carry *exactly* the source's presentation timestamps (also for
variable frame rate footage) and *exactly* the pixels the compositor produced.  A plain raw-video
pipe cannot do that (it has no timestamps, so FFmpeg would re-time the frames to a constant rate).
Instead PyAV muxes the frames as ``rawvideo`` into a NUT stream - which stores a time base and a PTS
per frame - and that stream is written to the standard input of an ``ffmpeg`` process which encodes
it with timestamps passed through (see :meth:`app.encoding.profiles.EncodePlan.command`).

Behaviour that matters to callers:

* :meth:`NutPipeEncoder.write` returns quickly: frames go through a bounded queue to a writer thread,
  so the render loop only blocks when the encoder is the bottleneck (back-pressure) and several
  encoders can be fed from one loop.
* Any failure - a rejected option, FFmpeg dying mid-stream, a full disk - surfaces from the next
  :meth:`write` or from :meth:`finish` as an error that carries FFmpeg's own messages.
* :meth:`abort` kills FFmpeg and deletes the partial file; an encoder never leaves a process behind.

Pitfalls this module handles (each was reproduced while validating the design):

* FFmpeg's stderr must be drained concurrently or the whole pipeline deadlocks after a few frames.
* ``container.close()`` does not close the pipe; FFmpeg waits for end-of-file until stdin is closed.
* On a broken pipe PyAV raises ``BrokenPipeError``, ``OSError(22)`` or an ``av.error`` exception
  depending on timing; a container that is not explicitly closed afterwards can crash the
  interpreter when it is garbage-collected.
* NUT cannot store negative timestamps (the muxer would silently shift *all* of them): a negative
  first PTS is shifted to zero here and the shift is reported in :attr:`pts_shift` so the final mux
  can undo it.
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
from fractions import Fraction
from pathlib import Path
from typing import Any, List, Optional, Sequence

import av
import numpy as np

from ..core.errors import EncodingError
from ..processing.planes import planes_to_frame, semiplanar_to_frame, to_semiplanar
from ..utils.fs import safe_unlink
from .ffmpeg import CREATE_NO_WINDOW, FfmpegTools, _kill, _StderrCollector, process_failure
from .profiles import PACKED_RGB8, EncodePlan, pipe_layout

from ..utils.avlog import silence_pyav_logging

log = logging.getLogger(__name__)

__all__ = ["NutPipeEncoder"]

_END = object()

#: Upper bound for raw frames waiting in the queue of one encoder (bytes).  Keeps memory bounded
#: for large frames: a 4K 10-bit frame is about 25 MB.
_MAX_QUEUE_BYTES = 192 * 1024 * 1024


class NutPipeEncoder:
    """Encodes one video-only file from numpy planes and exact integer timestamps.

    Usage::

        encoder = NutPipeEncoder(tools, path, plan, source_time_base)
        encoder.start()
        for planes, pts in frames:
            encoder.write(planes, pts)        # planes in ``plan.fmt``; pts in source time-base ticks
        encoder.finish()                      # or encoder.abort() on failure

    ``time_base`` MUST be the source stream's time base and ``pts`` the source PTS: the NUT muxer only
    refines a time base by an integer factor, so every timestamp stays exact.
    """

    def __init__(
        self,
        tools: FfmpegTools,
        output: Path,
        plan: EncodePlan,
        time_base: Fraction,
        *,
        queue_frames: int = 24,
        name: str = "main",
    ) -> None:
        self.tools = tools
        self.output = Path(output)
        self.plan = plan
        self.time_base = Fraction(time_base)
        if self.time_base <= 0:
            raise ValueError(f"invalid time base {time_base!r}")
        self.name = name
        self.fmt = plan.fmt
        self.width = int(plan.width)
        self.height = int(plan.height)
        layout = pipe_layout(plan.pipe_pix_fmt)
        if layout is None or layout[0].name != plan.fmt.name:
            raise EncodingError(
                f"Internal error: pipe format {plan.pipe_pix_fmt} cannot be produced from {plan.fmt.name} planes."
            )
        self._layout = layout[1]                       # planar | semiplanar | packed
        frame_bytes = max(1, self.fmt.frame_bytes(self.width, self.height))
        depth = max(2, min(int(queue_frames), _MAX_QUEUE_BYTES // frame_bytes))
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=depth)
        self.command: List[str] = plan.command(tools, self.output)
        self.frames_written = 0
        #: Ticks added to every PTS because the first one was negative (0 for normal files).
        self.pts_shift = 0
        self._last_pts: Optional[int] = None
        self._proc: Optional["subprocess.Popen[bytes]"] = None
        self._stderr: Optional[_StderrCollector] = None
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self._aborted = threading.Event()
        self._finished = False
        self._started = False
        self._what = f"Encoding the {name} video" if name != "main" else "Encoding the video"

    # ------------------------------------------------------------------------------ life cycle
    def start(self) -> None:
        """Launch FFmpeg and the writer thread."""

        if self._started:
            return
        self._started = True
        self.output.parent.mkdir(parents=True, exist_ok=True)
        log.debug("Encoder %s: %s", self.name, subprocess.list2cmdline(self.command))
        try:
            # bufsize=-1: stdin becomes a BufferedWriter whose write() always consumes everything
            # (libavformat's I/O layer does not retry short writes of a raw pipe).
            self._proc = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                bufsize=-1, creationflags=CREATE_NO_WINDOW,
            )
        except OSError as exc:
            raise EncodingError(
                f"FFmpeg could not be started for the {self.name} output: {exc}",
                details={"command": self.command},
            ) from exc
        self._stderr = _StderrCollector(self._proc.stderr, name=f"vmt-ffmpeg-stderr-{self.name}")
        self._stderr.start()
        self._thread = threading.Thread(target=self._run, name=f"vmt-encoder-{self.name}", daemon=True)
        self._thread.start()

    def _failure(self) -> BaseException:
        """The error that describes why FFmpeg stopped (waits briefly for its exit code and messages)."""

        proc = self._proc
        returncode: Optional[int] = None
        if proc is not None:
            try:
                returncode = proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                _kill(proc)
                returncode = proc.returncode
        if self._stderr is not None:
            self._stderr.join(timeout=10)
        text = self._stderr.text() if self._stderr is not None else ""
        return process_failure(
            self._what, self.command, returncode, text,
            hint="See the FFmpeg messages in the error details. An unsupported encoder option, a pixel format "
            "the encoder does not accept or a full disk are the usual causes.",
            extra={"frames_written": self.frames_written, "output": str(self.output)},
        )

    def _classify(self, exc: BaseException) -> BaseException:
        """Turn an exception of the writer thread into the error reported to the caller.

        Writing to the pipe fails when FFmpeg has exited (a rejected option, a full disk ...): then
        FFmpeg's own message is the useful one.  If FFmpeg is still running the problem is on our
        side; it is stopped and the original exception is reported.
        """

        proc = self._proc
        assert proc is not None
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        if proc.poll() is not None:
            return self._failure()
        _kill(proc)
        error = EncodingError(
            f"Writing frame {self.frames_written} of the {self.name} output to the encoder failed: "
            f"{type(exc).__name__}: {exc}",
            details={"command": self.command, "frames_written": self.frames_written},
        )
        error.__cause__ = exc
        return error

    def _run(self) -> None:
        """Writer thread: owns the PyAV container from creation to close."""

        proc = self._proc
        assert proc is not None and proc.stdin is not None
        container = None
        stream = None
        try:
            # write_index=0: the pipe cannot be rewound and the index would grow in memory with
            # every frame.  avoid_negative_ts=disabled: never let the muxer shift timestamps on its own.
            silence_pyav_logging()
            container = av.open(
                proc.stdin, mode="w", format="nut", buffer_size=1 << 20,
                options={"avoid_negative_ts": "disabled", "write_index": "0"},
            )
            rate = self.plan.frame_rate if self.plan.frame_rate and 0 < self.plan.frame_rate < 1000 else Fraction(30)
            stream = container.add_stream("rawvideo", rate=rate)
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = self.plan.pipe_pix_fmt
            stream.codec_context.time_base = self.time_base
            stream.time_base = self.time_base
            while True:
                item = self._queue.get()
                if item is _END:
                    break
                if self._aborted.is_set():
                    continue
                for packet in stream.encode(item):
                    container.mux(packet)
                self.frames_written += 1
            if not self._aborted.is_set():
                for packet in stream.encode(None):
                    container.mux(packet)
        except BaseException as exc:  # noqa: BLE001 - classified, then re-raised to the producer
            if not self._aborted.is_set():
                self._error = self._classify(exc)
                log.debug("Encoder %s stopped: %r", self.name, exc)
            self._drain()
        finally:
            # ALWAYS close the container explicitly (see the module docstring), then signal
            # end-of-file to FFmpeg by closing the pipe.
            if container is not None:
                try:
                    container.close()
                except BaseException as exc:  # noqa: BLE001
                    if self._error is None and not self._aborted.is_set():
                        self._error = self._failure()
                    log.debug("Encoder %s: closing the NUT stream failed: %r", self.name, exc)
            try:
                proc.stdin.close()
            except BaseException as exc:  # noqa: BLE001 - OSError(22) on a dead pipe
                if self._error is None and not self._aborted.is_set():
                    self._error = self._failure()
                log.debug("Encoder %s: closing the pipe failed: %r", self.name, exc)

    def _drain(self) -> None:
        """After a failure: keep consuming so that a producer blocked in ``write`` is released."""

        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._aborted.is_set() or self._finished:
                    return
                continue
            if item is _END:
                return

    # ------------------------------------------------------------------------------ frames
    def _packed_frame(self, planes: Sequence[np.ndarray], pts: int) -> "av.VideoFrame":
        """Interleave planar 8-bit RGB (G, B, R[, A]) into a packed layout such as ``bgr0``."""

        pixel_bytes, ir, ig, ib, ia = PACKED_RGB8[self.plan.pipe_pix_fmt]
        frame = av.VideoFrame(self.width, self.height, self.plan.pipe_pix_fmt)
        plane = frame.planes[0]
        view = np.frombuffer(plane, dtype=np.uint8).reshape(plane.height, plane.line_size)
        packed = view[:, : self.width * pixel_bytes].reshape(self.height, self.width, pixel_bytes)
        packed[..., ig] = planes[0]
        packed[..., ib] = planes[1]
        packed[..., ir] = planes[2]
        if ia is not None:
            packed[..., ia] = planes[3] if len(planes) > 3 else 255
        elif pixel_bytes == 4:
            packed[..., 3] = 255                      # padding byte of bgr0 / rgb0
        frame.pts = pts
        frame.time_base = self.time_base
        return frame

    def _make_frame(self, planes: Sequence[np.ndarray], pts: int) -> "av.VideoFrame":
        fmt = self.fmt
        if len(planes) != fmt.num_planes:
            raise ValueError(f"{self.name}: {fmt.name} needs {fmt.num_planes} planes, got {len(planes)}")
        for index, plane in enumerate(planes):
            expected = fmt.plane_size(index, self.width, self.height)
            if plane.shape != expected or plane.dtype != fmt.dtype:
                raise ValueError(
                    f"{self.name}: plane {index} of {fmt.name} must be {fmt.dtype} {expected}, "
                    f"got {plane.dtype} {plane.shape}"
                )
        if self._layout == "planar":
            return planes_to_frame(planes, fmt, pts, self.time_base)
        if self._layout == "semiplanar":
            return semiplanar_to_frame(to_semiplanar(planes, fmt, self.plan.pipe_pix_fmt), self.plan.pipe_pix_fmt,
                                       pts, self.time_base)
        return self._packed_frame(planes, pts)

    def write(self, planes: Sequence[np.ndarray], pts: int, duration: int = 0) -> None:
        """Queue one frame. *planes* are in ``plan.fmt`` (strictly checked); *pts* in time-base ticks.

        The samples are copied before this method returns, so the caller may reuse its arrays.
        *duration* is accepted for symmetry with the decoder but NUT does not carry frame durations.
        """

        if not self._started:
            raise RuntimeError("NutPipeEncoder.write() called before start()")
        if self._finished or self._aborted.is_set():
            raise RuntimeError("NutPipeEncoder.write() called after finish() / abort()")
        if self._error is not None:
            raise self._error
        pts = int(pts)
        if self._last_pts is None and pts < 0:
            self.pts_shift = -pts
            log.info("Encoder %s: first timestamp is negative (%d); the timeline is shifted by %d ticks "
                     "for encoding and shifted back when the audio is added.", self.name, pts, self.pts_shift)
        pts += self.pts_shift
        if self._last_pts is not None and pts <= self._last_pts:
            raise ValueError(
                f"{self.name}: timestamps must be strictly increasing ({pts - self.pts_shift} after "
                f"{self._last_pts - self.pts_shift})"
            )
        frame = self._make_frame(planes, pts)          # copies the samples into FFmpeg-owned memory
        self._last_pts = pts
        while True:
            if self._error is not None:
                raise self._error
            if self._aborted.is_set():
                raise RuntimeError("the encoder was aborted")
            try:
                self._queue.put(frame, timeout=0.2)
                return
            except queue.Full:
                if self._thread is not None and not self._thread.is_alive() and self._error is None:
                    raise EncodingError(f"The encoder thread of the {self.name} output stopped unexpectedly.")

    def finish(self) -> None:
        """Flush everything, wait for FFmpeg and verify that a non-empty file was written."""

        if self._finished:
            if self._error is not None:
                raise self._error
            return
        if not self._started:
            raise RuntimeError("NutPipeEncoder.finish() called before start()")
        self._finished = True
        proc, thread = self._proc, self._thread
        assert proc is not None and thread is not None
        while thread.is_alive():
            try:
                self._queue.put(_END, timeout=0.2)
                break
            except queue.Full:
                continue
        thread.join()
        returncode = proc.wait()
        if self._stderr is not None:
            self._stderr.join(timeout=10)
        if self._error is None and returncode != 0:
            self._error = self._failure()
        if self._error is None:
            try:
                size = self.output.stat().st_size
            except OSError:
                size = 0
            if size <= 0:
                self._error = EncodingError(
                    f"FFmpeg finished but wrote no data for the {self.name} output.",
                    details={"command": self.command, "stderr_tail": self._stderr.text()[-4000:] if self._stderr else ""},
                )
        self._close_stderr()
        if self._error is not None:
            safe_unlink(self.output)
            raise self._error
        text = self._stderr.text().strip() if self._stderr is not None else ""
        if text:
            log.debug("Encoder %s messages:\n%s", self.name, text[-2000:])

    def abort(self) -> None:
        """Stop immediately: kill FFmpeg, release the writer thread, delete the partial file."""

        if not self._started:
            return
        already_done = self._finished and (self._proc is None or self._proc.poll() is not None)
        self._aborted.set()
        self._finished = True
        proc, thread = self._proc, self._thread
        if proc is not None and proc.poll() is None:
            _kill(proc)
        if thread is not None and thread.is_alive():
            try:
                self._queue.put_nowait(_END)
            except queue.Full:
                pass                                   # the thread is draining and will see the flag
            thread.join(timeout=30)
        if self._stderr is not None:
            self._stderr.join(timeout=10)
        self._close_stderr()
        if not already_done or self._error is not None:
            safe_unlink(self.output)

    def _close_stderr(self) -> None:
        proc = self._proc
        if proc is not None and proc.stderr is not None:
            try:
                proc.stderr.close()
            except OSError:
                pass

    def __enter__(self) -> "NutPipeEncoder":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.finish()
        else:
            self.abort()
