"""Progress reporting and cancellation primitives shared by the pipeline, the CLI and the API.

All numbers reported here are *measured*: ``frame`` is the count of frames that have really been
processed, ``fps`` is derived from wall-clock time over a sliding window, and ``eta_seconds`` is the
remaining frame count divided by that measured rate.  Nothing in this module may fabricate progress.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, Optional, Tuple

from .errors import JobCancelledError


class Stage(str, Enum):
    """Pipeline stages, in execution order."""

    QUEUED = "queued"
    PROBE = "probe"                # metadata analysis + timeline scan
    SUBJECT = "subject"            # subject identification / prompt resolution
    MODEL_LOAD = "model_load"      # weights -> device
    BACKTRACK = "backtrack"        # optional reverse tracking of frames before the reference frame
    RENDER = "render"              # decode -> matte -> refine -> composite -> encode (streaming)
    MUX = "mux"                    # audio / metadata mux
    VERIFY = "verify"              # quality verification
    EXPORT = "export"              # deliverables moved into place, audio export
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressEvent:
    """A snapshot of processing progress. Serialisable with :meth:`to_dict`."""

    stage: Stage = Stage.QUEUED
    message: str = ""
    frame: int = 0                           # frames fully processed in the current stage
    total_frames: Optional[int] = None       # exact frame count of the stage, if known
    percent: Optional[float] = None          # 0..100 for the current stage
    overall_percent: Optional[float] = None  # 0..100 across all stages (weighted)
    fps: Optional[float] = None              # measured end-to-end processing rate (frames / second)
    inference_fps: Optional[float] = None    # measured model-only rate (frames / second of model time)
    eta_seconds: Optional[float] = None
    elapsed_seconds: float = 0.0
    batch_size: Optional[int] = None
    device: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_utilization: Optional[float] = None  # percent, when the driver exposes it
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    ram_used_mb: Optional[float] = None      # resident memory of this process
    ram_system_percent: Optional[float] = None
    warnings: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["stage"] = self.stage.value
        return data


ProgressCallback = Callable[[ProgressEvent], None]


class CancelToken:
    """Thread-safe cooperative cancellation flag.

    Long running loops call :meth:`raise_if_cancelled` between units of work (one batch, one
    download chunk, ...).  The pipeline guarantees that cancellation removes partial outputs.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = "cancelled by user"

    def cancel(self, reason: str = "cancelled by user") -> None:
        self._reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise JobCancelledError(f"Processing {self._reason}.")

    def wait(self, timeout: float) -> bool:
        """Sleep up to *timeout* seconds; return True if cancelled meanwhile."""

        return self._event.wait(timeout)


class RateMeter:
    """Sliding-window rate estimator (units per second) based on wall-clock samples."""

    def __init__(self, window_seconds: float = 5.0) -> None:
        self.window_seconds = float(window_seconds)
        self._samples: Deque[Tuple[float, float]] = deque()
        self._start: Optional[float] = None
        self._first_count: float = 0.0

    def reset(self) -> None:
        self._samples.clear()
        self._start = None
        self._first_count = 0.0

    def update(self, count: float, now: Optional[float] = None) -> None:
        """Record that *count* units (cumulative) were complete at time *now*."""

        t = time.perf_counter() if now is None else now
        if self._start is None:
            self._start = t
            self._first_count = count
        self._samples.append((t, count))
        cutoff = t - self.window_seconds
        while len(self._samples) > 2 and self._samples[1][0] <= cutoff:
            self._samples.popleft()

    @property
    def rate(self) -> Optional[float]:
        """Windowed rate, or None until two samples at least 50 ms apart exist."""

        if len(self._samples) < 2:
            return None
        (t0, c0), (t1, c1) = self._samples[0], self._samples[-1]
        dt = t1 - t0
        if dt < 0.05 or c1 <= c0:
            return None
        return (c1 - c0) / dt

    @property
    def average_rate(self) -> Optional[float]:
        """Rate since the first sample."""

        if self._start is None or not self._samples:
            return None
        t1, c1 = self._samples[-1]
        dt = t1 - self._start
        if dt <= 0 or c1 <= self._first_count:
            return None
        return (c1 - self._first_count) / dt


class ProgressReporter:
    """Helper that owns a callback, throttles emission and fills in measured rates.

    The pipeline creates one reporter per run and calls :meth:`stage` when entering a stage and
    :meth:`advance` as work completes.  ``min_interval`` throttles callback invocations; stage
    changes and the final event of a stage are always delivered.
    """

    #: Relative weight of each stage for ``overall_percent``.
    STAGE_WEIGHTS: Dict[Stage, float] = {
        Stage.PROBE: 1.0,
        Stage.SUBJECT: 2.0,
        Stage.MODEL_LOAD: 2.0,
        Stage.BACKTRACK: 0.0,   # set dynamically when backtracking is needed
        Stage.RENDER: 88.0,
        Stage.MUX: 2.0,
        Stage.EXPORT: 1.0,
        Stage.VERIFY: 4.0,
    }
    #: Execution order.  Verification comes before the deliverables are moved into place: a file
    #: that fails its integrity checks is never delivered.
    STAGE_ORDER = (
        Stage.PROBE,
        Stage.SUBJECT,
        Stage.MODEL_LOAD,
        Stage.BACKTRACK,
        Stage.RENDER,
        Stage.MUX,
        Stage.VERIFY,
        Stage.EXPORT,
    )

    def __init__(
        self,
        callback: Optional[ProgressCallback] = None,
        *,
        min_interval: float = 0.2,
        stats_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        self._callback = callback
        self._min_interval = float(min_interval)
        self._stats_provider = stats_provider
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self._last_emit = 0.0
        self._meter = RateMeter()
        self._weights = dict(self.STAGE_WEIGHTS)
        self.event = ProgressEvent()
        self.model_seconds = 0.0     # accumulated wall time spent inside model.process()
        self.model_frames = 0

    # -- configuration -----------------------------------------------------------------
    def set_stage_weight(self, stage: Stage, weight: float) -> None:
        self._weights[stage] = float(weight)

    def set_static(self, **fields: Any) -> None:
        """Set fields that persist across events (device, gpu_name, batch_size, ...)."""

        with self._lock:
            for key, value in fields.items():
                setattr(self.event, key, value)

    # -- stage / progress --------------------------------------------------------------
    def stage(self, stage: Stage, message: str = "", total_frames: Optional[int] = None) -> None:
        with self._lock:
            self._meter.reset()
            self.event.stage = stage
            self.event.message = message
            self.event.frame = 0
            self.event.total_frames = total_frames
            self.event.percent = 0.0 if stage not in (Stage.DONE,) else 100.0
            self.event.fps = None
            self.event.eta_seconds = None
        self._emit(force=True)

    def advance(self, frame: int, message: Optional[str] = None, force: bool = False, **extras: Any) -> None:
        """Report that *frame* frames (cumulative within the stage) are complete."""

        with self._lock:
            self._meter.update(frame)
            ev = self.event
            ev.frame = int(frame)
            if message is not None:
                ev.message = message
            if ev.total_frames:
                ev.percent = max(0.0, min(100.0, 100.0 * frame / ev.total_frames))
            rate = self._meter.rate or self._meter.average_rate
            ev.fps = rate
            if rate and ev.total_frames:
                ev.eta_seconds = max(0.0, (ev.total_frames - frame) / rate)
            if self.model_seconds > 0 and self.model_frames > 0:
                ev.inference_fps = self.model_frames / self.model_seconds
            if extras:
                ev.extras.update(extras)
        self._emit(force=force)

    def note_model_time(self, seconds: float, frames: int) -> None:
        with self._lock:
            self.model_seconds += float(seconds)
            self.model_frames += int(frames)

    def message(self, text: str) -> None:
        with self._lock:
            self.event.message = text
        self._emit(force=True)

    def warn(self, count: int = 1) -> None:
        with self._lock:
            self.event.warnings += count

    def finish(self, stage: Stage, message: str = "") -> None:
        """Emit a terminal event (DONE / FAILED / CANCELLED)."""

        with self._lock:
            self.event.stage = stage
            self.event.message = message
            if stage is Stage.DONE:
                self.event.percent = 100.0
                self.event.overall_percent = 100.0
                self.event.eta_seconds = 0.0
        self._emit(force=True, terminal=True)

    # -- internals ---------------------------------------------------------------------
    def _overall(self, ev: ProgressEvent) -> Optional[float]:
        if ev.stage in (Stage.DONE,):
            return 100.0
        if ev.stage not in self._weights:
            return ev.overall_percent
        total = sum(self._weights[s] for s in self.STAGE_ORDER)
        if total <= 0:
            return None
        done = 0.0
        for s in self.STAGE_ORDER:
            if s is ev.stage:
                done += self._weights[s] * ((ev.percent or 0.0) / 100.0)
                break
            done += self._weights[s]
        return max(0.0, min(100.0, 100.0 * done / total))

    def _emit(self, force: bool = False, terminal: bool = False) -> None:
        now = time.perf_counter()
        with self._lock:
            if not force and (now - self._last_emit) < self._min_interval:
                return
            self._last_emit = now
            ev = self.event
            ev.elapsed_seconds = now - self._t0
            ev.timestamp = time.time()
            if not terminal:
                ev.overall_percent = self._overall(ev)
            if self._stats_provider is not None:
                try:
                    for key, value in self._stats_provider().items():
                        if hasattr(ev, key):
                            setattr(ev, key, value)
                except Exception:  # stats are best-effort, never break processing
                    pass
            snapshot = ProgressEvent(**{**ev.__dict__, "extras": dict(ev.extras)})
        if self._callback is not None:
            self._callback(snapshot)
