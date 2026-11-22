"""Job management for the web interface: queue, persistence, cancellation, live progress.

Jobs are processed one at a time by a single worker thread (one GPU, bounded memory).  Every job has
its own folder ``<data_dir>/jobs/<id>/``:

* ``job.json``    - the record, rewritten atomically on every state change
* ``work/``       - the matte cache (kept, so a job can be resumed or re-rendered with new settings)
* ``out/``        - deliverables
* ``preview.mp4`` - the 2x2 preview video (original / subject / matte / composite)
* ``report.json`` - the processing report
* ``job.log``     - the log of this job

Jobs that were queued or running when the server stopped are marked ``interrupted`` on the next start
and can be resumed; their cached mattes are reused.
"""

from __future__ import annotations

import faulthandler
import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..core.config import CONTAINER_EXTENSIONS, CODECS, AppSettings, BackgroundMode, ProcessingConfig
from ..core.errors import ConfigError, JobCancelledError, JobNotFoundError, VmtError
from ..core.events import CancelToken, ProgressEvent, Stage
from ..core.log import add_job_log_handler
from ..core.report import ProcessingReport
from ..processing.live import LivePreview
from ..utils.fs import atomic_write_text, safe_filename, safe_rmtree
from .library import VideoLibrary

log = logging.getLogger(__name__)

ACTIVE = ("queued", "running")
FINISHED = ("completed", "completed_with_warnings", "failed", "cancelled", "interrupted")
RESUMABLE = ("failed", "cancelled", "interrupted")


class _StallWatch:
    """Diagnosis of hangs: when a running job reports no progress for a long time, the stacks of all
    threads are appended to a file.

    It uses the watchdog of ``faulthandler``, a C-level thread that needs no interpreter lock, so it
    still works when the Python side of the process is completely blocked (a deadlock inside a native
    library, for example).  A dump during a legitimately silent phase is harmless.
    """

    def __init__(self, path: Path, seconds: float) -> None:
        self.path = path
        self.seconds = float(seconds)
        self._file: Any = None
        self._armed_at = 0.0
        self._lock = threading.Lock()

    def arm(self, header: Optional[str] = None) -> None:
        if self.seconds <= 0:
            return
        now = time.monotonic()
        with self._lock:
            if header is None and now - self._armed_at < min(5.0, self.seconds / 4.0):
                return
            self._armed_at = now
            try:
                if self._file is None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    self._file = open(self.path, "a", encoding="utf-8", buffering=1)
                if header:
                    self._file.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} {header} ---\n")
                faulthandler.dump_traceback_later(self.seconds, repeat=True, file=self._file)
            except (OSError, ValueError, RuntimeError) as exc:
                log.debug("stall watchdog not armed: %s", exc)

    def disarm(self) -> None:
        with self._lock:
            self._armed_at = 0.0
            try:
                faulthandler.cancel_dump_traceback_later()
            except Exception:  # pragma: no cover - best effort
                pass

    def close(self) -> None:
        self.disarm()
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
                self._file = None


@dataclass
class JobRecord:
    id: str
    video_id: str
    name: str                                  # display name (source file name)
    config: Dict[str, Any]                     # ProcessingConfig dump
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: Dict[str, Any] = field(default_factory=dict)      # last ProgressEvent
    error: Optional[Dict[str, Any]] = None
    outputs: List[Dict[str, Any]] = field(default_factory=list)  # {kind, label, filename, size_bytes, url}
    warnings: List[str] = field(default_factory=list)
    cache_job_id: Optional[str] = None         # job whose work/ folder (matte cache) this job uses
    output_name: str = ""                      # file name of the main deliverable
    has_report: bool = False
    has_preview: bool = False
    summary: Dict[str, Any] = field(default_factory=dict)        # compact result facts for the job list

    def to_dict(self, cache_available: Optional[bool] = None) -> Dict[str, Any]:
        base = f"/api/jobs/{self.id}"
        done = self.status in ("completed", "completed_with_warnings")
        return {
            # What the interface may offer for this job (so it need not infer it from the status).
            "resumable": self.status in RESUMABLE,
            "rerenderable": done,
            "cache_available": cache_available,
            "id": self.id,
            "video_id": self.video_id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "config": self.config,
            "progress": self.progress,
            "error": self.error,
            "outputs": self.outputs,
            "warnings": self.warnings,
            "cache_job_id": self.cache_job_id,
            "output_name": self.output_name,
            "summary": self.summary,
            "report_url": f"{base}/report" if self.has_report else None,
            "preview_url": f"{base}/preview" if self.has_preview else None,
            "live_url": f"{base}/live.jpg",
            "events_url": f"{base}/events",
        }


class JobManager:
    """Owns the job queue and the single worker thread."""

    def __init__(self, settings: AppSettings, library: VideoLibrary, autostart: bool = True) -> None:
        self.settings = settings
        self.library = library
        self.root = settings.data_dir / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: Dict[str, JobRecord] = {}
        self._tokens: Dict[str, CancelToken] = {}
        self._live: Dict[str, LivePreview] = {}
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._listeners: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self._before_job: List[Callable[[], None]] = []
        self._stop = threading.Event()
        self._last_save: Dict[str, float] = {}
        self._save_lock = threading.Lock()
        self._stall = _StallWatch(settings.data_dir / "logs" / "stall-traces.log", settings.stall_dump_seconds)
        self.current_job_id: Optional[str] = None
        self._load()
        self._worker = threading.Thread(target=self._run, name="vmt-job-worker", daemon=True)
        if autostart:
            self._worker.start()

    # ------------------------------------------------------------------------------ persistence
    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _save(self, job: JobRecord) -> None:
        # Serialised: the worker thread (progress) and request threads (cancel, resume) both save.
        with self._save_lock:
            data = {k: getattr(job, k) for k in JobRecord.__dataclass_fields__}
            try:
                atomic_write_text(self.job_dir(job.id) / "job.json", json.dumps(data, default=str))
            except OSError as exc:
                log.warning("Could not persist job %s: %s", job.id, exc)

    def describe(self, job: JobRecord) -> Dict[str, Any]:
        """The job as the API returns it, including whether its matte cache still exists."""

        work = self.job_dir(job.cache_job_id or job.id) / "work"
        try:
            cached = work.is_dir() and any(work.iterdir())
        except OSError:
            cached = False
        return job.to_dict(cache_available=bool(cached))

    def _load(self) -> None:
        for entry in sorted(self.root.iterdir()):
            path = entry / "job.json"
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                job = JobRecord(**{k: data[k] for k in JobRecord.__dataclass_fields__ if k in data})
            except Exception as exc:
                log.warning("Ignoring unreadable job folder %s: %s", entry.name, exc)
                continue
            if job.status in ACTIVE:
                job.status = "interrupted"
                job.finished_at = time.time()
                job.error = {
                    "code": "interrupted",
                    "message": "The server stopped while this job was queued or running.",
                    "hint": "Resume the job: mattes that were already computed are reused.",
                }
                self._save(job)
            self._jobs[job.id] = job

    # ------------------------------------------------------------------------------ queries
    def list(self) -> List[JobRecord]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"Job '{job_id}' does not exist.")
        return job

    def live(self, job_id: str) -> Optional[LivePreview]:
        with self._lock:
            return self._live.get(job_id)

    def busy(self) -> bool:
        return self.current_job_id is not None

    def report(self, job_id: str) -> ProcessingReport:
        self.get(job_id)
        path = self.job_dir(job_id) / "report.json"
        if not path.is_file():
            raise JobNotFoundError(f"Job '{job_id}' has no report yet.")
        return ProcessingReport.load(path)

    def output_path(self, job_id: str, filename: str) -> Path:
        """Resolve a deliverable of a job; refuses anything outside the job's output folder."""

        self.get(job_id)
        out = (self.job_dir(job_id) / "out").resolve()
        name = safe_filename(filename)
        path = (out / name).resolve()
        if out not in path.parents or not path.exists():
            raise JobNotFoundError(f"Job '{job_id}' has no file named '{filename}'.")
        return path

    # ------------------------------------------------------------------------------ events
    def subscribe(self, job_id: str, callback: Callable[[Dict[str, Any]], None]) -> Callable[[], None]:
        """Register a listener for a job's events; returns the function that removes it."""

        with self._lock:
            self._listeners.setdefault(job_id, []).append(callback)

        def unsubscribe() -> None:
            with self._lock:
                listeners = self._listeners.get(job_id, [])
                if callback in listeners:
                    listeners.remove(callback)

        return unsubscribe

    def _emit(self, job: JobRecord, kind: str) -> None:
        payload = {"type": kind, "job": self.describe(job)}
        with self._lock:
            listeners = list(self._listeners.get(job.id, []))
        for callback in listeners:
            try:
                callback(payload)
            except Exception as exc:  # a broken subscriber must never stop a job
                log.debug("job listener failed: %s", exc)

    def add_before_job_hook(self, hook: Callable[[], None]) -> None:
        """Called on the worker thread right before a job starts (used to free interactive helper models)."""

        self._before_job.append(hook)

    # ------------------------------------------------------------------------------ creating
    def _output_name(self, source_name: str, config: ProcessingConfig) -> str:
        enc = config.encoding
        if enc.container:
            ext = CONTAINER_EXTENSIONS[enc.container]
        elif enc.codec is not None:
            ext = CONTAINER_EXTENSIONS[CODECS[enc.codec].default_container]
        else:
            ext = ".mov" if config.background is BackgroundMode.TRANSPARENT else ".mp4"
        stem = Path(safe_filename(source_name)).stem or "video"
        return f"{stem}_{config.background.value}{ext}"

    def create(self, video_id: str, config: ProcessingConfig, cache_job_id: Optional[str] = None) -> JobRecord:
        video = self.library.get(video_id)
        if not Path(video.path).is_file():
            raise ConfigError(f"The source file of this video is no longer available: {video.path}")
        if cache_job_id is not None:
            self.get(cache_job_id)
        job_id = uuid.uuid4().hex[:12]
        folder = self.job_dir(job_id)
        (folder / "out").mkdir(parents=True, exist_ok=False)
        job = JobRecord(
            id=job_id,
            video_id=video_id,
            name=video.filename,
            config=config.model_dump(mode="json"),
            cache_job_id=cache_job_id,
            output_name=self._output_name(video.filename, config),
        )
        with self._lock:
            self._jobs[job_id] = job
            self._tokens[job_id] = CancelToken()
        self._save(job)
        self._queue.put(job_id)
        self._emit(job, "created")
        return job

    def rerender(self, job_id: str, config: ProcessingConfig) -> JobRecord:
        """New job for the same video that reuses the cached mattes of *job_id* when settings allow."""

        source = self.get(job_id)
        root = source.cache_job_id or source.id
        return self.create(source.video_id, config, cache_job_id=root)

    def resume(self, job_id: str) -> JobRecord:
        """Queue an interrupted / failed / cancelled job again (its matte cache is reused)."""

        job = self.get(job_id)
        if job.status in ACTIVE:
            raise ConfigError("The job is already queued or running.")
        if job.status in ("completed", "completed_with_warnings"):
            raise ConfigError("The job is already complete; use re-render to apply new settings.")
        with self._lock:
            job.status = "queued"
            job.error = None
            job.finished_at = None
            job.progress = {}
            self._tokens[job_id] = CancelToken()
        self._save(job)
        self._queue.put(job_id)
        self._emit(job, "queued")
        return job

    # ------------------------------------------------------------------------------ control
    def cancel(self, job_id: str) -> JobRecord:
        job = self.get(job_id)
        with self._lock:
            token = self._tokens.get(job_id)
            if job.status == "queued":
                job.status = "cancelled"
                job.finished_at = time.time()
                self._save(job)
            elif job.status == "running" and token is not None:
                token.cancel("cancelled by user")
        self._emit(job, "cancel_requested")
        return job

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        if job.status == "running":
            raise ConfigError("A running job cannot be deleted; cancel it first.")
        with self._lock:
            dependants = [j.id for j in self._jobs.values() if j.cache_job_id == job_id and j.status in ACTIVE]
            if dependants:
                raise ConfigError("Another queued job still uses this job's matte cache; cancel it first.")
            if job.status == "queued":
                job.status = "cancelled"
            self._jobs.pop(job_id, None)
            self._tokens.pop(job_id, None)
            self._live.pop(job_id, None)
            self._listeners.pop(job_id, None)
        if not safe_rmtree(self.job_dir(job_id)):
            log.warning("Could not remove the folder of job %s completely (files in use).", job_id)

    def delete_for_video(self, video_id: str) -> None:
        for job in self.list():
            if job.video_id == video_id:
                if job.status == "running":
                    raise ConfigError("A job for this video is running; cancel it first.")
                self.delete(job.id)

    def shutdown(self, timeout: float = 15.0) -> None:
        self._stop.set()
        with self._lock:
            current = self.current_job_id
            token = self._tokens.get(current) if current else None
        if token is not None:
            token.cancel("stopped because the server is shutting down")
        self._queue.put(None)
        if self._worker.is_alive():
            self._worker.join(timeout)
        self._stall.close()

    # ------------------------------------------------------------------------------ worker
    def _run(self) -> None:
        while not self._stop.is_set():
            job_id = self._queue.get()
            if job_id is None:
                break
            with self._lock:
                job = self._jobs.get(job_id)
                token = self._tokens.get(job_id)
            if job is None or job.status != "queued" or token is None:
                continue                     # deleted or cancelled while waiting
            try:
                self._process(job, token)
            except Exception:                # _process handles everything; this is a last resort
                log.exception("Unexpected failure in the job worker")

    def _on_progress(self, job: JobRecord, event: ProgressEvent) -> None:
        self._stall.arm()
        job.progress = event.to_dict()
        now = time.time()
        if now - self._last_save.get(job.id, 0.0) > 3.0:
            self._last_save[job.id] = now
            self._save(job)
        self._emit(job, "progress")

    def _process(self, job: JobRecord, token: CancelToken) -> None:
        from ..processing.pipeline import MattingPipeline   # heavy import (PyTorch) on first use

        folder = self.job_dir(job.id)
        work = self.job_dir(job.cache_job_id or job.id) / "work"
        live = LivePreview()
        with self._lock:
            self._live[job.id] = live
            self.current_job_id = job.id
            job.status = "running"
            job.started_at = time.time()
            job.finished_at = None
            job.error = None
            job.outputs = []
            job.warnings = []
        self._save(job)
        self._emit(job, "started")
        for hook in self._before_job:
            try:
                hook()
            except Exception as exc:
                log.debug("before-job hook failed: %s", exc)
        handler = add_job_log_handler(folder / "job.log")
        report: Optional[ProcessingReport] = None
        status, error = "failed", None
        self._stall.arm(f"job {job.id} ({job.name}) started")
        try:
            video = self.library.get(job.video_id)
            config = ProcessingConfig.model_validate(job.config)
            # Jobs keep their matte cache so they can be resumed / re-rendered; outputs may be replaced
            # when a job is resumed.
            config = config.model_copy(update={"overwrite": True, "resume": True})
            pipeline = MattingPipeline(self.settings)
            report = pipeline.run(
                Path(video.path),
                folder / "out" / job.output_name,
                config,
                progress=lambda event: self._on_progress(job, event),
                cancel=token,
                work_dir=work,
                live=live,
                preview_path=folder / "preview.mp4",
                report_path=folder / "report.json",
            )
            status = report.status
        except JobCancelledError as exc:
            status = "interrupted" if self._stop.is_set() else "cancelled"
            error = exc.to_dict()
            report = getattr(exc, "report", None)
        except VmtError as exc:
            status = "failed"
            error = exc.to_dict()
            report = getattr(exc, "report", None)
            log.error("Job %s failed: %s", job.id, exc)
        except Exception as exc:  # pragma: no cover - the pipeline wraps unexpected errors itself
            status = "failed"
            error = {"code": "error", "message": f"{type(exc).__name__}: {exc}", "hint": None}
            log.exception("Job %s crashed", job.id)
        finally:
            self._stall.disarm()
            logging.getLogger().removeHandler(handler)
            handler.close()
            job.finished_at = time.time()
            job.has_report = (folder / "report.json").is_file()
            job.has_preview = (folder / "preview.mp4").is_file()
            if report is not None:
                job.warnings = list(report.warnings)
                job.outputs = self._collect_outputs(job, report)
                t = report.timing
                job.summary = {
                    "frames": t.frames_processed,
                    "frames_from_cache": t.frames_from_cache,
                    "seconds": round(t.total_seconds, 2),
                    "average_fps": t.average_fps,
                    "average_inference_fps": t.average_inference_fps,
                    "checks_failed": len(report.failed_checks),
                    "checks_warned": len(report.warned_checks),
                    "checks_total": len(report.checks),
                    "checks_run": bool(report.checks),
                    "device": report.hardware.device,
                    "model": (report.model or {}).get("id"),
                }
            # The status is published last, so nobody sees a finished job with stale results.
            with self._lock:
                job.error = error
                job.status = status
            with self._lock:
                self.current_job_id = None
            self._save(job)
            self._emit(job, "finished")

    def _collect_outputs(self, job: JobRecord, report: ProcessingReport) -> List[Dict[str, Any]]:
        out_dir = self.job_dir(job.id) / "out"
        result: List[Dict[str, Any]] = []
        for item in report.outputs:
            if item.kind == "report":
                continue
            path = Path(item.path)
            if not path.exists():
                continue
            try:
                rel = path.resolve().relative_to(out_dir.resolve())
            except ValueError:
                continue
            name = rel.as_posix()
            is_dir = path.is_dir()
            result.append(
                {
                    "kind": item.kind,
                    "label": item.label,
                    "filename": name,
                    "size_bytes": item.size_bytes,
                    "codec": item.codec,
                    "container": item.container,
                    "frames": item.frames,
                    "is_folder": is_dir,
                    "url": f"/api/jobs/{job.id}/files/{name}" + (".zip" if is_dir else ""),
                }
            )
        return result
