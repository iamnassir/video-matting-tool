"""FastAPI application: REST API for the local web interface plus the static UI.

``create_app(settings)`` builds a self-contained application object; nothing is global, so tests can
create as many independent instances as they like.

All endpoints live under ``/api``; everything else is served from ``app/ui/static``.  The server is
meant to run on the loopback interface for the person sitting at the machine: there is no
authentication, and endpoints such as "register a local file path" are only safe in that setting
(they are refused for non-loopback clients unless ``allow_path_import`` is set deliberately).
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import logging
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..core.config import (
    BACKGROUND_COLORS,
    CODECS,
    OUTPUT_CONTAINERS,
    QUALITY_PRESETS,
    REFINEMENT_PRESETS,
    SUPPORTED_INPUT_EXTENSIONS,
    AppSettings,
    BackgroundMode,
    ProcessingConfig,
    QualityPreset,
    SubjectPrompt,
    VideoCodec,
    apply_cache_environment,
    load_settings,
)
from ..core.errors import (
    ConfigError,
    DiskSpaceError,
    InputError,
    JobNotFoundError,
    LicenseError,
    ModelNotInstalledError,
    PromptNotSupportedError,
    ResourceBusyError,
    SubjectNotFoundError,
    UnknownModelError,
    VmtError,
    WeightsMissingError,
)
from ..core.events import CancelToken
from ..core.hardware import gpu_stats, ram_stats, system_info
from ..jobs.library import VideoLibrary
from ..jobs.manager import JobManager
from ..utils.fs import safe_filename

log = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parents[1] / "ui" / "static"


# =========================================================================== request bodies


class ImportBody(BaseModel):
    path: str = Field(..., description="Absolute path of a video file on this computer.")


class JobBody(BaseModel):
    video_id: str
    config: Dict[str, Any] = Field(default_factory=dict, description="ProcessingConfig (missing keys use the defaults).")


class BatchBody(BaseModel):
    video_ids: List[str]
    config: Dict[str, Any] = Field(default_factory=dict)


class RerenderBody(BaseModel):
    config: Dict[str, Any] = Field(default_factory=dict)


class SubjectPreviewBody(BaseModel):
    prompt: Dict[str, Any] = Field(default_factory=dict, description="SubjectPrompt")
    model: Optional[str] = Field(None, description="Matting model the subject is meant for.")


class DownloadBody(BaseModel):
    variant: Optional[str] = None
    accept_license: bool = False


# =========================================================================== helpers


def _status_for(exc: VmtError) -> int:
    if isinstance(exc, (JobNotFoundError, UnknownModelError)):
        return 404
    if isinstance(exc, ResourceBusyError):
        return 409
    if isinstance(exc, (WeightsMissingError, ModelNotInstalledError, PromptNotSupportedError)):
        return 409
    if isinstance(exc, LicenseError):
        return 403
    if isinstance(exc, DiskSpaceError):
        return 507
    if isinstance(exc, SubjectNotFoundError):
        return 422
    if isinstance(exc, (ConfigError, InputError)):
        return 400
    return 500


def _validation_text(exc: Exception) -> str:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            parts = []
            for e in errors():
                loc = ".".join(str(x) for x in e.get("loc", ()))
                msg = str(e.get("msg", "")).replace("Value error, ", "")
                parts.append(f"{loc}: {msg}" if loc else msg)
            return "; ".join(parts)
        except Exception:
            pass
    return str(exc)


def _merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _is_loopback(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return host in ("127.0.0.1", "::1", "localhost", "testclient")


class _DownloadTasks:
    """Explicit, user-triggered model downloads running in background threads."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._tokens: Dict[str, CancelToken] = {}

    def start(self, model_id: str, variant: Optional[str], accept_license: bool) -> Dict[str, Any]:
        from ..models import registry

        info = registry.get_model_info(model_id, self.settings)
        targets = [(info, info.variant(variant))]
        for dep in info.depends_on:
            dep_info = registry.get_model_info(dep, self.settings)
            targets.append((dep_info, dep_info.variant(None)))
        restricted = [t[0] for t in targets if t[0].license.commercial_use != "allowed"]
        if restricted and not accept_license:
            names = "; ".join(f"{i.id}: {i.license.name} (commercial use {i.license.commercial_use})" for i in restricted)
            raise LicenseError(
                f"Accept the licence terms before downloading: {names}.",
                hint="Tick 'I have read and accept the licence' and try again.",
                details={"licenses": {i.id: i.license.to_dict() for i in restricted}},
            )
        task_id = uuid.uuid4().hex[:10]
        task = {
            "id": task_id, "model": model_id, "variant": targets[0][1].id, "status": "running",
            "downloaded": 0, "total": sum((w.size_bytes or 0) for _, v in targets for w in v.weights),
            "file": "", "error": None, "started_at": time.time(),
        }
        token = CancelToken()
        with self._lock:
            self._tasks[task_id] = task
            self._tokens[task_id] = token

        def run() -> None:
            wm = registry.weights_manager(self.settings)
            done_before = 0
            try:
                for target_info, target_variant in targets:
                    def on_progress(done: int, total: Optional[int], name: str, _base: int = done_before) -> None:
                        task["downloaded"] = _base + done
                        task["file"] = name
                    paths = wm.ensure(target_info, target_variant, download=True, progress=on_progress, cancel=token)
                    done_before += sum(p.stat().st_size for p in paths)
                    task["downloaded"] = done_before
                task["status"] = "completed"
            except VmtError as exc:
                task["status"] = "cancelled" if token.cancelled else "failed"
                task["error"] = exc.to_dict()
            except Exception as exc:  # pragma: no cover
                task["status"] = "failed"
                task["error"] = {"code": "error", "message": str(exc), "hint": None}
            task["finished_at"] = time.time()

        threading.Thread(target=run, name="vmt-download", daemon=True).start()
        return dict(task)

    def get(self, task_id: str) -> Dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            raise JobNotFoundError(f"Download task '{task_id}' does not exist.")
        return dict(task)

    def cancel(self, task_id: str) -> Dict[str, Any]:
        with self._lock:
            token = self._tokens.get(task_id)
        if token is not None:
            token.cancel()
        return self.get(task_id)


class _SubjectService:
    """Interactive subject preview. Keeps its helper models loaded between clicks and frees them
    before a job starts or after a period of inactivity."""

    IDLE_SECONDS = 240.0

    def __init__(self, settings: AppSettings, library: VideoLibrary, jobs: JobManager) -> None:
        self.settings = settings
        self.library = library
        self.jobs = jobs
        self._lock = threading.Lock()
        self._selector: Any = None
        self._last_used = 0.0
        jobs.add_before_job_hook(self.release)

    def release(self) -> None:
        with self._lock:
            if self._selector is not None:
                try:
                    self._selector.close()
                finally:
                    self._selector = None
                from ..processing.batching import release_device_memory

                release_device_memory(None)

    def release_if_idle(self) -> None:
        if self._selector is not None and (time.time() - self._last_used) > self.IDLE_SECONDS:
            self.release()

    def preview(self, video_id: str, prompt_data: Dict[str, Any], model_id: Optional[str]) -> Dict[str, Any]:
        if self.jobs.busy():
            raise ResourceBusyError(
                "A job is being processed; the subject preview is available again when it has finished.",
                hint="You can still queue the job: the selection is resolved when the job starts.",
            )
        try:
            prompt = SubjectPrompt.model_validate(prompt_data or {})
        except Exception as exc:
            raise ConfigError(f"Invalid subject selection: {_validation_text(exc)}") from exc
        from ..core.hardware import select_device
        from ..models import registry
        from ..tracking.subject import SubjectSelector

        info = registry.get_model_info(model_id, self.settings) if model_id else None
        frame = self.library.frame_rgb(video_id, prompt.frame_index)
        height, width = frame.shape[:2]
        with self._lock:
            if self._selector is None:
                device, _, _ = select_device(self.settings.default_device, True)
                self._selector = SubjectSelector(self.settings, device)
            self._last_used = time.time()
            selector = self._selector
            selector.warnings.clear()
            moved: List[str] = []
            if prompt.mode == "mask":
                # A cut-out picture of another frame: find the frame it shows (see align_mask_frame).
                frames = _LibraryFrames(self.library, video_id)
                prompt, moved = selector.align_mask_frame(prompt, frames, frames.frame_count, budget_seconds=30.0)
                if moved:
                    frame = self.library.frame_rgb(video_id, prompt.frame_index)
            subject = selector.resolve_on_frame(prompt, frame, info, want_mask=True)
            notes = list(selector.warnings) + moved + list(subject.notes)
            backends = selector.backends()
        if info is not None and info.capabilities.subject_domain == "human" and prompt.mode != "auto":
            notes.append(
                f"{info.name} mattes people only. If the selection is something else (a vehicle, a product, "
                "an animal), choose the SAM 2 model, which follows any object."
            )
        if subject.mask is None:
            if prompt.mode == "auto":
                notes.append("No preview mask: the subject detector is not installed, so the model picks the primary "
                             "subject itself when the job runs (python main.py models download detector).")
            elif not backends.get("sam2"):
                notes.append("No preview mask: turning a box or clicks into a mask needs SAM 2, which is not installed "
                             "(python scripts/install_models.py sam2). The selection is still used when the job runs.")
        overlay = None
        area = None
        if subject.mask is not None:
            mask = np.clip(subject.mask, 0.0, 1.0)
            area = float((mask > 0.5).mean())
            scale = min(1.0, 960.0 / max(height, width))
            if scale < 1.0:
                mask = cv2.resize(mask, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
            rgba = np.zeros(mask.shape + (4,), np.uint8)
            rgba[..., 0] = 32                      # BGRA for OpenCV: a cyan tint
            rgba[..., 1] = 220
            rgba[..., 2] = 255
            rgba[..., 3] = (mask * 150).astype(np.uint8)
            rgba = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            ok, buf = cv2.imencode(".png", rgba, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if ok:
                overlay = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
        return {
            "frame_index": subject.frame_index,
            "width": width,
            "height": height,
            "box": [float(v) for v in subject.box] if subject.box else None,
            "points": [[float(x), float(y), int(label)] for x, y, label in subject.points],
            "source": subject.source,
            "score": subject.score,
            "label": subject.label,
            "has_mask": subject.mask is not None,
            "area_fraction": area,
            "overlay_png": overlay,
            "notes": notes,
            "backends": backends,
        }


class _LibraryFrames:
    """The frames of one library video with the reader interface the subject selector expects."""

    def __init__(self, library: Any, video_id: str) -> None:
        self._library = library
        self._video_id = video_id
        self.frame_count = int(library.probe(video_id).metadata.frame_count)

    def frame_rgb(self, index: int, max_side: Optional[int] = None) -> np.ndarray:
        return self._library.frame_rgb(self._video_id, index, max_side)


# =========================================================================== application


def create_app(settings: Optional[AppSettings] = None) -> FastAPI:
    settings = settings or load_settings()
    apply_cache_environment(settings)
    import tempfile

    tempfile.tempdir = str(settings.tmp)       # anything spooled to disk stays inside the data directory

    library = VideoLibrary(settings)
    jobs = JobManager(settings, library)
    downloads = _DownloadTasks(settings)
    subjects = _SubjectService(settings, library, jobs)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        subjects.release()
        library.release_readers()
        jobs.shutdown()

    app = FastAPI(title="Video Matting Tool", version=__version__, lifespan=lifespan, docs_url="/api/docs",
                  openapi_url="/api/openapi.json", redoc_url=None)
    app.state.settings = settings
    app.state.library = library
    app.state.jobs = jobs

    @app.exception_handler(VmtError)
    async def _vmt_error(request: Request, exc: VmtError) -> JSONResponse:
        status = _status_for(exc)
        if status >= 500:
            log.error("%s %s failed: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=status, content={"error": exc.to_dict()})

    api = APIRouter(prefix="/api")

    def _config(data: Dict[str, Any]) -> ProcessingConfig:
        base = settings.default_processing_config().model_dump(mode="json")
        try:
            return ProcessingConfig.model_validate(_merge(base, data or {}))
        except Exception as exc:
            raise ConfigError(f"Invalid settings: {_validation_text(exc)}") from exc

    # ------------------------------------------------------------------ general
    @api.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @api.get("/system")
    def get_system() -> Dict[str, Any]:
        info = system_info()
        try:
            from ..encoding.ffmpeg import find_ffmpeg

            tools = find_ffmpeg(settings)
            info["ffmpeg"] = {"version": tools.version, "path": tools.ffmpeg, "available": True}
        except VmtError as exc:
            info["ffmpeg"] = {"available": False, "error": exc.message, "hint": exc.hint}
        info["version"] = __version__
        info["data_dir"] = str(settings.data_dir)
        info["models_dir"] = str(settings.models_dir)
        return info

    @api.get("/system/stats")
    def get_stats() -> Dict[str, Any]:
        subjects.release_if_idle()
        out: Dict[str, Any] = dict(ram_stats())
        try:
            import torch

            if torch.cuda.is_available():
                out.update(gpu_stats(0))
        except Exception:
            pass
        out["busy"] = jobs.busy()
        out["current_job_id"] = jobs.current_job_id
        return out

    @api.get("/options")
    def get_options() -> Dict[str, Any]:
        available: Dict[str, bool] = {}
        try:
            from ..encoding.ffmpeg import find_ffmpeg

            tools = find_ffmpeg(settings)
            available = {c.value: tools.has_encoder(CODECS[c].ffmpeg_encoder) for c in VideoCodec}
        except VmtError:
            pass
        return {
            "version": __version__,
            "defaults": settings.default_processing_config().model_dump(mode="json"),
            "quality_presets": [
                {"id": q.value, "label": QUALITY_PRESETS[q].label, "description": QUALITY_PRESETS[q].description,
                 "refinement_preset": QUALITY_PRESETS[q].refinement_preset}
                for q in QualityPreset
            ],
            "refinement_presets": {name: dataclasses.asdict(p) for name, p in REFINEMENT_PRESETS.items()},
            "codecs": [
                {"id": c.value, "label": CODECS[c].label, "group": CODECS[c].group, "containers": list(CODECS[c].containers),
                 "supports_alpha": CODECS[c].supports_alpha, "lossless": CODECS[c].lossless, "hardware": CODECS[c].hardware,
                 "rate_control": CODECS[c].rate_control, "description": CODECS[c].description,
                 "available": available.get(c.value, False)}
                for c in VideoCodec
            ],
            "backgrounds": [
                {"id": b.value, "label": b.value.capitalize(),
                 "color": ("#%02X%02X%02X" % BACKGROUND_COLORS[b]) if b in BACKGROUND_COLORS else None}
                for b in BackgroundMode
            ],
            "containers": list(OUTPUT_CONTAINERS),
            "input_extensions": list(SUPPORTED_INPUT_EXTENSIONS),
            "allow_path_import": settings.allow_path_import,
        }

    # ------------------------------------------------------------------ models
    @api.get("/models")
    def list_models() -> List[Dict[str, Any]]:
        from ..models import registry

        out = []
        for info in registry.list_model_infos(settings, include_auxiliary=True):
            out.append({**info.to_dict(), "status": registry.model_status(info.id, settings)})
        return out

    @api.post("/models/{model_id}/download")
    def download_model(model_id: str, body: DownloadBody) -> Dict[str, Any]:
        return downloads.start(model_id, body.variant, body.accept_license)

    @api.get("/tasks/{task_id}")
    def get_task(task_id: str) -> Dict[str, Any]:
        return downloads.get(task_id)

    @api.post("/tasks/{task_id}/cancel")
    def cancel_task(task_id: str) -> Dict[str, Any]:
        return downloads.cancel(task_id)

    # ------------------------------------------------------------------ videos
    @api.get("/videos")
    def list_videos() -> List[Dict[str, Any]]:
        return [v.to_dict() for v in library.list()]

    @api.post("/videos/upload")
    async def upload_video(request: Request, filename: str = Query(..., description="Original file name")) -> Dict[str, Any]:
        """Upload a video as the raw request body (``Content-Type: application/octet-stream``)."""

        length = request.headers.get("content-length")
        expected = int(length) if length and length.isdigit() else None
        record = await library.add_upload(request.stream(), filename, expected)
        return record.to_dict()

    @api.post("/videos/import")
    def import_video(body: ImportBody, request: Request) -> Dict[str, Any]:
        if not settings.allow_path_import or not _is_loopback(request):
            raise ConfigError("Registering local file paths is disabled.",
                              hint="Upload the file instead, or enable allow_path_import and connect from this computer.")
        return library.add_path(body.path).to_dict()

    @api.get("/videos/{video_id}")
    def get_video(video_id: str) -> Dict[str, Any]:
        return library.get(video_id).to_dict()

    @api.delete("/videos/{video_id}")
    def delete_video(video_id: str) -> Dict[str, Any]:
        jobs.delete_for_video(video_id)
        library.delete(video_id)
        return {"deleted": video_id}

    @api.get("/videos/{video_id}/frame")
    def video_frame(video_id: str, index: int = Query(0, ge=0), max_side: int = Query(1280, ge=64, le=4096)) -> Response:
        data = library.frame_jpeg(video_id, index, max_side)
        return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"})

    @api.get("/videos/{video_id}/preview")
    def video_preview(video_id: str) -> FileResponse:
        record = library.get(video_id)
        path = library.proxy_path(video_id)
        if record.proxy != "ready" or not path.is_file():
            raise JobNotFoundError("The browser preview of this video is not available"
                                   + (f": {record.proxy_error}" if record.proxy_error else " yet."))
        return FileResponse(path, media_type="video/mp4")

    @api.post("/videos/{video_id}/subject-preview")
    def subject_preview(video_id: str, body: SubjectPreviewBody) -> Dict[str, Any]:
        return subjects.preview(video_id, body.prompt, body.model)

    # ------------------------------------------------------------------ jobs
    @api.get("/jobs")
    def list_jobs() -> List[Dict[str, Any]]:
        return [jobs.describe(j) for j in jobs.list()]

    @api.post("/jobs")
    def create_job(body: JobBody) -> Dict[str, Any]:
        return jobs.describe(jobs.create(body.video_id, _config(body.config)))

    @api.post("/jobs/batch")
    def create_batch(body: BatchBody) -> List[Dict[str, Any]]:
        if not body.video_ids:
            raise ConfigError("No videos selected.")
        config = _config(body.config)
        for video_id in body.video_ids:
            library.get(video_id)                       # validate all before queuing any
        return [jobs.describe(jobs.create(video_id, config)) for video_id in body.video_ids]

    @api.get("/jobs/{job_id}")
    def get_job(job_id: str) -> Dict[str, Any]:
        return jobs.describe(jobs.get(job_id))

    @api.delete("/jobs/{job_id}")
    def delete_job(job_id: str) -> Dict[str, Any]:
        jobs.delete(job_id)
        return {"deleted": job_id}

    @api.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> Dict[str, Any]:
        return jobs.describe(jobs.cancel(job_id))

    @api.post("/jobs/{job_id}/resume")
    def resume_job(job_id: str) -> Dict[str, Any]:
        return jobs.describe(jobs.resume(job_id))

    @api.post("/jobs/{job_id}/rerender")
    def rerender_job(job_id: str, body: RerenderBody) -> Dict[str, Any]:
        source = jobs.get(job_id)
        merged = _merge(source.config, body.config or {})
        return jobs.describe(jobs.rerender(job_id, _config(merged)))

    @api.get("/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request) -> StreamingResponse:
        """Server-sent events: one ``snapshot`` followed by ``progress`` / ``finished`` events."""

        job = jobs.get(job_id)
        loop = asyncio.get_running_loop()
        pending: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=64)

        def push(payload: Dict[str, Any]) -> None:       # called from the worker thread
            def put() -> None:
                if pending.full():
                    try:
                        pending.get_nowait()             # drop the oldest progress event, keep the newest
                    except asyncio.QueueEmpty:
                        pass
                pending.put_nowait(payload)

            try:
                loop.call_soon_threadsafe(put)
            except RuntimeError:
                pass                                     # the event loop is closing

        unsubscribe = jobs.subscribe(job_id, push)

        async def stream():
            try:
                yield _sse({"type": "snapshot", "job": jobs.describe(job)})
                if job.status not in ("queued", "running"):
                    return
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        payload = await asyncio.wait_for(pending.get(), timeout=10.0)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield _sse(payload)
                    if payload.get("type") == "finished":
                        return
            finally:
                unsubscribe()

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @api.get("/jobs/{job_id}/live.jpg")
    def job_live(job_id: str) -> Response:
        jobs.get(job_id)
        live = jobs.live(job_id)
        latest = live.latest() if live is not None else None
        if latest is None:
            return Response(status_code=204)
        data, frame_index, version = latest
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store", "X-Frame-Index": str(frame_index), "X-Version": str(version)})

    @api.get("/jobs/{job_id}/preview")
    def job_preview(job_id: str) -> FileResponse:
        jobs.get(job_id)
        path = jobs.job_dir(job_id) / "preview.mp4"
        if not path.is_file():
            raise JobNotFoundError("This job has no preview video yet.")
        return FileResponse(path, media_type="video/mp4")

    @api.get("/jobs/{job_id}/report")
    def job_report(job_id: str) -> Dict[str, Any]:
        return jobs.report(job_id).model_dump(mode="json")

    @api.get("/jobs/{job_id}/report.txt")
    def job_report_text(job_id: str) -> PlainTextResponse:
        return PlainTextResponse(jobs.report(job_id).to_text())

    @api.get("/jobs/{job_id}/log")
    def job_log(job_id: str, tail: int = Query(400, ge=1, le=20000)) -> PlainTextResponse:
        jobs.get(job_id)
        path = jobs.job_dir(job_id) / "job.log"
        if not path.is_file():
            return PlainTextResponse("")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return PlainTextResponse("\n".join(lines[-tail:]))

    @api.get("/jobs/{job_id}/files/{name:path}")
    def job_file(job_id: str, name: str) -> FileResponse:
        if name.endswith(".zip"):
            folder_name = name[:-4]
            try:
                folder = jobs.output_path(job_id, folder_name)
            except JobNotFoundError:
                folder = None
            if folder is not None and folder.is_dir():
                archive = folder.with_suffix(".zip")
                if not archive.is_file():
                    tmp = archive.with_suffix(".zip.tmp")
                    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
                        for item in sorted(folder.iterdir()):
                            if item.is_file():
                                zf.write(item, arcname=f"{folder.name}/{item.name}")
                    tmp.replace(archive)
                return FileResponse(archive, media_type="application/zip", filename=archive.name)
        path = jobs.output_path(job_id, name)
        if path.is_dir():
            raise JobNotFoundError(f"'{name}' is a folder; download '{name}.zip' instead.")
        return FileResponse(path, filename=safe_filename(path.name))

    app.include_router(api)

    @app.get("/api/{rest:path}", include_in_schema=False)
    def _unknown_api(rest: str) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": {"code": "not_found", "message": f"Unknown API path /api/{rest}"}})

    if UI_DIR.is_dir() and (UI_DIR / "index.html").is_file():
        app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
    else:
        @app.get("/", include_in_schema=False)
        def _no_ui() -> PlainTextResponse:
            return PlainTextResponse("The web interface files are missing (app/ui/static). The API is available under /api.")

    return app


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"
