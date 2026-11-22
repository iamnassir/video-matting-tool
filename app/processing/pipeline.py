"""The streaming matting pipeline.

``MattingPipeline.run`` takes one input file to one verified output file::

    probe -> subject -> model load -> [backtrack] -> render -> mux -> verify -> deliver

The render stage is a single streaming pass: a decoder thread feeds batches of native-plane frames
through a bounded queue; the main thread converts them to R'G'B' on the inference device, obtains
the matte (from the model or from the matte cache), refines it, composites every requested target in
the native pixel domain and hands the result to one encoder per target, each of which pipes
timestamp-exact raw frames into an ``ffmpeg`` process.  Memory use is bounded by the queue sizes; the
video is never loaded as a whole.

Failure handling: every exit path aborts the encoders, removes partial files, leaves nothing under
the final output name and still produces a :class:`~app.core.report.ProcessingReport`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import queue
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .. import __version__
from ..compositing.colors import ColorConverter, planes_from_numpy, planes_to_numpy
from ..compositing.compositor import Compositor, Target, preview_atlas, rgb8_to_yuv420p
from ..core.config import (
    CODECS,
    CONTAINER_EXTENSIONS,
    QUALITY_PRESETS,
    SUPPORTED_INPUT_EXTENSIONS,
    AppSettings,
    ProcessingConfig,
    VideoCodec,
    apply_cache_environment,
    load_settings,
    parse_resolution,
    resolve_codec_and_container,
)
from ..core.errors import (
    ConfigError,
    CorruptedVideoError,
    GpuOutOfMemoryError,
    InvalidPathError,
    JobCancelledError,
    OutputExistsError,
    OutputPathError,
    VerificationError,
    VmtError,
)
from ..core.events import CancelToken, ProgressCallback, ProgressReporter, Stage
from ..core.hardware import ResourceMonitor, cpu_name, ram_stats, select_device
from ..core.pixfmt import ColorInfo, PixFmt, ResolvedColor, make_pixfmt
from ..core.report import HardwareSummary, OutputFile, ProcessingReport
from ..core.types import DecodedFrame, ProbeResult, ResolvedSubject, box_from_mask
from ..encoding.ffmpeg import FfmpegTools, find_ffmpeg
from ..encoding.muxer import extract_audio, mux_output
from ..encoding.nut_pipe import NutPipeEncoder
from ..encoding.profiles import EncodePlan, plan_encoding
from ..encoding.sinks import PngSequenceWriter
from ..models import registry
from ..models.base import MattingModel, SequenceInfo
from ..quality.verify import FrameSample, VerifyInput, verify_output
from ..refinement.refiner import MatteRefiner
from ..tracking.component_tracker import ComponentSubjectTracker
from ..tracking.monitor import TrackingMonitor
from ..tracking.subject import SubjectSelector
from ..utils import torch_ops
from ..utils.fs import (
    atomic_replace,
    atomic_write_text,
    ensure_parent_writable,
    free_bytes,
    human_bytes,
    safe_rmtree,
    safe_unlink,
)
from .batching import (
    BatchController,
    auto_batch_size,
    chunked_with_recovery,
    composite_chunk_size,
    process_with_recovery,
    release_device_memory,
)
from .live import LivePreview
from .matte_store import MatteStore
from .probe import probe
from .profiling import StageClock
from .reader import VideoReader

log = logging.getLogger(__name__)

#: Frames a recurrent model is fed before the resume point to rebuild its temporal state.
RESUME_WARMUP_FRAMES = 12
#: Frames sampled (evenly) for the pixel-level verification of the finished file.
VERIFY_SAMPLES = 10
_END = object()


# =========================================================================== helper threads


class _FrameFeeder(threading.Thread):
    """Decodes ahead of the render loop and delivers batches through a bounded queue."""

    def __init__(self, frames: Iterator[DecodedFrame], batch_size: int, max_batches: int) -> None:
        super().__init__(name="vmt-decoder", daemon=True)
        self._frames = frames
        self._batch_size = max(1, int(batch_size))
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, int(max_batches)))
        self._stop_event = threading.Event()

    def _put(self, item: Any) -> bool:
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def run(self) -> None:
        try:
            batch: List[DecodedFrame] = []
            for frame in self._frames:
                if self._stop_event.is_set():
                    return
                batch.append(frame)
                if len(batch) >= self._batch_size:
                    if not self._put(batch):
                        return
                    batch = []
            if batch:
                self._put(batch)
            self._put(_END)
        except BaseException as exc:  # delivered to the consumer, which re-raises it
            self._put(exc)

    def __iter__(self) -> Iterator[List[DecodedFrame]]:
        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                if not self.is_alive() and self._queue.empty():
                    return
                continue
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def stop(self) -> None:
        self._stop_event.set()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=10.0)
        close = getattr(self._frames, "close", None)
        if close is not None and not self.is_alive():
            try:
                close()
            except Exception:  # pragma: no cover
                pass


class _MatteWriter:
    """Writes raw mattes to the matte store on a background thread (compression off the hot path)."""

    def __init__(self, store: MatteStore, max_pending: int = 32) -> None:
        self._store = store
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max_pending)
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name="vmt-matte-writer", daemon=True)
        self._last_flush = time.perf_counter()
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _END:
                    return
                if self._error is None:
                    index, matte = item
                    self._store.put(index, matte)
                    if time.perf_counter() - self._last_flush > 10.0:
                        self._store.flush()          # bound the work lost by a crash to ~10 s
                        self._last_flush = time.perf_counter()
            except BaseException as exc:
                self._error = exc
            finally:
                self._queue.task_done()

    def put(self, index: int, matte: np.ndarray) -> None:
        if self._error is not None:
            raise self._error
        self._queue.put((index, matte))

    def drain(self) -> None:
        """Block until everything queued so far is in the store (so it can be read back)."""

        self._queue.join()
        if self._error is not None:
            raise self._error

    def close(self, flush: bool = True) -> None:
        if self._thread.is_alive():
            self._queue.put(_END)
            self._thread.join()
        if flush and self._error is None:
            self._store.flush()
        if self._error is not None:
            raise self._error


# =========================================================================== sinks


@dataclass
class _VideoSink:
    """One encoded deliverable fed by the render loop."""

    name: str                       # main | alpha_matte | prores_4444 | green_screen | preview
    label: str
    plan: EncodePlan
    intermediate: Path              # video-only file written by the encoder
    final: Path                     # delivered file
    with_audio: bool
    target: Optional[Target] = None  # compositor target; None for the preview atlas
    audio_mode: str = "auto"        # passed to the muxer
    keep_rotation: bool = True
    encoder: Optional[NutPipeEncoder] = None
    staged: Optional[Path] = None    # muxed file waiting for verification / delivery
    mux_info: Dict[str, Any] = field(default_factory=dict)


def _matte_codec_for(codec: VideoCodec) -> VideoCodec:
    return {
        VideoCodec.H264_NVENC: VideoCodec.H264,
        VideoCodec.HEVC_NVENC: VideoCodec.H265,
        VideoCodec.PRORES_4444: VideoCodec.PRORES_422_HQ,
    }.get(codec, codec)


def _suffixed(path: Path, suffix: str, extension: Optional[str] = None) -> Path:
    """``clip_green.mp4`` + ``_alpha`` -> ``clip_green_alpha.mp4``."""

    ext = extension if extension is not None else path.suffix
    return path.with_name(f"{path.stem}{suffix}{ext}")


# =========================================================================== pipeline


class MattingPipeline:
    """Runs matting jobs. Stateless between runs; safe to reuse."""

    def __init__(self, settings: Optional[AppSettings] = None) -> None:
        self.settings = settings or load_settings()

    def run(
        self,
        input_path: Any,
        output_path: Any,
        config: Optional[ProcessingConfig] = None,
        *,
        progress: Optional[ProgressCallback] = None,
        cancel: Optional[CancelToken] = None,
        work_dir: Optional[Any] = None,
        live: Optional[LivePreview] = None,
        preview_path: Optional[Any] = None,
        report_path: Optional[Any] = None,
        save_report: bool = True,
    ) -> ProcessingReport:
        """Process one video.

        Returns the :class:`ProcessingReport`.  On failure a :class:`VmtError` is raised whose
        ``report`` attribute holds the (failed) report; the output path is left untouched.

        * ``work_dir``     - scratch directory (matte cache, intermediates). Default: a hidden folder
          next to the output.  It is removed after a successful run unless ``config.keep_work_dir``.
        * ``live``         - receives live preview images while rendering.
        * ``preview_path`` - if given, the 2x2 preview video is written there (used by the web UI).
        * ``report_path``  - where to save the JSON report (default ``<output stem>.report.json``).
        """

        config = config or self.settings.default_processing_config()
        cancel = cancel or CancelToken()
        requested = config.device if config.device != "auto" else self.settings.default_device
        device_request = requested
        fallback_reason: Optional[str] = None
        while True:
            run = _Run(
                self.settings, Path(input_path), Path(output_path), config, progress, cancel,
                Path(work_dir) if work_dir else None, live, Path(preview_path) if preview_path else None,
                Path(report_path) if report_path else None, save_report, device_request, requested, fallback_reason,
            )
            try:
                return run.execute()
            except GpuOutOfMemoryError as exc:
                if run.device_type == "cpu" or not config.allow_cpu_fallback or device_request == "cpu":
                    raise
                fallback_reason = f"GPU out of memory ({exc.message}) - continued on the CPU"
                log.warning("%s. Mattes already computed are reused.", fallback_reason)
                device_request = "cpu"


class _Run:
    """State of one processing attempt."""

    def __init__(
        self,
        settings: AppSettings,
        input_path: Path,
        output_path: Path,
        config: ProcessingConfig,
        progress: Optional[ProgressCallback],
        cancel: CancelToken,
        work_dir: Optional[Path],
        live: Optional[LivePreview],
        preview_path: Optional[Path],
        report_path: Optional[Path],
        save_report: bool,
        device_request: str,
        requested_device: str,
        fallback_reason: Optional[str],
    ) -> None:
        self.settings = settings
        self.input = input_path
        self.output = output_path
        self.config = config
        self.cancel = cancel
        self.live = live
        self.preview_path = preview_path
        self.report_path = report_path
        self.save_report = save_report
        self.device_request = device_request
        self.requested_device = requested_device
        self.fallback_reason = fallback_reason
        self.force_resume = fallback_reason is not None      # a CPU retry must reuse the cached mattes
        self.user_work_dir = work_dir

        self.report = ProcessingReport(
            input_path=str(input_path), output_path=str(output_path), config=config.model_dump(mode="json")
        )
        self.monitor: Optional[ResourceMonitor] = None
        self.reporter = ProgressReporter(progress, stats_provider=self._stats)
        self.stage_seconds: Dict[str, float] = {}
        self.device_type = "cpu"

        # resources created during the run (all closed in _cleanup)
        self.tools: Optional[FfmpegTools] = None
        self.reader: Optional[VideoReader] = None
        self.model: Optional[MattingModel] = None
        self.store: Optional[MatteStore] = None
        self.matte_writer: Optional[_MatteWriter] = None
        self.feeder: Optional[_FrameFeeder] = None
        self.sinks: List[_VideoSink] = []
        self.png: Optional[PngSequenceWriter] = None
        self.png_dir: Optional[Path] = None
        self.png_stage: Optional[Path] = None
        self.work_dir: Optional[Path] = None
        self.delivered: List[Path] = []

        self.subject: Optional[ResolvedSubject] = None
        self.tracker: Optional[ComponentSubjectTracker] = None
        self.sequence_started = False
        self.written_pts: List[int] = []
        self.samples: List[FrameSample] = []
        self.frames_from_cache = 0
        self.model_frames = 0
        self.presence_low = 0

    # ---------------------------------------------------------------------------- utilities
    def _stats(self) -> Dict[str, Any]:
        return self.monitor.stats() if self.monitor is not None else {}

    class _Timer:
        def __init__(self, run: "_Run", name: str) -> None:
            self.run, self.name = run, name

        def __enter__(self) -> "_Run._Timer":
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *exc: Any) -> None:
            self.run.stage_seconds[self.name] = self.run.stage_seconds.get(self.name, 0.0) + time.perf_counter() - self.t0

    def _timed(self, name: str) -> "_Run._Timer":
        return _Run._Timer(self, name)

    def _warn(self, text: str) -> None:
        if text and text not in self.report.warnings:
            log.warning(text)
            self.report.add_warning(text)
            self.reporter.warn()

    # ---------------------------------------------------------------------------- entry point
    def execute(self) -> ProcessingReport:
        t_start = time.perf_counter()
        error: Optional[BaseException] = None
        try:
            with torch.inference_mode():
                self._prepare()
                self._backtrack()
                self._render()
            self._mux()
            self._verify()
            self._deliver()
            self.report.finalize_status()
        except JobCancelledError as exc:
            error = exc
            self.report.status = "cancelled"
            self.report.error = exc.to_dict()
        except VmtError as exc:
            error = exc
            self.report.status = "failed"
            self.report.error = exc.to_dict()
        except Exception as exc:  # unexpected: wrap, keep the traceback in the log
            log.exception("Unexpected error while processing %s", self.input)
            error = VmtError(
                f"Unexpected error: {type(exc).__name__}: {exc}",
                hint="Run again with --verbose and check the log; this is a bug worth reporting.",
            )
            error.__cause__ = exc
            self.report.status = "failed"
            self.report.error = error.to_dict()
        finally:
            self._cleanup(failed=error is not None)
            self._finish_report(time.perf_counter() - t_start)

        if error is None:
            self.reporter.finish(Stage.DONE, "Finished")
            return self.report
        setattr(error, "report", self.report)
        if isinstance(error, GpuOutOfMemoryError):
            raise error                      # the caller may retry on the CPU; no terminal event yet
        self.reporter.finish(
            Stage.CANCELLED if isinstance(error, JobCancelledError) else Stage.FAILED, str(error)
        )
        raise error

    # ---------------------------------------------------------------------------- prepare
    def _prepare(self) -> None:
        cfg, settings = self.config, self.settings
        apply_cache_environment(settings)
        self.tools = find_ffmpeg(settings)
        self.report.hardware.ffmpeg = self.tools.version

        # ---- paths -------------------------------------------------------------------
        inp = self.input
        if not inp.exists():
            raise InvalidPathError(f"Input file not found: {inp}")
        if not inp.is_file():
            raise InvalidPathError(f"Input path is not a file: {inp}")
        out = ensure_parent_writable(self.output)
        if out.exists() and out.is_dir():
            raise OutputPathError(f"The output path is a directory: {out}", hint="Give a file name, or use --output-directory.")
        try:
            same = out.exists() and out.samefile(inp)
        except OSError:
            same = False
        if same or out.resolve() == inp.resolve():
            raise OutputPathError("The output path is the same file as the input.", hint="Choose a different output name.")
        if out.exists() and not cfg.overwrite:
            raise OutputExistsError(f"Output file already exists: {out}")

        # ---- probe -------------------------------------------------------------------
        self.reporter.stage(Stage.PROBE, "Analysing the input")
        with self._timed("probe"):
            self.probe: ProbeResult = probe(inp, ffprobe=self.tools.ffprobe)
        meta, video = self.probe.metadata, self.probe.metadata.video
        self.meta = meta
        self.report.input = meta.to_dict()
        for text in meta.warnings:
            self._warn(text)
        if inp.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
            self._warn(
                f"Input extension '{inp.suffix}' is not one of the officially supported containers "
                f"({', '.join(SUPPORTED_INPUT_EXTENSIONS)}); processing continues because FFmpeg can read it."
            )
        if meta.frame_count <= 0:
            raise CorruptedVideoError("The video stream contains no decodable frames.")
        self.total = min(meta.frame_count, cfg.max_frames) if cfg.max_frames else meta.frame_count
        self.truncated = self.total < meta.frame_count
        if self.truncated:
            self._warn(f"max_frames is set: only the first {self.total} of {meta.frame_count} frames are processed.")
        if video.is_interlaced:
            self._warn(
                f"The source is interlaced (field order {video.field_order}). Frames are matted as woven "
                "frames and the output is flagged progressive; deinterlace first for best edges on motion."
            )
        if video.color.is_hdr:
            self._warn(
                f"HDR source ({video.color.transfer}): pixel values and colour tags are passed through "
                "unchanged; no tone mapping is applied and dynamic HDR metadata (e.g. Dolby Vision) is not carried over."
            )
        if meta.subtitle_streams or meta.data_streams:
            self._warn(
                f"The source has {meta.subtitle_streams} subtitle and {meta.data_streams} data stream(s); "
                "only video and audio are written to the output."
            )
        self.rotation = video.rotation
        self.dw, self.dh = video.display_width, video.display_height
        self.fps = float(meta.frame_rate) if meta.frame_rate else 30.0

        # ---- output format -----------------------------------------------------------
        self.codec, self.container = resolve_codec_and_container(cfg, out, video.bit_depth)
        spec = cfg.encoding.resolution
        ow, oh, resized = parse_resolution(spec, video.width, video.height)
        if resized and "x" in spec and self.rotation in (90, 270):
            ow, oh = oh, ow            # a WxH request refers to the picture as displayed
            resized = (ow, oh) != (video.width, video.height)
        self.out_w, self.out_h, self.resized = ow, oh, resized
        if resized:
            self._warn(f"Output is scaled to {ow}x{oh} on explicit request (source {video.width}x{video.height}).")

        # ---- device ------------------------------------------------------------------
        self.device, self.device_info, fallback = select_device(self.device_request, cfg.allow_cpu_fallback)
        self.device_type = self.device.type
        reason = self.fallback_reason or fallback
        if reason:
            self._warn(f"Not running on the requested device: {reason}")
        self.monitor = ResourceMonitor(self.device, self.device_info)
        self.clock = StageClock(
            self.device, enabled=os.environ.get("VMT_PROFILE", "") == "1" or log.isEnabledFor(logging.DEBUG)
        )
        if self.device_type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass
        hw = self.report.hardware
        hw.device = str(self.device)
        hw.device_name = self.device_info.name
        hw.requested_device = self.requested_device
        hw.fallback_reason = reason
        hw.vram_total_mb = self.device_info.total_memory_mb if self.device_type != "cpu" else None
        hw.vram_free_at_start_mb = self.device_info.free_memory_mb if self.device_type != "cpu" else None
        ram = ram_stats()
        hw.ram_total_mb = ram["ram_total_mb"]
        hw.cpu = cpu_name()
        import os as _os

        hw.cpu_threads = _os.cpu_count()
        hw.os = platform.platform()
        hw.python = platform.python_version()
        hw.torch = torch.__version__
        hw.cuda = torch.version.cuda if self.device_type == "cuda" else None
        self.reporter.set_static(device=str(self.device), gpu_name=self.device_info.name if self.device_type != "cpu" else None)

        # ---- reader ------------------------------------------------------------------
        self.reader = VideoReader(inp, self.probe, tolerate_errors=cfg.tolerate_decode_errors)
        self.fmt: PixFmt = self.reader.fmt
        self.color: ResolvedColor = self.reader.color
        if self.color.assumed:
            log.info("Source colour tags missing (%s): assuming %s, %s range.", ", ".join(self.color.assumed),
                     self.color.matrix, "full" if self.color.full_range else "limited")
        if not self.reader.working.exact:
            self._warn(
                f"Source pixel format {video.pix_fmt} has no exact planar equivalent; it is converted to "
                f"{self.fmt.name} on input ({self.reader.working.note})."
            )

        # ---- work directory ----------------------------------------------------------
        if self.user_work_dir is not None:
            self.work_dir = self.user_work_dir
        else:
            tag = hashlib.sha256(f"{inp.resolve()}|{out.resolve()}".encode("utf-8")).hexdigest()[:10]
            self.work_dir = out.parent / ".vmt_work" / f"{out.stem}-{tag}"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # ---- model (licence, dependencies, weights) -------------------------------------
        self.model = registry.create_model(
            cfg.model,
            settings=settings,
            device=self.device,
            variant=cfg.variant,
            quality=cfg.quality,
            precision=cfg.precision,
            options=cfg.model_options,
            commercial_use=cfg.commercial_use,
            download=cfg.download_models,
            cancel=self.cancel,
            warnings=self.report.warnings,
        )
        info = self.model.info
        caps = info.capabilities
        self.report.hardware.precision = self.model.precision
        self.report.model = {
            "id": info.id,
            "name": info.name,
            "variant": self.model.variant.id,
            "license": info.license.to_dict(),
            "alpha_type": caps.alpha_type,
            "subject_domain": caps.subject_domain,
            "limitations": list(info.limitations),
            "homepage": info.homepage,
        }
        self.isolate = bool(cfg.subject.isolate and not caps.tracks_single_subject)

        # ---- matte store -------------------------------------------------------------
        self.matte_bits = 16 if int(cfg.model_options.get("matte_cache_bits", 8)) == 16 else 8
        self.matte_max = float((1 << self.matte_bits) - 1)
        stat = inp.stat()
        fingerprint = {
            "app": __version__.split(".")[0],
            "input": {"path": str(inp.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            "frames": meta.frame_count,
            "display_size": [self.dw, self.dh],
            "rotation": self.rotation,
            "raw": cfg.fingerprint_payload(),
            "variant": self.model.variant.id,
            "isolate": self.isolate,
        }
        store_dir = self.work_dir / "mattes"
        use_cache = cfg.resume or self.force_resume
        if not use_cache:
            safe_rmtree(store_dir)
        self.store = MatteStore(store_dir, self.dw, self.dh, fingerprint, bits=self.matte_bits)
        if self.store.reset_reason and use_cache:
            log.info("Existing matte cache was not reusable: %s", self.store.reset_reason)
        self.store_writes = use_cache
        self.matte_writer = _MatteWriter(self.store)
        self.cached_end = min(self.store.contiguous_end(0), self.total) if use_cache else 0
        self.report.resumed = self.cached_end > 0
        model_needed = self.cached_end < self.total

        # ---- subject -----------------------------------------------------------------
        self.reporter.stage(Stage.SUBJECT, "Identifying the subject")
        with self._timed("subject"):
            prompt = cfg.subject.to_pixels(self.dw, self.dh)
            if model_needed:
                selector = SubjectSelector(
                    settings, self.device, commercial_use=cfg.commercial_use, download=cfg.download_models,
                    cancel=self.cancel, warnings=self.report.warnings,
                )
                try:
                    self.subject = selector.resolve(prompt, self.reader, info, frame_limit=self.total)
                finally:
                    selector.close()
                release_device_memory(self.device)
                for note in self.subject.notes:
                    self._warn(note)
                summary = self.subject.summary()
                atomic_write_text(self.work_dir / "subject.json", _json(summary))
            else:
                self.subject = ResolvedSubject(frame_index=prompt.frame_index, source="cache")
                summary = _load_json(self.work_dir / "subject.json") or self.subject.summary()
            self.report.subject = summary
        self.ref = min(max(0, self.subject.frame_index), self.total - 1)
        if cfg.subject.mode in ("points", "box", "mask", "text") and not caps.tracks_single_subject:
            if not self.isolate:
                self._warn(
                    f"Model '{info.id}' has no subject selection and isolation is switched off: "
                    "every person the model detects is kept; the subject prompt is ignored."
                )
            else:
                self._warn(
                    f"Model '{info.id}' has no built-in subject selection: the selected subject is isolated by "
                    "connected-component tracking, which cannot separate people who touch or overlap."
                )

        # ---- where the model sequence starts (fresh run or resume) -----------------------
        self.seq_start: Optional[int] = None
        self.seq_subject: Optional[ResolvedSubject] = self.subject
        self.tracker_seed: Optional[ResolvedSubject] = self.subject
        if model_needed:
            if self.cached_end <= self.ref:
                self.seq_start = self.ref
            elif caps.tracks_single_subject or caps.needs_initial_mask or caps.native_prompts:
                j = self.store.last_nonempty_before(self.cached_end)
                if j is None or j < self.ref:
                    self.seq_start = self.ref       # no usable matte to re-seed from: redo from the reference frame
                else:
                    mask = self.store.get(j).astype(np.float32) / self.matte_max
                    self.seq_start = j
                    self.seq_subject = ResolvedSubject(
                        frame_index=j, mask=mask, box=box_from_mask(mask), source="resume:cached-matte"
                    )
            else:
                self.seq_start = max(self.ref, self.cached_end - RESUME_WARMUP_FRAMES)
                if self.seq_start > self.ref and self.store.has(self.seq_start):
                    mask = self.store.get(self.seq_start).astype(np.float32) / self.matte_max
                    if float((mask > 0.5).mean()) > 0.0005:
                        self.tracker_seed = ResolvedSubject(
                            frame_index=self.seq_start, mask=mask, box=box_from_mask(mask), source="resume:cached-matte"
                        )
            if self.report.resumed:
                log.info("Resuming: %d frame(s) come from the matte cache; the model restarts at frame %d.",
                         self.cached_end, self.seq_start)

        # ---- load the model --------------------------------------------------------------
        if model_needed:
            self.reporter.stage(Stage.MODEL_LOAD, f"Loading {info.name}")
            with self._timed("model_load"):
                self.model.load()
            self.report.model["runtime"] = self.model.runtime_summary()
        self.cancel.raise_if_cancelled()

        # ---- processing objects ------------------------------------------------------
        params = cfg.refinement_params()
        self.refiner = MatteRefiner(params, self.dw, self.dh, alpha_is_mask=(caps.alpha_type == "mask"))
        self.track_monitor = TrackingMonitor(self.dw, self.dh, self.fps)
        self.converter = ColorConverter(self.fmt, self.color)
        # Blur-Fusion uses box sizes of 90 and 6 pixels (radii 45 and 3) at around 1080p; larger
        # frames scale them up, smaller frames keep at least half of them.
        scale = max(0.5, min(video.width, video.height) / 1080.0)
        self.compositor = Compositor(
            self.fmt, self.color, video.width, video.height,
            decontaminate=params.decontaminate,
            decontaminate_radius=45.0 * scale,
            decontaminate_refine_radius=3.0 * scale,
        )
        free_mb = self.device_info.free_memory_mb if self.device_type != "cpu" else None
        self.controller = BatchController(
            auto_batch_size(self.dw, self.dh, self.device_type, free_mb, caps.max_batch_size, cfg.batch_size)
        )
        self.loop_batch = max(1, self.controller.size)
        self.comp_chunk = composite_chunk_size(video.width, video.height, self.device_type, free_mb)
        self.reporter.set_static(batch_size=self.controller.size)

        self._plan_sinks()
        self._check_disk_space()

        if self.total > 0:
            count = min(VERIFY_SAMPLES, self.total)
            self.sample_indices = set(int(round(x)) for x in np.linspace(0, self.total - 1, count))
        else:
            self.sample_indices = set()

    # ---------------------------------------------------------------------------- sinks
    def _plan_sinks(self) -> None:
        cfg, meta, video = self.config, self.meta, self.meta.video
        bg = cfg.background_rgb()
        self.background = bg
        size = (self.out_w, self.out_h) if self.resized else None
        common = dict(
            source_fmt=self.fmt, color=video.color, resolved_color=self.color, width=self.out_w, height=self.out_h,
            frame_rate=meta.frame_rate, time_base=video.time_base, is_vfr=meta.is_vfr, encoding=cfg.encoding,
            quality=cfg.quality, tools=self.tools,
        )
        ext = CONTAINER_EXTENSIONS[self.container]
        out = self.output

        def add(name: str, label: str, plan: EncodePlan, target: Optional[Target], final: Path,
                with_audio: bool, audio_mode: str = "auto", keep_rotation: bool = True) -> _VideoSink:
            if name not in ("main", "preview") or (name == "preview" and self.preview_path is None):
                if final.exists() and not cfg.overwrite:
                    raise OutputExistsError(f"Output file already exists: {final}")
            sink = _VideoSink(
                name=name, label=label, plan=plan, target=target, final=final, with_audio=with_audio,
                audio_mode=audio_mode, keep_rotation=keep_rotation,
                intermediate=self.work_dir / f"{name}_video{plan.extension}",
            )
            for text in plan.warnings:
                self._warn(text)
            self.sinks.append(sink)
            return sink

        plan = plan_encoding(purpose="main", codec=self.codec, container=self.container, with_alpha=bg is None, **common)
        if bg is None:
            target = Target("main", "alpha", plan.fmt, size=size, alpha_mode=cfg.exports.alpha_mode)
        else:
            target = Target("main", "composite", plan.fmt, background=bg, size=size)
        main = add("main", "Main output", plan, target, out, cfg.encoding.preserve_audio,
                   cfg.encoding.audio_codec)
        self.main_sink = main
        self.report.encoding = {
            "codec": self.codec.value,
            "container": self.container,
            "encoder": plan.encoder,
            "pix_fmt": plan.fmt.name,
            "lossless": plan.lossless,
            "video_args": list(plan.video_args),
            "background": "transparent" if bg is None else "#%02X%02X%02X" % bg,
        }

        ex = cfg.exports
        if ex.alpha_matte:
            codec = ex.alpha_matte_codec or _matte_codec_for(self.codec)
            container = self.container if self.container in CODECS[codec].containers else CODECS[codec].default_container
            mplan = plan_encoding(purpose="alpha_matte", codec=codec, container=container, with_alpha=False, **common)
            add("alpha_matte", "Alpha matte video", mplan, Target("alpha_matte", "matte", mplan.fmt, size=size),
                _suffixed(out, "_alpha", CONTAINER_EXTENSIONS[container]), with_audio=False)
        if ex.prores_4444 and not (self.codec is VideoCodec.PRORES_4444 and bg is None):
            pplan = plan_encoding(purpose="prores_4444", codec=VideoCodec.PRORES_4444, container="mov", with_alpha=True, **common)
            add("prores_4444", "Transparent ProRes 4444", pplan,
                Target("prores_4444", "alpha", pplan.fmt, size=size, alpha_mode=ex.alpha_mode),
                _suffixed(out, "_prores4444", ".mov"), cfg.encoding.preserve_audio, cfg.encoding.audio_codec)
        if ex.green_screen and bg != (0, 255, 0):
            gcodec = self.codec if not (bg is None and not CODECS[self.codec].supports_alpha) else VideoCodec.H264
            gcontainer = self.container if self.container in CODECS[gcodec].containers else CODECS[gcodec].default_container
            gplan = plan_encoding(purpose="green_screen", codec=gcodec, container=gcontainer, with_alpha=False, **common)
            add("green_screen", "Green-screen video", gplan,
                Target("green_screen", "composite", gplan.fmt, background=(0, 255, 0), size=size),
                _suffixed(out, "_green", CONTAINER_EXTENSIONS[gcontainer]), cfg.encoding.preserve_audio,
                cfg.encoding.audio_codec)
        if ex.png_sequence:
            self.png_dir = out.with_name(f"{out.stem}_png")
            if self.png_dir.exists() and not cfg.overwrite:
                raise OutputExistsError(f"Output folder already exists: {self.png_dir}")
            self.png_stage = out.with_name(f"{out.stem}_png.partial")
            safe_rmtree(self.png_stage)
            self.png = PngSequenceWriter(
                self.png_stage, bit_depth=ex.png_bit_depth, compression=ex.png_compression, cancel=self.cancel
            )

        self.preview_sink: Optional[_VideoSink] = None
        self.tile = torch_ops.fit_size(self.dw, self.dh, self.settings.preview_tile_max_side, multiple=2)
        if ex.matte_preview or self.preview_path is not None:
            tw, th = self.tile
            pfmt = make_pixfmt("yuv", 8, 1, 1)
            pcolor = ColorInfo(range="tv", space="bt709", primaries="bt709", transfer="bt709")
            pplan = plan_encoding(
                purpose="preview", codec=VideoCodec.H264, container="mp4", with_alpha=False,
                source_fmt=pfmt, color=pcolor, resolved_color=ResolvedColor("bt709", 0.2126, 0.0722, False),
                width=2 * tw, height=2 * th, frame_rate=meta.frame_rate, time_base=video.time_base,
                is_vfr=meta.is_vfr, encoding=cfg.encoding, quality=cfg.quality, tools=self.tools,
            )
            final = self.preview_path if self.preview_path is not None else _suffixed(out, "_preview", ".mp4")
            self.preview_sink = add(
                "preview", "Preview (original / subject / matte / composite)", pplan, None, final,
                with_audio=True, audio_mode="aac", keep_rotation=False,
            )

    def _check_disk_space(self) -> None:
        """Refuse to start when the volumes clearly cannot hold the results."""

        meta, video = self.meta, self.meta.video
        duration = float(meta.video_duration or meta.duration or 0.0) * (self.total / max(1, meta.frame_count))
        raw_frame = self.fmt.frame_bytes(self.out_w, self.out_h)
        need_out = 0.0
        need_work = 0.0
        for sink in self.sinks:
            est = sink.plan.estimated_bytes(duration, self.total, raw_frame, meta.bit_rate)
            need_work += est            # video-only intermediate
            need_out += est             # muxed file (the intermediate is deleted afterwards)
        if self.png is not None:
            need_out += self.total * self.out_w * self.out_h * (0.6 if self.config.exports.png_bit_depth == 8 else 1.6)
        if self.store_writes:
            need_work += max(0, self.total - self.cached_end) * (self.dw * self.dh) * 0.05
        need_out += float(sum(a.bit_rate or 192_000 for a in meta.audio)) / 8.0 * duration * len(
            [s for s in self.sinks if s.with_audio]
        )
        margin = 64 * 1024 * 1024
        out_dir, work = self.output.parent, self.work_dir
        try:
            same_volume = out_dir.resolve().anchor == work.resolve().anchor
        except OSError:
            same_volume = True
        checks = [(out_dir, need_out + need_work + margin)] if same_volume else [
            (out_dir, need_out + margin), (work, need_work + margin)
        ]
        from ..core.errors import DiskSpaceError

        for path, needed in checks:
            free = free_bytes(path)
            if free < needed:
                raise DiskSpaceError(
                    f"Not enough free disk space: about {human_bytes(needed)} are needed in {path} "
                    f"but only {human_bytes(free)} are free.",
                    details={"path": str(path), "needed_bytes": int(needed), "free_bytes": int(free)},
                )
            if free < 1.5 * needed:
                self._warn(f"Disk space is tight: about {human_bytes(needed)} needed, {human_bytes(free)} free in {path}.")

    # ---------------------------------------------------------------------------- tensors
    def _to_device(self, batch: Sequence[DecodedFrame]) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Upload a batch: native planes, R'G'B' in coded orientation and in display orientation."""

        arrays = [np.stack([f.planes[i] for f in batch]) for i in range(self.fmt.color_planes)]
        planes = planes_from_numpy(arrays, self.device)
        rgb_coded = self.converter.planes_to_rgb(planes)
        rgb_display = torch_ops.rotate_to_display(rgb_coded, self.rotation)
        if self.rotation:
            rgb_display = rgb_display.contiguous()
        return planes, rgb_coded, rgb_display

    def _quantize(self, alpha: torch.Tensor) -> torch.Tensor:
        """Snap a matte to the precision of the matte cache, so cached and live mattes are identical."""

        return torch.round(alpha.float().clamp(0.0, 1.0) * self.matte_max) / self.matte_max

    def _matte_to_numpy(self, matte: torch.Tensor) -> np.ndarray:
        scaled = torch.round(matte * self.matte_max)
        if self.matte_bits == 8:
            return scaled.to(torch.uint8).cpu().numpy()
        return scaled.to(torch.int32).cpu().numpy().astype(np.uint16)

    def _new_tracker(self, seed: Optional[ResolvedSubject]) -> Optional[ComponentSubjectTracker]:
        if not self.isolate:
            return None
        tracker = ComponentSubjectTracker(self.dw, self.dh)
        if seed is not None and (seed.mask is not None or seed.box is not None or seed.points):
            tracker.reset(seed_mask=seed.mask, seed_box=seed.box, seed_points=seed.points)
        else:
            tracker.reset()
        return tracker

    def _sequence_info(self, start: int, reverse: bool, subject: Optional[ResolvedSubject], frames: int) -> SequenceInfo:
        return SequenceInfo(
            width=self.dw, height=self.dh, fps=self.fps, num_frames=frames, start_index=start, reverse=reverse,
            subject=subject, inference_max_side=QUALITY_PRESETS[self.config.quality].inference_max_side,
        )

    def _run_model(self, frames: torch.Tensor, indices: Sequence[int]) -> Tuple[torch.Tensor, List[Optional[float]]]:
        t0 = time.perf_counter()
        outputs = process_with_recovery(self.model, frames, indices, self.controller, self.device, on_reduce=self._warn)
        if self.device_type == "cuda":
            torch.cuda.synchronize(self.device)
        self.reporter.note_model_time(time.perf_counter() - t0, len(indices))
        self.model_frames += len(indices)
        alpha = torch.cat([o.alpha for o in outputs], dim=0).float()
        presence: List[Optional[float]] = []
        for o in outputs:
            n = int(o.alpha.shape[0])
            presence.extend(list(o.presence) if o.presence is not None else [None] * n)
        self.reporter.set_static(batch_size=self.controller.size)
        return alpha, presence

    # ---------------------------------------------------------------------------- backtrack
    def _backtrack(self) -> None:
        """Track the subject from the reference frame back to frame 0 and cache those mattes."""

        if self.seq_start is None or self.ref <= 0 or self.cached_end > self.ref:
            return
        if self.store.covers(0, self.ref):
            return
        caps = self.model.info.capabilities
        if not self.config.subject.track_backward:
            self._warn(f"Frames 0-{self.ref - 1} precede the subject's reference frame and backward tracking is off: they are left empty.")
            return
        if not caps.supports_reverse:
            self._warn(f"Model '{self.model.info.id}' cannot track backwards: frames 0-{self.ref - 1} are left empty.")
            return
        weight = 88.0 * self.ref / max(1, self.total)
        self.reporter.set_stage_weight(Stage.BACKTRACK, weight)
        self.reporter.stage(Stage.BACKTRACK, "Tracking the subject back to the first frame", total_frames=self.ref + 1)
        with self._timed("backtrack"):
            try:
                frames_iter = self.reader.frames_reverse(self.ref + 1, 0)
            except VmtError as exc:
                self._warn(f"Backward tracking is not possible for this file ({exc.message}); frames 0-{self.ref - 1} are left empty.")
                return
            self.model.start_sequence(self._sequence_info(self.ref, True, self.subject, self.ref + 1))
            tracker = self._new_tracker(self.subject)
            feeder = _FrameFeeder(frames_iter, self.loop_batch, 2)
            feeder.start()
            done = 0
            try:
                for batch in feeder:
                    self.cancel.raise_if_cancelled()
                    indices = [f.index for f in batch]
                    _, _, rgb_display = self._to_device(batch)
                    alpha, _ = self._run_model(rgb_display, indices)
                    if tracker is not None:
                        alpha = tracker.apply(alpha, indices)
                    q = self._quantize(alpha)
                    for k, idx in enumerate(indices):
                        if idx < self.ref:
                            self.matte_writer.put(idx, self._matte_to_numpy(q[k, 0]))
                    done += len(batch)
                    self.reporter.advance(done)
            finally:
                feeder.stop()
                self.model.end_sequence()
            self.matte_writer.drain()
            self.store.flush()

    # ---------------------------------------------------------------------------- render
    def _start_sinks(self) -> None:
        for sink in self.sinks:
            sink.encoder = NutPipeEncoder(
                self.tools, sink.intermediate, sink.plan, self.meta.video.time_base,
                queue_frames=self.settings.encode_queue_frames, name=sink.name,
            )
            sink.encoder.start()

    def _mattes(self, indices: List[int], rgb_display: torch.Tensor) -> Tuple[torch.Tensor, List[Optional[float]]]:
        """Raw mattes (display orientation, quantised) for consecutive frames: cache first, model otherwise."""

        n = len(indices)
        alpha = torch.zeros((n, 1, self.dh, self.dw), device=self.device, dtype=torch.float32)
        presence: List[Optional[float]] = [None] * n
        first = None
        if self.seq_start is not None:
            first = next((k for k, i in enumerate(indices) if i >= self.seq_start), None)
        if first is not None:
            idxs = indices[first:]
            if not self.sequence_started:
                self.model.start_sequence(
                    self._sequence_info(idxs[0], False, self.seq_subject, self.total - idxs[0])
                )
                self.tracker = self._new_tracker(self.tracker_seed)
                self.sequence_started = True
            with self.clock("model"):
                a, pres = self._run_model(rgb_display[first:], idxs)
            if self.tracker is not None:
                with self.clock("isolate"):
                    a = self.tracker.apply(a, idxs)
            q = self._quantize(a)
            alpha[first:] = q
            presence[first:] = pres
            if self.store_writes:
                with self.clock("matte_cache"):
                    for j, idx in enumerate(idxs):
                        if idx >= self.cached_end:
                            self.matte_writer.put(idx, self._matte_to_numpy(q[j, 0]))
        # Frames already in the cache always use the cached matte (identical to the earlier run);
        # frames before the model's start that were never inferred stay empty.
        for k, idx in enumerate(indices):
            from_cache = idx < self.cached_end or (self.seq_start is not None and idx < self.seq_start)
            if from_cache and self.store.has(idx):
                matte = self.store.get(idx).astype(np.float32) / self.matte_max
                alpha[k, 0] = torch.from_numpy(matte).to(self.device)
                if idx < self.cached_end:
                    self.frames_from_cache += 1
        return alpha, presence

    def _capture_sample(self, frame: DecodedFrame, alpha_coded: torch.Tensor) -> None:
        """Remember what the output must look like on this frame (for pixel-level verification)."""

        margin = max(4.0, 12.0 * min(self.meta.video.width, self.meta.video.height) / 1080.0)
        a = alpha_coded[None]
        background = (torch_ops.max_filter(a, margin) <= 0.0)[0, 0].cpu().numpy()
        core = (torch_ops.min_filter(a, margin) >= 1.0)[0, 0].cpu().numpy()
        self.samples.append(
            FrameSample(
                index=len(self.written_pts), pts=frame.pts, source_planes=[p.copy() for p in frame.planes],
                background_mask=background, core_mask=core,
            )
        )

    def _render(self) -> None:
        cfg = self.config
        self.reporter.stage(Stage.RENDER, "Matting and compositing", total_frames=self.total)
        targets = [s for s in self.sinks if s.target is not None]
        want_png = self.png is not None
        with self._timed("render"):
            self._start_sinks()
            if self.png is not None:
                self.png.start()
            self.feeder = _FrameFeeder(
                self.reader.frames(0, self.total), self.loop_batch,
                max(1, self.settings.decode_queue_frames // max(1, self.loop_batch)),
            )
            self.feeder.start()
            done = 0
            clock = self.clock
            batches = iter(self.feeder)
            while True:
                with clock("decode_wait"):
                    batch = next(batches, None)
                if batch is None:
                    break
                self.cancel.raise_if_cancelled()
                indices = [f.index for f in batch]
                with clock("upload"):
                    planes, rgb_coded, rgb_display = self._to_device(batch)
                raw, presence = self._mattes(indices, rgb_display)
                with clock("refine"):
                    # One call per batch: the refiner keeps its temporal state across calls and
                    # produces the same result as frame-by-frame processing.
                    refined = self.refiner.refine(raw, rgb_display, indices)
                with clock("monitor"):
                    complete = all(p is not None for p in presence)
                    self.track_monitor.update(refined, indices, presence if complete else None)
                    alpha_coded = torch_ops.rotate_to_coded(refined, self.rotation)
                    if self.rotation:
                        alpha_coded = alpha_coded.contiguous()

                def composite(start: int, stop: int) -> Dict[str, Any]:
                    with clock("composite"):
                        ctx = self.compositor.begin(
                            [p[start:stop] for p in planes], alpha_coded[start:stop], rgb_coded[start:stop]
                        )
                        rendered = {sink.name: self.compositor.render(ctx, sink.target) for sink in targets}
                        rgba = None
                        if want_png:
                            rgba = self.compositor.render_rgba(ctx, cfg.exports.png_bit_depth, cfg.exports.alpha_mode)
                    with clock("download"):
                        out: Dict[str, Any] = {
                            sink.name: planes_to_numpy(rendered[sink.name], sink.target.fmt) for sink in targets
                        }
                        if rgba is not None:
                            arr = rgba.cpu().numpy()
                            out["__png__"] = arr.astype(np.uint16) if cfg.exports.png_bit_depth == 16 else arr
                    return out

                chunks = chunked_with_recovery(composite, len(batch), self.comp_chunk, self.device)
                atlas_np = None
                if self.preview_sink is not None or (self.live is not None and self.live.wants_frame()):
                    with clock("preview"):
                        atlas = preview_atlas(rgb_display, refined, self.background, self.tile)
                        if self.preview_sink is not None:
                            atlas_planes = planes_to_numpy(rgb8_to_yuv420p(atlas), self.preview_sink.plan.fmt)
                        if self.live is not None and self.live.wants_frame():
                            atlas_np = atlas[-1].cpu().numpy()

                k = 0
                for chunk in chunks:
                    count = len(next(iter(chunk.values()))[0]) if chunk else 0
                    for j in range(count):
                        frame = batch[k]
                        if frame.index in self.sample_indices:
                            with clock("verify_samples"):
                                self._capture_sample(frame, alpha_coded[k])
                        with clock("encode_queue"):
                            for sink in targets:
                                sink.encoder.write([plane[j] for plane in chunk[sink.name]], frame.pts, frame.duration)
                            if want_png:
                                self.png.write(chunk["__png__"][j], len(self.written_pts))
                            if self.preview_sink is not None:
                                self.preview_sink.encoder.write(
                                    [plane[k] for plane in atlas_planes], frame.pts, frame.duration
                                )
                        self.written_pts.append(frame.pts)
                        k += 1
                if atlas_np is not None:
                    self.live.update(atlas_np, indices[-1])
                done += len(batch)
                self.reporter.advance(done)

            self.reporter.advance(done, message="Finishing the encoders", force=True)
            if self.sequence_started:
                # Facts that are only known once frames were processed (working resolution, the
                # configuration left after an out-of-memory reduction) belong in the report.
                try:
                    self.report.model["runtime"] = self.model.runtime_summary()
                except Exception as exc:  # pragma: no cover - the summary is informational
                    log.debug("runtime summary failed: %s", exc)
                self.model.end_sequence()
            with clock("encoder_flush"):
                for sink in self.sinks:
                    sink.encoder.finish()
                if self.png is not None:
                    self.png.finish()
            self.matte_writer.close()
            self.matte_writer = None

        skipped = list(getattr(self.reader, "skipped_frames", []) or [])
        if skipped:
            self._warn(
                f"{len(skipped)} frame(s) could not be decoded and are missing from the output "
                f"(frame indices: {_compact(skipped)})."
            )
        for text in getattr(self.reader, "warnings", []) or []:
            self._warn(text)
        if not self.written_pts:
            raise CorruptedVideoError("No frame could be decoded from the input.")
        if len(self.written_pts) != self.total and not skipped:
            self._warn(
                f"The container announced {self.total} frames but {len(self.written_pts)} were decoded; "
                "the output contains every decoded frame."
            )

    # ---------------------------------------------------------------------------- mux
    def _audio_end(self) -> Optional[Fraction]:
        """End time for audio when only part of the video is processed (max_frames)."""

        if not self.truncated or not self.written_pts:
            return None
        tl = self.probe.timeline
        last = min(len(self.written_pts) - 1, len(tl) - 1)
        end_ticks = int(self.written_pts[-1]) + max(1, tl.frame_duration(last))
        return Fraction(end_ticks) * self.meta.video.time_base

    def _mux(self) -> None:
        self.reporter.stage(Stage.MUX, "Adding audio and metadata")
        with self._timed("mux"):
            audio_end = self._audio_end()
            for sink in self.sinks:
                self.cancel.raise_if_cancelled()
                staged = sink.final.with_name(f"{sink.final.stem}.partial{sink.final.suffix}")
                staged.parent.mkdir(parents=True, exist_ok=True)
                result = mux_output(
                    self.tools,
                    video_only=sink.intermediate,
                    source=self.meta,
                    output=staged,
                    container=sink.plan.container,
                    encoding=self.config.encoding,
                    include_audio=sink.with_audio and self.meta.has_audio,
                    audio_mode=sink.audio_mode,
                    rotation=None if sink.keep_rotation else 0,      # None = copy the source's display matrix
                    copy_metadata=sink.name != "preview",
                    audio_end=audio_end,
                    video_shift=Fraction(sink.encoder.pts_shift) * self.meta.video.time_base,
                    keep_aspect=sink.keep_rotation and not self.resized,
                    cancel=self.cancel,
                )
                sink.staged = staged
                sink.mux_info = {
                    "audio_mode": result.audio_mode, "audio_streams": result.audio_streams, "command": result.command,
                    "time_shift": result.time_shift,
                }
                for text in result.warnings:
                    self._warn(f"{sink.label}: {text}" if sink.name != "main" else text)
                safe_unlink(sink.intermediate)
            self.report.encoding["audio"] = self.main_sink.mux_info.get("audio_mode", "none")
            self.report.encoding["mux_command"] = self.main_sink.mux_info.get("command")

    # ---------------------------------------------------------------------------- verify
    def _verify(self) -> None:
        self.reporter.stage(Stage.VERIFY, "Verifying the output", total_frames=100)
        with self._timed("verify"):
            main = self.main_sink
            data = VerifyInput(
                source=self.meta,
                source_pts=np.asarray(self.written_pts, dtype=np.int64),
                source_time_base=self.meta.video.time_base,
                output_path=main.staged,
                codec=self.codec,
                container=self.container,
                src_fmt=self.fmt,
                out_fmt=main.plan.fmt,
                resolved_color=self.color,
                width=self.out_w,
                height=self.out_h,
                resized=self.resized,
                background=self.background,
                lossless=main.plan.lossless,
                audio_mode=main.mux_info.get("audio_mode", "none"),
                expect_audio_streams=main.mux_info.get("audio_streams", 0),
                truncated=self.truncated,
                samples=self.samples,
                refine_stats=self.refiner.statistics(),
                tracking_report=self.track_monitor.report(),
                time_shift=main.mux_info.get("time_shift", Fraction(0)),
            )
            result = verify_output(
                data, self.tools, cancel=self.cancel,
                progress=lambda fraction: self.reporter.advance(int(fraction * 100), force=False),
            )
            self.report.checks = list(result.checks)
            self.report.comparison = list(result.comparison)
            self.report.output = result.output.to_dict() if result.output is not None else None
            failed = [c for c in result.checks if c.status.value == "fail"]
            if result.critical:
                names = ", ".join(c.name for c in failed) or "integrity"
                raise VerificationError(
                    f"The encoded file failed verification ({names}) and was not delivered.",
                    hint="See the report for details. Try another codec / container, or run with --verbose.",
                    details={"checks": [c.model_dump(mode="json") for c in failed]},
                )
            if failed and self.config.strict_verification:
                rejected = self.output.with_name(f"{self.output.stem}.rejected{self.output.suffix}")
                safe_unlink(rejected)
                atomic_replace(main.staged, rejected)
                main.staged = None
                raise VerificationError(
                    f"Quality verification failed ({', '.join(c.name for c in failed)}); strict verification is on. "
                    f"The file was kept as {rejected.name} for inspection.",
                    details={"rejected_file": str(rejected), "checks": [c.model_dump(mode="json") for c in failed]},
                )

    # ---------------------------------------------------------------------------- deliver
    def _deliver(self) -> None:
        self.reporter.stage(Stage.EXPORT, "Writing the deliverables")
        with self._timed("export"):
            frames = len(self.written_pts)
            for sink in self.sinks:
                if sink.staged is None:
                    continue
                atomic_replace(sink.staged, sink.final)
                sink.staged = None
                self.delivered.append(sink.final)
                if sink.name == "preview" and not self.config.exports.matte_preview:
                    continue        # internal preview for the UI, not a requested deliverable
                self.report.outputs.append(
                    OutputFile(
                        kind="matte_preview" if sink.name == "preview" else sink.name, label=sink.label,
                        path=str(sink.final), size_bytes=sink.final.stat().st_size, codec=sink.plan.codec.value,
                        container=sink.plan.container, frames=frames,
                    )
                )
            if self.png is not None and self.png_dir is not None:
                safe_rmtree(self.png_dir)
                atomic_replace(self.png_stage, self.png_dir)
                self.delivered.append(self.png_dir)
                size = sum(p.stat().st_size for p in self.png_dir.glob("*.png"))
                self.report.outputs.append(
                    OutputFile(kind="png_sequence", label="PNG sequence (RGBA)", path=str(self.png_dir),
                               size_bytes=size, codec="png", container="folder", frames=frames)
                )
            if self.config.exports.audio:
                if self.meta.has_audio:
                    audio = extract_audio(self.tools, self.meta, self.output.with_name(f"{self.output.stem}_audio"),
                                          audio_end=self._audio_end(), cancel=self.cancel)
                    for path in audio:
                        self.delivered.append(path)
                        self.report.outputs.append(
                            OutputFile(kind="audio", label="Original audio (stream copy)", path=str(path),
                                       size_bytes=path.stat().st_size, container=path.suffix.lstrip("."))
                        )
                else:
                    self._warn("Audio export was requested but the source has no audio stream.")

    # ---------------------------------------------------------------------------- report / cleanup
    def _finish_report(self, total_seconds: float) -> None:
        rep = self.report
        t = rep.timing
        t.total_seconds = float(total_seconds)
        t.stages = {k: round(v, 3) for k, v in self.stage_seconds.items()}
        t.frames_processed = len(self.written_pts)
        t.frames_from_cache = int(self.frames_from_cache)
        render = self.stage_seconds.get("render", 0.0)
        if render > 0 and self.written_pts:
            t.average_fps = len(self.written_pts) / render
        if self.reporter.model_seconds > 0 and self.reporter.model_frames > 0:
            t.average_inference_fps = self.reporter.model_frames / self.reporter.model_seconds
        if getattr(self, "clock", None) is not None and self.clock.enabled:
            t.render_breakdown = self.clock.summary(len(self.written_pts) or None)
            if t.render_breakdown:
                log.debug("Render stage breakdown: %s", t.render_breakdown)
        duration = float(self.meta.video_duration or 0.0) if hasattr(self, "meta") else 0.0
        if duration > 0 and total_seconds > 0 and self.written_pts and hasattr(self, "total"):
            t.realtime_factor = duration * (len(self.written_pts) / max(1, self.meta.frame_count)) / total_seconds
        hw = rep.hardware
        if self.monitor is not None:
            hw.vram_peak_allocated_mb = self.monitor.peak_vram_allocated_mb()
            self.monitor.stats()
            hw.ram_peak_process_mb = self.monitor.peak_ram_mb or None
        if hasattr(self, "controller"):
            hw.final_batch_size = self.controller.size
            hw.batch_reductions = self.controller.reductions + self.controller.memory_reductions
        if hasattr(self, "refiner"):
            stats: Dict[str, Any] = {"refinement": self.refiner.statistics(), "tracking": self.track_monitor.report()}
            if self.tracker is not None:
                stats["isolation_events"] = self.tracker.events()
            rep.matte_stats = stats
            for text in stats["tracking"].get("warnings", []) or []:
                rep.add_warning(text)
        if rep.status not in ("failed", "cancelled"):
            rep.finalize_status()
        if self.save_report:
            path = self.report_path or self.output.with_name(f"{self.output.stem}.report.json")
            try:
                rep.outputs = [o for o in rep.outputs if o.kind != "report"]
                rep.outputs.append(OutputFile(kind="report", label="Processing report", path=str(path)))
                rep.save(path)
                rep.outputs[-1].size_bytes = path.stat().st_size
            except OSError as exc:
                log.warning("Could not write the report to %s: %s", path, exc)

    def _cleanup(self, failed: bool) -> None:
        """Release every resource; on failure remove partial files (but keep the matte cache)."""

        if self.feeder is not None:
            self.feeder.stop()
        for sink in self.sinks:
            try:
                if sink.encoder is not None and failed:
                    sink.encoder.abort()
            except Exception as exc:  # pragma: no cover - best effort
                log.debug("encoder abort failed: %s", exc)
            if failed:
                safe_unlink(sink.intermediate)
                if sink.staged is not None:
                    safe_unlink(sink.staged)
        if self.png is not None and failed:
            try:
                self.png.abort()
            except Exception as exc:  # pragma: no cover
                log.debug("png abort failed: %s", exc)
            if self.png_stage is not None:
                safe_rmtree(self.png_stage)
        if self.matte_writer is not None:
            try:
                self.matte_writer.close(flush=True)       # keep what was inferred: it makes resume possible
            except Exception as exc:
                log.debug("matte writer close failed: %s", exc)
            self.matte_writer = None
        if self.store is not None:
            try:
                self.store.close()
            except Exception as exc:
                log.debug("matte store close failed: %s", exc)
        if self.model is not None:
            try:
                if self.model.loaded:
                    self.model.end_sequence()
                self.model.unload()
            except Exception as exc:  # pragma: no cover
                log.debug("model unload failed: %s", exc)
        if self.reader is not None:
            try:
                self.reader.close()
            except Exception as exc:  # pragma: no cover
                log.debug("reader close failed: %s", exc)
        release_device_memory(getattr(self, "device", None))
        if self.work_dir is not None:
            keep = self.config.keep_work_dir or self.user_work_dir is not None
            if not failed and not keep:
                safe_rmtree(self.work_dir)
                parent = self.work_dir.parent
                try:
                    if parent.name == ".vmt_work" and not any(parent.iterdir()):
                        parent.rmdir()
                except OSError:
                    pass
            elif failed and not (self.config.resume or self.force_resume) and not keep:
                safe_rmtree(self.work_dir)


# =========================================================================== small helpers


def _json(data: Any) -> str:
    import json

    return json.dumps(data, indent=1, default=str)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _compact(indices: Sequence[int], limit: int = 12) -> str:
    """``[1, 2, 3, 7, 9, 10]`` -> ``"1-3, 7, 9-10"`` (truncated after *limit* ranges)."""

    if not indices:
        return ""
    ordered = sorted(set(int(i) for i in indices))
    ranges: List[str] = []
    start = prev = ordered[0]
    for i in ordered[1:] + [None]:  # type: ignore[list-item]
        if i is not None and i == prev + 1:
            prev = i
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        if i is not None:
            start = prev = i
    if len(ranges) > limit:
        return ", ".join(ranges[:limit]) + f", ... ({len(ordered)} in total)"
    return ", ".join(ranges)
