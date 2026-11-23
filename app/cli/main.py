"""Command line interface.

    python main.py process --input in.mp4 --output out_green.mp4 --model rvm --background green
    python main.py process --input clips/ --output-directory out/ --quality high        (batch)
    python main.py serve                                                                (web UI)
    python main.py models list | info | download | verify | remove
    python main.py probe in.mp4
    python main.py system

Heavy modules (PyTorch, the pipeline) are imported inside the commands so ``--help`` stays instant.
Exit codes: 0 = success, 1 = error, 2 = finished but a quality check failed, 130 = cancelled.
"""

from __future__ import annotations

import glob
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .. import __version__
from ..core.config import (
    CODECS,
    CONTAINER_EXTENSIONS,
    QUALITY_PRESETS,
    REFINEMENT_PRESETS,
    SUPPORTED_INPUT_EXTENSIONS,
    AppSettings,
    BackgroundMode,
    EncodingConfig,
    ExportOptions,
    ProcessingConfig,
    QualityPreset,
    RefinementConfig,
    VideoCodec,
    load_settings,
    parse_subject_spec,
)
from ..core.errors import ConfigError, JobCancelledError, VmtError
from ..core.events import CancelToken, ProgressEvent, Stage
from ..core.log import make_stdio_safe, setup_logging

EXIT_OK, EXIT_ERROR, EXIT_QC, EXIT_CANCELLED = 0, 1, 2, 130

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode=None,
    help="AI video matting and green-screen conversion that preserves resolution, timing, audio and pixels.",
)
models_app = typer.Typer(no_args_is_help=True, help="Inspect, download and verify model weights (downloads are always explicit).")
app.add_typer(models_app, name="models")

# soft_wrap: never hard-wrap long lines (paths, error messages) at the terminal width; when the
# output is redirected that width is an arbitrary 80 columns and would break lines mid-sentence.
console = Console(highlight=False, soft_wrap=True)
err_console = Console(stderr=True, highlight=False, soft_wrap=True)

_state: Dict[str, Any] = {"config": None, "verbose": False, "quiet": False, "log_file": None}


# =========================================================================== shared helpers


def _settings(**overrides: Any) -> AppSettings:
    return load_settings(_state["config"], **overrides)


def _init_logging(settings: AppSettings) -> None:
    level = "DEBUG" if _state["verbose"] else ("ERROR" if _state["quiet"] else "WARNING")
    setup_logging(level, _state["log_file"] or settings.log_file)


def _fail(exc: BaseException) -> "typer.Exit":
    """Print an application error in a consistent, readable form and return the exit exception."""

    if isinstance(exc, JobCancelledError):
        err_console.print("Cancelled.")
        return typer.Exit(EXIT_CANCELLED)
    if isinstance(exc, VmtError):
        err_console.print(f"ERROR [{exc.code}]: {exc.message}", style="bold red", markup=False)
        if exc.hint:
            err_console.print(f"  hint: {exc.hint}", markup=False)
        tail = exc.details.get("stderr_tail") if isinstance(exc.details, dict) else None
        if tail and _state["verbose"]:
            err_console.print(str(tail), markup=False)
        return typer.Exit(EXIT_ERROR)
    err_console.print(f"ERROR: {type(exc).__name__}: {exc}", style="bold red", markup=False)
    return typer.Exit(EXIT_ERROR)


def _mb(value: Optional[float]) -> str:
    return "-" if value is None else f"{value / 1024.0:.1f} GB" if value >= 1024 else f"{value:.0f} MB"


def _bytes(n: Optional[float]) -> str:
    if not n:
        return "-"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return "-"


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"video-matting-tool {__version__}")
        raise typer.Exit(EXIT_OK)


@app.callback()
def _main(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Settings file (YAML). Default: configs/default.yaml, then VMT_CONFIG."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging (debug level, FFmpeg command lines, stack traces)."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print errors and the final result."),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Also write the log to this file."),
    version: Optional[bool] = typer.Option(None, "--version", callback=_print_version, is_eager=True, help="Print the version and exit."),
) -> None:
    make_stdio_safe()
    _state.update(config=config, verbose=verbose, quiet=quiet, log_file=log_file)


# =========================================================================== process


def _expand_inputs(items: List[str], recursive: bool) -> List[Path]:
    """Resolve files, folders and glob patterns to a de-duplicated, ordered list of video files."""

    found: List[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            matches = sorted(q for q in p.glob(pattern) if q.is_file() and q.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS)
            if not matches:
                raise ConfigError(f"No supported video files found in folder {p}.",
                                  hint=f"Supported extensions: {', '.join(SUPPORTED_INPUT_EXTENSIONS)}")
            found.extend(matches)
        elif any(ch in item for ch in "*?["):
            matches = sorted(Path(m) for m in glob.glob(item, recursive=recursive) if Path(m).is_file())
            if not matches:
                raise ConfigError(f"The pattern {item!r} matches no files.")
            found.extend(matches)
        else:
            found.append(p)          # existence is checked by the pipeline, with a proper error
    unique: List[Path] = []
    seen = set()
    for p in found:
        key = str(p.resolve()).lower() if os.name == "nt" else str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _parse_model_options(pairs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"Invalid --model-option {pair!r}; expected key=value.")
        key, value = pair.split("=", 1)
        key = key.strip()
        text = value.strip()
        try:
            out[key] = json.loads(text)
        except ValueError:
            out[key] = text
    return out


def _default_extension(cfg: ProcessingConfig) -> str:
    if cfg.encoding.container:
        return CONTAINER_EXTENSIONS[cfg.encoding.container]
    if cfg.encoding.codec is not None:
        return CONTAINER_EXTENSIONS[CODECS[cfg.encoding.codec].default_container]
    return ".mov" if cfg.background is BackgroundMode.TRANSPARENT else ".mp4"


class _ProgressView:
    """Renders pipeline progress events as a live progress bar, or as JSON lines for automation."""

    def __init__(self, label: str, json_lines: bool, quiet: bool) -> None:
        self.label = label
        self.json_lines = json_lines
        self.quiet = quiet
        self.progress: Optional[Progress] = None
        self.task = None
        self.stage: Optional[Stage] = None

    def __enter__(self) -> "_ProgressView":
        if not self.json_lines and not self.quiet:
            self.progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                TextColumn("{task.fields[detail]}"),
                TimeElapsedColumn(),
                console=err_console,
                transient=False,
            )
            self.progress.start()
            self.task = self.progress.add_task(self.label, total=100.0, detail="")
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.progress is not None:
            self.progress.stop()

    def __call__(self, event: ProgressEvent) -> None:
        if self.json_lines:
            sys.stdout.write(json.dumps({"file": self.label, **event.to_dict()}) + "\n")
            sys.stdout.flush()
            return
        if self.progress is None or self.task is None:
            return
        parts: List[str] = []
        if event.total_frames:
            parts.append(f"{event.frame}/{event.total_frames}")
        if event.fps:
            parts.append(f"{event.fps:.1f} fps")
        if event.eta_seconds is not None and event.stage is Stage.RENDER:
            eta = int(event.eta_seconds)
            parts.append(f"ETA {eta // 60}:{eta % 60:02d}")
        if event.gpu_utilization is not None:
            parts.append(f"GPU {event.gpu_utilization:.0f}%")
        if event.gpu_memory_used_mb is not None and event.gpu_memory_total_mb:
            parts.append(f"VRAM {event.gpu_memory_used_mb / 1024:.1f}/{event.gpu_memory_total_mb / 1024:.1f} GB")
        if event.batch_size and event.stage is Stage.RENDER:
            parts.append(f"batch {event.batch_size}")
        description = f"{self.label}: {event.message or event.stage.value}"
        self.progress.update(
            self.task,
            completed=float(event.overall_percent or 0.0),
            description=description[:60],
            detail="  ".join(parts),
        )


@app.command()
def process(
    input: List[str] = typer.Option(..., "--input", "-i", help="Input video file, folder or glob pattern. Repeat the option for several inputs."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file (only with a single input). The extension selects the container."),
    output_directory: Optional[Path] = typer.Option(None, "--output-directory", "-d", help="Folder for the outputs of a batch: <input name>_<background>.<ext>."),
    recursive: bool = typer.Option(False, "--recursive", help="Search input folders recursively."),
    # model / device
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model: rvm | sam2 | matanyone | sam2_rvm | sam2_matanyone (see 'models list')."),
    variant: Optional[str] = typer.Option(None, "--variant", help="Model variant (default: chosen by --quality)."),
    quality: Optional[str] = typer.Option(None, "--quality", help="Quality preset: fast | balanced | high | max."),
    device: Optional[str] = typer.Option(None, "--device", help="Inference device: auto | cuda | cuda:N | mps | cpu."),
    precision: Optional[str] = typer.Option(None, "--precision", help="auto | fp32 | fp16."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Frames per inference batch (default: automatic, reduced automatically when memory runs out)."),
    model_option: List[str] = typer.Option([], "--model-option", "-O", help="Adapter specific option key=value (repeatable), e.g. -O downsample_ratio=0.4."),
    # background
    background: Optional[str] = typer.Option(None, "--background", "-b", help="green | blue | black | white | transparent | custom  (default green = #00FF00)."),
    background_color: Optional[str] = typer.Option(None, "--background-color", help="#RRGGBB for --background custom."),
    # subject
    subject: Optional[str] = typer.Option(None, "--subject", help="auto | box:x1,y1,x2,y2 | points:x,y;x,y,- | text:<description> | mask:<image>"),
    subject_frame: Optional[int] = typer.Option(None, "--subject-frame", help="Frame the subject selection refers to (0-based, default 0)."),
    keep_all_subjects: bool = typer.Option(False, "--keep-all-subjects", help="Do not restrict the matte to the primary / selected subject."),
    no_track_backward: bool = typer.Option(False, "--no-track-backward", help="Do not track the subject backwards from --subject-frame to the first frame."),
    # refinement
    mask_refinement: Optional[str] = typer.Option(None, "--mask-refinement", help="Refinement preset: off | light | balanced | strong."),
    edge_feather: Optional[float] = typer.Option(None, "--edge-feather", help="Edge feathering radius in pixels (at 1080p; scaled to the frame size)."),
    temporal_smoothing: Optional[float] = typer.Option(None, "--temporal-smoothing", help="Temporal smoothing strength 0..0.95 (0 = off)."),
    erode: Optional[float] = typer.Option(None, "--erode", help="Shrink the matte by this many pixels."),
    dilate: Optional[float] = typer.Option(None, "--dilate", help="Grow the matte by this many pixels."),
    smooth: Optional[float] = typer.Option(None, "--smooth", help="Gaussian smoothing sigma in pixels."),
    decontaminate: Optional[float] = typer.Option(None, "--decontaminate", help="Edge colour decontamination 0..1 (0 = keep source pixels at edges exactly)."),
    black_point: Optional[float] = typer.Option(None, "--black-point", help="Alpha at or below this value becomes fully transparent."),
    white_point: Optional[float] = typer.Option(None, "--white-point", help="Alpha at or above this value becomes fully opaque."),
    motion_compensation: Optional[bool] = typer.Option(None, "--motion-compensation/--no-motion-compensation", help="Optical-flow compensated temporal smoothing (slower)."),
    # encoding
    output_codec: Optional[str] = typer.Option(None, "--output-codec", help="h264 | h265 | h264_nvenc | hevc_nvenc | prores_422_hq | prores_4444 | ffv1 | vp9 (default: automatic)."),
    crf: Optional[int] = typer.Option(None, "--crf", help="Constant quality value (lower = better; default from --quality)."),
    bitrate: Optional[str] = typer.Option(None, "--bitrate", help="Target video bitrate, e.g. 20M. Overrides --crf."),
    max_bitrate: Optional[str] = typer.Option(None, "--max-bitrate", help="Peak bitrate for --bitrate mode."),
    preset: Optional[str] = typer.Option(None, "--preset", help="Encoder speed preset (x264/x265: ultrafast..veryslow, NVENC: p1..p7)."),
    pix_fmt: Optional[str] = typer.Option(None, "--pix-fmt", help="Output pixel format (default: same as the source where the codec allows)."),
    resolution: Optional[str] = typer.Option(None, "--resolution", help="source (default) | <N>p | WxH. Scaling only happens when you ask for it."),
    preserve_audio: Optional[bool] = typer.Option(None, "--preserve-audio/--no-preserve-audio", help="Keep the original audio (stream copy when the container allows). Default: on."),
    audio_codec: Optional[str] = typer.Option(None, "--audio-codec", help="auto | copy | aac | pcm | opus | flac."),
    # exports
    export_alpha: bool = typer.Option(False, "--export-alpha", help="Also write a grayscale alpha matte video."),
    export_png_sequence: bool = typer.Option(False, "--export-png-sequence", help="Also write an RGBA PNG image sequence."),
    export_prores4444: bool = typer.Option(False, "--export-prores4444", help="Also write a transparent ProRes 4444 .mov."),
    export_green: bool = typer.Option(False, "--export-green", help="Also write a green-screen video (when the main background is not green)."),
    export_audio: bool = typer.Option(False, "--export-audio", help="Also extract the original audio (stream copy)."),
    export_matte_preview: bool = typer.Option(False, "--export-matte-preview", help="Also write a low-resolution 2x2 preview video."),
    # run control
    max_frames: Optional[int] = typer.Option(None, "--max-frames", help="Process only the first N frames (drafts)."),
    resume: Optional[bool] = typer.Option(None, "--resume/--no-resume", help="Reuse cached mattes of an interrupted run. Default: on."),
    keep_cache: bool = typer.Option(False, "--keep-cache", help="Keep the matte cache after success (allows quick re-rendering with other settings)."),
    overwrite: bool = typer.Option(False, "--overwrite", "-y", help="Replace existing output files."),
    download_models: bool = typer.Option(False, "--download-models", help="Download missing model weights for this run."),
    commercial: Optional[bool] = typer.Option(None, "--commercial/--non-commercial", help="Refuse (or allow) models whose licence forbids commercial use."),
    strict: bool = typer.Option(False, "--strict", help="Treat failed quality checks as errors."),
    tolerate_decode_errors: bool = typer.Option(False, "--tolerate-decode-errors", help="Skip undecodable frames instead of aborting."),
    report: Optional[Path] = typer.Option(None, "--report", help="Where to write the JSON report (single input). Default: <output>.report.json."),
    no_report: bool = typer.Option(False, "--no-report", help="Do not write a report file."),
    json_progress: bool = typer.Option(False, "--json-progress", help="Print progress as JSON lines on stdout (for automation)."),
    stop_on_error: bool = typer.Option(False, "--stop-on-error", help="Batch: stop at the first failed file instead of continuing."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging (same as the global option before the command)."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print errors and the batch summary."),
) -> None:
    """Remove the background of one or more videos and composite the subject onto a solid colour."""

    if verbose:
        _state["verbose"] = True
    if quiet:
        _state["quiet"] = True
    try:
        settings = _settings()
        _init_logging(settings)
        inputs = _expand_inputs(input, recursive)
        if not inputs:
            raise ConfigError("No input files.")
        if output is not None and len(inputs) > 1:
            raise ConfigError("--output can only be used with a single input; use --output-directory for batches.")
        if output is None and output_directory is None:
            raise ConfigError("Specify --output (single file) or --output-directory (batch).")
        if report is not None and len(inputs) > 1:
            raise ConfigError("--report can only be used with a single input.")

        base = settings.default_processing_config()
        data = base.model_dump(mode="python")

        def put(section: Optional[str], key: str, value: Any) -> None:
            if value is None:
                return
            (data[section] if section else data)[key] = value

        put(None, "model", model)
        put(None, "variant", variant)
        put(None, "quality", quality)
        put(None, "device", device)
        put(None, "precision", precision)
        put(None, "batch_size", batch_size)
        put(None, "background", background)
        put(None, "background_color", background_color)
        put(None, "max_frames", max_frames)
        put(None, "resume", resume)
        put(None, "commercial_use", commercial)
        if model_option:
            data["model_options"] = {**data.get("model_options", {}), **_parse_model_options(model_option)}
        if keep_cache:
            data["keep_work_dir"] = True
        if overwrite:
            data["overwrite"] = True
        if download_models:
            data["download_models"] = True
        if strict:
            data["strict_verification"] = True
        if tolerate_decode_errors:
            data["tolerate_decode_errors"] = True

        if subject is not None:
            data["subject"] = parse_subject_spec(subject).model_dump(mode="python")
        put("subject", "frame_index", subject_frame)
        if keep_all_subjects:
            data["subject"]["isolate"] = False
        if no_track_backward:
            data["subject"]["track_backward"] = False

        put("refinement", "preset", mask_refinement)
        put("refinement", "edge_feather", edge_feather)
        put("refinement", "temporal_smoothing", temporal_smoothing)
        put("refinement", "erode", erode)
        put("refinement", "dilate", dilate)
        put("refinement", "gaussian_smoothing", smooth)
        put("refinement", "decontaminate", decontaminate)
        put("refinement", "black_point", black_point)
        put("refinement", "white_point", white_point)
        put("refinement", "motion_compensation", motion_compensation)

        put("encoding", "codec", output_codec)
        put("encoding", "crf", crf)
        put("encoding", "bitrate", bitrate)
        put("encoding", "max_bitrate", max_bitrate)
        put("encoding", "preset", preset)
        put("encoding", "pix_fmt", pix_fmt)
        put("encoding", "resolution", resolution)
        put("encoding", "preserve_audio", preserve_audio)
        put("encoding", "audio_codec", audio_codec)

        for flag, key in (
            (export_alpha, "alpha_matte"), (export_png_sequence, "png_sequence"), (export_prores4444, "prores_4444"),
            (export_green, "green_screen"), (export_audio, "audio"), (export_matte_preview, "matte_preview"),
        ):
            if flag:
                data["exports"][key] = True
        try:
            cfg = ProcessingConfig.model_validate(data)
        except Exception as exc:
            raise ConfigError(f"Invalid options: {_validation_text(exc)}") from exc

        jobs: List[Tuple[Path, Path]] = []
        ext = _default_extension(cfg)
        for src in inputs:
            if output is not None:
                dst = output
            else:
                dst = output_directory / f"{src.stem}_{cfg.background.value}{ext}"
            jobs.append((src, dst))
        targets = [str(d.resolve()).lower() for _, d in jobs]
        if len(set(targets)) != len(targets):
            raise ConfigError("Several inputs would be written to the same output file (identical file names).",
                              hint="Process the folders separately or rename the inputs.")
    except VmtError as exc:
        raise _fail(exc)

    from ..processing.pipeline import MattingPipeline  # heavy import: PyTorch

    pipeline = MattingPipeline(settings)
    cancel = CancelToken()

    def on_sigint(signum: int, frame: Any) -> None:
        if cancel.cancelled:
            raise KeyboardInterrupt
        err_console.print("\nStopping after the current batch (Ctrl+C again to force quit) ...")
        cancel.cancel("cancelled by user")

    previous = signal.signal(signal.SIGINT, on_sigint)
    results: List[Dict[str, Any]] = []
    worst = EXIT_OK
    try:
        for number, (src, dst) in enumerate(jobs, 1):
            label = src.name if len(jobs) == 1 else f"[{number}/{len(jobs)}] {src.name}"
            t0 = time.perf_counter()
            entry: Dict[str, Any] = {"input": str(src), "output": str(dst), "status": "failed", "seconds": 0.0, "detail": ""}
            try:
                with _ProgressView(label, json_progress, _state["quiet"]) as view:
                    rep = pipeline.run(
                        src, dst, cfg, progress=view, cancel=cancel,
                        report_path=report if len(jobs) == 1 else None, save_report=not no_report,
                    )
                entry["status"] = rep.status
                failed_checks = rep.failed_checks
                entry["detail"] = (
                    f"{rep.timing.frames_processed} frames, "
                    f"{(rep.timing.average_fps or 0):.1f} fps"
                    + (f", {len(failed_checks)} check(s) FAILED" if failed_checks else "")
                    + (f", {len(rep.warnings)} warning(s)" if rep.warnings else "")
                )
                if failed_checks:
                    worst = max(worst, EXIT_QC)
                if not _state["quiet"] and not json_progress:
                    console.print(rep.to_text(), markup=False)
            except JobCancelledError:
                entry["status"] = "cancelled"
                results.append(entry)
                worst = EXIT_CANCELLED
                break
            except VmtError as exc:
                entry["detail"] = f"[{exc.code}] {exc.message}"
                err_console.print(f"ERROR [{exc.code}] {src.name}: {exc.message}", style="bold red", markup=False)
                if exc.hint:
                    err_console.print(f"  hint: {exc.hint}", markup=False)
                tail = exc.details.get("stderr_tail") if isinstance(exc.details, dict) else None
                if tail and _state["verbose"]:
                    err_console.print(str(tail), markup=False)
                worst = max(worst, EXIT_ERROR) if worst != EXIT_CANCELLED else worst
                if stop_on_error:
                    entry["seconds"] = time.perf_counter() - t0
                    results.append(entry)
                    break
            entry["seconds"] = time.perf_counter() - t0
            results.append(entry)
    finally:
        signal.signal(signal.SIGINT, previous)

    if len(jobs) > 1 and not json_progress:
        table = Table(title="Batch summary", show_lines=False)
        for column in ("Input", "Status", "Time", "Result"):
            table.add_column(column, overflow="fold")
        for e in results:
            table.add_row(Path(e["input"]).name, e["status"], f"{e['seconds']:.1f} s", e["detail"])
        for src, _ in jobs[len(results):]:
            table.add_row(src.name, "not started", "-", "")
        console.print(table)
    if json_progress:
        sys.stdout.write(json.dumps({"summary": results}) + "\n")
    raise typer.Exit(worst)


def _validation_text(exc: Exception) -> str:
    """Condense a pydantic ValidationError into one readable line per problem."""

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


# =========================================================================== serve


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host", help="Interface to bind (default 127.0.0.1: only this computer)."),
    port: Optional[int] = typer.Option(None, "--port", help="Port (default 8765)."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the web interface in the default browser."),
) -> None:
    """Start the local web interface."""

    try:
        settings = _settings(host=host, port=port)
        _init_logging(settings)
    except VmtError as exc:
        raise _fail(exc)
    import uvicorn

    from ..api.server import create_app

    url = f"http://{'127.0.0.1' if settings.host in ('0.0.0.0', '::') else settings.host}:{settings.port}/"
    if settings.host not in ("127.0.0.1", "localhost", "::1"):
        err_console.print(
            f"WARNING: binding to {settings.host} exposes the tool (uploads, files, GPU) to other machines on the network.",
            style="yellow", markup=False,
        )
    console.print(f"Video Matting Tool {__version__} - web interface at {url}  (Ctrl+C to stop)", markup=False)
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(
        create_app(settings), host=settings.host, port=settings.port,
        log_level="debug" if _state["verbose"] else "warning",
    )


# =========================================================================== models


def _model_rows(settings: AppSettings) -> List[Dict[str, Any]]:
    from ..models import registry

    rows = []
    for info in registry.list_model_infos(settings, include_auxiliary=True):
        rows.append({"info": info, "status": registry.model_status(info.id, settings)})
    return rows


@models_app.command("list")
def models_list(json_out: bool = typer.Option(False, "--json", help="Machine readable output.")) -> None:
    """List the available models, their licences and whether they are ready to use."""

    try:
        settings = _settings()
        _init_logging(settings)
        rows = _model_rows(settings)
    except VmtError as exc:
        raise _fail(exc)
    if json_out:
        console.print_json(json.dumps([{**r["info"].to_dict(), "status": r["status"]} for r in rows], default=str))
        return
    table = Table(title="Matting models (--model)")
    helpers = Table(title="Helper models (subject detection, text prompts)")
    for column in ("Id", "Name", "Output", "Subjects", "Licence", "Commercial use", "Package", "Weights"):
        table.add_column(column, overflow="fold")
    for column in ("Id", "Name", "Licence", "Commercial use", "Package", "Weights"):
        helpers.add_column(column, overflow="fold")
    for r in rows:
        info, st = r["info"], r["status"]
        if info.kind == "auxiliary":
            ready = [v for v, s in st["variants"].items() if s["ready"]]
            helpers.add_row(
                info.id, info.name, info.license.name, info.license.commercial_use,
                "installed" if st["installed"] else "missing: " + ", ".join(st["missing_requirements"] or ["dependency"]),
                ", ".join(ready) if ready else "not downloaded",
            )
            continue
        ready = [v for v, s in st["variants"].items() if s["ready"]]
        weights = ", ".join(ready) if ready else ("not downloaded" if info.variants and any(v.weights for v in info.variants) else "-")
        if info.depends_on:
            weights = "ready" if st["ready"] else "needs: " + ", ".join(info.depends_on)
        table.add_row(
            info.id, info.name,
            "true alpha" if info.capabilities.alpha_type == "alpha" else "mask (refined to alpha)",
            "people" if info.capabilities.subject_domain == "human" else "anything",
            info.license.name, info.license.commercial_use,
            "installed" if st["installed"] else "missing: " + ", ".join(st["missing_requirements"] or ["dependency"]),
            weights,
        )
    console.print(table)
    if helpers.row_count:
        console.print(helpers)
    console.print("Details: python main.py models info <id>     Download: python main.py models download <id>", markup=False)


@models_app.command("info")
def models_info(model_id: str = typer.Argument(..., help="Model id, e.g. rvm")) -> None:
    """Show licence, limitations, variants and weight files of a model."""

    try:
        settings = _settings()
        _init_logging(settings)
        from ..models import registry

        info = registry.get_model_info(model_id, settings)
        status = registry.model_status(model_id, settings)
    except VmtError as exc:
        raise _fail(exc)
    p = lambda text="": console.print(text, markup=False)   # noqa: E731
    p(f"{info.name}  [{info.id}]")
    p(info.description)
    p(f"Homepage : {info.homepage}")
    p(f"Licence  : {info.license.name}  ({info.license.url})")
    p(f"Commercial use: {info.license.commercial_use}")
    if info.license.notes:
        p(f"  {info.license.notes}")
    caps = info.capabilities
    if info.kind == "auxiliary":
        p("Role     : helper model for subject selection (not selectable with --model)")
    else:
        p(f"Output   : {'true soft alpha' if caps.alpha_type == 'alpha' else 'segmentation mask, refined to a soft matte'}")
        p(f"Subjects : {'people only' if caps.subject_domain == 'human' else 'any object'}; "
          f"prompts: {', '.join(sorted(caps.native_prompts)) or 'none (automatic)'}"
          f"{'; needs a first-frame mask' if caps.needs_initial_mask else ''}")
    p(f"Package  : {'installed' if status['installed'] else 'NOT installed - ' + (info.install_hint or '')}")
    p("Variants :")
    for v in info.variants:
        st = status["variants"].get(v.id, {})
        mark = "ready" if st.get("ready") else "not downloaded"
        default = " (default)" if v.id == info.default_variant else ""
        p(f"  - {v.id}{default}: {v.label} [{mark}]" + (f", about {v.vram_hint_mb} MB VRAM at 1080p" if v.vram_hint_mb else ""))
        for w in v.weights:
            p(f"      {w.filename}  {_bytes(w.size_bytes)}  {w.url}")
            if w.sha256:
                p(f"      sha256 {w.sha256}")
    if info.quality_variants:
        p("Quality presets: " + ", ".join(f"{k} -> {v}" for k, v in info.quality_variants.items()))
    if info.limitations:
        p("Limitations:")
        for text in info.limitations:
            p(f"  - {text}")
    if info.citation:
        p(f"Citation : {info.citation}")


@models_app.command("download")
def models_download(
    model_id: str = typer.Argument(..., help="Model id, e.g. rvm"),
    variant: Optional[str] = typer.Option(None, "--variant", help="Variant to download (default: the model's default variant)."),
    all_variants: bool = typer.Option(False, "--all", help="Download every variant."),
    accept_license: bool = typer.Option(False, "--accept-license", help="Confirm that you have read and accept the model's licence terms."),
) -> None:
    """Download and verify model weights from their official source."""

    try:
        settings = _settings()
        _init_logging(settings)
        from ..models import registry

        info = registry.get_model_info(model_id, settings)
        targets: List[Tuple[Any, Any]] = []          # (ModelInfo, ModelVariant)
        variants = list(info.variants) if all_variants else [info.variant(variant)]
        for v in variants:
            targets.append((info, v))
        for dep in info.depends_on:
            dep_info = registry.get_model_info(dep, settings)
            dep_variant = dep_info.variant(None)
            if (dep_info, dep_variant) not in targets:
                targets.append((dep_info, dep_variant))
        licences = {t[0].id: t[0].license for t in targets}
        restricted = [f"{mid}: {lic.name} ({lic.commercial_use})" for mid, lic in licences.items() if lic.commercial_use != "allowed"]
        if restricted and not accept_license:
            err_console.print("The following licences have conditions you must accept before downloading:", markup=False)
            for mid, lic in licences.items():
                if lic.commercial_use != "allowed":
                    err_console.print(f"  {mid}: {lic.name} - commercial use {lic.commercial_use}. {lic.notes}", markup=False)
                    err_console.print(f"     {lic.url}", markup=False)
            err_console.print("Re-run with --accept-license to confirm.", markup=False)
            raise typer.Exit(EXIT_ERROR)
        wm = registry.weights_manager(settings)
        cancel = CancelToken()
        previous = signal.signal(signal.SIGINT, lambda *_: cancel.cancel())
        try:
            for target_info, target_variant in targets:
                if not target_variant.weights:
                    continue
                with Progress(TextColumn("{task.description}"), BarColumn(bar_width=None), TaskProgressColumn(),
                              TextColumn("{task.fields[detail]}"), console=err_console) as bar:
                    task = bar.add_task(f"{target_info.id}/{target_variant.id}", total=100.0, detail="")

                    def on_progress(done: int, total: Optional[int], name: str) -> None:
                        pct = 100.0 * done / total if total else 0.0
                        bar.update(task, completed=pct, detail=f"{name}  {_bytes(done)} / {_bytes(total)}")

                    paths = wm.ensure(target_info, target_variant, download=True, progress=on_progress, cancel=cancel)
                    bar.update(task, completed=100.0)
                for path in paths:
                    console.print(f"  ready: {path}", markup=False)
        finally:
            signal.signal(signal.SIGINT, previous)
        missing = registry.missing_modules(info)
        if missing:
            console.print(f"Note: the Python package(s) {', '.join(missing)} are not installed yet. {info.install_hint}", markup=False)
    except VmtError as exc:
        raise _fail(exc)


@models_app.command("verify")
def models_verify(model_id: Optional[str] = typer.Argument(None, help="Model id (default: all models)")) -> None:
    """Re-check the SHA-256 of every downloaded weight file."""

    try:
        settings = _settings()
        _init_logging(settings)
        from ..models import registry

        ids = [model_id] if model_id else [i.id for i in registry.list_model_infos(settings, include_auxiliary=True)]
        wm = registry.weights_manager(settings)
        bad = 0
        for mid in ids:
            info = registry.get_model_info(mid, settings)
            for v in info.variants:
                for entry in wm.status(info.id, v, full=True):
                    if not entry["present"]:
                        continue
                    ok = entry["ok"]
                    bad += 0 if ok else 1
                    console.print(f"{'OK  ' if ok else 'BAD '} {mid}/{v.id}  {entry['filename']}: {entry['detail']}", markup=False)
        if bad:
            raise typer.Exit(EXIT_ERROR)
    except VmtError as exc:
        raise _fail(exc)


@models_app.command("remove")
def models_remove(
    model_id: str = typer.Argument(..., help="Model id"),
    variant: Optional[str] = typer.Option(None, "--variant", help="Variant (default: every variant)."),
) -> None:
    """Delete downloaded weight files of a model."""

    try:
        settings = _settings()
        _init_logging(settings)
        from ..models import registry

        info = registry.get_model_info(model_id, settings)
        wm = registry.weights_manager(settings)
        variants = [info.variant(variant)] if variant else list(info.variants)
        freed = sum(wm.remove(info.id, v) for v in variants)
        console.print(f"Removed {_bytes(freed)} of weights for {model_id}.", markup=False)
    except VmtError as exc:
        raise _fail(exc)


# =========================================================================== probe / system / presets


@app.command()
def probe(
    video: Path = typer.Argument(..., help="Video file to analyse"),
    json_out: bool = typer.Option(False, "--json", help="Machine readable output."),
) -> None:
    """Show the metadata the tool extracts from a video (resolution, exact timing, colour, audio)."""

    try:
        settings = _settings()
        _init_logging(settings)
        from ..encoding.ffmpeg import find_ffmpeg
        from ..processing.probe import probe as run_probe

        tools = find_ffmpeg(settings)
        result = run_probe(video, ffprobe=tools.ffprobe)
    except VmtError as exc:
        raise _fail(exc)
    meta = result.metadata
    if json_out:
        console.print_json(json.dumps(meta.to_dict(), default=str))
        return
    v = meta.video
    p = lambda text="": console.print(text, markup=False)   # noqa: E731
    p(f"File      : {meta.path}  ({_bytes(meta.size_bytes)})")
    p(f"Container : {meta.container}  ({meta.format_long_name})")
    p(f"Video     : {v.codec} {v.profile or ''}  {v.width}x{v.height}  {v.pix_fmt}  {v.bit_depth}-bit")
    if v.rotation:
        p(f"            rotation {v.rotation} deg -> displayed as {v.display_width}x{v.display_height}")
    rate = f"{float(meta.frame_rate):.4f} fps" if meta.frame_rate else "unknown"
    p(f"Timing    : {meta.frame_count} frames, {rate} ({'VARIABLE frame rate' if meta.is_vfr else 'constant frame rate'}), "
      f"time base {v.time_base}, duration {float(meta.video_duration or 0):.3f} s")
    if meta.is_vfr and meta.vfr:
        p(f"            frame durations {meta.vfr.get('min_frame_duration', 0) * 1000:.2f} .. "
          f"{meta.vfr.get('max_frame_duration', 0) * 1000:.2f} ms")
    c = v.color
    p(f"Colour    : range {c.range}, matrix {c.space}, primaries {c.primaries}, transfer {c.transfer}"
      + ("  [HDR]" if c.is_hdr else ""))
    if v.field_order not in ("progressive", "unknown"):
        p(f"            interlaced ({v.field_order})")
    if meta.audio:
        for a in meta.audio:
            p(f"Audio     : #{a.index} {a.codec} {a.sample_rate or '?'} Hz {a.channel_layout or a.channels or '?'}"
              f"  {(a.bit_rate or 0) // 1000} kb/s  {float(a.duration or 0):.3f} s")
    else:
        p("Audio     : none")
    for text in meta.warnings:
        p(f"Warning   : {text}")


@app.command()
def system(json_out: bool = typer.Option(False, "--json", help="Machine readable output.")) -> None:
    """Show detected hardware (GPU, VRAM, RAM), PyTorch and FFmpeg."""

    try:
        settings = _settings()
        _init_logging(settings)
        from ..core.hardware import system_info
        from ..encoding.ffmpeg import find_ffmpeg

        info = system_info()
        try:
            tools = find_ffmpeg(settings)
            info["ffmpeg"] = {
                "version": tools.version, "ffmpeg": tools.ffmpeg, "ffprobe": tools.ffprobe,
                "encoders": {c.value: tools.has_encoder(CODECS[c].ffmpeg_encoder) for c in VideoCodec},
            }
        except VmtError as exc:
            info["ffmpeg"] = {"error": exc.message, "hint": exc.hint}
    except VmtError as exc:
        raise _fail(exc)
    if json_out:
        console.print_json(json.dumps(info, default=str))
        return
    p = lambda text="": console.print(text, markup=False)   # noqa: E731
    p(f"Platform : {info['platform']}")
    p(f"CPU      : {info['cpu']} ({info['cpu_threads']} threads)")
    p(f"RAM      : {_mb(info.get('ram_total_mb'))} total, {_mb(info.get('ram_available_mb'))} available")
    p(f"Python   : {info['python']}    PyTorch: {info.get('torch')}    CUDA runtime: {info.get('cuda_runtime') or 'none'}")
    p("Devices  :")
    for d in info["devices"]:
        mem = f"{_mb(d['free_memory_mb'])} free of {_mb(d['total_memory_mb'])}" if d.get("total_memory_mb") else ""
        state = "" if d["available"] else "  [NOT USABLE]"
        p(f"  - {d['id']:<7} {d['name']}  {mem}  {d['detail']}{state}")
    p(f"Default device: {info['default_device']}")
    ff = info.get("ffmpeg", {})
    if "error" in ff:
        p(f"FFmpeg   : NOT FOUND - {ff['error']}  ({ff.get('hint')})")
    else:
        p(f"FFmpeg   : {ff['version']}  ({ff['ffmpeg']})")
        enc = ", ".join(f"{name}{'' if ok else ' (unavailable)'}" for name, ok in ff["encoders"].items())
        p(f"Encoders : {enc}")


@app.command()
def presets() -> None:
    """List the quality presets, refinement presets and output codecs."""

    table = Table(title="Quality presets (--quality)")
    for column in ("Name", "Description", "Refinement preset"):
        table.add_column(column, overflow="fold")
    for q in QualityPreset:
        info = QUALITY_PRESETS[q]
        table.add_row(q.value, info.description, info.refinement_preset)
    console.print(table)

    fields = ("temporal_smoothing", "edge_feather", "black_point", "white_point", "decontaminate", "min_artifact_area", "max_hole_area")
    table = Table(title="Refinement presets (--mask-refinement)")
    table.add_column("Name")
    for f in fields:
        table.add_column(f, overflow="fold")
    for name, params in REFINEMENT_PRESETS.items():
        table.add_row(name, *[str(getattr(params, f)) for f in fields])
    console.print(table)

    table = Table(title="Output codecs (--output-codec)")
    for column in ("Id", "Codec", "Use", "Containers", "Alpha", "Notes"):
        table.add_column(column, overflow="fold")
    for codec in VideoCodec:
        c = CODECS[codec]
        table.add_row(codec.value, c.label, c.group, ", ".join(c.containers), "yes" if c.supports_alpha else "no", c.description)
    console.print(table)


def run() -> None:
    """Entry point used by ``main.py`` and the console script."""

    make_stdio_safe()
    try:
        app()
    except KeyboardInterrupt:
        err_console.print("Interrupted.")
        sys.exit(EXIT_CANCELLED)


if __name__ == "__main__":  # pragma: no cover
    run()
