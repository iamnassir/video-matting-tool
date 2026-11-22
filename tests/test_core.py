"""Unit tests of the core contracts: configuration, timelines, pixel formats, reports, progress."""

from __future__ import annotations

import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from app.core import config as cfg
from app.core.config import (
    BackgroundMode,
    EncodingConfig,
    ProcessingConfig,
    QualityPreset,
    RefinementConfig,
    SubjectPrompt,
    VideoCodec,
    load_settings,
    parse_resolution,
    parse_subject_spec,
    resolve_codec_and_container,
)
from app.core.errors import ConfigError, GpuOutOfMemoryError, VmtError, is_cuda_oom
from app.core.events import CancelToken, ProgressReporter, RateMeter, Stage
from app.core.pixfmt import ColorInfo, make_pixfmt, parse_working_pixfmt, quantization, resolve_color
from app.core.report import CheckStatus, ProcessingReport, QualityCheck
from app.core.types import FrameTimeline, box_from_mask, clamp_box, parse_fraction


# --------------------------------------------------------------------------- configuration


def test_defaults_are_the_documented_ones():
    c = ProcessingConfig()
    assert c.model == "rvm" and c.background is BackgroundMode.GREEN
    assert c.background_rgb() == (0, 255, 0)                       # pure chroma green #00FF00
    assert c.encoding.resolution == "source" and c.encoding.preserve_audio is True
    assert c.subject.mode == "auto" and c.quality is QualityPreset.BALANCED


@pytest.mark.parametrize(
    "background,expected",
    [("green", (0, 255, 0)), ("blue", (0, 0, 255)), ("black", (0, 0, 0)), ("white", (255, 255, 255)), ("transparent", None)],
)
def test_background_colours(background, expected):
    assert ProcessingConfig(background=background).background_rgb() == expected


def test_custom_background_needs_a_colour():
    assert ProcessingConfig(background="custom", background_color="336699").background_rgb() == (0x33, 0x66, 0x99)
    with pytest.raises(ValueError):
        ProcessingConfig(background="custom")
    with pytest.raises(ValueError):
        ProcessingConfig(background="custom", background_color="#12")


def test_transparent_needs_an_alpha_codec():
    with pytest.raises(ValueError, match="alpha"):
        ProcessingConfig(background="transparent", encoding={"codec": "h264"})
    ProcessingConfig(background="transparent", encoding={"codec": "prores_4444"})


def test_codec_container_validation():
    with pytest.raises(ValueError, match="cannot be stored"):
        ProcessingConfig(encoding={"codec": "prores_422_hq", "container": "mp4"})
    with pytest.raises(ValueError):
        ProcessingConfig(device="gpu0")
    assert ProcessingConfig(device="CUDA:1").device == "cuda:1"


@pytest.mark.parametrize(
    "name,depth,expected",
    [
        ("out.mp4", 8, (VideoCodec.H264, "mp4")),
        ("out.mp4", 10, (VideoCodec.H265, "mp4")),             # deeper than 8 bits -> HEVC automatically
        ("out.mov", 8, (VideoCodec.H264, "mov")),
        ("out.mkv", 8, (VideoCodec.H264, "mkv")),
        ("out.webm", 8, (VideoCodec.VP9, "webm")),
    ],
)
def test_automatic_codec_choice(name, depth, expected):
    assert resolve_codec_and_container(ProcessingConfig(), Path(name), depth) == expected


def test_automatic_codec_for_transparency():
    c = ProcessingConfig(background="transparent")
    assert resolve_codec_and_container(c, Path("o.mov")) == (VideoCodec.PRORES_4444, "mov")
    assert resolve_codec_and_container(c, Path("o.webm")) == (VideoCodec.VP9, "webm")
    assert resolve_codec_and_container(c, Path("o.mkv")) == (VideoCodec.FFV1, "mkv")
    with pytest.raises(ConfigError):
        resolve_codec_and_container(c, Path("o.mp4"))
    with pytest.raises(ConfigError):
        resolve_codec_and_container(ProcessingConfig(), Path("o.avi"))
    with pytest.raises(ConfigError):
        resolve_codec_and_container(ProcessingConfig(encoding={"codec": "ffv1"}), Path("o.mp4"))


def test_parse_resolution():
    assert parse_resolution("source", 1920, 1080) == (1920, 1080, False)
    assert parse_resolution("1080p", 1920, 1080) == (1920, 1080, False)          # already that size: no scaling
    assert parse_resolution("720p", 1920, 1080) == (1280, 720, True)
    assert parse_resolution("720p", 1080, 1920) == (720, 1280, True)             # portrait: short side is the width
    assert parse_resolution("1280x720", 1920, 1080) == (1280, 720, True)
    with pytest.raises(ValueError):
        EncodingConfig(resolution="big")


@pytest.mark.parametrize(
    "spec,mode",
    [(None, "auto"), ("auto", "auto"), ("box:10,20,300,400", "box"), ("points:100,200;50,60,-", "points"),
     ("text:the woman in red", "text"), ("mask:first.png", "mask")],
)
def test_parse_subject_spec_modes(spec, mode):
    assert parse_subject_spec(spec).mode == mode


def test_parse_subject_spec_details():
    box = parse_subject_spec("box:10,20,300,400")
    assert box.box == (10.0, 20.0, 300.0, 400.0) and not box.normalized
    pts = parse_subject_spec("points:0.5,0.4;0.1,0.1,-")
    assert pts.normalized and [p.label for p in pts.points] == [1, 0]
    scaled = pts.to_pixels(1000, 500)
    assert (scaled.points[0].x, scaled.points[0].y) == (500.0, 200.0) and not scaled.normalized
    as_json = parse_subject_spec('{"mode": "box", "box": [1, 2, 30, 40], "frame_index": 7}')
    assert as_json.frame_index == 7
    for bad in ("box:1,2,3", "points:5", "points:1,2,?", "text:", "circle:1,2,3", "points:10,10,-"):
        with pytest.raises(ConfigError):
            parse_subject_spec(bad)


def test_subject_prompt_validation():
    with pytest.raises(ValueError):
        SubjectPrompt(mode="box")
    with pytest.raises(ValueError):
        SubjectPrompt(mode="box", box=(5, 5, 5, 40))                 # zero width
    with pytest.raises(ValueError):
        SubjectPrompt(mode="points", points=[{"x": 1, "y": 1, "label": 0}])     # only a negative click


def test_refinement_presets_and_overrides():
    balanced = RefinementConfig().resolve(QualityPreset.BALANCED)
    assert balanced.enabled and balanced.temporal_smoothing == pytest.approx(0.3) and balanced.edge_feather == 0.0
    off = RefinementConfig(preset="off").resolve()
    assert not off.enabled and off.black_point == 0.0 and off.white_point == 1.0
    partly = RefinementConfig(preset="off", edge_feather=2.0).resolve()
    assert partly.enabled and partly.edge_feather == 2.0             # an explicit tool re-enables refinement
    strong = RefinementConfig(preset="strong", temporal_smoothing=0.1).resolve()
    assert strong.temporal_smoothing == pytest.approx(0.1) and strong.motion_compensation is True
    assert balanced.px(10.0, 3840, 2160) == pytest.approx(20.0)      # pixel quantities are 1080p-referenced
    assert balanced.px(10.0, 1080, 1920) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        RefinementConfig(temporal_smoothing=1.5)


def test_matte_fingerprint_ignores_render_settings():
    a = ProcessingConfig(background="green", refinement={"edge_feather": 1.0}, encoding={"crf": 20}, max_frames=10)
    b = ProcessingConfig(background="blue", refinement={"edge_feather": 3.0}, encoding={"crf": 30})
    assert a.fingerprint_payload() == b.fingerprint_payload()        # same mattes can be reused
    c = ProcessingConfig(model="sam2")
    assert a.fingerprint_payload() != c.fingerprint_payload()


def test_settings_precedence(tmp_path, monkeypatch):
    user = tmp_path / "user.yaml"
    user.write_text("port: 9001\nmodels_dir: my_models\ndefaults:\n  quality: high\n", encoding="utf-8")
    s = load_settings(user)
    assert s.port == 9001 and s.models_dir == cfg.PROJECT_ROOT / "my_models"
    assert s.default_processing_config().quality is QualityPreset.HIGH
    monkeypatch.setenv("VMT_PORT", "9002")
    assert load_settings(user).port == 9002                           # environment beats the file
    assert load_settings(user, port=9003).port == 9003                # explicit override beats both
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("defaults:\n  quality: ultra\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(bad).default_processing_config()


# --------------------------------------------------------------------------- timelines


def _timeline(pts, tb):
    pts = np.asarray(pts, dtype=np.int64)
    return FrameTimeline(tb, pts, np.zeros(len(pts), np.int64), np.zeros(len(pts), bool))


def test_constant_frame_rate_timeline():
    tl = _timeline(np.arange(0, 1001 * 30, 1001), Fraction(1, 30000))
    assert not tl.is_vfr and tl.average_rate == Fraction(30000, 1001)
    assert tl.duration == Fraction(1001 * 30, 30000) and tl.time(3) == Fraction(3003, 30000)


def test_rounded_timestamps_are_not_variable_frame_rate():
    # 29.97 fps stored in milliseconds (Matroska): deltas alternate 33 / 34 ms but the rate is constant
    ms = np.round(np.arange(300) * 1000 / 29.97).astype(np.int64)
    tl = _timeline(ms, Fraction(1, 1000))
    assert len(set(np.diff(ms))) > 1 and not tl.is_vfr


def test_true_variable_frame_rate_is_detected():
    pts = np.cumsum(np.array([33, 33, 50, 17, 33, 100, 33] * 10))
    tl = _timeline(pts, Fraction(1, 1000))
    assert tl.is_vfr and tl.vfr_stats()["distinct_durations"] >= 4


def test_timeline_helpers():
    tl = FrameTimeline(Fraction(1, 1000), np.array([0, 40, 80, 120]), np.array([40, 40, 40, 40]), np.array([1, 0, 1, 0]))
    assert tl.index_of_pts(79) == 2 and tl.index_of_pts(10_000) == 3
    assert tl.preceding_keyframe(1) == 0 and tl.preceding_keyframe(3) == 2
    assert tl.frame_duration(3) == 40 and tl.duration == Fraction(160, 1000)
    assert len(_timeline([], Fraction(1, 25))) == 0 and _timeline([], Fraction(1, 25)).duration == 0


def test_small_helpers():
    assert parse_fraction("30000/1001") == Fraction(30000, 1001)
    assert parse_fraction("0/0") is None and parse_fraction("N/A") is None and parse_fraction(None) is None
    mask = np.zeros((10, 20), np.float32)
    mask[2:5, 3:9] = 1.0
    assert box_from_mask(mask) == (3.0, 2.0, 9.0, 5.0) and box_from_mask(np.zeros((4, 4))) is None
    assert clamp_box((30, 5, -5, 50), 20, 10) == (0.0, 5.0, 20.0, 10.0)


# --------------------------------------------------------------------------- pixel formats and colour


@pytest.mark.parametrize("name", ["yuv420p", "yuv422p10le", "yuva444p10le", "gbrp", "gbrap12le", "gray16le", "yuv440p"])
def test_pixel_format_names_round_trip(name):
    assert parse_working_pixfmt(name).name == name


def test_pixel_format_geometry():
    fmt = parse_working_pixfmt("yuv420p10le")
    assert fmt.plane_size(0, 1921, 1081) == (1081, 1921) and fmt.plane_size(1, 1921, 1081) == (541, 961)
    assert fmt.dtype.itemsize == 2 and fmt.max_value == 1023 and fmt.num_planes == 3
    assert parse_working_pixfmt("yuv420p").frame_bytes(1920, 1080) == 1920 * 1080 * 3 // 2
    assert make_pixfmt("yuv", 8, 1, 1).with_alpha().name == "yuva420p"
    for bad in ("yuvj420p", "nv12", "rgb24", "yuv420p10be"):
        with pytest.raises(ValueError):
            parse_working_pixfmt(bad)


def test_colour_resolution_heuristics():
    fmt = parse_working_pixfmt("yuv420p")
    hd = resolve_color(ColorInfo(), fmt, 1920, 1080)
    assert hd.matrix == "bt709" and not hd.full_range and set(hd.assumed) == {"matrix", "range"}
    sd = resolve_color(ColorInfo(), fmt, 720, 576)
    assert sd.matrix == "smpte170m"
    tagged = resolve_color(ColorInfo(range="pc", space="bt2020nc"), fmt, 3840, 2160)
    assert tagged.matrix == "bt2020nc" and tagged.full_range and tagged.assumed == ()
    hinted = resolve_color(ColorInfo(), fmt, 1920, 1080, full_range_hint=True)        # yuvj420p
    assert hinted.full_range and "range" not in hinted.assumed
    assert ColorInfo(transfer="smpte2084").is_hdr and not ColorInfo(transfer="bt709").is_hdr


def test_quantization_levels():
    assert quantization(parse_working_pixfmt("yuv420p"), False) == (16.0, 219.0, 128.0, 224.0)
    assert quantization(parse_working_pixfmt("yuv420p10le"), False) == (64.0, 876.0, 512.0, 896.0)
    assert quantization(parse_working_pixfmt("yuv420p"), True) == (0.0, 255.0, 128.0, 255.0)


# --------------------------------------------------------------------------- report, progress, errors


def test_report_status_and_serialisation(tmp_path):
    report = ProcessingReport(input_path="in.mp4", output_path="out.mp4")
    report.add_check(QualityCheck(id="resolution", name="Resolution", status=CheckStatus.PASS, detail="ok"))
    report.finalize_status()
    assert report.status == "completed"
    report.add_check(QualityCheck(id="resolution", name="Resolution", status=CheckStatus.FAIL, detail="wrong"))
    assert len(report.checks) == 1                                    # same id replaces
    report.add_warning("careful")
    report.add_warning("careful")
    report.finalize_status()
    assert report.status == "completed_with_warnings" and report.warnings == ["careful"]
    path = report.save(tmp_path / "r" / "report.json")
    again = ProcessingReport.load(path)
    assert again.failed_checks[0].detail == "wrong" and json.loads(path.read_text(encoding="utf-8"))["status"] == report.status
    text = again.to_text()
    assert "PROCESSING REPORT" in text and "[FAIL" in text and text.isascii()
    report.status = "failed"
    report.finalize_status()
    assert report.status == "failed"                                  # a failure is never overwritten


def test_progress_reporter_measures_rates():
    events = []
    reporter = ProgressReporter(events.append, min_interval=0.0)
    reporter.stage(Stage.RENDER, "working", total_frames=100)
    reporter.advance(10)
    time.sleep(0.12)
    reporter.advance(60)
    last = events[-1]
    assert last.frame == 60 and last.percent == pytest.approx(60.0)
    assert last.fps and 150 < last.fps < 2000 and last.eta_seconds is not None and last.eta_seconds < 1.0
    overall = [e.overall_percent for e in events if e.overall_percent is not None]
    assert overall == sorted(overall)
    reporter.finish(Stage.DONE, "ok")
    assert events[-1].stage is Stage.DONE and events[-1].overall_percent == 100.0


def test_progress_is_throttled():
    events = []
    reporter = ProgressReporter(events.append, min_interval=10.0)
    reporter.stage(Stage.RENDER, "", total_frames=1000)               # stage changes are always delivered
    for i in range(200):
        reporter.advance(i)
    assert len(events) == 1


def test_rate_meter_and_cancel_token():
    meter = RateMeter(window_seconds=1.0)
    assert meter.rate is None
    meter.update(0, now=0.0)
    meter.update(50, now=0.5)
    assert meter.rate == pytest.approx(100.0)
    token = CancelToken()
    token.raise_if_cancelled()
    token.cancel("stopped for a test")
    with pytest.raises(VmtError) as info:
        token.raise_if_cancelled()
    assert info.value.code == "cancelled" and "stopped for a test" in info.value.message


def test_errors_carry_code_hint_and_details():
    error = GpuOutOfMemoryError("no memory", details={"device": "cuda:0"})
    data = error.to_dict()
    assert data["code"] == "gpu_out_of_memory" and data["hint"] and data["details"] == {"device": "cuda:0"}
    assert "hint" in str(error)
    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert is_cuda_oom(RuntimeError("MPS backend out of memory"))
    assert not is_cuda_oom(RuntimeError("shape mismatch")) and not is_cuda_oom(ValueError("out of memory"))
