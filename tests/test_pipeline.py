"""End-to-end acceptance tests of the processing engine.

Every test runs the real pipeline (probe -> decode -> matte -> refine -> composite -> encode -> mux
-> verify) on a synthetic clip.  The matte comes from the weight-free test model (a keyer for the
magenta test subject), so these tests need neither a GPU nor downloaded weights; everything else is
the production code path.  Real models are covered in ``test_models_e2e.py``.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from app.core.config import ProcessingConfig
from app.core.errors import (
    InvalidPathError,
    JobCancelledError,
    OutputExistsError,
    UnknownModelError,
    VmtError,
)
from app.core.events import CancelToken, Stage
from app.core.pixfmt import parse_working_pixfmt, resolve_color, ColorInfo
from tests.helpers import media

pytestmark = pytest.mark.usefixtures("test_models")


# --------------------------------------------------------------------------- helpers


def make_config(**overrides) -> ProcessingConfig:
    data = {"model": "test_chroma", "device": "cpu", "quality": "balanced"}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value
    return ProcessingConfig.model_validate(data)


def run(app_settings, clip, out: Path, **overrides):
    from app.processing.pipeline import MattingPipeline

    return MattingPipeline(app_settings).run(clip.path, out, make_config(**overrides))


def test_matroska_time_step_does_not_depend_on_the_reported_time_base():
    """Matroska and WebM count in milliseconds whatever time base a demuxer reports for the stream."""

    from fractions import Fraction

    from app.quality.verify import _time_step

    ms = Fraction(1, 1000)
    for container in ("mkv", "webm"):
        assert _time_step(container, ms) == ms
        assert _time_step(container, Fraction(1, 1_000_000_000)) == ms      # finer than the stored step
        assert _time_step(container, Fraction(1, 30000)) == ms
        assert _time_step(container, Fraction(1, 25)) == Fraction(1, 25)     # coarser: taken as it is
    for container in ("mp4", "mov", "other", None):
        assert _time_step(container, Fraction(1, 30000)) == Fraction(1, 30000)


def code_values(fmt_name: str, width: int, height: int, rgb=(0, 255, 0), color: ColorInfo = ColorInfo(), full_range_hint=None):
    """Exact code values of a background colour for a stream with the given tags."""

    from app.compositing.colors import ColorConverter

    fmt = parse_working_pixfmt(fmt_name)
    resolved = resolve_color(color, fmt, width, height, full_range_hint)
    return ColorConverter(fmt, resolved).color_code_values(rgb)


def masks_for(clip, index: int, margin: float = 10.0):
    """Ground-truth regions of the synthetic clip: far background and deep subject core."""

    alpha = media.subject_alpha(index, clip.width, clip.height, clip.frames, soft=1.0)
    yy, xx = np.mgrid[0 : clip.height, 0 : clip.width].astype(np.float32)
    cx, cy = media.subject_center(index, clip.width, clip.height, clip.frames)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radius = media.subject_radius(clip.width, clip.height)
    background = r > radius + margin
    core = r < radius - margin
    if clip.facts.get("two_subjects"):
        background &= media.distractor_alpha(clip.width, clip.height) < 0.001
        d = np.sqrt((xx - clip.width * 0.88) ** 2 + (yy - clip.height * 0.18) ** 2)
        background &= d > 0.09 * min(clip.width, clip.height) + margin
    return background, core, alpha


def chroma_mask(mask: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Reduce a luma-resolution mask to the resolution of *plane* (all contributing samples set)."""

    if mask.shape == plane.shape:
        return mask
    fy = int(round(mask.shape[0] / plane.shape[0]))
    fx = int(round(mask.shape[1] / plane.shape[1]))
    h, w = plane.shape
    padded = np.zeros((h * fy, w * fx), dtype=bool)
    padded[: mask.shape[0], : mask.shape[1]] = mask
    return padded.reshape(h, fy, w, fx).all(axis=(1, 3))


def video_stream(path: Path) -> dict:
    return [s for s in media.ffprobe(path)["streams"] if s["codec_type"] == "video"][0]


def audio_streams(path: Path) -> list:
    return [s for s in media.ffprobe(path)["streams"] if s["codec_type"] == "audio"]


def assert_no_failed_checks(report):
    failed = [f"{c.id}: {c.detail}" for c in report.checks if c.status.value == "fail"]
    assert not failed, failed


# --------------------------------------------------------------------------- the main promise


def test_green_screen_preserves_timing_audio_and_pixels(app_settings, clip_cfr, tmp_path):
    out = tmp_path / "out_green.mp4"
    report = run(app_settings, clip_cfr, out)

    assert out.is_file() and report.status in ("completed", "completed_with_warnings")
    assert_no_failed_checks(report)
    assert report.timing.frames_processed == clip_cfr.frames

    v = video_stream(out)
    assert (v["width"], v["height"]) == (clip_cfr.width, clip_cfr.height)
    assert v["codec_name"] == "h264" and v["pix_fmt"] == "yuv420p"

    # every frame keeps its exact presentation time
    assert media.packet_pts(out) == media.packet_pts(clip_cfr.path)

    # audio is stream-copied
    a = audio_streams(out)
    assert len(a) == 1 and a[0]["codec_name"] == "aac"
    assert report.encoding["audio"] == "copy"

    # the background is uniform chroma green, the subject keeps its pixels
    green = code_values("yuv420p", clip_cfr.width, clip_cfr.height)
    result = media.decode_planes(out)
    source = media.decode_planes(clip_cfr.path)
    assert len(result) == clip_cfr.frames
    for index in (0, clip_cfr.frames // 2, clip_cfr.frames - 1):
        background, core, _ = masks_for(clip_cfr, index)
        for p, (plane, src) in enumerate(zip(result[index], source[index])):
            bg = plane[chroma_mask(background, plane)].astype(int)
            assert abs(bg.mean() - green[p]) <= 1.0 and np.abs(bg - green[p]).max() <= 3      # lossy codec tolerance
            fg_mask = chroma_mask(core, plane)
            diff = np.abs(plane[fg_mask].astype(int) - src[fg_mask].astype(int))
            assert diff.mean() < 2.0                                                          # subject unchanged

    # a report was written next to the output and no scratch files were left behind
    saved = json.loads((tmp_path / "out_green.report.json").read_text(encoding="utf-8"))
    assert saved["model"]["id"] == "test_chroma" and saved["input"]["frame_count"] == clip_cfr.frames
    assert {c["id"] for c in saved["checks"]} >= {"resolution", "frame_count", "timestamps", "background_color"}
    assert saved["comparison"] and saved["hardware"]["device"] == "cpu"
    assert not (tmp_path / ".vmt_work").exists()
    assert not list(tmp_path.glob("*.partial*"))


def test_lossless_output_is_bit_exact(app_settings, clip_cfr, tmp_path):
    """With a lossless codec: opaque pixels are the source pixels, the background is one exact value."""

    out = tmp_path / "exact.mkv"
    report = run(app_settings, clip_cfr, out, encoding={"codec": "ffv1"})
    assert_no_failed_checks(report)
    assert video_stream(out)["codec_name"] == "ffv1"
    green = code_values("yuv420p", clip_cfr.width, clip_cfr.height)
    result = media.decode_planes(out)
    source = media.decode_planes(clip_cfr.path)
    for index in range(0, clip_cfr.frames, 7):
        background, core, _ = masks_for(clip_cfr, index)
        for p, (plane, src) in enumerate(zip(result[index], source[index])):
            bg = plane[chroma_mask(background, plane)]
            assert bg.min() == bg.max() == green[p]                       # perfectly uniform, exact value
            fg_mask = chroma_mask(core, plane)
            assert np.array_equal(plane[fg_mask], src[fg_mask])           # bit-exact source pixels


def test_variable_frame_rate_is_preserved(app_settings, clip_vfr, tmp_path):
    out = tmp_path / "vfr.mp4"
    report = run(app_settings, clip_vfr, out)
    assert_no_failed_checks(report)
    times = media.packet_pts(out)
    assert times == media.packet_pts(clip_vfr.path)
    deltas = {b - a for a, b in zip(times, times[1:])}
    assert len(deltas) >= 5                                               # still genuinely variable
    assert report.input["is_vfr"] is True and report.output["is_vfr"] is True


def test_10bit_hdr_source_keeps_depth_and_tags(app_settings, clip_10bit, tmp_path):
    out = tmp_path / "hdr.mov"
    report = run(app_settings, clip_10bit, out)
    assert_no_failed_checks(report)
    v = video_stream(out)
    assert v["codec_name"] == "hevc" and v["pix_fmt"] == "yuv420p10le"    # automatic codec for >8-bit sources
    assert v.get("color_space") == "bt2020nc" and v.get("color_transfer") == "arib-std-b67"
    assert v.get("color_primaries") == "bt2020" and v.get("color_range") == "tv"
    assert media.packet_pts(out) == media.packet_pts(clip_10bit.path)
    assert any("HDR" in w for w in report.warnings)
    green = code_values(
        "yuv420p10le", clip_10bit.width, clip_10bit.height,
        color=ColorInfo(range="tv", space="bt2020nc", primaries="bt2020", transfer="arib-std-b67"),
    )
    frame = media.decode_planes(out, max_frames=1)[0]
    background, _, _ = masks_for(clip_10bit, 0)
    for p, plane in enumerate(frame):
        bg = plane[chroma_mask(background, plane)].astype(int)
        assert abs(bg.mean() - green[p]) <= 4.0


def test_rotation_metadata_is_preserved(app_settings, clip_rotated, tmp_path):
    out = tmp_path / "portrait.mp4"
    report = run(app_settings, clip_rotated, out)
    assert_no_failed_checks(report)
    v = video_stream(out)
    assert (v["width"], v["height"]) == (clip_rotated.width, clip_rotated.height)      # coded size untouched
    src_rot = [sd.get("rotation") for sd in video_stream(clip_rotated.path).get("side_data_list", []) if "rotation" in sd]
    out_rot = [sd.get("rotation") for sd in v.get("side_data_list", []) if "rotation" in sd]
    assert src_rot and out_rot == src_rot
    # the matte was computed on the upright picture and turned back: the subject is where it was
    frame = media.decode_planes(out, max_frames=1)[0]
    background, core, _ = masks_for(clip_rotated, 0)
    green = code_values("yuv420p", clip_rotated.width, clip_rotated.height)
    assert abs(frame[0][background].astype(int).mean() - green[0]) <= 1.5
    assert abs(frame[0][core].astype(int).mean() - green[0]) > 20


def test_source_without_audio_and_millisecond_time_base(app_settings, clip_noaudio_mkv, tmp_path):
    out = tmp_path / "silent.mp4"
    report = run(app_settings, clip_noaudio_mkv, out)
    assert_no_failed_checks(report)
    assert not audio_streams(out)
    assert report.encoding["audio"] == "none"
    assert media.packet_pts(out) == media.packet_pts(clip_noaudio_mkv.path)


def test_webm_source(app_settings, clip_webm, tmp_path):
    out = tmp_path / "from_webm.mp4"
    report = run(app_settings, clip_webm, out)
    assert_no_failed_checks(report)
    assert media.packet_pts(out) == media.packet_pts(clip_webm.path)
    assert len(audio_streams(out)) == 1


def test_odd_dimensions_and_444(app_settings, clip_odd_444, tmp_path):
    out = tmp_path / "odd.mkv"
    report = run(app_settings, clip_odd_444, out, encoding={"codec": "ffv1"})
    assert_no_failed_checks(report)
    v = video_stream(out)
    assert (v["width"], v["height"], v["pix_fmt"]) == (321, 179, "yuv444p")


# --------------------------------------------------------------------------- backgrounds and exports


@pytest.mark.parametrize("background,rgb", [("blue", (0, 0, 255)), ("black", (0, 0, 0)), ("white", (255, 255, 255))])
def test_solid_backgrounds(app_settings, clip_cfr, tmp_path, background, rgb):
    out = tmp_path / f"{background}.mkv"
    report = run(app_settings, clip_cfr, out, background=background, encoding={"codec": "ffv1"})
    assert_no_failed_checks(report)
    expected = code_values("yuv420p", clip_cfr.width, clip_cfr.height, rgb)
    frame = media.decode_planes(out, max_frames=1)[0]
    bg_mask, _, _ = masks_for(clip_cfr, 0)
    for p, plane in enumerate(frame):
        bg = plane[chroma_mask(bg_mask, plane)]
        assert bg.min() == bg.max() == expected[p]


def test_custom_background_colour(app_settings, clip_cfr, tmp_path):
    out = tmp_path / "custom.mkv"
    run(app_settings, clip_cfr, out, background="custom", background_color="#336699", encoding={"codec": "ffv1"})
    expected = code_values("yuv420p", clip_cfr.width, clip_cfr.height, (0x33, 0x66, 0x99))
    frame = media.decode_planes(out, max_frames=1)[0]
    bg_mask, _, _ = masks_for(clip_cfr, 0)
    assert int(frame[0][bg_mask].max()) == int(frame[0][bg_mask].min()) == expected[0]


def test_transparent_prores_4444(app_settings, clip_cfr, tmp_path):
    out = tmp_path / "alpha.mov"
    report = run(app_settings, clip_cfr, out, background="transparent")
    assert_no_failed_checks(report)
    v = video_stream(out)
    assert v["codec_name"] == "prores" and "yuva444p" in v["pix_fmt"]
    with av.open(str(out)) as container:
        frame = next(container.decode(container.streams.video[0]))
        alpha_plane = frame.planes[3]
        bits = frame.format.components[3].bits
        a = np.frombuffer(alpha_plane, dtype="<u2").reshape(alpha_plane.height, alpha_plane.line_size // 2)[:, : alpha_plane.width]
        a = a.astype(np.float32) / float((1 << bits) - 1)
    background, core, _ = masks_for(clip_cfr, 0)
    assert a[background].max() < 0.02 and a[core].min() > 0.98


def test_all_exports(app_settings, clip_cfr, tmp_path):
    out = tmp_path / "main.mp4"
    report = run(
        app_settings, clip_cfr, out, background="blue",
        exports={"alpha_matte": True, "png_sequence": True, "prores_4444": True, "green_screen": True,
                 "audio": True, "matte_preview": True},
    )
    assert_no_failed_checks(report)
    kinds = {o.kind for o in report.outputs}
    assert kinds >= {"main", "alpha_matte", "png_sequence", "prores_4444", "green_screen", "audio", "matte_preview", "report"}
    for o in report.outputs:
        assert Path(o.path).exists(), o.path

    matte = tmp_path / "main_alpha.mp4"
    assert media.packet_pts(matte) == media.packet_pts(clip_cfr.path)
    luma = media.decode_planes(matte, max_frames=1)[0][0].astype(int)
    background, core, _ = masks_for(clip_cfr, 0)
    assert luma[background].mean() < 20 and luma[core].mean() > 230          # limited-range black / white

    pngs = sorted((tmp_path / "main_png").glob("*.png"))
    assert len(pngs) == clip_cfr.frames
    import cv2

    rgba = cv2.imdecode(np.fromfile(str(pngs[0]), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert rgba.shape == (clip_cfr.height, clip_cfr.width, 4)
    assert rgba[..., 3][background].max() == 0 and rgba[..., 3][core].min() == 255

    assert video_stream(tmp_path / "main_prores4444.mov")["codec_name"] == "prores"
    green = code_values("yuv420p", clip_cfr.width, clip_cfr.height)
    gframe = media.decode_planes(tmp_path / "main_green.mp4", max_frames=1)[0]
    assert abs(gframe[0][background].astype(int).mean() - green[0]) <= 1.5

    audio_file = [Path(o.path) for o in report.outputs if o.kind == "audio"][0]
    info = media.ffprobe(audio_file)
    assert [s["codec_name"] for s in info["streams"]] == ["aac"]

    preview = video_stream(tmp_path / "main_preview.mp4")
    assert preview["codec_name"] == "h264" and preview["width"] % 2 == 0 and preview["height"] % 2 == 0
    assert len(media.packet_pts(tmp_path / "main_preview.mp4")) == clip_cfr.frames


def test_explicit_resize(app_settings, clip_cfr, tmp_path):
    out = tmp_path / "small.mp4"
    report = run(app_settings, clip_cfr, out, encoding={"resolution": "90p"})
    v = video_stream(out)
    assert (v["width"], v["height"]) == (160, 90)
    assert any("scaled" in w.lower() for w in report.warnings)
    assert media.packet_pts(out) == media.packet_pts(clip_cfr.path)


# --------------------------------------------------------------------------- subject handling


def test_primary_subject_is_isolated(app_settings, clip_two_subjects, tmp_path):
    """The model keys both magenta blobs; isolation must keep only the primary (large, central) one."""

    yy, xx = np.mgrid[0 : clip_two_subjects.height, 0 : clip_two_subjects.width].astype(np.float32)
    blob = np.sqrt((xx - clip_two_subjects.width * 0.88) ** 2 + (yy - clip_two_subjects.height * 0.18) ** 2) < 6
    green = code_values("yuv420p", clip_two_subjects.width, clip_two_subjects.height)

    out = tmp_path / "isolated.mkv"
    run(app_settings, clip_two_subjects, out, encoding={"codec": "ffv1"})
    frame = media.decode_planes(out, max_frames=3)[2]
    assert np.all(frame[0][blob] == green[0])                              # the second blob is gone

    out_all = tmp_path / "all.mkv"
    run(app_settings, clip_two_subjects, out_all, encoding={"codec": "ffv1"}, subject={"isolate": False})
    frame_all = media.decode_planes(out_all, max_frames=3)[2]
    assert not np.any(frame_all[0][blob] == green[0])                      # kept when isolation is off


def test_selecting_the_other_subject_with_a_box(app_settings, clip_two_subjects, tmp_path):
    w, h = clip_two_subjects.width, clip_two_subjects.height
    box = [w * 0.88 - 22, h * 0.18 - 22, w * 0.88 + 22, h * 0.18 + 22]
    out = tmp_path / "small_blob.mkv"
    run(app_settings, clip_two_subjects, out, encoding={"codec": "ffv1"}, subject={"mode": "box", "box": box})
    frame = media.decode_planes(out, max_frames=3)[2]
    green = code_values("yuv420p", w, h)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    blob = np.sqrt((xx - w * 0.88) ** 2 + (yy - h * 0.18) ** 2) < 6
    _, core, _ = masks_for(clip_two_subjects, 2)
    assert not np.any(frame[0][blob] == green[0])                          # the selected blob is kept
    assert np.all(frame[0][core] == green[0])                              # the big one is removed


def test_reference_frame_in_the_middle_tracks_backwards(app_settings, clip_cfr, tmp_path):
    cx, cy = media.subject_center(12, clip_cfr.width, clip_cfr.height, clip_cfr.frames)
    out = tmp_path / "backtrack.mkv"
    report = run(
        app_settings, clip_cfr, out, model="test_mask", encoding={"codec": "ffv1"},
        subject={"mode": "points", "points": [{"x": cx, "y": cy, "label": 1}], "frame_index": 12},
    )
    assert report.timing.frames_processed == clip_cfr.frames
    assert "backtrack" in report.timing.stages
    green = code_values("yuv420p", clip_cfr.width, clip_cfr.height)
    frames = media.decode_planes(out)
    for index in (0, 5, 11, 12, 30):                                       # subject present before AND after frame 12
        _, core, _ = masks_for(clip_cfr, index)
        assert not np.any(frames[index][0][core] == green[0]), index


def test_mask_model_gets_soft_edges(app_settings, clip_cfr, tmp_path):
    out = tmp_path / "mask_model.mp4"
    cx, cy = media.subject_center(0, clip_cfr.width, clip_cfr.height, clip_cfr.frames)
    report = run(
        app_settings, clip_cfr, out, model="test_mask", exports={"alpha_matte": True},
        subject={"mode": "points", "points": [{"x": cx, "y": cy, "label": 1}]},   # this model needs a prompt
    )
    assert_no_failed_checks(report)
    luma = media.decode_planes(tmp_path / "mask_model_alpha.mp4", max_frames=1)[0][0].astype(int)
    soft = ((luma > 40) & (luma < 210)).sum()
    assert soft > 30                                                       # the hard mask was refined to a soft matte


# --------------------------------------------------------------------------- robustness


def test_partial_render_with_max_frames(app_settings, clip_cfr, tmp_path):
    out = tmp_path / "draft.mp4"
    report = run(app_settings, clip_cfr, out, max_frames=10)
    assert report.timing.frames_processed == 10
    assert media.packet_pts(out) == media.packet_pts(clip_cfr.path)[:10]
    duration = float(media.ffprobe(out)["format"]["duration"])
    assert duration < 0.6                                                  # audio was cut with the video (10 frames = 0.33 s)


def test_batch_size_is_reduced_on_out_of_memory(app_settings, clip_cfr, tmp_path, test_models):
    test_models["chroma"].oom_above_batch = 2
    out = tmp_path / "oom.mp4"
    report = run(app_settings, clip_cfr, out, batch_size=8)
    assert_no_failed_checks(report)
    assert report.hardware.batch_reductions >= 1 and report.hardware.final_batch_size <= 2
    assert any("batch size reduced" in w.lower() for w in report.warnings)
    assert media.packet_pts(out) == media.packet_pts(clip_cfr.path)


def test_cancel_then_resume_reuses_cached_mattes(app_settings, clip_cfr, tmp_path):
    from app.processing.pipeline import MattingPipeline

    out = tmp_path / "resumed.mkv"
    cfg = make_config(encoding={"codec": "ffv1"}, batch_size=4)
    token = CancelToken()

    def cancel_midway(event):
        if event.stage is Stage.RENDER and event.frame >= 16:
            token.cancel()

    with pytest.raises(JobCancelledError):
        MattingPipeline(app_settings).run(clip_cfr.path, out, cfg, progress=cancel_midway, cancel=token)
    assert not out.exists() and not list(tmp_path.glob("*.partial*"))      # nothing under the final name
    assert (tmp_path / ".vmt_work").exists()                               # ... but the matte cache survived

    report = MattingPipeline(app_settings).run(clip_cfr.path, out, cfg)
    assert report.resumed and report.timing.frames_from_cache >= 12
    assert report.timing.frames_processed == clip_cfr.frames
    assert_no_failed_checks(report)

    reference = tmp_path / "reference.mkv"
    MattingPipeline(app_settings).run(clip_cfr.path, reference, cfg)
    a, b = media.decode_planes(out), media.decode_planes(reference)
    assert all(np.array_equal(x, y) for fa, fb in zip(a, b) for x, y in zip(fa, fb))   # identical to an uninterrupted run


def test_rerender_with_new_settings_does_not_run_the_model(app_settings, clip_cfr, tmp_path, test_models):
    from app.processing.pipeline import MattingPipeline

    work = tmp_path / "work"
    first = MattingPipeline(app_settings).run(clip_cfr.path, tmp_path / "first.mp4", make_config(), work_dir=work)
    assert first.timing.frames_from_cache == 0

    seen = []
    original_load = test_models["chroma"].load

    def tracking_load(self):
        seen.append("load")
        return original_load(self)

    test_models["chroma"].load = tracking_load
    try:
        second = MattingPipeline(app_settings).run(
            clip_cfr.path, tmp_path / "second.mp4",
            make_config(background="blue", refinement={"edge_feather": 2.0}, encoding={"crf": 24}), work_dir=work,
        )
    finally:
        test_models["chroma"].load = original_load
    assert second.timing.frames_from_cache == clip_cfr.frames and not seen
    blue = code_values("yuv420p", clip_cfr.width, clip_cfr.height, (0, 0, 255))
    frame = media.decode_planes(tmp_path / "second.mp4", max_frames=1)[0]
    background, _, _ = masks_for(clip_cfr, 0)
    assert abs(frame[0][background].astype(int).mean() - blue[0]) <= 1.5


def test_progress_events_are_real(app_settings, clip_cfr, tmp_path):
    from app.processing.pipeline import MattingPipeline

    events = []
    MattingPipeline(app_settings).run(clip_cfr.path, tmp_path / "p.mp4", make_config(), progress=events.append)
    stages = [e.stage for e in events]
    for stage in (Stage.PROBE, Stage.SUBJECT, Stage.RENDER, Stage.MUX, Stage.VERIFY, Stage.DONE):
        assert stage in stages
    render = [e for e in events if e.stage is Stage.RENDER]
    assert render[-1].frame == clip_cfr.frames and render[-1].total_frames == clip_cfr.frames
    frames = [e.frame for e in render]
    assert frames == sorted(frames)                                        # monotonic, measured
    overall = [e.overall_percent for e in events if e.overall_percent is not None]
    assert overall == sorted(overall) and overall[-1] == 100.0
    assert any(e.fps for e in render)


# --------------------------------------------------------------------------- errors


def test_missing_input(app_settings, tmp_path):
    from app.processing.pipeline import MattingPipeline

    with pytest.raises(InvalidPathError):
        MattingPipeline(app_settings).run(tmp_path / "nope.mp4", tmp_path / "o.mp4", make_config())


def test_existing_output_is_not_overwritten_without_permission(app_settings, clip_cfr, tmp_path):
    out = tmp_path / "exists.mp4"
    out.write_bytes(b"keep me")
    with pytest.raises(OutputExistsError):
        run(app_settings, clip_cfr, out)
    assert out.read_bytes() == b"keep me"
    report = run(app_settings, clip_cfr, out, overwrite=True)
    assert report.timing.frames_processed == clip_cfr.frames and out.stat().st_size > 1000


def test_unknown_model(app_settings, clip_cfr, tmp_path):
    with pytest.raises(UnknownModelError) as info:
        run(app_settings, clip_cfr, tmp_path / "o.mp4", model="does_not_exist")
    assert "test_chroma" in (info.value.hint or "")
    assert not (tmp_path / "o.mp4").exists()


def test_not_a_video_file(app_settings, tmp_path):
    from app.processing.pipeline import MattingPipeline

    bogus = tmp_path / "text.mp4"
    bogus.write_text("this is not a video")
    with pytest.raises(VmtError) as info:
        MattingPipeline(app_settings).run(bogus, tmp_path / "o.mp4", make_config())
    assert info.value.code in ("unsupported_format", "corrupted_video", "no_video_stream")
    assert getattr(info.value, "report", None) is not None and info.value.report.status == "failed"
    assert not (tmp_path / "o.mp4").exists()


def test_truncated_video_is_reported(app_settings, clip_cfr, tmp_path):
    from app.processing.pipeline import MattingPipeline

    data = clip_cfr.path.read_bytes()
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(data[: len(data) // 3])                             # moov atom at the end is gone
    with pytest.raises(VmtError) as info:
        MattingPipeline(app_settings).run(broken, tmp_path / "o.mp4", make_config())
    assert info.value.code in ("corrupted_video", "unsupported_format")
    assert info.value.message and not (tmp_path / "o.mp4").exists()


def test_encoder_failure_leaves_no_output(app_settings, clip_cfr, tmp_path):
    out = tmp_path / "bad.mp4"
    with pytest.raises(VmtError) as info:
        run(app_settings, clip_cfr, out, encoding={"extra_args": ["-this_option_does_not_exist", "1"]})
    assert info.value.code in ("ffmpeg_failed", "encoding_error")
    assert "stderr_tail" in info.value.details
    assert not out.exists() and not list(tmp_path.glob("*.partial*"))
