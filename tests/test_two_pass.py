"""Two-pass models: the pipeline mattes the whole clip, lets the model revise, then renders.

The model here is a weight-free test fixture (the keyer of the other pipeline tests) whose
``revise`` blanks the left half of every matte, so the effect of the revision is visible in the
output file.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
import pytest

from app.core.errors import JobCancelledError
from app.core.events import CancelToken, Stage
from app.models.base import LicenseInfo, ModelCapabilities, ModelInfo, ModelVariant
from tests.helpers import media
from tests.helpers.dummy_model import ChromaTestModel
from tests.test_pipeline import assert_no_failed_checks, code_values, make_config, masks_for


class TwoPassTestModel(ChromaTestModel):
    info = ModelInfo(
        id="test_two_pass",
        name="Test two-pass model (test-suite only)",
        description="The test keyer, revised after the whole clip was seen. Not a real model.",
        license=LicenseInfo(name="n/a (test fixture)", url="", commercial_use="allowed"),
        capabilities=ModelCapabilities(
            alpha_type="alpha", temporal="none", subject_domain="any", max_batch_size=1,
            native_prompts=frozenset({"points", "box", "mask"}), tracks_single_subject=True, two_pass=True,
        ),
        variants=(ModelVariant(id="default", label="default", weights=()),),
        default_variant="default",
        limitations=("Test fixture.",),
    )

    #: what happened, across instances (reset by the fixture)
    log: List[tuple] = []
    #: seconds each frame takes (lets a test cancel in the middle of the first pass)
    delay = 0.0

    def start_sequence(self, seq) -> None:
        super().start_sequence(seq)
        type(self).log.append(("sequence", seq.start_index, bool(seq.reverse), bool(seq.will_revise)))

    def process(self, frames, frame_indices):
        type(self).log.append(("frame", int(frame_indices[0])))
        if type(self).delay:
            time.sleep(type(self).delay)
        return super().process(frames, frame_indices)

    def revise(self, indices, load, save, check=None) -> None:
        type(self).log.append(("revise", list(indices)))
        for index in indices:
            matte = load(index)
            assert matte.dtype == np.float32 and 0.0 <= float(matte.min()) and float(matte.max()) <= 1.0
            matte = matte.copy()
            matte[:, : matte.shape[1] // 2] = 0.0
            save(index, matte)
            if check is not None:
                check()


@pytest.fixture()
def two_pass(test_models):
    from app.models import registry

    registry.register_model(TwoPassTestModel)
    TwoPassTestModel.log = []
    TwoPassTestModel.delay = 0.0
    yield TwoPassTestModel
    TwoPassTestModel.delay = 0.0
    registry._REGISTRY.pop(TwoPassTestModel.info.id, None)


def run(app_settings, clip, out, **overrides):
    from app.processing.pipeline import MattingPipeline

    cfg = make_config(model="test_two_pass", encoding={"codec": "ffv1"}, **overrides)
    return MattingPipeline(app_settings).run(clip.path, out, cfg)


def left_half_is_background(clip, path):
    green = code_values("yuv420p", clip.width, clip.height)
    half = clip.width // 2
    frames = media.decode_planes(path)
    assert len(frames) == clip.frames
    kept = 0
    for index, planes in enumerate(frames):
        assert np.all(planes[0][:, : half - 8] == green[0]), index       # revised away ...
        _, core, _ = masks_for(clip, index)
        right = core.copy()
        right[:, : half + 8] = False
        if right.any():                                                  # ... the rest is the model's matte
            assert not np.any(planes[0][right] == green[0]), index
            kept += 1
    assert kept > 0
    return frames


def subject_prompt(clip, frame):
    cx, cy = media.subject_center(frame, clip.width, clip.height, clip.frames)
    return {"mode": "points", "points": [{"x": cx, "y": cy, "label": 1}], "frame_index": frame}


def test_whole_clip_is_matted_revised_and_then_rendered(app_settings, clip_cfr, tmp_path, two_pass):
    out = tmp_path / "two_pass.mkv"
    report = run(app_settings, clip_cfr, out, subject=subject_prompt(clip_cfr, 12))
    assert report.timing.frames_processed == clip_cfr.frames
    assert not report.resumed and report.timing.frames_from_cache == 0
    assert_no_failed_checks_except_tracking(report)                                    # timestamps, audio, pixels

    log = two_pass.log
    sequences = [entry for entry in log if entry[0] == "sequence"]
    assert sequences == [("sequence", 12, True, True), ("sequence", 12, False, True)]   # back to 0, then on to the end
    frames = [entry[1] for entry in log if entry[0] == "frame"]
    assert frames == list(range(12, -1, -1)) + list(range(12, clip_cfr.frames))
    revisions = [entry for entry in log if entry[0] == "revise"]
    assert revisions == [("revise", list(range(clip_cfr.frames)))]                      # once, with every frame
    assert log.index(revisions[0]) == len(log) - 1                                     # after all frames
    left_half_is_background(clip_cfr, out)


def test_reference_on_the_first_frame_needs_one_sequence(app_settings, clip_cfr, tmp_path, two_pass):
    out = tmp_path / "from_start.mkv"
    run(app_settings, clip_cfr, out, subject=subject_prompt(clip_cfr, 0))
    assert [entry for entry in two_pass.log if entry[0] == "sequence"] == [("sequence", 0, False, True)]
    assert sum(1 for entry in two_pass.log if entry[0] == "revise") == 1
    left_half_is_background(clip_cfr, out)


def test_a_finished_run_is_rerendered_from_the_revised_cache(app_settings, clip_cfr, tmp_path, two_pass):
    from app.processing.pipeline import MattingPipeline

    work = tmp_path / "work"
    cfg = make_config(model="test_two_pass", encoding={"codec": "ffv1"}, subject=subject_prompt(clip_cfr, 12))
    first = tmp_path / "first.mkv"
    MattingPipeline(app_settings).run(clip_cfr.path, first, cfg, work_dir=work)
    two_pass.log = []
    second = tmp_path / "second.mkv"
    report = MattingPipeline(app_settings).run(clip_cfr.path, second, cfg, work_dir=work)
    assert report.resumed and report.timing.frames_from_cache == clip_cfr.frames
    assert two_pass.log == []                                                          # the model was not needed
    a, b = left_half_is_background(clip_cfr, first), media.decode_planes(second)
    assert all(np.array_equal(x, y) for fa, fb in zip(a, b) for x, y in zip(fa, fb))


def test_an_interrupted_run_starts_again(app_settings, clip_cfr, tmp_path, two_pass):
    """Raw mattes of an unfinished run cannot be revised later: nothing of them is reused."""

    from app.processing.pipeline import MattingPipeline

    out = tmp_path / "interrupted.mkv"
    cfg = make_config(model="test_two_pass", encoding={"codec": "ffv1"}, subject=subject_prompt(clip_cfr, 12))
    token = CancelToken()
    two_pass.delay = 0.03                                 # slow enough for progress events to arrive

    def cancel_midway(event):
        if event.stage is Stage.BACKTRACK and event.frame >= 8:
            token.cancel()

    with pytest.raises(JobCancelledError):
        MattingPipeline(app_settings).run(clip_cfr.path, out, cfg, progress=cancel_midway, cancel=token)
    assert not out.exists()
    assert not any(entry[0] == "revise" for entry in two_pass.log)
    assert 8 <= sum(1 for entry in two_pass.log if entry[0] == "frame") < clip_cfr.frames

    two_pass.log = []
    two_pass.delay = 0.0
    report = MattingPipeline(app_settings).run(clip_cfr.path, out, cfg)
    assert not report.resumed and report.timing.frames_from_cache == 0
    assert sum(1 for entry in two_pass.log if entry[0] == "frame") == clip_cfr.frames + 1   # frame 12 starts both passes
    assert sum(1 for entry in two_pass.log if entry[0] == "revise") == 1
    assert_no_failed_checks_except_tracking(report)
    left_half_is_background(clip_cfr, out)


def assert_no_failed_checks_except_tracking(report):
    """The revision of this test model removes the subject from many frames on purpose."""

    failed = [c.id for c in report.checks if c.status.value == "fail" and "track" not in c.id and "subject" not in c.id]
    assert not failed, failed


def test_progress_covers_the_whole_clip_before_rendering(app_settings, clip_cfr, tmp_path, two_pass):
    from app.processing.pipeline import MattingPipeline

    events = []
    cfg = make_config(model="test_two_pass", encoding={"codec": "ffv1"}, subject=subject_prompt(clip_cfr, 12))
    MattingPipeline(app_settings).run(clip_cfr.path, tmp_path / "progress.mkv", cfg, progress=events.append)
    tracking = [e for e in events if e.stage is Stage.BACKTRACK]
    assert tracking and tracking[0].total_frames == clip_cfr.frames
    assert max(e.frame for e in tracking) == clip_cfr.frames
    overall = [e.overall_percent for e in events]
    assert all(b >= a - 1e-6 for a, b in zip(overall, overall[1:]))                    # never runs backwards
    assert any("outline" in (e.message or "").lower() for e in tracking)


def test_single_pass_models_are_not_announced_a_revision(app_settings, clip_cfr, tmp_path, test_models):
    from app.processing.pipeline import MattingPipeline

    out = tmp_path / "single.mkv"
    cfg = make_config(model="test_mask", encoding={"codec": "ffv1"}, subject=subject_prompt(clip_cfr, 12))
    report = MattingPipeline(app_settings).run(clip_cfr.path, out, cfg)
    assert_no_failed_checks(report)
    assert "backtrack" in report.timing.stages
