"""Tests for the tracking health monitor (app/tracking/monitor.py)."""

from __future__ import annotations

import json
import math
from typing import Optional

import pytest
import torch

from app.core.errors import ConfigError
from app.tracking.component_tracker import TrackingError
from app.tracking.monitor import TrackingMonitor

W, H = 320, 180


def blob(cx: float, cy: float, radius: float, soft: float = 2.0, w: int = W, h: int = H) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    d = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return ((radius - d) / (2.0 * soft) + 0.5).clamp(0.0, 1.0)[None, None]


def empty(w: int = W, h: int = H) -> torch.Tensor:
    return torch.zeros(1, 1, h, w)


def feed(monitor: TrackingMonitor, frames, start: int = 0, presence: Optional[list] = None) -> None:
    for i, a in enumerate(frames):
        monitor.update(a, [start + i], None if presence is None else [presence[i]])


def test_coverage_numbers():
    monitor = TrackingMonitor(W, H, fps=25.0)
    radii = [20.0, 30.0, 40.0]
    monitor.update(torch.cat([blob(160.0, 90.0, r) for r in radii]), [0, 1, 2])          # one batch
    report = monitor.report()
    expected = [math.pi * r * r / (W * H) for r in radii]
    assert report["frames"] == 3
    assert report["coverage_min"] == pytest.approx(expected[0], rel=0.03)
    assert report["coverage_max"] == pytest.approx(expected[2], rel=0.03)
    assert report["coverage_mean"] == pytest.approx(sum(expected) / 3.0, rel=0.03)
    assert report["empty_frames"] == 0 and report["absent_segments"] == []
    assert report["suspected_switches"] == [] and report["warnings"] == []
    assert report["low_presence_frames"] is None                # the model did not report presence
    json.dumps(report)                                           # goes into the processing report as is

    # an untouched monitor reports zeros, not an error
    fresh = TrackingMonitor(W, H, fps=25.0).report()
    assert fresh["frames"] == 0 and fresh["coverage_mean"] == 0.0 and fresh["warnings"] == []


def test_absent_segments_are_detected_and_described():
    monitor = TrackingMonitor(W, H, fps=30.0)
    subject = blob(160.0, 90.0, 35.0)
    frames = [subject] * 10 + [empty()] * 12 + [subject] * 5 + [empty()] + [subject] * 4
    feed(monitor, frames, start=100)
    report = monitor.report()
    assert report["frames"] == 32
    assert report["empty_frames"] == 13
    assert report["absent_segments"] == [[110, 121], [127, 127]]
    assert report["coverage_min"] == 0.0
    assert "Subject not visible in frames 110-121 (0.40 s)." in report["warnings"]
    assert "Subject not visible in frame 127 (0.03 s)." in report["warnings"]
    assert len(report["warnings"]) == 2
    assert report["suspected_switches"] == []                   # same place before and after the gap

    # a faint residue far below the "empty" threshold still counts as absent
    monitor.update(torch.full((1, 1, H, W), 2e-5), [132])
    assert monitor.report()["absent_segments"][-1] == [132, 132]

    # durations: one second and more is given with one decimal
    long_gap = TrackingMonitor(W, H, fps=29.97)
    feed(long_gap, [subject] * 2 + [empty()] * 29 + [subject] * 2, start=118)
    assert "Subject not visible in frames 120-148 (1.0 s)." in long_gap.report()["warnings"]
    # without a usable frame rate the sentence simply has no duration
    no_fps = TrackingMonitor(W, H, fps=0.0)
    feed(no_fps, [subject, empty(), empty(), subject])
    assert no_fps.report()["warnings"] == ["Subject not visible in frames 1-2."]


def test_all_frames_empty_gives_one_clear_warning():
    monitor = TrackingMonitor(W, H, fps=24.0)
    feed(monitor, [empty()] * 48)
    report = monitor.report()
    assert report["empty_frames"] == 48 and report["absent_segments"] == [[0, 47]]
    assert report["coverage_max"] == 0.0
    assert len(report["warnings"]) == 1
    assert report["warnings"][0].startswith("No subject was found in any of the 48 frames (2.0 s)")


def test_suspected_switch_is_detected_on_a_jump_across_the_frame():
    monitor = TrackingMonitor(W, H, fps=30.0)
    top_left = blob(40.0, 35.0, 25.0)
    bottom_right = blob(285.0, 150.0, 25.0)
    feed(monitor, [top_left] * 6 + [bottom_right] * 6)
    report = monitor.report()
    assert report["suspected_switches"] == [6]
    assert report["absent_segments"] == []
    assert len(report["warnings"]) == 1 and "Possible subject switch at frame 6" in report["warnings"][0]
    distance = math.hypot(285 - 40, 150 - 35) / math.hypot(W, H)
    assert f"{distance * 100:.0f} %" in report["warnings"][0]

    # the jump is also found across a gap in which the subject was not visible
    gap = TrackingMonitor(W, H, fps=30.0)
    feed(gap, [top_left] * 3 + [empty()] * 4 + [bottom_right] * 3)
    assert gap.report()["suspected_switches"] == [7]
    assert gap.report()["absent_segments"] == [[3, 6]]


def test_ordinary_motion_and_growth_are_not_switches():
    # a subject that crosses the whole frame in small steps
    walking = TrackingMonitor(W, H, fps=30.0)
    feed(walking, [blob(30.0 + 13.0 * i, 90.0, 25.0) for i in range(21)])
    assert walking.report()["suspected_switches"] == []

    # a subject that suddenly becomes much larger: the centroid moves far, but the boxes overlap
    growing = TrackingMonitor(W, H, fps=30.0)
    small = blob(30.0, 30.0, 20.0)
    huge = torch.ones(1, 1, H, W)
    huge[..., :, :20] = 0.0
    feed(growing, [small, small, huge, huge])
    assert growing.report()["suspected_switches"] == []

    # a moderate jump below the distance threshold (a third of the diagonal) is not reported by default ...
    moderate = TrackingMonitor(W, H, fps=30.0)
    feed(moderate, [blob(60.0, 90.0, 20.0)] * 2 + [blob(180.0, 90.0, 20.0)] * 2)
    assert moderate.report()["suspected_switches"] == []
    # ... but the threshold can be tightened
    strict = TrackingMonitor(W, H, fps=30.0, switch_distance=0.25)
    feed(strict, [blob(60.0, 90.0, 20.0)] * 2 + [blob(180.0, 90.0, 20.0)] * 2)
    assert strict.report()["suspected_switches"] == [2]


def test_low_presence_and_full_frames_are_reported():
    monitor = TrackingMonitor(W, H, fps=30.0)
    subject = blob(160.0, 90.0, 35.0)
    presence = [0.9, 0.95, 0.2, 0.1, 0.3, 0.9, 0.4, 0.99]
    feed(monitor, [subject] * 8, presence=presence)
    report = monitor.report()
    assert report["low_presence_frames"] == 4
    assert report["low_presence_segments"] == [[2, 4], [6, 6]]
    assert any("low confidence that the subject is present in 4 frames" in w and "frames 2-4" in w for w in report["warnings"])

    # presence reported for only some batches: frames without a value do not count as low
    partial = TrackingMonitor(W, H, fps=30.0)
    partial.update(subject, [0])
    partial.update(subject, [1], presence=[0.1])
    assert partial.report()["low_presence_frames"] == 1

    full = TrackingMonitor(W, H, fps=30.0)
    feed(full, [subject, torch.ones(1, 1, H, W), torch.ones(1, 1, H, W)])
    report = full.report()
    assert report["full_frames"] == 2 and report["coverage_max"] == pytest.approx(1.0)
    assert any("covers the whole picture in 2 frames" in w for w in report["warnings"])


def test_many_problems_are_summarised_not_listed_endlessly():
    monitor = TrackingMonitor(W, H, fps=30.0)
    subject = blob(160.0, 90.0, 35.0)
    frames = []
    for _ in range(20):
        frames += [subject, subject, empty()]
    feed(monitor, frames)
    report = monitor.report()
    assert len(report["absent_segments"]) == 20
    assert len(report["warnings"]) == TrackingMonitor.MAX_LISTED + 1
    assert "8 more segments (8 frames, 0.27 s)" in report["warnings"][-1]


def test_updates_in_any_order_replace_earlier_records_and_do_not_touch_the_input():
    monitor = TrackingMonitor(W, H, fps=30.0)
    subject = blob(160.0, 90.0, 35.0)
    before = subject.clone()
    monitor.update(torch.cat([subject, empty()]), [5, 4])        # out of order within a batch
    monitor.update(subject, [3])
    monitor.update(subject, [4])                                 # frame 4 is rendered again: now visible
    assert torch.equal(subject, before)
    report = monitor.report()
    assert report["frames"] == 3 and report["empty_frames"] == 0 and report["absent_segments"] == []
    # frames that were never shown are unknown, not absent: runs do not bridge index gaps
    monitor.update(torch.cat([empty(), empty()]), [8, 10])
    assert monitor.report()["absent_segments"] == [[8, 8], [10, 10]]


def test_odd_sizes_other_dtypes_and_validation():
    w, h = 251, 143
    monitor = TrackingMonitor(w, h, fps=30.0)
    a = blob(120.0, 70.0, 30.0, w=w, h=h)
    monitor.update(a.half(), [0])
    monitor.update(a.double(), [1])
    nan = a.clone()
    nan[0, 0, :4, :4] = float("nan")
    monitor.update(nan, [2])
    report = monitor.report()
    assert report["frames"] == 3
    assert report["coverage_mean"] == pytest.approx(math.pi * 900.0 / (w * h), rel=0.04)
    assert all(v == v for v in (report["coverage_mean"], report["coverage_min"], report["coverage_max"]))

    with pytest.raises(TrackingError):
        monitor.update(torch.zeros(1, 1, h, w + 1), [3])
    with pytest.raises(TrackingError):
        monitor.update(torch.zeros(h, w), [3])
    with pytest.raises(TrackingError):
        monitor.update(torch.zeros(2, 1, h, w), [3])
    with pytest.raises(TrackingError):
        monitor.update(torch.zeros(1, 1, h, w), [3], presence=[0.5, 0.5])
    with pytest.raises(ConfigError):
        TrackingMonitor(0, h, 30.0)
    with pytest.raises(ConfigError):
        TrackingMonitor(w, h, 30.0, switch_distance=0.0)
    monitor.update(torch.zeros(0, 1, h, w), [])                  # an empty batch is not an error
    assert monitor.report()["frames"] == 3


def test_tracker_refiner_monitor_chain_on_the_synthetic_test_clip():
    """The three stages as the pipeline chains them, on the shared two-subject test picture: the
    keyer mattes both magenta discs, the tracker keeps the main one, the refiner cleans the matte and
    the monitor finds nothing to complain about."""

    from app.core.config import REFINEMENT_PRESETS
    from app.refinement.refiner import MatteRefiner
    from app.tracking.component_tracker import ComponentSubjectTracker
    from tests.helpers import media
    from tests.helpers.dummy_model import chroma_alpha

    frames = 24
    tracker = ComponentSubjectTracker(W, H)
    refiner = MatteRefiner(REFINEMENT_PRESETS["balanced"], W, H)
    monitor = TrackingMonitor(W, H, fps=29.97)
    for start in range(0, frames, 4):                                        # batches of four frames
        indices = list(range(start, start + 4))
        rgb = torch.stack(
            [torch.from_numpy(media.synth_frame(i, W, H, frames, two_subjects=True)).permute(2, 0, 1) for i in indices]
        ).float() / 255.0
        raw = chroma_alpha(rgb)
        isolated = tracker.apply(raw, indices)
        refined = refiner.refine(isolated, rgb, indices)
        monitor.update(refined, indices, presence=[1.0] * 4)
        for k, i in enumerate(indices):
            truth = torch.from_numpy(media.subject_alpha(i, W, H, frames))[None, None]
            distractor = torch.from_numpy(media.distractor_alpha(W, H))[None, None]
            assert float(raw[k : k + 1][distractor == 1.0].min()) > 0.9                 # the keyer does matte the distractor
            assert bool((refined[k : k + 1][distractor > 0.0] == 0.0).all())            # ... the chain removes it completely
            # the subject stays as the keyer delivered it (the keyer's edge is a little harder than the truth)
            assert float((refined[k : k + 1] - truth).abs().mean()) < 0.01
            assert float((refined[k : k + 1] - isolated[k : k + 1]).abs().mean()) < 0.002
            assert bool((refined[k : k + 1][truth == 1.0] == 1.0).all())                # exactly opaque core
            far = torch.from_numpy(media.subject_alpha(i, W, H, frames, soft=14.0))[None, None] == 0.0
            assert bool((refined[k : k + 1][far] == 0.0).all())                         # exactly empty background
    assert tracker.state == "tracking" and [e["event"] for e in tracker.events()] == ["acquired"]
    report = monitor.report()
    assert report["frames"] == frames and report["warnings"] == []
    assert report["empty_frames"] == 0 and report["suspected_switches"] == [] and report["low_presence_frames"] == 0
    expected = math.pi * media.subject_radius(W, H) ** 2 / (W * H)
    assert report["coverage_mean"] == pytest.approx(expected, rel=0.05)
    stats = refiner.statistics()
    assert stats["frames"] == frames and stats["area_jumps"] == 0
