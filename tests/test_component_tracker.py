"""Tests for the connected-component subject tracker (app/tracking/component_tracker.py).

The mattes are synthetic blobs with soft edges; "kept" always means bit-identical to the blob that
was put in, "removed" means exactly zero.
"""

from __future__ import annotations

import json
import time
from typing import List, Sequence, Tuple

import numpy as np
import pytest
import torch

from app.core.errors import ConfigError, SubjectError, SubjectNotFoundError, VmtError
from app.tracking.component_tracker import ComponentSubjectTracker, TrackingError

# 320 x 180 is reduced to a 256 x 144 working map (ratio 1.25: cells and pixels do not line up, which
# is exactly what the exactness guarantees have to survive).
W, H = 320, 180


def blob(cx: float, cy: float, radius: float, soft: float = 2.0, w: int = W, h: int = H) -> torch.Tensor:
    """[1, 1, h, w] disc: exactly 1 inside, exactly 0 outside, linear ramp of ``2 * soft`` px."""

    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    d = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return ((radius - d) / (2.0 * soft) + 0.5).clamp(0.0, 1.0)[None, None]


def empty(w: int = W, h: int = H) -> torch.Tensor:
    return torch.zeros(1, 1, h, w)


PRIMARY = dict(cx=150.0, cy=95.0, radius=42.0)       # large, central
OTHER = dict(cx=285.0, cy=30.0, radius=14.0)         # small, in the corner


def two_blobs() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    primary, other = blob(**PRIMARY), blob(**OTHER)
    return torch.maximum(primary, other), primary, other


def event_names(tracker: ComponentSubjectTracker) -> List[str]:
    return [e["event"] for e in tracker.events()]


# --------------------------------------------------------------------------- acquisition


def test_auto_keeps_only_the_primary_subject_and_leaves_it_untouched():
    both, primary, other = two_blobs()
    before = both.clone()
    tracker = ComponentSubjectTracker(W, H)
    assert tracker.state == "searching" and tracker.events() == []
    for i in range(4):
        out = tracker.apply(both, [i])
        assert torch.equal(both, before)                         # input untouched
        assert out.dtype == torch.float32 and out.shape == both.shape
        assert torch.equal(out, primary)                         # kept values bit-identical, the other blob exactly 0
        assert tracker.state == "tracking"
    events = tracker.events()
    assert [e["event"] for e in events] == ["acquired"]
    assert events[0]["frame"] == 0 and events[0]["mode"] == "auto"
    x1, y1, x2, y2 = events[0]["bbox"]                           # full-resolution pixels around the primary blob
    assert x1 <= 150 - 40 and x2 >= 150 + 40 and y1 <= 95 - 40 and y2 >= 95 + 40
    assert x1 >= 150 - 50 and x2 <= 150 + 50
    assert 0.08 < events[0]["area"] < 0.14                       # pi * 42^2 / (320 * 180) = 0.096
    json.dumps(events)


def test_single_subject_is_a_no_op_and_empty_frames_stay_empty():
    primary = blob(**PRIMARY)
    tracker = ComponentSubjectTracker(W, H)
    assert torch.equal(tracker.apply(empty(), [0]), empty())     # nothing there yet
    assert tracker.state == "searching"
    out = tracker.apply(primary, [1])
    assert out is primary                                        # only the subject: the input itself comes back
    assert tracker.state == "tracking" and tracker.events()[0]["frame"] == 1


@pytest.mark.parametrize("size", [(320, 180), (960, 540), (251, 143), (200, 120), (180, 320)])
def test_kept_and_removed_regions_are_exact_at_any_resolution(size):
    w, h = size
    sx, sy = w / 320.0, h / 180.0
    s = min(sx, sy)
    primary = blob(150 * sx, 95 * sy, 42 * s, soft=2.0, w=w, h=h)
    # a faint halo around the subject (soft hair, the tail of motion blur) below the component threshold
    halo = 0.25 * blob(150 * sx, 95 * sy, 47 * s, soft=2.0, w=w, h=h)
    subject = torch.maximum(primary, halo)
    other = blob(285 * sx, 30 * sy, 14 * s, w=w, h=h)
    close = blob((150 + 42 + 9) * sx + 8, 150 * sy, 7 * s, w=w, h=h)            # a second blob only a few px away
    ghost = 0.2 * blob(40 * sx, 150 * sy, 12 * s, w=w, h=h)                     # faint matte far from everything
    scene = torch.maximum(torch.maximum(torch.maximum(subject, other), close), ghost)
    before = scene.clone()
    tracker = ComponentSubjectTracker(w, h)
    out = tracker.apply(scene, [0])
    assert tracker.state == "tracking" and torch.equal(scene, before)
    solid_subject = subject > 0.3
    assert torch.equal(out[solid_subject], scene[solid_subject])                # multiplied by exactly 1.0
    assert bool((out[other > 0.3] == 0.0).all()) and bool((out[close > 0.3] == 0.0).all())   # exactly 0
    assert bool((out[other > 0.0] == 0.0).all())                                # the fringe goes with its blob
    assert bool((out[ghost > 0.0] == 0.0).all())                                # faint matte elsewhere is removed
    assert bool((out <= scene).all()) and float(out.min()) >= 0.0               # the tracker only ever removes
    # The subject's own faint halo survives untouched wherever no removed blob is nearer.
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    away_from_close = (torch.sqrt((xx - ((150 + 42 + 9) * sx + 8)) ** 2 + (yy - 150 * sy) ** 2) > 40 * s)[None, None]
    halo_only = (halo > 0.0) & (primary == 0.0) & away_from_close
    assert int(halo_only.sum()) > 50 and torch.equal(out[halo_only], scene[halo_only])


def test_faint_matte_outside_the_subject_zone_is_removed_without_touching_the_subject():
    primary = blob(**PRIMARY)
    haze = torch.zeros(1, 1, H, W)
    haze[..., 150:170, 10:60] = 0.02                             # far away, never a component
    tracker = ComponentSubjectTracker(W, H)
    out = tracker.apply(torch.maximum(primary, haze), [0])
    assert torch.equal(out, primary)
    # haze right next to the subject belongs to its fringe zone and stays
    near = torch.zeros(1, 1, H, W)
    near[..., 90:100, 192:197] = 0.02                            # just outside the blob's edge at x = 192
    scene = torch.maximum(primary, near)
    assert tracker.apply(scene, [1]) is scene


def test_seed_box_selects_the_small_blob_instead():
    both, primary, other = two_blobs()
    tracker = ComponentSubjectTracker(W, H)
    tracker.reset(seed_box=(265.0, 10.0, 305.0, 50.0))
    for i in range(3):
        assert torch.equal(tracker.apply(both, [i]), other)
    assert tracker.state == "tracking"
    assert tracker.events()[0]["mode"] == "seed"
    # a sloppy box that covers the blob only partly still selects it
    tracker.reset(seed_box=(280.0, 0.0, 320.0, 28.0))
    assert torch.equal(tracker.apply(both, [0]), other)


def test_seed_mask_and_points_select_the_subject():
    both, primary, other = two_blobs()
    tracker = ComponentSubjectTracker(W, H)

    tracker.reset(seed_mask=(other[0, 0] > 0.5).numpy())                       # bool mask
    assert torch.equal(tracker.apply(both, [0]), other)
    tracker.reset(seed_mask=(other[0, 0] * 255).to(torch.uint8).numpy())       # 8-bit mask
    assert torch.equal(tracker.apply(both, [0]), other)
    tracker.reset(seed_mask=primary[0, 0].numpy())                             # float mask
    assert torch.equal(tracker.apply(both, [0]), primary)

    tracker.reset(seed_points=[(285.0, 30.0, 1)])                              # a click on the small blob
    assert torch.equal(tracker.apply(both, [0]), other)
    tracker.reset(seed_points=[(150.0, 95.0, 0)])                              # "not this one": auto picks the other
    assert torch.equal(tracker.apply(both, [0]), other)
    tracker.reset(seed_points=[(150.0, 95.0, 1), (285.0, 30.0, 1)])            # two clicks: both are the subject
    assert tracker.apply(both, [0]) is both

    # a seed that matches nothing keeps searching instead of guessing
    tracker.reset(seed_box=(5.0, 120.0, 40.0, 170.0))
    assert torch.equal(tracker.apply(both, [0]), empty())
    assert tracker.state == "searching"


def test_invalid_seeds_and_arguments_are_rejected():
    tracker = ComponentSubjectTracker(W, H)
    with pytest.raises(SubjectNotFoundError):
        tracker.reset(seed_mask=np.zeros((H, W), np.float32))                  # empty mask
    with pytest.raises(SubjectError):
        tracker.reset(seed_mask=np.zeros((3, H, W), np.float32))
    with pytest.raises(SubjectError):
        tracker.reset(seed_box=(400.0, 10.0, 500.0, 50.0))                     # outside the frame
    with pytest.raises(SubjectError):
        tracker.reset(seed_box=(1.0, 2.0, 3.0))                                # not a box
    with pytest.raises(SubjectError):
        tracker.reset(seed_points=[(10.0, 10.0, 5)])
    with pytest.raises(TrackingError):
        tracker.apply(torch.zeros(1, 1, H, W + 2), [0])
    with pytest.raises(TrackingError):
        tracker.apply(torch.zeros(H, W), [0])
    with pytest.raises(TrackingError):
        tracker.apply(torch.zeros(2, 1, H, W), [0])
    assert issubclass(TrackingError, VmtError)
    for bad in (dict(work_side=4), dict(threshold=0.0), dict(threshold=1.0), dict(min_area=1.5), dict(lost_patience_frames=0)):
        with pytest.raises(ConfigError):
            ComponentSubjectTracker(W, H, **bad)
    with pytest.raises(ConfigError):
        ComponentSubjectTracker(0, H)


# --------------------------------------------------------------------------- lost / re-acquired


def run(tracker: ComponentSubjectTracker, frames: Sequence[torch.Tensor], start: int = 0) -> List[torch.Tensor]:
    return [tracker.apply(a, [start + i]) for i, a in enumerate(frames)]


def test_lost_subject_gives_empty_output_and_the_other_blob_is_never_adopted():
    both, primary, other = two_blobs()
    tracker = ComponentSubjectTracker(W, H, lost_patience_frames=6)           # the search covers the frame after 6 frames
    for out in run(tracker, [both] * 8):
        assert torch.equal(out, primary)

    # The primary disappears for 20 frames while the other blob stays where it is.
    for i in range(8, 28):
        out = tracker.apply(other, [i])
        assert torch.equal(out, empty())                         # nothing is kept
        assert tracker.state == "lost"                           # ... and the bystander is NEVER adopted
    assert event_names(tracker) == ["acquired", "lost"]
    lost = tracker.events()[1]
    assert lost["frame"] == 8 and lost["tracked_frames"] == 8

    # The primary re-appears near its last position.
    back = blob(PRIMARY["cx"] + 10.0, PRIMARY["cy"] - 6.0, PRIMARY["radius"])
    out = tracker.apply(torch.maximum(back, other), [28])
    assert torch.equal(out, back)
    assert tracker.state == "tracking"
    assert event_names(tracker) == ["acquired", "lost", "reacquired"]
    reacquired = tracker.events()[2]
    assert reacquired["frame"] == 28 and reacquired["lost_frames"] == 20
    for out in run(tracker, [torch.maximum(back, other)] * 3, start=29):
        assert torch.equal(out, back)


def test_bystander_is_not_adopted_even_when_it_moves_or_the_wait_is_long():
    both, primary, other = two_blobs()
    tracker = ComponentSubjectTracker(W, H, lost_patience_frames=4)
    run(tracker, [both] * 5)
    # The bystander wanders through the whole frame - right across the place where the subject was.
    for i in range(5, 85):
        t = (i - 5) / 79.0
        wanderer = blob(285.0 - 250.0 * t, 30.0 + 110.0 * np.sin(np.pi * t), 14.0)
        assert torch.equal(tracker.apply(wanderer, [i]), empty())
        assert tracker.state == "lost"
    assert event_names(tracker) == ["acquired", "lost"]


def test_reacquisition_waits_for_the_search_region_to_reach_a_far_component():
    both, primary, other = two_blobs()
    tracker = ComponentSubjectTracker(W, H, lost_patience_frames=20)
    run(tracker, [both] * 4)
    # The subject vanishes and a new blob shows up at the far left edge, about 40 % of the diagonal away.
    far = blob(22.0, 150.0, 20.0)
    scene = torch.maximum(far, other)
    frames_until_found = None
    for i in range(4, 40):
        out = tracker.apply(scene, [i])
        if tracker.state == "tracking":
            frames_until_found = i - 4
            assert torch.equal(out, far)
            break
        assert torch.equal(out, empty())
    # not at once (the search region grows gradually), but well within the patience
    assert frames_until_found is not None and 2 <= frames_until_found <= 12
    assert event_names(tracker) == ["acquired", "lost", "reacquired"]
    assert tracker.events()[2]["distance"] > 0.1


def test_tiny_specks_are_never_acquired():
    both, primary, other = two_blobs()
    tracker = ComponentSubjectTracker(W, H, lost_patience_frames=3)
    speck = blob(150.0, 95.0, 2.0, soft=0.5)                     # far below min_area, right where the subject was
    assert torch.equal(tracker.apply(speck, [0]), empty()) and tracker.state == "searching"
    run(tracker, [both] * 3, start=1)
    for i in range(4, 12):
        assert torch.equal(tracker.apply(torch.maximum(speck, other), [i]), empty())
        assert tracker.state == "lost"


# --------------------------------------------------------------------------- following


def test_steadily_moving_subject_is_followed_past_a_bystander():
    tracker = ComponentSubjectTracker(W, H)
    other = blob(285.0, 150.0, 14.0)
    for i in range(40):
        subject = blob(40.0 + 6.0 * i, 60.0 + 20.0 * np.sin(i / 6.0), 30.0)   # 6 px per frame across the frame
        out = tracker.apply(torch.maximum(subject, other), [i])
        assert torch.equal(out, subject), f"frame {i}"
        assert tracker.state == "tracking"
    assert event_names(tracker) == ["acquired"]


def test_fast_subject_is_followed_within_the_motion_margin_and_found_again_beyond_it():
    tracker = ComponentSubjectTracker(W, H, lost_patience_frames=10)
    other = blob(285.0, 150.0, 14.0)
    subject = blob(60.0, 60.0, 16.0)
    tracker.apply(torch.maximum(subject, other), [0])
    # 40 px in one frame: more than the subject's own diameter, no overlap, but inside the motion margin
    jumped = blob(100.0, 60.0, 16.0)
    assert torch.equal(tracker.apply(torch.maximum(jumped, other), [1]), jumped)
    assert tracker.state == "tracking"
    # 110 px in one frame is not plausible motion: the subject is lost first, then found again
    teleported = blob(210.0, 60.0, 16.0)
    assert torch.equal(tracker.apply(torch.maximum(teleported, other), [2]), empty())
    assert tracker.state == "lost"
    found = False
    for i in range(3, 12):
        out = tracker.apply(torch.maximum(teleported, other), [i])
        if tracker.state == "tracking":
            assert torch.equal(out, teleported)
            found = True
            break
    assert found
    assert event_names(tracker) == ["acquired", "lost", "reacquired"]


def test_skipped_frames_widen_the_motion_margin():
    subject = blob(60.0, 90.0, 16.0)
    other = blob(285.0, 150.0, 14.0)
    far = blob(150.0, 90.0, 16.0)                                # 90 px away
    consecutive = ComponentSubjectTracker(W, H)
    consecutive.apply(torch.maximum(subject, other), [0])
    consecutive.apply(torch.maximum(far, other), [1])
    assert consecutive.state == "lost"                           # too far for one frame
    skipping = ComponentSubjectTracker(W, H)
    skipping.apply(torch.maximum(subject, other), [0])
    assert torch.equal(skipping.apply(torch.maximum(far, other), [5]), far)   # plausible over five frames
    assert skipping.state == "tracking"
    # descending frame numbers (backward tracking) work the same way
    backward = ComponentSubjectTracker(W, H)
    for i, index in enumerate(range(30, 20, -1)):
        moved = blob(60.0 + 6.0 * i, 90.0, 16.0)
        assert torch.equal(backward.apply(torch.maximum(moved, other), [index]), moved)


def test_merged_people_are_kept_together_and_separated_again():
    """The documented limitation: touching mattes are one component.  Both are kept while they
    touch; when they part, the other person is dropped again."""

    tracker = ComponentSubjectTracker(W, H)
    subject = blob(110.0, 90.0, 34.0)

    def visitor(x: float) -> torch.Tensor:
        return blob(x, 90.0, 20.0)

    positions = [250.0, 230.0, 210.0, 190.0, 170.0, 160.0, 160.0, 170.0, 190.0, 210.0, 230.0, 250.0]
    for i, x in enumerate(positions):
        other = visitor(x)
        scene = torch.maximum(subject, other)
        out = tracker.apply(scene, [i])
        touching = x - 110.0 < 34.0 + 20.0 + 2.0
        if touching:
            assert torch.equal(out, scene), f"frame {i}"         # one component: cannot be separated
        else:
            assert torch.equal(out, subject), f"frame {i}"       # apart: only the subject
        assert tracker.state == "tracking"
    assert event_names(tracker) == ["acquired"]


def test_newcomer_beside_the_subject_is_removed_and_never_absorbed():
    """Somebody who enters the picture right next to the subject starts as a speck.  The speck is
    kept (it could be a strand of hair), but it is not remembered as the subject: as soon as it is
    big enough to be someone it is removed, and it is not adopted when the subject leaves."""

    tracker = ComponentSubjectTracker(W, H, lost_patience_frames=4)
    subject = blob(**PRIMARY)                                    # right edge at x = 192
    radii = [0.0, 1.0, 1.0, 1.0, 6.0, 8.0, 10.0, 10.0, 10.0]

    def newcomer_at(radius: float) -> torch.Tensor:
        # grows away from the subject: its near edge stays 9 px off the subject's edge, never touching
        return blob(201.0 + radius, 95.0, radius, soft=0.5) if radius > 0 else empty()

    for i, radius in enumerate(radii):
        newcomer = newcomer_at(radius)
        scene = torch.maximum(subject, newcomer)
        out = tracker.apply(scene, [i])
        if radius <= 1.0:
            assert torch.equal(out, scene), f"frame {i}"         # a speck beside the subject: kept
        else:
            assert torch.equal(out, subject), f"frame {i}"       # somebody: removed
        assert tracker.state == "tracking"
    grown = newcomer_at(10.0)
    for i in range(len(radii), len(radii) + 20):
        assert torch.equal(tracker.apply(grown, [i]), empty())   # the subject left: not adopted
        assert tracker.state == "lost"
    assert event_names(tracker) == ["acquired", "lost"]


def test_detached_fragments_next_to_the_subject_are_kept():
    tracker = ComponentSubjectTracker(W, H)
    subject = blob(**PRIMARY)
    other = blob(**OTHER)
    for i in range(6):
        angle = 0.5 * i - 1.0                                    # a speck 8 px off the edge that moves around it
        strand = blob(150.0 + 50.0 * np.cos(angle), 95.0 + 50.0 * np.sin(angle), 1.0, soft=0.5)
        far_speck = blob(30.0, 160.0, 1.0, soft=0.5)             # a speck that is nowhere near the subject
        scene = torch.maximum(torch.maximum(subject, strand), torch.maximum(other, far_speck))
        out = tracker.apply(scene, [i])
        assert torch.equal(out, torch.maximum(subject, strand)), f"frame {i}"


def test_occluded_subject_is_followed_until_it_is_gone_and_found_when_it_returns():
    """The subject slides behind something (its matte shrinks to nothing), stays hidden and comes
    out again a little further on; a bystander is visible all the time."""

    tracker = ComponentSubjectTracker(W, H, lost_patience_frames=30)
    other = blob(40.0, 40.0, 16.0)
    wall = 170.0                                                 # everything right of x = 170 is hidden
    frames = []
    for i in range(40):
        cx = 60.0 + 6.0 * i                                      # walks to the right, 6 px per frame
        a = blob(cx, 110.0, 28.0)
        a[..., :, int(wall) : int(wall) + 70] = 0.0              # the occluder is 70 px wide
        frames.append(a)
    full_area = float(blob(60.0, 110.0, 28.0).sum())
    states = []
    for i, a in enumerate(frames):
        scene = torch.maximum(a, other)
        out = tracker.apply(scene, [i])
        states.append(tracker.state)
        if tracker.state == "tracking":
            assert torch.equal(out, a), f"frame {i}"             # the visible part of the subject, nothing else
        else:
            assert torch.equal(out, empty()), f"frame {i}"       # hidden, a last sliver, or not found yet: nothing
            # never while much of it is visible (the search region needs a moment to reach the far
            # side of a 70 px wide occluder, hence not simply "nothing visible")
            assert float(a.sum()) < 0.25 * full_area, f"frame {i}"
    assert states[0] == "tracking" and states[-1] == "tracking"
    assert "lost" in states
    assert event_names(tracker) == ["acquired", "lost", "reacquired"]
    hidden = [i for i, a in enumerate(frames) if float(a.max()) == 0.0]
    assert hidden and all(states[i] == "lost" for i in hidden)
    # it is picked up again within a few frames of becoming visible
    reacquired = tracker.events()[2]["frame"]
    assert reacquired - (hidden[-1] + 1) <= 3


def test_batches_are_equivalent_to_single_frames():
    frames = []
    other = blob(285.0, 150.0, 14.0)
    for i in range(12):
        if 5 <= i < 8:
            frames.append(other)                                 # the subject is gone for three frames
        else:
            frames.append(torch.maximum(blob(60.0 + 8.0 * i, 70.0, 26.0), other))
    single = ComponentSubjectTracker(W, H, lost_patience_frames=5)
    expected = torch.cat(run(single, frames))
    batched = ComponentSubjectTracker(W, H, lost_patience_frames=5)
    stacked = torch.cat(frames)
    got = torch.cat([batched.apply(stacked[0:5], [0, 1, 2, 3, 4]), batched.apply(stacked[5:12], list(range(5, 12)))])
    assert torch.equal(got, expected)
    assert batched.events() == single.events()
    assert event_names(single) == ["acquired", "lost", "reacquired"]
    assert bool((expected[5:8] == 0.0).all())


def test_reset_starts_a_new_sequence_and_keeps_the_event_log():
    both, primary, other = two_blobs()
    tracker = ComponentSubjectTracker(W, H)
    run(tracker, [both] * 3)
    tracker.reset(seed_box=(265.0, 10.0, 305.0, 50.0))
    assert tracker.state == "searching"
    assert torch.equal(tracker.apply(both, [0]), other)          # the old subject / negatives are forgotten
    assert event_names(tracker) == ["acquired", "acquired"]
    assert [e["mode"] for e in tracker.events()] == ["auto", "seed"]
    events = tracker.events()
    events.clear()                                               # the returned list is a copy
    assert len(tracker.events()) == 2


def test_half_precision_and_nan_input():
    both, primary, other = two_blobs()
    tracker = ComponentSubjectTracker(W, H)
    out = tracker.apply(both.half(), [0])
    assert out.dtype == torch.float32 and torch.equal(out, primary.half().float())
    broken = both.clone()
    broken[0, 0, :3, :3] = float("nan")                          # NaN compares false: not part of any component
    assert tracker.apply(broken, [1]).shape == both.shape and tracker.state == "tracking"


# --------------------------------------------------------------------------- timing (no assertion on speed)


def test_component_tracker_timing_1080p(record_property):
    """Measure the tracker on 1920x1080 frames on the CPU (two people, one is removed)."""

    w, h = 1920, 1080
    tracker = ComponentSubjectTracker(w, h)
    other = blob(1650.0, 300.0, 120.0, soft=3.0, w=w, h=h)
    frames = 8
    elapsed = 0.0
    for i in range(frames + 1):
        subject = blob(700.0 + 12.0 * i, 560.0, 300.0, soft=3.0, w=w, h=h)
        scene = torch.maximum(subject, other)
        t0 = time.perf_counter()
        out = tracker.apply(scene, [i])
        if i > 0:
            elapsed += time.perf_counter() - t0
        assert torch.equal(out, subject)
    ms = elapsed / frames * 1000.0
    record_property("component_tracker_1080p_cpu_ms_per_frame", round(ms, 1))
    print(f"\n[timing] ComponentSubjectTracker 1920x1080 CPU ({torch.get_num_threads()} threads): {ms:.1f} ms/frame")
    assert tracker.state == "tracking"
