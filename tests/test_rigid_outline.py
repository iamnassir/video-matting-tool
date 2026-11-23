"""The outline guard of the SAM 2 + MatAnyone hybrid and the rigid outline of a whole clip.

Everything here is plain NumPy / OpenCV: no model, no GPU.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.models.hybrid.outline import guard_matte, solid_outline
from app.tracking.rigid import OutlineMosaic, RigidTracker, scale_affine

LIMITS = dict(margin=4.0, rim=40.0, sliver=6.0)


def body(height=300, width=400):
    """A solid block with one window well inside it."""

    outline = np.zeros((height, width), bool)
    outline[40:260, 50:350] = True
    alpha = outline.astype(np.float32)
    alpha[110:200, 130:270] = 0.0                      # the window: 70 px from the top edge
    return outline, alpha


# --------------------------------------------------------------------------- outline and guard


def test_solid_outline_fills_what_is_enclosed_only():
    mask = np.zeros((60, 80), np.float32)
    mask[10:50, 10:70] = 1.0
    mask[20:30, 20:30] = 0.0                           # enclosed: an opening
    mask[25:35, 60:70] = 0.0                           # open to the outside: part of the shape
    out = solid_outline(mask)
    assert out.dtype == np.bool_
    assert out[25, 25] and not out[30, 65]
    assert not out[0, 0] and out[15, 15]
    # A subject that touches the frame border encloses nothing through the border.
    edge = np.zeros((40, 40), bool)
    edge[:, :10] = True
    edge[:, 30:] = True
    assert not solid_outline(edge)[20, 20]


def test_guard_removes_everything_outside_the_outline():
    outline, alpha = body()
    alpha[5:30, 5:395] = 1.0                           # something the matte took outside
    alpha[100:150, 360:395] = 1.0
    out = guard_matte(alpha, outline, **LIMITS)
    assert out[10, 200] == 0.0 and out[120, 380] == 0.0
    assert out[150, 80] == 1.0                         # the subject itself passes unchanged
    assert float(out[~cv2.dilate(outline.astype(np.uint8), np.ones((19, 19), np.uint8)).astype(bool)].max()) == 0.0


def test_guard_keeps_a_window_open():
    outline, alpha = body()
    out = guard_matte(alpha, outline, **LIMITS)
    assert float(out[110:200, 130:270].max()) == 0.0
    # ... also when the window comes close to the edge (inside the rim zone) but reaches deeper
    alpha[60:200, 130:270] = 0.0                       # now 20 px from the top edge
    out = guard_matte(alpha, outline, **LIMITS)
    assert float(out[64:196, 134:266].max()) == 0.0


def test_guard_restores_a_piece_dropped_at_the_rim():
    outline, alpha = body()
    alpha[40:70, 100:300] = 0.0                        # the matte lost a 30 px strip under the top edge
    out = guard_matte(alpha, outline, **LIMITS)
    assert float(out[40:70, 100:300].min()) == 1.0
    assert float(out[110:200, 130:270].max()) == 0.0   # the window is still open


def test_guard_makes_a_faint_patch_in_the_rim_solid():
    outline, alpha = body()
    alpha[55:65, 150:200] = 0.6                        # half-transparent pocket, not touching the edge
    alpha[50:58, 280:300] = 0.0                        # enclosed hole inside the rim zone
    out = guard_matte(alpha, outline, **LIMITS)
    assert float(out[55:65, 150:200].min()) == 1.0
    assert float(out[50:58, 280:300].min()) == 1.0


def test_guard_leaves_a_thin_gap_at_the_edge_to_the_matte():
    outline, alpha = body()
    alpha[40:43, 50:350] = 0.0                         # the matte's own edge lies 3 px inside the outline
    out = guard_matte(alpha, outline, **LIMITS)
    assert float(out[40:43, 60:340].max()) == 0.0
    assert float(out[43:60, 60:340].min()) == 1.0


def test_guard_closes_a_broken_window_only_along_the_edge():
    outline, alpha = body()
    alpha[40:110, 190:210] = 0.0                       # the matte connected the window to the outside
    out = guard_matte(alpha, outline, **LIMITS)
    assert float(out[44:78, 192:208].min()) == 1.0     # closed within the rim ...
    assert float(out[120:190, 140:260].max()) == 0.0   # ... the window itself stays open


def test_guard_without_rim_repair_and_with_an_empty_outline():
    outline, alpha = body()
    alpha[40:70, 100:300] = 0.0
    out = guard_matte(alpha, outline, margin=4.0, rim=0.0, sliver=6.0)
    assert float(out[40:70, 100:300].max()) == 0.0
    assert float(guard_matte(alpha, np.zeros_like(outline), **LIMITS).max()) == 0.0


# --------------------------------------------------------------------------- rigid motion


def textured_scene(rng, height=270, width=480):
    """A textured object on a canvas and a function that renders it moved, over a changing background."""

    canvas = cv2.GaussianBlur(rng.integers(0, 255, (height, width), dtype=np.uint8), (0, 0), 1.2)
    canvas = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX)
    shape = np.zeros((height, width), np.uint8)
    cv2.ellipse(shape, (width // 2, height // 2), (150, 80), 0, 0, 360, 255, -1)

    def render(motion, seed):
        back = cv2.GaussianBlur(np.random.default_rng(seed).integers(0, 255, (height, width), dtype=np.uint8), (0, 0), 3.0)
        moved = cv2.warpAffine(canvas, motion[:2], (width, height), flags=cv2.INTER_LINEAR)
        where = cv2.warpAffine(shape, motion[:2], (width, height), flags=cv2.INTER_NEAREST) > 0
        return np.where(where, moved, back).astype(np.uint8), where

    return shape > 0, render


def motion_at(step):
    """A drift with a slow zoom and a slight turn."""

    angle = np.deg2rad(0.08 * step)
    scale = 1.0 + 0.0015 * step
    cos, sin = scale * np.cos(angle), scale * np.sin(angle)
    centre = np.array([240.0, 135.0])
    linear = np.array([[cos, -sin], [sin, cos]])
    shift = centre - linear @ centre + np.array([1.4 * step, -0.5 * step])
    return np.array([[cos, -sin, shift[0]], [sin, cos, shift[1]], [0.0, 0.0, 1.0]])


def test_rigid_tracker_follows_an_object_without_drift():
    rng = np.random.default_rng(7)
    shape, render = textured_scene(rng)
    first, _ = render(np.eye(3), 0)
    inner = cv2.erode(shape.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    tracker = RigidTracker(first, inner)
    assert tracker.initial > 50 and not tracker.lost
    corners = np.array([[100.0, 60.0, 1.0], [380.0, 60.0, 1.0], [100.0, 210.0, 1.0], [380.0, 210.0, 1.0]])
    worst = 0.0
    for step in range(1, 61):
        frame, _ = render(motion_at(step), step)
        found = tracker.step(frame)
        assert found is not None, step
        error = np.abs(corners @ found[:2].T - corners @ motion_at(step)[:2].T).max()
        worst = max(worst, float(error))
    assert worst < 1.0                                  # sixty frames on, still within a pixel
    assert tracker.anchored_steps > 40                  # measured against the reference frame, not chained


def test_rigid_tracker_reports_when_the_object_is_gone():
    rng = np.random.default_rng(3)
    shape, render = textured_scene(rng)
    first, _ = render(np.eye(3), 0)
    tracker = RigidTracker(first, cv2.erode(shape.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    assert tracker.step(render(motion_at(1), 1)[0]) is not None
    noise = np.random.default_rng(99).integers(0, 255, first.shape, dtype=np.uint8)
    assert tracker.step(noise) is None and tracker.lost
    assert tracker.step(render(motion_at(2), 2)[0]) is None          # and stays lost
    # Nothing to hold on to at all: lost from the start.
    blank = RigidTracker(np.full_like(first, 128), np.ones(first.shape, bool))
    assert blank.lost and blank.step(first) is None


def test_scale_affine_is_the_same_motion_at_another_size():
    motion = motion_at(20)
    half = scale_affine(motion, 0.5)
    point = np.array([120.0, 80.0, 1.0])
    assert np.allclose((half @ np.array([60.0, 40.0, 1.0]))[:2], 0.5 * (motion @ point)[:2])


# --------------------------------------------------------------------------- one outline for the clip


def moving_outlines(count=60, size=(180, 320), intruder_frames=(), wobble=0.0):
    height, width = size
    base = np.zeros((height, width), np.uint8)
    cv2.rectangle(base, (60, 50), (300, 150), 255, -1)           # the right part leaves the reference frame later
    cv2.circle(base, (110, 50), 30, 255, -1)
    outlines, motions = [], []
    for step in range(count):
        motion = np.array([[1.0, 0.0, -1.5 * step], [0.0, 1.0, 0.3 * step], [0.0, 0.0, 1.0]])
        # the object is wider than the reference frame shows: draw it on a wide canvas first
        wide = np.zeros((height, width + 200), np.uint8)
        wide[:, :width] = base
        cv2.rectangle(wide, (300, 50), (420, 150), 255, -1)
        shown = cv2.warpAffine(wide, motion[:2], (width, height), flags=cv2.INTER_NEAREST) > 0
        if wobble:
            radius = int(round(wobble * (1 + np.sin(step))))
            if radius:
                shown = cv2.dilate(shown.astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) > 0
        if step in intruder_frames:
            shown[5:50, 150:300] = True                           # a van behind the roof
        outlines.append(shown)
        motions.append(motion)
    return outlines[0].copy() if 0 not in intruder_frames else base > 0, outlines, motions


def test_mosaic_outvotes_a_passing_intruder_and_covers_what_enters_later():
    reference, outlines, motions = moving_outlines(intruder_frames=range(20, 32))
    shape = OutlineMosaic((180, 320))
    result = shape.build(outlines, motions, reference)
    assert result is not None and result.agreement > 0.93
    assert result.mosaic.shape[1] > 320                           # larger than one frame
    carried = shape.carry(result, motions[25])
    assert not carried[5:45, 160:290].any()                       # the intruder is gone
    clean = moving_outlines()[1][25]
    union = (carried | clean).sum()
    assert (carried & clean).sum() / union > 0.97
    # A late frame shows a part of the object the reference frame never did: this point lies 388 px
    # from the left in the reference frame's coordinates, and that frame is 320 px wide.
    late = shape.carry(result, motions[59])
    assert late[100, 300] and moving_outlines()[1][59][100, 300]


def test_mosaic_keeps_the_given_outline_on_the_reference_frame():
    reference, outlines, motions = moving_outlines()
    given = reference.copy()
    given[60:80, 150:200] = False                                 # the user's outline differs from the vote
    shape = OutlineMosaic((180, 320))
    result = shape.build(outlines, motions, given)
    assert not shape.carry(result, motions[0])[70, 175]
    assert shape.carry(result, motions[0])[120, 175]


def test_mosaic_shows_when_the_shape_is_not_rigid():
    reference, outlines, motions = moving_outlines(wobble=9.0)
    result = OutlineMosaic((180, 320)).build(outlines, motions, reference)
    assert result is not None and result.agreement < 0.93


def test_mosaic_gives_up_when_the_subject_wanders_too_far():
    reference, outlines, motions = moving_outlines(count=4)
    motions[3] = np.array([[1.0, 0.0, -5000.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert OutlineMosaic((180, 320)).build(outlines, motions, reference) is None
    assert OutlineMosaic((180, 320)).build([], [], reference) is None
