"""Tests for the temporal filter (app/refinement/temporal.py) and the matte refiner
(app/refinement/refiner.py).

Synthetic sequences only: a soft disc ("subject") on a textured picture, with controlled flicker
and controlled motion, so every claim of the filter can be measured.  CPU only, seeded.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import List, Tuple

import pytest
import torch

from app.core.config import REFINEMENT_PRESETS, RefinementParams
from app.core.errors import ConfigError
from app.refinement import ops
from app.refinement.ops import RefinementError
from app.refinement.refiner import MatteRefiner
from app.refinement.temporal import TemporalFilter
from app.utils import torch_ops

H, W = 128, 192

# --------------------------------------------------------------------------- synthetic material


def disc(cy: float, cx: float, radius: float, soft: float = 1.5, h: int = H, w: int = W) -> torch.Tensor:
    """[1, 1, h, w] disc: exactly 1 inside, exactly 0 outside, linear ramp of ``2 * soft`` px."""

    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    d = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return ((radius - d) / (2.0 * soft) + 0.5).clamp(0.0, 1.0)[None, None]


def texture(seed: int, h: int = H, w: int = W, sigma: float = 2.0) -> torch.Tensor:
    """[1, 3, h, w] smooth random colour texture in 0..1."""

    g = torch.Generator().manual_seed(seed)
    t = torch_ops.gaussian_blur(torch.rand(1, 3, h, w, generator=g), sigma)
    lo, hi = t.amin(dim=(2, 3), keepdim=True), t.amax(dim=(2, 3), keepdim=True)
    return (t - lo) / (hi - lo)


def composite(alpha: torch.Tensor, fg: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
    return bg * (1.0 - alpha) + fg * alpha


def static_flicker_sequence(frames: int = 30, amplitude: float = 0.1, seed: int = 7):
    """A still picture whose matte flickers by +-amplitude in the edge band.

    Returns (mattes, picture, clean matte, band mask).
    """

    base = disc(64.0, 96.0, 30.0, soft=2.5)
    band = (base > 0.0) & (base < 1.0)
    picture = composite(base, texture(2) * 0.4 + 0.55, texture(1) * 0.5 + 0.1)
    g = torch.Generator().manual_seed(seed)
    mattes = []
    for _ in range(frames):
        noise = (torch.rand(1, 1, H, W, generator=g) * 2.0 - 1.0) * amplitude
        mattes.append((base + noise * band).clamp(0.0, 1.0))
    return mattes, picture, base, band


def moving_sequence(frames: int = 20, step: int = 6, soft: float = 2.0):
    """A subject that translates ``step`` px per frame over a still background (the picture moves with it)."""

    bg = texture(1) * 0.5 + 0.1
    fg = texture(2) * 0.4 + 0.55
    mattes, pictures = [], []
    for t in range(frames):
        a = disc(64.0, 40.0 + step * t, 26.0, soft=soft)
        mattes.append(a)
        pictures.append(composite(a, torch.roll(fg, shifts=step * t, dims=-1), bg))
    return mattes, pictures


def pan_sequence(frames: int = 20, step: int = 6, amplitude: float = 0.1, seed: int = 3):
    """A camera pan: picture and matte translate together; the matte flickers in its edge band.

    Returns (noisy mattes, clean mattes, pictures).
    """

    wide = W + step * frames + 8
    matte_wide = disc(64.0, 60.0, 26.0, soft=1.0, w=wide)
    scene = composite(matte_wide, texture(12, w=wide) * 0.3 + 0.65, texture(11, w=wide) * 0.6 + 0.2)
    g = torch.Generator().manual_seed(seed)
    noisy, clean, pictures = [], [], []
    for t in range(frames):
        x0 = step * (frames - 1 - t)
        a = matte_wide[..., x0 : x0 + W].clone()
        band = torch_ops.max_filter(((a > 0.0) & (a < 1.0)).float(), 1) > 0.0
        noise = (torch.rand(1, 1, H, W, generator=g) * 2.0 - 1.0) * amplitude
        noisy.append((a + noise * band).clamp(0.0, 1.0))
        clean.append(a)
        pictures.append(scene[..., x0 : x0 + W].clone())
    return noisy, clean, pictures


def flicker_of(sequence: List[torch.Tensor], mask: torch.Tensor) -> float:
    """Mean absolute frame-to-frame change inside ``mask``."""

    total = sum(float((sequence[t] - sequence[t - 1]).abs()[mask].mean()) for t in range(1, len(sequence)))
    return total / (len(sequence) - 1)


def plain_ema(sequence: List[torch.Tensor], strength: float) -> List[torch.Tensor]:
    out, prev = [], None
    for a in sequence:
        prev = a if prev is None else strength * prev + (1.0 - strength) * a
        out.append(prev)
    return out


def mean_error(outputs: List[torch.Tensor], targets: List[torch.Tensor], skip: int = 1) -> float:
    pairs = list(zip(outputs, targets))[skip:]
    return sum(float((o - t).abs().mean()) for o, t in pairs) / len(pairs)


# =========================================================================== TemporalFilter


def test_temporal_strength_zero_is_an_exact_pass_through():
    mattes, picture, _, _ = static_flicker_sequence(frames=6)
    tf = TemporalFilter(W, H, 0.0)
    assert tf.motion_map is None
    for i, a in enumerate(mattes):
        before = a.clone()
        out = tf.apply(a, picture)
        assert out is a and torch.equal(a, before)              # the very same tensor, untouched
        # the analysis keeps running: the refiner and the statistics need the motion map
        assert (tf.motion_map is None) == (i == 0)
    assert tf.motion_map.shape == (1, 1, *tf.analysis_size)
    assert float(tf.motion_map.max()) == 0.0                    # a still picture


def test_temporal_first_frame_and_reset_pass_through():
    mattes, picture, _, _ = static_flicker_sequence(frames=6)
    tf = TemporalFilter(W, H, 0.7)
    assert tf.apply(mattes[0], picture) is mattes[0]            # first frame: unchanged
    second = tf.apply(mattes[1], picture)
    assert not torch.equal(second, mattes[1])                   # second frame: blended with the first
    tf.reset()
    assert tf.motion_map is None and tf.motion_blur_map is None
    assert tf.apply(mattes[2], picture) is mattes[2]            # first frame after a reset: unchanged
    # ... and the state really restarted: the next output only depends on frames 2 and 3
    fresh = TemporalFilter(W, H, 0.7)
    fresh.apply(mattes[2], picture)
    assert torch.equal(tf.apply(mattes[3], picture), fresh.apply(mattes[3], picture))


@pytest.mark.parametrize("compensate", [False, True])
def test_temporal_reduces_flicker_of_a_static_scene_by_more_than_half(compensate):
    mattes, picture, base, band = static_flicker_sequence(frames=30, amplitude=0.1)
    inputs = [m.clone() for m in mattes]
    tf = TemporalFilter(W, H, 0.6, motion_compensation=compensate)
    g = torch.Generator().manual_seed(21)
    outputs = []
    for a in mattes:
        # a little sensor noise in the picture must not be mistaken for motion
        noisy_picture = (picture + 0.008 * torch.randn(1, 3, H, W, generator=g)).clamp(0.0, 1.0)
        outputs.append(tf.apply(a, noisy_picture))
    assert all(torch.equal(a, b) for a, b in zip(mattes, inputs))        # inputs untouched

    raw, filtered = flicker_of(mattes, band), flicker_of(outputs, band)
    assert raw > 0.05
    assert filtered < 0.5 * raw                                  # theory for strength 0.6: about 0.33
    # the filtered matte is also closer to the clean one
    assert mean_error(outputs, [base] * 30, skip=5) < 0.6 * mean_error(mattes, [base] * 30, skip=5)
    # a mild setting still helps, monotonically
    mild = TemporalFilter(W, H, 0.3, motion_compensation=compensate)
    mild_out = [mild.apply(a, picture) for a in mattes]
    assert filtered < flicker_of(mild_out, band) < 0.75 * raw

    # exact 0 / 1 preservation: stable opaque and transparent pixels never change at all
    opaque, transparent = base == 1.0, base == 0.0
    assert int(opaque.sum()) > 1000 and int(transparent.sum()) > 1000
    for out in outputs:
        assert bool((out[opaque] == 1.0).all()) and bool((out[transparent] == 0.0).all())
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


@pytest.mark.parametrize("compensate", [False, True])
def test_temporal_constant_matte_in_a_static_scene_never_drifts(compensate):
    a = disc(64.0, 96.0, 30.0, soft=3.0)
    picture = composite(a, texture(2), texture(1))
    tf = TemporalFilter(W, H, 0.9, motion_compensation=compensate)
    for _ in range(25):
        assert torch.equal(tf.apply(a, picture), a)             # bit-exact, however long it runs


@pytest.mark.parametrize("strength", [0.3, 0.6, 0.9])
def test_temporal_does_not_trail_behind_a_moving_subject(strength):
    mattes, pictures = moving_sequence(frames=20, step=6)
    ema_lag = mean_error(plain_ema(mattes, strength), mattes)
    assert ema_lag > 0.005                                       # a plain recursive filter does trail

    tf = TemporalFilter(W, H, strength)
    outputs = [tf.apply(a, p) for a, p in zip(mattes, pictures)]
    lag = mean_error(outputs, mattes)
    assert lag < ema_lag / 3.0                                   # requirement
    assert lag < ema_lag / 20.0                                  # what the motion gate really achieves
    assert max(float((o - a).abs().max()) for o, a in zip(outputs, mattes)) < 0.05      # no ghost anywhere

    # Even without the image-motion gate the matte-change gate alone prevents ghosting of the bulk.
    blind = TemporalFilter(W, H, strength, motion_aware=False, preserve_motion_blur=False)
    blind_out = [blind.apply(a, p) for a, p in zip(mattes, pictures)]
    assert lag <= mean_error(blind_out, mattes) < ema_lag / 3.0

    # The motion map marks the moving subject and nothing far away from it.
    motion = tf.motion_map
    assert motion is not None and float(motion.max()) > 0.9
    assert float(motion[..., :, :8].max()) == 0.0


def test_temporal_motion_compensation_runs_and_does_not_degrade_motion():
    mattes, pictures = moving_sequence(frames=20, step=6)
    plain = TemporalFilter(W, H, 0.6)
    comp = TemporalFilter(W, H, 0.6, motion_compensation=True)
    lag_plain = mean_error([plain.apply(a, p) for a, p in zip(mattes, pictures)], mattes)
    comp_out = [comp.apply(a, p) for a, p in zip(mattes, pictures)]
    lag_comp = mean_error(comp_out, mattes)
    ema_lag = mean_error(plain_ema(mattes, 0.6), mattes)
    assert lag_comp < ema_lag / 3.0
    assert lag_comp <= lag_plain + 0.002 * ema_lag + 1e-4        # compensation does not add lag
    assert all(float(o.min()) >= 0.0 and float(o.max()) <= 1.0 for o in comp_out)
    # stable plateaus stay exact under warping
    always_one = torch.stack(mattes).amin(dim=0) == 1.0
    always_zero = torch.stack(mattes).amax(dim=0) == 0.0
    for out in comp_out:
        assert bool((out[always_one] == 1.0).all()) and bool((out[always_zero] == 0.0).all())


def test_temporal_motion_compensation_smooths_along_a_camera_pan():
    """Everything moves 6 px per frame.  Without compensation nothing may be blended (and nothing
    is); with compensation the flicker is smoothed along the motion."""

    noisy, clean, pictures = pan_sequence(frames=20, step=6, amplitude=0.1)
    raw_error = mean_error(noisy, clean, skip=3)

    gated = TemporalFilter(W, H, 0.6, preserve_motion_blur=False)
    gated_error = mean_error([gated.apply(a, p) for a, p in zip(noisy, pictures)], clean, skip=3)
    assert gated_error == pytest.approx(raw_error, rel=0.02)     # moving picture: passes through, no trails

    comp = TemporalFilter(W, H, 0.6, motion_compensation=True, preserve_motion_blur=False)
    comp_error = mean_error([comp.apply(a, p) for a, p in zip(noisy, pictures)], clean, skip=3)
    assert comp_error < 0.9 * raw_error                          # smoothed along the motion

    # a plain recursive filter makes it worse (it trails) - compensation is not the same as blending
    blind = TemporalFilter(W, H, 0.6, motion_aware=False, preserve_motion_blur=False)
    blind_error = mean_error([blind.apply(a, p) for a, p in zip(noisy, pictures)], clean, skip=3)
    assert blind_error > 1.3 * raw_error


def test_temporal_motion_blur_zone_is_never_blended():
    """A wide semi-transparent region in a moving picture area keeps its current values."""

    h, w, step, frames = 96, 160, 5, 8
    ramp = torch.linspace(0.0, 1.0, 40).view(1, 1, 1, 40)
    mattes, pictures = [], []
    g = torch.Generator().manual_seed(5)
    for t in range(frames):
        a = torch.zeros(1, 1, h, w)
        x0 = 20 + step * t
        a[..., 20:76, x0 : x0 + 40] = ramp                        # 40 px wide motion-blur-like ramp ...
        a[..., 20:76, x0 + 40 : x0 + 70] = 1.0                    # ... in front of a solid body
        noise = 0.05 * (torch.rand(1, 1, h, w, generator=g) - 0.5) * ((a > 0) & (a < 1))
        mattes.append((a + noise).clamp(0.0, 1.0))
        pictures.append(composite(a, torch.roll(texture(2, h, w), step * t, -1), texture(1, h, w) * 0.4))
    keep = TemporalFilter(w, h, 0.8, motion_aware=False, preserve_motion_blur=True)
    blend = TemporalFilter(w, h, 0.8, motion_aware=False, preserve_motion_blur=False)
    keep_out = [keep.apply(a, p) for a, p in zip(mattes, pictures)]
    blend_out = [blend.apply(a, p) for a, p in zip(mattes, pictures)]
    assert keep.motion_blur_map is not None and float(keep.motion_blur_map.max()) > 0.9
    assert blend.motion_blur_map is None

    def change(outputs: List[torch.Tensor]) -> float:
        total = 0.0
        for t in range(2, frames):
            x0 = 20 + step * t
            region = (slice(None), slice(None), slice(26, 70), slice(x0 + 8, x0 + 32))      # inside the ramp
            total += float((outputs[t][region] - mattes[t][region]).abs().mean())
        return total / (frames - 2)

    assert change(blend_out) > 0.05                              # a recursive filter smears and delays the ramp
    assert change(keep_out) < 1e-3                               # protected: the blur keeps its current shape
    assert change(keep_out) < change(blend_out) / 50.0


def test_temporal_heals_after_a_non_finite_frame_and_validates_input():
    a = disc(64.0, 96.0, 30.0)
    picture = texture(1)
    tf = TemporalFilter(W, H, 0.5)
    tf.apply(a, picture)
    bad = a.clone()
    bad[0, 0, 10:20, 10:20] = float("nan")
    tf.apply(bad, picture)
    assert not bool(torch.isnan(tf.apply(a, picture)).any())     # the NaN did not survive in the recursion
    assert torch.equal(tf.apply(a, picture), a)

    with pytest.raises(RefinementError):
        tf.apply(torch.cat([a, a]), torch.cat([picture, picture]))           # one frame at a time
    with pytest.raises(RefinementError):
        tf.apply(a[..., :-1], picture)
    with pytest.raises(RefinementError):
        tf.apply(a, picture[:, :2])
    for bad_strength in (-0.1, 1.0, float("nan")):
        with pytest.raises(ConfigError):
            TemporalFilter(W, H, bad_strength)
    with pytest.raises(ConfigError):
        TemporalFilter(0, H, 0.3)
    # tiny and odd frames work, also with optical flow
    tiny = TemporalFilter(23, 17, 0.5, motion_compensation=True)
    g = torch.Generator().manual_seed(1)
    for _ in range(4):
        out = tiny.apply(torch.rand(1, 1, 17, 23, generator=g), torch.rand(1, 3, 17, 23, generator=g))
    assert out.shape == (1, 1, 17, 23) and tiny.analysis_size == (17, 23)
    assert TemporalFilter(1920, 1080, 0.3).analysis_size == (270, 480)       # 1080p is analysed at 1/4 scale
    assert TemporalFilter(3840, 2160, 0.3).analysis_size == (270, 480)


# =========================================================================== MatteRefiner


def flicker_batch(frames: int = 12, seed: int = 7) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mattes, picture, base, _ = static_flicker_sequence(frames=frames, seed=seed)
    return torch.cat(mattes), picture.expand(frames, -1, -1, -1).clone(), base


def test_refiner_disabled_is_the_identity():
    alpha, rgb, _ = flicker_batch(5)
    alpha[0, 0, 0, 0] = 1.3                                      # out-of-range samples are clamped
    alpha[0, 0, 0, 1] = -0.2
    before = alpha.clone()
    refiner = MatteRefiner(REFINEMENT_PRESETS["off"], W, H)
    out = refiner.refine(alpha, rgb, range(5))
    assert torch.equal(alpha, before)                            # input untouched
    assert out is not alpha and out.dtype == torch.float32 and out.shape == alpha.shape
    assert torch.equal(out, before.clamp(0.0, 1.0))
    stats = refiner.statistics()
    assert stats["frames"] == 5 and stats["enabled"] is False
    assert stats["flicker_mean"] == pytest.approx(stats["raw_flicker_mean"])
    assert stats["components_removed"] == 0 and stats["holes_filled"] == 0


@pytest.mark.parametrize("name", sorted(REFINEMENT_PRESETS))
@pytest.mark.parametrize("scaled", [True, False])
def test_refiner_presets_run_on_a_batch_and_keep_exact_plateaus(name, scaled):
    params = replace(REFINEMENT_PRESETS[name], resolution_scaling=scaled)
    alpha, rgb, base = flicker_batch(4)
    before_alpha, before_rgb = alpha.clone(), rgb.clone()
    refiner = MatteRefiner(params, W, H)
    out = refiner.refine(alpha, rgb, [0, 1, 2, 3])
    assert torch.equal(alpha, before_alpha) and torch.equal(rgb, before_rgb)
    assert out.shape == alpha.shape and out.dtype == torch.float32 and out.device == alpha.device
    assert bool(torch.isfinite(out).all())
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    core = disc(64.0, 96.0, 30.0 - 12.0, soft=0.01) == 1.0       # well inside the subject
    outside = disc(64.0, 96.0, 30.0 + 12.0, soft=0.01) == 0.0    # well outside
    for i in range(4):
        assert bool((out[i : i + 1][core] == 1.0).all())         # exactly opaque: source pixels are copied
        assert bool((out[i : i + 1][outside] == 0.0).all())      # exactly transparent: constant background
    assert refiner.statistics()["frames"] == 4


def test_refiner_levels_make_near_plateaus_exact():
    """Model mattes are rarely exactly 0 / 1: the levels step must produce the exact values."""

    g = torch.Generator().manual_seed(9)
    base = disc(64.0, 96.0, 30.0, soft=2.0)
    haze = 0.012 * torch.rand(3, 1, H, W, generator=g)
    alpha = (base * (1.0 - haze) + (1.0 - base) * haze).clamp(0.0, 1.0)        # 0..0.012 outside, 0.988..1 inside
    assert int((alpha == 0.0).sum()) < 100 and int((alpha == 1.0).sum()) < 100
    rgb = composite(base, texture(2), texture(1)).expand(3, -1, -1, -1)
    out = MatteRefiner(REFINEMENT_PRESETS["balanced"], W, H).refine(alpha, rgb, [0, 1, 2])
    assert bool((out[:, :, base[0, 0] == 1.0] == 1.0).all())
    assert bool((out[:, :, base[0, 0] == 0.0] == 0.0).all())


def test_refiner_turns_a_mask_into_a_soft_matte():
    hard = (disc(64.0, 96.0, 30.0) > 0.5).float()
    rgb = composite(disc(64.0, 96.0, 31.5), texture(2) * 0.3 + 0.65, texture(1) * 0.3 + 0.05)
    params = REFINEMENT_PRESETS["balanced"]

    def intermediate(x: torch.Tensor) -> int:
        return int(torch.unique(x[(x > 0.0) & (x < 1.0)]).numel())

    assert intermediate(hard) == 0
    as_mask = MatteRefiner(params, W, H, alpha_is_mask=True).refine(hard, rgb, [0])
    assert intermediate(as_mask) > 50                            # a soft edge appeared
    assert bool((as_mask[disc(64.0, 96.0, 18.0, soft=0.01) == 1.0] == 1.0).all())      # far interior: exactly 1
    assert bool((as_mask[disc(64.0, 96.0, 42.0, soft=0.01) == 0.0] == 0.0).all())      # far exterior: exactly 0
    assert float((as_mask - hard).abs().mean()) < 0.02           # an edge treatment, not a different matte

    # "auto" leaves the matte of an alpha model alone ...
    as_alpha = MatteRefiner(params, W, H, alpha_is_mask=False).refine(hard, rgb, [0])
    assert intermediate(as_alpha) == 0
    # ... unless the guided filter is requested explicitly; "none" switches it off for masks too.
    forced = MatteRefiner(replace(params, mask_to_alpha="guided_filter"), W, H).refine(hard, rgb, [0])
    assert intermediate(forced) > 50
    off = MatteRefiner(replace(params, mask_to_alpha="none"), W, H, alpha_is_mask=True).refine(hard, rgb, [0])
    assert intermediate(off) == 0


def _hair_scene() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A body with a plain soft edge and thin semi-transparent strands growing out of it."""

    alpha = torch.zeros(1, 1, H, W)
    alpha[..., 70:, :] = 1.0
    alpha[..., 68, :] = 0.5
    alpha[..., 69, :] = 0.85
    strands = torch.zeros(1, 1, H, W, dtype=torch.bool)
    for k, x in enumerate(range(100, 180, 8)):
        strands[..., 30:68, x : x + 1 + (k % 2)] = True          # 1 and 2 px wide strands, right half only
    alpha = torch.where(strands, torch.full_like(alpha, 0.7), alpha)
    rgb = composite(alpha, texture(2), texture(1))
    return alpha, rgb, strands


def test_refiner_preserve_hair_keeps_thin_strands():
    alpha, rgb, strands = _hair_scene()
    base = RefinementParams(
        erode=1.5, edge_feather=2.0, resolution_scaling=False, temporal_smoothing=0.0,
        remove_small_artifacts=False, fill_holes=False, black_point=0.0, white_point=1.0,
    )
    protected = MatteRefiner(replace(base, preserve_hair=True), W, H).refine(alpha, rgb, [0])
    wiped = MatteRefiner(replace(base, preserve_hair=False), W, H).refine(alpha, rgb, [0])
    original = float(alpha[strands].mean())
    kept, lost = float(protected[strands].mean()), float(wiped[strands].mean())
    assert original == pytest.approx(0.7)
    assert kept > 0.9 * original                                 # strands survive almost untouched
    assert float((protected[strands] == alpha[strands]).float().mean()) > 0.9      # most of them bit for bit
    assert lost < 0.25 * original                                # without protection erode + feather wipe them out
    assert kept > 4.0 * lost
    # Away from the hair the operators act exactly as configured: the plain edge is eroded in both.
    plain = (slice(None), slice(None), slice(60, 80), slice(8, 60))
    assert torch.allclose(protected[plain], wiped[plain], atol=1e-6)
    assert float(wiped[plain].sum()) < float(alpha[plain].sum()) - 40.0
    # nothing appears where there was nothing, and the body stays opaque
    assert bool((protected[..., :20, :] == 0.0).all()) and bool((protected[..., 90:, :] == 1.0).all())


@pytest.mark.parametrize("preserve_hair", [False, True])
def test_refiner_spatial_operators_on_the_active_crop_equal_the_full_frame(preserve_hair):
    """The refiner runs the spatial operators only on the part of the frame that is not flat.  That
    is an optimisation and must not be visible: the result equals the operators on the whole frame."""

    alpha = disc(60.0, 70.0, 26.0, soft=2.0)
    for k, x in enumerate(range(52, 90, 7)):
        alpha[..., 18:36, x : x + 1 + (k % 2)] = 0.7              # strands above the disc
    rgb = composite(alpha, texture(2), texture(1))
    params = RefinementParams(
        erode=1.5, dilate=1.0, gaussian_smoothing=1.2, edge_feather=2.0, preserve_hair=preserve_hair,
        resolution_scaling=False, temporal_smoothing=0.0, remove_small_artifacts=False, fill_holes=False,
        black_point=0.0, white_point=1.0,
    )
    out = MatteRefiner(params, W, H).refine(alpha, rgb, [0])

    expected = ops.feather(ops.smooth(ops.dilate(ops.erode(alpha, 1.5), 1.0), 1.2), 2.0)
    if preserve_hair:
        # the protected zone reaches as far as the operators can carry a sample
        reach = 2 + 1 + ops.smooth_support(1.2) + ops.smooth_support(2.0 / 2.0)
        protect = ops.detail_map(alpha, MatteRefiner.DETAIL_RADIUS, 3 + reach)
        expected = torch.lerp(expected, alpha, protect)
    assert torch.allclose(out, expected.clamp(0.0, 1.0), atol=1e-6)
    assert not torch.allclose(out, alpha, atol=1e-2)             # the operators did act
    assert bool((out[..., :, 150:] == 0.0).all())                # the flat part of the frame is untouched


def test_refiner_resets_automatically_on_a_gap_in_the_frame_indices():
    alpha, rgb, _ = flicker_batch(6)
    params = replace(REFINEMENT_PRESETS["balanced"], temporal_smoothing=0.7)

    continuous = MatteRefiner(params, W, H)
    continuous.refine(alpha[0:3], rgb[0:3], [0, 1, 2])
    blended = continuous.refine(alpha[3:4], rgb[3:4], [3])

    gapped = MatteRefiner(params, W, H)
    gapped.refine(alpha[0:3], rgb[0:3], [0, 1, 2])
    after_gap = gapped.refine(alpha[3:4], rgb[3:4], [10])        # frame 10 does not follow frame 2

    fresh = MatteRefiner(params, W, H).refine(alpha[3:4], rgb[3:4], [10])
    assert torch.equal(after_gap, fresh)                         # nothing of frames 0-2 leaked in
    assert not torch.equal(blended, fresh)                       # without the gap the history is used
    assert gapped.statistics()["temporal_resets"] == 1 and continuous.statistics()["temporal_resets"] == 0
    # a gap inside one batch and going backwards are handled the same way
    again = MatteRefiner(params, W, H)
    again.refine(alpha[0:4], rgb[0:4], [0, 1, 5, 6])
    again.refine(alpha[4:5], rgb[4:5], [3])
    assert again.statistics()["temporal_resets"] == 2
    # explicit reset: same effect, statistics are kept
    manual = MatteRefiner(params, W, H)
    manual.refine(alpha[0:3], rgb[0:3], [0, 1, 2])
    manual.reset()
    assert torch.equal(manual.refine(alpha[3:4], rgb[3:4], [3]), fresh)
    assert manual.statistics()["frames"] == 4


@pytest.mark.parametrize("name", ["balanced", "strong"])
def test_refiner_batch_equals_frame_by_frame(name):
    alpha, rgb, _ = flicker_batch(7)
    alpha[2, 0, 12:14, 170:172] = 0.8                                         # a 4 px speck for the clean-up
    alpha[4, 0, 60:63, 90:93] = 0.0                                           # a 9 px hole
    params = REFINEMENT_PRESETS[name]
    batched = MatteRefiner(params, W, H)
    whole = batched.refine(alpha, rgb, list(range(7)))
    single = MatteRefiner(params, W, H)
    parts = [single.refine(alpha[i : i + 1], rgb[i : i + 1], [i]) for i in range(7)]
    assert torch.equal(whole, torch.cat(parts))                  # bit-identical
    mixed = MatteRefiner(params, W, H)
    chunks = [mixed.refine(alpha[0:3], rgb[0:3], [0, 1, 2]), mixed.refine(alpha[3:7], rgb[3:7], [3, 4, 5, 6])]
    assert torch.equal(whole, torch.cat(chunks))
    assert batched.statistics() == single.statistics() == mixed.statistics()
    assert batched.statistics()["components_removed"] == 1 and batched.statistics()["holes_filled"] == 1


def test_refiner_statistics_report_the_flicker_improvement():
    alpha, rgb, _ = flicker_batch(30)
    expected_keys = {
        "frames", "components_removed", "holes_filled", "flicker_mean", "flicker_p95", "flicker_max",
        "raw_flicker_mean", "raw_flicker_p95", "raw_flicker_max", "area_jumps",
    }
    results = {}
    for name in ("light", "balanced", "strong"):
        refiner = MatteRefiner(REFINEMENT_PRESETS[name], W, H)
        assert expected_keys <= set(refiner.statistics())        # present before the first frame, all zero
        assert refiner.statistics()["frames"] == 0 and refiner.statistics()["flicker_mean"] == 0.0
        for start in range(0, 30, 6):
            refiner.refine(alpha[start : start + 6], rgb[start : start + 6], list(range(start, start + 6)))
        stats = refiner.statistics()
        json.dumps(stats)                                        # serialisable for the report
        assert expected_keys <= set(stats)
        assert stats["frames"] == 30 and stats["transitions"] == 29
        assert stats["raw_flicker_mean"] > 0.0
        assert stats["flicker_mean"] <= stats["flicker_p95"] <= stats["flicker_max"]
        assert stats["raw_flicker_mean"] <= stats["raw_flicker_p95"] <= stats["raw_flicker_max"]
        assert stats["flicker_mean"] < stats["raw_flicker_mean"]                 # the refined matte flickers less
        assert stats["flicker_max"] < stats["raw_flicker_max"]
        assert stats["area_jumps"] == 0
        results[name] = stats["flicker_mean"] / stats["raw_flicker_mean"]
    assert results["strong"] < results["balanced"] < results["light"] < 1.0
    assert results["balanced"] < 0.75 and results["strong"] < 0.5


def test_refiner_statistics_count_area_jumps_only_in_a_still_picture():
    base = disc(64.0, 96.0, 30.0)
    picture = composite(base, texture(2), texture(1))
    params = REFINEMENT_PRESETS["balanced"]

    # The matte collapses for one frame although nothing in the picture changes: two jumps (out, back).
    refiner = MatteRefiner(params, W, H)
    for i in range(8):
        a = disc(64.0, 96.0, 12.0) if i == 4 else base
        refiner.refine(a, picture, [i])
    assert refiner.statistics()["area_jumps"] == 2

    # A subject that walks into the picture changes the matte area just as fast - but the picture moves.
    walker = MatteRefiner(params, W, H)
    bg, fg = texture(1) * 0.4, texture(2) * 0.4 + 0.6
    for i in range(10):
        a = disc(64.0, -20.0 + 12.0 * i, 26.0)
        walker.refine(a, composite(a, fg, bg), [i])
    assert walker.statistics()["area_jumps"] == 0


def test_refiner_preserve_motion_blur_relaxes_the_levels_in_moving_blur_only():
    """Strong levels harden every soft edge - except wide semi-transparent ramps that are moving."""

    h, w = 96, 160
    ramp = torch.linspace(0.0, 1.0, 40).view(1, 1, 1, 40)

    def frame(t: int, moving: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.zeros(1, 1, h, w)
        x0 = 20 + (4 * t if moving else 0)
        a[..., 20:76, x0 : x0 + 40] = ramp
        a[..., 20:76, x0 + 40 : x0 + 70] = 1.0
        shift = 4 * t if moving else 0
        return a, composite(a, torch.roll(texture(2, h, w), shift, -1), texture(1, h, w) * 0.4)

    def soft_pixels(params: RefinementParams, moving: bool) -> int:
        refiner = MatteRefiner(params, w, h)
        out = None
        for t in range(4):
            a, rgb = frame(t, moving)
            out = refiner.refine(a, rgb, [t])
        return int(((out > 0.02) & (out < 0.98)).sum())

    hard = RefinementParams(black_point=0.3, white_point=0.7, temporal_smoothing=0.0,
                            remove_small_artifacts=False, fill_holes=False)
    with_blur = soft_pixels(replace(hard, preserve_motion_blur=True), moving=True)
    without = soft_pixels(replace(hard, preserve_motion_blur=False), moving=True)
    still = soft_pixels(replace(hard, preserve_motion_blur=True), moving=False)
    assert without == pytest.approx(still, rel=0.1)              # a still ramp is hardened either way
    assert with_blur > 1.6 * without                             # the moving blur keeps its width


def test_refiner_sanitises_input_and_validates_arguments():
    alpha, rgb, _ = flicker_batch(3)
    alpha[1, 0, 5:9, 5:9] = float("nan")
    alpha[2, 0, 5:9, 5:9] = float("inf")
    refiner = MatteRefiner(REFINEMENT_PRESETS["balanced"], W, H)
    out = refiner.refine(alpha, rgb, [0, 1, 2])
    assert bool(torch.isfinite(out).all()) and float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert bool(torch.isfinite(refiner.refine(alpha[0:1], rgb[0:1], [3])).all())
    assert all(v == v for v in refiner.statistics().values() if isinstance(v, float))      # no NaN in the report

    with pytest.raises(RefinementError):
        refiner.refine(alpha[..., :-2], rgb[..., :-2], [4, 5, 6])            # wrong frame size
    with pytest.raises(RefinementError):
        refiner.refine(alpha, rgb, [4, 5])                                   # index count
    with pytest.raises(RefinementError):
        refiner.refine(alpha[:, 0], rgb, [4, 5, 6])                          # missing channel dimension
    with pytest.raises(RefinementError):
        refiner.refine(alpha, rgb[:, :1], [4, 5, 6])
    with pytest.raises(ConfigError):
        MatteRefiner(replace(REFINEMENT_PRESETS["balanced"], mask_to_alpha="magic"), W, H)
    with pytest.raises(ConfigError):
        MatteRefiner(replace(REFINEMENT_PRESETS["balanced"], black_point=0.6, white_point=0.5), W, H)
    # an empty batch is not an error
    empty = refiner.refine(alpha[:0], rgb[:0], [])
    assert empty.shape == (0, 1, H, W)
    # half precision input is accepted and comes back as float32
    half = MatteRefiner(REFINEMENT_PRESETS["balanced"], W, H).refine(alpha[0:1].half(), rgb[0:1].half(), [0])
    assert half.dtype == torch.float32


def test_refiner_odd_frame_size_and_full_or_empty_mattes():
    h, w = 97, 151
    rgb = texture(4, h, w).expand(3, -1, -1, -1)
    for name in ("light", "balanced", "strong"):
        refiner = MatteRefiner(REFINEMENT_PRESETS[name], w, h)
        empty = refiner.refine(torch.zeros(3, 1, h, w), rgb, [0, 1, 2])
        assert torch.equal(empty, torch.zeros(3, 1, h, w))
        full = refiner.refine(torch.ones(3, 1, h, w), rgb, [3, 4, 5])
        assert torch.equal(full, torch.ones(3, 1, h, w))
        soft = disc(48.0, 75.0, 30.0, h=h, w=w).expand(3, -1, -1, -1)
        out = refiner.refine(soft, rgb, [6, 7, 8])
        assert out.shape == (3, 1, h, w) and float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    masked = MatteRefiner(REFINEMENT_PRESETS["balanced"], w, h, alpha_is_mask=True)
    assert torch.equal(masked.refine(torch.ones(1, 1, h, w), rgb[0:1], [0]), torch.ones(1, 1, h, w))
    assert torch.equal(masked.refine(torch.zeros(1, 1, h, w), rgb[0:1], [1]), torch.zeros(1, 1, h, w))


# =========================================================================== timing (no assertion on speed)


def test_refiner_timing_1080p_balanced(record_property):
    """Measure the refiner on a 1920x1080 frame with the 'balanced' preset on the CPU."""

    h, w = 1080, 1920
    frames = 6
    g = torch.Generator().manual_seed(0)
    base = disc(540.0, 960.0, 330.0, soft=3.0, h=h, w=w)
    picture = composite(base, texture(2, h, w, sigma=6.0), texture(1, h, w, sigma=6.0))
    band = (base > 0.0) & (base < 1.0)
    refiner = MatteRefiner(REFINEMENT_PRESETS["balanced"], w, h)
    elapsed = 0.0
    for i in range(frames + 1):
        alpha = (base + 0.1 * (torch.rand(1, 1, h, w, generator=g) - 0.5) * band).clamp(0.0, 1.0)
        rgb = (picture + 0.01 * torch.randn(1, 3, h, w, generator=g)).clamp(0.0, 1.0)
        t0 = time.perf_counter()
        out = refiner.refine(alpha, rgb, [i])
        if i > 0:                                               # the first frame warms up caches and has no history
            elapsed += time.perf_counter() - t0
    ms = elapsed / frames * 1000.0
    record_property("refiner_balanced_1080p_cpu_ms_per_frame", round(ms, 1))
    print(f"\n[timing] MatteRefiner 'balanced' 1920x1080 CPU ({torch.get_num_threads()} threads): {ms:.1f} ms/frame")
    assert out.shape == (1, 1, h, w)
    stats = refiner.statistics()
    assert stats["frames"] == frames + 1 and stats["flicker_mean"] < stats["raw_flicker_mean"]
