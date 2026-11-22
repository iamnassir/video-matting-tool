"""Blur-Fusion foreground colour estimation: exactness rules, blending, batch independence and
the actual decontamination effect on synthetic edges."""

from __future__ import annotations

import pytest
import torch

import app.refinement.foreground as foreground_module
from app.refinement.foreground import estimate_foreground

RED = (1.0, 0.0, 0.0)
BLUE = (0.0, 0.0, 1.0)


def soft_disc(height: int, width: int, softness: float, radius_fraction: float = 0.3) -> torch.Tensor:
    """Matte ``[1, 1, H, W]`` of a centred disc whose edge is a linear ramp *softness* pixels wide."""

    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32), torch.arange(width, dtype=torch.float32), indexing="ij"
    )
    dist = torch.sqrt((xx - width * 0.5) ** 2 + (yy - height * 0.5) ** 2)
    return ((min(height, width) * radius_fraction - dist) / softness + 0.5).clamp(0.0, 1.0).view(1, 1, height, width)


def mix(alpha: torch.Tensor, fg, bg) -> torch.Tensor:
    """Observed picture ``a*F + (1-a)*B`` for constant colours or ``[1, 3, H, W]`` pictures."""

    f = fg if torch.is_tensor(fg) else torch.tensor(fg).view(1, 3, 1, 1)
    b = bg if torch.is_tensor(bg) else torch.tensor(bg).view(1, 3, 1, 1)
    return alpha * f + (1.0 - alpha) * b


def band_of(alpha: torch.Tensor, low: float = 0.2, high: float = 0.8) -> torch.Tensor:
    return ((alpha > low) & (alpha < high)).expand(-1, 3, -1, -1)


def noisy_scene(height: int, width: int, softness: float, seed: int):
    gen = torch.Generator().manual_seed(seed)
    alpha = soft_disc(height, width, softness)
    rgb = (mix(alpha, RED, BLUE) + 0.08 * torch.rand(1, 3, height, width, generator=gen)).clamp(0.0, 1.0)
    return rgb, alpha


# --------------------------------------------------------------------------- rule 1


@pytest.mark.parametrize("strength", [1.0, 0.4])
def test_opaque_pixels_are_returned_exactly(strength):
    rgb, alpha = noisy_scene(96, 128, softness=8.0, seed=1)
    out = estimate_foreground(rgb, alpha, strength=strength, radius=20.0, refine_radius=3.0)
    opaque = (alpha >= 1.0).expand_as(rgb)
    assert int(opaque.sum()) > 3000
    assert torch.equal(out[opaque], rgb[opaque])
    assert not torch.equal(out, rgb)                              # ... but the band did change


def test_matte_values_above_one_count_as_opaque():
    rgb, alpha = noisy_scene(64, 96, softness=6.0, seed=2)
    hot = alpha * 1.5                                             # out-of-range matte: >= 1 inside the subject
    out = estimate_foreground(rgb, hot, radius=12.0, refine_radius=2.0)
    opaque = (hot >= 1.0).expand_as(rgb)
    assert torch.equal(out[opaque], rgb[opaque])
    assert torch.equal(out, estimate_foreground(rgb, hot.clamp(0.0, 1.0), radius=12.0, refine_radius=2.0))


def test_full_matte_returns_the_input_values_and_empty_matte_is_well_defined():
    gen = torch.Generator().manual_seed(3)
    rgb = torch.rand(2, 3, 63, 97, generator=gen)
    full = estimate_foreground(rgb, torch.ones(2, 1, 63, 97), radius=10.0, refine_radius=2.0)
    assert torch.equal(full, rgb)
    empty = estimate_foreground(rgb, torch.zeros(2, 1, 63, 97), radius=10.0, refine_radius=2.0)
    assert empty.shape == rgb.shape and empty.dtype == torch.float32
    assert bool(torch.isfinite(empty).all())
    assert float(empty.max()) == 0.0                              # no subject anywhere: nothing to extend, black


# --------------------------------------------------------------------------- rule 2


def test_strength_blends_between_input_and_estimate():
    rgb, alpha = noisy_scene(96, 128, softness=8.0, seed=4)
    kwargs = dict(radius=20.0, refine_radius=3.0)
    full = estimate_foreground(rgb, alpha, strength=1.0, **kwargs)

    assert estimate_foreground(rgb, alpha, strength=0.0, **kwargs) is rgb          # unchanged, not even copied
    assert estimate_foreground(rgb, alpha, strength=-1.0, **kwargs) is rgb
    assert estimate_foreground(rgb, alpha, strength=float("nan"), **kwargs) is rgb
    assert torch.equal(estimate_foreground(rgb, alpha, strength=3.0, **kwargs), full)

    half = estimate_foreground(rgb, alpha, strength=0.5, **kwargs)
    assert float((half - 0.5 * (rgb + full)).abs().max()) < 1e-6
    quarter = estimate_foreground(rgb, alpha, strength=0.25, **kwargs)
    assert float((quarter - (0.75 * rgb + 0.25 * full)).abs().max()) < 1e-6

    band = band_of(alpha)
    blue = [float(x[:, 2:3][band[:, :1]].mean()) for x in (rgb, quarter, half, full)]
    assert blue[0] > blue[1] > blue[2] > blue[3]                  # more strength, less old background


# --------------------------------------------------------------------------- rule 3


def test_batch_equals_frame_by_frame(monkeypatch):
    frames, mattes = [], []
    for seed, softness in ((5, 3.0), (6, 7.0), (7, 14.0)):
        rgb, alpha = noisy_scene(63, 97, softness=softness, seed=seed)
        frames.append(rgb)
        mattes.append(alpha)
    mattes.append(torch.zeros(1, 1, 63, 97))                     # a frame without any subject
    frames.append(frames[0].flip(-1))
    mattes.append(torch.ones(1, 1, 63, 97))                      # a fully opaque frame
    frames.append(frames[1].flip(-2))
    rgb, alpha = torch.cat(frames), torch.cat(mattes)
    kwargs = dict(radius=12.0, refine_radius=2.0)

    batch = estimate_foreground(rgb, alpha, **kwargs)
    for index in range(rgb.shape[0]):
        single = estimate_foreground(rgb[index:index + 1], alpha[index:index + 1], **kwargs)
        assert torch.equal(batch[index:index + 1], single), f"frame {index} depends on its batch"

    # the internal memory-bounding split must not change anything either
    monkeypatch.setattr(foreground_module, "_CHUNK_PIXELS", 2 * 63 * 97)
    assert torch.equal(estimate_foreground(rgb, alpha, **kwargs), batch)
    monkeypatch.setattr(foreground_module, "_CHUNK_PIXELS", 1)
    assert torch.equal(estimate_foreground(rgb, alpha, **kwargs), batch)


# --------------------------------------------------------------------------- rule 4


@pytest.mark.parametrize(
    "height, width, softness, radius, refine",
    [
        (96, 128, 8.0, 24.0, 3.0),
        (128, 96, 5.0, 12.0, 2.0),
        (256, 256, 12.0, 40.0, 3.0),
    ],
)
def test_red_disc_over_blue_is_decontaminated(height, width, softness, radius, refine):
    alpha = soft_disc(height, width, softness)
    rgb = mix(alpha, RED, BLUE)
    out = estimate_foreground(rgb, alpha, radius=radius, refine_radius=refine)

    band = (alpha > 0.2) & (alpha < 0.8)
    assert int(band.sum()) > 300
    observed_blue = float(rgb[:, 2:3][band].mean())
    estimated_blue = float(out[:, 2:3][band].mean())
    assert 0.4 < observed_blue < 0.6                              # the observed colour really is a mixture
    assert estimated_blue < 0.25 * observed_blue, (observed_blue, estimated_blue)  # required: > 60 % less; measured: ~90 %
    assert float(out[:, 0:1][band].mean()) > 0.85                 # and it is red now
    assert float(out[:, 1:2][band].abs().max()) < 1e-4            # no colour appears that was in neither layer

    red = torch.tensor(RED).view(1, 3, 1, 1)
    error_before = float(((rgb - red) ** 2)[band_of(alpha)].mean())
    error_after = float(((out - red) ** 2)[band_of(alpha)].mean())
    assert error_after < 0.1 * error_before


def test_textured_layers_error_to_true_foreground_is_reduced():
    height, width = 128, 160
    gen = torch.Generator().manual_seed(8)
    yy, xx = torch.meshgrid(torch.arange(height, dtype=torch.float32), torch.arange(width, dtype=torch.float32), indexing="ij")
    true_fg = torch.stack([0.75 + 0.15 * torch.sin(xx * 0.21), 0.25 + 0.1 * torch.cos(yy * 0.17), 0.2 + 0.1 * torch.sin((xx + yy) * 0.11)])
    true_bg = torch.stack([0.15 + 0.1 * torch.cos(xx * 0.09), 0.3 + 0.15 * torch.sin(yy * 0.13), 0.8 + 0.1 * torch.cos((xx - yy) * 0.07)])
    true_fg = (true_fg[None] + 0.02 * torch.rand(1, 3, height, width, generator=gen)).clamp(0, 1)
    true_bg = (true_bg[None] + 0.02 * torch.rand(1, 3, height, width, generator=gen)).clamp(0, 1)
    alpha = soft_disc(height, width, softness=9.0)
    rgb = mix(alpha, true_fg, true_bg)

    out = estimate_foreground(rgb, alpha, radius=30.0, refine_radius=3.0)
    band = band_of(alpha, 0.1, 0.9)
    error_before = float(((rgb - true_fg) ** 2)[band].mean())
    error_after = float(((out - true_fg) ** 2)[band].mean())
    assert error_after < 0.15 * error_before, (error_before, error_after)


def test_uniform_picture_is_a_fixed_point():
    """If foreground and background have the same colour there is nothing to remove."""

    alpha = soft_disc(96, 128, softness=10.0)
    colour = torch.tensor([0.8, 0.25, 0.35]).view(1, 3, 1, 1)
    rgb = colour.expand(1, 3, 96, 128).contiguous()
    out = estimate_foreground(rgb, alpha, radius=20.0, refine_radius=3.0)
    visible = (alpha > 0.02).expand_as(rgb)
    assert float((out - rgb)[visible].abs().max()) < 1e-3


def test_hard_matte_keeps_the_subject_and_extends_its_colour_outwards():
    alpha = (soft_disc(96, 128, softness=1.0) >= 0.5).to(torch.float32)
    rgb = mix(alpha, RED, BLUE)
    out = estimate_foreground(rgb, alpha, radius=20.0, refine_radius=3.0)
    inside = alpha.bool().expand_as(rgb)
    assert torch.equal(out[inside], rgb[inside])
    # pixels just outside the subject (within one pixel) carry the subject's colour, not the old background
    grown = torch.nn.functional.max_pool2d(alpha, kernel_size=3, stride=1, padding=1)
    rim = (grown > 0) & (alpha == 0)
    assert int(rim.sum()) > 100
    assert float(out[:, 0:1][rim].min()) > 0.9 and float(out[:, 2:3][rim].max()) < 0.1


# --------------------------------------------------------------------------- working copy / options


def test_reduced_working_copy_is_close_to_full_resolution_processing():
    height, width = 192, 256
    alpha = soft_disc(height, width, softness=10.0)
    rgb = (mix(alpha, RED, BLUE) + 0.05 * torch.rand(1, 3, height, width, generator=torch.Generator().manual_seed(9))).clamp(0, 1)
    kwargs = dict(radius=36.0, refine_radius=3.0)
    exact = estimate_foreground(rgb, alpha, max_working_side=None, **kwargs)
    assert torch.equal(exact, estimate_foreground(rgb, alpha, max_working_side=0, **kwargs))
    assert torch.equal(exact, estimate_foreground(rgb, alpha, max_working_side=-1, **kwargs))
    assert torch.equal(exact, estimate_foreground(rgb, alpha, max_working_side=4096, **kwargs))   # nothing to reduce
    reduced = estimate_foreground(rgb, alpha, max_working_side=64, **kwargs)                       # 4x smaller stage 1
    band = band_of(alpha)
    assert float((reduced - exact)[band].abs().mean()) < 0.01
    assert float(reduced[:, 2:3][band[:, :1]].mean()) < 0.4 * float(rgb[:, 2:3][band[:, :1]].mean())
    opaque = (alpha >= 1.0).expand_as(rgb)
    assert torch.equal(reduced[opaque], rgb[opaque])


def test_single_stage_and_degenerate_radii():
    rgb, alpha = noisy_scene(64, 96, softness=6.0, seed=10)
    two = estimate_foreground(rgb, alpha, radius=14.0, refine_radius=2.0)
    one = estimate_foreground(rgb, alpha, radius=14.0, refine_radius=0.0)          # stage 2 skipped
    assert not torch.equal(one, two)
    band = (alpha > 0.2) & (alpha < 0.8)
    for out in (one, two, estimate_foreground(rgb, alpha, radius=0.0, refine_radius=0.3),
                estimate_foreground(rgb, alpha, radius=5000.0, refine_radius=5000.0)):
        assert out.shape == rgb.shape and bool(torch.isfinite(out).all())
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
        assert float(out[:, 2:3][band].mean()) < float(rgb[:, 2:3][band].mean())


@pytest.mark.parametrize("height, width, batch", [(63, 97, 2), (97, 63, 1), (1, 40, 1), (40, 1, 2), (5, 7, 3)])
def test_odd_and_tiny_sizes(height, width, batch):
    gen = torch.Generator().manual_seed(11)
    rgb = torch.rand(batch, 3, height, width, generator=gen)
    alpha = torch.rand(batch, 1, height, width, generator=gen)
    out = estimate_foreground(rgb, alpha)                          # default radii are far larger than the frame
    assert out.shape == rgb.shape and out.dtype == torch.float32 and out.device == rgb.device
    assert bool(torch.isfinite(out).all()) and float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_inputs_are_not_modified():
    rgb, alpha = noisy_scene(48, 64, softness=5.0, seed=12)
    rgb_kept, alpha_kept = rgb.clone(), alpha.clone()
    for strength in (1.0, 0.5):
        estimate_foreground(rgb, alpha, strength=strength, radius=10.0, refine_radius=2.0)
    assert torch.equal(rgb, rgb_kept) and torch.equal(alpha, alpha_kept)


def test_undefined_matte_samples_are_treated_as_background():
    rgb, alpha = noisy_scene(48, 64, softness=5.0, seed=13)
    broken = alpha.clone()
    broken[0, 0, :4, :4] = float("nan")
    assert float(alpha[0, 0, :4, :4].max()) == 0.0                # that corner is background anyway
    out = estimate_foreground(rgb, broken, radius=10.0, refine_radius=2.0)
    assert bool(torch.isfinite(out).all())
    assert torch.equal(out, estimate_foreground(rgb, alpha, radius=10.0, refine_radius=2.0))


def test_other_float_dtypes_are_accepted():
    rgb, alpha = noisy_scene(32, 48, softness=4.0, seed=14)
    reference = estimate_foreground(rgb, alpha, radius=8.0, refine_radius=2.0)
    out = estimate_foreground(rgb.double(), alpha.double(), radius=8.0, refine_radius=2.0)
    assert out.dtype == torch.float32 and torch.equal(out, reference)
    assert torch.equal(estimate_foreground(rgb, alpha.half(), radius=8.0, refine_radius=2.0),
                       estimate_foreground(rgb, alpha.half().float(), radius=8.0, refine_radius=2.0))


@pytest.mark.parametrize(
    "rgb_shape, alpha_shape",
    [
        ((1, 3, 8, 8), (1, 1, 8, 9)),
        ((1, 3, 8, 8), (2, 1, 8, 8)),
        ((1, 3, 8, 8), (1, 3, 8, 8)),
        ((1, 4, 8, 8), (1, 1, 8, 8)),
        ((3, 8, 8), (1, 8, 8)),
    ],
)
def test_shape_validation(rgb_shape, alpha_shape):
    with pytest.raises(ValueError):
        estimate_foreground(torch.zeros(rgb_shape), torch.zeros(alpha_shape))
