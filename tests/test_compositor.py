"""Native-domain compositor: pixel preservation (I4), constant background (I5), edge blending,
transparent / matte targets, scaling, RGBA export, previews, batch independence and a timing probe."""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import app.compositing.compositor as compositor_module
from app.compositing.colors import (
    ColorConverter,
    planes_from_numpy,
    planes_to_numpy,
    to_integer_planes,
)
from app.compositing.compositor import CompositeContext, Compositor, Target, preview_atlas, rgb8_to_yuv420p
from app.core.errors import ConfigError
from app.core.pixfmt import MATRIX_COEFFICIENTS, PixFmt, ResolvedColor, parse_working_pixfmt
from app.utils.torch_ops import resize

GREEN, BLUE, RED, BLACK, WHITE = (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 0, 0), (255, 255, 255)
W, H = 97, 63                                   # odd on purpose: partial chroma footprints everywhere


# --------------------------------------------------------------------------- helpers


def fmt(name: str) -> PixFmt:
    return parse_working_pixfmt(name)


def color(matrix: str = "bt709", full: bool = False) -> ResolvedColor:
    kr, kb = MATRIX_COEFFICIENTS[matrix]
    return ResolvedColor(matrix=matrix, kr=kr, kb=kb, full_range=full)


def color_for(pix: PixFmt) -> ResolvedColor:
    return color("bt709", full=pix.family == "rgb")


def random_planes(pix: PixFmt, width: int, height: int, batch: int = 1, seed: int = 0) -> List[torch.Tensor]:
    """Random integer code values over the whole code range (float32 planes, colour planes only)."""

    gen = torch.Generator().manual_seed(seed)
    return [
        torch.randint(0, pix.max_value + 1, (batch, 1) + pix.plane_size(i, width, height), generator=gen).to(torch.float32)
        for i in range(pix.color_planes)
    ]


def picture_planes(pix: PixFmt, col: ResolvedColor, width: int, height: int, batch: int = 1, seed: int = 0):
    """A smooth, slightly noisy picture well inside the gamut -> integer-valued float planes."""

    gen = torch.Generator().manual_seed(seed)
    coarse = torch.rand(batch, 3, max(2, height // 8), max(2, width // 8), generator=gen)
    rgb = F.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False)
    rgb = (0.2 + 0.6 * rgb + 0.03 * (torch.rand(batch, 3, height, width, generator=gen) - 0.5)).clamp(0.15, 0.85)
    planes = to_integer_planes(ColorConverter(pix, col).rgb_to_planes(rgb), pix)
    return [p.to(torch.float32) for p in planes]


def uniform_planes(pix: PixFmt, col: ResolvedColor, rgb8, width: int, height: int, batch: int = 1):
    codes = ColorConverter(pix, col).color_code_values(rgb8)
    return [
        torch.full((batch, 1) + pix.plane_size(i, width, height), float(codes[i])) for i in range(pix.color_planes)
    ], codes


def mixed_matte(height: int = H, width: int = W) -> torch.Tensor:
    """Hard opaque area, soft ramp, hard clear area, plus hard-edged blocks at odd coordinates."""

    xx = torch.arange(width, dtype=torch.float32).view(1, width).expand(height, width)
    a = ((40.5 - xx) / 9.0 + 0.5).clamp(0.0, 1.0).clone()       # 1 for x <= 36, 0 for x >= 45
    a[5:20, 61:80] = 1.0                                        # opaque block inside the clear area
    a[41:50, 70:90] = 0.37                                      # flat semi-transparent block
    a[30:33, 3:9] = 0.0                                         # hole inside the opaque area
    return a.view(1, 1, height, width)


def soft_disc(height: int, width: int, radius: float, softness: float, cx=None, cy=None) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32), torch.arange(width, dtype=torch.float32), indexing="ij"
    )
    cx = (width - 1) / 2.0 if cx is None else cx
    cy = (height - 1) / 2.0 if cy is None else cy
    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return ((radius - dist) / softness + 0.5).clamp(0.0, 1.0).view(1, 1, height, width)


def pooled(alpha: np.ndarray, sub_w: int, sub_h: int, fn) -> np.ndarray:
    """Reference footprint pooling of one ``[H, W]`` matte with plain loops (independent of torch)."""

    sh, sw = 1 << sub_h, 1 << sub_w
    h, w = alpha.shape
    out = np.empty(((h + sh - 1) // sh, (w + sw - 1) // sw), dtype=np.float64)
    for j in range(out.shape[0]):
        for i in range(out.shape[1]):
            out[j, i] = fn(alpha[j * sh:(j + 1) * sh, i * sw:(i + 1) * sw])
    return out


def local_range(plane: np.ndarray, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """Minimum and maximum of every sample's ``(2r+1)^2`` neighbourhood."""

    if radius == 0:
        return plane, plane
    padded = np.pad(plane, radius, mode="edge")
    h, w = plane.shape
    views = [padded[dy:dy + h, dx:dx + w] for dy in range(2 * radius + 1) for dx in range(2 * radius + 1)]
    return np.min(views, axis=0), np.max(views, axis=0)


def sub_of(pix: PixFmt, index: int) -> Tuple[int, int]:
    return (pix.chroma_w, pix.chroma_h) if pix.is_chroma_plane(index) else (0, 0)


def codes_of(values: np.ndarray, peak: float) -> np.ndarray:
    """``round_half_away(values * peak)`` with the product formed in float32, as the compositor does.

    The single-precision product is exactly representable in float64, so the rounding itself is
    done without any floating-point ambiguity."""

    product = values.astype(np.float32) * np.float32(peak)
    return np.floor(product.astype(np.float64) + 0.5)


# --------------------------------------------------------------------------- 1. opaque core (I4)


@pytest.mark.parametrize(
    "name",
    ["yuv420p", "yuv422p", "yuv444p", "yuv440p", "yuv420p10le", "yuv422p10le", "yuv444p10le", "yuv420p12le",
     "yuv444p16le", "gbrp", "gbrp10le", "gray", "gray16le"],
)
def test_fully_opaque_matte_returns_the_source_bit_for_bit(name):
    pix = fmt(name)
    rng = np.random.default_rng(1)
    arrays = [
        rng.integers(0, pix.max_value + 1, size=(2,) + pix.plane_size(i, W, H)).astype(pix.dtype)
        for i in range(pix.color_planes)
    ]
    planes = planes_from_numpy(arrays, "cpu")
    comp = Compositor(pix, color_for(pix), W, H, decontaminate=1.0)
    for size in (None, (W, H)):
        ctx = comp.begin(planes, torch.ones(2, 1, H, W))
        out = comp.render(ctx, Target("main", "composite", pix, background=GREEN, size=size))
        assert len(out) == pix.color_planes
        for index, (plane, source) in enumerate(zip(out, planes)):
            assert plane.shape == (2, 1) + pix.plane_size(index, W, H)
            assert not plane.is_floating_point()
            assert torch.equal(plane.to(torch.float32), source)
        for result, original in zip(planes_to_numpy(out, pix), arrays):
            assert result.dtype == pix.dtype and result.tobytes() == original.tobytes()


# --------------------------------------------------------------------------- 2. background constant (I5)


@pytest.mark.parametrize("name", ["yuv420p", "yuv420p10le", "yuv444p", "yuv422p10le", "gbrp", "gray"])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("background", [GREEN, BLUE, BLACK, WHITE], ids=["green", "blue", "black", "white"])
def test_empty_matte_gives_one_constant_per_plane(name, full, background):
    pix = fmt(name)
    col = color("bt709", full)
    planes = random_planes(pix, W, H, batch=2, seed=2)
    comp = Compositor(pix, col, W, H)
    ctx = comp.begin(planes, torch.zeros(2, 1, H, W))
    out = comp.render(ctx, Target("main", "composite", pix, background=background))
    expected = ColorConverter(pix, col).color_code_values(background)
    assert len(out) == len(expected) == pix.color_planes
    for index, plane in enumerate(out):
        assert plane.shape == (2, 1) + pix.plane_size(index, W, H)
        assert int(plane.min()) == int(plane.max()) == expected[index]


@pytest.mark.parametrize(
    "name, full, background, expected",
    [
        ("yuv420p", False, GREEN, [173, 42, 26]),
        ("yuv420p", True, GREEN, [182, 30, 12]),
        ("yuv420p10le", False, GREEN, [691, 167, 105]),
        ("yuv420p10le", False, WHITE, [940, 512, 512]),
        ("yuv420p", True, BLACK, [0, 128, 128]),
        ("yuv420p", False, BLUE, [32, 240, 118]),
        ("gbrp", True, GREEN, [255, 0, 0]),
    ],
)
def test_background_constant_literal_values(name, full, background, expected):
    pix = fmt(name)
    comp = Compositor(pix, color("bt709", full), 32, 16)
    ctx = comp.begin(random_planes(pix, 32, 16), torch.zeros(1, 1, 16, 32))
    out = comp.render(ctx, Target("main", "composite", pix, background=background))
    assert [int(p.unique().item()) for p in out] == expected


# --------------------------------------------------------------------------- 3. mixed matte


@pytest.mark.parametrize("name", ["yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "gbrp", "gray"])
@pytest.mark.parametrize("decontaminate", [1.0, 0.0])
def test_mixed_matte_core_is_source_background_is_constant_band_is_in_between(name, decontaminate):
    pix = fmt(name)
    col = color_for(pix)
    planes = picture_planes(pix, col, W, H, batch=2, seed=3)
    alpha = torch.cat([mixed_matte(), mixed_matte().flip(-1).flip(-2)])
    comp = Compositor(pix, col, W, H, decontaminate=decontaminate, decontaminate_radius=10.0, decontaminate_refine_radius=2.0)
    ctx = comp.begin(planes, alpha)
    out = comp.render(ctx, Target("main", "composite", pix, background=GREEN))
    constants = comp.converter.color_code_values(GREEN)

    for index, plane in enumerate(out):
        sub_w, sub_h = sub_of(pix, index)
        for frame in range(2):
            result = plane[frame, 0].numpy().astype(np.float64)
            source = planes[index][frame, 0].numpy().astype(np.float64)
            matte = alpha[frame, 0].numpy()
            core = pooled(matte, sub_w, sub_h, np.min) == 1.0
            clear = pooled(matte, sub_w, sub_h, np.max) == 0.0
            band = ~(core | clear)
            assert core.sum() > 100 and clear.sum() > 100 and band.sum() > 50
            assert np.array_equal(result[core], source[core])                  # I4: bit-exact source
            assert np.all(result[clear] == constants[index])                   # I5: one constant
            if decontaminate == 0.0:
                # without colour estimation every band sample is a mixture of nearby source samples
                # and the background constant (subsampled chroma: the neighbouring samples take part
                # through the interpolation of the source chroma)
                low, high = local_range(source, 1 if (sub_w or sub_h) else 0)
                low = np.minimum(low, constants[index]) - 1.5
                high = np.maximum(high, constants[index]) + 1.5
                assert np.all(result[band] >= low[band]) and np.all(result[band] <= high[band])


@pytest.mark.parametrize("name", ["yuv420p", "yuv444p10le", "yuv422p"])
def test_band_of_a_uniform_picture_is_the_linear_blend_in_every_plane(name):
    """With one colour everywhere the foreground estimate is that colour, so the band must be
    exactly ``K + a * (S - K)`` per plane (chroma: with the matte averaged over the footprint)."""

    pix = fmt(name)
    col = color("bt709", False)
    planes, codes = uniform_planes(pix, col, (200, 60, 90), W, H)
    alpha = mixed_matte()
    comp = Compositor(pix, col, W, H, decontaminate=1.0, decontaminate_radius=12.0, decontaminate_refine_radius=2.0)
    out = comp.render(comp.begin(planes, alpha), Target("main", "composite", pix, background=GREEN))
    constants = comp.converter.color_code_values(GREEN)
    matte = alpha[0, 0].numpy().astype(np.float64)
    for index, plane in enumerate(out):
        weight = pooled(matte, *sub_of(pix, index), np.mean)
        expected = constants[index] + weight * (codes[index] - constants[index])
        result = plane[0, 0].numpy().astype(np.float64)
        assert np.abs(result - expected).max() <= 1.5
        low, high = min(codes[index], constants[index]), max(codes[index], constants[index])
        assert result.min() >= low and result.max() <= high                   # never outside source..background


def test_footprint_masks_match_reference_pooling():
    pix = fmt("yuv420p")
    alpha = torch.cat([mixed_matte(), soft_disc(H, W, 20.0, 5.0)])
    comp = Compositor(pix, color(), W, H)
    ctx = comp.begin(random_planes(pix, W, H, batch=2), alpha)
    for sub_w, sub_h in ((0, 0), (1, 1), (1, 0), (0, 1)):
        core, clear = ctx.masks(sub_w, sub_h)
        assert core.dtype == torch.bool and clear.dtype == torch.bool
        for frame in range(2):
            matte = alpha[frame, 0].numpy()
            assert np.array_equal(core[frame, 0].numpy(), pooled(matte, sub_w, sub_h, np.min) == 1.0)
            assert np.array_equal(clear[frame, 0].numpy(), pooled(matte, sub_w, sub_h, np.max) == 0.0)
        assert ctx.masks(sub_w, sub_h)[0] is core                              # cached


# --------------------------------------------------------------------------- other target formats


def test_composite_into_a_deeper_444_format_restores_the_core_without_a_matrix():
    src, dst = fmt("yuv420p"), fmt("yuv444p10le")
    col = color("bt709", False)
    planes = picture_planes(src, col, W, H, seed=4)
    alpha = mixed_matte()
    comp = Compositor(src, col, W, H, decontaminate_radius=10.0, decontaminate_refine_radius=2.0)
    out = comp.render(comp.begin(planes, alpha), Target("deep", "composite", dst, background=GREEN))
    assert [p.dtype for p in out] == [torch.int16] * 3 and all(p.shape == (1, 1, H, W) for p in out)
    core = (alpha == 1.0)[0, 0]
    clear = (alpha == 0.0)[0, 0]
    assert torch.equal(out[0][0, 0][core].to(torch.float32), planes[0][0, 0][core] * 4.0)   # luma: exactly x4
    for index in (1, 2):
        up = F.interpolate(planes[index], scale_factor=2, mode="bilinear", align_corners=False)[..., :H, :W] * 4.0
        expected = torch.floor(up + 0.5)
        assert torch.equal(out[index][0, 0][core].to(torch.float32), expected[0, 0][core])
    for plane, constant in zip(out, [691, 167, 105]):
        assert bool((plane[0, 0][clear] == constant).all())


def test_gray_and_cross_family_targets():
    # gray source -> gray target: exact; gray source -> yuv target: luma exact, neutral chroma in the core
    gray = fmt("gray")
    col = color("bt709", False)
    (luma,) = random_planes(gray, W, H, seed=5)
    comp = Compositor(gray, col, W, H)
    ones, zeros = torch.ones(1, 1, H, W), torch.zeros(1, 1, H, W)
    out = comp.render(comp.begin([luma], ones), Target("g", "composite", fmt("yuv420p"), background=GREEN))
    assert torch.equal(out[0].to(torch.float32), luma)
    assert int(out[1].min()) == int(out[1].max()) == 128 and int(out[2].min()) == int(out[2].max()) == 128
    out = comp.render(comp.begin([luma], zeros), Target("g", "composite", fmt("yuv420p"), background=GREEN))
    assert [int(p.unique().item()) for p in out] == [173, 42, 26]

    # yuv source -> gray target: the luma plane, chroma dropped
    yuv = fmt("yuv420p")
    planes = random_planes(yuv, W, H, seed=6)
    comp = Compositor(yuv, col, W, H)
    (only,) = comp.render(comp.begin(planes, ones), Target("y", "composite", gray, background=GREEN))
    assert torch.equal(only.to(torch.float32), planes[0])

    # yuv (limited) source -> RGB target: goes through the matrix (not bit-exact), full-range RGB
    rgb_fmt = fmt("gbrp")
    planes = picture_planes(yuv, col, W, H, seed=7)
    comp = Compositor(yuv, col, W, H)
    out = comp.render(comp.begin(planes, zeros), Target("r", "composite", rgb_fmt, background=WHITE))
    assert [int(p.unique().item()) for p in out] == [255, 255, 255]
    out = comp.render(comp.begin(planes, ones), Target("r", "composite", rgb_fmt, background=WHITE))
    reference = to_integer_planes(comp.converter.rgb_to_planes(comp.converter.planes_to_rgb(planes), rgb_fmt), rgb_fmt)
    assert all(torch.equal(a, b) for a, b in zip(out, reference))

    # RGB source -> yuv target: matrix conversion, video range
    planes = picture_planes(rgb_fmt, color("bt709", True), W, H, seed=8)
    comp = Compositor(rgb_fmt, color("bt709", True), W, H)
    out = comp.render(comp.begin(planes, zeros), Target("v", "composite", yuv, background=GREEN))
    assert [int(p.unique().item()) for p in out] == [173, 42, 26]
    out = comp.render(comp.begin(planes, ones), Target("v", "composite", yuv, background=GREEN))
    reference = to_integer_planes(comp.converter.rgb_to_planes(comp.converter.planes_to_rgb(planes), yuv), yuv)
    assert all(torch.equal(a, b) for a, b in zip(out, reference))
    assert 16 <= int(out[0].min()) and int(out[0].max()) <= 235


def test_composite_target_with_alpha_format_gets_an_opaque_alpha_plane():
    src = fmt("yuv420p")
    planes = picture_planes(src, color(), W, H, seed=9)
    comp = Compositor(src, color(), W, H, decontaminate_radius=10.0, decontaminate_refine_radius=2.0)
    ctx = comp.begin(planes, mixed_matte())
    plain = comp.render(ctx, Target("p", "composite", src, background=GREEN))
    with_alpha = comp.render(ctx, Target("a", "composite", fmt("yuva420p"), background=GREEN))
    assert len(with_alpha) == 4 and with_alpha[3].shape == (1, 1, H, W)
    assert int(with_alpha[3].min()) == 255
    assert all(torch.equal(a, b) for a, b in zip(plain, with_alpha[:3]))
    matte = comp.render(ctx, Target("m", "matte", fmt("yuva444p10le")))
    assert len(matte) == 4 and int(matte[3].min()) == int(matte[3].max()) == 1023


# --------------------------------------------------------------------------- 7. alpha targets


def test_alpha_target_yuva444p10le_from_yuv420p_source():
    src, dst = fmt("yuv420p"), fmt("yuva444p10le")
    col = color("bt709", False)
    planes = picture_planes(src, col, W, H, batch=2, seed=10)
    alpha = torch.cat([mixed_matte(), soft_disc(H, W, 20.0, 7.0)])
    comp = Compositor(src, col, W, H, decontaminate_radius=10.0, decontaminate_refine_radius=2.0)
    ctx = comp.begin(planes, alpha)
    straight = comp.render(ctx, Target("alpha", "alpha", dst))
    premultiplied = comp.render(ctx, Target("alpha", "alpha", dst, alpha_mode="premultiplied"))

    for out in (straight, premultiplied):
        assert len(out) == 4
        assert all(p.dtype == torch.int16 and p.shape == (2, 1, H, W) for p in out)
        # alpha plane: full range, round(a * 1023)
        assert np.array_equal(out[3].numpy(), codes_of(alpha.numpy(), 1023.0))
        assert int(out[3].max()) == 1023 and int(out[3].min()) == 0
        assert out[3][0, 0, 45, 75].item() == 379 and out[3][0, 0, 0, 40].item() == 568   # 0.37 and 5/9 of 1023
        core, clear = alpha == 1.0, alpha == 0.0
        # opaque core: source luma x4 exactly, chroma = interpolated source chroma x4
        assert torch.equal(out[0][core].to(torch.float32), planes[0][core] * 4.0)
        for index in (1, 2):
            up = F.interpolate(planes[index], scale_factor=2, mode="bilinear", align_corners=False)[..., :H, :W] * 4.0
            assert torch.equal(out[index][core].to(torch.float32), torch.floor(up + 0.5)[core])
        # fully transparent samples: black, in both modes
        for plane, black in zip(out[:3], [64, 512, 512]):
            assert bool((plane[clear] == black).all())

    # in the band the premultiplied colour is the straight colour pulled towards black
    band = (alpha > 0.05) & (alpha < 0.95)
    assert int(band.sum()) > 300
    luma_straight = straight[0][band].to(torch.float32)
    luma_premultiplied = premultiplied[0][band].to(torch.float32)
    assert bool((luma_premultiplied <= luma_straight + 1).all())
    assert float((luma_straight - luma_premultiplied).mean()) > 20.0
    expected = 64.0 + alpha[band] * (luma_straight - 64.0)
    assert float((luma_premultiplied - expected).abs().max()) <= 1.5


def test_straight_alpha_carries_the_foreground_colour_in_the_band():
    src, dst = fmt("yuv444p"), fmt("yuva444p")
    col = color("bt709", False)
    planes, codes = uniform_planes(src, col, (200, 60, 90), W, H)
    alpha = soft_disc(H, W, 20.0, 9.0)
    comp = Compositor(src, col, W, H, decontaminate_radius=12.0, decontaminate_refine_radius=2.0)
    out = comp.render(comp.begin(planes, alpha), Target("alpha", "alpha", dst))
    visible = (alpha > 0.0)[0, 0]
    for plane, code in zip(out[:3], codes):
        assert int((plane[0, 0][visible].to(torch.int32) - code).abs().max()) <= 1       # F, not a*F
    subsampled = comp.render(comp.begin(planes, alpha), Target("alpha", "alpha", fmt("yuva420p")))
    assert [tuple(p.shape[-2:]) for p in subsampled] == [(H, W), (32, 49), (32, 49), (H, W)]
    assert torch.equal(subsampled[0], out[0]) and torch.equal(subsampled[3], out[3])


def test_straight_alpha_chroma_ignores_colour_under_transparent_pixels():
    """4:2:0 straight alpha, hard matte: a chroma sample shared by subject and removed background
    must carry the subject's chroma only (alpha-weighted reduction), or the edge gets a colour fringe."""

    src, dst = fmt("yuv444p"), fmt("yuva420p")
    col = color("bt709", False)
    conv = ColorConverter(src, col)
    width, height = 48, 32
    alpha = torch.zeros(1, 1, height, width)
    alpha[..., 5:21, 7:31] = 1.0                                             # hard edges that cut through 2x2 footprints
    red, blue = conv.color_code_values(RED), conv.color_code_values(BLUE)
    planes = [torch.where(alpha > 0, float(r), float(b)) for r, b in zip(red, blue)]   # red subject on blue
    comp = Compositor(src, col, width, height)
    for mode in ("straight", "premultiplied"):
        y, cb, cr, a = comp.render(comp.begin(planes, alpha), Target("web", "alpha", dst, alpha_mode=mode))
        assert torch.equal(a, (alpha * 255).to(torch.uint8))
        assert bool((y[alpha > 0] == red[0]).all()) and bool((y[alpha == 0] == 16).all())
        touched = F.max_pool2d(alpha, 2, 2) > 0                              # footprints with a visible pixel
        full = -F.max_pool2d(-alpha, 2, 2) > 0                               # footprints that are all subject
        straddling = touched & ~full
        assert int(straddling.sum()) > 20 and int(full.sum()) > 50
        assert bool((cb[~touched] == 128).all()) and bool((cr[~touched] == 128).all())
        assert bool((cb[full] == red[1]).all()) and bool((cr[full] == red[2]).all())
        if mode == "straight":
            assert bool((cb[straddling] == red[1]).all()) and bool((cr[straddling] == red[2]).all())
        else:                                                                # premultiplied: pulled towards neutral
            assert bool((cr[straddling] < red[2]).all()) and bool((cr[straddling] > 128).all())


def test_planar_rgba_target_agrees_with_render_rgba():
    """A ``gbrap`` alpha target and ``render_rgba`` are two layouts of the same picture."""

    src = fmt("yuv420p")
    col = color("bt709", False)
    planes = picture_planes(src, col, W, H, batch=2, seed=21)
    alpha = torch.cat([mixed_matte(), mixed_matte().flip(-2)])
    comp = Compositor(src, col, W, H, decontaminate_radius=10.0, decontaminate_refine_radius=2.0)
    ctx = comp.begin(planes, alpha)
    for bits, name in ((8, "gbrap"), (16, "gbrap16le")):
        for mode in ("straight", "premultiplied"):
            g, b, r, a = comp.render(ctx, Target("planar", "alpha", fmt(name), alpha_mode=mode))
            rgba = comp.render_rgba(ctx, bits, mode)
            assert g.dtype == rgba.dtype
            stacked = torch.cat([r, g, b, a], dim=1).permute(0, 2, 3, 1)
            assert torch.equal(stacked, rgba)
    again = comp.render(ctx, Target("planar", "alpha", fmt("gbrap")))        # rendering is repeatable
    assert all(torch.equal(x, y) for x, y in zip(again, comp.render(ctx, Target("planar", "alpha", fmt("gbrap")))))


def test_alpha_target_requires_an_alpha_format_and_valid_mode():
    with pytest.raises(ConfigError):
        Target("bad", "alpha", fmt("yuv444p10le"))
    with pytest.raises(ConfigError):
        Target("bad", "alpha", fmt("yuva444p10le"), alpha_mode="associated")


# --------------------------------------------------------------------------- 8. matte targets


def test_matte_target_levels():
    src = fmt("yuv420p")
    width, height = 9, 7
    planes = random_planes(src, width, height, seed=11)
    alpha = torch.zeros(1, 1, height, width)
    alpha[0, 0, 0, :3] = torch.tensor([1.0, 0.5, 0.25])
    comp = Compositor(src, color(), width, height)
    ctx = comp.begin(planes, alpha)

    y, cb, cr = comp.render(ctx, Target("matte", "matte", fmt("yuv420p")))
    assert y.dtype == torch.uint8 and y.shape == (1, 1, 7, 9) and cb.shape == (1, 1, 4, 5) == cr.shape
    assert y[0, 0, 0, :4].tolist() == [235, 126, 71, 16]                    # 16 + 219 * a, halves away from zero
    assert int(y[0, 0, 1:].max()) == 16
    assert int(cb.min()) == int(cb.max()) == 128 and int(cr.min()) == int(cr.max()) == 128

    y10, cb10, cr10 = comp.render(ctx, Target("matte", "matte", fmt("yuv422p10le")))
    assert y10.dtype == torch.int16 and cb10.shape == (1, 1, 7, 5)
    assert y10[0, 0, 0, :4].tolist() == [940, 502, 283, 64]
    assert int(cb10.min()) == int(cr10.max()) == 512

    (grey,) = comp.render(ctx, Target("matte", "matte", fmt("gray")))
    assert grey[0, 0, 0, :4].tolist() == [255, 128, 64, 0]                  # gray: full range
    (grey16,) = comp.render(ctx, Target("matte", "matte", fmt("gray16le")))
    assert grey16.dtype == torch.int32 and grey16[0, 0, 0, :4].tolist() == [65535, 32768, 16384, 0]
    rgb = comp.render(ctx, Target("matte", "matte", fmt("gbrp")))
    assert len(rgb) == 3 and all(p[0, 0, 0, :4].tolist() == [255, 128, 64, 0] for p in rgb)

    # full-range sources still get a video-level matte: the matte is a picture of its own
    full = Compositor(src, color("bt709", True), width, height)
    y_full, _, _ = full.render(full.begin(planes, alpha), Target("matte", "matte", fmt("yuv420p")))
    assert torch.equal(y_full, y)


# --------------------------------------------------------------------------- 9. scaling


def _scaling_scene(name: str = "yuv420p", width: int = 96, height: int = 64, radius: float = 14.0):
    pix = fmt(name)
    col = color("bt709", False)
    planes = picture_planes(pix, col, width, height, seed=12)
    alpha = soft_disc(height, width, radius, 3.0)
    comp = Compositor(pix, col, width, height, decontaminate_radius=8.0, decontaminate_refine_radius=2.0)
    return pix, comp, comp.begin(planes, alpha), planes, alpha


def _far_from_centre(plane_h: int, plane_w: int, distance: float) -> torch.Tensor:
    """Mask of the samples of a ``plane_h x plane_w`` plane that lie at least *distance* source
    pixels (of the 96x64 source frame the plane was scaled from) away from the frame centre."""

    yy, xx = torch.meshgrid(
        (torch.arange(plane_h, dtype=torch.float32) + 0.5) * (64.0 / plane_h),
        (torch.arange(plane_w, dtype=torch.float32) + 0.5) * (96.0 / plane_w),
        indexing="ij",
    )
    return torch.sqrt((xx - 48.0) ** 2 + (yy - 32.0) ** 2) >= distance


@pytest.mark.parametrize("size", [(48, 32), (192, 128), (72, 48)])
def test_scaling_returns_requested_size_and_keeps_pure_background_constant(size):
    pix, comp, ctx, _, _ = _scaling_scene()
    out = comp.render(ctx, Target("small", "composite", pix, background=GREEN, size=size))
    width, height = size
    assert [tuple(p.shape) for p in out] == [
        (1, 1, height, width), (1, 1, height // 2, width // 2), (1, 1, height // 2, width // 2)
    ]
    assert all(p.dtype == torch.uint8 for p in out)
    # samples whose whole filter support lies in the pure background are exactly the constant
    for plane, constant, distance in zip(out, [173, 42, 26], [30.0, 36.0, 36.0]):
        far = _far_from_centre(plane.shape[-2], plane.shape[-1], distance)
        assert int(far.sum()) > 40
        assert bool((plane[0, 0][far] == constant).all())
    centre = out[0][0, 0, height // 2 - 2:height // 2 + 2, width // 2 - 2:width // 2 + 2]
    assert bool((centre != 173).any())                                      # the subject is still there


def test_scaling_pads_with_the_background_constant_and_preserves_aspect():
    pix, comp, ctx, planes, _ = _scaling_scene()
    out = comp.render(ctx, Target("square", "composite", pix, background=GREEN, size=(64, 64)))
    assert [tuple(p.shape[-2:]) for p in out] == [(64, 64), (32, 32), (32, 32)]
    for plane, constant, rows in zip(out, [173, 42, 26], [8, 4, 4]):
        assert bool((plane[0, 0, :rows] == constant).all()) and bool((plane[0, 0, -rows:] == constant).all())
    assert bool((out[0][0, 0, 28:36] != 173).any())

    # geometry, measured with a fully opaque matte rendered as a matte picture: a 96x64 frame fitted
    # into 64x64 is 64 wide and 42.67 high, centred
    opaque = comp.begin(planes, torch.ones(1, 1, 64, 96))
    luma = comp.render(opaque, Target("m", "matte", pix, size=(64, 64)))[0][0, 0]
    white_rows = (luma == 235).all(dim=1)
    black_rows = (luma == 16).all(dim=1)
    assert int(white_rows.sum()) in (42, 43, 44)
    assert int(white_rows.sum()) + int(black_rows.sum()) == 64             # nothing in between: hard picture edge
    first = int(torch.nonzero(white_rows)[0])
    last = int(torch.nonzero(white_rows)[-1])
    assert abs(first - (63 - last)) <= 2                                    # centred
    chroma = comp.render(opaque, Target("m", "matte", pix, size=(64, 64)))[1]
    assert int(chroma.min()) == int(chroma.max()) == 128


def test_scaling_absorbs_rounding_of_the_requested_size_instead_of_padding():
    pix = fmt("yuv420p")
    col = color()
    planes, codes = uniform_planes(pix, col, (200, 60, 90), 98, 64)
    comp = Compositor(pix, col, 98, 64)
    ctx = comp.begin(planes, torch.ones(1, 1, 64, 98))
    # 98x64 scaled by 1.5 is 147x96; an even "96p" request asks for 146x96 - no bars may appear
    out = comp.render(ctx, Target("up", "composite", pix, background=GREEN, size=(146, 96)))
    assert [tuple(p.shape[-2:]) for p in out] == [(96, 146), (48, 73), (48, 73)]
    for plane, code in zip(out, codes):
        assert int(plane.min()) == int(plane.max()) == code                 # a constant picture stays that constant


def test_pure_padding_does_not_resample_and_keeps_the_picture_exact():
    pix = fmt("yuv420p")
    col = color()
    planes = random_planes(pix, 96, 64, seed=13)
    kept = [p.clone() for p in planes]
    comp = Compositor(pix, col, 96, 64)
    ctx = comp.begin(planes, torch.ones(1, 1, 64, 96))
    out = comp.render(ctx, Target("pad", "composite", pix, background=BLUE, size=(128, 64)))
    assert [tuple(p.shape[-2:]) for p in out] == [(64, 128), (32, 64), (32, 64)]
    assert torch.equal(out[0][..., 16:112].to(torch.float32), planes[0])
    assert torch.equal(out[1][..., 8:56].to(torch.float32), planes[1])
    for plane, constant, margin in zip(out, [32, 240, 118], [16, 8, 8]):
        assert bool((plane[..., :margin] == constant).all()) and bool((plane[..., -margin:] == constant).all())
    assert all(torch.equal(a, b) for a, b in zip(planes, kept))            # the source planes were not touched


def test_scaled_alpha_and_matte_targets_pad_with_transparent_black():
    pix, comp, ctx, _, _ = _scaling_scene()
    for mode in ("straight", "premultiplied"):
        out = comp.render(ctx, Target("a", "alpha", fmt("yuva420p"), size=(64, 64), alpha_mode=mode))
        assert [tuple(p.shape[-2:]) for p in out] == [(64, 64), (32, 32), (32, 32), (64, 64)]
        for plane, constant, rows in zip(out, [16, 128, 128, 0], [8, 4, 4, 8]):
            assert bool((plane[0, 0, :rows] == constant).all()) and bool((plane[0, 0, -rows:] == constant).all())
        assert int(out[3].max()) == 255
    matte = comp.render(ctx, Target("m", "matte", pix, size=(64, 64)))
    assert bool((matte[0][0, 0, :8] == 16).all()) and bool((matte[0][0, 0, -8:] == 16).all())
    assert int(matte[1].min()) == int(matte[2].max()) == 128
    assert int(matte[0].max()) == 235


@pytest.mark.parametrize("name", ["yuva444p", "yuva420p", "yuva444p10le"])
def test_scaled_straight_alpha_has_no_dark_fringe(name):
    """Colour next to transparency must be resampled premultiplied: a uniformly coloured subject
    keeps its colour in every visible sample of the scaled straight-alpha output."""

    src = fmt("yuv444p")
    dst = fmt(name)
    col = color("bt709", False)
    planes, source_codes = uniform_planes(src, col, (230, 200, 60), 96, 64)
    alpha = soft_disc(64, 96, 20.0, 4.0)
    comp = Compositor(src, col, 96, 64, decontaminate_radius=8.0, decontaminate_refine_radius=2.0)
    ctx = comp.begin(planes, alpha)
    codes = [c << (dst.bit_depth - src.bit_depth) for c in source_codes]   # the source colour at the target depth
    straight = comp.render(ctx, Target("s", "alpha", dst, size=(48, 32)))
    premultiplied = comp.render(ctx, Target("p", "alpha", dst, size=(48, 32), alpha_mode="premultiplied"))
    assert torch.equal(straight[3], premultiplied[3])
    peak = dst.max_value
    alpha_plane = straight[3].to(torch.float32)
    assert int(alpha_plane.max()) == peak and int(alpha_plane.min()) == 0
    for index in range(3):
        plane_alpha, plane_peak = alpha_plane, alpha_plane
        if dst.is_chroma_plane(index) and dst.is_subsampled:
            plane_alpha = F.avg_pool2d(alpha_plane, 2, 2)
            plane_peak = F.max_pool2d(alpha_plane, 2, 2)
        visible = plane_alpha >= peak * (16.0 / 255.0)
        edge = visible & (plane_alpha <= peak * 0.8)
        assert int(edge.sum()) >= 8
        deviation = (straight[index].to(torch.float32) - codes[index]).abs()
        assert float(deviation[visible].max()) <= 1.0                       # the subject's colour, to the code value
        black = comp.converter.color_code_values(BLACK, dst)[index]
        transparent = plane_peak == 0
        assert int(transparent.sum()) > 50 and bool((straight[index][transparent] == black).all())
    # the premultiplied rendering of the same edge is far from the subject colour (pulled to black)
    edge = (alpha_plane >= peak * (16.0 / 255.0)) & (alpha_plane <= peak * 0.8)
    gap = (premultiplied[0].to(torch.float32) - codes[0]).abs()[edge].mean()
    assert float(gap) > 20.0 * peak / 255.0


def test_scaling_validation():
    pix, comp, ctx, _, _ = _scaling_scene()
    with pytest.raises(ConfigError):
        comp.render(ctx, Target("odd", "composite", pix, background=GREEN, size=(49, 32)))
    with pytest.raises(ConfigError):
        comp.render(ctx, Target("odd", "composite", pix, background=GREEN, size=(48, 33)))
    with pytest.raises(ConfigError):
        comp.render(ctx, Target("odd", "composite", fmt("yuv422p"), background=GREEN, size=(49, 32)))
    out = comp.render(ctx, Target("ok", "composite", fmt("yuv422p"), background=GREEN, size=(48, 33)))
    assert [tuple(p.shape[-2:]) for p in out] == [(33, 48), (33, 24), (33, 24)]
    out = comp.render(ctx, Target("ok", "composite", fmt("yuv444p"), background=GREEN, size=(49, 33)))
    assert [tuple(p.shape[-2:]) for p in out] == [(33, 49)] * 3
    # an odd size equal to the (odd) source size is not a resize and stays legal for 4:2:0
    odd = Compositor(pix, color(), W, H)
    octx = odd.begin(random_planes(pix, W, H), torch.ones(1, 1, H, W))
    assert odd.render(octx, Target("same", "composite", pix, background=GREEN, size=(W, H)))[1].shape == (1, 1, 32, 49)
    for bad in ((0, 32), (48,), (48, 32, 3), "big"):
        with pytest.raises(ConfigError):
            Target("bad", "composite", pix, background=GREEN, size=bad)


# --------------------------------------------------------------------------- 10. decontamination


@pytest.mark.parametrize("name", ["yuv444p", "yuv420p", "yuv422p10le"])
def test_decontamination_removes_the_old_background_from_the_edge_band(name):
    pix = fmt(name)
    col = color("bt709", False)
    conv = ColorConverter(pix, col)
    width, height = 128, 96
    alpha = soft_disc(height, width, 30.0, 8.0)
    red = torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1)
    blue = torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1)
    observed = alpha * red + (1.0 - alpha) * blue                         # red disc photographed on blue
    planes = [p.to(torch.float32) for p in to_integer_planes(conv.rgb_to_planes(observed), pix)]
    target = Target("main", "composite", pix, background=GREEN)

    results = {}
    for strength in (1.0, 0.0):
        comp = Compositor(pix, col, width, height, decontaminate=strength, decontaminate_radius=24.0, decontaminate_refine_radius=3.0)
        out = comp.render(comp.begin(planes, alpha), target)
        results[strength] = (out, conv.planes_to_rgb(out))

    band = ((alpha > 0.2) & (alpha < 0.8))[0, 0]
    assert int(band.sum()) > 500
    blue_clean = float(results[1.0][1][0, 2][band].mean())
    blue_dirty = float(results[0.0][1][0, 2][band].mean())
    assert blue_dirty > 0.12                                               # the halo is real without decontamination
    assert blue_clean < 0.4 * blue_dirty, (blue_clean, blue_dirty)         # ... and at least 60 % weaker with it
    # red + green explain the band once the blue is gone
    assert float(results[1.0][1][0, 0][band].mean()) > float(results[0.0][1][0, 0][band].mean())

    # the opaque core and the pure background are identical in both renderings (and exact)
    constants = conv.color_code_values(GREEN)
    matte = alpha[0, 0].numpy()
    for index in range(pix.color_planes):
        clean = results[1.0][0][index][0, 0].numpy()
        dirty = results[0.0][0][index][0, 0].numpy()
        core = pooled(matte, *sub_of(pix, index), np.min) == 1.0
        clear = pooled(matte, *sub_of(pix, index), np.max) == 0.0
        assert core.sum() > 100 and np.array_equal(clean[core], dirty[core])
        assert np.array_equal(clean[core], planes[index][0, 0].numpy()[core])
        assert np.all(clean[clear] == constants[index]) and np.all(dirty[clear] == constants[index])


def test_foreground_estimate_is_computed_once_and_only_when_needed(monkeypatch):
    calls = []
    real = compositor_module.estimate_foreground

    def counting(rgb, alpha, **kwargs):
        calls.append((rgb.shape[0], kwargs))
        return real(rgb, alpha, **kwargs)

    monkeypatch.setattr(compositor_module, "estimate_foreground", counting)
    pix = fmt("yuv420p")
    planes = picture_planes(pix, color(), W, H, batch=3, seed=14)
    soft = mixed_matte()
    hard = (soft_disc(H, W, 20.0, 1.0) >= 0.5).to(torch.float32)
    targets = [
        Target("main", "composite", pix, background=GREEN),
        Target("blue", "composite", pix, background=BLUE),
        Target("alpha", "alpha", fmt("yuva444p10le")),
        Target("matte", "matte", pix),
    ]

    comp = Compositor(pix, color(), W, H, decontaminate=0.7, decontaminate_radius=11.0, decontaminate_refine_radius=2.0)
    ctx = comp.begin(planes, torch.cat([soft, hard, soft.flip(-1)]))
    for target in targets:
        comp.render(ctx, target)
    comp.render_rgba(ctx)
    assert len(calls) == 1                                                 # shared by every target of the batch
    assert calls[0][0] == 2                                                # only the two frames that have a band
    assert calls[0][1] == {"strength": 0.7, "radius": 11.0, "refine_radius": 2.0}
    assert ctx.band_frames.tolist() == [True, False, True]

    for strength, alpha in (
        (0.0, torch.cat([soft, hard, soft])),                              # decontamination switched off
        (1.0, torch.cat([hard, hard, hard])),                              # binary matte: no band anywhere
        (1.0, torch.ones(3, 1, H, W)),
        (1.0, torch.zeros(3, 1, H, W)),
    ):
        calls.clear()
        comp = Compositor(pix, color(), W, H, decontaminate=strength)
        ctx = comp.begin(planes, alpha)
        for target in targets:
            comp.render(ctx, target)
        assert calls == []
        assert ctx.foreground is ctx.rgb


def test_precomputed_rgb_is_used_and_inputs_are_never_modified():
    pix = fmt("yuv420p")
    planes = picture_planes(pix, color(), W, H, batch=2, seed=15)
    alpha = torch.cat([mixed_matte(), soft_disc(H, W, 18.0, 6.0)])
    comp = Compositor(pix, color(), W, H, decontaminate_radius=10.0, decontaminate_refine_radius=2.0)
    rgb = comp.converter.planes_to_rgb(planes)
    kept_planes, kept_alpha, kept_rgb = [p.clone() for p in planes], alpha.clone(), rgb.clone()

    ctx = comp.begin(planes, alpha, rgb)
    assert isinstance(ctx, CompositeContext) and ctx.rgb is rgb and ctx.batch == 2
    targets = [
        Target("main", "composite", pix, background=GREEN),
        Target("deep", "composite", fmt("yuv444p10le"), background=GREEN),
        Target("alpha", "alpha", fmt("yuva444p10le")),
        Target("pre", "alpha", fmt("yuva420p"), alpha_mode="premultiplied"),
        Target("matte", "matte", pix),
        Target("small", "composite", fmt("yuv444p"), background=GREEN, size=(49, 33)),
        Target("small-alpha", "alpha", fmt("yuva444p"), size=(49, 33)),
    ]
    with_rgb = [comp.render(ctx, t) for t in targets] + [[comp.render_rgba(ctx)]]
    assert all(torch.equal(a, b) for a, b in zip(planes, kept_planes))
    assert torch.equal(alpha, kept_alpha) and torch.equal(rgb, kept_rgb)

    lazy = comp.begin(planes, alpha)                                        # derives R'G'B' itself
    without_rgb = [comp.render(lazy, t) for t in targets] + [[comp.render_rgba(lazy)]]
    for a, b in zip(with_rgb, without_rgb):
        assert all(torch.equal(x, y) for x, y in zip(a, b))

    as_integers = comp.begin([p.to(torch.uint8) for p in planes], alpha.double())
    assert all(torch.equal(x, y) for x, y in zip(comp.render(as_integers, targets[0]), with_rgb[0]))


def test_matte_is_sanitised():
    pix = fmt("yuv444p")
    planes = picture_planes(pix, color(), 16, 8, seed=16)
    alpha = torch.full((1, 1, 8, 16), 0.5)
    alpha[0, 0, 0, 0:4] = torch.tensor([float("nan"), -0.3, 1.7, float("inf")])
    alpha[0, 0, 1, 0] = float("-inf")
    kept = alpha.clone()
    comp = Compositor(pix, color(), 16, 8, decontaminate=0.0)
    ctx = comp.begin(planes, alpha)
    assert torch.equal(alpha[0, 0, 2:], kept[0, 0, 2:]) and bool(torch.isnan(alpha[0, 0, 0, 0]))   # input untouched
    assert ctx.alpha[0, 0, 0, :4].tolist() == [0.0, 0.0, 1.0, 1.0] and float(ctx.alpha[0, 0, 1, 0]) == 0.0
    out = comp.render(ctx, Target("main", "composite", pix, background=GREEN))
    constants = [173, 42, 26]
    for plane, source, constant in zip(out, planes, constants):
        assert plane[0, 0, 0, :2].tolist() == [constant, constant]         # undefined / negative -> background
        assert plane[0, 0, 0, 2:4].tolist() == source[0, 0, 0, 2:4].tolist()   # above one -> source
        assert int(plane[0, 0, 1, 0]) == constant


# --------------------------------------------------------------------------- 11. RGBA, preview, yuv420p


def test_render_rgba_values_and_dtypes():
    pix = fmt("yuv444p")
    col = color("bt709", False)
    planes = picture_planes(pix, col, W, H, batch=2, seed=17)
    alpha = torch.cat([mixed_matte(), soft_disc(H, W, 20.0, 6.0)])
    comp = Compositor(pix, col, W, H, decontaminate=0.0)                    # F == rgb: values are predictable
    ctx = comp.begin(planes, alpha)
    rgb = ctx.rgb.permute(0, 2, 3, 1).numpy()                               # float32 [B, H, W, 3]
    a = alpha[:, 0].numpy()                                                 # float32 [B, H, W]

    for bits, dtype, peak in ((8, torch.uint8, 255.0), (16, torch.int32, 65535.0)):
        straight = comp.render_rgba(ctx, bits)
        premultiplied = comp.render_rgba(ctx, bits, "premultiplied")
        alpha_codes = codes_of(a, peak)
        for out in (straight, premultiplied):
            assert out.shape == (2, H, W, 4) and out.dtype == dtype and out.is_contiguous()
            assert np.array_equal(out[..., 3].numpy(), alpha_codes)         # A = round(a * max)
            assert int(out[..., 3].max()) == int(peak) and int(out[..., 3].min()) == 0
        visible = alpha_codes > 0
        assert visible.sum() > 2000 and (~visible).sum() > 2000
        expected = codes_of(rgb, peak)
        assert np.array_equal(straight[..., :3].numpy()[visible], expected[visible])       # straight: F itself
        assert int(straight[..., :3].numpy()[~visible].max()) == 0                         # transparent black
        expected = codes_of(rgb * a[..., None], peak)
        assert np.array_equal(premultiplied[..., :3].numpy(), expected)                    # premultiplied: a * F
        assert int(premultiplied[..., :3].numpy()[a == 0.0].max()) == 0
    assert int(comp.render_rgba(ctx, 8)[0, 45, 75, 3]) == 94                # 0.37 * 255 = 94.35
    assert int(comp.render_rgba(ctx, 16)[0, 45, 75, 3]) == 24248            # 0.37 * 65535 = 24247.95

    with pytest.raises(ConfigError):
        comp.render_rgba(ctx, 10)
    with pytest.raises(ConfigError):
        comp.render_rgba(ctx, 8, "weird")


def test_render_rgba_uses_the_foreground_estimate():
    pix = fmt("yuv444p")
    col = color("bt709", False)
    conv = ColorConverter(pix, col)
    alpha = soft_disc(96, 128, 30.0, 8.0)
    observed = alpha * torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1) + (1.0 - alpha) * torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1)
    planes = [p.to(torch.float32) for p in to_integer_planes(conv.rgb_to_planes(observed), pix)]
    band = ((alpha > 0.2) & (alpha < 0.8))[0, 0]
    blue = {}
    for strength in (1.0, 0.0):
        comp = Compositor(pix, col, 128, 96, decontaminate=strength, decontaminate_radius=24.0, decontaminate_refine_radius=3.0)
        rgba = comp.render_rgba(comp.begin(planes, alpha))
        blue[strength] = float(rgba[0, ..., 2][band].to(torch.float32).mean())
    assert blue[0.0] > 100.0 and blue[1.0] < 0.25 * blue[0.0]


def test_preview_atlas_tiles():
    height, width, tile_w, tile_h = 72, 128, 64, 36
    gen = torch.Generator().manual_seed(18)
    coarse = torch.rand(2, 3, 9, 16, generator=gen)
    rgb = F.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False)
    alpha = torch.cat([soft_disc(height, width, 22.0, 6.0), soft_disc(height, width, 15.0, 2.0, cx=40.0, cy=30.0)])
    atlas = preview_atlas(rgb, alpha, GREEN, (tile_w, tile_h))
    assert atlas.shape == (2, 2 * tile_h, 2 * tile_w, 3) and atlas.dtype == torch.uint8 and atlas.is_contiguous()

    small_rgb = resize(rgb, (tile_h, tile_w), mode="bilinear", antialias=True).clamp(0, 1)
    small_alpha = resize(alpha, (tile_h, tile_w), mode="bilinear", antialias=True).clamp(0, 1)
    original = atlas[:, :tile_h, :tile_w].to(torch.float32)
    segmented = atlas[:, :tile_h, tile_w:].to(torch.float32)
    matte = atlas[:, tile_h:, :tile_w].to(torch.float32)
    composite = atlas[:, tile_h:, tile_w:].to(torch.float32)

    # top-left: the frame, area-averaged down
    assert float((original - small_rgb.permute(0, 2, 3, 1) * 255.0).abs().max()) <= 0.51
    # bottom-left: the matte as grey
    assert torch.equal(matte[..., 0], matte[..., 1]) and torch.equal(matte[..., 0], matte[..., 2])
    assert float((matte[..., 0] - small_alpha[:, 0] * 255.0).abs().max()) <= 0.51
    assert int(matte.max()) == 255 and int(matte.min()) == 0
    # bottom-right: simple composite over the background colour
    expected = small_alpha * small_rgb + (1.0 - small_alpha) * torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1)
    assert float((composite - expected.permute(0, 2, 3, 1) * 255.0).abs().max()) <= 0.51
    outside = small_alpha[:, 0] == 0.0
    assert int(outside.sum()) > 500 and bool((composite[outside] == torch.tensor([0.0, 255.0, 0.0])).all())
    # top-right: subject in colour, the rest dimmed grey, bright outline on the 0.5 contour
    outline = (segmented == torch.tensor([255.0, 235.0, 40.0])).all(dim=-1)
    assert 40 < int(outline.sum()) < 0.2 * outline.numel()
    hard = (small_alpha >= 0.5).to(torch.float32)
    ring = (F.max_pool2d(hard, 3, 1, 1) > hard)[:, 0]                       # pixels just outside the 0.5 contour
    assert bool((outline == ring).all())
    assert not bool((outline & (small_alpha[:, 0] >= 0.5)).any())           # the outline never covers the subject
    inside = (small_alpha[:, 0] == 1.0) & ~outline
    assert int(inside.sum()) > 100 and float((segmented[inside] - original[inside]).abs().max()) <= 1.0
    away = outside & ~outline
    grey = segmented[away]
    assert bool((grey[:, 0] == grey[:, 1]).all()) and bool((grey[:, 0] == grey[:, 2]).all())    # desaturated
    luma = (0.2126 * small_rgb[:, 0] + 0.7152 * small_rgb[:, 1] + 0.0722 * small_rgb[:, 2])[away] * 255.0
    assert float((grey[:, 0] - 0.35 * luma).abs().max()) <= 0.6                                # dimmed to 35 %


def test_preview_atlas_checkerboard_and_validation():
    rgb = torch.full((1, 3, 40, 60), 0.8)
    alpha = torch.zeros(1, 1, 40, 60)
    alpha[..., 10:30, 20:40] = 1.0
    atlas = preview_atlas(rgb, alpha, None, (60, 40))                       # tile == frame size: no resampling
    assert atlas.shape == (1, 80, 120, 3)
    composite = atlas[0, 40:, 60:]
    assert bool((composite[..., 0] == composite[..., 1]).all()) and bool((composite[..., 0] == composite[..., 2]).all())
    transparent = composite[..., 0][alpha[0, 0] == 0.0]
    assert sorted(transparent.unique().tolist()) == [107, 158]              # the two greys of the checkerboard
    assert bool((composite[10:30, 20:40] == 204).all())                     # the subject on top of it (0.8 * 255)
    board = composite[..., 0]
    assert int(board[0, 0]) != int(board[0, 59]) or int(board[0, 0]) != int(board[39, 0])   # it really alternates

    bigger = preview_atlas(rgb, alpha, BLUE, (90, 60))                      # tiles larger than the frame are fine
    assert bigger.shape == (1, 120, 180, 3)
    kept_rgb, kept_alpha = rgb.clone(), alpha.clone()
    preview_atlas(rgb, alpha, GREEN, (30, 20))
    assert torch.equal(rgb, kept_rgb) and torch.equal(alpha, kept_alpha)
    with pytest.raises(ValueError):
        preview_atlas(rgb, alpha[:, :, :20], GREEN, (30, 20))
    with pytest.raises(ValueError):
        preview_atlas(rgb[0], alpha, GREEN, (30, 20))
    with pytest.raises(ConfigError):
        preview_atlas(rgb, alpha, GREEN, (0, 20))
    with pytest.raises(ConfigError):
        preview_atlas(rgb, alpha, (0, 300, 0), (30, 20))


def test_rgb8_to_yuv420p_reference_values():
    white = torch.full((2, 6, 8, 3), 255, dtype=torch.uint8)
    y, cb, cr = rgb8_to_yuv420p(white)
    assert (y.shape, cb.shape, cr.shape) == ((2, 1, 6, 8), (2, 1, 3, 4), (2, 1, 3, 4))
    assert y.dtype == cb.dtype == cr.dtype == torch.uint8
    assert [int(p.unique().item()) for p in (y, cb, cr)] == [235, 128, 128]

    for rgb8, expected in ((GREEN, [173, 42, 26]), (BLACK, [16, 128, 128]), (RED, [63, 102, 240]), (BLUE, [32, 240, 118])):
        picture = torch.tensor(rgb8, dtype=torch.uint8).view(1, 1, 1, 3).expand(1, 4, 6, 3)
        assert [int(p.unique().item()) for p in rgb8_to_yuv420p(picture)] == expected

    # chroma is the average of each 2x2 block: one red and three white pixels
    block = torch.full((1, 2, 2, 3), 255, dtype=torch.uint8)
    block[0, 0, 0] = torch.tensor([255, 0, 0], dtype=torch.uint8)
    kept = block.clone()
    y, cb, cr = rgb8_to_yuv420p(block)
    assert y.flatten().tolist() == [63, 235, 235, 235]
    assert int(cb) == 122 and int(cr) == 156                               # 128 - 25.66 / 4, 128 + 112 / 4
    assert torch.equal(block, kept)

    with pytest.raises(ConfigError):
        rgb8_to_yuv420p(torch.zeros(1, 5, 8, 3, dtype=torch.uint8))
    with pytest.raises(ConfigError):
        rgb8_to_yuv420p(torch.zeros(1, 4, 7, 3, dtype=torch.uint8))
    with pytest.raises(ValueError):
        rgb8_to_yuv420p(torch.zeros(1, 3, 4, 8, dtype=torch.uint8))


# --------------------------------------------------------------------------- 12. batch independence


def _batch_scene():
    src = fmt("yuv420p")
    col = color("bt709", False)
    planes = picture_planes(src, col, W, H, batch=5, seed=19)
    alpha = torch.cat([
        mixed_matte(),
        torch.zeros(1, 1, H, W),                                           # no subject
        soft_disc(H, W, 22.0, 9.0),
        torch.ones(1, 1, H, W),                                            # all subject
        (soft_disc(H, W, 15.0, 1.0, cx=30.0, cy=20.0) >= 0.5).to(torch.float32),   # hard matte, no band
    ])
    targets = [
        Target("main", "composite", src, background=GREEN),
        Target("deep", "composite", fmt("yuv444p10le"), background=BLUE),
        Target("rgb", "composite", fmt("gbrp"), background=WHITE),
        Target("alpha", "alpha", fmt("yuva444p10le")),
        Target("alpha-web", "alpha", fmt("yuva420p")),
        Target("alpha-pre", "alpha", fmt("yuva420p"), alpha_mode="premultiplied"),
        Target("matte", "matte", src),
        Target("small", "composite", src, background=GREEN, size=(48, 32)),
        Target("small-alpha", "alpha", fmt("yuva420p"), size=(64, 64)),
    ]
    comp = Compositor(src, col, W, H, decontaminate=1.0, decontaminate_radius=10.0, decontaminate_refine_radius=2.0)
    return comp, planes, alpha, targets


def test_rendering_in_memory_bounded_steps_does_not_change_the_result(monkeypatch):
    comp, planes, alpha, targets = _batch_scene()
    ctx = comp.begin(planes, alpha)
    assert ctx.frame_clear == [False, True, False, False, False]
    assert ctx.frame_opaque == [False, False, False, True, False]
    assert not ctx.all_clear and not ctx.all_opaque
    assert comp._steps(ctx) == [(0, 5)]                                    # small frames: the whole batch at once
    reference = [comp.render(ctx, t) for t in targets] + [[comp.render_rgba(ctx, 8), comp.render_rgba(ctx, 16)]]
    for pixels, steps in ((2 * W * H, [(0, 2), (2, 4), (4, 5)]), (W * H, [(i, i + 1) for i in range(5)]), (1, None)):
        monkeypatch.setattr(compositor_module, "_CHUNK_PIXELS", pixels)
        fresh = comp.begin(planes, alpha)
        if steps is not None:
            assert comp._steps(fresh) == steps
        again = [comp.render(fresh, t) for t in targets] + [[comp.render_rgba(fresh, 8), comp.render_rgba(fresh, 16)]]
        for expected, result in zip(reference, again):
            assert len(expected) == len(result)
            assert all(torch.equal(a, b) for a, b in zip(expected, result))


def test_rendering_a_batch_equals_rendering_frame_by_frame():
    comp, planes, alpha, targets = _batch_scene()
    ctx = comp.begin(planes, alpha)
    batch = [comp.render(ctx, t) for t in targets]
    batch_rgba = [comp.render_rgba(ctx, 8), comp.render_rgba(ctx, 16, "premultiplied")]

    for frame in range(alpha.shape[0]):
        single = comp.begin([p[frame:frame + 1] for p in planes], alpha[frame:frame + 1])
        for target, expected in zip(targets, batch):
            out = comp.render(single, target)
            for index, plane in enumerate(out):
                assert torch.equal(plane, expected[index][frame:frame + 1]), (target.name, frame, index)
        assert torch.equal(comp.render_rgba(single, 8), batch_rgba[0][frame:frame + 1])
        assert torch.equal(comp.render_rgba(single, 16, "premultiplied"), batch_rgba[1][frame:frame + 1])

    # a different split of the same frames (as after a resume) renders the same pixels as well
    pair = comp.begin([p[2:4] for p in planes], alpha[2:4])
    for target, expected in zip(targets, batch):
        assert all(torch.equal(plane, expected[i][2:4]) for i, plane in enumerate(comp.render(pair, target)))


# --------------------------------------------------------------------------- validation


def test_target_validation_and_hashing():
    pix = fmt("yuv420p")
    with pytest.raises(ConfigError):
        Target("x", "overlay", pix, background=GREEN)
    with pytest.raises(ConfigError):
        Target("x", "composite", pix)                                      # no background
    for bad in ((0, 256, 0), (1, 2), "green", (0.5, None, 3)):
        with pytest.raises(ConfigError):
            Target("x", "composite", pix, background=bad)
    a = Target("x", "composite", pix, background=[0, 255, 0], size=[64, 48])
    b = Target("x", "composite", pix, background=(0, 255, 0), size=(64, 48))
    assert a == b and hash(a) == hash(b) and a.background == (0, 255, 0) and a.size == (64, 48)
    assert len({a, b, Target("y", "matte", pix)}) == 2
    assert Target("m", "matte", pix).background is None and Target("m", "matte", pix).alpha_mode == "straight"


def test_begin_and_render_validation():
    pix = fmt("yuv420p")
    planes = random_planes(pix, 32, 16)
    alpha = torch.ones(1, 1, 16, 32)
    comp = Compositor(pix, color(), 32, 16)
    with pytest.raises(ValueError):
        comp.begin(planes[:2], alpha)
    with pytest.raises(ValueError):
        comp.begin([planes[0], planes[0], planes[0]], alpha)               # chroma at luma size
    with pytest.raises(ValueError):
        comp.begin(planes, torch.ones(1, 1, 32, 16))                       # matte in the wrong orientation
    with pytest.raises(ValueError):
        comp.begin(planes, torch.ones(2, 1, 16, 32))
    with pytest.raises(ValueError):
        comp.begin(planes, alpha, torch.zeros(1, 3, 16, 31))
    with pytest.raises(ValueError):
        comp.begin([p[:0] for p in planes], alpha[:0])                     # empty batch
    with pytest.raises(ConfigError):
        Compositor(pix, color(), 0, 16)
    other = Compositor(pix, color(), 32, 16)
    with pytest.raises(ValueError):
        other.render(comp.begin(planes, alpha), Target("x", "composite", pix, background=GREEN))
    # an alpha plane of the source is ignored
    with_alpha = Compositor(fmt("yuva420p"), color(), 32, 16)
    out = with_alpha.render(with_alpha.begin(planes + [alpha * 255.0], alpha), Target("x", "composite", pix, background=GREEN))
    assert all(torch.equal(o.to(torch.float32), p) for o, p in zip(out, planes))
    # out-of-range strength is clamped, not an error
    assert Compositor(pix, color(), 32, 16, decontaminate=7.0).decontaminate == 1.0
    assert Compositor(pix, color(), 32, 16, decontaminate=-1.0).decontaminate == 0.0


# --------------------------------------------------------------------------- 13. timing probe


@pytest.mark.slow
def test_timing_1080p_yuv420p_cpu(capsys):
    """Measures begin + render(composite) for one 1920x1080 yuv420p frame on the CPU.

    Reports milliseconds; there is deliberately no assertion on speed (machines differ), only on
    the result being a valid frame.
    """

    width, height = 1920, 1080
    pix = fmt("yuv420p")
    col = color("bt709", False)
    gen = torch.Generator().manual_seed(20)
    coarse = torch.rand(1, 3, height // 8, width // 8, generator=gen)
    rgb = F.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False)
    planes = [p.to(torch.float32) for p in to_integer_planes(ColorConverter(pix, col).rgb_to_planes(rgb), pix)]
    del coarse, rgb
    alpha = soft_disc(height, width, 320.0, 12.0)
    target = Target("main", "composite", pix, background=GREEN)

    lines = []
    for strength in (1.0, 0.0):
        comp = Compositor(pix, col, width, height, decontaminate=strength)
        timings = []
        out = None
        for _ in range(4):                                                  # first run warms up
            start = time.perf_counter()
            ctx = comp.begin(planes, alpha)
            out = comp.render(ctx, target)
            timings.append((time.perf_counter() - start) * 1000.0)
            del ctx
        steady = sorted(timings[1:])
        lines.append(
            f"[timing] 1920x1080 yuv420p, cpu ({torch.get_num_threads()} threads), begin + render(composite), "
            f"decontaminate={strength}: median {steady[1]:.1f} ms, min {steady[0]:.1f} ms"
        )
        assert [tuple(p.shape) for p in out] == [(1, 1, 1080, 1920), (1, 1, 540, 960), (1, 1, 540, 960)]
        assert int(out[0][0, 0, 0, 0]) == 173 and int(out[1][0, 0, 0, 0]) == 42 and int(out[2][0, 0, 0, 0]) == 26
        assert torch.equal(out[0][0, 0, 500:580, 900:1020].to(torch.float32), planes[0][0, 0, 500:580, 900:1020])
    with capsys.disabled():
        print()
        for line in lines:
            print(line)
