"""Colour conversion: reference code values, libswscale cross-check, round trips, plane-domain
conversions and the integer / numpy boundary."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from app.compositing.colors import (
    ColorConverter,
    _round_code,
    convert_planes,
    downsample_chroma,
    planes_from_numpy,
    planes_to_numpy,
    to_integer_planes,
    upsample_chroma,
)
from app.core.errors import ConfigError
from app.core.pixfmt import MATRIX_COEFFICIENTS, PixFmt, ResolvedColor, parse_working_pixfmt, quantization

GREEN, BLUE, RED, BLACK, WHITE = (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 0, 0), (255, 255, 255)


def fmt(name: str) -> PixFmt:
    return parse_working_pixfmt(name)


def color(matrix: str = "bt709", full: bool = False) -> ResolvedColor:
    kr, kb = MATRIX_COEFFICIENTS[matrix]
    return ResolvedColor(matrix=matrix, kr=kr, kb=kb, full_range=full)


def random_planes(pix: PixFmt, width: int, height: int, batch: int = 1, seed: int = 0):
    """Random integer code values over the whole code range, as float32 planes."""

    gen = torch.Generator().manual_seed(seed)
    planes = []
    for index in range(pix.num_planes):
        h, w = pix.plane_size(index, width, height)
        planes.append(torch.randint(0, pix.max_value + 1, (batch, 1, h, w), generator=gen).to(torch.float32))
    return planes


def analytic_rgb(y, cb, cr, pix: PixFmt, col: ResolvedColor) -> np.ndarray:
    """Float64 reference of the Y'CbCr -> R'G'B' conversion for co-sited (4:4:4) samples."""

    yo, ys, co, cs = quantization(pix, col.full_range)
    kr, kb = col.kr, col.kb
    kg = 1.0 - kr - kb
    yn = (y.astype(np.float64) - yo) / ys
    cbn = (cb.astype(np.float64) - co) / cs
    crn = (cr.astype(np.float64) - co) / cs
    r = yn + 2.0 * (1.0 - kr) * crn
    b = yn + 2.0 * (1.0 - kb) * cbn
    g = (yn - kr * r - kb * b) / kg
    return np.clip(np.stack([r, g, b], axis=1), 0.0, 1.0)


# --------------------------------------------------------------------------- constant colours


@pytest.mark.parametrize(
    "name, matrix, full, rgb8, expected",
    [
        # the reference values of the specification
        ("yuv420p", "bt709", False, GREEN, [173, 42, 26]),
        ("yuv420p", "smpte170m", False, GREEN, [145, 54, 34]),
        ("yuv420p", "bt709", True, GREEN, [182, 30, 12]),
        ("yuv420p10le", "bt709", False, GREEN, [691, 167, 105]),
        ("gbrp", "bt709", True, GREEN, [255, 0, 0]),
        # well-known broadcast values
        ("yuv420p", "bt709", False, WHITE, [235, 128, 128]),
        ("yuv420p", "bt709", False, BLACK, [16, 128, 128]),
        ("yuv420p", "bt709", False, BLUE, [32, 240, 118]),
        ("yuv420p", "bt709", False, RED, [63, 102, 240]),
        ("yuv444p", "bt709", True, WHITE, [255, 128, 128]),
        ("yuv444p", "bt709", True, BLACK, [0, 128, 128]),
        ("yuv422p10le", "bt709", False, WHITE, [940, 512, 512]),
        ("yuv422p10le", "bt709", False, BLACK, [64, 512, 512]),
        ("yuv420p10le", "bt709", True, WHITE, [1023, 512, 512]),
        ("yuv444p16le", "bt709", False, WHITE, [60160, 32768, 32768]),
        ("gray", "bt709", False, GREEN, [173]),
        ("gray", "bt709", True, GREEN, [182]),
        ("gbrp10le", "bt709", True, WHITE, [1023, 1023, 1023]),
        ("gbrp", "bt709", True, (10, 20, 30), [20, 30, 10]),
        ("gbrap", "bt709", True, RED, [0, 0, 255]),
    ],
)
def test_color_code_values_reference(name, matrix, full, rgb8, expected):
    conv = ColorConverter(fmt(name), color(matrix, full))
    values = conv.color_code_values(rgb8)
    assert values == expected
    assert all(isinstance(v, int) for v in values)


def test_color_code_values_for_another_format_uses_same_matrix_and_range():
    conv = ColorConverter(fmt("yuv420p"), color("bt709", False))
    assert conv.color_code_values(GREEN, fmt("yuv444p10le")) == [691, 167, 105]
    assert conv.color_code_values(GREEN, fmt("yuva444p10le")) == [691, 167, 105]      # alpha adds no colour plane
    assert conv.color_code_values(GREEN, fmt("gray")) == [173]
    full = ColorConverter(fmt("yuv420p"), color("bt709", True))
    # 1023 * 0.7152 = 731.65; 512 - 1023 * 0.385428 = 117.71; 512 - 1023 * 0.454153 = 47.40
    assert full.color_code_values(GREEN, fmt("yuv420p10le")) == [732, 118, 47]


def test_color_code_values_rounds_half_away_from_zero_and_clamps():
    # 8-bit mid grey, limited range: 16 + 219 * 128 / 255 = 125.929... -> 126, neutral chroma
    assert ColorConverter(fmt("yuv444p"), color("bt709", False)).color_code_values((128, 128, 128)) == [126, 128, 128]
    nine = ColorConverter(fmt("gray9le"), color("bt709", True))
    assert nine.color_code_values((127, 127, 127)) == [254]             # 511 * 127 / 255 = 254.498
    assert nine.color_code_values((1, 1, 1)) == [2]                     # 511 / 255 = 2.004
    # the scalar rounding rule itself: halves go away from zero (never to even), float noise just
    # below an exact half does not flip the result, and the code range is enforced
    assert _round_code(2.5, 255) == 3 and _round_code(3.5, 255) == 4
    assert _round_code(254.5, 511) == 255 and _round_code(254.49999999999997, 511) == 255
    assert _round_code(254.4999, 511) == 254
    assert _round_code(-0.5, 255) == 0 and _round_code(-7.0, 255) == 0
    assert _round_code(300.2, 255) == 255 and _round_code(1023.5, 1023) == 1023


@pytest.mark.parametrize("bad", [(0, 256, 0), (-1, 0, 0), (1, 2), "green", None])
def test_color_code_values_rejects_invalid_colours(bad):
    conv = ColorConverter(fmt("yuv420p"), color())
    with pytest.raises(ConfigError):
        conv.color_code_values(bad)


def test_converter_rejects_invalid_matrix():
    with pytest.raises(ConfigError):
        ColorConverter(fmt("yuv420p"), ResolvedColor("broken", 0.7, 0.4, False))


def test_range_used_for_other_families():
    yuv_limited = ColorConverter(fmt("yuv420p"), color("bt709", False))
    assert yuv_limited.full_range_for(fmt("yuv444p10le")) is False
    assert yuv_limited.full_range_for(fmt("gray")) is False
    assert yuv_limited.full_range_for(fmt("gbrp")) is True          # RGB derived from video-range Y'CbCr is full range
    assert yuv_limited.color_code_values(WHITE, fmt("gbrp")) == [255, 255, 255]
    rgb_full = ColorConverter(fmt("gbrp"), color("bt709", True))
    assert rgb_full.full_range_for(fmt("gbrp10le")) is True
    assert rgb_full.full_range_for(fmt("yuv420p")) is False         # Y'CbCr derived from RGB uses video range
    assert rgb_full.color_code_values(GREEN, fmt("yuv420p")) == [173, 42, 26]
    rgb_limited = ColorConverter(fmt("gbrp"), color("bt709", False))
    assert rgb_limited.color_code_values(WHITE) == [235, 235, 235]
    assert rgb_limited.color_code_values(BLACK) == [16, 16, 16]


# --------------------------------------------------------------------------- planes -> R'G'B'


@pytest.mark.parametrize(
    "name, matrix, full, tolerance",
    [
        ("yuv444p", "bt709", False, 2e-6),
        ("yuv444p", "smpte170m", False, 2e-6),
        ("yuv444p", "bt709", True, 2e-6),
        ("yuv444p10le", "bt2020nc", False, 2e-6),
        ("yuv444p12le", "bt709", False, 2e-6),
        ("yuv444p16le", "bt709", False, 2e-6),
    ],
)
def test_planes_to_rgb_matches_analytic_formula(name, matrix, full, tolerance):
    pix, col = fmt(name), color(matrix, full)
    planes = random_planes(pix, 37, 23, batch=2, seed=3)
    rgb = ColorConverter(pix, col).planes_to_rgb(planes)
    assert rgb.shape == (2, 3, 23, 37) and rgb.dtype == torch.float32
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0
    ref = analytic_rgb(planes[0][:, 0].numpy(), planes[1][:, 0].numpy(), planes[2][:, 0].numpy(), pix, col)
    assert np.abs(rgb.numpy().astype(np.float64) - ref).max() < tolerance


_SWS_BILINEAR, _SWS_FULL_CHR_H_INT, _SWS_ACCURATE_RND = 2, 0x2000, 0x40000


@pytest.mark.parametrize("matrix, sws_name", [("bt709", "ITU709"), ("smpte170m", "ITU601")])
@pytest.mark.parametrize("mode", ["default", "accurate"])
def test_planes_to_rgb_matches_libswscale(matrix, sws_name, mode):
    """Cross-check against FFmpeg's libswscale (through PyAV) for yuv420p limited range.

    Luma is random per pixel; chroma is constant in 8x8-sample blocks and only pixels inside a
    block are compared, because the two implementations interpolate chroma differently at edges.
    libswscale's default table-driven path truncates (up to almost 3 code values low); with
    ``SWS_FULL_CHR_H_INT | SWS_ACCURATE_RND`` it is accurate to its 8-bit rounding.
    """

    av = pytest.importorskip("av")
    from av.video import reformatter as rf

    rng = np.random.default_rng(7)
    height, width = 96, 128
    y = rng.integers(16, 236, size=(height, width), dtype=np.uint8)
    blocks = rng.integers(16, 241, size=(2, height // 16, width // 16), dtype=np.uint8)
    cb = np.kron(blocks[0], np.ones((8, 8), np.uint8))
    cr = np.kron(blocks[1], np.ones((8, 8), np.uint8))

    frame = av.VideoFrame.from_ndarray(
        np.concatenate([y.reshape(-1), cb.reshape(-1), cr.reshape(-1)]).reshape(height * 3 // 2, width),
        format="yuv420p",
    )
    space = getattr(rf.Colorspace, sws_name)
    frame.colorspace = space
    frame.color_range = rf.ColorRange.MPEG
    if mode == "default":
        flags, tolerance = rf.Interpolation.BILINEAR, 3.0
    else:
        flags, tolerance = _SWS_BILINEAR | _SWS_FULL_CHR_H_INT | _SWS_ACCURATE_RND, 0.75
    try:
        reference = frame.reformat(
            format="rgb24", src_colorspace=space, dst_colorspace=space,
            src_color_range=rf.ColorRange.MPEG, dst_color_range=rf.ColorRange.JPEG, interpolation=flags,
        ).to_ndarray()
    except (TypeError, ValueError, KeyError) as exc:     # raw swscale flags not accepted by this PyAV build
        pytest.skip(f"PyAV does not accept these swscale flags: {exc}")

    conv = ColorConverter(fmt("yuv420p"), color(matrix, False))
    planes = planes_from_numpy([y[None], cb[None], cr[None]], "cpu")
    mine = conv.planes_to_rgb(planes)[0].permute(1, 2, 0).numpy().astype(np.float64) * 255.0

    rows, cols = np.mgrid[0:height, 0:width]
    crow, ccol = (rows // 2) % 8, (cols // 2) % 8
    inside = (crow >= 1) & (crow <= 6) & (ccol >= 1) & (ccol <= 6)
    assert inside.sum() > 5000
    diff = np.abs(mine - reference.astype(np.float64))[inside]
    assert diff.max() <= tolerance, f"max deviation from libswscale: {diff.max():.3f} code values"
    if mode == "accurate":
        assert np.mean(np.floor(mine + 0.5)[inside] == reference[inside]) > 0.98


def test_chroma_upsampling_is_bilinear_and_centre_sited():
    row = torch.tensor([0.0, 16.0]).view(1, 1, 1, 2)
    up = upsample_chroma(row, 1, 0, 1, 4)
    assert up.flatten().tolist() == [0.0, 4.0, 12.0, 16.0]
    col = torch.tensor([0.0, 16.0]).view(1, 1, 2, 1)
    assert upsample_chroma(col, 0, 1, 4, 1).flatten().tolist() == [0.0, 4.0, 12.0, 16.0]
    odd = upsample_chroma(row, 1, 0, 1, 3)                       # odd width: the last luma column is cropped
    assert odd.flatten().tolist() == [0.0, 4.0, 12.0]
    same = torch.rand(1, 2, 4, 4)
    assert upsample_chroma(same, 0, 0, 4, 4) is same


@pytest.mark.parametrize("name", ["yuv420p", "yuv422p", "yuv440p", "yuv420p10le"])
def test_subsampled_planes_to_rgb_equals_444_with_upsampled_chroma(name):
    pix = fmt(name)
    width, height = 97, 63                                       # odd on purpose
    planes = random_planes(pix, width, height, batch=2, seed=5)
    col = color("bt709", False)
    rgb = ColorConverter(pix, col).planes_to_rgb(planes)
    assert rgb.shape == (2, 3, height, width)

    ch, cw = planes[1].shape[-2:]
    full = [
        F.interpolate(p, size=(ch << pix.chroma_h, cw << pix.chroma_w), mode="bilinear", align_corners=False)[
            ..., :height, :width
        ]
        for p in planes[1:3]
    ]
    reference = ColorConverter(pix.with_chroma(0, 0), col).planes_to_rgb([planes[0], full[0], full[1]])
    assert torch.equal(rgb, reference)

    constant = [torch.full_like(planes[0], 120.0), torch.full_like(planes[1], 90.0), torch.full_like(planes[2], 200.0)]
    flat = ColorConverter(pix, col).planes_to_rgb(constant)
    assert float((flat - flat[:, :, :1, :1]).abs().max()) == 0.0   # constant chroma stays constant


def test_planes_to_rgb_rgb_and_gray_families():
    pix = fmt("gbrp")
    g = torch.tensor([[0.0, 255.0]]).view(1, 1, 1, 2)
    b = torch.tensor([[51.0, 0.0]]).view(1, 1, 1, 2)
    r = torch.tensor([[255.0, 102.0]]).view(1, 1, 1, 2)
    rgb = ColorConverter(pix, color("bt709", True)).planes_to_rgb([g, b, r])
    assert torch.allclose(rgb[0, :, 0, 0], torch.tensor([1.0, 0.0, 0.2]))
    assert torch.allclose(rgb[0, :, 0, 1], torch.tensor([0.4, 1.0, 0.0]))

    ten = ColorConverter(fmt("gbrp10le"), color("bt709", True))
    assert float(ten.planes_to_rgb([torch.full((1, 1, 2, 2), 1023.0)] * 3).min()) == 1.0

    limited = ColorConverter(fmt("gbrp"), color("bt709", False))
    out = limited.planes_to_rgb([torch.full((1, 1, 1, 1), 235.0), torch.full((1, 1, 1, 1), 16.0), torch.full((1, 1, 1, 1), 125.5)])
    assert torch.allclose(out.flatten(), torch.tensor([0.5, 1.0, 0.0]))

    gray = ColorConverter(fmt("gray"), color("bt709", False))
    y = torch.tensor([16.0, 125.5, 235.0, 255.0, 0.0]).view(1, 1, 1, 5)
    grey = gray.planes_to_rgb([y])
    assert grey.shape == (1, 3, 1, 5)
    assert torch.allclose(grey[0, 0], torch.tensor([[0.0, 0.5, 1.0, 1.0, 0.0]]))
    assert torch.equal(grey[:, 0], grey[:, 1]) and torch.equal(grey[:, 0], grey[:, 2])
    grey[:, 0].fill_(7.0)                                        # the three channels do not share memory
    assert float(grey[:, 1].max()) <= 1.0


def test_planes_to_rgb_ignores_alpha_accepts_integer_dtypes_and_keeps_inputs():
    pix = fmt("yuva420p")
    planes = random_planes(pix, 32, 16, batch=2, seed=11)
    assert len(planes) == 4
    conv = ColorConverter(pix, color())
    kept = [p.clone() for p in planes]
    rgb = conv.planes_to_rgb(planes)
    assert all(torch.equal(a, b) for a, b in zip(planes, kept))            # inputs untouched
    assert torch.equal(rgb, conv.planes_to_rgb(planes[:3]))                 # alpha plane ignored
    assert torch.equal(rgb, conv.planes_to_rgb([p.to(torch.uint8) for p in planes]))
    assert torch.equal(rgb, conv.planes_to_rgb([p.to(torch.int32) for p in planes]))
    assert torch.allclose(rgb, conv.planes_to_rgb([p.to(torch.float64) for p in planes]))


def test_planes_to_rgb_validates_shapes():
    conv = ColorConverter(fmt("yuv420p"), color())
    planes = random_planes(fmt("yuv420p"), 32, 16)
    with pytest.raises(ValueError):
        conv.planes_to_rgb(planes[:2])
    with pytest.raises(ValueError):
        conv.planes_to_rgb([planes[0], planes[1], planes[0]])              # chroma at luma size
    with pytest.raises(ValueError):
        conv.planes_to_rgb([p[:, 0] for p in planes])                      # missing channel dimension
    with pytest.raises(ValueError):
        conv.rgb_to_planes(torch.rand(1, 4, 8, 8))


# --------------------------------------------------------------------------- R'G'B' -> planes


@pytest.mark.parametrize(
    "name, matrix, full",
    [
        ("yuv444p", "bt709", False),
        ("yuv444p", "smpte170m", True),
        ("yuv444p10le", "bt709", False),
        ("yuv444p10le", "bt2020nc", True),
        ("gbrp", "bt709", True),
        ("gbrp10le", "bt709", True),
        ("gray", "bt709", False),
    ],
)
def test_round_trip_on_444_formats(name, matrix, full):
    """planes -> R'G'B' -> planes returns the same code values (within 1) for in-gamut pictures."""

    pix, col = fmt(name), color(matrix, full)
    conv = ColorConverter(pix, col)
    gen = torch.Generator().manual_seed(21)

    # (a) colours strictly inside the gamut: no clamping anywhere, so only float rounding remains
    inner = torch.rand(2, 3, 33, 47, generator=gen) * 0.9 + 0.05
    planes = to_integer_planes(conv.rgb_to_planes(inner), pix)
    again = conv.rgb_to_planes(conv.planes_to_rgb(planes))
    for original, back in zip(planes, again):
        assert float((back - original.to(torch.float32)).abs().max()) < 0.01

    # (b) the whole unit cube including its faces: rounding can leave the gamut by half a code
    #     value, which the 0..1 clamp of R'G'B' turns into at most one code value
    cube = torch.rand(2, 3, 33, 47, generator=gen)
    cube[:, :, :4] = torch.round(cube[:, :, :4])                 # saturated primaries / black / white
    planes = to_integer_planes(conv.rgb_to_planes(cube), pix)
    again = to_integer_planes(conv.rgb_to_planes(conv.planes_to_rgb(planes)), pix)
    for original, back in zip(planes, again):
        assert int((back.to(torch.int64) - original.to(torch.int64)).abs().max()) <= 1


@pytest.mark.parametrize("name, full", [("yuv420p", False), ("yuv420p", True), ("yuv420p10le", False), ("yuv444p10le", True),
                                        ("gbrp", True), ("gray10le", False)])
@pytest.mark.parametrize("matrix", ["bt709", "smpte170m"])
def test_constant_picture_rounds_to_color_code_values(name, full, matrix):
    """The float32 frame conversion of a flat colour agrees with the float64 constants."""

    pix = fmt(name)
    conv = ColorConverter(pix, color(matrix, full))
    for rgb8 in (GREEN, BLUE, BLACK, WHITE, RED, (12, 200, 99)):
        picture = (torch.tensor(rgb8, dtype=torch.float32) / 255.0).view(1, 3, 1, 1).expand(2, 3, 9, 13)
        planes = to_integer_planes(conv.rgb_to_planes(picture), pix)
        expected = conv.color_code_values(rgb8)
        assert len(planes) == pix.color_planes
        for index, plane in enumerate(planes):
            assert plane.shape[-2:] == pix.plane_size(index, 13, 9)
            assert int(plane.min()) == int(plane.max()) == expected[index]


def test_rgb_to_planes_box_averages_chroma_over_footprints_for_odd_sizes():
    width, height = 7, 5
    gen = torch.Generator().manual_seed(4)
    rgb = torch.rand(1, 3, height, width, generator=gen)
    conv = ColorConverter(fmt("yuv444p"), color("bt709", False))
    y444, cb444, cr444 = conv.rgb_to_planes(rgb)
    for name in ("yuv420p", "yuv422p", "yuv440p"):
        pix = fmt(name)
        y, cb, cr = conv.rgb_to_planes(rgb, pix)
        assert torch.equal(y, y444)                              # luma does not depend on the subsampling
        assert cb.shape[-2:] == pix.plane_size(1, width, height) == cr.shape[-2:]
        sh, sw = 1 << pix.chroma_h, 1 << pix.chroma_w
        for full, small in ((cb444, cb), (cr444, cr)):
            for j in range(small.shape[-2]):
                for i in range(small.shape[-1]):
                    footprint = full[0, 0, j * sh:(j + 1) * sh, i * sw:(i + 1) * sw]
                    assert footprint.numel() in (1, 2, 4)
                    assert abs(float(small[0, 0, j, i]) - float(footprint.double().mean())) < 1e-4
    assert downsample_chroma(cb444, 0, 0) is cb444


def test_rgb_to_planes_weighted_chroma_reduction():
    """``chroma_weight`` keeps pixels with zero weight out of the chroma sample they share."""

    conv = ColorConverter(fmt("yuv444p"), color("bt709", False))
    target = fmt("yuv420p")
    red = torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1)
    blue = torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1)
    weight = torch.zeros(1, 1, 5, 7)
    weight[0, 0, 0, 0] = 1.0                                     # one visible pixel in the first 2x2 footprint
    weight[0, 0, 2:4, 2:4] = torch.tensor([[0.5, 0.25], [0.0, 0.25]])
    rgb = weight.gt(0) * red + weight.le(0) * blue
    gen = torch.Generator().manual_seed(6)
    rgb[:, :, 2:4, 2:4] = torch.rand(1, 3, 2, 2, generator=gen)

    y, cb, cr = conv.rgb_to_planes(rgb, target, chroma_weight=weight)
    y_plain, cb_plain, cr_plain = conv.rgb_to_planes(rgb, target)
    assert torch.equal(y, y_plain)                               # luma is never weighted
    assert cb.shape == cb_plain.shape == (1, 1, 3, 4)
    # footprint (0, 0): only the red pixel counts -> red's chroma (102.336, 240), not the red/blue mix
    assert abs(float(cb[0, 0, 0, 0]) - 102.336) < 1e-2 and abs(float(cr[0, 0, 0, 0]) - 240.0) < 1e-2
    assert abs(float(cb_plain[0, 0, 0, 0]) - (102.336 + 3 * 240.0) / 4) < 1e-2
    # footprint (1, 1): the weighted mean of its four pixels
    _, cb444, cr444 = conv.rgb_to_planes(rgb)
    w = weight[0, 0, 2:4, 2:4].double()
    for full, small in ((cb444, cb), (cr444, cr)):
        expected = float((full[0, 0, 2:4, 2:4].double() * w).sum() / w.sum())
        assert abs(float(small[0, 0, 1, 1]) - expected) < 1e-3
    # footprints without any weight fall back to the plain average and stay finite
    assert bool(torch.isfinite(cb).all()) and bool(torch.isfinite(cr).all())
    assert float(cb[0, 0, 2, 3]) == float(cb_plain[0, 0, 2, 3]) and float(cr[0, 0, 0, 2]) == float(cr_plain[0, 0, 0, 2])

    # equal weights are the plain box average; formats without subsampled chroma ignore the weight
    for value in (1.0, 0.3):
        even = conv.rgb_to_planes(rgb, target, chroma_weight=torch.full((1, 1, 5, 7), value))
        assert all(float((a - b).abs().max()) < 1e-4 for a, b in zip(even, (y_plain, cb_plain, cr_plain)))
    for name in ("yuv444p", "gbrp", "gray"):
        plain = conv.rgb_to_planes(rgb, fmt(name))
        weighted = conv.rgb_to_planes(rgb, fmt(name), chroma_weight=weight)
        assert all(torch.equal(a, b) for a, b in zip(plain, weighted))
    with pytest.raises(ValueError):
        conv.rgb_to_planes(rgb, target, chroma_weight=torch.zeros(1, 1, 5, 6))


def test_rgb_to_planes_is_unrounded_and_keeps_input():
    conv = ColorConverter(fmt("yuv420p"), color("bt709", False))
    rgb = torch.full((1, 3, 4, 4), 0.5)
    kept = rgb.clone()
    y, cb, cr = conv.rgb_to_planes(rgb)
    assert torch.equal(rgb, kept)
    assert y.dtype == torch.float32 and abs(float(y[0, 0, 0, 0]) - 125.5) < 1e-4      # 16 + 219 / 2, not rounded
    assert abs(float(cb.mean()) - 128.0) < 1e-4 and abs(float(cr.mean()) - 128.0) < 1e-4
    g, b, r = ColorConverter(fmt("gbrp"), color("bt709", True)).rgb_to_planes(
        torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1, 1)
    )
    assert torch.allclose(torch.cat([r, g, b]).flatten(), torch.tensor([25.5, 51.0, 76.5]))


# --------------------------------------------------------------------------- plane-domain conversion


def test_convert_planes_8_to_10_bit_limited_is_exactly_times_four():
    src, dst = fmt("yuv420p"), fmt("yuv420p10le")
    planes = random_planes(src, 97, 63, batch=2, seed=1)
    out = convert_planes(planes, src, dst, full_range=False)
    assert len(out) == 3
    for a, b in zip(planes, out):
        assert b.dtype == torch.float32 and torch.equal(b, a * 4.0)
    twelve = convert_planes(planes, src, fmt("yuv420p12le"), full_range=False)
    assert all(torch.equal(b, a * 16.0) for a, b in zip(planes, twelve))


@pytest.mark.parametrize("name", ["yuv420p", "yuv422p10le", "yuv444p", "gbrp", "gray", "gray16le"])
def test_convert_planes_identity_returns_the_same_values(name):
    pix = fmt(name)
    planes = random_planes(pix, 97, 63, batch=2, seed=2)
    for full in (False, True):
        out = convert_planes(planes, pix, pix, full_range=full)
        assert len(out) == pix.color_planes
        assert all(b is a for a, b in zip(planes, out))                    # not even a copy
    as_int = [p.to(torch.int32) for p in planes]
    out = convert_planes(as_int, pix, pix, full_range=False)
    assert all(b.dtype == torch.float32 and torch.equal(b, a) for a, b in zip(planes, out))


def test_convert_planes_ignores_alpha_on_both_sides():
    src, dst = fmt("yuva420p"), fmt("yuv420p")
    planes = random_planes(src, 32, 16, seed=3)
    out = convert_planes(planes, src, dst, full_range=False)
    assert len(out) == 3 and all(b is a for a, b in zip(planes, out))
    out = convert_planes(planes[:3], dst, src, full_range=False)           # no alpha plane is invented
    assert len(out) == 3 and all(b is a for a, b in zip(planes, out))


def test_convert_planes_420_to_444_keeps_luma_exact_and_interpolates_chroma():
    src, dst = fmt("yuv420p"), fmt("yuv444p")
    width, height = 97, 63
    planes = random_planes(src, width, height, batch=2, seed=4)
    y, cb, cr = convert_planes(planes, src, dst, full_range=False)
    assert y is planes[0]
    assert cb.shape == y.shape == cr.shape
    for small, big in ((planes[1], cb), (planes[2], cr)):
        ref = F.interpolate(small, scale_factor=2, mode="bilinear", align_corners=False)[..., :height, :width]
        assert torch.equal(big, ref)
    # together with a depth increase: luma exactly x4, chroma interpolated then x4
    y10, cb10, _ = convert_planes(planes, src, fmt("yuv444p10le"), full_range=False)
    assert torch.equal(y10, planes[0] * 4.0) and torch.equal(cb10, cb * 4.0)


def test_convert_planes_444_to_420_box_averages_and_mixed_subsampling():
    src = fmt("yuv444p")
    width, height = 9, 7
    planes = random_planes(src, width, height, seed=5)
    y, cb, cr = convert_planes(planes, src, fmt("yuv420p"), full_range=False)
    assert y is planes[0] and cb.shape[-2:] == (4, 5)
    assert abs(float(cb[0, 0, 0, 0]) - float(planes[1][0, 0, 0:2, 0:2].mean())) < 1e-4
    assert abs(float(cb[0, 0, 3, 4]) - float(planes[1][0, 0, 6, 8])) < 1e-4           # 1x1 corner footprint
    assert abs(float(cr[0, 0, 3, 0]) - float(planes[2][0, 0, 6, 0:2].mean())) < 1e-4   # 1x2 bottom footprint

    # 4:2:2 -> 4:2:0 only averages vertically; 4:2:0 -> 4:2:2 only interpolates vertically
    p422 = random_planes(fmt("yuv422p"), width, height, seed=6)
    _, cb420, _ = convert_planes(p422, fmt("yuv422p"), fmt("yuv420p"), full_range=False)
    assert cb420.shape[-2:] == (4, 5)
    assert abs(float(cb420[0, 0, 1, 2]) - float(p422[1][0, 0, 2:4, 2].mean())) < 1e-4
    p420 = random_planes(fmt("yuv420p"), width, height, seed=7)
    _, cb422, _ = convert_planes(p420, fmt("yuv420p"), fmt("yuv422p"), full_range=False)
    assert cb422.shape[-2:] == (7, 5)
    ref = F.interpolate(p420[1], size=(8, 5), mode="bilinear", align_corners=False)[..., :7, :]
    assert torch.equal(cb422, ref)


def test_convert_planes_depth_decrease_rounds_and_clamps():
    src, dst = fmt("gray10le"), fmt("gray")
    values = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 690.0, 691.0, 1021.0, 1022.0, 1023.0]).view(1, 1, 1, -1)
    (out,) = convert_planes([values], src, dst, full_range=False)
    assert out.flatten().tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 173.0, 173.0, 255.0, 255.0, 255.0]
    (full,) = convert_planes([values], src, dst, full_range=True)
    assert full.flatten().tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 172.0, 172.0, 255.0, 255.0, 255.0]


def test_convert_planes_full_range_scales_by_peak_ratio():
    src, dst = fmt("gbrp"), fmt("gbrp10le")
    values = torch.tensor([0.0, 1.0, 128.0, 255.0]).view(1, 1, 1, 4)
    out = convert_planes([values, values, values], src, dst, full_range=True)
    assert len(out) == 3
    assert torch.allclose(out[0].flatten(), torch.tensor([0.0, 1023.0 / 255.0, 128.0 * 1023.0 / 255.0, 1023.0]), atol=1e-3)
    assert to_integer_planes(out, dst)[2].flatten().tolist() == [0, 4, 514, 1023]
    limited = convert_planes([values, values, values], src, dst, full_range=False)
    assert limited[1].flatten().tolist() == [0.0, 4.0, 512.0, 1020.0]


def test_convert_planes_gray_to_yuv_adds_neutral_chroma_and_yuv_to_gray_drops_it():
    width, height = 9, 7
    (y,) = random_planes(fmt("gray"), width, height, batch=2, seed=8)
    out = convert_planes([y], fmt("gray"), fmt("yuv420p"), full_range=False)
    assert out[0] is y
    assert out[1].shape == (2, 1, 4, 5) and float(out[1].min()) == float(out[1].max()) == 128.0
    assert torch.equal(out[1], out[2]) and out[1] is not out[2]
    ten = convert_planes([y], fmt("gray"), fmt("yuv444p10le"), full_range=True)
    assert ten[1].shape == y.shape and float(ten[2].min()) == float(ten[2].max()) == 512.0

    planes = random_planes(fmt("yuv420p"), width, height, seed=9)
    (luma,) = convert_planes(planes, fmt("yuv420p"), fmt("gray"), full_range=False)
    assert luma is planes[0]


def test_convert_planes_rejects_matrix_conversions_and_bad_shapes():
    yuv = random_planes(fmt("yuv420p"), 16, 8)
    with pytest.raises(ConfigError):
        convert_planes(yuv, fmt("yuv420p"), fmt("gbrp"), full_range=False)
    with pytest.raises(ConfigError):
        convert_planes(random_planes(fmt("gbrp"), 16, 8), fmt("gbrp"), fmt("yuv444p"), full_range=True)
    with pytest.raises(ConfigError):
        convert_planes(random_planes(fmt("gbrp"), 16, 8), fmt("gbrp"), fmt("gray"), full_range=True)
    with pytest.raises(ValueError):
        convert_planes(yuv[:1], fmt("yuv420p"), fmt("yuv444p"), full_range=False)
    with pytest.raises(ValueError):
        convert_planes([yuv[0], yuv[0], yuv[0]], fmt("yuv420p"), fmt("yuv444p"), full_range=False)


def test_convert_planes_does_not_modify_inputs():
    src = fmt("yuv444p10le")
    planes = random_planes(src, 15, 11, seed=10)
    kept = [p.clone() for p in planes]
    for target in ("yuv420p", "yuv422p", "yuv420p12le", "gray", "yuv444p"):
        for full in (False, True):
            convert_planes(planes, src, fmt(target), full_range=full)
    assert all(torch.equal(a, b) for a, b in zip(planes, kept))


# --------------------------------------------------------------------------- integer boundary


def test_to_integer_planes_rounds_half_away_from_zero_and_clamps():
    below_half = float(np.nextafter(np.float32(0.5), np.float32(0.0)))     # the float just below 0.5
    values = torch.tensor(
        [0.0, 0.4, 0.5, below_half, 1.5, 2.5, 3.49, 254.5, 255.4, 255.5, 300.0, -0.4, -0.5, -3.0, float("inf")]
    ).view(1, 1, 1, -1)
    kept = values.clone()
    (out,) = to_integer_planes([values], fmt("yuv420p"))
    assert out.dtype == torch.uint8 and out.shape == values.shape
    assert out.flatten().tolist() == [0, 0, 1, 0, 2, 3, 3, 255, 255, 255, 255, 0, 0, 0, 255]
    assert torch.equal(values, kept)                                        # input untouched


@pytest.mark.parametrize(
    "name, dtype, peak",
    [
        ("yuv420p", torch.uint8, 255),
        ("gray9le", torch.int16, 511),
        ("yuv420p10le", torch.int16, 1023),
        ("yuv422p12le", torch.int16, 4095),
        ("yuv444p14le", torch.int16, 16383),
        ("yuv420p16le", torch.int32, 65535),
        ("gray16le", torch.int32, 65535),
        ("gbrap", torch.uint8, 255),
    ],
)
def test_to_integer_planes_dtypes_and_peaks(name, dtype, peak):
    pix = fmt(name)
    values = torch.tensor([0.0, 0.5, peak - 0.5, float(peak), peak + 7.0]).view(1, 1, 1, -1)
    (out,) = to_integer_planes([values], pix)
    assert out.dtype == dtype
    assert out.flatten().tolist() == [0, 1, peak, peak, peak]
    (same,) = to_integer_planes([out], pix)                                 # integer input passes through
    assert same.dtype == dtype and torch.equal(same, out)


def test_planes_from_numpy_uint8_and_uint16_full_range():
    rng = np.random.default_rng(0)
    a8 = rng.integers(0, 256, size=(2, 5, 7), dtype=np.uint8)
    a16 = rng.integers(0, 65536, size=(2, 5, 7), dtype=np.uint16).astype("<u2")
    a16[0, 0, :4] = [0, 32767, 32768, 65535]
    p8, p16 = planes_from_numpy([a8, a16], torch.device("cpu"))
    for plane, arr in ((p8, a8), (p16, a16)):
        assert plane.dtype == torch.float32 and plane.shape == (2, 1, 5, 7) and plane.device.type == "cpu"
        assert np.array_equal(plane[:, 0].numpy(), arr.astype(np.float32))
    assert p16[0, 0, 0, :4].tolist() == [0.0, 32767.0, 32768.0, 65535.0]
    before = p8.clone()
    a8[:] = 0                                                               # the tensors never alias the arrays
    assert torch.equal(p8, before)


def test_planes_from_numpy_handles_big_endian_views_and_read_only_arrays():
    base = np.arange(2 * 6 * 8, dtype=np.uint16).reshape(2, 6, 8) * 601
    big = base.astype(">u2")
    (plane,) = planes_from_numpy([big], "cpu")
    assert np.array_equal(plane[:, 0].numpy(), base.astype(np.float32))

    sliced = base[:, ::2, ::-1]                                             # non-contiguous, negative stride
    (plane,) = planes_from_numpy([sliced], "cpu")
    assert np.array_equal(plane[:, 0].numpy(), sliced.astype(np.float32))

    frozen = np.frombuffer(bytes(range(48)), dtype=np.uint8).reshape(2, 4, 6)
    assert not frozen.flags.writeable
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                     # no "array is not writable" warning
        (plane,) = planes_from_numpy([frozen], "cpu")
    assert np.array_equal(plane[:, 0].numpy(), frozen.astype(np.float32))

    with pytest.raises(ValueError):
        planes_from_numpy([np.zeros((4, 6), np.uint8)], "cpu")
    with pytest.raises(ValueError):
        planes_from_numpy([np.zeros((1, 4, 6), np.complex64)], "cpu")


@pytest.mark.parametrize("name", ["yuv420p", "yuv420p10le", "yuv444p16le", "gray12le", "gbrap"])
def test_numpy_round_trip_is_lossless(name):
    pix = fmt(name)
    rng = np.random.default_rng(3)
    width, height, batch = 13, 9, 2
    arrays = []
    for index in range(pix.num_planes):
        h, w = pix.plane_size(index, width, height)
        arrays.append(rng.integers(0, pix.max_value + 1, size=(batch, h, w)).astype(pix.dtype))
    planes = planes_from_numpy(arrays, "cpu")
    back = planes_to_numpy(to_integer_planes(planes, pix), pix)
    assert len(back) == pix.num_planes
    for original, result in zip(arrays, back):
        assert result.dtype == pix.dtype and result.shape == original.shape
        assert result.flags.c_contiguous
        assert result.tobytes() == original.tobytes()                       # bit for bit, little-endian


def test_planes_to_numpy_accepts_float_and_non_contiguous_planes():
    pix = fmt("yuv420p10le")
    stacked = torch.tensor([[1.4, 2.5], [1022.6, 2000.0]]).view(1, 2, 1, 2).expand(2, 2, 3, 2).contiguous()
    views = [stacked[:, 0:1], stacked[:, 1:2]]                              # channel slices: not contiguous for B > 1
    out = planes_to_numpy(views, pix)
    assert all(a.dtype == np.dtype("<u2") and a.flags.c_contiguous and a.shape == (2, 3, 2) for a in out)
    assert out[0][0, 0].tolist() == [1, 3] and out[1][1, 2].tolist() == [1023, 1023]
    with pytest.raises(ValueError):
        planes_to_numpy([torch.zeros(1, 2, 3, 4, dtype=torch.uint8)], fmt("yuv420p"))
