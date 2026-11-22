"""Unit tests for the matte refinement operators (app/refinement/ops.py).

All tensors are small and synthetic; everything runs on the CPU and is deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.core.errors import ConfigError, VmtError
from app.refinement import ops
from app.refinement.ops import RefinementError

# --------------------------------------------------------------------------- synthetic mattes


def soft_disc(h: int, w: int, cy: float, cx: float, radius: float, soft: float = 1.5) -> torch.Tensor:
    """[1, 1, h, w] disc with a linear edge ramp of ``2 * soft`` pixels (exact 1 inside, exact 0 outside)."""

    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    d = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return ((radius - d) / (2.0 * soft) + 0.5).clamp(0.0, 1.0)[None, None]


def hard_rect(h: int, w: int, y0: int, y1: int, x0: int, x1: int) -> torch.Tensor:
    a = torch.zeros(1, 1, h, w)
    a[..., y0:y1, x0:x1] = 1.0
    return a


def half_crossing(row: torch.Tensor) -> float:
    """Sub-pixel boundary position at which a left-to-right decreasing row falls through 0.5."""

    idx = (row >= 0.5).nonzero().flatten()
    assert idx.numel() > 0
    i = int(idx.max())
    if i + 1 >= row.numel():
        return float(i + 1)
    a, b = float(row[i]), float(row[i + 1])
    return i + 0.5 + (a - 0.5) / max(a - b, 1e-9)


def two_tone_image(h: int, w: int, edge: int, noise: float = 0.0, seed: int = 0) -> torch.Tensor:
    """[1, 3, h, w] picture with a strong vertical colour edge between column ``edge - 1`` and ``edge``."""

    img = torch.empty(1, 3, h, w)
    img[..., :edge] = torch.tensor([0.75, 0.55, 0.45]).view(1, 3, 1, 1)
    img[..., edge:] = torch.tensor([0.15, 0.35, 0.25]).view(1, 3, 1, 1)
    if noise > 0.0:
        g = torch.Generator().manual_seed(seed)
        img = (img + noise * torch.randn(1, 3, h, w, generator=g)).clamp(0.0, 1.0)
    return img


# --------------------------------------------------------------------------- levels


def test_levels_is_exact_at_both_ends_and_linear_between():
    black, white = 0.015, 0.985
    g = torch.Generator().manual_seed(1)
    a = torch.rand(2, 1, 37, 53, generator=g)
    # Values exactly at, just below and just above the two points (as float32).
    b32, w32 = torch.tensor(black, dtype=torch.float32), torch.tensor(white, dtype=torch.float32)
    specials = torch.stack(
        [
            b32, torch.nextafter(b32, torch.tensor(0.0)), torch.nextafter(b32, torch.tensor(1.0)),
            w32, torch.nextafter(w32, torch.tensor(0.0)), torch.nextafter(w32, torch.tensor(1.0)),
            torch.tensor(0.0), torch.tensor(1.0),
        ]
    )
    a[0, 0, 0, : specials.numel()] = specials
    before = a.clone()
    out = ops.levels(a, black, white)

    assert torch.equal(a, before)                              # input untouched
    assert out.dtype == torch.float32 and out.shape == a.shape
    assert bool((out[a <= b32] == 0.0).all())                  # exactly 0.0 at and below the black point
    assert bool((out[a >= w32] == 1.0).all())                  # exactly 1.0 at and above the white point
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    inner = (a > b32) & (a < w32)
    expected = (a[inner].double() - black) / (white - black)
    assert torch.allclose(out[inner].double(), expected, atol=2e-6)
    assert float(out[0, 0, 0, 2]) > 0.0 and float(out[0, 0, 0, 4]) < 1.0   # the neighbours of the points are NOT snapped
    # monotonic
    order = torch.argsort(a.flatten())
    assert bool((torch.diff(out.flatten()[order]) >= 0).all())


def test_levels_identity_per_pixel_and_invalid():
    a = torch.tensor([-0.2, 0.0, 0.3, 1.0, 1.4]).view(1, 1, 1, 5)
    assert torch.equal(ops.levels(a, 0.0, 1.0), a.clamp(0.0, 1.0))

    # Per-pixel levels: the left half uses (0.1, 0.9), the right half the identity (0, 1).
    a = torch.linspace(0.0, 1.0, 11).repeat(2).view(1, 1, 2, 11)
    black = torch.zeros(1, 1, 2, 11)
    white = torch.ones(1, 1, 2, 11)
    black[..., 0, :] = 0.1
    white[..., 0, :] = 0.9
    out = ops.levels(a, black, white)
    assert torch.allclose(out[0, 0, 1], a[0, 0, 1])
    assert float(out[0, 0, 0, 1]) == 0.0 and float(out[0, 0, 0, 9]) == 1.0      # 0.1 -> 0, 0.9 -> 1 exactly
    assert out[0, 0, 0, 5] == pytest.approx(0.5, abs=1e-6)

    for bad in [(0.6, 0.4), (0.5, 0.5), (-0.1, 0.9), (0.1, 1.1)]:
        with pytest.raises(ConfigError):
            ops.levels(a, *bad)
    with pytest.raises(RefinementError):
        ops.levels(torch.zeros(4, 4), 0.1, 0.9)
    assert issubclass(RefinementError, VmtError)


# --------------------------------------------------------------------------- erode / dilate


@pytest.mark.parametrize("radius", [1.0, 2.0, 2.5, 0.4])
def test_erode_dilate_monotonic_and_dual(radius):
    g = torch.Generator().manual_seed(2)
    a = torch.rand(2, 1, 45, 61, generator=g)                  # odd size, batch 2
    a[0] = soft_disc(45, 61, 22.0, 30.0, 14.0)[0]
    before = a.clone()
    er, di = ops.erode(a, radius), ops.dilate(a, radius)
    assert torch.equal(a, before)
    assert bool((er <= a + 1e-7).all()) and bool((a <= di + 1e-7).all())
    # a larger radius erodes / dilates at least as much
    assert bool((ops.erode(a, radius + 1.0) <= er + 1e-7).all())
    assert bool((ops.dilate(a, radius + 1.0) >= di - 1e-7).all())
    # duality: eroding the matte is dilating its complement
    assert torch.allclose(er, 1.0 - ops.dilate(1.0 - a, radius), atol=1e-6)
    assert torch.allclose(di, 1.0 - ops.erode(1.0 - a, radius), atol=1e-6)


def test_erode_dilate_fractional_plateaus_and_noop():
    a = soft_disc(64, 96, 32.0, 48.0, 20.0)
    assert ops.erode(a, 0.0) is a and ops.dilate(a, 0.0) is a          # nothing to do: the input itself
    lo, hi, mid = ops.dilate(a, 1.0), ops.dilate(a, 2.0), ops.dilate(a, 1.5)
    assert bool((mid >= lo - 1e-7).all()) and bool((mid <= hi + 1e-7).all())
    assert torch.allclose(mid, 0.5 * (lo + hi), atol=1e-6)
    # plateaus stay exact, the subject shrinks / grows
    for out in (ops.erode(a, 2.5), ops.dilate(a, 2.5)):
        assert bool((out[..., 28:36, 44:52] == 1.0).all())
        assert bool((out[..., :4, :] == 0.0).all())
    assert float(ops.erode(a, 2.0).sum()) < float(a.sum()) < float(ops.dilate(a, 2.0).sum())
    for const in (0.0, 1.0, 0.37):
        c = torch.full((1, 1, 17, 23), const)
        assert torch.equal(ops.erode(c, 1.5), c) and torch.equal(ops.dilate(c, 1.5), c)
    with pytest.raises(ConfigError):
        ops.erode(a, -1.0)


# --------------------------------------------------------------------------- feather / smooth


@pytest.mark.parametrize("op, amount", [(ops.smooth, 1.7), (ops.feather, 3.0), (ops.smooth, 0.4), (ops.feather, 0.6)])
def test_feather_and_smooth_leave_constant_images_unchanged(op, amount):
    for const in (0.0, 1.0):
        c = torch.full((2, 1, 33, 47), const)
        assert torch.equal(op(c, amount), c)                    # bit-exact: the compositor needs exact 0 / 1
    c = torch.full((1, 1, 33, 47), 0.37)
    assert torch.allclose(op(c, amount), c, atol=1e-6)
    assert op(c, 0.0) is c


def test_feather_softens_the_edge_only():
    a = hard_rect(80, 120, 20, 60, 30, 90)
    before = a.clone()
    out = ops.feather(a, 4.0)
    assert torch.equal(a, before)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert bool((out[..., 30:50, 45:75] == 1.0).all())          # interior: exactly opaque
    assert bool((out[..., :10, :] == 0.0).all()) and bool((out[..., :, :18] == 0.0).all())   # far outside: exactly 0
    row = out[0, 0, 40]
    assert 0.0 < float(row[28]) < 0.5 < float(row[31]) < 1.0     # a soft ramp across the former hard edge
    assert half_crossing(row.flip(0)) == pytest.approx(120 - 30, abs=0.05)     # the 50 % contour did not move
    assert float(out.sum()) == pytest.approx(float(a.sum()), rel=2e-3)          # symmetric: no shrinking / growing
    # about 95 % of the transition lies within +- radius of the edge
    assert float(row[30 - 4 - 1]) < 0.03 and float(row[30 + 4]) > 0.97
    # smooth is the same filter parameterised by sigma
    assert torch.allclose(ops.smooth(a, 2.0), out, atol=1e-7)
    assert int(torch.unique(ops.smooth(a, 1.0)).numel()) > 10


# --------------------------------------------------------------------------- guided filter


@pytest.mark.parametrize("offset", [-3, 3, -2, 2])
@pytest.mark.parametrize("noise", [0.0, 0.03])
def test_guided_filter_snaps_the_mask_edge_to_the_image_edge(offset, noise):
    h, w, edge = 96, 160, 80
    img = two_tone_image(h, w, edge, noise=noise, seed=5)
    mask = torch.zeros(1, 1, h, w)
    mask[..., : edge + offset] = 1.0                            # mask edge 2-3 px away from the image edge
    before_mask, before_img = mask.clone(), img.clone()

    out = ops.guided_filter(img, mask, 6.0, 1e-3)

    assert torch.equal(mask, before_mask) and torch.equal(img, before_img)
    assert out.shape == mask.shape and out.dtype == torch.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert bool((out[..., : edge - 30] == 1.0).all())           # far interior: exactly 1
    assert bool((out[..., edge + 30 :] == 0.0).all())           # far exterior: exactly 0
    positions = torch.tensor([half_crossing(out[0, 0, y]) for y in range(20, h - 20)])
    mask_edge, image_edge = float(edge + offset), float(edge)
    # The matte edge moved from the mask edge to (almost) the image edge.
    assert float((positions - image_edge).abs().mean()) < 0.6
    assert float((positions - image_edge).abs().max()) < abs(mask_edge - image_edge) - 1.0
    # The result is a soft matte, not a mask.
    assert int(torch.unique(out).numel()) > 10


def test_guided_filter_variants_batch_and_errors():
    h, w, edge = 71, 125, 60                                    # odd size
    img = torch.cat([two_tone_image(h, w, edge, 0.02, seed=1), two_tone_image(h, w, edge, 0.02, seed=2)])
    mask = torch.zeros(2, 1, h, w)
    mask[..., : edge - 3] = 1.0
    reference = None
    for subsample in (None, 1, 2, 3):
        out = ops.guided_filter(img, mask, 6.0, 1e-3, subsample=subsample)
        assert out.shape == mask.shape
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
        assert bool((out[..., :25] == 1.0).all()) and bool((out[..., 95:] == 0.0).all())
        for b in range(2):
            pos = torch.tensor([half_crossing(out[b, 0, y]) for y in range(15, h - 15)])
            assert float(pos.mean()) > edge - 1.6                # moved at least half of the 3 px offset towards the edge
        # each frame of the batch is filtered on its own
        single = ops.guided_filter(img[1:2], mask[1:2], 6.0, 1e-3, subsample=subsample)
        assert torch.allclose(out[1:2], single, atol=1e-5)
        reference = out if reference is None else reference
    # a constant source stays exactly constant whatever the picture shows
    g = torch.Generator().manual_seed(3)
    busy = torch.rand(1, 3, 40, 56, generator=g)
    assert torch.equal(ops.guided_filter(busy, torch.ones(1, 1, 40, 56), 5.0, 1e-3), torch.ones(1, 1, 40, 56))
    assert torch.equal(ops.guided_filter(busy, torch.zeros(1, 1, 40, 56), 5.0, 1e-3), torch.zeros(1, 1, 40, 56))
    # radius 0 switches the filter off
    soft = torch.rand(1, 1, 40, 56, generator=g)
    assert torch.equal(ops.guided_filter(busy, soft, 0.0, 1e-3), soft)

    with pytest.raises(RefinementError):
        ops.guided_filter(busy[:, :2], soft, 4.0, 1e-3)          # not an RGB guide
    with pytest.raises(RefinementError):
        ops.guided_filter(busy[..., :30, :], soft, 4.0, 1e-3)    # size mismatch
    with pytest.raises(ConfigError):
        ops.guided_filter(busy, soft, 4.0, 0.0)
    with pytest.raises(ConfigError):
        ops.guided_filter(busy, soft, 4.0, 1e-3, subsample=0)


def test_guided_filter_on_a_flat_picture_is_a_plain_smoother():
    """Where the picture has no edge the filter box-filters the matte: same mass, no overshoot."""

    a = soft_disc(72, 100, 36.0, 50.0, 20.0, soft=2.0)
    flat = torch.full((1, 3, 72, 100), 0.5)
    out = ops.guided_filter(flat, a, 4.0, 1e-3)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert float(out.sum()) == pytest.approx(float(a.sum()), rel=0.02)
    # Beyond the support of the filter (two cascaded windows) the plateaus are exact.
    assert bool((out[..., 34:39, 48:53] == 1.0).all()) and bool((out[..., :3, :] == 0.0).all())


# --------------------------------------------------------------------------- mask_to_alpha


def _transition_width(row: torch.Tensor) -> int:
    return int(((row > 0.05) & (row < 0.95)).sum())


@pytest.mark.parametrize("noise", [0.0, 0.03])
def test_mask_to_alpha_keeps_the_mask_crisp_where_the_picture_has_no_edge(noise):
    """The plain guided filter turns an unsupported mask edge into a wide ramp; mask_to_alpha does not."""

    h, w, edge = 96, 160, 77
    g = torch.Generator().manual_seed(11)
    flat = (torch.full((1, 3, h, w), 0.45) + noise * torch.randn(1, 3, h, w, generator=g)).clamp(0.0, 1.0)
    mask = torch.zeros(1, 1, h, w)
    mask[..., :edge] = 1.0
    before = mask.clone()
    plain = ops.guided_filter(flat, mask, 6.0, 1e-3)
    out = ops.mask_to_alpha(flat, mask, 6.0, 1e-3, soften_px=0.75)
    assert torch.equal(mask, before)
    rows = range(20, h - 20)
    plain_width = sum(_transition_width(plain[0, 0, y]) for y in rows) / len(rows)
    width = sum(_transition_width(out[0, 0, y]) for y in rows) / len(rows)
    assert plain_width > 15.0                                   # the documented weakness of the plain filter
    assert 1.0 <= width <= 4.0                                  # anti-aliased, but still a crisp edge
    positions = torch.tensor([half_crossing(out[0, 0, y]) for y in rows])
    assert float((positions - edge).abs().max()) < 0.2          # and it stayed where the mask put it
    assert bool((out[..., : edge - 30] == 1.0).all()) and bool((out[..., edge + 30 :] == 0.0).all())
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert int(torch.unique(out).numel()) > 2                   # soft, not binary


@pytest.mark.parametrize("offset", [-3, 3])
def test_mask_to_alpha_still_snaps_to_a_strong_image_edge(offset):
    h, w, edge = 96, 160, 80
    img = two_tone_image(h, w, edge, noise=0.02, seed=6)
    mask = torch.zeros(1, 1, h, w)
    mask[..., : edge + offset] = 1.0
    out = ops.mask_to_alpha(img, mask, 6.0, 1e-3, soften_px=0.75)
    positions = torch.tensor([half_crossing(out[0, 0, y]) for y in range(20, h - 20)])
    assert float((positions - edge).abs().mean()) < 0.6         # moved from the mask edge to the image edge
    assert bool((out[..., : edge - 30] == 1.0).all()) and bool((out[..., edge + 30 :] == 0.0).all())
    # where the picture explains the mask the result IS the guided filter
    assert torch.allclose(out, ops.guided_filter(img, mask, 6.0, 1e-3), atol=0.05)


def test_mask_to_alpha_leaves_an_unsupported_soft_matte_alone():
    """Without a softening fallback (soft input) and without image support nothing changes."""

    a = soft_disc(72, 100, 36.0, 50.0, 20.0, soft=2.0)
    flat = torch.full((1, 3, 72, 100), 0.5)
    out = ops.mask_to_alpha(flat, a, 6.0, 1e-3)
    assert torch.allclose(out, a, atol=1e-6)
    assert float((ops.guided_filter(flat, a, 6.0, 1e-3) - a).abs().max()) > 0.2      # the plain filter would blur it
    with pytest.raises(ConfigError):
        ops.mask_to_alpha(flat, a, 6.0, 1e-3, soften_px=-1.0)
    with pytest.raises(RefinementError):
        ops.mask_to_alpha(flat[:, :1], a, 6.0, 1e-3)


# --------------------------------------------------------------------------- remove_small_components


def _blob_with_speck(h: int = 96, w: int = 144):
    main = soft_disc(h, w, 50.0, 60.0, 26.0, soft=2.0)
    speck = 0.9 * soft_disc(h, w, 14.0, 126.0, 2.5, soft=2.0)            # small blob with a soft fringe
    return torch.maximum(main, speck), main, speck


def test_remove_small_components_removes_speck_with_fringe_and_keeps_main_blob():
    a, main, speck = _blob_with_speck()
    assert float(speck.max()) > 0.8 and int(((speck > 0) & (speck < 0.05)).sum()) > 0     # there IS a faint fringe
    before = a.clone()
    out, removed = ops.remove_small_components(a, min_area_px=200.0)
    assert torch.equal(a, before)
    assert removed == 1
    assert torch.equal(out, main)                                # speck and fringe gone, main blob bit-identical
    assert bool((out[..., :30, 110:] == 0.0).all())

    # A threshold below the speck's size keeps everything (and returns the input itself).
    same, removed = ops.remove_small_components(a, min_area_px=5.0)
    assert removed == 0 and same is a
    # Disabled.
    same, removed = ops.remove_small_components(a, min_area_px=0.0)
    assert removed == 0 and same is a


def test_remove_small_components_never_removes_the_largest_and_handles_batches():
    h, w = 64, 96
    small = soft_disc(h, w, 30.0, 40.0, 5.0)                    # the only real content: about 80 px
    tiny = soft_disc(h, w, 10.0, 80.0, 1.5)
    frame0 = torch.maximum(small, tiny)
    frame1 = torch.zeros(1, 1, h, w)                             # empty matte
    frame2 = torch.ones(1, 1, h, w)                              # full matte
    batch = torch.cat([frame0, frame1, frame2])
    out, removed = ops.remove_small_components(batch, min_area_px=10_000.0)
    assert removed == 1
    assert torch.equal(out[0:1], small)                          # the largest survives although it is "small"
    assert torch.equal(out[1:2], frame1) and torch.equal(out[2:3], frame2)


def test_remove_small_components_on_a_block_reduced_map():
    """A single isolated pixel must stay visible when the labelling runs at reduced resolution."""

    h, w = 150, 262                                              # not a multiple of the block size
    main = soft_disc(h, w, 80.0, 100.0, 40.0)
    a = main.clone()
    a[0, 0, 7, 255] = 1.0                                        # one-pixel speck near the corner
    a[0, 0, 143, 259] = 0.6                                      # and one in the partial last block
    out, removed = ops.remove_small_components(a, min_area_px=30.0, max_side=64)        # block = 5
    assert removed == 2
    assert torch.equal(out, main)
    # areas are counted in full-resolution pixels: a 6x6 block is not "small" for a limit of 30
    b = main.clone()
    b[0, 0, 10:16, 240:246] = 1.0
    out, removed = ops.remove_small_components(b, min_area_px=30.0, max_side=64)
    assert removed == 0 and torch.equal(out, b)
    out, removed = ops.remove_small_components(b, min_area_px=40.0, max_side=64)
    assert removed == 1 and torch.equal(out, main)


# --------------------------------------------------------------------------- fill_small_holes


def test_fill_small_holes_fills_enclosed_hole_only():
    h, w = 96, 144
    body = hard_rect(h, w, 16, 80, 24, 120)
    hole = soft_disc(h, w, 48.0, 60.0, 6.0, soft=2.0)            # soft-edged hole, about 110 px
    a = body * (1.0 - hole)
    assert float(a[0, 0, 48, 60]) == 0.0
    before = a.clone()
    out, filled = ops.fill_small_holes(a, max_area_px=400.0)
    assert torch.equal(a, before)
    assert filled == 1
    assert torch.equal(out, body)                                # hole AND its soft fringe are exactly 1, rest identical

    # Larger than the limit: untouched.
    same, filled = ops.fill_small_holes(a, max_area_px=60.0)
    assert filled == 0 and same is a
    # Disabled.
    same, filled = ops.fill_small_holes(a, max_area_px=0.0)
    assert filled == 0 and same is a


def test_fill_small_holes_ignores_bays_and_border_regions():
    h, w = 96, 144
    # A bay: a notch that is open to the outer background.
    bay = hard_rect(h, w, 16, 80, 24, 120)
    bay[..., 40:52, 24:70] = 0.0
    out, filled = ops.fill_small_holes(bay, max_area_px=5000.0)
    assert filled == 0 and out is bay
    # A pocket closed by the frame border instead of by the subject is not a hole either.
    pocket = hard_rect(h, w, 0, 60, 30, 110)
    pocket[..., 0:20, 60:80] = 0.0
    out, filled = ops.fill_small_holes(pocket, max_area_px=5000.0)
    assert filled == 0 and out is pocket
    # A gap bounded by semi-transparent matte (hair, motion blur) is not enclosed by solid matte.
    veil = 0.6 * hard_rect(h, w, 16, 80, 24, 120)
    veil[..., 44:52, 60:68] = 0.0
    out, filled = ops.fill_small_holes(veil, max_area_px=5000.0)
    assert filled == 0 and out is veil
    # A shallow dip that never falls below the hole threshold is left alone.
    dip = hard_rect(h, w, 16, 80, 24, 120)
    dip[..., 44:52, 60:68] = 0.7
    out, filled = ops.fill_small_holes(dip, max_area_px=5000.0)
    assert filled == 0 and out is dip
    # ... but the same patch with a real hole inside is filled completely, shallow part included.
    dip[..., 46:50, 62:66] = 0.1
    out, filled = ops.fill_small_holes(dip, max_area_px=5000.0)
    assert filled == 1 and torch.equal(out, hard_rect(h, w, 16, 80, 24, 120))


def test_fill_small_holes_batch_and_block_reduction():
    h, w = 150, 262
    body = soft_disc(h, w, 75.0, 130.0, 60.0)
    a = body.clone()
    a[0, 0, 70, 131] = 0.0                                       # one-pixel hole
    a[0, 0, 60:64, 100:104] = 0.2                                # 16 px hole
    big = body.clone()
    big[0, 0, 50:100, 100:150] = 0.0                             # 2500 px hole
    batch = torch.cat([a, big, torch.zeros(1, 1, h, w), torch.ones(1, 1, h, w)])
    out, filled = ops.fill_small_holes(batch, max_area_px=200.0, max_side=64)           # block = 5
    assert filled == 2
    assert torch.equal(out[0:1], body)
    assert torch.equal(out[1:], batch[1:])                       # too large / empty / full: untouched


# --------------------------------------------------------------------------- detail_map


def _strand_scene(h: int = 96, w: int = 160):
    """A solid body with a soft straight edge plus 1-2 px wide semi-transparent strands above it."""

    body = torch.zeros(1, 1, h, w)
    body[..., 60:, :] = 1.0
    body[..., 58, :] = 0.66
    body[..., 59, :] = 0.9                                       # soft edge
    strands = torch.zeros(1, 1, h, w)
    for k, x in enumerate(range(30, 130, 9)):
        width = 1 + (k % 2)                                      # alternating 1 and 2 px
        strands[..., 22:58, x : x + width] = 0.7
    return torch.maximum(body, strands), body, strands > 0


def test_detail_map_high_on_strands_low_on_flat_and_plain_edges():
    a, body, strand_mask = _strand_scene()
    before = a.clone()
    d = ops.detail_map(a, 2.0)
    assert torch.equal(a, before)
    assert d.shape == a.shape and float(d.min()) >= 0.0 and float(d.max()) <= 1.0
    assert float(d[strand_mask].mean()) > 0.95                   # strands: fully protected
    assert float(d[strand_mask].min()) > 0.5
    assert bool((d[..., 75:, :] == 0.0).all())                   # inside the body: exactly 0
    assert bool((d[..., :8, :] == 0.0).all())                    # empty background: exactly 0
    assert float(d[..., 22:58, :16].max()) == 0.0                # background beside the hair

    # A plain soft edge of a large shape - straight, diagonal or round - is not "detail".
    for plain in (
        body,
        soft_disc(120, 120, 60.0, 60.0, 40.0, soft=2.0),
        soft_disc(120, 120, 60.0, 60.0, 40.3, soft=0.01),        # hard edge
        (torch.arange(120.0).view(1, -1) + torch.arange(120.0).view(-1, 1) < 130.0).float()[None, None],   # diagonal
    ):
        assert float(ops.detail_map(plain, 2.0).max()) < 0.05

    # The spread parameter widens the protected zone around the strands.
    wide = ops.detail_map(a, 2.0, spread_px=8.0)
    assert float(wide[0, 0, 40, 30 + 5]) > 0.9 > float(d[0, 0, 40, 30 + 5])
    assert bool((wide >= d - 1e-6)[..., 22:58, :].all())
    assert ops.detail_map_support(2.0, 8.0) > ops.detail_map_support(2.0) >= 4


def test_detail_map_sees_thin_gaps_and_ignores_matte_noise():
    h, w = 80, 120
    solid = hard_rect(h, w, 10, 70, 10, 110)
    gaps = solid.clone()
    gaps[..., 20:60, 50:52] = 0.2                                # a 2 px wide dark gap inside solid matte
    d = ops.detail_map(gaps, 2.0)
    assert float(d[..., 25:55, 50:52].min()) > 0.9
    assert float(d[..., 25:55, 20:30].max()) == 0.0
    # Noise of a few percent on a plateau or on a ramp is not detail.
    g = torch.Generator().manual_seed(4)
    noisy = (0.5 + 0.015 * torch.randn(1, 1, h, w, generator=g)).clamp(0.0, 1.0)
    assert float(ops.detail_map(noisy, 2.0).max()) < 0.05
    ramp = torch.linspace(0.0, 1.0, w).view(1, 1, 1, w).expand(1, 1, h, w)
    noisy_ramp = (ramp + 0.015 * torch.randn(1, 1, h, w, generator=g)).clamp(0.0, 1.0)
    assert float(ops.detail_map(noisy_ramp, 2.0).max()) < 0.05


def test_detail_map_ignores_edges_that_leave_the_frame():
    """A subject cut off by the frame border (diagonal shoulder line, any angle) is a plain edge."""

    h, w = 90, 130
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    for slope in (1.0, -1.0, 0.4, -2.5):
        plane = ((yy - h / 2) - slope * (xx - w / 2)).clamp(-1.0, 1.0) * 0.5 + 0.5      # soft straight edge through the frame
        assert float(ops.detail_map(plane[None, None], 2.0).max()) == 0.0
        assert float(ops.detail_map((plane > 0.5).float()[None, None], 3.0).max()) == 0.0


# --------------------------------------------------------------------------- shared contract


def test_operators_accept_other_dtypes_and_reject_bad_shapes():
    a = soft_disc(40, 56, 20.0, 28.0, 10.0)
    half = a.to(torch.float16)
    for out in (
        ops.levels(half, 0.1, 0.9),
        ops.erode(half, 1.0),
        ops.smooth(half, 1.0),
        ops.detail_map(half, 2.0),
        ops.remove_small_components(half, 10.0)[0],
        ops.fill_small_holes(half, 10.0)[0],
    ):
        assert out.dtype == torch.float32 and out.shape == a.shape
    for fn in (
        lambda x: ops.erode(x, 1.0),
        lambda x: ops.feather(x, 1.0),
        lambda x: ops.detail_map(x),
        lambda x: ops.remove_small_components(x, 10.0),
        lambda x: ops.fill_small_holes(x, 10.0),
    ):
        with pytest.raises(RefinementError):
            fn(torch.zeros(1, 3, 8, 8))
        with pytest.raises(RefinementError):
            fn(torch.zeros(8, 8))


# --------------------------------------------------------------------------- clean-up: reference check
#
# remove_small_components / fill_small_holes only look at the bounding boxes of candidate regions.
# These straightforward whole-map versions define what they must return, bit for bit.

def _reference_remove(alpha, min_area, threshold=0.05, max_side=960):
    import cv2

    a = alpha.to(torch.float32)
    batch, _, height, width = a.shape
    block = ops._reduction_block(height, width, max_side)
    counts = ops._counts_to_numpy(ops._block_counts(a > threshold, block), block)[:, 0]
    kernel = ops._fringe_kernel(block)
    remove = np.zeros(counts.shape, dtype=bool)
    removed = 0
    for i in range(batch):
        support = (counts[i] > 0).astype(np.uint8)
        n_labels, labels = cv2.connectedComponents(support, connectivity=8, ltype=cv2.CV_32S)
        if n_labels <= 2:
            continue
        areas = np.bincount(labels.ravel(), weights=counts[i].ravel().astype(np.float64), minlength=n_labels)
        small = areas < min_area
        small[0] = False
        small[1 + int(np.argmax(areas[1:]))] = False
        if not small.any():
            continue
        gone = small[labels]
        remove[i] = cv2.dilate(gone.astype(np.uint8), kernel).astype(bool) & ~(support.astype(bool) & ~gone)
        removed += int(small.sum())
    if not removed:
        return a, 0
    keep = torch.from_numpy(~remove).to(a.dtype).unsqueeze(1)
    return a * ops._block_upsample(keep, block, height, width), removed


def _reference_fill(alpha, max_area, threshold=0.5, rim=0.9, max_side=960):
    import cv2

    a = alpha.to(torch.float32)
    batch, _, height, width = a.shape
    block = ops._reduction_block(height, width, max_side)
    counts = ops._counts_to_numpy(ops._block_counts(torch.cat([a < rim, a < threshold], dim=1), block), block)
    kernel = ops._fringe_kernel(block)
    low_h, low_w = counts.shape[-2:]
    fill = np.zeros((batch, low_h, low_w), dtype=bool)
    filled = 0
    for i in range(batch):
        basin, core = counts[i, 0], counts[i, 1]
        background = (basin > 0).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=4, ltype=cv2.CV_32S)
        left, top = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
        right, bottom = left + stats[:, cv2.CC_STAT_WIDTH], top + stats[:, cv2.CC_STAT_HEIGHT]
        enclosed = ~((left == 0) | (top == 0) | (right == low_w) | (bottom == low_h))
        enclosed[0] = False
        basin_area = np.bincount(labels.ravel(), weights=basin.ravel().astype(np.float64), minlength=n_labels)
        core_area = np.bincount(labels.ravel(), weights=core.ravel().astype(np.float64), minlength=n_labels)
        hole = enclosed & (core_area > 0) & (basin_area <= max_area)
        if not hole.any():
            continue
        region = hole[labels]
        fill[i] = region | (cv2.dilate(region.astype(np.uint8), kernel).astype(bool) & ~background.astype(bool))
        filled += int(hole.sum())
    if not filled:
        return a, 0
    mask = ops._block_upsample(torch.from_numpy(fill).unsqueeze(1), block, height, width)
    return torch.where(mask, torch.ones((), dtype=a.dtype), a), filled


def _busy_matte(rng, height, width, specks):
    import cv2

    m = np.zeros((height, width), np.float32)
    for _ in range(int(rng.integers(1, 5))):
        centre = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        axes = (int(rng.integers(6, max(8, width // 3))), int(rng.integers(6, max(8, height // 2))))
        cv2.ellipse(m, centre, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    for _ in range(int(rng.integers(0, 30))):
        centre = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        cv2.circle(m, centre, int(rng.integers(1, max(3, width // 12))), float(rng.choice([0.0, 0.0, 0.3, 0.7])), -1)
    for _ in range(specks):
        centre = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        cv2.circle(m, centre, int(rng.integers(0, 5)), float(rng.uniform(0.06, 1.0)), -1)
    return m


@pytest.mark.parametrize("size", [(90, 160), (270, 480), (1080, 1920)])
@pytest.mark.parametrize("specks", [0, 40, 1200])        # 1200: more regions than the per-region loop takes
def test_cleanup_matches_the_whole_map_reference(size, specks):
    rng = np.random.default_rng(size[0] * 7 + specks)
    for soft in (False, True):
        m = _busy_matte(rng, size[0], size[1], specks)
        if soft:
            import cv2

            m = cv2.GaussianBlur(m, (0, 0), 1.5)
        alpha = torch.from_numpy(m)[None, None]
        for area in (1.0, 40.0, 900.0, 1e9):
            got, count = ops.remove_small_components(alpha, area)
            want, want_count = _reference_remove(alpha, area)
            assert count == want_count and torch.equal(got, want)
            got, count = ops.fill_small_holes(alpha, area)
            want, want_count = _reference_fill(alpha, area)
            assert count == want_count and torch.equal(got, want)
