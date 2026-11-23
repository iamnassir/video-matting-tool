"""Mask images as people really supply them: pasted paths with quotes, grey masks saved as colour
JPEGs, transparent cut-outs, colour cut-outs on a uniform background (the still of a green-screen
result) and cut-outs that show another frame than the chosen reference frame.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import SubjectPrompt, parse_subject_spec
from app.core.errors import InvalidPathError, SubjectError
from app.tracking.subject import SubjectSelector, enclosed_openings, load_mask_image, read_mask_image
from app.utils.fs import clean_path_text
from tests.helpers import media

W, H, FRAMES = 320, 180, 36
GREEN = (25, 185, 51)            # R, G, B of a typical keyed background (not exactly #00FF00)


def _write(path: Path, rgb_or_grey: np.ndarray, quality: int = 90) -> Path:
    img = rgb_or_grey if rgb_or_grey.ndim == 2 or rgb_or_grey.shape[2] == 4 else cv2.cvtColor(rgb_or_grey, cv2.COLOR_RGB2BGR)
    if rgb_or_grey.ndim == 3 and rgb_or_grey.shape[2] == 4:
        img = cv2.cvtColor(rgb_or_grey, cv2.COLOR_RGBA2BGRA)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if path.suffix.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(path.suffix, img, params)
    assert ok
    buf.tofile(str(path))
    return path


def _ellipse(height: int = H, width: int = W) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    cv2.ellipse(mask, (width // 2, height // 2), (width // 4, height // 3), 0, 0, 360, 255, -1)
    return mask


def _photo(height: int = H, width: int = W, seed: int = 3) -> np.ndarray:
    """A picture with smooth, varied colours and no flat area (stands in for a photograph)."""

    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, (height // 12 + 2, width // 12 + 2, 3), dtype=np.uint8)
    return cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)


def _cutout(frame_rgb: np.ndarray, keep: np.ndarray, colour=GREEN) -> np.ndarray:
    out = np.empty_like(frame_rgb)
    out[:] = np.array(colour, np.uint8)
    out[keep > 0] = frame_rgb[keep > 0]
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0.5, b > 0.5
    return float((a & b).sum()) / max(1.0, float((a | b).sum()))


# --------------------------------------------------------------------------- pasted paths


@pytest.mark.parametrize("text, expected", [
    ('"C:\\Users\\me\\Downloads\\mask one.png"', "C:\\Users\\me\\Downloads\\mask one.png"),      # Explorer: Copy as path
    ("  'masks/first frame.png' \n", "masks/first frame.png"),
    ("file:///C:/Users/me/a%20mask.png", "C:/Users/me/a mask.png"),
    ("file://server/share/mask.png", "//server/share/mask.png"),
    ("file:///home/me/mask.png", "/home/me/mask.png"),
    ("plain.png", "plain.png"),
    ('"', '"'),
    ("", ""),
    (None, ""),
])
def test_clean_path_text(text, expected):
    assert clean_path_text(text) == expected


def test_subject_prompt_accepts_a_quoted_mask_path():
    assert SubjectPrompt(mode="mask", mask_path='"C:\\a b\\m.png"').mask_path == "C:\\a b\\m.png"
    assert parse_subject_spec('mask:"C:\\a b\\m.png"').mask_path == "C:\\a b\\m.png"
    with pytest.raises(ValueError):
        SubjectPrompt(mode="mask", mask_path='""')              # nothing left after tidying


def test_quoted_path_is_read_and_a_missing_file_is_named_without_quotes(tmp_path):
    path = _write(tmp_path / "grey mask.png", _ellipse())
    mask = load_mask_image(f'"{path}"', W, H)
    assert mask.shape == (H, W) and mask.dtype == np.float32
    assert _iou(mask, _ellipse()) > 0.99
    with pytest.raises(InvalidPathError) as err:
        load_mask_image(f'"{tmp_path / "nothing.png"}"', W, H)
    assert '"' not in err.value.message


# --------------------------------------------------------------------------- what counts as a mask


def test_grey_mask_saved_as_a_colour_jpeg_is_still_a_grey_mask(tmp_path):
    grey = _ellipse()
    path = _write(tmp_path / "mask.jpg", np.dstack([grey, grey, grey]), quality=70)
    image = read_mask_image(str(path), W, H)
    assert image.picture is None and image.notes == []
    assert _iou(image.mask, grey) > 0.98


def test_transparent_cutout_uses_its_alpha(tmp_path):
    rgba = np.dstack([_photo(), _ellipse()])
    image = read_mask_image(str(_write(tmp_path / "cutout.png", rgba)), W, H)
    assert image.picture is None
    assert _iou(image.mask, _ellipse()) > 0.99


def test_fully_opaque_alpha_is_ignored(tmp_path):
    grey = _ellipse()
    rgba = np.dstack([grey, grey, grey, np.full_like(grey, 255)])
    mask = load_mask_image(str(_write(tmp_path / "opaque.png", rgba)), W, H)
    assert _iou(mask, grey) > 0.99                      # not "everything", which the opaque alpha would give


@pytest.mark.parametrize("colour", [GREEN, (0, 255, 0), (20, 40, 235), (230, 30, 220)])
@pytest.mark.parametrize("ext", [".png", ".jpg"])
def test_colour_cutout_on_a_uniform_background_becomes_a_mask(tmp_path, colour, ext):
    grey = cv2.cvtColor(_photo(seed=5), cv2.COLOR_RGB2GRAY)
    subject = np.dstack([grey, grey, np.clip(grey.astype(np.int16) + 18, 0, 255).astype(np.uint8)])   # a greyish subject
    path = _write(tmp_path / f"cutout{ext}", _cutout(subject, _ellipse(), colour))
    image = read_mask_image(str(path), W, H)
    assert _iou(image.mask, _ellipse()) > 0.96
    assert image.picture is not None and image.picture.shape == (H, W, 3)
    assert len(image.notes) == 1 and "uniform #" in image.notes[0]


def test_cutout_is_resized_to_the_frame_and_a_different_shape_is_reported(tmp_path):
    big = cv2.resize(_cutout(_photo(seed=5) // 3 + 90, _ellipse()), (W * 2, H * 2), interpolation=cv2.INTER_NEAREST)
    same_shape = read_mask_image(str(_write(tmp_path / "big.png", big)), W, H)
    assert same_shape.mask.shape == (H, W) and same_shape.picture.shape == (H, W, 3)
    assert not any("different shape" in n for n in same_shape.notes)
    square = read_mask_image(str(_write(tmp_path / "square.png", big[:, : H * 2])), W, H)
    assert any("different shape" in n for n in square.notes)


def test_a_photograph_is_not_a_mask(tmp_path):
    with pytest.raises(SubjectError) as err:
        read_mask_image(str(_write(tmp_path / "holiday.jpg", _photo())), W, H)
    assert "colour picture, not a mask" in err.value.message
    assert "black and white" in (err.value.hint or "")


def test_a_photograph_with_a_natural_green_area_is_not_a_cutout(tmp_path):
    photo = _photo(seed=9)
    lawn = np.zeros((H, W, 3), np.float32)
    lawn[..., 1] = np.linspace(90, 210, W)[None, :]                 # green, but shaded: not one flat colour
    lawn[..., 0] = 30
    lawn[..., 2] = 40
    photo[H // 2:] = lawn[H // 2:].astype(np.uint8)
    with pytest.raises(SubjectError):
        read_mask_image(str(_write(tmp_path / "garden.jpg", photo)), W, H)


# --------------------------------------------------------------------------- selector behaviour


class _Frames:
    """The reader interface the selector uses, over frames held in memory."""

    def __init__(self, frames):
        self.frames = frames
        self.frame_count = len(frames)
        self.calls = []

    def frame_rgb(self, index, max_side=None):
        self.calls.append(index)
        return self.frames[index]


@pytest.fixture()
def selector(app_settings):
    s = SubjectSelector(app_settings, "cpu")
    yield s
    s.close()


@pytest.fixture(scope="module")
def frames():
    return [media.synth_frame(i, W, H, FRAMES) for i in range(FRAMES)]


def _subject_box(index: int, margin: float = 1.6) -> np.ndarray:
    """A box around the moving subject of the synthetic clip in frame *index*."""

    cx, cy = media.subject_center(index, W, H, FRAMES)
    r = media.subject_radius(W, H) * margin
    keep = np.zeros((H, W), np.uint8)
    keep[max(0, int(cy - r)): int(cy + r), max(0, int(cx - r)): int(cx + r)] = 255
    return keep


BLUE = (20, 40, 235)


def test_full_frame_and_empty_masks_are_refused(tmp_path, selector, frames):
    white = _write(tmp_path / "white.png", np.full((H, W), 255, np.uint8))
    with pytest.raises(SubjectError) as err:
        selector.resolve_on_frame(SubjectPrompt(mode="mask", mask_path=str(white)), frames[0])
    assert "whole frame" in err.value.message
    black = _write(tmp_path / "black.png", np.zeros((H, W), np.uint8))
    with pytest.raises(SubjectError):
        selector.resolve_on_frame(SubjectPrompt(mode="mask", mask_path=str(black)), frames[0])


def _ring() -> np.ndarray:
    ring = np.zeros((H, W), np.uint8)
    cv2.rectangle(ring, (40, 30), (280, 150), 255, -1)
    cv2.rectangle(ring, (110, 60), (210, 120), 0, -1)                # a window
    return ring


def test_enclosed_openings():
    assert enclosed_openings(_ellipse() / 255.0) == 0.0
    assert enclosed_openings(np.zeros((H, W), np.float32)) == 0.0
    share = enclosed_openings(_ring() / 255.0)
    assert share == pytest.approx((101 * 61) / (241 * 121), abs=0.01)
    notch = _ring()
    notch[60:121, 200:281] = 0                                       # the window now opens to the outside
    assert enclosed_openings(notch / 255.0) < 0.01


def test_mask_with_openings_and_a_whole_object_model(tmp_path, selector, frames):
    """A segmentation model closes openings again: the selection says so and names the alternative."""

    from app.models.base import ModelCapabilities, ModelInfo
    from app.models.registry import get_model_info

    ring = _write(tmp_path / "ring.png", _ring())
    solid = _write(tmp_path / "solid.png", _ellipse())
    sam2 = get_model_info("sam2", selector.settings)
    assert sam2.capabilities.alpha_type == "mask"
    subject = selector.resolve_on_frame(SubjectPrompt(mode="mask", mask_path=str(ring)), frames[0], sam2)
    assert any("see-through openings" in n and "MatAnyone" in n for n in subject.notes)
    subject = selector.resolve_on_frame(SubjectPrompt(mode="mask", mask_path=str(solid)), frames[0], sam2)
    assert not any("openings" in n for n in subject.notes)
    matting = ModelInfo(
        id="any_matting", name="Any matting model", description="", license=sam2.license,
        capabilities=ModelCapabilities(alpha_type="alpha"), variants=sam2.variants, default_variant=sam2.default_variant,
    )
    subject = selector.resolve_on_frame(SubjectPrompt(mode="mask", mask_path=str(ring)), frames[0], matting)
    assert not any("openings" in n for n in subject.notes)


def test_cutout_of_the_reference_frame_is_used_as_it_is(tmp_path, selector, frames):
    keep = _subject_box(0)
    path = _write(tmp_path / "frame0.jpg", _cutout(frames[0], keep, BLUE))
    reader = _Frames(frames)
    prompt = SubjectPrompt(mode="mask", mask_path=f'"{path}"')
    aligned, notes = selector.align_mask_frame(prompt, reader, FRAMES)
    assert aligned.frame_index == 0 and notes == []
    assert reader.calls == [0]                                      # no search when the picture fits
    subject = selector.resolve(prompt, reader, None)
    assert subject.frame_index == 0 and subject.source == "user:mask"
    assert _iou(subject.mask, keep) > 0.95
    assert not any("line up" in n for n in subject.notes)


@pytest.mark.parametrize("shown, chosen", [(20, 0), (4, 30), (35, 0)])
def test_cutout_of_another_frame_moves_the_reference_frame(tmp_path, selector, frames, shown, chosen):
    keep = _subject_box(shown)
    path = _write(tmp_path / "later.jpg", _cutout(frames[shown], keep, BLUE))
    reader = _Frames(frames)
    prompt = SubjectPrompt(mode="mask", mask_path=str(path), frame_index=chosen)
    subject = selector.resolve(prompt, reader, None)
    assert subject.frame_index == shown
    assert _iou(subject.mask, keep) > 0.95
    assert any(f"shows frame {shown}" in n and f"not frame {chosen}" in n for n in subject.notes)
    assert not any("line up" in n for n in subject.notes)


def test_black_and_white_mask_is_never_moved(tmp_path, selector, frames):
    path = _write(tmp_path / "bw.png", _subject_box(20))
    reader = _Frames(frames)
    subject = selector.resolve(SubjectPrompt(mode="mask", mask_path=str(path)), reader, None)
    assert subject.frame_index == 0 and subject.notes == []
    assert set(reader.calls) == {0}


def test_cutout_that_matches_no_frame_is_reported(tmp_path, selector, frames):
    other = _photo(seed=11)
    path = _write(tmp_path / "other.png", _cutout(other, _subject_box(10), BLUE))
    reader = _Frames(frames)
    subject = selector.resolve(SubjectPrompt(mode="mask", mask_path=str(path)), reader, None)
    assert subject.frame_index == 0
    assert any("does not line up with frame 0" in n for n in subject.notes)
    assert len(set(reader.calls)) == FRAMES                         # every frame was looked at


def test_search_stops_at_its_time_budget(tmp_path, selector, frames):
    other = _photo(seed=11)
    path = _write(tmp_path / "other.png", _cutout(other, _subject_box(10), BLUE))
    reader = _Frames(frames)
    prompt = SubjectPrompt(mode="mask", mask_path=str(path))
    aligned, notes = selector.align_mask_frame(prompt, reader, FRAMES, budget_seconds=0.0)
    assert aligned.frame_index == 0 and notes == []
    assert len(reader.calls) <= 2


def test_border_line_and_specks_of_a_cutout_are_not_subject(tmp_path):
    """Pictures from image tools often carry a dark line along a border: it is not the subject."""

    frame = _photo()
    picture = _cutout(frame, _ellipse())
    clean = read_mask_image(str(_write(tmp_path / "clean.png", picture)), W, H).mask
    picture[:2, :] = (20, 20, 20)                      # a dark line along the top border
    picture[30:32, 300:303] = (200, 30, 30)            # a stray speck
    path = _write(tmp_path / "lined.png", picture)
    mask = read_mask_image(str(path), W, H).mask
    assert float(mask[:4, :].max()) == 0.0
    assert float(mask[26:36, 296:307].max()) == 0.0
    assert _iou(mask, clean) > 0.995                   # the subject itself is untouched
    # A subject that really touches the border stays.
    touching = _ellipse()
    touching[:60, 100:220] = 255
    path = _write(tmp_path / "touching.png", _cutout(frame, touching))
    kept = read_mask_image(str(path), W, H).mask
    assert float(kept[:3, 110:210].mean()) > 0.8 and _iou(kept, touching) > 0.85


def test_mask_with_openings_and_a_model_for_people(tmp_path, selector, frames):
    """MatAnyone keeps openings but was made for people: the note names the model for objects."""

    from app.models.registry import get_model_info

    ring = _write(tmp_path / "ring.png", _ring())
    subject = selector.resolve_on_frame(SubjectPrompt(mode="mask", mask_path=str(ring)), frames[0],
                                        get_model_info("matanyone", selector.settings))
    assert any("see-through openings" in n and "sam2_matanyone" in n for n in subject.notes)
    subject = selector.resolve_on_frame(SubjectPrompt(mode="mask", mask_path=str(ring)), frames[0],
                                        get_model_info("sam2_matanyone", selector.settings))
    assert not any("openings" in n for n in subject.notes)        # the model made for this says nothing
