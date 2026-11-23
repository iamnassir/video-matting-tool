"""Subject selection: turn a user prompt into a concrete subject on a reference frame.

``SubjectPrompt`` (configuration) describes *how* the user wants the subject chosen: automatically,
by clicks, by a box, by a mask image or by a text description.  :class:`SubjectSelector` resolves it
against the actual video into a :class:`~app.core.types.ResolvedSubject` in display orientation.

Resolution strategy, by capability of the matting model that will consume the subject:

* models with native prompts (SAM 2 family) receive boxes / clicks / masks directly;
* models that need a first-frame mask (MatAnyone) get one from a *mask backend*;
* models without any prompt support (RVM) use the subject only to seed the pipeline's
  connected-component tracker, for which a box, clicks or a mask are all sufficient.

Backends are optional and discovered lazily (see :meth:`SubjectSelector.backends`), so the selector
degrades with a clear message instead of failing when an optional model is not installed:

=============  ==========================================  =======================================
prompt         preferred backend                           fallbacks
=============  ==========================================  =======================================
auto           primary-subject detector                    the matting model's own first matte
box / points   SAM 2 image predictor                       detector instance matching the prompt,
                                                           GrabCut (box only)
text           open-vocabulary detector -> box             (none: clear error)
mask           image file                                  -
=============  ==========================================  =======================================
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..core.config import AppSettings, SubjectPrompt
from ..core.errors import (
    ConfigError,
    InvalidPathError,
    PromptNotSupportedError,
    SubjectError,
    SubjectNotFoundError,
    VmtError,
)
from ..core.events import CancelToken
from ..core.types import ResolvedSubject, box_from_mask, clamp_box
from ..models.base import ModelInfo
from ..utils.fs import clean_path_text

log = logging.getLogger(__name__)

#: A mask backend: ``fn(frame_rgb_uint8, prompt_in_pixels) -> ResolvedSubject | None``.
MaskBackend = Callable[[np.ndarray, SubjectPrompt], Optional[ResolvedSubject]]


#: A picture is a grey-scale mask when fewer than this share of its pixels are visibly coloured.
_MASK_COLOURED_SHARE = 0.005
#: R, G, B may differ by this much (8-bit scale) in a "grey" pixel: JPEG noise stays far below it.
_MASK_GREY_TOLERANCE = 24
#: A uniform background colour has to cover at least this share of a colour picture.
_KEY_MIN_SHARE = 0.03
_NOT_A_MASK_HINT = (
    "A mask image shows the reference frame in black and white (white = subject, black = background). "
    "A cut-out with a transparent background, or on one uniform background colour such as chroma green, "
    "is accepted too. To point at the subject instead, use Click or Box."
)


#: A cut-out lines up with a frame when its subject pixels differ by less than this on average
#: (8-bit scale).  Measured: 8 for a JPEG cut-out of the right frame, 35-50 for other frames of a
#: moving shot.
_CUTOUT_MATCH = 16.0


@dataclass
class MaskImage:
    """A mask image as read for one frame size."""

    mask: np.ndarray                                    # float32 [H, W] in 0..1
    notes: List[str] = field(default_factory=list)
    #: The picture itself (RGB uint8 [H, W, 3]) when the mask came from a colour cut-out.  Its
    #: subject pixels are pixels of the video, which makes it possible to check which frame it shows.
    picture: Optional[np.ndarray] = None


class _CutoutMatcher:
    """Compares video frames with a colour cut-out, inside the cut-out's subject, on small copies."""

    def __init__(self, image: MaskImage, side: int = 384) -> None:
        assert image.picture is not None
        height, width = image.mask.shape
        scale = min(1.0, side / float(max(height, width)))
        self.side = side
        self.size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        self.picture = cv2.resize(image.picture, self.size, interpolation=cv2.INTER_AREA).astype(np.int16)
        core = (cv2.resize(image.mask, self.size, interpolation=cv2.INTER_AREA) > 0.9).astype(np.uint8)
        self.core = cv2.erode(core, np.ones((3, 3), np.uint8)) > 0      # edge pixels are mixed with the background colour
        self.usable = int(self.core.sum()) >= 64

    def difference(self, frame_rgb: np.ndarray) -> float:
        if (frame_rgb.shape[1], frame_rgb.shape[0]) != self.size:
            frame_rgb = cv2.resize(frame_rgb, self.size, interpolation=cv2.INTER_AREA)
        return float(np.abs(frame_rgb.astype(np.int16) - self.picture)[self.core].mean())


def enclosed_openings(mask: np.ndarray) -> float:
    """Share of the area inside a mask's outline that is open (background enclosed by the subject).

    Car windows, the gap between an arm and the body, the inside of a handle: 0.0 for a solid shape.
    """

    binary = (np.asarray(mask) > 0.5).astype(np.uint8)
    height, width = binary.shape
    scale = min(1.0, 512.0 / max(height, width))
    if scale < 1.0:
        binary = cv2.resize(binary, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                            interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    inside = int(filled.sum())
    return float(int((filled & (1 - binary)).sum())) / inside if inside else 0.0


def _unit_scale(img: np.ndarray) -> float:
    return float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 1.0


def _to_8bit(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(img.astype(np.float32) * (255.0 / _unit_scale(img)), 0, 255).astype(np.uint8)


def _drop_hairlines(mask: np.ndarray) -> np.ndarray:
    """Remove parts of a cut-out mask that are only a few pixels thick and stand alone.

    Pictures made by image tools often carry a dark line along one border or a stray speck; such
    a part is not the background colour, so it would count as subject.  A part is removed when
    nothing of it survives an erosion by three pixels (the frame border counts as background).
    """

    binary = (mask > 0.5).astype(np.uint8)
    core = cv2.erode(binary, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    if count <= 1:
        return mask
    keep = np.zeros(count, bool)
    keep[np.unique(labels[core > 0])] = True
    thin = (binary > 0) & ~keep[labels]
    if not thin.any():
        return mask
    out = mask.copy()
    out[cv2.dilate(thin.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = 0.0   # with its soft edge
    return out


def _mask_from_cutout(bgr: np.ndarray, name: str) -> Tuple[np.ndarray, str]:
    """Mask of a colour cut-out: everything that is not the uniform background colour.

    The background colour is the most frequent saturated colour.  It is accepted only if it is
    flat (one colour, not a photographed lawn or sky) and covers a real share of the picture;
    otherwise the picture is an ordinary photograph and cannot serve as a mask.
    """

    h, w = bgr.shape[:2]
    scale = min(1.0, 512.0 / max(h, w))
    small = cv2.resize(bgr, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    luma, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    saturated = np.hypot(cr - 128.0, cb - 128.0) >= 40.0
    not_a_mask = SubjectError(f"{name} is a colour picture, not a mask.", hint=_NOT_A_MASK_HINT)
    if float(saturated.mean()) < _KEY_MIN_SHARE:
        raise not_a_mask
    hist, _, _ = np.histogram2d(cr[saturated], cb[saturated], bins=64, range=[[0, 256], [0, 256]])
    i, j = np.unravel_index(int(hist.argmax()), hist.shape)
    near = np.hypot(cr - (i * 4 + 2), cb - (j * 4 + 2)) <= 10.0
    key_cr, key_cb = float(np.median(cr[near])), float(np.median(cb[near]))
    strength = float(np.hypot(key_cr - 128.0, key_cb - 128.0))
    distance = np.hypot(cr - key_cr, cb - key_cb)
    wide = distance <= 0.3 * strength
    share = float(wide.mean())
    tight = float((distance <= 8.0).mean())
    flat = share > 0 and tight >= 0.85 * share and float(luma[wide].std()) <= 14.0
    if share < _KEY_MIN_SHARE or not flat:
        raise not_a_mask
    key_bgr = np.median(small[wide].reshape(-1, 3), axis=0)
    # Full-resolution mask: 0 at the background colour, 1 from a little more than half way to grey.
    full = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    dist = np.hypot(full[..., 1] - key_cr, full[..., 2] - key_cb)
    low, high = 0.3 * strength, 0.55 * strength
    mask = np.clip((dist - low) / max(high - low, 1e-3), 0.0, 1.0)
    mask = cv2.medianBlur((mask * 255.0 + 0.5).astype(np.uint8), 5).astype(np.float32) / 255.0   # compression specks
    mask = _drop_hairlines(mask)
    colour = "#{:02X}{:02X}{:02X}".format(int(round(key_bgr[2])), int(round(key_bgr[1])), int(round(key_bgr[0])))
    note = (
        f"The mask image is a colour picture on a uniform {colour} background, so everything that is not "
        f"that colour was taken as the subject ({100.0 * float((mask > 0.5).mean()):.0f} % of the picture)."
    )
    return mask, note


def read_mask_image(path: str, width: int, height: int) -> MaskImage:
    """Read a mask image for a ``width x height`` display frame.

    Returns the mask as float32 ``[height, width]`` in 0..1 with notes about how the picture was
    understood.  Accepted, in this order:

    * a picture with transparency: the alpha channel is the mask;
    * a grey-scale picture (any bit depth, also stored as RGB): white = subject;
    * a colour cut-out on one uniform background colour (for example the still of a green-screen
      result): everything that is not the background colour is the subject.

    Any other colour picture is rejected, because reading a photograph as "brightness = subject"
    would silently select most of the frame.  The mask is resized to the frame when its size differs
    (a mask drawn on a proxy or at another resolution).
    """

    p = Path(clean_path_text(path))
    if not p.is_file():
        raise InvalidPathError(f"Subject mask file not found: {p}", hint="Enter the full path of an image file on this computer.")
    data = np.fromfile(str(p), dtype=np.uint8)      # cv2.imread cannot open non-ASCII Windows paths
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    if img is None:
        raise SubjectError(f"Could not read the subject mask image {p}.", hint="Use a PNG, JPEG or TIFF image.")
    notes: List[str] = []
    picture = None
    alpha = None
    if img.ndim == 3 and img.shape[2] in (2, 4):
        alpha = img[:, :, -1]
        img = img[:, :, 0] if img.shape[2] == 2 else img[:, :, :3]
        if int(alpha.min()) == int(alpha.max()):
            alpha = None                                # fully opaque (or empty): the picture itself is the mask
    if alpha is not None:
        mask = alpha.astype(np.float32) / _unit_scale(alpha)
    elif img.ndim == 2:
        mask = img.astype(np.float32) / _unit_scale(img)
    else:
        bgr = _to_8bit(img)
        spread = bgr.max(axis=2).astype(np.int16) - bgr.min(axis=2)
        if float((spread > _MASK_GREY_TOLERANCE).mean()) < _MASK_COLOURED_SHARE:
            mask = img.max(axis=2).astype(np.float32) / _unit_scale(img)
        else:
            mask, note = _mask_from_cutout(bgr, p.name)
            notes.append(note)
            picture = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if mask.shape != (height, width):
        mh, mw = mask.shape
        if abs((mw / mh) / (width / height) - 1.0) > 0.02:
            notes.append(
                f"The mask image ({mw}x{mh}) has a different shape than the video frame ({width}x{height}); "
                "it was stretched to fit and may not line up with the picture."
            )
        log.info("Subject mask %s is %dx%d; resizing to the frame size %dx%d.", p.name, mw, mh, width, height)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        if picture is not None:
            picture = cv2.resize(picture, (width, height), interpolation=cv2.INTER_AREA)
    return MaskImage(mask=np.clip(mask, 0.0, 1.0), notes=notes, picture=picture)


def load_mask_image(path: str, width: int, height: int) -> np.ndarray:
    """The mask of :func:`read_mask_image` without the notes."""

    return read_mask_image(path, width, height).mask


def grabcut_mask(frame_rgb: np.ndarray, box: Tuple[float, float, float, float], iterations: int = 4) -> Optional[np.ndarray]:
    """Classical GrabCut segmentation inside a box. A last-resort mask source that needs no model."""

    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)
    if (x2 - x1) < 4 or (y2 - y1) < 4:
        return None
    scale = min(1.0, 640.0 / max(h, w))     # GrabCut is slow at full HD; work on a reduced copy
    small = cv2.resize(frame_rgb, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    rect = (
        int(round(x1 * scale)), int(round(y1 * scale)),
        max(2, int(round((x2 - x1) * scale))), max(2, int(round((y2 - y1) * scale))),
    )
    mask = np.zeros(small.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), mask, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error as exc:
        log.debug("GrabCut failed: %s", exc)
        return None
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.float32)
    if fg.sum() < 16:
        return None
    return cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR)


class SubjectSelector:
    """Resolves subject prompts. Owns optional helper models and frees them in :meth:`close`."""

    def __init__(
        self,
        settings: AppSettings,
        device: Any,
        *,
        commercial_use: Optional[bool] = None,
        download: Optional[bool] = None,
        cancel: Optional[CancelToken] = None,
        warnings: Optional[List[str]] = None,
    ) -> None:
        self.settings = settings
        self.device = device
        self.commercial_use = commercial_use
        self.download = download
        self.cancel = cancel
        self.warnings = warnings if warnings is not None else []
        self._backends: Dict[str, Any] = {}

    # ------------------------------------------------------------------ backends (optional)
    def backends(self) -> Dict[str, bool]:
        """Which optional backends can be used right now (package installed and weights present)."""

        out = {"sam2": False, "detector": False, "text": False, "grabcut": True}
        try:
            from . import backends as _b  # provided by the optional subject-backends module

            out.update(_b.available(self.settings))
        except ImportError:
            pass
        except Exception as exc:  # a broken optional backend must not break selection
            log.debug("Subject backends unavailable: %s", exc)
        return out

    def _backend(self, name: str) -> Optional[Any]:
        if name in self._backends:
            return self._backends[name]
        backend = None
        try:
            from . import backends as _b

            backend = _b.create(
                name, self.settings, self.device, commercial_use=self.commercial_use,
                download=self.download, cancel=self.cancel, warnings=self.warnings,
            )
        except ImportError:
            backend = None
        except VmtError as exc:
            log.info("Subject backend '%s' is not usable: %s", name, exc.message)
            self.warnings.append(f"Subject backend '{name}' is not usable: {exc.message}")
            backend = None
        self._backends[name] = backend
        return backend

    def close(self) -> None:
        """Release helper models (frees VRAM before the matting model is loaded)."""

        for backend in self._backends.values():
            try:
                if backend is not None and hasattr(backend, "close"):
                    backend.close()
            except Exception as exc:  # pragma: no cover - best effort
                log.debug("closing subject backend failed: %s", exc)
        self._backends.clear()

    # ------------------------------------------------------------------ mask pictures
    def align_mask_frame(
        self, prompt: SubjectPrompt, reader: Any, total: int, *, budget_seconds: float = 45.0
    ) -> Tuple[SubjectPrompt, List[str]]:
        """Make sure a colour cut-out used as the mask belongs to the reference frame.

        A cut-out (the subject on a uniform background colour) carries the video's own pixels, so it
        can be compared with the frames.  When it does not show the chosen reference frame, the video
        is searched for the frame it does show and that frame becomes the reference frame; the note
        returned with the new prompt says so.  Black-and-white masks carry no picture and are taken
        as given.  ``reader`` needs ``frame_rgb(index, max_side=...)``.

        The search reads the video once, front to back from the reference frame, for at most
        ``budget_seconds``; when it finds nothing the prompt is returned unchanged (the mismatch is
        then reported by :meth:`resolve_on_frame`).
        """

        if prompt.mode != "mask" or total <= 1:
            return prompt, []
        first = reader.frame_rgb(prompt.frame_index)
        height, width = first.shape[:2]
        image = read_mask_image(prompt.mask_path or "", width, height)
        if image.picture is None:
            return prompt, []
        matcher = _CutoutMatcher(image)
        if not matcher.usable or matcher.difference(first) <= _CUTOUT_MATCH:
            return prompt, []
        del first
        started = time.monotonic()
        best_index, best = -1, float("inf")
        checked = 0
        worse_since_match = 0
        order = list(range(prompt.frame_index + 1, total)) + list(range(0, prompt.frame_index))
        for index in order:
            if self.cancel is not None:
                self.cancel.raise_if_cancelled()
            if time.monotonic() - started >= budget_seconds:
                break
            if index == 0:
                worse_since_match = 0
            difference = matcher.difference(reader.frame_rgb(index, max_side=matcher.side))
            checked += 1
            if difference < best:
                best_index, best = index, difference
                worse_since_match = 0
            elif best <= _CUTOUT_MATCH:
                worse_since_match += 1
                if worse_since_match >= 12:                 # the best frame is behind us
                    break
        if best_index >= 0 and best <= _CUTOUT_MATCH:
            log.info("Mask picture matches frame %d (difference %.1f); reference frame moved from %d.",
                     best_index, best, prompt.frame_index)
            note = (
                f"The mask picture shows frame {best_index} of the video, not frame {prompt.frame_index}, "
                f"so frame {best_index} is used as the reference frame."
            )
            return prompt.model_copy(update={"frame_index": best_index}), [note]
        log.info("Mask picture matches no frame (%d checked, best difference %.1f).", checked, best)
        return prompt, []

    # ------------------------------------------------------------------ resolution
    def resolve_on_frame(
        self,
        prompt: SubjectPrompt,
        frame_rgb: np.ndarray,
        model_info: Optional[ModelInfo] = None,
        *,
        want_mask: bool = False,
    ) -> ResolvedSubject:
        """Resolve *prompt* on one display-oriented RGB frame (``uint8 [H, W, 3]``).

        ``want_mask`` asks for a mask even if the matting model would not need one (used by the
        interactive preview in the UI).  Raises :class:`SubjectNotFoundError` when nothing matches.
        """

        height, width = frame_rgb.shape[:2]
        prompt = prompt.to_pixels(width, height)
        caps = model_info.capabilities if model_info is not None else None
        needs_mask = want_mask or bool(caps and caps.needs_initial_mask)
        native = set(caps.native_prompts) if caps else set()

        if prompt.mode == "mask":
            image = read_mask_image(prompt.mask_path or "", width, height)
            mask, mask_notes = image.mask, list(image.notes)
            if image.picture is not None:
                matcher = _CutoutMatcher(image)
                if matcher.usable and matcher.difference(frame_rgb) > _CUTOUT_MATCH:
                    mask_notes.append(
                        f"The mask picture does not line up with frame {prompt.frame_index}: its subject looks "
                        "different from the video there. A mask has to show the reference frame, so move the "
                        "reference frame to the frame the picture was made from."
                    )
            if caps is not None and model_info is not None and model_info.id != "sam2_matanyone":
                openings = enclosed_openings(mask)
                if openings >= 0.03 and caps.alpha_type == "mask":
                    mask_notes.append(
                        f"The mask has see-through openings ({100.0 * openings:.0f} % of the area inside its outline). "
                        f"{model_info.name} follows an object as a whole and closes such openings again in the "
                        "frames after the reference frame. The 'SAM 2 + MatAnyone' model (sam2_matanyone) keeps them."
                    )
                elif openings >= 0.03 and caps.subject_domain == "human":
                    mask_notes.append(
                        f"The mask has see-through openings ({100.0 * openings:.0f} % of the area inside its outline). "
                        f"{model_info.name} was made for people. For an object with openings - a car, a piece of "
                        "furniture - the 'SAM 2 + MatAnyone' model (sam2_matanyone) holds the outline steady and "
                        "keeps out what passes behind."
                    )
            covered = float((mask > 0.5).mean())
            if covered <= 0.0:
                raise SubjectNotFoundError("The supplied subject mask is empty.", hint=_NOT_A_MASK_HINT)
            if covered >= 0.995:
                raise SubjectError(
                    "The supplied subject mask covers the whole frame, so there is no background to remove.",
                    hint=_NOT_A_MASK_HINT,
                )
            return ResolvedSubject(
                frame_index=prompt.frame_index, mask=mask, box=box_from_mask(mask), source="user:mask", score=1.0,
                notes=mask_notes,
            )

        if prompt.mode == "text":
            grounder = self._backend("text")
            if grounder is None:
                raise PromptNotSupportedError(
                    "Text-based subject selection needs the optional open-vocabulary detector, which is not installed.",
                    hint="Install it (see docs/MODELS.md, 'Text prompts') or select the subject with a box or clicks.",
                )
            found = grounder.detect(frame_rgb, prompt.text or "")
            if found is None:
                raise SubjectNotFoundError(f"Nothing matching the description {prompt.text!r} was found in frame {prompt.frame_index}.")
            box, score = found
            prompt = prompt.model_copy(update={"mode": "box", "box": box})
            subject = self.resolve_on_frame(prompt, frame_rgb, model_info, want_mask=want_mask)
            subject.source = f"text:{getattr(grounder, 'name', 'grounding')}+{subject.source}"
            subject.text = prompt.text
            subject.score = score if subject.score is None else subject.score
            return subject

        if prompt.mode in ("box", "points"):
            box = clamp_box(prompt.box, width, height) if prompt.box is not None else None
            points = [(float(p.x), float(p.y), int(p.label)) for p in prompt.points]
            for x, y, _ in points:
                if not (0.0 <= x <= width and 0.0 <= y <= height):
                    raise ConfigError(
                        f"Subject point ({x:.0f}, {y:.0f}) lies outside the {width}x{height} frame.",
                        hint="Coordinates refer to the frame as displayed (after rotation).",
                    )
            subject = ResolvedSubject(
                frame_index=prompt.frame_index, box=box, points=points, source=f"user:{prompt.mode}"
            )
            kinds_ok = (prompt.mode in native) or (prompt.mode == "points" and "points" in native) or (
                prompt.mode == "box" and "box" in native
            )
            # Clicks without a box are ambiguous (the shirt or the person?).  When SAM 2 is available
            # they are resolved to the whole subject first, so that the matting model starts from an
            # unambiguous box instead of guessing on its own.
            clicks_only = prompt.mode == "points" and box is None
            if kinds_ok and not needs_mask and not clicks_only:
                return subject
            mask_subject = self._mask_from_prompt(frame_rgb, prompt)
            if mask_subject is not None:
                mask_subject.frame_index = prompt.frame_index
                mask_subject.box = mask_subject.box or box
                mask_subject.points = points
                return mask_subject
            if needs_mask and not want_mask:
                raise PromptNotSupportedError(
                    "This model needs a mask of the subject, and no backend is available to create one from "
                    f"a {prompt.mode} prompt.",
                    hint="Install the SAM 2 package and weights (python main.py models download sam2), "
                    "or provide a mask image with --subject mask:<file>.",
                )
            return subject

        # ---- automatic ------------------------------------------------------------------
        detector = self._backend("detector")
        if detector is not None:
            found = detector.primary_subject(frame_rgb)
            if found is not None:
                found.frame_index = prompt.frame_index
                if needs_mask:
                    # The detector's mask is coarse: let SAM 2 draw the precise one from its box.
                    # (Models with native box prompts do that themselves on the reference frame.)
                    refined = self._refine_with_sam2(frame_rgb, found)
                    if refined is not None:
                        refined.frame_index = prompt.frame_index
                        refined.label = refined.label or found.label
                        refined.notes.extend(n for n in found.notes if n not in refined.notes)
                        return refined
                return found
            if needs_mask or (caps is not None and caps.tracks_single_subject):
                raise SubjectNotFoundError(f"No subject was detected in frame {prompt.frame_index}.")
        elif needs_mask or (caps is not None and caps.tracks_single_subject):
            raise PromptNotSupportedError(
                "Automatic subject detection is not available for this model: the subject detector is not installed.",
                hint="Download it with 'python main.py models download detector', or select the subject with "
                "--subject box:x1,y1,x2,y2 / points:x,y.",
            )
        # Models that matte without a prompt (RVM): the pipeline's component tracker picks the
        # primary subject from the model's own first matte.
        return ResolvedSubject(frame_index=prompt.frame_index, source="auto:model")

    def _refine_with_sam2(self, frame_rgb: np.ndarray, subject: ResolvedSubject) -> Optional[ResolvedSubject]:
        sam = self._backend("sam2")
        if sam is None or subject.box is None:
            return None
        try:
            out = sam.segment(frame_rgb, box=subject.box, points=[])
        except VmtError as exc:
            log.info("SAM 2 refinement of the detected subject failed: %s", exc.message)
            return None
        if out is None:
            return None
        out.source = f"{subject.source}+{out.source}"
        out.box = out.box or subject.box
        return out

    def _whole_subject_for_clicks(
        self, frame_rgb: np.ndarray, clicked: ResolvedSubject, points: List[Tuple[float, float, int]]
    ) -> Optional[ResolvedSubject]:
        """Clicks that landed on a *part* of a detected object select the whole object.

        A click on a waistcoat makes SAM 2 return the waistcoat; in a matting tool it means "this
        person".  When the subject detector is installed and one of its instances honours every
        click (contains the positive ones, none of the negative ones) while the clicked mask is a
        clearly smaller region inside that instance, SAM 2 is asked again with the instance's box
        plus the clicks.  Returns None when the rule does not apply; the clicked mask is used then.
        """

        if clicked.mask is None or not self.backends().get("detector"):
            return None
        detector, sam = self._backend("detector"), self._backend("sam2")
        if detector is None or sam is None:
            return None
        try:
            instance = detector.match_prompt(frame_rgb, box=None, points=points)
        except VmtError as exc:
            log.info("Subject detection for the clicked point failed: %s", exc.message)
            return None
        if instance is None or instance.mask is None or instance.box is None:
            return None
        part, whole = clicked.mask > 0.5, instance.mask > 0.5
        part_area, whole_area = float(part.sum()), float(whole.sum())
        if part_area <= 0.0 or whole_area <= 0.0:
            return None
        if float((part & whole).sum()) / part_area < 0.8 or part_area >= 0.6 * whole_area:
            return None                     # already the whole object, or something else entirely
        try:
            refined = sam.segment(frame_rgb, box=instance.box, points=points)
        except VmtError as exc:
            log.info("SAM 2 could not segment the detected %s: %s", instance.label, exc.message)
            return None
        if refined is None or refined.mask is None:
            return None
        height, width = refined.mask.shape[:2]
        for x, y, label in points:
            xi, yi = min(width - 1, max(0, int(round(x)))), min(height - 1, max(0, int(round(y))))
            if bool(refined.mask[yi, xi] > 0.5) != bool(label):
                return None                 # the larger mask contradicts a click: keep what was clicked
        label = instance.label or "object"
        refined.source = "sam2:points+maskrcnn"
        refined.label = instance.label
        refined.notes.append(
            f"The click landed on a part of a detected {label} ({100.0 * part_area / whole_area:.0f} % of it), so the "
            f"whole {label} was selected. Draw a box or supply a mask image to select only a part."
        )
        return refined

    def _mask_from_prompt(self, frame_rgb: np.ndarray, prompt: SubjectPrompt) -> Optional[ResolvedSubject]:
        """Create a mask for a box / points prompt with the best backend available."""

        points = [(float(p.x), float(p.y), int(p.label)) for p in prompt.points]
        sam = self._backend("sam2")
        if sam is not None:
            out = sam.segment(frame_rgb, box=prompt.box, points=points)
            if out is not None:
                if prompt.box is None and points:
                    whole = self._whole_subject_for_clicks(frame_rgb, out, points)
                    if whole is not None:
                        return whole
                return out
        detector = self._backend("detector")
        if detector is not None:
            out = detector.match_prompt(frame_rgb, box=prompt.box, points=points)
            if out is not None:
                return out
        if prompt.box is not None:
            mask = grabcut_mask(frame_rgb, prompt.box)
            if mask is not None:
                self.warnings.append(
                    "The subject mask was created with the classical GrabCut fallback (no segmentation model "
                    "available); install SAM 2 for a more accurate selection."
                )
                return ResolvedSubject(mask=mask, box=box_from_mask(mask) or prompt.box, source="grabcut:box", score=None)
        return None

    def resolve(
        self,
        prompt: SubjectPrompt,
        reader: Any,
        model_info: ModelInfo,
        *,
        frame_limit: Optional[int] = None,
        search_frames: int = 90,
    ) -> ResolvedSubject:
        """Resolve *prompt* against a video (``reader`` is an ``app.processing.reader.VideoReader``).

        In automatic mode, when nothing is found on the reference frame, the following frames are
        searched (at a stride, up to ``search_frames`` candidates) and the first frame with a subject
        becomes the reference frame.
        """

        total = reader.frame_count if frame_limit is None else min(reader.frame_count, frame_limit)
        if prompt.frame_index >= total:
            raise ConfigError(
                f"Subject reference frame {prompt.frame_index} is beyond the last frame ({total - 1}).",
            )
        moved: List[str] = []
        if prompt.mode == "mask":
            prompt, moved = self.align_mask_frame(prompt, reader, total)
        frame = reader.frame_rgb(prompt.frame_index)
        try:
            subject = self.resolve_on_frame(prompt, frame, model_info)
            subject.notes[:0] = moved
            return subject
        except SubjectNotFoundError:
            if prompt.mode != "auto":
                raise
        stride = max(1, min(15, (total - prompt.frame_index) // max(1, search_frames)))
        index = prompt.frame_index + stride
        tried = 0
        while index < total and tried < search_frames:
            if self.cancel is not None:
                self.cancel.raise_if_cancelled()
            candidate = prompt.model_copy(update={"frame_index": index})
            try:
                subject = self.resolve_on_frame(candidate, reader.frame_rgb(index), model_info)
                subject.notes.append(
                    f"No subject in frame {prompt.frame_index}; first found in frame {index}."
                )
                return subject
            except SubjectNotFoundError:
                pass
            index += stride
            tried += 1
        raise SubjectNotFoundError(
            f"No subject was detected between frame {prompt.frame_index} and frame {min(index, total - 1)}."
        )
