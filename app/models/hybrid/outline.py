"""SAM 2 + MatAnyone hybrid for objects with see-through parts (a car with windows, a chair, a bicycle).

Neither model can do this alone:

* SAM 2 follows an object as a whole.  Its outline is fairly stable, but it closes openings: the
  scenery seen through a window stays in the picture.
* MatAnyone decides pixel by pixel.  It keeps openings open, but it was trained on people: around a
  car it also takes passing vehicles, and where the object has the colour of the background (a
  white roof against a white sky) its edge wanders from frame to frame.

The hybrid runs both on every frame.  SAM 2 is given the *solid* outline of the selection (every
opening filled) and MatAnyone the selection as it is::

    outline = SAM 2 mask, smoothed, holes filled
    alpha   = MatAnyone alpha
    alpha   = 1 in the rim of the outline, except in an opening proper           (see guard_matte)
    alpha   = 0 outside the outline

So the outline comes from SAM 2 and everything inside it - the openings, the driver behind the
glass - from MatAnyone.

**One outline for the whole clip.**  SAM 2 can still be fooled for a moment: a van that passes
directly behind a car looks like more car.  A rigid object does not change its shape, though, it
only moves.  The adapter therefore also measures the object's motion (``app.tracking.rigid``) and,
once the whole clip has been seen, puts all outlines on top of each other in the object's own
coordinates and takes the majority (:meth:`Sam2MatAnyoneAdapter.revise`).  That one shape, moved
with the object, replaces the frame-by-frame outlines - if it explains them well; for a subject
that changes its shape (a person) the frame-by-frame outlines are kept.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import zlib
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from ...core.errors import ModelLoadError, SubjectError
from ...tracking.rigid import OutlineMosaic, RigidTracker, scale_affine
from ...utils import torch_ops
from ..base import (
    LicenseInfo,
    MatteOutput,
    MattingModel,
    ModelCapabilities,
    ModelInfo,
    ModelVariant,
    SequenceInfo,
)
from ..registry import register_model

log = logging.getLogger(__name__)

#: GPU memory SAM 2 needs while tracking, beyond its weights (small variant, measured: 0.5 GB);
#: MatAnyone chooses its working size from what is left.
_SAM2_RESERVE_MB = 600.0


def solid_outline(mask: np.ndarray) -> np.ndarray:
    """*mask* with every enclosed opening filled, as a boolean array.

    An opening is background that cannot be reached from the border of the frame.
    """

    binary = np.asarray(mask)
    if binary.dtype != np.bool_:
        binary = binary > 0.5
    height, width = binary.shape
    padded = np.ones((height + 2, width + 2), np.uint8)           # a ring of background around the frame
    padded[1:-1, 1:-1] = (~binary).astype(np.uint8)
    cv2.floodFill(padded, None, (0, 0), 2)                        # everything reachable from outside
    return padded[1:-1, 1:-1] != 2


def guard_matte(alpha: np.ndarray, outline: np.ndarray, *, margin: float, rim: float, sliver: float) -> np.ndarray:
    """Limit a matte to an outline and make the rim of the outline solid again.

    ``alpha``    float32 ``[H, W]`` in 0..1 (the per-pixel matte, with its openings).
    ``outline``  bool ``[H, W]``, solid (no holes).
    ``margin``   pixels over which the matte fades to zero outside the outline.
    ``rim``      depth in pixels of the rim zone, measured inwards from the outline's edge.
    ``sliver``   a gap between the matte and the outline thinner than this is left alone: there the
                 matte's own, finer edge is kept.

    Inside the rim zone everything that is not fully opaque is made opaque, except an *opening
    proper*: a transparent area that is enclosed by the subject and reaches deeper than the rim (a
    window).  What is repaired is therefore a piece the matte dropped at the edge, a faint patch
    in a part that has the colour of the background, or - when the matte wrongly connected an
    opening to the outside - the strip along the edge, never the opening as a whole.
    """

    alpha = np.asarray(alpha, dtype=np.float32)
    outline = np.asarray(outline, dtype=bool)
    if not outline.any():
        return np.zeros_like(alpha)
    result = alpha
    solid = outline.astype(np.uint8)
    if rim > 0:
        faint = outline & (alpha < 0.9)
        if faint.any():
            depth = cv2.distanceTransform(solid, cv2.DIST_L2, 3)   # 1 on the outline's edge, growing inwards
            zone = depth <= rim
            protected = np.zeros_like(outline)
            clear = outline & (alpha < 0.5)
            if clear.any():
                count, labels = cv2.connectedComponents(clear.astype(np.uint8), connectivity=4)
                reaches_deep = np.zeros(count, bool)
                reaches_deep[np.unique(labels[clear & ~zone])] = True
                touches_edge = np.zeros(count, bool)
                touches_edge[np.unique(labels[clear & (depth <= 1.5)])] = True
                opening = reaches_deep & ~touches_edge
                opening[0] = False
                protected = opening[labels]
                if protected.any():                               # ... with its soft edge
                    size = 2 * max(2, int(round(sliver / 2.0))) + 1
                    protected = cv2.dilate(protected.astype(np.uint8), np.ones((size, size), np.uint8)) > 0
            candidates = faint & zone & ~protected
            core = candidates & (depth > sliver)
            if core.any():
                size = 2 * int(math.ceil(2.0 * sliver)) + 1
                near = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))) > 0
                result = np.where(candidates & (core | near), np.float32(1.0), alpha)
    margin = max(1.0, float(margin))
    size = 2 * int(round(margin)) + 1
    gate = cv2.dilate(solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))).astype(np.float32)
    gate = cv2.GaussianBlur(gate, (0, 0), max(0.8, margin / 3.0))
    gate[outline] = 1.0                                           # the matte passes unchanged inside
    return (result * gate).astype(np.float32)


def _variant(variant_id: str, label: str, sam2: str, text: str, vram: int) -> ModelVariant:
    return ModelVariant(id=variant_id, label=label, weights=(), description=text, vram_hint_mb=vram,
                        options={"sam2_variant": sam2, "matanyone_variant": "matanyone2"})


@register_model
class Sam2MatAnyoneAdapter(MattingModel):
    info = ModelInfo(
        id="sam2_matanyone",
        name="SAM 2 + MatAnyone (objects with openings)",
        description=(
            "For an object you can see through in places - a car with its windows, a chair, a bicycle: SAM 2 "
            "holds the outline steady and keeps passing things out, MatAnyone keeps the openings open. "
            "Needs a mask or a selection of the object; non-commercial licence."
        ),
        license=LicenseInfo(
            name="NTU S-Lab License 1.0 (MatAnyone) + Apache-2.0 (SAM 2)",
            url="https://github.com/pq-yang/MatAnyone/blob/main/LICENSE",
            commercial_use="prohibited",
            notes=(
                "The stricter component governs: MatAnyone's code and weights may be used for research and other "
                "non-commercial purposes only. SAM 2 is Apache-2.0."
            ),
        ),
        capabilities=ModelCapabilities(
            alpha_type="alpha",
            native_prompts=frozenset({"mask"}),
            needs_initial_mask=True,
            subject_domain="any",
            temporal="memory",
            max_batch_size=1,
            supports_fp16=True,
            devices=frozenset({"cuda", "cpu"}),
            supports_reverse=True,
            tracks_single_subject=True,
            two_pass=True,
        ),
        variants=(
            _variant("balanced", "SAM 2.1 small + MatAnyone 2 (recommended)", "small",
                     "Stable outline tracking with the current MatAnyone weights.", 2800),
            _variant("fast", "SAM 2.1 tiny + MatAnyone 2", "tiny", "Slightly faster outline tracking.", 2700),
        ),
        default_variant="balanced",
        quality_variants={"fast": "fast", "balanced": "balanced", "high": "balanced", "max": "balanced"},
        limitations=(
            "Non-commercial licence (MatAnyone, NTU S-Lab License 1.0): refused when commercial use is switched on.",
            "Needs a mask of the object on the reference frame, with the openings cut out. Boxes and clicks are "
            "turned into a mask by the helper models, but that mask has no openings.",
            "The outline is as hard as SAM 2 draws it: fine hair at the edge is not preserved. For people use "
            "'matanyone' or 'sam2_rvm'.",
            "Something that passes behind an opening (seen through a window) is judged by MatAnyone alone and can "
            "stay visible for some frames.",
            "The outline is held as one fixed shape that moves with the object. That suits rigid things; for a "
            "subject that changes its shape the tool notices and falls back to frame-by-frame outlines "
            "(option rigid_outline=off forces that).",
            "Where the object has exactly the colour of what is behind it, the outline is a best guess.",
            "Two networks run on every frame, and the whole clip is analysed before the first frame is written: "
            "slower and more memory-hungry than either model alone.",
        ),
        requires=("sam2", "hydra", "matanyone", "omegaconf"),
        install_hint="Install both packages with:  python scripts/install_models.py sam2  and  "
        "python scripts/install_models.py matanyone --accept-license",
        homepage="https://github.com/pq-yang/MatAnyone",
        citation="Ravi et al., SAM 2 (ICLR 2025); Yang et al., MatAnyone (CVPR 2025).",
        depends_on=("sam2", "matanyone"),
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sam2: Optional[MattingModel] = None
        self._matte: Optional[MattingModel] = None
        #: all four in pixels at 1080p
        self._margin = float(self.options.get("gate_margin", 4.0))
        self._rim = float(self.options.get("rim_depth", 44.0)) if self.options.get("restore_rim", True) else 0.0
        self._sliver = float(self.options.get("sliver", 6.0))
        self._smooth = float(self.options.get("outline_smoothing", 3.0))
        #: "auto" = one outline for the clip when it explains the single outlines well; "on" / "off"
        self._rigid = str(self.options.get("rigid_outline", "auto")).strip().lower()
        if self._rigid in ("true", "1", "yes"):
            self._rigid = "on"
        elif self._rigid in ("false", "0", "no"):
            self._rigid = "off"
        self._rigid_agreement = float(self.options.get("rigid_agreement", 0.93))
        #: pixels at 1080p by which the one outline of the clip is drawn in: it is right to within a
        #: pixel or two, and a sliver of background at the edge shows more than a sliver of subject missing
        self._choke = float(self.options.get("outline_choke", 2.0))
        self._track: Optional[Tuple[int, MatteOutput]] = None      # SAM 2's answer for the frame in work
        self._size = (0, 0)
        self._revising = False
        self._reference: Optional[np.ndarray] = None               # the solid outline given for the reference frame
        self._features: Optional[np.ndarray] = None                # where motion features may be taken (work size)
        self._work_scale = 1.0
        self._tracker: Optional[RigidTracker] = None
        self._motions: Dict[int, np.ndarray] = {}                  # frame -> reference-to-frame motion (full size)
        self._outlines: Dict[int, bytes] = {}                      # frame -> SAM 2 outline, packed
        self._restored_peak = 0.0
        self._rigid_report: Dict[str, Any] = {"mode": self._rigid, "used": False}

    # ------------------------------------------------------------------------- life cycle
    def load(self) -> None:
        if self.factory is None:
            raise ModelLoadError(
                "The hybrid adapter must be created through the model registry (it builds its two component models).",
                hint="Use app.models.registry.create_model('sam2_matanyone', ...).",
            )
        passthrough = {k: v for k, v in self.options.items()
                       if k in ("internal_short_side", "n_warmup", "mem_every", "max_mem_frames")}
        passthrough.setdefault("reserve_vram_mb", _SAM2_RESERVE_MB)
        self._sam2 = self.factory("sam2", self.variant.options["sam2_variant"], {})
        self._matte = self.factory("matanyone", self.variant.options["matanyone_variant"], passthrough)
        self._sam2.load()
        try:
            self._matte.load()
        except BaseException:
            self._sam2.unload()
            self._sam2 = None
            raise
        self.loaded = True

    def unload(self) -> None:
        for model in (self._sam2, self._matte):
            if model is not None:
                try:
                    model.unload()
                except Exception as exc:  # pragma: no cover - best effort
                    log.debug("unloading a component failed: %s", exc)
        self._sam2 = self._matte = None
        self._track = None
        self._tracker = None
        self._motions.clear()
        self._outlines.clear()
        super().unload()

    def start_sequence(self, seq: SequenceInfo) -> None:
        if self._sam2 is None or self._matte is None:
            raise ModelLoadError("Hybrid adapter used before load().")
        subject = seq.subject
        if subject is None or subject.mask is None:
            raise SubjectError(
                "This model needs a mask of the object on the reference frame, but none was provided.",
                hint="Give a mask image with the openings cut out (--subject mask:<file>), or select the object "
                "with a box or clicks (needs the SAM 2 helper).",
            )
        mask = np.asarray(subject.mask) > 0.5
        outline = solid_outline(mask)
        if not outline.any():
            raise SubjectError("The subject mask is empty.", hint="Select the subject on a frame where it is visible.")
        whole = dataclasses.replace(subject, mask=outline.astype(np.float32), box=None, points=[])
        self._track = None
        size = (int(seq.height), int(seq.width))
        # The first sequence of a clip starts on the reference frame with the user's selection; a
        # second one (the other direction in time) starts on the same frame and continues the
        # records.  Anything else is a new clip for this adapter.
        continues = (self._revising and bool(seq.will_revise) and size == self._size
                     and self._reference is not None and self._reference.shape == outline.shape
                     and bool((self._reference == outline).all()))
        if not continues:
            self._motions.clear()
            self._outlines.clear()
            self._restored_peak = 0.0
            self._rigid_report = {"mode": self._rigid, "used": False}
        self._revising = bool(seq.will_revise)
        self._size = size
        self._reference = outline
        self._tracker = None
        self._features = None
        if self._revising and self._rigid != "off":
            height, width = size
            self._work_scale = min(1.0, 540.0 / min(height, width))
            small = (max(1, int(round(width * self._work_scale))), max(1, int(round(height * self._work_scale))))
            # Features come from the subject proper: away from its edge and from the openings, where
            # the background shows.
            margin = max(3, int(round(0.012 * min(height, width))))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
            inner = cv2.erode(mask.astype(np.uint8), kernel)
            self._features = cv2.resize(inner, small, interpolation=cv2.INTER_NEAREST) > 0
        self._sam2.start_sequence(dataclasses.replace(seq, subject=whole))
        self._matte.start_sequence(seq)

    def end_sequence(self) -> None:
        self._track = None
        self._tracker = None
        for model in (self._sam2, self._matte):
            if model is not None:
                model.end_sequence()

    # ------------------------------------------------------------------------- inference
    def _limits(self) -> Dict[str, float]:
        scale = min(self._size) / 1080.0
        return {"margin": max(1.0, self._margin * scale), "rim": self._rim * scale,
                "sliver": max(2.0, self._sliver * scale)}

    def _follow(self, index: int, frame: torch.Tensor) -> None:
        """Measure the subject's motion on this frame (for the one outline of the clip)."""

        if self._features is None:
            return
        height, width = self._size
        small = (max(1, int(round(height * self._work_scale))), max(1, int(round(width * self._work_scale))))
        luma = (0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2])[None, None].to(torch.float32)
        if small != (height, width):
            luma = torch.nn.functional.interpolate(luma, size=small, mode="area")
        gray = (luma[0, 0].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
        if self._tracker is None:
            self._tracker = RigidTracker(gray, self._features)
            self._motions[index] = np.eye(3, dtype=np.float64)        # the reference frame itself
            return
        motion = self._tracker.step(gray)
        if motion is not None:
            self._motions[index] = scale_affine(motion, 1.0 / self._work_scale)

    def process(self, frames: torch.Tensor, frame_indices: Sequence[int]) -> MatteOutput:
        if self._sam2 is None or self._matte is None:
            raise ModelLoadError("Hybrid adapter used before load().")
        index = int(frame_indices[0])
        # SAM 2 must see every frame exactly once.  When MatAnyone fails afterwards (for example out
        # of memory) and the frame is tried again, SAM 2's answer for it is reused; MatAnyone rebuilds
        # its own memory after a failure.  A retry of the same frame is therefore safe.
        if self._track is not None and self._track[0] == index:
            track = self._track[1]
        else:
            track = self._sam2.process(frames, frame_indices)
            self._track = (index, track)
        matte = self._matte.process(frames, frame_indices)
        self._track = None

        scale = min(self._size) / 1080.0
        # The mask comes from a 256 x 256 grid: smoothing before the threshold removes the steps.
        probability = torch_ops.gaussian_blur(track.alpha.to(torch.float32), max(0.6, self._smooth * scale))
        outline = solid_outline((probability[0, 0] > 0.5).cpu().numpy())
        if self._revising:
            # The outline is decided in revise(), with the whole clip in view: keep what that needs
            # and hand out MatAnyone's matte untouched.
            self._outlines[index] = zlib.compress(np.packbits(outline).tobytes(), 1)
            self._follow(index, frames[0])
            return MatteOutput(alpha=matte.alpha.to(torch.float32).clamp(0.0, 1.0), presence=track.presence,
                               is_mask=False)
        alpha = matte.alpha[0, 0].to(torch.float32).cpu().numpy()
        guarded = guard_matte(alpha, outline, **self._limits())
        self._restored_peak = max(self._restored_peak, float(np.mean(np.maximum(guarded - alpha, 0.0))))
        out = torch.from_numpy(guarded).to(frames.device)[None, None].clamp_(0.0, 1.0)
        return MatteOutput(alpha=out, presence=track.presence, is_mask=False)

    # ------------------------------------------------------------------------- the whole clip
    def _outline_of(self, index: int) -> np.ndarray:
        height, width = self._size
        raw = np.frombuffer(zlib.decompress(self._outlines[index]), np.uint8)
        return np.unpackbits(raw, count=height * width).reshape(height, width).astype(bool)

    def revise(self, indices: Sequence[int], load: Callable[[int], np.ndarray], save: Callable[[int, np.ndarray], None],
               check: Optional[Callable[[], None]] = None) -> None:
        frames = [int(i) for i in indices if int(i) in self._outlines]
        if not frames:
            return
        mosaic = None
        shape: Optional[OutlineMosaic] = None
        followed = [i for i in frames if i in self._motions]
        report: Dict[str, Any] = {"mode": self._rigid, "used": False, "frames": len(frames),
                                  "frames_followed": len(followed)}
        if self._rigid != "off":
            enough = len(followed) >= max(2, int(0.6 * len(frames)))
            if followed and (enough or self._rigid == "on"):
                # At most about 400 votes: plenty for a majority, and quick on long clips.
                voters = followed[::max(1, len(followed) // 400)]
                shape = OutlineMosaic(self._size, scale=min(0.5, 540.0 / min(self._size)))

                def outlines() -> Iterator[np.ndarray]:
                    for i in voters:
                        if check is not None:
                            check()
                        yield self._outline_of(i)

                mosaic = shape.build(outlines(), [self._motions[i] for i in voters], self._reference)
                if mosaic is None:
                    report["reason"] = "the subject moved too far for one shape"
                else:
                    report["agreement"] = round(mosaic.agreement, 4)
                    if self._rigid == "auto" and mosaic.agreement < self._rigid_agreement:
                        report["reason"] = "the outline changes its shape from frame to frame"
                        mosaic = None
            else:
                report["reason"] = "the subject's motion could not be followed"
        report["used"] = mosaic is not None
        limits = self._limits()
        tight = dict(limits, margin=min(limits["margin"], max(1.0, 2.0 * min(self._size) / 1080.0)))
        choke = int(round(self._choke * min(self._size) / 1080.0))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * choke + 1, 2 * choke + 1)) if choke > 0 else None
        carried = 0
        for i in frames:
            if check is not None:
                check()
            if mosaic is not None and shape is not None and i in self._motions:
                outline = shape.carry(mosaic, self._motions[i])
                if kernel is not None:
                    # (the frame border is not an edge of the subject: nothing is taken off there)
                    outline = cv2.erode(outline.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                                        borderValue=1) > 0
                carried += 1
                use = tight
            else:
                outline = self._outline_of(i)
                use = limits
            alpha = np.asarray(load(i), dtype=np.float32)
            guarded = guard_matte(alpha, outline, **use)
            self._restored_peak = max(self._restored_peak, float(np.mean(np.maximum(guarded - alpha, 0.0))))
            save(i, guarded)
        report["frames_with_the_one_outline"] = carried
        self._rigid_report = report
        self._motions.clear()
        self._outlines.clear()
        self._revising = False
        log.info("Outline for the whole clip: %s", report)

    def reduce_memory(self) -> bool:
        return bool(self._matte is not None and self._matte.reduce_memory())

    def runtime_summary(self) -> Dict[str, Any]:
        summary = super().runtime_summary()
        summary["options"].pop("quality", None)
        summary.update({
            "gate_margin_px_at_1080p": self._margin,
            "rim_depth_px_at_1080p": self._rim,
            "outline_smoothing_px_at_1080p": self._smooth,
            "outline_choke_px_at_1080p": self._choke,
            "largest_rim_repair_share": round(self._restored_peak, 5),
            "rigid_outline": dict(self._rigid_report),
        })
        if self._sam2 is not None:
            summary["sam2"] = self._sam2.runtime_summary()
        if self._matte is not None:
            summary["matanyone"] = self._matte.runtime_summary()
        return summary
