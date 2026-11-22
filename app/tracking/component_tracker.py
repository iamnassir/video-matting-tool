"""Subject isolation for models that matte every person: connected-component tracking of the matte.

Models without prompt support (Robust Video Matting) matte *everybody* in the picture.  When the
user selected one subject, this tracker removes the others: it follows the connected components of
the matte from frame to frame, keeps the ones that belong to the selected subject and zeroes the
rest.  It never changes the matte of the subject itself - kept regions are multiplied by exactly
1.0 - and it never looks at the picture, only at the matte.

How it works
------------
* The matte is thresholded and reduced to a small binary map (long side ``work_side``) in which a
  cell is set as soon as one of its pixels is; components are labelled there on the CPU with OpenCV.
* **Acquisition.**  With a seed (mask, box or clicks from the subject selection) the components
  under the seed become the subject.  Without one the primary subject is chosen on the first frame
  that has a component: largest ``area x centrality``.
* **Association.**  From then on a component belongs to the subject when it overlaps the subject's
  previous region; when nothing overlaps (fast motion), the component within a motion margin of a
  few percent of the frame diagonal takes over.  Specks next to the subject (a detached strand, a
  fingertip) are kept with it but not remembered as part of it.
* **Negative tracks.**  Every other sizeable component seen while the subject is tracked is another
  person and gets a track of its own, updated by the same overlap rule.  A component that belongs to
  a negative track can never become the subject - this is what stops the selection from jumping to
  a bystander when the subject is occluded or leaves the frame.
* **Output.**  The matte is multiplied by a keep map: 1 on the subject's components and in a zone
  of 2.5 % of the frame diagonal around them (soft fringe, hair, motion-blur tails), 0 on every other
  component and on faint matte elsewhere in the frame.
* **Lost.**  When nothing is left of the subject (or only a remnant below 5 % of its recent size:
  following a speck would turn the returning subject into "someone else") the state becomes
  ``"lost"``: nothing is kept (the output is empty), the search region grows from the motion margin
  to the whole frame over ``lost_patience_frames`` frames, and the subject is re-acquired by the best
  new component inside it - large, and close to where the subject was last seen.  Components of
  negative tracks stay excluded for as long as it takes; after the patience has run out the tracker
  simply keeps waiting.

Inherent limitation
-------------------
A connected component is all this method can see.  **When two people touch or overlap, their mattes
merge into one component and cannot be separated**: while they touch, both are kept (dropping the
merged component would drop the subject).  The other person's track is frozen at the place where
it was last seen alone, so the split is usually resolved correctly, but if the two swap places
while merged the tracker cannot know who is who, and somebody who is first seen attached to the
subject (emerging from behind it) is taken for a part of it.  The same holds for people standing
closer together than about one cell of the working map (0.4 % of the frame's long side).  Models
that segment a prompted subject themselves (the SAM 2 based adapters) do not have this limitation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch

from ..core.errors import ConfigError, SubjectError, SubjectNotFoundError, VmtError
from ..core.types import clamp_box
from ..utils import torch_ops

log = logging.getLogger(__name__)

__all__ = ["ComponentSubjectTracker", "TrackingError"]

# Roles a component can have in one frame.  Values >= 0 are indices into the negative tracks.
_NEW = -1
_SUBJECT = -2
#: A speck right next to the subject (a detached strand, a fingertip): kept in the output, but not
#: remembered as part of the subject - otherwise somebody who enters the frame beside the subject
#: would start as such a speck and grow into "the subject".
_FRAGMENT = -3

_KERNEL3 = np.ones((3, 3), dtype=np.uint8)


class TrackingError(VmtError):
    """The tracking helpers were called with data that violates their contract (shape, size).

    ``app.core.errors`` has no class for internal contract violations between engine modules, so the
    tracking package defines this one.
    """

    code = "tracking_error"
    default_hint = "This is a programming error in the caller, not a problem with the video."


@dataclass
class _Negative:
    """Another person: the region it occupied when it was last seen as a component of its own."""

    mask: np.ndarray          # bool [h, w] in the working map
    area: float               # cells
    last_seen: int            # frame index


class ComponentSubjectTracker:
    """Keeps only the tracked subject's connected components of the matte.

    Parameters
    ----------
    width, height : frame size in display orientation.
    work_side     : long side of the reduced binary map the tracking runs on.
    threshold     : matte value above which a pixel belongs to a component.
    min_area      : fraction of the frame area a component needs to be acquired as the subject or to
                    be tracked as another person (smaller ones are specks: never adopted, kept only
                    as fragments right next to the subject).
    lost_patience_frames : frames over which the search region grows to the whole frame after the
                    subject was lost.
    """

    #: Motion margin per frame as a fraction of the frame diagonal.
    MOTION_MARGIN = 0.04
    #: Width of the zone around the subject's components that is kept with them (its soft fringe,
    #: hair, motion-blur tails), as a fraction of the frame diagonal.
    FRINGE_MARGIN = 0.025
    #: The margin grows with the number of skipped frames up to this factor.
    MAX_GAP_SCALE = 8
    #: A component covering at least this share of the subject's last region contains the subject.
    CONTAIN_FRACTION = 0.3
    #: A re-acquired component must have at least this share of the subject's recent area.
    REACQUIRE_FRACTION = 0.05
    #: Per-frame decay of the remembered subject area (a slowly decaying maximum).
    REFERENCE_DECAY = 0.97
    #: Frames a negative track may be unseen and still be matched by proximity instead of overlap.
    NEGATIVE_ACTIVE_GAP = 3
    #: Upper bound of simultaneously remembered other people.
    MAX_NEGATIVES = 32

    def __init__(
        self,
        width: int,
        height: int,
        *,
        work_side: int = 256,
        threshold: float = 0.3,
        min_area: float = 0.0005,
        lost_patience_frames: int = 45,
    ) -> None:
        if int(width) < 1 or int(height) < 1:
            raise ConfigError(f"Invalid frame size {width}x{height} for the subject tracker.")
        if int(work_side) < 16:
            raise ConfigError(f"work_side must be at least 16, got {work_side!r}.")
        if not (0.0 < float(threshold) < 1.0):
            raise ConfigError(f"threshold must be between 0 and 1 (exclusive), got {threshold!r}.")
        if not (0.0 <= float(min_area) < 1.0):
            raise ConfigError(f"min_area is a fraction of the frame area (0 <= min_area < 1), got {min_area!r}.")
        if int(lost_patience_frames) < 1:
            raise ConfigError(f"lost_patience_frames must be >= 1, got {lost_patience_frames!r}.")
        self.width = int(width)
        self.height = int(height)
        self.threshold = float(threshold)
        self._w, self._h = torch_ops.fit_size(self.width, self.height, int(work_side))
        self._diag = math.hypot(self._w, self._h)
        self._margin = max(2.0, self.MOTION_MARGIN * self._diag)
        self._fringe = max(3.0, self.FRINGE_MARGIN * self._diag)
        self._min_area_cells = max(1.0, float(min_area) * self._w * self._h)
        self._patience = int(lost_patience_frames)
        # Remember someone who left for a good while (they often come back through the same door),
        # but not for ever: a stale region must not block the place for the rest of the video.
        self._negative_ttl = max(4 * self._patience, 120)
        self._events: List[Dict[str, Any]] = []
        self.reset()

    # ------------------------------------------------------------------ public API
    @property
    def state(self) -> str:
        """``"searching"`` (no subject yet), ``"tracking"`` or ``"lost"``."""

        return self._state

    def events(self) -> List[Dict[str, Any]]:
        """Tracking events in the order they happened.

        ``{"frame": int, "event": "acquired" | "lost" | "reacquired", ...}`` plus details (``area`` as
        a fraction of the frame, ``bbox`` in full-resolution pixels, ``mode``, ``lost_frames``).
        Events are kept across :meth:`reset` so one tracker can report a backward and a forward
        pass together.
        """

        return [dict(e) for e in self._events]

    def reset(
        self,
        seed_mask: Optional[np.ndarray] = None,
        seed_box: Optional[Tuple[float, float, float, float]] = None,
        seed_points: Sequence[Tuple[float, float, int]] = (),
    ) -> None:
        """Start a new sequence, optionally seeded with the selected subject.

        ``seed_mask`` (H x W, non-zero / > 0.5 = subject), ``seed_box`` (x1, y1, x2, y2) and
        ``seed_points`` ((x, y, label) with label 1 = subject, 0 = not the subject) are in
        full-resolution display pixels of the first frame that will be passed to :meth:`apply`.
        Without any seed the primary subject is chosen automatically.
        """

        self._state = "searching"
        self._subject: Optional[np.ndarray] = None       # bool [h, w]: last known region of the subject
        self._subject_area = 0.0
        self._reference_area = 0.0
        self._negatives: List[_Negative] = []
        self._last_index: Optional[int] = None
        self._first_index: Optional[int] = None
        self._lost_index: Optional[int] = None
        self._tracked_frames = 0

        self._seed_region: Optional[np.ndarray] = None
        self._seed_positive: List[Tuple[float, float]] = []
        self._seed_negative: List[Tuple[float, float]] = []
        sx, sy = self._w / float(self.width), self._h / float(self.height)

        if seed_mask is not None:
            mask = np.asarray(seed_mask)
            if mask.ndim != 2 or mask.size == 0:
                raise SubjectError(f"The subject seed mask must be a 2-D array (H x W), got shape {mask.shape}.")
            values = mask.astype(np.float32)
            peak = float(values.max())
            if not math.isfinite(peak) or peak <= 0.0:
                raise SubjectNotFoundError("The subject seed mask is empty: it does not mark any pixel.")
            binary = (values > (0.5 if peak <= 1.0 else 0.5 * peak)).astype(np.float32)
            coverage = cv2.resize(binary, (self._w, self._h), interpolation=cv2.INTER_AREA)
            region = coverage >= 0.25
            if not region.any():
                region = coverage > 0.0
            if not region.any():
                raise SubjectNotFoundError("The subject seed mask is empty: it does not mark any pixel.")
            self._seed_region = region
        elif seed_box is not None:
            try:
                x1, y1, x2, y2 = clamp_box(seed_box, self.width, self.height)
            except (TypeError, ValueError) as exc:
                raise SubjectError(f"The subject seed box must be (x1, y1, x2, y2), got {seed_box!r}.") from exc
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 - x1 <= 0.0 or y2 - y1 <= 0.0:
                raise SubjectError(
                    f"The subject seed box {tuple(seed_box)} does not overlap the {self.width}x{self.height} frame.",
                    hint="Give the box in display pixels of the reference frame: x1, y1, x2, y2.",
                )
            region = np.zeros((self._h, self._w), dtype=bool)
            cx1, cy1 = int(math.floor(x1 * sx)), int(math.floor(y1 * sy))
            cx2 = min(self._w, max(cx1 + 1, int(math.ceil(x2 * sx))))
            cy2 = min(self._h, max(cy1 + 1, int(math.ceil(y2 * sy))))
            region[min(cy1, self._h - 1) : cy2, min(cx1, self._w - 1) : cx2] = True
            self._seed_region = region

        for point in seed_points:
            try:
                x, y, label = float(point[0]), float(point[1]), int(point[2])
            except (TypeError, ValueError, IndexError) as exc:
                raise SubjectError(f"Seed points must be (x, y, label) triples, got {point!r}.") from exc
            if label not in (0, 1) or not (math.isfinite(x) and math.isfinite(y)):
                raise SubjectError(f"Invalid seed point {point!r}: label must be 1 (subject) or 0 (not the subject).")
            cell = (min(max(x * sx, 0.0), self._w - 1.0), min(max(y * sy, 0.0), self._h - 1.0))
            (self._seed_positive if label == 1 else self._seed_negative).append(cell)

        self._seeded = self._seed_region is not None or bool(self._seed_positive)

    def apply(self, alpha: torch.Tensor, frame_indices: Sequence[int]) -> torch.Tensor:
        """Zero every component of the matte that does not belong to the tracked subject.

        ``alpha`` ``[B, 1, H, W]`` in 0..1 on any device, ``frame_indices`` the frame numbers of the
        batch in processing order.  Returns a float32 tensor of the same shape on the same device:
        pixels of kept components and of their fringe zone are numerically unchanged (multiplied by
        exactly 1.0), pixels of removed components are exactly 0, as is everything outside the
        fringe zone of the subject; the transition is a short ramp in the gap between a kept and a
        removed component or at the rim of the zone.  When nothing has to be removed in the whole
        batch the input tensor itself is returned; the input is never modified.
        """

        if not isinstance(alpha, torch.Tensor) or alpha.dim() != 4 or alpha.shape[1] != 1:
            got = tuple(alpha.shape) if isinstance(alpha, torch.Tensor) else type(alpha).__name__
            raise TrackingError(f"apply() expects alpha of shape [B, 1, H, W], got {got}.")
        if tuple(alpha.shape[-2:]) != (self.height, self.width):
            raise TrackingError(
                f"The matte is {alpha.shape[-1]}x{alpha.shape[-2]} but the tracker was created for "
                f"{self.width}x{self.height}.",
                hint="Create the ComponentSubjectTracker with the display-oriented frame size of the video.",
            )
        if len(frame_indices) != alpha.shape[0]:
            raise TrackingError(
                f"{len(frame_indices)} frame indices were given for a batch of {alpha.shape[0]} frames."
            )
        a = alpha if alpha.dtype == torch.float32 else alpha.to(torch.float32)
        if a.shape[0] == 0:
            return a

        # Two binary maps: pixels of components (above the threshold) and pixels with any matte at all.
        binary = torch.cat([a > self.threshold, a > 0.0], dim=1)
        if (self.height, self.width) != (self._h, self._w):
            # "Any" pooling: a cell is set when one of its pixels is, so every pixel of a component
            # lies inside a cell of that component and thin limbs stay connected.
            binary = torch_ops.resize(binary.to(torch.float32), (self._h, self._w), mode="area") > 0.0
        cells = binary.to(torch.uint8).cpu().numpy()

        keep_maps: List[Optional[np.ndarray]] = []
        for i, index in enumerate(frame_indices):
            keep_maps.append(self._step(np.ascontiguousarray(cells[i, 0]), cells[i, 1].astype(bool), int(index)))
        if all(k is None for k in keep_maps):
            return a

        ones = np.ones((self._h, self._w), dtype=np.float32)
        stacked = np.stack([ones if k is None else k for k in keep_maps])[:, None]
        keep = torch.from_numpy(stacked).to(a.device)
        if (self.height, self.width) != (self._h, self._w):
            keep = torch_ops.resize(keep, (self.height, self.width), mode="bilinear")
        # After bilinear up-sampling every pixel inside a kept component's cell is >= 0.625 and
        # every pixel inside a removed component's cell is <= 0.375 (see _keep_map), so this ramp
        # makes them exactly 1.0 and exactly 0.0.
        keep = (keep * 5.0 - 2.0).clamp_(0.0, 1.0)
        return a * keep

    # ------------------------------------------------------------------ one frame
    def _step(self, cells: np.ndarray, occupied: np.ndarray, index: int) -> Optional[np.ndarray]:
        """Advance the tracker by one frame.  Returns the keep map, or ``None`` to keep everything.

        ``cells`` is the binary component map (uint8), ``occupied`` marks cells with any matte at all.
        """

        gap = 1 if self._last_index is None else max(1, abs(index - self._last_index))
        self._last_index = index
        if self._first_index is None:
            self._first_index = index
        reach = float(min(self._diag, self._margin * min(gap, self.MAX_GAP_SCALE)))

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(cells, connectivity=8, ltype=cv2.CV_32S)
        areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
        areas[0] = 0.0
        roles, merged = self._associate(labels, n, areas, reach, index)

        if self._state == "tracking":
            remaining = float(areas[roles == _SUBJECT].sum())
            if remaining < self.REACQUIRE_FRACTION * self._reference_area:
                # Nothing overlaps the subject any more - or only a remnant far too small to be it
                # (the last sliver before an occlusion, a speck the model left behind).  Following a
                # speck would be worse than waiting: the real subject would come back as "someone
                # else" and be excluded.
                roles[(roles == _SUBJECT) | (roles == _FRAGMENT)] = _NEW
                self._state = "lost"
                self._lost_index = index
                self._events.append(
                    {
                        "frame": index,
                        "event": "lost",
                        "tracked_frames": self._tracked_frames,
                        "last_bbox": self._bbox(self._subject),
                    }
                )
                log.info("subject tracker: subject lost at frame %d", index)
        elif self._state == "lost":
            found = self._reacquire(labels, n, areas, roles, reach, index)
            if found is not None:
                distance, lost_frames = found
                self._state = "tracking"
                self._tracked_frames = 0
                region = np.isin(labels, np.flatnonzero(roles == _SUBJECT))
                self._events.append(
                    {
                        "frame": index,
                        "event": "reacquired",
                        "lost_frames": lost_frames,
                        "distance": round(distance / self._diag, 4),
                        "area": round(float(region.mean()), 6),
                        "bbox": self._bbox(region),
                    }
                )
                log.info("subject tracker: subject re-acquired at frame %d after %d frames", index, lost_frames)
        else:
            mode = self._acquire(labels, n, areas, centroids, roles, index)
            if mode is not None:
                self._state = "tracking"
                self._tracked_frames = 0
                region = np.isin(labels, np.flatnonzero(roles == _SUBJECT))
                self._events.append(
                    {
                        "frame": index,
                        "event": "acquired",
                        "mode": mode,
                        "area": round(float(region.mean()), 6),
                        "bbox": self._bbox(region),
                    }
                )
                log.info("subject tracker: subject acquired at frame %d (%s)", index, mode)

        subject_labels = np.flatnonzero(roles == _SUBJECT)
        kept: Optional[np.ndarray] = None
        self._update_negatives(labels, roles, index)
        if self._state == "tracking" and subject_labels.size:
            core = np.isin(labels, subject_labels)
            fragments = np.flatnonzero(roles == _FRAGMENT)
            kept = core | np.isin(labels, fragments) if fragments.size else core
            self._update_subject(core, merged)
            # Everything else of a relevant size that is visible while the subject is tracked is
            # somebody (or something) else and must never be adopted later.
            for j in np.flatnonzero(roles == _NEW):
                if j > 0 and areas[j] >= self._min_area_cells:
                    self._negatives.append(_Negative(mask=labels == j, area=float(areas[j]), last_seen=index))
            if len(self._negatives) > self.MAX_NEGATIVES:
                self._negatives.sort(key=lambda neg: (neg.last_seen, neg.area), reverse=True)
                del self._negatives[self.MAX_NEGATIVES :]
        return self._keep_map(labels, kept, occupied)

    # ------------------------------------------------------------------ association
    def _associate(
        self, labels: np.ndarray, n: int, areas: np.ndarray, reach: float, index: int
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Give every component a role: the subject, one of the negative tracks, or new.

        Returns ``(roles, merged)`` where ``merged`` lists the regions of negative tracks that are
        currently fused with the subject's component (two people touching).
        """

        roles = np.full(n, _NEW, dtype=np.int64)
        merged: List[np.ndarray] = []
        if n <= 1:
            return roles, merged

        # Tracks: (role id, region, near reach or None).  The subject, when tracked, is track 0.
        tracks: List[Tuple[int, np.ndarray, Optional[float]]] = []
        tracking = self._state == "tracking" and self._subject is not None
        if tracking:
            assert self._subject is not None
            tracks.append((_SUBJECT, self._subject, reach))
        for k, neg in enumerate(self._negatives):
            unseen = max(1, abs(index - neg.last_seen))
            near = self._margin * min(unseen, self.MAX_GAP_SCALE) if unseen <= self.NEGATIVE_ACTIVE_GAP else None
            tracks.append((k, neg.mask, near))
        if not tracks:
            return roles, merged

        direct = np.zeros((len(tracks), n), dtype=np.float64)
        near_overlap = np.zeros((len(tracks), n), dtype=np.float64)
        for t, (_, region, near_reach) in enumerate(tracks):
            direct[t] = np.bincount(labels[region], minlength=n)
            if near_reach is not None:
                distance = cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 3)
                near_overlap[t] = np.bincount(labels[distance <= near_reach], minlength=n)
        direct[:, 0] = 0.0
        near_overlap[:, 0] = 0.0

        def best_track(column: np.ndarray) -> int:
            """Track with the largest overlap; a tie between the subject and a negative goes to the negative."""

            if tracking and len(tracks) > 1 and column[1:].max() >= column[0]:
                return 1 + int(np.argmax(column[1:]))
            return int(np.argmax(column))

        has_direct = np.zeros(len(tracks), dtype=bool)
        merged_ids: Set[int] = set()
        for j in range(1, n):
            column = direct[:, j]
            if not column.any():
                continue
            if tracking and column[0] >= self.CONTAIN_FRACTION * max(self._subject_area, 1.0):
                # This component contains the subject.  If it also covers other people they have
                # merged with the subject and cannot be separated (see the module docstring).
                roles[j] = _SUBJECT
                has_direct[0] = True
                merged_ids.update(tracks[t][0] for t in range(1, len(tracks)) if column[t] > 0)
                continue
            t = best_track(column)
            roles[j] = tracks[t][0]
            has_direct[t] = True

        # Components that only come close to a track: either the track moved fast (it has no
        # overlapping component), or they are fragments, or something new appeared next to it.
        jumped: List[Tuple[float, int]] = []
        for j in range(1, n):
            if roles[j] != _NEW or direct[:, j].any():
                continue
            column = near_overlap[:, j]
            if not column.any():
                continue
            t = best_track(column)
            role = tracks[t][0]
            if has_direct[t]:
                if areas[j] < self._min_area_cells:      # a speck right next to its owner: a fragment
                    roles[j] = _FRAGMENT if role == _SUBJECT else role
            elif role == _SUBJECT:
                jumped.append((float(column[t]), j))
            else:
                roles[j] = role
        if jumped:
            best = max(jumped)[1]
            roles[best] = _SUBJECT
            for _, j in jumped:
                if j != best and areas[j] < self._min_area_cells:
                    roles[j] = _FRAGMENT

        # A negative that has a component of its own this frame is not merged, whatever touched it.
        for k in merged_ids:
            if not (roles == k).any():
                merged.append(self._negatives[k].mask)
                self._negatives[k].last_seen = index     # hidden inside the subject's component, not gone
        return roles, merged

    def _component_distances(self, labels: np.ndarray, n: int, region: np.ndarray) -> np.ndarray:
        """Smallest distance (cells) from every component to ``region``."""

        distance = cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 3)
        nearest = np.full(n, np.inf, dtype=np.float64)
        np.minimum.at(nearest, labels.ravel(), distance.ravel().astype(np.float64))
        return nearest

    def _attach_fragments(self, labels: np.ndarray, n: int, areas: np.ndarray, roles: np.ndarray) -> None:
        """Specks right next to the freshly chosen subject are kept with it (detached strands, fingers)."""

        chosen = np.isin(labels, np.flatnonzero(roles == _SUBJECT))
        nearest = self._component_distances(labels, n, chosen)
        for j in range(1, n):
            if roles[j] == _NEW and areas[j] < self._min_area_cells and nearest[j] <= self._margin:
                roles[j] = _FRAGMENT

    # ------------------------------------------------------------------ acquisition
    def _acquire(
        self, labels: np.ndarray, n: int, areas: np.ndarray, centroids: np.ndarray, roles: np.ndarray, index: int
    ) -> Optional[str]:
        """Choose the subject for the first time.  Returns ``"seed"`` / ``"auto"`` or None."""

        if n <= 1:
            return None
        # Clicks that say "not this one" create negative tracks straight away.
        for x, y in self._seed_negative:
            j = self._label_at(labels, x, y)
            if j > 0 and roles[j] == _NEW:
                roles[j] = len(self._negatives)
                self._negatives.append(_Negative(mask=labels == j, area=float(areas[j]), last_seen=index))

        available = roles == _NEW
        available[0] = False
        if self._seeded:
            chosen: List[int] = []
            if self._seed_region is not None:
                region = self._seed_region
                elapsed = abs(index - (self._first_index if self._first_index is not None else index))
                if elapsed > 0:
                    # The subject may have moved since the reference frame, but a seed is a deliberate
                    # choice: the slack stays local instead of growing to the whole frame.
                    slack = self._margin * min(elapsed, 3)
                    region = cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 3) <= slack
                overlap = np.bincount(labels[region], minlength=n).astype(np.float64)
                overlap[~available] = 0.0
                inside = (overlap > 0.0) & (overlap >= 0.5 * areas)
                chosen = [int(j) for j in np.flatnonzero(inside)]
                if not chosen and overlap.max() > 0.0:
                    chosen = [int(np.argmax(overlap))]
            for x, y in self._seed_positive:
                j = self._label_at(labels, x, y)
                if j > 0 and available[j] and j not in chosen:
                    chosen.append(j)
            if not chosen:
                return None
            roles[chosen] = _SUBJECT
            mode = "seed"
        else:
            centre_x, centre_y = (self._w - 1) / 2.0, (self._h - 1) / 2.0
            best: Optional[Tuple[float, int]] = None
            for j in np.flatnonzero(available):
                if areas[j] < self._min_area_cells:
                    continue
                offset = math.hypot(centroids[j][0] - centre_x, centroids[j][1] - centre_y) / (self._diag / 2.0)
                centrality = 1.0 - 0.75 * min(1.0, offset) ** 2
                score = areas[j] * centrality
                if best is None or score > best[0]:
                    best = (score, int(j))
            if best is None:
                return None
            roles[best[1]] = _SUBJECT
            mode = "auto"
        self._attach_fragments(labels, n, areas, roles)
        return mode

    def _reacquire(
        self, labels: np.ndarray, n: int, areas: np.ndarray, roles: np.ndarray, reach: float, index: int
    ) -> Optional[Tuple[float, int]]:
        """Look for the lost subject.  Returns ``(distance in cells, frames lost)`` when found."""

        if n <= 1 or self._subject is None:
            return None
        lost_frames = abs(index - (self._lost_index if self._lost_index is not None else index))
        radius = reach + (self._diag - reach) * min(1.0, lost_frames / float(self._patience))
        # Big enough to be the subject again: a fraction of its recent size and at least min_area
        # (unless the user deliberately seeded a subject that is smaller than that).
        minimum = max(
            self.REACQUIRE_FRACTION * self._reference_area,
            min(self._min_area_cells, 0.5 * self._reference_area),
        )
        nearest = self._component_distances(labels, n, self._subject)
        best: Optional[Tuple[float, int]] = None
        for j in range(1, n):
            if roles[j] != _NEW or areas[j] < minimum or nearest[j] > radius:
                continue
            # Prefer large components close to where the subject was last seen.
            score = areas[j] / (1.0 + (nearest[j] / (0.25 * self._diag)) ** 2)
            if best is None or score > best[0]:
                best = (score, j)
        if best is None:
            return None
        roles[best[1]] = _SUBJECT
        distance = float(nearest[best[1]])
        self._attach_fragments(labels, n, areas, roles)
        return distance, lost_frames

    def _label_at(self, labels: np.ndarray, x: float, y: float, radius: int = 2) -> int:
        """Component under a click, or the most frequent one within a few cells of it."""

        xi, yi = int(x), int(y)
        j = int(labels[yi, xi])
        if j > 0:
            return j
        window = labels[max(0, yi - radius) : yi + radius + 1, max(0, xi - radius) : xi + radius + 1]
        hits = window[window > 0]
        return int(np.bincount(hits).argmax()) if hits.size else 0

    # ------------------------------------------------------------------ state updates
    def _update_subject(self, kept: np.ndarray, merged_masks: List[np.ndarray]) -> None:
        region = kept
        if merged_masks:
            # While merged with another person, remember the subject without the place where that
            # person was last seen alone: after the split the two regions tell the components apart.
            blocked = np.zeros_like(kept)
            for mask in merged_masks:
                blocked |= mask
            blocked = cv2.dilate(blocked.astype(np.uint8), _KERNEL3).astype(bool)
            trimmed = kept & ~blocked
            if trimmed.sum() >= 0.2 * kept.sum():
                region = trimmed
        self._subject = region
        self._subject_area = float(region.sum())
        self._reference_area = max(self._subject_area, self._reference_area * self.REFERENCE_DECAY)
        self._tracked_frames += 1

    def _update_negatives(self, labels: np.ndarray, roles: np.ndarray, index: int) -> None:
        """Move every negative track to its components of this frame and forget stale tracks."""

        for k, neg in enumerate(self._negatives):
            own = np.flatnonzero(roles == k)
            if own.size:
                neg.mask = np.isin(labels, own)
                neg.area = float(neg.mask.sum())
                neg.last_seen = index
        self._negatives = [neg for neg in self._negatives if abs(index - neg.last_seen) <= self._negative_ttl]

    # ------------------------------------------------------------------ output
    def _keep_map(
        self, labels: np.ndarray, kept: Optional[np.ndarray], occupied: np.ndarray
    ) -> Optional[np.ndarray]:
        """Cell map: 1 = keep, 0 = remove, values between only in gaps and at the rim of the kept zone.

        The kept zone is the subject's components grown by the fringe margin, so the subject keeps
        its soft edge, hair and motion-blur tails, while faint matte elsewhere in the frame (the
        ghost of a bystander that never reached the component threshold) is removed.  Inside the
        zone a cell goes to the nearer of "kept components" and "removed components", so the fringe
        of a removed person leaves with that person.

        Exactness.  Two different components are never 8-neighbours, so a kept component cell only
        has neighbours of value 1 or 0.5 and a removed one only 0 or 0.5.  Bilinear up-sampling
        weights the cell a pixel lies in with at least 0.25, which bounds the up-sampled value by
        0.25 + 0.75 * 0.5 = 0.625 from below inside kept cells and by 0.75 * 0.5 = 0.375 from above
        inside removed cells; :meth:`apply` turns that into exactly 1.0 and exactly 0.0.

        Returns ``None`` when nothing in the frame lies outside the kept zone.
        """

        if kept is None:
            return np.zeros(labels.shape, dtype=np.float32)        # searching / lost: nothing is kept
        removed = (labels > 0) & ~kept
        to_kept = cv2.distanceTransform((~kept).astype(np.uint8), cv2.DIST_L2, 3)
        # Full weight up to the fringe margin, then a two-cell fade instead of a cut.
        keep = np.clip((self._fringe + 2.0 - to_kept) / 2.0, 0.0, 1.0).astype(np.float32)
        if not removed.any():
            if not (occupied & (keep < 1.0)).any():
                return None                                          # only the subject (and its fringe) is there
            return keep
        to_removed = cv2.distanceTransform((~removed).astype(np.uint8), cv2.DIST_L2, 3)
        keep[to_kept > to_removed] = 0.0
        tie = to_kept == to_removed
        keep[tie] = np.minimum(keep[tie], 0.5)
        near_kept = cv2.dilate(kept.astype(np.uint8), _KERNEL3).astype(bool)
        near_removed = cv2.dilate(removed.astype(np.uint8), _KERNEL3).astype(bool)
        keep[near_kept & ~near_removed] = 1.0
        keep[near_removed & ~near_kept] = 0.0
        keep[near_kept & near_removed] = 0.5
        keep[kept] = 1.0
        keep[removed] = 0.0
        return keep

    def _bbox(self, region: Optional[np.ndarray]) -> Optional[List[float]]:
        """Bounding box of a working-map region in full-resolution pixels ``[x1, y1, x2, y2]``."""

        if region is None or not region.any():
            return None
        ys, xs = np.nonzero(region)
        sx, sy = self.width / float(self._w), self.height / float(self._h)
        return [
            round(float(xs.min()) * sx, 1),
            round(float(ys.min()) * sy, 1),
            round(float(xs.max() + 1) * sx, 1),
            round(float(ys.max() + 1) * sy, 1),
        ]
