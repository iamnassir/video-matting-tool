"""Rigid motion of a subject, and its outline as one shape for the whole clip.

A car, a product on a turntable arm, a piece of furniture on a dolly: such a subject does not change
its shape, it only moves in the picture.  Two things follow from that, and both are used to keep an
outline steady where a segmentation model wavers (the subject has the colour of the background) or
is fooled (something similar passes behind it):

* :class:`RigidTracker` measures, frame by frame, the affine motion that takes the reference frame
  to the current one.  It follows corner features on the subject with pyramidal Lucas-Kanade, keeps
  the consistent ones (RANSAC) and, to stop errors adding up over hundreds of frames, measures again
  directly against the reference frame whenever enough of its features are still visible.
* :class:`OutlineMosaic` puts the outlines of all frames on top of each other in the subject's own
  coordinates and takes the majority.  A van that is behind the subject for twenty frames out of
  two hundred is outvoted; a wobble of the edge averages out.  The mosaic is larger than the frame,
  so parts of the subject that enter the picture later are covered too.

Everything here is plain NumPy / OpenCV on the CPU and independent of any model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def scale_affine(matrix: np.ndarray, factor: float) -> np.ndarray:
    """The same motion for pictures *factor* times the size (coordinates multiplied by *factor*)."""

    out = np.array(matrix, dtype=np.float64, copy=True)
    out[:2, 2] *= float(factor)
    return out


class RigidTracker:
    """Affine motion from the reference frame to each following frame.

    ``gray``          the reference frame, ``uint8 [H, W]`` (a reduced size is enough: 540 lines).
    ``feature_mask``  where features may be taken: the subject proper, away from its edge and from
                      anything seen through it.

    :meth:`step` must be called with consecutive frames (either direction in time).  It returns the
    3 x 3 matrix that maps reference coordinates to the frame's coordinates, or ``None`` once the
    subject could not be followed; after that it keeps returning ``None``.
    """

    def __init__(self, gray: np.ndarray, feature_mask: np.ndarray, max_points: int = 700) -> None:
        self._reference = gray
        self._previous = gray
        self._mask = np.asarray(feature_mask, dtype=bool).astype(np.uint8) * 255
        self._max_points = int(max_points)
        self._anchors = self._detect(gray, self._mask, self._max_points)
        self._points = self._anchors.copy()
        self._total = np.eye(3, dtype=np.float64)
        self.initial = int(len(self._anchors))
        self.lost = self.initial < 12
        self.anchored_steps = 0
        self.steps = 0

    @staticmethod
    def _detect(gray: np.ndarray, mask: np.ndarray, count: int, existing: Optional[np.ndarray] = None) -> np.ndarray:
        if count <= 0:
            return np.zeros((0, 1, 2), np.float32)
        if existing is not None and len(existing):
            mask = mask.copy()
            for x, y in existing.reshape(-1, 2):
                cv2.circle(mask, (int(round(float(x))), int(round(float(y)))), 7, 0, -1)
        points = cv2.goodFeaturesToTrack(gray, maxCorners=int(count), qualityLevel=0.01, minDistance=7, mask=mask,
                                         blockSize=5)
        return np.zeros((0, 1, 2), np.float32) if points is None else points.astype(np.float32)

    @staticmethod
    def _flow(a: np.ndarray, b: np.ndarray, points: np.ndarray, guess: Optional[np.ndarray] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
        """Follow *points* from image *a* to *b*; returns (new positions, which ones are trustworthy)."""

        if guess is None:
            moved, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None, **_LK)
        else:
            moved, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, points, guess.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **_LK)
        if moved is None:
            return points, np.zeros(len(points), bool)
        back, ok_back, _ = cv2.calcOpticalFlowPyrLK(b, a, moved, points.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **_LK)
        if back is None:
            return moved, np.zeros(len(points), bool)
        error = np.linalg.norm((back - points).reshape(-1, 2), axis=1)
        return moved, (ok.ravel() == 1) & (ok_back.ravel() == 1) & (error < 1.0)

    @staticmethod
    def _fit(src: np.ndarray, dst: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        if len(src) < 8:
            return None, np.zeros(len(src), bool)
        matrix, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.5,
                                               maxIters=2000, confidence=0.995, refineIters=10)
        if matrix is None:
            return None, np.zeros(len(src), bool)
        return np.vstack([matrix, [0.0, 0.0, 1.0]]), inliers.ravel().astype(bool)

    def step(self, gray: np.ndarray) -> Optional[np.ndarray]:
        if self.lost:
            return None
        self.steps += 1
        # 1. from the previous frame (small motion, reliable)
        moved, good = self._flow(self._previous, gray, self._points)
        chained, inliers = self._fit(self._points[good], moved[good])
        if chained is None or int(inliers.sum()) < 12 or float(inliers.mean()) < 0.4:
            self.lost = True
            return None
        total = chained @ self._total
        points = moved[good][inliers].reshape(-1, 1, 2)
        # 2. directly from the reference frame, starting from that estimate: no accumulated error
        if len(self._anchors) >= 12:
            ones = np.ones((len(self._anchors), 1), np.float32)
            guess = (np.hstack([self._anchors.reshape(-1, 2), ones]) @ total[:2].T.astype(np.float32)).reshape(-1, 1, 2)
            moved_a, good_a = self._flow(self._reference, gray, self._anchors, guess.astype(np.float32))
            direct, inliers_a = self._fit(self._anchors[good_a], moved_a[good_a])
            if direct is not None and int(inliers_a.sum()) >= max(20, self.initial // 4):
                total = direct
                self.anchored_steps += 1
        self._total = total
        self._previous = gray
        if len(points) < max(60, self.initial // 2):
            height, width = gray.shape
            where = cv2.warpAffine(self._mask, total[:2], (width, height), flags=cv2.INTER_NEAREST)
            fresh = self._detect(gray, where, self._max_points - len(points), points)
            points = np.concatenate([points, fresh], axis=0)
        self._points = points
        return total.copy()


@dataclass
class MosaicResult:
    mosaic: np.ndarray                    # float32 [H, W] in 0..1 on the canvas
    origin: Tuple[float, float]           # canvas position of the reference frame's top-left corner
    agreement: float                      # median overlap (IoU) of the carried outline with the frames' own
    frames_used: int


class OutlineMosaic:
    """The outline of a rigid subject as one shape, voted over all frames.

    ``size``     (height, width) of the frames.
    ``scale``    the votes are collected at this fraction of the frame size.
    """

    def __init__(self, size: Tuple[int, int], scale: float = 0.5, max_growth: float = 6.0) -> None:
        self.height, self.width = int(size[0]), int(size[1])
        self.scale = float(scale)
        self.max_growth = float(max_growth)

    def _canvas(self, motions: Sequence[np.ndarray]) -> Optional[Tuple[float, float, int, int]]:
        corners = np.array([[0, 0, 1], [self.width, 0, 1], [0, self.height, 1], [self.width, self.height, 1]], np.float64)
        low = np.array([0.0, 0.0])
        high = np.array([float(self.width), float(self.height)])
        for motion in motions:
            try:
                inverse = np.linalg.inv(motion)
            except np.linalg.LinAlgError:
                return None
            mapped = corners @ inverse[:2].T
            low = np.minimum(low, mapped.min(axis=0))
            high = np.maximum(high, mapped.max(axis=0))
        origin_x, origin_y = float(np.ceil(-low[0])), float(np.ceil(-low[1]))
        canvas_w = int(np.ceil(high[0] + origin_x)) + 1
        canvas_h = int(np.ceil(high[1] + origin_y)) + 1
        if canvas_w * canvas_h > self.max_growth * self.width * self.height:
            return None                                    # the subject wandered too far for one canvas
        return origin_x, origin_y, canvas_w, canvas_h

    def _to_canvas(self, motion: np.ndarray, origin: Tuple[float, float]) -> np.ndarray:
        """Frame coordinates (at vote scale) -> canvas coordinates (at vote scale)."""

        shift = np.array([[1.0, 0.0, origin[0]], [0.0, 1.0, origin[1]], [0.0, 0.0, 1.0]])
        return scale_affine(shift @ np.linalg.inv(motion), self.scale)

    def build(self, outlines: Iterable[np.ndarray], motions: Sequence[np.ndarray],
              reference_outline: Optional[np.ndarray] = None) -> Optional[MosaicResult]:
        """Vote.  The i-th outline (bool ``[H, W]``) belongs to ``motions[i]`` (reference -> frame).

        ``outlines`` may be a generator: each outline is reduced to the vote size at once, so a
        long clip never has all its full-size outlines in memory.

        ``reference_outline`` is the outline given for the reference frame itself: inside the
        reference frame's rectangle it replaces the vote (it is what the user asked for).  Towards
        the border of that rectangle it gives way to the vote gradually, so that the two meet
        without a step where the subject continues beyond the reference frame.
        """

        if not len(motions):
            return None
        canvas = self._canvas(motions)
        if canvas is None:
            return None
        origin_x, origin_y, canvas_w, canvas_h = canvas
        origin = (origin_x, origin_y)
        small = (max(1, int(round(canvas_w * self.scale))), max(1, int(round(canvas_h * self.scale))))
        frame_small = (max(1, int(round(self.width * self.scale))), max(1, int(round(self.height * self.scale))))
        votes = np.zeros((small[1], small[0]), np.float32)
        seen = np.zeros_like(votes)
        ones = np.ones((frame_small[1], frame_small[0]), np.float32)
        # The agreement is judged on a sample of at most about 60 frames.
        every = max(1, len(motions) // 60)
        sample: List[Tuple[np.ndarray, np.ndarray]] = []
        used = 0
        for number, (outline, motion) in enumerate(zip(outlines, motions)):
            picture = cv2.resize(np.asarray(outline, dtype=np.float32), frame_small, interpolation=cv2.INTER_AREA)
            if number % every == 0:
                sample.append((picture > 0.5, motion))
            matrix = self._to_canvas(motion, origin)[:2]
            votes += cv2.warpAffine(picture, matrix, small, flags=cv2.INTER_LINEAR)
            seen += cv2.warpAffine(ones, matrix, small, flags=cv2.INTER_LINEAR)
            used += 1
        if not used:
            return None
        share = np.where(seen > 0.5, votes / np.maximum(seen, 1e-6), 0.0).astype(np.float32)
        mosaic = cv2.resize(share, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)
        if reference_outline is not None:
            x0, y0 = int(origin_x), int(origin_y)
            band = max(2.0, 0.04 * min(self.width, self.height))
            xs = np.minimum(np.arange(self.width), self.width - 1 - np.arange(self.width)).astype(np.float32)
            ys = np.minimum(np.arange(self.height), self.height - 1 - np.arange(self.height)).astype(np.float32)
            weight = np.clip(np.minimum(ys[:, None], xs[None, :]) / band, 0.0, 1.0)
            inside = mosaic[y0:y0 + self.height, x0:x0 + self.width]
            given = np.asarray(reference_outline, dtype=np.float32)
            mosaic[y0:y0 + self.height, x0:x0 + self.width] = weight * given + (1.0 - weight) * inside
        mosaic = cv2.GaussianBlur(mosaic, (0, 0), 1.5)
        result = MosaicResult(mosaic=mosaic, origin=origin, agreement=0.0, frames_used=used)
        # How well does the one shape explain the single outlines?
        overlaps = []
        small_mosaic = cv2.resize(mosaic, small, interpolation=cv2.INTER_AREA)
        for own, motion in sample:
            matrix = np.linalg.inv(self._to_canvas(motion, origin))[:2]
            carried = cv2.warpAffine(small_mosaic, matrix, frame_small, flags=cv2.INTER_LINEAR) > 0.5
            union = int((carried | own).sum())
            overlaps.append(float((carried & own).sum()) / union if union else 1.0)
        result.agreement = float(np.median(overlaps))
        return result

    def carry(self, result: MosaicResult, motion: np.ndarray) -> np.ndarray:
        """The outline on one frame: the mosaic moved with the subject (bool ``[H, W]``)."""

        shift = np.array([[1.0, 0.0, -result.origin[0]], [0.0, 1.0, -result.origin[1]], [0.0, 0.0, 1.0]])
        matrix = (motion @ shift)[:2]
        return cv2.warpAffine(result.mosaic, matrix, (self.width, self.height), flags=cv2.INTER_LINEAR) > 0.5
