"""Tracking health monitor: cheap per-frame statistics of the final matte for the processing report.

The pipeline shows this module every finished matte.  From a small copy of each frame it records
how much of the picture the subject covers, where it is (centroid, bounding box) and, when the
model reports it, how confident the model is that the subject is present.  :meth:`TrackingMonitor.report`
condenses that into numbers and into plain sentences a user can act on - "Subject not visible in
frames 120-148 (1.0 s)" - so that a matte that silently went empty or jumped to somebody else does
not go unnoticed in a long render.

Only eight numbers per frame are kept, in compact arrays (about 40 bytes per frame): the whole
video is never held in memory, and a film-length render costs a few megabytes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..core.errors import ConfigError
from ..utils import torch_ops
from .component_tracker import TrackingError

__all__ = ["TrackingMonitor"]

# Columns of a frame record.
_COVERAGE, _CX, _CY, _X1, _Y1, _X2, _Y2, _PRESENCE = range(8)


class TrackingMonitor:
    """Collects coverage / position statistics of the final matte and reports tracking problems.

    Parameters
    ----------
    width, height : frame size in display orientation.
    fps           : nominal frame rate, used only to express frame counts as seconds in the warnings
                    (the footage may be variable frame rate; the durations are approximate then).
    switch_distance : centroid jump, as a fraction of the frame diagonal, from which a jump between
                    two consecutive visible frames with disjoint bounding boxes is reported as a
                    suspected subject switch.
    """

    #: Long side of the copy the statistics are computed on.
    WORK_SIDE = 160
    #: Coverage (fraction of the frame area) below which a frame counts as empty: no subject visible.
    EMPTY_COVERAGE = 1e-4
    #: Coverage above which the matte covers practically the whole frame (nothing was separated).
    FULL_COVERAGE = 0.995
    #: Matte value from which a cell of the small copy belongs to the bounding box.
    BOX_THRESHOLD = 0.25
    #: Presence reported by the model below this value counts as "low".
    LOW_PRESENCE = 0.5
    #: At most this many sentences per kind of problem are listed individually.
    MAX_LISTED = 12

    def __init__(self, width: int, height: int, fps: float, *, switch_distance: float = 0.5) -> None:
        if int(width) < 1 or int(height) < 1:
            raise ConfigError(f"Invalid frame size {width}x{height} for the tracking monitor.")
        if not (0.0 < float(switch_distance) <= 1.5):
            raise ConfigError(f"switch_distance is a fraction of the frame diagonal, got {switch_distance!r}.")
        self.width = int(width)
        self.height = int(height)
        try:
            fps_value = float(fps)
        except (TypeError, ValueError):
            fps_value = 0.0
        self.fps = fps_value if math.isfinite(fps_value) and fps_value > 0.0 else 0.0
        self.switch_distance = float(switch_distance)
        self._w, self._h = torch_ops.fit_size(self.width, self.height, self.WORK_SIDE)
        # Records in arrival order, one chunk per update() call: frame indices and [n, 8] float32 rows
        # (coverage, centroid x / y, box x1 / y1 / x2 / y2 in full-resolution pixels, presence or nan).
        self._chunks: List[Tuple[np.ndarray, np.ndarray]] = []
        self._presence_reported = False

    # ------------------------------------------------------------------ collection
    def update(
        self,
        alpha: torch.Tensor,
        frame_indices: Sequence[int],
        presence: Optional[Sequence[float]] = None,
    ) -> None:
        """Record a batch of final mattes.

        ``alpha`` ``[B, 1, H, W]`` in 0..1 on any device (not modified), ``frame_indices`` their frame
        numbers, ``presence`` the model's per-frame confidence that the subject is present (or None
        when the model cannot tell).  A frame passed twice replaces its earlier record.
        """

        if not isinstance(alpha, torch.Tensor) or alpha.dim() != 4 or alpha.shape[1] != 1:
            got = tuple(alpha.shape) if isinstance(alpha, torch.Tensor) else type(alpha).__name__
            raise TrackingError(f"update() expects alpha of shape [B, 1, H, W], got {got}.")
        if tuple(alpha.shape[-2:]) != (self.height, self.width):
            raise TrackingError(
                f"The matte is {alpha.shape[-1]}x{alpha.shape[-2]} but the monitor was created for "
                f"{self.width}x{self.height}."
            )
        batch = alpha.shape[0]
        if len(frame_indices) != batch:
            raise TrackingError(f"{len(frame_indices)} frame indices were given for a batch of {batch} frames.")
        if presence is not None and len(presence) != batch:
            raise TrackingError(f"{len(presence)} presence values were given for a batch of {batch} frames.")
        if batch == 0:
            return

        low = torch_ops.resize(alpha.detach().to(torch.float32), (self._h, self._w), mode="area")
        cells = np.nan_to_num(low[:, 0].cpu().numpy(), nan=0.0).clip(0.0, 1.0)
        sx, sy = self.width / float(self._w), self.height / float(self._h)
        xs = (np.arange(self._w, dtype=np.float64) + 0.5) * sx
        ys = (np.arange(self._h, dtype=np.float64) + 0.5) * sy

        rows_out = np.full((batch, 8), np.nan, dtype=np.float32)
        for i in range(batch):
            frame = cells[i].astype(np.float64)
            total = float(frame.sum())
            rows_out[i, _COVERAGE] = total / frame.size
            if total > 0.0:
                rows_out[i, _CX] = (frame.sum(axis=0) * xs).sum() / total
                rows_out[i, _CY] = (frame.sum(axis=1) * ys).sum() / total
                solid = frame > min(self.BOX_THRESHOLD, 0.5 * float(frame.max()))
                cols = np.flatnonzero(solid.any(axis=0))
                rows = np.flatnonzero(solid.any(axis=1))
                rows_out[i, _X1:] = (cols[0] * sx, rows[0] * sy, (cols[-1] + 1) * sx, (rows[-1] + 1) * sy, np.nan)
            if presence is not None and presence[i] is not None:
                rows_out[i, _PRESENCE] = float(presence[i])
                self._presence_reported = True
        self._chunks.append((np.asarray([int(i) for i in frame_indices], dtype=np.int64), rows_out))
        if len(self._chunks) > 512:
            self._compact()

    # ------------------------------------------------------------------ report
    def report(self) -> Dict[str, Any]:
        """Summary of everything seen so far (JSON-serialisable).

        Keys: ``frames``, ``coverage_mean`` / ``coverage_min`` / ``coverage_max`` (fraction of the
        frame area), ``empty_frames``, ``absent_segments`` (``[start, end]`` frame index pairs,
        inclusive, where no subject is visible), ``suspected_switches`` (frame indices),
        ``low_presence_frames`` (count, or None when the model does not report presence) with
        ``low_presence_segments``, ``full_frames`` (matte covers the whole picture) and ``warnings``
        (sentences for the processing report).
        """

        index_array, records = self._compact()
        indices = [int(i) for i in index_array]
        frames = len(indices)
        out: Dict[str, Any] = {
            "frames": frames,
            "coverage_mean": 0.0,
            "coverage_min": 0.0,
            "coverage_max": 0.0,
            "empty_frames": 0,
            "absent_segments": [],
            "suspected_switches": [],
            "low_presence_frames": None,
            "low_presence_segments": [],
            "full_frames": 0,
            "warnings": [],
        }
        if frames == 0:
            return out

        coverage = records[:, _COVERAGE].astype(np.float64)
        empty = coverage < self.EMPTY_COVERAGE
        out["coverage_mean"] = float(coverage.mean())
        out["coverage_min"] = float(coverage.min())
        out["coverage_max"] = float(coverage.max())
        out["empty_frames"] = int(empty.sum())
        out["full_frames"] = int((coverage > self.FULL_COVERAGE).sum())
        absent = self._segments(indices, empty)
        out["absent_segments"] = absent

        # Suspected switches: the matte is somewhere else entirely than in the previous visible frame.
        diagonal = math.hypot(self.width, self.height)
        switches: List[int] = []
        jumps: Dict[int, float] = {}
        previous: Optional[np.ndarray] = None
        for row, index in enumerate(indices):
            if empty[row]:
                continue
            record = records[row]
            if previous is not None:
                distance = math.hypot(record[_CX] - previous[_CX], record[_CY] - previous[_CY]) / diagonal
                disjoint = (
                    record[_X1] >= previous[_X2]
                    or previous[_X1] >= record[_X2]
                    or record[_Y1] >= previous[_Y2]
                    or previous[_Y1] >= record[_Y2]
                )
                if distance > self.switch_distance and disjoint:
                    switches.append(index)
                    jumps[index] = float(distance)
            previous = record
        out["suspected_switches"] = switches

        low_segments: List[List[int]] = []
        if self._presence_reported:
            low = np.nan_to_num(records[:, _PRESENCE].astype(np.float64), nan=1.0) < self.LOW_PRESENCE
            out["low_presence_frames"] = int(low.sum())
            low_segments = self._segments(indices, low)
            out["low_presence_segments"] = low_segments

        # ---------------------------------------------------------------- warnings
        warnings: List[str] = []
        if out["empty_frames"] == frames:
            warnings.append(
                f"No subject was found in any of the {frames} frames{self._seconds(frames)}: "
                f"the output contains only the background."
            )
        else:
            for start, end in absent[: self.MAX_LISTED]:
                warnings.append(f"Subject not visible in {self._span(start, end)}.")
            if len(absent) > self.MAX_LISTED:
                rest = absent[self.MAX_LISTED :]
                frames_rest = sum(e - s + 1 for s, e in rest)
                warnings.append(
                    f"Subject not visible in {len(rest)} more segments "
                    f"({frames_rest} frames{self._seconds(frames_rest, bare=True)})."
                )
        for index in switches[: self.MAX_LISTED]:
            warnings.append(
                f"Possible subject switch at frame {index}: the matte jumped across {jumps[index] * 100.0:.0f} % "
                f"of the frame diagonal to a place that does not overlap its previous position."
            )
        if len(switches) > self.MAX_LISTED:
            warnings.append(f"{len(switches) - self.MAX_LISTED} more possible subject switches were detected.")
        if out["low_presence_frames"]:
            count = int(out["low_presence_frames"])
            first = low_segments[0]
            where = self._span(first[0], first[1], seconds=False)
            warnings.append(
                f"The model reported low confidence that the subject is present in {count} "
                f"frame{'s' if count != 1 else ''}{self._seconds(count)}, first in {where}."
            )
        if out["full_frames"]:
            count = int(out["full_frames"])
            warnings.append(
                f"The matte covers the whole picture in {count} frame{'s' if count != 1 else ''}"
                f"{self._seconds(count)}: nothing was separated from the background there."
            )
        out["warnings"] = warnings
        return out

    # ------------------------------------------------------------------ helpers
    def _compact(self) -> Tuple[np.ndarray, np.ndarray]:
        """Merge all chunks into one, sorted by frame index; a frame seen twice keeps its last record."""

        if not self._chunks:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 8), dtype=np.float32)
        indices = np.concatenate([c[0] for c in self._chunks])
        records = np.concatenate([c[1] for c in self._chunks])
        order = np.argsort(indices, kind="stable")              # stable: later updates stay behind earlier ones
        indices, records = indices[order], records[order]
        last = np.ones(indices.shape[0], dtype=bool)
        last[:-1] = indices[1:] != indices[:-1]
        indices, records = indices[last], records[last]
        self._chunks = [(indices, records)]
        return indices, records

    @staticmethod
    def _segments(indices: Sequence[int], flags: Sequence[bool]) -> List[List[int]]:
        """Maximal runs of flagged frames with consecutive indices, as inclusive ``[start, end]`` pairs."""

        segments: List[List[int]] = []
        start: Optional[int] = None
        last = 0
        for index, flag in zip(indices, flags):
            if flag and start is not None and index == last + 1:
                last = index
                continue
            if start is not None:
                segments.append([start, last])
                start = None
            if flag:
                start = last = index
        if start is not None:
            segments.append([start, last])
        return segments

    def _seconds(self, frame_count: int, bare: bool = False) -> str:
        """`` (1.0 s)`` for a number of frames, or an empty string when the frame rate is unknown."""

        if self.fps <= 0.0:
            return ""
        seconds = frame_count / self.fps
        text = f"{seconds:.1f} s" if seconds >= 0.95 else f"{seconds:.2f} s"
        return f", {text}" if bare else f" ({text})"

    def _span(self, start: int, end: int, seconds: bool = True) -> str:
        count = end - start + 1
        text = f"frame {start}" if count == 1 else f"frames {start}-{end}"
        return text + (self._seconds(count) if seconds else "")
