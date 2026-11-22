"""Live preview channel between a running pipeline and its observers (the web UI).

The render loop offers its most recent preview atlas (original / subject overlay / matte /
composite) at a limited rate; observers fetch the latest JPEG.  Nothing here is simulated: an image
only exists once a frame has really been processed.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class LivePreview:
    def __init__(self, min_interval: float = 0.5, jpeg_quality: int = 82) -> None:
        self.min_interval = float(min_interval)
        self.jpeg_quality = int(jpeg_quality)
        self._lock = threading.Lock()
        self._last_time = 0.0
        self._jpeg: Optional[bytes] = None
        self._frame_index = -1
        self._tile: Tuple[int, int] = (0, 0)
        self._version = 0

    def wants_frame(self) -> bool:
        """True when enough time has passed since the last accepted frame (cheap, lock-free read)."""

        return (time.perf_counter() - self._last_time) >= self.min_interval

    def update(self, atlas_rgb: np.ndarray, frame_index: int) -> None:
        """Publish a new atlas (``uint8 [2*th, 2*tw, 3]`` RGB) for frame *frame_index*."""

        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(np.ascontiguousarray(atlas_rgb), cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()
            self._frame_index = int(frame_index)
            self._tile = (atlas_rgb.shape[1] // 2, atlas_rgb.shape[0] // 2)
            self._last_time = time.perf_counter()
            self._version += 1

    def latest(self) -> Optional[Tuple[bytes, int, int]]:
        """``(jpeg bytes, frame index, version)`` of the newest preview, or None before the first frame."""

        with self._lock:
            if self._jpeg is None:
                return None
            return self._jpeg, self._frame_index, self._version

    @property
    def tile_size(self) -> Tuple[int, int]:
        """``(tile width, tile height)`` of the atlas quadrants (0, 0 before the first frame)."""

        with self._lock:
            return self._tile

    def clear(self) -> None:
        with self._lock:
            self._jpeg = None
            self._frame_index = -1
            self._version += 1
