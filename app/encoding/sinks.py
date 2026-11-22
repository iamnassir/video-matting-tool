"""Deliverables that are not video files: the RGBA PNG image sequence.

PNG compression is the slow part (tens of milliseconds per 1080p frame), so frames are encoded by a
small pool of worker threads fed through a bounded queue: the render loop only blocks when the
workers cannot keep up, and memory stays bounded.  ``cv2.imencode`` releases the GIL, so the pool
scales with cores.  Files are written with plain Python I/O because ``cv2.imwrite`` cannot open
paths with non-ASCII characters on Windows.
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np

from ..core.errors import DiskSpaceError, StorageError
from ..core.events import CancelToken

log = logging.getLogger(__name__)

__all__ = ["PngSequenceWriter"]

_END = object()


class PngSequenceWriter:
    """Writes RGBA frames as ``frame_000000.png``, ``frame_000001.png`` ... into a folder.

    ``write(rgba, index)`` takes ``[H, W, 4]`` arrays in R, G, B, A order, ``uint8`` for 8-bit PNGs or
    ``uint16`` for 16-bit PNGs (the alpha association - straight or premultiplied - is whatever the
    caller rendered).  Errors in a worker surface from the next ``write`` or from ``finish``.
    """

    def __init__(
        self,
        directory: Path,
        *,
        bit_depth: int = 8,
        compression: int = 3,
        workers: int = 4,
        pattern: str = "frame_{index:06d}.png",
        cancel: Optional[CancelToken] = None,
    ) -> None:
        if bit_depth not in (8, 16):
            raise ValueError("bit_depth must be 8 or 16")
        self.directory = Path(directory)
        self.bit_depth = int(bit_depth)
        self.dtype = np.dtype(np.uint8) if bit_depth == 8 else np.dtype(np.uint16)
        self.compression = max(0, min(9, int(compression)))
        self.pattern = pattern
        self.cancel = cancel
        self._workers = max(1, int(workers))
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=self._workers * 2)
        self._threads: List[threading.Thread] = []
        self._error: Optional[BaseException] = None
        self._stop = threading.Event()
        self._started = False
        self._closed = False
        self._lock = threading.Lock()
        self.frames_written = 0

    def path_for(self, index: int) -> Path:
        return self.directory / self.pattern.format(index=int(index))

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError(f"Cannot create the PNG sequence folder {self.directory}: {exc}") from exc
        for number in range(self._workers):
            thread = threading.Thread(target=self._run, name=f"vmt-png-{number}", daemon=True)
            thread.start()
            self._threads.append(thread)

    # ------------------------------------------------------------------------------ workers
    def _run(self) -> None:
        params = [cv2.IMWRITE_PNG_COMPRESSION, self.compression]
        while True:
            item = self._queue.get()
            try:
                if item is _END:
                    return
                if self._stop.is_set() or self._error is not None:
                    continue
                bgra, index = item
                ok, buffer = cv2.imencode(".png", bgra, params)
                if not ok:
                    raise StorageError(f"PNG encoding of frame {index} failed.")
                path = self.path_for(index)
                try:
                    with open(path, "wb") as handle:
                        handle.write(buffer.tobytes())
                except OSError as exc:
                    if getattr(exc, "errno", None) == 28 or "space" in str(exc).lower():
                        raise DiskSpaceError(f"The disk is full: could not write {path}.") from exc
                    raise StorageError(f"Could not write {path}: {exc}") from exc
                with self._lock:
                    self.frames_written += 1
            except BaseException as exc:  # noqa: BLE001 - reported to the producer
                if self._error is None:
                    self._error = exc
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------------------ producer side
    def write(self, rgba: np.ndarray, index: int) -> None:
        """Queue one frame. The pixel data is copied, so the caller may reuse its array."""

        if not self._started or self._closed:
            raise RuntimeError("PngSequenceWriter.write() needs start() first and no finish() / abort() before")
        if self._error is not None:
            raise self._error
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError(f"expected an [H, W, 4] RGBA array, got shape {rgba.shape}")
        if rgba.dtype != self.dtype:
            raise ValueError(f"expected {self.dtype} samples for {self.bit_depth}-bit PNGs, got {rgba.dtype}")
        bgra = np.ascontiguousarray(rgba[..., [2, 1, 0, 3]])       # OpenCV's channel order; this is the copy
        while True:
            if self._error is not None:
                raise self._error
            if self.cancel is not None:
                self.cancel.raise_if_cancelled()
            if self._stop.is_set():
                raise RuntimeError("the PNG writer was aborted")
            try:
                self._queue.put((bgra, int(index)), timeout=0.2)
                return
            except queue.Full:
                continue

    def finish(self) -> None:
        """Wait until every queued frame is on disk; raise the first worker error, if any."""

        if self._closed:
            if self._error is not None:
                raise self._error
            return
        self._closed = True
        if self._started:
            for _ in self._threads:
                self._queue.put(_END)
            for thread in self._threads:
                thread.join()
        if self._error is not None:
            raise self._error

    def abort(self) -> None:
        """Stop the workers without finishing the queue (files already written are left in place)."""

        self._stop.set()
        if self._closed:
            return
        self._closed = True
        if not self._started:
            return
        try:
            while True:
                self._queue.get_nowait()
                self._queue.task_done()
        except queue.Empty:
            pass
        for _ in self._threads:
            self._queue.put(_END)
        for thread in self._threads:
            thread.join(timeout=30)
