"""Video library: the uploaded / registered source videos of the web interface.

Each video lives in ``<data_dir>/videos/<id>/``:

* ``video.json``  - the record (file name, location, probed metadata)
* ``source.<ext>`` - the uploaded file (absent when a local path was registered instead of uploaded)
* ``proxy.mp4``   - a small browser-playable preview (H.264/AAC), created in the background

Uploads are streamed straight to that folder in chunks; nothing is buffered in memory or in the
system temp directory.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import cv2
import numpy as np

from ..core.config import SUPPORTED_INPUT_EXTENSIONS, AppSettings
from ..core.errors import (
    ConfigError,
    DiskSpaceError,
    InvalidPathError,
    JobNotFoundError,
    StorageError,
    UnsupportedFormatError,
    VmtError,
)
from ..core.types import ProbeResult
from ..utils.fs import atomic_write_text, clean_path_text, free_bytes, safe_filename, safe_rmtree, safe_unlink

log = logging.getLogger(__name__)


@dataclass
class VideoRecord:
    id: str
    filename: str
    path: str                              # where the source file is
    size_bytes: int
    imported: bool = False                 # True = registered by path (the file is not ours to delete)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    proxy: str = "pending"                 # pending | ready | failed | unavailable
    proxy_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "imported": self.imported,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "proxy": self.proxy,
            "proxy_error": self.proxy_error,
            "preview_url": f"/api/videos/{self.id}/preview" if self.proxy == "ready" else None,
            "frame_url": f"/api/videos/{self.id}/frame",
        }


class VideoLibrary:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.root = settings.data_dir / "videos"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._records: Dict[str, VideoRecord] = {}
        self._probes: Dict[str, ProbeResult] = {}
        self._readers: Dict[str, Any] = {}          # id -> (VideoReader, lock) for frame extraction
        self._load()

    # ------------------------------------------------------------------------------ persistence
    def _dir(self, video_id: str) -> Path:
        return self.root / video_id

    def _save(self, record: VideoRecord) -> None:
        data = {k: v for k, v in record.to_dict().items() if k not in ("preview_url", "frame_url")}
        atomic_write_text(self._dir(record.id) / "video.json", json.dumps(data, default=str))

    def _load(self) -> None:
        for entry in sorted(self.root.iterdir()):
            meta = entry / "video.json"
            if not meta.is_file():
                continue
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                record = VideoRecord(**{k: data[k] for k in VideoRecord.__dataclass_fields__ if k in data})
                if not Path(record.path).is_file():
                    log.warning("Library video %s: source file %s is missing.", record.id, record.path)
                if record.proxy == "pending":
                    record.proxy = "ready" if (entry / "proxy.mp4").is_file() else "unavailable"
                self._records[record.id] = record
            except Exception as exc:
                log.warning("Ignoring unreadable library entry %s: %s", entry.name, exc)

    # ------------------------------------------------------------------------------ queries
    def list(self) -> List[VideoRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)

    def get(self, video_id: str) -> VideoRecord:
        with self._lock:
            record = self._records.get(video_id)
        if record is None:
            raise JobNotFoundError(f"Video '{video_id}' does not exist.")
        return record

    def probe(self, video_id: str) -> ProbeResult:
        """Probe result of a library video (cached in memory)."""

        record = self.get(video_id)
        with self._lock:
            cached = self._probes.get(video_id)
        if cached is not None:
            return cached
        from ..encoding.ffmpeg import find_ffmpeg
        from ..processing.probe import probe

        result = probe(Path(record.path), ffprobe=find_ffmpeg(self.settings).ffprobe)
        with self._lock:
            self._probes[video_id] = result
        return result

    # ------------------------------------------------------------------------------ adding
    def _register(self, record: VideoRecord) -> VideoRecord:
        """Probe a new file, store its record and start building the browser proxy."""

        try:
            result = self.probe_path(Path(record.path))
        except VmtError:
            self._discard(record)
            raise
        record.metadata = result.metadata.to_dict()
        with self._lock:
            self._records[record.id] = record
            self._probes[record.id] = result
        self._save(record)
        threading.Thread(target=self._build_proxy, args=(record.id,), name="vmt-proxy", daemon=True).start()
        return record

    def probe_path(self, path: Path) -> ProbeResult:
        from ..encoding.ffmpeg import find_ffmpeg
        from ..processing.probe import probe

        return probe(path, ffprobe=find_ffmpeg(self.settings).ffprobe)

    def _discard(self, record: VideoRecord) -> None:
        safe_rmtree(self._dir(record.id))

    def _check_name(self, filename: str) -> str:
        name = safe_filename(filename or "video", default="video")
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_INPUT_EXTENSIONS:
            raise UnsupportedFormatError(
                f"'{name}' is not a supported video file (extension '{suffix or 'none'}').",
            )
        return name

    async def add_upload(self, chunks: AsyncIterator[bytes], filename: str, expected_size: Optional[int] = None) -> VideoRecord:
        """Store an upload by streaming it to disk, then register it."""

        name = self._check_name(filename)
        limit = self.settings.max_upload_bytes
        if expected_size is not None:
            if limit is not None and expected_size > limit:
                raise ConfigError(f"The upload is larger than the configured limit of {limit} bytes.")
            free = free_bytes(self.root)
            if expected_size + 256 * 1024 * 1024 > free:
                raise DiskSpaceError(
                    f"Not enough free disk space for the upload: {expected_size / 1e9:.2f} GB needed, "
                    f"{free / 1e9:.2f} GB free in {self.root}.",
                    hint="Free up space, or register the file by its local path instead of uploading it.",
                )
        video_id = uuid.uuid4().hex[:12]
        folder = self._dir(video_id)
        folder.mkdir(parents=True, exist_ok=False)
        target = folder / f"source{Path(name).suffix.lower()}"
        written = 0
        try:
            with open(target, "wb") as fh:
                async for chunk in chunks:
                    if not chunk:
                        continue
                    written += len(chunk)
                    if limit is not None and written > limit:
                        raise ConfigError(f"The upload exceeds the configured limit of {limit} bytes.")
                    fh.write(chunk)
        except OSError as exc:
            safe_rmtree(folder)
            raise StorageError(f"Could not store the upload: {exc}",
                               hint="Check free disk space in the data directory.") from exc
        except BaseException:
            safe_rmtree(folder)
            raise
        if written == 0:
            safe_rmtree(folder)
            raise ConfigError("The upload was empty.")
        if expected_size is not None and written != expected_size:
            safe_rmtree(folder)
            raise ConfigError(f"The upload was interrupted: received {written} of {expected_size} bytes.")
        record = VideoRecord(id=video_id, filename=name, path=str(target), size_bytes=written)
        return self._register(record)

    def add_path(self, path: str) -> VideoRecord:
        """Register a file that already exists on this computer (no copy)."""

        p = Path(clean_path_text(path)).expanduser()
        if not p.is_file():
            raise InvalidPathError(f"File not found: {p}")
        name = self._check_name(p.name)
        video_id = uuid.uuid4().hex[:12]
        self._dir(video_id).mkdir(parents=True, exist_ok=False)
        record = VideoRecord(id=video_id, filename=name, path=str(p.resolve()), size_bytes=p.stat().st_size, imported=True)
        return self._register(record)

    # ------------------------------------------------------------------------------ proxy / frames
    def proxy_path(self, video_id: str) -> Path:
        return self._dir(video_id) / "proxy.mp4"

    def _build_proxy(self, video_id: str) -> None:
        try:
            record = self.get(video_id)
            from ..encoding.ffmpeg import find_ffmpeg
            from ..encoding.muxer import make_proxy

            make_proxy(find_ffmpeg(self.settings), Path(record.path), self.proxy_path(video_id), max_side=960)
            record.proxy = "ready"
        except Exception as exc:
            log.warning("Could not create the browser preview for %s: %s", video_id, exc)
            try:
                record = self.get(video_id)
            except VmtError:
                return
            record.proxy = "failed"
            record.proxy_error = str(exc)
        with self._lock:
            if video_id in self._records:
                self._save(record)

    def frame_jpeg(self, video_id: str, index: int, max_side: Optional[int] = 1280, quality: int = 88) -> bytes:
        """JPEG of frame *index* in display orientation (for the subject-selection canvas)."""

        frame = self.frame_rgb(video_id, index, max_side)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise StorageError("Could not encode the frame preview.")
        return buf.tobytes()

    def frame_rgb(self, video_id: str, index: int, max_side: Optional[int] = None) -> np.ndarray:
        """Display-oriented RGB frame (``uint8 [H, W, 3]``) of a library video."""

        record = self.get(video_id)
        result = self.probe(video_id)
        count = result.metadata.frame_count
        if index < 0 or index >= count:
            raise ConfigError(f"Frame {index} does not exist (the video has {count} frames).")
        with self._lock:
            entry = self._readers.get(video_id)
            if entry is None:
                from ..processing.reader import VideoReader

                entry = (VideoReader(Path(record.path), result), threading.Lock())
                self._readers[video_id] = entry
                while len(self._readers) > 3:              # keep only a few decoders open
                    old_id = next(iter(self._readers))
                    if old_id == video_id:
                        break
                    old_reader, _ = self._readers.pop(old_id)
                    try:
                        old_reader.close()
                    except Exception:
                        pass
        reader, lock = entry
        with lock:
            return reader.frame_rgb(index, max_side=max_side)

    def release_readers(self) -> None:
        with self._lock:
            readers = list(self._readers.values())
            self._readers.clear()
        for reader, lock in readers:
            with lock:
                try:
                    reader.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------------------ removal
    def delete(self, video_id: str) -> None:
        record = self.get(video_id)
        with self._lock:
            entry = self._readers.pop(video_id, None)
            self._records.pop(video_id, None)
            self._probes.pop(video_id, None)
        if entry is not None:
            reader, lock = entry
            with lock:
                try:
                    reader.close()
                except Exception:
                    pass
        if not safe_rmtree(self._dir(record.id)):
            log.warning("Could not remove %s completely (files in use).", self._dir(record.id))
