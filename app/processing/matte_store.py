"""Disk-backed cache of raw mattes (one grey image per frame).

Purpose
-------
* **Resume**: an interrupted run continues from the last matte that reached the disk instead of
  re-running the model from the first frame.
* **Re-render**: refinement / background / encoding changes reuse the stored mattes; the model is not
  executed again.
* **Backward tracking**: mattes for frames before the subject's reference frame are produced by a
  reverse pass and read back during the forward render.

Layout: ``<dir>/index.json`` plus ``chunk_NNNNNN.mattes`` files.  A chunk holds up to
``chunk_frames`` losslessly compressed mattes (8 or 16 bit grey, display orientation, full
resolution) in a trivial record format::

    b"VMTMATTE2\\n"  then per frame:
        <uint32 frame index> <uint16 x0> <uint16 y0> <uint16 w> <uint16 h> <uint32 byte length>
        <zlib (level 1) of the w x h crop that contains every non-zero sample>

Mattes are exact zeros outside the subject, so only the bounding box of the non-zero region is
compressed (an all-zero matte costs 16 bytes).  This is several times faster than PNG, which matters
because the cache sits in the hot path of fast models.

Chunk files and the index are written to a temporary name and renamed, so a crash can never leave a
half-written chunk that later reads back as valid.

The store is tied to a *fingerprint*: a hash of the input file identity and of every setting that
influences the raw matte.  Opening a directory whose fingerprint differs discards its content.

Thread safety: all public methods take an internal lock, so ``put`` may be called from a writer
thread while the render loop calls ``get``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import threading
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from ..core.errors import StorageError
from ..utils.fs import atomic_replace, atomic_write_text, safe_rmtree, safe_unlink

log = logging.getLogger(__name__)

STORE_VERSION = 3
_MAGIC = b"VMTMATTE2\n"
_RECORD = struct.Struct("<IHHHHI")   # frame index, x0, y0, w, h, payload length
_CHUNK_SUFFIX = ".mattes"

#: One stored matte: ``(x0, y0, w, h, zlib bytes of the crop)``.
Record = Tuple[int, int, int, int, bytes]


def fingerprint_hash(fingerprint: Dict[str, Any]) -> str:
    """Stable SHA-256 of a JSON-serialisable fingerprint."""

    text = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def encode_matte(matte: np.ndarray, level: int = 1) -> Record:
    """Losslessly pack a grey matte: crop to the bounding box of its non-zero samples and deflate."""

    rows = np.flatnonzero(matte.any(axis=1))
    if rows.size == 0:
        return (0, 0, 0, 0, b"")
    cols = np.flatnonzero(matte.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    crop = np.ascontiguousarray(matte[y0:y1, x0:x1])
    if crop.dtype.itemsize > 1 and crop.dtype.byteorder == ">":   # always store little-endian
        crop = crop.astype(crop.dtype.newbyteorder("<"))
    return (x0, y0, x1 - x0, y1 - y0, zlib.compress(crop.tobytes(), level))


def decode_matte(record: Record, width: int, height: int, dtype: np.dtype) -> np.ndarray:
    """Inverse of :func:`encode_matte`; raises ``ValueError`` / ``zlib.error`` on damaged data."""

    x0, y0, w, h, payload = record
    dt = np.dtype(dtype)
    out = np.zeros((height, width), dtype=dt)
    if w == 0 or h == 0:
        return out
    if x0 + w > width or y0 + h > height:
        raise ValueError("matte record lies outside the frame")
    raw = zlib.decompress(payload)
    if len(raw) != w * h * dt.itemsize:
        raise ValueError("matte record has the wrong size")
    src_dt = dt.newbyteorder("<") if dt.itemsize > 1 else dt
    out[y0 : y0 + h, x0 : x0 + w] = np.frombuffer(raw, dtype=src_dt).reshape(h, w)
    return out


def _read_chunk_file(path: Path) -> Dict[int, Record]:
    """Parse a chunk file into ``{frame index: record}``; raises ``ValueError`` if malformed."""

    blob = path.read_bytes()
    if not blob.startswith(_MAGIC):
        raise ValueError("bad magic")
    out: Dict[int, Record] = {}
    pos = len(_MAGIC)
    end = len(blob)
    while pos < end:
        if pos + _RECORD.size > end:
            raise ValueError("truncated record header")
        index, x0, y0, w, h, length = _RECORD.unpack_from(blob, pos)
        pos += _RECORD.size
        if pos + length > end:
            raise ValueError("truncated record payload")
        out[index] = (x0, y0, w, h, blob[pos : pos + length])
        pos += length
    return out


def _write_chunk_file(path: Path, records: Dict[int, Record]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(_MAGIC)
            for index in sorted(records):
                x0, y0, w, h, payload = records[index]
                fh.write(_RECORD.pack(index, x0, y0, w, h, len(payload)))
                fh.write(payload)
            fh.flush()
        atomic_replace(tmp, path)
    except OSError:
        safe_unlink(tmp)
        raise


class MatteStore:
    def __init__(
        self,
        directory: Path,
        width: int,
        height: int,
        fingerprint: Dict[str, Any],
        *,
        bits: int = 8,
        chunk_frames: int = 48,
        compression: int = 1,
    ) -> None:
        if bits not in (8, 16):
            raise ValueError("bits must be 8 or 16")
        if width > 65535 or height > 65535:
            raise ValueError("frame size exceeds the matte cache limit of 65535 pixels per side")
        self.directory = Path(directory)
        self.width = int(width)
        self.height = int(height)
        self.bits = int(bits)
        self.chunk_frames = int(chunk_frames)
        self.compression = int(compression)
        self.fingerprint = dict(fingerprint)
        self.fingerprint_hash = fingerprint_hash(self.fingerprint)
        self.dtype = np.dtype(np.uint8) if bits == 8 else np.dtype(np.uint16)
        self.max_value = 255 if bits == 8 else 65535
        self.reused = False                  # True when existing content was accepted
        self.reset_reason: Optional[str] = None
        self._lock = threading.RLock()
        self._on_disk: Dict[int, Set[int]] = {}           # chunk id -> frame indices stored in its file
        self._pending: Dict[int, Dict[int, Record]] = {}  # chunk id -> {index: record} newer than the file
        self._dirty: Set[int] = set()                     # chunk ids whose file is behind the pending buffer
        self._cache: "OrderedDict[int, Dict[int, Record]]" = OrderedDict()   # small read cache of chunk records
        self._closed = False
        self._open()

    # ------------------------------------------------------------------------------- setup
    @property
    def index_path(self) -> Path:
        return self.directory / "index.json"

    def _chunk_path(self, chunk_id: int) -> Path:
        return self.directory / f"chunk_{chunk_id:06d}{_CHUNK_SUFFIX}"

    def _open(self) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError(f"Cannot create the matte cache directory {self.directory}: {exc}") from exc
        if not self.index_path.is_file():
            if any(self.directory.iterdir()):
                self._wipe("no index file")
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._wipe(f"unreadable index ({exc})")
            return
        mismatch = None
        if data.get("version") != STORE_VERSION:
            mismatch = "different cache version"
        elif data.get("fingerprint_hash") != self.fingerprint_hash:
            mismatch = "input file or matte-relevant settings changed"
        elif (data.get("width"), data.get("height")) != (self.width, self.height):
            mismatch = "frame size changed"
        elif data.get("bits") != self.bits or data.get("chunk_frames") != self.chunk_frames:
            mismatch = "cache layout changed"
        if mismatch:
            self._wipe(mismatch)
            return
        kept = 0
        for key, indices in (data.get("chunks") or {}).items():
            chunk_id = int(key)
            path = self._chunk_path(chunk_id)
            if not path.is_file():
                continue
            try:
                stored = set(_read_chunk_file(path).keys())
            except (OSError, ValueError) as exc:
                log.warning("Discarding damaged matte chunk %s: %s", path.name, exc)
                safe_unlink(path)
                continue
            present = {int(i) for i in indices} & stored
            if present:
                self._on_disk[chunk_id] = present
                kept += len(present)
        self.reused = kept > 0
        if kept:
            log.info("Matte cache: %d frame(s) available in %s", kept, self.directory)

    def _wipe(self, reason: str) -> None:
        self.reset_reason = reason
        log.info("Matte cache reset (%s): %s", reason, self.directory)
        for entry in list(self.directory.iterdir()):
            ok = safe_rmtree(entry) if entry.is_dir() else safe_unlink(entry)
            if not ok:
                raise StorageError(f"Cannot clear the matte cache {self.directory}: {entry.name} is in use.")
        self._on_disk.clear()
        self._pending.clear()
        self._dirty.clear()
        self._cache.clear()

    # ------------------------------------------------------------------------------- queries
    def has(self, index: int) -> bool:
        chunk_id = index // self.chunk_frames
        with self._lock:
            return index in self._pending.get(chunk_id, ()) or index in self._on_disk.get(chunk_id, ())

    def indices(self) -> List[int]:
        with self._lock:
            out: Set[int] = set()
            for s in self._on_disk.values():
                out |= s
            for d in self._pending.values():
                out |= set(d.keys())
            return sorted(out)

    def __len__(self) -> int:
        return len(self.indices())

    def contiguous_end(self, start: int = 0) -> int:
        """Smallest index ``>= start`` that is NOT stored (so ``[start, end)`` is fully available)."""

        i = int(start)
        with self._lock:
            while self.has(i):
                i += 1
        return i

    def covers(self, start: int, stop: int) -> bool:
        """True if every index in ``[start, stop)`` is stored."""

        return self.contiguous_end(start) >= stop

    def disk_bytes(self) -> int:
        total = 0
        try:
            for entry in self.directory.iterdir():
                if entry.is_file():
                    total += entry.stat().st_size
        except OSError:
            pass
        return total

    # ------------------------------------------------------------------------------- write
    def _encode(self, matte: np.ndarray) -> Record:
        if matte.ndim != 2 or matte.shape != (self.height, self.width):
            raise ValueError(f"matte shape {matte.shape} does not match the store ({self.height}, {self.width})")
        if matte.dtype != self.dtype:
            raise ValueError(f"matte dtype {matte.dtype} does not match the store ({self.dtype})")
        return encode_matte(matte, self.compression)

    def put(self, index: int, matte: np.ndarray) -> None:
        """Store the matte of frame *index* (``uint8`` or ``uint16`` grey, ``height x width``)."""

        record = self._encode(matte)   # compression happens outside the lock
        chunk_id = index // self.chunk_frames
        with self._lock:
            if self._closed:
                raise StorageError("The matte cache is closed.")
            bucket = self._pending.setdefault(chunk_id, {})
            bucket[int(index)] = record
            self._dirty.add(chunk_id)
            total = len(set(bucket) | self._on_disk.get(chunk_id, set()))
            if total >= self.chunk_frames:
                self._write_chunk(chunk_id)
                self._write_index()

    def _load_chunk(self, chunk_id: int) -> Dict[int, Record]:
        cached = self._cache.get(chunk_id)
        if cached is not None:
            self._cache.move_to_end(chunk_id)
            return cached
        data: Dict[int, Record] = {}
        path = self._chunk_path(chunk_id)
        if path.is_file():
            try:
                data = _read_chunk_file(path)
            except (OSError, ValueError) as exc:
                raise StorageError(f"Matte cache chunk {path} is damaged: {exc}") from exc
        self._remember(chunk_id, data)
        return data

    def _remember(self, chunk_id: int, data: Dict[int, Record]) -> None:
        self._cache[chunk_id] = data
        self._cache.move_to_end(chunk_id)
        while len(self._cache) > 3:
            self._cache.popitem(last=False)

    def _write_chunk(self, chunk_id: int) -> None:
        pending = self._pending.get(chunk_id, {})
        if not pending:
            self._dirty.discard(chunk_id)
            return
        merged = dict(self._load_chunk(chunk_id)) if self._on_disk.get(chunk_id) else {}
        merged.update(pending)
        path = self._chunk_path(chunk_id)
        try:
            _write_chunk_file(path, merged)
        except OSError as exc:
            raise StorageError(
                f"Cannot write the matte cache chunk {path}: {exc}",
                hint="Check free disk space in the work directory, or run with --no-resume.",
            ) from exc
        self._on_disk[chunk_id] = set(merged.keys())
        self._pending.pop(chunk_id, None)
        self._dirty.discard(chunk_id)
        self._remember(chunk_id, merged)

    def _write_index(self) -> None:
        data = {
            "version": STORE_VERSION,
            "width": self.width,
            "height": self.height,
            "bits": self.bits,
            "chunk_frames": self.chunk_frames,
            "fingerprint_hash": self.fingerprint_hash,
            "fingerprint": self.fingerprint,
            "chunks": {str(k): sorted(v) for k, v in sorted(self._on_disk.items())},
        }
        try:
            atomic_write_text(self.index_path, json.dumps(data, default=str))
        except OSError as exc:
            raise StorageError(f"Cannot write the matte cache index {self.index_path}: {exc}") from exc

    def flush(self) -> None:
        """Persist everything buffered so far (also incomplete chunks)."""

        with self._lock:
            if not self._dirty:
                return
            for chunk_id in sorted(self._dirty):
                self._write_chunk(chunk_id)
            self._write_index()

    # ------------------------------------------------------------------------------- read
    def get(self, index: int) -> np.ndarray:
        """Return the stored matte of frame *index*; raises ``KeyError`` if it is not stored."""

        chunk_id = index // self.chunk_frames
        with self._lock:
            record = self._pending.get(chunk_id, {}).get(index)
            if record is None:
                if index not in self._on_disk.get(chunk_id, ()):
                    raise KeyError(index)
                record = self._load_chunk(chunk_id).get(index)
                if record is None:
                    raise KeyError(index)
        try:
            return decode_matte(record, self.width, self.height, self.dtype)
        except (ValueError, zlib.error) as exc:
            raise StorageError(f"Matte {index} in the cache is damaged: {exc}") from exc

    def get_many(self, indices: Iterable[int]) -> List[np.ndarray]:
        return [self.get(i) for i in indices]

    def last_nonempty_before(self, index: int, min_fraction: float = 0.0005, max_lookback: int = 600) -> Optional[int]:
        """Index of the latest stored frame ``< index`` whose matte covers at least ``min_fraction``
        of the frame, searching back at most ``max_lookback`` frames. Used to re-seed a model on resume."""

        need = max(1.0, min_fraction * self.width * self.height)
        thresh = self.max_value // 2
        i = index - 1
        stop = max(-1, index - 1 - max_lookback)
        while i > stop:
            if self.has(i) and int((self.get(i) > thresh).sum()) >= need:
                return i
            i -= 1
        return None

    # ------------------------------------------------------------------------------- life cycle
    def drop_from(self, start: int) -> None:
        """Forget every stored matte with index ``>= start`` (used when cached frames must be redone)."""

        with self._lock:
            for chunk_id in list(self._pending):
                bucket = self._pending[chunk_id]
                for idx in [i for i in bucket if i >= start]:
                    del bucket[idx]
                if not bucket:
                    del self._pending[chunk_id]
            for chunk_id in list(self._on_disk):
                keep = {i for i in self._on_disk[chunk_id] if i < start}
                if keep == self._on_disk[chunk_id]:
                    continue
                records = {i: r for i, r in self._load_chunk(chunk_id).items() if i in keep}
                self._cache.pop(chunk_id, None)
                path = self._chunk_path(chunk_id)
                if not records:
                    safe_unlink(path)
                    del self._on_disk[chunk_id]
                    continue
                try:
                    _write_chunk_file(path, records)
                except OSError as exc:
                    raise StorageError(f"Cannot rewrite the matte cache chunk {path}: {exc}") from exc
                self._on_disk[chunk_id] = keep
            self._dirty = {c for c in self._dirty if self._pending.get(c)}
            self._write_index()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self.flush()
            finally:
                self._closed = True
                self._cache.clear()

    def delete(self) -> None:
        """Remove the cache directory entirely."""

        with self._lock:
            self._closed = True
            self._pending.clear()
            self._on_disk.clear()
            self._cache.clear()
        safe_rmtree(self.directory)

    def __enter__(self) -> "MatteStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
