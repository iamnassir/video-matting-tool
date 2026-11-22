"""File-system helpers: atomic writes that survive Windows sharing violations, disk-space checks,
safe names.

On Windows, replacing or deleting a file fails with ``PermissionError`` while another process has it
open - typically an antivirus on-access scanner or the search indexer that looks at a file right after
it was written or read.  These conditions clear within milliseconds, so the helpers here retry
briefly instead of failing the whole job.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Optional, Union

from ..core.errors import DiskSpaceError, OutputPathError, StorageError

PathLike = Union[str, os.PathLike]

_RETRY_ERRNOS = {13, 16, 32}          # EACCES, EBUSY, ERROR_SHARING_VIOLATION (as errno on some paths)
_RETRY_WINERRORS = {5, 32, 33}        # ACCESS_DENIED, SHARING_VIOLATION, LOCK_VIOLATION


def _is_transient(exc: OSError) -> bool:
    if isinstance(exc, PermissionError):
        return True
    if getattr(exc, "winerror", None) in _RETRY_WINERRORS:
        return True
    return exc.errno in _RETRY_ERRNOS


def clean_path_text(text: Optional[str]) -> str:
    """Tidy a path that a person typed or pasted.

    Windows Explorer's "Copy as path" wraps the path in double quotes, and pasted text often carries
    spaces or a line break.  Neither can be part of a real path at its ends, so they are removed;
    a ``file://`` address (dragged from a browser or file manager) becomes a plain path.
    """

    s = (text or "").strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    if s.lower().startswith("file://"):
        from urllib.parse import unquote, urlparse

        parsed = urlparse(s)
        local = unquote(parsed.path)
        if re.match(r"^/[A-Za-z]:", local):                 # file:///C:/folder/file.png
            local = local[1:]
        elif parsed.netloc and parsed.netloc.lower() != "localhost":
            local = f"//{parsed.netloc}{local}"             # file://server/share/file.png
        s = local
    return s


def atomic_replace(src: PathLike, dst: PathLike, retries: int = 40, delay: float = 0.05) -> None:
    """``os.replace`` with retries for transient sharing violations (about 2-4 s in total)."""

    last: Optional[OSError] = None
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except OSError as exc:
            if not _is_transient(exc):
                raise
            last = exc
            time.sleep(min(0.25, delay * (1 + attempt // 4)))
    assert last is not None
    raise last


def safe_unlink(path: PathLike, retries: int = 20, delay: float = 0.05) -> bool:
    """Delete a file if it exists; retry transient sharing violations. Returns True if it is gone."""

    p = Path(path)
    for attempt in range(retries):
        try:
            p.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError as exc:
            if not _is_transient(exc):
                return False
            time.sleep(min(0.25, delay * (1 + attempt // 4)))
    return not p.exists()


def safe_rmtree(path: PathLike, retries: int = 10, delay: float = 0.1) -> bool:
    """Remove a directory tree; retry transient failures. Returns True if it is gone."""

    p = Path(path)
    for _ in range(retries):
        if not p.exists():
            return True
        shutil.rmtree(p, ignore_errors=True)
        if not p.exists():
            return True
        time.sleep(delay)
    return not p.exists()


def atomic_write_bytes(path: PathLike, data: bytes) -> None:
    """Write *data* to ``path`` via a temporary sibling file and an atomic rename."""

    p = Path(path)
    # Unique per writer: two threads (or processes) replacing the same file must never share a
    # temporary file, or their contents could interleave.
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
        atomic_replace(tmp, p)
    except OSError:
        safe_unlink(tmp)
        raise


def atomic_write_text(path: PathLike, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def free_bytes(path: PathLike) -> int:
    """Free space on the volume that holds *path* (walks up to the first existing parent)."""

    p = Path(path).resolve()
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free


def require_free_space(path: PathLike, needed_bytes: int, what: str = "this operation") -> None:
    """Raise :class:`DiskSpaceError` unless at least *needed_bytes* are free at *path*."""

    try:
        free = free_bytes(path)
    except OSError as exc:
        raise StorageError(f"Cannot determine free disk space for {path}: {exc}") from exc
    if free < needed_bytes:
        raise DiskSpaceError(
            f"Not enough free disk space for {what}: about {needed_bytes / 1e9:.2f} GB needed, "
            f"{free / 1e9:.2f} GB free on the volume of {path}.",
            details={"path": str(path), "needed_bytes": int(needed_bytes), "free_bytes": int(free)},
        )


def ensure_parent_writable(path: PathLike) -> Path:
    """Create the parent directory of *path* and verify that files can be created in it."""

    p = Path(path)
    parent = p.parent if str(p.parent) else Path(".")
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputPathError(f"Cannot create the output directory {parent}: {exc}") from exc
    probe = parent / f".vmt_write_test_{os.getpid()}"
    try:
        with open(probe, "wb") as fh:
            fh.write(b"0")
    except OSError as exc:
        raise OutputPathError(
            f"The output directory {parent} is not writable: {exc}",
            hint="Choose another output location or fix the folder permissions.",
        ) from exc
    finally:
        safe_unlink(probe)
    return p


_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_filename(name: str, default: str = "file", max_length: int = 150) -> str:
    """Reduce an arbitrary (possibly user supplied) name to a safe single path component."""

    base = os.path.basename(str(name).replace("\\", "/"))
    base = _UNSAFE.sub("_", base).strip(" .")
    if not base:
        base = default
    stem, dot, ext = base.rpartition(".")
    if dot and stem:
        if stem.upper() in _RESERVED:
            stem = "_" + stem
        ext = ext[:16]
        stem = stem[: max(1, max_length - len(ext) - 1)]
        return f"{stem}.{ext}"
    if base.upper() in _RESERVED:
        base = "_" + base
    return base[:max_length]


def human_bytes(n: float) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{n} B"
