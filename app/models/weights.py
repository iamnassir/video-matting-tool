"""Model weight management: explicit, verified downloads from official sources.

Policy
------
* Weights are never fetched implicitly.  A download happens only when the user asks for it
  (``python main.py models download <id>``, the *Download* button in the UI, ``--download-models``
  on a run, or ``auto_download_models: true`` in the settings).
* Every file is downloaded from the official URL recorded in the adapter's :class:`ModelInfo`
  (overridable per file in ``weight_overrides`` for mirrors), written to ``<file>.part``, checked
  against the pinned size and SHA-256, and only then moved into place.
* Files live in ``<models_dir>/<model id>/<filename>``; a manually placed file with the right name
  is accepted after the same verification.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .. import __version__
from ..core.errors import (
    ChecksumMismatchError,
    DiskSpaceError,
    WeightsDownloadError,
    WeightsMissingError,
)
from ..core.events import CancelToken
from .base import ModelInfo, ModelVariant, WeightFile

log = logging.getLogger(__name__)

#: ``progress(downloaded_bytes, total_bytes_or_None, filename)``
DownloadProgress = Callable[[int, Optional[int], str], None]

_CHUNK = 1024 * 1024
_USER_AGENT = f"video-matting-tool/{__version__}"


def sha256_file(path: Path, cancel: Optional[CancelToken] = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            if cancel is not None:
                cancel.raise_if_cancelled()
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class WeightsManager:
    def __init__(self, models_dir: Path, overrides: Optional[Dict[str, Dict[str, str]]] = None) -> None:
        self.models_dir = Path(models_dir)
        self.overrides = dict(overrides or {})

    # -- locations -----------------------------------------------------------------------
    def dir_for(self, model_id: str) -> Path:
        return self.models_dir / model_id

    def path(self, model_id: str, weight: WeightFile) -> Path:
        return self.dir_for(model_id) / weight.filename

    def resolved(self, weight: WeightFile) -> WeightFile:
        """Apply per-file overrides (mirror URL, replacement checksum) from the settings."""

        ov = self.overrides.get(weight.filename)
        if not ov:
            return weight
        return WeightFile(
            filename=weight.filename,
            url=ov.get("url", weight.url),
            size_bytes=int(ov["size_bytes"]) if "size_bytes" in ov else (None if "url" in ov else weight.size_bytes),
            sha256=ov.get("sha256", None if "url" in ov and "sha256" not in ov else weight.sha256),
            description=weight.description,
            source=ov.get("source", "user-configured override" if "url" in ov else weight.source),
        )

    # -- verification --------------------------------------------------------------------
    def _marker(self, path: Path) -> Path:
        return path.with_name(path.name + ".verified.json")

    def _marker_valid(self, path: Path, weight: WeightFile) -> bool:
        marker = self._marker(path)
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
            st = path.stat()
            if "mtime_ns" in data:
                same_time = int(data["mtime_ns"]) == st.st_mtime_ns
            else:                                   # marker written by an earlier version
                same_time = abs(float(data.get("mtime", -1)) - st.st_mtime) < 1e-3
            return (
                data.get("size") == st.st_size
                and same_time
                and (weight.sha256 is None or data.get("sha256") == weight.sha256.lower())
            )
        except Exception:
            return False

    def _drop_marker(self, path: Path) -> None:
        """Forget that *path* was verified (called as soon as a check of it fails).

        Size and modification time alone cannot prove that a file is unchanged: a file rewritten
        within one tick of the file system's clock keeps both.  Once a check has failed, nothing
        may fall back on the old marker.
        """

        try:
            self._marker(path).unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.debug("could not remove verification marker for %s: %s", path, exc)

    def _write_marker(self, path: Path, digest: Optional[str]) -> None:
        try:
            st = path.stat()
            self._marker(path).write_text(
                json.dumps({"size": st.st_size, "mtime": st.st_mtime, "mtime_ns": st.st_mtime_ns, "sha256": digest}),
                encoding="utf-8",
            )
        except OSError as exc:  # marker is an optimisation only
            log.debug("could not write verification marker for %s: %s", path, exc)

    def check_file(self, model_id: str, weight: WeightFile, full: bool = False) -> Dict[str, Any]:
        """Describe the on-disk state of one weight file.

        ``full=True`` forces a SHA-256 recomputation; otherwise a previously verified file is trusted
        as long as its size and modification time are unchanged.
        """

        w = self.resolved(weight)
        path = self.path(model_id, w)
        info: Dict[str, Any] = {
            "filename": w.filename,
            "path": str(path),
            "url": w.url,
            "expected_bytes": w.size_bytes,
            "present": path.is_file(),
            "ok": False,
            "detail": "",
        }
        if not info["present"]:
            info["detail"] = "not downloaded"
            return info
        size = path.stat().st_size
        info["bytes"] = size
        if w.size_bytes is not None and size != w.size_bytes:
            self._drop_marker(path)
            info["detail"] = f"size mismatch: {size} bytes on disk, expected {w.size_bytes}"
            return info
        if w.sha256 is None:
            info["ok"] = True
            info["detail"] = "present (no pinned checksum)"
            return info
        if not full and self._marker_valid(path, w):
            info["ok"] = True
            info["detail"] = "verified"
            return info
        digest = sha256_file(path)
        if digest.lower() != w.sha256.lower():
            self._drop_marker(path)
            info["detail"] = f"SHA-256 mismatch: {digest} (expected {w.sha256})"
            return info
        self._write_marker(path, digest)
        info["ok"] = True
        info["detail"] = "verified"
        return info

    def status(self, model_id: str, variant: ModelVariant, full: bool = False) -> List[Dict[str, Any]]:
        return [self.check_file(model_id, w, full=full) for w in variant.weights]

    def is_ready(self, model_id: str, variant: ModelVariant) -> bool:
        """Cheap readiness test (no hashing): every file exists with the expected size."""

        for weight in variant.weights:
            w = self.resolved(weight)
            p = self.path(model_id, w)
            if not p.is_file():
                return False
            if w.size_bytes is not None and p.stat().st_size != w.size_bytes:
                return False
        return True

    # -- download ------------------------------------------------------------------------
    def download(
        self,
        model_id: str,
        weight: WeightFile,
        progress: Optional[DownloadProgress] = None,
        cancel: Optional[CancelToken] = None,
        retries: int = 3,
        timeout: float = 30.0,
    ) -> Path:
        """Download one file (resumable) and verify it. Returns the final path."""

        w = self.resolved(weight)
        dest = self.path(model_id, w)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.check_file(model_id, weight).get("ok"):
            return dest
        if dest.exists():
            log.warning("Replacing invalid weight file %s", dest)
            dest.unlink()
        part = dest.with_name(dest.name + ".part")

        if w.size_bytes:
            have = part.stat().st_size if part.exists() else 0
            free = shutil.disk_usage(dest.parent).free
            need = w.size_bytes - have + 64 * _CHUNK
            if free < need:
                raise DiskSpaceError(
                    f"Not enough free space in {dest.parent} to download {w.filename}: "
                    f"{need / 1e6:.0f} MB needed, {free / 1e6:.0f} MB free.",
                    details={"path": str(dest.parent), "needed_bytes": need, "free_bytes": free},
                )

        last_error: Optional[BaseException] = None
        for attempt in range(1, retries + 1):
            if cancel is not None:
                cancel.raise_if_cancelled()
            try:
                self._fetch(w, part, progress, cancel, timeout)
                last_error = None
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                log.warning("Download attempt %d/%d for %s failed: %s", attempt, retries, w.filename, exc)
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404, 410):
                    break  # retrying cannot help
                if attempt < retries:
                    if cancel is not None and cancel.wait(2.0 * attempt):
                        cancel.raise_if_cancelled()
                    elif cancel is None:
                        time.sleep(2.0 * attempt)
        if last_error is not None:
            raise WeightsDownloadError(
                f"Could not download {w.filename} from {w.url}: {last_error}",
                details={"url": w.url, "partial_file": str(part)},
            ) from last_error

        size = part.stat().st_size
        if w.size_bytes is not None and size != w.size_bytes:
            part.unlink(missing_ok=True)
            raise ChecksumMismatchError(
                f"Downloaded {w.filename} has {size} bytes, expected {w.size_bytes}.",
                details={"url": w.url},
            )
        digest = sha256_file(part, cancel)
        if w.sha256 is not None and digest.lower() != w.sha256.lower():
            part.unlink(missing_ok=True)
            raise ChecksumMismatchError(
                f"SHA-256 of downloaded {w.filename} is {digest}, expected {w.sha256}.",
                details={"url": w.url},
            )
        os.replace(part, dest)
        self._write_marker(dest, digest)
        log.info("Downloaded %s (%0.1f MB, sha256 %s)", dest, size / 1e6, digest[:16])
        return dest

    def _fetch(
        self,
        w: WeightFile,
        part: Path,
        progress: Optional[DownloadProgress],
        cancel: Optional[CancelToken],
        timeout: float,
    ) -> None:
        have = part.stat().st_size if part.exists() else 0
        if w.size_bytes is not None and have > w.size_bytes:
            part.unlink()
            have = 0
        if w.size_bytes is not None and have == w.size_bytes:
            return
        headers = {"User-Agent": _USER_AGENT, "Accept": "*/*"}
        if have > 0:
            headers["Range"] = f"bytes={have}-"
        request = urllib.request.Request(w.url, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and have > 0:   # range not satisfiable: restart from scratch
                part.unlink(missing_ok=True)
                return self._fetch(w, part, progress, cancel, timeout)
            raise
        with response:
            status = getattr(response, "status", 200)
            mode = "ab"
            if have > 0 and status != 206:     # server ignored the Range header
                have = 0
                mode = "wb"
            length = response.headers.get("Content-Length")
            total = (int(length) + have) if length and length.isdigit() else w.size_bytes
            with open(part, mode) as fh:
                done = have
                last = 0.0
                while True:
                    if cancel is not None:
                        cancel.raise_if_cancelled()
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    now = time.perf_counter()
                    if progress is not None and (now - last) >= 0.2:
                        last = now
                        progress(done, total, w.filename)
            if progress is not None:
                progress(done, total, w.filename)

    # -- high level ----------------------------------------------------------------------
    def missing(self, model_id: str, variant: ModelVariant) -> List[WeightFile]:
        return [w for w in variant.weights if not self.check_file(model_id, w).get("ok")]

    def ensure(
        self,
        info: ModelInfo,
        variant: ModelVariant,
        *,
        download: bool,
        progress: Optional[DownloadProgress] = None,
        cancel: Optional[CancelToken] = None,
    ) -> List[Path]:
        """Make sure every file of *variant* is present and verified; download only if allowed."""

        missing = self.missing(info.id, variant)
        if missing and not download:
            total = sum(w.size_bytes or 0 for w in missing)
            names = ", ".join(w.filename for w in missing)
            raise WeightsMissingError(
                f"Weights for model '{info.id}' (variant '{variant.id}') are not installed: {names}"
                + (f" ({total / 1e6:.0f} MB)" if total else "")
                + ".",
                hint=(
                    f"Download them explicitly with:  python main.py models download {info.id} --variant {variant.id}"
                    f"   (licence: {info.license.name}; see 'python main.py models info {info.id}'). "
                    "Or re-run with --download-models."
                ),
                details={"model": info.id, "variant": variant.id, "files": [w.to_dict() for w in missing]},
            )
        for w in missing:
            log.info("Downloading %s for model %s from %s", w.filename, info.id, self.resolved(w).url)
            self.download(info.id, w, progress=progress, cancel=cancel)
        return [self.path(info.id, self.resolved(w)) for w in variant.weights]

    def remove(self, model_id: str, variant: ModelVariant) -> int:
        """Delete the files of a variant. Returns the number of bytes freed."""

        freed = 0
        for weight in variant.weights:
            p = self.path(model_id, weight)
            for candidate in (p, self._marker(p), p.with_name(p.name + ".part")):
                if candidate.exists():
                    freed += candidate.stat().st_size
                    candidate.unlink()
        return freed
