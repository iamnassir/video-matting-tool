"""FFmpeg / ffprobe discovery and process helpers.

Everything that touches the ``ffmpeg`` and ``ffprobe`` executables goes through this module so that

* the binaries are located once (``AppSettings.ffmpeg_path`` / ``ffprobe_path``, the ``VMT_FFMPEG`` /
  ``VMT_FFPROBE`` environment variables, else ``PATH``) and verified to run,
* every child process is started the same way: argument lists (never a shell), no console window
  on Windows, ``stdin`` detached, ``stderr`` drained concurrently,
* failures become :class:`~app.core.errors.FfmpegFailedError` (or :class:`DiskSpaceError` when
  FFmpeg reports a full disk) carrying the command, the return code and the tail of FFmpeg's own
  error output in ``details`` - the only reliable description of what went wrong.

The recipes in :mod:`app.encoding.profiles` and :mod:`app.encoding.muxer` were validated with
FFmpeg 8.1.  Several of them depend on options that older releases do not have (``-fps_mode``,
``-enc_time_base demux``, ``-display_rotation``), so releases older than 6.1 are refused up front
instead of failing in the middle of a job.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

from ..core.config import AppSettings
from ..core.errors import DiskSpaceError, FfmpegFailedError, FfmpegNotFoundError
from ..core.events import CancelToken

log = logging.getLogger(__name__)

__all__ = [
    "CREATE_NO_WINDOW",
    "FfmpegTools",
    "ffmpeg_query",
    "ffprobe_json",
    "find_ffmpeg",
    "is_disk_full",
    "process_failure",
    "run_ffmpeg",
    "signed_returncode",
    "stderr_tail",
]

#: ``creationflags`` for every child process: no console window pops up when the application runs
#: as a service or behind a GUI (0 on platforms without the flag).
CREATE_NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0)

#: Oldest FFmpeg release whose command line has every option the encode / mux recipes use.
MIN_FFMPEG_VERSION: Tuple[int, int] = (6, 1)

#: Lines of FFmpeg's stderr kept in ``details["stderr_tail"]``.
STDERR_TAIL_LINES = 40

#: Upper bound for the stderr text kept in memory per process (FFmpeg can be very chatty).
_MAX_STDERR_BYTES = 512 * 1024

_DISK_FULL_PATTERNS = (
    "no space left on device",
    "not enough space on the disk",
    "there is not enough space",
    "disk quota exceeded",
    "enospc",
)

PathLike = Union[str, "os.PathLike[str]"]


# --------------------------------------------------------------------------- tools


@dataclass(frozen=True)
class FfmpegTools:
    """The located executables plus what was learned by running them once.

    ``encoders`` holds the names of every video *and* audio encoder of this build (the muxer needs
    to know about ``aac`` / ``libopus`` as much as the planner needs ``libx264``).
    """

    ffmpeg: str
    ffprobe: str
    version: str
    encoders: FrozenSet[str]

    def has_encoder(self, name: str) -> bool:
        """True if this FFmpeg build contains the encoder *name* (e.g. ``"libx265"``, ``"h264_nvenc"``)."""

        return name in self.encoders


_CACHE: Dict[Tuple[str, str], FfmpegTools] = {}
_CACHE_LOCK = threading.Lock()


def _exe_name(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def _locate(name: str, configured: Optional[str], sibling_of: Optional[str]) -> str:
    """Resolve one executable: an explicit file or folder, else next to *sibling_of*, else ``PATH``."""

    if configured:
        text = os.path.expandvars(os.path.expanduser(str(configured).strip().strip('"')))
        candidate = Path(text)
        if candidate.is_dir():
            candidate = candidate / _exe_name(name)
        if candidate.is_file():
            return str(candidate)
        found = shutil.which(text)
        if found:
            return found
        raise FfmpegNotFoundError(
            f"{name} was not found at the configured location: {configured}",
            hint=f"Point VMT_{name.upper()} (or '{name}_path' in the configuration file) at the {_exe_name(name)} "
            "executable, or remove the setting to use the one on PATH.",
            details={"configured": str(configured)},
        )
    if sibling_of:
        candidate = Path(sibling_of).with_name(_exe_name(name))
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise FfmpegNotFoundError(f"{name} was not found on PATH.")


def _run_once(cmd: Sequence[str], what: str, timeout: float = 30.0) -> str:
    """Run a short informational command and return its stdout; failure means 'not usable'."""

    try:
        result = subprocess.run(
            list(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, creationflags=CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FfmpegNotFoundError(
            f"{what} could not be started ({cmd[0]}): {exc}",
            details={"command": list(cmd)},
        ) from exc
    if result.returncode != 0:
        raise FfmpegNotFoundError(
            f"{what} is installed but does not run correctly ({cmd[0]} exited with code "
            f"{signed_returncode(result.returncode)}).",
            details={"command": list(cmd), "returncode": result.returncode,
                     "stderr_tail": stderr_tail(result.stderr.decode("utf-8", "replace"))},
        )
    return result.stdout.decode("utf-8", "replace")


_VERSION_RE = re.compile(r"^\s*(?:ffmpeg|ffprobe)\s+version\s+(\S+)", re.IGNORECASE | re.MULTILINE)
_ENCODER_RE = re.compile(r"^\s*([VAS])[A-Z.]{5}\s+(\S+)\s")


def _parse_version(text: str) -> str:
    match = _VERSION_RE.search(text)
    return match.group(1) if match else "unknown"


def _parse_encoders(text: str) -> FrozenSet[str]:
    """Names of the video and audio encoders in the output of ``ffmpeg -encoders``."""

    names = set()
    seen_separator = False
    for line in text.splitlines():
        if not seen_separator:
            seen_separator = line.strip().startswith("---")
            continue
        match = _ENCODER_RE.match(line)
        if match and match.group(1) in ("V", "A"):
            names.add(match.group(2))
    return frozenset(names)


def _check_version(version: str, path: str) -> None:
    match = re.match(r"n?(\d+)\.(\d+)", version)
    if match is None:
        log.debug("FFmpeg version %r is not a release number (development build); assuming it is recent.", version)
        return
    found = (int(match.group(1)), int(match.group(2)))
    if found < MIN_FFMPEG_VERSION:
        raise FfmpegNotFoundError(
            f"FFmpeg {version} ({path}) is too old: version {MIN_FFMPEG_VERSION[0]}.{MIN_FFMPEG_VERSION[1]} or newer "
            "is required for timestamp-exact encoding.",
            hint="Install a current FFmpeg build (https://ffmpeg.org/download.html) and make sure it comes first on "
            "PATH, or set VMT_FFMPEG / VMT_FFPROBE.",
            details={"version": version, "path": path},
        )


def ffmpeg_query(tools: FfmpegTools, args: Sequence[str], timeout: float = 30.0) -> str:
    """Run a short informational ffmpeg command (``-h encoder=NAME`` ...) and return its stdout.

    Raises :class:`FfmpegFailedError` when the command cannot be run or exits with an error.
    """

    command = [tools.ffmpeg, "-hide_banner", *[str(a) for a in args]]
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, creationflags=CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FfmpegFailedError(
            f"ffmpeg could not be queried ({' '.join(command[1:])}): {exc}",
            details={"command": command, "returncode": None, "stderr_tail": ""},
        ) from exc
    if result.returncode != 0:
        raise process_failure("Querying ffmpeg", command, result.returncode, result.stderr.decode("utf-8", "replace"))
    return result.stdout.decode("utf-8", "replace")


def find_ffmpeg(settings: Optional[AppSettings] = None) -> FfmpegTools:
    """Locate ``ffmpeg`` and ``ffprobe``, make sure both run and list the available encoders.

    Search order per executable: the configured path (a file, or a folder containing it), next to
    a configured ``ffmpeg``, then ``PATH``.  Without *settings* the ``VMT_FFMPEG`` / ``VMT_FFPROBE``
    environment variables are honoured.  The result is cached per pair of resolved paths, so the
    three probing processes are started once per installation and not once per job.

    Raises :class:`FfmpegNotFoundError` (with an installation hint) when an executable is missing,
    cannot be started, exits with an error or is too old.
    """

    if settings is not None:
        ffmpeg_cfg, ffprobe_cfg = settings.ffmpeg_path, settings.ffprobe_path
    else:
        ffmpeg_cfg, ffprobe_cfg = os.environ.get("VMT_FFMPEG") or None, os.environ.get("VMT_FFPROBE") or None
    ffmpeg = _locate("ffmpeg", ffmpeg_cfg, None)
    ffprobe = _locate("ffprobe", ffprobe_cfg, ffmpeg if ffmpeg_cfg else None)
    key = (ffmpeg, ffprobe)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached

    version = _parse_version(_run_once([ffmpeg, "-hide_banner", "-version"], "ffmpeg"))
    _check_version(version, ffmpeg)
    probe_version = _parse_version(_run_once([ffprobe, "-hide_banner", "-version"], "ffprobe"))
    if probe_version != version:
        log.warning("ffmpeg (%s) and ffprobe (%s) come from different FFmpeg builds.", version, probe_version)
    encoders = _parse_encoders(_run_once([ffmpeg, "-hide_banner", "-encoders"], "ffmpeg"))
    if not encoders:
        raise FfmpegNotFoundError(
            f"The encoder list of {ffmpeg} could not be read; this does not look like a usable FFmpeg build.",
            details={"path": ffmpeg},
        )
    tools = FfmpegTools(ffmpeg=ffmpeg, ffprobe=ffprobe, version=version, encoders=encoders)
    log.info("Using FFmpeg %s (%s), %d encoders.", version, ffmpeg, len(encoders))
    with _CACHE_LOCK:
        _CACHE[key] = tools
    return tools


# --------------------------------------------------------------------------- failure reporting


def signed_returncode(returncode: Optional[int]) -> Optional[int]:
    """Windows reports FFmpeg's negative AVERROR exit codes as unsigned 32-bit numbers
    (4294967274 for -22); turn them back into the value FFmpeg meant."""

    if returncode is None:
        return None
    return returncode - (1 << 32) if returncode > 0x7FFFFFFF else returncode


def stderr_tail(text: str, lines: int = STDERR_TAIL_LINES) -> str:
    """The last non-empty *lines* of FFmpeg's stderr (progress updates separated by ``\\r`` count as lines)."""

    if not text:
        return ""
    parts = [part.rstrip() for part in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    kept = [part for part in parts if part.strip()]
    return "\n".join(kept[-lines:])


def is_disk_full(text: str) -> bool:
    """True if FFmpeg's error output says that the target volume ran out of space."""

    lowered = (text or "").lower()
    return any(pattern in lowered for pattern in _DISK_FULL_PATTERNS)


def _last_line(tail: str) -> str:
    """The most informative single line of a stderr tail: the last one that is not pure bookkeeping."""

    noise = ("terminating thread", "task finished with error", "nothing was written into output file",
             "conversion failed", "error while filtering", "exiting normally")
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    for line in reversed(lines):
        if not any(marker in line.lower() for marker in noise):
            return line
    return lines[-1] if lines else ""


def process_failure(
    what: str,
    command: Sequence[str],
    returncode: Optional[int],
    stderr_text: str,
    *,
    hint: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Exception:
    """Build the exception for a failed FFmpeg process (the caller raises it).

    ``what`` names the operation in user terms ("Encoding the main output", "Adding the audio").
    A full disk is reported as :class:`DiskSpaceError`, everything else as
    :class:`FfmpegFailedError`; both carry ``command``, ``returncode`` and ``stderr_tail``.
    """

    tail = stderr_tail(stderr_text)
    details: Dict[str, Any] = {
        "command": [str(part) for part in command],
        "returncode": signed_returncode(returncode),
        "stderr_tail": tail,
    }
    if extra:
        details.update(extra)
    if is_disk_full(stderr_text):
        return DiskSpaceError(f"{what} failed: the disk is full.", details=details)
    reason = _last_line(tail)
    message = f"{what} failed (FFmpeg exit code {signed_returncode(returncode)})"
    message += f": {reason}" if reason else "."
    return FfmpegFailedError(
        message,
        hint=hint or "See the FFmpeg messages in the error details; check the encoder options and the output location.",
        details=details,
    )


class _StderrCollector(threading.Thread):
    """Drains a child's stderr on a thread.

    Without a concurrent reader FFmpeg blocks as soon as the pipe buffer (a few kilobytes) is full,
    and the whole pipeline deadlocks.  Only the most recent output is kept, in bounded memory.
    """

    def __init__(self, stream: Any, name: str = "vmt-ffmpeg-stderr") -> None:
        super().__init__(name=name, daemon=True)
        self._stream = stream
        self._chunks = bytearray()
        self._lock = threading.Lock()

    def run(self) -> None:
        try:
            while True:
                chunk = self._stream.read1(65536) if hasattr(self._stream, "read1") else self._stream.read(65536)
                if not chunk:
                    break
                with self._lock:
                    self._chunks += chunk
                    if len(self._chunks) > 2 * _MAX_STDERR_BYTES:
                        del self._chunks[: len(self._chunks) - _MAX_STDERR_BYTES]
        except (OSError, ValueError):       # the pipe is closed under us when the process is killed
            pass

    def text(self) -> str:
        with self._lock:
            return bytes(self._chunks).decode("utf-8", "replace")


def _kill(proc: "subprocess.Popen[bytes]") -> None:
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:      # pragma: no cover - a killed process always exits
        log.warning("ffmpeg (pid %s) did not exit after being killed.", proc.pid)


def run_ffmpeg(
    tools: FfmpegTools,
    args: Sequence[str],
    *,
    cancel: Optional[CancelToken] = None,
    timeout: Optional[float] = None,
    what: str = "ffmpeg",
) -> str:
    """Run ``ffmpeg -hide_banner -nostdin -y <args>`` and return its stderr text.

    The process is polled so that *cancel* can stop it: it is killed and
    :class:`~app.core.errors.JobCancelledError` is raised (the caller removes partial files).
    A non-zero exit raises :class:`FfmpegFailedError` whose message starts with *what* and whose
    ``details`` hold ``command``, ``returncode`` and ``stderr_tail``; "No space left on device"
    becomes :class:`DiskSpaceError`.  *timeout* (seconds) kills a process that runs too long.
    """

    command: List[str] = [tools.ffmpeg, "-hide_banner", "-nostdin", "-y", *[str(a) for a in args]]
    if cancel is not None:
        cancel.raise_if_cancelled()
    log.debug("Running: %s", subprocess.list2cmdline(command))
    try:
        proc = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            creationflags=CREATE_NO_WINDOW,
        )
    except OSError as exc:
        raise FfmpegNotFoundError(
            f"ffmpeg could not be started ({tools.ffmpeg}): {exc}", details={"command": command}
        ) from exc
    collector = _StderrCollector(proc.stderr)
    collector.start()
    deadline = None if timeout is None else time.monotonic() + float(timeout)
    try:
        while True:
            try:
                returncode = proc.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                pass
            if cancel is not None and cancel.cancelled:
                _kill(proc)
                collector.join(timeout=5)
                cancel.raise_if_cancelled()
            if deadline is not None and time.monotonic() > deadline:
                _kill(proc)
                collector.join(timeout=5)
                raise FfmpegFailedError(
                    f"{what} timed out after {float(timeout):.0f} s and was stopped.",
                    details={"command": command, "returncode": None, "stderr_tail": stderr_tail(collector.text())},
                )
    except BaseException:
        if proc.poll() is None:            # KeyboardInterrupt and friends: never leave ffmpeg running
            _kill(proc)
        raise
    finally:
        if proc.poll() is not None:
            collector.join(timeout=10)
            if proc.stderr is not None:
                try:
                    proc.stderr.close()
                except OSError:
                    pass
    text = collector.text()
    if returncode != 0:
        raise process_failure(what, command, returncode, text)
    return text


def ffprobe_json(ffprobe: str, path: PathLike, extra: Sequence[str] = ()) -> Dict[str, Any]:
    """``ffprobe -v error -of json -show_format -show_streams <extra> <path>`` as a dictionary.

    *extra* may add selections such as ``-select_streams a:0`` or ``-show_entries packet=pts``.
    Raises :class:`FfmpegFailedError` (with ffprobe's message in ``details['stderr_tail']``) when the
    file cannot be read or the output is not valid JSON.
    """

    command = [str(ffprobe), "-v", "error", "-of", "json", "-show_format", "-show_streams",
               *[str(a) for a in extra], str(path)]
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=CREATE_NO_WINDOW,
        )
    except OSError as exc:
        raise FfmpegNotFoundError(f"ffprobe could not be started ({ffprobe}): {exc}", details={"command": command}) from exc
    stderr_text = result.stderr.decode("utf-8", "replace")
    if result.returncode != 0:
        tail = stderr_tail(stderr_text)
        reason = _last_line(tail)
        raise FfmpegFailedError(
            f"ffprobe could not read {path}" + (f": {reason}" if reason else "."),
            hint="Check that the file exists, is readable and is a media file.",
            details={"command": command, "returncode": signed_returncode(result.returncode), "stderr_tail": tail},
        )
    try:
        data = json.loads(result.stdout.decode("utf-8", "replace") or "{}")
    except ValueError as exc:
        raise FfmpegFailedError(
            f"ffprobe returned unreadable output for {path}.",
            details={"command": command, "returncode": 0, "stderr_tail": stderr_tail(stderr_text)},
        ) from exc
    if not isinstance(data, dict):
        raise FfmpegFailedError(
            f"ffprobe returned unexpected output for {path}.",
            details={"command": command, "returncode": 0, "stderr_tail": stderr_tail(stderr_text)},
        )
    return data
