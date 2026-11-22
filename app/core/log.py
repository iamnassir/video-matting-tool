"""Logging setup shared by the CLI and the server."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"
_configured = False


def make_stdio_safe() -> None:
    """Make stdout / stderr tolerant of characters the console code page cannot encode.

    Windows consoles often use a legacy code page; without this a single unusual character in a
    file name would abort the program with UnicodeEncodeError.
    """

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None, quiet_libraries: bool = True) -> None:
    """Configure the root logger once. Later calls only adjust the level / add a file handler."""

    global _configured
    root = logging.getLogger()
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    root.setLevel(numeric)
    if not _configured:
        make_stdio_safe()
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
        root.addHandler(handler)
        _configured = True
    if log_file is not None:
        log_file = Path(log_file)
        already = any(
            isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == log_file.resolve()
            for h in root.handlers
        )
        if not already:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FORMAT, "%Y-%m-%d %H:%M:%S"))
            root.addHandler(fh)
    if quiet_libraries:
        for noisy in ("PIL", "matplotlib", "urllib3", "httpx", "httpcore", "hydra", "iopath", "fvcore", "asyncio", "multipart"):
            logging.getLogger(noisy).setLevel(max(numeric, logging.WARNING))
        # PyAV forwards FFmpeg's log; keep only real problems unless debugging
        logging.getLogger("libav").setLevel(logging.ERROR if numeric > logging.DEBUG else logging.INFO)


def add_job_log_handler(path: Path, level: int = logging.INFO) -> logging.Handler:
    """Attach a per-job log file; the caller removes it with ``logging.getLogger().removeHandler``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT, "%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(handler)
    return handler
