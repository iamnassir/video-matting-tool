"""Keeps PyAV's Python log callback switched off.

PyAV can forward FFmpeg's log messages to Python's ``logging``.  Its callback has to take the
interpreter lock (GIL), and FFmpeg calls it from its decoder worker threads.  When the thread that
holds the GIL is at that moment waiting for those workers - which is what closing or freeing a
multi-threaded decoder does - the process deadlocks for good: no Python thread runs any more, a web
server stops answering.  PyAV's own documentation notes that the callback "sometimes doesn't play
nicely with multi-threaded workflows".

PyAV leaves the callback off by default, but ``import torchvision`` switches it on
(``torchvision.io`` calls ``av.logging.set_level(av.logging.ERROR)``), and SAM 2, MatAnyone, the
subject detector and ``transformers`` all import torchvision.  This module switches the callback off
and makes later attempts to switch it on ineffective.  Nothing in this application reads FFmpeg's
messages through PyAV: decode errors arrive as exceptions, and the FFmpeg command-line tools report
through their own stderr.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

_lock = threading.Lock()
_guarded = False


def silence_pyav_logging() -> None:
    """Switch PyAV's log forwarding off and keep it off. Safe to call often and from any thread."""

    global _guarded
    try:
        import av.logging as av_logging
    except Exception:  # pragma: no cover - PyAV is a core dependency
        return
    with _lock:
        if not _guarded:
            original = av_logging.set_level

            def set_level(level=None):  # noqa: ANN001 - mirrors PyAV's signature
                """Replacement installed by the Video Matting Tool: log forwarding stays off (it can
                deadlock multi-threaded decoding)."""

                original(None)

            set_level.__wrapped__ = original  # type: ignore[attr-defined]
            try:
                av_logging.set_level = set_level
                _guarded = True
            except Exception as exc:  # pragma: no cover
                log.debug("could not guard av.logging.set_level: %s", exc)
            original(None)
        elif av_logging.get_level() is not None:
            # Somebody kept a reference to the original function and used it.
            getattr(av_logging.set_level, "__wrapped__", av_logging.set_level)(None)
