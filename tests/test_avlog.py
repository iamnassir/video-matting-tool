"""PyAV's Python log callback must stay off: with it, closing a multi-threaded decoder can deadlock
the whole process (decoder threads wait for the GIL, the GIL holder waits for the decoder threads)."""

from __future__ import annotations

import av
import av.logging
import pytest

import app  # noqa: F401 - installs the guard
from app.processing.probe import open_container
from app.utils.avlog import silence_pyav_logging


def test_log_forwarding_cannot_be_switched_on():
    silence_pyav_logging()
    assert av.logging.get_level() is None
    av.logging.set_level(av.logging.ERROR)            # what `import torchvision` does
    assert av.logging.get_level() is None
    original = getattr(av.logging.set_level, "__wrapped__")
    original(av.logging.ERROR)                        # somebody who kept the original function
    assert av.logging.get_level() is not None
    silence_pyav_logging()
    assert av.logging.get_level() is None


def test_torchvision_import_does_not_enable_it():
    pytest.importorskip("torchvision")
    import torchvision.io  # noqa: F401

    silence_pyav_logging()
    assert av.logging.get_level() is None


@pytest.mark.timeout(120)
def test_closing_a_threaded_decoder_early_never_hangs(clip_10bit):
    av.logging.set_level(av.logging.ERROR)            # ineffective, see above
    for _ in range(60):
        container = open_container(clip_10bit.path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frames = 0
        for packet in container.demux(stream):
            if packet.size == 0:
                continue
            frames += len(packet.decode())
            if frames >= 2:
                break
        container.close()
    assert av.logging.get_level() is None
