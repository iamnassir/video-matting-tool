"""Stage timing for the render loop.

GPU work is asynchronous: a PyTorch call returns as soon as the work is queued, and the time is
paid by whichever later call first needs a result (typically a transfer to the CPU).  Ordinary
profilers therefore blame the wrong stage.  :class:`StageClock` measures honestly by synchronising
the device at the boundaries of every stage - which itself costs a little throughput, so it is only
active on request (``VMT_PROFILE=1``, verbose logging, or the ``profile`` model option) and is a
no-op otherwise.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, Optional


class _Timer:
    __slots__ = ("clock", "name", "t0")

    def __init__(self, clock: "StageClock", name: str) -> None:
        self.clock = clock
        self.name = name
        self.t0 = 0.0

    def __enter__(self) -> "_Timer":
        self.clock._sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.clock._sync()
        self.clock.seconds[self.name] += time.perf_counter() - self.t0
        self.clock.calls[self.name] += 1


class _Noop:
    __slots__ = ()

    def __enter__(self) -> "_Noop":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


_NOOP = _Noop()


class StageClock:
    """``with clock("refine"): ...`` accumulates wall time per stage when enabled."""

    def __init__(self, device: Any = None, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self.seconds: Dict[str, float] = defaultdict(float)
        self.calls: Dict[str, int] = defaultdict(int)
        self._cuda = self.enabled and getattr(device, "type", "cpu") == "cuda"

    def _sync(self) -> None:
        if self._cuda:
            import torch

            torch.cuda.synchronize(self.device)

    def __call__(self, name: str) -> Any:
        return _Timer(self, name) if self.enabled else _NOOP

    def summary(self, frames: Optional[int] = None) -> Dict[str, Any]:
        """``{stage: {"seconds": s, "ms_per_frame": m}}`` ordered by cost (empty when disabled)."""

        out: Dict[str, Any] = {}
        for name, seconds in sorted(self.seconds.items(), key=lambda item: -item[1]):
            entry: Dict[str, Any] = {"seconds": round(seconds, 4), "calls": self.calls[name]}
            if frames:
                entry["ms_per_frame"] = round(1000.0 * seconds / frames, 3)
            out[name] = entry
        return out
