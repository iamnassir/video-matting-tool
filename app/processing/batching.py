"""Batch sizing and out-of-memory recovery for the render loop.

Inference batch size is chosen from the frame size and the memory that is really free on the
device, then adapted at run time: when PyTorch reports an out-of-memory condition the batch is
halved and the same frames are retried (adapters are retry-safe by contract).  At batch size 1 the
adapter is asked for a cheaper configuration (``reduce_memory``).  Only when that is impossible does
the pipeline give up on the device (and, if allowed, restart on the CPU re-using every matte that is
already in the matte store).
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, TypeVar

from ..core.errors import GpuOutOfMemoryError, is_cuda_oom

log = logging.getLogger(__name__)

T = TypeVar("T")


def release_device_memory(device: Any = None) -> None:
    """Drop unreferenced tensors and return cached blocks to the driver."""

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch, "mps", None)
        if mps is not None and getattr(device, "type", "") == "mps":
            mps.empty_cache()
    except Exception:  # pragma: no cover - best effort
        pass


def auto_batch_size(
    width: int,
    height: int,
    device_type: str,
    free_memory_mb: Optional[float],
    model_max: Optional[int] = None,
    requested: Optional[int] = None,
) -> int:
    """Initial number of frames per model call.

    ``requested`` (the user's ``--batch-size``) wins, capped by the model's own limit.  Otherwise the
    size follows the pixel count per frame and is scaled down when little device memory is free.
    This is only a starting point: the pipeline shrinks it automatically on out-of-memory.
    """

    if requested is not None:
        size = max(1, int(requested))
    else:
        megapixels = (width * height) / 1e6
        if device_type == "cpu":
            size = 2 if megapixels <= 1.0 else 1
        else:
            if megapixels <= 0.6:        # up to ~960x540
                size = 12
            elif megapixels <= 1.0:      # 720p
                size = 8
            elif megapixels <= 2.2:      # 1080p
                size = 4
            elif megapixels <= 4.0:      # 1440p
                size = 2
            else:                        # 4K and above
                size = 1
            if free_memory_mb is not None:
                if free_memory_mb < 1500:
                    size = 1
                elif free_memory_mb < 3000:
                    size = max(1, size // 4)
                elif free_memory_mb < 6000:
                    size = max(1, size // 2)
    if model_max is not None:
        size = min(size, int(model_max))
    return max(1, size)


def composite_chunk_size(width: int, height: int, device_type: str, free_memory_mb: Optional[float]) -> int:
    """Frames composited per call (compositing needs several full-resolution float images per frame)."""

    megapixels = (width * height) / 1e6
    if device_type == "cpu":
        return 1
    if megapixels <= 1.0:
        size = 4
    elif megapixels <= 2.2:
        size = 2
    else:
        size = 1
    if free_memory_mb is not None and free_memory_mb < 3000:
        size = 1
    return size


@dataclass
class BatchController:
    """Current inference batch size plus a record of how often it had to be reduced."""

    size: int
    min_size: int = 1
    reductions: int = 0
    memory_reductions: int = 0     # times the adapter switched to a cheaper configuration

    def reduce(self) -> bool:
        """Halve the batch size. Returns False when it is already at the minimum."""

        if self.size <= self.min_size:
            return False
        self.size = max(self.min_size, self.size // 2)
        self.reductions += 1
        return True


def process_with_recovery(
    model: Any,
    frames: Any,
    frame_indices: Sequence[int],
    controller: BatchController,
    device: Any,
    on_reduce: Optional[Callable[[str], None]] = None,
) -> List[Any]:
    """Run ``model.process`` over *frames* in sub-batches of ``controller.size`` with OOM recovery.

    Returns the list of ``MatteOutput`` objects (one per sub-batch, in order).  Raises
    :class:`GpuOutOfMemoryError` when a single frame does not fit even after the adapter reduced its
    memory use.
    """

    total = int(frames.shape[0])
    outputs: List[Any] = []
    pos = 0
    while pos < total:
        n = min(controller.size, total - pos)
        try:
            out = model.process(frames[pos : pos + n], list(frame_indices[pos : pos + n]))
        except Exception as exc:  # noqa: BLE001 - classified below
            if not is_cuda_oom(exc):
                raise
            del exc
            release_device_memory(device)
            if n > 1 and controller.reduce():
                msg = f"Out of device memory: inference batch size reduced to {controller.size}."
                log.warning(msg)
                if on_reduce is not None:
                    on_reduce(msg)
                continue
            if model.reduce_memory():
                controller.memory_reductions += 1
                msg = "Out of device memory at batch size 1: the model switched to a lower internal resolution."
                log.warning(msg)
                if on_reduce is not None:
                    on_reduce(msg)
                continue
            raise GpuOutOfMemoryError(
                "The inference device ran out of memory even for a single frame.",
                details={"device": str(device), "batch_size": n},
            ) from None
        outputs.append(out)
        pos += n
    return outputs


def chunked_with_recovery(
    fn: Callable[[int, int], T],
    total: int,
    chunk: int,
    device: Any,
    what: str = "compositing",
) -> List[T]:
    """Call ``fn(start, stop)`` over ``range(total)`` in chunks, halving the chunk on OOM.

    ``fn`` must be stateless (safe to call again for the same range).
    """

    results: List[T] = []
    pos = 0
    size = max(1, int(chunk))
    while pos < total:
        n = min(size, total - pos)
        try:
            results.append(fn(pos, pos + n))
        except Exception as exc:  # noqa: BLE001
            if not is_cuda_oom(exc):
                raise
            del exc
            release_device_memory(device)
            if n > 1:
                size = max(1, n // 2)
                log.warning("Out of device memory during %s: chunk reduced to %d frame(s).", what, size)
                continue
            raise GpuOutOfMemoryError(
                f"The inference device ran out of memory during {what} of a single frame.",
                details={"device": str(device)},
            ) from None
        pos += n
    return results
