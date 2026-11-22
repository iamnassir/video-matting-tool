"""Hardware detection, device selection and live resource statistics.

Supports NVIDIA CUDA, Apple Silicon MPS and CPU.  All probing is defensive: a broken driver or a
missing optional dependency degrades to "unknown" instead of raising, except in
:func:`select_device` where an explicitly requested but unusable device is an error (or a reported
fallback when the caller allows it).
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .errors import ConfigError, CudaInitError, DeviceError

log = logging.getLogger(__name__)

_MB = 1024.0 * 1024.0


@dataclass
class DeviceInfo:
    id: str                               # "cuda:0", "mps", "cpu"
    type: str                             # "cuda" | "mps" | "cpu"
    name: str
    available: bool = True
    total_memory_mb: Optional[float] = None
    free_memory_mb: Optional[float] = None
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- NVML (optional)

_nvml_lock = threading.Lock()
_nvml_state: Dict[str, Any] = {"tried": False, "ok": False, "module": None}


def _nvml():
    """Return the initialised pynvml module or None."""

    with _nvml_lock:
        if not _nvml_state["tried"]:
            _nvml_state["tried"] = True
            try:
                import pynvml  # provided by the nvidia-ml-py package

                pynvml.nvmlInit()
                _nvml_state["module"] = pynvml
                _nvml_state["ok"] = True
            except Exception as exc:  # not installed, no NVIDIA driver, ...
                log.debug("NVML unavailable: %s", exc)
        return _nvml_state["module"] if _nvml_state["ok"] else None


def _nvidia_smi_stats(index: int) -> Optional[Dict[str, float]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [
                exe,
                f"--id={index}",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if out.returncode != 0:
            return None
        parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        vals = [float(p) if p.replace(".", "", 1).isdigit() else float("nan") for p in parts]
        return {
            "gpu_utilization": vals[0],
            "gpu_memory_used_mb": vals[1],
            "gpu_memory_total_mb": vals[2],
            "gpu_temperature_c": vals[3],
        }
    except Exception:
        return None


def gpu_stats(index: int = 0) -> Dict[str, Optional[float]]:
    """Live utilisation and memory of CUDA device *index* (values are None when unavailable)."""

    stats: Dict[str, Optional[float]] = {
        "gpu_utilization": None,
        "gpu_memory_used_mb": None,
        "gpu_memory_total_mb": None,
        "gpu_temperature_c": None,
    }
    nv = _nvml()
    if nv is not None:
        try:
            handle = nv.nvmlDeviceGetHandleByIndex(index)
            mem = nv.nvmlDeviceGetMemoryInfo(handle)
            stats["gpu_memory_used_mb"] = mem.used / _MB
            stats["gpu_memory_total_mb"] = mem.total / _MB
            try:
                stats["gpu_utilization"] = float(nv.nvmlDeviceGetUtilizationRates(handle).gpu)
            except Exception:
                pass
            try:
                stats["gpu_temperature_c"] = float(nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
            return stats
        except Exception as exc:
            log.debug("NVML query failed: %s", exc)
    smi = _nvidia_smi_stats(index)
    if smi:
        stats.update(smi)
        return stats
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(index)
            stats["gpu_memory_used_mb"] = (total - free) / _MB
            stats["gpu_memory_total_mb"] = total / _MB
    except Exception:
        pass
    return stats


# --------------------------------------------------------------------------- RAM / CPU


def ram_stats() -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "ram_total_mb": None,
        "ram_available_mb": None,
        "ram_system_percent": None,
        "ram_used_mb": None,       # resident set size of this process
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        out["ram_total_mb"] = vm.total / _MB
        out["ram_available_mb"] = vm.available / _MB
        out["ram_system_percent"] = float(vm.percent)
        out["ram_used_mb"] = psutil.Process(os.getpid()).memory_info().rss / _MB
    except Exception as exc:
        log.debug("psutil unavailable: %s", exc)
    return out


def cpu_name() -> str:
    name = platform.processor() or ""
    try:
        if sys.platform == "win32":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip() or name
        elif sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5)
            name = out.stdout.strip() or name
        else:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
    except Exception:
        pass
    return name or platform.machine()


# --------------------------------------------------------------------------- devices


def _nvml_describe(index: int) -> Optional[Tuple[str, float, float]]:
    """``(name, total MB, free MB)`` of a GPU from the driver, without touching the CUDA runtime."""

    nv = _nvml()
    if nv is None:
        return None
    try:
        handle = nv.nvmlDeviceGetHandleByIndex(index)
        name = nv.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        memory = nv.nvmlDeviceGetMemoryInfo(handle)
        return str(name), memory.total / _MB, memory.free / _MB
    except Exception as exc:
        log.debug("NVML device query failed: %s", exc)
        return None


def list_devices() -> List[DeviceInfo]:
    """Enumerate usable inference devices, best first. The CPU is always present."""

    devices: List[DeviceInfo] = []
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a hard requirement
        return [DeviceInfo("cpu", "cpu", cpu_name(), True, detail=f"PyTorch not importable: {exc}")]

    try:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                info = DeviceInfo(f"cuda:{i}", "cuda", "CUDA device", True)
                try:
                    # With a single GPU, name and memory come from the driver (NVML): asking the
                    # CUDA runtime would create a CUDA context, which permanently occupies a few
                    # hundred MB of VRAM in a process that may only want to *list* devices (the
                    # idle web server).  With several GPUs the driver's numbering can differ from
                    # CUDA's, so the runtime is asked there.
                    described = _nvml_describe(i) if count == 1 else None
                    if described is not None:
                        info.name, info.total_memory_mb, info.free_memory_mb = described
                        info.detail = f"CUDA {torch.version.cuda}"
                    else:
                        info.name = torch.cuda.get_device_name(i)
                        free, total = torch.cuda.mem_get_info(i)
                        info.total_memory_mb = total / _MB
                        info.free_memory_mb = free / _MB
                        # On Windows (WDDM) the CUDA runtime reports this process's budget, not what
                        # other applications already occupy.  The driver-level figure is the truthful
                        # one: beyond it the driver silently spills into slow shared system memory.
                        g = gpu_stats(i)
                        if g["gpu_memory_used_mb"] is not None and g["gpu_memory_total_mb"]:
                            info.free_memory_mb = max(
                                0.0, min(info.free_memory_mb, g["gpu_memory_total_mb"] - g["gpu_memory_used_mb"])
                            )
                        cap = torch.cuda.get_device_capability(i)
                        info.detail = f"compute capability {cap[0]}.{cap[1]}, CUDA {torch.version.cuda}"
                except Exception as exc:
                    info.available = False
                    info.detail = f"initialisation failed: {exc}"
                devices.append(info)
        elif getattr(torch.version, "cuda", None) is None:
            log.debug("This PyTorch build has no CUDA support.")
    except Exception as exc:
        devices.append(DeviceInfo("cuda:0", "cuda", "CUDA device", False, detail=f"initialisation failed: {exc}"))

    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            devices.append(DeviceInfo("mps", "mps", "Apple Silicon GPU (MPS)", True, detail="Metal Performance Shaders"))
    except Exception:
        pass

    ram = ram_stats()
    devices.append(
        DeviceInfo(
            "cpu",
            "cpu",
            cpu_name(),
            True,
            total_memory_mb=ram["ram_total_mb"],
            free_memory_mb=ram["ram_available_mb"],
            detail=f"{os.cpu_count() or 1} logical cores",
        )
    )
    return devices


def _probe_device(device_id: str, attempts: int = 3) -> None:
    """Run a tiny computation on the device to make sure it really works.

    A GPU that is shared with other applications can fail transiently while they allocate memory,
    so the test is repeated before the device is given up (falling back to the CPU makes a job tens
    of times slower: it must not happen because of a hiccup).
    """

    import time

    import torch

    dev = torch.device(device_id)
    last: Optional[BaseException] = None
    for attempt in range(max(1, attempts)):
        try:
            x = torch.ones((1, 3, 8, 8), device=dev)
            y = torch.nn.functional.avg_pool2d(x, 2).sum().item()
            if abs(y - 48.0) > 1e-3:
                raise RuntimeError(f"device self-test returned {y}")
            return
        except Exception as exc:  # noqa: BLE001 - retried, then reported by the caller
            last = exc
            log.warning("Self-test of %s failed (attempt %d of %d): %s", device_id, attempt + 1, attempts, exc)
            try:
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
            time.sleep(1.0)
    assert last is not None
    raise last


def select_device(requested: str = "auto", allow_cpu_fallback: bool = True) -> Tuple[Any, DeviceInfo, Optional[str]]:
    """Choose the inference device.

    Returns ``(torch.device, DeviceInfo, fallback_reason)``.  ``fallback_reason`` is None when the
    requested device (or, for ``auto``, the best available accelerator) is used; otherwise it explains
    why the CPU was selected.  Raises :class:`CudaInitError` / :class:`DeviceError` when the requested
    device is unusable and ``allow_cpu_fallback`` is false.
    """

    import torch

    req = (requested or "auto").strip().lower()
    devices = list_devices()
    by_id = {d.id: d for d in devices}
    cpu = by_id["cpu"]

    def fallback(reason: str, error_cls=DeviceError) -> Tuple[Any, DeviceInfo, Optional[str]]:
        if not allow_cpu_fallback:
            raise error_cls(reason, hint="Run with --device cpu, or allow CPU fallback.")
        log.warning("%s Falling back to CPU.", reason)
        return torch.device("cpu"), cpu, reason

    if req == "cpu":
        return torch.device("cpu"), cpu, None

    if req == "auto":
        for d in devices:
            if d.type in ("cuda", "mps") and d.available:
                try:
                    _probe_device(d.id)
                    return torch.device(d.id), d, None
                except Exception as exc:
                    log.warning("Device %s failed its self-test: %s", d.id, exc)
        accel = [d for d in devices if d.type != "cpu"]
        if accel:
            return torch.device("cpu"), cpu, "No working GPU: " + "; ".join(f"{d.id}: {d.detail}" for d in accel)
        return torch.device("cpu"), cpu, None

    if req == "cuda":
        req = "cuda:0"
    if req.startswith("cuda"):
        if getattr(torch.version, "cuda", None) is None:
            return fallback("This PyTorch build has no CUDA support (CPU-only wheel).", CudaInitError)
        info = by_id.get(req)
        if info is None:
            n = sum(1 for d in devices if d.type == "cuda")
            if n == 0:
                return fallback("CUDA was requested but no NVIDIA GPU / driver is available.", CudaInitError)
            raise ConfigError(f"Device {req} does not exist; {n} CUDA device(s) found.")
        if not info.available:
            return fallback(f"CUDA device {req} could not be initialised ({info.detail}).", CudaInitError)
        try:
            _probe_device(req)
        except Exception as exc:
            return fallback(f"CUDA device {req} failed its self-test: {exc}.", CudaInitError)
        return torch.device(req), info, None

    if req == "mps":
        info = by_id.get("mps")
        if info is None:
            return fallback("MPS was requested but is not available on this machine.")
        try:
            _probe_device("mps")
        except Exception as exc:
            return fallback(f"MPS failed its self-test: {exc}.")
        return torch.device("mps"), info, None

    raise ConfigError(f"Unknown device {requested!r}; use auto, cuda, cuda:N, mps or cpu.")


def resolve_precision(precision: str, device: Any, supports_fp16: bool = True) -> str:
    """Resolve ``auto`` to a concrete precision. Half precision is only used on CUDA."""

    p = (precision or "auto").lower()
    dev_type = getattr(device, "type", str(device))
    if p == "fp16":
        if dev_type != "cuda" or not supports_fp16:
            log.info("fp16 is not supported for this model/device combination; using fp32.")
            return "fp32"
        return "fp16"
    if p == "fp32":
        return "fp32"
    return "fp16" if (dev_type == "cuda" and supports_fp16) else "fp32"


def cuda_index(device: Any) -> int:
    idx = getattr(device, "index", None)
    return int(idx) if idx is not None else 0


class ResourceMonitor:
    """Provides the live stats block attached to progress events and tracks peaks for the report."""

    def __init__(self, device: Any, device_info: Optional[DeviceInfo] = None, min_interval: float = 1.0) -> None:
        self.device = device
        self.device_info = device_info
        self.min_interval = float(min_interval)
        self._last_time = 0.0
        self._last: Dict[str, Any] = {}
        self.peak_ram_mb = 0.0
        self._lock = threading.Lock()

    def stats(self) -> Dict[str, Any]:
        import time

        now = time.perf_counter()
        with self._lock:
            if self._last and (now - self._last_time) < self.min_interval:
                return dict(self._last)
            out: Dict[str, Any] = {"device": str(self.device)}
            if self.device_info is not None:
                out["gpu_name"] = self.device_info.name if self.device_info.type != "cpu" else None
            if getattr(self.device, "type", "cpu") == "cuda":
                g = gpu_stats(cuda_index(self.device))
                out["gpu_utilization"] = g["gpu_utilization"]
                out["gpu_memory_used_mb"] = g["gpu_memory_used_mb"]
                out["gpu_memory_total_mb"] = g["gpu_memory_total_mb"]
            r = ram_stats()
            out["ram_used_mb"] = r["ram_used_mb"]
            out["ram_system_percent"] = r["ram_system_percent"]
            if r["ram_used_mb"]:
                self.peak_ram_mb = max(self.peak_ram_mb, float(r["ram_used_mb"]))
            self._last, self._last_time = out, now
            return dict(out)

    def peak_vram_allocated_mb(self) -> Optional[float]:
        try:
            import torch

            if getattr(self.device, "type", "cpu") == "cuda":
                return torch.cuda.max_memory_allocated(self.device) / _MB
        except Exception:
            pass
        return None


def system_info() -> Dict[str, Any]:
    """Static description of the machine for ``GET /api/system`` and ``main.py system``."""

    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu_name(),
        "cpu_threads": os.cpu_count(),
    }
    info.update({k: v for k, v in ram_stats().items() if k in ("ram_total_mb", "ram_available_mb")})
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_runtime"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    except Exception as exc:
        info["torch"] = f"unavailable: {exc}"
    devices = list_devices()
    info["devices"] = [d.to_dict() for d in devices]
    best = next((d for d in devices if d.type != "cpu" and d.available), devices[-1])
    info["default_device"] = best.id
    return info
