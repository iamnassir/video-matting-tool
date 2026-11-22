"""Hydra-free, download-free construction of the MatAnyone network.

The official ways to obtain a ``matanyone.MatAnyone`` network are unusable inside this application:

* ``matanyone.utils.get_default_model.get_matanyone_model`` composes its configuration with Hydra,
  which is process-global state: a second call in the same process fails ("GlobalHydra is already
  initialized").  It also leaves ``pretrained_resnet`` unset, see below.
* ``MatAnyone.from_pretrained`` goes through the Hugging Face hub cache (``HF_HOME``).
* Both build their ResNet-50 / ResNet-18 backbones with ``pretrained=True`` unless the configuration
  says otherwise, which downloads about 149 MB of ImageNet weights into the torch hub cache
  (``TORCH_HOME``) - weights that are overwritten by the checkpoint a moment later.

This module therefore builds the network from a configuration that is embedded here (the exact
architecture the two official checkpoints were trained with, see :data:`NETWORK_CONFIG`), switches
the backbone download off, installs a guard that turns any download attempt into an error, and loads
the verified ``.pth`` file with ``strict=True`` so that an architecture / weight mismatch can never go
unnoticed.  Nothing is read from or written to any cache directory.

It also owns the workaround for the package's module-level ``device`` global
(``matanyone/model/matanyone.py``: ``device = get_default_device()``).  ``MatAnyone.encode_image``
moves its normalisation buffers to *that* device on every call, i.e. to CUDA whenever a CUDA device
exists - regardless of where the network lives.  A network on the CPU (CPU fallback, tests) or on a
second GPU would therefore fail.  :func:`bound_device` points the global at the device of the
network that is about to run and serialises such sections, so CPU and GPU instances can be used in
the same process, in any order.

Line references in comments are to pq-yang/MatAnyone at commit :data:`PINNED_COMMIT`.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Tuple

from ...core.errors import GpuOutOfMemoryError, ModelLoadError, ModelNotInstalledError, is_cuda_oom

if TYPE_CHECKING:  # pragma: no cover
    import torch

log = logging.getLogger(__name__)

#: Commit of pq-yang/MatAnyone this adapter was written and tested against.
PINNED_COMMIT = "e5ddc534c1fff9bb9e54cf476095d29071b7cb4f"

#: The only supported way to install the package: without its declared dependencies.  They are
#: training / GUI packages (gradio, PySide6, cchardet, netifaces ...) that either do not build on
#: Windows or would replace numpy / OpenCV / FastAPI of this application.
INSTALL_COMMAND = (
    f'pip install --no-deps "matanyone @ git+https://github.com/pq-yang/MatAnyone@{PINNED_COMMIT}"'
)

#: Complete configuration of the network and of ``InferenceCore``.
#:
#: ``model`` is ``matanyone/config/model/base.yaml`` with its two interpolations resolved; the other
#: keys are the inference settings of ``matanyone/config/eval_matanyone_config.yaml`` (identical to
#: the ``config.json`` of the Hugging Face repositories PeiqingYang/MatAnyone and
#: PeiqingYang/MatAnyone2).  It is embedded rather than read from the installed package because the
#: architecture is a property of the pinned checkpoints, not of whatever YAML happens to be
#: installed; the strict state-dict load proves that the two agree.
NETWORK_CONFIG: Dict[str, Any] = {
    "amp": False,                 # not read by any inference code; fp16 is applied by the adapter
    "flip_aug": False,
    "max_internal_size": -1,      # must stay off: its mask branch crashes on the 2-D matting mask
    "save_all": True,
    "use_all_masks": False,
    "use_long_term": False,
    "mem_every": 5,
    "max_mem_frames": 5,
    "long_term": {
        "count_usage": True,
        "max_mem_frames": 10,
        "min_mem_frames": 5,
        "num_prototypes": 128,
        "max_num_tokens": 10000,
        "buffer_tokens": 2000,
    },
    "top_k": 30,
    "stagger_updates": 5,
    "chunk_size": -1,
    "save_scores": False,
    "save_aux": False,
    "visualize": False,
    "output_dir": None,
    "model": {
        # Never fetch ImageNet backbones (big_modules.py:58 and :137 default this to True).
        "pretrained_resnet": False,
        "pixel_mean": [0.485, 0.456, 0.406],
        "pixel_std": [0.229, 0.224, 0.225],
        "pixel_dim": 256,
        "key_dim": 64,
        "value_dim": 256,
        "sensory_dim": 256,
        "embed_dim": 256,
        "pixel_encoder": {"type": "resnet50", "ms_dims": [1024, 512, 256, 64, 3]},
        "mask_encoder": {"type": "resnet18", "final_dim": 256},
        "pixel_pe_scale": 32,
        "pixel_pe_temperature": 128,
        "object_transformer": {
            "embed_dim": 256,
            "ff_dim": 2048,
            "num_heads": 8,
            "num_blocks": 3,
            "num_queries": 16,
            "read_from_pixel": {"input_norm": False, "input_add_pe": False, "add_pe_to_qkv": [True, True, False]},
            "read_from_past": {"add_pe_to_qkv": [True, True, False]},
            "read_from_memory": {"add_pe_to_qkv": [True, True, False]},
            "read_from_query": {"add_pe_to_qkv": [True, True, False], "output_norm": False},
            "query_self_attention": {"add_pe_to_qkv": [True, True, False]},
            "pixel_self_attention": {"add_pe_to_qkv": [True, True, False]},
        },
        "object_summarizer": {"embed_dim": 256, "num_summaries": 16, "add_pe": True},
        "aux_loss": {
            "sensory": {"enabled": True, "weight": 0.01},
            "query": {"enabled": True, "weight": 0.01},
        },
        "mask_decoder": {"up_dims": [256, 128, 128, 64, 16]},
    },
}

#: Serialises everything that depends on process-global state of the ``matanyone`` package: the
#: module-level ``device`` and the temporary download guard.  Re-entrant, so the adapter can nest.
_GLOBAL_LOCK = threading.RLock()


def import_matanyone() -> Tuple[Any, Any, Any]:
    """Import the optional package and return ``(InferenceCore, MatAnyone, model_module)``.

    ``model_module`` is ``matanyone.model.matanyone``, the module that holds the ``device`` global.
    The import happens here, never at module import time of the adapter, so the application starts
    without the package and reports a precise, actionable error when it is really needed.
    """

    try:
        import matanyone.model.matanyone as model_module
        from matanyone import InferenceCore, MatAnyone
    except ImportError as exc:
        raise ModelNotInstalledError(
            f"The 'matanyone' package (or one of the modules it imports) is not available: {exc}",
            hint=f"Install it without its GUI / training dependencies:  {INSTALL_COMMAND}   "
            "It needs torch, torchvision, omegaconf, huggingface_hub, safetensors, imageio, tqdm, Pillow and "
            "OpenCV, all of which are part of this application's environment.",
            details={"missing": getattr(exc, "name", None)},
        ) from exc
    return InferenceCore, MatAnyone, model_module


def network_config(overrides: Optional[Dict[str, Any]] = None) -> Any:
    """A fresh OmegaConf copy of :data:`NETWORK_CONFIG`, optionally with top-level overrides merged in."""

    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ModelNotInstalledError(
            f"MatAnyone needs the 'omegaconf' package: {exc}",
            hint='Install it with:  pip install "omegaconf>=2.3,<2.4"',
            details={"missing": "omegaconf"},
        ) from exc
    cfg = OmegaConf.create(copy.deepcopy(NETWORK_CONFIG))
    if overrides:
        cfg = OmegaConf.merge(cfg, overrides)
    return cfg


class _DownloadForbidden:
    """Stand-in for ``torch.utils.model_zoo`` inside ``matanyone.model.utils.resnet`` while a network
    is being built: any attempt to fetch backbone weights becomes an error instead of a silent
    download into the user profile."""

    @staticmethod
    def load_url(url: str, *args: Any, **kwargs: Any) -> Any:
        raise ModelLoadError(
            f"MatAnyone tried to download backbone weights from {url}; downloads are not permitted here.",
            hint="This is a bug in the network configuration (pretrained_resnet must be false); "
            f"make sure the pinned package version is installed:  {INSTALL_COMMAND}",
            details={"url": url},
        )


@contextlib.contextmanager
def _forbid_downloads() -> Iterator[None]:
    """Second line of defence behind ``pretrained_resnet: False`` (see :class:`_DownloadForbidden`)."""

    try:
        from matanyone.model.utils import resnet as resnet_module
    except ImportError:  # the package layout changed; the configuration switch is all we have
        yield
        return
    with _GLOBAL_LOCK:
        missing = object()
        original = getattr(resnet_module, "model_zoo", missing)
        if original is missing:
            yield
            return
        resnet_module.model_zoo = _DownloadForbidden
        try:
            yield
        finally:
            resnet_module.model_zoo = original


def read_state_dict(weights_path: Path) -> Dict[str, Any]:
    """Read an official ``.pth`` checkpoint (a plain ``OrderedDict[str, Tensor]``) onto the CPU.

    ``weights_only=True`` restricts unpickling to tensors and containers, so even a tampered file
    cannot execute code (the file is additionally pinned by SHA-256 in the weights manager).
    """

    import torch

    path = Path(weights_path)
    if not path.is_file():
        raise ModelLoadError(
            f"MatAnyone weights file not found: {path}",
            hint="Download it with 'python main.py models download matanyone'.",
        )
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelLoadError(
            f"Could not read the MatAnyone weights from {path}: {exc}",
            hint="Verify the file with 'python main.py models verify matanyone' or download it again.",
        ) from exc
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ModelLoadError(
            f"{path} is not a MatAnyone checkpoint (expected a state dict of tensors).",
            hint="Verify the file with 'python main.py models verify matanyone' or download it again.",
        )
    return state


def build_network(weights_path: Path, device: "torch.device") -> Any:
    """Build ``matanyone.MatAnyone`` (single-object matting network) and load *weights_path* into it.

    Works for both official checkpoints (``matanyone.pth`` and ``matanyone2.pth`` hold the same 562
    tensors).  No Hydra, no network access, no cache directory.  The network is returned in eval
    mode on *device* with float32 weights; half precision is applied per call with autocast, as the
    official scripts do.
    """

    import torch

    _, MatAnyone, _ = import_matanyone()
    cfg = network_config()
    state = read_state_dict(weights_path)
    # Build outside inference mode even when the caller is inside it (the pipeline loads models in
    # an inference-mode block): parameters created in inference mode could not be used by code
    # that runs with gradients disabled only through no_grad.
    with _GLOBAL_LOCK, torch.inference_mode(False), torch.no_grad():
        try:
            with _forbid_downloads():
                net = MatAnyone(cfg, single_object=True)
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(
                f"Could not construct the MatAnyone network: {type(exc).__name__}: {exc}",
                hint=f"The installed 'matanyone' package may not be the supported version. Reinstall it:  {INSTALL_COMMAND}",
            ) from exc
        try:
            net.load_state_dict(state, strict=True)
        except Exception as exc:
            raise ModelLoadError(
                f"The weights in {Path(weights_path).name} do not match the MatAnyone network: {exc}",
                hint="Verify the file with 'python main.py models verify matanyone'; if it is intact, the installed "
                f"'matanyone' package is not the supported version:  {INSTALL_COMMAND}",
            ) from exc
        del state
        try:
            net = net.to(device).eval()
        except Exception as exc:
            if is_cuda_oom(exc):
                # Not a broken installation: the device is full.  The pipeline answers this error
                # class with its CPU fallback (when the user allows it).
                raise GpuOutOfMemoryError(
                    f"Not enough free memory on {device} to load the MatAnyone network (about 0.3 GB).",
                    details={"device": str(device)},
                ) from None
            raise ModelLoadError(f"Could not move the MatAnyone network to {device}: {exc}") from exc
    return net


@contextlib.contextmanager
def bound_device(device: "torch.device") -> Iterator[None]:
    """Run a block with the package's module-level ``device`` pointing at *device*.

    ``MatAnyone.encode_image`` (matanyone.py:89-90) moves the normalisation buffers of the network to
    that global on every call.  Every forward pass of a network must therefore happen inside this
    context with the device the network lives on.  The lock makes interleaved use of several
    networks on different devices safe, also from different threads.
    """

    _, _, model_module = import_matanyone()
    with _GLOBAL_LOCK:
        previous = getattr(model_module, "device", None)
        model_module.device = device
        try:
            yield
        finally:
            model_module.device = previous


def create_inference_core(network: Any, device: "torch.device", *, mem_every: int = 5, max_mem_frames: int = 5) -> Any:
    """A fresh ``InferenceCore`` (= empty temporal memory) for one sequence.

    The settings are merged into a per-sequence copy of the configuration, so the network's own
    ``cfg`` is never modified.  ``max_mem_frames`` counts the permanent first-frame memory plus the
    FIFO of recent memory frames; ``mem_every`` is the spacing of memory frames.
    """

    InferenceCore, _, _ = import_matanyone()
    from omegaconf import OmegaConf

    cfg = OmegaConf.merge(
        network.cfg,
        {
            "mem_every": int(mem_every),
            "max_mem_frames": int(max_mem_frames),
            "max_internal_size": -1,
            "flip_aug": False,
            "use_long_term": False,
            "chunk_size": -1,
            "save_aux": False,
        },
    )
    with bound_device(device):
        core = InferenceCore(network, cfg=cfg, device=device)
    # The store only ever holds the features of the frame being processed; the adapter empties it
    # itself when a step fails, so the destructor's "leaking" warning would only be noise.
    core.image_feature_store.no_warning = True
    return core
