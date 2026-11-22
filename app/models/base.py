"""Model adapter contract.

Every matting / segmentation model is wrapped in a :class:`MattingModel` subclass and registered
with :func:`app.models.registry.register_model`.  The pipeline only talks to this interface, so
models are interchangeable and third parties can add models as plugins.

Life cycle (driven by the pipeline)::

    model = create_model("rvm", variant=None, device=device, precision="auto", ...)
    model.load()                                  # weights -> device (raises WeightsMissingError ...)
    model.start_sequence(SequenceInfo(...))       # reset temporal state, receive the subject
    for batch in frames:                          # strictly consecutive frames, presentation order
        out = model.process(frames, indices)      # -> MatteOutput (alpha B x 1 x H x W in 0..1)
    model.end_sequence()
    model.unload()                                # free VRAM / RAM

Contract details
----------------
* ``frames`` is a float32 tensor ``[B, 3, H, W]`` in 0..1, non-linear R'G'B' (as displayed), on
  ``self.device``, in **display orientation**, at full source resolution.  Adapters do their own
  resizing / normalisation and must return alpha at exactly ``H x W``.
* A sequence always starts at the subject's reference frame: the first frame passed to
  :meth:`process` after :meth:`start_sequence` is the frame the prompt refers to.  For backward
  tracking the pipeline starts a second sequence with ``reverse=True`` and feeds frames in
  descending index order.
* ``process`` must be **retry safe**: if it raises (for example CUDA out of memory), the adapter's
  temporal state must be unchanged so the pipeline can call it again with a smaller batch.
  Adapters with ``max_batch_size == 1`` are always called with a single frame.
* Adapters never read video files, never composite and never touch original pixels: they only
  estimate a matte.
* ``reduce_memory`` is called after an out-of-memory error at batch size 1; return True if the
  adapter switched to a cheaper configuration (lower internal resolution ...) and a retry makes
  sense.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..core.config import QualityPreset
from ..core.types import ResolvedSubject

if TYPE_CHECKING:  # pragma: no cover
    import torch


# --------------------------------------------------------------------------- static metadata


@dataclass(frozen=True)
class LicenseInfo:
    name: str                       # e.g. "GPL-3.0", "Apache-2.0", "NTU S-Lab License 1.0"
    url: str
    commercial_use: str             # "allowed" | "conditional" | "prohibited"
    spdx: Optional[str] = None
    notes: str = ""                 # obligations / caveats in plain language

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "spdx": self.spdx,
            "url": self.url,
            "commercial_use": self.commercial_use,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class WeightFile:
    """One downloadable file. ``filename`` is also the name on disk under ``<models_dir>/<model id>/``."""

    filename: str
    url: str                        # official source URL
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None    # pinned checksum; verified after download
    description: str = ""
    source: str = ""                # human readable origin, e.g. "GitHub release v1.0.0 (official)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "url": self.url,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "description": self.description,
            "source": self.source,
        }


@dataclass(frozen=True)
class ModelVariant:
    id: str
    label: str
    weights: Tuple[WeightFile, ...]
    description: str = ""
    vram_hint_mb: Optional[int] = None        # rough VRAM need at 1080p
    options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "vram_hint_mb": self.vram_hint_mb,
            "weights": [w.to_dict() for w in self.weights],
        }


@dataclass(frozen=True)
class ModelCapabilities:
    #: "alpha" = the network predicts a true soft alpha matte (hair, motion blur);
    #: "mask"  = it predicts a segmentation mask that the refinement stage turns into a soft matte.
    alpha_type: str
    #: prompt kinds the model consumes natively: subset of {"points", "box", "mask"}
    native_prompts: FrozenSet[str] = frozenset()
    #: the model cannot start without a mask of the subject on the reference frame
    needs_initial_mask: bool = False
    #: "human" = trained for people only; "any" = class agnostic
    subject_domain: str = "any"
    #: how temporal consistency is obtained: "recurrent" | "memory" | "none"
    temporal: str = "none"
    #: largest batch the adapter accepts in one process() call (None = no limit)
    max_batch_size: Optional[int] = None
    supports_fp16: bool = True
    devices: FrozenSet[str] = frozenset({"cuda", "cpu"})
    #: can run a sequence in descending frame order (needed for backward tracking)
    supports_reverse: bool = True
    #: follows one selected subject and ignores other people / objects
    tracks_single_subject: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha_type": self.alpha_type,
            "native_prompts": sorted(self.native_prompts),
            "needs_initial_mask": self.needs_initial_mask,
            "subject_domain": self.subject_domain,
            "temporal": self.temporal,
            "max_batch_size": self.max_batch_size,
            "supports_fp16": self.supports_fp16,
            "devices": sorted(self.devices),
            "supports_reverse": self.supports_reverse,
            "tracks_single_subject": self.tracks_single_subject,
        }


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    description: str
    license: LicenseInfo
    capabilities: ModelCapabilities
    variants: Tuple[ModelVariant, ...]
    default_variant: str
    #: variant chosen for each quality preset
    quality_variants: Dict[str, str] = field(default_factory=dict)
    limitations: Tuple[str, ...] = ()
    #: importable Python modules the adapter needs beyond the base requirements
    requires: Tuple[str, ...] = ()
    install_hint: str = ""
    homepage: str = ""
    citation: str = ""
    #: ids of other registered models this adapter delegates to (e.g. sam2_rvm -> sam2, rvm)
    depends_on: Tuple[str, ...] = ()
    #: "matting" = selectable with --model; "auxiliary" = helper used internally (subject detector,
    #: text grounding) whose weights are managed with the same commands
    kind: str = "matting"

    def variant(self, variant_id: Optional[str]) -> ModelVariant:
        vid = variant_id or self.default_variant
        for v in self.variants:
            if v.id == vid:
                return v
        from ..core.errors import ConfigError

        raise ConfigError(
            f"Model '{self.id}' has no variant '{vid}'.",
            hint=f"Available variants: {', '.join(v.id for v in self.variants)}",
        )

    def variant_for_quality(self, quality: QualityPreset) -> str:
        return self.quality_variants.get(quality.value, self.default_variant)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "license": self.license.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "variants": [v.to_dict() for v in self.variants],
            "default_variant": self.default_variant,
            "quality_variants": dict(self.quality_variants),
            "limitations": list(self.limitations),
            "requires": list(self.requires),
            "install_hint": self.install_hint,
            "homepage": self.homepage,
            "citation": self.citation,
            "depends_on": list(self.depends_on),
            "kind": self.kind,
        }


# --------------------------------------------------------------------------- runtime types


@dataclass
class SequenceInfo:
    """Describes the frame sequence an adapter is about to receive."""

    width: int                               # display-oriented frame width handed to the model
    height: int
    fps: float                               # nominal frame rate (informational; footage may be VFR)
    num_frames: Optional[int] = None         # frames in this sequence, if known
    start_index: int = 0                     # absolute index of the first frame of the sequence
    reverse: bool = False                    # frames arrive in descending index order
    subject: Optional[ResolvedSubject] = None
    inference_max_side: Optional[int] = None  # quality-preset hint: cap for internal working resolution


@dataclass
class MatteOutput:
    """Result of :meth:`MattingModel.process` for one batch."""

    alpha: "torch.Tensor"                    # [B, 1, H, W] float32 in 0..1 on the input device
    #: per-frame confidence in 0..1 that the subject is present (None = the model cannot tell)
    presence: Optional[List[float]] = None
    #: True if alpha is a hard / near-binary mask that still needs mask-to-alpha refinement
    is_mask: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)


class MattingModel(ABC):
    """Base class of all model adapters. See the module docstring for the contract."""

    info: ClassVar[ModelInfo]

    def __init__(
        self,
        *,
        variant: Optional[str],
        device: "torch.device",
        precision: str,
        weights_dir: Path,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.variant = self.info.variant(variant)
        self.device = device
        self.precision = precision            # "fp32" | "fp16" (already resolved from "auto")
        self.weights_dir = Path(weights_dir)  # <models_dir>/<model id>
        self.options: Dict[str, Any] = dict(options or {})
        self.loaded = False
        #: Set by the registry: ``factory(model_id, variant=None, options=None) -> MattingModel``.
        #: Composite adapters use it to build the adapters they delegate to (same device, precision,
        #: licence and download policy).  None when an adapter is constructed directly (tests).
        self.factory: Optional[Any] = None

    # -- availability --------------------------------------------------------------------
    @classmethod
    def missing_requirements(cls) -> List[str]:
        """Names of required Python modules that cannot be imported (empty list = installed)."""

        import importlib.util

        missing = []
        for name in cls.info.requires:
            try:
                if importlib.util.find_spec(name) is None:
                    missing.append(name)
            except (ImportError, ValueError):
                missing.append(name)
        return missing

    def weight_path(self, weight: WeightFile) -> Path:
        return self.weights_dir / weight.filename

    # -- life cycle ----------------------------------------------------------------------
    @abstractmethod
    def load(self) -> None:
        """Load weights onto ``self.device``. Must set ``self.loaded = True``."""

    @abstractmethod
    def start_sequence(self, seq: SequenceInfo) -> None:
        """Reset all temporal state and prepare for a new sequence."""

    @abstractmethod
    def process(self, frames: "torch.Tensor", frame_indices: Sequence[int]) -> MatteOutput:
        """Estimate mattes for consecutive frames. See the module docstring."""

    def end_sequence(self) -> None:
        """Release per-sequence state. Default: nothing."""

    def reduce_memory(self) -> bool:
        """Switch to a cheaper configuration after OOM at batch size 1. Default: not possible."""

        return False

    def unload(self) -> None:
        """Free the model. Subclasses should drop references then call ``super().unload()``."""

        self.loaded = False
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best effort
            pass

    # -- introspection -------------------------------------------------------------------
    def runtime_summary(self) -> Dict[str, Any]:
        """Facts about the loaded configuration for the processing report."""

        return {
            "model": self.info.id,
            "variant": self.variant.id,
            "device": str(self.device),
            "precision": self.precision,
            "options": dict(self.options),
        }

    def __enter__(self) -> "MattingModel":
        if not self.loaded:
            self.load()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.end_sequence()
        finally:
            self.unload()
