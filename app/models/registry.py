"""Model registry and plugin loader.

Built-in adapters register themselves on import of their module.  Third-party adapters can be added
without touching this package in two ways:

1. list an importable module in ``plugins:`` of the configuration file; importing it must call
   :func:`register_model` on a :class:`~app.models.base.MattingModel` subclass;
2. expose an entry point in the group ``video_matting_tool.models`` whose target is either such a
   class or a module that registers one.

See ``docs/MODELS.md`` for a complete plugin example.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from ..core.config import AppSettings, QualityPreset
from ..core.errors import LicenseError, ModelNotInstalledError, UnknownModelError
from ..core.events import CancelToken
from ..core.hardware import resolve_precision
from .base import MattingModel, ModelInfo
from .weights import DownloadProgress, WeightsManager

log = logging.getLogger(__name__)

_REGISTRY: Dict[str, Type[MattingModel]] = {}
#: Helper models that are not matting adapters (subject detector, text grounding).  They share the
#: weight management, licence checks and the ``models`` commands, but cannot be chosen as ``--model``.
_AUXILIARY: Dict[str, ModelInfo] = {}
_LOAD_ERRORS: Dict[str, str] = {}
_builtin_loaded = False
_plugins_loaded: set = set()

#: Built-in adapter modules, in display order.  Each must import cleanly even when its optional
#: third-party dependency is absent (heavy imports happen inside ``load()``).
BUILTIN_ADAPTER_MODULES: Tuple[str, ...] = (
    "app.models.rvm.adapter",
    "app.models.sam2.adapter",
    "app.models.matanyone.adapter",
    "app.models.hybrid.adapter",
    "app.models.hybrid.outline",
    "app.tracking.backends",       # registers the auxiliary subject-selection models
)

ENTRY_POINT_GROUP = "video_matting_tool.models"


def register_model(cls: Type[MattingModel]) -> Type[MattingModel]:
    """Class decorator / function that adds an adapter to the registry."""

    info = getattr(cls, "info", None)
    if not isinstance(info, ModelInfo):
        raise TypeError(f"{cls.__name__} must define a class attribute 'info' of type ModelInfo")
    if not (isinstance(cls, type) and issubclass(cls, MattingModel)):
        raise TypeError(f"{cls!r} is not a MattingModel subclass")
    ids = [v.id for v in info.variants]
    if info.default_variant not in ids:
        raise ValueError(f"model '{info.id}': default_variant '{info.default_variant}' is not among its variants {ids}")
    for quality, variant in info.quality_variants.items():
        if variant not in ids:
            raise ValueError(f"model '{info.id}': quality_variants[{quality!r}] = '{variant}' is not a variant")
    existing = _REGISTRY.get(info.id)
    if existing is not None and existing is not cls:
        log.warning("Model id '%s' re-registered: %s replaces %s", info.id, cls.__name__, existing.__name__)
    _REGISTRY[info.id] = cls
    return cls


def register_auxiliary(info: ModelInfo) -> ModelInfo:
    """Register a helper model (``info.kind == "auxiliary"``): weights and licence are managed like
    those of matting models, but it is used internally (subject detection, text prompts)."""

    if not isinstance(info, ModelInfo):
        raise TypeError("register_auxiliary expects a ModelInfo")
    if info.kind != "auxiliary":
        raise ValueError(f"model '{info.id}': auxiliary models must set kind='auxiliary'")
    if info.id in _REGISTRY:
        raise ValueError(f"model id '{info.id}' is already used by a matting model")
    ids = [v.id for v in info.variants]
    if info.default_variant not in ids:
        raise ValueError(f"model '{info.id}': default_variant '{info.default_variant}' is not among its variants {ids}")
    _AUXILIARY[info.id] = info
    return info


def missing_modules(info: ModelInfo) -> List[str]:
    """Names of Python modules listed in ``info.requires`` that cannot be imported."""

    import importlib.util

    missing = []
    for name in info.requires:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


def _import_adapter(module: str) -> None:
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == module:      # the adapter module itself is absent (optional component)
            log.debug("Model adapter module %s is not present.", module)
            return
        _LOAD_ERRORS[module] = f"{type(exc).__name__}: {exc}"
        log.error("Could not import model adapter module %s: %s", module, exc)
    except Exception as exc:  # a broken adapter must not take the application down
        _LOAD_ERRORS[module] = f"{type(exc).__name__}: {exc}"
        log.error("Could not import model adapter module %s: %s", module, exc, exc_info=log.isEnabledFor(logging.DEBUG))


def load_builtin_models() -> None:
    global _builtin_loaded
    if _builtin_loaded:
        return
    _builtin_loaded = True
    for module in BUILTIN_ADAPTER_MODULES:
        _import_adapter(module)


def load_plugins(modules: Optional[List[str]] = None) -> None:
    """Import plugin modules from the settings and from installed entry points."""

    for module in modules or []:
        if module not in _plugins_loaded:
            _plugins_loaded.add(module)
            _import_adapter(module)
    if "__entry_points__" in _plugins_loaded:
        return
    _plugins_loaded.add("__entry_points__")
    try:
        from importlib import metadata

        eps = metadata.entry_points()
        selected = eps.select(group=ENTRY_POINT_GROUP) if hasattr(eps, "select") else eps.get(ENTRY_POINT_GROUP, [])
        for ep in selected:
            try:
                obj = ep.load()
                if isinstance(obj, type) and issubclass(obj, MattingModel):
                    register_model(obj)
            except Exception as exc:
                _LOAD_ERRORS[f"entry point {ep.name}"] = f"{type(exc).__name__}: {exc}"
                log.error("Could not load model plugin entry point %s: %s", ep.name, exc)
    except Exception as exc:  # pragma: no cover - importlib.metadata oddities
        log.debug("Entry point discovery failed: %s", exc)


def ensure_loaded(settings: Optional[AppSettings] = None) -> None:
    load_builtin_models()
    load_plugins(list(settings.plugins) if settings is not None else None)


def load_errors() -> Dict[str, str]:
    """Adapter modules / plugins that failed to import, with the reason."""

    return dict(_LOAD_ERRORS)


def available_model_ids(settings: Optional[AppSettings] = None) -> List[str]:
    ensure_loaded(settings)
    return list(_REGISTRY.keys())


def get_model_class(model_id: str, settings: Optional[AppSettings] = None) -> Type[MattingModel]:
    """The adapter class of a *matting* model (auxiliary helper models have no adapter class)."""

    ensure_loaded(settings)
    try:
        return _REGISTRY[model_id]
    except KeyError:
        hint = f"Available models: {', '.join(sorted(_REGISTRY)) or '(none registered)'}"
        if model_id in _AUXILIARY:
            hint = f"'{model_id}' is a helper model used for subject selection, not a matting model. {hint}"
        raise UnknownModelError(f"Unknown model '{model_id}'.", hint=hint) from None


def get_model_info(model_id: str, settings: Optional[AppSettings] = None) -> ModelInfo:
    """Metadata of a matting model or an auxiliary helper model."""

    ensure_loaded(settings)
    if model_id in _AUXILIARY:
        return _AUXILIARY[model_id]
    return get_model_class(model_id, settings).info


def list_model_infos(settings: Optional[AppSettings] = None, include_auxiliary: bool = False) -> List[ModelInfo]:
    ensure_loaded(settings)
    infos = [cls.info for cls in _REGISTRY.values()]
    if include_auxiliary:
        infos.extend(_AUXILIARY.values())
    return infos


def weights_manager(settings: AppSettings) -> WeightsManager:
    return WeightsManager(settings.models_dir, settings.weight_overrides)


def model_status(model_id: str, settings: AppSettings, full: bool = False) -> Dict[str, Any]:
    """Installation state of a model: Python package importable? which variants have weights?"""

    info = get_model_info(model_id, settings)
    wm = weights_manager(settings)
    missing = missing_modules(info)
    variants: Dict[str, Any] = {}
    for v in info.variants:
        files = wm.status(info.id, v, full=full) if full else [
            {
                "filename": w.filename,
                "present": wm.path(info.id, w).is_file(),
                "expected_bytes": wm.resolved(w).size_bytes,
                "url": wm.resolved(w).url,
            }
            for w in v.weights
        ]
        variants[v.id] = {
            "ready": wm.is_ready(info.id, v) if not full else all(f.get("ok") for f in files),
            "files": files,
            "download_bytes": sum((wm.resolved(w).size_bytes or 0) for w in v.weights if not wm.path(info.id, w).is_file()),
        }
    deps: Dict[str, Any] = {}
    for dep in info.depends_on:
        try:
            deps[dep] = model_status(dep, settings, full=full)
        except UnknownModelError as exc:
            deps[dep] = {"installed": False, "ready": False, "error": exc.message}
    installed = not missing and all(d.get("installed", False) for d in deps.values())
    default_ready = variants.get(info.default_variant, {}).get("ready", False)
    any_ready = any(v["ready"] for v in variants.values()) if variants else True
    deps_ready = all(d.get("ready", False) for d in deps.values())
    return {
        "id": info.id,
        "installed": installed,
        "missing_requirements": missing,
        "install_hint": info.install_hint if missing else "",
        "variants": variants,
        "ready": installed and any_ready and deps_ready,
        "default_variant_ready": installed and default_ready and deps_ready,
        "dependencies": deps,
        "commercial_use": info.license.commercial_use,
    }


def check_license(info: ModelInfo, commercial_use: bool) -> Optional[str]:
    """Enforce the licence policy.

    Raises :class:`LicenseError` when the installation is flagged for commercial use and the model's
    licence prohibits it.  Returns a warning string for licences with conditions, else None.
    """

    lic = info.license
    if lic.commercial_use == "prohibited":
        if commercial_use:
            raise LicenseError(
                f"Model '{info.id}' is licensed under {lic.name}, which does not permit commercial use.",
                hint="Choose a model whose licence allows commercial use (see 'python main.py models list'), "
                "or obtain a commercial licence from the authors and disable commercial_use for this run.",
                details={"license": lic.to_dict()},
            )
        return f"Model '{info.id}' ({lic.name}) is for non-commercial use only. {lic.notes}".strip()
    if lic.commercial_use == "conditional":
        return f"Model '{info.id}' is licensed under {lic.name}: {lic.notes}".strip()
    return None


def ensure_auxiliary(
    model_id: str,
    *,
    settings: AppSettings,
    variant: Optional[str] = None,
    commercial_use: Optional[bool] = None,
    download: Optional[bool] = None,
    download_progress: Optional[DownloadProgress] = None,
    cancel: Optional[CancelToken] = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Licence, dependency and weight checks for a helper model; returns ``{filename: path}``.

    Mirrors :func:`create_model` for models that are not matting adapters: nothing is downloaded
    unless the caller (or the settings) allows it.
    """

    ensure_loaded(settings)
    info = _AUXILIARY.get(model_id)
    if info is None:
        raise UnknownModelError(
            f"Unknown helper model '{model_id}'.",
            hint=f"Available helper models: {', '.join(sorted(_AUXILIARY)) or '(none registered)'}",
        )
    commercial = settings.commercial_use if commercial_use is None else bool(commercial_use)
    allow_download = settings.auto_download_models if download is None else bool(download)
    note = check_license(info, commercial)
    if note and warnings is not None and note not in warnings:
        warnings.append(note)
    missing = missing_modules(info)
    if missing:
        raise ModelNotInstalledError(
            f"Helper model '{info.id}' needs Python package(s) that are not installed: {', '.join(missing)}.",
            hint=info.install_hint or None,
            details={"missing": missing},
        )
    variant_obj = info.variant(variant)
    wm = weights_manager(settings)
    paths = wm.ensure(info, variant_obj, download=allow_download, progress=download_progress, cancel=cancel)
    return {w.filename: p for w, p in zip(variant_obj.weights, paths)}


def create_model(
    model_id: str,
    *,
    settings: AppSettings,
    device: Any,
    variant: Optional[str] = None,
    quality: QualityPreset = QualityPreset.BALANCED,
    precision: str = "auto",
    options: Optional[Dict[str, Any]] = None,
    commercial_use: Optional[bool] = None,
    download: Optional[bool] = None,
    download_progress: Optional[DownloadProgress] = None,
    cancel: Optional[CancelToken] = None,
    warnings: Optional[List[str]] = None,
) -> MattingModel:
    """Instantiate (but do not load) an adapter after licence, dependency and weight checks.

    ``warnings`` (if given) receives licence notices that must be shown to the user / recorded in
    the report.  Composite adapters obtain their sub-models through ``model.factory``.
    """

    cls = get_model_class(model_id, settings)
    info = cls.info
    commercial = settings.commercial_use if commercial_use is None else bool(commercial_use)
    allow_download = settings.auto_download_models if download is None else bool(download)

    note = check_license(info, commercial)
    if note and warnings is not None and note not in warnings:
        warnings.append(note)

    missing = missing_modules(info)
    if missing:
        raise ModelNotInstalledError(
            f"Model '{info.id}' needs Python package(s) that are not installed: {', '.join(missing)}.",
            hint=info.install_hint or None,
            details={"missing": missing},
        )

    variant_id = variant or info.variant_for_quality(quality)
    variant_obj = info.variant(variant_id)
    wm = weights_manager(settings)
    wm.ensure(info, variant_obj, download=allow_download, progress=download_progress, cancel=cancel)

    resolved_precision = resolve_precision(precision, device, info.capabilities.supports_fp16)
    model = cls(
        variant=variant_obj.id,
        device=device,
        precision=resolved_precision,
        weights_dir=wm.dir_for(info.id),
        # "quality" lets adapters derive preset-dependent defaults (internal resolution ...);
        # variant options and explicit user options override it.
        options={"quality": quality.value, **variant_obj.options, **(options or {})},
    )

    def factory(dep_id: str, dep_variant: Optional[str] = None, dep_options: Optional[Dict[str, Any]] = None) -> MattingModel:
        return create_model(
            dep_id,
            settings=settings,
            device=device,
            variant=dep_variant,
            quality=quality,
            precision=precision,
            options=dep_options,
            commercial_use=commercial,
            download=allow_download,
            download_progress=download_progress,
            cancel=cancel,
            warnings=warnings,
        )

    model.factory = factory  # type: ignore[attr-defined]
    return model


def _reset_for_tests() -> None:
    """Clear the registry (unit tests only)."""

    global _builtin_loaded
    _REGISTRY.clear()
    _AUXILIARY.clear()
    _LOAD_ERRORS.clear()
    _plugins_loaded.clear()
    _builtin_loaded = False
