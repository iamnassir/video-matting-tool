"""Configuration models (Pydantic v2) shared by the CLI, the API and the pipeline.

Three layers:

* :class:`AppSettings`      - installation level settings (directories, FFmpeg location, server,
  model download policy).  Loaded from ``configs/default.yaml``, an optional user file and
  ``VMT_*`` environment variables.
* :class:`ProcessingConfig` - everything that describes *one* processing run.  This is the JSON
  body of ``POST /api/jobs`` and what the CLI builds from its flags.
* presets                   - quality presets and refinement presets fill in every value the user
  left unset (``None``).  Explicit values always win over presets.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =========================================================================== enums / tables


class BackgroundMode(str, Enum):
    GREEN = "green"
    BLUE = "blue"
    BLACK = "black"
    WHITE = "white"
    TRANSPARENT = "transparent"
    CUSTOM = "custom"


#: Exact 8-bit R'G'B' values of the built-in backgrounds.  Green is pure chroma green #00FF00.
BACKGROUND_COLORS: Dict[BackgroundMode, Tuple[int, int, int]] = {
    BackgroundMode.GREEN: (0, 255, 0),
    BackgroundMode.BLUE: (0, 0, 255),
    BackgroundMode.BLACK: (0, 0, 0),
    BackgroundMode.WHITE: (255, 255, 255),
}


class QualityPreset(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"
    MAX = "max"


class VideoCodec(str, Enum):
    H264 = "h264"                    # libx264            - high quality delivery
    H265 = "h265"                    # libx265            - high quality delivery
    H264_NVENC = "h264_nvenc"        # NVIDIA hardware    - fast delivery
    HEVC_NVENC = "hevc_nvenc"        # NVIDIA hardware    - fast delivery
    PRORES_422_HQ = "prores_422_hq"  # prores_ks profile 3 - professional intermediate
    PRORES_4444 = "prores_4444"      # prores_ks profile 4 - alpha channel output
    FFV1 = "ffv1"                    # lossless (optionally with alpha)
    VP9 = "vp9"                      # libvpx-vp9 (optionally with alpha) for the web


@dataclass(frozen=True)
class CodecInfo:
    codec: VideoCodec
    label: str
    group: str                       # delivery | intermediate | alpha | lossless
    ffmpeg_encoder: str
    containers: Tuple[str, ...]      # allowed output containers, first is the default
    supports_alpha: bool = False
    lossless: bool = False
    hardware: bool = False
    rate_control: str = "crf"        # crf | cq | none
    description: str = ""

    @property
    def default_container(self) -> str:
        return self.containers[0]


CODECS: Dict[VideoCodec, CodecInfo] = {
    VideoCodec.H264: CodecInfo(
        VideoCodec.H264, "H.264 (libx264)", "delivery", "libx264", ("mp4", "mov", "mkv"),
        description="High quality delivery. Plays everywhere. CRF or target bitrate.",
    ),
    VideoCodec.H265: CodecInfo(
        VideoCodec.H265, "H.265 / HEVC (libx265)", "delivery", "libx265", ("mp4", "mov", "mkv"),
        description="High quality delivery at lower bitrates, 10-bit capable. CRF or target bitrate.",
    ),
    VideoCodec.H264_NVENC: CodecInfo(
        VideoCodec.H264_NVENC, "H.264 (NVIDIA NVENC)", "delivery", "h264_nvenc", ("mp4", "mov", "mkv"),
        hardware=True, rate_control="cq",
        description="GPU encoder: fastest, slightly lower quality per bit than libx264.",
    ),
    VideoCodec.HEVC_NVENC: CodecInfo(
        VideoCodec.HEVC_NVENC, "H.265 / HEVC (NVIDIA NVENC)", "delivery", "hevc_nvenc", ("mp4", "mov", "mkv"),
        hardware=True, rate_control="cq",
        description="GPU encoder: fastest HEVC, 10-bit capable.",
    ),
    VideoCodec.PRORES_422_HQ: CodecInfo(
        VideoCodec.PRORES_422_HQ, "Apple ProRes 422 HQ", "intermediate", "prores_ks", ("mov", "mkv"),
        rate_control="none",
        description="Professional intermediate (10-bit 4:2:2). Best choice for keying in an NLE.",
    ),
    VideoCodec.PRORES_4444: CodecInfo(
        VideoCodec.PRORES_4444, "Apple ProRes 4444 (alpha)", "alpha", "prores_ks", ("mov", "mkv"),
        supports_alpha=True, rate_control="none",
        description="10-bit 4:4:4 with an alpha channel. Use for transparent output.",
    ),
    VideoCodec.FFV1: CodecInfo(
        VideoCodec.FFV1, "FFV1 (lossless)", "lossless", "ffv1", ("mkv",),
        supports_alpha=True, lossless=True, rate_control="none",
        description="Mathematically lossless archive / intermediate, optional alpha. Large files.",
    ),
    VideoCodec.VP9: CodecInfo(
        VideoCodec.VP9, "VP9 (libvpx-vp9)", "delivery", "libvpx-vp9", ("webm", "mkv"),
        supports_alpha=True,
        description="Web delivery; supports transparency in WebM.",
    ),
}

SUPPORTED_INPUT_EXTENSIONS: Tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")
OUTPUT_CONTAINERS: Tuple[str, ...] = ("mp4", "mov", "mkv", "webm")
CONTAINER_EXTENSIONS: Dict[str, str] = {"mp4": ".mp4", "mov": ".mov", "mkv": ".mkv", "webm": ".webm"}
EXTENSION_CONTAINERS: Dict[str, str] = {".mp4": "mp4", ".m4v": "mp4", ".mov": "mov", ".mkv": "mkv", ".webm": "webm"}

# =========================================================================== subject selection


class PointPrompt(BaseModel):
    """A click on the reference frame. ``label`` 1 = this is the subject, 0 = this is NOT the subject."""

    model_config = ConfigDict(extra="forbid")
    x: float
    y: float
    label: int = 1

    @field_validator("label")
    @classmethod
    def _label(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("label must be 1 (positive) or 0 (negative)")
        return v


class SubjectPrompt(BaseModel):
    """How the foreground subject is chosen.

    Coordinates refer to the reference frame ``frame_index`` in *display orientation* (what a viewer
    sees).  They are pixels unless ``normalized`` is true, in which case they are fractions of the
    display width / height.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["auto", "points", "box", "mask", "text"] = "auto"
    frame_index: int = Field(0, ge=0, description="Reference frame the prompt refers to (0-based).")
    points: List[PointPrompt] = Field(default_factory=list)
    box: Optional[Tuple[float, float, float, float]] = Field(None, description="x1, y1, x2, y2")
    text: Optional[str] = None
    mask_path: Optional[str] = Field(None, description="Path to a mask image (white = subject) for the reference frame.")
    normalized: bool = False
    isolate: bool = Field(
        True,
        description="Keep only the selected / primary subject. For models without prompt support (RVM) "
        "this enables connected-component subject tracking; disable to keep every person the model mattes.",
    )
    track_backward: bool = Field(
        True, description="When the reference frame is not the first frame, also track the subject backwards to frame 0."
    )

    @field_validator("mask_path", mode="before")
    @classmethod
    def _tidy_mask_path(cls, value: Any) -> Any:
        # A path pasted with Explorer's "Copy as path" arrives wrapped in quotes.
        if isinstance(value, str):
            from ..utils.fs import clean_path_text

            return clean_path_text(value) or None
        return value

    @model_validator(mode="after")
    def _check(self) -> "SubjectPrompt":
        if self.mode == "points" and not any(p.label == 1 for p in self.points):
            raise ValueError("subject mode 'points' needs at least one positive point")
        if self.mode == "box" and self.box is None:
            raise ValueError("subject mode 'box' needs box=(x1, y1, x2, y2)")
        if self.mode == "text" and not (self.text and self.text.strip()):
            raise ValueError("subject mode 'text' needs a non-empty text")
        if self.mode == "mask" and not self.mask_path:
            raise ValueError("subject mode 'mask' needs mask_path")
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            if x2 == x1 or y2 == y1:
                raise ValueError("box must have a non-zero area")
        return self

    def to_pixels(self, width: int, height: int) -> "SubjectPrompt":
        """Return a copy with pixel coordinates for a ``width x height`` display frame."""

        if not self.normalized:
            return self
        pts = [PointPrompt(x=p.x * width, y=p.y * height, label=p.label) for p in self.points]
        box = None
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            box = (x1 * width, y1 * height, x2 * width, y2 * height)
        return self.model_copy(update={"points": pts, "box": box, "normalized": False})


_NUM = r"[-+]?\d+(?:\.\d+)?"


def parse_subject_spec(spec: Optional[str]) -> SubjectPrompt:
    """Parse the CLI ``--subject`` value.

    Accepted forms::

        auto
        box:x1,y1,x2,y2
        points:x,y;x,y,-;x,y,+          (a bare x,y is a positive click; '-' or 0 marks a negative click)
        text:the woman in the red jacket
        mask:path/to/first_frame_mask.png
        {"mode": "box", "box": [..]}    (JSON form of SubjectPrompt)

    Coordinates are pixels of the reference frame as displayed; if every coordinate is within 0..1
    they are treated as fractions of the frame size.
    """

    if spec is None:
        return SubjectPrompt()
    s = spec.strip()
    if not s or s.lower() == "auto":
        return SubjectPrompt()
    try:
        if s.startswith("{"):
            return SubjectPrompt.model_validate(json.loads(s))
        kind, _, rest = s.partition(":")
        kind = kind.strip().lower()
        rest = rest.strip()
        if kind == "box":
            vals = [float(v) for v in re.findall(_NUM, rest)]
            if len(vals) != 4:
                raise ValueError("box needs four numbers: x1,y1,x2,y2")
            norm = all(0.0 <= v <= 1.0 for v in vals)
            return SubjectPrompt(mode="box", box=(vals[0], vals[1], vals[2], vals[3]), normalized=norm)
        if kind in ("points", "point", "clicks", "click"):
            pts: List[PointPrompt] = []
            for chunk in [c for c in rest.split(";") if c.strip()]:
                parts = [p.strip() for p in chunk.split(",") if p.strip()]
                if len(parts) < 2:
                    raise ValueError(f"cannot parse point {chunk!r}; expected x,y[,+|-]")
                label = 1
                if len(parts) >= 3:
                    flag = parts[2].lower()
                    if flag in ("-", "0", "neg", "negative", "n"):
                        label = 0
                    elif flag in ("+", "1", "pos", "positive", "p"):
                        label = 1
                    else:
                        raise ValueError(f"unknown point label {parts[2]!r}; use + or -")
                pts.append(PointPrompt(x=float(parts[0]), y=float(parts[1]), label=label))
            norm = bool(pts) and all(0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0 for p in pts)
            return SubjectPrompt(mode="points", points=pts, normalized=norm)
        if kind == "text":
            return SubjectPrompt(mode="text", text=rest)
        if kind == "mask":
            return SubjectPrompt(mode="mask", mask_path=rest)
    except ConfigError:
        raise
    except Exception as exc:  # pydantic ValidationError, ValueError, JSON errors
        raise ConfigError(
            f"Invalid --subject value {spec!r}: {exc}",
            hint="Use auto | box:x1,y1,x2,y2 | points:x,y;x,y,- | text:<description> | mask:<path>",
        ) from exc
    raise ConfigError(
        f"Invalid --subject value {spec!r}",
        hint="Use auto | box:x1,y1,x2,y2 | points:x,y;x,y,- | text:<description> | mask:<path>",
    )


# =========================================================================== matte refinement


@dataclass(frozen=True)
class RefinementParams:
    """Fully resolved refinement parameters (no ``None``).  Pixel quantities are expressed for a
    1080-line frame and scaled with :meth:`px` unless ``resolution_scaling`` is false."""

    enabled: bool = True
    mask_to_alpha: str = "auto"            # auto | guided_filter | none  (auto: guided filter for mask models only)
    guided_radius: float = 6.0
    guided_eps: float = 1e-3
    remove_small_artifacts: bool = True
    min_artifact_area: float = 0.0005      # fraction of the frame area
    fill_holes: bool = True
    max_hole_area: float = 0.0005          # fraction of the frame area
    erode: float = 0.0
    dilate: float = 0.0
    edge_feather: float = 0.0
    gaussian_smoothing: float = 0.0        # sigma
    temporal_smoothing: float = 0.3        # 0 = off .. 1 = maximum
    motion_aware: bool = True              # gate temporal smoothing by local image motion (no trails on moving edges)
    motion_compensation: bool = False      # warp the previous matte along optical flow before blending (slower)
    preserve_hair: bool = True
    preserve_motion_blur: bool = True
    black_point: float = 0.015             # alpha <= black_point becomes exactly 0
    white_point: float = 0.985             # alpha >= white_point becomes exactly 1
    decontaminate: float = 1.0             # 0 = strict source pixels at edges, 1 = full foreground colour estimation
    resolution_scaling: bool = True

    def px(self, value: float, width: int, height: int) -> float:
        """Scale a 1080p-referenced pixel quantity to the actual frame size."""

        if not self.resolution_scaling:
            return float(value)
        return float(value) * (min(width, height) / 1080.0)


REFINEMENT_PRESETS: Dict[str, RefinementParams] = {
    "off": RefinementParams(
        enabled=False, mask_to_alpha="none", remove_small_artifacts=False, fill_holes=False,
        temporal_smoothing=0.0, motion_aware=False, preserve_hair=False, preserve_motion_blur=False,
        black_point=0.0, white_point=1.0, decontaminate=0.0,
    ),
    "light": RefinementParams(
        min_artifact_area=0.0002, max_hole_area=0.0002, temporal_smoothing=0.15,
        black_point=0.01, white_point=0.99, decontaminate=0.6,
    ),
    "balanced": RefinementParams(),
    "strong": RefinementParams(
        min_artifact_area=0.002, max_hole_area=0.002, edge_feather=1.0, temporal_smoothing=0.55,
        motion_compensation=True, black_point=0.03, white_point=0.97, decontaminate=1.0,
    ),
}


class RefinementConfig(BaseModel):
    """User facing refinement settings. Every ``None`` is filled from ``preset``.

    Pixel quantities (erode, dilate, edge_feather, gaussian_smoothing, guided_radius) are expressed
    for a 1080-line frame and scaled to the real resolution unless ``resolution_scaling`` is false.
    """

    model_config = ConfigDict(extra="forbid")

    preset: Optional[Literal["off", "light", "balanced", "strong"]] = Field(
        None, description="Base preset. None = chosen by the quality preset."
    )
    mask_to_alpha: Optional[Literal["auto", "guided_filter", "none"]] = None
    guided_radius: Optional[float] = Field(None, ge=1.0, le=64.0)
    guided_eps: Optional[float] = Field(None, gt=0.0, le=1.0)
    remove_small_artifacts: Optional[bool] = None
    min_artifact_area: Optional[float] = Field(None, ge=0.0, le=0.5)
    fill_holes: Optional[bool] = None
    max_hole_area: Optional[float] = Field(None, ge=0.0, le=0.5)
    erode: Optional[float] = Field(None, ge=0.0, le=64.0)
    dilate: Optional[float] = Field(None, ge=0.0, le=64.0)
    edge_feather: Optional[float] = Field(None, ge=0.0, le=64.0)
    gaussian_smoothing: Optional[float] = Field(None, ge=0.0, le=32.0)
    temporal_smoothing: Optional[float] = Field(None, ge=0.0, le=0.95)
    motion_aware: Optional[bool] = None
    motion_compensation: Optional[bool] = None
    preserve_hair: Optional[bool] = None
    preserve_motion_blur: Optional[bool] = None
    black_point: Optional[float] = Field(None, ge=0.0, le=0.45)
    white_point: Optional[float] = Field(None, ge=0.55, le=1.0)
    decontaminate: Optional[float] = Field(None, ge=0.0, le=1.0)
    resolution_scaling: Optional[bool] = None

    def resolve(self, quality: "QualityPreset" = QualityPreset.BALANCED) -> RefinementParams:
        preset_name = self.preset or QUALITY_PRESETS[quality].refinement_preset
        base = REFINEMENT_PRESETS[preset_name]
        values = {k: getattr(base, k) for k in base.__dataclass_fields__}
        for name in type(self).model_fields:
            if name == "preset":
                continue
            v = getattr(self, name)
            if v is not None:
                values[name] = v
        # Any explicit non-preset value switches refinement on, except with an explicit 'off' preset
        # where individual tools can still be re-enabled one by one.
        explicit = [n for n in type(self).model_fields if n != "preset" and getattr(self, n) is not None]
        if explicit:
            values["enabled"] = True
        return RefinementParams(**values)


# =========================================================================== encoding / exports


class EncodingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    codec: Optional[VideoCodec] = Field(
        None, description="None = automatic: from the output extension (H.264 for mp4/mov/mkv, VP9 for webm; "
        "ProRes 4444 / VP9 / FFV1 when the background is transparent)."
    )
    container: Optional[Literal["mp4", "mov", "mkv", "webm"]] = Field(
        None, description="None = from the output file extension, else the codec's default container."
    )
    crf: Optional[int] = Field(None, ge=0, le=63, description="Constant quality (x264/x265/VP9 CRF, NVENC CQ). None = from quality preset.")
    bitrate: Optional[str] = Field(None, description="Target video bitrate such as '20M' or '8000k'. Overrides crf.")
    max_bitrate: Optional[str] = None
    preset: Optional[str] = Field(None, description="Encoder speed preset (x264/x265: ultrafast..veryslow, NVENC: p1..p7).")
    pix_fmt: Optional[str] = Field(None, description="Output pixel format. None = same as source when the codec supports it.")
    resolution: str = Field("source", description="'source' (default, no scaling), '<N>p' (e.g. 1080p) or 'WxH'.")
    preserve_audio: bool = True
    audio_codec: Literal["auto", "copy", "aac", "pcm", "opus", "flac"] = Field(
        "auto", description="auto = stream copy when the container allows it, otherwise re-encode (reported as a warning)."
    )
    audio_bitrate: str = "256k"
    faststart: bool = True
    extra_args: List[str] = Field(default_factory=list, description="Extra FFmpeg output options for the video encoder.")

    @field_validator("bitrate", "max_bitrate", "audio_bitrate")
    @classmethod
    def _bitrate(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not re.fullmatch(r"\d+(?:\.\d+)?[kKmM]?", v.strip()):
            raise ValueError("bitrate must look like 8000k, 20M or 5000000")
        return v.strip()

    @field_validator("resolution")
    @classmethod
    def _resolution(cls, v: str) -> str:
        s = v.strip().lower()
        if s in ("source", "original", ""):
            return "source"
        if re.fullmatch(r"\d{2,5}p", s) or re.fullmatch(r"\d{2,5}x\d{2,5}", s):
            return s
        raise ValueError("resolution must be 'source', '<N>p' (e.g. 1080p) or 'WxH' (e.g. 1920x1080)")


class ExportOptions(BaseModel):
    """Optional extra deliverables written next to the main output (``<stem>_<suffix>``)."""

    model_config = ConfigDict(extra="forbid")

    alpha_matte: bool = Field(False, description="Grayscale matte video (white = subject).")
    png_sequence: bool = Field(False, description="RGBA PNG image sequence in a folder.")
    prores_4444: bool = Field(False, description="Transparent Apple ProRes 4444 .mov with alpha.")
    green_screen: bool = Field(False, description="Additional green-screen video when the main output uses another background.")
    audio: bool = Field(False, description="Original audio track(s), stream-copied without re-encoding.")
    matte_preview: bool = Field(False, description="Low resolution 2x2 preview video: original / subject overlay / matte / composite.")
    alpha_matte_codec: Optional[VideoCodec] = Field(None, description="Codec of the matte video. None = same family as the main output.")
    png_bit_depth: Literal[8, 16] = 8
    png_compression: int = Field(3, ge=0, le=9)
    alpha_mode: Literal["straight", "premultiplied"] = Field(
        "straight", description="Alpha association for transparent outputs (ProRes 4444, PNG, VP9, FFV1)."
    )

    def any_enabled(self) -> bool:
        return any((self.alpha_matte, self.png_sequence, self.prores_4444, self.green_screen, self.audio, self.matte_preview))


# =========================================================================== quality presets


@dataclass(frozen=True)
class QualityPresetInfo:
    name: QualityPreset
    label: str
    description: str
    refinement_preset: str
    #: generic encoder effort index 0..3 mapped per codec by app.encoding.profiles
    encoder_effort: int
    #: inference working-size hint: long-side cap for models that can run below source resolution
    #: (None = no cap).  The matte is always produced at full source resolution.
    inference_max_side: Optional[int]


QUALITY_PRESETS: Dict[QualityPreset, QualityPresetInfo] = {
    QualityPreset.FAST: QualityPresetInfo(
        QualityPreset.FAST, "Fast", "Lightest model variant, minimal refinement, fast encode. For previews and drafts.",
        "light", 0, 1280,
    ),
    QualityPreset.BALANCED: QualityPresetInfo(
        QualityPreset.BALANCED, "Balanced", "Standard model variant with balanced refinement. Good default.",
        "balanced", 1, 1920,
    ),
    QualityPreset.HIGH: QualityPresetInfo(
        QualityPreset.HIGH, "High", "Stronger model variant, full-detail inference, slower high quality encode.",
        "balanced", 2, 2560,
    ),
    QualityPreset.MAX: QualityPresetInfo(
        QualityPreset.MAX, "Maximum", "Largest model variant that fits in memory, native-resolution inference, near-lossless encode.",
        "balanced", 3, None,
    ),
}


# =========================================================================== processing config


class ProcessingConfig(BaseModel):
    """Complete description of one processing run."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    model: str = Field("rvm", description="Model adapter id: rvm | sam2 | matanyone | sam2_rvm | <plugin id>.")
    variant: Optional[str] = Field(None, description="Model variant. None = chosen by the quality preset.")
    quality: QualityPreset = QualityPreset.BALANCED
    device: str = Field("auto", description="auto | cuda | cuda:N | mps | cpu")
    precision: Literal["auto", "fp32", "fp16"] = "auto"
    batch_size: Optional[int] = Field(None, ge=1, le=64, description="Frames per inference batch. None = automatic.")
    background: BackgroundMode = BackgroundMode.GREEN
    background_color: Optional[str] = Field(None, description="#RRGGBB, required when background = custom.")
    subject: SubjectPrompt = Field(default_factory=SubjectPrompt)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    encoding: EncodingConfig = Field(default_factory=EncodingConfig)
    exports: ExportOptions = Field(default_factory=ExportOptions)
    model_options: Dict[str, Any] = Field(default_factory=dict, description="Adapter specific options (see docs/MODELS.md).")
    max_frames: Optional[int] = Field(None, ge=1, description="Process only the first N frames (drafts / tests). Output is then shorter than the source by design.")
    resume: bool = Field(True, description="Cache raw mattes so an interrupted run can continue and settings can be re-rendered without re-running the model.")
    keep_work_dir: bool = False
    overwrite: bool = False
    allow_cpu_fallback: bool = Field(True, description="Fall back to the CPU when the GPU cannot be initialised or runs out of memory at the smallest batch.")
    tolerate_decode_errors: bool = False
    strict_verification: bool = Field(False, description="Treat failed quality checks as an error (the output is kept with a .rejected suffix).")
    commercial_use: Optional[bool] = Field(None, description="Refuse models whose licence forbids commercial use. None = installation default.")
    download_models: Optional[bool] = Field(None, description="Allow downloading missing weights for this run. None = installation default.")

    @field_validator("background_color")
    @classmethod
    def _hex(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        s = v.strip()
        if not re.fullmatch(r"#?[0-9a-fA-F]{6}", s):
            raise ValueError("background_color must be a hex colour like #00FF00")
        return "#" + s.lstrip("#").upper()

    @model_validator(mode="after")
    def _check(self) -> "ProcessingConfig":
        if self.background is BackgroundMode.CUSTOM and not self.background_color:
            raise ValueError("background 'custom' needs background_color (#RRGGBB)")
        codec = self.encoding.codec
        if self.background is BackgroundMode.TRANSPARENT and codec is not None and not CODECS[codec].supports_alpha:
            raise ValueError(
                f"background 'transparent' needs a codec with an alpha channel "
                f"(prores_4444, ffv1 or vp9), not {codec.value}"
            )
        if codec is not None and self.encoding.container is not None:
            if self.encoding.container not in CODECS[codec].containers:
                raise ValueError(
                    f"codec {codec.value} cannot be stored in .{self.encoding.container}; "
                    f"allowed: {', '.join(CODECS[codec].containers)}"
                )
        dev = self.device.strip().lower()
        if not re.fullmatch(r"auto|cpu|mps|cuda(:\d+)?", dev):
            raise ValueError("device must be auto, cpu, mps, cuda or cuda:N")
        self.device = dev
        return self

    # -- helpers -------------------------------------------------------------------------
    def background_rgb(self) -> Optional[Tuple[int, int, int]]:
        """8-bit R'G'B' of the background, or None for transparent output."""

        if self.background is BackgroundMode.TRANSPARENT:
            return None
        if self.background is BackgroundMode.CUSTOM:
            s = (self.background_color or "#00FF00").lstrip("#")
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
        return BACKGROUND_COLORS[self.background]

    def refinement_params(self) -> RefinementParams:
        return self.refinement.resolve(self.quality)

    def fingerprint_payload(self) -> Dict[str, Any]:
        """The subset of settings that determines the *raw* matte (used to validate cached mattes)."""

        return {
            "model": self.model,
            "variant": self.variant,
            "quality": self.quality.value,
            "precision": self.precision,
            "subject": self.subject.model_dump(mode="json"),
            "model_options": self.model_options,
        }


def resolve_codec_and_container(
    config: ProcessingConfig, output_path: Optional[Path], source_bit_depth: int = 8
) -> Tuple[VideoCodec, str]:
    """Pick the concrete video codec and container for the main output.

    Order of precedence: explicit ``encoding.codec`` / ``encoding.container`` -> output file
    extension -> defaults.  Raises :class:`ConfigError` for impossible combinations.

    Automatic codec choice: H.264 for 8-bit sources and H.265 for sources deeper than 8 bits (10-bit
    H.264 is poorly supported by players, 10-bit HEVC is not), VP9 for .webm; for a transparent
    background ProRes 4444 (.mov), VP9 (.webm) or FFV1 (.mkv).
    """

    enc = config.encoding
    ext_container = None
    if output_path is not None and output_path.suffix:
        ext_container = EXTENSION_CONTAINERS.get(output_path.suffix.lower())
        if ext_container is None:
            raise ConfigError(
                f"Unsupported output extension {output_path.suffix!r}.",
                hint="Use .mp4, .mov, .mkv or .webm.",
            )
    transparent = config.background is BackgroundMode.TRANSPARENT
    container = enc.container or ext_container
    codec = enc.codec
    if codec is None:
        if transparent:
            codec = {
                None: VideoCodec.PRORES_4444,
                "mov": VideoCodec.PRORES_4444,
                "webm": VideoCodec.VP9,
                "mkv": VideoCodec.FFV1,
            }.get(container)
            if codec is None:
                raise ConfigError(
                    f"Transparent output cannot be stored in .{container}.",
                    hint="Use a .mov (ProRes 4444), .webm (VP9) or .mkv (FFV1) output file.",
                )
        elif container == "webm":
            codec = VideoCodec.VP9
        else:
            codec = VideoCodec.H265 if source_bit_depth > 8 else VideoCodec.H264
    info = CODECS[codec]
    if transparent and not info.supports_alpha:
        raise ConfigError(
            f"Codec {codec.value} has no alpha channel; transparent output needs prores_4444, ffv1 or vp9."
        )
    if container is None:
        container = info.default_container
    if container not in info.containers:
        raise ConfigError(
            f"Codec {codec.value} cannot be stored in a .{container} file.",
            hint=f"Use one of: {', '.join('.' + c for c in info.containers)} or pick another codec.",
        )
    if enc.container and ext_container and enc.container != ext_container:
        raise ConfigError(
            f"Output extension {output_path.suffix!r} does not match container '{enc.container}'."
        )
    return codec, container


def parse_resolution(spec: str, src_width: int, src_height: int) -> Tuple[int, int, bool]:
    """Resolve ``encoding.resolution`` to ``(width, height, resized)`` in coded orientation.

    * ``source``  -> unchanged.
    * ``<N>p``    -> the short side becomes N, aspect ratio kept, even dimensions.
    * ``WxH``     -> exact size (the frame is scaled to fit and padded with the background colour).
    """

    s = spec.strip().lower()
    if s == "source":
        return src_width, src_height, False
    if s.endswith("p"):
        n = int(s[:-1])
        if n < 16:
            raise ConfigError(f"Invalid resolution {spec!r}.")
        short, long_ = (src_height, src_width) if src_height <= src_width else (src_width, src_height)
        if n == short:
            return src_width, src_height, False
        scale = n / short
        new_long = int(round(long_ * scale / 2.0)) * 2
        new_short = int(round(n / 2.0)) * 2
        w, h = (new_long, new_short) if src_height <= src_width else (new_short, new_long)
        return w, h, True
    w_s, h_s = s.split("x")
    w, h = int(w_s), int(h_s)
    if w < 16 or h < 16:
        raise ConfigError(f"Invalid resolution {spec!r}.")
    return w, h, (w, h) != (src_width, src_height)


# =========================================================================== application settings


class AppSettings(BaseModel):
    """Installation level settings."""

    model_config = ConfigDict(extra="ignore")

    models_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "models")
    data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    cache_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / ".cache")
    temp_dir: Optional[Path] = Field(None, description="Scratch directory. None = <data_dir>/tmp.")
    ffmpeg_path: Optional[str] = None
    ffprobe_path: Optional[str] = None
    auto_download_models: bool = Field(False, description="Download missing weights without asking. Off by default: downloads are explicit.")
    commercial_use: bool = Field(False, description="Installation is used commercially: refuse non-commercial models.")
    default_device: str = "auto"
    host: str = "127.0.0.1"
    port: int = 8765
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    plugins: List[str] = Field(default_factory=list, description="Importable modules that register additional model adapters.")
    weight_overrides: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Per weight-file overrides, keyed by file name: {'url': ..., 'sha256': ...} (mirrors, pinned copies).",
    )
    preview_tile_max_side: int = Field(640, ge=160, le=1280)
    decode_queue_frames: int = Field(24, ge=2, le=512)
    encode_queue_frames: int = Field(24, ge=2, le=512)
    max_upload_bytes: Optional[int] = None
    allow_path_import: bool = Field(True, description="Allow registering a local file path instead of uploading (loopback only).")
    stall_dump_seconds: float = Field(
        300.0, ge=0.0,
        description="Web server: when a running job reports no progress for this long, the stacks of all threads are "
        "written to <data_dir>/logs/stall-traces.log (diagnosis of hangs). 0 = off.",
    )
    defaults: Dict[str, Any] = Field(default_factory=dict, description="Default ProcessingConfig overrides.")

    @property
    def tmp(self) -> Path:
        return self.temp_dir or (self.data_dir / "tmp")

    def ensure_dirs(self) -> None:
        for p in (self.models_dir, self.data_dir, self.cache_dir, self.tmp):
            p.mkdir(parents=True, exist_ok=True)

    def default_processing_config(self) -> ProcessingConfig:
        try:
            return ProcessingConfig.model_validate(self.defaults or {})
        except Exception as exc:
            raise ConfigError(f"Invalid 'defaults' section in the configuration file: {exc}") from exc


_ENV_MAP = {
    "VMT_MODELS_DIR": "models_dir",
    "VMT_DATA_DIR": "data_dir",
    "VMT_CACHE_DIR": "cache_dir",
    "VMT_TEMP_DIR": "temp_dir",
    "VMT_FFMPEG": "ffmpeg_path",
    "VMT_FFPROBE": "ffprobe_path",
    "VMT_AUTO_DOWNLOAD": "auto_download_models",
    "VMT_COMMERCIAL_USE": "commercial_use",
    "VMT_DEVICE": "default_device",
    "VMT_HOST": "host",
    "VMT_PORT": "port",
    "VMT_LOG_LEVEL": "log_level",
    "VMT_LOG_FILE": "log_file",
    "VMT_STALL_DUMP_SECONDS": "stall_dump_seconds",
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except OSError as exc:
        raise ConfigError(f"Cannot read configuration file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Configuration file {path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file {path} must contain a mapping at the top level.")
    return data


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_settings(config_path: Optional[os.PathLike] = None, **overrides: Any) -> AppSettings:
    """Load settings: built-in defaults < configs/default.yaml < user file < VMT_* env < overrides.

    Relative paths inside configuration files are resolved against the project root.
    """

    data: Dict[str, Any] = {}
    default_file = PROJECT_ROOT / "configs" / "default.yaml"
    if default_file.is_file():
        data = _deep_merge(data, _load_yaml(default_file))
    user_file = config_path or os.environ.get("VMT_CONFIG")
    if user_file:
        p = Path(user_file)
        if not p.is_file():
            raise ConfigError(f"Configuration file not found: {p}")
        data = _deep_merge(data, _load_yaml(p))
    for env, key in _ENV_MAP.items():
        if os.environ.get(env) not in (None, ""):
            data[key] = os.environ[env]
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    for key in ("models_dir", "data_dir", "cache_dir", "temp_dir", "log_file"):
        v = data.get(key)
        if v:
            p = Path(os.path.expandvars(os.path.expanduser(str(v))))
            data[key] = p if p.is_absolute() else (PROJECT_ROOT / p)
    try:
        return AppSettings.model_validate(data)
    except Exception as exc:
        raise ConfigError(f"Invalid application settings: {exc}") from exc


def apply_cache_environment(settings: AppSettings) -> None:
    """Point third-party caches and temp files at the application's own directories.

    Libraries such as torch.hub, torchvision and huggingface_hub otherwise write into the user
    profile; redirecting them keeps every download inside the configured directories.  Variables the
    user already set are respected.
    """

    settings.ensure_dirs()
    os.environ.setdefault("TORCH_HOME", str(settings.cache_dir / "torch"))
    os.environ.setdefault("HF_HOME", str(settings.cache_dir / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
