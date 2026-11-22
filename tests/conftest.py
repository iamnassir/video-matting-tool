"""Shared pytest fixtures.

All media used by the tests is synthesised into pytest's base temp directory, which
``pyproject.toml`` places inside the project (``.cache/pytest_tmp``) so nothing is written to the
system temp folder.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.helpers import media  # noqa: E402

requires_ffmpeg = pytest.mark.skipif(
    media.FFMPEG is None or media.FFPROBE is None, reason="ffmpeg / ffprobe not found on PATH"
)


def pytest_collection_modifyitems(config, items):
    """Skip ``gpu`` tests automatically when no CUDA device is usable."""

    if not any("gpu" in item.keywords for item in items):
        return
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if has_cuda:
        return
    skip = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


# --------------------------------------------------------------------------- real footage (optional)

import json as _json
import os as _os

#: Folder with real test clips and their ``manifest.json`` (not part of the repository; fetch with
#: ``python scripts/download_test_assets.py`` or point VMT_TEST_MEDIA at your own copy).
REAL_MEDIA_DIR = Path(_os.environ.get("VMT_TEST_MEDIA", str(ROOT / "tests" / "data" / "real")))


class RealMedia:
    """Access to the optional real-footage clips. Tests that need a clip are skipped when it is absent."""

    def __init__(self, root: Path) -> None:
        self.root = root
        try:
            self.manifest = _json.loads((root / "manifest.json").read_text(encoding="utf-8")).get("clips", {})
        except (OSError, ValueError):
            self.manifest = {}

    def path(self, clip_id: str) -> Path:
        entry = self.manifest.get(clip_id)
        if entry is None:
            pytest.skip(f"real test clip '{clip_id}' is not listed in {self.root / 'manifest.json'}")
        p = self.root / entry["path"]
        if not p.is_file():
            pytest.skip(f"real test clip '{clip_id}' is not downloaded ({p})")
        return p

    def info(self, clip_id: str) -> dict:
        self.path(clip_id)
        return dict(self.manifest[clip_id])

    def file(self, relative: str) -> Path:
        p = self.root / relative
        if not p.is_file():
            pytest.skip(f"real test file '{relative}' is not available")
        return p


@pytest.fixture(scope="session")
def real_media() -> RealMedia:
    return RealMedia(REAL_MEDIA_DIR)


# --------------------------------------------------------------------------- settings


@pytest.fixture()
def app_settings(tmp_path):
    """Isolated AppSettings: data / cache / temp in a per-test folder, real project models dir."""

    from app.core.config import AppSettings

    settings = AppSettings(
        models_dir=ROOT / "models",
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        temp_dir=tmp_path / "tmp",
    )
    settings.ensure_dirs()
    return settings


@pytest.fixture()
def test_models():
    """Register the weight-free test adapters for the duration of a test."""

    from app.models import registry
    from tests.helpers.dummy_model import ChromaTestModel, MaskTestModel

    registry.ensure_loaded()
    registry.register_model(ChromaTestModel)
    registry.register_model(MaskTestModel)
    ChromaTestModel.oom_above_batch = None
    yield {"chroma": ChromaTestModel, "mask": MaskTestModel}
    ChromaTestModel.oom_above_batch = None
    registry._REGISTRY.pop(ChromaTestModel.info.id, None)
    registry._REGISTRY.pop(MaskTestModel.info.id, None)


# --------------------------------------------------------------------------- media


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("media")


class Clip:
    """A generated clip plus the facts tests assert against."""

    def __init__(self, path: Path, pts, time_base: Fraction, width: int, height: int, frames: int, **facts):
        self.path = Path(path)
        self.pts = list(pts)
        self.time_base = time_base
        self.width = width
        self.height = height
        self.frames = frames
        self.facts = facts

    def __fspath__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"Clip({self.path.name}, {self.width}x{self.height}, {self.frames} frames)"


def _clip(media_dir: Path, name: str, *, audio: str = "aac", ext: str = "mp4", rotation: int = 0,
          audio_offset: float = 0.0, **video_kwargs) -> Clip:
    if media.FFMPEG is None or media.FFPROBE is None:
        pytest.skip("ffmpeg / ffprobe not found on PATH")
    width = video_kwargs.setdefault("width", 320)
    height = video_kwargs.setdefault("height", 180)
    frames = video_kwargs.setdefault("frames", 36)
    fps = video_kwargs.setdefault("fps", Fraction(30000, 1001))
    tb = video_kwargs.get("time_base") or Fraction(1, fps.numerator if fps.denominator != 1 else fps.numerator * 1000)
    video_kwargs["time_base"] = tb
    out = media_dir / f"{name}.{ext}"
    stage = media_dir / f"{name}_video.{ext}"
    pts = media.write_video(stage, **video_kwargs)
    current = stage
    if audio:
        with_audio = media_dir / f"{name}_av.{ext}"
        media.add_audio(current, with_audio, codec=audio, audio_offset=audio_offset)
        current = with_audio
    if rotation:
        rotated = media_dir / f"{name}_rot.{ext}"
        media.with_rotation(current, rotated, rotation)
        current = rotated
    if out.exists():
        out.unlink()
    current.rename(out)
    for leftover in (stage, media_dir / f"{name}_av.{ext}", media_dir / f"{name}_rot.{ext}"):
        if leftover.exists() and leftover != out:
            leftover.unlink()
    return Clip(out, pts, tb, width, height, frames, audio=audio, rotation=rotation,
                vfr=video_kwargs.get("vfr", False), pix_fmt=video_kwargs.get("pix_fmt", "yuv420p"),
                fps=fps, two_subjects=video_kwargs.get("two_subjects", False))


@pytest.fixture(scope="session")
def clip_cfr(media_dir) -> Clip:
    """H.264 yuv420p, 29.97 fps constant, AAC audio, MP4."""

    return _clip(media_dir, "cfr")


@pytest.fixture(scope="session")
def clip_vfr(media_dir) -> Clip:
    """H.264 with genuinely irregular frame durations (variable frame rate), AAC audio, MP4."""

    return _clip(media_dir, "vfr", vfr=True, time_base=Fraction(1, 90000))


@pytest.fixture(scope="session")
def clip_noaudio_mkv(media_dir) -> Clip:
    """H.264 in Matroska without an audio stream."""

    return _clip(media_dir, "noaudio", audio="", ext="mkv", time_base=Fraction(1, 1000), fps=Fraction(25, 1))


@pytest.fixture(scope="session")
def clip_10bit(media_dir) -> Clip:
    """HEVC yuv420p10le tagged BT.2020 / HLG in MOV with AAC audio."""

    return _clip(
        media_dir, "tenbit", ext="mov", codec="libx265", pix_fmt="yuv420p10le",
        options={"x265-params": "log-level=error"},
        color={"colorspace": "bt2020nc", "color_primaries": "bt2020", "color_trc": "arib-std-b67", "color_range": "tv"},
    )


@pytest.fixture(scope="session")
def clip_rotated(media_dir) -> Clip:
    """Portrait phone-style clip: coded 320x180 with a 90 degree display rotation, AAC audio."""

    return _clip(media_dir, "rotated", rotation=90)


@pytest.fixture(scope="session")
def clip_two_subjects(media_dir) -> Clip:
    """Like clip_cfr but with a second, smaller magenta blob that must not be selected."""

    return _clip(media_dir, "two", two_subjects=True)


@pytest.fixture(scope="session")
def clip_odd_444(media_dir) -> Clip:
    """Odd dimensions and 4:4:4 chroma (exercises plane-size rounding)."""

    return _clip(media_dir, "odd444", width=321, height=179, pix_fmt="yuv444p", audio="")


@pytest.fixture(scope="session")
def clip_webm(media_dir) -> Clip:
    """VP9 + Opus in WebM (millisecond time base)."""

    return _clip(
        media_dir, "web", ext="webm", codec="libvpx-vp9", audio="libopus", time_base=Fraction(1, 1000),
        fps=Fraction(30, 1), options={"crf": "20", "b:v": "0", "deadline": "realtime", "cpu-used": "8"},
    )
