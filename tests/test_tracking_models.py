"""SAM 2, the SAM 2 + RVM hybrid, MatAnyone and the subject-selection helpers.

The metadata tests always run.  The inference tests use the real packages, the real official weights
and real footage on a CUDA device; each one is skipped automatically when a package, a weights file,
the test footage or the GPU is missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from app.core.config import QualityPreset, SubjectPrompt, load_settings
from app.core.errors import LicenseError, SubjectError
from app.core.types import ResolvedSubject
from app.models import registry
from app.models.base import SequenceInfo
from app.models.matanyone.adapter import internal_size, prepare_mask

REAL = Path(__file__).parent / "data" / "real"
JOKER = REAL / "matanyone_sample1" / "test-sample1.mp4"
JOKER_MASK = REAL / "matanyone_sample1" / "test-sample1.png"
COUPLE = REAL / "matanyone_media_sample1_720p" / "test-sample1-720p.mp4"
WOMAN_BOX = (247.0, 186.0, 559.0, 720.0)
MAN_BOX = (651, 135, 1037, 720)
BEDROOM = REAL / "sam2_bedroom" / "bedroom.mp4"
GIRL_BOX = (308.0, 0.0, 518.0, 398.0)


# --------------------------------------------------------------------------- metadata (always run)

def test_registered_models_and_licences():
    ids = {info.id for info in registry.list_model_infos(load_settings())}
    assert {"rvm", "sam2", "sam2_rvm", "matanyone", "sam2_matanyone"} <= ids
    sam2 = registry.get_model_info("sam2")
    assert sam2.license.commercial_use == "allowed" and sam2.capabilities.alpha_type == "mask"
    assert sam2.capabilities.tracks_single_subject and {"points", "box", "mask"} <= set(sam2.capabilities.native_prompts)
    for variant in sam2.variants:
        for w in variant.weights:
            assert w.url.startswith("https://dl.fbaipublicfiles.com/segment_anything_2/")
            assert w.sha256 and len(w.sha256) == 64 and w.size_bytes
    hybrid = registry.get_model_info("sam2_rvm")
    assert hybrid.depends_on == ("sam2", "rvm") and hybrid.capabilities.alpha_type == "alpha"
    assert all(not v.weights for v in hybrid.variants)
    mat = registry.get_model_info("matanyone")
    assert mat.license.commercial_use == "prohibited" and mat.capabilities.needs_initial_mask
    for variant in mat.variants:
        for w in variant.weights:
            assert w.url.startswith("https://github.com/pq-yang/MatAnyone")
            assert w.sha256 and len(w.sha256) == 64 and w.size_bytes
    outline = registry.get_model_info("sam2_matanyone")
    assert outline.depends_on == ("sam2", "matanyone") and all(not v.weights for v in outline.variants)
    assert outline.license.commercial_use == "prohibited"            # the stricter component governs
    assert outline.capabilities.two_pass and outline.capabilities.needs_initial_mask
    assert outline.capabilities.subject_domain == "any" and not sam2.capabilities.two_pass
    for info in (sam2, hybrid, mat, outline):
        assert info.limitations and info.install_hint


def test_matanyone_refused_for_commercial_use():
    with pytest.raises(LicenseError):
        registry.create_model("matanyone", settings=load_settings(), device=torch.device("cpu"), commercial_use=True)


def test_auxiliary_models_listed():
    infos = {i.id: i for i in registry.list_model_infos(load_settings(), include_auxiliary=True)}
    assert infos["detector"].kind == "auxiliary" and infos["text_grounding"].kind == "auxiliary"
    assert "detector" not in {i.id for i in registry.list_model_infos(load_settings())}


def test_matanyone_mask_preparation_and_internal_size():
    mask = np.zeros((90, 160), np.float32)
    mask[20:70, 40:100] = 1.0
    mask[40:44, 60:64] = 0.0                       # a small hole is closed
    out = prepare_mask(mask, 9)
    assert out.dtype == np.float32 and set(np.unique(out)) <= {0.0, 255.0}
    assert out[42, 62] == 255.0 and out[5, 5] == 0.0
    assert internal_size(720, 1280, 480) == (480, 853)
    assert internal_size(1920, 1080, 432) == (768, 432)
    assert internal_size(360, 640, 720) == (360, 640)          # never enlarged


# --------------------------------------------------------------------------- real inference

def _need(model_id: str, variant=None):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    settings = load_settings()
    todo = [model_id]
    while todo:
        info = registry.get_model_info(todo.pop(), settings)
        todo.extend(info.depends_on)
        missing = registry.missing_modules(info)
        if missing:
            pytest.skip(f"package(s) not installed for '{info.id}': {', '.join(missing)}")
        v = info.variant(variant if info.id == model_id else None)
        if v.weights and not registry.weights_manager(settings).is_ready(info.id, v):
            pytest.skip(f"weights of '{info.id}' ({v.id}) are not installed")
    return settings


def _frames(path: Path, count: int, start: int = 0):
    """First *count* frames as uint8 RGB arrays."""

    if not path.is_file():
        pytest.skip(f"test footage missing: {path.name} (python scripts/download_test_assets.py)")
    import av

    out = []
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index >= start:
                out.append(frame.to_ndarray(format="rgb24"))
            if len(out) >= count:
                break
    return out


def _tensor(frame: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(frame).to(device).permute(2, 0, 1)[None].float().div_(255.0)


def _mask_image(path: Path, height: int, width: int) -> np.ndarray:
    from app.tracking.subject import load_mask_image

    if not path.is_file():
        pytest.skip(f"reference mask missing: {path.name}")
    return load_mask_image(str(path), width, height)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0.5, b > 0.5
    return float((a & b).sum()) / max(1.0, float((a | b).sum()))


def _run(model, frames, subject, device, **seq):
    h, w = frames[0].shape[:2]
    model.start_sequence(SequenceInfo(width=w, height=h, fps=30.0, num_frames=len(frames), subject=subject, **seq))
    outputs = []
    with torch.inference_mode():
        for i, frame in enumerate(frames):
            outputs.append(model.process(_tensor(frame, device), [i]))
    model.end_sequence()
    return outputs


def _create(model_id, settings, **kwargs):
    model = registry.create_model(model_id, settings=settings, device=torch.device("cuda"),
                                  quality=QualityPreset.BALANCED, precision="auto", **kwargs)
    with torch.inference_mode():
        model.load()
    return model


@pytest.mark.gpu
@pytest.mark.models
def test_sam2_tracks_the_prompted_subject():
    settings = _need("sam2")
    frames = _frames(BEDROOM, 8)
    model = _create("sam2", settings)
    try:
        subject = ResolvedSubject(frame_index=0, box=GIRL_BOX, source="test:box")
        outputs = _run(model, frames, subject, model.device)
        h, w = frames[0].shape[:2]
        for out in outputs:
            assert out.is_mask and out.alpha.shape == (1, 1, h, w) and out.alpha.dtype == torch.float32
            assert 0.0 <= float(out.alpha.min()) and float(out.alpha.max()) <= 1.0
            assert out.presence and out.presence[0] > 0.5
        first = (outputs[0].alpha[0, 0] > 0.5).cpu().numpy()
        assert 0.02 < first.mean() < 0.5
        x1, y1, x2, y2 = (int(v) for v in GIRL_BOX)
        inside = first[max(0, y1 - 12): y2 + 12, max(0, x1 - 12): x2 + 12].sum()
        assert inside / first.sum() > 0.97                       # nothing outside the prompted box
        # The girl moves quickly, so the masks of two frames need not overlap much; the track must
        # stay one object of similar size.
        last = (outputs[-1].alpha[0, 0] > 0.5).cpu().numpy()
        assert 0.4 < last.sum() / first.sum() < 2.5
    finally:
        model.unload()


@pytest.mark.gpu
@pytest.mark.models
def test_hybrid_keeps_only_the_selected_person():
    settings = _need("sam2_rvm")
    frames = _frames(COUPLE, 8)
    model = _create("sam2_rvm", settings)
    try:
        subject = ResolvedSubject(frame_index=0, box=WOMAN_BOX, source="test:box")
        outputs = _run(model, frames, subject, model.device)
        x1, y1, x2, y2 = MAN_BOX
        for out in outputs[2:]:                                  # RVM needs a frame or two to settle
            assert not out.is_mask
            alpha = out.alpha[0, 0]
            assert float(alpha[y1:y2, x1 + 40: x2].mean()) < 0.02     # the other person is gone
            wx1, wy1, wx2, wy2 = (int(v) for v in WOMAN_BOX)
            assert float(alpha[wy1:wy2, wx1:wx2].mean()) > 0.25       # the selected one is there
            soft = ((alpha > 0.05) & (alpha < 0.95)).float().mean()
            assert float(soft) > 0.0005                              # a soft matte, not a hard mask
    finally:
        model.unload()


@pytest.mark.gpu
@pytest.mark.models
def test_matanyone_propagates_the_mask_and_recovers():
    settings = _need("matanyone")
    frames = _frames(JOKER, 10)
    h, w = frames[0].shape[:2]
    mask = _mask_image(JOKER_MASK, h, w)
    model = _create("matanyone", settings, options={"n_warmup": 4})
    try:
        with pytest.raises(SubjectError):
            model.start_sequence(SequenceInfo(width=w, height=h, fps=30.0, subject=ResolvedSubject(frame_index=0, box=(0, 0, 9, 9))))
        subject = ResolvedSubject(frame_index=0, mask=mask, source="test:mask")
        outputs = _run(model, frames, subject, model.device)
        for out in outputs:
            assert not out.is_mask and out.alpha.shape == (1, 1, h, w) and out.alpha.dtype == torch.float32
            assert 0.0 <= float(out.alpha.min()) and float(out.alpha.max()) <= 1.0
        assert outputs[0].alpha is not outputs[1].alpha
        assert _iou(outputs[0].alpha[0, 0].cpu().numpy(), mask) > 0.9
        assert _iou(outputs[-1].alpha[0, 0].cpu().numpy(), mask) > 0.6
        assert model.runtime_summary()["internal_short_side"]

        # Memory reduction in the middle of a sequence: the matte carries on from the last result.
        model.start_sequence(SequenceInfo(width=w, height=h, fps=30.0, subject=subject))
        with torch.inference_mode():
            before = model.process(_tensor(frames[0], model.device), [0]).alpha.clone()
            side = model.runtime_summary()["internal_short_side"]
            if model.reduce_memory():
                assert model.runtime_summary()["internal_short_side"] < side
            after = model.process(_tensor(frames[1], model.device), [1]).alpha
        assert _iou(before[0, 0].cpu().numpy(), after[0, 0].cpu().numpy()) > 0.85
        model.end_sequence()
    finally:
        model.unload()


@pytest.mark.gpu
@pytest.mark.models
def test_subject_selection_backends():
    settings = _need("sam2")
    from app.tracking import backends
    from app.tracking.subject import SubjectSelector

    ready = backends.available(settings)
    if not ready.get("sam2"):
        pytest.skip("SAM 2 image backend not available")
    frame = _frames(JOKER, 1)[0]
    h, w = frame.shape[:2]
    reference = _mask_image(JOKER_MASK, h, w)
    info = registry.get_model_info("matanyone")
    notes: list = []
    selector = SubjectSelector(settings, torch.device("cuda"), warnings=notes)
    try:
        box = selector.resolve_on_frame(SubjectPrompt(mode="box", box=(380, 58, 782, 720)), frame, info)
        assert box.mask is not None and box.source.startswith("sam2") and _iou(box.mask, reference) > 0.9
        if ready.get("detector"):
            # One click on the waistcoat must select the whole person, not the garment.
            click = selector.resolve_on_frame(SubjectPrompt(mode="points", points=[{"x": 600, "y": 420}]), frame, info)
            assert click.mask is not None and _iou(click.mask, reference) > 0.85
            auto = selector.resolve_on_frame(SubjectPrompt(mode="auto"), frame, info)
            assert auto.mask is not None and auto.label == "person"
    finally:
        selector.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_matanyone_working_size_follows_the_free_memory(tmp_path, monkeypatch):
    from app.models.matanyone.adapter import MatAnyoneAdapter

    def adapter(device="cuda", **options):
        return MatAnyoneAdapter(variant=None, device=torch.device(device), precision="fp16", weights_dir=tmp_path,
                                options=options)

    model = adapter()
    for free, side in ((9000, 1080), (4800, 1080), (3200, 864), (2200, 720), (1500, 576), (1000, 480), (300, 432)):
        monkeypatch.setattr(model, "_free_vram_mb", lambda free=free: float(free))
        assert model._choose_side(2160, 3840, None) == side, free
    monkeypatch.setattr(model, "_free_vram_mb", lambda: 9000.0)
    assert model._choose_side(720, 1280, None) == 720              # never above the source
    assert model._choose_side(2160, 3840, 1280) == 720             # the quality preset caps the long side
    monkeypatch.setattr(model, "_free_vram_mb", lambda: None)
    assert model._choose_side(2160, 3840, None) == 1080            # unknown: assume there is room
    # Memory set aside for a network that runs alongside (the SAM 2 + MatAnyone hybrid).
    shared = adapter(reserve_vram_mb=600)
    monkeypatch.setattr(shared, "_free_vram_mb", lambda: 2500.0)
    assert shared._choose_side(1080, 1920, None) == 576 and shared.runtime_summary()["vram_free_when_chosen_mb"] == 1900
    assert adapter(internal_short_side=864)._choose_side(1080, 1920, None) == 864
    assert adapter(internal_short_side=4000)._choose_side(1080, 1920, None) == 1080
    assert adapter("cpu")._choose_side(1080, 1920, None) == 432


@pytest.mark.gpu
@pytest.mark.models
def test_outline_hybrid_runs_frame_by_frame_and_revises_a_clip():
    """SAM 2 + MatAnyone on real footage: single pass, and the two passes with the revision."""

    settings = _need("sam2_matanyone")
    frames = _frames(JOKER, 10)
    h, w = frames[0].shape[:2]
    mask = _mask_image(JOKER_MASK, h, w)
    model = _create("sam2_matanyone", settings, options={"n_warmup": 4})
    try:
        with pytest.raises(SubjectError):
            model.start_sequence(SequenceInfo(width=w, height=h, fps=30.0, subject=ResolvedSubject(frame_index=0, box=(0, 0, 9, 9))))
        subject = ResolvedSubject(frame_index=0, mask=mask, source="test:mask")

        # 1. frame by frame: the matte never leaves SAM 2's outline (plus the soft margin)
        outputs = _run(model, frames, subject, model.device)
        for out in outputs:
            assert not out.is_mask and out.alpha.shape == (1, 1, h, w) and out.alpha.dtype == torch.float32
            assert out.presence is not None
        assert _iou(outputs[0].alpha[0, 0].cpu().numpy(), mask) > 0.9
        assert _iou(outputs[-1].alpha[0, 0].cpu().numpy(), mask) > 0.6
        summary = model.runtime_summary()
        assert summary["sam2"]["working_size"] and summary["matanyone"]["internal_short_side"]
        assert summary["rigid_outline"]["used"] is False

        # 2. the whole clip: raw mattes first, then the revision
        raw = _run(model, frames, subject, model.device, will_revise=True)
        store = {i: out.alpha[0, 0].cpu().numpy().copy() for i, out in enumerate(raw)}
        saved = {}
        model.revise(list(store), lambda i: store[i], lambda i, matte: saved.__setitem__(i, matte))
        assert sorted(saved) == list(range(len(frames)))
        report = model.runtime_summary()["rigid_outline"]
        assert report["frames"] == len(frames) and report["frames_followed"] >= 1
        for i, matte in saved.items():
            assert matte.shape == (h, w) and matte.dtype == np.float32
            assert 0.0 <= float(matte.min()) and float(matte.max()) <= 1.0
            assert _iou(matte, outputs[i].alpha[0, 0].cpu().numpy()) > 0.9      # close to the frame-by-frame result
        # A second revision has nothing left to work on.
        again = {}
        model.revise(list(store), lambda i: store[i], lambda i, matte: again.__setitem__(i, matte))
        assert again == {}

        # 3. forced off: the frame-by-frame outlines are used
        off = _create("sam2_matanyone", settings, options={"n_warmup": 4, "rigid_outline": "off"})
        try:
            raw = _run(off, frames[:4], subject, off.device, will_revise=True)
            kept = {}
            off.revise(range(4), lambda i: raw[i].alpha[0, 0].cpu().numpy(), lambda i, matte: kept.__setitem__(i, matte))
            assert len(kept) == 4 and off.runtime_summary()["rigid_outline"] == {
                "mode": "off", "used": False, "frames": 4, "frames_followed": 0, "frames_with_the_one_outline": 0}
        finally:
            off.unload()
    finally:
        model.unload()
