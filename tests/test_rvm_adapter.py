"""Robust Video Matting adapter: contract tests against the real, official weights.

Skipped automatically when the weights are not installed
(``python main.py models download rvm``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from app.core.config import QualityPreset, load_settings
from app.core.errors import ConfigError
from app.models import registry
from app.models.base import MatteOutput, SequenceInfo

pytestmark = pytest.mark.models


def _create(variant="mobilenetv3", device="cpu", precision="auto", **options):
    settings = load_settings()
    info = registry.get_model_info("rvm", settings)
    if not registry.weights_manager(settings).is_ready("rvm", info.variant(variant)):
        pytest.skip(f"RVM weights for variant '{variant}' are not installed")
    return registry.create_model(
        "rvm", settings=settings, device=torch.device(device), variant=variant,
        quality=QualityPreset.BALANCED, precision=precision, options=options,
    )


@pytest.fixture()
def rvm_cpu():
    model = _create()
    model.load()
    yield model
    model.unload()


def _frames(n, h=144, w=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand((n, 3, h, w), generator=g)


def test_registry_metadata():
    info = registry.get_model_info("rvm")
    assert info.capabilities.alpha_type == "alpha"
    assert info.capabilities.subject_domain == "human"
    assert not info.capabilities.native_prompts and not info.capabilities.tracks_single_subject
    assert info.license.commercial_use == "conditional" and "GPL" in info.license.name
    for variant in info.variants:
        for w in variant.weights:
            assert w.url.startswith("https://github.com/PeterL1n/RobustVideoMatting/releases/download/")
            assert w.sha256 and len(w.sha256) == 64 and w.size_bytes
    assert info.variant_for_quality(QualityPreset.FAST) == "mobilenetv3"
    assert info.variant_for_quality(QualityPreset.MAX) == "resnet50"
    assert info.limitations


def test_output_contract(rvm_cpu):
    rvm_cpu.start_sequence(SequenceInfo(width=256, height=144, fps=30.0))
    out = rvm_cpu.process(_frames(3), [0, 1, 2])
    assert isinstance(out, MatteOutput) and not out.is_mask
    assert out.alpha.shape == (3, 1, 144, 256) and out.alpha.dtype == torch.float32
    assert float(out.alpha.min()) >= 0.0 and float(out.alpha.max()) <= 1.0


def test_odd_frame_size(rvm_cpu):
    rvm_cpu.start_sequence(SequenceInfo(width=251, height=139, fps=25.0))
    out = rvm_cpu.process(_frames(2, 139, 251), [0, 1])
    assert out.alpha.shape == (2, 1, 139, 251)


def test_batched_equals_sequential(rvm_cpu):
    frames = _frames(5, seed=3)
    rvm_cpu.start_sequence(SequenceInfo(width=256, height=144, fps=30.0))
    batched = rvm_cpu.process(frames, list(range(5))).alpha
    rvm_cpu.start_sequence(SequenceInfo(width=256, height=144, fps=30.0))
    single = torch.cat([rvm_cpu.process(frames[i : i + 1], [i]).alpha for i in range(5)])
    assert torch.allclose(batched, single, atol=2e-4)
    # and split across two calls: the recurrent state must carry over between process() calls
    rvm_cpu.start_sequence(SequenceInfo(width=256, height=144, fps=30.0))
    split = torch.cat([rvm_cpu.process(frames[:2], [0, 1]).alpha, rvm_cpu.process(frames[2:], [2, 3, 4]).alpha])
    assert torch.allclose(batched, split, atol=2e-4)


def test_temporal_state_is_used_and_reset(rvm_cpu):
    frames = _frames(4, seed=5)
    rvm_cpu.start_sequence(SequenceInfo(width=256, height=144, fps=30.0))
    rvm_cpu.process(frames[:3], [0, 1, 2])
    with_history = rvm_cpu.process(frames[3:], [3]).alpha
    rvm_cpu.start_sequence(SequenceInfo(width=256, height=144, fps=30.0))   # reset
    without_history = rvm_cpu.process(frames[3:], [0]).alpha
    assert not torch.allclose(with_history, without_history, atol=1e-4)


def test_process_is_retry_safe(rvm_cpu, monkeypatch):
    frames = _frames(4, seed=7)
    rvm_cpu.start_sequence(SequenceInfo(width=256, height=144, fps=30.0))
    rvm_cpu.process(frames[:1], [0])
    reference_state = [r.clone() for r in rvm_cpu._rec]

    real_call = rvm_cpu._call
    calls = {"n": 0}

    def flaky(src, rec, ratio):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory. (simulated)")
        return real_call(src, rec, ratio)

    monkeypatch.setattr(rvm_cpu, "_call", flaky)
    with pytest.raises(RuntimeError):
        rvm_cpu.process(frames[1:], [1, 2, 3])
    for before, after in zip(reference_state, rvm_cpu._rec):
        assert torch.equal(before, after)                      # state untouched by the failed call
    retried = torch.cat([rvm_cpu.process(frames[i : i + 1], [i]).alpha for i in (1, 2, 3)])

    monkeypatch.setattr(rvm_cpu, "_call", real_call)
    rvm_cpu.start_sequence(SequenceInfo(width=256, height=144, fps=30.0))
    clean = torch.cat([rvm_cpu.process(frames[i : i + 1], [i]).alpha for i in range(4)])[1:]
    assert torch.allclose(retried, clean, atol=2e-4)


def test_size_change_resets_state(rvm_cpu):
    rvm_cpu.start_sequence(SequenceInfo(width=256, height=144, fps=30.0))
    rvm_cpu.process(_frames(2), [0, 1])
    out = rvm_cpu.process(_frames(1, 160, 288), [2])            # must not crash on stale state shapes
    assert out.alpha.shape == (1, 1, 160, 288)


def test_reduce_memory_lowers_internal_resolution(rvm_cpu):
    before = rvm_cpu._ratio(1080, 1920)
    assert rvm_cpu.reduce_memory() is True
    assert rvm_cpu._ratio(1080, 1920) < before
    steps = 0
    while rvm_cpu.reduce_memory():
        steps += 1
        assert steps < 20
    assert rvm_cpu._internal_side >= 256                       # never below the supported minimum
    out = rvm_cpu.process(_frames(1), [0])
    assert out.alpha.shape == (1, 1, 144, 256)


def test_downsample_ratio_option_and_validation():
    model = _create(downsample_ratio=0.4)
    assert model._ratio(1080, 1920) == pytest.approx(0.4)
    with pytest.raises(ConfigError):
        _create(downsample_ratio=1.5)


def test_runtime_summary(rvm_cpu):
    summary = rvm_cpu.runtime_summary()
    assert summary["model"] == "rvm" and summary["variant"] == "mobilenetv3"
    assert summary["precision"] == "fp32" and summary["dtype"] == "float32"    # no half precision on the CPU


@pytest.mark.gpu
def test_cuda_fp16_matches_cpu_fp32():
    frames = _frames(3, 288, 512, seed=11)
    cpu = _create()
    cpu.load()
    cpu.start_sequence(SequenceInfo(width=512, height=288, fps=30.0))
    ref = cpu.process(frames, [0, 1, 2]).alpha
    cpu.unload()

    gpu = _create(device="cuda")
    assert gpu.precision == "fp16"
    gpu.load()
    try:
        gpu.start_sequence(SequenceInfo(width=512, height=288, fps=30.0))
        out = gpu.process(frames.cuda(), [0, 1, 2]).alpha
        assert out.device.type == "cuda" and out.dtype == torch.float32
        # half precision on a random-noise input: small numerical differences are expected
        assert float((out.cpu() - ref).abs().mean()) < 0.02
        # a permuted (channels-last strided) input must be accepted and give the same result
        permuted = frames.permute(0, 2, 3, 1).contiguous().cuda().permute(0, 3, 1, 2)
        gpu.start_sequence(SequenceInfo(width=512, height=288, fps=30.0))
        again = gpu.process(permuted, [0, 1, 2]).alpha
        assert torch.allclose(out, again, atol=1e-3)
    finally:
        gpu.unload()
