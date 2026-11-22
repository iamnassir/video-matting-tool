"""A deterministic, weight-free matting adapter used ONLY by the test-suite.

It is not a real model and is never registered outside tests: it keys the magenta subject of the
synthetic test clips (see ``tests/helpers/media.py``).  It exists so that pipeline, API and CLI tests
can exercise the complete engine in seconds on any machine, without a GPU or downloaded weights.
Real models are covered by the tests marked ``models`` / ``gpu``.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

from app.models.base import (
    LicenseInfo,
    MatteOutput,
    MattingModel,
    ModelCapabilities,
    ModelInfo,
    ModelVariant,
    SequenceInfo,
)

TEST_MODEL_ID = "test_chroma"


def chroma_alpha(frames: torch.Tensor) -> torch.Tensor:
    """Matte of the magenta test subject: large where min(R, B) exceeds G."""

    r, g, b = frames[:, 0:1], frames[:, 1:2], frames[:, 2:3]
    return ((torch.minimum(r, b) - g - 0.10) / 0.45).clamp(0.0, 1.0)


class ChromaTestModel(MattingModel):
    info = ModelInfo(
        id=TEST_MODEL_ID,
        name="Test chroma keyer (test-suite only)",
        description="Deterministic keyer for the synthetic magenta test subject. Not a real model.",
        license=LicenseInfo(name="n/a (test fixture)", url="", commercial_use="allowed"),
        capabilities=ModelCapabilities(alpha_type="alpha", temporal="none", subject_domain="any"),
        variants=(ModelVariant(id="default", label="default", weights=()),),
        default_variant="default",
        limitations=("Test fixture: only keys pure magenta.",),
    )

    #: set to an integer N to raise a simulated CUDA out-of-memory error whenever a batch larger
    #: than N frames is submitted (lets tests exercise the automatic batch reduction)
    oom_above_batch = None
    #: frame indices processed, in call order (for assertions)
    calls: List[List[int]]

    def load(self) -> None:
        self.calls = []
        self.sequences: List[SequenceInfo] = []
        self.loaded = True

    def start_sequence(self, seq: SequenceInfo) -> None:
        self.sequences.append(seq)

    def process(self, frames: torch.Tensor, frame_indices: Sequence[int]) -> MatteOutput:
        if self.oom_above_batch is not None and frames.shape[0] > self.oom_above_batch:
            raise RuntimeError("CUDA out of memory. (simulated by the test model)")
        self.calls.append([int(i) for i in frame_indices])
        alpha = chroma_alpha(frames.float())
        presence = [float(a.max() > 0.5) for a in alpha]
        return MatteOutput(alpha=alpha, presence=presence)


class MaskTestModel(ChromaTestModel):
    """Same keyer but reports a hard mask, to exercise the mask-to-alpha refinement path."""

    info = ModelInfo(
        id="test_mask",
        name="Test mask model (test-suite only)",
        description="Binary variant of the test keyer. Not a real model.",
        license=LicenseInfo(name="n/a (test fixture)", url="", commercial_use="allowed"),
        capabilities=ModelCapabilities(
            alpha_type="mask", temporal="none", subject_domain="any", max_batch_size=1,
            native_prompts=frozenset({"points", "box", "mask"}), tracks_single_subject=True,
        ),
        variants=(ModelVariant(id="default", label="default", weights=()),),
        default_variant="default",
    )

    def process(self, frames: torch.Tensor, frame_indices: Sequence[int]) -> MatteOutput:
        out = super().process(frames, frame_indices)
        out.alpha = (out.alpha > 0.5).float()
        out.is_mask = True
        return out
