"""Lossless bridge between PyAV frames and numpy planes."""

from __future__ import annotations

from fractions import Fraction

import av
import numpy as np
import pytest

from app.core.pixfmt import parse_working_pixfmt
from app.processing.planes import (
    blank_planes,
    frame_to_planes,
    planes_to_frame,
    semiplanar_to_frame,
    to_semiplanar,
    working_format_for,
)
from tests.helpers import media

DIRECT = [
    "yuv420p", "yuv422p", "yuv444p", "yuv440p", "yuv420p10le", "yuv422p10le", "yuv444p10le", "yuv420p12le",
    "yuv444p16le", "yuva420p", "yuva444p10le", "gbrp", "gbrp10le", "gbrap", "gray", "gray10le", "gray16le",
]


def random_planes(name, width, height, seed=0):
    fmt = parse_working_pixfmt(name)
    rng = np.random.default_rng(seed)
    return fmt, [
        rng.integers(0, fmt.max_value + 1, size=fmt.plane_size(i, width, height)).astype(fmt.dtype)
        for i in range(fmt.num_planes)
    ]


def fill(frame, seed=0):
    """Fill every plane buffer of a frame with reproducible random bytes."""

    rng = np.random.default_rng(seed)
    for plane in frame.planes:
        buf = np.frombuffer(plane, dtype=np.uint8)
        buf[:] = rng.integers(0, 256, size=buf.shape, dtype=np.uint8)
    return frame


@pytest.mark.parametrize("name", DIRECT)
@pytest.mark.parametrize("size", [(64, 48), (97, 63), (2, 2)])
def test_direct_round_trip_is_bit_exact(name, size):
    width, height = size
    fmt, planes = random_planes(name, width, height, seed=sum(map(ord, name)))
    frame = planes_to_frame(planes, fmt, pts=1234, time_base=Fraction(1, 90000))
    assert frame.format.name == name and (frame.width, frame.height) == (width, height)
    assert frame.pts == 1234 and frame.time_base == Fraction(1, 90000)
    wf = working_format_for(name)
    assert wf.method == "direct" and wf.exact and wf.fmt == fmt
    back = frame_to_planes(frame)
    assert len(back) == fmt.num_planes
    for a, b in zip(planes, back):
        assert a.dtype == b.dtype and np.array_equal(a, b)
        assert b.flags.c_contiguous and b.flags.owndata           # safe to keep after the frame is gone


def test_planes_to_frame_is_strict():
    fmt, planes = random_planes("yuv420p", 32, 16)
    with pytest.raises(ValueError):
        planes_to_frame(planes[:2], fmt)
    with pytest.raises(ValueError):
        planes_to_frame([planes[0].astype(np.uint16), planes[1], planes[2]], fmt)
    with pytest.raises(ValueError):
        planes_to_frame([planes[0], planes[1][:, :-1], planes[2]], fmt)


def test_full_range_alias_is_a_relabel_not_a_conversion():
    frame = fill(av.VideoFrame(48, 32, "yuvj420p"), seed=3)
    wf = working_format_for("yuvj420p")
    assert wf.fmt.name == "yuv420p" and wf.method == "relabel" and wf.full_range_hint is True and wf.exact
    planes = frame_to_planes(frame)
    for i, plane in enumerate(planes):
        p = frame.planes[i]
        raw = np.frombuffer(p, np.uint8).reshape(p.height, p.line_size)[:, : p.width]
        assert np.array_equal(plane, raw)                          # byte-identical: no range rescaling


@pytest.mark.parametrize(
    "name,target",
    [
        ("nv12", "yuv420p"), ("nv21", "yuv420p"), ("p010le", "yuv420p10le"), ("yuyv422", "yuv422p"),
        ("uyvy422", "yuv422p"), ("rgb24", "gbrp"), ("bgr24", "gbrp"), ("rgba", "gbrap"), ("bgra", "gbrap"),
        ("argb", "gbrap"), ("rgb48le", "gbrp16le"), ("yuv420p10be", "yuv420p10le"), ("yuv444p16be", "yuv444p16le"),
    ],
)
def test_repacking_matches_libswscale(name, target):
    """Pure repacks must give exactly what libswscale gives for the same (matrix-free) conversion."""

    frame = fill(av.VideoFrame(64, 48, name), seed=7)
    if name == "p010le":       # only the high 10 bits of each 16-bit word are meaningful
        for plane in frame.planes:
            buf = np.frombuffer(plane, dtype="<u2")
            buf[:] = buf & 0xFFC0
    if name.endswith("10be"):
        for plane in frame.planes:
            buf = np.frombuffer(plane, dtype=">u2")
            buf[:] = buf & 0x03FF
    wf = working_format_for(name)
    assert wf.fmt.name == target and wf.exact
    ours = frame_to_planes(frame)
    reference = frame_to_planes(frame.reformat(format=target))
    assert len(ours) == len(reference)
    for a, b in zip(ours, reference):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert np.array_equal(a, b)


def test_odd_width_packed_yuv():
    frame = fill(av.VideoFrame(34, 16, "yuyv422"), seed=11)
    ours = frame_to_planes(frame)
    reference = frame_to_planes(frame.reformat(format="yuv422p"))
    for a, b in zip(ours, reference):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("name,pipe", [("yuv420p", "nv12"), ("yuv420p10le", "p010le")])
def test_semiplanar_round_trip(name, pipe):
    fmt, planes = random_planes(name, 64, 48, seed=5)
    packed = to_semiplanar(planes, fmt, pipe)
    frame = semiplanar_to_frame(packed, pipe, pts=7, time_base=Fraction(1, 1000))
    assert frame.format.name == pipe and frame.pts == 7
    back = frame_to_planes(frame)
    for a, b in zip(planes, back):
        assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        to_semiplanar(planes, fmt, "nv16")                         # 4:2:0 data cannot become 4:2:2


def test_heavy_subsampling_is_flagged_inexact():
    wf = working_format_for("yuv411p")
    assert wf.fmt.name == "yuv422p" and not wf.exact and wf.note
    frame = fill(av.VideoFrame(64, 32, "yuv411p"), seed=2)
    planes = frame_to_planes(frame)
    assert [p.shape for p in planes] == [(32, 64), (32, 32), (32, 32)]
    assert np.array_equal(planes[1][:, 0], planes[1][:, 1])        # chroma samples replicated


def test_swscale_fallback_formats():
    for name in ("pal8", "rgb565le", "monob"):
        wf = working_format_for(name)
        assert wf.method == "swscale" and not wf.exact and wf.fmt.family == "rgb"
    frame = fill(av.VideoFrame(32, 16, "rgb565le"), seed=1)
    planes = frame_to_planes(frame)
    assert len(planes) == 3 and all(p.shape == (16, 32) and p.dtype == np.uint8 for p in planes)


def test_unknown_format_is_rejected():
    from app.core.errors import UnsupportedFormatError

    with pytest.raises(UnsupportedFormatError):
        working_format_for("not_a_format")


def test_decoded_frames_with_row_padding(clip_cfr, clip_10bit, clip_odd_444):
    """Real decoder output (padded line sizes, reused buffers) equals an independent extraction."""

    for clip in (clip_cfr, clip_10bit, clip_odd_444):
        reference = media.decode_planes(clip.path, max_frames=5)
        ours = []
        with av.open(str(clip.path)) as container:
            stream = container.streams.video[0]
            wf = working_format_for(stream.codec_context.pix_fmt)
            for frame in container.decode(stream):
                ours.append(frame_to_planes(frame, wf))
                if len(ours) == 5:
                    break
        assert len(ours) == len(reference) == 5
        for frame_planes, ref_planes in zip(ours, reference):
            for a, b in zip(frame_planes, ref_planes):
                assert np.array_equal(a, b)
        # frames must differ from each other: proves the arrays were copied, not aliased to one buffer
        assert not np.array_equal(ours[0][0], ours[4][0])


def test_blank_planes():
    fmt = parse_working_pixfmt("yuv420p")
    planes = blank_planes(fmt, 33, 17, [173, 42, 26])
    assert [p.shape for p in planes] == [(17, 33), (9, 17), (9, 17)]
    assert int(planes[0][0, 0]) == 173 and int(planes[1].max()) == 42 and int(planes[2].min()) == 26
