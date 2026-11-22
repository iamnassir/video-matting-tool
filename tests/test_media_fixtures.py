"""Sanity checks of the synthetic test media (the fixtures everything else relies on)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from tests.helpers import media


def _stream(info, kind):
    return [s for s in info["streams"] if s["codec_type"] == kind]


def test_cfr_clip(clip_cfr):
    info = media.ffprobe(clip_cfr.path)
    v = _stream(info, "video")[0]
    assert (v["width"], v["height"], v["pix_fmt"], v["codec_name"]) == (320, 180, "yuv420p", "h264")
    assert _stream(info, "audio")[0]["codec_name"] == "aac"
    times = media.packet_pts(clip_cfr.path)
    assert len(times) == clip_cfr.frames
    assert times == [Fraction(p) * clip_cfr.time_base for p in clip_cfr.pts]
    deltas = {b - a for a, b in zip(times, times[1:])}
    assert deltas == {Fraction(1001, 30000)}


def test_vfr_clip_is_really_vfr(clip_vfr):
    times = media.packet_pts(clip_vfr.path)
    assert times == [Fraction(p) * clip_vfr.time_base for p in clip_vfr.pts]
    deltas = {b - a for a, b in zip(times, times[1:])}
    assert len(deltas) >= 5 and max(deltas) / min(deltas) >= 5


def test_10bit_clip_tags(clip_10bit):
    v = _stream(media.ffprobe(clip_10bit.path), "video")[0]
    assert v["pix_fmt"] == "yuv420p10le" and v["codec_name"] == "hevc"
    assert v.get("color_space") == "bt2020nc" and v.get("color_transfer") == "arib-std-b67"
    assert v.get("color_primaries") == "bt2020" and v.get("color_range") == "tv"


def test_rotated_clip_has_display_matrix(clip_rotated):
    v = _stream(media.ffprobe(clip_rotated.path), "video")[0]
    assert (v["width"], v["height"]) == (320, 180)
    rotations = [sd.get("rotation") for sd in v.get("side_data_list", []) if "rotation" in sd]
    assert rotations and abs(int(rotations[0])) == 90


def test_other_containers(clip_noaudio_mkv, clip_webm, clip_odd_444):
    assert not _stream(media.ffprobe(clip_noaudio_mkv.path), "audio")
    web = media.ffprobe(clip_webm.path)
    assert _stream(web, "video")[0]["codec_name"] == "vp9" and _stream(web, "audio")[0]["codec_name"] == "opus"
    odd = _stream(media.ffprobe(clip_odd_444.path), "video")[0]
    assert (odd["width"], odd["height"], odd["pix_fmt"]) == (321, 179, "yuv444p")


def test_subject_is_where_the_helpers_say(clip_cfr, clip_two_subjects):
    frames = media.decode_planes(clip_cfr.path, max_frames=1)
    y = frames[0][0].astype(np.float32)
    alpha = media.subject_alpha(0, clip_cfr.width, clip_cfr.height, clip_cfr.frames)
    inside, outside = y[alpha > 0.99].mean(), y[alpha < 0.01].mean()
    assert abs(inside - outside) > 20            # the subject is clearly distinguishable in luma
    two = media.synth_frame(0, 320, 180, 36, two_subjects=True)
    d = media.distractor_alpha(320, 180) > 0.99
    assert np.abs(two[d].astype(int) - np.array(media.SUBJECT_RGB)).max() <= 2


def test_test_model_keys_the_subject():
    import torch

    from tests.helpers.dummy_model import chroma_alpha

    rgb = media.synth_frame(5, 320, 180, 36)
    t = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255.0
    a = chroma_alpha(t)[0, 0].numpy()
    truth = media.subject_alpha(5, 320, 180, 36)
    assert a[truth > 0.99].min() > 0.99
    assert a[truth < 0.01].max() < 0.01
