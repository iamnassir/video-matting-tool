"""Tests of ``app.processing.probe``: static metadata (ffprobe) and the exact frame timeline.

The specification under test, in short:

* one ffprobe JSON call fills ``VideoMetadata`` / ``VideoStreamInfo`` / ``AudioStreamInfo``;
* ``FrameTimeline`` holds one entry per frame *the decoder will deliver*, in presentation order, as
  integer timestamps in the stream time base - whatever the container does (B-frames, edit lists,
  packets without timestamps, hidden frames, start offsets, broken timestamps);
* unreadable input raises the documented ``VmtError`` subclasses and nothing else.

The reference for "what the decoder delivers" is always something independent of the code under test:
``ffprobe`` packet lists (``media.packet_pts``), the ``ffmpeg`` CLI (``-vf showinfo``) or a plain PyAV
decode.

Besides the tests this module holds the *derived clips* (``Derived``): special-purpose files built
once per session from the ``conftest`` fixtures.  ``tests/test_reader.py`` imports them from here so
that both modules exercise the very same files.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import av
import numpy as np
import pytest

from app.core.errors import (
    CorruptedVideoError,
    FfmpegFailedError,
    FfmpegNotFoundError,
    InputError,
    InvalidPathError,
    NoVideoStreamError,
    UnsupportedFormatError,
    VmtError,
)
from app.processing import probe as probe_module
from app.processing.probe import TimestampFixer, probe
from tests.helpers import media

# =========================================================================== shared helpers


def ffmpeg(*args: object) -> None:
    """Run the ffmpeg CLI quietly; raises with its stderr when it fails."""

    media.run([media.FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *[str(a) for a in args]])


def cli_frame_times(path: Path) -> List[Fraction]:
    """Presentation times (seconds) of the frames the ``ffmpeg`` CLI itself delivers for *path*.

    ``-copyts`` keeps the container timestamps and ``showinfo`` prints the timestamp every decoded
    frame carries when it enters the filter graph, i.e. after FFmpeg's own handling of missing
    timestamps.  This is the ground truth for "ffmpeg-equivalent" timelines.
    """

    result = subprocess.run(
        [media.FFMPEG, "-hide_banner", "-nostdin", "-copyts", "-i", str(path), "-map", "0:v:0", "-vf", "showinfo",
         "-fps_mode", "passthrough", "-f", "null", "-"],
        capture_output=True,
    )
    text = result.stderr.decode("utf-8", "replace")
    base = re.search(r"config in time_base: (\d+)/(\d+)", text)
    assert result.returncode == 0 and base, f"ffmpeg showinfo failed for {path}:\n{text[-1500:]}"
    time_base = Fraction(int(base.group(1)), int(base.group(2)))
    return [Fraction(int(m.group(1))) * time_base for m in re.finditer(r"\] n:\s*\d+ pts:\s*(-?\d+)\s+pts_time:", text)]


def timeline_times(result) -> List[Fraction]:
    """The probed timeline as exact seconds."""

    return [Fraction(int(p)) * result.timeline.time_base for p in result.timeline.pts]


def video_entry(path: Path, index: int = 0) -> dict:
    """ffprobe's description of the *index*-th video stream of *path*."""

    return [s for s in media.ffprobe(path)["streams"] if s["codec_type"] == "video"][index]


def remux_with_timestamps(src: Path, dst: Path, pts: Sequence[int], dts: Optional[Sequence[int]] = None) -> None:
    """Stream-copy the video of *src* into *dst*, replacing the packet timestamps (demux order).

    This is how clips with deliberately broken timestamps are made: no encoder would produce them.
    """

    with av.open(str(src)) as source, av.open(str(dst), mode="w") as target:
        in_stream = source.streams.video[0]
        out_stream = target.add_stream_from_template(in_stream)
        out_stream.time_base = in_stream.time_base
        position = 0
        for packet in source.demux(in_stream):
            if packet.size == 0:
                continue
            packet.stream = out_stream
            packet.pts = int(pts[position])
            packet.dts = int((dts or pts)[position])
            position += 1
            target.mux(packet)
    assert position == len(pts), f"{src.name} has {position} packets, {len(pts)} timestamps were given"


def truncated_copy(src: Path, dst: Path, fraction: float) -> Path:
    """The first *fraction* of the bytes of *src* (an interrupted copy / download)."""

    data = Path(src).read_bytes()
    Path(dst).write_bytes(data[: int(len(data) * fraction)])
    return Path(dst)


def damaged_copy(src: Path, dst: Path, fraction: float, length: int, fill: int = 0xFF) -> Path:
    """A copy of *src* in which *length* bytes starting at *fraction* of the file are overwritten."""

    shutil.copyfile(src, dst)
    with open(dst, "r+b") as handle:
        handle.seek(int(Path(src).stat().st_size * fraction))
        handle.write(bytes([fill]) * length)
    return Path(dst)


# --------------------------------------------------------------------------- derived clips

#: Timestamps of ``duplicate_pts.mkv`` (two packets repeat the timestamp of their predecessor) and of
#: ``backward_pts.mkv`` (packet 30 lies before its predecessor although the stream has no B-frames;
#: the anomaly is placed far behind the frames the probe decodes for its plausibility check).
DUPLICATE_PTS = [0, 40, 80, 120, 160, 160, 240, 280, 320, 360, 360, 440]
BACKWARD_PTS = [200 + 40 * i for i in range(40)]
BACKWARD_PTS[30] = BACKWARD_PTS[29] - 15
BACKWARD_DTS = [40 * i for i in range(40)]
#: Presentation delay patched into ``negative_start.mkv`` (Matroska CodecDelay), in milliseconds.
NEGATIVE_START_MS = 100
#: Frames and frame rate of the generated raw H.264 stream.
RAW_FRAMES, RAW_FPS = 20, 25
UNICODE_NAME = "vidéo テスト 视频 clip (kopie).mp4"

_BUILDERS: Dict[str, Callable[["Derived", Path], None]] = {}


def _builds(*names: str):
    def register(function: Callable[["Derived", Path], None]):
        for name in names:
            _BUILDERS[name] = function
        return function

    return register


class Derived:
    """Special-purpose clips, built on first use and kept for the rest of the test session.

    ``path(name)`` returns the file; every name is listed in ``_BUILDERS``.  The files live in a
    sub-folder of the session's media directory, so nothing is rebuilt between test modules.
    """

    def __init__(self, folder: Path, base: Path) -> None:
        self.folder = Path(folder)
        self.base = Path(base)          # the plain CFR clip (H.264 with B-frames + AAC, 36 frames)
        self.folder.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        target = self.folder / name
        if not target.exists():
            staging = self.folder / f"staging_{name}"
            _BUILDERS[name](self, staging)
            staging.replace(target)
        return target

    def scratch(self, name: str) -> Path:
        """A helper file next to the clips (intermediate results of a builder)."""

        return self.folder / f"scratch_{name}"


@_builds("mpeg4_bframes.avi")
def _build_avi_mpeg4(d: Derived, out: Path) -> None:
    ffmpeg("-i", d.base, "-an", "-c:v", "mpeg4", "-bf", "2", "-g", "12", "-q:v", "3", out)


@_builds("mjpeg.avi")
def _build_avi_mjpeg(d: Derived, out: Path) -> None:
    ffmpeg("-i", d.base, "-an", "-c:v", "mjpeg", "-q:v", "3", out)


@_builds("raw.h264")
def _build_raw_h264(d: Derived, out: Path) -> None:
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=size=160x120:rate={RAW_FPS}", "-frames:v", RAW_FRAMES, "-pix_fmt", "yuv420p",
           "-c:v", "libx264", "-preset", "veryfast", "-bf", "2", "-g", "10", "-f", "h264", out)


@_builds("ntsc.mkv")
def _build_ntsc_mkv(d: Derived, out: Path) -> None:
    ffmpeg("-i", d.base, "-an", "-c", "copy", out)          # 29.97 fps squeezed into Matroska's 1 ms ticks


@_builds("transport.ts")
def _build_ts(d: Derived, out: Path) -> None:
    ffmpeg("-i", d.base, "-c", "copy", "-f", "mpegts", out)


@_builds("start_offset.mp4")
def _build_start_mp4(d: Derived, out: Path) -> None:
    media.write_video(out, frames=12, width=160, height=90, start_pts=30000, time_base=Fraction(1, 30000))


@_builds("start_offset.mkv")
def _build_start_mkv(d: Derived, out: Path) -> None:
    media.write_video(out, frames=12, width=160, height=90, start_pts=1000, time_base=Fraction(1, 1000), fps=Fraction(25))


@_builds("negative_start.mkv")
def _build_negative_start(d: Derived, out: Path) -> None:
    # No muxer writes negative presentation times on request, but Matroska has an element that
    # shifts a track towards them: CodecDelay.  The 7-byte Language element ("und") of the only
    # track is replaced by a 7-byte CodecDelay of 100 ms, which moves every timestamp 100 ms back.
    source = d.scratch("negative_start_source.mkv")
    media.write_video(source, frames=24, width=160, height=90, time_base=Fraction(1, 1000), fps=Fraction(25))
    data = bytearray(source.read_bytes())
    language = bytes.fromhex("22b59c83756e64")
    if data.count(language) != 1:
        pytest.skip("this FFmpeg build writes the Matroska track header differently; cannot patch in a CodecDelay")
    position = data.find(language)
    data[position: position + 7] = bytes.fromhex("56aa84") + (NEGATIVE_START_MS * 1_000_000).to_bytes(4, "big")
    out.write_bytes(bytes(data))


@_builds("edit_list.mp4")
def _build_edit_list(d: Derived, out: Path) -> None:
    # Negative timestamps make the MP4 muxer write an edit list that hides the first three frames.
    media.write_video(out, frames=12, width=160, height=90, start_pts=-3003, time_base=Fraction(1, 30000))


@_builds("one_frame.mp4", "one_frame.mkv")
def _build_one_frame(d: Derived, out: Path) -> None:
    media.write_video(out, frames=1, width=64, height=48)


@_builds("two_frames.mp4", "two_frames.mkv")
def _build_two_frames(d: Derived, out: Path) -> None:
    media.write_video(out, frames=2, width=64, height=48)


@_builds("two_audio.mp4")
def _build_two_audio(d: Derived, out: Path) -> None:
    ffmpeg("-i", d.base, "-f", "lavfi", "-t", "0.8", "-i", "sine=frequency=440:sample_rate=48000",
           "-f", "lavfi", "-t", "0.8", "-i", "sine=frequency=880:sample_rate=44100",
           "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0", "-c:v", "copy",
           "-c:a:0", "aac", "-b:a:0", "96k", "-c:a:1", "aac", "-b:a:1", "64k",
           "-metadata:s:a:0", "language=eng", "-metadata:s:a:0", "title=Main",
           "-metadata:s:a:1", "language=deu", "-metadata:s:a:1", "title=Kommentar",
           "-disposition:a:0", "default", "-disposition:a:1", "0", out)


@_builds("audio_offset.mp4")
def _build_audio_offset(d: Derived, out: Path) -> None:
    media.add_audio(d.base, out, audio_offset=0.5)


@_builds("duplicate_pts.mkv")
def _build_duplicate_pts(d: Derived, out: Path) -> None:
    source = d.scratch("no_bframes_12.mkv")
    media.write_video(source, frames=12, width=64, height=48, time_base=Fraction(1, 1000), fps=Fraction(25), options={"bf": "0"})
    remux_with_timestamps(source, out, DUPLICATE_PTS)


@_builds("backward_pts.mkv")
def _build_backward_pts(d: Derived, out: Path) -> None:
    source = d.scratch("no_bframes_40.mkv")
    media.write_video(source, frames=40, width=64, height=48, time_base=Fraction(1, 1000), fps=Fraction(25), options={"bf": "0"})
    remux_with_timestamps(source, out, BACKWARD_PTS, BACKWARD_DTS)


@_builds("vp8_altref.webm")
def _build_vp8_altref(d: Derived, out: Path) -> None:
    # Two-pass VP8 produces "alt-ref" packets: decoded but never shown (packets > frames).
    common = ["-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30", "-frames:v", "120", "-c:v", "libvpx", "-b:v", "150k",
              "-auto-alt-ref", "1", "-lag-in-frames", "25", "-arnr-maxframes", "7", "-deadline", "good", "-cpu-used", "2",
              "-g", "120", "-passlogfile", d.scratch("vp8_pass")]
    ffmpeg(*common, "-pass", "1", "-f", "null", "-")
    ffmpeg(*common, "-pass", "2", "-f", "webm", out)


@_builds("subtitles.mkv")
def _build_subtitles(d: Derived, out: Path) -> None:
    srt = d.scratch("subs.srt")
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    ffmpeg("-i", d.base, "-i", srt, "-map", "0", "-map", "1", "-c", "copy", "-c:s", "srt", out)


@_builds("timecode.mov")
def _build_timecode(d: Derived, out: Path) -> None:
    ffmpeg("-i", d.base, "-map", "0", "-c", "copy", "-timecode", "01:00:00:00", out)       # adds a tmcd data track


@_builds("cover_art.mp4")
def _build_cover_art(d: Derived, out: Path) -> None:
    cover = d.scratch("cover.png")
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=64x48", "-frames:v", "1", cover)
    ffmpeg("-i", d.base, "-i", cover, "-map", "0", "-map", "1", "-c", "copy", "-disposition:v:1", "attached_pic", out)


@_builds("two_video.mkv")
def _build_two_video(d: Derived, out: Path) -> None:
    small = d.scratch("small.mp4")
    media.write_video(small, frames=6, width=64, height=48)
    ffmpeg("-i", small, "-i", d.base, "-map", "0:v:0", "-map", "1:v:0", "-map", "1:a:0", "-c", "copy", out)


@_builds("hdr10.mp4")
def _build_hdr10(d: Derived, out: Path) -> None:
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25", "-frames:v", "6", "-pix_fmt", "yuv420p10le", "-c:v", "libx265",
           "-x265-params", "log-level=error:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
           "master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=1000,400",
           "-tag:v", "hvc1", out)


def _build_display_matrix(*options: str) -> Callable[[Derived, Path], None]:
    def build(d: Derived, out: Path) -> None:
        ffmpeg(*options, "-i", d.base, "-map", "0", "-c", "copy", out)

    return build


#: file name -> (ffmpeg input options, clockwise rotation a player applies, picture is mirrored)
DISPLAY_MATRIX_CLIPS = {
    "rotation_ccw90.mp4": (("-display_rotation", "90"), 270, False),
    "rotation_180.mp4": (("-display_rotation", "180"), 180, False),
    "rotation_ccw270.mp4": (("-display_rotation", "270"), 90, False),
    "rotation_cw90.mov": (("-display_rotation", "-90"), 90, False),
    "rotation_ccw90.mkv": (("-display_rotation", "90"), 270, False),
    # A mirrored matrix has no unique angle.  The probe follows av_display_rotation_get (ffprobe's
    # "rotation"), and the muxer rebuilds the matrix as "that rotation, then a vertical flip": a
    # horizontal flip is therefore 180 degrees + vertical flip, a vertical flip 0 degrees + flip.
    "hflip.mp4": (("-display_hflip",), 180, True),
    "vflip.mp4": (("-display_vflip",), 0, True),
    "rotation_ccw90_hflip.mp4": (("-display_rotation", "90", "-display_hflip"), 270, True),
    "rotation_ccw30.mp4": (("-display_rotation", "30"), 0, False),
}
for _name, (_options, _, _) in DISPLAY_MATRIX_CLIPS.items():
    _BUILDERS[_name] = _build_display_matrix(*_options)


@_builds("faststart.mp4")
def _build_faststart(d: Derived, out: Path) -> None:
    ffmpeg("-i", d.base, "-c", "copy", "-movflags", "+faststart", out)      # index (moov) in front of the data


@_builds(UNICODE_NAME, "no_extension")
def _build_copy(d: Derived, out: Path) -> None:
    shutil.copyfile(d.base, out)


@_builds("quicktime_no_extension")
def _build_quicktime(d: Derived, out: Path) -> None:
    ffmpeg("-i", d.base, "-c", "copy", "-f", "mov", out)


@_builds("long.mp4")
def _build_long(d: Derived, out: Path) -> None:
    # 18000 frames (10 minutes at 29.97 fps) for a few megabytes: a tiny clip, stream-copied 300 times.
    unit = d.scratch("long_unit.mp4")
    media.write_video(unit, frames=60, width=64, height=48)
    ffmpeg("-stream_loop", "299", "-i", unit, "-c", "copy", out)


@_builds("audio_only.m4a")
def _build_audio_only(d: Derived, out: Path) -> None:
    ffmpeg("-f", "lavfi", "-t", "0.5", "-i", "sine=frequency=440", "-c:a", "aac", "-f", "ipod", out)


@_builds("cover_only.mp3")
def _build_cover_only(d: Derived, out: Path) -> None:
    cover = d.scratch("cover_for_mp3.png")
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=64x48", "-frames:v", "1", cover)
    ffmpeg("-f", "lavfi", "-t", "0.5", "-i", "sine=frequency=440", "-i", cover, "-map", "0:a", "-map", "1:v",
           "-c:a", "libmp3lame", "-c:v", "png", "-disposition:v", "attached_pic", "-id3v2_version", "3", "-f", "mp3", out)


@pytest.fixture(scope="session")
def derived(media_dir, clip_cfr) -> Derived:
    """The derived clips (see :class:`Derived`)."""

    return Derived(media_dir / "video_input", clip_cfr.path)


# =========================================================================== static metadata


def test_one_ffprobe_call_and_every_container_field(clip_cfr, monkeypatch):
    calls = []
    real_run = subprocess.run

    def spy(command, *args, **kwargs):
        calls.append(list(command))
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(probe_module.subprocess, "run", spy)
    result = probe(clip_cfr.path)
    assert len(calls) == 1, "static metadata must come from a single ffprobe call"
    assert {"-show_format", "-show_streams"} <= set(calls[0]) and "json" in calls[0]

    meta = result.metadata
    info = media.ffprobe(clip_cfr.path)["format"]
    assert Path(meta.path) == clip_cfr.path and Path(meta.path).is_absolute()
    assert meta.filename == clip_cfr.path.name
    assert meta.size_bytes == clip_cfr.path.stat().st_size
    assert meta.format_name == info["format_name"] and meta.format_long_name == info["format_long_name"]
    assert meta.container == "mp4"
    assert meta.duration == Fraction(info["duration"]) and meta.start_time == 0
    assert meta.bit_rate == int(info["bit_rate"])
    assert meta.tags["major_brand"] == "isom"
    assert (meta.subtitle_streams, meta.data_streams, meta.extra_video_streams) == (0, 0, 0)
    assert meta.warnings == ()


def test_video_stream_fields(clip_cfr):
    meta = probe(clip_cfr).metadata                  # a path-like object is accepted as well
    video, entry = meta.video, video_entry(clip_cfr.path)
    assert video.index == 0
    assert (video.codec, video.profile, video.codec_tag) == ("h264", entry["profile"], "avc1")
    assert video.codec_long == entry["codec_long_name"]
    assert (video.width, video.height) == (clip_cfr.width, clip_cfr.height) == (video.display_width, video.display_height)
    assert (video.pix_fmt, video.bit_depth, video.has_alpha) == ("yuv420p", 8, False)
    assert video.time_base == clip_cfr.time_base == Fraction(1, 30000)
    assert video.field_order == "progressive" and not video.is_interlaced
    assert video.rotation == 0
    assert (video.start_pts, video.start_time) == (0, 0)
    assert video.duration == Fraction(int(entry["duration_ts"]), 30000)
    assert video.nb_frames == 36
    assert video.avg_frame_rate == video.r_frame_rate == Fraction(30000, 1001)
    assert video.bit_rate == int(entry["bit_rate"])
    assert video.tags["handler_name"] == "VideoHandler"
    assert video.hdr == {}
    # The fixture carries no colour tags: they must be reported as unknown, never guessed.
    color = video.color
    assert (color.range, color.space, color.primaries, color.transfer) == ("unknown",) * 4
    assert color.chroma_location == entry.get("chroma_location", "unspecified")


def test_audio_stream_fields(clip_cfr, clip_noaudio_mkv):
    meta = probe(clip_cfr.path).metadata
    entry = [s for s in media.ffprobe(clip_cfr.path)["streams"] if s["codec_type"] == "audio"][0]
    assert meta.has_audio and len(meta.audio) == 1
    audio = meta.audio[0]
    assert (audio.index, audio.codec, audio.profile) == (1, "aac", "LC")
    assert audio.codec_long == entry["codec_long_name"]
    assert (audio.sample_rate, audio.channels, audio.channel_layout, audio.sample_fmt) == (48000, 1, "mono", "fltp")
    assert audio.time_base == Fraction(1, 48000)
    assert audio.start_time == 0
    assert audio.duration == Fraction(int(entry["duration_ts"]), 48000)
    assert audio.nb_frames == int(entry["nb_frames"]) and audio.bit_rate == int(entry["bit_rate"])
    assert audio.language == "und" and audio.is_default is True
    assert audio.tags["handler_name"] == "SoundHandler"

    silent = probe(clip_noaudio_mkv.path).metadata
    assert silent.audio == () and not silent.has_audio


def test_two_audio_streams_and_audio_offset(derived):
    first, second = probe(derived.path("two_audio.mp4")).metadata.audio
    assert (first.index, second.index) == (1, 2)
    assert (first.sample_rate, second.sample_rate) == (48000, 44100)
    assert (first.language, second.language) == ("eng", "deu")
    assert (first.title, second.title) == ("Main", "Kommentar")
    assert (first.is_default, second.is_default) == (True, False)
    assert first.duration == Fraction(4, 5) and abs(second.duration - Fraction(4, 5)) < Fraction(1, 20)

    late = probe(derived.path("audio_offset.mp4")).metadata
    assert late.video.start_time == 0
    # 0.5 s offset; AAC priming may move the first packet by a frame or two.
    assert abs(late.audio[0].start_time - Fraction(1, 2)) < Fraction(1, 20)


def test_colour_tags_bit_depth_and_hdr(clip_10bit, clip_webm, derived):
    video = probe(clip_10bit.path).metadata.video
    assert (video.codec, video.pix_fmt, video.bit_depth, video.has_alpha) == ("hevc", "yuv420p10le", 10, False)
    assert (video.color.range, video.color.space, video.color.primaries, video.color.transfer) == (
        "tv", "bt2020nc", "bt2020", "arib-std-b67")
    assert video.color.is_hdr
    assert video.field_order == "unknown"            # ffprobe does not say: nothing is invented

    # Partially tagged stream: what is tagged is reported, the rest stays unknown.
    web = probe(clip_webm.path).metadata.video
    assert (web.color.range, web.color.space, web.color.primaries, web.color.transfer) == ("tv", "unknown", "unknown", "unknown")

    # HDR10 static metadata that exists only in the bitstream (x265 SEI) is found as well.
    hdr = probe(derived.path("hdr10.mp4")).metadata.video
    assert hdr.color.transfer == "smpte2084" and hdr.color.is_hdr
    assert hdr.hdr["content_light_level"]["max_content"] == 1000
    assert hdr.hdr["content_light_level"]["max_average"] == 400
    assert Fraction(hdr.hdr["mastering_display"]["max_luminance"]) == 1000
    assert Fraction(hdr.hdr["mastering_display"]["red_x"]) == Fraction(34000, 50000)


@pytest.mark.parametrize("name", sorted(DISPLAY_MATRIX_CLIPS))
def test_rotation_is_the_clockwise_angle_a_player_applies(derived, name):
    _, clockwise, mirrored = DISPLAY_MATRIX_CLIPS[name]
    meta = probe(derived.path(name)).metadata
    video = meta.video
    assert video.rotation == clockwise
    assert (video.width, video.height) == (320, 180)                 # coded size never changes
    expected_display = (180, 320) if clockwise in (90, 270) else (320, 180)
    assert (video.display_width, video.display_height) == expected_display
    mirror_warnings = [w for w in meta.warnings if "mirror" in w.lower()]
    odd_warnings = [w for w in meta.warnings if "multiple of 90" in w]
    assert bool(mirror_warnings) == mirrored
    assert bool(odd_warnings) == (name == "rotation_ccw30.mp4")
    if not mirrored and name != "rotation_ccw30.mp4":
        assert meta.warnings == ()


def test_rotation_follows_ffprobe_sign_convention(clip_rotated):
    # The fixture is made with "-display_rotation 90": ffprobe reports +90 (counter-clockwise), so a
    # player turns the coded picture 270 degrees clockwise.
    side = [sd for sd in video_entry(clip_rotated.path).get("side_data_list", []) if "rotation" in sd]
    assert int(side[0]["rotation"]) == 90
    video = probe(clip_rotated.path).metadata.video
    assert video.rotation == 270
    assert (video.display_width, video.display_height) == (clip_rotated.height, clip_rotated.width)


def test_rotation_helper_handles_legacy_tag_and_garbage():
    rotation = probe_module._rotation
    assert rotation({}, {"rotate": "90"}) == (90, [])              # the legacy tag already is clockwise
    assert rotation({}, {"rotate": "-90"}) == (270, [])
    assert rotation({}, {"rotate": "sideways"}) == (0, [])
    assert rotation({}, {}) == (0, [])
    assert rotation({"side_data_list": [{"side_data_type": "Display Matrix", "rotation": "nan"}]}, {}) == (0, [])
    # Only the matrix is given (no rotation number): it is evaluated like av_display_rotation_get does.
    matrix = "\n00000000:            0       65536           0\n00000001:       -65536           0           0\n00000002:            0           0  1073741824\n"
    assert rotation({"side_data_list": [{"side_data_type": "Display Matrix", "displaymatrix": matrix}]}, {}) == (90, [])


@pytest.mark.parametrize(
    "fixture, expected",
    [("clip_cfr", "mp4"), ("clip_10bit", "mov"), ("clip_noaudio_mkv", "mkv"), ("clip_webm", "webm")],
)
def test_container_is_normalised_for_fixtures(request, fixture, expected):
    assert probe(request.getfixturevalue(fixture).path).metadata.container == expected


@pytest.mark.parametrize(
    "name, expected",
    [("mjpeg.avi", "avi"), ("transport.ts", "other"), ("raw.h264", "other"),
     ("no_extension", "mp4"), ("quicktime_no_extension", "mov")],
)
def test_container_is_normalised_for_derived_clips(derived, name, expected):
    assert probe(derived.path(name)).metadata.container == expected


def test_container_normalisation_rules():
    normalise = probe_module._normalise_container
    mov_family = "mov,mp4,m4a,3gp,3g2,mj2"
    assert normalise(mov_family, Path("a.MP4"), {}, []) == "mp4"
    assert normalise(mov_family, Path("a.mov"), {"major_brand": "isom"}, []) == "mov"
    assert normalise(mov_family, Path("a.bin"), {"major_brand": "qt  "}, []) == "mov"
    assert normalise(mov_family, Path("a.bin"), {"major_brand": "mp42"}, []) == "mp4"
    vp9 = [{"codec_type": "video", "codec_name": "vp9"}, {"codec_type": "audio", "codec_name": "opus"}]
    h264 = [{"codec_type": "video", "codec_name": "h264"}]
    assert normalise("matroska,webm", Path("a.mkv"), {}, vp9) == "mkv"
    assert normalise("matroska,webm", Path("a.webm"), {}, h264) == "webm"
    assert normalise("matroska,webm", Path("a.bin"), {}, vp9) == "webm"       # decided by the codecs
    assert normalise("matroska,webm", Path("a.bin"), {}, h264) == "mkv"
    assert normalise("avi", Path("a.avi"), {}, []) == "avi"
    assert normalise("mpegts", Path("a.ts"), {}, []) == "other"
    assert normalise("", Path("a.mp4"), {}, []) == "other"


def test_stream_counts_and_primary_stream(derived):
    subs = probe(derived.path("subtitles.mkv")).metadata
    assert (subs.subtitle_streams, subs.data_streams, subs.extra_video_streams) == (1, 0, 0)

    timecode = probe(derived.path("timecode.mov")).metadata
    assert (timecode.subtitle_streams, timecode.data_streams, timecode.extra_video_streams) == (0, 1, 0)

    # Cover art is a "video" stream for FFmpeg, but never the one to process - and not worth a warning.
    cover = probe(derived.path("cover_art.mp4")).metadata
    assert cover.extra_video_streams == 1 and (cover.video.width, cover.video.height) == (320, 180)
    assert cover.frame_count == 36 and not [w for w in cover.warnings if "video streams" in w]

    # Two real video streams: the first one wins and the user is told.
    two = probe(derived.path("two_video.mkv")).metadata
    assert two.video.index == 0 and (two.video.width, two.video.height) == (64, 48)
    assert two.frame_count == 6 and two.extra_video_streams == 1
    assert any("2 video streams" in w for w in two.warnings)


def test_metadata_to_dict_is_json_serialisable(clip_cfr, clip_10bit, clip_vfr, clip_rotated, derived):
    paths = [clip_cfr.path, clip_10bit.path, clip_vfr.path, clip_rotated.path, derived.path("hdr10.mp4"),
             derived.path("mpeg4_bframes.avi"), derived.path("two_audio.mp4"), derived.path("one_frame.mkv")]
    for path in paths:
        result = probe(path)
        data = json.loads(json.dumps(result.metadata.to_dict()))
        assert data["frame_count"] == len(result.timeline) and data["filename"] == path.name
        assert data["video"]["time_base"] == f"{result.timeline.time_base.numerator}/{result.timeline.time_base.denominator}"
        assert isinstance(data["warnings"], list) and isinstance(data["vfr"], dict)
        json.dumps(result.scan.to_dict())
    first = probe(clip_cfr.path).metadata.to_dict()
    assert first["frame_rate"] == "30000/1001" and abs(first["frame_rate_float"] - 29.97003) < 1e-5
    assert first["video"]["display_width"] == 320 and first["has_audio"] is True


# =========================================================================== timeline: healthy files


@pytest.mark.parametrize(
    "fixture", ["clip_cfr", "clip_vfr", "clip_noaudio_mkv", "clip_10bit", "clip_rotated", "clip_two_subjects",
                "clip_odd_444", "clip_webm"],
)
def test_timeline_is_exact_for_every_fixture(request, fixture):
    clip = request.getfixturevalue(fixture)
    result = probe(clip.path)
    timeline, meta = result.timeline, result.metadata
    assert timeline.time_base == clip.time_base == meta.video.time_base
    assert timeline.pts.tolist() == clip.pts                      # exactly the timestamps that were written
    assert timeline_times(result) == media.packet_pts(clip.path)  # ... and what ffprobe sees
    assert meta.frame_count == len(timeline) == clip.frames
    assert timeline.pts.dtype == np.int64 and timeline.keyframes.dtype == bool and timeline.durations.dtype == np.int64
    assert bool(np.all(np.diff(timeline.pts) > 0))
    assert bool(timeline.keyframes[0]) and not timeline.synthesized
    assert meta.is_vfr == timeline.is_vfr == clip.facts["vfr"]
    assert meta.video_duration == timeline.duration
    assert result.scan.method == "packets" and result.scan.packets == clip.frames
    assert (meta.video.width, meta.video.height, meta.video.pix_fmt) == (clip.width, clip.height, clip.facts["pix_fmt"])


def test_b_frames_are_reordered_and_key_frames_flagged(clip_cfr):
    data = media.ffprobe(clip_cfr.path, "-select_streams", "v:0", "-show_entries", "packet=pts,flags")
    stored = [int(p["pts"]) for p in data["packets"]]
    assert stored != sorted(stored), "precondition: the clip stores B-frames out of presentation order"
    timeline = probe(clip_cfr.path).timeline
    assert timeline.pts.tolist() == sorted(stored)
    keys = sorted(int(p["pts"]) for p in data["packets"] if "K" in p["flags"])
    assert timeline.pts[timeline.keyframes].tolist() == keys
    assert np.flatnonzero(timeline.keyframes).tolist() == [0, 12, 24]          # GOP of 12 in the fixture


def test_frame_rate_is_an_exact_fraction(clip_cfr, clip_noaudio_mkv, clip_webm, clip_vfr, derived):
    assert probe(clip_cfr.path).metadata.frame_rate == Fraction(30000, 1001)
    assert probe(clip_noaudio_mkv.path).metadata.frame_rate == Fraction(25)
    # Matroska / WebM round every timestamp to a millisecond.  The mean over such a timeline is not
    # the frame rate; the container's own declaration is, as long as the timeline agrees with it.
    assert probe(clip_webm.path).metadata.frame_rate == Fraction(30)
    ntsc = probe(derived.path("ntsc.mkv"))
    assert ntsc.timeline.time_base == Fraction(1, 1000) and not ntsc.metadata.is_vfr
    assert ntsc.metadata.frame_rate == Fraction(30000, 1001)
    # A genuinely variable clip has no nominal rate: the mean over the exact timeline is reported.
    vfr = probe(clip_vfr.path)
    assert vfr.metadata.frame_rate == vfr.timeline.average_rate.limit_denominator(100_000)
    assert vfr.metadata.frame_rate != vfr.metadata.video.r_frame_rate


def test_vfr_statistics(clip_vfr, clip_cfr, clip_webm):
    result = probe(clip_vfr.path)
    stats, tb = result.metadata.vfr, float(clip_vfr.time_base)
    deltas = np.diff(np.asarray(clip_vfr.pts))
    assert result.metadata.is_vfr and stats["is_vfr"] is True and stats["frames"] == clip_vfr.frames
    assert stats["min_frame_duration"] == pytest.approx(float(deltas.min()) * tb)
    assert stats["max_frame_duration"] == pytest.approx(float(deltas.max()) * tb)
    assert stats["distinct_durations"] == len(set(deltas.tolist())) >= 5
    assert stats["timestamps_synthesized"] is False

    steady = probe(clip_cfr.path).metadata.vfr
    assert steady["is_vfr"] is False and steady["distinct_durations"] == 1 and steady["cfr_fit_error_ticks"] == 0
    # 30 fps in millisecond ticks alternates between 33 and 34 ms: constant frame rate all the same.
    rounded = probe(clip_webm.path).metadata
    assert not rounded.is_vfr and rounded.vfr["distinct_durations"] == 2


def test_edit_list_hides_frames(derived):
    path = derived.path("edit_list.mp4")
    result = probe(path)
    meta = result.metadata
    assert meta.video.nb_frames == 12, "precondition: the container still declares every stored frame"
    assert meta.frame_count == len(result.timeline) == 9
    assert result.scan.discarded == 3 and result.scan.method == "packets"
    assert timeline_times(result) == cli_frame_times(path)
    assert result.timeline.pts[0] == 0
    warning = [w for w in meta.warnings if "12" in w and "9" in w]
    assert warning and "edit list" in warning[0]
    assert meta.frame_rate == Fraction(30000, 1001) and not meta.is_vfr


@pytest.mark.parametrize("name, start", [("start_offset.mp4", Fraction(1)), ("start_offset.mkv", Fraction(1))])
def test_non_zero_start_time(derived, name, start):
    path = derived.path(name)
    result = probe(path)
    timeline, video = result.timeline, result.metadata.video
    assert timeline.start == start and timeline.time(0) == start
    assert (video.start_pts, video.start_time) == (int(timeline.pts[0]), start)
    assert result.metadata.start_time == start
    assert timeline_times(result) == media.packet_pts(path) == cli_frame_times(path)
    assert result.metadata.frame_count == 12 and not result.metadata.is_vfr
    # The duration is that of the frames, not the distance from zero.
    assert result.metadata.video_duration == timeline.duration < Fraction(1)


def test_negative_first_timestamp(derived):
    path = derived.path("negative_start.mkv")
    result = probe(path)
    timeline = result.timeline
    assert timeline.pts.tolist() == [40 * i - NEGATIVE_START_MS for i in range(24)]
    assert timeline.start == Fraction(-NEGATIVE_START_MS, 1000)
    assert timeline_times(result) == media.packet_pts(path) == cli_frame_times(path)
    assert result.scan.method == "packets" and not timeline.synthesized
    assert result.metadata.video.start_pts == -NEGATIVE_START_MS
    assert result.metadata.frame_rate == Fraction(25) and not result.metadata.is_vfr
    assert result.metadata.video_duration == Fraction(24, 25)


@pytest.mark.parametrize("name", ["one_frame.mp4", "one_frame.mkv", "two_frames.mp4", "two_frames.mkv"])
def test_one_and_two_frame_clips(derived, name):
    path = derived.path(name)
    frames = 1 if name.startswith("one") else 2
    result = probe(path)
    meta, timeline = result.metadata, result.timeline
    assert meta.frame_count == len(timeline) == frames
    assert timeline_times(result) == media.packet_pts(path)
    assert not meta.is_vfr and meta.vfr["is_vfr"] is False and meta.vfr["frames"] == frames
    assert meta.frame_rate == Fraction(30000, 1001)             # from the container: too few frames to measure
    assert meta.video_duration is not None and meta.video_duration >= 0
    # Known exactly from the timeline even where ffprobe cannot tell (very short Matroska files).
    assert meta.video.start_pts == 0 and meta.video.start_time == 0
    assert bool(timeline.keyframes[0])
    json.dumps(meta.to_dict())


def test_scan_timeline_false_reads_only_the_container_claims(clip_cfr, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the timeline must not be scanned")

    monkeypatch.setattr(probe_module, "_build_timeline", forbidden)
    result = probe(clip_cfr.path, scan_timeline=False)
    assert len(result.timeline) == 0 and result.timeline.time_base == clip_cfr.time_base
    assert result.metadata.frame_count == 36 and result.metadata.frame_rate == Fraction(30000, 1001)
    assert result.metadata.vfr["scanned"] is False and result.scan.method == "skipped"
    assert (result.metadata.video.width, result.metadata.video.height) == (320, 180)


# =========================================================================== timeline: awkward files


def test_avi_with_b_frames_has_no_usable_packet_timestamps(derived):
    path = derived.path("mpeg4_bframes.avi")
    packets = media.ffprobe(path, "-select_streams", "v:0", "-show_entries", "packet=pts")["packets"]
    assert any("pts" not in p for p in packets), "precondition: some packets carry no PTS"
    result = probe(path)
    meta, timeline = result.metadata, result.timeline
    assert meta.container == "avi" and meta.frame_count == len(timeline) == 36
    assert timeline.synthesized is True and result.scan.method == "decode"
    assert meta.vfr["timestamps_synthesized"] is True
    assert timeline_times(result) == cli_frame_times(path)        # exactly what ffmpeg itself would use
    assert bool(np.all(np.diff(timeline.pts) > 0))
    assert any("no presentation timestamp" in w for w in meta.warnings)
    assert meta.frame_rate == Fraction(30000, 1001) and not meta.is_vfr


def test_avi_without_b_frames_takes_the_fast_path(derived):
    path = derived.path("mjpeg.avi")
    result = probe(path)
    assert result.scan.method == "packets" and not result.timeline.synthesized
    assert result.metadata.frame_count == 36 and bool(result.timeline.keyframes.all())
    assert timeline_times(result) == cli_frame_times(path) == media.packet_pts(path)
    assert result.metadata.warnings == ()


def test_raw_elementary_stream_gets_ffmpeg_equivalent_timestamps(derived):
    path = derived.path("raw.h264")
    result = probe(path)
    meta, timeline = result.metadata, result.timeline
    assert meta.container == "other" and meta.frame_count == RAW_FRAMES
    assert timeline.synthesized is True and result.scan.method == "decode"
    assert "no timestamps" in result.scan.reason
    times = timeline_times(result)
    assert times == cli_frame_times(path)
    assert times == [Fraction(i, RAW_FPS) for i in range(RAW_FRAMES)]
    assert meta.frame_rate == Fraction(RAW_FPS) and not meta.is_vfr
    assert any("no presentation timestamp" in w for w in meta.warnings)


def test_duplicate_timestamps_are_made_strictly_increasing(derived):
    path = derived.path("duplicate_pts.mkv")
    stored = [int(p["pts"]) for p in media.ffprobe(path, "-select_streams", "v:0", "-show_entries", "packet=pts")["packets"]]
    assert stored == DUPLICATE_PTS, "precondition: the file really stores duplicate timestamps"
    result = probe(path)
    # A timestamp that does not exceed its predecessor is moved one tick behind it - nothing else moves.
    assert result.timeline.pts.tolist() == [0, 40, 80, 120, 160, 161, 240, 280, 320, 360, 361, 440]
    assert result.scan.method == "decode" and result.scan.adjusted_timestamps == 2
    assert result.timeline.synthesized is True
    assert any("strictly increasing" in w for w in result.metadata.warnings)
    assert result.metadata.frame_count == 12


def test_timestamps_that_run_backwards_are_not_mistaken_for_b_frames(derived):
    path = derived.path("backward_pts.mkv")
    stored = [int(p["pts"]) for p in media.ffprobe(path, "-select_streams", "v:0", "-show_entries", "packet=pts")["packets"]]
    assert stored == BACKWARD_PTS, "precondition: packet 30 is stamped earlier than packet 29"
    result = probe(path)
    # The stream has no B-frames, so the decoder delivers the frames in stored order: frame 30 comes
    # after frame 29 and its timestamp is moved behind it.  Sorting the packets would swap them.
    expected = list(BACKWARD_PTS)
    expected[30] = BACKWARD_PTS[29] + 1
    assert result.timeline.pts.tolist() == expected
    assert result.scan.method == "decode" and result.scan.adjusted_timestamps == 1
    assert any("strictly increasing" in w for w in result.metadata.warnings)


def test_vp8_hidden_frames_are_not_counted(derived):
    path = derived.path("vp8_altref.webm")
    counts = media.ffprobe(path, "-select_streams", "v:0", "-count_frames", "-count_packets",
                           "-show_entries", "stream=nb_read_frames,nb_read_packets")["streams"][0]
    packets, frames = int(counts["nb_read_packets"]), int(counts["nb_read_frames"])
    assert packets > frames == 120, "precondition: the stream contains alt-ref packets that are never shown"
    result = probe(path)
    assert result.metadata.frame_count == len(result.timeline) == frames
    assert timeline_times(result) == cli_frame_times(path)
    assert result.scan.packets == packets
    assert result.metadata.frame_rate == Fraction(30) and not result.metadata.is_vfr
    # Hidden frames are recognised from the packet header: no need to decode the whole file.
    # The alt-ref packets share the timestamp of the frame that follows them, so the packet scan is
    # not trusted and the stream is decoded once; the count and the times are exact either way.
    assert result.scan.method in ("packets", "decode")


def test_transport_stream_and_fragmented_start(derived):
    path = derived.path("transport.ts")
    result = probe(path)
    assert result.metadata.frame_count == 36 and result.timeline.pts[0] > 0      # MPEG-TS never starts at zero
    assert timeline_times(result) == media.packet_pts(path) == cli_frame_times(path)
    assert result.metadata.frame_rate == Fraction(30000, 1001)


# =========================================================================== real footage


def test_real_clips_agree_with_the_manifest(real_media):
    checked = 0
    for clip_id, entry in real_media.manifest.items():
        facts = entry.get("video")
        if not facts or not (real_media.root / entry["path"]).is_file():
            continue
        result = probe(real_media.root / entry["path"])
        meta, video = result.metadata, result.metadata.video
        assert meta.frame_count == len(result.timeline) == facts["nb_frames_decoded"], clip_id
        assert (video.width, video.height, video.pix_fmt) == (facts["width"], facts["height"], facts["pix_fmt"]), clip_id
        assert meta.frame_rate == Fraction(facts["avg_frame_rate"]), clip_id
        assert not meta.is_vfr, clip_id
        keys = int(result.timeline.keyframes.sum())
        # An edit list may hide a key frame; otherwise every key frame of the container is presented.
        assert keys == facts["keyframes"] or ("edit_list" in facts and 0 < keys < facts["keyframes"]), clip_id
        assert (video.color.range, video.color.space) == (facts["color_range"], facts["colorspace"]), clip_id
        assert meta.has_audio == (entry.get("audio") is not None), clip_id
        checked += 1
    if not checked:
        pytest.skip("no real test clips with recorded facts are available")


def test_real_edit_list_clip(real_media):
    path = real_media.path("M3_matanyone_sample3_trench_run")
    result = probe(path)
    meta = result.metadata
    assert meta.video.nb_frames == 650 and meta.frame_count == len(result.timeline) == 600
    assert result.scan.discarded == 50 and result.scan.method == "packets"
    assert result.timeline.pts[0] == 0 and not bool(result.timeline.keyframes[0])     # its key frame is hidden
    assert any("650" in w and "600" in w and "edit list" in w for w in meta.warnings)
    assert meta.frame_rate == Fraction(24) and not meta.is_vfr


def test_real_rotation_and_reflection_clip(real_media):
    meta = probe(real_media.path("T1_rotate_and_reflect")).metadata
    assert meta.video.rotation == 90 and (meta.video.width, meta.video.height) == (1080, 720)
    assert (meta.video.display_width, meta.video.display_height) == (720, 1080)
    assert any("mirror" in w.lower() for w in meta.warnings)


def test_real_phone_clip(real_media):
    result = probe(real_media.path("T2_phone_portrait_4k60"))
    meta, video = result.metadata, result.metadata.video
    assert video.rotation == 90 and (video.display_width, video.display_height) == (2160, 3840)
    assert (video.pix_fmt, video.color.range, video.color.space) == ("yuvj420p", "pc", "bt470bg")
    assert meta.is_vfr and meta.vfr["distinct_durations"] > 2             # a phone's frame timing wanders
    assert meta.frame_count == 18 and meta.has_audio
    assert timeline_times(result) == media.packet_pts(real_media.path("T2_phone_portrait_4k60"))


def test_real_iphone_dolby_vision_clip(real_media):
    result = probe(real_media.path("T3_iphone_dolby_vision_hlg"))
    meta, video = result.metadata, result.metadata.video
    assert meta.container == "mov" and (video.codec, video.pix_fmt, video.bit_depth) == ("hevc", "yuv420p10le", 10)
    assert (video.color.space, video.color.primaries, video.color.transfer) == ("bt2020nc", "bt2020", "arib-std-b67")
    assert video.color.is_hdr and video.rotation == 90
    assert video.hdr["dolby_vision"]["dv_profile"] == 8 and "ambient_viewing_environment" in video.hdr
    assert meta.data_streams == 3 and meta.subtitle_streams == 0
    assert (video.nb_frames, meta.frame_count) == (5, 4)                  # one packet is hidden by the edit list
    json.dumps(meta.to_dict())


def test_real_full_range_and_long_gop_clips(real_media):
    full = probe(real_media.path("B4_intel_classroom_1080p_fullrange")).metadata
    assert (full.video.pix_fmt, full.video.color.range, full.video.color.space) == ("yuvj420p", "pc", "unknown")
    assert not full.has_audio and full.frame_count == 984
    long_gop = probe(real_media.path("B3_sam2_bedroom_540p"))
    assert long_gop.metadata.frame_count == 200 and np.flatnonzero(long_gop.timeline.keyframes).tolist() == [0]
    derived_phone = probe(real_media.path("D1_derived_phone_portrait")).metadata
    assert derived_phone.video.rotation == 90 and derived_phone.warnings == ()


# =========================================================================== errors


def test_missing_and_non_file_paths(tmp_path):
    with pytest.raises(InvalidPathError) as missing:
        probe(tmp_path / "does not exist.mp4")
    assert "not found" in missing.value.message and missing.value.code == "invalid_path"
    with pytest.raises(InvalidPathError) as folder:
        probe(tmp_path)
    assert "not a file" in folder.value.message
    with pytest.raises(InvalidPathError):
        probe(12345)
    with pytest.raises(InvalidPathError):
        probe(None)


def test_ffprobe_missing(clip_cfr, tmp_path, monkeypatch):
    with pytest.raises(FfmpegNotFoundError) as explicit:
        probe(clip_cfr.path, ffprobe=str(tmp_path / "no_such_ffprobe.exe"))
    assert "no_such_ffprobe" in explicit.value.message and explicit.value.hint

    monkeypatch.setenv("VMT_FFPROBE", str(tmp_path / "also_missing.exe"))
    with pytest.raises(FfmpegNotFoundError):
        probe(clip_cfr.path)

    monkeypatch.delenv("VMT_FFPROBE")
    monkeypatch.setattr(probe_module.shutil, "which", lambda name: None)
    with pytest.raises(FfmpegNotFoundError) as on_path:
        probe(clip_cfr.path)
    assert "PATH" in on_path.value.message


def test_subprocess_failures_never_escape(clip_cfr, monkeypatch):
    def vanished(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(probe_module.subprocess, "run", vanished)
    with pytest.raises(FfmpegNotFoundError):
        probe(clip_cfr.path)

    def hangs(command, *args, **kwargs):
        raise subprocess.TimeoutExpired(command, 120.0, stderr=b"still reading")

    monkeypatch.setattr(probe_module.subprocess, "run", hangs)
    with pytest.raises(FfmpegFailedError) as timeout:
        probe(clip_cfr.path)
    assert "did not finish" in timeout.value.message and timeout.value.details["stderr_tail"] == "still reading"

    def denied(*args, **kwargs):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(probe_module.subprocess, "run", denied)
    with pytest.raises(FfmpegFailedError):
        probe(clip_cfr.path)

    def garbage(command, *args, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=b"{ not json", stderr=b"")

    monkeypatch.setattr(probe_module.subprocess, "run", garbage)
    with pytest.raises(FfmpegFailedError):
        probe(clip_cfr.path)


@pytest.mark.parametrize("suffix", [".mp4", ".mov", ".mkv", ".webm", ".avi"])
def test_text_file_with_a_video_extension_is_not_media(tmp_path, suffix):
    path = tmp_path / f"notes{suffix}"
    path.write_text("These are my notes, not a video.\n" * 300, encoding="utf-8")
    with pytest.raises(UnsupportedFormatError) as error:
        probe(path)
    assert path.name in error.value.message and error.value.code == "unsupported_format"
    assert "stderr_tail" in error.value.details and error.value.hint


def test_other_non_media_files(tmp_path):
    binary = tmp_path / "random.mp4"
    binary.write_bytes(bytes((i * 7919 + 13) % 251 for i in range(40_000)))
    with pytest.raises(UnsupportedFormatError):
        probe(binary)
    # FFmpeg would happily render a text file as ANSI art; that is not a video either.
    text = tmp_path / "readme.txt"
    text.write_text("plain text\n" * 100, encoding="utf-8")
    with pytest.raises(UnsupportedFormatError) as error:
        probe(text)
    assert "text" in error.value.message


def test_files_without_a_video_stream(derived):
    with pytest.raises(NoVideoStreamError) as audio:
        probe(derived.path("audio_only.m4a"))
    assert "audio only" in audio.value.message and audio.value.code == "no_video_stream"
    with pytest.raises(NoVideoStreamError) as cover:
        probe(derived.path("cover_only.mp3"))
    assert "attached picture" in cover.value.message


def test_empty_file(tmp_path):
    path = tmp_path / "empty.mp4"
    path.write_bytes(b"")
    with pytest.raises(CorruptedVideoError) as error:
        probe(path)
    assert "empty" in error.value.message


def test_truncated_mp4_without_index(clip_cfr, tmp_path):
    path = truncated_copy(clip_cfr.path, tmp_path / "interrupted.mp4", 0.6)
    with pytest.raises(CorruptedVideoError) as error:
        probe(path)
    assert "moov" in error.value.message and "truncated" in error.value.message
    assert "moov atom not found" in error.value.details["stderr_tail"]         # what ffprobe said
    assert error.value.hint and error.value.code == "corrupted_video"


def test_header_without_frames(clip_noaudio_mkv, clip_webm, tmp_path):
    for clip in (clip_noaudio_mkv, clip_webm):
        path = truncated_copy(clip.path, tmp_path / f"header_only{clip.path.suffix}", 0.01)
        with pytest.raises(CorruptedVideoError) as error:
            probe(path)
        assert "No video frame" in error.value.message or "cannot read" in error.value.message
        assert error.value.details


def test_truncated_matroska_is_read_up_to_the_cut(clip_noaudio_mkv, tmp_path):
    path = truncated_copy(clip_noaudio_mkv.path, tmp_path / "cut.mkv", 0.6)
    result = probe(path)
    assert 10 < result.metadata.frame_count < clip_noaudio_mkv.frames
    got = result.timeline.pts.tolist()
    # Every presented frame is a frame of the source, in order.  A reference frame that was stored
    # before the cut is still presented although the B-frames in front of it were lost (FFmpeg
    # delivers it as well), so the tail may have a gap.
    assert got == sorted(set(got)) and set(got) <= set(clip_noaudio_mkv.pts)
    assert got[:10] == clip_noaudio_mkv.pts[:10]
    assert any("truncated" in w for w in result.metadata.warnings)


def test_truncated_mp4_with_leading_index_is_read_up_to_the_cut(derived, tmp_path):
    whole = derived.path("faststart.mp4")
    path = truncated_copy(whole, tmp_path / "cut_faststart.mp4", 0.6)
    result = probe(path)
    meta = result.metadata
    # The index still lists all 36 frames; only those whose data is complete are presented.  The
    # packet cut in half by the truncation is not one of them.
    complete = [p for p in av_packets(path) if not p["corrupt"]]
    assert meta.video.nb_frames == 36 and meta.frame_count == len(complete) >= 15
    assert result.timeline.pts.tolist() == sorted(p["pts"] for p in complete)
    assert any("36" in w for w in meta.warnings) and any("truncated" in w for w in meta.warnings)


def av_packets(path: Path) -> List[dict]:
    """Video packets as PyAV demuxes them (reference for flags that ffprobe does not print)."""

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return [{"pts": p.pts, "corrupt": p.is_corrupt, "key": p.is_keyframe}
                for p in container.demux(stream) if p.size]


def test_damage_in_the_middle_is_left_to_the_reader(clip_cfr, clip_noaudio_mkv, tmp_path):
    # A block of garbage in the middle of the file does not stop the analysis: the container is still
    # readable, so the probe describes it and the reader reports the frames that fail to decode.
    for clip in (clip_cfr, clip_noaudio_mkv):
        path = damaged_copy(clip.path, tmp_path / f"damaged{clip.path.suffix}", 0.5, 4096)
        result = probe(path)
        assert 0 < result.metadata.frame_count <= clip.frames
        assert bool(np.all(np.diff(result.timeline.pts) > 0))


def test_no_low_level_exception_escapes(clip_cfr, clip_noaudio_mkv, clip_webm, derived, tmp_path):
    sources = [clip_cfr.path, clip_noaudio_mkv.path, clip_webm.path, derived.path("mpeg4_bframes.avi"),
               derived.path("faststart.mp4")]
    outcomes = {"ok": 0, "error": 0}
    for source in sources:
        variants = [truncated_copy(source, tmp_path / f"t{int(f * 100)}_{source.name}", f) for f in (0.03, 0.35, 0.8)]
        variants += [damaged_copy(source, tmp_path / f"d{int(f * 100)}_{source.name}", f, 3000, fill)
                     for f, fill in ((0.0, 0x00), (0.02, 0xFF), (0.4, 0xA5), (0.97, 0x00))]
        for path in variants:
            try:
                result = probe(path)
            except VmtError as error:
                assert isinstance(error, (InputError, FfmpegFailedError)) and error.message
                outcomes["error"] += 1
            else:
                assert result.metadata.frame_count == len(result.timeline) > 0
                assert bool(np.all(np.diff(result.timeline.pts) > 0))
                json.dumps(result.metadata.to_dict())
                outcomes["ok"] += 1
    assert outcomes["ok"] and outcomes["error"], outcomes            # both branches were exercised


# =========================================================================== paths


def test_non_ascii_path_with_spaces(derived):
    path = derived.path(UNICODE_NAME)
    assert " " in path.name and not path.name.isascii()
    result = probe(str(path))
    assert result.metadata.frame_count == 36 and result.metadata.filename == UNICODE_NAME
    assert Path(result.metadata.path) == path
    assert json.loads(json.dumps(result.metadata.to_dict()))["filename"] == UNICODE_NAME


def test_relative_path_is_made_absolute(clip_cfr, monkeypatch):
    monkeypatch.chdir(clip_cfr.path.parent)
    result = probe(clip_cfr.path.name)
    assert Path(result.metadata.path) == clip_cfr.path


# =========================================================================== timestamp fixer


def test_fixer_passes_good_timestamps_through():
    fixer = TimestampFixer(Fraction(1, 1000))
    assert [fixer(p, p - 40, 40) for p in (-80, -40, 0, 40, 1000)] == [-80, -40, 0, 40, 1000]
    assert (fixer.frames, fixer.synthesized, fixer.adjusted) == (5, 0, 0)


def test_fixer_moves_duplicates_and_backward_steps_by_one_tick():
    fixer = TimestampFixer(Fraction(1, 1000))
    assert [fixer(p, p, 40) for p in (0, 40, 40, 30, 200)] == [0, 40, 41, 42, 200]
    assert (fixer.synthesized, fixer.adjusted) == (2, 2)


def test_fixer_extrapolates_missing_timestamps_like_ffmpeg():
    # No timestamps at all (raw stream): frame duration from the codec's frame rate.
    raw = TimestampFixer(Fraction(1, 1_200_000), codec_rate=Fraction(25), ts_unreliable=True)
    assert [raw(None, None, 0) for _ in range(4)] == [0, 48000, 96000, 144000]
    assert raw.synthesized == 4 and raw.adjusted == 0
    # Timestamps present but declared unreliable by the format are ignored as well.
    ignored = TimestampFixer(Fraction(1, 1_200_000), codec_rate=Fraction(25), ts_unreliable=True)
    assert [ignored(7, 7, 0), ignored(9, 9, 0)] == [0, 48000]
    # AVI-style: decoding timestamps only; the best-effort value is the DTS, gaps continue by duration.
    avi = TimestampFixer(Fraction(1001, 30000))
    assert [avi(None, d, 1) for d in (0, 1, 2)] == [0, 1, 2]
    assert avi(None, None, 1) == 3


# =========================================================================== speed (reported, not asserted)


def test_probe_speed_on_a_long_clip(derived, capsys):
    path = derived.path("long.mp4")
    started = time.perf_counter()
    result = probe(path)
    elapsed = time.perf_counter() - started
    frames = result.metadata.frame_count
    assert frames == 18000 and result.scan.method == "packets"
    with capsys.disabled():
        print(f"\n[speed] probe of {frames} frames ({float(result.metadata.video_duration):.0f} s of video): "
              f"{elapsed:.3f} s total, timeline scan {result.scan.seconds:.3f} s "
              f"({frames / max(result.scan.seconds, 1e-9):,.0f} packets/s)")


@pytest.mark.slow
def test_probe_speed_when_the_whole_stream_must_be_decoded(real_media, tmp_path, capsys):
    # H.264 in AVI carries no presentation timestamps: the timeline needs a full decode.
    source = real_media.path("A2_cbp_videoblog_speech")
    path = tmp_path / "speech.avi"
    ffmpeg("-i", source, "-an", "-c:v", "copy", path)
    started = time.perf_counter()
    result = probe(path)
    elapsed = time.perf_counter() - started
    assert result.scan.method == "decode" and result.metadata.frame_count == 954
    with capsys.disabled():
        print(f"\n[speed] probe with full decode, {result.metadata.frame_count} frames 1024x576: {elapsed:.3f} s "
              f"({result.metadata.frame_count / elapsed:,.0f} frames/s)")
