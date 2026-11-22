"""Synthetic test media.

Clips are generated on the fly (no binary assets in the repository).  The picture is a textured,
moving background with a magenta "subject" disc whose position is a known function of the frame
index, so tests can predict where the subject is and a trivial chroma model can matte it.
The video stream is written with PyAV using explicit per-frame timestamps (so true variable frame
rate is possible); audio and container variations are added with the ffmpeg CLI.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import av
import numpy as np

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

SUBJECT_RGB = (235, 20, 235)          # magenta: min(R, B) - G is large only on the subject


def subject_center(index: int, width: int, height: int, frames: int) -> Tuple[float, float]:
    """Centre of the main subject disc in frame *index* (moves left to right with a slight bob)."""

    t = index / max(1, frames - 1)
    cx = width * (0.25 + 0.5 * t)
    cy = height * (0.5 + 0.08 * math.sin(2.0 * math.pi * t))
    return cx, cy


def subject_radius(width: int, height: int) -> float:
    return 0.22 * min(width, height)


def subject_alpha(index: int, width: int, height: int, frames: int, soft: float = 2.5) -> np.ndarray:
    """Ground-truth matte (float32 0..1) of the main subject for frame *index*."""

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = subject_center(index, width, height, frames)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip((subject_radius(width, height) - r) / soft + 0.5, 0.0, 1.0).astype(np.float32)


def distractor_alpha(width: int, height: int, soft: float = 2.0) -> np.ndarray:
    """A second, smaller magenta blob in the top-right corner (for subject-isolation tests)."""

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    r = np.sqrt((xx - width * 0.88) ** 2 + (yy - height * 0.18) ** 2)
    return np.clip((0.09 * min(width, height) - r) / soft + 0.5, 0.0, 1.0).astype(np.float32)


def synth_frame(index: int, width: int, height: int, frames: int, two_subjects: bool = False) -> np.ndarray:
    """RGB uint8 frame: busy moving background (never magenta) + magenta subject disc(s)."""

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    ph = index * 0.21
    # Green always dominates red and blue by a clear margin, so the background can never be
    # mistaken for the magenta subject, not even after lossy compression.
    r = 0.26 + 0.14 * np.sin(xx * 0.11 + ph) * np.cos(yy * 0.07)
    g = 0.70 + 0.14 * np.sin(yy * 0.13 - ph * 0.7) + 0.06 * np.sin((xx + yy) * 0.31 + ph)
    b = 0.24 + 0.14 * np.cos(xx * 0.05 - yy * 0.09 + ph * 0.5)
    rng = np.random.default_rng(1000 + index)
    noise = rng.normal(0.0, 0.012, size=(height, width)).astype(np.float32)
    bg = np.stack([r + noise, g + noise, b + noise], axis=-1)
    bg = np.clip(bg, 0.0, 1.0)
    a = subject_alpha(index, width, height, frames)
    if two_subjects:
        a = np.maximum(a, distractor_alpha(width, height))
    fg = np.asarray(SUBJECT_RGB, dtype=np.float32) / 255.0
    out = bg * (1.0 - a[..., None]) + fg[None, None, :] * a[..., None]
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def make_pts(frames: int, fps: Fraction, time_base: Fraction, vfr: bool, start: int = 0) -> List[int]:
    """Presentation timestamps (in *time_base* ticks). ``vfr`` produces genuinely irregular spacing."""

    step = Fraction(1, 1) / fps / time_base
    if not vfr:
        return [start + int(round(i * step)) for i in range(frames)]
    pattern = [1.0, 1.0, 2.0, 0.5, 1.0, 1.5, 0.5, 3.0, 1.0, 1.0, 0.75, 1.25]
    pts, t = [], float(start)
    for i in range(frames):
        pts.append(int(round(t)))
        t += float(step) * pattern[i % len(pattern)]
    for i in range(1, frames):                       # strictly increasing after rounding
        if pts[i] <= pts[i - 1]:
            pts[i] = pts[i - 1] + 1
    return pts


def write_video(
    path: Path,
    *,
    width: int = 320,
    height: int = 180,
    frames: int = 36,
    fps: Fraction = Fraction(30000, 1001),
    time_base: Optional[Fraction] = None,
    pix_fmt: str = "yuv420p",
    codec: str = "libx264",
    vfr: bool = False,
    start_pts: int = 0,
    two_subjects: bool = False,
    options: Optional[Dict[str, str]] = None,
    color: Optional[Dict[str, str]] = None,
) -> List[int]:
    """Encode a video-only file with PyAV and return the list of PTS that were written.

    ``color`` may carry ffmpeg-style tags, e.g. ``{"colorspace": "bt709", "color_range": "tv"}``;
    they are applied with a stream-copy pass because PyAV does not expose all of them uniformly.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tb = time_base or Fraction(1, fps.numerator if fps.denominator != 1 else fps.numerator * 1000)
    pts_list = make_pts(frames, fps, tb, vfr, start_pts)
    raw = path if not color else path.with_name(path.stem + "_untagged" + path.suffix)
    container = av.open(str(raw), mode="w")
    try:
        stream = container.add_stream(codec, rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = pix_fmt
        stream.time_base = tb
        stream.codec_context.time_base = tb
        opts = {"crf": "14", "preset": "veryfast", "bf": "2", "g": "12"} if codec in ("libx264", "libx265") else {}
        opts.update(options or {})
        if opts:
            stream.options = opts
        for i, pts in enumerate(pts_list):
            rgb = synth_frame(i, width, height, frames, two_subjects)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format=pix_fmt)
            frame.pts = pts
            frame.time_base = tb
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    if color:
        args = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw), "-map", "0", "-c", "copy"]
        if any(k in color for k in ("colorspace", "color_primaries", "color_trc", "color_range")):
            bsf_name = {"libx264": "h264_metadata", "libx265": "hevc_metadata"}.get(codec)
            ids = {
                "colorspace": ("matrix_coefficients", {"bt709": 1, "smpte170m": 6, "bt470bg": 5, "bt2020nc": 9}),
                "color_primaries": ("colour_primaries", {"bt709": 1, "smpte170m": 6, "bt470bg": 5, "bt2020": 9}),
                "color_trc": ("transfer_characteristics", {"bt709": 1, "smpte170m": 6, "arib-std-b67": 18, "smpte2084": 16}),
            }
            parts = []
            for key, (field, table) in ids.items():
                if key in color:
                    parts.append(f"{field}={table[color[key]]}")
            if "color_range" in color:
                parts.append(f"video_full_range_flag={1 if color['color_range'] in ('pc', 'jpeg', 'full') else 0}")
            if bsf_name and parts:
                args += ["-bsf:v", f"{bsf_name}=" + ":".join(parts)]
        args.append(str(path))
        run(args)
        raw.unlink(missing_ok=True)
    return pts_list


def run(args: Sequence[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(list(args), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(map(str, args))}\n{result.stderr[-2000:]}")
    return result


def add_audio(
    video: Path,
    out: Path,
    *,
    codec: str = "aac",
    frequency: int = 440,
    sample_rate: int = 48000,
    audio_offset: float = 0.0,
    extra: Sequence[str] = (),
) -> Path:
    """Mux a sine tone (same length as the video) into *out*, stream-copying the video."""

    dur = float(ffprobe(video)["format"]["duration"])
    args = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video)]
    if audio_offset:
        args += ["-itsoffset", str(audio_offset)]
    args += [
        "-f", "lavfi", "-t", f"{dur:.6f}", "-i", f"sine=frequency={frequency}:sample_rate={sample_rate}",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", codec,
    ]
    if codec in ("aac", "libopus", "libmp3lame"):
        args += ["-b:a", "96k"]
    args += list(extra) + [str(out)]
    run(args)
    return out


def with_rotation(video: Path, out: Path, degrees: int) -> Path:
    """Stream-copy *video* adding a display-matrix rotation (ffmpeg's -display_rotation convention)."""

    run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-display_rotation", str(degrees), "-i", str(video),
         "-map", "0", "-c", "copy", str(out)])
    return out


def ffprobe(path: Path, *extra: str) -> dict:
    result = run([FFPROBE, "-v", "error", "-of", "json", "-show_format", "-show_streams", *extra, str(path)])
    return json.loads(result.stdout)


def packet_pts(path: Path, stream: str = "v:0") -> List[Fraction]:
    """Exact presentation times (seconds, as Fractions) of every packet of a stream, sorted."""

    data = ffprobe(path, "-select_streams", stream, "-show_entries", "packet=pts:stream=time_base")
    tb = Fraction(data["streams"][0]["time_base"])
    return sorted(Fraction(int(p["pts"])) * tb for p in data.get("packets", []) if "pts" in p)


def decode_planes(path: Path, max_frames: Optional[int] = None) -> List[List[np.ndarray]]:
    """Decode a file with PyAV and return native planes per frame (no colour conversion)."""

    out: List[List[np.ndarray]] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            planes = []
            for p in frame.planes:
                bps = 2 if p.line_size >= 2 * p.width and frame.format.components[0].bits > 8 else 1
                dtype = np.uint8 if bps == 1 else np.dtype("<u2")
                arr = np.frombuffer(p, dtype=dtype).reshape(p.height, p.line_size // bps)[:, : p.width]
                planes.append(arr.copy())
            out.append(planes)
            if max_frames is not None and len(out) >= max_frames:
                break
    return out
