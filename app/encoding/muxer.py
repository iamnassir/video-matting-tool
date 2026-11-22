"""Final assembly: the re-encoded video plus the source's audio, rotation and metadata.

The encoder writes a *video-only* file whose frames carry the source's presentation timestamps.
:func:`mux_output` stream-copies that video (it is never re-encoded here) together with the audio
streams of the source into the deliverable.

What the recipe guarantees (each point was validated with packet-level and content-level tests):

* **Audio is stream-copied** whenever the container can hold the codec; otherwise that stream is
  re-encoded and a warning says why.
* **A/V offset is preserved.**  A plain two-input ``ffmpeg -i video -i source`` copy destroys sync
  whenever the video does not start first, because FFmpeg rebases every input to its own start.
  ``-copyts`` keeps the absolute timestamps of both inputs instead.  If *every* stream of the source
  starts late (transport-stream captures often start at 1.4 s or at one hour) the same offset is
  subtracted from both inputs, so the result starts at zero like a player would show the source,
  with the relationship between the streams untouched.  The applied shift is reported in
  :attr:`MuxResult.time_shift`.
* **Rotation** metadata (the display matrix, including mirroring) is recreated on the copied video
  stream; pixels are never rotated.
* **Metadata**: global tags, creation time, chapters, the video stream's language / handler name and
  a QuickTime time code are carried over.  Data streams (phone metadata tracks) are never mapped:
  they break muxing.

The module also extracts audio streams without re-encoding (:func:`extract_audio`) and builds the
small browser preview of a source video (:func:`make_proxy`).
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..core.config import EncodingConfig
from ..core.errors import AudioMuxError, ConfigError, FfmpegFailedError, JobCancelledError, VmtError
from ..core.events import CancelToken
from ..core.types import AudioStreamInfo, VideoMetadata
from ..utils.fs import atomic_replace, safe_unlink
from .ffmpeg import CREATE_NO_WINDOW, FfmpegTools, ffprobe_json, run_ffmpeg
from .profiles import MUXER_NAMES

log = logging.getLogger(__name__)

__all__ = ["MuxResult", "extract_audio", "make_proxy", "mux_output"]

_MAX_TIMESCALE = (1 << 31) - 1
_MOV_LIKE = ("mp4", "mov")

#: Audio codecs each container can hold at all (verified by muxing and reading back).
_MUXABLE: Dict[str, FrozenSet[str]] = {
    "mp4": frozenset({"aac", "mp3", "ac3", "eac3", "opus", "vorbis", "flac", "alac", "dts",
                      "pcm_s16le", "pcm_s24le", "pcm_s16be", "pcm_f32le"}),
    "mov": frozenset({"aac", "mp3", "ac3", "eac3", "vorbis", "alac", "dts", "mp2", "amr_nb",
                      "pcm_s16le", "pcm_s24le", "pcm_s16be", "pcm_f32le", "pcm_mulaw"}),
    "webm": frozenset({"opus", "vorbis"}),
}
#: The conservative subset that players reliably play; used by the automatic mode.
_COPY_POLICY: Dict[str, FrozenSet[str]] = {
    "mp4": frozenset({"aac", "mp3", "ac3", "eac3", "alac", "opus", "flac"}),
    "mov": frozenset({"aac", "mp3", "ac3", "eac3", "alac", "pcm_s16le", "pcm_s24le", "pcm_s16be", "pcm_f32le"}),
    "webm": frozenset({"opus", "vorbis"}),
}
_LOSSLESS_AUDIO = ("flac", "alac", "truehd", "mlp", "wavpack", "ape", "tta")

#: Encoder and the containers it may be written to, per forced ``audio_codec`` setting.
_FORCED: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "aac": ("aac", ("mp4", "mov", "mkv")),
    "opus": ("libopus", ("mp4", "mkv", "webm")),
    "flac": ("flac", ("mp4", "mkv")),
    "pcm": ("pcm_s24le", ("mov", "mkv")),
}

#: File extension for a stream-copied audio export, by codec (everything else goes to Matroska audio).
_AUDIO_EXTENSIONS: Dict[str, str] = {
    "aac": ".m4a", "alac": ".m4a", "mp3": ".mp3", "opus": ".opus", "vorbis": ".ogg", "flac": ".flac",
    "ac3": ".ac3", "eac3": ".eac3", "pcm_s16le": ".wav", "pcm_s24le": ".wav", "pcm_s32le": ".wav",
    "pcm_f32le": ".wav", "pcm_u8": ".wav",
}


@dataclass
class MuxResult:
    path: Path
    audio_mode: str                     # "copy" | "transcode" | "none"
    audio_streams: int
    warnings: List[str]
    command: List[str]
    #: Seconds added to every source timestamp in the output (0, or minus the common start offset).
    time_shift: Fraction = Fraction(0)
    #: Per audio stream: source codec, action ("copy" / "transcode") and output codec.
    audio_plan: List[Dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- helpers


def _is_lossless(codec: str) -> bool:
    return codec.startswith("pcm_") or codec in _LOSSLESS_AUDIO


def _can_copy(codec: str, container: str, strict: bool) -> bool:
    """*strict* = the automatic policy (players must cope); otherwise "the container accepts it"."""

    if container == "mkv":
        return True                      # Matroska holds every codec FFmpeg can demux
    table = _COPY_POLICY if strict else _MUXABLE
    return codec in table.get(container, frozenset())


def _fallback_encoder(container: str, stream: AudioStreamInfo, bitrate: str) -> Tuple[str, List[str]]:
    """Encoder name and options used when a stream cannot be copied into *container*."""

    if container == "webm":
        return "libopus", ["-b:a:{i}", bitrate]
    if container == "mov" and _is_lossless(stream.codec):
        return "pcm_s24le", []
    if container == "mkv" and _is_lossless(stream.codec):
        return "flac", []
    return "aac", ["-b:a:{i}", bitrate]


def _display_matrix(tools: FfmpegTools, path: str) -> Tuple[Optional[int], bool]:
    """``(rotation, flipped)`` of the first real video stream as FFmpeg reports it.

    *rotation* is the counter-clockwise angle of the display matrix - the same number that
    ``-display_rotation`` takes - or None when the file has no display matrix.
    """

    try:
        data = ffprobe_json(tools.ffprobe, path, ["-select_streams", "V:0", "-show_entries", "stream_side_data_list"])
    except VmtError as exc:
        log.debug("Could not read the display matrix of %s: %s", path, exc)
        return None, False
    for stream in data.get("streams", []):
        for side in stream.get("side_data_list") or []:
            if "rotation" not in side and "displaymatrix" not in side:
                continue
            flipped = False
            numbers: List[int] = []
            for token in str(side.get("displaymatrix", "")).split():
                if token.endswith(":"):
                    continue
                try:
                    numbers.append(int(token))
                except ValueError:
                    pass
            if len(numbers) == 9:
                flipped = (numbers[0] * numbers[4] - numbers[1] * numbers[3]) < 0
            try:
                rotation = int(round(float(side.get("rotation", 0))))
            except (TypeError, ValueError):
                rotation = 0
            return rotation, flipped
    return None, False


def _rotation_args(tools: FfmpegTools, source: VideoMetadata, rotation: Optional[int], warnings: List[str]) -> List[str]:
    """Input options for the video-only file that attach the wanted display matrix."""

    if rotation is None:
        angle, flipped = _display_matrix(tools, source.path)
        if angle is None:
            return []
        if flipped:
            return ["-display_rotation:v:0", str(-angle), "-display_vflip:v:0"]
        return ["-display_rotation:v:0", str(angle)] if angle else []
    clockwise = int(rotation) % 360
    if clockwise == 0:
        return []
    if clockwise % 90:
        warnings.append(f"Rotation {rotation} is not a multiple of 90 degrees; it was not written.")
        return []
    counter_clockwise = -clockwise if clockwise <= 180 else 360 - clockwise
    return ["-display_rotation:v:0", str(counter_clockwise)]


def _hidden_audio_lead(tools: FfmpegTools, source: VideoMetadata) -> float:
    """Seconds of audio that an MP4 edit list hides before the stream start (0 if none / unknown).

    Matroska and WebM cannot hide such lead-in audio, so it would become audible and shift the
    sound; ``-copypriorss 0`` drops it when it is longer than a codec's priming.
    """

    lead = 0.0
    for number, stream in enumerate(source.audio):
        if stream.start_time is None:
            continue
        try:
            result = subprocess.run(
                [tools.ffprobe, "-v", "error", "-select_streams", f"a:{number}", "-read_intervals", "%+#1",
                 "-show_entries", "packet=pts_time", "-of", "csv=p=0", source.path],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, creationflags=CREATE_NO_WINDOW,
            )
            first = result.stdout.strip().splitlines()[0].split(",")[0] if result.stdout.strip() else ""
            if first and first != "N/A":
                lead = max(lead, float(stream.start_time) - float(first))
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            continue
    return lead


def _movie_timescale(time_base: Fraction) -> Optional[int]:
    """A movie time scale that represents video ticks exactly and audio offsets to ~11 microseconds."""

    den = int(time_base.denominator)
    if den <= 0 or den > _MAX_TIMESCALE:
        return None
    factor = max(1, -(-90000 // den))             # ceil(90000 / den)
    scale = den * factor
    return scale if scale <= _MAX_TIMESCALE else den


def _seconds(value: Fraction) -> str:
    return f"{float(value):.6f}"


# --------------------------------------------------------------------------- the mux


def mux_output(
    tools: FfmpegTools,
    *,
    video_only: Path,
    source: VideoMetadata,
    output: Path,
    container: str,
    encoding: EncodingConfig,
    include_audio: bool = True,
    audio_mode: str = "auto",
    rotation: Optional[int] = None,
    copy_metadata: bool = True,
    audio_end: Optional[Fraction] = None,
    video_shift: Fraction = Fraction(0),
    keep_aspect: bool = True,
    cancel: Optional[CancelToken] = None,
) -> MuxResult:
    """Write *output*: the video stream of *video_only* (copied) plus the audio of the source.

    * ``audio_mode``: ``auto`` (copy where the container allows, else re-encode with a warning),
      ``copy`` (copy or raise :class:`AudioMuxError`), or ``aac`` / ``opus`` / ``flac`` / ``pcm``
      (always re-encode to that codec).  ``include_audio=False`` writes no audio.
    * ``rotation``: None = reproduce the source's display matrix (rotation and mirroring); 0 = none;
      another multiple of 90 = that clockwise rotation.
    * ``audio_end``: cut the audio at this source time in seconds (partial renders).
    * ``video_shift``: seconds by which the timeline of *video_only* is ahead of the source's (the
      encoder shifts a negative first timestamp to zero); it is undone here.
    * ``keep_aspect``: carry over a non-square pixel aspect ratio (only valid for unscaled output).

    A failure leaves no file at *output*.
    """

    if container not in MUXER_NAMES:
        raise ConfigError(f"Unknown output container {container!r}.", hint="Use mp4, mov, mkv or webm.")
    video_only, output = Path(video_only), Path(output)
    warnings: List[str] = []
    mode = (audio_mode or "auto").lower()
    streams: Sequence[AudioStreamInfo] = source.audio if include_audio else ()

    # ---- what happens to each audio stream --------------------------------------------------
    plan: List[Dict[str, Any]] = []
    for number, stream in enumerate(streams):
        entry: Dict[str, Any] = {"index": number, "source_codec": stream.codec, "action": "copy", "codec": stream.codec,
                                 "args": []}
        if mode in _FORCED:
            encoder, containers = _FORCED[mode]
            if container not in containers:
                raise AudioMuxError(
                    f"Audio codec '{mode}' cannot be stored in a .{container} file.",
                    hint=f"Use audio codec 'auto', or one of the containers {', '.join('.' + c for c in containers)}.",
                )
            entry.update(action="transcode", codec=encoder,
                         args=[] if encoder in ("flac", "pcm_s24le") else ["-b:a:{i}", encoding.audio_bitrate])
        elif mode == "copy":
            if not _can_copy(stream.codec, container, strict=False):
                raise AudioMuxError(
                    f"The {stream.codec} audio of the source cannot be stream-copied into a .{container} file.",
                    hint="Use audio codec 'auto' (re-encodes only when necessary) or an .mkv output, which "
                    "accepts every audio codec.",
                )
        elif mode == "auto":
            if not _can_copy(stream.codec, container, strict=True):
                encoder, args = _fallback_encoder(container, stream, encoding.audio_bitrate)
                entry.update(action="transcode", codec=encoder, args=args)
                warnings.append(
                    f"Audio stream {number + 1} ({stream.codec}) cannot be stored unchanged in .{container}: "
                    f"it was re-encoded to {encoder}. Choose an .mkv output to keep the original audio bit-exact."
                )
        else:
            raise ConfigError(f"Unknown audio mode {audio_mode!r}.", hint="Use auto, copy, aac, opus, flac or pcm.")
        plan.append(entry)

    # ---- timeline: keep absolute timestamps; remove a start offset common to all streams ------
    starts = [source.video.start_time] + [a.start_time for a in source.audio]
    known = [s for s in starts if s is not None]
    earliest = min(known) if known else Fraction(0)
    shift = -earliest if earliest > Fraction(1, 1_000_000) else Fraction(0)
    video_offset = shift - Fraction(video_shift)

    rotation_args = _rotation_args(tools, source, rotation, warnings)
    den_scale = _movie_timescale(source.video.time_base)

    def build(optional_metadata: bool, audio_plan: List[Dict[str, Any]]) -> List[str]:
        args: List[str] = ["-v", "error", "-copyts"]
        if video_offset != 0:
            args += ["-itsoffset", _seconds(video_offset)]
        args += [*rotation_args, "-i", str(video_only)]
        if audio_plan or copy_metadata:
            if shift != 0:
                args += ["-itsoffset", _seconds(shift)]
            args += ["-i", source.path]
        args += ["-map", "0:v:0"]
        codec_args = ["-c:v", "copy"]
        for out_index, entry in enumerate(audio_plan):
            args += ["-map", f"1:a:{entry['index']}"]
            if entry["action"] == "copy":
                codec_args += [f"-c:a:{out_index}", "copy"]
            else:
                codec_args += [f"-c:a:{out_index}", entry["codec"]]
                codec_args += [part.replace("{i}", str(out_index)) for part in entry["args"]]
        args += codec_args
        sar = source.video.sample_aspect_ratio
        dar = source.video.display_aspect_ratio
        if keep_aspect and sar and sar != 1 and dar:
            args += ["-aspect:v:0", f"{dar.numerator}:{dar.denominator}"]
        if copy_metadata:
            args += ["-map_metadata", "1"]
            args += ["-map_chapters", "1"] if optional_metadata else ["-map_chapters", "-1"]
            language = source.video.tags.get("language")
            if language and language != "und":
                args += ["-metadata:s:v:0", f"language={language}"]
            handler = source.video.tags.get("handler_name")
            if handler and container in _MOV_LIKE:
                args += ["-metadata:s:v:0", f"handler_name={handler}"]
        else:
            args += ["-map_metadata", "-1", "-map_chapters", "-1"]
        if container in _MOV_LIKE:
            flags: List[str] = []
            if encoding.faststart:
                flags.append("faststart")
            if copy_metadata and optional_metadata:
                timecode = source.video.tags.get("timecode") or source.tags.get("timecode")
                if timecode:
                    args += ["-timecode", str(timecode)]        # recreates the QuickTime time-code track
                if any(key.startswith("com.") for key in source.tags):
                    flags.append("use_metadata_tags")           # keeps com.apple.quicktime.* keys
                    args += ["-metadata", "major_brand=", "-metadata", "minor_version=", "-metadata", "compatible_brands="]
            if flags:
                args += ["-movflags", "+" + "+".join(flags)]
            if den_scale is not None:
                # Exact video ticks; start offsets (stored in movie time units) to microseconds.
                args += ["-video_track_timescale", str(source.video.time_base.denominator), "-movie_timescale", str(den_scale)]
        elif audio_plan and source.container in _MOV_LIKE and _hidden_audio_lead(tools, source) > 0.05:
            args += ["-copypriorss", "0"]
            warnings.append(
                "The source hides lead-in audio with an edit list, which this container cannot express: "
                "the hidden audio packets were dropped so the sound stays in sync."
            )
        if audio_end is not None:
            limit = Fraction(audio_end) + shift
            if limit > 0:
                args += ["-t", _seconds(limit)]
        args += ["-f", MUXER_NAMES[container], str(output)]
        return args

    def run(optional_metadata: bool, audio_plan: List[Dict[str, Any]]) -> List[str]:
        args = build(optional_metadata, audio_plan)
        run_ffmpeg(tools, args, cancel=cancel, what="Adding audio and metadata" if audio_plan else "Writing the final file")
        return [tools.ffmpeg, "-hide_banner", "-nostdin", "-y", *args]

    output.parent.mkdir(parents=True, exist_ok=True)
    safe_unlink(output)
    command: List[str] = []
    try:
        try:
            command = run(True, plan)
        except FfmpegFailedError as first:
            # Optional metadata (time code, chapters, vendor keys) must never cost the job.
            safe_unlink(output)
            log.info("Mux with full metadata failed (%s); retrying with essential metadata only.", first.message)
            try:
                command = run(False, plan)
                warnings.append("Some metadata of the source (chapters, time code or vendor tags) could not be "
                                "carried over to the output.")
            except FfmpegFailedError as second:
                safe_unlink(output)
                copied = [e for e in plan if e["action"] == "copy"]
                if mode != "auto" or not copied:
                    raise _audio_error(second, plan) from second
                # The container refused a stream it normally accepts: re-encode the audio.
                fallback: List[Dict[str, Any]] = []
                for entry in plan:
                    if entry["action"] == "copy":
                        encoder, args = _fallback_encoder(container, streams[entry["index"]], encoding.audio_bitrate)
                        entry = {**entry, "action": "transcode", "codec": encoder, "args": args}
                    fallback.append(entry)
                try:
                    command = run(False, fallback)
                except FfmpegFailedError as third:
                    safe_unlink(output)
                    raise _audio_error(third, fallback) from third
                plan = fallback
                warnings.append(
                    f"The audio could not be stream-copied into the .{container} file and was re-encoded "
                    f"({second.message})."
                )
    except (JobCancelledError, VmtError):
        safe_unlink(output)
        raise
    if not output.is_file() or output.stat().st_size == 0:
        safe_unlink(output)
        raise FfmpegFailedError("FFmpeg reported success but the final file is missing or empty.",
                                details={"command": command, "stderr_tail": ""})
    if not plan:
        mode_out = "none"
    else:
        mode_out = "copy" if all(e["action"] == "copy" for e in plan) else "transcode"
    return MuxResult(
        path=output, audio_mode=mode_out, audio_streams=len(plan), warnings=list(dict.fromkeys(warnings)),
        command=command, time_shift=shift,
        audio_plan=[{k: e[k] for k in ("index", "source_codec", "action", "codec")} for e in plan],
    )


def _audio_error(error: FfmpegFailedError, plan: List[Dict[str, Any]]) -> VmtError:
    """Present a mux failure as an audio problem when audio streams were involved."""

    if not plan:
        return error
    codecs = ", ".join(sorted({str(e["source_codec"]) for e in plan}))
    return AudioMuxError(
        f"The audio ({codecs}) could not be added to the output: {error.message}",
        hint="Try an .mkv output, a different audio codec, or --no-preserve-audio.",
        details=dict(error.details),
    )


# --------------------------------------------------------------------------- audio export


def extract_audio(
    tools: FfmpegTools,
    source: VideoMetadata,
    out_stem: Path,
    *,
    audio_end: Optional[Fraction] = None,
    cancel: Optional[CancelToken] = None,
) -> List[Path]:
    """Write every audio stream of the source to its own file without re-encoding.

    The first stream becomes ``<out_stem>.<ext>``, further ones ``<out_stem>_2.<ext>`` ...; the
    extension follows the codec (``.m4a`` for AAC / ALAC, ``.wav`` for PCM, ``.opus``, ``.flac`` ...)
    with Matroska audio (``.mka``) as the fallback that holds anything.
    """

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for number, stream in enumerate(source.audio):
        suffix = "" if number == 0 else f"_{number + 1}"
        candidates = [_AUDIO_EXTENSIONS.get(stream.codec, ".mka"), ".mka"]
        last_error: Optional[VmtError] = None
        for extension in dict.fromkeys(candidates):
            target = out_stem.with_name(f"{out_stem.name}{suffix}{extension}")
            partial = target.with_name(f"{target.stem}.partial{target.suffix}")
            args = ["-v", "error", "-i", source.path, "-map", f"0:a:{number}", "-vn", "-sn", "-dn", "-c", "copy"]
            if audio_end is not None:
                start = source.start_time or Fraction(0)
                limit = Fraction(audio_end) - (start if start > 0 else Fraction(0))
                if limit > 0:
                    args += ["-t", _seconds(limit)]
            args.append(str(partial))
            try:
                run_ffmpeg(tools, args, cancel=cancel, what="Extracting the audio")
                atomic_replace(partial, target)
                written.append(target)
                last_error = None
                break
            except JobCancelledError:
                safe_unlink(partial)
                raise
            except VmtError as exc:
                safe_unlink(partial)
                last_error = exc
        if last_error is not None:
            raise AudioMuxError(
                f"Audio stream {number + 1} ({stream.codec}) could not be extracted: {last_error.message}",
                details=dict(last_error.details),
            ) from last_error
    return written


# --------------------------------------------------------------------------- browser preview


def make_proxy(
    tools: FfmpegTools,
    source: Path,
    output: Path,
    *,
    max_side: int = 960,
    cancel: Optional[CancelToken] = None,
) -> Path:
    """Create a small browser-playable preview of any supported input (H.264 + AAC in MP4).

    The picture is auto-rotated to display orientation, scaled so the long side is at most
    *max_side* (even dimensions) and converted to 8-bit BT.709.  HDR sources are converted without
    tone mapping, so they look flat: this file is for the user interface only and is never used as a
    processing source.
    """

    source, output = Path(source), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.stem}.partial{output.suffix or '.mp4'}")
    side = max(64, int(max_side))
    scale = (
        f"scale=w='if(gte(iw,ih),2*trunc(min(iw,{side})/2),-2)':"
        f"h='if(gte(iw,ih),-2,2*trunc(min(ih,{side})/2))':flags=bilinear,format=yuv420p"
    )
    args = [
        "-v", "error", "-i", str(source),
        "-map", "0:V:0", "-map", "0:a:0?", "-sn", "-dn",
        "-vf", scale,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p", "-g", "48",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
        "-c:a", "aac", "-b:a", "128k", "-ac", "2",
        "-movflags", "+faststart", "-f", "mp4", str(partial),
    ]
    try:
        run_ffmpeg(tools, args, cancel=cancel, what="Creating the browser preview")
        atomic_replace(partial, output)
    except BaseException:
        safe_unlink(partial)
        raise
    return output
