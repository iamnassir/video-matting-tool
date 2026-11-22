# Video Matting Tool

[![tests](https://github.com/iamnassir/video-matting-tool/actions/workflows/tests.yml/badge.svg)](https://github.com/iamnassir/video-matting-tool/actions/workflows/tests.yml)

Removes the background of a video with AI matting models and composites the subject onto pure
chroma green (`#00FF00`), another solid colour, or transparency, while preserving what an editor
needs to be untouched: resolution, orientation, every frame and its timestamp, duration, colour
tags, bit depth and the original audio.

It has a command line, a local web interface, and a model layer with four interchangeable models.

```
python main.py process --input input.mp4 --output output_green.mp4 --model rvm --background green --device cuda --quality high
python main.py serve        # web interface at http://127.0.0.1:8765 (Windows: double-click start_web.bat)
```

## Install on Windows (three steps)

The tool runs on your own computer; videos never leave it. It needs about 6 GB of disk space with an
NVIDIA graphics card (about 1.5 GB without one, where it works but slowly).

1. Once, in a terminal (Start menu, "Terminal"): `winget install Python.Python.3.10` and
   `winget install Gyan.FFmpeg`. Close the terminal afterwards.
2. Download the project from https://github.com/iamnassir/video-matting-tool (green **Code** button,
   **Download ZIP**, then unpack it) and double-click `install.bat`. It creates its own Python
   environment inside the project folder, installs PyTorch (the NVIDIA build when an NVIDIA card is
   present, otherwise the processor build) and, after you accepted its licence, downloads the
   default model (120 MB). With an NVIDIA card the download is about 2.5 GB, so give it time.
3. Double-click `start_web.bat`. The tool opens in your browser at http://127.0.0.1:8765. Keep the
   black window open while you work; closing it stops the tool.

Linux, macOS, optional models and manual steps: [docs/INSTALL.md](docs/INSTALL.md).

## What it does

* **Models** (adapters behind one interface, see [docs/MODELS.md](docs/MODELS.md)):
  Robust Video Matting, SAM 2.1, a SAM 2 + RVM hybrid, MatAnyone / MatAnyone 2, and plugins.
  Weights come only from the official sources, are pinned by SHA-256 and are downloaded only on
  request. Licences are enforced (`--commercial` refuses non-commercial models).
* **Subject selection**: automatic, clicks (positive / negative), box, mask image, text description,
  on any reference frame, with backward tracking to the start of the clip.
* **Input**: MP4, MOV, MKV, AVI, WebM and anything else FFmpeg reads; constant and variable frame
  rate, rotation and mirror metadata, 8-bit and 10-bit, limited and full range, HDR tags.
* **Timestamp-exact output**: one output frame per source frame with the identical timestamp;
  frame rate is never assumed constant, nothing is resampled, interpolated or scaled unless asked.
* **Audio**: stream copy (bit-identical) whenever the container allows; sync verified per packet.
* **Matte refinement**: feather, erode / dilate, smoothing, motion-aware temporal smoothing, hole
  filling, speckle removal, mask-to-alpha conversion, edge colour decontamination (no halos, no
  spill). Off by default beyond a conservative preset.
* **Outputs**: H.264, H.265 (software and NVENC), ProRes 422 HQ, ProRes 4444 (with alpha), FFV1,
  VP9; alpha matte video, RGBA PNG sequence, original audio, side-by-side preview.
* **Hardware**: CUDA, Apple MPS, CPU. Streaming with bounded memory; automatic batch reduction and
  CPU fallback on out-of-memory.
* **Reliability**: resumable runs (cached mattes), cancel at any time, no partial file ever left
  under the final name, automatic verification and a report after every run.

## Quick start (manual)

```
python -m venv .venv && .venv\Scripts\activate
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -c constraints.txt
python main.py models download rvm --all --accept-license     # GPL-3.0, see docs/MODELS.md
python main.py process -i input.mp4 -o output_green.mp4
```

Full instructions, CPU and macOS variants, optional models: [docs/INSTALL.md](docs/INSTALL.md).
Recipes for the CLI, the web interface and the API: [docs/USAGE.md](docs/USAGE.md).
Problems: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Verified behaviour

Measured on the development machine (Windows 11, RTX 3060 12 GB shared with other applications,
PyTorch 2.5.1, FFmpeg 8.1) with real footage:

| Run | Result |
|---|---|
| RVM, 954 frames 1024x576 | 34 s; 117 fps inference, 33 fps end to end; all timestamps identical, audio bit-identical, uniform green, foreground PSNR 46 dB |
| RVM, 1080p | 80-92 fps inference, about 20 fps end to end (batch size 1, GPU shared) |
| SAM 2.1 small, selected child through an occlusion | about 8 fps, 0.7 GB VRAM |
| SAM 2 + RVM, one of two people who touch | 6-7.5 fps, only the selected person kept |
| MatAnyone 2, 720p, official sample | 10-12 fps inference; only the selected person, soft hair |
| Awkward inputs | rotated and mirrored phone clips, 4K variable frame rate, 10-bit HLG Dolby Vision MOV, 1080p60, vertical video, clips with edit lists: all hard checks pass |

Every run prints this kind of table and writes it to `<output>.report.json`:

```
 Frame count             79                      79                      yes
 Duration                2.633 s                 2.633 s                 yes
 [PASS] Frame timestamps: All 79 frame times match the source exactly.
 [PASS] Audio streams: 1 stream(s) stream-copied bit-identically (aac 44100 Hz 2 ch).
 [PASS] Background colour: ... uniform #00FF00 ...
 [PASS] Foreground preservation: The subject matches the source: luma PSNR 49.4 dB ...
```

## Known limitations

* **Pixel preservation and lossy codecs**: opaque subject pixels are passed through bit-exactly in
  the source pixel format up to the encoder. With H.264 / H.265 the encoder then compresses them
  (the report states the PSNR); FFV1 output is lossless. In 8-bit limited-range YUV the closest
  colour to `#00FF00` can decode as `#00FF01`.
* **Model limits** are listed per model in [docs/MODELS.md](docs/MODELS.md): RVM mattes all people
  and only people; SAM 2 produces masks, not hair-level alpha; MatAnyone is non-commercial.
* **SAM2Matting** was evaluated and not integrated (licence and dependency conflicts); `sam2_rvm`
  and `matanyone` cover the use case.
* **Text prompts** return a best guess even for phrases that match nothing; preview the selection.
* **Speed**: end-to-end speed at 1080p is limited by refinement and compositing (about 26 ms per
  frame, of which 10 ms is the edge colour decontamination), not by RVM itself.
* **Truncated files** are processed up to the cut with a clear warning; an MP4 that lost its index
  cannot be read.
* **Not tested here**: Apple MPS, Linux, the Dockerfile (no Docker on the development machine).
* The web server has no authentication and binds to localhost by default.

## Project layout

```
main.py                 entry point (CLI)
app/core/               configuration, errors, events, hardware, pixel formats, report
app/models/             adapter contract, registry, weights manager; rvm/ sam2/ hybrid/ matanyone/
app/tracking/           subject selection, helper models, component tracker, tracking monitor
app/processing/         probe, reader, matte cache, batching, pipeline
app/refinement/         matte refinement, foreground decontamination
app/compositing/        colour conversion and compositing in the native pixel format
app/encoding/           FFmpeg profiles, timestamp-exact pipe encoder, muxer, PNG writer
app/quality/            output verification
app/jobs/, app/api/     job queue and HTTP API
app/ui/static/          web interface (no build step)
app/cli/                command line
configs/default.yaml    settings
scripts/                optional model installer, test footage downloader
tests/                  automated tests
docs/                   documentation
```

## Tests

```
pip install -r requirements-dev.txt
python -m pytest
```

Tests that need model weights, a GPU, real footage or a browser skip themselves when those are not
available. On the development machine, with everything installed, the whole suite
(about 700 tests, 5 minutes) passes.

## Licence notes

The models are third-party works under their own licences (GPL-3.0, Apache-2.0, NTU S-Lab 1.0,
BSD-3-Clause); see [docs/MODELS.md](docs/MODELS.md) before commercial deployment.
