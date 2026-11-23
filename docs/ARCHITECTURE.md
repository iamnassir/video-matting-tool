# Architecture

This document is the engineering contract of the Video Matting Tool: how data flows, which module
owns what, the exact public interfaces between modules, and the invariants every module must keep.

## 1. Goals and invariants

The tool removes the background of a video and composites the subject onto a uniform background
(pure chroma green `#00FF00` by default) while preserving the source's technical properties.

Invariants (verified automatically after every run, see section 9):

| # | Invariant |
|---|-----------|
| I1 | Output coded resolution, rotation metadata and aspect ratio equal the source (unless a resize is explicitly requested). |
| I2 | Output has exactly one frame per source frame, with the **same presentation timestamps**. Timestamps are integers in the source stream time base from decode to encode; a constant frame rate is never assumed. No interpolation, duplication or dropping. |
| I3 | Audio is stream-copied when the container allows it and keeps its offset to the video. |
| I4 | **Pixel preservation**: where the matte is fully opaque, output samples are the source samples, copied in the source's native pixel format without any colour-space round trip (bit-exact before encoding). The model only ever produces a matte; no pixel is generated, upscaled, denoised or "enhanced". |
| I5 | Where the matte is zero the output is one constant colour: every sample of a plane has the same code value. No gradients, noise or texture. |
| I6 | A failed run never leaves a file under the final output name. Errors carry a code, a message and a hint. |
| I7 | The whole video is never held in memory. Frames stream through bounded queues. |

## 2. Data flow

```
 input file
   │  probe.py            ffprobe + PyAV packet scan  -> VideoMetadata + FrameTimeline (exact PTS list)
   ▼
 reader.py                PyAV decode -> DecodedFrame(index, pts, native planes)      [decoder thread]
   │  bounded queue
   ▼
 pipeline.py (render loop, one batch at a time, on the inference device)
   ├─ ColorConverter.planes_to_rgb      native planes -> R'G'B' float (model input only)
   ├─ rotate to display orientation     lossless 90-degree turns
   ├─ MattingModel.process              -> raw alpha (or MatteStore.read when resuming)
   ├─ ComponentSubjectTracker.apply     subject isolation for models without prompt support
   ├─ MatteStore.write                  raw-matte cache (resume / re-render)
   ├─ MatteRefiner.refine               mask->alpha, cleanup, morphology, temporal filter, levels
   ├─ TrackingMonitor.update            coverage / flicker / lost-subject statistics
   ├─ rotate back to coded orientation
   └─ Compositor                        native-domain composite per output target
   │  bounded queue per sink
   ▼
 NutPipeEncoder (per target)            PyAV muxes raw planes + exact PTS into NUT -> ffmpeg CLI stdin
   ▼                                    ffmpeg encodes the video-only intermediate
 muxer.py                               ffmpeg stream-copies video + original audio + metadata
   ▼
 verify.py                              quality verification -> ProcessingReport
   ▼
 final output (atomically renamed into place)
```

Stages reported to the user (`app.core.events.Stage`): `probe → subject → model_load → [backtrack] →
render → mux → export → verify → done`.

## 3. Conventions

* **Coded vs display orientation.** Decoded frames are in coded orientation (PyAV never
  auto-rotates). `VideoStreamInfo.rotation` is the clockwise angle (0/90/180/270) a player applies.
  Models, subject prompts, the matte store and the refiner work in **display orientation**; the
  pipeline converts with `torch.rot90` and rotates the matte back before compositing. The output
  keeps the source's coded size and rotation metadata.
* **Working format.** Each source pixel format maps to a planar little-endian working format with the
  same subsampling and bit depth (`app.core.pixfmt.PixFmt`), e.g. `yuv420p`, `yuv422p10le`, `gbrp`.
  `yuvj*` formats are relabelled (same planes, full range), `nv12`-style and packed formats are
  repacked without changing sample values.
* **Colour.** `resolve_color()` picks matrix and range from the stream tags, falling back to the usual
  player heuristics (BT.709 for HD, BT.601 for SD, limited range for YUV). The output carries the
  source tags verbatim. HDR transfer functions (PQ / HLG) are passed through untouched: no tone
  mapping.
* **Tensors.** Alpha is `float32 [B,1,H,W]` in 0..1. RGB is `float32 [B,3,H,W]` in 0..1 (non-linear
  R'G'B'). Planes on the device are `float32 [B,1,h,w]` holding integer code values (exact up to
  2^24). Compositing happens in non-linear R'G'B' (the video-industry convention).
* **Pixel quantities** in refinement settings are specified for a 1080-line frame and scaled by
  `min(width, height) / 1080` (`RefinementParams.px`).
* **Errors.** Modules raise subclasses of `app.core.errors.VmtError` at their boundaries.
* **Threads.** PyAV decode, FFmpeg encode and CUDA all release the GIL; the pipeline uses one decoder
  thread, the main inference thread and one writer thread per sink, connected by bounded queues.
* **No hidden downloads, no hidden caches.** Weights are downloaded only on explicit request and
  stored under `models_dir`; third-party caches are redirected under `cache_dir`.

## 4. Package map

| Package | Responsibility |
|---------|----------------|
| `app/core` | Contracts: config models, data types, pixel formats, errors, progress events, hardware detection, report schema. No dependency on other `app` packages. |
| `app/processing` | Video input (`probe`, `planes`, `reader`), the streaming `pipeline`, the raw-matte cache (`matte_store`). |
| `app/models` | `base` (adapter interface), `registry` (plugins), `weights` (verified downloads), adapters in `rvm/`, `sam2/`, `matanyone/`, `hybrid/`. |
| `app/tracking` | Subject selection (`subject`, `auto_detect`, `text_grounding`), subject isolation (`component_tracker`), tracking health (`monitor`), rigid motion and the one outline of a clip (`rigid`). |
| `app/refinement` | Matte refinement operators (`ops`), temporal filtering (`temporal`), the configurable `refiner`, foreground colour estimation (`foreground`). |
| `app/compositing` | Colour conversion (`colors`) and native-domain compositing (`compositor`). |
| `app/encoding` | FFmpeg discovery (`ffmpeg`), codec profiles (`profiles`), the timestamp-exact encoder (`nut_pipe`), audio/metadata mux (`muxer`), extra deliverables (`sinks`). |
| `app/quality` | Output verification (`verify`) and report assembly (`report_builder`). |
| `app/jobs` | Job queue, persistence, cancellation, live previews. |
| `app/api` | FastAPI application and routes. |
| `app/ui` | Static single-page web interface served by the API. |
| `app/cli` | Typer command line interface. |

Dependency direction: `core` ← everything; `processing.pipeline` depends on all engine packages;
`jobs` depends on `processing`; `api` and `cli` depend on `jobs` / `processing`. Engine packages never
import `api`, `cli` or `jobs`.

## 5. Model adapters

See `app/models/base.py` for the binding contract. Summary:

* An adapter declares a `ModelInfo` (licence, capabilities, variants with official weight URLs and
  pinned SHA-256, limitations) and implements `load / start_sequence / process / end_sequence /
  unload`, optionally `reduce_memory`.
* Input: display-oriented `float32 [B,3,H,W]` R'G'B' at full resolution on the adapter's device.
  Output: `MatteOutput.alpha` at the same size, plus optional per-frame `presence`.
* A sequence starts at the subject's reference frame. Backward tracking is a second sequence with
  `reverse=True`.
* A model with `capabilities.two_pass` decides with the whole clip in view: the pipeline runs it
  over every frame into the matte cache first (sequences announced with `will_revise`), calls
  `revise(indices, load, save)` once, and only then renders, entirely from the cache.
* `process` is retry-safe (state unchanged if it raises).
* Capabilities drive the pipeline: `alpha_type` (`alpha` = true matte, `mask` = needs mask-to-alpha
  refinement), `native_prompts`, `needs_initial_mask`, `tracks_single_subject`, `max_batch_size`.

Plugins register with `app.models.registry.register_model` (config `plugins:` list or the
`video_matting_tool.models` entry-point group).

## 6. Subject selection

`SubjectPrompt` (config) → `SubjectSelector.resolve()` → `ResolvedSubject` (mask and/or box and points
on the reference frame, display orientation). Resolution chain:

| Prompt | How it is resolved |
|--------|--------------------|
| `auto` | Primary-subject detector on the reference frame (and following frames until a subject appears). |
| `box` / `points` | Passed through to models with native prompt support; otherwise converted to a mask with the SAM 2 image predictor (fallbacks: detector instance matching the prompt, GrabCut for boxes). |
| `text` | Open-vocabulary detector → box → as above. Optional dependency. |
| `mask` | User-supplied image, resized to the display frame. |

Models without prompt support (RVM) are restricted to the selected subject by
`ComponentSubjectTracker` (connected-component tracking of the matte), which also keeps "negative"
tracks of other people so the selection does not jump to them.

## 7. Matte store, resume and re-render

`MatteStore` keeps the raw matte (after subject isolation, before refinement, display orientation,
8-bit by default) in chunk files under the work directory together with a fingerprint of the input
file and of every setting that influences the raw matte. Consequences:

* **Resume**: an interrupted run restarts the encoder from frame 0 but takes mattes from the store for
  the frames already inferred, re-seeds the model from the last stored matte and continues.
* **Re-render**: changing only refinement, background or encoding settings reuses every stored matte;
  the model is not run again.
* **Backward tracking**: mattes for frames before the reference frame are produced by a reverse pass
  and read back from the store during the forward render.
* **Two-pass models**: every matte goes through the store, the model revises them there, and a
  marker records that the revision happened. Without the marker (an interrupted run) the store is
  emptied and the run starts again: raw mattes cannot be revised later, because what the model
  noted about the clip is gone with the process.

The live path quantises the raw matte exactly like the store, so resumed and uninterrupted runs
render identically from the same mattes.

## 8. Compositing

For a target background colour `B` (R'G'B'), source colour `I` and matte `α`:

1. Foreground colour `F` = `I` where `α` is 1; in the semi-transparent band `F` is estimated with
   Blur-Fusion foreground estimation (scaled by `decontaminate`) so the old background does not
   leave a halo.
2. `C = α·F + (1−α)·B` in float R'G'B', converted to the output planes with the resolved matrix/range.
3. **Opaque core restore**: wherever every contributing matte sample is exactly 1, the output sample
   is copied from the source plane (bit-exact, invariant I4).
4. **Background constant**: wherever every contributing matte sample is exactly 0, the output sample
   is the pre-computed integer code value of `B` for that plane (invariant I5).

Transparent targets (ProRes 4444, VP9/FFV1 with alpha, PNG) carry `F` in the colour planes and `α`
in the alpha plane (straight alpha by default, premultiplied on request).

## 9. Verification and report

`app/quality/verify.py` checks the finished file against the source: decodability, resolution,
rotation, pixel format, colour tags, frame count, per-frame timestamps, frame rate / VFR, duration,
audio streams (stream-copy bit-identity, A/V offset), background colour and uniformity on sampled
frames, foreground preservation (PSNR in the opaque core), black frames, matte flicker and tracking
events. Results are `QualityCheck` entries in the `ProcessingReport`, which also records input and
output metadata, the model and its licence, hardware, timings, average inference FPS, warnings and
all output paths.

Critical failures (undecodable / truncated output) abort the run and the file is not delivered. Other
failed checks mark the report `completed_with_warnings` (or raise with `strict_verification`).

## PyAV log forwarding is kept off

PyAV can forward FFmpeg's log messages to Python logging. Its callback takes the interpreter lock and
is called from FFmpeg's decoder worker threads; when the thread holding the lock is waiting for those
workers (closing or freeing a multi-threaded decoder) the process deadlocks permanently. PyAV leaves
the callback off by default, but `import torchvision` switches it on, and every model except RVM
imports torchvision. `app/utils/avlog.py` switches it off when the package is imported and before
every container is opened, and makes later `av.logging.set_level(...)` calls ineffective.
`tests/test_avlog.py` covers this. Do not enable PyAV log forwarding in plugins.

## Stall diagnosis (web server)

When a running job reports no progress for `stall_dump_seconds` (default 300), the stacks of all
threads are appended to `<data_dir>/logs/stall-traces.log` by `faulthandler`'s watchdog, which works
even when the interpreter itself is blocked.
