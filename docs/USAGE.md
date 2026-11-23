# Usage

## Command line

```
python main.py process --input input.mp4 --output output_green.mp4 --model rvm --background green --device cuda --quality high
```

### Common recipes

```
# Fast draft of the first 150 frames
python main.py process -i in.mp4 -o draft.mp4 --model rvm --quality fast --max-frames 150

# One of several people: click on them (x,y in pixels of the displayed frame); "-" marks an exclusion click
python main.py process -i in.mp4 -o out.mp4 --model sam2_rvm --subject "points:400,520;840,520,-"

# A box, on frame 120 instead of the first frame (frames before it are tracked backwards)
python main.py process -i in.mp4 -o out.mp4 --model sam2 --subject "box:247,186,559,720" --subject-frame 120

# Describe the subject
python main.py process -i in.mp4 -o out.mp4 --model sam2_rvm --subject "text:the woman"

# Your own mask image (white = subject) with the highest-quality model (non-commercial licence)
python main.py process -i in.mp4 -o out.mp4 --model matanyone --subject "mask:first_frame_mask.png"

# A still of the result you want (the subject cut out on flat green) also works as the mask;
# the frame it was made from is found automatically
python main.py process -i in.mp4 -o out.mp4 --model matanyone --subject "mask:wanted_look.jpg"

# A car with its driver, everything else green - also what is seen through the windows
# (the mask is a still with the car cut out and the windows cut open)
python main.py process -i car.mp4 -o car_green.mp4 --model sam2_matanyone --subject "mask:car_cutout.jpg"

# Other backgrounds
python main.py process -i in.mp4 -o out_blue.mp4 --background blue
python main.py process -i in.mp4 -o out.mp4 --background custom --background-color "#1E90FF"
python main.py process -i in.mp4 -o out.mov --background transparent          # ProRes 4444 with alpha

# Extra deliverables next to the main output
python main.py process -i in.mp4 -o out.mp4 --export-alpha --export-prores4444 --export-png-sequence --export-audio

# Mastering codecs
python main.py process -i in.mov -o out.mov --output-codec prores_422_hq
python main.py process -i in.mkv -o out.mkv --output-codec ffv1                # lossless
python main.py process -i in.mp4 -o out.mp4 --output-codec h265 --crf 16 --preset slow

# Batch: a folder or a glob; one failed file does not stop the others
python main.py process -i footage/ --recursive --output-directory out/ --model rvm
python main.py process -i "shots/*.mov" -d out/ --model rvm --stop-on-error
```

### Mask images

`--subject mask:<file>` (web interface: *Mask image*) accepts three kinds of picture:

| Picture | How it is read |
|---|---|
| Black and white (any format, also a grey JPEG) | white = subject, black = background |
| Cut-out with transparency (PNG with alpha) | the alpha channel is the mask |
| Colour cut-out on one flat background colour, for example the still of a green-screen result | everything that is not the background colour is the subject |

Any other colour picture is refused ("is a colour picture, not a mask") instead of being read as
"bright = subject". A mask that covers the whole frame or nothing is refused as well.

The mask has to belong to the reference frame (`--subject-frame`, default 0). A black-and-white mask
is taken as given. A colour cut-out carries the video's own pixels, so the tool checks it: when it
shows another frame than the chosen one, the video is searched (for at most 45 s) and the frame it
shows becomes the reference frame; the report says so. If no frame matches, the report warns that
the picture does not line up.

Paths can be pasted the way Windows gives them (`"C:\Users\me\Downloads\mask.png"`, quotes included).

Which model for a mask: for an object with see-through parts (car windows, the gaps of a chair)
choose `sam2_matanyone`: it holds the outline as one steady shape and keeps the openings open. For
a person, `matanyone` (it also keeps the gap between arm and body). `sam2` alone follows the
object as a whole and fills openings in again after the reference frame. A dark line along the
border of a cut-out picture, as image tools sometimes leave it, is ignored.

### Matte refinement

`--mask-refinement off|light|balanced|strong` selects a preset; single values override it:

| Option | Effect |
|---|---|
| `--edge-feather PX` | feather the edge (pixels at 1080p, scaled to the frame size) |
| `--erode PX` / `--dilate PX` | shrink / grow the matte |
| `--smooth SIGMA` | Gaussian smoothing of the matte |
| `--temporal-smoothing 0..0.95` | flicker reduction; motion-aware, so moving edges are not smeared |
| `--motion-compensation` | optical-flow compensated temporal smoothing (slower) |
| `--black-point` / `--white-point` | clamp nearly transparent / nearly opaque alpha |
| `--decontaminate 0..1` | remove the old background colour from semi-transparent edge pixels (prevents halos and spill) |

`off` passes the model's matte through untouched. The defaults are conservative: nothing is smoothed
unless a preset or option asks for it.

### What is preserved

Resolution, display orientation (rotation metadata, including mirrored phone footage), every frame
with its original timestamp (variable frame rate stays variable), duration, colour tags, bit depth,
and the audio streams as bit-identical copies whenever the container allows it. Scaling only happens
with an explicit `--resolution`. Each run ends with an automatic verification and a report
(`<output>.report.json`, printed as a table):

```
[PASS] Frame timestamps: All 954 frame times match the source exactly.
[PASS] Audio streams: 1 stream(s) stream-copied bit-identically (aac 44100 Hz 2 ch).
[PASS] Background colour: ... uniform #00FF00 ...
[PASS] Foreground preservation: The subject matches the source: luma PSNR 46 dB ...
```

`--strict` turns a failed check into an error. Exit codes: 0 success (warnings are listed in the
report), 2 the output was written but a quality check failed, 1 error, 130 cancelled.

A note on "pure green": the composite is built with one constant code value per plane, so lossless
outputs (FFV1, ProRes 4444, PNG) contain exactly `#00FF00`. In 8-bit limited-range YUV (H.264/H.265)
the nearest representable colour decodes to `#00FF01` in some decoders; the report states what was
measured.

### Resume and re-render

Mattes are cached while a job runs. If a run is interrupted (Ctrl+C, crash, power loss), running the
same command again continues where it stopped (`--no-resume` starts over). `--keep-cache` keeps the
mattes after success so the same video can be rendered again with another background, codec or
refinement in a fraction of the time. A failed or cancelled run never leaves a file under the final
output name.

### Other commands

```
python main.py probe input.mp4          # everything the tool reads from the file
python main.py system                   # hardware, PyTorch, FFmpeg encoders
python main.py presets                  # quality presets, refinement presets, codecs
python main.py models list
python main.py process --help
```

`--json-progress` prints progress as JSON lines for automation; `-v` adds FFmpeg command lines.

## Web interface

```
python main.py serve                    # http://127.0.0.1:8765  (--port, --host, --no-open)
```

On Windows, double-clicking `start_web.bat` does the same and opens the browser. The page only
answers while the server is running: "connection refused" in the browser means it has not been
started (or its window was closed).

1. **Source**: drop one or more videos (or register a local path without copying it). The metadata
   table and a browser preview appear.
2. **Subject**: scrub to a frame, choose Auto, Click (right click = exclude), Box, Text or Mask, and
   press *Preview selection* to see exactly what the engine will track.
3. **Settings**: model, quality, background, refinement, codec, audio, exports. Options that cannot
   work together are disabled with the reason.
4. **Process**: start; the page shows the stage, frame counter, percentage, fps, ETA, GPU and memory
   readings and a live picture (original / subject / matte / composite). Cancel and Resume work at
   any time.
5. **Results**: the player compares original, subject, matte and composite (side by side, wipe,
   grid); downloads; the full report; *Re-render* with other settings reuses the cached mattes.

The server binds to this computer only by default. It has no authentication: do not expose it to a
network you do not trust.

### HTTP API

The interface uses a small JSON API that can be scripted directly; the schema is at
`http://127.0.0.1:8765/api/docs`.

```
POST /api/videos/import      {"path": "D:/clips/a.mp4"}            -> {"id": ...}
POST /api/jobs               {"video_id": "...", "config": {"model": "rvm", "background": "green"}}
GET  /api/jobs/{id}          status, progress, outputs, summary
GET  /api/jobs/{id}/events   server-sent events
POST /api/jobs/{id}/cancel | /resume | /rerender
GET  /api/jobs/{id}/report   the processing report
GET  /api/jobs/{id}/files/{name}
```
