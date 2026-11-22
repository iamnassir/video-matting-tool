# Troubleshooting

Every error has a code, a message and usually a hint. `-v` adds stack traces and the FFmpeg command
lines. The report (`<output>.report.json`) records warnings and the result of every check.

| Symptom / code | Cause | What to do |
|---|---|---|
| `ffmpeg_not_found` | `ffmpeg` / `ffprobe` not on `PATH` | Install FFmpeg 7+, or set `VMT_FFMPEG` and `VMT_FFPROBE`. |
| `weights_missing` | Weights not downloaded | `python main.py models download <id>` (or `--download-models` for this run). |
| `model_not_installed` | Optional package missing | `python scripts/install_models.py sam2` / `matanyone --accept-license` / `text`. |
| `weights_checksum_mismatch` | Corrupt or changed download | `python main.py models remove <id>` and download again. |
| `license_restriction` | `--commercial` with a non-commercial model | Use `rvm`, `sam2` or `sam2_rvm`. |
| `subject_not_found` | No subject on the reference frame | Choose another `--subject-frame`, or select the subject with a box / clicks. |
| Exit code 2, "Subject tracking" check failed | The model found no subject in any frame (for example a shot without a person with `rvm`) | Use `sam2` with a selection for non-human subjects. |
| `prompt_not_supported` | The model needs a mask and no helper model is installed | Install SAM 2 and the detector, or pass `--subject mask:<file>`. |
| "No working GPU ... using the CPU" | CUDA failed to initialise or has no free memory | `python main.py system`; close other GPU applications; check the driver. |
| Very slow, GPU nearly idle | The job fell back to the CPU, or another program holds the VRAM | Look at the `Device:` line of the report. |
| "batch size reduced" warnings | The GPU ran out of memory and the tool recovered | Nothing; for speed, free VRAM or use `--quality fast`. |
| `CUBLAS_STATUS_EXECUTION_FAILED` in the log | cuBLAS could not allocate on a nearly full GPU | Treated as out-of-memory and retried automatically. |
| `insufficient_disk_space` | Not enough room for output plus intermediates | Free space or choose another output folder / `VMT_TEMP_DIR`. |
| `output_exists` | The output file is there | `-y` / `--overwrite`. |
| `encoder_unavailable` | This FFmpeg build lacks the encoder (for example NVENC) | `python main.py system` lists what is available; pick another `--output-codec`. |
| Audio was re-encoded (warning) | The container cannot hold the source codec (for example PCM in MP4) | Use `.mov` / `.mkv`, or accept the stated codec. |
| Background reads `#00FF01` | 8-bit limited-range YUV cannot represent `#00FF00` exactly | Use FFV1, ProRes 4444 or PNG for an exact value; keyers do not care about one code value. |
| Halo or old background colour at edges | Semi-transparent pixels carry the old background | `--decontaminate 0.8`, or `--mask-refinement strong`. |
| Edges look too soft / too hard | Refinement preset | `--mask-refinement light` or `off`; `--edge-feather`. |
| Flicker | Model instability on hard footage | `--temporal-smoothing 0.5`; try `matanyone` or `sam2_rvm`. |
| Another person appears in the result | `rvm` mattes all people; they touched | Use `sam2_rvm` or `matanyone` with a selection. |
| A click selected only a garment | Single clicks are ambiguous | Install the detector (then the whole person is selected), add a second click, or draw a box. |
| Hand-held object missing (`rvm`) | RVM is trained on people | `sam2_rvm` (fills from the SAM 2 mask) or `sam2`. |
| The browser says "connection refused" | The web server is not running | `python main.py serve`, or double-click `start_web.bat`; keep its window open. |
| "is a colour picture, not a mask" | The mask image is an ordinary photograph | Supply a black-and-white mask, a cut-out with transparency, or a cut-out on one flat background colour (see "Mask images" in USAGE.md); or select with Click / Box. |
| "The mask picture does not line up with frame N" | The cut-out shows a frame that was not found in the video (or another video) | Make the picture from a frame of this video; the tool then finds the frame itself. |
| Car windows (or other openings) are not cut out | `sam2` follows whole objects and closes openings; `rvm` only mattes people | Use `matanyone` with a mask image in which the openings are cut out. |
| Something passing behind the subject is kept for a few frames | It touches the subject's outline and looks similar | Trim those frames, or key the area out afterwards; a second reference frame is not supported yet. |
| Rotated phone video looks rotated in the result | It should not: rotation metadata is preserved | Check the "Orientation" line of the report and please report the file. |
| "probably truncated or damaged" warning | The file ends early (interrupted download or recording) | The frames that can be read are processed and the output verifies against them; get a complete copy for the rest. An MP4 without its index (`moov`) cannot be read at all. |
| `unsupported_format`: "is a text file, not a video" | A renamed non-video file | Choose a real video file. |
| A job in the web interface seems frozen | | Look at `data/logs/stall-traces.log`: when a running job reports no progress for 5 minutes (`stall_dump_seconds`), the stacks of all threads are written there. Restart the server; the job shows as *interrupted* and can be resumed. |
| "The source has a constant frame rate but the output a variable one" for an MKV or WebM output | A development build of FFmpeg newer than 9.1 is installed; it reports Matroska time differently | The file is correct. Use a released FFmpeg, or an MP4 / MOV output. |
| Browser cannot play the source preview | The proxy is still being created or failed | The frame view and processing still work. |

## Interrupted runs

Run the same command again: cached mattes are reused. If the cache belongs to different settings it
is discarded automatically. `--no-resume` forces a fresh start.

## Reporting a problem

Include the output of `python main.py system`, `python main.py probe <file>`, the command line, and
the `.report.json` if one was written.
