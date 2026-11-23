# Installation

Tested on Windows 11 with Python 3.10, PyTorch 2.5.1 (CUDA 12.1) and FFmpeg 8.1 on an NVIDIA card,
and without a graphics card on clean Windows (Python 3.10 and 3.12, FFmpeg 9.1) and Linux (Python
3.10, FFmpeg 9.0) machines. macOS uses the same steps; the Apple MPS and Docker paths are
implemented but were not tested.

## Windows: the short way

Install Python and FFmpeg once (`winget install Python.Python.3.10`, `winget install Gyan.FFmpeg`),
then double-click `install.bat` in the project folder and, when it has finished, `start_web.bat`.
`install.bat` performs sections 2 to 4 below: it creates `.venv`, installs PyTorch (the CUDA build
when `nvidia-smi` finds an NVIDIA card, otherwise the CPU build; `set VMT_TORCH=cpu` or `cuda`
overrides the choice), installs `requirements.txt` and downloads the RVM weights. Running it again
is safe; finished steps are skipped.

## 1. Prerequisites

* Python 3.10, 3.11 or 3.12
* FFmpeg 7 or newer with `ffmpeg` and `ffprobe` on `PATH` (or set `VMT_FFMPEG` / `VMT_FFPROBE`)
  * Windows: `winget install Gyan.FFmpeg`
  * Linux: `sudo apt install ffmpeg`
  * macOS: `brew install ffmpeg`
* NVIDIA GPU (optional): a current driver. No separate CUDA toolkit is needed.

## 2. Environment and PyTorch

```
cd video_matting_tool
python -m venv .venv
.venv\Scripts\activate            # Linux / macOS: source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the PyTorch build for your hardware **first**:

```
# NVIDIA GPU (CUDA 12.1)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon (MPS)
pip install torch==2.5.1 torchvision==0.20.1
```

Then the application:

```
pip install -r requirements.txt -c constraints.txt
python main.py system             # shows GPU, VRAM, PyTorch, FFmpeg and its encoders
```

`constraints.txt` keeps later installs from replacing NumPy, OpenCV or PyAV;
`scripts/install_models.py` additionally pins the PyTorch build that is installed, whichever it is.

PyTorch 2.5.1 with CUDA 12.1 supports NVIDIA cards up to the RTX 40 series. On a newer card that
this build cannot drive, the tool reports that the GPU failed its self-test and uses the processor.

## 3. Models

Downloads are explicit. A model whose licence has conditions is only downloaded with
`--accept-license`; without it the command prints the conditions and stops. The smallest useful
setup is RVM (GPL-3.0; two variants, 15 MB + 103 MB, used by the fast and the high quality presets):

```
python main.py models download rvm --all --accept-license
```

Optional models (see [MODELS.md](MODELS.md) for what each one is for and its licence):

```
python scripts/install_models.py sam2
python main.py models download sam2
python main.py models download detector --accept-license    # automatic subject detection

python scripts/install_models.py matanyone --accept-license     # non-commercial licence
python main.py models download matanyone --accept-license
# with both sam2 and matanyone installed, the model for objects with openings is ready too:
#   --model sam2_matanyone   (a car with its windows; nothing more to download)

python scripts/install_models.py text                # text prompts
python main.py models download text_grounding
```

`scripts/install_models.py` installs the upstream packages at pinned commits **without** their own
dependency lists, because those would replace the GPU build of PyTorch with a CPU build and pull in
GUI and training packages. `--dry-run` prints the commands.

## 4. First run

```
python main.py process --input clip.mp4 --output clip_green.mp4 --model rvm
python main.py serve              # web interface at http://127.0.0.1:8765
```

## GPU and CPU

* `--device auto` (default) uses CUDA when it works, then MPS, then the CPU. A GPU that is present
  but unusable (driver problem, no free memory) is reported and the CPU is used instead.
* Half precision is used automatically on CUDA.
* CPU-only operation works for every model. Measured with RVM at 576p on the development machine:
  about 8-9 fps inference. SAM 2 and MatAnyone are much slower on a CPU (well under 1 fps).
* When the GPU runs out of memory the batch is halved automatically, then the model is asked for a
  cheaper configuration, and finally the job continues on the CPU with the mattes computed so far.
* Only one job runs at a time in the web interface.

## Directories

| Setting (`configs/default.yaml`) | Environment | Default | Content |
|---|---|---|---|
| `models_dir` | `VMT_MODELS_DIR` | `models/` | weights |
| `data_dir` | `VMT_DATA_DIR` | `data/` | uploads, jobs, reports, logs (web interface) |
| `cache_dir` | `VMT_CACHE_DIR` | `.cache/` | third-party caches (`TORCH_HOME`, `HF_HOME`) |
| `temp_dir` | `VMT_TEMP_DIR` | `data/tmp` | scratch files |

Nothing is written outside these directories and the output locations you choose. Put them on a
drive with space: an intermediate file is written next to each output, and the matte cache of a job
needs roughly 5-40 MB per minute of 1080p video.

## Docker

A `Dockerfile` (CUDA runtime base, FFmpeg, the pinned requirements) is included:

```
docker build -t video-matting-tool .
docker run --gpus all -p 8765:8765 -v vmt-data:/app/data -v vmt-models:/app/models video-matting-tool
```

It was written for the NVIDIA Container Toolkit and has **not been built or run** on the development
machine (no Docker there). Treat it as a starting point.

## Tests

```
pip install -r requirements-dev.txt
python -m pytest                       # everything; model and GPU tests skip when unavailable
python -m pytest -m "not gpu and not ui"
python scripts/download_test_assets.py  # optional: real test footage used by some tests
```
