# syntax=docker/dockerfile:1
#
# Video Matting Tool - container image (web interface + CLI).
#
#   docker build -t video-matting-tool .
#   docker run --gpus all -p 8765:8765 -v vmt-models:/app/models -v vmt-data:/app/data video-matting-tool
#   docker run --gpus all -v "$PWD:/work" -v vmt-models:/app/models video-matting-tool \
#       python main.py process --input /work/in.mp4 --output /work/out_green.mp4 --download-models
#
# NOTE: this Dockerfile was written for the validated software stack (PyTorch 2.5.1 / CUDA 12.1,
# FFmpeg 8.1) but could not be built on the development machine (no Docker there): treat the first
# build as untested.  GPU use needs the NVIDIA Container Toolkit on the host; without "--gpus all" the
# tool runs on the CPU.
#
# Model weights are NOT part of the image (licences differ per model and downloads are explicit):
# keep /app/models on a volume and download with "python main.py models download <id>".

ARG TORCH_IMAGE=pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
FROM ${TORCH_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# A current static FFmpeg build: distribution packages are too old for the timestamp-exact
# encode path (the tool needs FFmpeg 7 or newer; it was developed against 8.1).
ARG FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl xz-utils git \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL "${FFMPEG_URL}" -o /tmp/ffmpeg.tar.xz \
 && mkdir -p /opt/ffmpeg \
 && tar -xJf /tmp/ffmpeg.tar.xz -C /opt/ffmpeg --strip-components=1 \
 && ln -s /opt/ffmpeg/bin/ffmpeg /usr/local/bin/ffmpeg \
 && ln -s /opt/ffmpeg/bin/ffprobe /usr/local/bin/ffprobe \
 && rm /tmp/ffmpeg.tar.xz \
 && ffmpeg -hide_banner -version | head -n 1

WORKDIR /app

# Dependencies first (better layer caching).  The constraints file is regenerated from the PyTorch
# that ships in the base image so pip can never replace it.
COPY requirements.txt ./
RUN python -c "import torch, torchvision; print('torch==' + torch.__version__); print('torchvision==' + torchvision.__version__); print('numpy==1.26.4')" > constraints.txt \
 && cat constraints.txt \
 && pip install -r requirements.txt -c constraints.txt \
 && python -c "import torch, av, cv2, fastapi; print('torch', torch.__version__, '| av', av.__version__, '| cv2', cv2.__version__)"

COPY app ./app
COPY configs ./configs
COPY scripts ./scripts
COPY main.py pyproject.toml README.md ./

# Optional model packages.  SAM 2 (Apache-2.0) and text prompts are installed by default; MatAnyone is
# non-commercial and therefore opt-in:  docker build --build-arg INSTALL_MATANYONE=1 ...
ARG INSTALL_SAM2=1
ARG INSTALL_TEXT=1
ARG INSTALL_MATANYONE=0
RUN if [ "${INSTALL_SAM2}" = "1" ]; then python scripts/install_models.py sam2; fi \
 && if [ "${INSTALL_TEXT}" = "1" ]; then python scripts/install_models.py text; fi \
 && if [ "${INSTALL_MATANYONE}" = "1" ]; then python scripts/install_models.py matanyone --accept-license; fi

RUN mkdir -p /app/models /app/data /app/.cache
VOLUME ["/app/models", "/app/data"]

# The server must listen on all interfaces inside the container; publish the port only on the
# host's loopback interface ("-p 127.0.0.1:8765:8765") unless you really want to expose it.
ENV VMT_HOST=0.0.0.0 \
    VMT_PORT=8765
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${VMT_PORT}/api/health" || exit 1

CMD ["python", "main.py", "serve", "--no-open"]
