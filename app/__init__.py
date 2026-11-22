"""Video Matting Tool: AI video matting and green-screen conversion."""

__version__ = "1.0.0"
__app_name__ = "video-matting-tool"

# PyAV's Python log callback can deadlock multi-threaded decoding (see app/utils/avlog.py); it is
# switched off before anything opens a video and kept off even when a library (torchvision) asks
# for it later.
from .utils.avlog import silence_pyav_logging as _silence_pyav_logging  # noqa: E402

_silence_pyav_logging()
