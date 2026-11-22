"""Exception hierarchy for the whole application.

Every error carries:

* ``code``    - a stable, machine readable identifier (used by the API and the CLI exit report)
* ``message`` - a human readable description of what went wrong
* ``hint``    - an optional, actionable suggestion for the user
* ``details`` - optional structured context (paths, ffmpeg stderr tail, sizes, ...)

Rule for all modules: never swallow an unexpected exception silently and never let a partially
written file survive under its final name.  Convert low level failures (``av.error.*``,
``subprocess`` failures, ``torch`` OOM, ``OSError``) into one of these classes at module
boundaries so that callers can react and users get a meaningful message.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class VmtError(Exception):
    """Base class of all application errors."""

    code: str = "error"
    default_hint: Optional[str] = None

    def __init__(
        self,
        message: str,
        *,
        hint: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint if hint is not None else self.default_hint
        self.details: Dict[str, Any] = dict(details or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "type": type(self).__name__,
            "message": self.message,
            "hint": self.hint,
            "details": self.details,
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.hint:
            return f"{self.message} (hint: {self.hint})"
        return self.message


# --------------------------------------------------------------------------- configuration


class ConfigError(VmtError):
    """Invalid or inconsistent configuration / command line arguments."""

    code = "config_error"


# --------------------------------------------------------------------------- input


class InputError(VmtError):
    """Problem with the input file."""

    code = "input_error"


class InvalidPathError(InputError):
    code = "invalid_path"


class UnsupportedFormatError(InputError):
    code = "unsupported_format"
    default_hint = "Supported containers: MP4, MOV, MKV, AVI, WebM. Re-wrap or transcode the file with FFmpeg first."


class NoVideoStreamError(InputError):
    code = "no_video_stream"


class CorruptedVideoError(InputError):
    code = "corrupted_video"
    default_hint = (
        "The file could not be decoded cleanly. Try repairing it with "
        "'ffmpeg -err_detect ignore_err -i in -c copy out', or pass --tolerate-decode-errors "
        "to skip undecodable frames (the report will list them)."
    )


# --------------------------------------------------------------------------- models


class ModelError(VmtError):
    code = "model_error"


class UnknownModelError(ModelError):
    code = "unknown_model"


class ModelNotInstalledError(ModelError):
    """The Python package that implements the model is not importable."""

    code = "model_not_installed"


class WeightsMissingError(ModelError):
    """Model weights are not on disk and automatic download is disabled."""

    code = "weights_missing"


class WeightsDownloadError(ModelError):
    code = "weights_download_failed"
    default_hint = "Check the network connection, or download the file manually and place it in the models directory."


class ChecksumMismatchError(ModelError):
    code = "weights_checksum_mismatch"
    default_hint = "The file is corrupt or was replaced. Delete it and download it again."


class LicenseError(ModelError):
    """The selected model's licence does not permit the requested use."""

    code = "license_restriction"


class ModelLoadError(ModelError):
    code = "model_load_failed"


class PromptNotSupportedError(ModelError):
    """The requested subject-selection method is not available for the selected model."""

    code = "prompt_not_supported"


# --------------------------------------------------------------------------- subject selection


class SubjectError(VmtError):
    code = "subject_error"


class SubjectNotFoundError(SubjectError):
    code = "subject_not_found"
    default_hint = "Select the subject manually with a box or clicks, or choose a frame where the subject is visible."


# --------------------------------------------------------------------------- devices


class DeviceError(VmtError):
    code = "device_error"


class CudaInitError(DeviceError):
    code = "cuda_init_failed"
    default_hint = "Update the NVIDIA driver, or run with --device cpu."


class GpuOutOfMemoryError(DeviceError):
    code = "gpu_out_of_memory"
    default_hint = (
        "Close other GPU applications, lower --batch-size, choose a lighter model variant or quality preset, "
        "or run with --device cpu."
    )


# --------------------------------------------------------------------------- encoding / muxing


class EncodingError(VmtError):
    code = "encoding_error"


class FfmpegNotFoundError(EncodingError):
    code = "ffmpeg_not_found"
    default_hint = "Install FFmpeg (https://ffmpeg.org/download.html) and make sure ffmpeg and ffprobe are on PATH, or set VMT_FFMPEG / VMT_FFPROBE."


class EncoderUnavailableError(EncodingError):
    code = "encoder_unavailable"


class FfmpegFailedError(EncodingError):
    """An ffmpeg / ffprobe process exited with an error. ``details['stderr_tail']`` holds its last lines."""

    code = "ffmpeg_failed"


class AudioMuxError(EncodingError):
    code = "audio_mux_failed"


# --------------------------------------------------------------------------- storage


class StorageError(VmtError):
    code = "storage_error"


class DiskSpaceError(StorageError):
    code = "insufficient_disk_space"
    default_hint = "Free up disk space or choose another output / work directory."


class OutputPathError(StorageError):
    code = "invalid_output_path"


class OutputExistsError(OutputPathError):
    code = "output_exists"
    default_hint = "Pass --overwrite to replace the existing file."


# --------------------------------------------------------------------------- verification / jobs


class VerificationError(VmtError):
    """A hard quality check failed and strict verification is enabled."""

    code = "verification_failed"


class JobCancelledError(VmtError):
    """Processing was cancelled by the user. Not a failure."""

    code = "cancelled"


class JobNotFoundError(VmtError):
    code = "job_not_found"


class ResourceBusyError(VmtError):
    """The requested operation needs a resource (the GPU, a file) that is in use right now."""

    code = "busy"


class ResumeError(VmtError):
    """Cached state exists but cannot be reused (different input, model or settings)."""

    code = "resume_mismatch"


def is_cuda_oom(exc: BaseException) -> bool:
    """Return True if *exc* is a CUDA / MPS out-of-memory condition raised by PyTorch."""

    name = type(exc).__name__
    if name == "OutOfMemoryError":
        return True
    text = str(exc).lower()
    needles = (
        "out of memory",
        "cuda error: out of memory",
        "cudnn_status_alloc_failed",
        "cudnn_status_internal_error: alloc",
        # cuBLAS reports a failed workspace allocation this way when the GPU is (nearly) full,
        # which is the normal situation on a card shared with other applications.
        "cublas_status_alloc_failed",
        "cublas_status_execution_failed",
        "mps backend out of memory",
        "unable to allocate",
        "not enough memory",
    )
    return isinstance(exc, (RuntimeError, MemoryError)) and any(n in text for n in needles)
