"""Processing report: what was done, on what hardware, how long it took and whether the output
passed quality verification.  Produced by the pipeline for every run (also failed ones)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .. import __version__


class CheckStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIPPED = "skipped"


class QualityCheck(BaseModel):
    """Outcome of one automated verification step."""

    model_config = ConfigDict(extra="forbid")

    id: str                                   # stable identifier, e.g. "resolution", "frame_count", "timestamps"
    name: str                                 # human readable title
    status: CheckStatus
    detail: str = ""                          # one-sentence explanation of the result
    expected: Any = None
    measured: Any = None


class ComparisonRow(BaseModel):
    """One line of the input-versus-output property table."""

    model_config = ConfigDict(extra="forbid")

    property: str
    input: Any = None
    output: Any = None
    match: Optional[bool] = None              # None = informational (not expected to be equal)
    note: str = ""


class OutputFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str                                 # main | alpha_matte | png_sequence | prores_4444 | green_screen | audio | matte_preview | report
    label: str
    path: str
    size_bytes: int = 0
    codec: Optional[str] = None
    container: Optional[str] = None
    frames: Optional[int] = None


class HardwareSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str = "cpu"                       # cuda:0 | mps | cpu
    device_name: str = ""
    requested_device: str = "auto"
    fallback_reason: Optional[str] = None     # set when the run did not use the requested device
    precision: str = "fp32"
    vram_total_mb: Optional[float] = None
    vram_free_at_start_mb: Optional[float] = None
    vram_peak_allocated_mb: Optional[float] = None
    ram_total_mb: Optional[float] = None
    ram_peak_process_mb: Optional[float] = None
    cpu: str = ""
    cpu_threads: Optional[int] = None
    os: str = ""
    python: str = ""
    torch: str = ""
    cuda: Optional[str] = None
    ffmpeg: str = ""
    final_batch_size: Optional[int] = None
    batch_reductions: int = 0                 # times the batch size was reduced after out-of-memory


class TimingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_seconds: float = 0.0
    stages: Dict[str, float] = Field(default_factory=dict)   # seconds per stage
    frames_processed: int = 0
    frames_from_cache: int = 0                # frames whose matte came from the resume cache
    average_fps: Optional[float] = None       # frames / wall-clock seconds of the render stage
    average_inference_fps: Optional[float] = None   # frames / seconds spent inside the model
    realtime_factor: Optional[float] = None   # video duration / total processing time
    #: Where the render stage spent its time, per stage (only measured in profiling / verbose runs,
    #: because honest GPU timing needs synchronisation points that cost a little throughput).
    render_breakdown: Dict[str, Any] = Field(default_factory=dict)


class ProcessingReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_version: str = __version__
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    status: str = "completed"                 # completed | completed_with_warnings | failed | cancelled
    input_path: str = ""
    output_path: Optional[str] = None
    input: Dict[str, Any] = Field(default_factory=dict)      # VideoMetadata.to_dict()
    output: Optional[Dict[str, Any]] = None                  # VideoMetadata.to_dict() of the result
    comparison: List[ComparisonRow] = Field(default_factory=list)
    model: Dict[str, Any] = Field(default_factory=dict)      # id, name, variant, licence, alpha_type, limitations...
    subject: Dict[str, Any] = Field(default_factory=dict)    # ResolvedSubject.summary()
    hardware: HardwareSummary = Field(default_factory=HardwareSummary)
    timing: TimingSummary = Field(default_factory=TimingSummary)
    config: Dict[str, Any] = Field(default_factory=dict)     # ProcessingConfig dump
    encoding: Dict[str, Any] = Field(default_factory=dict)   # codec, container, pix_fmt, ffmpeg args, audio handling
    checks: List[QualityCheck] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    outputs: List[OutputFile] = Field(default_factory=list)
    matte_stats: Dict[str, Any] = Field(default_factory=dict)   # coverage, flicker index, tracking events
    resumed: bool = False
    error: Optional[Dict[str, Any]] = None    # VmtError.to_dict() when status == failed

    # -- helpers -------------------------------------------------------------------------
    def add_warning(self, text: str) -> None:
        if text and text not in self.warnings:
            self.warnings.append(text)

    def add_check(self, check: QualityCheck) -> None:
        self.checks = [c for c in self.checks if c.id != check.id] + [check]

    @property
    def failed_checks(self) -> List[QualityCheck]:
        return [c for c in self.checks if c.status is CheckStatus.FAIL]

    @property
    def warned_checks(self) -> List[QualityCheck]:
        return [c for c in self.checks if c.status is CheckStatus.WARN]

    def finalize_status(self) -> None:
        """Derive ``status`` from checks and warnings (unless already failed / cancelled)."""

        if self.status in ("failed", "cancelled"):
            return
        if self.failed_checks or self.warnings or self.warned_checks:
            self.status = "completed_with_warnings"
        else:
            self.status = "completed"

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent, ensure_ascii=False)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "ProcessingReport":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_text(self) -> str:
        """Plain-text rendering for terminals and log files (ASCII only)."""

        lines: List[str] = []
        add = lines.append
        add("=" * 78)
        add(f" PROCESSING REPORT - status: {self.status.upper()}")
        add("=" * 78)
        add(f" Input : {self.input_path}")
        add(f" Output: {self.output_path or '-'}")
        m = self.model
        if m:
            lic = (m.get("license") or {}).get("name", "?")
            add(f" Model : {m.get('name', m.get('id', '?'))} [{m.get('variant', '?')}]  licence: {lic}")
        hw = self.hardware
        add(f" Device: {hw.device} {('(' + hw.device_name + ')') if hw.device_name else ''}  precision: {hw.precision}")
        if hw.fallback_reason:
            add(f"         fallback: {hw.fallback_reason}")
        t = self.timing
        fps = f"{t.average_fps:.2f}" if t.average_fps else "-"
        ifps = f"{t.average_inference_fps:.2f}" if t.average_inference_fps else "-"
        add(f" Time  : {t.total_seconds:.1f} s total, {t.frames_processed} frames, {fps} fps end-to-end, {ifps} fps inference")
        if t.frames_from_cache:
            add(f"         {t.frames_from_cache} frames reused from the matte cache")
        if self.comparison:
            add("-" * 78)
            add(f" {'Property':<24}{'Input':<24}{'Output':<24}OK")
            for row in self.comparison:
                flag = "" if row.match is None else ("yes" if row.match else "NO")
                add(f" {row.property:<24}{_short(row.input):<24}{_short(row.output):<24}{flag}")
        if self.checks:
            add("-" * 78)
            add(" Quality verification")
            for c in self.checks:
                add(f"  [{c.status.value.upper():<7}] {c.name}: {c.detail}")
        if self.warnings:
            add("-" * 78)
            add(" Warnings")
            for w in self.warnings:
                add(f"  - {w}")
        if self.outputs:
            add("-" * 78)
            add(" Files")
            for o in self.outputs:
                add(f"  {o.label:<26} {o.path}  ({_size(o.size_bytes)})")
        if self.error:
            add("-" * 78)
            add(f" ERROR [{self.error.get('code')}]: {self.error.get('message')}")
            if self.error.get("hint"):
                add(f"   hint: {self.error['hint']}")
        add("=" * 78)
        return "\n".join(lines)


def _short(value: Any, width: int = 22) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)
    return text if len(text) <= width else text[: width - 2] + ".."


def _size(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{n} B"
