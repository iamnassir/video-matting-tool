"""Command line interface tests (in-process through Typer's test runner).

``process`` runs the real pipeline with the weight-free test model on the CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app import __version__
from app.cli.main import EXIT_ERROR, EXIT_OK, app
from tests.helpers import media

runner = CliRunner()


def invoke(*args: str):
    return runner.invoke(app, [str(a) for a in args])


def process_args(clip, out: Path, *extra: str):
    return ("process", "--input", clip.path, "--output", out, "--model", "test_chroma", "--device", "cpu", *extra)


# --------------------------------------------------------------------------- information commands


def test_version_and_help():
    result = invoke("--version")
    assert result.exit_code == EXIT_OK and __version__ in result.output
    result = invoke("--help")
    assert result.exit_code == EXIT_OK
    for command in ("process", "serve", "models", "probe", "system", "presets"):
        assert command in result.output
    result = invoke("process", "--help")
    for option in ("--input", "--output", "--model", "--background", "--device", "--quality", "--subject",
                   "--mask-refinement", "--edge-feather", "--temporal-smoothing", "--output-codec", "--preserve-audio",
                   "--export-alpha", "--batch-size", "--output-directory", "--verbose"):
        assert option in result.output, option


def test_models_commands():
    result = invoke("models", "list")
    assert result.exit_code == EXIT_OK and "rvm" in result.output and "GPL" in result.output
    result = invoke("models", "list", "--json")
    listed = json.loads(result.output)
    assert any(m["id"] == "rvm" and "status" in m for m in listed)
    result = invoke("models", "info", "rvm")
    assert result.exit_code == EXIT_OK
    assert "Limitations" in result.output and "sha256" in result.output and "github.com/PeterL1n" in result.output
    result = invoke("models", "info", "does_not_exist")
    assert result.exit_code == EXIT_ERROR and "unknown_model" in result.output


def test_model_download_requires_licence_acknowledgement():
    result = invoke("models", "download", "rvm")
    assert result.exit_code == EXIT_ERROR and "--accept-license" in result.output and "GPL" in result.output


def test_presets_and_system():
    result = invoke("presets")
    assert result.exit_code == EXIT_OK
    for word in ("balanced", "strong", "prores_4444", "ffv1"):
        assert word in result.output
    result = invoke("system", "--json")
    info = json.loads(result.output)
    assert info["devices"][-1]["id"] == "cpu" and "version" in info["ffmpeg"]


def test_probe(clip_vfr, clip_rotated):
    result = invoke("probe", clip_vfr.path)
    assert result.exit_code == EXIT_OK and "VARIABLE frame rate" in result.output and "aac" in result.output
    result = invoke("probe", clip_rotated.path, "--json")
    data = json.loads(result.output)
    assert data["video"]["rotation"] in (90, 270) and data["video"]["display_width"] == clip_rotated.height
    result = invoke("probe", "no_such_file.mp4")
    assert result.exit_code == EXIT_ERROR and "invalid_path" in result.output


# --------------------------------------------------------------------------- process


@pytest.mark.usefixtures("test_models")
def test_process_single_file(clip_cfr, tmp_path):
    out = tmp_path / "out green.mp4"                                   # a space in the path on purpose
    result = invoke(*process_args(clip_cfr, out, "--export-alpha"))
    assert result.exit_code == EXIT_OK, result.output
    assert out.is_file() and (tmp_path / "out green_alpha.mp4").is_file()
    assert "PROCESSING REPORT" in result.output and "Frame timestamps" in result.output
    report = json.loads((tmp_path / "out green.report.json").read_text(encoding="utf-8"))
    assert report["timing"]["frames_processed"] == clip_cfr.frames
    assert media.packet_pts(out) == media.packet_pts(clip_cfr.path)


@pytest.mark.usefixtures("test_models")
def test_process_options_reach_the_pipeline(clip_cfr, tmp_path):
    out = tmp_path / "custom.mkv"
    report_path = tmp_path / "reports" / "r.json"
    result = invoke(*process_args(
        clip_cfr, out, "--background", "blue", "--output-codec", "ffv1", "--mask-refinement", "strong",
        "--edge-feather", "1.5", "--temporal-smoothing", "0.2", "--batch-size", "3", "--max-frames", "12",
        "--no-preserve-audio", "--report", report_path, "--quiet",
    ))
    assert result.exit_code == EXIT_OK, result.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = report["config"]
    assert config["background"] == "blue" and config["encoding"]["codec"] == "ffv1"
    assert config["refinement"]["preset"] == "strong" and config["refinement"]["edge_feather"] == 1.5
    assert config["refinement"]["temporal_smoothing"] == 0.2 and config["batch_size"] == 3
    assert report["timing"]["frames_processed"] == 12 and report["encoding"]["audio"] == "none"
    info = media.ffprobe(out)
    assert [s["codec_name"] for s in info["streams"]] == ["ffv1"]
    assert "PROCESSING REPORT" not in result.output                       # --quiet


@pytest.mark.usefixtures("test_models")
def test_batch_processing_continues_after_a_bad_file(clip_cfr, clip_vfr, tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_text("not a video")
    out_dir = tmp_path / "batch out"
    result = invoke("process", "--input", clip_cfr.path, "--input", bad, "--input", clip_vfr.path,
                    "--output-directory", out_dir, "--model", "test_chroma", "--device", "cpu", "--quiet")
    assert result.exit_code == EXIT_ERROR                                   # one file failed ...
    assert (out_dir / f"{clip_cfr.path.stem}_green.mp4").is_file()          # ... the others were still processed
    assert (out_dir / f"{clip_vfr.path.stem}_green.mp4").is_file()
    assert not (out_dir / "broken_green.mp4").exists()
    assert "Batch summary" in result.output and "failed" in result.output


@pytest.mark.usefixtures("test_models")
def test_batch_stop_on_error_and_folder_input(clip_cfr, tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "a_bad.mp4").write_text("not a video")
    (folder / "b_good.mp4").write_bytes(clip_cfr.path.read_bytes())
    (folder / "notes.txt").write_text("ignored: not a video extension")
    out_dir = tmp_path / "out"
    result = invoke("process", "--input", folder, "--output-directory", out_dir, "--model", "test_chroma",
                    "--device", "cpu", "--stop-on-error", "--quiet")
    assert result.exit_code == EXIT_ERROR
    assert not (out_dir / "b_good_green.mp4").exists() and "not started" in result.output
    result = invoke("process", "--input", folder, "--output-directory", out_dir, "--model", "test_chroma",
                    "--device", "cpu", "--quiet")
    assert (out_dir / "b_good_green.mp4").is_file()


@pytest.mark.usefixtures("test_models")
def test_json_progress_lines(clip_cfr, tmp_path):
    result = invoke(*process_args(clip_cfr, tmp_path / "j.mp4", "--json-progress"))
    assert result.exit_code == EXIT_OK
    lines = [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]
    stages = [line["stage"] for line in lines if "stage" in line]
    assert "render" in stages and stages[-1] == "done"
    assert any(line.get("frame") == clip_cfr.frames for line in lines)
    assert "summary" in lines[-1] and lines[-1]["summary"][0]["status"].startswith("completed")


@pytest.mark.usefixtures("test_models")
def test_existing_output_needs_overwrite(clip_cfr, tmp_path):
    out = tmp_path / "keep.mp4"
    out.write_bytes(b"precious")
    result = invoke(*process_args(clip_cfr, out, "--quiet"))
    assert result.exit_code == EXIT_ERROR and "output_exists" in result.output and "--overwrite" in result.output
    assert out.read_bytes() == b"precious"
    result = invoke(*process_args(clip_cfr, out, "--quiet", "--overwrite"))
    assert result.exit_code == EXIT_OK and out.stat().st_size > 1000


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (("--input", "a.mp4"), "Specify --output"),
        (("--input", "a.mp4", "--input", "b.mp4", "--output", "o.mp4"), "single input"),
        (("--input", "a.mp4", "--output", "o.mp4", "--background", "transparent", "--output-codec", "h264"), "alpha"),
        (("--input", "a.mp4", "--output", "o.mp4", "--subject", "box:1,2,3"), "four numbers"),
        (("--input", "a.mp4", "--output", "o.mp4", "--quality", "ultra"), "quality"),
        (("--input", "a.mp4", "--output", "o.mp4", "--model-option", "novalue"), "key=value"),
        (("--input", "no_match_*.mp4", "--output-directory", "o"), "matches no files"),
    ],
)
def test_invalid_arguments_are_explained(arguments, expected):
    result = invoke("process", *arguments)
    assert result.exit_code == EXIT_ERROR
    assert expected in result.output and "Traceback" not in result.output


def test_missing_input_file_is_reported(tmp_path):
    result = invoke("process", "--input", tmp_path / "nope.mp4", "--output", tmp_path / "o.mp4", "--model", "rvm",
                    "--device", "cpu")
    assert result.exit_code == EXIT_ERROR and "invalid_path" in result.output and "Traceback" not in result.output
