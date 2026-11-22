"""Install the optional model packages into the current Python environment.

    python scripts/install_models.py sam2                      # SAM 2.1 (Apache-2.0)
    python scripts/install_models.py matanyone --accept-license  # MatAnyone / MatAnyone 2 (non-commercial)
    python scripts/install_models.py text                      # text prompts (Grounding DINO via transformers)
    python scripts/install_models.py all --accept-license
    python scripts/install_models.py sam2 --dry-run            # only print the commands

Why a script instead of a requirements file: the upstream projects pin their own dependency sets
(PyTorch versions, GUI and training packages).  Installing them the usual way would replace the
hardware-specific PyTorch build with a CPU-only one from PyPI, pull a second OpenCV and NumPy 2, and
fail on packages that need a C compiler.  This script installs exactly what inference needs:

* the model package itself from its official repository at a pinned commit, with ``--no-deps``;
* the few runtime dependencies, under ``constraints.txt`` plus the PyTorch build that is installed
  (CUDA, CPU or Apple), so PyTorch / NumPy cannot be touched;
* and it verifies afterwards that PyTorch is unchanged.

Model *weights* are a separate, explicit step:  ``python main.py models download <id>``.
Run this script with the interpreter of the environment you want to install into.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS = ROOT / "constraints.txt"
#: Placeholder in the pip steps below; replaced by the file :func:`effective_constraints` writes.
CONSTRAINTS_ARG = "{constraints}"

SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
MATANYONE_COMMIT = "e5ddc534c1fff9bb9e54cf476095d29071b7cb4f"

PACKAGES: Dict[str, Dict[str, object]] = {
    "sam2": {
        "title": "SAM 2.1 (facebookresearch/sam2)",
        "license": "Apache-2.0 (code and checkpoints): commercial use allowed.",
        "restricted": False,
        "needs_git": True,
        "env": {"SAM2_BUILD_CUDA": "0"},     # never try to compile the optional CUDA extension
        "steps": [
            ["install", "setuptools>=61,<81", "wheel"],
            # --no-build-isolation: otherwise pip downloads a second (CPU) PyTorch just to build the wheel
            ["install", "--no-deps", "--no-build-isolation",
             f"git+https://github.com/facebookresearch/sam2.git@{SAM2_COMMIT}"],
            ["install", "-c", CONSTRAINTS_ARG, "hydra-core>=1.3.2", "iopath==0.1.10"],
        ],
        "check": "import sam2, hydra; from sam2.build_sam import build_sam2_video_predictor",
        "next": "python main.py models download sam2",
    },
    "matanyone": {
        "title": "MatAnyone / MatAnyone 2 (pq-yang/MatAnyone)",
        "license": "NTU S-Lab License 1.0: research / NON-COMMERCIAL use only. Commercial use needs written "
                   "permission from the authors (https://github.com/pq-yang/MatAnyone/blob/main/LICENSE).",
        "restricted": True,
        "needs_git": True,
        "env": {},
        "steps": [
            ["install", "--no-deps", f"git+https://github.com/pq-yang/MatAnyone@{MATANYONE_COMMIT}"],
            ["install", "-c", CONSTRAINTS_ARG, "omegaconf>=2.3,<2.4", "safetensors>=0.4.3", "imageio>=2.31"],
        ],
        "check": "import matanyone, omegaconf",
        "next": "python main.py models download matanyone --accept-license",
        "note": "'pip check' will list MatAnyone's GUI and training dependencies as missing. That is expected: "
                "they are not needed for inference and are deliberately not installed.",
    },
    "text": {
        "title": "Text prompts (Hugging Face transformers for Grounding DINO)",
        "license": "transformers: Apache-2.0. Grounding DINO weights: Apache-2.0.",
        "restricted": False,
        "needs_git": False,
        "env": {},
        "steps": [
            # last 4.x line: works with Python 3.10 and PyTorch 2.5 (5.x needs newer versions of both)
            ["install", "-c", CONSTRAINTS_ARG, "transformers==4.57.6", "tokenizers==0.22.2", "huggingface-hub==0.36.2"],
        ],
        "check": "import transformers; from transformers import AutoModelForZeroShotObjectDetection",
        "next": "python main.py models download text_grounding",
    },
}


def torch_version() -> str:
    out = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__, torch.version.cuda)"],
        capture_output=True, text=True,
    )
    return out.stdout.strip() if out.returncode == 0 else ""


def effective_constraints() -> Path:
    """``constraints.txt`` plus the PyTorch / torchvision builds installed right now.

    Pinning the installed build (``2.5.1+cu121``, ``2.5.1+cpu``, ``2.5.1`` ...) makes pip keep it
    whatever a dependency asks for, on every kind of machine.
    """

    text = CONSTRAINTS.read_text(encoding="utf-8")
    probe = subprocess.run(
        [sys.executable, "-c", "import torch, torchvision; print(torch.__version__); print(torchvision.__version__)"],
        capture_output=True, text=True,
    )
    versions = probe.stdout.split() if probe.returncode == 0 else []
    if len(versions) >= 2:
        text += f"\n# the installed PyTorch build\ntorch=={versions[0]}\ntorchvision=={versions[1]}\n"
    target = ROOT / ".cache" / "constraints.effective.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def run_pip(args: Sequence[str], env: Dict[str, str], dry_run: bool) -> int:
    constraints = str(effective_constraints())
    command = [sys.executable, "-m", "pip", *[constraints if a == CONSTRAINTS_ARG else a for a in args]]
    print("   $ " + " ".join(f'"{a}"' if " " in a else a for a in command))
    if dry_run:
        return 0
    full_env = dict(os.environ)
    full_env.update(env)
    full_env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    return subprocess.run(command, env=full_env).returncode


def install(name: str, accept_license: bool, dry_run: bool) -> bool:
    spec = PACKAGES[name]
    print(f"\n== {spec['title']} ==")
    print(f"   licence: {spec['license']}")
    if spec["restricted"] and not accept_license:
        print("   NOT INSTALLED: this licence has conditions. Re-run with --accept-license to confirm that you "
              "have read and accept them.")
        return False
    if spec["needs_git"] and shutil.which("git") is None:
        print("   NOT INSTALLED: 'git' was not found on PATH (needed to fetch the pinned commit).")
        return False
    before = torch_version()
    if not before and not dry_run:
        print("   NOT INSTALLED: PyTorch is not installed in this environment. Install it first (docs/INSTALL.md).")
        return False
    for step in spec["steps"]:     # type: ignore[union-attr]
        code = run_pip(step, spec["env"], dry_run)   # type: ignore[arg-type]
        if code != 0:
            print(f"   FAILED (pip exit code {code}). Nothing else was changed by this script.")
            return False
    if dry_run:
        return True
    after = torch_version()
    if after != before:
        print(f"   ERROR: PyTorch changed during the installation ({before!r} -> {after!r}).")
        print("   Reinstall the PyTorch build for your hardware (docs/INSTALL.md) and report this.")
        return False
    check = subprocess.run([sys.executable, "-c", str(spec["check"])], capture_output=True, text=True)
    if check.returncode != 0:
        print("   ERROR: the package was installed but cannot be imported:")
        print("   " + check.stderr.strip().splitlines()[-1] if check.stderr.strip() else "   (no error output)")
        return False
    print(f"   OK (PyTorch still {after}).")
    if spec.get("note"):
        print(f"   note: {spec['note']}")
    print(f"   next: {spec['next']}")
    return True


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Install optional model packages without disturbing PyTorch.")
    parser.add_argument("packages", nargs="+", choices=[*PACKAGES.keys(), "all"], help="what to install")
    parser.add_argument("--accept-license", action="store_true", help="confirm restricted (non-commercial) licences")
    parser.add_argument("--dry-run", action="store_true", help="print the pip commands without running them")
    args = parser.parse_args(argv)
    if not CONSTRAINTS.is_file():
        print(f"constraints file not found: {CONSTRAINTS}")
        return 1
    names = list(PACKAGES) if "all" in args.packages else list(dict.fromkeys(args.packages))
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch_version() or 'NOT INSTALLED'}")
    results = {name: install(name, args.accept_license, args.dry_run) for name in names}
    print("\nSummary: " + ", ".join(f"{n}: {'ok' if ok else 'not installed'}" for n, ok in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
