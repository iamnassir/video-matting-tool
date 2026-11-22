"""Run a command while holding a machine-wide named lock.

    python scripts/with_lock.py torch -- python -m pytest tests/test_compositor.py

Purpose: memory-heavy processes (anything that imports PyTorch, anything that touches the GPU) must
not run concurrently on a workstation with little free RAM / VRAM.  Developers, CI steps and
automated agents wrap such commands with this script so they queue instead of piling up.

The lock is an OS file lock on ``.cache/locks/<name>.lock``: it is released automatically when the
process exits, even if it crashes, so it can never go stale.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_DIR = ROOT / ".cache" / "locks"


def _acquire(path: Path, timeout: float, poll: float = 0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+b")
    deadline = time.monotonic() + timeout
    announced = False
    while True:
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except OSError:
            if time.monotonic() >= deadline:
                fh.close()
                raise TimeoutError(f"could not acquire lock '{path.stem}' within {timeout:.0f} s")
            if not announced:
                print(f"[with_lock] waiting for lock '{path.stem}' ...", file=sys.stderr, flush=True)
                announced = True
            time.sleep(poll)


def main(argv: list) -> int:
    if "--" not in argv or len(argv) < 3:
        print("usage: with_lock.py <lock-name> [--timeout SECONDS] -- <command> [args...]", file=sys.stderr)
        return 2
    split = argv.index("--")
    head, command = argv[:split], argv[split + 1 :]
    name = head[0]
    timeout = 3600.0
    if "--timeout" in head:
        timeout = float(head[head.index("--timeout") + 1])
    if not command:
        print("with_lock.py: no command given", file=sys.stderr)
        return 2
    # "python" means "the interpreter running this script" (the project's virtual environment);
    # relative executables are resolved because Windows' CreateProcess does not accept ./ paths.
    exe = command[0]
    if exe.lower() in ("python", "python3", "python.exe"):
        command[0] = sys.executable
    elif any(sep in exe for sep in ("/", "\\")):
        command[0] = str(Path(exe).resolve())
    else:
        import shutil

        command[0] = shutil.which(exe) or exe
    try:
        handle = _acquire(LOCK_DIR / f"{name}.lock", timeout)
    except TimeoutError as exc:
        print(f"[with_lock] {exc}", file=sys.stderr)
        return 75
    try:
        t0 = time.monotonic()
        result = subprocess.run(command)
        print(f"[with_lock] '{name}' held for {time.monotonic() - t0:.1f} s, exit code {result.returncode}", file=sys.stderr)
        return result.returncode
    finally:
        handle.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
