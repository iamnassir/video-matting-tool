"""Fetch the optional real-footage test clips into ``tests/data/real/``.

    python scripts/download_test_assets.py                 # everything (about 85 MB)
    python scripts/download_test_assets.py A2_cbp_videoblog_speech B3_sam2_bedroom_540p
    python scripts/download_test_assets.py --list

The clips are described in ``tests/real_media_manifest.json`` (source URL, SHA-256, licence, subject
prompts).  They are third-party footage used for local testing only and are never part of the
repository; read each clip's licence in the manifest before using it for anything else.  Every file
is verified against its pinned SHA-256 after the download.

The two ``D1_*`` entries are derived locally with FFmpeg from clip ``C1`` (the same picture stored
sideways with rotation metadata, like a phone recording) because no licensed phone original with a
person was available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "real_media_manifest.json"
TARGET = ROOT / "tests" / "data" / "real"
USER_AGENT = "video-matting-tool-test-assets/1.0"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected_sha: str, expected_bytes: int) -> bool:
    if dest.is_file() and dest.stat().st_size == expected_bytes and sha256(dest) == expected_sha:
        print(f"   ok (already present)   {dest.relative_to(TARGET)}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as fh:
            shutil.copyfileobj(response, fh, length=1 << 20)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        part.unlink(missing_ok=True)
        print(f"   FAILED to download {url}: {exc}")
        return False
    digest = sha256(part)
    if digest != expected_sha:
        part.unlink(missing_ok=True)
        print(f"   FAILED checksum for {dest.name}: got {digest}, expected {expected_sha} (the source file changed)")
        return False
    part.replace(dest)
    print(f"   downloaded {dest.stat().st_size / 1e6:6.1f} MB   {dest.relative_to(TARGET)}")
    return True


def derive_rotated(source: Path, dest: Path) -> bool:
    """First two seconds of *source*, stored sideways with a -90 degree display matrix (phone style)."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("   SKIPPED derived clip: ffmpeg not found on PATH")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    sideways = dest.with_name(dest.stem + "_sideways.mp4")
    steps = [
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-t", "2", "-vf", "transpose=2",
         "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy", str(sideways)],
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-display_rotation", "-90", "-i", str(sideways),
         "-map", "0", "-c", "copy", str(dest)],
    ]
    try:
        for step in steps:
            result = subprocess.run(step, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"   FAILED to derive {dest.name}: {result.stderr.strip()[-300:]}")
                return False
    finally:
        sideways.unlink(missing_ok=True)
    print(f"   derived    {dest.stat().st_size / 1e6:6.1f} MB   {dest.relative_to(TARGET)}")
    return True


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Download the optional real-footage test clips.")
    parser.add_argument("clips", nargs="*", help="clip ids (default: all)")
    parser.add_argument("--list", action="store_true", help="list the available clips and exit")
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clips: Dict[str, dict] = manifest["clips"]
    if args.list:
        for cid, entry in clips.items():
            origin = entry.get("source_url") or "derived locally from C1"
            print(f"{cid:<50} {entry['bytes'] / 1e6:6.1f} MB  {origin}")
            print(f"{'':<50} licence: {entry.get('license')}")
        return 0
    wanted = args.clips or list(clips)
    unknown = [c for c in wanted if c not in clips]
    if unknown:
        print(f"unknown clip id(s): {', '.join(unknown)}  (use --list)")
        return 2
    TARGET.mkdir(parents=True, exist_ok=True)
    ok = True
    local: Dict[str, dict] = {}
    for cid in wanted:
        entry = clips[cid]
        dest = TARGET / entry["path"]
        print(f"{cid}")
        if entry.get("source_url"):
            good = fetch(entry["source_url"], dest, entry["sha256"], entry["bytes"])
        elif dest.is_file():
            good = True
            print(f"   ok (already present)   {dest.relative_to(TARGET)}")
        else:
            base = clips.get("C1_man_walking_vertical")
            base_path = TARGET / base["path"] if base else None
            if base_path is None or not base_path.is_file():
                print("   SKIPPED: needs clip C1_man_walking_vertical first")
                good = False
            else:
                good = derive_rotated(base_path, dest)
        ok = ok and good
        if good:
            local[cid] = entry
    for rel, entry in manifest.get("extra_files", {}).items():
        print(f"{rel}")
        ok = fetch(entry["source_url"], TARGET / rel, entry["sha256"], entry["bytes"]) and ok
    existing: Dict[str, dict] = {}
    index = TARGET / "manifest.json"
    if index.is_file():
        try:
            existing = json.loads(index.read_text(encoding="utf-8")).get("clips", {})
        except ValueError:
            existing = {}
    existing.update(local)
    index.write_text(json.dumps({"schema": manifest["schema"], "clips": existing}, indent=1), encoding="utf-8")
    print("\nAll requested files are in place." if ok else "\nSome files could not be fetched (see above).")
    print(f"Folder: {TARGET}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
