"""Decode the supplied Morgan video and run the shared reference preparation."""

import hashlib
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

folder = Path(__file__).resolve().parent
source = folder / "from_a_tiktok.mp4"
ffmpeg = shutil.which("ffmpeg")
if not ffmpeg:
    raise RuntimeError("FFmpeg is required.")
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
subprocess.run(
    [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "24000",
        "-c:a",
        "pcm_s16le",
        str(folder / "from_a_tiktok.wav"),
    ],
    check=True,
)
sys.argv = [str(folder.parent / "prepare-reference.py"), "--folder", str(folder)]
runpy.run_path(sys.argv[0], run_name="__main__")
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash, "Original video changed"
report_path = folder / "reference-report.json"
report = json.loads(report_path.read_text())
report["original_video"] = {"filename": source.name, "sha256": source_hash}
report_path.write_text(json.dumps(report, indent=2) + "\n")
