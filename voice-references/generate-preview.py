"""Generate a continuous preview in one Qwen3 synthesis call."""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--folder", type=Path, default=Path(__file__).resolve().parent / "pepper")
parser.add_argument("--text", default="amir-demo.txt", help="Text file inside this folder.")
parser.add_argument("--reference", default="reference", help="Reference WAV/TXT basename inside this folder.")
parser.add_argument("--output", default="amir-demo", help="Output basename inside this folder.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--language", default="english", help="Qwen3 generation language; default english.")
parser.add_argument("--match-level", help="Optional comparison WAV inside this folder; match its RMS playback level.")
args = parser.parse_args()
folder = args.folder.expanduser().resolve()
for value in (args.text, args.reference, args.output, *([args.match_level] if args.match_level else [])):
    if Path(value).name != value:
        parser.error("Use filenames/basenames inside the selected voice folder.")
model_path = Path.home() / ".cache/speech-to-speech/qwen3-tts-base-6bit"
model = load_model(str(model_path))
mx.random.seed(args.seed)
text = (folder / args.text).read_text().strip()
started = time.perf_counter()
chunks = []
first_audio = None
for result in model.generate(
    text=text,
    ref_audio=str(folder / f"{args.reference}.wav"),
    ref_text=(folder / f"{args.reference}.txt").read_text().strip(),
    lang_code=args.language,
    stream=True,
    streaming_interval=12 / 12.5,
    max_tokens=700,
    verbose=False,
):
    if first_audio is None:
        first_audio = time.perf_counter() - started
        print(f"First generated audio after {first_audio:.2f}s", flush=True)
    chunks.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
    sample_rate = result.sample_rate
assert chunks, "No generated audio"
audio = np.concatenate(chunks)
assert np.isfinite(audio).all()
level_gain = 1.0
if args.match_level:
    comparison, comparison_rate = sf.read(folder / args.match_level)
    assert comparison_rate == sample_rate, "Comparison sample rate must match"
    level_gain = float(np.sqrt(np.mean(comparison**2)) / max(np.sqrt(np.mean(audio**2)), 1e-10))
    audio *= level_gain
peak = float(np.max(np.abs(audio)))
scale = min(1.0, 10 ** (-1 / 20) / max(peak, 1e-10))
audio *= scale
sf.write(folder / f"{args.output}.wav", audio, sample_rate, subtype="PCM_16")
subprocess.run(
    [
        shutil.which("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(folder / f"{args.output}.wav"),
        "-c:a",
        "libmp3lame",
        "-b:a",
        "128k",
        str(folder / f"{args.output}.mp3"),
    ],
    check=True,
)
report = {
    "text": text,
    "model": str(model_path),
    "reference": f"{args.reference}.wav",
    "seed": args.seed,
    "language": args.language,
    "sample_rate": sample_rate,
    "audio_seconds": round(len(audio) / sample_rate, 3),
    "generation_seconds": round(time.perf_counter() - started, 3),
    "first_audio_seconds": round(first_audio, 3),
    "peak_dbfs": round(float(20 * np.log10(max(np.max(np.abs(audio)), 1e-10))), 2),
    "clipped_samples": int(np.sum(np.abs(audio) >= 1)),
    "note": "AI-generated preview; the entire supplied text is synthesized in one call.",
    "synthesis_calls": 1,
    "text_file": args.text,
    "level_match_reference": args.match_level,
    "level_match_gain_db": round(float(20 * np.log10(level_gain)), 3),
    "style_method": "Alternate upbeat reference and expressive punctuation; no direct emotion instruction."
    if args.reference == "reference-upbeat"
    else "Original reference and dialogue.",
}
(folder / f"{args.output}-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
