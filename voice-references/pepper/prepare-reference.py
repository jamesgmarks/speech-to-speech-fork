"""Build Pepper-only Qwen3 references; source clips are never overwritten.

Run from the repository root with .venv/bin/python. Requires FFmpeg and the
project's numpy/scipy/soundfile dependencies. The RNNoise model is cached in work/.
"""

import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags

FOLDER = Path(__file__).resolve().parent
WORK = FOLDER / "work"
MODEL_URL = "https://raw.githubusercontent.com/GregorR/rnnoise-models/master/somnolent-hogwash-2018-09-01/sh.rnnn"
FFMPEG = shutil.which("ffmpeg")
if not FFMPEG:
    raise RuntimeError("FFmpeg is required.")
WORK.mkdir(exist_ok=True)
model = WORK / "sh.rnnn"
if not model.exists():
    urllib.request.urlretrieve(MODEL_URL, model)
manifest = json.loads((FOLDER / "segments.json").read_text())
rate = manifest["sample_rate"]
segments = manifest["segments"]
sources = {}
metrics = {}
for name in sorted({segment["source"] for segment in segments}):
    source = FOLDER / f"{name}.wav"
    decoded = WORK / f"{name}-pcm.wav"
    cleaned = WORK / f"{name}-clean.wav"
    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(rate),
            "-c:a",
            "pcm_f32le",
            str(decoded),
        ],
        check=True,
    )
    # A partial wet/dry mix keeps breath and consonant detail. No gating,
    # pitch changes, time stretching, or aggressive speech-band filtering.
    filters = f"aresample=48000,highpass=f=70,arnndn=m={model}:mix=0.75,aresample={rate}"
    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(decoded),
            "-af",
            filters,
            "-ac",
            "1",
            "-ar",
            str(rate),
            "-c:a",
            "pcm_f32le",
            str(cleaned),
        ],
        check=True,
    )
    raw, _ = sf.read(decoded)
    clean, _ = sf.read(cleaned)
    length = min(len(raw), len(clean), rate * 8)
    correlations = correlate(clean[:length], raw[:length], method="fft")
    lags = correlation_lags(length, length)
    mask = np.abs(lags) < rate * 0.1
    lag = int(lags[mask][correlations[mask].argmax()])
    if lag > 0:
        clean = np.r_[clean[lag:], np.zeros(lag)]
    elif lag < 0:
        clean = np.r_[np.zeros(-lag), clean[:lag]]
    clean = clean[: len(raw)]
    if len(clean) < len(raw):
        clean = np.pad(clean, (0, len(raw) - len(clean)))
    sf.write(WORK / f"{name}-clean-aligned.wav", clean, rate, subtype="FLOAT")
    speech = np.concatenate(
        [clean[round(s["start"] * rate) : round(s["end"] * rate)] for s in segments if s["source"] == name]
    )
    rms = float(np.sqrt(np.mean(speech**2)))
    gain_db = float(np.clip(-20 - 20 * np.log10(max(rms, 1e-10)), -12, 6))
    sources[name] = (raw, clean, 10 ** (gain_db / 20))
    metrics[name] = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seconds": len(raw) / rate,
        "filter_delay_seconds": lag / rate,
        "scene_gain_db": round(gain_db, 3),
    }


def assemble(chosen, cleaned=True):
    parts = [np.zeros(round(rate * 0.12))]
    edits = []
    offset = len(parts[0])
    for index, segment in enumerate(chosen):
        if index:
            gap = np.zeros(round(rate * manifest["gap_seconds"]))
            parts.append(gap)
            offset += len(gap)
        raw, clean, gain = sources[segment["source"]]
        audio = (clean if cleaned else raw)[round(segment["start"] * rate) : round(segment["end"] * rate)].copy() * gain
        assert len(audio) > 0
        fade = min(round(rate * manifest["edge_fade_ms"] / 1000), len(audio) // 2)
        audio[:fade] *= np.linspace(0, 1, fade)
        audio[-fade:] *= np.linspace(1, 0, fade)
        edits.append({**segment, "output_start": offset / rate, "output_end": (offset + len(audio)) / rate})
        parts.append(audio)
        offset += len(audio)
    parts.append(np.zeros(round(rate * 0.15)))
    joined = np.concatenate(parts)
    # Peak normalization leaves 3 dB headroom, without compression.
    peak = float(np.max(np.abs(joined)))
    scale = 10 ** (-3 / 20) / max(peak, 1e-10)
    joined *= scale
    return joined, edits, scale


by_id = {segment["id"]: segment for segment in segments}
chosen = [by_id[name] for name in manifest["reference_order"]]
report = {
    "sample_rate": rate,
    "channels": 1,
    "subtype": "PCM_16",
    "sources": metrics,
    "denoiser_model_url": MODEL_URL,
    "denoiser_model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
    "processing": "70 Hz high-pass; RNNoise mix 0.75; scene-level RMS matching; 6 ms edge fades; 200 ms joins; -3 dBFS peak",
    "artifacts": {},
}
for filename, selection, cleaned in [
    ("reference", chosen, True),
    ("reference-unprocessed", chosen, False),
    ("all-pepper-lines", segments, True),
    ("reference-upbeat", [by_id[name] for name in manifest["upbeat_reference_order"]], True),
]:
    audio, edits, scale = assemble(selection, cleaned)
    sf.write(FOLDER / f"{filename}.wav", audio, rate, subtype="PCM_16")
    text = " ".join(s["text"] for s in selection)
    (FOLDER / f"{filename}.txt").write_text(text + "\n")
    report["artifacts"][filename] = {
        "seconds": round(len(audio) / rate, 3),
        "peak_dbfs": -3,
        "clipped_samples": int(np.sum(np.abs(audio) >= 1)),
        "final_gain_db": round(20 * np.log10(scale), 3),
        "transcript": text,
        "edits": edits,
    }
    print(filename, report["artifacts"][filename]["seconds"], "seconds", flush=True)
    if filename == "all-pepper-lines":
        for edit in edits:
            start = round(edit["output_start"] * rate)
            end = round(edit["output_end"] * rate)
            sf.write(WORK / (edit["id"] + ".wav"), audio[start:end], rate, subtype="PCM_16")

# Measure an actual inter-line pause, rather than the inserted digital silence.
raw, clean, _ = sources["trash"]
quiet = slice(round(11.45 * rate), round(12.08 * rate))
raw_rms = np.sqrt(np.mean(raw[quiet] ** 2))
clean_rms = np.sqrt(np.mean(clean[quiet] ** 2))
report["pause_noise_attenuation_db"] = round(float(20 * np.log10(raw_rms / max(clean_rms, 1e-10))), 2)
(FOLDER / "reference-report.json").write_text(json.dumps(report, indent=2) + "\n")
print("Measured pause background reduction:", report["pause_noise_attenuation_db"], "dB")
