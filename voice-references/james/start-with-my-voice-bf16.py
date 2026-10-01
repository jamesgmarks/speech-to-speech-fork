"""Run the local voice app with James's recorded voice reference."""

import os
from pathlib import Path

folder = Path(__file__).resolve().parent
root = folder.parent.parent
cache = Path.home() / ".cache/huggingface/hub"


def snapshot(model):
    base = cache / ("models--" + model.replace("/", "--"))
    return str(base / "snapshots" / (base / "refs/main").read_text().strip())


command = [
    str(root / ".venv/bin/speech-to-speech"),
    "serve",
    "--mac-optimal-settings",
    "--llm_backend",
    "mlx-lm",
    "--model_name",
    snapshot("Qwen/Qwen3-4B-Instruct-2507"),
    "--qwen3_tts_model_name",
    snapshot("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"),
    "--qwen3_tts_mlx_quantization",
    "bf16",
    "--qwen3_tts_voice_profiles",
    str(root / "voice-references/voices.json"),
    "--qwen3_tts_ref_audio",
    str(folder / "reference.wav"),
    "--qwen3_tts_ref_text",
    (folder / "reference.txt").read_text().strip(),
    "--host",
    "127.0.0.1",
]
os.chdir(root)
os.execv(command[0], command)
