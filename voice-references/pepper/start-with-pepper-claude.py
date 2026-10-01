"""Run the voice app with Claude Code and the prepared Pepper reference.

Append CLI flags to override settings, e.g. --claude_agent_max_tokens 4096.
"""

import os
import sys
from pathlib import Path

folder = Path(__file__).resolve().parent
root = folder.parent.parent
command = [
    str(root / ".venv/bin/speech-to-speech"),
    "serve",
    "--mac-optimal-settings",
    "--llm_backend",
    "claude-agent-sdk",
    "--model_name",
    "sonnet",
    "--claude_agent_cwd",
    str(root),
    "--claude_agent_max_tokens",
    "2048",
    "--claude_agent_effort",
    "low",
    "--claude_agent_max_retries",
    "1",
    "--stream_batch_sentences",
    "1",
    "--qwen3_tts_model_name",
    str(Path.home() / ".cache/speech-to-speech/qwen3-tts-base-6bit"),
    "--qwen3_tts_mlx_quantization",
    "6bit",
    "--qwen3_tts_streaming_chunk_size",
    "12",
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
os.execv(command[0], command + sys.argv[1:])
