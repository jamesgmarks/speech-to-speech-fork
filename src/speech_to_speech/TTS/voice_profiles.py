"""Server-owned voice references; clients select IDs, never transcript paths."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class VoiceCatalogEvent(BaseModel):
    type: Literal["speech_to_speech.voices"] = "speech_to_speech.voices"
    voices: list[dict[str, Any]]
    default: str | None = None


class _ProfileSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^custom:[a-z0-9][a-z0-9_-]*$")
    name: str = Field(min_length=1)
    ref_audio: str = Field(min_length=1)
    ref_text_file: str = Field(min_length=1)


@dataclass(frozen=True)
class VoiceProfile:
    id: str
    name: str
    ref_audio: Path
    ref_text: str


def load_voice_profiles(manifest: str | Path | None) -> dict[str, VoiceProfile]:
    if manifest is None:
        return {}
    path = Path(manifest).expanduser().resolve()
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Voice profiles must be a nonempty JSON array.")
    profiles = {}
    for entry in entries:
        spec = _ProfileSpec.model_validate(entry)
        if spec.id in profiles:
            raise ValueError(f"Duplicate voice profile ID: {spec.id}")
        audio = (path.parent / Path(spec.ref_audio).expanduser()).resolve()
        if not audio.is_file():
            raise FileNotFoundError(f"Voice profile {spec.id} audio does not exist: {audio}")
        transcript = (path.parent / Path(spec.ref_text_file).expanduser()).read_text(encoding="utf-8").strip()
        if not transcript or not spec.name.strip():
            raise ValueError(f"Voice profile {spec.id} needs a name and nonempty reference transcript.")
        profiles[spec.id] = VoiceProfile(spec.id, spec.name.strip(), audio, transcript)
    return profiles
