"""Backend-wide settings shared by every client (currently just the agent instructions)."""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

MAX_INSTRUCTIONS_CHARS = 20_000

# Recovered from the original browser-local value; used the first time the server runs.
SEED_AGENT_INSTRUCTIONS = (
    "You are a competent personal assistant and orchestrator. \n\n"
    "While you have a plethora of tools at your disposal, you should do the minimum possible thing on your turn, "
    "opting instead to delegate the bulk of any execution to peer sessions (that you are able to launch via the "
    "custom tool provided to you) where possible, keeping your personal activity to a minimum in order to be "
    "highly responsive in your own conversation thread.\n\n"
    "With these guardrails taken into account, one of your key purposes right now is to assist in developing "
    "reusable tools and systems for managing your user's regular concerns and then employing those tools and "
    "systems to reduce the effort that your user needs to output in order to accomplish the same tasks.\n\n"
    "You'll provide UI capability to represent relevant data to the user when the user most needs it."
)


class StaleSettings(Exception):
    """The caller's etag no longer matches the stored value."""


def etag_for(text: str) -> str:
    return '"' + hashlib.sha256(text.encode()).hexdigest()[:16] + '"'


class SettingsStore:
    """Persisted in ``settings.json`` under ``directory``; in-memory only when no directory is configured."""

    def __init__(self, directory: str | Path | None) -> None:
        self._path = Path(directory).expanduser() / "settings.json" if directory else None
        self._lock = Lock()
        self._agent_instructions = SEED_AGENT_INSTRUCTIONS
        self._updated_at: str | None = None
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self._path.exists():
            saved = json.loads(self._path.read_text())
            if not isinstance(saved, dict) or not isinstance(saved.get("agent_instructions"), str):
                raise ValueError(f"{self._path} is invalid; fix or remove it.")
            self._agent_instructions = saved["agent_instructions"]
            self._updated_at = saved.get("updated_at")
        else:
            self._write_locked()

    @property
    def agent_instructions(self) -> str:
        return self._agent_instructions

    def snapshot(self) -> dict[str, str | None]:
        with self._lock:
            return {
                "agent_instructions": self._agent_instructions,
                "updated_at": self._updated_at,
                "etag": etag_for(self._agent_instructions),
            }

    def set_agent_instructions(self, text: str, if_match: str | None) -> dict[str, str | None]:
        if len(text) > MAX_INSTRUCTIONS_CHARS:
            raise ValueError(f"Instructions are limited to {MAX_INSTRUCTIONS_CHARS} characters")
        with self._lock:
            if if_match is not None and if_match != etag_for(self._agent_instructions):
                raise StaleSettings
            self._agent_instructions = text
            self._updated_at = datetime.now(timezone.utc).isoformat()
            self._write_locked()
        return self.snapshot()

    def _write_locked(self) -> None:
        if self._path is None:
            return
        body = json.dumps({"agent_instructions": self._agent_instructions, "updated_at": self._updated_at})
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".settings-")
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(body)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self._path)
        except BaseException:
            os.unlink(tmp)
            raise
