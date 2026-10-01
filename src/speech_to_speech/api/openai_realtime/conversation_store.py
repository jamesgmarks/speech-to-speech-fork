"""Durable, opt-in conversation context shared by all pipeline slots."""

import json
import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import UUID

from speech_to_speech.LLM.chat import Chat


class ConversationStore:
    def __init__(self, directory: str) -> None:
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = Lock()
        self._active: set[str] = set()
        self._saved: dict[str, str] = {}

    @staticmethod
    def validate_key(key: str) -> str:
        if str(UUID(key)) != key:
            raise ValueError("conversation_key must be a canonical UUID")
        return key

    def acquire(
        self, key: str, backend: str, size: int, previous_backend: str | None = None
    ) -> tuple[Chat, bool, bool]:
        key = self.validate_key(key)
        with self._lock:
            if key in self._active:
                raise ValueError("This conversation is already connected. Disconnect its other tab first.")
            path = self.directory / f"{key}.json"
            saved = json.loads(path.read_text()) if path.exists() else None
            if saved is not None and (
                not isinstance(saved, dict)
                or saved.get("version") != 1
                or not isinstance(saved.get("items"), list)
                or not isinstance(saved.get("backend"), str)
            ):
                raise ValueError("Saved conversation is invalid; it has been preserved for recovery.")
            reset = bool((saved and saved["backend"] != backend) or (previous_backend and previous_backend != backend))
            items = saved["items"] if saved and not reset else []
            chat = Chat.from_persistent_snapshot(items, size)
            # Persist a backend switch immediately, including switches back to a
            # previously used backend before the first new user message arrives.
            self._save_locked(key, backend, chat.persistent_snapshot())
            self._active.add(key)
            return chat, bool(items), reset

    def save(self, key: str, backend: str, chat: Chat) -> None:
        items = chat.persistent_snapshot()
        with self._lock:
            self._save_locked(key, backend, items)

    def release(self, key: str) -> None:
        with self._lock:
            self._active.discard(key)
            self._saved.pop(key, None)

    def _save_locked(self, key: str, backend: str, items: list[dict[str, Any]]) -> None:
        content = json.dumps({"version": 1, "backend": backend, "items": items}, ensure_ascii=False)
        if self._saved.get(key) == content:
            return
        fd, temporary = tempfile.mkstemp(prefix=f".{key}.", dir=self.directory)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.directory / f"{key}.json")
            if os.name == "posix":
                directory_fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            self._saved[key] = content
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
