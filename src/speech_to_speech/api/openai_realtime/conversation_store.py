"""Durable, opt-in conversation context shared by all pipeline slots."""

import json
import os
import re
import tempfile
import wave
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import UUID, uuid4

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

    @staticmethod
    def audio_url(key: str, audio_id: str) -> str:
        return f"/v1/conversation/audio/{key}/{audio_id}.wav"

    def save_audio(
        self, key: str, backend: str, clips: dict[str, bytes], item_audio: dict[str, str], sample_rate: int
    ) -> None:
        """Publish original generated speech atomically, separately from model context."""
        self.validate_key(key)
        with self._lock:
            path = self.directory / f"{key}.json"
            saved = json.loads(path.read_text())
            if saved.get("backend") != backend:
                return
            directory = self.directory / "audio" / key
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            for audio_id, pcm in clips.items():
                if not re.fullmatch(r"[a-f0-9]{64}", audio_id):
                    raise ValueError("Invalid audio ID")
                fd, temporary = tempfile.mkstemp(prefix=".audio-", dir=directory)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        with wave.open(stream, "wb") as output:
                            output.setnchannels(1)
                            output.setsampwidth(2)
                            output.setframerate(sample_rate)
                            output.writeframes(pcm)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, directory / f"{audio_id}.wav")
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            saved["audio_files"] = list(dict.fromkeys([*saved.get("audio_files", []), *clips]))
            saved.setdefault("item_audio", {}).update(item_audio)
            self._write_json_locked(f"{key}.json", saved)
            self._saved.pop(key, None)

    def audio_path(self, key: str, audio_id: str) -> Path:
        self.validate_key(key)
        if not re.fullmatch(r"[a-f0-9]{64}", audio_id):
            raise ValueError("Invalid audio ID")
        with self._lock:
            manifest = self.directory / f"{key}.json"
            if not manifest.is_file() or audio_id not in json.loads(manifest.read_text()).get("audio_files", []):
                raise FileNotFoundError("Recording unavailable")
            path = self.directory / "audio" / key / f"{audio_id}.wav"
            if not path.is_file():
                raise FileNotFoundError("Recording unavailable")
            return path

    def current(self, backend: str, size: int) -> dict[str, Any]:
        """Return the deployment's durable conversation without claiming a slot.

        On first use, adopt the most recently saved conversation so enabling
        this mode keeps the user's existing context. The pointer never follows
        subsequent writes to old conversations.
        """
        with self._lock:
            key = self._current_key_locked()
            return self._view_locked(key, backend, size)

    def start_new(self, expected_key: str, backend: str, size: int) -> dict[str, Any]:
        with self._lock:
            current = self._current_key_locked()
            if self.validate_key(expected_key) != current:
                raise ValueError("The conversation changed in another browser. Refresh before resetting it.")
            if current in self._active:
                raise ValueError("Disconnect the conversation in its other browser before starting a fresh one.")
            key = str(uuid4())
            self._save_locked(key, backend, [])
            self._write_json_locked("current.json", {"version": 1, "key": key})
            return self._view_locked(key, backend, size)

    def _current_key_locked(self) -> str:
        pointer = self.directory / "current.json"
        if pointer.exists():
            saved = json.loads(pointer.read_text())
            if not isinstance(saved, dict) or saved.get("version") != 1 or not isinstance(saved.get("key"), str):
                raise ValueError("Saved current conversation is invalid; it has been preserved for recovery.")
            return self.validate_key(saved["key"])
        candidates = []
        for path in self.directory.glob("*.json"):
            try:
                self.validate_key(path.stem)
            except ValueError:
                continue
            candidates.append(path)
        key = max(candidates, key=lambda path: path.stat().st_mtime_ns).stem if candidates else str(uuid4())
        self._write_json_locked("current.json", {"version": 1, "key": key})
        return key

    def _view_locked(self, key: str, backend: str, size: int) -> dict[str, Any]:
        path = self.directory / f"{key}.json"
        saved = json.loads(path.read_text()) if path.exists() else None
        if saved is not None and (
            not isinstance(saved, dict)
            or saved.get("version") != 1
            or not isinstance(saved.get("items"), list)
            or not isinstance(saved.get("backend"), str)
        ):
            raise ValueError("Saved conversation is invalid; it has been preserved for recovery.")
        reset = bool(saved and saved["backend"] != backend)
        if reset and key in self._active:
            raise ValueError("Disconnect the conversation before changing its backing LLM.")
        items = saved["items"] if saved and not reset else []
        chat = Chat.from_persistent_snapshot(items, size)
        self._save_locked(key, backend, chat.persistent_snapshot())
        persisted = json.loads(path.read_text())
        history = []
        for row in persisted.get("display_history", []):
            message = {"role": row["role"], "text": row["text"]}
            audio_id = persisted.get("item_audio", {}).get(row["id"])
            if audio_id:
                message["audio_url"] = self.audio_url(key, audio_id)
            history.append(message)
        return {"key": key, "backend": backend, "resumed": bool(items), "reset": reset, "history": history}

    def _save_locked(self, key: str, backend: str, items: list[dict[str, Any]]) -> None:
        content = json.dumps({"version": 1, "backend": backend, "items": items}, ensure_ascii=False)
        if self._saved.get(key) == content:
            return
        path = self.directory / f"{key}.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        history = previous.get("display_history", []) if previous.get("backend") == backend else []
        # Keep the readable transcript independently of the model's bounded
        # context. IDs let changed committed text update in place without
        # duplicating messages at every checkpoint.
        by_id = {row["id"]: row for row in history}
        for item in items:
            if item.get("type") != "message" or item.get("role") not in {"user", "assistant"}:
                continue
            text = "\n".join(part.get("text", "") for part in item.get("content", []) if isinstance(part, dict))
            if text and isinstance(item.get("id"), str):
                row = {"id": item["id"], "role": item["role"], "text": text}
                if row["id"] in by_id:
                    by_id[row["id"]].update(row)
                else:
                    history.append(row)
                    by_id[row["id"]] = row
        value = json.loads(content)
        value["display_history"] = history
        if previous.get("backend") == backend:
            for field in ("audio_files", "item_audio"):
                if field in previous:
                    value[field] = previous[field]
        self._write_json_locked(f"{key}.json", value)
        self._saved[key] = content

    def _write_json_locked(self, name: str, value: dict[str, Any]) -> None:
        fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=self.directory)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(value, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.directory / name)
            if os.name == "posix":
                directory_fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
