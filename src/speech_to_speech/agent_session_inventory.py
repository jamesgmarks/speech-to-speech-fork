"""Read-only, app-scoped inventory of managed and native Claude sessions."""

import asyncio
import json
import shutil
from time import monotonic
from typing import Any


def resolve_session_target(target: str, sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve exact IDs/names first, then unambiguous spoken name prefixes."""
    if not isinstance(target, str) or not target.strip() or len(target) > 4096 or "\0" in target:
        raise ValueError("Choose a session name or ID.")
    target = target.strip().casefold()

    def normalized(value: str) -> str:
        return "".join(c for c in value.casefold() if c.isalnum())

    exact = [
        s
        for s in sessions
        if target in {str(s.get(k, "")).casefold() for k in ("name", "session_id", "native_session_id")}
    ]
    matches = exact
    if not matches:
        key = normalized(target)
        matches = [s for s in sessions if normalized(str(s.get("name", ""))) == key]
        if not matches and len(key) >= 3:
            matches = [s for s in sessions if normalized(str(s.get("name", ""))).startswith(key)]
    if len(matches) == 1:
        return matches[0]
    if matches:
        choices = ", ".join(f"{s.get('name')} ({s.get('session_id')})" for s in matches)
        raise ValueError(f"Ambiguous session target. Choose one of: {choices}")
    raise ValueError("No session matches that name or ID. List sessions to choose an exact target.")


class AgentSessionInventory:
    def __init__(self, registry: Any = None):
        self.registry = registry
        self._lock = asyncio.Lock()
        self._cached: list[dict[str, Any]] = []
        self._error: str | None = None
        self._expires = 0.0

    async def list(self) -> dict[str, Any]:
        if self.registry is None:
            return {"sessions": [], "enabled": False, "discovery_error": None}
        async with self._lock:
            if monotonic() >= self._expires:
                self._cached = []
                self._error = None
                executable = shutil.which("claude")
                if executable is None:
                    self._error = "Claude CLI is unavailable; external session discovery is disabled."
                else:
                    process = None
                    try:
                        process = await asyncio.create_subprocess_exec(
                            executable,
                            "agents",
                            "--json",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        stdout, _ = await asyncio.wait_for(process.communicate(), 3)
                        if process.returncode or len(stdout) > 1048576:
                            raise ValueError("Claude CLI did not return a valid session inventory.")
                        entries = json.loads(stdout)
                        if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
                            raise ValueError("Claude CLI returned an unsupported session inventory.")
                        self._cached = entries
                    except (OSError, ValueError, TimeoutError) as exc:
                        self._error = str(exc) or "External session discovery timed out."
                    finally:
                        if process is not None and process.returncode is None:
                            try:
                                process.kill()
                            except ProcessLookupError:
                                pass
                            await process.wait()
                self._expires = monotonic() + 2
            sessions = [
                {**s, "source": "app", "provider": "claude", "working_with": True} for s in self.registry.list()
            ]
            native_ids = {s["native_session_id"] for s in sessions if s.get("native_session_id")}
            contacts = self.registry.contacts()
            for entry in self._cached:
                native_id = entry.get("sessionId") or entry.get("id")
                if not native_id or native_id in native_ids:
                    continue
                name = entry.get("name") or native_id
                sessions.append(
                    {
                        "session_id": native_id,
                        "native_session_id": native_id,
                        "name": name,
                        "directory": entry.get("cwd", ""),
                        "state": entry.get("status") or entry.get("state") or "unknown",
                        "kind": entry.get("kind", ""),
                        "source": "external",
                        "provider": "claude",
                        "working_with": native_id in contacts or name in contacts,
                        "latest_reply": "",
                        "queued_messages": 0,
                    }
                )
            return {"sessions": sessions, "enabled": True, "discovery_error": self._error}
