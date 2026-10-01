"""App-owned independent Claude clients with persistent contexts and mailboxes.

These are root SDK connections, never native Agent children. Each client's
connect/disconnect is owned by one async task, while its continuous reader also
accepts native peer turns when no app message is active.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any
from uuid import uuid4

from claude_agent_sdk import AssistantMessage, ResultMessage, SystemMessage, TextBlock

Sink = Callable[[str, str, str, str], None]


@dataclass
class PeerMessage:
    message_id: str
    text: str
    sink: Sink


class ClaudePeerSession:
    def __init__(self, name: str, directory: str, factory: Callable[[ClaudePeerSession], Any], timeout: float):
        self.id = "peer_" + uuid4().hex
        self.name = name
        self.directory = directory
        self.native_session_id: str | None = None
        self.state = "starting"
        self.latest_reply = ""
        self.error = ""
        self.recipient: Any = None
        self.permission_responder: Any = None
        self.waiting: Callable[[], bool] = lambda: False
        self.closed = Event()
        self.finished = Event()
        self.connected = Event()
        self._queue: Queue[PeerMessage] = Queue(maxsize=8)
        self._lock = Lock()
        self._active: PeerMessage | None = None
        self._last_sink: Sink | None = None
        self._timeout = timeout
        self._factory = factory
        self.worker = Thread(target=self._run, name="claude-independent-session", daemon=True)

    def start(self) -> None:
        self.worker.start()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.id,
                "native_session_id": self.native_session_id,
                "name": self.name,
                "directory": self.directory,
                "state": "waiting_for_permission" if self.state == "working" and self.waiting() else self.state,
                "queued_messages": self._queue.qsize(),
                "latest_reply": self.latest_reply[-8192:],
                "error": self.error,
            }

    def send(self, text: str, sink: Sink, recipient: Any, *, permission_responder: Any = None) -> dict[str, str]:
        if not isinstance(text, str) or not text.strip() or "\0" in text or len(text) > 65536:
            raise ValueError("A message must contain 1–65536 characters without NUL bytes.")
        request = PeerMessage("peer_message_" + uuid4().hex, text, sink)
        with self._lock:
            if self.closed.is_set() or self.finished.is_set():
                raise ValueError("This agent session has stopped. Create a new session.")
            if (
                self._active is not None
                and recipient is not self.recipient
                and getattr(self.recipient, "connection_active", True)
            ):
                raise ValueError("This session is busy for another voice call. Wait until it is idle.")
            try:
                self._queue.put_nowait(request)
            except Full as exc:
                raise ValueError("This session already has eight queued messages.") from exc
            self.recipient = recipient
            if permission_responder is not None:
                self.permission_responder = permission_responder
            self._last_sink = sink
            sink(request.message_id, "running", f"{self.name}: queued message", "")
        return {"session_id": self.id, "message_id": request.message_id, "status": "queued"}

    def _complete(self, result: str, status: str = "completed", *, unsolicited: bool = False) -> None:
        with self._lock:
            request = None if unsolicited else self._active
            if not unsolicited:
                self._active = None
                self.state = "idle"
            elif self._active is None:
                self.state = "idle"
            self.latest_reply = result[-32768:]
            sink = request.sink if request is not None else self._last_sink
        if sink is not None and not self.closed.is_set():
            sink(
                request.message_id if request else "peer_notice_" + uuid4().hex,
                status,
                f"Agent session {self.name}",
                result,
            )

    async def _session(self) -> None:
        async with self._factory(self) as client:
            self.connected.set()
            with self._lock:
                self.state = "idle"

            async def receive() -> None:
                text = ""
                async for message in client.receive_messages():
                    if getattr(message, "parent_tool_use_id", None) is not None:
                        continue
                    if isinstance(message, SystemMessage) and message.subtype == "init":
                        native_id = message.data.get("session_id")
                        if isinstance(native_id, str):
                            self.native_session_id = native_id
                    elif isinstance(message, AssistantMessage):
                        with self._lock:
                            self.state = "working"
                        text = (
                            text + "\n" + "".join(b.text for b in message.content if isinstance(b, TextBlock))
                        ).strip()[-32768:]
                    elif isinstance(message, ResultMessage):
                        self.native_session_id = message.session_id
                        origin = getattr(message, "origin", None) or {}
                        unsolicited = origin.get("kind") not in (None, "human")
                        status = "failed" if message.is_error or message.subtype != "success" else "completed"
                        answer = message.result or text or "; ".join(message.errors or []) or f"Session turn {status}."
                        self._complete(answer, status, unsolicited=unsolicited)
                        text = ""
                if not self.closed.is_set():
                    raise RuntimeError("Independent Claude session disconnected unexpectedly.")

            receiver = asyncio.create_task(receive())
            deadline = monotonic() + self._timeout
            previous = monotonic()
            try:
                while not self.closed.is_set():
                    if receiver.done():
                        await receiver
                        return
                    now = monotonic()
                    if self.waiting():
                        deadline += now - previous
                    previous = now
                    with self._lock:
                        active = self._active
                    if active is None:
                        try:
                            message = self._queue.get_nowait()
                        except Empty:
                            await asyncio.sleep(0.05)
                            continue
                        with self._lock:
                            self._active = message
                            self.state = "working"
                        deadline = monotonic() + self._timeout
                        await client.query(message.text)
                    elif now >= deadline:
                        raise TimeoutError(f"Agent session turn exceeded {self._timeout:g} seconds.")
                    await asyncio.sleep(0.05)
            finally:
                try:
                    await asyncio.wait_for(client.interrupt(), 2)
                finally:
                    receiver.cancel()
                    await asyncio.gather(receiver, return_exceptions=True)

    def _run(self) -> None:
        async def run() -> None:
            owner = asyncio.create_task(self._session())
            start = monotonic()
            previous = start
            active_id: str | None = None
            deadline = start + self._timeout
            try:
                while not owner.done():
                    now = monotonic()
                    with self._lock:
                        current_id = self._active.message_id if self._active else None
                    if current_id != active_id:
                        active_id = current_id
                        deadline = now + self._timeout
                    elif self.waiting():
                        deadline += now - previous
                    previous = now
                    if active_id is not None and now >= deadline:
                        raise TimeoutError(f"Agent session turn exceeded {self._timeout:g} seconds.")
                    if not self.connected.is_set() and monotonic() - start >= min(30, self._timeout):
                        raise TimeoutError("Independent Claude session did not connect in time.")
                    if self.closed.is_set():
                        # Normal shutdown lets the owner interrupt/disconnect;
                        # startup or a stuck query is bounded as well.
                        try:
                            await asyncio.wait_for(asyncio.shield(owner), 3)
                        except TimeoutError:
                            owner.cancel()
                        return
                    await asyncio.wait({owner}, timeout=0.05)
                await owner
            finally:
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)

        try:
            asyncio.run(run())
        except Exception as exc:
            self.error = str(exc)
            self._complete(str(exc), "failed")
        finally:
            with self._lock:
                self.state = "stopped" if self.closed.is_set() else "failed"
            while True:
                try:
                    request = self._queue.get_nowait()
                except Empty:
                    break
                if not self.closed.is_set():
                    request.sink(
                        request.message_id, "failed", f"Agent session {self.name}", self.error or "Session stopped."
                    )
            self.finished.set()

    def stop(self) -> None:
        with self._lock:
            already_closed = self.closed.is_set()
            self.closed.set()
            active = self._active
        if already_closed:
            self.worker.join(timeout=4)
            return
        if active is not None:
            active.sink(active.message_id, "cancelled", f"Agent session {self.name}", "")
        while True:
            try:
                queued = self._queue.get_nowait()
            except Empty:
                break
            queued.sink(queued.message_id, "cancelled", f"Agent session {self.name}", "")
        self.worker.join(timeout=4)


class ClaudeSessionRegistry:
    """Shared by this app's pipeline units; no discovery of arbitrary CLI sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, ClaudePeerSession] = {}
        self._lock = Lock()
        self.external_contacts: set[str] = set()

    def find(self, name: str) -> ClaudePeerSession | None:
        with self._lock:
            return next(
                (
                    s
                    for s in self._sessions.values()
                    if s.name == name and not s.closed.is_set() and not s.finished.is_set()
                ),
                None,
            )

    def track_external(self, target: str) -> None:
        with self._lock:
            self.external_contacts.add(target)

    def contacts(self) -> set[str]:
        with self._lock:
            return set(self.external_contacts)

    def create(
        self, name: str, directory: str, factory: Callable[[ClaudePeerSession], Any], timeout: float, limit: int = 4
    ) -> ClaudePeerSession:
        if not isinstance(name, str) or not name.strip() or len(name) > 80 or any(ord(c) < 32 for c in name):
            raise ValueError("Use a session name of 1–80 characters without control characters.")
        with self._lock:
            active = [s for s in self._sessions.values() if not s.closed.is_set() and not s.finished.is_set()]
            if len(active) >= limit:
                raise ValueError("The independent agent session limit has been reached. Stop a session first.")
            if any(s.name == name for s in active):
                raise ValueError("An active agent session already uses that name.")
            session = ClaudePeerSession(name, directory, factory, timeout)
            self._sessions[session.id] = session
            session.start()
            return session

    def get(self, session_id: str) -> ClaudePeerSession:
        with self._lock:
            if session_id not in self._sessions:
                raise ValueError("Unknown app-owned agent session ID.")
            return self._sessions[session_id]

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [s.snapshot() for s in self._sessions.values()]

    def shutdown(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.stop()


app_claude_sessions = ClaudeSessionRegistry()
