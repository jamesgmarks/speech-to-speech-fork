"""Independent root sessions retain history and receive asynchronous messages."""

import asyncio
import json
from queue import Queue
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace
from uuid import uuid4

import pytest

pytest.importorskip("claude_agent_sdk")
from claude_agent_sdk import AssistantMessage, ResultMessage, SystemMessage, TextBlock

from speech_to_speech.LLM.claude_sessions import ClaudeSessionRegistry


def result(text, origin=None, session_id="native-root-session"):
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id=session_id,
        result=text,
        origin=origin,
    )


def eventually(predicate, timeout=3):
    start = monotonic()
    while not predicate() and monotonic() - start < timeout:
        sleep(0.01)
    assert predicate()


class Client:
    def __init__(self, gate=None, native_session_id="native-root-session"):
        self.native_session_id = native_session_id
        self.gate = gate
        self.prompts = []
        self.disconnected = False
        self.interrupted = False
        self.owner = None
        self.permission_modes = []
        self.permission_error = None

    async def __aenter__(self):
        self.owner = asyncio.current_task()
        self.loop = asyncio.get_running_loop()
        self.messages = asyncio.Queue()
        await self.messages.put(SystemMessage(subtype="init", data={"session_id": self.native_session_id}))
        return self

    async def __aexit__(self, *args):
        assert self.owner is asyncio.current_task()
        self.disconnected = True

    async def query(self, text):
        self.prompts.append(text)
        if self.gate is not None:
            while not self.gate.is_set():
                await asyncio.sleep(0.01)
        answer = f"Context: {' | '.join(self.prompts)}"
        await self.messages.put(AssistantMessage(content=[TextBlock(text=answer)], model="sonnet"))
        await self.messages.put(result(answer, session_id=self.native_session_id))

    async def receive_messages(self):
        while True:
            yield await self.messages.get()

    async def interrupt(self):
        self.interrupted = True

    async def set_permission_mode(self, mode):
        assert asyncio.get_running_loop() is self.loop
        if self.permission_error:
            raise RuntimeError(self.permission_error)
        self.permission_modes.append(mode)

    def inject_peer_reply(self, text):
        self.loop.call_soon_threadsafe(self.messages.put_nowait, result(text, {"kind": "peer", "from": "another-root"}))


@pytest.fixture
def registry():
    registry = ClaudeSessionRegistry()
    yield registry
    registry.shutdown()


def test_one_client_remembers_multiple_messages_and_native_peer_notices(registry):
    client = Client()
    peer = registry.create("research", "/tmp", lambda _: client, timeout=2)
    events = Queue()

    def sink(*args):
        events.put(args)

    recipient = SimpleNamespace(connection_active=True)
    peer.send("Remember violet", sink, recipient)
    assert events.get(timeout=2)[1] == "running"
    assert "Remember violet" in events.get(timeout=2)[3]
    assert not client.disconnected
    peer.send("What did I say?", sink, recipient)
    assert events.get(timeout=2)[1] == "running"
    assert "Remember violet | What did I say?" in events.get(timeout=2)[3]
    assert peer.snapshot()["native_session_id"] == "native-root-session"
    client.inject_peer_reply("External session reports the migration is complete.")
    assert "migration is complete" in events.get(timeout=2)[3]
    peer.stop()
    assert client.interrupted and client.disconnected


def test_busy_sessions_queue_messages_and_reconnect_without_losing_context(registry):
    gate = Event()
    client = Client(gate)
    peer = registry.create("long-task", "/tmp", lambda _: client, timeout=2)
    events = Queue()
    first = SimpleNamespace(connection_active=True)
    second = SimpleNamespace(connection_active=True)
    peer.send("first", lambda *args: events.put(args), first)
    eventually(lambda: bool(client.prompts))
    eventually(lambda: peer.native_session_id == "native-root-session")
    with pytest.raises(ValueError, match="another voice call"):
        peer.send("unrelated caller", lambda *args: None, second)
    first.connection_active = False
    peer.send("reconnected", lambda *args: events.put(args), second)
    gate.set()
    eventually(lambda: len(client.prompts) == 2 and peer.snapshot()["state"] == "idle")
    assert peer.snapshot()["latest_reply"] == "Context: first | reconnected"
    assert not client.disconnected


def test_session_limits_unknown_targets_and_stopping_queued_work(registry):
    gate = Event()
    client = Client(gate)
    peer = registry.create("only", "/tmp", lambda _: client, timeout=5, limit=1)
    with pytest.raises(ValueError, match="limit"):
        registry.create("other", "/tmp", lambda _: Client(), timeout=2, limit=1)
    with pytest.raises(ValueError, match="Unknown"):
        registry.get("an-external-cli-id")
    events = Queue()
    owner = SimpleNamespace(connection_active=True)
    peer.send("working", lambda *args: events.put(args), owner)
    eventually(lambda: bool(client.prompts))
    peer.send("queued", lambda *args: events.put(args), owner)
    peer.stop()
    assert peer.finished.is_set() and peer.snapshot()["state"] == "stopped"
    statuses = [events.get_nowait()[1] for _ in range(events.qsize())]
    assert statuses.count("cancelled") == 2
    with pytest.raises(ValueError, match="stopped"):
        peer.send("later", lambda *args: None, owner)


def test_permission_wait_does_not_consume_turn_deadline(registry):
    gate = Event()
    client = Client(gate)
    peer = registry.create("approval", "/tmp", lambda _: client, timeout=0.1)
    peer.waiting = lambda: True
    events = Queue()
    peer.send("approve", lambda *args: events.put(args), SimpleNamespace(connection_active=True))
    eventually(lambda: bool(client.prompts))
    sleep(0.2)
    assert not peer.finished.is_set()
    gate.set()
    eventually(lambda: peer.snapshot()["state"] == "idle")


def test_timeout_reports_failure_and_releases_the_client(registry):
    client = Client(Event())
    peer = registry.create("timeout", "/tmp", lambda _: client, timeout=0.1)
    events = Queue()
    peer.send("blocked", lambda *args: events.put(args), SimpleNamespace(connection_active=True))
    assert events.get(timeout=2)[1] == "running"
    assert "exceeded" in events.get(timeout=2)[3]
    eventually(lambda: peer.finished.is_set())
    assert client.disconnected


def test_stalled_startup_can_be_stopped(registry):
    class Stalled(Client):
        async def __aenter__(self):
            await asyncio.Event().wait()

    peer = registry.create("startup", "/tmp", lambda _: Stalled(), timeout=0.1)
    peer.stop()
    assert peer.finished.is_set()


async def test_live_mode_change_reaches_busy_client_without_interrupt_or_context_loss(registry):
    gate = Event()
    client = Client(gate)
    peer = registry.create("worker", "/tmp", lambda _: client, timeout=3, permission_mode="acceptEdits")
    other_client = Client()
    other = registry.create("other", "/tmp", lambda _: other_client, timeout=3)
    peer.send("Keep working", lambda *args: None, SimpleNamespace(connection_active=True))
    eventually(lambda: bool(client.prompts) and other.connected.is_set())
    changed = await peer.set_permission_mode("plan")
    assert changed["permission_mode"] == "plan" and changed["permission_control"]
    assert client.permission_modes == ["plan"]
    assert other.snapshot()["permission_mode"] == "default" and other_client.permission_modes == []
    assert not client.interrupted and not client.disconnected
    gate.set()
    eventually(lambda: peer.snapshot()["state"] == "idle")
    peer.send("Continue", lambda *args: None, peer.recipient)
    eventually(lambda: len(client.prompts) == 2)
    assert client.prompts == ["Keep working", "Continue"]


async def test_mode_is_only_changed_on_acknowledgement_and_stopped_sessions_reject_changes(registry):
    client = Client()
    peer = registry.create("worker", "/tmp", lambda _: client, timeout=2)
    eventually(lambda: peer.connected.is_set())
    client.permission_error = "Managed policy rejected this mode"
    with pytest.raises(RuntimeError, match="Managed policy"):
        await peer.set_permission_mode("bypassPermissions")
    assert peer.snapshot()["permission_mode"] == "default"
    with pytest.raises(ValueError, match="Unsupported"):
        await peer.set_permission_mode("bogus")
    peer.stop()
    with pytest.raises(ValueError, match="not connected"):
        await peer.set_permission_mode("acceptEdits")
    assert not peer.snapshot()["permission_control"]


async def test_service_restart_restores_session_identity_mode_and_native_resume_target(tmp_path):
    path = tmp_path / "sessions.json"
    registry = ClaudeSessionRegistry(str(path))
    native_id = str(uuid4())
    first = Client(native_session_id=native_id)
    peer = registry.create("worker", str(tmp_path), lambda _: first, timeout=3)
    try:
        eventually(lambda: peer.native_session_id == native_id)
        await peer.set_permission_mode("plan")
        registry.shutdown()
        assert path.stat().st_mode & 0o777 == 0o600
        saved = json.loads(path.read_text())["sessions"]
        assert saved[0]["session_id"] == peer.id and saved[0]["permission_mode"] == "plan"
        registry.shutdown()  # shared pipeline handlers may each clean up
        registry.checkpoint()  # a late worker notification must not erase it
        assert json.loads(path.read_text())["sessions"] == saved
        fresh = ClaudeSessionRegistry(str(path))
        restored_client = Client(native_session_id=native_id)
        targets = []

        def restore_factory(restored):
            targets.append((restored.id, restored.native_session_id, restored.permission_mode))
            return restored_client

        try:
            fresh.restore(restore_factory, timeout=3, limit=4)
            restored = fresh.get(peer.id)
            eventually(lambda: restored.connected.is_set())
            assert targets == [(peer.id, native_id, "plan")]
            assert restored.snapshot()["permission_control"]
            restored.stop()
            assert json.loads(path.read_text())["sessions"] == []
        finally:
            fresh.shutdown()
    finally:
        registry.shutdown()


def test_invalid_session_state_is_preserved_for_recovery(tmp_path):
    path = tmp_path / "sessions.json"
    path.write_text("invalid json")
    registry = ClaudeSessionRegistry(str(path))
    registry.shutdown()
    assert path.read_text() == "invalid json"
