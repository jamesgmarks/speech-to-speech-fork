"""Independent root sessions retain history and receive asynchronous messages."""

import asyncio
from queue import Queue
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace

import pytest

pytest.importorskip("claude_agent_sdk")
from claude_agent_sdk import AssistantMessage, ResultMessage, SystemMessage, TextBlock

from speech_to_speech.LLM.claude_sessions import ClaudeSessionRegistry


def result(text, origin=None):
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="native-root-session",
        result=text,
        origin=origin,
    )


def eventually(predicate, timeout=3):
    start = monotonic()
    while not predicate() and monotonic() - start < timeout:
        sleep(0.01)
    assert predicate()


class Client:
    def __init__(self, gate=None):
        self.gate = gate
        self.prompts = []
        self.disconnected = False
        self.interrupted = False
        self.owner = None

    async def __aenter__(self):
        self.owner = asyncio.current_task()
        self.loop = asyncio.get_running_loop()
        self.messages = asyncio.Queue()
        await self.messages.put(SystemMessage(subtype="init", data={"session_id": "native-root-session"}))
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
        await self.messages.put(result(answer))

    async def receive_messages(self):
        while True:
            yield await self.messages.get()

    async def interrupt(self):
        self.interrupted = True

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
