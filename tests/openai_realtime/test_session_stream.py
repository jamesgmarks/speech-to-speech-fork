"""The read-only session stream: levels, transcripts and tool calls mirrored to local observers."""

import asyncio
import base64
from queue import Queue
from threading import Event as ThreadingEvent
from types import SimpleNamespace

import numpy as np
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.session_stream import (
    LEVEL_FRAME_S,
    STREAM_VERSION,
    ObservedTransport,
    StreamHub,
    Subscription,
    frame_levels,
    is_local_observer,
    pcm16_level,
)
from speech_to_speech.api.openai_realtime.transports import SessionTransport
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import AssistantOutputEvent, AssistantResponseDoneEvent
from speech_to_speech.pipeline.messages import (
    AUDIO_RESPONSE_DONE,
    AssistantTextPart,
    AssistantToolCallPart,
    AudioOutput,
)


def tone(seconds: float, sample_rate: int, amplitude: float = 0.5) -> bytes:
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    return (np.sin(2 * np.pi * 220 * t) * amplitude * 32767).astype("<i2").tobytes()


def drain(subscription: Subscription) -> list[dict]:
    messages = []
    while not subscription.queue.empty():
        messages.append(subscription.queue.get_nowait())
    return messages


class TestLevels:
    def test_silence_is_zero_and_full_scale_is_one(self):
        assert pcm16_level(b"\x00\x00" * 100) == 0.0
        assert pcm16_level(b"") == 0.0
        assert pcm16_level(np.full(100, 32767, "<i2").tobytes()) == 1.0

    def test_louder_audio_reads_higher_and_speech_sits_mid_scale(self):
        quiet, normal, loud = (pcm16_level(tone(0.1, 16000, amplitude)) for amplitude in (0.01, 0.1, 0.8))
        assert 0.0 < quiet < normal < loud <= 1.0
        assert 0.4 < normal < 0.9

    def test_an_odd_trailing_byte_is_ignored(self):
        assert pcm16_level(tone(0.1, 16000) + b"\x01") == pcm16_level(tone(0.1, 16000))

    def test_frames_are_one_per_frame_interval(self):
        levels = frame_levels(tone(0.2, 16000), 16000)
        assert len(levels) == 4 and all(level > 0 for level in levels)
        assert len(frame_levels(tone(0.06, 16000), 16000)) == 2  # a short tail is its own frame


class TestObserverGate:
    @pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
    def test_a_local_non_browser_process_is_allowed(self, host):
        assert is_local_observer({}, host)

    @pytest.mark.parametrize(
        ("headers", "host"),
        [
            ({}, "192.168.1.20"),
            ({}, None),
            ({"origin": "http://localhost:5173"}, "127.0.0.1"),  # any web page, even a local one
            ({"origin": "null"}, "127.0.0.1"),
            ({"x-forwarded-for": "203.0.113.9"}, "127.0.0.1"),  # a reverse proxy on this machine
            ({"cf-connecting-ip": "203.0.113.9"}, "127.0.0.1"),  # a Cloudflare tunnel
            ({"forwarded": "for=203.0.113.9"}, "127.0.0.1"),
        ],
    )
    def test_everything_else_is_refused(self, headers, host):
        assert not is_local_observer(headers, host)


class TestSubscription:
    def test_a_full_queue_drops_new_levels_but_evicts_old_content_for_new_content(self, monkeypatch):
        monkeypatch.setattr("speech_to_speech.api.openai_realtime.session_stream.SUBSCRIBER_QUEUE_SIZE", 2)
        subscription = Subscription()
        subscription.offer({"type": "transcript", "n": 1})
        subscription.offer({"type": "transcript", "n": 2})
        subscription.offer({"type": "audio.level", "level": 0.5})
        assert [m["n"] for m in drain(subscription)] == [1, 2]
        subscription.offer({"type": "transcript", "n": 1})
        subscription.offer({"type": "transcript", "n": 2})
        subscription.offer({"type": "transcript", "n": 3})
        assert [m["n"] for m in drain(subscription)] == [2, 3]


def event(type_: str, **fields):
    return SimpleNamespace(type=type_, **fields)


class TestHub:
    def test_hello_lists_active_sessions(self):
        hub = StreamHub()
        assert hub.hello() == {"type": "stream.hello", "v": STREAM_VERSION, "active_sessions": []}
        hub.session_started("s1", "websocket")
        assert hub.hello()["active_sessions"] == ["s1"]

    def test_user_transcripts_grow_then_finish(self):
        hub = StreamHub()
        subscription = hub.subscribe()
        hub.observe_server_event(
            "s1", event("conversation.item.input_audio_transcription.delta", item_id="i1", delta="hel")
        )
        hub.observe_server_event(
            "s1", event("conversation.item.input_audio_transcription.delta", item_id="i1", delta="lo")
        )
        hub.observe_server_event(
            "s1", event("conversation.item.input_audio_transcription.completed", item_id="i1", transcript="Hello.")
        )
        assert [(m["text"], m["final"], m["role"]) for m in drain(subscription)] == [
            ("hel", False, "user"),
            ("hello", False, "user"),
            ("Hello.", True, "user"),
        ]

    def test_assistant_audio_and_text_transcripts(self):
        hub = StreamHub()
        subscription = hub.subscribe()
        hub.observe_server_event("s1", event("response.output_audio_transcript.delta", item_id="a1", delta="Hi "))
        hub.observe_server_event("s1", event("response.output_audio_transcript.delta", item_id="a1", delta="there"))
        hub.observe_server_event(
            "s1", event("response.output_audio_transcript.done", item_id="a1", transcript="Hi there")
        )
        hub.observe_server_event("s1", event("response.output_text.done", item_id="a2", text="Typed answer"))
        assert [(m["item_id"], m["text"], m["final"]) for m in drain(subscription)] == [
            ("a1", "Hi ", False),
            ("a1", "Hi there", False),
            ("a1", "Hi there", True),
            ("a2", "Typed answer", True),
        ]

    def test_tool_calls_keep_their_full_arguments_and_results_are_named(self):
        hub = StreamHub()
        subscription = hub.subscribe()
        arguments = '{"query": "' + "x" * 5000 + '"}'
        hub.observe_server_event(
            "s1", event("response.function_call_arguments.done", call_id="c1", name="search", arguments=arguments)
        )
        hub.observe_tool_result("s1", "c1", "three results")
        hub.observe_tool_result("s1", "unknown", "?")
        call, result, orphan = drain(subscription)
        assert call == {
            "type": "tool",
            "session_id": "s1",
            "phase": "call",
            "call_id": "c1",
            "name": "search",
            "arguments": arguments,
        }
        assert (result["phase"], result["name"], result["output"]) == ("result", "search", "three results")
        assert orphan["name"] == ""

    def test_speech_and_unrelated_events(self):
        hub = StreamHub()
        subscription = hub.subscribe()
        hub.observe_server_event("s1", event("input_audio_buffer.speech_started"))
        hub.observe_server_event("s1", event("input_audio_buffer.speech_stopped"))
        hub.observe_server_event("s1", event("response.created"))
        assert [m["active"] for m in drain(subscription)] == [True, False]

    def test_user_levels_are_throttled_to_one_peak_per_frame(self):
        now = [10.0]
        hub = StreamHub(clock=lambda: now[0])
        subscription = hub.subscribe()
        quiet, loud = tone(0.02, 24000, 0.05), tone(0.02, 24000, 0.8)
        hub.observe_user_audio("s1", quiet)  # first chunk goes straight out
        hub.observe_user_audio("s1", loud)
        hub.observe_user_audio("s1", quiet)  # within the same frame: held back, peak kept
        assert len(drain(subscription)) == 1
        now[0] += LEVEL_FRAME_S
        hub.observe_user_audio("s1", quiet)
        (level,) = drain(subscription)
        assert level["source"] == "user" and level["level"] == pcm16_level(loud)

    def test_nothing_is_computed_without_observers(self):
        hub = StreamHub()
        hub.observe_user_audio("s1", tone(0.02, 24000))
        assert hub._input_meters == {}

    def test_ending_a_session_zeroes_the_meters_and_forgets_its_state(self):
        hub = StreamHub()
        hub.session_started("s1", "websocket")
        subscription = hub.subscribe()
        hub.observe_server_event("s1", event("response.output_text.delta", item_id="a1", delta="x"))
        hub.observe_server_event(
            "s1", event("response.function_call_arguments.done", call_id="c1", name="t", arguments="{}")
        )
        drain(subscription)
        hub.session_ended("s1")
        assert [(m["type"], m.get("source"), m.get("level")) for m in drain(subscription)] == [
            ("audio.level", "user", 0.0),
            ("audio.level", "assistant", 0.0),
            ("session.ended", None, None),
        ]
        assert hub._transcripts == {} and hub._tool_names == {} and hub.active_sessions == []
        hub.session_ended("s1")  # idempotent: no second "ended"
        assert drain(subscription) == []


class FakeTransport(SessionTransport):
    kind = "websocket"

    def __init__(self):
        self.events, self.chunks, self.discarded = [], [], 0

    async def send_events(self, events):
        self.events.extend(events)

    async def send_audio_chunk(self, service, session_id, pcm, response_key=None):
        self.chunks.append(pcm)

    def discard_pending_audio(self):
        self.discarded += 1

    async def close(self):
        pass


class TestObservedTransport:
    async def test_passes_everything_through_unchanged(self):
        inner, hub = FakeTransport(), StreamHub()
        transport = ObservedTransport(inner, hub)
        transport.start("s1")
        marker = event("response.created")
        await transport.send_events([marker])
        await transport.send_audio_chunk(None, "s1", b"\x00\x00", None)
        transport.discard_pending_audio()
        assert (inner.events, inner.chunks, inner.discarded, transport.kind) == (
            [marker],
            [b"\x00\x00"],
            1,
            "websocket",
        )

    async def test_a_mirroring_failure_never_breaks_the_session(self, monkeypatch):
        inner, hub = FakeTransport(), StreamHub()
        transport = ObservedTransport(inner, hub)
        transport.start("s1")
        monkeypatch.setattr(hub, "observe_server_event", lambda *_: 1 / 0)
        await transport.send_events([event("response.created")])
        assert len(inner.events) == 1

    async def test_assistant_levels_follow_playback_time_not_generation_time(self):
        hub = StreamHub()
        subscription = hub.subscribe()
        transport = ObservedTransport(FakeTransport(), hub)
        transport.start("s1")
        drain(subscription)
        audio = tone(0.1, 16000)  # TTS hands over two 100 ms chunks at once
        await transport.send_audio_chunk(None, "s1", audio)
        await transport.send_audio_chunk(None, "s1", audio)
        await asyncio.sleep(0.02)
        assert len(drain(subscription)) <= 1  # not all four frames at once
        await asyncio.sleep(0.45)
        levels = [m["level"] for m in drain(subscription) if m["source"] == "assistant"]
        # frames 1..4 were played in order, then silence once, since the first chunk's own "done" marker was superseded
        assert levels[-1] == 0.0 and sum(1 for level in levels if level == 0.0) == 1 and len(levels) >= 3
        transport.finish()

    async def test_barge_in_silences_the_meter_and_cancels_pending_frames(self):
        hub = StreamHub()
        subscription = hub.subscribe()
        transport = ObservedTransport(FakeTransport(), hub)
        transport.start("s1")
        await transport.send_audio_chunk(None, "s1", tone(1.0, 16000))
        await asyncio.sleep(0.02)
        drain(subscription)
        transport.discard_pending_audio()
        assert [m["level"] for m in drain(subscription)] == [0.0]
        await asyncio.sleep(0.2)
        assert drain(subscription) == []  # the rest of that second never plays
        transport.finish()

    async def test_does_no_level_work_when_nobody_is_listening(self):
        transport = ObservedTransport(FakeTransport(), StreamHub())
        transport.start("s1")
        await transport.send_audio_chunk(None, "s1", tone(0.5, 16000))
        assert transport._level_timers == set()


@pytest.fixture
def setup():
    text_prompt_queue: Queue = Queue()
    should_listen = ThreadingEvent()
    should_listen.set()
    service = RealtimeService(text_prompt_queue=text_prompt_queue, should_listen=should_listen)
    output_queue: Queue = Queue()
    unit = PipelineUnit(
        index=0,
        service=service,
        cancel_scope=CancelScope(),
        should_listen=should_listen,
        response_playing=ThreadingEvent(),
        input_queue=Queue(),
        output_queue=output_queue,
        text_output_queue=Queue(),
        text_prompt_queue=text_prompt_queue,
        handlers=[],
    )
    return create_app(pool=[unit], stop_event=ThreadingEvent()), output_queue


def local_client(app) -> TestClient:
    return TestClient(app, client=("127.0.0.1", 50000))


class TestStreamEndpoint:
    def test_mirrors_a_live_session_end_to_end(self, setup):
        app, output_queue = setup
        with local_client(app) as client, client.websocket_connect("/v1/stream") as observer:
            assert observer.receive_json() == {"type": "stream.hello", "v": STREAM_VERSION, "active_sessions": []}
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                ws.send_json(
                    {"type": "input_audio_buffer.append", "audio": base64.b64encode(tone(0.1, 24000, 0.7)).decode()}
                )

                key = "response_1"
                tool = {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": '{"city": "Toronto"}'}
                output_queue.put(AssistantOutputEvent(response_key=key, parts=[AssistantTextPart(text="Checking.")]))
                output_queue.put(AssistantOutputEvent(response_key=key, parts=[AssistantToolCallPart(tool=tool)]))
                output_queue.put(AssistantResponseDoneEvent(response_key=key))
                output_queue.put(AudioOutput(audio=tone(0.1, 16000, 0.6), response_key=key))
                output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, response_key=key))
                while ws.receive_json()["type"] != "response.done":
                    pass
                ws.send_json(
                    {
                        "type": "conversation.item.create",
                        "item": {"type": "function_call_output", "call_id": "c1", "output": "12 degrees"},
                    }
                )

            messages = []
            while not messages or messages[-1]["type"] != "session.ended":
                messages.append(observer.receive_json())

        session_ids = {m["session_id"] for m in messages}
        assert len(session_ids) == 1
        kinds = [m["type"] for m in messages]
        assert kinds[0] == "session.started" and messages[0]["transport"] == "websocket"
        levels = {(m["source"], m["level"] > 0) for m in messages if m["type"] == "audio.level"}
        assert {("user", True), ("assistant", True)} <= levels
        finals = [m["text"] for m in messages if m["type"] == "transcript" and m["final"]]
        assert finals == ["Checking."]
        tools = [m for m in messages if m["type"] == "tool"]
        assert [(t["phase"], t["name"]) for t in tools] == [("call", "lookup"), ("result", "lookup")]
        assert tools[0]["arguments"] == '{"city": "Toronto"}' and tools[1]["output"] == "12 degrees"

    def test_a_late_observer_is_told_which_sessions_are_live(self, setup):
        app, _ = setup
        with local_client(app) as client, client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            with client.websocket_connect("/v1/stream") as observer:
                assert len(observer.receive_json()["active_sessions"]) == 1

    def test_the_observer_cannot_talk_to_the_session(self, setup):
        app, _ = setup
        with local_client(app) as client, client.websocket_connect("/v1/stream") as observer:
            observer.receive_json()
            observer.send_text('{"type": "response.create"}')
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                observer.receive_json()  # session.started: the session is idle, not triggered to respond

    @pytest.mark.parametrize(
        "headers",
        [
            {"Origin": "http://localhost:5173"},
            {"Origin": "https://evil.example"},
            {"X-Forwarded-For": "203.0.113.9"},
            {"CF-Connecting-IP": "203.0.113.9"},
        ],
    )
    def test_browsers_and_proxied_requests_are_refused(self, setup, headers):
        app, _ = setup
        with local_client(app) as client, pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/v1/stream", headers=headers):
                pass

    def test_a_remote_client_is_refused(self, setup):
        app, _ = setup
        with TestClient(app, client=("203.0.113.9", 50000)) as client, pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/v1/stream"):
                pass
