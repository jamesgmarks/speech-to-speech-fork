import wave
from hashlib import sha256
from uuid import uuid4

import pytest
from openai.types.realtime import ResponseDoneEvent

from speech_to_speech.api.openai_realtime.conversation_store import ConversationStore
from speech_to_speech.api.openai_realtime.service import PIPELINE_SAMPLE_RATE, RealtimeService
from speech_to_speech.LLM.chat import make_assistant_message
from speech_to_speech.pipeline.events import AssistantOutputEvent


def connection(directory):
    store = ConversationStore(str(directory))
    key = store.current("claude-agent-sdk", 30)["key"]
    service = RealtimeService(llm_backend="claude-agent-sdk", chat_size=30)
    service.conversation_store = store
    conn = service.register(key)
    return service, conn, key


def speak(service, conn, text, pcm, response_key, status="completed"):
    chat = service._state(conn).runtime_config.chat
    chat.add_provisional_generation_items(response_key, [make_assistant_message(text)])
    service.dispatch_pipeline_event(conn, AssistantOutputEvent(text=text, response_key=response_key))
    service.encode_audio_chunk(conn, pcm[:10], response_key)
    service.encode_audio_chunk(conn, pcm[10:], response_key)
    events = service.finish_response(conn, status=status, response_key=response_key)
    return next(event.response.model_dump() for event in events if isinstance(event, ResponseDoneEvent))


def read_recording(store, url):
    key, filename = url.rsplit("/", 2)[-2:]
    path = store.audio_path(key, filename.removesuffix(".wav"))
    with wave.open(str(path)) as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getframerate() == PIPELINE_SAMPLE_RATE
        pcm = audio.readframes(audio.getnframes())
    assert path.stat().st_mode & 0o777 == 0o600
    return pcm


def test_original_audio_survives_server_restart_and_keeps_turn_identity(tmp_path):
    service, conn, key = connection(tmp_path)
    first = b"\x00\x01" * 200
    second = b"\x00\x02" * 300
    one = speak(service, conn, "Same words.", first, "one")
    two = speak(service, conn, "Same words.", second, "two")
    assert read_recording(service.conversation_store, one["speech_to_speech_audio_url"]) == first
    service.unregister(conn)
    fresh = ConversationStore(str(tmp_path))
    history = fresh.current("claude-agent-sdk", 30)["history"]
    assert len(history) == 2
    assert [read_recording(fresh, row["audio_url"]) for row in history] == [first, second]
    assert history[0]["audio_url"] != history[1]["audio_url"]
    assert two["speech_to_speech_audio_url"] == history[1]["audio_url"]
    # Checkpoints preserve audio metadata rather than overwriting it.
    chat, _, _ = fresh.acquire(key, "claude-agent-sdk", 30)
    fresh.save(key, "claude-agent-sdk", chat)
    assert fresh.current("claude-agent-sdk", 30)["history"] == history
    fresh.release(key)
    fresh.current("other-backend", 30)
    with pytest.raises(FileNotFoundError):
        fresh.audio_path(key, one["speech_to_speech_audio_url"].rsplit("/", 1)[1].removesuffix(".wav"))


def test_cancelled_response_keeps_generated_clip_without_committing_context(tmp_path):
    service, conn, _ = connection(tmp_path)
    pcm = b"\x01\x02" * 50
    response = speak(service, conn, "Interrupted speech.", pcm, "cancelled", "cancelled")
    assert read_recording(service.conversation_store, response["speech_to_speech_audio_url"]) == pcm
    assert service.conversation_store.current("claude-agent-sdk", 30)["history"] == []
    next_response = speak(service, conn, "Next reply.", b"\x03\x04" * 60, "next")
    assert read_recording(service.conversation_store, next_response["speech_to_speech_audio_url"]) == b"\x03\x04" * 60


def test_messages_around_tools_have_separate_replays_and_live_bubble_has_combined_audio(tmp_path):
    service, conn, _ = connection(tmp_path)
    state = service._state(conn)
    state.runtime_config.chat.add_provisional_generation_items(
        "tools", [make_assistant_message("Checking."), make_assistant_message("Finished.")]
    )
    service.dispatch_pipeline_event(conn, AssistantOutputEvent(text="Checking.", response_key="tools"))
    service.encode_audio_chunk(conn, b"\x01\x02" * 20, "tools")
    service.dispatch_pipeline_event(conn, AssistantOutputEvent(
        response_key="tools", text="", tools=[{
            "type": "function_call", "id": "fc_tool", "call_id": "call_tool", "name": "test", "arguments": "{}"
        }]
    ))
    service.dispatch_pipeline_event(conn, AssistantOutputEvent(text="Finished.", response_key="tools"))
    service.encode_audio_chunk(conn, b"\x03\x04" * 30, "tools")
    events = service.finish_response(conn, response_key="tools")
    done = next(event.response.model_dump() for event in events if isinstance(event, ResponseDoneEvent))
    store = service.conversation_store
    assert read_recording(store, done["speech_to_speech_audio_url"]) == b"\x01\x02" * 20 + b"\x03\x04" * 30
    assert [read_recording(store, row["audio_url"]) for row in store.current("claude-agent-sdk", 30)["history"]] == [
        b"\x01\x02" * 20, b"\x03\x04" * 30
    ]


def test_capture_is_bounded_and_invalid_or_unpublished_paths_are_rejected(tmp_path):
    service, conn, key = connection(tmp_path)
    state = service._state(conn)
    service.record_output_audio(conn, "item", b"\x00\x00" * 10)
    state.replay_audio_bytes = PIPELINE_SAMPLE_RATE * 2 * 300
    service.record_output_audio(conn, "item", b"\x00\x00")
    assert state.replay_audio_overflow and not state.replay_audio
    assert service.publish_response_audio(conn, "resp", "completed") is None
    store = service.conversation_store
    with pytest.raises(ValueError):
        store.audio_path("../../escape", "a" * 64)
    with pytest.raises(ValueError):
        store.audio_path(key, "../../escape")
    with pytest.raises(FileNotFoundError):
        store.audio_path(str(uuid4()), sha256(b"unknown").hexdigest())


async def test_webrtc_replay_uses_original_pipeline_audio_not_a_partial_media_recording(tmp_path):
    aiortc = pytest.importorskip("aiortc")
    from speech_to_speech.api.openai_realtime.webrtc_session import WebRTCSession

    service, conn, _ = connection(tmp_path)
    service._state(conn).runtime_config.chat.add_provisional_generation_items("rtc", [make_assistant_message("RTC reply.")])
    service.dispatch_pipeline_event(conn, AssistantOutputEvent(text="RTC reply.", response_key="rtc"))

    async def noop(*args):
        pass

    session = WebRTCSession(aiortc.RTCPeerConnection(), on_client_event=noop, on_audio=lambda pcm: None,
                            on_open=noop, on_closed=lambda: None)
    try:
        pcm = b"\x01\x02" * 1600
        await session.send_audio_chunk(service, conn, pcm[:1000], "rtc")
        await session.send_audio_chunk(service, conn, pcm[1000:], "rtc")
        events = service.finish_response(conn, response_key="rtc")
        done = next(event.response.model_dump() for event in events if isinstance(event, ResponseDoneEvent))
        assert read_recording(service.conversation_store, done["speech_to_speech_audio_url"]) == pcm
        assert session._track.buffered_bytes > 0  # RTP still has audio queued when response.done arrives.
    finally:
        await session.close()


async def test_audio_endpoint_serves_wav_ranges_but_rejects_unpublished_recordings(tmp_path):
    from threading import Event

    import httpx

    from speech_to_speech.api.openai_realtime.websocket_router import create_app

    service, conn, key = connection(tmp_path)
    done = speak(service, conn, "Replay.", b"\x01\x02" * 1600, "http")
    app = create_app([], Event(), conversation_store_dir=str(tmp_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        response = await http.get(done["speech_to_speech_audio_url"], headers={"Range": "bytes=0-3"})
        assert response.status_code == 206
        assert response.content == b"RIFF"
        assert response.headers["cache-control"] == "private, no-store"
        response = await http.get(f"/v1/conversation/audio/{key}/{'a' * 64}.wav")
        assert response.status_code == 404
