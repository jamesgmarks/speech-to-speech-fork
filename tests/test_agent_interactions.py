from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from speech_to_speech.agent_interactions import (
    AgentInteractions,
    AgentPermissionReply,
    AgentPermissionRequested,
)
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.events import AgentPermissionEvent, TranscriptionCompletedEvent
from speech_to_speech.pipeline.messages import Transcription
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.VAD.vad_handler import VADHandler


@pytest.mark.parametrize(
    "phrase,decision",
    [
        ("Approve request.", "allow"),
        ("Allow this request!", "allow"),
        ("deny request", "deny"),
        ("Reject the request", "deny"),
    ],
)
def test_explicit_voice_decisions(phrase, decision):
    broker = AgentInteractions()
    request = broker.open("response", "Bash", {"command": "echo hi"})
    assert broker.respond_voice(request.request_id, phrase) is None
    assert broker.decision(request)[0] == decision


@pytest.mark.parametrize(
    "phrase", ["yes", "okay", "do not approve request", "I said approve request yesterday", "maybe"]
)
def test_ambiguous_or_negated_speech_does_not_approve(phrase):
    broker = AgentInteractions()
    request = broker.open("response", "Bash", {})
    assert broker.respond_voice(request.request_id, phrase)
    assert broker.decision(request)[0] is None


def test_stale_and_duplicate_replies_cannot_approve_new_requests():
    broker = AgentInteractions()
    first = broker.open("first", "Bash", {})
    broker.close(first)
    second = broker.open("second", "Bash", {})
    assert broker.respond(first.request_id, "allow")
    assert broker.decision(second)[0] is None
    assert broker.respond(second.request_id, "deny") is None
    assert broker.respond(second.request_id, "allow")
    assert broker.decision(second)[0] == "deny"


def test_single_question_can_receive_free_text_by_voice():
    broker = AgentInteractions()
    request = broker.open("r", "AskUserQuestion", {"questions": [{"question": "Which format?"}]})
    assert broker.respond_voice(request.request_id, "Use a short summary") is None
    assert broker.decision(request)[1] == {"Which format?": "Use a short summary"}


def test_multiple_questions_require_complete_explicit_answers():
    broker = AgentInteractions()
    request = broker.open(
        "r", "AskUserQuestion", {"questions": [{"question": "Which format?"}, {"question": "Which scope?"}]}
    )
    assert broker.respond(request.request_id, "allow", {"Which format?": "Summary"})
    assert broker.respond_voice(request.request_id, "Summary")
    assert broker.respond(request.request_id, "allow", {"Which format?": "Summary", "Which scope?": ["A", "B"]}) is None


def test_simultaneous_requests_have_no_implicit_voice_target():
    broker = AgentInteractions()
    first = broker.open("r", "Bash", {})
    assert broker.voice_target() == first.request_id
    broker.open("r", "Write", {})
    assert broker.voice_target() is None


def test_permission_voice_stays_out_of_chat_and_does_not_queue_a_new_response():
    queue = Queue()
    tracker = SpeculativeTurnTracker()
    turn, revision = tracker.start_turn()
    service = RealtimeService(text_prompt_queue=queue, speculative_turns=tracker)
    session = service.register()
    state = service._state(session)
    state.in_response = True
    state.current_response_key = "r"
    request = state.runtime_config.agent_interactions.open("r", "Bash", {})
    before = state.runtime_config.chat.to_transformers_chat()
    events = service.dispatch_pipeline_event(
        session, TranscriptionCompletedEvent(transcript="approve request", agent_request_id=request.request_id)
    )
    assert events[0].error is None
    assert queue.empty()
    assert state.in_response
    assert state.current_response_key == "r"
    assert tracker.is_latest(turn, revision)
    assert state.runtime_config.chat.to_transformers_chat() == before


def test_request_before_first_text_opens_implicit_response_and_allows_reply():
    service = RealtimeService()
    session = service.register()
    state = service._state(session)
    state.mark_response_pending("r")
    request = state.runtime_config.agent_interactions.open("r", "Write", {"file_path": "file.py"})
    events = service.dispatch_pipeline_event(
        session,
        AgentPermissionEvent(
            response_key="r",
            event=AgentPermissionRequested(
                request_id=request.request_id, tool_name="Write", input=request.input, timeout_s=300
            ),
        ),
    )
    assert [e.type for e in events] == ["response.created", "speech_to_speech.agent.permission.requested"]
    assert events[1].response_id == state.current_response_id
    assert (
        service.handle_agent_permission_reply(
            session,
            AgentPermissionReply(
                type="speech_to_speech.agent.permission.reply", request_id=request.request_id, decision="allow"
            ),
        )
        is None
    )


def test_replies_are_scoped_to_the_connected_session():
    service = RealtimeService()
    first, second = service.register(), service.register()
    state = service._state(first)
    state.in_response = True
    state.current_response_key = "r"
    request = state.runtime_config.agent_interactions.open("r", "Bash", {})
    reply = AgentPermissionReply(
        type="speech_to_speech.agent.permission.reply", request_id=request.request_id, decision="allow"
    )
    assert service.handle_agent_permission_reply(second, reply)
    assert state.runtime_config.agent_interactions.decision(request)[0] is None
    service.unregister(first)
    assert state.runtime_config.agent_interactions.decision(request)[0] == "deny"


@pytest.mark.parametrize("streaming", [False, True])
def test_permission_audio_bypasses_listening_gate_and_speculative_turn_creation(streaming):
    config = RuntimeConfig()
    request = config.agent_interactions.open("r", "Bash", {})
    vad = VADHandler.__new__(VADHandler)
    vad.should_listen = Event()  # Normal response speech has closed this gate.
    vad.sample_rate = 16000
    vad._total_samples = 0
    vad._apply_runtime_turn_detection = lambda cfg: None
    vad.iterator = SimpleNamespace(reset_states=lambda: None)

    class Iterator:
        min_silence_samples = 1024

        def reset_states(self):
            pass

        def __call__(self, audio):
            return [torch.ones(4096)]

    vad.iterator = Iterator()
    commands = []
    if streaming:
        vad.streaming_stt_sink = SimpleNamespace(
            discard_utterance=lambda: commands.append("discard"),
            start_turn=lambda *args: commands.append(("start", args)),
            append_audio=lambda pcm: commands.append(("audio", pcm)),
            commit_boundary=lambda *args: commands.append(("commit", args)),
        )
    tracker = SpeculativeTurnTracker()
    turn, revision = tracker.start_turn()
    vad.speculative_turns = tracker
    outputs = list(vad.process((np.ones(512, dtype=np.int16).tobytes(), config)))
    assert outputs[0].agent_request_id == request.request_id
    assert outputs[0].turn_id is None
    assert tracker.is_latest(turn, revision)
    handler = BaseHandler(Event(), Queue(), Queue())
    transcript = handler.output_for_queue(Transcription(text="approve request"), outputs[0])
    assert transcript.agent_request_id == request.request_id

    assert vad.iterator.min_silence_samples == 6400
    if streaming:
        assert commands[0] == "discard"
        assert commands[1] == ("start", (None, None))
        assert len(commands[2][1]) == 4096 * 2
        assert commands[3] == ("commit", (None, None))
    config.agent_interactions.respond(request.request_id, "deny")
    assert list(vad.process((bytes(1024), config))) == []
    assert vad.iterator.min_silence_samples == 1024
    assert tracker.is_latest(turn, revision)
