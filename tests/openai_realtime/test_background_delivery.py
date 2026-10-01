"""Durable worker notifications stay independent of foreground voice turns."""

import pytest
from openai.types.realtime import ResponseCreateEvent

from speech_to_speech.agent_interactions import AgentPermissionReply, AgentPermissionRequested
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.events import AgentBackgroundEvent, AgentPermissionEvent, SpeechStartedEvent


def _completed(config, job_id="job-1"):
    return AgentBackgroundEvent(
        runtime_config=config,
        job_id=job_id,
        status="completed",
        description="Inspect the code",
        result="The adapter closes the SDK client at the end of each turn.",
    )


def test_completion_is_public_without_reopening_origin_response(service, conn_id, runtime_config, text_prompt_queue):
    st = service._state(conn_id)
    st.speculative_user_turn_id = "already-closed-turn"
    event = _completed(runtime_config)
    wire = service.dispatch_pipeline_event(conn_id, event)
    assert wire[0].type == "speech_to_speech.agent.background"
    assert wire[0].result == event.result
    assert not st.in_response
    created = service.maybe_start_background_delivery(conn_id)
    assert created.type == "response.created"
    request = text_prompt_queue.get_nowait()
    assert request.runtime_config is runtime_config
    assert request.turn_id is None and request.turn_revision is None
    assert request.response.metadata == {"background_job_id": event.job_id}
    assert event.result in request.response.instructions
    assert "Do not run tools" in request.response.instructions
    assert st.background_delivery_queue == [event.job_id]
    service.finish_response(conn_id)
    assert not st.background_delivery_queue
    assert service.dispatch_pipeline_event(conn_id, event) == []
    assert service.maybe_start_background_delivery(conn_id) is None


@pytest.mark.parametrize("busy", ["response", "pending", "speech", "queued_input", "permission"])
def test_result_waits_for_foreground_and_user_input(service, conn_id, runtime_config, text_prompt_queue, busy):
    st = service._state(conn_id)
    if busy == "response":
        service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        text_prompt_queue.get_nowait()
    elif busy == "pending":
        st.mark_response_pending("foreground")
    elif busy == "speech":
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id="speaking", turn_revision=0))
    elif busy == "queued_input":
        text_prompt_queue.put("foreground")
    else:
        runtime_config.agent_interactions.open("background:worker", "Bash", {"command": "ls"})
    status = service.dispatch_pipeline_event(conn_id, _completed(runtime_config))
    assert status[0].status == "completed"
    assert service.maybe_start_background_delivery(conn_id) is None
    assert st.background_delivery_queue == ["job-1"]


def test_interrupted_delivery_waits_for_next_turn_and_retries_once(service, conn_id, runtime_config, text_prompt_queue):
    service.dispatch_pipeline_event(conn_id, _completed(runtime_config))
    service.maybe_start_background_delivery(conn_id)
    first = text_prompt_queue.get_nowait()
    service.finish_response(conn_id, status="cancelled")
    assert service.maybe_start_background_delivery(conn_id) is None
    assert service._state(conn_id).background_delivery_queue == ["job-1"]
    service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
    text_prompt_queue.get_nowait()
    service.finish_response(conn_id)
    assert service.maybe_start_background_delivery(conn_id) is not None
    second = text_prompt_queue.get_nowait()
    assert second.response_key != first.response_key
    service.finish_response(conn_id)
    assert service.maybe_start_background_delivery(conn_id) is None


def test_completions_are_ordered_and_session_isolated(service, conn_id, runtime_config, text_prompt_queue):
    assert service.dispatch_pipeline_event(conn_id, _completed(RuntimeConfig(), "old-session")) == []
    for job_id in ("job-1", "job-2"):
        service.dispatch_pipeline_event(conn_id, _completed(runtime_config, job_id))
    for job_id in ("job-1", "job-2"):
        service.maybe_start_background_delivery(conn_id)
        assert text_prompt_queue.get_nowait().response.metadata == {"background_job_id": job_id}
        service.finish_response(conn_id)
    assert service.maybe_start_background_delivery(conn_id) is None


def test_background_permission_remains_answerable_after_foreground_closed(service, conn_id, runtime_config):
    broker = runtime_config.agent_interactions
    pending = broker.open("background:worker", "Bash", {"command": "ls"})
    event = AgentPermissionEvent(
        runtime_config=runtime_config,
        response_key=pending.response_key,
        event=AgentPermissionRequested(
            request_id=pending.request_id, tool_name="Bash", input=pending.input, timeout_s=30
        ),
    )
    assert service.dispatch_pipeline_event(conn_id, event)[0].response_id is None
    assert not service._state(conn_id).in_response
    reply = AgentPermissionReply(
        type="speech_to_speech.agent.permission.reply", request_id=pending.request_id, decision="allow"
    )
    assert service.handle_agent_permission_reply(conn_id, reply) is None
    assert broker.decision(pending)[0] == "allow"
    assert service.handle_agent_permission_reply(conn_id, reply).error.type == "stale_agent_request"


def test_background_permission_cannot_cross_session(service, conn_id, runtime_config):
    old_config = RuntimeConfig()
    pending = old_config.agent_interactions.open("background:worker", "Bash", {"command": "ls"})
    event = AgentPermissionEvent(
        runtime_config=old_config,
        response_key=pending.response_key,
        event=AgentPermissionRequested(
            request_id=pending.request_id, tool_name="Bash", input=pending.input, timeout_s=30
        ),
    )
    assert service.dispatch_pipeline_event(conn_id, event) == []
    reply = AgentPermissionReply(
        type="speech_to_speech.agent.permission.reply", request_id=pending.request_id, decision="allow"
    )
    assert service.handle_agent_permission_reply(conn_id, reply).error.type == "stale_agent_request"


@pytest.mark.parametrize("status", ["running", "cancelled", "failed"])
def test_job_status_without_result_does_not_make_empty_spoken_response(service, conn_id, runtime_config, status):
    event = AgentBackgroundEvent(runtime_config=runtime_config, job_id="job-1", status=status)
    assert service.dispatch_pipeline_event(conn_id, event)[0].status == status
    assert service.maybe_start_background_delivery(conn_id) is None


def test_failed_job_has_a_spoken_error_notification(service, conn_id, runtime_config, text_prompt_queue):
    event = _completed(runtime_config).model_copy(update={"status": "failed", "result": "The worker timed out."})
    service.dispatch_pipeline_event(conn_id, event)
    assert service.maybe_start_background_delivery(conn_id) is not None
    request = text_prompt_queue.get_nowait()
    assert "Status: failed" in request.response.instructions
    assert event.result in request.response.instructions


def test_foreground_permission_is_still_stale_after_turn_closes(service, conn_id, runtime_config):
    pending = runtime_config.agent_interactions.open("old-foreground", "Bash", {"command": "ls"})
    reply = AgentPermissionReply(
        type="speech_to_speech.agent.permission.reply", request_id=pending.request_id, decision="allow"
    )
    assert service.handle_agent_permission_reply(conn_id, reply).error.type == "stale_agent_request"
