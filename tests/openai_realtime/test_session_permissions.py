from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock

from starlette.testclient import TestClient

from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.pipeline.cancel_scope import CancelScope


def test_permission_control_routes_exact_app_session_and_rejects_invalid_or_cross_origin_requests():
    changed = AsyncMock(return_value={"session_id": "peer_one", "permission_mode": "acceptEdits"})
    peer = SimpleNamespace(set_permission_mode=changed)
    targets = []

    def get(session_id):
        targets.append(session_id)
        if session_id != "peer_one":
            raise ValueError("Unknown session")
        return peer

    registry = SimpleNamespace(get=get)
    unit = PipelineUnit(
        index=0,
        service=RealtimeService(),
        cancel_scope=CancelScope(),
        should_listen=Event(),
        response_playing=Event(),
        input_queue=Queue(),
        output_queue=Queue(),
        text_output_queue=Queue(),
        text_prompt_queue=Queue(),
        handlers=[SimpleNamespace(session_tools=True, _peer_sessions=registry)],
    )
    with TestClient(create_app(pool=[unit], stop_event=Event())) as client:
        url = "/v1/agent/sessions/peer_one/permission-mode"
        response = client.post(url, json={"mode": "acceptEdits"}, headers={"Origin": "http://testserver"})
        assert response.status_code == 200 and response.json()["permission_mode"] == "acceptEdits"
        changed.assert_awaited_once_with("acceptEdits")
        assert client.post(url, json={"mode": "bogus"}).status_code == 422
        assert client.post(url, json={"mode": "plan", "url": "http://other"}).status_code == 422
        assert client.post(url, json={"mode": "plan"}, headers={"Origin": "https://other.example"}).status_code == 403
        assert client.post(url, json={"mode": "plan"}, headers={"Origin": "null"}).status_code == 403
        assert changed.await_count == 1 and targets == ["peer_one"]
        assert client.post("/v1/agent/sessions/external/permission-mode", json={"mode": "plan"}).status_code == 404
        changed.side_effect = ValueError("This session has disconnected")
        assert client.post(url, json={"mode": "plan"}).status_code == 409
        changed.side_effect = RuntimeError("Policy refuses this mode")
        response = client.post(url, json={"mode": "bypassPermissions"})
        assert response.status_code == 502 and "Policy refuses" in response.json()["detail"]
        changed.side_effect = TimeoutError()
        assert client.post(url, json={"mode": "plan"}).status_code == 504
