from queue import Queue
from threading import Event as ThreadingEvent

import pytest
from fastapi.testclient import TestClient
from openai.types.realtime import RealtimeSessionCreateRequest, SessionUpdateEvent

from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.settings_store import (
    MAX_INSTRUCTIONS_CHARS,
    SEED_AGENT_INSTRUCTIONS,
    SettingsStore,
    StaleSettings,
)
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.pipeline.cancel_scope import CancelScope

from .test_cloudflare_access import TUNNEL_ORIGIN, gate, token

LOCAL = "http://localhost:5173"


def make_unit() -> PipelineUnit:
    text_prompt_queue: Queue = Queue()
    should_listen = ThreadingEvent()
    should_listen.set()
    return PipelineUnit(
        index=0,
        service=RealtimeService(text_prompt_queue=text_prompt_queue, should_listen=should_listen),
        cancel_scope=CancelScope(),
        should_listen=should_listen,
        response_playing=ThreadingEvent(),
        input_queue=Queue(),
        output_queue=Queue(),
        text_output_queue=Queue(),
        text_prompt_queue=text_prompt_queue,
        handlers=[],
    )


@pytest.fixture
def unit() -> PipelineUnit:
    return make_unit()


@pytest.fixture
def client(unit, tmp_path) -> TestClient:
    app = create_app(
        pool=[unit], stop_event=ThreadingEvent(), conversation_store_dir=str(tmp_path), session_gate=gate()
    )
    return TestClient(app)


class TestStore:
    def test_seeds_on_first_run_and_persists_privately(self, tmp_path):
        store = SettingsStore(tmp_path / "s")
        assert store.agent_instructions == SEED_AGENT_INSTRUCTIONS
        assert (tmp_path / "s").stat().st_mode & 0o777 == 0o700
        assert (tmp_path / "s" / "settings.json").stat().st_mode & 0o777 == 0o600

    def test_a_saved_value_survives_restart(self, tmp_path):
        SettingsStore(tmp_path).set_agent_instructions("be brief", None)
        assert SettingsStore(tmp_path).agent_instructions == "be brief"

    def test_etag_mismatch_is_rejected_and_match_accepted(self, tmp_path):
        store = SettingsStore(tmp_path)
        etag = store.snapshot()["etag"]
        store.set_agent_instructions("one", etag)
        with pytest.raises(StaleSettings):
            store.set_agent_instructions("two", etag)
        assert store.agent_instructions == "one"

    def test_overlong_instructions_are_refused(self, tmp_path):
        with pytest.raises(ValueError):
            SettingsStore(tmp_path).set_agent_instructions("x" * (MAX_INSTRUCTIONS_CHARS + 1), None)

    def test_corrupt_file_is_not_overwritten(self, tmp_path):
        (tmp_path / "settings.json").write_text('{"nope": 1}')
        with pytest.raises(ValueError):
            SettingsStore(tmp_path)
        assert (tmp_path / "settings.json").read_text() == '{"nope": 1}'

    def test_memory_only_without_a_directory(self):
        store = SettingsStore(None)
        store.set_agent_instructions("hi", None)
        assert store.agent_instructions == "hi"


class TestEndpoints:
    def test_get_returns_seed_and_etag(self, client):
        body = client.get("/v1/settings").json()
        assert body["agent_instructions"] == SEED_AGENT_INSTRUCTIONS
        assert body["etag"]

    def test_put_saves_and_returns_a_new_etag(self, client):
        before = client.get("/v1/settings").json()["etag"]
        saved = client.put(
            "/v1/settings/agent-instructions", json={"agent_instructions": "new"}, headers={"If-Match": before}
        )
        assert saved.status_code == 200
        assert saved.headers["etag"] == saved.json()["etag"] != before
        assert client.get("/v1/settings").json()["agent_instructions"] == "new"

    def test_stale_put_gets_412_with_current_value(self, client):
        stale = client.get("/v1/settings").json()["etag"]
        client.put("/v1/settings/agent-instructions", json={"agent_instructions": "a"})
        late = client.put(
            "/v1/settings/agent-instructions", json={"agent_instructions": "b"}, headers={"If-Match": stale}
        )
        assert late.status_code == 412
        assert late.json()["agent_instructions"] == "a"

    @pytest.mark.parametrize("body", [{}, {"agent_instructions": 3}, [1]])
    def test_bad_bodies_are_422(self, client, body):
        assert client.put("/v1/settings/agent-instructions", json=body).status_code == 422

    def test_remote_origin_needs_access_token(self, client):
        headers = {"Origin": TUNNEL_ORIGIN}
        assert client.get("/v1/settings", headers=headers).status_code == 403
        put = client.put("/v1/settings/agent-instructions", json={"agent_instructions": "x"}, headers=headers)
        assert put.status_code == 403
        ok = client.get("/v1/settings", headers={**headers, "cf-access-jwt-assertion": token()})
        assert ok.status_code == 200

    def test_unlisted_origin_is_refused(self, client):
        assert client.get("/v1/settings", headers={"Origin": "https://evil.example"}).status_code == 403
        assert client.get("/v1/voices", headers={"Origin": "https://evil.example"}).status_code == 403


class TestPrecedence:
    def test_new_sessions_start_with_the_shared_value(self, unit, client):
        client.put("/v1/settings/agent-instructions", json={"agent_instructions": "shared"})
        conn = unit.service.register()
        assert unit.service._state(conn).runtime_config.session.instructions == "shared"

    def test_client_instructions_are_ignored(self, unit, client):
        client.put("/v1/settings/agent-instructions", json={"agent_instructions": "shared"})
        conn = unit.service.register()
        update = SessionUpdateEvent(
            type="session.update", session=RealtimeSessionCreateRequest(type="realtime", instructions="mine")
        )
        assert unit.service.handle_session_update(conn, update) is None
        assert unit.service._state(conn).runtime_config.session.instructions == "shared"

    def test_a_save_reaches_running_sessions(self, unit, client):
        conn = unit.service.register()
        client.put("/v1/settings/agent-instructions", json={"agent_instructions": "live"})
        assert unit.service._state(conn).runtime_config.session.instructions == "live"


class TestCors:
    def test_allowed_origin_is_echoed_with_credentials(self, client):
        for path in ("/v1/voices", "/v1/settings"):
            response = client.get(path, headers={"Origin": LOCAL})
            assert response.headers["access-control-allow-origin"] == LOCAL
            assert response.headers["access-control-allow-credentials"] == "true"
            assert response.headers["vary"] == "Origin"

    def test_no_origin_gets_no_cors_headers(self, client):
        assert "access-control-allow-origin" not in client.get("/v1/voices").headers

    def test_preflight_for_put_with_if_match(self, client):
        response = client.options(
            "/v1/settings/agent-instructions",
            headers={
                "Origin": LOCAL,
                "Access-Control-Request-Method": "PUT",
                "Access-Control-Request-Headers": "content-type, if-match",
            },
        )
        assert response.status_code == 204
        assert response.headers["access-control-allow-origin"] == LOCAL
        assert "PUT" in response.headers["access-control-allow-methods"]
        assert "If-Match" in response.headers["access-control-allow-headers"]

    def test_preflight_from_unlisted_origin_is_refused(self, client):
        for path in ("/v1/voices", "/v1/settings", "/v1/settings/agent-instructions"):
            response = client.options(path, headers={"Origin": "https://evil.example"})
            assert response.status_code == 403
            assert "access-control-allow-origin" not in response.headers

    def test_remote_preflight_needs_no_token_but_the_request_does(self, client):
        assert client.options("/v1/settings", headers={"Origin": TUNNEL_ORIGIN}).status_code == 204
        assert client.get("/v1/settings", headers={"Origin": TUNNEL_ORIGIN}).status_code == 403

    def test_expose_etag(self, client):
        assert client.get("/v1/settings", headers={"Origin": LOCAL}).headers["access-control-expose-headers"] == "ETag"
