from queue import Queue
from threading import Event as ThreadingEvent

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from speech_to_speech.api.openai_realtime.origin_policy import (
    ALLOWED_ORIGINS_ENV,
    DEFAULT_ALLOWED_ORIGINS,
    allowed_origins_from_env,
    normalize_origin,
    origin_allowed,
    parse_allowed_origins,
)
from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.pipeline.cancel_scope import CancelScope

DEFAULTS = parse_allowed_origins(None)


class TestPolicy:
    def test_defaults_cover_the_dashboard_the_backends_own_pages_and_the_tunnel_hostname(self):
        for origin in (
            "http://localhost:4317",
            "http://127.0.0.1:4317",
            "http://localhost:7860",
            "http://127.0.0.1:7860",
            "http://localhost:7861",
            "http://localhost:8765",
            "https://voice.onlyjapaneseconversation.com",
        ):
            assert origin_allowed(origin, DEFAULTS), origin
        assert DEFAULTS == frozenset(DEFAULT_ALLOWED_ORIGINS)

    def test_requests_without_an_origin_are_non_browser_clients_and_allowed(self):
        assert origin_allowed(None, DEFAULTS)

    @pytest.mark.parametrize(
        "origin",
        [
            "https://evil.example",
            "null",
            "",
            "http://localhost:9999",  # right host, wrong port
            "https://localhost:7860",  # right host and port, wrong scheme
            "http://voice.onlyjapaneseconversation.com",  # right host, wrong scheme
            "https://voice.onlyjapaneseconversation.com.evil.example",
            "https://evilvoice.onlyjapaneseconversation.com",
            "http://localhost:7860.evil.example",
        ],
    )
    def test_other_origins_are_refused(self, origin):
        assert not origin_allowed(origin, DEFAULTS)

    def test_matching_ignores_case_and_a_trailing_slash(self):
        assert origin_allowed("HTTP://LocalHost:7860/", DEFAULTS)
        assert (
            normalize_origin(" https://Voice.OnlyJapaneseConversation.com/path ")
            == "https://voice.onlyjapaneseconversation.com"
        )

    def test_a_configured_list_replaces_the_defaults(self):
        configured = parse_allowed_origins("https://dash.example.com, http://localhost:3000/ ,not-an-origin,")
        assert configured == frozenset({"https://dash.example.com", "http://localhost:3000"})
        assert origin_allowed("https://dash.example.com", configured)
        assert not origin_allowed("http://localhost:7860", configured)
        assert origin_allowed(None, configured)

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_unset_or_blank_keeps_the_defaults(self, value):
        assert parse_allowed_origins(value) == DEFAULTS

    def test_reads_the_environment(self, monkeypatch):
        monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://only.example")
        assert allowed_origins_from_env() == frozenset({"https://only.example"})
        monkeypatch.delenv(ALLOWED_ORIGINS_ENV)
        assert allowed_origins_from_env() == DEFAULTS


def make_app(**kwargs):
    text_prompt_queue: Queue = Queue()
    should_listen = ThreadingEvent()
    should_listen.set()
    unit = PipelineUnit(
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
    return create_app(pool=[unit], stop_event=ThreadingEvent(), **kwargs)


class TestRealtimeEndpoint:
    def test_an_allowed_origin_gets_a_session(self):
        with (
            TestClient(make_app()) as client,
            client.websocket_connect("/v1/realtime", headers={"origin": "http://localhost:4317"}) as ws,
        ):
            assert ws.receive_json()["type"] == "session.created"

    def test_a_client_with_no_origin_gets_a_session(self):
        with TestClient(make_app()) as client, client.websocket_connect("/v1/realtime") as ws:
            assert ws.receive_json()["type"] == "session.created"

    @pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://localhost:9999"])
    def test_a_browser_page_from_anywhere_else_is_refused_before_a_slot_is_claimed(self, origin):
        app = make_app()
        with TestClient(app) as client:
            with pytest.raises(WebSocketDisconnect) as refused:
                with client.websocket_connect("/v1/realtime", headers={"origin": origin}):
                    pass
            assert refused.value.code == 1008
            # The refused attempt left the only slot free.
            with client.websocket_connect("/v1/realtime") as ws:
                assert ws.receive_json()["type"] == "session.created"

    def test_the_allowlist_can_be_configured(self):
        app = make_app(allowed_origins=frozenset({"http://localhost:3000"}))
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime", headers={"origin": "http://localhost:3000"}) as ws:
                assert ws.receive_json()["type"] == "session.created"
            with (
                pytest.raises(WebSocketDisconnect),
                client.websocket_connect("/v1/realtime", headers={"origin": "http://localhost:7860"}),
            ):
                pass

    def test_the_environment_configures_the_default(self, monkeypatch):
        monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://dash.example.com")
        with (
            TestClient(make_app()) as client,
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect("/v1/realtime", headers={"origin": "http://localhost:4317"}),
        ):
            pass


class TestWebRtcEndpoint:
    def test_a_browser_page_from_anywhere_else_is_refused(self):
        with TestClient(make_app()) as client:
            response = client.post(
                "/v1/realtime/calls",
                content="v=0",
                headers={"origin": "https://evil.example", "content-type": "application/sdp"},
            )
        assert response.status_code == 403

    def test_an_allowed_origin_is_not_refused_on_origin_grounds(self):
        with TestClient(make_app()) as client:
            response = client.post(
                "/v1/realtime/calls",
                content="v=0",
                headers={"origin": "http://localhost:4317", "content-type": "text/plain"},
            )
        assert response.status_code != 403
