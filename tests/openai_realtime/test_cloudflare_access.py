import time
from queue import Queue
from threading import Event as ThreadingEvent

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from speech_to_speech.api.openai_realtime.cloudflare_access import (
    AccessDenied,
    CloudflareAccessConfig,
    CloudflareAccessVerifier,
    SessionGate,
    access_issuer,
    extract_access_token,
    is_local_origin,
)
from speech_to_speech.api.openai_realtime.origin_policy import DEFAULT_ALLOWED_ORIGINS
from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.pipeline.cancel_scope import CancelScope

TEAM = "personal"
AUD = "aud-tag-1"
ISSUER = f"https://{TEAM}.cloudflareaccess.com"
TUNNEL_ORIGIN = "https://voice.onlyjapaneseconversation.com"
ALLOWED = frozenset(DEFAULT_ALLOWED_ORIGINS)

SIGNING_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def token(key=SIGNING_KEY, algorithm="RS256", **claims) -> str:
    now = int(time.time())
    payload = {"iss": ISSUER, "aud": [AUD], "iat": now, "exp": now + 300, "email": "james@example.com", **claims}
    return jwt.encode({k: v for k, v in payload.items() if v is not None}, key, algorithm=algorithm)


def verifier() -> CloudflareAccessVerifier:
    return CloudflareAccessVerifier(TEAM, frozenset({AUD}), key_resolver=lambda _token: SIGNING_KEY.public_key())


class TestVerifier:
    def test_accepts_a_valid_token_and_returns_its_claims(self):
        assert verifier().verify(token())["email"] == "james@example.com"

    def test_accepts_a_service_token_jwt(self):
        claims = verifier().verify(token(email=None, common_name="client-id.access"))
        assert claims["common_name"] == "client-id.access"

    @pytest.mark.parametrize(
        "bad",
        [
            token(exp=int(time.time()) - 3600),
            token(aud=["some-other-app"]),
            token(iss="https://other.cloudflareaccess.com"),
            token(key=OTHER_KEY),
            token(exp=None),
            token(aud=None),
            "not-a-jwt",
            jwt.encode(
                {"iss": ISSUER, "aud": [AUD], "exp": int(time.time()) + 300}, key="secret" * 8, algorithm="HS256"
            ),
        ],
        ids=["expired", "wrong-audience", "wrong-issuer", "wrong-signature", "no-exp", "no-aud", "garbage", "hs256"],
    )
    def test_rejects_anything_else(self, bad):
        with pytest.raises(AccessDenied):
            verifier().verify(bad)

    def test_rejects_an_unsigned_token(self):
        unsigned = jwt.encode({"iss": ISSUER, "aud": [AUD], "exp": int(time.time()) + 300}, key=None, algorithm="none")
        with pytest.raises(AccessDenied):
            verifier().verify(unsigned)

    def test_fails_closed_when_the_keys_cannot_be_fetched(self):
        def unreachable(_token):
            raise OSError("no route to host")

        with pytest.raises(AccessDenied, match="could not be verified"):
            CloudflareAccessVerifier(TEAM, frozenset({AUD}), key_resolver=unreachable).verify(token())


class TestTeamDomain:
    @pytest.mark.parametrize(
        "team", ["personal", "personal.cloudflareaccess.com", "https://Personal.cloudflareaccess.com/"]
    )
    def test_the_team_can_be_a_name_or_its_full_domain(self, team):
        assert access_issuer(team) == ISSUER

    def test_a_custom_team_domain_is_kept(self):
        assert access_issuer("access.example.com") == "https://access.example.com"


class TestConfig:
    def test_required_by_default_and_read_from_the_environment(self):
        assert CloudflareAccessConfig.from_env({}) == CloudflareAccessConfig(required=True)
        config = CloudflareAccessConfig.from_env({"S2S_CF_ACCESS_TEAM": TEAM, "S2S_CF_ACCESS_AUD": "a, b ,"})
        assert config.team == TEAM and config.audiences == {"a", "b"} and config.configured

    @pytest.mark.parametrize("value", ["false", "0", "No", "off"])
    def test_can_be_switched_off(self, value):
        assert not CloudflareAccessConfig.from_env({"S2S_CF_ACCESS_REQUIRED": value}).required


class TestTokenSources:
    def test_the_edge_header_wins_and_the_cookie_is_the_fallback(self):
        assert extract_access_token({"cf-access-jwt-assertion": " abc "}) == "abc"
        assert extract_access_token({"cookie": "theme=dark; CF_Authorization=xyz; other=1"}) == "xyz"
        assert extract_access_token({"cf-access-jwt-assertion": "h", "cookie": "CF_Authorization=c"}) == "h"
        assert extract_access_token({"cookie": "theme=dark"}) is None
        assert extract_access_token({}) is None

    @pytest.mark.parametrize(
        ("origin", "local"),
        [
            ("http://localhost:5173", True),
            ("http://127.0.0.1:4317", True),
            ("http://LOCALHOST:7860", True),
            (TUNNEL_ORIGIN, False),
            ("http://localhost.evil.example", False),
            ("http://127.0.0.1.evil.example", False),
        ],
    )
    def test_only_localhost_and_loopback_origins_are_local(self, origin, local):
        assert is_local_origin(origin) is local


def gate(**overrides) -> SessionGate:
    return SessionGate(
        ALLOWED,
        overrides.pop("access", CloudflareAccessConfig(required=True, team=TEAM, audiences=frozenset({AUD}))),
        overrides.pop("verifier", verifier()),
    )


class TestGate:
    async def test_no_origin_is_a_non_browser_client(self):
        assert await gate().denial({}) is None

    async def test_local_origins_need_no_token(self):
        for origin in ("http://localhost:5173", "http://127.0.0.1:4317"):
            assert await gate().denial({"origin": origin}) is None

    async def test_a_remote_origin_needs_a_valid_token(self):
        assert await gate().denial({"origin": TUNNEL_ORIGIN}) == "Cloudflare Access token required"
        assert await gate().denial({"origin": TUNNEL_ORIGIN, "cf-access-jwt-assertion": token()}) is None
        assert await gate().denial({"origin": TUNNEL_ORIGIN, "cookie": f"CF_Authorization={token()}"}) is None
        bad = await gate().denial({"origin": TUNNEL_ORIGIN, "cf-access-jwt-assertion": token(key=OTHER_KEY)})
        assert bad == "Cloudflare Access token invalid"

    async def test_anything_through_the_tunnel_needs_a_token_even_without_a_remote_origin(self):
        for extra in ({"cf-connecting-ip": "203.0.113.9"}, {"cf-ray": "abc"}, {"x-forwarded-for": "203.0.113.9"}):
            assert await gate().denial(extra) == "Cloudflare Access token required"  # a tool: no Origin
            forged = {"origin": "http://localhost:5173", **extra}  # a client claiming to be local
            assert await gate().denial(forged) == "Cloudflare Access token required"
            assert await gate().denial({**forged, "cf-access-jwt-assertion": token()}) is None

    async def test_a_valid_token_does_not_make_an_unlisted_origin_allowed(self):
        assert (
            await gate().denial({"origin": "https://evil.example", "cf-access-jwt-assertion": token()})
            == "Origin not allowed"
        )

    async def test_fails_closed_when_required_but_not_configured(self):
        unconfigured = SessionGate(ALLOWED, CloudflareAccessConfig(required=True))
        assert (
            await unconfigured.denial({"origin": TUNNEL_ORIGIN}) == "Cloudflare Access isn't configured on the backend"
        )
        assert await unconfigured.denial({"origin": "http://localhost:5173"}) is None

    async def test_the_token_check_can_be_switched_off(self):
        relaxed = SessionGate(ALLOWED, CloudflareAccessConfig(required=False))
        assert await relaxed.denial({"origin": TUNNEL_ORIGIN}) is None
        assert await relaxed.denial({"origin": "https://evil.example"}) == "Origin not allowed"


def make_app(session_gate: SessionGate):
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
    return create_app(pool=[unit], stop_event=ThreadingEvent(), session_gate=session_gate)


class TestEndpoints:
    def test_the_tunnel_origin_gets_a_session_only_with_a_valid_token(self):
        with TestClient(make_app(gate())) as client:
            with (
                pytest.raises(WebSocketDisconnect) as refused,
                client.websocket_connect("/v1/realtime", headers={"origin": TUNNEL_ORIGIN}),
            ):
                pass
            assert refused.value.reason == "Cloudflare Access token required"
            with client.websocket_connect(
                "/v1/realtime", headers={"origin": TUNNEL_ORIGIN, "cf-access-jwt-assertion": token()}
            ) as ws:
                assert ws.receive_json()["type"] == "session.created"

    def test_a_forged_token_is_refused(self):
        with (
            TestClient(make_app(gate())) as client,
            pytest.raises(WebSocketDisconnect) as refused,
            client.websocket_connect(
                "/v1/realtime", headers={"origin": TUNNEL_ORIGIN, "cf-access-jwt-assertion": token(key=OTHER_KEY)}
            ),
        ):
            pass
        assert refused.value.reason == "Cloudflare Access token invalid"

    def test_localhost_and_no_origin_need_no_token(self):
        with TestClient(make_app(gate())) as client:
            with client.websocket_connect("/v1/realtime", headers={"origin": "http://localhost:5173"}) as ws:
                assert ws.receive_json()["type"] == "session.created"

    def test_the_webrtc_endpoint_applies_the_same_gate(self):
        headers = {"origin": TUNNEL_ORIGIN, "content-type": "text/plain"}
        with TestClient(make_app(gate())) as client:
            assert client.post("/v1/realtime/calls", content="v=0", headers=headers).status_code == 403
            ok = client.post(
                "/v1/realtime/calls", content="v=0", headers={**headers, "cf-access-jwt-assertion": token()}
            )
            assert ok.status_code != 403
