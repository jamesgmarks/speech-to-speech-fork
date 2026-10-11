"""Cloudflare Access verification for sessions that arrive from a non-local origin or through the tunnel.

Browsers on this machine (``localhost`` / ``127.0.0.1`` origins) need nothing. A page on any other
origin, such as the tunnel hostname, must carry a valid Cloudflare Access JWT. The browser never
handles that token itself: after the Access login Cloudflare sets an HttpOnly ``CF_Authorization``
cookie for the hostname, the browser sends it with the WebSocket handshake, and Cloudflare's edge
forwards it to the origin as the ``Cf-Access-Jwt-Assertion`` header. Service tokens
(``CF-Access-Client-Id`` / ``-Secret``) are exchanged for the same JWT at the edge, so verifying
the JWT covers both; a browser can't set those headers on a WebSocket anyway.

The JWT is verified here, not trusted because it arrived through the tunnel: RS256 signature against
the team's published keys, issuer, audience (the Access application tag) and expiry.
"""

import asyncio
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.cookies import CookieError, SimpleCookie
from typing import Any
from urllib.parse import urlsplit

import jwt
from jwt import PyJWKClient

from speech_to_speech.api.openai_realtime.origin_policy import origin_allowed

logger = logging.getLogger(__name__)

REQUIRED_ENV = "S2S_CF_ACCESS_REQUIRED"
TEAM_ENV = "S2S_CF_ACCESS_TEAM"
AUDIENCE_ENV = "S2S_CF_ACCESS_AUD"

ACCESS_HEADER = "cf-access-jwt-assertion"
ACCESS_COOKIE = "CF_Authorization"
LOCAL_HOSTNAMES = frozenset({"localhost", "127.0.0.1"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_CLOCK_LEEWAY_SECONDS = 30


class AccessDenied(Exception):
    """The token is missing or not acceptable; the message is safe to show the client."""


@dataclass(frozen=True)
class CloudflareAccessConfig:
    """``required`` (default on): non-local browser origins must present a verified token. ``team`` and
    ``audiences`` say how to verify it; with ``required`` on and either missing, such origins are refused."""

    required: bool = True
    team: str | None = None
    audiences: frozenset[str] = frozenset()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "CloudflareAccessConfig":
        source = os.environ if env is None else env
        required = source.get(REQUIRED_ENV, "").strip().lower() not in _FALSE_VALUES
        audiences = frozenset(tag.strip() for tag in source.get(AUDIENCE_ENV, "").split(",") if tag.strip())
        return cls(required=required, team=source.get(TEAM_ENV, "").strip() or None, audiences=audiences)

    @property
    def configured(self) -> bool:
        return bool(self.team and self.audiences)


KeyResolver = Callable[[str], Any]


def access_issuer(team: str) -> str:
    """The team's Access domain as an issuer URL. ``team`` may be the team name or its full domain (``https://`` optional)."""
    host = (urlsplit(team if "//" in team else f"//{team}").hostname or team).lower()
    return f"https://{host if '.' in host else f'{host}.cloudflareaccess.com'}"


class CloudflareAccessVerifier:
    def __init__(self, team: str, audiences: frozenset[str], key_resolver: KeyResolver | None = None) -> None:
        self.issuer = access_issuer(team)
        self._audiences = sorted(audiences)
        if key_resolver is None:
            # Keys are cached (and re-fetched when a token names an unknown key id).
            jwks = PyJWKClient(f"{self.issuer}/cdn-cgi/access/certs", cache_keys=True, lifespan=3600, timeout=5)
            key_resolver = lambda token: jwks.get_signing_key_from_jwt(token).key  # noqa: E731
        self._resolve_key = key_resolver

    def verify(self, token: str) -> dict[str, Any]:
        """Blocking (it may fetch keys): run it off the event loop. Returns the claims or raises ``AccessDenied``."""
        try:
            return jwt.decode(
                token,
                self._resolve_key(token),
                algorithms=["RS256"],  # never the token's own choice: no ``none``, no HS256 key confusion
                audience=self._audiences,
                issuer=self.issuer,
                leeway=_CLOCK_LEEWAY_SECONDS,
                options={"require": ["exp", "iss", "aud"]},
            )
        except jwt.PyJWTError as error:
            logger.warning("Rejected Cloudflare Access token: %s", error)
            raise AccessDenied("Cloudflare Access token invalid") from error
        except Exception as error:  # noqa: BLE001 - e.g. the key endpoint is unreachable: fail closed
            logger.error("Could not verify a Cloudflare Access token: %s", error)
            raise AccessDenied("Cloudflare Access token could not be verified") from error


def is_local_origin(origin: str) -> bool:
    return (urlsplit(origin.strip()).hostname or "").lower() in LOCAL_HOSTNAMES


# Added by Cloudflare's tunnel/edge (or any reverse proxy) to every request it forwards; a client can't remove them.
PROXY_MARKER_HEADERS = ("cf-connecting-ip", "cf-ray", "x-forwarded-for", "forwarded")


def arrived_via_proxy(headers: Any) -> bool:
    return any(headers.get(name) for name in PROXY_MARKER_HEADERS)


def extract_access_token(headers: Any) -> str | None:
    """The Access JWT as forwarded by Cloudflare's edge, or failing that the cookie it was issued in."""
    header = (headers.get(ACCESS_HEADER) or "").strip()
    if header:
        return header
    try:
        cookies: SimpleCookie = SimpleCookie(headers.get("cookie") or "")
    except CookieError:
        return None
    morsel = cookies.get(ACCESS_COOKIE)
    return morsel.value.strip() or None if morsel else None


class SessionGate:
    """Decides whether a client may open a voice session. A browser's origin must be allowed. Anything remote
    (a non-local origin, or any request that came through the tunnel or a proxy) must carry a valid Cloudflare
    Access token. A local request with no ``Origin`` is a non-browser tool and needs nothing."""

    def __init__(
        self,
        allowed_origins: frozenset[str],
        access: CloudflareAccessConfig | None = None,
        verifier: CloudflareAccessVerifier | None = None,
    ) -> None:
        self._allowed_origins = allowed_origins
        self._access = access if access is not None else CloudflareAccessConfig.from_env()
        self._verifier = verifier
        if self._verifier is None and self._access.required and self._access.configured:
            assert self._access.team is not None
            self._verifier = CloudflareAccessVerifier(self._access.team, self._access.audiences)
        if self._access.required and self._verifier is None:
            logger.warning(
                "Cloudflare Access verification is on but %s/%s aren't set: browsers on non-local origins will be refused",
                TEAM_ENV,
                AUDIENCE_ENV,
            )

    @property
    def allowed_origins(self) -> frozenset[str]:
        return self._allowed_origins

    async def denial(self, headers: Any) -> str | None:
        """Why the request must be refused, or None if it may proceed."""
        origin = headers.get("origin")
        proxied = arrived_via_proxy(headers)
        if origin is None and not proxied:
            return None  # a non-browser client on this machine
        if origin is not None and not origin_allowed(origin, self._allowed_origins):
            return "Origin not allowed"
        # Remote if the page is on another origin, or the request came through the tunnel at all (which also
        # catches a non-browser client, or one forging a localhost Origin, going through it).
        remote = proxied or origin is None or not is_local_origin(origin)
        if not remote or not self._access.required:
            return None
        if self._verifier is None:
            return "Cloudflare Access isn't configured on the backend"
        token = extract_access_token(headers)
        if token is None:
            return "Cloudflare Access token required"
        try:
            await asyncio.to_thread(self._verifier.verify, token)
        except AccessDenied as denied:
            return str(denied)
        return None
