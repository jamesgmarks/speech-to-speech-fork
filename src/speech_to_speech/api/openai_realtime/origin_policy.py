"""Which web pages may open a voice session from a browser.

A browser always sends ``Origin`` on a WebSocket handshake and on cross-site
POSTs, and any web page can reach ``localhost``. Without a check, any site open
in the user's browser could start a session. Requests with no ``Origin`` are
non-browser clients (scripts, ``websocat``) and are allowed: a browser can't
omit the header, and the session is otherwise unauthenticated on loopback.
"""

import os
from collections.abc import Iterable
from urllib.parse import urlsplit

ALLOWED_ORIGINS_ENV = "S2S_ALLOWED_ORIGINS"

DEFAULT_ALLOWED_ORIGINS: tuple[str, ...] = (
    # Personal OS dashboard: its daemon (4317) and the Vite dev/preview server (5173).
    "http://localhost:4317",
    "http://127.0.0.1:4317",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    # This backend's own pages: the demo frontends (7860 local, 7861 public) and the API port itself.
    "http://localhost:7860",
    "http://127.0.0.1:7860",
    "http://localhost:7861",
    "http://127.0.0.1:7861",
    "http://localhost:8765",
    "http://127.0.0.1:8765",
    # The protected Cloudflare Tunnel hostname.
    "https://voice.onlyjapaneseconversation.com",
)


def normalize_origin(origin: str) -> str:
    """Lower-case ``scheme://host[:port]`` with any trailing slash or path dropped; "" if it isn't an origin."""
    parts = urlsplit(origin.strip())
    if not parts.scheme or not parts.netloc:
        return ""
    return f"{parts.scheme}://{parts.netloc}".lower()


def parse_allowed_origins(value: str | None) -> frozenset[str]:
    """A comma-separated list replaces the defaults; unset or blank keeps them. Entries that aren't origins are ignored."""
    entries: Iterable[str] = DEFAULT_ALLOWED_ORIGINS if value is None or not value.strip() else value.split(",")
    return frozenset(origin for origin in map(normalize_origin, entries) if origin)


def allowed_origins_from_env() -> frozenset[str]:
    return parse_allowed_origins(os.environ.get(ALLOWED_ORIGINS_ENV))


def origin_allowed(origin: str | None, allowed: frozenset[str]) -> bool:
    """No ``Origin`` header means a non-browser client. Anything else must match exactly (``null`` never does)."""
    if origin is None:
        return True
    return normalize_origin(origin) in allowed
