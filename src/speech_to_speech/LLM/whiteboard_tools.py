"""Voice-assistant tools for the Personal OS whiteboard widget.

The Personal OS daemon holds the board: it serves the latest flattened snapshot as a PNG and accepts drawing
commands. These helpers are the thin HTTP client for that. `POS_DAEMON_URL` (default http://127.0.0.1:4317) and
the token in `~/.personal-os/token` (or `POS_TOKEN`) say where and how to reach it.
"""

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_DAEMON_URL = "http://127.0.0.1:4317"
TIMEOUT_S = 10

READ_TOOL_NAME = "read_whiteboard"
DRAW_TOOL_NAME = "draw_on_whiteboard"

READ_TOOL_DESCRIPTION = (
    "Look at James's Personal OS whiteboard: returns the current drawing (all visible layers flattened) as an image, "
    "plus the layer names. Pass `layer` (a layer's id, name, or stack index; 0 is the bottom) to see just that one layer on its own "
    "over the paper color, hidden layers included. Read-only. Fails if no Whiteboard widget is open on a dashboard."
)
READ_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "layer": {
            "type": ["string", "integer"],
            "description": "Optional. The layer to view alone: its id, name, or stack index (0 is the bottom). Omit for the whole board.",
        },
    },
    "additionalProperties": False,
}
DRAW_TOOL_DESCRIPTION = (
    "Draw on James's Personal OS whiteboard with a batch of commands, applied in order. The canvas is 1280x800 pixels, "
    "origin top-left; colors are #rrggbb; every command may name a `layer` (id or name; default: the active layer). "
    "Commands (each an object with a `type`): "
    "stroke {points:[{x,y,pressure?}], color, size}; erase {points, size}; line|arrow {from:{x,y}, to:{x,y}, color, size} (an arrow has its head at `to`); "
    "rect|ellipse {x, y, width, height, color, size, fill?}; polygon {points, color, size, fill?}; "
    "fill {x, y, color, tolerance?}; text {x, y, text, color, fontSize?}; clear {}; "
    "layer.add {name?}; layer.delete {layer}; layer.move {layer, to}; layer.visibility {layer, visible}; "
    "layer.select {layer}; layer.rename {layer, name}; layer.color {layer, color?} (a #rrggbb label for the layer chip; omit color to clear); undo {}; redo {}; "
    'pixels {x, y, width, height, layer?, plus EITHER palette:["#rrggbb"|"#rrggbbaa"|"transparent"|null, ...up to 36] and rows:["0110", ...] '
    '(one string per pixel row, one character per pixel: 0-9 then a-z indexes the palette; "." is always a transparent pixel) OR data:<base64 of raw bytes, row by row> with sourceWidth, sourceHeight and format:"rgba"(default)|"rgb"}. '
    "`pixels` draws a whole sprite or picture in ONE command: the pixel grid is placed at x,y and scaled (crisp, not smoothed) to width x height board pixels, "
    "like a pasted image; it lands on the layer as one undo step and is then selected. The grid is at most 128x128 pixels (anything larger is rejected, so split it into several pixels commands). "
    "Big pictures: tile many blocks on ONE layer, each at its own x,y; edges are rounded, so blocks laid edge to edge (next x = previous x + width) leave no seams, "
    "later blocks draw over earlier ones (transparent pixels let the earlier ones show; add replace:true to overwrite the area completely), and blocks sent close together undo as one step. "
    'TRANSPARENCY is supported, pixels do NOT all need a color: a palette entry of "transparent" (or null) paints nothing, so whatever is already on the layer and the layers below shows through; '
    '"#rrggbbaa" gives partial transparency (aa is hex alpha: 00 invisible, 80 half, ff opaque); in raw format rgba the 4th byte of each pixel is alpha (rgb is always opaque). '
    "Transparent pixels also let earlier tiles show; replace:true instead erases the whole block area first, so transparent pixels there become empty. "
    'Prefer palette+rows: e.g. a 4x4 heart-ish sprite as palette ["transparent","#ef4444"], rows ["0110","1111","0110","0000"]. '
    "Build diagrams from these: boxes (rect with a text label inside), arrows between them, ellipses for nodes, "
    "a separate layer for your own additions so James's drawing is never covered (layer.add then draw, or name `layer`). "
    "Text is positioned by its top-left corner; fontSize 32 is about 20 px wide per character. Fill with `fill` on a shape, or a fill command on a region. "
    "Use read_whiteboard afterwards to check the result. Drawing lands on the widget within about a second."
)
DRAW_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "commands": {"type": "array", "minItems": 1, "maxItems": 200, "items": {"type": "object"}},
    },
    "required": ["commands"],
    "additionalProperties": False,
}


class WhiteboardError(RuntimeError):
    """The whiteboard couldn't be reached or refused the request; the message is fit to say aloud."""


def _daemon_url() -> str:
    return os.environ.get("POS_DAEMON_URL", DEFAULT_DAEMON_URL).rstrip("/")


def _token() -> str:
    token = os.environ.get("POS_TOKEN")
    if token:
        return token.strip()
    try:
        return (Path.home() / ".personal-os" / "token").read_text().strip()
    except OSError as exc:
        raise WhiteboardError("Can't read the Personal OS token (~/.personal-os/token).") from exc


def _request(path: str, body: bytes | None = None, query: dict[str, str] | None = None) -> tuple[bytes, Any]:
    url = f"{_daemon_url()}{path}?{urllib.parse.urlencode({'token': _token(), **(query or {})})}"
    request = urllib.request.Request(url, data=body, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310 - fixed local daemon URL
            return response.read(), response.headers
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            pass
        raise WhiteboardError(f"The whiteboard said {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise WhiteboardError("Can't reach the Personal OS daemon. Is it running?") from exc


def read_whiteboard(layer: str | int | None = None) -> dict[str, Any]:
    """An MCP tool result holding the snapshot PNG as an image: the whole board, or just `layer` when given."""
    png, headers = _request("/whiteboard/snapshot.png", query=None if layer is None else {"layer": str(layer)})
    content: list[dict[str, Any]] = [
        {"type": "image", "data": base64.b64encode(png).decode("ascii"), "mimeType": "image/png"}
    ]
    layers = headers.get("x-whiteboard-layers")
    if layers:
        try:
            listing = json.loads(urllib.parse.unquote(layers))
        except ValueError:
            listing = []
        names = ", ".join(
            f"{layer['name']}{' [' + layer['color'] + ']' if layer.get('color') else ''}"
            f"{' (active)' if layer.get('active') else ''}{'' if layer.get('visible', True) else ' (hidden)'}"
            for layer in listing
        )
        content.append({"type": "text", "text": f"Canvas 1280x800. Layers, bottom to top: {names}."})
    return {"content": content}


def draw_on_whiteboard(commands: list[dict[str, Any]]) -> dict[str, Any]:
    body, _ = _request("/whiteboard/commands", json.dumps({"commands": commands}).encode("utf-8"))
    return {"content": [{"type": "text", "text": body.decode("utf-8")}]}
