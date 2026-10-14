import base64
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from speech_to_speech.LLM import whiteboard_tools

PNG = b"\x89PNG\r\n\x1a\nfake"


@pytest.fixture
def daemon(monkeypatch):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, status, content_type, body):
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            seen.append(("GET", self.path, None))
            if "token=tok" not in self.path:
                return self._reply(401, "application/json", b'{"error":{"message":"Unauthorized"}}')
            if self.path.startswith("/whiteboard/snapshot.png"):
                self.send_response(200)
                self.send_header("content-type", "image/png")
                self.send_header(
                    "x-whiteboard-layers",
                    "%5B%7B%22id%22%3A%22a%22%2C%22name%22%3A%22Layer%201%22%2C%22visible%22%3Atrue%2C%22active%22%3Atrue%2C%22color%22%3A%22%23ef4444%22%7D%5D",
                )
                self.end_headers()
                return self.wfile.write(PNG)
            self._reply(404, "text/plain", b"")

        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            seen.append(("POST", self.path, json.loads(body)))
            if b"bad" in body:
                return self._reply(
                    400, "application/json", b'{"error":{"message":"commands.0.color: Use a #rrggbb color"}}'
                )
            self._reply(202, "application/json", b'{"ok":true,"batchId":7}')

    server = HTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("POS_DAEMON_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("POS_TOKEN", "tok")
    yield seen
    server.shutdown()


def test_reading_returns_the_snapshot_as_an_mcp_image(daemon):
    result = whiteboard_tools.read_whiteboard()
    image, layers = result["content"]
    assert layers["text"] == "Canvas 1280x800. Layers, bottom to top: Layer 1 [#ef4444] (active)."
    assert image["type"] == "image" and image["mimeType"] == "image/png"
    assert base64.b64decode(image["data"]) == PNG
    assert daemon == [("GET", "/whiteboard/snapshot.png?token=tok", None)]


def test_drawing_posts_the_commands_and_reports_the_batch(daemon):
    commands = [{"type": "layer.add", "name": "Agent"}]
    result = whiteboard_tools.draw_on_whiteboard(commands)
    assert json.loads(result["content"][0]["text"]) == {"ok": True, "batchId": 7}
    assert daemon == [("POST", "/whiteboard/commands?token=tok", {"commands": commands})]


def test_a_pixel_block_is_passed_through_and_the_tool_documents_it(daemon):
    block = {"type": "pixels", "x": 0, "y": 0, "width": 8, "height": 8, "palette": ["#000000"], "rows": ["0"]}
    whiteboard_tools.draw_on_whiteboard([block])
    assert daemon == [("POST", "/whiteboard/commands?token=tok", {"commands": [block]})]
    assert "pixels {" in whiteboard_tools.DRAW_TOOL_DESCRIPTION and "128x128" in whiteboard_tools.DRAW_TOOL_DESCRIPTION
    for syntax in ("TRANSPARENCY", '"transparent"', "|null", "#rrggbbaa", "replace:true", '"." is always'):
        assert syntax in whiteboard_tools.DRAW_TOOL_DESCRIPTION


def test_a_refused_request_explains_why(daemon):
    with pytest.raises(whiteboard_tools.WhiteboardError, match="#rrggbb"):
        whiteboard_tools.draw_on_whiteboard([{"type": "bad"}])


def test_an_unreachable_daemon_is_reported_plainly(monkeypatch):
    monkeypatch.setenv("POS_DAEMON_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("POS_TOKEN", "tok")
    with pytest.raises(whiteboard_tools.WhiteboardError, match="Can't reach"):
        whiteboard_tools.read_whiteboard()
