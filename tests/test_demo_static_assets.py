import json
import re
import sys
from pathlib import Path

from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
from static_assets import DemoStaticFiles  # noqa: E402


def write_assets(root):
    (root / "ui").mkdir()
    (root / "index.html").write_text(
        '<link rel="stylesheet" href="style.css"><script type="module" src="main.js?v=old"></script>'
    )
    (root / "main.js").write_text('import { value } from "./ui/value.js"; document.body.dataset.value = value;')
    (root / "ui/value.js").write_text('export const value = "old";')
    (root / "style.css").write_text("body { color: red; }")


def import_map(response):
    return json.loads(re.search(r'<script type="importmap">(.*?)</script>', response.text).group(1))["imports"]


def test_dependency_changes_invalidate_entry_css_and_transitive_modules(tmp_path):
    write_assets(tmp_path)
    app = Starlette()
    app.mount("/", DemoStaticFiles(directory=tmp_path, html=True))
    with TestClient(app) as client:
        first = client.get("/")
        first_map = import_map(first)
        assert first.headers["cache-control"] == "no-store"
        assert "?v=old" not in first.text
        assert set(first_map) == {"./main.js", "./ui/value.js"}
        assert client.get(first_map["./ui/value.js"][1:]).headers["cache-control"] == "no-cache"
        (tmp_path / "ui/value.js").write_text('export const value = "new";')
        second = client.get("/")
        assert import_map(second)["./main.js"] != first_map["./main.js"]
        assert import_map(second)["./ui/value.js"] != first_map["./ui/value.js"]
        assert re.search(r'href="([^"]+)"', first.text).group(1) != re.search(r'href="([^"]+)"', second.text).group(1)


def test_old_physical_html_validators_cannot_hide_updated_dependencies(tmp_path):
    write_assets(tmp_path)
    old_app = Starlette()
    old_app.mount("/", StaticFiles(directory=tmp_path, html=True))
    with TestClient(old_app) as client:
        cached = client.get("/")
    app = Starlette()
    app.mount("/", DemoStaticFiles(directory=tmp_path, html=True))
    with TestClient(app) as client:
        response = client.get(
            "/", headers={"If-None-Match": cached.headers["etag"], "If-Modified-Since": cached.headers["last-modified"]}
        )
        assert response.status_code == 200
        assert import_map(response)
