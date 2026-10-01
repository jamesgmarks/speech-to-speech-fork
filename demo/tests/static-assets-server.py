"""Browser fixture for upgrading an already cached, unbundled frontend."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import uvicorn
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route
from starlette.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from static_assets import DemoStaticFiles  # noqa: E402


class CachedLegacyFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = (
            "no-store" if str(path) in (".", "", "index.html") else "public, max-age=86400"
        )
        return response


with TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "ui").mkdir()
    (root / "index.html").write_text(
        '<link rel="stylesheet" href="style.css"><body><script type="module" src="main.js?v=old"></script></body>'
    )
    (root / "main.js").write_text('import { value } from "./ui/value.js"; document.body.dataset.value = value;')
    (root / "ui/value.js").write_text('export const value = "old";')
    (root / "style.css").write_text("body { color: rgb(255, 0, 0); }")
    current = CachedLegacyFiles(directory=root, html=True)

    async def deploy(request):
        global current
        (root / "ui/value.js").write_text('export const value = "new";')
        (root / "style.css").write_text("body { color: rgb(0, 0, 255); }")
        current = DemoStaticFiles(directory=root, html=True)
        return Response("deployed")

    app = Starlette(routes=[Route("/deploy", deploy, methods=["POST"])])

    async def files(scope, receive, send):
        await current(scope, receive, send)

    app.mount("/", files)
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
