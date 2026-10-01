"""Keep an unbundled module graph on one frontend revision, including upgrades."""

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

from starlette.responses import FileResponse, HTMLResponse
from starlette.staticfiles import StaticFiles


class DemoStaticFiles(StaticFiles):
    def _versioned_html(self, path: str) -> str:
        root = Path(self.directory)
        assets = []
        for directory, children, files in os.walk(root):
            children[:] = [
                name
                for name in children
                if not name.startswith(".") and name not in {"node_modules", "vendor", "tests", "__pycache__"}
            ]
            assets.extend(Path(directory) / name for name in files if Path(name).suffix in {".js", ".mjs", ".css"})
        if (root / "package.json").exists():
            assets.append(root / "package.json")
        digest = hashlib.sha256()
        for asset in sorted(assets):
            digest.update(asset.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(asset.read_bytes())
        revision = digest.hexdigest()[:16]
        imports = {
            "./" + asset.relative_to(root).as_posix(): "./" + asset.relative_to(root).as_posix() + "?v=" + revision
            for asset in assets
            if asset.suffix in {".js", ".mjs"}
        }
        source = Path(path).read_text()
        source = re.sub(r'href="style\.css(?:\?[^\"]*)?"', f'href="style.css?v={revision}"', source)
        source = re.sub(r'src="main\.js(?:\?[^\"]*)?"', f'src="main.js?v={revision}"', source)
        source = re.sub(
            r'src="vendor/openai-realtime-agents\.umd\.js(?:\?[^\"]*)?"',
            f'src="vendor/openai-realtime-agents.umd.js?v={revision}"',
            source,
        )
        import_map = '<script type="importmap">' + json.dumps({"imports": imports}) + "</script>\n"
        return source.replace('<script type="module"', import_map + '<script type="module"', 1)

    async def get_response(self, path, scope):
        # The physical HTML can be unchanged when a dependency changes. Its
        # original file ETag must not produce a 304 for the old import map.
        scope = {
            **scope,
            "headers": [
                (name, value)
                for name, value in scope["headers"]
                if name.lower() not in {b"if-none-match", b"if-modified-since"}
            ],
        }
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse) and str(response.path).endswith(".html"):
            body = await asyncio.to_thread(self._versioned_html, str(response.path))
            return HTMLResponse(body, headers={"Cache-Control": "no-store"})
        response.headers["Cache-Control"] = "no-cache"
        return response
