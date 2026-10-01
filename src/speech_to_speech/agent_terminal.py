"""Launch an independent interactive agent in a visible macOS terminal.

The voice adapter selects the provider; model tool arguments cannot select a
different executable or arbitrary shell command. No SDK session, transcript,
app-only tools, or approval overrides are passed to the new CLI conversation.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

AgentProvider = Literal["claude", "codex"]
TERMINAL_TOOL_NAME = "launch_agent_terminal"
TERMINAL_TOOL_DESCRIPTION = (
    "Open a new visible terminal in the given directory, running a fresh, independent "
    "interactive session of this voice assistant's backing CLI. Optionally provide "
    "an initial prompt to hand off a task. The terminal stays open independently of "
    "this voice call. Its conversation and results are not sent back to this call. "
    "Use this when the user requests a separate terminal or interactive agent session; "
    "use background agents for tasks whose results should return here."
)
TERMINAL_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "directory": {
            "type": "string",
            "description": "Existing absolute directory, ~/path, or path relative to the voice workspace.",
        },
        "prompt": {"type": "string", "description": "Optional initial task. Omit to open an idle interactive session."},
    },
    "required": ["directory"],
    "additionalProperties": False,
}


def _text(value: Any, name: str, *, limit: int = 16384) -> str:
    if not isinstance(value, str) or "\0" in value or len(value) > limit:
        raise ValueError(f"{name} must be text without NUL bytes and at most {limit} characters.")
    return value


class AgentTerminalLauncher:
    def __init__(self, provider: AgentProvider, *, cwd: str | None = None, model: str | None = None):
        if provider not in {"claude", "codex"}:
            raise ValueError("Unsupported interactive agent provider.")
        self.provider = provider
        self.cwd = Path(cwd or os.getcwd()).expanduser().resolve()
        self.model = model

    def launch(self, directory: str, prompt: str = "") -> dict[str, Any]:
        if sys.platform != "darwin":
            raise RuntimeError("Opening an agent terminal currently requires macOS Terminal.app.")
        directory = _text(directory, "directory", limit=4096)
        prompt = _text(prompt, "prompt")
        if not directory.strip():
            raise ValueError("Choose an existing directory.")
        target = Path(directory).expanduser()
        if not target.is_absolute():
            target = self.cwd / target
        target = target.resolve(strict=True)
        if not target.is_dir():
            raise ValueError("The requested path is not a directory.")
        executable = shutil.which(self.provider)
        if executable is None:
            raise RuntimeError(f"Install {self.provider} and make it available on the voice service's PATH first.")
        # Interactive CLIs must not mistake the new terminal for a nested SDK
        # run. Preserve normal CLI authentication/settings and permission rules.
        argv = ["/usr/bin/env", "-u", "CLAUDECODE", "-u", "CLAUDE_CODE_ENTRYPOINT", executable]
        if self.provider == "codex":
            argv.append("--no-daemon")
        if self.model:
            argv.extend(["--model", _text(self.model, "model", limit=256)])
        if prompt:
            argv.extend(["--", prompt])
        command = "cd -- " + shlex.quote(str(target)) + " && " + shlex.join(argv)
        # JSON string quoting is compatible with AppleScript strings here:
        # non-ASCII stays literal and quotes/backslashes/newlines are escaped.
        literal = json.dumps(command, ensure_ascii=False)
        script = (
            'tell application "Terminal"\n'
            f"set agentTab to do script {literal}\n"
            "activate\n"
            "return id of window 1\n"
            "end tell"
        )
        try:
            completed = subprocess.run(
                ["/usr/bin/osascript", "-e", script], capture_output=True, text=True, timeout=15, check=True
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Terminal did not confirm opening within 15 seconds. Check Terminal before retrying."
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip()[:1000]
            raise RuntimeError(f"Could not open Terminal. Check macOS Automation permissions. {detail}") from exc
        return {
            "status": "launch_requested",
            "provider": self.provider,
            "directory": str(target),
            "terminal": "Terminal.app",
            "window_id": completed.stdout.strip(),
            "initial_prompt_provided": bool(prompt),
            "independent": True,
        }


def merge_mcp_config(config: str | None, app_server: Any) -> dict[str, Any]:
    """Add this SDK client's server without writing global/project settings."""
    servers: dict[str, Any] = {}
    if config:
        raw = config if config.lstrip().startswith("{") else Path(config).expanduser().read_text()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("mcpServers"), dict):
            raise ValueError("MCP configuration must contain an mcpServers object.")
        servers.update(parsed["mcpServers"])
    if "speech_to_speech" in servers:
        raise ValueError("The speech_to_speech MCP server name is reserved for this app.")
    servers["speech_to_speech"] = app_server
    return servers
