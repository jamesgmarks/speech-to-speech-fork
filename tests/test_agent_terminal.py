"""Interactive handoffs preserve arguments and do not alter CLI permissions."""

import json
import shlex
import subprocess
from types import SimpleNamespace

import pytest

import speech_to_speech.agent_terminal as terminal


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_launch_quotes_untrusted_paths_and_prompts_and_starts_fresh(tmp_path, monkeypatch, provider):
    target = tmp_path / "space ' \" $(touch nope) `echo nope` ; & café"
    target.mkdir()
    prompt = "--resume old; $(touch nope)\n\"hello\" 'friend'"
    captured = []
    monkeypatch.setattr(terminal.sys, "platform", "darwin")
    monkeypatch.setattr(terminal.shutil, "which", lambda name: f"/local bin/{name}")
    monkeypatch.setattr(
        terminal.subprocess, "run", lambda argv, **kw: captured.append((argv, kw)) or SimpleNamespace(stdout="123\n")
    )
    result = terminal.AgentTerminalLauncher(provider, cwd=str(tmp_path), model="a-model").launch(target.name, prompt)
    argv, options = captured[0]
    assert argv[:2] == ["/usr/bin/osascript", "-e"]
    command = json.loads(argv[2].splitlines()[1].removeprefix("set agentTab to do script "))
    assert shlex.split(command) == [
        "cd",
        "--",
        str(target),
        "&&",
        "/usr/bin/env",
        "-u",
        "CLAUDECODE",
        "-u",
        "CLAUDE_CODE_ENTRYPOINT",
        f"/local bin/{provider}",
        *(["--no-daemon"] if provider == "codex" else []),
        "--model",
        "a-model",
        "--",
        prompt,
    ]
    assert options["check"] and options["timeout"] == 15
    assert result["provider"] == provider and result["directory"] == str(target)
    assert result["independent"] and result["window_id"] == "123"


def test_plain_interactive_session_omits_prompt_and_model(tmp_path, monkeypatch):
    monkeypatch.setattr(terminal.sys, "platform", "darwin")
    monkeypatch.setattr(terminal.shutil, "which", lambda _: "/usr/bin/claude")
    captured = []
    monkeypatch.setattr(
        terminal.subprocess, "run", lambda argv, **kw: captured.append(argv) or SimpleNamespace(stdout="1")
    )
    assert not terminal.AgentTerminalLauncher("claude", cwd=str(tmp_path)).launch(".")["initial_prompt_provided"]
    command = json.loads(captured[0][2].splitlines()[1].removeprefix("set agentTab to do script "))
    assert shlex.split(command)[-1] == "/usr/bin/claude"


@pytest.mark.parametrize("directory,prompt", [("", ""), ("\0", ""), (".", "\0"), (".", "x" * 16385), ("missing", "")])
def test_invalid_arguments_never_open_terminal(tmp_path, monkeypatch, directory, prompt):
    monkeypatch.setattr(terminal.sys, "platform", "darwin")
    monkeypatch.setattr(terminal.subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected terminal launch"))
    with pytest.raises((ValueError, FileNotFoundError)):
        terminal.AgentTerminalLauncher("claude", cwd=str(tmp_path)).launch(directory, prompt)


def test_file_is_not_a_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(terminal.sys, "platform", "darwin")
    file = tmp_path / "file"
    file.touch()
    with pytest.raises(ValueError, match="not a directory"):
        terminal.AgentTerminalLauncher("claude").launch(str(file))


def test_missing_cli_and_unsupported_platform_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(terminal.sys, "platform", "darwin")
    monkeypatch.setattr(terminal.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="Install claude"):
        terminal.AgentTerminalLauncher("claude").launch(str(tmp_path))
    monkeypatch.setattr(terminal.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="macOS"):
        terminal.AgentTerminalLauncher("codex").launch(str(tmp_path))


@pytest.mark.parametrize("timeout", [False, True])
def test_terminal_errors_are_reported(tmp_path, monkeypatch, timeout):
    monkeypatch.setattr(terminal.sys, "platform", "darwin")
    monkeypatch.setattr(terminal.shutil, "which", lambda _: "/bin/claude")

    def fail(*args, **kwargs):
        if timeout:
            raise subprocess.TimeoutExpired("osascript", 15)
        raise subprocess.CalledProcessError(1, "osascript", stderr="Not authorized (-1743)")

    monkeypatch.setattr(terminal.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="Check Terminal" if timeout else "Automation permissions"):
        terminal.AgentTerminalLauncher("claude").launch(str(tmp_path))


def test_mcp_config_merge_preserves_external_servers_without_writing(tmp_path):
    path = tmp_path / "app.json"
    text = json.dumps({"mcpServers": {"other": {"command": "server", "env": {"TOKEN": "${TOKEN}"}}}})
    path.write_text(text)
    server = object()
    assert terminal.merge_mcp_config(str(path), server) == terminal.merge_mcp_config(text, server)
    assert terminal.merge_mcp_config(text, server)["other"]["env"] == {"TOKEN": "${TOKEN}"}
    assert path.read_text() == text
    assert terminal.merge_mcp_config(None, server) == {"speech_to_speech": server}


@pytest.mark.parametrize("config", ["[]", '{"mcpServers": []}', '{"mcpServers": {"speech_to_speech": {}}}'])
def test_mcp_config_rejects_invalid_or_reserved_server(config):
    with pytest.raises((ValueError, OSError)):
        terminal.merge_mcp_config(config, object())
