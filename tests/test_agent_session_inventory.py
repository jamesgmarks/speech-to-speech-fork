import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from speech_to_speech.agent_session_inventory import AgentSessionInventory


def registry():
    return SimpleNamespace(
        list=lambda: [{"session_id": "peer_a", "native_session_id": "native-a", "name": "worker", "state": "working"}],
        contacts=lambda: {"external-b"},
    )


async def test_inventory_deduplicates_native_sessions_and_marks_voice_contacts(monkeypatch):
    import speech_to_speech.agent_session_inventory as module

    monkeypatch.setattr(module.shutil, "which", lambda _: "/bin/claude")
    process = SimpleNamespace(
        returncode=0,
        communicate=AsyncMock(
            return_value=(
                json.dumps(
                    [
                        {"sessionId": "native-a", "name": "worker", "status": "idle"},
                        {
                            "sessionId": "external-b",
                            "name": "review",
                            "status": "busy",
                            "cwd": "/tmp",
                            "kind": "interactive",
                        },
                        {"sessionId": "external-c", "name": "unrelated", "status": "idle"},
                    ]
                ).encode(),
                b"",
            )
        ),
    )
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", spawn)
    inventory = AgentSessionInventory(registry())
    first, second = await asyncio.gather(inventory.list(), inventory.list())
    assert first == second
    assert len(first["sessions"]) == 3
    assert first["sessions"][0]["state"] == "working"
    assert first["sessions"][1]["working_with"]
    assert not first["sessions"][2]["working_with"]
    assert not first["sessions"][1]["permission_control"]
    assert first["sessions"][1]["permission_mode"] is None
    assert "no control connection" in first["sessions"][1]["permission_control_reason"]
    spawn.assert_awaited_once()
    assert spawn.call_args.args == ("/bin/claude", "agents", "--json")


@pytest.mark.parametrize("payload", [b"not-json", b"{}", b"[1]"])
async def test_inventory_keeps_managed_sessions_when_external_discovery_fails(monkeypatch, payload):
    import speech_to_speech.agent_session_inventory as module

    monkeypatch.setattr(module.shutil, "which", lambda _: "/bin/claude")
    monkeypatch.setattr(
        module.asyncio,
        "create_subprocess_exec",
        AsyncMock(
            return_value=SimpleNamespace(
                returncode=0,
                communicate=AsyncMock(return_value=(payload, b"")),
            )
        ),
    )
    result = await AgentSessionInventory(registry()).list()
    assert len(result["sessions"]) == 1
    assert result["discovery_error"]


async def test_disabled_backend_never_discovers_external_sessions(monkeypatch):
    import speech_to_speech.agent_session_inventory as module

    spawn = AsyncMock()
    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", spawn)
    assert await AgentSessionInventory().list() == {"sessions": [], "enabled": False, "discovery_error": None}
    spawn.assert_not_called()


async def test_discovery_timeout_reaps_only_its_child(monkeypatch):
    import speech_to_speech.agent_session_inventory as module

    monkeypatch.setattr(module.shutil, "which", lambda _: "/bin/claude")
    killed = []
    process = SimpleNamespace(
        returncode=None,
        communicate=AsyncMock(side_effect=TimeoutError),
        kill=lambda: killed.append(True),
        wait=AsyncMock(),
    )
    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    result = await AgentSessionInventory(registry()).list()
    assert result["discovery_error"]
    assert killed == [True]
    process.wait.assert_awaited_once()
