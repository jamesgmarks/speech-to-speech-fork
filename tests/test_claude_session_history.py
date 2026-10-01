import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("claude_agent_sdk")

import speech_to_speech.LLM.claude_session_history as history
from speech_to_speech.agent_session_inventory import resolve_session_target

NATIVE = "d8305e87-041c-4aa9-8314-845c493558c0"
SESSION = {
    "session_id": NATIVE,
    "native_session_id": NATIVE,
    "name": "james-b6",
    "directory": "/tmp/james",
    "state": "idle",
    "source": "external",
}


def message(role, content, id="message-1"):
    return SimpleNamespace(type=role, uuid=id, message={"content": content})


@pytest.fixture
def saved(monkeypatch):
    info = Mock(
        return_value=SimpleNamespace(
            last_modified=1790851490371, summary="Recent work", file_size=2048, cwd="/tmp/james", custom_title=None
        )
    )
    messages = Mock(
        return_value=[
            message("user", "What are you working on?", "user-1"),
            message(
                "assistant",
                [
                    {"type": "thinking", "thinking": "hidden", "signature": "private"},
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "mcp__pepper-rooms__send_message",
                        "input": {"room": "speech-to-speech-router", "body": "All quiet."},
                    },
                ],
            ),
            message(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": [
                            {"type": "text", "text": "Room message sent."},
                            {"type": "image", "data": "hidden-image"},
                        ],
                    }
                ],
            ),
            message(
                "assistant",
                [
                    {"type": "text", "text": "I replied in the room."},
                    {"type": "image", "source": {"data": "hidden-image"}},
                ],
            ),
        ]
    )
    monkeypatch.setattr(history, "get_session_info", info)
    monkeypatch.setattr(history, "get_session_messages", messages)
    return info, messages


async def test_recent_snapshot_exposes_reply_channel_and_tool_results_without_hidden_payloads(saved):
    inventory = SimpleNamespace(list=AsyncMock(return_value={"sessions": [SESSION]}))
    result = await history.read_agent_session(inventory, "James B6", 3)
    assert result["availability"] == "available"
    assert result["truncated"]
    assert result["session"]["directory"] == "/tmp/james"
    assert result["session"]["last_saved_at"] == "2026-10-01T10:44:50.371000+00:00"
    assert len(result["messages"]) == 3
    assert result["messages"][0]["blocks"][0]["name"] == "mcp__pepper-rooms__send_message"
    assert result["messages"][1]["blocks"][0]["content"] == "Room message sent."
    raw = json.dumps(result)
    assert "hidden" not in raw and "signature" not in raw
    saved[1].assert_called_once_with(NATIVE, directory="/tmp/james")


async def test_exited_session_can_be_read_by_exact_uuid(saved):
    inventory = SimpleNamespace(list=AsyncMock(return_value={"sessions": []}))
    result = await history.read_agent_session(inventory, NATIVE)
    assert result["session"]["source"] == "saved"
    assert result["session"]["state"] == "not_live"


async def test_unknown_path_cannot_select_an_arbitrary_file(saved):
    inventory = SimpleNamespace(list=AsyncMock(return_value={"sessions": []}))
    with pytest.raises(ValueError):
        await history.read_agent_session(inventory, "../../secrets.json")
    saved[0].assert_not_called()
    saved[1].assert_not_called()


async def test_ambiguity_preserves_choices_without_reading_logs(saved):
    other = {**SESSION, "name": "james-b7", "session_id": "other", "native_session_id": "other"}
    inventory = SimpleNamespace(list=AsyncMock(return_value={"sessions": [SESSION, other]}))
    with pytest.raises(ValueError, match="Ambiguous.*james-b6.*james-b7"):
        await history.read_agent_session(inventory, "James")
    saved[0].assert_not_called()


def test_spoken_target_resolution_prefers_exact_matches_and_rejects_ambiguity():
    assert resolve_session_target("James", [SESSION]) is SESSION
    assert resolve_session_target("James B6", [SESSION]) is SESSION
    assert resolve_session_target(NATIVE, [SESSION]) is SESSION
    exact = {**SESSION, "name": "James", "session_id": "different"}
    assert resolve_session_target("james", [SESSION, exact]) is exact
    with pytest.raises(ValueError, match="Ambiguous"):
        resolve_session_target("james", [SESSION, {**SESSION, "name": "james-other"}])


@pytest.mark.parametrize("limit", [0, 101, True, "20"])
async def test_invalid_limits_do_not_read_transcripts(saved, limit):
    with pytest.raises(ValueError, match="max_messages"):
        await history.read_agent_session(SimpleNamespace(), NATIVE, limit)
    saved[1].assert_not_called()


async def test_large_transcripts_return_metadata_without_full_parse(saved):
    saved[0].return_value.file_size = history.MAX_TRANSCRIPT_BYTES + 1
    inventory = SimpleNamespace(list=AsyncMock(return_value={"sessions": [SESSION]}))
    result = await history.read_agent_session(inventory, NATIVE)
    assert result["availability"] == "too_large"
    saved[1].assert_not_called()


async def test_output_budget_keeps_newest_evidence_and_marks_truncation(saved):
    saved[1].return_value = [
        message("assistant", [{"type": "text", "text": str(i) + "x" * 9000}], str(i)) for i in range(100)
    ]
    inventory = SimpleNamespace(list=AsyncMock(return_value={"sessions": [SESSION]}))
    result = await history.read_agent_session(inventory, NATIVE, 100)
    assert result["truncated"] and len(json.dumps(result, ensure_ascii=False)) < history.MAX_OUTPUT_CHARS
    assert result["messages"][-1]["id"] == "99"


async def test_unsaved_session_reports_unavailable_without_inference(saved):
    inventory = SimpleNamespace(list=AsyncMock(return_value={"sessions": [{**SESSION, "native_session_id": None}]}))
    result = await history.read_agent_session(inventory, "James")
    assert result["availability"] == "not_yet_saved"
    saved[0].assert_not_called()
    saved[1].assert_not_called()
