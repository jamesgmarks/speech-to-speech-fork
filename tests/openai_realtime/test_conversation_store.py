import json
from uuid import uuid4

import pytest

from speech_to_speech.api.openai_realtime.conversation_store import ConversationStore
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.LLM.chat import Chat, make_assistant_message, make_user_message


def connect(directory, key, backend="claude-agent-sdk", previous_backend=None):
    service = RealtimeService(llm_backend=backend, chat_size=30)
    service.conversation_store = ConversationStore(str(directory))
    conn_id = service.register(key, previous_backend)
    return service, conn_id, service._state(conn_id).runtime_config.chat


def test_context_survives_disconnect_and_fresh_server(tmp_path):
    key = str(uuid4())
    service, conn, chat = connect(tmp_path, key)
    chat.add_item(make_user_message("Remember that Amir prefers tea."))
    chat.add_item(make_assistant_message("I'll remember."))
    service.unregister(conn)
    fresh, resumed, restored = connect(tmp_path, key)
    assert restored.to_transformers_chat() == chat.to_transformers_chat()
    info = fresh.build_session_created(resumed).model_dump()["session"]["speech_to_speech_conversation"]
    assert info == {
        "key": key,
        "backend": "claude-agent-sdk",
        "resumed": True,
        "reset": False,
        "history": [
            {"role": "user", "text": "Remember that Amir prefers tea."},
            {"role": "assistant", "text": "I'll remember."},
        ],
    }


def test_checkpoint_survives_without_clean_shutdown(tmp_path):
    key = str(uuid4())
    service, conn, chat = connect(tmp_path, key)
    chat.add_item(make_user_message("A completed turn before a crash."))
    service.checkpoint(conn)
    _, _, restored = connect(tmp_path, key)
    assert restored.to_transformers_chat() == chat.to_transformers_chat()


def test_backend_change_resets_and_does_not_resurrect_old_backend(tmp_path):
    key = str(uuid4())
    service, conn, chat = connect(tmp_path, key)
    chat.add_item(make_user_message("Old conversation."))
    service.unregister(conn)
    other, conn, chat = connect(tmp_path, key, "responses-api", "claude-agent-sdk")
    assert chat.buffer == []
    assert other._state(conn).conversation_reset
    other.unregister(conn)
    _, _, restored = connect(tmp_path, key, "claude-agent-sdk", "responses-api")
    assert restored.buffer == []


def test_client_backend_hint_resets_separate_backend_store(tmp_path):
    key = str(uuid4())
    service, conn, chat = connect(tmp_path, key)
    chat.add_item(make_user_message("Must not return after switching back."))
    service.unregister(conn)
    _, _, restored = connect(tmp_path, key, previous_backend="responses-api")
    assert restored.buffer == []


def test_context_is_isolated_and_concurrent_attachment_is_rejected(tmp_path):
    store = ConversationStore(str(tmp_path))
    key = str(uuid4())
    chat, _, _ = store.acquire(key, "claude-agent-sdk", 30)
    chat.add_item(make_user_message("Private to this conversation."))
    store.save(key, "claude-agent-sdk", chat)
    with pytest.raises(ValueError, match="already connected"):
        store.acquire(key, "claude-agent-sdk", 30)
    other, resumed, _ = store.acquire(str(uuid4()), "claude-agent-sdk", 30)
    assert other.buffer == [] and not resumed
    store.release(key)
    restored, resumed, _ = store.acquire(key, "claude-agent-sdk", 30)
    assert restored.buffer and resumed


def test_invalid_key_and_corrupt_state_are_not_silently_reset(tmp_path):
    store = ConversationStore(str(tmp_path))
    with pytest.raises(ValueError):
        store.acquire("../../escape", "claude-agent-sdk", 30)
    key = str(uuid4())
    path = tmp_path / f"{key}.json"
    path.write_text("broken JSON")
    with pytest.raises(ValueError):
        store.acquire(key, "claude-agent-sdk", 30)
    assert path.read_text() == "broken JSON"


def test_unfinished_generation_is_excluded_until_committed():
    chat = Chat(30)
    user = chat.add_item(make_user_message("Hello."))
    chat.add_provisional_generation_items(
        "response_1", [make_assistant_message("Unheard reply.")], after_item_id=user.id
    )
    assert len(chat.persistent_snapshot()) == 1
    assert len(chat.buffer) == 2
    chat.finalize_provisional_generation("response_1")
    assert len(chat.persistent_snapshot()) == 2


def test_current_conversation_adopts_existing_context_and_survives_new_browsers_and_server(tmp_path):
    key = str(uuid4())
    service, conn, chat = connect(tmp_path, key)
    chat.add_item(make_user_message("An ongoing conversation across browser origins."))
    service.unregister(conn)
    store = ConversationStore(str(tmp_path))
    info = store.current("claude-agent-sdk", 30)
    assert info["key"] == key
    assert info["history"] == [{"role": "user", "text": "An ongoing conversation across browser origins."}]
    assert ConversationStore(str(tmp_path)).current("claude-agent-sdk", 30) == info
    # An unrelated old client saving a newer file cannot move the pointer.
    other = Chat(30)
    other.add_item(make_user_message("A different browser's old conversation."))
    store.save(str(uuid4()), "claude-agent-sdk", other)
    assert store.current("claude-agent-sdk", 30)["key"] == key


def test_shared_reset_is_explicit_and_stale_browsers_cannot_reset_it_again(tmp_path):
    store = ConversationStore(str(tmp_path))
    info = store.current("claude-agent-sdk", 30)
    chat, _, _ = store.acquire(info["key"], "claude-agent-sdk", 30)
    chat.add_item(make_user_message("Keep this until a deliberate reset."))
    store.save(info["key"], "claude-agent-sdk", chat)
    with pytest.raises(ValueError, match="Disconnect"):
        store.start_new(info["key"], "claude-agent-sdk", 30)
    store.release(info["key"])
    fresh = store.start_new(info["key"], "claude-agent-sdk", 30)
    assert fresh["key"] != info["key"] and fresh["history"] == []
    with pytest.raises(ValueError, match="changed in another browser"):
        store.start_new(info["key"], "claude-agent-sdk", 30)
    assert ConversationStore(str(tmp_path)).current("claude-agent-sdk", 30) == fresh


def test_shared_backend_switch_clears_context_and_transcript_without_resurrecting_old_history(tmp_path):
    store = ConversationStore(str(tmp_path))
    info = store.current("claude-agent-sdk", 30)
    chat = Chat(30)
    chat.add_item(make_user_message("Claude context."))
    store.save(info["key"], "claude-agent-sdk", chat)
    switched = store.current("responses-api", 30)
    assert switched["key"] == info["key"] and switched["reset"] and switched["history"] == []
    assert store.current("claude-agent-sdk", 30)["history"] == []


def test_readable_transcript_survives_model_context_eviction_and_service_restart(tmp_path):
    store = ConversationStore(str(tmp_path))
    key = store.current("claude-agent-sdk", 1)["key"]
    chat = Chat(1)
    for text in ["First turn", "Second turn", "Third turn"]:
        chat.add_item(make_user_message(text))
        store.save(key, "claude-agent-sdk", chat)
    info = ConversationStore(str(tmp_path)).current("claude-agent-sdk", 1)
    assert [row["text"] for row in info["history"]] == ["First turn", "Second turn", "Third turn"]
    saved = json.loads((tmp_path / f"{key}.json").read_text())
    assert len(saved["items"]) < len(info["history"])
    assert (tmp_path / "current.json").stat().st_mode & 0o777 == 0o600
