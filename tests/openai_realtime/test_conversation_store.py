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
