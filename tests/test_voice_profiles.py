import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler
from speech_to_speech.TTS.voice_profiles import load_voice_profiles


@pytest.fixture
def manifest(tmp_path):
    entries = []
    for name in ("james", "pepper", "pepper-upbeat"):
        (tmp_path / f"{name}.wav").write_bytes(b"reference fixture")
        (tmp_path / f"{name}.txt").write_text(f"Transcript for {name}.\n")
        entries.append(
            {"id": f"custom:{name}", "name": name.title(), "ref_audio": f"{name}.wav", "ref_text_file": f"{name}.txt"}
        )
    path = tmp_path / "voices.json"
    path.write_text(json.dumps(entries))
    return path


@pytest.fixture
def handler(manifest, monkeypatch):
    monkeypatch.setattr(
        Qwen3TTSHandler,
        "_setup_mlx",
        lambda self, name: setattr(self, "model", SimpleNamespace(config=SimpleNamespace(tts_model_type="base"))),
    )
    monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)
    monkeypatch.setattr("speech_to_speech.TTS.qwen3_tts_handler.platform", "darwin")
    obj = object.__new__(Qwen3TTSHandler)
    obj.setup(
        Event(),
        model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ref_audio=manifest.parent / "james.wav",
        ref_text="Transcript for james.",
        voice_profiles=manifest,
    )
    return obj


def config(voice):
    return SimpleNamespace(session=SimpleNamespace(audio=SimpleNamespace(output=SimpleNamespace(voice=voice))))


def test_manifest_resolves_relative_paths_and_reads_matching_transcript(manifest):
    profiles = load_voice_profiles(manifest)
    assert profiles["custom:pepper"].ref_audio == manifest.parent / "pepper.wav"
    assert profiles["custom:pepper"].ref_text == "Transcript for pepper."


@pytest.mark.parametrize("fault", ["duplicate", "missing_audio", "empty_transcript", "invalid_id", "empty_catalog"])
def test_manifest_rejects_invalid_profiles_at_startup(manifest, fault):
    entries = json.loads(manifest.read_text())
    if fault == "duplicate":
        entries.append(entries[0])
    elif fault == "missing_audio":
        entries[0]["ref_audio"] = "absent.wav"
    elif fault == "empty_transcript":
        (manifest.parent / "james.txt").write_text("  \n")
    elif fault == "invalid_id":
        entries[0]["id"] = "../james"
    else:
        entries = []
    manifest.write_text(json.dumps(entries))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_voice_profiles(manifest)


def test_switching_voices_pairs_audio_and_text_and_clears_cached_codes(handler):
    handler.ref_spk, handler.ref_rvq = Path("old.spk"), Path("old.rvq")
    handler._apply_session_voice_override("base", config("custom:pepper"))
    assert handler.ref_audio.name == "pepper.wav"
    assert handler.ref_text == "Transcript for pepper."
    assert handler.ref_spk is None and handler.ref_rvq is None
    handler.ref_spk = Path("pepper.spk")
    handler._apply_session_voice_override("base", config("custom:pepper"))
    assert handler.ref_spk == Path("pepper.spk")
    handler._apply_session_voice_override("base", config("custom:pepper-upbeat"))
    assert handler.ref_audio.name == "pepper-upbeat.wav"
    assert handler.ref_text == "Transcript for pepper-upbeat."
    assert handler.ref_spk is None
    handler._apply_session_voice_override("base", config("custom:james"))
    assert handler.ref_audio.name == "james.wav"
    assert handler.ref_text == "Transcript for james."


def test_response_override_is_temporary_and_session_end_restores_transcript(handler):
    response = SimpleNamespace(audio=SimpleNamespace(output=SimpleNamespace(voice="custom:pepper-upbeat")))
    handler._apply_session_voice_override("base", config("custom:pepper"), response)
    assert handler.ref_text == "Transcript for pepper-upbeat."
    handler._apply_session_voice_override("base", config("custom:pepper"))
    assert handler.ref_text == "Transcript for pepper."
    handler.on_session_end()
    assert handler.ref_audio.name == "james.wav"
    assert handler.ref_text == "Transcript for james."


def test_public_catalog_matches_initial_reference_without_exposing_paths(handler):
    catalog = handler.voice_catalog()
    assert catalog["default"] == "custom:james"
    assert [v["id"] for v in catalog["voices"]] == ["custom:james", "custom:pepper", "custom:pepper-upbeat"]
    assert all(set(v) == {"id", "name", "kind"} for v in catalog["voices"])


def test_unknown_profile_does_not_modify_reference(handler):
    with pytest.raises(ValueError, match="Unknown"):
        handler._apply_session_voice_override("base", config("custom:missing"))
    assert handler.ref_text == "Transcript for james."


def test_catalog_alone_selects_a_matching_initial_reference(handler, manifest):
    handler.setup(Event(), model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base", voice_profiles=manifest)
    assert handler.ref_audio.name == "james.wav"
    assert handler.ref_text == "Transcript for james."
    assert handler.voice_catalog()["default"] == "custom:james"


def test_custom_voice_models_advertise_supported_presets(handler):
    handler.model = SimpleNamespace(
        config=SimpleNamespace(tts_model_type="custom_voice"), get_supported_speakers=lambda: ["Ryan", "Aiden"]
    )
    assert handler.voice_catalog() == {
        "voices": [
            {"id": "Ryan", "name": "Ryan", "kind": "builtin"},
            {"id": "Aiden", "name": "Aiden", "kind": "builtin"},
        ],
        "default": "Aiden",
    }


def test_generation_receives_each_selected_reference_and_transcript(handler, monkeypatch):
    captured = []
    handler.model.generate = lambda **kwargs: iter(())
    monkeypatch.setattr(handler, "_prepare_mlx_ref_audio", lambda path: str(path))
    monkeypatch.setattr(handler, "_stream_mlx_generation", lambda fn, **kwargs: (captured.append(kwargs), iter(()))[1])
    for voice in ("custom:pepper", "custom:pepper-upbeat", "custom:james"):
        handler._apply_session_voice_override("base", config(voice))
        list(handler._process_voice_clone("Hello Amir."))
        name = voice.removeprefix("custom:")
        assert Path(captured[-1]["ref_audio"]).name == f"{name}.wav"
        assert captured[-1]["ref_text"] == f"Transcript for {name}."
