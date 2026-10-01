# Local voice references

`voices.json` registers **James**, **Pepper**, **Pepper (upbeat)**, and **Forrest Gump** for the
settings voice selector. Every entry points to a local WAV and its matching
transcript; the recordings, transcripts, generated samples, and processing
reports are excluded from Git. Keep those files in place on this computer.
The launchers and reproducible voice preparation scripts are tracked.

Start the Claude-backed app with James as the initial default:

```sh
.venv/bin/python voice-references/james/start-with-my-voice-claude.py
```

Then open the demo, choose **Settings → Voice**, select a voice, and **Save**.
The setting persists and applies to upcoming speech without a server restart.
The Pepper launcher starts with Pepper as its default and offers the same
choices. Only one backend should occupy the realtime port at a time.
Claude launchers group up to three sentences per speech generation to reduce
voice and level changes between separate performances. For minimum startup
latency, append `--stream_batch_sentences 1`.

To add a voice, create a directory with its reference audio and exact transcript
and add an entry to `voices.json`, giving it a unique `custom:` ID. Restart the
backend to load a changed catalog, then refresh the demo. Paths in the catalog
are relative to this directory. The local recordings must be supplied separately
when using these launchers on another computer.

See [Pepper's workflow](pepper/README.md) for the editing and sample-generation
steps. Its upbeat variant uses a different reference recording; Qwen3-TTS Base
does not expose the direct emotional instructions of CustomVoice/VoiceDesign.
See [Forrest's workflow](forest/README.md) for his reference edits and cleanup.
