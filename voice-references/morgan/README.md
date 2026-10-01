# Morgan local voice reference

The supplied `from_a_tiktok.mp4` is preserved unchanged. It contains a
51.35-second continuous narration; the registered **Morgan** voice uses
14.72–31.36 seconds of it, retaining six consecutive sentences and their natural
pauses. Local Parakeet alignment guided the cut boundaries, and transcription
of the prepared reference matched all the reference words.

`reference.wav` is 16.91 seconds, mono, 24 kHz, 16-bit PCM. Preparation uses a
70 Hz high-pass filter and a light 25% RNNoise wet/dry mix, six-millisecond edge
fades, short end padding, and peak normalization to −3 dBFS. Speed and pitch
are unchanged. The measured source pause at 6.8–7.5 seconds lost about 4.2 dB
of background level. `reference-unprocessed.wav` retains the same cut without
the filters for comparison; this workflow reduces the background, rather than
fully separating speech from music.

Rebuild the reference with the original video in this directory:

```sh
.venv/bin/python voice-references/morgan/prepare-reference.py
```

The wrapper decodes the video and calls the shared manifest-based preparation
script. `segments.json` defines the cut, transcript, and cleanup settings.
`reference-report.json` records processing measurements and the original
video's SHA-256 hash. The RNNoise model is reused from the local Pepper cache.

`preview.mp3` and `preview.wav` are an 11.68-second, three-sentence generated
sample addressed to Amir, synthesized in one call. Regenerate with:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python voice-references/generate-preview.py \
  --folder voice-references/morgan --text preview.txt --output preview \
  --language english --seed 42
```

The **Settings → Voice → Morgan** profile uses this reference and its matching
`reference.txt`, English as the automatic-language fallback, and seed 42 for
repeatable MLX sampling. The video, decoded audio, transcripts, generated
previews, work files, and reports stay local and are excluded from Git.
