# Pepper reference and Amir demo

`reference.wav` is the recommended starting reference: 23.62 seconds, mono,
24 kHz, 16-bit PCM. Supply its exact matching `reference.txt` through
`--qwen3_tts_ref_text` and the WAV through `--qwen3_tts_ref_audio`. Qwen's Base
voice-cloning interface takes reference audio plus its transcript:
https://github.com/QwenLM/Qwen3-TTS#voice-clone

The reference joins Pepper's opening and closing lines from `trash.wav` with
her full martini request from `olives.wav`. Longer uninterrupted utterances
provide more useful speech with fewer cuts than the short alternating lines
in `plans.wav`. A small repeated “a” in the martini request is retained in
both audio and transcript; the screenshot's simplified text omits it.

`all-pepper-lines.wav` and its matching TXT include Pepper's lines from all
three clips, with Tony's and Christine's dialogue removed. This longer version
is available for comparison, but the shorter reference is the default for
lower reference-processing cost.

`reference-unprocessed.wav` uses the identical edits without the high-pass
filter or neural noise suppression, for an A/B comparison. Noise suppression
can leave residual soundtrack and room noise and can alter breath detail;
these movie clips are not isolated studio recordings. The source WAV files
contain MPEG Layer III audio; output references are ordinary PCM WAV files.

Processing uses a 70 Hz high-pass filter and FFmpeg's RNNoise filter with a
75% processed / 25% original mix, followed by scene-level gain matching,
6 ms edge fades, 200 ms gaps, and peak normalization to -3 dBFS. Playback
speed and pitch are unchanged. The denoiser's approximately 10 ms latency is
removed before applying cuts. The RNNoise model comes from:
https://github.com/GregorR/rnnoise-models

`segments.json` records every source cut and text. `reference-report.json`
records source/model hashes, processing settings, output timing, and levels.
`work/verification.json` contains local Parakeet speech-recognition checks.
Recognition mistakes such as “Sark” for “Stark” were corrected using the
supplied dialogue. Neither ASR nor noise metrics establish subjective voice
similarity; listen to the generated sample to judge that.

## Amir sample

`amir-demo.mp3` is the shareable AI-generated demo. `amir-demo.wav` is the
lossless PCM export of the generation. Its original dialogue is in
`amir-demo.txt`; it is inspired by Pepper's brisk, lightly teasing style.
`amir-demo-report.json` records generation settings and timing.

Regenerate with the already cached local model:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python voice-references/pepper/generate-demo.py
```

Run the voice application with this reference and the Claude backend:

```sh
.venv/bin/python voice-references/pepper/start-with-pepper-claude.py
```

Stop an existing backend first if it uses the same port. CLI flags appended
to the launcher override its defaults.

To reproduce the audio edits:

```sh
.venv/bin/python voice-references/pepper/prepare-reference.py
```

Requires the project's audio dependencies and FFmpeg. Its small RNNoise model
is downloaded on the first run if `work/sh.rnnn` is missing. Originals are
preserved. Recordings, transcripts, generated audio, and reports stay local;
the scripts, cut manifest, and documentation are tracked.

## Upbeat comparison

`amir-demo-upbeat.mp3` and `amir-demo-upbeat.wav` are an alternate take of the
Amir dialogue. The words are the same, with brighter punctuation, and the
9.47-second `reference-upbeat.wav` uses Pepper's birthday exchange and gift
thank-you lines. Its exact transcript is `reference-upbeat.txt`.

The Qwen3-TTS Base model used for reference cloning has no supported natural
language emotion instruction. Qwen's CustomVoice and VoiceDesign variants
support `instruct`, but their controls do not apply to this Base cloning path.
Here, delivery is influenced indirectly by the reference performance and text;
this is a comparison to audition, rather than a guaranteed emotion setting.
No direct emotion instruction is sent to the Base model.

Regenerate this take with:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python voice-references/pepper/generate-demo.py \
  --text amir-demo-upbeat.txt --reference reference-upbeat --output amir-demo-upbeat \
  --match-level amir-demo.wav
```

The upbeat demo is RMS level-matched to the original Amir demo for a fair
listening comparison; the voice is not pitched or time-stretched.
