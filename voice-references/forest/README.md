# Forrest Gump reference

The five original files match the visited links in the supplied screenshot:
`gump.wav`, `love.wav`, `people_call_me.wav`, `rocks.wav`, and `shoes.wav`.
They are mono, 11,025 Hz, 8-bit PCM recordings. Originals are preserved.

`reference.wav` uses the combined **16.88-second** reference: the longer
introduction, the first complete shoe sentence, the rocks line, and the love
line. This is the reference the user preferred in the continuous, single-call
preview. `reference-multi-scene.wav` reproduces the same audio as a stable
comparison filename. The love-only reference is preserved separately as
`reference-single-clip.wav` (6.05 seconds); it produced a continuous preview
but lost the desired voice likeness in the user's listening comparison.
`all-forrest-excerpts.wav` contains all five selected excerpts (19.10 seconds).
The later clause of the shoes clip is excluded. Each output has its own
matching TXT file; always use the corresponding transcript when cloning.

Outputs are mono 24 kHz, 16-bit PCM with 6 ms edge fades, 200 ms gaps,
scene-level volume matching, and a -3 dBFS peak. Cleanup uses a 70 Hz high-pass
filter for low-frequency rumble. RNNoise is disabled for these low-resolution
clips: trials at 50% and 25% made the local ASR's recognition of Forrest's name
worse than the original in the initial mixed reference. The supplied dialogue
corrects ASR's compression of "My name's" into "Mine's" in that comparison.
ASR checks do not establish perceived voice similarity. There is no pitch or
playback-speed change, and resampling cannot
restore detail missing from the original 8-bit files.

`reference-unprocessed.wav` has the same edits without the high-pass filter,
for comparison. The report records source hashes, cut times, levels, and
measured reduction in the quiet pause within `love.wav`. `work/` contains
decoded clips and ASR checks. All audio, transcripts, reports, and work files
stay local; the cut manifest and preparation code are tracked.

Rebuild from the originals:

```sh
.venv/bin/python voice-references/prepare-reference.py --folder voice-references/forest
```

The shared workflow also preserves the existing Pepper outputs. `segments.json`
controls the selection order, edit boundaries, cleanup mix, and transcript.
The registered `custom:forrest` profile appears as **Forrest Gump** in the
demo's **Settings → Voice** selector after the backend reloads its catalog.
It explicitly uses `reference-multi-scene.wav` and its matching transcript,
English as the automatic-language fallback, and MLX seed 42. These match the
approved `preview-single-pass-mixed` generation. The earlier live setup used
automatic language and unseeded sampling, which did not reproduce that preview
even with the same reference. A controlled test reproduced the approved
English output within one PCM quantization step; automatic language produced
a different 7.12-second performance rather than the approved 8.88 seconds.
The live backend also uses a worker-local compiled sampler for seeded profiles:
the import-time MLX sampler otherwise captures the setup thread's random state.
`preview-live-approved-settings.mp3` captures the corrected live passage.
Its waveform correlation with the approved preview is 0.9999996 after accounting
for the pipeline's 24→16→24 kHz resampling and 163 ms leading-silence trim;
the remaining playback gain difference is 0.021 dB.

`preview.wav` and `preview.mp3` preserve the initial 7.78-second preview from
the Claude/Qwen3 pipeline, using the mixed reference and original text for Amir:
"Hello Amir. I hope your day is going just fine. You take your time now, and
we'll get these computers figured out together." These files and their report
stay local. The supplied movie dialogue is used only in the voice reference.
That preview was synthesized in three separate calls, one per sentence; its
approximate sentence levels rose from -22.82 to -20.67 to -17.50 dBFS RMS.

Two controlled comparisons use the same text and seed, with one continuous
synthesis call and overall RMS matched to the initial preview:

- `preview-single-pass-mixed.wav` / `.mp3`: preferred combined reference, 8.88 seconds.
- `preview-consistent.wav` / `.mp3`: continuous love reference, 6.56 seconds.

The default profile retains the preferred combined reference. Local Claude launchers
group up to three sentences per TTS call instead of one. This reduces separate
performances within short replies, at the cost of waiting for more LLM text
before speech starts; longer responses and tool boundaries can still use
multiple calls. Override with `--stream_batch_sentences 1` for earlier startup.
The comparison previews contain no per-sentence gain changes.
The preferred preview's three sentence levels are within approximately 0.5 dB
RMS of each other, compared with the original's approximately 5 dB rise.

Regenerate the preferred comparison in one synthesis call:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python voice-references/generate-preview.py \
  --folder voice-references/forest --text preview.txt --reference reference \
  --output preview-single-pass-mixed --match-level preview.wav
```
