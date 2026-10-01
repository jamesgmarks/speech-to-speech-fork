# Forrest Gump reference

The five original files match the visited links in the supplied screenshot:
`gump.wav`, `love.wav`, `people_call_me.wav`, `rocks.wav`, and `shoes.wav`.
They are mono, 11,025 Hz, 8-bit PCM recordings. Originals are preserved.

`reference.wav` joins four longer excerpts into a **16.88-second** reference:
the longer introduction, the first complete shoe sentence, the rocks line,
and the love line. The short greeting is included in `all-forrest-excerpts.wav`
(19.10 seconds) as an alternative; using both introductions adds repetition.
The later clause of the shoes clip is excluded. Each output has an exact
matching TXT file; always use its own transcript when cloning.

Outputs are mono 24 kHz, 16-bit PCM with 6 ms edge fades, 200 ms gaps,
scene-level volume matching, and a -3 dBFS peak. Cleanup uses a 70 Hz high-pass
filter for low-frequency rumble. RNNoise is disabled for these low-resolution
clips: trials at 50% and 25% made the local ASR's recognition of Forrest's name
worse than the original. The milder reference retains the name recognition;
ASR still compresses "My name's" into "Mine's", so the supplied dialogue is
used to correct that transcript. ASR checks do not establish perceived voice
similarity. There is no pitch or playback-speed change, and resampling cannot
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

`preview.wav` and `preview.mp3` are a 7.78-second AI-generated preview from
the running Claude/Qwen3 pipeline, using this profile and original text for Amir:
"Hello Amir. I hope your day is going just fine. You take your time now, and
we'll get these computers figured out together." These files and their report
stay local. The supplied movie dialogue is used only in the voice reference.
