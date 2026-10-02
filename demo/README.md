---
title: HF Realtime Voice
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Voice chat over WebSocket or WebRTC against HF speech-to-speech
hf_oauth: true
hf_oauth_expiration_minutes: 10080
---

# Realtime Voice Demo

Browser voice-chat UI for the
[huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech)
backend, speaking the OpenAI Realtime **GA** protocol over **WebSocket**
(default) or **WebRTC** (Settings → Transport, env-pinned deploys only — see
[WebRTC transport](#webrtc-transport)).

## Start the configured personal app

From the repository root, run `npm start` (or `npm start --prefix demo`). This
uses `.venv/bin/python` and the existing
[`start-with-my-voice-claude.py`](../voice-references/james/start-with-my-voice-claude.py)
configuration: Claude Sonnet, 2,048 response tokens, low effort, one retry,
Parakeet transcription, and the cached 6-bit Qwen3-TTS model with streaming chunk
size 12 and the local custom voice catalog.

The launcher waits for the backend, then starts the localhost UI on port 7860.
If `~/.cloudflared/speech-to-speech.yml` exists, it also starts the public frontend
on port 7861 and that protected tunnel. Shared conversation persistence is
enabled on both frontends. Cloudflare credentials, recordings, and conversation
files stay local. Logs are saved under `.cache/app-launcher/`. Press **Ctrl+C** to
stop the processes launched by this command.

Use `npm run start:local` to start only the backend and localhost UI, or
`npm run start:check` to check prerequisites without starting services. Backend
flags can override the existing model settings, for example:

```bash
npm start -- --claude_agent_max_tokens 4096
```

The launcher uses cached models by default (`HF_HUB_OFFLINE=1`). It needs the
project Python environment, local James reference files, installed demo npm
dependencies, and `cloudflared` when the tunnel is configured. Optional
`S2S_PYTHON`, `S2S_CLOUDFLARED`, and `S2S_TUNNEL_CONFIG` environment variables
override their locations.

Both choices run one `RealtimeSession` adapter over the pinned official
`@openai/agents` package's stock transport classes. The adapter keeps demo-only
queue, audio, visualization, device, camera, and metering behavior out of the
protocol implementation.

## Conversation and response timings

Conversation is visible by default alongside the voice controls, or below
them on smaller screens. Use its close button or the conversation toolbar
button to hide/show it; this preference persists across reloads. It stays open
while you interact elsewhere, including when dismissing Settings with Escape.

Expand **Server timings** beneath a response to inspect
E2E (estimated speech end to first generated audio), VAD end decision, Smart Turn
decision, transcription, response generation, voice synthesis to first audio,
and hold time before response. E2E is visible in the collapsed summary.
MLX lock wait is available only in macOS terminal logs.
Tool-only responses receive their own timing entry; follow-ups do not repeat STT.

These are server measurements, excluding browser buffering and playback. Stages
can overlap and must not be summed. Missing stages display **Unavailable**. Older
servers, absent metadata, and unsupported or malformed records leave the transcript
unchanged. Server timings become available when the response finishes, including
interrupted, failed, and incomplete responses.

Validation: `npm run test:agents:adapter` checks parsing and adapter behavior;
`npm run test:ui` checks the history display at desktop and phone widths (requires
`npx playwright install chromium`).

## Quick start (local)

1. **Start the speech-to-speech backend** (from the repo root;
   see the [backend README](https://github.com/huggingface/speech-to-speech/blob/main/src/speech_to_speech/api/openai_realtime/README.md)
   for more model combinations):

   ```bash
   uv run speech-to-speech serve \
     --stt parakeet-tdt \
     --llm_backend transformers \
     --tts kokoro \
     --model_name "Qwen/Qwen3-4B-Instruct-2507" \
     --llm_device mps \
     --llm_torch_dtype float16 \
     --enable_live_transcription
   ```

   The realtime server listens on `ws://localhost:8765/v1/realtime` by default
   (`--host` / `--port` to change).

2. **Install the pinned browser SDK and start this app**, pointing it at the
   backend with `SPEECH_TO_SPEECH_URL`:

   ```bash
   npm ci --prefix demo
   uv pip install -r demo/requirements.txt
   export SPEECH_TO_SPEECH_URL=ws://localhost:8765/v1/realtime
   export SERPER_API_KEY=...   # optional; web search is disabled without it
   export STARTUP_GREETING=... # optional; empty disables the automatic greeting
   uv run uvicorn --app-dir demo server:app --reload --port 7860
   ```

   Or with Docker:

   ```bash
   docker build -t s2s-demo demo/
   docker run -p 7860:7860 -e SPEECH_TO_SPEECH_URL=ws://host.docker.internal:8765/v1/realtime s2s-demo
   ```

   > **Docker + host backend: WebSocket and WebRTC need different hostnames.**
   > The two transports dial the backend from different network namespaces:
   >
   > - **WebRTC** is dialed **server-side** — the browser POSTs its SDP offer to
   >   the demo's `/api/calls` proxy, which forwards it from *inside the
   >   container*. There `host.docker.internal` resolves to your host, so the
   >   command above works.
   > - **WebSocket** is dialed **client-side** — the demo hands the URL straight
   >   to the browser, which opens the socket itself. The browser runs on your
   >   *host*, where `host.docker.internal` is not a real DNS name, so the
   >   connection never reaches the backend and the server logs nothing.
   >
   > Set `SPEECH_TO_SPEECH_PUBLIC_URL=ws://localhost:8765/v1/realtime` for
   > the browser while keeping `SPEECH_TO_SPEECH_URL` pointed at
   > `host.docker.internal`. Without Docker, the same localhost URL works
   > for both.

3. Open <http://localhost:7860/>, click the orb, allow the mic, talk.

> Browsers require **HTTPS or `localhost`** for `getUserMedia()` (mic + camera).
> `127.0.0.1` and `localhost` both work; plain `http://192.168.x.y` does NOT.

### Protected Cloudflare Tunnel

Keep the speech backend on loopback. Configure a self-hosted Cloudflare Access
application for the **entire public hostname**, with an Allow policy limited to
your login email, before publishing DNS or starting the tunnel. Enable an
identity provider (for example, email one-time PIN). The UI, API routes, and
WebSocket must share that protected hostname so browser authentication cookies
also protect voice connections.

Start a separate demo instance without disturbing the local UI or running agents:

```bash
export SPEECH_TO_SPEECH_URL=ws://127.0.0.1:8765/v1/realtime
export SPEECH_TO_SPEECH_PUBLIC_URL=wss://voice.example.com/v1/realtime
export SPEECH_TO_SPEECH_RTC=false
export SPEECH_TO_SPEECH_SHARED_CONVERSATION=true
export SPEECH_TO_SPEECH_CLIENT_TOOLS=false # SDK-backed deployment; native agent tools remain enabled
uv run uvicorn --app-dir demo server:app --host 127.0.0.1 --port 7861
```

The public URL is advertised to the browser only; voice catalogs, session lists,
and permission controls still use the private `SPEECH_TO_SPEECH_URL`. Disable
WebRTC for this deployment because the HTTP tunnel carries WebSockets but does
not carry the direct peer connection's UDP media.

`SPEECH_TO_SPEECH_SHARED_CONVERSATION=true` makes this a personal deployment with
one server-owned ongoing conversation. Enable it on the localhost frontend too:
both addresses and new browsers then load the same transcript before microphone
startup. On first use the server adopts its most recently saved conversation.
The durable `current.json` pointer stays fixed across reconnects and service or
machine restarts. **Settings → Start a new conversation** rotates it deliberately
for all browsers; changing the backing LLM clears its context and transcript.
Only one browser can attach its microphone to a conversation at a time. Idle
pages refresh the shared transcript without occupying a speech slot. This mode
is opt-in because all visitors to that deployment share the same conversation;
keep it protected with the personal Access policy above.

Use the example [tunnel configuration](cloudflared.example.yml), substituting
your tunnel ID, credentials path, hostname, Access team name, and application AUD
tag. Its origin validation rejects requests without a valid Access JWT on both
routes. Keep credentials and the populated deployment config outside the repo.

```bash
cloudflared tunnel --origincert ~/.cloudflared/personal-cert.pem create speech-to-speech
# Create and verify Access protection first, then publish the hostname:
cloudflared tunnel --origincert ~/.cloudflared/personal-cert.pem route dns speech-to-speech voice.example.com
cloudflared tunnel --config ~/.cloudflared/speech-to-speech.yml ingress validate
cloudflared tunnel --config ~/.cloudflared/speech-to-speech.yml run speech-to-speech
```

Confirm unauthenticated requests to `/`, `/api/config`, and `/v1/realtime` receive
the Access login redirect or a denial. Then sign in and test the microphone and
voice response over HTTPS. The Mac and local backend must remain running.

Smoke-test the backend from the shell:

```bash
websocat ws://localhost:8765/v1/realtime
# -> you should get a session.created event back immediately
```

## How it works

1. The adapter creates an official Agents SDK `RealtimeSession` with the stock
   WebSocket transport on the configured `/v1/realtime` URL.
2. The SDK performs `session.update` using the OpenAI Realtime **GA** schema.
3. The SDK streams mic audio as PCM16 24 kHz mono base64 chunks
   (`input_audio_buffer.append`, one frame every ~40 ms).
4. The browser keeps a bounded, in-memory copy of those sent frames and uses
   the server VAD boundaries to add replayable user recordings to conversation
   history. No recording is uploaded or persisted separately.
5. Server pushes `response.output_audio.delta` (PCM16 24 kHz mono base64)
   and transcript deltas.

The backend exposes one concurrent session per pipeline unit
(`--num_pipelines` to serve more).

## WebRTC transport

With `SPEECH_TO_SPEECH_URL` set, **Settings → Transport** offers WebRTC as an
alternative to the WebSocket. Same conversation, different plumbing:

1. The SDK's stock WebRTC transport adds the browser mic track and its data
   channel to an `RTCPeerConnection` and POSTs the SDP offer to the same-origin
   `/api/calls` proxy, which forwards it to the backend's
   `POST /v1/realtime/calls` (the OpenAI GA handshake). The proxy exists
   because the s2s server has no CORS middleware — and it forwards **only**
   to the env-pinned URL, never to a client-supplied one, so it can't be used
   as an open proxy. That's why the toggle is locked to WebSocket when the
   URL isn't pinned (user-typed URLs, LB mode).
2. Only the handshake goes through the proxy: the negotiated audio (Opus RTP
   both ways) and the data channel flow directly browser ↔ backend.
3. JSON events on the data channel are the same GA protocol as the WebSocket,
   minus the audio: mic audio rides the media track (never
   `input_audio_buffer.append`, which the backend rejects over WebRTC), and
   the assistant's voice arrives as a remote audio track (never
   `response.output_audio.delta`). Barge-in flushing is server-side.

Backend requirement: the `webrtc` extra
(`pip install "speech-to-speech[webrtc]"`), otherwise `/v1/realtime/calls`
answers 501 and the handshake fails with a clear message.

Caveats vs. WebSocket:

- **User recording replay**: conversation-history recordings currently use the
  exact PCM frames sent through `input_audio_buffer.append`, so they are
  available only on the WebSocket transport.
- **NAT**: host ICE candidates only by default — fine when browser and backend
  are on the same machine/LAN. Across the internet, set `RTC_ICE_SERVERS` on
  *this* app (a JSON list of `RTCIceServer` dicts, or comma-separated
  STUN/TURN URLs; served to the browser via `/api/config`) and
  `SPEECH_TO_SPEECH_ICE_SERVERS` on the backend. There is no TURN relay
  fallback, so symmetric-NAT setups may still not connect.
- **Noise gate**: implemented in the WebSocket capture worklet, so it's
  hidden on WebRTC — the raw mic track (with the browser's own
  `noiseSuppression`) is sent instead.
- **Camera snapshots** are re-encoded to fit one data-channel message
  (~60 KB), so the model may see a smaller frame than over WebSocket.
- **Load-balancer mode is WebSocket-only** for now.

## Connecting to a backend

Three modes, picked by env (`/api/config` tells the client which one is active):

- **`SPEECH_TO_SPEECH_URL` env** — the mode you want for local use, and the
  highest priority. The browser connects **directly** to this realtime
  WebSocket URL; it's shown read-only in Settings. Setting it disables the
  load-balancer logic entirely (no `/api/session` proxy, no queue, no
  metering, no sign-in). Unlike the LB address it is not a secret. Accepts a
  full `ws(s)://host/v1/realtime` URL or a bare host like `localhost:8765`
  (the app adds `/v1/realtime`).
- **Neither env set** — **Settings → Speech-to-speech server URL**: paste a
  full connect URL or a bare host, and the browser connects to it directly.
- **`LOAD_BALANCER_URL` env** — multi-compute deployments only: the browser
  POSTs the same-origin `/api/session` proxy, the server forwards to the LB,
  and the browser dials the per-session compute URL the LB hands back. The LB
  address never reaches the browser; the Settings URL field is hidden. On
  OAuth-enabled Spaces, the proxy forwards the signed-in user's HF access token
  to the LB through `X-Reachy-Mini-Authorization` so the backend can attribute
  usage. The token stays server-side; anonymous requests include no credential.

| `SPEECH_TO_SPEECH_URL` | `LOAD_BALANCER_URL` | `SPACE_ID` | Connection | URL field | Transport | Metering |
|:---:|:---:|:---:|---|---|---|---|
| ✅ | any | any | direct → pinned URL | visible, locked | WS or WebRTC | off |
| – | – | any | direct → user URL | editable | WS only | off |
| – | ✅ | ✅ | LB proxy | hidden | WS only | **on** |
| – | ✅ | – | LB proxy | hidden | WS only | off |

**Settings → Restart** reconnects with the current voice, instructions and URL.

## Startup greeting

By default, each new connection creates one hidden user item asking the model
for a brief greeting, then requests a response. Besides opening the conversation
naturally, this warms the same prompt prefix used by the first spoken turn.

Set `STARTUP_GREETING` to customize the hidden prompt, or set it to an empty
value to disable automatic generation. The adapter sends it once through the
new `RealtimeSession` after connection.

## Tools

The assistant can call two tools mid-conversation (toggle them from the **Tools**
button, top-right):

- **Web search** — Google results via Serper.dev, proxied server-side so the key
  never reaches the browser. Set `SERPER_API_KEY` as an env var / Space secret.
  Without it, the tool is disabled unless the user pastes their own key in the
  Tools panel.
- **Camera** — while enabled, a live self-view shows bottom-left; when the model
  calls the tool, the current frame is sent to the vision-language model so it can
  see what you're showing it.

## Usage limits (deployed Space only)

Conversation time is metered per UTC day by sign-in tier (see `limiter.py` /
`auth.py`), but **only on the deployed Space** — metering turns on only when BOTH
`LOAD_BALANCER_URL` and `SPACE_ID` (injected automatically by the HF Space
runtime) are present. Running locally — even with `LOAD_BALANCER_URL` exported —
leaves the app unmetered. Tunable via env:

| Env | Default | What |
|-----|---------|------|
| `LIMIT_ANON_SEC` | `300` | Daily seconds for anonymous visitors (5 min) |
| `LIMIT_FREE_SEC` | `600` | Daily seconds for signed-in non-PRO users (10 min) |
| `LB_HF_TOKEN` | _(falls back to user OAuth)_ | Optional Space secret sent in standard `Authorization` to authenticate requests at the HF Inference Endpoint ingress; per-user attribution continues through `X-Reachy-Mini-Authorization` |
| `UNLIMITED_ORGS` | _(adds to defaults)_ | Extra HF org names whose members get **unlimited** usage, like PRO |
| `USAGE_HASH_SECRET` | _(random)_ | HMAC secret for hashing identity keys + signing the anon cookie |

PRO members are always unlimited. Members of `cerebras`, `HuggingFaceM4`,
`smolagents`, and `pollen-robotics` are unlimited out of the box (shown as
"Team", not "PRO"); set `UNLIMITED_ORGS=my-team` to add more. Matched
case-insensitively against the user's organisations from HF OAuth.

## Settings (stored in `localStorage`)

| Key | What |
|-----|------|
| Speech-to-speech server URL | Direct realtime WebSocket URL (hidden/locked when pinned by env) |
| Transport | WebSocket (default) or WebRTC; selectable only with an env-pinned URL |
| Microphone | Input device for capture. Applies on the next conversation / Restart. |
| Speakers | Output device for assistant audio. Chrome/Edge can switch live via `AudioContext.setSinkId`; other browsers keep the system default. |
| Voice | Qwen3-TTS speaker name (Aiden, Ryan, Dylan, Eric, Ono_Anna, Serena, Sohee, Uncle_Fu, Vivian) |
| Instructions | System prompt sent in `session.update` once the connection opens |

LocalStorage keys are namespaced `s2s.ws.*` (plus `s2s.transport` for the
transport pick, and `s2s.audio.inputId` / `s2s.audio.outputId` for devices).

## Files

| File | Role |
|------|------|
| `index.html` | Single page, orb + settings modal (identical UI to the WebRTC app) |
| `main.js` | State machine, settings, tools, camera, noise-gate UI wiring |
| `ui/chat.js` | `ChatView`: history panel, ephemeral bubbles, transcript/tool streaming, user recording replay |
| `ui/account.js` | `Account`: HF login chip + popover, daily-limit modal |
| `ui/dom.js` | Shared helpers: `$`, `escHtml`, `truncateError`, `DEBUG` |
| `auth.py` | HF OAuth + per-request identity (tier, hashed keys) |
| `limiter.py` | SQLite per-day talk-time budget (chunked server-clock reservation) |
| `s2s-realtime-client.js` | Narrow demo adapter around one Agents SDK `RealtimeSession` and SDK transports with browser-owned WebSocket truncation |
| `package.json` / `package-lock.json` | Exact official Agents SDK and browser-test dependency pins |
| `ws/codec.js` | base64 <-> PCM helpers + transcript extraction (pure) |
| `ws/user-audio-recorder.js` | Bounded sent-PCM buffer + VAD slicing + browser-playable WAV wrapping |
| `ws/orb-visualizer.js` | `OrbVisualiser`: FFT bands -> orb CSS custom properties |
| `worklets/mic-capture.js` | AudioWorklet: 48 kHz Float32 -> 24 kHz Int16 PCM, posts ~40 ms chunks |
| `worklets/audio-playback.js` | AudioWorklet: 24 kHz Float32 ring buffer -> 48 kHz, linear interp, fade in/out |
| `style.css` | Orb animations, layout, dark theme (verbatim from the WebRTC app) |

## Audio pipeline notes

- **WebSocket startup buffer**: Settings → Playback startup buffer (ms) controls
  how much assistant audio is accumulated before each response starts playing.
  The default is 0 (immediate playback). The setting is saved in this browser
  and applies on the next conversation; it does not affect WebRTC or the Python
  client's `--playback-buffer-ms` setting. Missing, invalid, or negative values
  fall back to the default.
  Buffering counts PCM samples, not elapsed time. After the threshold is reached,
  later chunks stream immediately without rebuffering. Completed short responses
  and valid incomplete responses release their remaining audio; interruption,
  cancellation, failure, and disconnect discard pending audio. Each new response
  gets a fresh startup gate without cutting off already released audio. A failed
  or cancelled response discards only its private startup buffer; already released
  audio drains unless an interruption or disconnect clears the shared queue.
  In [issue #557](https://github.com/huggingface/speech-to-speech/issues/557),
  1200 ms resolved glitches in the reporter's local TTS setup. This is a tuning
  example, not a universal optimum: larger values add startup latency, and no
  finite reserve can prevent all underruns from sustained slower-than-realtime
  generation.
- **Input**: `getUserMedia({ echoCancellation, noiseSuppression, autoGainControl })`
  feeds the `mic-capture` worklet at the `AudioContext` rate. The worklet
  resamples to 24 kHz (boxcar lowpass + decimation on the 48 -> 24 fast
  path, linear interpolation fallback for odd rates) and packs Int16 LE.
- **Browser cache safety**: the entry module, realtime client, and both audio
  worklet URLs share the `audio-24k-v2` cache key. The client also waits for
  the capture worklet to report the same version and a 24 kHz output rate
  before opening a session. When the browser-audio contract or sample rate
  changes, bump the key in `index.html`, `main.js`, and
  `AUDIO_WORKLET_VERSION` in `s2s-realtime-client.js` together.
- **User replay**: the WebSocket client retains only a bounded copy of PCM it
  actually sends. `speech_started` / `speech_stopped` timestamps select each
  utterance, which is wrapped as an in-memory WAV and attached to the user row.
  Starting playback temporarily mutes outgoing mic audio to prevent feedback.
- **Output**: `response.output_audio.delta` decodes to Int16 -> Float32
  and is posted to the `audio-playback` worklet. The worklet maintains a
  per-context ring buffer, linearly interpolates 24 -> 48, and applies
  short 32-frame fades on entry/exit to suppress clicks.
- **Barge-in**: server VAD and explicit SDK interruptions clear the WebSocket
  playback queue. The worklet acknowledges the clear with rendered PCM counts
  per item/content part, excluding startup delay and underrun silence. The client
  sends `conversation.item.truncate` for unheard audio using those counts (zero
  before playback starts), retaining identities even after audio/response done.
  This overrides the SDK's receipt-time interruption clock. Server VAD cancels
  the in-flight response; explicit interruptions use the SDK cancellation hook.
  The local server currently accepts truncation events without changing its
  conversation history; these client counts do not establish server-side truncation.
  See the [Realtime interruption guidance](https://developers.openai.com/api/docs/guides/realtime-conversations#handling-interruptions).

## Credits

- Backend: [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech)
- UI verbatim from `amir-tfrere/minimal-conversation-app-s2s-backend` (Pollen Robotics × Hugging Face)

For a backend that executes its own tools, such as `claude-agent-sdk`, set
`SPEECH_TO_SPEECH_CLIENT_TOOLS=false` on the demo server. The demo then omits
browser search/camera function definitions; the backend's native tools remain
available.

### Custom voice choices

For Qwen3-TTS Base cloning, register voices on the backend with
`--qwen3_tts_voice_profiles /path/to/voices.json` (see the main README).
The demo loads the pinned backend's catalog through `GET /api/voices` and
shows it in **Settings → Voice**. **Save** changes the active session's voice
and remembers the choice. Unsupported saved presets fall back to the backend's
default; Base profiles each use their own reference audio and transcript.

Direct and load-balanced WebSocket connections, as well as WebRTC, also
discover choices after connecting: the demo negotiates the
`speech_to_speech.voices` session extension and receives a same-named event
with `voices` (`id`, `name`, `kind`) and `default`. The HTTP proxy forwards
only to `SPEECH_TO_SPEECH_URL`, never to browser-supplied addresses. Older
backends retain the existing preset selector.

### Claude tool approvals

Use the sidebar's **Sessions** tab to see app-owned independent sessions and
other live Claude sessions. It refreshes activity every two seconds while visible,
shows directories and the latest app-owned reply, and marks sessions linked to
the voice agent. Select a session to send it a follow-up through the connected
voice agent and its usual permissions. Conversation and Sessions share the
nonmodal sidebar; the orb, microphone and settings remain usable. Managed
sessions survive voice-call disconnects until stopped or the speech backend exits.
The read-only `/api/agent-sessions` proxy only accesses `SPEECH_TO_SPEECH_URL`;
external discovery requires a Claude CLI supporting `agents --json`.

With the `claude-agent-sdk` backend, pending native tool requests appear in a
Claude requests panel with exact tool inputs and **Allow once** / **Deny**.
You can also say **“approve request”** or **“deny request”** while one request is
pending and your microphone is unmuted. These replies resume the same agent
response. With multiple requests, use the buttons to identify the request.
`AskUserQuestion` provides answer fields; a single question accepts a spoken
answer as well. The server denies unanswered prompts after
`--claude_agent_permission_timeout_s` (300 seconds by default).

Both WebSocket and WebRTC carry the following custom events. Clients receive
`speech_to_speech.agent.permission.requested` (`request_id`, `response_id`,
`tool_name`, `input`, `title`, `description`, `timeout_s`) and reply using:

```json
{
  "type": "speech_to_speech.agent.permission.reply",
  "event_id": "my-reply",
  "request_id": "the-request-id",
  "decision": "allow"
}
```

Use `"deny"` to decline. For `AskUserQuestion`, an allow reply also includes
`"answers"`, mapping each question's full text to an answer string or an array
of choices for a multi-select question. Replies are scoped to the current
response and session; stale or duplicate replies produce a correlated error.
`speech_to_speech.agent.permission.resolved` reports `allowed`, `denied`,
`timed_out`, or `cancelled`. Spoken replies produce
`speech_to_speech.agent.permission.voice` with the recognized transcript and
an error if it did not resolve the request. Pending cards clear when the
response ends or the call disconnects. Normal Claude permission rules may
already approve or deny actions without an interactive prompt.
