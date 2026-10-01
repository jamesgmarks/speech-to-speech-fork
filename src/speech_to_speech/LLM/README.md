# LLM Summary

## Available LLM backends (`--llm_backend`)

Runtime-supported values in `s2s_pipeline.py`:

- `transformers` → `language_model.py` (Transformers backend)
- `mlx-lm` → `language_model.py` (MLX backend)
- `responses-api` → `responses_api_language_model.py`
- `chat-completions` → `chat_completions_language_model.py`
- `claude-agent-sdk` → `claude_agent_sdk_language_model.py`

## Usage

### 1) Transformers (`--llm_backend transformers`)

- Handler: `LanguageModelHandler`
- Typical use: local GPU/CPU inference using Hugging Face Transformers
- Backend-specific args prefix: `--llm_*`
- Shared args (from base): `--model_name`, `--chat_size`, `--init_chat_prompt`, `--enable_lang_prompt`

```bash
speech-to-speech serve \
  --llm_backend transformers \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --llm_device cuda \
  --llm_torch_dtype float16 \
  --llm_gen_max_new_tokens 128
```

Common options:
- `--llm_gen_min_new_tokens`
- `--llm_gen_temperature`
- `--llm_gen_do_sample`
- `--chat_size`
- `--init_chat_prompt`

### 2) MLX-LM (`--llm_backend mlx-lm`)

- Handler: `LanguageModelHandler`
- Typical use: Apple Silicon local inference
- Backend-specific args prefix: same as Transformers (`--llm_*`)

```bash
speech-to-speech serve \
  --llm_backend mlx-lm \
  --model_name mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --llm_device mps \
  --llm_gen_max_new_tokens 128
```

Common options:
- `--llm_gen_temperature`
- `--llm_gen_do_sample`
- `--chat_size`
- `--init_chat_prompt`

### 3) OpenAI-compatible API (`--llm_backend responses-api`)

- Handler: `ResponsesApiModelHandler`
- Typical use: remote model serving via OpenAI-compatible endpoints
- Backend-specific args prefix: `--responses_api_*`
- Shared args (from base): `--model_name`, `--chat_size`, `--init_chat_prompt`, `--enable_lang_prompt`

```bash
speech-to-speech serve \
  --llm_backend responses-api \
  --model_name gpt-5.6-terra \
  --responses_api_api_key YOUR_API_KEY \
  --responses_api_base_url https://api.example.com/v1 \
  --responses_api_stream true
```

Common options:
- `--chat_size`
- `--init_chat_prompt`
- `--user_role`

## LLM Behavior

When STT is set to language auto-detection (`--language auto`), LLM handlers can receive `(text, language_code)` and prepend a language control instruction like:

- `Please reply to my message in <language>.`

This helps the assistant respond in the detected language. The behavior is opt-in via `--enable_lang_prompt` (shared across all backends); it defaults to `False`.

## Setup

### CUDA setup

```bash
speech-to-speech serve \
  --llm_backend transformers \
  --model_name microsoft/Phi-3-mini-4k-instruct
```

### Local Mac setup

```bash
speech-to-speech local \
  --mac-optimal-settings \
  --model_name mlx-community/Qwen3-4B-Instruct-2507-4bit
```

`--mac-optimal-settings` sets `--llm_backend mlx-lm` and defaults the model to `mlx-community/Qwen3-4B-Instruct-2507-4bit` if not overridden. The command independently selects whether to run only the server or compose it with the audio client.

### Realtime (OpenAI-compatible) setup

Run the server, then connect with the packaged audio client:

```bash
# 1. Start the pipeline server
speech-to-speech serve \
  --llm_backend mlx-lm \
  --model_name mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --host 0.0.0.0 \
  --port 8765

# 2. Connect with the audio client
speech-to-speech talk --url ws://127.0.0.1:8765/v1/realtime
```

Or with `--mac-optimal-settings` on Apple Silicon:

```bash
speech-to-speech serve \
  --mac-optimal-settings \
  --host 0.0.0.0 \
  --port 8765
```

### Remote API setup

```bash
speech-to-speech serve \
  --llm_backend responses-api \
  --model_name gpt-5.6-terra \
  --responses_api_api_key YOUR_API_KEY
```

## Claude Code Agent SDK

Install `speech-to-speech[claude-agent-sdk]` and select
`--llm_backend claude-agent-sdk`. All Claude Code tools are available with
normal Claude Code permission rules. `--claude_agent_cwd` selects the workspace;
`--claude_agent_allowed_tools` supplies auto-approval rules without restricting
the toolset. CLI authentication/provider environment is inherited by the SDK.
Pending tool approvals are sent to the connected client through `can_use_tool`.
The demo offers Allow once/Deny buttons and explicit spoken approvals; agent
questions accept form answers or a spoken answer when there is one question.
`--claude_agent_permission_timeout_s` bounds human wait (default 300 seconds),
which is excluded from the generation deadline. No responder, timeout, or
cancellation grants approval.

Tune `--claude_agent_max_tokens`, `--claude_agent_effort`,
`--claude_agent_thinking`, `--claude_agent_thinking_budget_tokens`,
`--claude_agent_max_turns`, `--claude_agent_request_timeout_s`, and
`--claude_agent_max_retries` to manage response time. Streaming defaults to
one sentence per spoken batch. Output-token limits apply per model request,
including thinking and tool-call output; Realtime token-limit overrides are
also supported.

Each public response starts a fresh SDK client with a snapshot of the pipeline
conversation. Internal SDK tool transcripts are not persisted across voice
turns, and Realtime client function tools, images, and direct audio input are
unsupported. Hidden response prefetch is skipped. See the
[main README](../../../README.md#claude-code-agent-sdk) for examples and all
configuration options.
