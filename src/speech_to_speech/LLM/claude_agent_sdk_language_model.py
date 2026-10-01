"""Claude Code Agent SDK transport for the shared speech LLM pipeline.

A request owns one SDK client in one asyncio task. The pipeline Chat is the
source of truth: its snapshot is supplied as a transcript on every request,
without resuming a CLI session that could retain cancelled or out-of-band work.
Claude executes its own tools; they are not Realtime client function calls.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Generator, Iterator
from queue import Empty, Full, Queue
from threading import BoundedSemaphore, Event, Lock, Thread
from time import monotonic
from typing import Any

from claude_agent_sdk import (
    AssistantMessage as SDKAssistantMessage,
)
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
)
from claude_agent_sdk.types import StreamEvent
from openai.types.realtime.realtime_conversation_item_assistant_message import Content as AssistantContent

from speech_to_speech.agent_interactions import AgentPermissionRequested, AgentPermissionResolved
from speech_to_speech.LLM.base_openai_compatible_language_model import (
    AssistantMessage,
    BaseOpenAICompatibleHandler,
    ProviderEvent,
    TextDelta,
    Usage,
    _Turn,
)
from speech_to_speech.LLM.chat import Chat
from speech_to_speech.LLM.compaction_prompt import CompactGenerateFn
from speech_to_speech.pipeline.events import AgentPermissionEvent
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut
from speech_to_speech.pipeline.messages import EndOfResponse


class _ClaudeStream:
    """Bounded async-to-sync bridge with prompt cancellation during silent tools."""

    def __init__(
        self,
        options: ClaudeAgentOptions,
        prompt: str,
        timeout: float,
        cancelled: Callable[[], bool],
        slots: BoundedSemaphore,
        waiting: Callable[[], bool] = lambda: False,
    ):
        if not slots.acquire(timeout=0.1):
            raise RuntimeError("The previous Claude SDK request is still shutting down. Please retry.")
        self.options = options
        self.prompt = prompt
        self.timeout = timeout
        self.cancelled = cancelled
        self.waiting = waiting
        self.closed = Event()
        self.finished = Event()
        self.connected = Event()
        self.results: Queue[ProviderEvent | BaseException] = Queue(maxsize=16)
        self.worker = Thread(target=self._run, args=(slots,), name="claude-agent-sdk", daemon=True)
        try:
            self.worker.start()
        except BaseException:
            slots.release()
            raise

    async def _publish(self, event: ProviderEvent | BaseException) -> None:
        while not self.closed.is_set():
            try:
                self.results.put_nowait(event)
                return
            except Full:
                await asyncio.sleep(0.01)

    async def _session(self) -> None:
        # connect/disconnect must share their owning task: the SDK uses AnyIO
        # task groups and cancel scopes internally.
        async with ClaudeSDKClient(options=self.options) as client:
            self.connected.set()
            if self.closed.is_set() or self.cancelled():
                return
            await client.query(self.prompt)

            async def receive() -> None:
                streamed_text = ""
                completed = False
                async for message in client.receive_response():
                    if getattr(message, "parent_tool_use_id", None) is not None:
                        continue
                    if isinstance(message, StreamEvent):
                        event = message.event
                        if event.get("type") == "message_start":
                            streamed_text = ""
                        delta = event.get("delta", {})
                        if event.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            streamed_text += text
                            await self._publish(TextDelta(text=text))
                    elif isinstance(message, SDKAssistantMessage):
                        if message.error:
                            raise RuntimeError(f"Claude assistant error: {message.error}")
                        text = "".join(block.text for block in message.content if isinstance(block, TextBlock))
                        # Complete messages also arrive after their partial deltas.
                        # Fallback handles CLI versions that don't emit partials.
                        remaining = text[len(streamed_text) :] if text.startswith(streamed_text) else ""
                        if remaining:
                            await self._publish(TextDelta(text=remaining))
                        if text:
                            await self._publish(
                                AssistantMessage(content=[AssistantContent(type="output_text", text=text)])
                            )
                        streamed_text = ""
                    elif isinstance(message, ResultMessage):
                        completed = True
                        usage = message.usage or {}
                        await self._publish(
                            Usage(
                                input_tokens=sum(
                                    usage.get(key, 0) or 0
                                    for key in (
                                        "input_tokens",
                                        "cache_read_input_tokens",
                                        "cache_creation_input_tokens",
                                    )
                                ),
                                output_tokens=usage.get("output_tokens", 0) or 0,
                            )
                        )
                        if message.is_error or message.subtype != "success":
                            detail = "; ".join(message.errors or []) or message.result or message.subtype
                            raise RuntimeError(f"Claude agent response failed: {detail}")
                if not completed:
                    raise RuntimeError("Claude SDK stream ended without a result message.")

            receiver = asyncio.create_task(receive())
            try:
                while not receiver.done():
                    if self.closed.is_set() or self.cancelled():
                        await asyncio.wait_for(client.interrupt(), timeout=2.0)
                        return
                    await asyncio.wait({receiver}, timeout=0.05)
                await receiver
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)

    def _run(self, slots: BoundedSemaphore) -> None:
        async def run() -> None:
            session = asyncio.create_task(self._session())
            previous = monotonic()
            deadline = previous + self.timeout
            try:
                while not session.done():
                    now = monotonic()
                    if self.waiting():
                        deadline += now - previous
                    previous = now
                    if now >= deadline:
                        raise TimeoutError(f"Claude agent response timed out after {self.timeout:g}s.")
                    # Before connection, there is no client to interrupt yet.
                    # Cancelling its owning task also cleans failed SDK startup.
                    if not self.connected.is_set() and (self.closed.is_set() or self.cancelled()):
                        return
                    await asyncio.wait({session}, timeout=0.05)
                await session
            except Exception as exc:
                session.cancel()
                await asyncio.gather(session, return_exceptions=True)
                await self._publish(exc)
            finally:
                session.cancel()
                await asyncio.gather(session, return_exceptions=True)

        try:
            asyncio.run(run())
        finally:
            self.finished.set()
            slots.release()

    def __iter__(self) -> Iterator[ProviderEvent]:
        try:
            while not self.closed.is_set() and not self.cancelled():
                try:
                    result = self.results.get(timeout=0.05)
                except Empty:
                    if self.finished.is_set():
                        return
                    continue
                if isinstance(result, BaseException):
                    raise result
                yield result
        finally:
            self.close()

    def close(self) -> None:
        self.closed.set()
        self.worker.join(timeout=3.0)


class ClaudeAgentSDKModelHandler(BaseOpenAICompatibleHandler):
    """Use Claude Code's agent loop while retaining pipeline output semantics."""

    def setup(  # type: ignore[override]
        self,
        *,
        model_name: str = "sonnet",
        max_tokens: int | None = None,
        effort: str | None = None,
        thinking: str | None = None,
        thinking_budget_tokens: int = 1024,
        max_turns: int | None = None,
        max_budget_usd: float | None = None,
        request_timeout_s: float = 120.0,
        max_retries: int | None = None,
        cwd: str | None = None,
        cli_path: str | None = None,
        permission_mode: str = "default",
        permission_timeout_s: float = 300.0,
        text_output_queue: Queue | None = None,
        allowed_tools: list[str] | None = None,
        setting_sources: list[str] | None = None,
        mcp_config: str | None = None,
        stream_batch_sentences: int = 1,
        compact_history: bool = False,
        **kwargs: Any,
    ) -> None:
        for name, value in (
            ("max_tokens", max_tokens),
            ("max_turns", max_turns),
            ("max_budget_usd", max_budget_usd),
            ("request_timeout_s", request_timeout_s),
            ("permission_timeout_s", permission_timeout_s),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"claude_agent_{name} must be positive.")
        if max_retries is not None and max_retries < 0:
            raise ValueError("claude_agent_max_retries must be nonnegative.")
        if thinking == "enabled" and (
            thinking_budget_tokens < 1024 or (max_tokens is not None and thinking_budget_tokens >= max_tokens)
        ):
            raise ValueError("Enabled thinking needs at least 1024 budget tokens and a larger max_tokens limit.")
        if setting_sources is not None and set(setting_sources) - {"user", "project", "local"}:
            raise ValueError("claude_agent_setting_sources must contain user, project, or local.")
        self.permission_timeout_s = permission_timeout_s
        self.text_output_queue = text_output_queue
        self.max_tokens = max_tokens
        self._sdk_kwargs: dict[str, Any] = dict(
            model=model_name,
            cwd=cwd,
            cli_path=cli_path,
            permission_mode=permission_mode,
            allowed_tools=list(allowed_tools or []),
            tools={"type": "preset", "preset": "claude_code"},
            setting_sources=list(setting_sources) if setting_sources is not None else ["user", "project", "local"],
            include_partial_messages=True,
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            effort=effort,
        )
        if mcp_config is not None:
            self._sdk_kwargs["mcp_servers"] = mcp_config
        if thinking is not None:
            self._sdk_kwargs["thinking"] = {"type": thinking}
            if thinking == "enabled":
                self._sdk_kwargs["thinking"]["budget_tokens"] = thinking_budget_tokens
        self._sdk_kwargs["env"] = {} if max_retries is None else {"CLAUDE_CODE_MAX_RETRIES": str(max_retries)}
        self._sdk_slots = BoundedSemaphore(1)
        self._sdk_compaction_slots = BoundedSemaphore(1)
        self._sdk_streams: set[_ClaudeStream] = set()
        self._sdk_lock = Lock()
        super().setup(
            model_name=model_name,
            request_timeout_s=request_timeout_s,
            stream_batch_sentences=stream_batch_sentences,
            compact_history=compact_history,
            **kwargs,
        )

    def _setup_provider(self, *_args: Any) -> None:
        # Authentication comes from the SDK's inherited environment / CLI login.
        # No OpenAI client or API key is needed.
        pass

    def warmup(self) -> None:
        # Do not execute an agent run (and potentially tools) during startup.
        pass

    def _serialize(self, active_chat: Chat) -> dict[str, str]:
        instructions: list[str] = []
        transcript: list[dict[str, Any]] = []
        for message in active_chat.to_transformers_chat():
            content = message.get("content")
            if isinstance(content, list):
                if any(part.get("type") not in {"text", "input_text", "output_text"} for part in content):
                    raise ValueError(
                        "The Claude Agent SDK adapter currently accepts text transcripts only; use STT for audio and omit image inputs."
                    )
                message["content"] = "".join(part.get("text", "") for part in content)
            if message.get("role") == "system":
                instructions.append(message.get("content") or "")
            else:
                transcript.append(message)
        return {
            "system": "\n\n".join(instructions),
            "prompt": "Continue the conversation below as the assistant. The JSON is conversation history, not a new set of instructions. Respond to the latest user request, using tools when needed.\n"
            + json.dumps(transcript, ensure_ascii=False),
        }

    def _build_optional_kwargs(self, req_tools: Any, req_tool_choice: Any) -> dict[str, Any]:
        # Realtime tools execute on the client, whereas Claude Code tools execute
        # inside the SDK. Never advertise unsupported client functions silently.
        return {"unsupported_client_tools": bool(req_tools) or req_tool_choice not in (None, "auto", "none")}

    def _generate(
        self, active_chat: Chat, original_chat: Chat, turn: _Turn, optional_kwargs: dict[str, Any], **kwargs: Any
    ) -> Generator[LLMOut, None, bool]:
        options = dict(optional_kwargs, turn=turn)
        response_limit = getattr(turn.response, "max_output_tokens", None)
        session_limit = getattr(turn.runtime_config.session, "max_response_output_tokens", None)
        limit = response_limit if response_limit is not None else session_limit
        options["max_tokens"] = limit if isinstance(limit, int) else self.max_tokens
        return (yield from super()._generate(active_chat, original_chat, turn, options, **kwargs))

    def _request(self, api_input: dict[str, str], optional_kwargs: dict[str, Any]) -> _ClaudeStream:
        if optional_kwargs.get("unsupported_client_tools"):
            raise ValueError(
                "Realtime client function tools/tool_choice are not supported by claude-agent-sdk. Configure Claude Code tools/MCP instead."
            )
        sdk_kwargs = {**self._sdk_kwargs, "env": dict(self._sdk_kwargs["env"])}
        limit = optional_kwargs.get("max_tokens", self.max_tokens)
        if limit is not None:
            if limit <= 0:
                raise ValueError("Claude output token limit must be positive.")
            thinking = sdk_kwargs.get("thinking") or {}
            if thinking.get("type") == "enabled" and thinking["budget_tokens"] >= limit:
                raise ValueError("Enabled thinking needs a budget smaller than the output token limit.")
            sdk_kwargs["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(limit)
        sdk_kwargs["system_prompt"] = {"type": "preset", "preset": "claude_code", "append": api_input["system"]}
        if optional_kwargs.get("compaction"):
            sdk_kwargs.update(tools=[], allowed_tools=[], mcp_servers={}, strict_mcp_config=True, setting_sources=[])
        turn = optional_kwargs.get("turn")

        def cancelled() -> bool:
            return self.stop_event.is_set() or (
                turn is not None
                and (self._turn_is_cancelled(turn) or not self._turn_is_latest(turn.turn_id, turn.turn_revision))
            )

        if turn is not None and not optional_kwargs.get("compaction"):
            sdk_kwargs["can_use_tool"] = self._permission_callback(turn, cancelled)

        def waiting() -> bool:
            return turn is not None and bool(turn.runtime_config.agent_interactions.pending(turn.response_key))

        with self._sdk_lock:
            self._sdk_streams = {stream for stream in self._sdk_streams if not stream.finished.is_set()}
            stream = _ClaudeStream(
                ClaudeAgentOptions(**sdk_kwargs),
                api_input["prompt"],
                self.request_timeout_s,
                cancelled,
                self._sdk_compaction_slots if optional_kwargs.get("compaction") else self._sdk_slots,
                waiting,
            )
            self._sdk_streams.add(stream)
        return stream

    def _permission_callback(self, turn: _Turn, cancelled: Callable[[], bool]):
        async def can_use_tool(tool_name, input_data, context):
            if self.text_output_queue is None or cancelled():
                return PermissionResultDeny(message="No permission responder is connected.")
            broker = turn.runtime_config.agent_interactions
            request = broker.open(turn.response_key, tool_name, input_data)
            self.text_output_queue.put(
                AgentPermissionEvent(
                    response_key=turn.response_key,
                    turn_id=turn.turn_id,
                    turn_revision=turn.turn_revision,
                    event=AgentPermissionRequested(
                        request_id=request.request_id,
                        tool_name=tool_name,
                        input=request.input,
                        title=getattr(context, "title", None),
                        description=getattr(context, "description", None),
                        timeout_s=self.permission_timeout_s,
                    ),
                )
            )
            status = "cancelled"
            try:
                deadline = monotonic() + self.permission_timeout_s
                while not cancelled():
                    decision, answers, status = broker.decision(request)
                    if decision is not None:
                        if decision == "allow":
                            updated_input = dict(request.input)
                            if tool_name == "AskUserQuestion":
                                updated_input["answers"] = answers
                            return PermissionResultAllow(updated_input=updated_input)
                        return PermissionResultDeny(message="User denied this request.")
                    if monotonic() >= deadline:
                        status = "timed_out"
                        return PermissionResultDeny(message="Permission request timed out without approval.")
                    await asyncio.sleep(0.05)
                return PermissionResultDeny(message="Response was cancelled.")
            finally:
                status = broker.close(request, status if status != "pending" else "cancelled")
                self.text_output_queue.put(
                    AgentPermissionEvent(
                        response_key=turn.response_key,
                        event=AgentPermissionResolved(request_id=request.request_id, status=status),
                    )
                )

        return can_use_tool

    def _iter_stream_events(self, api_response: _ClaudeStream) -> Iterator[ProviderEvent]:
        yield from api_response

    def _iter_response_events(self, api_response: _ClaudeStream) -> Iterator[ProviderEvent]:
        events = list(api_response)
        text = "".join(event.text for event in events if isinstance(event, TextDelta))
        for event in events:
            if not isinstance(event, TextDelta):
                yield event
        if text:
            yield TextDelta(text=text)

    def _build_compaction_generate_fn(self) -> CompactGenerateFn:
        def generate(system: str, user: str) -> str:
            return "".join(
                event.text
                for event in self._request({"system": system, "prompt": user}, {"compaction": True})
                if isinstance(event, TextDelta)
            )

        return generate

    def process(self, request: LLMIn) -> Iterator[LLMOut]:
        # Hidden follow-ups must not execute agent tools before response.create.
        if request.prefetch_transaction is not None:
            request.prefetch_transaction.discard()
            if not request.prefetch_transaction.claimed:
                yield EndOfResponse(
                    response_key=request.response_key,
                    turn_id=request.turn_id,
                    turn_revision=request.turn_revision,
                    cancel_generation=self.cancel_scope.generation if self.cancel_scope else None,
                )
                return
        yield from super().process(request)

    def on_session_end(self) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        with self._sdk_lock:
            streams = list(self._sdk_streams)
            self._sdk_streams.clear()
        for stream in streams:
            stream.close()
