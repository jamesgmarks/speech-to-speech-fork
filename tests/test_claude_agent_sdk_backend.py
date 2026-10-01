"""Adapter contract tests; no credentials, subprocesses, GPU, or API calls."""

from __future__ import annotations

import asyncio
import json
from queue import Queue
from threading import Event
from time import monotonic
from types import SimpleNamespace

import pytest

pytest.importorskip("claude_agent_sdk")
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock
from claude_agent_sdk.types import StreamEvent, SystemMessage
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.realtime.realtime_session_create_request import RealtimeSessionCreateRequest

import speech_to_speech.LLM.claude_agent_sdk_language_model as adapter
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import EndOfResponse, GenerateResponseRequest, LLMResponseChunk, TokenUsage


def result(**kwargs):
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="sdk-session",
        **kwargs,
    )


def delta(text, **kwargs):
    return StreamEvent(
        uuid="event",
        session_id="sdk-session",
        event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}},
        **kwargs,
    )


class FakeClient:
    instances = []
    messages = []
    permission = None
    permission_result = None
    stall = False
    stall_connect = False
    cancelled_connect = False
    entered = Event()

    def __init__(self, options):
        self.options = options
        self.interrupted = False
        self.disconnected = False
        self.owner = None
        self.prompt = None
        self.__class__.instances.append(self)

    async def __aenter__(self):
        self.owner = asyncio.current_task()
        self.entered.set()
        if self.stall_connect:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.__class__.cancelled_connect = True
                raise
        return self

    async def __aexit__(self, *args):
        assert self.owner is asyncio.current_task(), "SDK lifecycle crossed asyncio tasks"
        self.disconnected = True

    async def query(self, prompt):
        self.prompt = prompt

    async def interrupt(self):
        self.interrupted = True

    async def receive_response(self):
        if self.permission is not None:
            from types import SimpleNamespace

            self.__class__.permission_result = await self.options.can_use_tool(*self.permission, SimpleNamespace())
        for message in self.messages:
            await asyncio.sleep(0)
            yield message
        if self.stall:
            await asyncio.Event().wait()

    async def receive_messages(self):
        async for message in self.receive_response():
            yield message


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch):
    FakeClient.instances = []
    FakeClient.messages = [
        delta("Hello."),
        AssistantMessage(content=[TextBlock(text="Hello.")], model="sonnet"),
        result(usage={"input_tokens": 3, "output_tokens": 2}),
    ]
    FakeClient.permission = None
    FakeClient.permission_result = None
    FakeClient.stall = False
    FakeClient.stall_connect = False
    FakeClient.cancelled_connect = False
    FakeClient.entered = Event()
    monkeypatch.setattr(adapter, "ClaudeSDKClient", FakeClient)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def handler(**kwargs):
    return adapter.ClaudeAgentSDKModelHandler(Event(), Queue(), Queue(), setup_kwargs=kwargs)


def request(*, chat=None, response=None, **session_kwargs):
    chat = chat or Chat(10)
    if not chat.buffer:
        chat.add_item(make_user_message("Hello, Claude."))
    session = RealtimeSessionCreateRequest(type="realtime", instructions="Speak clearly.", **session_kwargs)
    return GenerateResponseRequest(runtime_config=RuntimeConfig(chat=chat, session=session), response=response)


def text(outputs):
    return "".join(output.text for output in outputs if isinstance(output, LLMResponseChunk))


@pytest.mark.parametrize("stream", [True, False])
def test_pipeline_text_not_duplicated_and_history_committed(stream):
    h = handler(stream=stream)
    req = request(response=RealtimeResponseCreateParams(output_modalities=["text"]))
    outputs = list(h.process(req))
    assert text(outputs) == "Hello."
    assert len([o for o in outputs if isinstance(o, EndOfResponse)]) == 1
    assert outputs[-1].error is None
    assert req.runtime_config.chat.to_transformers_chat()[-1]["content"] == "Hello."
    assert [(o.input_tokens, o.output_tokens) for o in outputs if isinstance(o, TokenUsage)] == [(3, 2)]
    assert FakeClient.instances[0].disconnected


def test_all_tools_and_settings_are_available_and_limits_reach_sdk():
    h = handler(
        max_tokens=2048,
        effort="low",
        thinking="enabled",
        thinking_budget_tokens=1024,
        max_turns=4,
        max_budget_usd=0.5,
        max_retries=1,
        cwd="/tmp/workspace",
        permission_mode="acceptEdits",
        allowed_tools=["Bash(ls *)"],
        mcp_config="/tmp/mcp.json",
        terminal_tool=False,
        session_tools=False,
    )
    list(h.process(request()))
    options = FakeClient.instances[0].options
    assert options.tools == {"type": "preset", "preset": "claude_code"}
    assert options.disallowed_tools == []
    assert options.allowed_tools == ["Bash(ls *)"]
    assert options.env == {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "2048", "CLAUDE_CODE_MAX_RETRIES": "1"}
    assert options.thinking == {"type": "enabled", "budget_tokens": 1024}
    assert options.effort == "low"
    assert options.cwd == "/tmp/workspace"
    assert options.mcp_servers == "/tmp/mcp.json"
    assert options.max_turns == 4
    assert options.max_budget_usd == 0.5
    assert options.system_prompt["preset"] == "claude_code"
    assert "Speak clearly." in options.system_prompt["append"]
    assert options.setting_sources == ["user", "project", "local"]


def test_terminal_server_is_client_local_and_preserves_external_mcp(tmp_path):
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({"mcpServers": {"external": {"command": "existing-server"}}}))
    h = handler(mcp_config=str(config))
    list(h.process(request()))
    first = FakeClient.instances[-1].options
    list(h.process(request()))
    second = FakeClient.instances[-1].options
    assert first.mcp_servers["external"] == {"command": "existing-server"}
    assert first.mcp_servers["speech_to_speech"]["type"] == "sdk"
    assert first.mcp_servers["speech_to_speech"]["instance"] is not second.mcp_servers["speech_to_speech"]["instance"]
    assert "mcp__speech_to_speech__launch_agent_terminal" in first.system_prompt["append"]
    assert first.tools == {"type": "preset", "preset": "claude_code"}
    assert first.allowed_tools == []  # Normal permissions, no silent preapproval.
    assert h._sdk_kwargs["mcp_servers"] == str(config)


@pytest.mark.parametrize("decision", ["allow", "deny"])
def test_terminal_tool_uses_existing_permission_ui_and_only_launches_on_allow(monkeypatch, tmp_path, decision):
    from threading import Thread
    from types import SimpleNamespace

    tools = []
    original = adapter.create_sdk_mcp_server

    def server(**kwargs):
        tools.extend(kwargs["tools"])
        return original(**kwargs)

    monkeypatch.setattr(adapter, "create_sdk_mcp_server", server)
    calls = []
    events = Queue()
    h = handler(text_output_queue=events)
    monkeypatch.setattr(
        h._terminal_launcher, "launch", lambda *args: calls.append(args) or {"status": "launch_requested"}
    )

    async def receive(self):
        args = {"directory": str(tmp_path), "prompt": "Review the tests."}
        approval = await self.options.can_use_tool(
            "mcp__speech_to_speech__launch_agent_terminal", args, SimpleNamespace()
        )
        if approval.behavior == "allow":
            terminal_tool = next(t for t in tools if t.name == "launch_agent_terminal")
            response = await terminal_tool.handler(approval.updated_input)
            assert json.loads(response["content"][0]["text"])["status"] == "launch_requested"
        for message in self.messages:
            yield message

    monkeypatch.setattr(FakeClient, "receive_response", receive)
    req = request()
    outputs = []
    worker = Thread(target=lambda: outputs.extend(h.process(req)))
    worker.start()
    event = events.get(timeout=2)
    assert event.event.tool_name == "mcp__speech_to_speech__launch_agent_terminal"
    assert not calls
    assert req.runtime_config.agent_interactions.respond(event.event.request_id, decision) is None
    worker.join(2)
    assert not worker.is_alive() and outputs[-1].error is None
    assert calls == ([(str(tmp_path), "Review the tests.")] if decision == "allow" else [])
    h.cleanup()


@pytest.mark.parametrize(
    "session_limit,response_limit,expected", [(512, 128, "128"), (512, None, "512"), ("inf", None, "256")]
)
def test_realtime_token_limit_overrides_cli_limit(session_limit, response_limit, expected):
    h = handler(max_tokens=256)
    response = RealtimeResponseCreateParams(max_output_tokens=response_limit) if response_limit else None
    list(h.process(request(response=response, max_response_output_tokens=session_limit)))
    assert FakeClient.instances[0].options.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == expected


def test_complete_message_fallback_and_subagent_text_and_tools_are_not_spoken():
    FakeClient.messages = [
        delta("Subagent text", parent_tool_use_id="tool_1"),
        AssistantMessage(content=[TextBlock(text="Hidden")], model="sonnet", parent_tool_use_id="tool_1"),
        AssistantMessage(
            content=[ToolUseBlock(id="tool_2", name="Read", input={"file_path": "file.py"})], model="sonnet"
        ),
        AssistantMessage(content=[TextBlock(text="Done.")], model="sonnet"),
        result(),
    ]
    outputs = list(handler().process(request()))
    assert text(outputs) == "Done."
    assert outputs[-1].error is None


def test_multiple_agent_turns_stream_once_in_order():
    FakeClient.messages = [
        delta("Checking."),
        AssistantMessage(content=[TextBlock(text="Checking.")], model="sonnet"),
        delta(" Done."),
        AssistantMessage(content=[TextBlock(text=" Done.")], model="sonnet"),
        result(),
    ]
    outputs = list(handler().process(request(response=RealtimeResponseCreateParams(output_modalities=["text"]))))
    assert text(outputs) == "Checking. Done."


@pytest.mark.parametrize("failure", ["sdk", "missing_result", "assistant"])
def test_failures_emit_terminal_and_do_not_commit_history(failure):
    if failure == "sdk":
        FakeClient.messages = [
            ResultMessage(
                subtype="error_max_turns",
                duration_ms=1,
                duration_api_ms=1,
                is_error=True,
                num_turns=1,
                session_id="s",
                errors=["Turn limit reached"],
            )
        ]
    elif failure == "missing_result":
        FakeClient.messages = [AssistantMessage(content=[TextBlock(text="Incomplete.")], model="sonnet")]
    else:
        FakeClient.messages = [AssistantMessage(content=[], model="sonnet", error="authentication_failed")]
    req = request()
    before = req.runtime_config.chat.to_transformers_chat()
    outputs = list(handler().process(req))
    assert isinstance(outputs[-1], EndOfResponse)
    assert outputs[-1].error
    assert req.runtime_config.chat.to_transformers_chat() == before
    assert FakeClient.instances[0].disconnected


def test_interrupt_during_silent_tool_work():
    scope = CancelScope()
    FakeClient.messages = [delta("Hello.")]
    FakeClient.stall = True
    h = handler(cancel_scope=scope)
    req = request(response=RealtimeResponseCreateParams(output_modalities=["text"]))
    outputs = h.process(req)
    assert next(outputs).text == "Hello."
    scope.cancel()
    start = monotonic()
    remaining = list(outputs)
    assert monotonic() - start < 1.0
    assert isinstance(remaining[-1], EndOfResponse)
    assert FakeClient.instances[0].interrupted
    assert FakeClient.instances[0].disconnected
    assert len(req.runtime_config.chat.buffer) == 1


def test_timeout_during_silent_work_terminates_response():
    FakeClient.messages = []
    FakeClient.stall = True
    outputs = list(handler(request_timeout_s=0.1).process(request()))
    assert "timed out" in outputs[-1].error
    assert FakeClient.instances[0].disconnected


def test_fresh_sessions_use_only_current_pipeline_history():
    h = handler()
    list(h.process(request()))
    h.on_session_end()
    chat = Chat(10)
    chat.add_item(make_user_message("New session"))
    list(h.process(request(chat=chat)))
    assert len(FakeClient.instances) == 2
    second = FakeClient.instances[1]
    assert second.options.resume is None
    transcript = json.loads(second.prompt.split("\n", 1)[1])
    assert transcript == [{"role": "user", "content": "New session"}]


def test_client_function_tools_fail_clearly_without_sdk_launch():
    outputs = list(handler().process(request(tools=[{"type": "function", "name": "custom", "parameters": {}}])))
    assert "Realtime client function tools" in outputs[-1].error
    assert not FakeClient.instances


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": 0},
        {"max_turns": -1},
        {"request_timeout_s": 0},
        {"max_retries": -1},
        {"max_budget_usd": 0},
        {"setting_sources": ["invalid"]},
        {"thinking": "enabled", "max_tokens": 1024},
    ],
)
def test_invalid_settings_fail_early(kwargs):
    with pytest.raises(ValueError):
        handler(**kwargs)


@pytest.mark.parametrize("claimed", [False, True])
def test_hidden_prefetch_does_not_run_tools_but_claimed_work_can_run(claimed):
    from speech_to_speech.pipeline.messages import ResponsePrefetchTransaction

    req = request()
    req.prefetch_transaction = ResponsePrefetchTransaction()
    if claimed:
        assert req.prefetch_transaction.claim()
    outputs = list(handler().process(req))
    assert isinstance(outputs[-1], EndOfResponse)
    assert bool(FakeClient.instances) == claimed
    assert req.prefetch_transaction.discarded is not claimed


def test_dynamic_output_limit_validates_thinking_budget():
    h = handler(thinking="enabled", thinking_budget_tokens=1024, max_tokens=2048)
    outputs = list(h.process(request(max_response_output_tokens=512)))
    assert "budget smaller" in outputs[-1].error
    assert not FakeClient.instances


def test_cancel_during_sdk_startup_releases_worker():
    from threading import Thread

    scope = CancelScope()
    FakeClient.stall_connect = True
    h = handler(cancel_scope=scope)
    outputs = []
    worker = Thread(target=lambda: outputs.extend(h.process(request())))
    worker.start()
    assert FakeClient.entered.wait(1)
    scope.cancel()
    worker.join(1)
    assert not worker.is_alive()
    assert FakeClient.cancelled_connect
    assert isinstance(outputs[-1], EndOfResponse)
    assert h._sdk_slots.acquire(timeout=0.1)
    h._sdk_slots.release()


def test_background_compaction_does_not_occupy_the_foreground_worker_slot():
    FakeClient.messages = []
    FakeClient.stall = True
    h = handler()
    compaction = h._request({"system": "Summarize", "prompt": "History"}, {"compaction": True})
    try:
        assert FakeClient.entered.wait(1)
        foreground = h._request({"system": "Respond", "prompt": "Hello"}, {})
        assert foreground is not compaction
    finally:
        h.cleanup()
    assert compaction.finished.is_set()
    assert foreground.finished.is_set()


def test_empty_browser_tool_list_with_none_choice_keeps_native_tools():
    outputs = list(handler().process(request(tools=[], tool_choice="none")))
    assert outputs[-1].error is None
    assert FakeClient.instances[0].options.tools == {"type": "preset", "preset": "claude_code"}


@pytest.mark.parametrize("decision", ["allow", "deny"])
def test_sdk_waits_for_session_permission_and_resumes_same_response(decision):
    from threading import Thread
    from time import sleep

    events = Queue()
    FakeClient.permission = ("Bash", {"command": "echo hello"})
    h = handler(text_output_queue=events, request_timeout_s=0.1)
    req = request()
    outputs = []
    worker = Thread(target=lambda: outputs.extend(h.process(req)))
    worker.start()
    event = events.get(timeout=1)
    assert event.event.type == "speech_to_speech.agent.permission.requested"
    sleep(0.2)  # Human approval time does not consume the generation deadline.
    assert worker.is_alive()
    assert req.runtime_config.agent_interactions.respond(event.event.request_id, decision) is None
    worker.join(1)
    assert not worker.is_alive()
    assert outputs[-1].error is None
    assert FakeClient.permission_result.behavior == decision
    assert len(FakeClient.instances) == 1
    assert events.get(timeout=1).event.status == ("allowed" if decision == "allow" else "denied")
    assert not req.runtime_config.agent_interactions.pending()


def test_unanswered_permission_times_out_with_denial():
    events = Queue()
    FakeClient.permission = ("Write", {"file_path": "file.py"})
    req = request()
    list(handler(text_output_queue=events, permission_timeout_s=0.1).process(req))
    assert FakeClient.permission_result.behavior == "deny"
    assert events.get().event.type.endswith("requested")
    assert events.get().event.status == "timed_out"
    assert not req.runtime_config.agent_interactions.pending()


def test_cancel_while_waiting_for_approval_does_not_allow_the_tool():
    from threading import Thread

    events = Queue()
    scope = CancelScope()
    FakeClient.permission = ("Write", {"file_path": "file.py"})
    req = request()
    outputs = []
    h = handler(text_output_queue=events, cancel_scope=scope)
    worker = Thread(target=lambda: outputs.extend(h.process(req)))
    worker.start()
    prompt = events.get(timeout=1)
    scope.cancel()
    worker.join(1)
    assert not worker.is_alive()
    assert not req.runtime_config.agent_interactions.pending()
    assert req.runtime_config.agent_interactions.respond(prompt.event.request_id, "allow")
    assert isinstance(outputs[-1], EndOfResponse)


class DelayedWorkerClient(FakeClient):
    release = Event()
    worker_permission = False
    terminal_patch = False

    async def receive_messages(self):
        if self is not self.instances[0]:
            async for message in super().receive_messages():
                yield message
            return
        yield SystemMessage(
            subtype="task_started",
            data={
                "task_id": "worker-1",
                "description": "Inspect the architecture",
                "tool_use_id": "spawn-1",
            },
        )
        yield delta("I've started the investigation.")
        yield AssistantMessage(content=[TextBlock(text="I've started the investigation.")], model="sonnet")
        yield result()
        while not self.release.is_set():
            await asyncio.sleep(0.01)
        if self.worker_permission:
            from types import SimpleNamespace

            self.__class__.permission_result = await self.options.can_use_tool(
                "Read", {"file_path": "README.md"}, SimpleNamespace()
            )
        if self.terminal_patch:
            yield SystemMessage(subtype="task_updated", data={"task_id": "worker-1", "patch": {"status": "completed"}})
            yield AssistantMessage(
                content=[TextBlock(text="The pipeline uses a bounded background queue.")], model="sonnet"
            )
            yield result()
        else:
            yield AssistantMessage(
                content=[TextBlock(text="The pipeline uses a bounded background queue.")],
                model="sonnet",
                parent_tool_use_id="spawn-1",
            )
            yield SystemMessage(
                subtype="task_notification",
                data={
                    "task_id": "worker-1",
                    "status": "completed",
                    "summary": "Investigation completed",
                    "output_file": "",
                },
            )


def delayed_sdk(monkeypatch):
    DelayedWorkerClient.release = Event()
    DelayedWorkerClient.worker_permission = False
    DelayedWorkerClient.terminal_patch = False
    monkeypatch.setattr(adapter, "ClaudeSDKClient", DelayedWorkerClient)


@pytest.mark.parametrize("terminal_patch", [False, True])
def test_background_worker_survives_foreground_and_intervening_turn(monkeypatch, terminal_patch):
    delayed_sdk(monkeypatch)
    DelayedWorkerClient.terminal_patch = terminal_patch
    events = Queue()
    scope = CancelScope()
    h = handler(text_output_queue=events, cancel_scope=scope)
    req = request()
    try:
        outputs = list(h.process(req))
        assert text(outputs) == "I've started the investigation."
        assert outputs[-1].error is None
        first = FakeClient.instances[0]
        assert not first.disconnected
        started = events.get(timeout=1)
        assert started.status == "running"
        assert started.runtime_config is req.runtime_config
        # A spoken interruption or a new voice turn must not close this owner.
        scope.cancel()
        scope.new_response()
        second = request(chat=req.runtime_config.chat)
        assert text(list(h.process(second))) == "Hello."
        assert not first.interrupted
        assert not first.disconnected
        DelayedWorkerClient.release.set()
        completed = events.get(timeout=1)
        assert completed.status == "completed"
        assert completed.result == "The pipeline uses a bounded background queue."
        assert completed.runtime_config is req.runtime_config
    finally:
        h.cleanup()
    assert first.disconnected


def test_background_permission_remains_answerable_after_speech_cancel(monkeypatch):
    delayed_sdk(monkeypatch)
    DelayedWorkerClient.worker_permission = True
    events = Queue()
    scope = CancelScope()
    h = handler(text_output_queue=events, cancel_scope=scope)
    req = request()
    try:
        list(h.process(req))
        assert events.get(timeout=1).status == "running"
        scope.cancel()
        DelayedWorkerClient.release.set()
        prompt = events.get(timeout=1)
        assert prompt.response_key.startswith("background:")
        assert prompt.runtime_config is req.runtime_config
        assert prompt.turn_id is None
        assert req.runtime_config.agent_interactions.respond(prompt.event.request_id, "allow") is None
        assert events.get(timeout=1).event.status == "allowed"
        assert events.get(timeout=1).status == "completed"
        assert DelayedWorkerClient.permission_result.behavior == "allow"
    finally:
        h.cleanup()


def test_call_teardown_stops_native_background_owner(monkeypatch):
    delayed_sdk(monkeypatch)
    events = Queue()
    h = handler(text_output_queue=events)
    req = request()
    list(h.process(req))
    owner = FakeClient.instances[0]
    assert events.get(timeout=1).status == "running"
    h.on_session_end()
    assert owner.interrupted
    assert owner.disconnected
    assert not req.runtime_config.agent_interactions.pending()
    assert events.empty()


def test_background_timeout_reports_failure_without_blocking_next_turn(monkeypatch):
    delayed_sdk(monkeypatch)
    events = Queue()
    h = handler(text_output_queue=events, background_timeout_s=0.1)
    try:
        outputs = list(h.process(request()))
        assert outputs[-1].error is None
        assert events.get(timeout=1).status == "running"
        failure = events.get(timeout=1)
        assert failure.status == "failed"
        assert "timed out" in failure.result
        assert text(list(h.process(request()))) == "Hello."
    finally:
        h.cleanup()


def test_native_agent_jsonl_output_exposes_only_final_assistant_text(tmp_path):
    from speech_to_speech.LLM.claude_background import _task_output

    output = tmp_path / "job.output"
    output.write_text(
        "\n".join(
            json.dumps(frame)
            for frame in [
                {"message": {"role": "user", "content": "Hidden worker prompt"}},
                {"message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "Hidden"}]}},
                {"message": {"role": "assistant", "content": [{"type": "tool_use", "input": {"secret": "Hidden"}}]}},
                {"message": {"role": "assistant", "content": [{"type": "text", "text": "The actual findings."}]}},
            ]
        )
    )
    assert _task_output(str(output)) == "The actual findings."


def test_native_command_output_is_bounded(tmp_path):
    from speech_to_speech.LLM.claude_background import _task_output

    output = tmp_path / "job.output"
    output.write_text("x" * 100000 + "\nCommand finished successfully.")
    extracted = _task_output(str(output))
    assert len(extracted) == 32768
    assert extracted.endswith("Command finished successfully.")


def test_duplicate_terminal_patch_then_notification_keeps_actual_findings(monkeypatch):
    delayed_sdk(monkeypatch)

    class PatchFirstClient(DelayedWorkerClient):
        async def receive_messages(self):
            async for message in super().receive_messages():
                if isinstance(message, AssistantMessage) and message.parent_tool_use_id == "spawn-1":
                    yield SystemMessage(
                        subtype="task_updated",
                        data={
                            "task_id": "worker-1",
                            "patch": {"status": "completed"},
                        },
                    )
                yield message

    monkeypatch.setattr(adapter, "ClaudeSDKClient", PatchFirstClient)
    events = Queue()
    h = handler(text_output_queue=events)
    try:
        list(h.process(request()))
        assert events.get(timeout=1).status == "running"
        DelayedWorkerClient.release.set()
        completed = events.get(timeout=1)
        assert completed.status == "completed"
        assert completed.result == "The pipeline uses a bounded background queue."
    finally:
        h.cleanup()
    assert events.empty()


def test_native_json_command_output_is_not_mistaken_for_agent_transcript(tmp_path):
    from speech_to_speech.LLM.claude_background import _task_output

    output = tmp_path / "job.output"
    output.write_text('{"passed": 4, "failed": 0}')
    assert _task_output(str(output)) == '{"passed": 4, "failed": 0}'


def test_workers_nested_background_command_is_not_an_independent_announcement(monkeypatch):
    delayed_sdk(monkeypatch)

    class NestedCommandClient(DelayedWorkerClient):
        async def receive_messages(self):
            async for message in super().receive_messages():
                if isinstance(message, AssistantMessage) and message.parent_tool_use_id == "spawn-1":
                    yield AssistantMessage(
                        content=[ToolUseBlock(id="nested-command", name="Bash", input={"command": "sleep 5"})],
                        model="sonnet",
                        parent_tool_use_id="spawn-1",
                    )
                    yield SystemMessage(
                        subtype="task_started",
                        data={
                            "task_id": "inner-command",
                            "description": "Wait five seconds",
                            "tool_use_id": "nested-command",
                        },
                    )
                    yield SystemMessage(
                        subtype="task_notification",
                        data={
                            "task_id": "inner-command",
                            "status": "completed",
                            "summary": "Command done",
                            "output_file": "",
                        },
                    )
                yield message

    monkeypatch.setattr(adapter, "ClaudeSDKClient", NestedCommandClient)
    events = Queue()
    h = handler(text_output_queue=events)
    try:
        list(h.process(request()))
        assert events.get(timeout=1).status == "running"
        DelayedWorkerClient.release.set()
        completed = events.get(timeout=1)
        assert completed.job_id == "worker-1"
        assert completed.status == "completed"
        assert completed.result == "The pipeline uses a bounded background queue."
    finally:
        h.cleanup()
    assert events.empty()


async def test_external_inventory_and_history_tools_are_immediate_read_only_and_client_local(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    external = {
        "source": "external",
        "name": "james-b6",
        "session_id": "native-id",
        "native_session_id": "native-id",
        "directory": "/tmp/james",
        "state": "idle",
    }
    inventory = SimpleNamespace(
        list=AsyncMock(return_value={"sessions": [external], "enabled": True, "discovery_error": None})
    )
    monkeypatch.setattr(adapter, "AgentSessionInventory", lambda registry: inventory)
    read = AsyncMock(return_value={"messages": [{"role": "assistant", "text": "Recent reply."}]})
    monkeypatch.setattr(adapter, "read_agent_session", read)
    h = handler()
    req = request()
    tools = {t.name: t for t in h._independent_session_tools(SimpleNamespace(runtime_config=req.runtime_config))}
    result = await tools["list_external_agent_sessions"].handler({})
    assert json.loads(result["content"][0]["text"])["sessions"][0]["directory"] == "/tmp/james"
    result = await tools["read_agent_session"].handler({"target": "James B6", "max_messages": 7})
    assert "Recent reply" in result["content"][0]["text"]
    read.assert_awaited_once_with(inventory, "James B6", 7)
    assert FakeClient.instances == []  # No persistent router/model turn for reads.
    req.runtime_config.connection_active = False
    denied = await tools["read_agent_session"].handler({"target": "James"})
    assert denied["isError"]
    assert read.await_count == 1


async def test_external_send_resolves_native_id_and_requires_native_reply_channel(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    external = {
        "source": "external",
        "name": "james-b6",
        "session_id": "native-id",
        "native_session_id": "native-id",
        "directory": "/tmp/james",
        "state": "idle",
    }
    inventory = SimpleNamespace(list=AsyncMock(return_value={"sessions": [external]}))
    monkeypatch.setattr(adapter, "AgentSessionInventory", lambda registry: inventory)
    peer = SimpleNamespace(send=Mock(return_value={"status": "queued"}))
    registry = SimpleNamespace(find=lambda name: peer, track_external=Mock())
    h = handler()
    h._peer_sessions = registry
    req = request()
    tools = {t.name: t for t in h._independent_session_tools(SimpleNamespace(runtime_config=req.runtime_config))}
    reply = await tools["send_external_agent_message"].handler({"target": "native-id", "message": "Status please."})
    result = json.loads(reply["content"][0]["text"])
    assert result["target"] == "james-b6" and result["delivery"] == "pending"
    prompt = peer.send.call_args.args[0]
    assert '"target": "james-b6"' in prompt
    assert "built-in tool named exactly SendMessage" in prompt
    assert "from address" in prompt and "not an MCP" in prompt
    registry.track_external.assert_called_once_with("native-id")
    inventory.list.return_value["sessions"].append({**external, "name": "james-b7", "session_id": "other"})
    denied = await tools["send_external_agent_message"].handler({"target": "James", "message": "Status please."})
    assert denied["isError"] and "Ambiguous" in denied["content"][0]["text"]
    assert peer.send.call_count == 1


async def test_new_independent_session_inherits_voice_mode_and_can_switch_to_bypass(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock

    h = handler(permission_mode="default")
    req = request()
    req.runtime_config.agent_permission_mode = "acceptEdits"
    captured = {}

    def create(name, directory, factory, timeout, limit, *, permission_mode):
        peer = SimpleNamespace(
            name=name,
            directory=directory,
            permission_mode=permission_mode,
            native_session_id=None,
            send=Mock(return_value={"session_id": "peer_test", "status": "queued"}),
        )
        captured["client"] = factory(peer)
        return peer

    h._peer_sessions = SimpleNamespace(create=create)
    tools = {
        tool.name: tool for tool in h._independent_session_tools(SimpleNamespace(runtime_config=req.runtime_config))
    }
    response = await tools["create_agent_session"].handler(
        {"name": "worker", "directory": str(tmp_path), "prompt": "Inspect the code"}
    )
    assert not response.get("isError")
    options = captured["client"].options
    assert options.permission_mode == "acceptEdits"
    assert options.extra_args == {"name": "worker", "allow-dangerously-skip-permissions": None}
    assert options.tools == {"type": "preset", "preset": "claude_code"} and options.disallowed_tools == []
    assert h._sdk_kwargs["permission_mode"] == "default"


def test_peer_sessions_do_not_inherit_voice_output_token_cap():
    h = handler(max_tokens=2048, max_retries=1)
    options = h._create_peer_client(
        SimpleNamespace(
            name="peer", directory="/tmp", native_session_id=None, permission_mode="default", recipient=None
        )
    ).options
    assert options.env == {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "", "CLAUDE_CODE_MAX_RETRIES": "1"}
