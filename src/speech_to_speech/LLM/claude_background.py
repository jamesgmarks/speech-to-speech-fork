"""Keep Claude's native workers alive independently of a spoken response.

Each SDK client is connected and disconnected by one owning asyncio task. A
foreground iterator can finish while that task continues receiving worker
lifecycle messages. Only call teardown, a worker deadline, or explicit shutdown
closes a client that still owns jobs.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from queue import Empty, Full, Queue
from threading import BoundedSemaphore, Event, Lock, Thread
from time import monotonic
from typing import Any
from uuid import uuid4

from claude_agent_sdk import AssistantMessage as SDKAssistantMessage
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, TextBlock, ToolUseBlock
from claude_agent_sdk.types import StreamEvent, SystemMessage
from openai.types.realtime.realtime_conversation_item_assistant_message import Content as AssistantContent

from speech_to_speech.LLM.base_openai_compatible_language_model import AssistantMessage, ProviderEvent, TextDelta, Usage

BackgroundSink = Callable[[str, str, str, str], None]


def _task_output(path: str) -> str:
    """Read only the output path supplied by an SDK lifecycle frame, bounded.

    Agent output files can be JSONL transcripts. Extract assistant text rather
    than exposing instructions, thinking, or internal tool arguments.
    """
    if not path:
        return ""
    try:
        output = Path(path)
        if not output.is_file():
            return ""
        with output.open("rb") as source:
            source.seek(max(0, output.stat().st_size - 65536))
            text = source.read(65536).decode("utf-8", errors="replace")
    except OSError:
        return ""
    assistant_text = ""
    parsed = False
    for line in text.splitlines():
        try:
            frame = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(frame, dict):
            continue
        message = frame.get("message", frame)
        if not isinstance(message, dict):
            continue
        role = message.get("role", frame.get("type"))
        if role not in {"assistant", "user", "system"}:
            continue
        parsed = True
        if role != "assistant":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            candidate = content
        elif isinstance(content, list):
            candidate = "\n".join(
                block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            continue
        if candidate.strip():
            assistant_text = candidate
    return (assistant_text if parsed else text).strip()[-32768:]


class ClaudeStream:
    """Bounded foreground bridge with independently owned native SDK jobs."""

    def __init__(
        self,
        options: ClaudeAgentOptions,
        prompt: str,
        timeout: float,
        cancelled: Callable[[], bool],
        slots: BoundedSemaphore,
        waiting: Callable[[], bool] = lambda: False,
        *,
        client_factory: Callable[..., Any],
        background_sink: BackgroundSink | None = None,
        background_slots: BoundedSemaphore | None = None,
        background_timeout: float = 1800.0,
        stopped: Callable[[], bool] = lambda: False,
    ):
        if not slots.acquire(timeout=0.1):
            raise RuntimeError("The previous Claude SDK request is still shutting down. Please retry.")
        self.options = options
        self.prompt = prompt
        self.timeout = timeout
        self.cancelled = cancelled
        self.waiting = waiting
        self.client_factory = client_factory
        self.background_sink = background_sink
        self.background_slots = background_slots
        self.background_timeout = background_timeout
        self.stopped = stopped
        self.background_key = "background:" + uuid4().hex
        self.closed = Event()
        self.foreground_closed = Event()
        self.foreground_finished = Event()
        self.finished = Event()
        self.connected = Event()
        self.results: Queue[ProviderEvent | BaseException] = Queue(maxsize=16)
        self.jobs: dict[str, dict[str, str]] = {}
        self._lock = Lock()
        self._slots = slots
        self._slot_released = False
        self._background_slot_owned = False
        self._pending: dict[str, tuple[str, str]] = {}
        self._child_text: dict[str, str] = {}
        self._job_tools: dict[str, str] = {}
        self._nested_tool_ids: set[str] = set()
        self.worker = Thread(target=self._run, name="claude-agent-sdk", daemon=True)
        try:
            self.worker.start()
        except BaseException:
            slots.release()
            raise

    def _finish_foreground(self) -> None:
        self.foreground_finished.set()
        with self._lock:
            if not self._slot_released:
                self._slot_released = True
                self._slots.release()

    def has_jobs(self) -> bool:
        with self._lock:
            return bool(self.jobs or self._pending)

    def owns_background(self) -> bool:
        return self.foreground_finished.is_set() or self.foreground_closed.is_set()

    def job_snapshot(self) -> list[dict[str, str]]:
        with self._lock:
            return [
                {"job_id": job_id, "description": job["description"], "status": "running"}
                for job_id, job in self.jobs.items()
            ]

    def _emit_job(self, task_id: str, status: str, description: str, result: str = "") -> None:
        if self.background_sink is not None and not self.closed.is_set() and not self.stopped():
            self.background_sink(task_id, status, description, result)

    async def _publish(self, event: ProviderEvent | BaseException) -> None:
        while not self.closed.is_set() and not self.foreground_closed.is_set():
            try:
                self.results.put_nowait(event)
                return
            except Full:
                await asyncio.sleep(0.01)

    async def _lifecycle(self, message: SystemMessage) -> bool:
        if self.background_sink is None:
            return False
        data = message.data
        subtype = message.subtype
        task_id = getattr(message, "task_id", None) or data.get("task_id")
        if not task_id or subtype not in {"task_started", "task_progress", "task_notification", "task_updated"}:
            return False
        if subtype == "task_started":
            tool_use_id = getattr(message, "tool_use_id", None) or data.get("tool_use_id", "")
            if data.get("parent_tool_use_id") is not None or tool_use_id in self._nested_tool_ids:
                # A worker's shell commands and nested agents belong to that
                # worker. Reporting each as an independent user job duplicates
                # its final announcement and can expose intermediate output.
                return True
            if not self._background_slot_owned and self.background_slots is not None:
                if not self.background_slots.acquire(blocking=False):
                    raise RuntimeError("The background job limit has been reached. Finish an existing job first.")
                self._background_slot_owned = True
            description = getattr(message, "description", None) or data.get("description", "Background task")
            with self._lock:
                self.jobs[task_id] = {
                    "description": description,
                    "tool_use_id": tool_use_id,
                }
                self._job_tools[task_id] = self.jobs[task_id]["tool_use_id"]
            self._emit_job(task_id, "running", description)
            return True
        with self._lock:
            job = self.jobs.get(task_id)
            if job is None and task_id in self._pending:
                description, _ = self._pending[task_id]
                job = {"description": description, "tool_use_id": self._job_tools.get(task_id, "")}
        if job is None:
            return True  # Duplicate terminal frames are normal in SDK streams.
        status = getattr(message, "status", None) or data.get("status") or data.get("patch", {}).get("status")
        if status not in {"completed", "failed", "stopped", "killed"}:
            return True
        status = "cancelled" if status in {"stopped", "killed"} else status
        output = self._child_text.get(job["tool_use_id"], "")
        if not output:
            output = await asyncio.to_thread(
                _task_output, getattr(message, "output_file", None) or data.get("output_file", "")
            )
        summary = getattr(message, "summary", None) or data.get("summary", "")
        with self._lock:
            self.jobs.pop(task_id, None)
        if not self.owns_background():
            # The parent is still answering this request, so it owns reporting
            # these findings. Do not schedule a duplicate unsolicited response.
            self._emit_job(task_id, status, job["description"])
        elif output or status != "completed":
            self._emit_job(task_id, status, job["description"], output or summary or f"Task {status}.")
            self._pending.pop(task_id, None)
        else:
            # Some CLI versions only expose a terminal patch. Wait for the
            # parent's automatic continuation to produce usable findings.
            self._pending[task_id] = (job["description"], summary)
        return True

    async def _session(self) -> None:
        async with self.client_factory(options=self.options) as client:
            self.connected.set()
            if self.closed.is_set() or self.cancelled():
                return
            await client.query(self.prompt)

            async def receive() -> None:
                streamed_text = ""
                background_text = ""
                completed = False
                messages = client.receive_messages() if self.background_sink is not None else client.receive_response()
                async for message in messages:
                    if isinstance(message, SystemMessage) and await self._lifecycle(message):
                        if completed and not self.has_jobs():
                            return
                        continue
                    parent_id = getattr(message, "parent_tool_use_id", None)
                    if parent_id is not None:
                        if isinstance(message, StreamEvent):
                            block = message.event.get("content_block", {})
                            if block.get("type") == "tool_use" and block.get("id"):
                                self._nested_tool_ids.add(block["id"])
                        if isinstance(message, SDKAssistantMessage):
                            self._nested_tool_ids.update(
                                block.id for block in message.content if isinstance(block, ToolUseBlock)
                            )
                            text = "".join(block.text for block in message.content if isinstance(block, TextBlock))
                            if text:
                                self._child_text[parent_id] = text
                                for task_id, (description, _) in list(self._pending.items()):
                                    if self._job_tools.get(task_id) == parent_id:
                                        self._emit_job(task_id, "completed", description, text)
                                        self._pending.pop(task_id, None)
                                if completed and not self.has_jobs():
                                    return
                        continue
                    if isinstance(message, StreamEvent):
                        event = message.event
                        if event.get("type") == "message_start":
                            streamed_text = ""
                        delta = event.get("delta", {})
                        if event.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            streamed_text += text
                            if not self.owns_background():
                                await self._publish(TextDelta(text=text))
                    elif isinstance(message, SDKAssistantMessage):
                        if message.error:
                            raise RuntimeError(f"Claude assistant error: {message.error}")
                        text = "".join(block.text for block in message.content if isinstance(block, TextBlock))
                        if self.owns_background():
                            if text:
                                background_text += text
                        else:
                            remaining = text[len(streamed_text) :] if text.startswith(streamed_text) else ""
                            if remaining:
                                await self._publish(TextDelta(text=remaining))
                            if text:
                                await self._publish(
                                    AssistantMessage(content=[AssistantContent(type="output_text", text=text)])
                                )
                        streamed_text = ""
                    elif isinstance(message, ResultMessage):
                        usage = message.usage or {}
                        if not completed:
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
                        if completed or self.foreground_closed.is_set():
                            for task_id, (description, summary) in list(self._pending.items()):
                                self._emit_job(
                                    task_id,
                                    "completed",
                                    description,
                                    background_text
                                    or message.result
                                    or summary
                                    or "Task completed without a textual result.",
                                )
                                self._pending.pop(task_id, None)
                            background_text = ""
                        completed = True
                        self._finish_foreground()
                        if not self.has_jobs():
                            return
                if not completed and not self.foreground_closed.is_set():
                    raise RuntimeError("Claude SDK stream ended without a result message.")
                if self.has_jobs():
                    raise RuntimeError("Claude SDK stream ended before its background tasks finished.")

            receiver = asyncio.create_task(receive())
            try:
                while not receiver.done():
                    if self.closed.is_set() or self.stopped():
                        await asyncio.wait_for(client.interrupt(), timeout=2.0)
                        return
                    if self.cancelled() or self.foreground_closed.is_set():
                        if self.has_jobs():
                            self.foreground_closed.set()
                            self._finish_foreground()
                        elif not self.foreground_finished.is_set():
                            await asyncio.wait_for(client.interrupt(), timeout=2.0)
                            return
                    await asyncio.wait({receiver}, timeout=0.05)
                await receiver
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)

    def _run(self) -> None:
        async def run() -> None:
            session = asyncio.create_task(self._session())
            previous = monotonic()
            deadline = previous + self.timeout
            background_deadline: float | None = None
            try:
                while not session.done():
                    now = monotonic()
                    if self.waiting():
                        deadline += now - previous
                        if background_deadline is not None:
                            background_deadline += now - previous
                    previous = now
                    if self.owns_background():
                        if background_deadline is None:
                            background_deadline = now + self.background_timeout
                        if now >= background_deadline:
                            raise TimeoutError(f"Background Claude tasks timed out after {self.background_timeout:g}s.")
                    elif now >= deadline:
                        error = TimeoutError(f"Claude agent response timed out after {self.timeout:g}s.")
                        if self.has_jobs():
                            await self._publish(error)
                            self.foreground_closed.set()
                            self._finish_foreground()
                        else:
                            raise error
                    if not self.connected.is_set() and (self.closed.is_set() or self.cancelled()):
                        return
                    await asyncio.wait({session}, timeout=0.05)
                await session
            except Exception as exc:
                session.cancel()
                await asyncio.gather(session, return_exceptions=True)
                for task_id, job in list(self.jobs.items()):
                    self._emit_job(task_id, "failed", job["description"], str(exc))
                for task_id, (description, _) in list(self._pending.items()):
                    self._emit_job(task_id, "failed", description, str(exc))
                await self._publish(exc)
            finally:
                session.cancel()
                await asyncio.gather(session, return_exceptions=True)

        try:
            asyncio.run(run())
        finally:
            self._finish_foreground()
            if self._background_slot_owned and self.background_slots is not None:
                self.background_slots.release()
            self.finished.set()

    def __iter__(self) -> Iterator[ProviderEvent]:
        try:
            while not self.closed.is_set() and not self.cancelled():
                try:
                    result = self.results.get(timeout=0.05)
                except Empty:
                    if self.foreground_finished.is_set() or self.finished.is_set():
                        return
                    continue
                if isinstance(result, BaseException):
                    raise result
                yield result
        finally:
            self.close()

    def close(self) -> None:
        """Close speech consumption; keep native jobs and their permissions alive."""
        self.foreground_closed.set()
        if self.has_jobs() and not self.finished.is_set():
            self._finish_foreground()
            return
        self.shutdown()

    def shutdown(self) -> None:
        """Called only for session teardown or shutdown, not a spoken interruption."""
        self.closed.set()
        self.worker.join(timeout=3.0)
