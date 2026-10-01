"""Session-local rendezvous for agent questions and tool permission decisions."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class AgentPermissionReply(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["speech_to_speech.agent.permission.reply"]
    request_id: str
    decision: Literal["allow", "deny"]
    answers: dict[str, str | list[str]] | None = None
    event_id: str | None = None


AgentPermissionModeName = Literal["default", "acceptEdits", "bypassPermissions", "plan", "dontAsk", "auto"]


class AgentSessionPermissionModeSet(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    mode: AgentPermissionModeName


class AgentPermissionModeSet(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["speech_to_speech.agent.permission_mode.set"]
    mode: AgentPermissionModeName
    event_id: str | None = None


class AgentPermissionModeChanged(BaseModel):
    type: Literal["speech_to_speech.agent.permission_mode"] = "speech_to_speech.agent.permission_mode"
    mode: str
    event_id: str = Field(default_factory=lambda: uuid4().hex)


@dataclass
class AgentRequest:
    request_id: str
    response_key: str
    tool_name: str
    input: dict[str, Any]
    decision: Literal["allow", "deny"] | None = None
    answers: dict[str, str | list[str]] | None = None
    status: str = "pending"


class AgentInteractions:
    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: dict[str, AgentRequest] = {}

    def open(self, response_key: str, tool_name: str, input_data: dict[str, Any]) -> AgentRequest:
        request = AgentRequest(uuid4().hex, response_key, tool_name, deepcopy(input_data))
        with self._lock:
            self._requests[request.request_id] = request
        return request

    def pending(self, response_key: str | None = None) -> list[AgentRequest]:
        with self._lock:
            return [
                r
                for r in self._requests.values()
                if r.decision is None and (response_key is None or r.response_key == response_key)
            ]

    def voice_target(self) -> str | None:
        pending = self.pending()
        # Concurrent prompts require an explicit UI choice; never guess which
        # action a generic spoken approval refers to.
        return pending[0].request_id if len(pending) == 1 else None

    def respond(
        self, request_id: str, decision: Literal["allow", "deny"], answers: dict[str, str | list[str]] | None = None
    ) -> str | None:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None or request.decision is not None:
                return "This agent request is no longer awaiting a reply."
            if decision == "allow" and request.tool_name == "AskUserQuestion":
                questions = request.input.get("questions", [])
                expected = {q.get("question") for q in questions}
                if (
                    not questions
                    or not answers
                    or set(answers) != expected
                    or any(not value for value in answers.values())
                ):
                    return "Answer every question before submitting."
            elif answers is not None:
                return "Answers are only valid for an agent question."
            request.decision = decision
            request.answers = deepcopy(answers)
            request.status = "allowed" if decision == "allow" else "denied"
            return None

    def decision(self, request: AgentRequest) -> tuple[str | None, dict[str, str | list[str]] | None, str]:
        with self._lock:
            return request.decision, deepcopy(request.answers), request.status

    def respond_voice(self, request_id: str, transcript: str) -> str | None:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None or request.decision is not None:
                return "This agent request is no longer awaiting a reply."
            tool_name, input_data = request.tool_name, deepcopy(request.input)
        normalized = re.sub(r"[.!?,]", "", transcript.lower()).strip()
        if re.fullmatch(r"(?:deny|reject) (?:(?:the|this) )?request", normalized):
            return self.respond(request_id, "deny")
        if tool_name == "AskUserQuestion":
            questions = input_data.get("questions", [])
            if len(questions) == 1 and transcript.strip():
                return self.respond(request_id, "allow", {questions[0]["question"]: transcript.strip()})
            return "Use the on-screen form to answer multiple questions."
        if re.fullmatch(r"(?:approve|allow) (?:(?:the|this) )?request", normalized) or normalized == "allow once":
            return self.respond(request_id, "allow")
        return 'Say "approve request" or "deny request", or use the buttons.'

    def close(self, request: AgentRequest, status: str = "cancelled") -> str:
        with self._lock:
            self._requests.pop(request.request_id, None)
            return request.status if request.decision is not None else status

    def cancel_all(self) -> None:
        with self._lock:
            for request in self._requests.values():
                request.decision = "deny"
                request.status = "cancelled"


class AgentPermissionRequested(BaseModel):
    type: Literal["speech_to_speech.agent.permission.requested"] = "speech_to_speech.agent.permission.requested"
    request_id: str
    tool_name: str
    input: dict[str, Any]
    title: str | None = None
    description: str | None = None
    timeout_s: float
    response_id: str | None = None
    event_id: str = Field(default_factory=lambda: uuid4().hex)


class AgentPermissionResolved(BaseModel):
    type: Literal["speech_to_speech.agent.permission.resolved"] = "speech_to_speech.agent.permission.resolved"
    request_id: str
    status: str
    event_id: str = Field(default_factory=lambda: uuid4().hex)


class AgentPermissionVoice(BaseModel):
    type: Literal["speech_to_speech.agent.permission.voice"] = "speech_to_speech.agent.permission.voice"
    request_id: str
    transcript: str
    error: str | None = None
    event_id: str = Field(default_factory=lambda: uuid4().hex)
