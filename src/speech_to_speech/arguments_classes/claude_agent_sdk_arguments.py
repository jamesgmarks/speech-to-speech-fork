from dataclasses import dataclass, field
from typing import Literal, Optional

from speech_to_speech.arguments_classes.language_model_base_arguments import LanguageModelBaseArguments


@dataclass
class ClaudeAgentSDKArguments(LanguageModelBaseArguments):
    model_name: str = field(
        default="sonnet", metadata={"help": "Claude Code model ID or alias (e.g. sonnet, haiku, opus)."}
    )
    stream_batch_sentences: int = field(
        default=1, metadata={"help": "Sentences per spoken batch; 1 minimizes speech startup latency."}
    )
    compact_history: bool = field(
        default=False,
        metadata={"help": "Summarize older pipeline turns with an additional SDK request instead of trimming them."},
    )
    claude_agent_stream: bool = field(
        default=True, metadata={"help": "Stream assistant text into the speech pipeline."}
    )
    claude_agent_max_tokens: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum output tokens per model request, including thinking and tool calls. Uses CLAUDE_CODE_MAX_OUTPUT_TOKENS; unset preserves Claude Code's default. Not a total agent-run budget."
        },
    )
    claude_agent_effort: Optional[Literal["low", "medium", "high", "xhigh", "max"]] = field(
        default=None,
        metadata={"help": "Reasoning effort. Lower effort can reduce latency; unset preserves Claude Code's default."},
    )
    claude_agent_thinking: Optional[Literal["adaptive", "enabled", "disabled"]] = field(
        default=None, metadata={"help": "Thinking mode; model support varies. Unset preserves Claude Code's default."}
    )
    claude_agent_thinking_budget_tokens: int = field(
        default=1024,
        metadata={"help": "Thinking budget when --claude_agent_thinking enabled. Must be smaller than max_tokens."},
    )
    claude_agent_max_turns: Optional[int] = field(
        default=None, metadata={"help": "Maximum agentic model turns per response; unset leaves the SDK default."}
    )
    claude_agent_max_budget_usd: Optional[float] = field(
        default=None, metadata={"help": "Optional SDK cost limit per response."}
    )
    claude_agent_request_timeout_s: float = field(
        default=120.0,
        metadata={
            "help": "Generation deadline including CLI startup and tools, excluding human permission wait. Timeout interrupts the SDK."
        },
    )
    claude_agent_background_timeout_s: float = field(
        default=1800.0,
        metadata={
            "help": "Deadline for retained background SDK jobs after the voice response ends, excluding permission wait."
        },
    )
    claude_agent_max_background_sessions: int = field(
        default=4,
        metadata={"help": "Maximum simultaneous SDK clients owning native background jobs per connected call."},
    )
    claude_agent_orchestrator: bool = field(
        default=True,
        metadata={
            "help": "Prompt Claude to delegate substantial work in the background while keeping voice turns short. All native tools stay enabled."
        },
    )
    claude_agent_max_retries: Optional[int] = field(
        default=None,
        metadata={
            "help": "Claude Code API retry limit; unset preserves its default. Fewer retries reduce outage wait time."
        },
    )
    claude_agent_cwd: Optional[str] = field(
        default=None, metadata={"help": "Workspace directory for Claude Code tools and project settings."}
    )
    claude_agent_cli_path: Optional[str] = field(
        default=None, metadata={"help": "Optional Claude Code executable path; otherwise the SDK uses its bundled CLI."}
    )
    claude_agent_permission_mode: Literal["default", "acceptEdits", "plan", "bypassPermissions", "dontAsk", "auto"] = (
        field(
            default="default",
            metadata={
                "help": "Claude Code permission mode. All tools remain available; default retains normal approval rules. Pending approvals are shown in the demo and can be answered by voice."
            },
        )
    )
    claude_agent_session_state_path: str = field(
        default="~/.speech-to-speech/claude-sessions.json",
        metadata={
            "help": "Local app-owned SDK session identities and permission modes, resumed across service restarts."
        },
    )
    claude_agent_allowed_tools: list[str] = field(
        default_factory=list,
        metadata={"help": "Tool names or scoped rules to auto-approve. This does not limit the available toolset."},
    )
    claude_agent_setting_sources: list[str] = field(
        default_factory=lambda: ["user", "project", "local"],
        metadata={"help": "Claude Code settings sources to load: user, project, local."},
    )
    claude_agent_mcp_config: Optional[str] = field(
        default=None,
        metadata={"help": "Optional MCP configuration JSON file; configured Claude Code tools remain available."},
    )

    claude_agent_terminal_tool: bool = field(
        default=True,
        metadata={
            "help": "Provide this app's launch_agent_terminal tool to open independent interactive Claude Code sessions in macOS Terminal. Global and project tool registrations are unchanged."
        },
    )
    claude_agent_session_tools: bool = field(
        default=True,
        metadata={
            "help": "Provide app-only tools to create, list, message, and stop persistent independent Claude SDK sessions."
        },
    )
    claude_agent_max_independent_sessions: int = field(
        default=4,
        metadata={
            "help": "Maximum active app-owned independent SDK sessions. These survive voice disconnects until stopped or the app exits."
        },
    )

    claude_agent_permission_timeout_s: float = field(
        default=300.0,
        metadata={
            "help": "Time allowed to answer each permission prompt or agent question. Generation timeout pauses while waiting."
        },
    )
