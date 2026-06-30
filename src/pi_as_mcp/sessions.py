from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pi_as_mcp.pi_rpc import (
    DEFAULT_MODEL_VALIDATION_TIMEOUT_SECONDS,
    READ_ONLY_GUARD,
    PiRpcError,
    PiRpcRunner,
    ToolCall,
    ToolMode,
    assistant_message_text,
    compact_event,
    extract_message_text,
    extract_text_preview,
    guard_prompt,
    last_assistant_text,
    resolve_cwd,
    resolve_model,
    validate_timeout,
    validate_tool_mode,
)

ReplyBehavior = Literal["auto", "follow-up", "steer"]
ResponseVerbosity = Literal["summary", "normal", "debug"]
# Only agents that are spinning up or actively running a turn consume a
# concurrency slot. An "idle" agent is a live Pi worker that finished its turn
# and is just waiting for a possible follow-up reply; it does not block new
# delegations (otherwise a single-slot model jams after one delegation and never
# frees up, since idle workers are only removed on an explicit stop).
CONCURRENCY_COUNTED_STATUSES = {"starting", "running"}

# Turns are bounded by inactivity, not wall-clock: a turn that keeps streaming
# tokens, calling tools, or returning results stays alive indefinitely; only a
# running turn that produces nothing for this long is treated as stalled.
#
# The one fully-silent window is *prompt processing* (prefill) before the first
# token streams back. Local models running at the edge of a machine's capability
# can spend many minutes there, so this is generous (60 min) to avoid killing a
# healthy worker mid-prefill — which the model server logs as "Client
# disconnected. Stopping generation". Override via PI_AS_MCP_INACTIVITY_TIMEOUT_SECONDS.
def _default_inactivity_timeout_seconds() -> float:
    raw = os.environ.get("PI_AS_MCP_INACTIVITY_TIMEOUT_SECONDS")
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return 3600.0


DEFAULT_INACTIVITY_TIMEOUT_SECONDS = _default_inactivity_timeout_seconds()
# Separate, short bound for the prompt-accept handshake (Pi acks immediately).
PROMPT_ACK_TIMEOUT_SECONDS = 30
# How long a worker may sit idle (turn finished, awaiting a possible follow-up)
# before it is evicted: the Pi subprocess is killed to reclaim memory while the
# conversation is preserved on disk, so a later reply transparently respawns a
# worker that resumes the session by id. 0 disables eviction (workers stay
# resident until an explicit stop or parent death). Eviction only applies when
# session persistence is enabled (a session_dir is configured).
# Tradeoff: a resumed reply pays a cold cost (respawn pi + replay the on-disk
# session); raise this (or set 0) for interactive use, keep it low for fan-out.
DEFAULT_IDLE_EVICTION_SECONDS = 120

# (agent_id, record) -> None. Receives full-fidelity transcript records.
TranscriptSink = Callable[[str, dict[str, Any]], None]


def text_preview(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def describe_tool_call(tool_name: str, args: dict[str, Any]) -> str:
    if not args:
        return f"using {tool_name}"
    if tool_name in {"read", "ls"}:
        path = args.get("path") or args.get("filePath") or args.get("file_path")
        if isinstance(path, str):
            return f"{tool_name} {path}"
    if tool_name in {"grep", "find"}:
        pattern = args.get("pattern") or args.get("query") or args.get("glob")
        if isinstance(pattern, str):
            return f"{tool_name} {pattern!r}"
    if tool_name == "bash":
        command = args.get("command")
        if isinstance(command, str):
            return f"bash {text_preview(command, 120)!r}"
    return f"using {tool_name}"


def validate_response_verbosity(verbosity: str) -> ResponseVerbosity:
    if verbosity not in {"summary", "normal", "debug"}:
        raise PiRpcError("verbosity must be one of: summary, normal, debug")
    return verbosity  # type: ignore[return-value]


def _is_thinking_block_type(block_type: Any) -> bool:
    if not isinstance(block_type, str):
        return False
    lowered = block_type.lower()
    return "thinking" in lowered or "reasoning" in lowered


def _accumulate_message_update_parts(value: Any, parts: dict[str, list[str]], *, thinking: bool) -> None:
    """Walk a streamed message_update, bucketing deltas into text vs reasoning.

    Pi tags each delta with a ``type`` (``text_delta`` / ``thinking_delta``) and
    carries the actual characters in ``text``, so the channel is decided by the
    nearest ``type`` field, not by the key the string lives under. Some providers
    instead wrap the delta in an ``assistantMessageEvent`` envelope and carry the
    characters in ``content`` (e.g. ``{"type": "message_update",
    "assistantMessageEvent": {"type": "thinking_delta", "content": "..."}}``), so
    we descend into that envelope and treat a string ``content`` as a delta too.
    """
    if isinstance(value, str):
        parts["thinking" if thinking else "text"].append(value)
        return
    if isinstance(value, list):
        for item in value:
            _accumulate_message_update_parts(item, parts, thinking=thinking)
        return
    if not isinstance(value, dict):
        return

    block_type = value.get("type")
    local_thinking = thinking
    if isinstance(block_type, str) and block_type:
        if _is_thinking_block_type(block_type):
            local_thinking = True
        elif "text" in block_type.lower():
            local_thinking = False

    for key in ("thinking_delta", "thinkingDelta"):
        item = value.get(key)
        if isinstance(item, str):
            parts["thinking"].append(item)
    for key in ("text_delta", "textDelta"):
        item = value.get(key)
        if isinstance(item, str):
            parts["text"].append(item)
    text = value.get("text")
    if isinstance(text, str):
        parts["thinking" if local_thinking else "text"].append(text)
    reasoning = value.get("thinking")
    if isinstance(reasoning, str):
        parts["thinking"].append(reasoning)
    content = value.get("content")
    if isinstance(content, str):
        parts["thinking" if local_thinking else "text"].append(content)

    for nested_key in ("delta", "content", "assistantMessageEvent"):
        nested = value.get(nested_key)
        if isinstance(nested, dict | list):
            _accumulate_message_update_parts(nested, parts, thinking=local_thinking)


def extract_message_update_parts(value: Any) -> tuple[str, str]:
    """Return ``(text, thinking)`` deltas carried by a streamed message_update."""
    parts: dict[str, list[str]] = {"text": [], "thinking": []}
    _accumulate_message_update_parts(value, parts, thinking=False)
    return "".join(parts["text"]), "".join(parts["thinking"])


def extract_message_update_text(value: Any) -> str:
    text, thinking = extract_message_update_parts(value)
    return text + thinking


def extract_thinking_text(message: dict[str, Any] | None) -> str:
    """Pull reasoning content from a finalized message (mirror of text extraction)."""
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, list):
        return ""

    chunks: list[str] = []
    for item in content:
        if not isinstance(item, dict) or not _is_thinking_block_type(item.get("type")):
            continue
        for key in ("thinking", "text", "reasoning"):
            value = item.get(key)
            if isinstance(value, str) and value:
                chunks.append(value)
                break
    return "".join(chunks).strip()


def number_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return None
    return None


def int_usage_value(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = number_value(usage.get(key))
        if value is not None:
            return int(value)
    return 0


def float_usage_value(usage: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = number_value(usage.get(key))
        if value is not None:
            return value
    return None


def empty_usage() -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
        "context_used_tokens": None,
        "context_limit_tokens": None,
        "context_percent": None,
        "tokens_per_second": None,
    }


def usage_stats_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    context_usage = payload.get("contextUsage") if isinstance(payload.get("contextUsage"), dict) else {}
    context_source = context or context_usage

    stats = {
        "input_tokens": int_usage_value(
            payload,
            "input",
            "inputTokens",
            "input_tokens",
            "promptTokens",
            "prompt_tokens",
        ),
        "output_tokens": int_usage_value(
            payload,
            "output",
            "outputTokens",
            "output_tokens",
            "completionTokens",
            "completion_tokens",
        ),
        "cache_read_tokens": int_usage_value(
            payload,
            "cacheRead",
            "cache_read",
            "cacheReadTokens",
            "cache_read_tokens",
            "cachedInputTokens",
            "cached_input_tokens",
        ),
        "cache_write_tokens": int_usage_value(
            payload,
            "cacheWrite",
            "cache_write",
            "cacheWriteTokens",
            "cache_write_tokens",
        ),
        "total_tokens": int_usage_value(payload, "totalTokens", "total_tokens", "totalTokenCount"),
        "context_used_tokens": int_usage_value(
            payload,
            "contextUsed",
            "context_used",
            "contextTokens",
            "context_tokens",
        )
        or int_usage_value(context_source, "used", "tokens", "tokenCount", "token_count")
        or None,
        "context_limit_tokens": int_usage_value(
            payload,
            "contextLimit",
            "context_limit",
            "contextWindow",
            "context_window",
        )
        or int_usage_value(context_source, "limit", "window", "max", "maximum")
        or None,
        "context_percent": float_usage_value(
            payload,
            "contextPercent",
            "context_percent",
            "contextUsedPercent",
            "context_used_percent",
        )
        or float_usage_value(context_source, "percent", "percentage"),
        "tokens_per_second": float_usage_value(
            payload,
            "tokensPerSecond",
            "tokens_per_second",
            "outputTokensPerSecond",
            "output_tokens_per_second",
        ),
    }
    return stats


def usage_stats_from_message(message: Any) -> dict[str, Any] | None:
    if not isinstance(message, dict):
        return None
    payload = message.get("usage")
    if isinstance(payload, dict):
        return usage_stats_from_payload(payload)
    nested = message.get("message")
    if isinstance(nested, dict):
        return usage_stats_from_message(nested)
    return None


def usage_message_key(message: dict[str, Any], *, turn: int) -> str | None:
    for key in ("responseId", "response_id", "id"):
        value = message.get(key)
        if isinstance(value, str | int) and value:
            return f"{key}:{value}"

    nested = message.get("message")
    if isinstance(nested, dict):
        nested_key = usage_message_key(nested, turn=turn)
        if nested_key:
            return nested_key

    timestamp = message.get("timestamp")
    stop_reason = message.get("stopReason") or message.get("stop_reason")
    role = message.get("role")
    text = extract_message_text(message)
    if timestamp is None and not stop_reason and not text:
        return None
    return json.dumps(
        [turn, role, timestamp, stop_reason, text_preview(text, 96)],
        ensure_ascii=False,
        sort_keys=True,
    )


def merge_usage(total: dict[str, Any], stats: dict[str, Any]) -> None:
    for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "total_tokens"):
        total[key] = int(total.get(key) or 0) + int(stats.get(key) or 0)

    response_total = int(stats.get("total_tokens") or 0)
    if response_total <= 0:
        response_total = sum(
            int(stats.get(key) or 0)
            for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
        )

    context_used = stats.get("context_used_tokens") or response_total or None
    if context_used is not None:
        total["context_used_tokens"] = int(context_used)

    for key in ("context_limit_tokens", "context_percent", "tokens_per_second"):
        value = stats.get(key)
        if value is not None:
            total[key] = value


def usage_to_json(usage: dict[str, Any], *, elapsed_seconds: float) -> dict[str, Any]:
    data = empty_usage()
    data.update(usage)
    for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "total_tokens"):
        data[key] = int(data.get(key) or 0)

    if data["total_tokens"] <= 0:
        data["total_tokens"] = sum(
            data[key] for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
        )

    context_used = number_value(data.get("context_used_tokens"))
    context_limit = number_value(data.get("context_limit_tokens"))
    context_percent = number_value(data.get("context_percent"))
    if context_percent is None and context_used is not None and context_limit:
        context_percent = (context_used / context_limit) * 100
    if context_percent is not None and 0 < context_percent <= 1:
        context_percent *= 100

    data["context_used_tokens"] = int(context_used) if context_used is not None else None
    data["context_limit_tokens"] = int(context_limit) if context_limit is not None else None
    data["context_percent"] = round(context_percent, 2) if context_percent is not None else None

    tokens_per_second = number_value(data.get("tokens_per_second"))
    if tokens_per_second is None and elapsed_seconds > 0 and data["output_tokens"] > 0:
        tokens_per_second = data["output_tokens"] / elapsed_seconds
    data["tokens_per_second"] = round(tokens_per_second, 2) if tokens_per_second is not None else None
    return data


@dataclass
class SessionSnapshot:
    agent_id: str
    status: str
    created_at: float
    cwd: str
    provider: str
    model: str
    tool_mode: str
    final_text: str
    turn_count: int
    usage: dict[str, Any]
    tool_calls: list[ToolCall]
    event_counts: dict[str, int]
    active_listeners: int = 0
    stderr_tail: list[str] = field(default_factory=list)
    event_tail: list[dict[str, Any]] = field(default_factory=list)
    prompts: list["PromptRecord"] = field(default_factory=list)
    transcript: list["TranscriptItem"] = field(default_factory=list)
    error: str | None = None

    def to_json(self, *, verbosity: ResponseVerbosity = "normal") -> dict[str, Any]:
        validate_response_verbosity(verbosity)
        data: dict[str, Any] = {
            "agent_id": self.agent_id,
            "status": self.status,
            "created_at": self.created_at,
            "cwd": self.cwd,
            "provider": self.provider,
            "model": self.model,
            "tool_mode": self.tool_mode,
            "final_text": self.final_text,
            "turn_count": self.turn_count,
            "usage": self.usage,
            "event_counts": self.event_counts,
            "tool_call_count": len(self.tool_calls),
            "active_listeners": self.active_listeners,
            "observing_with_piw": self.active_listeners > 0,
        }
        if verbosity in {"normal", "debug"}:
            data["tool_calls"] = [tool.to_json() for tool in self.tool_calls]
            data["prompts"] = [prompt.to_json() for prompt in self.prompts]
            data["initial_request"] = self.prompts[0].to_json() if self.prompts else None
            data["transcript"] = [item.to_json() for item in self.transcript]
            if self.stderr_tail:
                data["stderr_tail"] = self.stderr_tail
        if verbosity == "debug" and self.event_tail:
            data["event_tail"] = self.event_tail
        if self.error:
            data["error"] = self.error
        return data


SessionSnapshotObserver = Callable[[str, SessionSnapshot], None]


@dataclass
class SessionSummary:
    agent_id: str
    status: str
    created_at: float
    state_seconds: float
    total_seconds: float
    last_activity_seconds: float
    last_action: str
    cwd: str
    model: str
    tool_mode: str
    turn_count: int
    usage: dict[str, Any]
    final_text_preview: str
    recent_actions: list[str]
    prompt_count: int
    initial_request_preview: str
    current_tool: str | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "agent_id": self.agent_id,
            "status": self.status,
            "created_at": self.created_at,
            "state_seconds": self.state_seconds,
            "total_seconds": self.total_seconds,
            "last_activity_seconds": self.last_activity_seconds,
            "last_action": self.last_action,
            "cwd": self.cwd,
            "model": self.model,
            "tool_mode": self.tool_mode,
            "turn_count": self.turn_count,
            "usage": self.usage,
            "final_text_preview": self.final_text_preview,
            "recent_actions": self.recent_actions,
            "prompt_count": self.prompt_count,
            "initial_request_preview": self.initial_request_preview,
        }
        if self.current_tool:
            data["current_tool"] = self.current_tool
        if self.error:
            data["error"] = self.error
        return data


@dataclass
class PromptRecord:
    turn: int
    behavior: str
    text: str
    accepted: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "behavior": self.behavior,
            "text": self.text,
            "accepted": self.accepted,
        }


@dataclass
class TranscriptItem:
    kind: str
    turn: int
    role: str | None = None
    text: str | None = None
    behavior: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    is_error: bool | None = None
    result_preview: str | None = None
    event_type: str | None = None

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "turn": self.turn,
        }
        for key, value in {
            "role": self.role,
            "text": self.text,
            "behavior": self.behavior,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "is_error": self.is_error,
            "result_preview": self.result_preview,
            "event_type": self.event_type,
        }.items():
            if value not in (None, {}, ""):
                data[key] = value
        return data


class PiAgentSession:
    def __init__(
        self,
        *,
        agent_id: str,
        runner: PiRpcRunner,
        prompt: str,
        cwd: str | None,
        model: str | None,
        provider: str | None,
        tool_mode: ToolMode,
        include_events: bool,
        observer: SessionSnapshotObserver | None = None,
        transcript_sink: TranscriptSink | None = None,
        unsafe_read_only: bool = False,
        session_dir: Path | None = None,
        idle_eviction_seconds: float = DEFAULT_IDLE_EVICTION_SECONDS,
    ) -> None:
        if not prompt.strip():
            raise PiRpcError("prompt must not be empty")
        validate_tool_mode(tool_mode)

        self.agent_id = agent_id
        self.runner = runner
        # When set, the worker persists its conversation to disk (keyed by
        # agent_id) so it can be evicted while idle and resumed on a later reply.
        self.session_dir = session_dir
        self.idle_eviction_seconds = idle_eviction_seconds
        self.cwd_path = resolve_cwd(cwd)
        self.model_spec = resolve_model(model, provider)
        self.tool_mode = tool_mode
        # Operator policy (config: agents.unsafe_read_only): a read-only request
        # runs with full tools but every prompt is guarded into its contract. The
        # reported tool_mode stays as requested; the worker just gets more tools.
        self._unsafe_read_only = unsafe_read_only and tool_mode == "read-only"
        self._guard: str | None = READ_ONLY_GUARD if self._unsafe_read_only else None
        self.include_events = include_events
        self.inactivity_timeout_seconds = DEFAULT_INACTIVITY_TIMEOUT_SECONDS
        self.created_at = time.time()
        self.created_monotonic = time.monotonic()
        self.state_since_monotonic = self.created_monotonic
        self.last_activity_monotonic = self.created_monotonic
        self.last_action = "starting Pi worker"
        self._recent_actions: deque[str] = deque(maxlen=8)
        self._record_action_locked(self.last_action)
        self.current_tool: str | None = None
        self._usage = empty_usage()
        self._seen_usage_keys: set[str] = set()

        self.model_context_tokens = runner.validate_model(
            self.model_spec.provider,
            self.model_spec.model,
            timeout_seconds=DEFAULT_MODEL_VALIDATION_TIMEOUT_SECONDS,
        )
        self._usage["context_limit_tokens"] = self.model_context_tokens

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._write_lock = threading.Lock()
        self._responses: dict[str, dict[str, Any]] = {}
        self._stderr_tail: deque[str] = deque(maxlen=25)
        self._event_tail: deque[dict[str, Any]] = deque(maxlen=50)
        self._event_counts: Counter[str] = Counter()
        self._tool_calls_by_id: dict[str, ToolCall] = {}
        self._tool_calls: list[ToolCall] = []
        self._prompts: list[PromptRecord] = []
        self._transcript: list[TranscriptItem] = []
        self._final_text = ""
        self._turn_count = 0
        self._running = False
        self._status = "starting"
        # Number of parents currently blocked in listen() on this agent — i.e.
        # actively watching it with `piw`/`wait`. Surfaced so the dashboard can
        # show whether a delegating agent is still tailing its subagent.
        self._active_listeners = 0
        self._error: str | None = None
        self._closed = False
        # True while the worker process has been deliberately killed to reclaim
        # memory but the agent remains resumable from its persisted session.
        self._evicted = False
        self._observer = observer
        self._transcript_sink = transcript_sink
        # Accumulates streamed assistant/reasoning deltas so the transcript log
        # records one coalesced reasoning entry per stream instead of one line
        # per token. Only ever touched by the stdout reader thread.
        self._stream_buffer = ""
        self.process: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

        # Spawn the Pi worker and start its IO readers. A single watchdog runs for
        # the whole agent lifetime (across evict/respawn cycles).
        with self._lock:
            self._start_worker_locked()
        self._watchdog_thread = threading.Thread(target=self._watchdog, daemon=True)
        self._watchdog_thread.start()

        self.send(prompt, behavior="auto")

    @property
    def _persist_enabled(self) -> bool:
        return self.session_dir is not None

    def _start_worker_locked(self) -> None:
        """Spawn (or respawn) the Pi RPC subprocess and start its IO readers.

        Called once at construction and again whenever a reply arrives for a
        worker that was evicted (or exited). With persistence enabled the new
        process resumes the same on-disk session by id, restoring prior context.
        """
        args = self.runner._build_args(
            self.model_spec.provider,
            self.model_spec.model,
            "full" if self._guard else self.tool_mode,
            session_id=self.agent_id if self._persist_enabled else None,
            session_dir=str(self.session_dir) if self._persist_enabled else None,
        )
        env = os.environ.copy()
        env["PI_OFFLINE"] = "1"
        # Lift undici's default 300s read timeouts in the Node worker so a long
        # prompt-processing (prefill) phase on a slow local model doesn't abort the
        # HTTP request mid-generation (server-side "Client disconnected"). The
        # preload resolves undici from Pi's own node_modules via PI_FETCH_DISPATCH_BASE.
        preload = Path(__file__).resolve().parent / "assets" / "disable_fetch_timeouts.mjs"
        if preload.is_file():
            env["PI_FETCH_DISPATCH_BASE"] = self.runner.pi_bin
            import_flag = f"--import={preload.as_uri()}"
            existing_node_options = env.get("NODE_OPTIONS", "").strip()
            env["NODE_OPTIONS"] = (
                f"{existing_node_options} {import_flag}".strip()
                if existing_node_options
                else import_flag
            )
        process: subprocess.Popen[str] = subprocess.Popen(
            args,
            cwd=str(self.cwd_path),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=os.name == "posix",
        )
        self.process = process
        self._evicted = False
        self._responses.clear()
        self._stdout_thread = threading.Thread(target=self._read_stdout, args=(process,), daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, args=(process,), daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _ensure_worker_locked(self) -> None:
        """Guarantee a live Pi worker is attached, respawning an evicted/exited
        one from its persisted session. Raises if the agent is gone for good."""
        process = self.process
        if process is not None and process.poll() is None and not self._evicted:
            return
        if self._closed:
            raise PiRpcError(f"agent {self.agent_id} is not running")
        if not self._persist_enabled:
            # No persisted session to resume from; an ephemeral worker that exited
            # cannot be revived.
            raise PiRpcError(f"agent {self.agent_id} is not running")
        self._set_status_locked("starting", "resuming persisted session")
        self._start_worker_locked()

    def send(
        self,
        prompt: str,
        *,
        behavior: ReplyBehavior = "auto",
    ) -> dict[str, Any]:
        if not prompt.strip():
            raise PiRpcError("prompt must not be empty")

        # Under an unsafe-* policy the tools are fully open; the contract is
        # enforced only by this prepended instruction. Apply it to every prompt
        # (initial and replies) so later turns stay guarded too.
        prompt = guard_prompt(self._guard, prompt)

        with self._lock:
            # Respawn a worker that was evicted (or exited) while idle, resuming
            # its persisted session so this reply continues the same conversation.
            self._ensure_worker_locked()
            is_running = self._running
            turn_count_before = self._turn_count
            turn = turn_count_before + 1
            effective_behavior = self._effective_behavior(behavior, is_running=is_running)
            prompt_record = PromptRecord(turn=turn, behavior=effective_behavior, text=prompt)
            self._prompts.append(prompt_record)
            self._append_transcript_locked(
                TranscriptItem(
                    kind="prompt",
                    turn=turn,
                    role="user",
                    text=prompt,
                    behavior=effective_behavior,
                    event_type="prompt",
                )
            )
            self._touch_locked(f"queued prompt: {text_preview(prompt)}")

        self._emit_transcript("prompt", {"text": prompt, "behavior": effective_behavior}, turn=turn)

        request_id = f"{self.agent_id}-{uuid.uuid4().hex}"
        command: dict[str, Any] = {
            "id": request_id,
            "type": "prompt",
            "message": prompt,
        }
        if effective_behavior == "steer":
            command["streamingBehavior"] = "steer"
        elif effective_behavior == "follow-up":
            command["streamingBehavior"] = "followUp"

        with self._write_lock:
            process = self.process
            if process is None or process.stdin is None:
                raise PiRpcError(f"agent {self.agent_id} stdin is closed")
            try:
                process.stdin.write(json.dumps(command, ensure_ascii=False) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise PiRpcError(f"agent {self.agent_id} pipe closed") from exc

        response = self._wait_for_response(request_id, timeout=PROMPT_ACK_TIMEOUT_SECONDS)
        with self._lock:
            prompt_record.accepted = bool(response.get("success"))
        if not response.get("success"):
            raise PiRpcError(str(response.get("error") or "Pi rejected prompt"))
        with self._lock:
            if self._turn_count == turn_count_before and self._status not in {"error", "timeout", "stopped"}:
                self._set_status_locked("running", "Pi accepted prompt")
        return response

    def snapshot(self, *, include_events: bool | None = None) -> SessionSnapshot:
        with self._lock:
            return self._snapshot_locked(include_events=include_events)

    def summary(self) -> SessionSummary:
        with self._lock:
            return self._summary_locked()

    def listen(
        self,
        *,
        after_turn_count: int,
        timeout_seconds: int,
        include_events: bool = False,
    ) -> tuple[SessionSnapshot, bool]:
        validate_timeout(timeout_seconds)
        if after_turn_count < 0:
            raise PiRpcError("after_turn_count must be >= 0")

        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            # Counted while blocked here so the dashboard (which polls) can show
            # that a parent is actively tailing this agent with piw.
            self._active_listeners += 1
            try:
                while True:
                    snapshot = self._snapshot_locked(include_events=include_events)
                    if snapshot.turn_count > after_turn_count or snapshot.status in {
                        "error",
                        "timeout",
                        "stopped",
                        "exited",
                    }:
                        return snapshot, False

                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return snapshot, True
                    self._condition.wait(timeout=remaining)
            finally:
                self._active_listeners -= 1

    def stop(self) -> SessionSnapshot:
        self._terminate(mark_status="stopped")
        return self.snapshot(include_events=True)

    def _wait_for_response(self, request_id: str, *, timeout: int) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while request_id not in self._responses:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PiRpcError(f"timed out waiting for Pi to accept request {request_id}")
                self._condition.wait(timeout=remaining)
            return self._responses.pop(request_id)

    def _read_stdout(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        observer_event: tuple[str, SessionSnapshot] | None = None
        terminal_status: str | None = None
        try:
            for line in process.stdout:
                if process is not self.process:
                    # A newer worker has taken over after a respawn; stop draining
                    # this dead pipe's leftovers into the live agent.
                    break
                self._handle_stdout_line(line.rstrip("\n").rstrip("\r"))
        finally:
            self._flush_stream_buffer(turn=self._turn_count + 1)
            with self._condition:
                # Only the currently-attached worker may declare the agent exited.
                # An evicted or replaced process draining its pipe must not clobber
                # the live (idle/resumable or freshly respawned) state.
                if (
                    process is self.process
                    and not self._closed
                    and not self._evicted
                    and self._status not in {"error", "timeout", "stopped"}
                ):
                    self._set_status_locked("exited", "Pi worker exited")
                    observer_event = ("agent_updated", self._snapshot_locked(include_events=False))
                    terminal_status = "exited"
                self._condition.notify_all()
        if terminal_status is not None:
            self._emit_transcript(
                "status", {"status": terminal_status, "error": self._error or ""}, turn=self._turn_count + 1
            )
        if observer_event is not None:
            self._notify_observer(*observer_event)

    def _read_stderr(self, process: subprocess.Popen[str]) -> None:
        assert process.stderr is not None
        for line in process.stderr:
            with self._lock:
                text = line.rstrip("\n").rstrip("\r")
                if text:
                    self._stderr_tail.append(text)

    def _handle_stdout_line(self, line: str) -> None:
        observer_event: tuple[str, SessionSnapshot] | None = None
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            with self._lock:
                self._event_counts["invalid_json"] += 1
                self._event_tail.append({"type": "invalid_json", "line": line[:500]})
            self._emit_transcript("invalid_json", {"line": line}, turn=self._turn_count + 1)
            return

        event_type = event.get("type")

        with self._condition:
            turn_for_record = self._turn_count + 1
            if isinstance(event_type, str):
                self._event_counts[event_type] += 1
            if self.include_events:
                self._event_tail.append(compact_event(event))

            if event_type == "response":
                request_id = event.get("id")
                if isinstance(request_id, str):
                    self._responses[request_id] = event
                self._condition.notify_all()
                return

            if event_type == "agent_start":
                self._set_status_locked("running", "Pi turn started")
                # A fresh attempt is underway; drop any error from a prior turn so
                # a recovered/re-delegated agent doesn't report a stale failure.
                self._clear_error_locked()

            if event_type == "message_update":
                self._record_message_update_locked(event)

            if event_type == "message_end":
                if not self._record_usage_from_message_locked(event.get("message")):
                    self._record_usage_from_event_locked(event)
                self._record_thinking_locked("message_end", event.get("message"))
                self._record_message_locked("message_end", event.get("message"))
                text = assistant_message_text(event.get("message"))
                if text:
                    self._final_text = text
                    self._touch_locked("received assistant message")

            if event_type == "turn_end":
                if not self._record_usage_from_message_locked(event.get("message")):
                    self._record_usage_from_event_locked(event)
                self._record_thinking_locked("turn_end", event.get("message"))
                self._record_message_locked("turn_end", event.get("message"))
                text = assistant_message_text(event.get("message"))
                if text:
                    self._final_text = text
                    self._touch_locked("turn ended")

            if event_type == "tool_execution_start":
                call_id = event.get("toolCallId")
                tool_name = str(event.get("toolName") or "unknown")
                args = event.get("args") if isinstance(event.get("args"), dict) else {}
                tool = ToolCall(
                    id=call_id if isinstance(call_id, str) else None,
                    name=tool_name,
                    args=args,
                )
                self._tool_calls.append(tool)
                if tool.id:
                    self._tool_calls_by_id[tool.id] = tool
                self.current_tool = tool.name
                self._append_transcript_locked(
                    TranscriptItem(
                        kind="tool_call",
                        turn=self._turn_count + 1,
                        tool_call_id=tool.id,
                        tool_name=tool.name,
                        tool_args=tool.args,
                        event_type="tool_execution_start",
                    )
                )
                self._touch_locked(describe_tool_call(tool.name, tool.args))

            if event_type == "tool_execution_end":
                call_id = event.get("toolCallId")
                tool = self._tool_calls_by_id.get(call_id) if isinstance(call_id, str) else None
                if tool is None:
                    tool = ToolCall(
                        id=call_id if isinstance(call_id, str) else None,
                        name=str(event.get("toolName") or "unknown"),
                    )
                    self._tool_calls.append(tool)
                tool.is_error = bool(event.get("isError"))
                tool.result_preview = extract_text_preview(event.get("result"))
                self.current_tool = None
                self._append_transcript_locked(
                    TranscriptItem(
                        kind="tool_result",
                        turn=self._turn_count + 1,
                        tool_call_id=tool.id,
                        tool_name=tool.name,
                        is_error=tool.is_error,
                        result_preview=tool.result_preview,
                        event_type="tool_execution_end",
                    )
                )
                suffix = "failed" if tool.is_error else "completed"
                self._touch_locked(f"{tool.name} {suffix}")

            if event_type == "agent_end":
                messages = event.get("messages")
                recorded_usage = False
                if isinstance(messages, list):
                    for message in messages:
                        recorded_usage = self._record_usage_from_message_locked(message) or recorded_usage
                    text = last_assistant_text(messages)
                    if text:
                        self._final_text = text
                if not recorded_usage:
                    self._record_usage_from_event_locked(event)
                self._turn_count += 1
                self.current_tool = None
                self._set_status_locked("idle", f"idle after turn {self._turn_count}")
                observer_event = ("agent_updated", self._snapshot_locked(include_events=False))

            # The provider hit a transient failure (rate limit, 5xx, timeout, ...)
            # and the runtime is retrying. Record the reason so a peek during the
            # retry window explains why the agent appears stuck.
            if event_type == "auto_retry_start":
                message = event.get("errorMessage") or event.get("error")
                if isinstance(message, str) and message.strip():
                    detail = message.strip()
                    attempt = event.get("attempt")
                    max_attempts = event.get("maxAttempts")
                    if isinstance(attempt, int) and isinstance(max_attempts, int):
                        detail = f"{detail} (auto-retry {attempt}/{max_attempts})"
                    self._record_provider_error_locked(detail)

            # Retries finished: clear on success, otherwise surface the final error
            # so the caller learns the agent produced no output *and why*.
            if event_type == "auto_retry_end":
                if event.get("success"):
                    self._clear_error_locked()
                else:
                    self._record_provider_error_locked(
                        event.get("finalError")
                        or event.get("errorMessage")
                        or event.get("error")
                    )

            # An explicit error event from the runtime/provider.
            if event_type == "error":
                self._record_provider_error_locked(
                    event.get("error")
                    or event.get("errorMessage")
                    or event.get("message")
                )

            # Defensive catch-all: any other event that carries an error payload
            # (provider-specific shapes) still gets forwarded rather than dropped.
            if event_type not in {
                "response",
                "auto_retry_start",
                "auto_retry_end",
                "error",
            }:
                fallback_error = event.get("finalError") or event.get("errorMessage")
                if isinstance(fallback_error, str) and fallback_error.strip():
                    self._record_provider_error_locked(fallback_error)

            self._condition.notify_all()
        if observer_event is not None:
            self._notify_observer(*observer_event)
        self._persist_transcript_event(event_type, event, turn=turn_for_record)

    def _derive_status_locked(self) -> str:
        """Compute the derived status string without building a snapshot.

        Single source of truth for status derivation; ``_snapshot_locked`` and
        the lightweight status accessors all use it so the result stays
        byte-for-byte identical across callers.
        """
        status = self._status
        process = self.process
        process_dead = process is None or process.poll() is not None
        if self._closed and self._status == "stopped":
            status = "stopped"
        elif self._closed:
            status = self._status
        elif self._evicted:
            # Worker reclaimed to save memory; the agent is idle and resumable
            # from its persisted session on the next reply.
            status = "idle"
        elif process_dead and status not in {"error", "timeout"}:
            status = "exited"
        elif self._running:
            status = "running"
        elif status in {"starting", "running"}:
            status = "idle"
        return status

    def _status_info_locked(self) -> tuple[str, str, str]:
        """Return ``(status, model, provider)`` without copying any lists.

        Cheap accessor for hot paths (e.g. the concurrency check) that only
        need the derived status and model/provider identity rather than a full
        ``SessionSnapshot``.
        """
        return (
            self._derive_status_locked(),
            self.model_spec.model,
            self.model_spec.provider,
        )

    def status_info(self) -> tuple[str, str, str]:
        """Locked public accessor for ``(status, model, provider)``."""
        with self._lock:
            return self._status_info_locked()

    def _snapshot_locked(self, *, include_events: bool | None = None) -> SessionSnapshot:
        status = self._derive_status_locked()

        return SessionSnapshot(
            agent_id=self.agent_id,
            status=status,
            created_at=self.created_at,
            cwd=str(self.cwd_path),
            provider=self.model_spec.provider,
            model=self.model_spec.model,
            tool_mode=self.tool_mode,
            final_text=self._final_text,
            turn_count=self._turn_count,
            usage=self._usage_snapshot_locked(),
            tool_calls=list(self._tool_calls),
            event_counts=dict(self._event_counts),
            active_listeners=self._active_listeners,
            stderr_tail=list(self._stderr_tail),
            event_tail=list(self._event_tail) if (self.include_events if include_events is None else include_events) else [],
            prompts=list(self._prompts),
            transcript=list(self._transcript),
            error=self._error,
        )

    def _summary_locked(self) -> SessionSummary:
        status = self._derive_status_locked()
        now = time.monotonic()
        return SessionSummary(
            agent_id=self.agent_id,
            status=status,
            created_at=self.created_at,
            state_seconds=round(now - self.state_since_monotonic, 3),
            total_seconds=round(now - self.created_monotonic, 3),
            last_activity_seconds=round(now - self.last_activity_monotonic, 3),
            last_action=self.last_action,
            cwd=str(self.cwd_path),
            model=self.model_spec.alias,
            tool_mode=self.tool_mode,
            turn_count=self._turn_count,
            usage=self._usage_snapshot_locked(now=now),
            final_text_preview=text_preview(self._final_text),
            recent_actions=list(self._recent_actions),
            prompt_count=len(self._prompts),
            initial_request_preview=text_preview(self._prompts[0].text) if self._prompts else "",
            current_tool=self.current_tool,
            error=self._error,
        )

    def _usage_snapshot_locked(self, *, now: float | None = None) -> dict[str, Any]:
        current = now if now is not None else time.monotonic()
        return usage_to_json(self._usage, elapsed_seconds=max(current - self.created_monotonic, 0.0))

    def _effective_behavior(self, behavior: ReplyBehavior, *, is_running: bool) -> str:
        if behavior == "steer":
            return "steer"
        if behavior == "follow-up" or (behavior == "auto" and is_running):
            return "follow-up"
        return "prompt"

    def _record_message_locked(self, event_type: str, message: Any) -> None:
        if not isinstance(message, dict):
            return
        role = message.get("role")
        role_text = role if isinstance(role, str) and role else "assistant"
        if role_text == "user":
            return
        text = extract_message_text(message)
        if not text:
            return
        self._append_transcript_locked(
            TranscriptItem(
                kind="message",
                turn=self._turn_count + 1,
                role=role_text,
                text=text,
                event_type=event_type,
            )
        )

    def _record_usage_from_message_locked(self, message: Any) -> bool:
        if not isinstance(message, dict):
            return False
        stats = usage_stats_from_message(message)
        if stats is None:
            return False
        key = usage_message_key(message, turn=self._turn_count + 1)
        if key:
            if key in self._seen_usage_keys:
                return True
            self._seen_usage_keys.add(key)
        merge_usage(self._usage, stats)
        return True

    def _record_usage_from_event_locked(self, event: dict[str, Any]) -> bool:
        payload = event.get("usage")
        if not isinstance(payload, dict):
            return False
        stats = usage_stats_from_payload(payload)
        key = json.dumps([self._turn_count + 1, payload], ensure_ascii=False, sort_keys=True, default=str)
        key = f"event-usage:{key}"
        if key in self._seen_usage_keys:
            return True
        self._seen_usage_keys.add(key)
        merge_usage(self._usage, stats)
        return True

    def _record_message_update_locked(self, event: dict[str, Any]) -> None:
        text, thinking = extract_message_update_parts(event)
        # Reasoning streams into its own item so the transcript keeps the model's
        # thinking distinct from its spoken answer (and the answer's finalized
        # message doesn't overwrite it).
        if thinking:
            self._append_stream_delta_locked("thinking_stream", thinking)
            self._touch_locked("receiving reasoning")
        if text:
            self._append_stream_delta_locked("message_stream", text)
            self._touch_locked("receiving assistant message")
        if not thinking and not text:
            # A streamed message_update means the model is actively producing
            # output even if we couldn't extract characters from this event's
            # shape (provider schema drift). Count it as activity so the
            # inactivity watchdog never starves mid-stream on a long generation.
            self._touch_locked("streaming")

    def _append_stream_delta_locked(self, kind: str, text: str) -> None:
        turn = self._turn_count + 1
        if self._transcript:
            previous = self._transcript[-1]
            if previous.kind == kind and previous.turn == turn:
                previous.text = (previous.text or "") + text
                return
        self._append_transcript_locked(
            TranscriptItem(
                kind=kind,
                turn=turn,
                role="assistant",
                text=text,
                event_type="message_update",
            )
        )

    def _record_thinking_locked(self, event_type: str, message: Any) -> None:
        thinking = extract_thinking_text(message)
        if not thinking:
            return
        turn = self._turn_count + 1
        finalized = TranscriptItem(
            kind="thinking",
            turn=turn,
            role="assistant",
            text=thinking,
            event_type=event_type,
        )
        # Replace this turn's streamed reasoning if we captured one; otherwise the
        # provider only delivered reasoning in the final message, so append it.
        for index in range(len(self._transcript) - 1, -1, -1):
            item = self._transcript[index]
            if item.turn != turn:
                break
            if item.kind in {"thinking", "thinking_stream"}:
                self._transcript[index] = finalized
                return
        self._append_transcript_locked(finalized)

    def _append_transcript_locked(self, item: TranscriptItem) -> None:
        if self._transcript:
            previous = self._transcript[-1]
            if (
                item.kind == "message"
                and previous.kind == "message_stream"
                and previous.turn == item.turn
                and previous.role == item.role
            ):
                self._transcript[-1] = item
                return
            if (
                previous.kind == item.kind
                and previous.turn == item.turn
                and previous.role == item.role
                and previous.text == item.text
                and previous.tool_call_id == item.tool_call_id
                and previous.tool_name == item.tool_name
                and previous.result_preview == item.result_preview
            ):
                return
        self._transcript.append(item)

    def _record_action_locked(self, action: str) -> None:
        if not self._recent_actions or self._recent_actions[-1] != action:
            self._recent_actions.append(action)

    def _touch_locked(self, action: str) -> None:
        self.last_activity_monotonic = time.monotonic()
        self.last_action = action
        self._record_action_locked(action)

    def _set_status_locked(self, status: str, action: str | None = None) -> None:
        if self._status != status:
            self.state_since_monotonic = time.monotonic()
        self._status = status
        self._running = status == "running"
        self.last_activity_monotonic = time.monotonic()
        if action:
            self.last_action = action
            self._record_action_locked(action)

    def _record_provider_error_locked(self, message: Any) -> None:
        """Surface a provider/runtime error (rate limit, quota, auth, etc.) so it
        reaches every snapshot the caller reads. Without this the message is only
        counted in event_counts and the actual reason is lost."""
        if not isinstance(message, str):
            return
        message = message.strip()
        if not message:
            return
        self._error = message
        self._touch_locked(text_preview(message, limit=160))

    def _clear_error_locked(self) -> None:
        self._error = None

    def _watchdog(self) -> None:
        # One watchdog runs for the whole agent lifetime (across evict/respawn
        # cycles); it returns only when the agent is permanently closed.
        while True:
            time.sleep(0.25)
            action: str | None = None
            with self._lock:
                if self._closed:
                    return
                process = self.process
                alive = process is not None and process.poll() is None
                idle_seconds = time.monotonic() - self.last_activity_monotonic
                # Inactivity, not wall-clock: a running turn that keeps streaming
                # tokens, calling tools, or returning results refreshes
                # last_activity and never trips this. Only a running turn that
                # goes silent for the whole window is treated as stalled.
                if self._running and alive and idle_seconds > self.inactivity_timeout_seconds:
                    action = "stall"
                # An idle (turn finished) worker that has sat quiet past the grace
                # window is evicted: kill the process to reclaim memory while the
                # persisted session keeps it resumable on the next reply.
                elif (
                    alive
                    and not self._running
                    and not self._evicted
                    and self._persist_enabled
                    and self.idle_eviction_seconds > 0
                    and idle_seconds > self.idle_eviction_seconds
                ):
                    action = "evict"

            if action == "stall":
                with self._condition:
                    self._error = f"Pi RPC stalled: no activity for {self.inactivity_timeout_seconds}s"
                    self._set_status_locked("timeout", self._error)
                    self._condition.notify_all()
                self._terminate(mark_status="timeout")
                return
            if action == "evict":
                self._evict()

    def _evict(self) -> None:
        """Kill an idle worker to reclaim memory while keeping the agent resumable.

        The conversation is already on disk (Pi appends each entry synchronously),
        so the next reply respawns a worker that resumes the session by id. The
        agent stays reported as ``idle``; only its process is gone."""
        observer_event: tuple[str, SessionSnapshot] | None = None
        with self._condition:
            process = self.process
            if self._closed or self._evicted or process is None or process.poll() is not None:
                return
            self._evicted = True
            self._running = False
            self.current_tool = None
            self._touch_locked("evicted idle worker (session persisted)")
            observer_event = ("agent_updated", self._snapshot_locked(include_events=False))
            self._condition.notify_all()
        self._kill_process(process)
        if observer_event is not None:
            self._notify_observer(*observer_event)

    def _kill_process(self, process: subprocess.Popen[str] | None) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin:
                try:
                    process.stdin.write(json.dumps({"type": "abort"}) + "\n")
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                process.terminate()
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()
            process.wait(timeout=3)

    def _terminate(self, *, mark_status: str) -> None:
        observer_event: tuple[str, SessionSnapshot] | None = None
        with self._condition:
            if self._closed:
                return
            self._closed = True
            process = self.process
            self._set_status_locked(mark_status, mark_status)
            self.current_tool = None
            observer_type = "agent_stopped" if mark_status == "stopped" else "agent_updated"
            observer_event = (observer_type, self._snapshot_locked(include_events=False))
            self._condition.notify_all()

        self._emit_transcript(
            "status", {"status": mark_status, "error": self._error or ""}, turn=self._turn_count + 1
        )

        self._kill_process(process)
        if observer_event is not None:
            self._notify_observer(*observer_event)

    def _emit_transcript(self, type_name: str, data: dict[str, Any], *, turn: int) -> None:
        sink = self._transcript_sink
        if sink is None:
            return
        timestamp = time.time()
        record = {
            "timestamp": round(timestamp, 6),
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
            "turn": turn,
            "type": type_name,
            "data": data,
        }
        try:
            sink(self.agent_id, record)
        except Exception:
            pass

    def _flush_stream_buffer(self, *, turn: int) -> None:
        if not self._stream_buffer:
            return
        text = self._stream_buffer
        self._stream_buffer = ""
        self._emit_transcript("stream", {"text": text}, turn=turn)

    def _persist_transcript_event(self, event_type: Any, event: dict[str, Any], *, turn: int) -> None:
        if self._transcript_sink is None:
            return
        if event_type == "message_update":
            # Coalesce streamed assistant/reasoning deltas into one record, flushed
            # when the next non-update event (or the worker exit) arrives. This keeps
            # every reasoning trace without writing one line per token.
            text = extract_message_update_text(event)
            if text:
                self._stream_buffer += text
            return
        self._flush_stream_buffer(turn=turn)
        if event_type == "response":
            # Internal prompt-accept handshake, not transcript content.
            return
        self._emit_transcript(str(event_type or "event"), event, turn=turn)

    def _notify_observer(self, event_type: str, snapshot: SessionSnapshot) -> None:
        if self._observer is None:
            return
        try:
            self._observer(event_type, snapshot)
        except Exception:
            pass


class SessionManager:
    def __init__(
        self,
        *,
        parent_id: str | None = None,
        owner_pid: int | None = None,
        observer: SessionSnapshotObserver | None = None,
        transcript_sink: TranscriptSink | None = None,
        session_dir: Path | None = None,
        idle_eviction_seconds: float = DEFAULT_IDLE_EVICTION_SECONDS,
    ) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, PiAgentSession] = {}
        self._runner = PiRpcRunner()
        self.parent_id = parent_id or uuid.uuid4().hex
        self.owner_pid = owner_pid
        self._observer = observer
        self._transcript_sink = transcript_sink
        self._session_dir = session_dir
        self._idle_eviction_seconds = idle_eviction_seconds

    def start(
        self,
        *,
        prompt: str,
        cwd: str | None,
        model: str | None,
        provider: str | None,
        tool_mode: ToolMode,
        include_events: bool,
        unsafe_read_only: bool = False,
    ) -> SessionSnapshot:
        agent_id = uuid.uuid4().hex[:12]
        session = PiAgentSession(
            agent_id=agent_id,
            runner=self._runner,
            prompt=prompt,
            cwd=cwd,
            model=model,
            provider=provider,
            tool_mode=tool_mode,
            include_events=include_events,
            observer=self._observer,
            transcript_sink=self._transcript_sink,
            unsafe_read_only=unsafe_read_only,
            session_dir=self._session_dir,
            idle_eviction_seconds=self._idle_eviction_seconds,
        )
        with self._lock:
            self._sessions[agent_id] = session
        return session.snapshot()

    def _cleanup_session_files(self, agent_id: str) -> None:
        """Delete the persisted session log(s) for a permanently-removed agent.

        Pi names files ``<timestamp>_<id>.jsonl`` under per-cwd subdirs, so match
        the agent_id anywhere in the tree. Best-effort: the session store is now
        durable (``~/.pi-as-mcp/sessions``), so a leftover file persists until a
        later cleanup rather than being reclaimed on reboot — it only costs disk."""
        if self._session_dir is None:
            return
        try:
            for path in self._session_dir.rglob(f"*{agent_id}*.jsonl"):
                try:
                    path.unlink()
                except OSError:
                    pass
        except OSError:
            pass

    def reply(
        self,
        agent_id: str,
        *,
        prompt: str,
        behavior: ReplyBehavior,
    ) -> SessionSnapshot:
        session = self._get(agent_id)
        session.send(prompt, behavior=behavior)
        return session.snapshot()

    def peek(self, agent_id: str, *, include_events: bool = False) -> SessionSnapshot:
        return self._get(agent_id).snapshot(include_events=include_events)

    def listen(
        self,
        agent_id: str,
        *,
        after_turn_count: int,
        timeout_seconds: int,
        include_events: bool = False,
    ) -> tuple[SessionSnapshot, bool]:
        return self._get(agent_id).listen(
            after_turn_count=after_turn_count,
            timeout_seconds=timeout_seconds,
            include_events=include_events,
        )

    def stop(self, agent_id: str) -> SessionSnapshot:
        session = self._get(agent_id)
        snapshot = session.stop()
        with self._lock:
            self._sessions.pop(agent_id, None)
        self._cleanup_session_files(agent_id)
        return snapshot

    def list(self) -> list[dict[str, Any]]:
        return self.summary()

    def summary(self) -> list[dict[str, Any]]:
        with self._lock:
            sessions = list(self._sessions.values())
        return [session.summary().to_json() for session in sessions]

    def active_model_count(self, *, provider: str, model: str, match_provider: bool) -> int:
        with self._lock:
            sessions = list(self._sessions.values())

        count = 0
        for session in sessions:
            session_status, session_model, session_provider = session.status_info()
            if session_status not in CONCURRENCY_COUNTED_STATUSES:
                continue
            if session_model != model:
                continue
            if match_provider and session_provider != provider:
                continue
            count += 1
        return count

    def has(self, agent_id: str) -> bool:
        with self._lock:
            return agent_id in self._sessions

    def close(self, *, reason: str = "parent closed") -> int:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session._terminate(mark_status=reason)
            self._cleanup_session_files(session.agent_id)
        return len(sessions)

    def _get(self, agent_id: str) -> PiAgentSession:
        with self._lock:
            session = self._sessions.get(agent_id)
        if session is None:
            raise PiRpcError(f"unknown agent_id: {agent_id}")
        return session
