from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import uuid
from typing import Any, NotRequired, TypedDict, cast

from mcp.server.fastmcp import FastMCP

from pi_as_mcp.config import load_config
from pi_as_mcp.pi_rpc import PiRpcError, ToolMode
from pi_as_mcp.daemon_client import DaemonClient
from pi_as_mcp.paths import runtime_dir
from pi_as_mcp.sessions import ReplyBehavior, ResponseVerbosity
from pi_as_mcp import skill

mcp = FastMCP("pi-as-mcp")
client = DaemonClient(default_parent_hint=f"mcp:{uuid.uuid4().hex}", parent_owner_pid=os.getpid())


class MonitorResult(TypedDict):
    agent_id: str
    status: str
    turn_count: int
    monitor_command: str
    monitor_after_turn_count: int
    queued_turn_expected: bool
    monitor_hint: str


class SiblingAgent(TypedDict):
    agent_id: str
    status: str
    model: str
    summary: str


class DelegateResult(MonitorResult):
    other_agents: list[SiblingAgent]
    other_agents_hint: str


class ModelAlias(TypedDict):
    alias: str
    provider: str
    model: str
    description: NotRequired[str]


class ModelsResult(TypedDict):
    models: list[ModelAlias]


class ScoreResult(TypedDict):
    agent_id: str
    score: int
    category: str
    comment: str
    sentiment: str
    recorded: bool


class ToolCallResult(TypedDict):
    id: str | None
    name: str
    args: dict[str, Any]
    is_error: bool | None
    result_preview: str | None


class SessionSnapshotBase(TypedDict):
    agent_id: str
    status: str
    cwd: str
    provider: str
    model: str
    tool_mode: str
    final_text: str
    turn_count: int
    event_counts: dict[str, int]
    tool_call_count: int


class SessionSnapshotResult(SessionSnapshotBase):
    tool_calls: list[ToolCallResult]
    stderr_tail: list[str]
    event_tail: list[dict[str, Any]]
    error: str


def ensure_wait_shim() -> str:
    path = runtime_dir() / "piw"
    body = f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -m pi_as_mcp.cli wait \"$@\"\n"
    if not path.exists() or path.read_text(encoding="utf-8") != body:
        # Write atomically with the mode already set, so a crash can never
        # leave a correct-looking but non-executable shim that the content
        # check above would forever consider up to date.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        tmp.chmod(0o700)
        tmp.replace(path)
    else:
        # Heal shims left non-executable by older versions.
        path.chmod(0o700)
    return str(path)


def wait_command(agent_id: str, *, after_turn_count: int = 0) -> str:
    parts = ["piw" if shutil.which("piw") else ensure_wait_shim(), agent_id]
    if after_turn_count > 0:
        parts.extend(["-a", str(after_turn_count)])
    return " ".join(shlex.quote(part) for part in parts)


def monitor_result(
    data: dict[str, Any],
    *,
    after_turn_count: int,
    queued_turn_expected: bool,
    monitor_hint: str,
) -> MonitorResult:
    return {
        "agent_id": str(data["agent_id"]),
        "status": str(data["status"]),
        "turn_count": int(data.get("turn_count", 0)),
        "monitor_command": wait_command(str(data["agent_id"]), after_turn_count=after_turn_count),
        "monitor_after_turn_count": after_turn_count,
        "queued_turn_expected": queued_turn_expected,
        "monitor_hint": monitor_hint,
    }


def snapshot_result(data: dict[str, Any]) -> SessionSnapshotResult:
    tool_calls = data.get("tool_calls")
    stderr_tail = data.get("stderr_tail")
    event_tail = data.get("event_tail")
    return {
        "agent_id": str(data["agent_id"]),
        "status": str(data["status"]),
        "cwd": str(data["cwd"]),
        "provider": str(data["provider"]),
        "model": str(data["model"]),
        "tool_mode": str(data["tool_mode"]),
        "final_text": str(data.get("final_text") or ""),
        "turn_count": int(data.get("turn_count", 0)),
        "event_counts": cast(dict[str, int], data.get("event_counts") or {}),
        "tool_call_count": int(data.get("tool_call_count", 0)),
        "tool_calls": cast(list[ToolCallResult], tool_calls if isinstance(tool_calls, list) else []),
        "stderr_tail": cast(list[str], stderr_tail if isinstance(stderr_tail, list) else []),
        "event_tail": cast(list[dict[str, Any]], event_tail if isinstance(event_tail, list) else []),
        "error": str(data.get("error") or ""),
    }


_score_tool_registered = False


def sync_score_tool() -> None:
    global _score_tool_registered
    enabled = load_config().agents.enable_score
    if enabled and not _score_tool_registered:
        mcp.add_tool(
            score,
            name="score",
            description=(
                "Rate a completed Pi subagent from 1 to 10. "
                ">5 is net-positive, <5 is net-negative."
            ),
        )
        _score_tool_registered = True
        return
    if not enabled and _score_tool_registered:
        mcp.remove_tool("score")
        _score_tool_registered = False


def score(agent_id: str, score: int, category: str, comment: str) -> ScoreResult:
    """Record a parent rating for a completed Pi subagent."""

    return cast(
        ScoreResult,
        client.request(
            "score",
            request_timeout_seconds=10,
            agent_id=agent_id,
            score=score,
            category=category,
            comment=comment,
        ),
    )


def other_agents_hint(siblings: list[SiblingAgent]) -> str:
    if not siblings:
        return ""
    stale = [s for s in siblings if s["status"] not in ("starting", "running")]
    if stale:
        return (
            f"You now have {len(siblings)} other Pi subagent(s); "
            f"{len(stale)} are idle/finished. Consider agent_stop to free them, "
            "or agent_reply to reuse one instead of spawning more."
        )
    return f"You have {len(siblings)} other Pi subagent(s) still active."


@mcp.tool()
def delegate(
    prompt: str,
    cwd: str | None = None,
    model: str | None = None,
    tool_mode: ToolMode = "read-only",
) -> DelegateResult:
    """Start a Pi subagent and return a short quiet monitor command for a background shell."""

    data = client.request(
        "delegate",
        request_timeout_seconds=10,
        prompt=prompt,
        cwd=cwd,
        model=model,
        tool_mode=tool_mode,
        include_events=False,
        verbosity="summary",
    )
    base = monitor_result(
        data,
        after_turn_count=0,
        queued_turn_expected=True,
        monitor_hint="waits for the first completed turn",
    )
    raw_siblings = data.get("other_agents")
    siblings = cast(
        list[SiblingAgent], raw_siblings if isinstance(raw_siblings, list) else []
    )
    return {
        **base,
        "other_agents": siblings,
        "other_agents_hint": other_agents_hint(siblings),
    }


@mcp.tool()
def agent_reply(
    agent_id: str,
    prompt: str,
    behavior: ReplyBehavior = "auto",
) -> MonitorResult:
    """Send another prompt and return a short quiet monitor command for a background shell."""

    data = client.request(
        "reply",
        request_timeout_seconds=15,
        agent_id=agent_id,
        prompt=prompt,
        behavior=behavior,
        verbosity="summary",
    )
    # The daemon captures the pre-prompt state atomically under the session
    # lock; a separate peek beforehand could race a completing turn and hand
    # back a monitor command that is satisfied by the *previous* turn.
    after_turn_count = int(data.get("reply_after_turn_count", data.get("turn_count", 0)))
    was_running = bool(data.get("reply_was_running", data.get("status") == "running"))
    if was_running:
        hint = "message was sent to the running turn; waits for that turn to complete"
    else:
        hint = "waits for the reply turn to complete"
    return monitor_result(
        data,
        after_turn_count=after_turn_count,
        queued_turn_expected=not was_running,
        monitor_hint=hint,
    )


@mcp.tool()
def agent_peek(agent_id: str, verbosity: ResponseVerbosity = "summary") -> SessionSnapshotResult:
    """Peek at a Pi subagent without waiting."""

    return snapshot_result(
        client.request(
            "peek",
            agent_id=agent_id,
            include_events=verbosity == "debug",
            verbosity=verbosity,
        ),
    )


@mcp.tool()
def agent_stop(agent_id: str, verbosity: ResponseVerbosity = "summary") -> SessionSnapshotResult:
    """Abort and remove a Pi subagent."""

    return snapshot_result(client.request("stop", agent_id=agent_id, verbosity=verbosity))


@mcp.tool()
def models() -> ModelsResult:
    """List sub-agent models pi-as-mcp exposes (Pi-enabled minus disabled)."""

    return cast(ModelsResult, client.request("models"))


@mcp.resource(
    skill.SKILL_RESOURCE_URI,
    name="cheap-subagents",
    title="Cheap sub-agents (pi-as-mcp)",
    description="How and when to delegate bounded tasks to cheaper Pi sub-agent models.",
    mime_type="text/markdown",
)
def cheap_subagents_skill() -> str:
    """Live-rendered skill: intro + auto-generated model roster + usage."""

    return skill.render_skill_body()


def sync_server_instructions() -> None:
    """Publish the generated skill as the MCP server's instructions.

    Read live at startup so the always-in-context instructions reflect the
    current config and Pi roster. Best-effort: a failure must not stop the
    server from serving its tools.
    """
    try:
        mcp._mcp_server.instructions = skill.render_server_instructions()
    except (PiRpcError, OSError, subprocess.SubprocessError):
        pass


def main() -> None:
    sync_score_tool()
    sync_server_instructions()
    mcp.run()


if __name__ == "__main__":
    main()
