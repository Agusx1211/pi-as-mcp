from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from pi_as_mcp.daemon_client import DaemonClient, DaemonClientError


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def add_common_agent_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cwd")
    parser.add_argument("--model")
    parser.add_argument("--tool-mode", default="read-only", choices=["none", "read-only", "write", "full"])
    parser.add_argument("--verbosity", default="summary", choices=["summary", "normal", "debug"])


def compact_wait_response(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "agent_id": data.get("agent_id"),
        "status": data.get("status"),
        "turn_count": data.get("turn_count", 0),
        "timed_out": bool(data.get("listen_timed_out", False)),
        "final_text": data.get("final_text") or "",
    }
    if "tool_call_count" in data:
        result["tool_call_count"] = data["tool_call_count"]
    if data.get("score_hint"):
        result["score_hint"] = data["score_hint"]
    if data.get("error"):
        result["error"] = data["error"]
    return result


def wait_for_agent(
    client: DaemonClient,
    *,
    agent_id: str,
    after_turn_count: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    started = time.monotonic()
    last: dict[str, Any] | None = None

    while True:
        listen_timeout = 900
        if timeout_seconds > 0:
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                if last is not None:
                    last["listen_timed_out"] = True
                    return last
                listen_timeout = 1
            else:
                listen_timeout = max(1, min(900, int(remaining)))

        data = client.request(
            "listen",
            request_timeout_seconds=listen_timeout + 5,
            agent_id=agent_id,
            after_turn_count=after_turn_count,
            timeout_seconds=listen_timeout,
            include_events=False,
            verbosity="summary",
        )
        last = data
        if not data.get("listen_timed_out"):
            return data


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="pi-agent", description="Manage local Pi subagents")
    sub = root.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="start a background Pi agent")
    start.add_argument("prompt")
    add_common_agent_args(start)

    delegate = sub.add_parser("delegate", help="start a background Pi agent")
    delegate.add_argument("prompt")
    add_common_agent_args(delegate)

    reply = sub.add_parser("reply", help="send another prompt to an agent")
    reply.add_argument("agent_id")
    reply.add_argument("prompt")
    reply.add_argument("--behavior", default="auto", choices=["auto", "follow-up", "steer"])
    reply.add_argument("--verbosity", default="summary", choices=["summary", "normal", "debug"])

    peek = sub.add_parser("peek", help="show one agent snapshot")
    peek.add_argument("agent_id")
    peek.add_argument("--verbosity", default="summary", choices=["summary", "normal", "debug"])

    listen = sub.add_parser("listen", help="wait until an agent completes another turn")
    listen.add_argument("agent_id")
    listen.add_argument("--after-turn", "--after-turn-count", dest="after_turn_count", type=int, default=0)
    listen.add_argument("--timeout-seconds", type=int, default=60)
    listen.add_argument("--verbosity", default="summary", choices=["summary", "normal", "debug"])

    wait = sub.add_parser("wait", aliases=["w"], help="wait quietly until an agent completes a turn")
    wait.add_argument("agent_id")
    wait.add_argument("--after-turn", "--after-turn-count", "-a", dest="after_turn_count", type=int, default=0)
    wait.add_argument(
        "--timeout-seconds",
        "-t",
        type=int,
        default=0,
        help="overall wait timeout; 0 waits until the agent completes",
    )

    stop = sub.add_parser("stop", help="stop an agent")
    stop.add_argument("agent_id")
    stop.add_argument("--verbosity", default="summary", choices=["summary", "normal", "debug"])

    summary = sub.add_parser("summary", help="summarize all live agents")
    summary.add_argument("--scoped", action="store_true", help="show only agents in this CLI parent scope")
    listing = sub.add_parser("list", help="alias for summary")
    listing.add_argument("--scoped", action="store_true", help="show only agents in this CLI parent scope")
    sub.add_parser("models", help="list Pi-enabled models")
    sub.add_parser("tui", help="open the interactive local agent TUI")

    health = sub.add_parser("health", help="check Pi backend")
    health.add_argument("--model")
    health.add_argument("--timeout-seconds", type=int, default=15)

    return root


def main() -> None:
    args = parser().parse_args()
    client = DaemonClient()
    try:
        command = args.command
        if command in {"start", "delegate"}:
            print_json(
                client.request(
                    command,
                    request_timeout_seconds=10,
                    prompt=args.prompt,
                    cwd=args.cwd,
                    model=args.model,
                    tool_mode=args.tool_mode,
                    include_events=args.verbosity == "debug",
                    verbosity=args.verbosity,
                )
            )
        elif command == "reply":
            print_json(
                client.request(
                    "reply",
                    request_timeout_seconds=15,
                    agent_id=args.agent_id,
                    prompt=args.prompt,
                    behavior=args.behavior,
                    verbosity=args.verbosity,
                )
            )
        elif command == "peek":
            print_json(
                client.request(
                    "peek",
                    agent_id=args.agent_id,
                    include_events=args.verbosity == "debug",
                    verbosity=args.verbosity,
                )
            )
        elif command == "listen":
            print_json(
                client.request(
                    "listen",
                    request_timeout_seconds=args.timeout_seconds + 5,
                    agent_id=args.agent_id,
                    after_turn_count=args.after_turn_count,
                    timeout_seconds=args.timeout_seconds,
                    include_events=args.verbosity == "debug",
                    verbosity=args.verbosity,
                )
            )
        elif command in {"wait", "w"}:
            print_json(
                compact_wait_response(
                    wait_for_agent(
                        client,
                        agent_id=args.agent_id,
                        after_turn_count=args.after_turn_count,
                        timeout_seconds=args.timeout_seconds,
                    )
                )
            )
        elif command == "stop":
            print_json(client.request("stop", agent_id=args.agent_id, verbosity=args.verbosity))
        elif command in {"summary", "list"}:
            print_json(client.request("summary" if args.scoped else "tui_summary"))
        elif command == "models":
            print_json(client.request("models"))
        elif command == "tui":
            from pi_as_mcp.tui import run_tui

            run_tui()
        elif command == "health":
            print_json(
                client.request(
                    "health",
                    request_timeout_seconds=args.timeout_seconds + 5,
                    model=args.model,
                    timeout_seconds=args.timeout_seconds,
                )
            )
    except DaemonClientError as exc:
        print_json({"error": str(exc)})
        sys.exit(1)


def wait_main() -> None:
    wait_parser = argparse.ArgumentParser(prog="piw", description="Wait quietly for a Pi subagent")
    wait_parser.add_argument("agent_id")
    wait_parser.add_argument("--after-turn", "--after-turn-count", "-a", dest="after_turn_count", type=int, default=0)
    wait_parser.add_argument(
        "--timeout-seconds",
        "-t",
        type=int,
        default=0,
        help="overall wait timeout; 0 waits until the agent completes",
    )
    args = wait_parser.parse_args()
    client = DaemonClient()
    try:
        print_json(
            compact_wait_response(
                wait_for_agent(
                    client,
                    agent_id=args.agent_id,
                    after_turn_count=args.after_turn_count,
                    timeout_seconds=args.timeout_seconds,
                )
            )
        )
    except DaemonClientError as exc:
        print_json({"error": str(exc)})
        sys.exit(1)


if __name__ == "__main__":
    main()
