from __future__ import annotations

import argparse
import json

from pi_as_mcp.sessions import SessionManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct async Pi RPC smoke test")
    parser.add_argument("prompt")
    parser.add_argument("--cwd")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--tool-mode", default="none")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--verbosity", default="summary", choices=["summary", "normal", "debug"])
    args = parser.parse_args()

    manager = SessionManager(parent_id="smoke")
    started = manager.start(
        prompt=args.prompt,
        cwd=args.cwd,
        provider=args.provider,
        model=args.model,
        tool_mode=args.tool_mode,
        include_events=args.verbosity == "debug",
    )
    try:
        result, timed_out = manager.listen(
            started.agent_id,
            after_turn_count=0,
            timeout_seconds=args.timeout_seconds,
            include_events=args.verbosity == "debug",
        )
        data = result.to_json(verbosity=args.verbosity)
        data["listen_timed_out"] = timed_out
        print(json.dumps(data, indent=2, ensure_ascii=False))
    finally:
        manager.stop(started.agent_id)


if __name__ == "__main__":
    main()
