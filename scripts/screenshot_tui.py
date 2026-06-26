#!/usr/bin/env python3
"""Generate a TUI screenshot with mock data for the README."""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from textual.app import App

from pi_as_mcp.tui import PiAgentTui

MOCK_NOW = time.time()
MOCK_CREATED = MOCK_NOW - 127.4

MOCK_REQUESTER_CLAUDE = {
    "scope_id": "scope-claude-7f3a2b",
    "label": "hint:mcp:7f3a2b",
    "kind": "Claude",
    "instance": "mcp:7f3a2b",
    "display": "Claude mcp:7f3a2b",
    "owner_pid": 48210,
    "peer_pid": 48210,
    "peer_process": {
        "pid": 48210,
        "ppid": 48190,
        "start_time": "2180923",
        "command": "node /home/user/.claude/local/claude",
    },
    "owner_process": {
        "pid": 48210,
        "ppid": 48190,
        "start_time": "2180923",
        "command": "node /home/user/.claude/local/claude",
    },
    "lineage": [
        {
            "pid": 48210,
            "ppid": 48190,
            "start_time": "2180923",
            "command": "node /home/user/.claude/local/claude",
        },
        {
            "pid": 48190,
            "ppid": 48100,
            "start_time": "2180900",
            "command": "/usr/bin/zsh",
        },
    ],
}

MOCK_REQUESTER_CODEX = {
    "scope_id": "scope-codex-e1d4f2",
    "label": "hint:mcp:e1d4f2",
    "kind": "Codex",
    "instance": "mcp:e1d4f2",
    "display": "Codex mcp:e1d4f2",
    "owner_pid": 49001,
    "peer_pid": 49001,
    "peer_process": {
        "pid": 49001,
        "ppid": 48900,
        "start_time": "2181100",
        "command": "node /home/user/.codex/codex",
    },
    "owner_process": None,
    "lineage": [
        {
            "pid": 49001,
            "ppid": 48900,
            "start_time": "2181100",
            "command": "node /home/user/.codex/codex",
        },
    ],
}

MOCK_AGENTS: list[dict[str, Any]] = [
    {
        "agent_id": "a8f2e1c3b4d6",
        "status": "idle",
        "created_at": MOCK_CREATED,
        "state_seconds": 12.5,
        "total_seconds": 127.4,
        "last_activity_seconds": 12.5,
        "last_action": "idle after turn 2",
        "cwd": "/home/user/acme-project",
        "model": "devstral-small-2",
        "tool_mode": "read-only",
        "turn_count": 2,
        "usage": {
            "input_tokens": 15234,
            "output_tokens": 2847,
            "cache_read_tokens": 8192,
            "cache_write_tokens": 4096,
            "total_tokens": 30369,
            "context_percent": 12.3,
            "tokens_per_second": 42.5,
        },
        "final_text_preview": "The error handling in `src/api/client.ts` has three areas for improvement...",
        "recent_actions": [
            "starting Pi worker",
            "read src/api/client.ts",
            "grep 'catch|throw|Error'",
            "received assistant message",
            "idle after turn 2",
        ],
        "prompt_count": 1,
        "initial_request_preview": "Analyze the error handling in src/api/client.ts and suggest improvements",
        "parent_scope_id": "scope-claude-7f3a2b",
        "requester": MOCK_REQUESTER_CLAUDE,
    },
    {
        "agent_id": "c3d4e5f6a7b8",
        "status": "running",
        "created_at": MOCK_CREATED + 30,
        "state_seconds": 8.2,
        "total_seconds": 97.4,
        "last_activity_seconds": 0.3,
        "last_action": "receiving assistant message",
        "cwd": "/home/user/acme-project",
        "model": "qwen3.6-35b-a3b",
        "tool_mode": "read-only",
        "turn_count": 0,
        "usage": {
            "input_tokens": 4210,
            "output_tokens": 891,
            "cache_read_tokens": 0,
            "cache_write_tokens": 2048,
            "total_tokens": 7149,
            "context_percent": 3.1,
            "tokens_per_second": 68.2,
        },
        "final_text_preview": "",
        "recent_actions": ["starting Pi worker", "receiving assistant message"],
        "prompt_count": 1,
        "initial_request_preview": "Check if pyproject.toml dependencies are up to date",
        "current_tool": None,
        "parent_scope_id": "scope-claude-7f3a2b",
        "requester": MOCK_REQUESTER_CLAUDE,
    },
    {
        "agent_id": "e5f6a7b8c3d4",
        "status": "idle",
        "created_at": MOCK_CREATED + 60,
        "state_seconds": 34.1,
        "total_seconds": 67.4,
        "last_activity_seconds": 34.1,
        "last_action": "idle after turn 1",
        "cwd": "/home/user/acme-project",
        "model": "mistral-small-4",
        "tool_mode": "write",
        "turn_count": 1,
        "usage": {
            "input_tokens": 8190,
            "output_tokens": 1523,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 9713,
            "context_percent": 6.7,
            "tokens_per_second": 55.0,
        },
        "final_text_preview": "Fixed the off-by-one error in the pagination logic.",
        "recent_actions": ["starting Pi worker", "edit src/pagination.ts", "idle after turn 1"],
        "prompt_count": 1,
        "initial_request_preview": "Fix the off-by-one error in src/pagination.ts line 42",
        "parent_scope_id": "scope-claude-7f3a2b",
        "requester": MOCK_REQUESTER_CLAUDE,
    },
    {
        "agent_id": "b1c2d3e4f5a6",
        "status": "running",
        "created_at": MOCK_CREATED + 15,
        "state_seconds": 5.8,
        "total_seconds": 112.4,
        "last_activity_seconds": 0.1,
        "last_action": "bash 'npm test -- --reporter=json'",
        "cwd": "/home/user/acme-project",
        "model": "devstral-2",
        "tool_mode": "full",
        "turn_count": 1,
        "usage": {
            "input_tokens": 22100,
            "output_tokens": 4310,
            "cache_read_tokens": 12000,
            "cache_write_tokens": 6000,
            "total_tokens": 44410,
            "context_percent": 18.5,
            "tokens_per_second": 38.1,
        },
        "final_text_preview": "",
        "recent_actions": [
            "starting Pi worker",
            "read package.json",
            "bash 'npm test -- --reporter=json'",
        ],
        "prompt_count": 2,
        "initial_request_preview": "Run the test suite and report any failures with suggested fixes",
        "current_tool": "bash",
        "parent_scope_id": "scope-codex-e1d4f2",
        "requester": MOCK_REQUESTER_CODEX,
    },
]

MOCK_STATS: dict[str, Any] = {
    "total_agents": 12,
    "observed_agents": 9,
    "unobserved_agents": 3,
    "scores": 6,
    "average_score": 7.33,
}

MOCK_DETAIL: dict[str, Any] = {
    "agent_id": "a8f2e1c3b4d6",
    "status": "idle",
    "created_at": MOCK_CREATED,
    "cwd": "/home/user/acme-project",
    "provider": "mistralai",
    "model": "devstral-small-2",
    "tool_mode": "read-only",
    "final_text": (
        "The error handling in `src/api/client.ts` has three areas for improvement:\n\n"
        "1. **Inconsistent catch blocks** — `fetchUser` (line 23) swallows errors silently "
        "with an empty `catch {}`, while `fetchOrders` (line 58) logs but doesn't rethrow. "
        "Both should propagate typed errors to callers.\n\n"
        "2. **Missing timeout handling** — None of the fetch calls set `AbortController` "
        "timeouts, so a hung upstream silently blocks the caller forever.\n\n"
        "3. **No retry logic** — Transient 5xx responses immediately surface as fatal errors. "
        "Adding exponential backoff for `fetchOrders` and `fetchInventory` would improve "
        "resilience without complicating the happy path."
    ),
    "turn_count": 2,
    "usage": {
        "input_tokens": 15234,
        "output_tokens": 2847,
        "cache_read_tokens": 8192,
        "cache_write_tokens": 4096,
        "total_tokens": 30369,
        "context_percent": 12.3,
        "tokens_per_second": 42.5,
        "context_used_tokens": None,
        "context_limit_tokens": None,
    },
    "event_counts": {
        "agent_start": 1,
        "message_update": 47,
        "message_end": 2,
        "tool_execution_start": 2,
        "tool_execution_end": 2,
        "agent_end": 2,
        "turn_end": 2,
    },
    "tool_call_count": 2,
    "active_listeners": 1,
    "observing_with_piw": True,
    "requester": MOCK_REQUESTER_CLAUDE,
    "stats": {
        "agent_id": "a8f2e1c3b4d6",
        "observed_by_parent": True,
        "observed_via": "listen",
        "observation_count": 1,
        "latest_score": {
            "score": 8,
            "category": "code-review",
            "comment": "Thorough analysis with actionable suggestions",
            "sentiment": "net-positive",
        },
        "scores": [
            {
                "score": 8,
                "category": "code-review",
                "comment": "Thorough analysis with actionable suggestions",
                "sentiment": "net-positive",
            },
        ],
    },
    "prompts": [
        {
            "turn": 1,
            "text": "Analyze the error handling in src/api/client.ts and suggest improvements",
            "behavior": "prompt",
            "accepted": True,
        },
    ],
    "transcript": [
        {
            "kind": "prompt",
            "turn": 1,
            "role": "user",
            "text": "Analyze the error handling in src/api/client.ts and suggest improvements",
            "behavior": "prompt",
        },
        {
            "kind": "thinking",
            "turn": 1,
            "role": "assistant",
            "text": (
                "I need to analyze the error handling patterns in this file. Let me start by "
                "reading the file to understand the current implementation, then look for common "
                "error handling anti-patterns like swallowed exceptions, missing error types, and "
                "inconsistent error propagation."
            ),
        },
        {
            "kind": "tool_call",
            "turn": 1,
            "tool_name": "read",
            "tool_args": {"path": "src/api/client.ts"},
        },
        {
            "kind": "tool_result",
            "turn": 1,
            "tool_name": "read",
            "is_error": False,
            "result_preview": (
                "import { ApiError, NetworkError } from './errors';\n"
                "import type { User, Order } from './types';\n\n"
                "const BASE_URL = process.env.API_URL ?? 'https://api.acme.dev';\n\n"
                "export async function fetchUser(id: string): Promise<User | null> {\n"
                "  try {\n"
                "    const res = await fetch(`${BASE_URL}/users/${id}`);\n"
                "    if (!res.ok) throw new ApiError(res.status, await res.text());\n"
                "    return await res.json();\n"
                "  } catch {\n"
                "    return null;\n"
                "  }\n"
                "}\n\n"
                "export async function fetchOrders(userId: string): Promise<Order[]> {\n"
                "  try {\n"
                "    const res = await fetch(`${BASE_URL}/orders?user=${userId}`);\n..."
            ),
        },
        {
            "kind": "tool_call",
            "turn": 1,
            "tool_name": "grep",
            "tool_args": {"pattern": "catch|throw|Error", "path": "src/api/"},
        },
        {
            "kind": "tool_result",
            "turn": 1,
            "tool_name": "grep",
            "is_error": False,
            "result_preview": (
                "src/api/client.ts:1: import { ApiError, NetworkError } from './errors';\n"
                "src/api/client.ts:9:     if (!res.ok) throw new ApiError(res.status, await res.text());\n"
                "src/api/client.ts:11:   } catch {\n"
                "src/api/client.ts:22:     if (!res.ok) throw new ApiError(res.status, await res.text());\n"
                "src/api/client.ts:25:   } catch (err) {\n"
                "src/api/client.ts:26:     console.error('fetchOrders failed:', err);\n"
                "src/api/errors.ts:1: export class ApiError extends Error {\n"
                "src/api/errors.ts:8: export class NetworkError extends Error {"
            ),
        },
        {
            "kind": "message",
            "turn": 1,
            "role": "assistant",
            "text": (
                "The error handling in `src/api/client.ts` has three areas for improvement:\n\n"
                "1. **Inconsistent catch blocks** — `fetchUser` (line 23) swallows errors silently "
                "with an empty `catch {}`, while `fetchOrders` (line 58) logs but doesn't rethrow. "
                "Both should propagate typed errors to callers.\n\n"
                "2. **Missing timeout handling** — None of the fetch calls set `AbortController` "
                "timeouts, so a hung upstream silently blocks the caller forever.\n\n"
                "3. **No retry logic** — Transient 5xx responses immediately surface as fatal errors. "
                "Adding exponential backoff for `fetchOrders` and `fetchInventory` would improve "
                "resilience without complicating the happy path."
            ),
        },
    ],
}


class MockPiAgentTui(PiAgentTui):
    """PiAgentTui with mock data, no daemon connection."""

    def __init__(self) -> None:
        # Bypass PiAgentTui.__init__ to avoid DaemonClient import issues.
        App.__init__(self)
        self.agents: list[dict[str, Any]] = list(MOCK_AGENTS)
        self.stats_summary: dict[str, Any] = dict(MOCK_STATS)
        self.selected_agent_id: str | None = "a8f2e1c3b4d6"
        self.detail: dict[str, Any] | None = dict(MOCK_DETAIL)
        self.error: str | None = None
        self.refresh_failed = False
        self.collapsed_parent_ids: set[str] = set()
        self.sidebar_cursor: tuple[str, str] | None = None
        self.status_frame = 0
        self._tree_structure_signature: tuple[tuple[str, tuple[str, ...]], ...] | None = None
        self._last_log_signature: str | None = None

    def on_mount(self) -> None:
        tree = self.agent_tree
        tree.show_root = False
        tree.auto_expand = False
        tree.show_horizontal_scrollbar = False
        tree.focus()
        self.reconcile_selection()
        self.render_all()

    def refresh_data(self) -> None:
        pass

    def load_detail(self) -> None:
        pass


async def main() -> None:
    app = MockPiAgentTui()
    async with app.run_test(size=(140, 42)) as pilot:
        # Let the app settle and render
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()
        svg = app.export_screenshot()
        output_dir = Path(__file__).resolve().parent.parent / "assets"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "tui.svg"
        output_path.write_text(svg, encoding="utf-8")
        print(f"Screenshot saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
