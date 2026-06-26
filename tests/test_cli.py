from __future__ import annotations

import sys

from pi_as_mcp import cli
from pi_as_mcp.cli import compact_wait_response


def test_compact_wait_response_keeps_only_monitor_output_fields() -> None:
    assert compact_wait_response(
        {
            "agent_id": "abc123",
            "status": "idle",
            "turn_count": 1,
            "listen_timed_out": False,
            "final_text": "done",
            "tool_call_count": 2,
            "score_hint": "rate me",
            "cwd": "/tmp/noise",
            "provider": "local",
            "event_counts": {"agent_start": 1},
        }
    ) == {
        "agent_id": "abc123",
        "status": "idle",
        "turn_count": 1,
        "timed_out": False,
        "final_text": "done",
        "tool_call_count": 2,
        "score_hint": "rate me",
    }


def test_summary_uses_global_agent_view_by_default(monkeypatch, capsys) -> None:
    calls: list[str] = []

    class FakeClient:
        def request(self, command: str, **kwargs):
            calls.append(command)
            return {"agents": []}

    monkeypatch.setattr(cli, "DaemonClient", FakeClient)
    monkeypatch.setattr(sys, "argv", ["pi-agent", "summary"])

    cli.main()

    assert calls == ["tui_summary"]
    assert '"agents": []' in capsys.readouterr().out


def test_summary_can_use_scoped_agent_view(monkeypatch) -> None:
    calls: list[str] = []

    class FakeClient:
        def request(self, command: str, **kwargs):
            calls.append(command)
            return {"agents": []}

    monkeypatch.setattr(cli, "DaemonClient", FakeClient)
    monkeypatch.setattr(sys, "argv", ["pi-agent", "summary", "--scoped"])

    cli.main()

    assert calls == ["summary"]
