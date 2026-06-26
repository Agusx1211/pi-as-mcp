from __future__ import annotations

import anyio

from pi_as_mcp.daemon_client import DaemonClientError
from pi_as_mcp.tui import (
    PiAgentTui,
    grouped_agents,
    model_label,
    observed_label,
    requester_display,
    score_label,
    short_text,
    stats_summary_label,
    status_icon,
    usage_stats_label,
    watching_label,
)


def test_requester_display_prefers_display_value() -> None:
    assert requester_display({"display": "Codex mcp:abc", "kind": "MCP"}) == "Codex mcp:abc"
    assert requester_display({"kind": "Claude", "instance": "mcp:def"}) == "Claude mcp:def"
    assert requester_display(None) == ""


def test_short_text_compacts_and_truncates() -> None:
    assert short_text("  hello\nworld  ") == "hello world"
    assert short_text("abcdef", limit=5) == "ab..."


def test_status_icon_animates_running_and_maps_common_states() -> None:
    assert [status_icon("running", frame=index) for index in range(4)] == ["|", "/", "-", "\\"]
    assert status_icon("idle") == "✓"
    assert status_icon("error") == "x"
    assert status_icon("unknown") == "•"


def test_model_label_prefers_provider_and_model() -> None:
    assert model_label({"provider": "local", "model": "example-model"}) == "local/example-model"
    assert model_label({"model": "example-model"}) == "example-model"
    assert model_label(None) == ""


def test_grouped_agents_uses_requester_scope() -> None:
    groups = grouped_agents(
        [
            {
                "agent_id": "b",
                "created_at": 100.0,
                "requester": {"scope_id": "scope-1", "display": "Codex one"},
            },
            {
                "agent_id": "a",
                "created_at": 300.0,
                "requester": {"scope_id": "scope-1", "display": "Codex one"},
            },
            {
                "agent_id": "c",
                "created_at": 200.0,
                "requester": {"scope_id": "scope-2", "display": "Claude two"},
            },
        ]
    )

    assert [(key, label) for key, label, _agents in groups] == [
        ("scope-1", "Codex one"),
        ("scope-2", "Claude two"),
    ]
    assert [agent["agent_id"] for _key, _label, agents in groups for agent in agents] == ["a", "b", "c"]


def test_grouped_agents_falls_back_to_lowest_total_seconds_as_newest() -> None:
    groups = grouped_agents(
        [
            {"agent_id": "old", "total_seconds": 50, "requester": {"scope_id": "scope-1"}},
            {"agent_id": "new", "total_seconds": 5, "requester": {"scope_id": "scope-1"}},
        ]
    )

    assert [agent["agent_id"] for _key, _label, agents in groups for agent in agents] == ["new", "old"]


def test_usage_stats_label_formats_token_totals() -> None:
    label = usage_stats_label(
        {
            "turn_count": 2,
            "usage": {
                "input_tokens": 1234,
                "output_tokens": 56,
                "cache_read_tokens": 1000,
                "cache_write_tokens": 2,
                "context_percent": 12.345,
                "tokens_per_second": 7.89,
            },
        }
    )

    assert label == "turns 2  in 1,234  out 56  cache 1,002  ctx 12.3%  t/s 7.9/s"


def test_usage_stats_label_handles_missing_usage() -> None:
    assert usage_stats_label({"turn_count": 0}) == "turns 0  in 0  out 0  cache 0  ctx --  t/s --"


def test_observed_and_score_labels_format_stats() -> None:
    stats = {
        "observed_by_parent": True,
        "observed_via": "listen",
        "observation_count": 2,
        "latest_score": {
            "score": 8,
            "category": "review",
            "sentiment": "net-positive",
        },
    }

    assert observed_label(stats) == "yes via listen · 2x"
    assert score_label(stats) == "8 · review · net-positive"
    assert observed_label({"observed_by_parent": False}) == "no"
    assert score_label({}) == "--"


def test_watching_label_reflects_active_piw_listeners() -> None:
    assert watching_label({"agent_id": "a", "active_listeners": 0}) == "no"
    assert watching_label({"agent_id": "a", "active_listeners": 1}) == "yes via piw"
    assert watching_label({"agent_id": "a", "active_listeners": 3}) == "yes via piw · 3x"
    assert watching_label({}) == "unknown"
    assert watching_label(None) == "unknown"


def test_stats_summary_label_formats_persisted_totals() -> None:
    assert (
        stats_summary_label(
            {
                "total_agents": 3,
                "observed_agents": 2,
                "unobserved_agents": 1,
                "scores": 4,
                "average_score": 7.25,
            }
        )
        == "agents 3  observed 2  unobserved 1  scores 4  avg score 7.25"
    )
    assert stats_summary_label({}) == "agents 0  observed 0  unobserved 0  scores 0"


def test_tui_reconcile_selection_keeps_existing_agent() -> None:
    app = PiAgentTui()
    app.agents = [{"agent_id": "first"}, {"agent_id": "second"}]
    app.selected_agent_id = "second"

    app.reconcile_selection()

    assert app.selected_agent_id == "second"


def test_tui_reconcile_selection_selects_first_when_missing() -> None:
    app = PiAgentTui()
    app.agents = [{"agent_id": "first"}]
    app.selected_agent_id = "missing"

    app.reconcile_selection()

    assert app.selected_agent_id == "first"


def test_tui_mounts_with_tree_navigation() -> None:
    class FakeClient:
        def request(self, command: str, **kwargs):
            if command == "tui_summary":
                return {
                    "agents": [
                        {
                            "agent_id": "agent-1",
                            "status": "running",
                            "provider": "local",
                            "model": "example-model",
                            "requester": {"scope_id": "scope-1", "display": "Codex fake"},
                        }
                    ]
                }
            if command == "inspect":
                return {
                    "agent_id": kwargs["agent_id"],
                    "status": "running",
                    "provider": "local",
                    "model": "example-model",
                    "cwd": "/tmp",
                    "turn_count": 0,
                    "tool_mode": "read-only",
                    "requester": {"display": "Codex fake", "lineage": []},
                    "prompts": [{"turn": 1, "text": "hello"}],
                    "transcript": [{"kind": "prompt", "turn": 1, "text": "hello"}],
                    "final_text": "",
                }
            raise AssertionError(command)

    async def check() -> None:
        app = PiAgentTui()
        app.client = FakeClient()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.selected_agent_id == "agent-1"
            assert app.agent_tree.cursor_node is not None
            await pilot.press("left")
            await pilot.press("right")

    anyio.run(check)


def test_tui_refresh_preserves_sidebar_cursor() -> None:
    class FakeClient:
        def request(self, command: str, **kwargs):
            if command == "tui_summary":
                return {
                    "agents": [
                        {
                            "agent_id": "agent-1",
                            "status": "running",
                            "provider": "local",
                            "model": "example-model",
                            "requester": {"scope_id": "scope-1", "display": "Codex fake"},
                        },
                        {
                            "agent_id": "agent-2",
                            "status": "idle",
                            "provider": "mistral",
                            "model": "mistral-medium-3.5",
                            "requester": {"scope_id": "scope-1", "display": "Codex fake"},
                        },
                    ]
                }
            if command == "inspect":
                return {
                    "agent_id": kwargs["agent_id"],
                    "status": "running",
                    "provider": "local",
                    "model": "example-model",
                    "cwd": "/tmp",
                    "turn_count": 0,
                    "tool_mode": "read-only",
                    "requester": {"display": "Codex fake", "lineage": []},
                    "prompts": [],
                    "transcript": [],
                    "final_text": "",
                }
            raise AssertionError(command)

    async def check() -> None:
        app = PiAgentTui()
        app.client = FakeClient()
        async with app.run_test() as pilot:
            await pilot.pause()
            parent_node = app.agent_tree.root.children[0]
            app.agent_tree.move_cursor(parent_node, animate=False)
            app.remember_sidebar_cursor(parent_node)
            app.refresh_data()
            await pilot.pause()
            assert app.agent_tree.cursor_node is not None
            assert app.agent_tree.cursor_node.data == {"type": "parent", "parent_id": "scope-1"}

            agent_node = app.agent_tree.root.children[0].children[1]
            app.agent_tree.move_cursor(agent_node, animate=False)
            app.remember_sidebar_cursor(agent_node)
            app.refresh_data()
            await pilot.pause()
            assert app.agent_tree.cursor_node is not None
            assert app.agent_tree.cursor_node.data == {"type": "agent", "agent_id": "agent-2"}

    anyio.run(check)


def test_tui_sidebar_label_puts_model_before_agent_id() -> None:
    app = PiAgentTui()
    label = app.agent_node_label(
        {
            "agent_id": "abcdef123456",
            "status": "idle",
            "model": "mistral-medium-3.5",
        }
    )

    plain = label.plain
    assert plain.startswith("✓ mistral-medium-3.5  abcdef12")


def test_tui_inspect_opts_out_of_observation() -> None:
    captured: list[dict[str, object]] = []

    class CapturingClient:
        def request(self, command: str, **kwargs):
            captured.append({"command": command, **kwargs})
            return {"agent_id": kwargs.get("agent_id"), "status": "idle"}

    app = PiAgentTui()
    app.client = CapturingClient()
    app.selected_agent_id = "agent-1"
    app.load_detail()

    inspect_calls = [call for call in captured if call["command"] == "inspect"]
    assert inspect_calls, "TUI should inspect the selected agent"
    # A passive dashboard must never record observation, or observation_count
    # climbs every poll and the stats log grows without bound.
    assert all(call.get("observe") is False for call in inspect_calls)


def test_tui_refresh_keeps_last_agents_on_transient_error() -> None:
    class FlakyClient:
        def __init__(self) -> None:
            self.fail = False

        def request(self, command: str, **kwargs):
            if command == "tui_summary":
                if self.fail:
                    raise DaemonClientError("timed out")
                return {
                    "agents": [
                        {
                            "agent_id": "agent-1",
                            "status": "running",
                            "provider": "local",
                            "model": "example-model",
                            "requester": {"scope_id": "scope-1", "display": "Codex fake"},
                        }
                    ],
                    "stats": {"total_agents": 5, "observed_agents": 4},
                }
            if command == "inspect":
                return {
                    "agent_id": kwargs.get("agent_id"),
                    "status": "running",
                    "provider": "local",
                    "model": "example-model",
                    "cwd": "/tmp",
                    "turn_count": 0,
                    "tool_mode": "read-only",
                    "requester": {"display": "Codex fake", "lineage": []},
                    "prompts": [],
                    "transcript": [],
                    "final_text": "",
                }
            raise AssertionError(command)

    async def check() -> None:
        app = PiAgentTui()
        client = FlakyClient()
        app.client = client
        async with app.run_test() as pilot:
            await pilot.pause()
            assert [agent["agent_id"] for agent in app.agents] == ["agent-1"]

            client.fail = True
            app.refresh_data()
            await pilot.pause()

            # The tree must not blank on a single failed poll.
            assert [agent["agent_id"] for agent in app.agents] == ["agent-1"]
            assert app.error is None
            assert app.refresh_failed is True

    anyio.run(check)


def test_tui_renders_thinking_block_distinctly() -> None:
    from rich.panel import Panel

    written: list[object] = []

    class FakeLog:
        def write(self, renderable: object) -> None:
            written.append(renderable)

    app = PiAgentTui()
    app.write_transcript_item(FakeLog(), {"kind": "thinking_stream", "turn": 2, "text": "weighing options"})
    app.write_transcript_item(FakeLog(), {"kind": "message", "turn": 2, "role": "assistant", "text": "done"})

    panel = written[0]
    assert isinstance(panel, Panel)
    assert "Thinking · turn 2 · streaming" in str(panel.title)
    # The spoken answer keeps its own (non-thinking) panel.
    assert "Thinking" not in str(written[1].title)


def test_tui_animation_tick_advances_only_when_needed() -> None:
    app = PiAgentTui()
    app.agents = [{"status": "idle"}]
    app.animate_status()
    assert app.status_frame == 0

    app.agents = [{"status": "running"}]
    app.render_tree = lambda: None  # type: ignore[method-assign]
    app.query_one = lambda *args, **kwargs: type("FakeStatic", (), {"update": lambda self, value: None})()
    app.animate_status()
    assert app.status_frame == 1
