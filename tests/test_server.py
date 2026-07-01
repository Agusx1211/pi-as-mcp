from __future__ import annotations

import json

import anyio

from pi_as_mcp import server, skill
from pi_as_mcp.server import mcp


def test_mcp_surface_is_small_and_structured() -> None:
    async def check() -> None:
        tools = {tool.name: tool for tool in await mcp.list_tools()}

        assert set(tools) == {
            "delegate",
            "agent_reply",
            "agent_peek",
            "agent_stop",
            "models",
        }
        assert tools["delegate"].outputSchema["title"] == "DelegateResult"
        delegate_inputs = tools["delegate"].inputSchema["properties"]
        assert "thinking" not in delegate_inputs
        assert "timeout_seconds" not in delegate_inputs
        assert "validate_model" not in delegate_inputs
        reply_inputs = tools["agent_reply"].inputSchema["properties"]
        assert "timeout_seconds" not in reply_inputs
        assert "monitor_command" in tools["delegate"].outputSchema["properties"]
        assert "monitor_after_turn_count" in tools["delegate"].outputSchema["properties"]
        assert "queued_turn_expected" in tools["delegate"].outputSchema["properties"]
        assert "other_agents" in tools["delegate"].outputSchema["properties"]
        assert "other_agents_hint" in tools["delegate"].outputSchema["properties"]
        assert "result" not in tools["delegate"].outputSchema["properties"]
        for name in {"agent_peek", "agent_stop"}:
            schema = tools[name].outputSchema
            for field in {"tool_calls", "stderr_tail", "event_tail"}:
                assert schema["properties"][field]["type"] == "array"
                assert "default" not in schema["properties"][field]
        # The only resource is the MCP-provided "cheap sub-agents" skill.
        resources = await mcp.list_resources()
        assert len(resources) == 1
        assert resources[0].name == "cheap-subagents"
        assert str(resources[0].uri) == skill.SKILL_RESOURCE_URI
        assert await mcp.list_resource_templates() == []
        assert await mcp.list_prompts() == []

    anyio.run(check)


def test_score_tool_is_config_gated(tmp_path, monkeypatch) -> None:
    async def check_tool_names() -> set[str]:
        return {tool.name for tool in await mcp.list_tools()}

    disabled = tmp_path / "disabled.json"
    disabled.write_text(json.dumps({"agents": {"enable_score": False}}), encoding="utf-8")
    enabled = tmp_path / "enabled.json"
    enabled.write_text(json.dumps({"agents": {"enable_score": True}}), encoding="utf-8")

    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(enabled))
    server.sync_score_tool()
    assert "score" in anyio.run(check_tool_names)

    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(disabled))
    server.sync_score_tool()
    assert "score" not in anyio.run(check_tool_names)


def test_structured_tools_return_schema_safe_content(monkeypatch) -> None:
    class FakeClient:
        def request(self, command: str, **kwargs):
            if command == "delegate":
                return {
                    "agent_id": "agent-1",
                    "status": "running",
                    "turn_count": 0,
                }
            if command == "reply":
                # Mirrors the daemon: the reply response carries the pre-prompt
                # state captured atomically under the session lock.
                was_idle = kwargs.get("agent_id") == "idle-agent"
                return {
                    "agent_id": str(kwargs.get("agent_id") or "agent-1"),
                    "status": "running",
                    "turn_count": 3 if was_idle else 0,
                    "reply_after_turn_count": 3 if was_idle else 0,
                    "reply_was_running": not was_idle,
                }
            if command in {"peek", "stop"}:
                is_idle_reply_probe = kwargs.get("agent_id") == "idle-agent"
                return {
                    "agent_id": str(kwargs.get("agent_id") or "agent-1"),
                    "status": "idle" if is_idle_reply_probe else "running",
                    "cwd": "/tmp/project",
                    "provider": "local",
                    "model": "example-model",
                    "tool_mode": "full",
                    "final_text": "",
                    "turn_count": 3 if is_idle_reply_probe else 0,
                    "event_counts": {},
                    "tool_call_count": 0,
                }
            if command == "models":
                return {
                    "models": [
                        {
                            "alias": "example-model",
                            "provider": "local",
                            "model": "example-model",
                        }
                    ]
                }
            raise AssertionError(command)

    async def check() -> None:
        monkeypatch.setattr(server, "client", FakeClient())
        monkeypatch.setattr(
            server,
            "wait_command",
            lambda agent_id, after_turn_count=0: f"piw {agent_id}"
            if after_turn_count == 0
            else f"piw {agent_id} -a {after_turn_count}",
        )

        _, delegate = await mcp.call_tool("delegate", {"prompt": "one"})
        assert delegate["monitor_command"] == "piw agent-1"
        assert delegate["monitor_after_turn_count"] == 0
        assert delegate["queued_turn_expected"] is True

        _, reply = await mcp.call_tool("agent_reply", {"agent_id": "agent-1", "prompt": "two"})
        assert reply["monitor_command"] == "piw agent-1"
        assert reply["monitor_after_turn_count"] == 0
        assert reply["queued_turn_expected"] is False
        assert "running turn" in reply["monitor_hint"]

        _, idle_reply = await mcp.call_tool("agent_reply", {"agent_id": "idle-agent", "prompt": "two"})
        assert idle_reply["monitor_command"] == "piw idle-agent -a 3"
        assert idle_reply["monitor_after_turn_count"] == 3
        assert idle_reply["queued_turn_expected"] is True

        for tool_name in {"agent_peek", "agent_stop"}:
            _, structured = await mcp.call_tool(tool_name, {"agent_id": "agent-1"})
            assert structured["tool_calls"] == []
            assert structured["stderr_tail"] == []
            assert structured["event_tail"] == []
            assert structured["error"] == ""

        _, models = await mcp.call_tool("models", {})
        assert models["models"][0]["provider"] == "local"

    anyio.run(check)


def test_score_tool_records_structured_rating(tmp_path, monkeypatch) -> None:
    enabled = tmp_path / "enabled.json"
    enabled.write_text(json.dumps({"agents": {"enable_score": True}}), encoding="utf-8")
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(enabled))

    class FakeClient:
        def request(self, command: str, **kwargs):
            assert command == "score"
            assert kwargs["agent_id"] == "agent-1"
            assert kwargs["score"] == 8
            assert kwargs["category"] == "review"
            assert kwargs["comment"] == "useful"
            return {
                "agent_id": "agent-1",
                "score": 8,
                "category": "review",
                "comment": "useful",
                "sentiment": "net-positive",
                "recorded": True,
            }

    async def check() -> None:
        monkeypatch.setattr(server, "client", FakeClient())
        server.sync_score_tool()
        _, result = await mcp.call_tool(
            "score",
            {
                "agent_id": "agent-1",
                "score": 8,
                "category": "review",
                "comment": "useful",
            },
        )
        assert result["recorded"] is True
        assert result["sentiment"] == "net-positive"

    try:
        anyio.run(check)
    finally:
        monkeypatch.delenv("PI_AS_MCP_CONFIG", raising=False)
        server.sync_score_tool()
