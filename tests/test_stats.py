from __future__ import annotations

from pi_as_mcp.stats import StatsStore


def test_stats_store_records_agent_observation_and_score(tmp_path) -> None:
    store = StatsStore(root=tmp_path)
    snapshot = {
        "agent_id": "agent-1",
        "status": "idle",
        "created_at": 100.0,
        "cwd": "/tmp/project",
        "provider": "local",
        "model": "example-model",
        "tool_mode": "read-only",
        "turn_count": 1,
        "tool_call_count": 2,
        "event_counts": {"agent_end": 1},
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "final_text": "done",
        "prompts": [{"turn": 1, "behavior": "prompt", "text": "summarize", "accepted": True}],
    }

    store.record_agent_snapshot(
        event_type="agent_started",
        snapshot=snapshot,
        requester={"display": "Codex mcp:abc"},
    )
    store.record_observed(
        agent_id="agent-1",
        via="listen",
        snapshot=snapshot,
        requester={"display": "Codex mcp:abc"},
    )
    score = store.record_score(
        agent_id="agent-1",
        score=8,
        category="review",
        comment="useful result",
        requester={"display": "Codex mcp:abc"},
    )

    stats = store.agent_stats("agent-1")
    summary = store.summary()

    assert score["sentiment"] == "net-positive"
    assert stats["observed_by_parent"] is True
    assert stats["observed_via"] == "listen"
    assert stats["prompts"][0]["text"] == "summarize"
    assert stats["latest_score"]["score"] == 8
    assert summary["total_agents"] == 1
    assert summary["observed_agents"] == 1
    assert summary["scores"] == 1
    assert summary["average_score"] == 8.0


def test_stats_store_does_not_mark_running_agent_observed(tmp_path) -> None:
    store = StatsStore(root=tmp_path)
    event = store.record_observed(
        agent_id="agent-1",
        via="peek",
        snapshot={"agent_id": "agent-1", "status": "running", "turn_count": 0, "final_text": ""},
        requester={},
    )

    assert event is None
    assert store.agent_stats("agent-1")["observed_by_parent"] is False


def test_stats_store_marks_returned_output_observed_even_if_agent_is_running(tmp_path) -> None:
    store = StatsStore(root=tmp_path)
    event = store.record_observed(
        agent_id="agent-1",
        via="peek",
        snapshot={"agent_id": "agent-1", "status": "running", "turn_count": 1, "final_text": "previous"},
        requester={},
    )

    assert event is not None
    assert store.agent_stats("agent-1")["observed_by_parent"] is True


def test_stats_store_start_event_does_not_erase_completed_snapshot(tmp_path) -> None:
    store = StatsStore(root=tmp_path)
    completed = {
        "agent_id": "agent-1",
        "status": "idle",
        "turn_count": 1,
        "final_text": "done",
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    started = {
        "agent_id": "agent-1",
        "status": "running",
        "turn_count": 0,
        "final_text": "",
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "prompts": [{"turn": 1, "behavior": "prompt", "text": "work", "accepted": True}],
    }

    store.record_agent_snapshot(event_type="agent_updated", snapshot=completed, requester={})
    store.record_agent_snapshot(event_type="agent_started", snapshot=started, requester={})

    stats = store.agent_stats("agent-1")
    assert stats["status"] == "idle"
    assert stats["turn_count"] == 1
    assert stats["final_text_preview"] == "done"
    assert stats["prompts"][0]["text"] == "work"
