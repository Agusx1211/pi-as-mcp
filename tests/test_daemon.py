from __future__ import annotations

import json
import os
import stat
import subprocess
import threading
import time
from pathlib import Path

import pytest

from pi_as_mcp.daemon import DaemonState, ParentIdentity, agent_spawn_rank
from pi_as_mcp.pi_rpc import PiRpcError


def write_config(tmp_path: Path, model_limits: dict[str, int]) -> Path:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"agents": {"concurrency_limits": {"models": model_limits}}}),
        encoding="utf-8",
    )
    return path


def write_score_config(tmp_path: Path, *, enabled: bool) -> Path:
    path = tmp_path / f"score-{enabled}.json"
    path.write_text(json.dumps({"agents": {"enable_score": enabled}}), encoding="utf-8")
    return path


def write_fake_pi(tmp_path: Path) -> Path:
    path = tmp_path / "fake-pi-daemon"
    path.write_text(
        """#!/usr/bin/env python3
import json
import sys
import time

if "--list-models" in sys.argv:
    print("provider   model                    context")
    print("local  example-model  128K")
    raise SystemExit(0)

line = sys.stdin.readline()
request = json.loads(line)
print(json.dumps({"id": request["id"], "type": "response", "command": "prompt", "success": True}), flush=True)
print(json.dumps({"type": "agent_start"}), flush=True)
while True:
    time.sleep(1)
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def write_finishing_fake_pi(tmp_path: Path) -> Path:
    path = tmp_path / "fake-pi-daemon-finished"
    path.write_text(
        """#!/usr/bin/env python3
import json
import sys
import time

if "--list-models" in sys.argv:
    print("provider   model                    context")
    print("local  example-model  128K")
    raise SystemExit(0)

line = sys.stdin.readline()
request = json.loads(line)
print(json.dumps({"id": request["id"], "type": "response", "command": "prompt", "success": True}), flush=True)
print(json.dumps({"type": "agent_start"}), flush=True)
message = {
    "role": "assistant",
    "content": [{"type": "text", "text": "done without parent read"}],
    "usage": {"input": 9, "output": 4, "totalTokens": 13},
}
print(json.dumps({"type": "message_end", "message": message}), flush=True)
print(json.dumps({"type": "agent_end", "messages": [message]}), flush=True)
while True:
    time.sleep(1)
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def write_recording_fake_pi(tmp_path: Path) -> Path:
    path = tmp_path / "fake-pi-recording"
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
import time

if "--list-models" in sys.argv:
    print("provider   model                    context")
    print("local  example-model  128K")
    raise SystemExit(0)

line = sys.stdin.readline()
request = json.loads(line)
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "daemon-worker-call.json"), "w") as handle:
    json.dump({"argv": sys.argv, "message": request.get("message")}, handle)
print(json.dumps({"id": request["id"], "type": "response", "command": "prompt", "success": True}), flush=True)
print(json.dumps({"type": "agent_start"}), flush=True)
while True:
    time.sleep(1)
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def write_slow_fake_pi(tmp_path: Path, *, delay: float) -> Path:
    """A worker whose model-validation spawn (`--list-models`) is slow, so the
    expensive part of a start is observable from another thread."""
    path = tmp_path / f"fake-pi-slow-{delay}"
    path.write_text(
        f"""#!/usr/bin/env python3
import json
import sys
import time

if "--list-models" in sys.argv:
    time.sleep({delay})
    print("provider   model                    context")
    print("local  example-model  128K")
    raise SystemExit(0)

line = sys.stdin.readline()
request = json.loads(line)
print(json.dumps({{"id": request["id"], "type": "response", "command": "prompt", "success": True}}), flush=True)
print(json.dumps({{"type": "agent_start"}}), flush=True)
while True:
    time.sleep(1)
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_daemon_start_does_not_hold_lock_during_spawn(tmp_path: Path) -> None:
    fake_pi = write_slow_fake_pi(tmp_path, delay=1.5)
    state = DaemonState()
    identity = ParentIdentity(scope_id="slow-spawn-scope", owner_pid=None, label="slow")
    try:
        state.manager_for(identity)._runner.pi_bin = str(fake_pi)

        result: dict[str, object] = {}

        def run() -> None:
            result["snapshot"] = state.start(
                identity,
                prompt="spawn slowly",
                cwd=str(tmp_path),
                model="local/example-model",
                provider=None,
                tool_mode="none",
                include_events=False,
            )

        worker = threading.Thread(target=run)
        worker.start()
        try:
            # Let the start enter the slow worker spawn (it briefly took the lock
            # to reserve a slot, then released it before spawning).
            time.sleep(0.3)
            # If the lock were held across the whole spawn this would block until
            # the ~1.5s spawn finished; we require it free almost immediately.
            acquired = state._lock.acquire(timeout=0.5)
            assert acquired, "daemon lock was held across the agent spawn"
            state._lock.release()
        finally:
            worker.join(timeout=10)

        assert isinstance(result.get("snapshot"), dict)
        assert result["snapshot"]["agent_id"]
    finally:
        state.close()


def test_daemon_concurrent_starts_respect_limit(tmp_path: Path, monkeypatch) -> None:
    fake_pi = write_slow_fake_pi(tmp_path, delay=0.7)
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(write_config(tmp_path, {"local/example-model": 1})))
    state = DaemonState()
    first_identity = ParentIdentity(scope_id="conc-first", owner_pid=None, label="first")
    second_identity = ParentIdentity(scope_id="conc-second", owner_pid=None, label="second")
    try:
        state.manager_for(first_identity)._runner.pi_bin = str(fake_pi)
        state.manager_for(second_identity)._runner.pi_bin = str(fake_pi)

        results: dict[str, tuple[str, object]] = {}
        barrier = threading.Barrier(2)

        def run(key: str, identity: ParentIdentity) -> None:
            barrier.wait()
            try:
                snapshot = state.start(
                    identity,
                    prompt=key,
                    cwd=str(tmp_path),
                    model="local/example-model",
                    provider=None,
                    tool_mode="none",
                    include_events=False,
                )
                results[key] = ("ok", snapshot)
            except PiRpcError as exc:
                results[key] = ("err", str(exc))

        threads = [
            threading.Thread(target=run, args=("first", first_identity)),
            threading.Thread(target=run, args=("second", second_identity)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        # The reservation (held only under the lock) keeps the limit honest even
        # while the winner is still spawning: exactly one start wins, one is rejected.
        kinds = sorted(value[0] for value in results.values())
        assert kinds == ["err", "ok"], results
        rejection = next(value[1] for value in results.values() if value[0] == "err")
        assert "concurrency limit reached" in str(rejection)
    finally:
        state.close()


def test_daemon_unsafe_read_only_config_upgrades_read_only_requests(tmp_path: Path, monkeypatch) -> None:
    fake_pi = write_recording_fake_pi(tmp_path)
    config = tmp_path / "unsafe.json"
    config.write_text(json.dumps({"agents": {"unsafe_read_only": True}}), encoding="utf-8")
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(config))
    monkeypatch.setenv("PI_AS_MCP_STATS_DIR", str(tmp_path / "stats"))

    state = DaemonState()
    identity = ParentIdentity(scope_id="unsafe-scope", owner_pid=None, label="x")
    try:
        state.manager_for(identity)._runner.pi_bin = str(fake_pi)
        state.start(
            identity,
            prompt="inspect the dirty changes",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="read-only",
            include_events=False,
        )

        call_path = tmp_path / "daemon-worker-call.json"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if call_path.exists():
                break
            time.sleep(0.05)
        else:
            raise AssertionError("worker did not record its launch")

        call = json.loads(call_path.read_text(encoding="utf-8"))
        argv = call["argv"]
        tools = argv[argv.index("--tools") + 1].split(",")
        # A plain read-only request was upgraded to full tools + guarded prompt.
        assert "bash" in tools
        assert call["message"].startswith("IMPORTANT — READ-ONLY")
    finally:
        state.close()


def test_daemon_reaps_sessions_when_owner_pid_exits(tmp_path: Path) -> None:
    fake_pi = write_fake_pi(tmp_path)
    owner = subprocess.Popen(["sleep", "60"])
    state = DaemonState()
    try:
        manager = state.manager_for(
            ParentIdentity(scope_id="owner-cleanup-test", owner_pid=owner.pid, label="fake-owner")
        )
        manager._runner.pi_bin = str(fake_pi)
        started = manager.start(
            prompt="keep running",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )
        worker_pid = manager._get(started.agent_id).process.pid
        assert manager.summary()[0]["status"] == "running"

        owner.terminate()
        owner.wait(timeout=5)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not manager.summary():
                break
            time.sleep(0.1)
        else:
            raise AssertionError("daemon did not reap dead owner scope")

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(worker_pid, 0)
            except OSError:
                break
            time.sleep(0.05)
        else:
            raise AssertionError(f"worker pid {worker_pid} survived owner cleanup")
    finally:
        try:
            owner.kill()
        except ProcessLookupError:
            pass
        state.close()


def test_daemon_can_find_agent_across_parent_scopes(tmp_path: Path) -> None:
    fake_pi = write_fake_pi(tmp_path)
    state = DaemonState()
    try:
        manager = state.manager_for(ParentIdentity(scope_id="mcp-scope", owner_pid=None, label="mcp"))
        manager._runner.pi_bin = str(fake_pi)
        started = manager.start(
            prompt="keep running",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )

        cli_manager = state.manager_for(ParentIdentity(scope_id="cli-scope", owner_pid=None, label="cli"))
        assert cli_manager is not manager
        assert state.manager_for_agent(started.agent_id) is manager
    finally:
        state.close()


def test_daemon_global_summary_includes_requester_identity(tmp_path: Path) -> None:
    fake_pi = write_fake_pi(tmp_path)
    state = DaemonState()
    try:
        manager = state.manager_for(
            ParentIdentity(
                scope_id="codex-mcp-scope",
                owner_pid=os.getpid(),
                label="hint:mcp:codex-instance",
                peer_pid=os.getpid(),
            )
        )
        manager._runner.pi_bin = str(fake_pi)
        started = manager.start(
            prompt="keep running",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )

        agents = state.global_summary()
        assert len(agents) == 1
        assert agents[0]["agent_id"] == started.agent_id
        assert agents[0]["parent_scope_id"] == "codex-mcp-scope"
        assert agents[0]["requester"]["label"] == "hint:mcp:codex-instance"
        assert agents[0]["requester"]["instance"] == "mcp:codex-instance"
        assert agents[0]["requester"]["peer_pid"] == os.getpid()
    finally:
        state.close()


def test_agent_spawn_rank_uses_created_at_then_total_seconds() -> None:
    assert agent_spawn_rank({"created_at": "1,234.5", "total_seconds": 1}) == 1234.5
    assert agent_spawn_rank({"created_at": True, "total_seconds": 2}) == -2
    assert agent_spawn_rank({"total_seconds": "3.5"}) == -3.5
    assert agent_spawn_rank({}) == 0


def test_daemon_global_summary_orders_latest_spawn_first(tmp_path: Path) -> None:
    fake_pi = write_fake_pi(tmp_path)
    state = DaemonState()
    try:
        manager = state.manager_for(ParentIdentity(scope_id="spawn-order-scope", owner_pid=None, label="cli"))
        manager._runner.pi_bin = str(fake_pi)
        first = manager.start(
            prompt="first",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )
        time.sleep(0.01)
        second = manager.start(
            prompt="second",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )

        agents = state.global_summary()
        assert [agent["agent_id"] for agent in agents[:2]] == [second.agent_id, first.agent_id]
        assert agents[0]["created_at"] > agents[1]["created_at"]
    finally:
        state.close()


def test_daemon_enforces_model_concurrency_limits_across_parent_scopes(
    tmp_path: Path, monkeypatch
) -> None:
    fake_pi = write_fake_pi(tmp_path)
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(write_config(tmp_path, {"local/example-model": 1})))
    state = DaemonState()
    first_identity = ParentIdentity(scope_id="first-scope", owner_pid=None, label="first")
    second_identity = ParentIdentity(scope_id="second-scope", owner_pid=None, label="second")
    try:
        state.manager_for(first_identity)._runner.pi_bin = str(fake_pi)
        state.manager_for(second_identity)._runner.pi_bin = str(fake_pi)

        first = state.start(
            first_identity,
            prompt="first",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )

        with pytest.raises(PiRpcError, match="concurrency limit reached"):
            state.start(
                second_identity,
                prompt="second",
                cwd=str(tmp_path),
                model="local/example-model",
                provider=None,
                tool_mode="none",
                include_events=False,
            )

        assert "concurrency" not in first
    finally:
        state.close()


def test_daemon_model_concurrency_limit_releases_after_stop(tmp_path: Path, monkeypatch) -> None:
    fake_pi = write_fake_pi(tmp_path)
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(write_config(tmp_path, {"example-model": 1})))
    state = DaemonState()
    first_identity = ParentIdentity(scope_id="first-scope", owner_pid=None, label="first")
    second_identity = ParentIdentity(scope_id="second-scope", owner_pid=None, label="second")
    try:
        first_manager = state.manager_for(first_identity)
        first_manager._runner.pi_bin = str(fake_pi)
        second_manager = state.manager_for(second_identity)
        second_manager._runner.pi_bin = str(fake_pi)

        first = state.start(
            first_identity,
            prompt="first",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )
        first_manager.stop(str(first["agent_id"]))

        second = state.start(
            second_identity,
            prompt="second",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )

        assert second["agent_id"] != first["agent_id"]
    finally:
        state.close()


def test_daemon_idle_agents_do_not_consume_concurrency(tmp_path: Path, monkeypatch) -> None:
    fake_pi = write_finishing_fake_pi(tmp_path)
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(write_config(tmp_path, {"example-model": 1})))
    state = DaemonState()
    identity = ParentIdentity(scope_id="idle-scope", owner_pid=None, label="idle")
    try:
        manager = state.manager_for(identity)
        manager._runner.pi_bin = str(fake_pi)

        first = state.start(
            identity,
            prompt="first",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )
        first_id = str(first["agent_id"])

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if state.agent_stats(first_id).get("status") == "idle":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("first agent never went idle")

        # The first agent is idle (live but done); it must not block a new
        # delegation even though the model limit is 1.
        second = state.start(
            identity,
            prompt="second",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )
        assert second["agent_id"] != first_id
    finally:
        state.close()


def test_daemon_delegate_lists_sibling_agents(tmp_path: Path, monkeypatch) -> None:
    fake_pi = write_finishing_fake_pi(tmp_path)
    monkeypatch.setenv("PI_AS_MCP_STATS_DIR", str(tmp_path / "stats"))
    state = DaemonState()
    identity = ParentIdentity(scope_id="sibling-scope", owner_pid=None, label="sibling")
    try:
        manager = state.manager_for(identity)
        manager._runner.pi_bin = str(fake_pi)

        first = state.start(
            identity,
            prompt="map the repo",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )
        first_id = str(first["agent_id"])

        second = state.start(
            identity,
            prompt="run the tests",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )
        second_id = str(second["agent_id"])

        siblings = state.sibling_overview(identity, exclude_agent_id=second_id)
        sibling_ids = {row["agent_id"] for row in siblings}
        assert sibling_ids == {first_id}
        only = siblings[0]
        assert only["status"]
        assert only["model"] == "example-model"
        assert "map the repo" in only["summary"]
    finally:
        state.close()


def test_daemon_records_start_stats(tmp_path: Path, monkeypatch) -> None:
    fake_pi = write_fake_pi(tmp_path)
    monkeypatch.setenv("PI_AS_MCP_STATS_DIR", str(tmp_path / "stats"))
    state = DaemonState()
    identity = ParentIdentity(scope_id="stats-scope", owner_pid=None, label="stats")
    try:
        state.manager_for(identity)._runner.pi_bin = str(fake_pi)
        started = state.start(
            identity,
            prompt="collect stats",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )

        stats = state.agent_stats(str(started["agent_id"]))

        assert stats["agent_id"] == started["agent_id"]
        assert stats["model"] == "example-model"
        assert stats["prompts"][0]["text"] == "collect stats"
        assert stats["observed_by_parent"] is False
    finally:
        state.close()


def test_daemon_records_completed_stats_before_parent_observes_output(tmp_path: Path, monkeypatch) -> None:
    fake_pi = write_finishing_fake_pi(tmp_path)
    monkeypatch.setenv("PI_AS_MCP_STATS_DIR", str(tmp_path / "stats"))
    state = DaemonState()
    identity = ParentIdentity(scope_id="unobserved-scope", owner_pid=None, label="stats")
    try:
        state.manager_for(identity)._runner.pi_bin = str(fake_pi)
        started = state.start(
            identity,
            prompt="finish quietly",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )
        agent_id = str(started["agent_id"])

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            stats = state.agent_stats(agent_id)
            if stats.get("turn_count") == 1:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("stats did not record completed turn")

        assert stats["status"] == "idle"
        assert stats["final_text_preview"] == "done without parent read"
        assert stats["usage"]["input_tokens"] == 9
        assert stats["observed_by_parent"] is False
        assert state.stats_summary()["unobserved_agents"] == 1
    finally:
        state.close()


def test_daemon_records_observed_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PI_AS_MCP_STATS_DIR", str(tmp_path / "stats"))
    state = DaemonState()
    identity = ParentIdentity(scope_id="observed-scope", owner_pid=None, label="observer")
    snapshot = {
        "agent_id": "agent-1",
        "status": "idle",
        "turn_count": 1,
        "final_text": "done",
    }

    state.record_agent_observed(via="listen", snapshot=snapshot, identity=identity)

    assert state.agent_stats("agent-1")["observed_by_parent"] is True
    assert state.agent_stats("agent-1")["observed_via"] == "listen"
    state.close()


def test_daemon_score_requires_enabled_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PI_AS_MCP_STATS_DIR", str(tmp_path / "stats"))
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(write_score_config(tmp_path, enabled=False)))
    state = DaemonState()
    identity = ParentIdentity(scope_id="score-scope", owner_pid=None, label="scorer")
    try:
        with pytest.raises(PiRpcError, match="disabled"):
            state.score_agent(
                identity,
                agent_id="agent-1",
                score=8,
                category="review",
                comment="good result",
            )
    finally:
        state.close()


def test_daemon_score_records_when_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PI_AS_MCP_STATS_DIR", str(tmp_path / "stats"))
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(write_score_config(tmp_path, enabled=True)))
    state = DaemonState()
    identity = ParentIdentity(scope_id="score-scope", owner_pid=None, label="scorer")
    try:
        assert state.score_hint("agent-1") is not None
        scored = state.score_agent(
            identity,
            agent_id="agent-1",
            score=3,
            category="research",
            comment="missed the main issue",
        )

        assert scored["sentiment"] == "net-negative"
        assert state.agent_stats("agent-1")["latest_score"]["score"] == 3
        assert state.score_hint("agent-1") is None
    finally:
        state.close()
