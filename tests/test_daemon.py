from __future__ import annotations

import json
import os
import signal
import socket
import stat
import subprocess
import struct
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from pi_as_mcp.daemon import (
    DaemonState,
    ParentIdentity,
    RequestHandler,
    agent_spawn_rank,
    cli_parent_identity_from_peer,
    exposed_model_aliases,
    parent_identity_from_peer,
    process_identity_matches,
    unix_socket_peer_pid,
)
from pi_as_mcp.pi_rpc import PiRpcError


def test_unix_socket_peer_pid_uses_linux_peer_credentials() -> None:
    peer_socket = Mock()
    peer_socket.getsockopt.return_value = struct.pack("=3i", 1234, 1000, 1000)

    assert unix_socket_peer_pid(peer_socket, platform_name="linux") == 1234
    peer_socket.getsockopt.assert_called_once_with(
        socket.SOL_SOCKET,
        socket.SO_PEERCRED,
        struct.calcsize("=3i"),
    )


def test_unix_socket_peer_pid_uses_darwin_local_peerpid(monkeypatch) -> None:
    peer_socket = Mock()
    peer_socket.getsockopt.return_value = struct.pack("=i", 4321)
    monkeypatch.delattr(socket, "SOL_LOCAL", raising=False)
    monkeypatch.delattr(socket, "LOCAL_PEERPID", raising=False)

    assert unix_socket_peer_pid(peer_socket, platform_name="darwin") == 4321
    peer_socket.getsockopt.assert_called_once_with(
        0,
        0x002,
        struct.calcsize("=i"),
    )


def test_unix_socket_peer_pid_fails_closed_when_credentials_are_unavailable() -> None:
    peer_socket = Mock()
    peer_socket.getsockopt.side_effect = OSError("not supported")

    with pytest.raises(PiRpcError, match="could not determine Unix socket peer PID"):
        unix_socket_peer_pid(peer_socket, platform_name="darwin")


def test_unix_socket_peer_pid_requires_linux_so_peercred(monkeypatch) -> None:
    monkeypatch.delattr(socket, "SO_PEERCRED")

    with pytest.raises(PiRpcError, match="SO_PEERCRED is unavailable"):
        unix_socket_peer_pid(Mock(), platform_name="linux")


def test_unix_socket_peer_pid_rejects_invalid_pid() -> None:
    peer_socket = Mock()
    peer_socket.getsockopt.return_value = struct.pack("=i", 0)

    with pytest.raises(PiRpcError, match="invalid peer PID 0"):
        unix_socket_peer_pid(peer_socket, platform_name="darwin")


def test_unix_socket_peer_pid_rejects_unsupported_platform() -> None:
    with pytest.raises(PiRpcError, match="unsupported on freebsd"):
        unix_socket_peer_pid(Mock(), platform_name="freebsd")


def test_peer_pid_supports_mcp_owner_validation_and_cli_scoping() -> None:
    peer_socket = Mock()
    peer_socket.getsockopt.return_value = struct.pack("=i", 4321)
    peer_pid = unix_socket_peer_pid(peer_socket, platform_name="darwin")

    mcp_identity = parent_identity_from_peer(
        peer_pid,
        parent_hint="mcp:instance",
        parent_owner_pid=4321,
    )
    cli_identity = parent_identity_from_peer(peer_pid)

    assert mcp_identity.owner_pid == 4321
    assert mcp_identity.peer_pid == 4321
    assert cli_identity.owner_pid is None
    assert cli_identity.peer_pid == 4321
    assert cli_identity.scope_id == parent_identity_from_peer(4321).scope_id


def _spawn_shell_with_cli_peers(
    tmp_path: Path, name: str
) -> tuple[subprocess.Popen[bytes], tuple[int, int]]:
    pid_file = tmp_path / f"{name}.pids"
    shell = subprocess.Popen(
        [
            "bash",
            "-c",
            'sleep 60 & first=$!; sleep 60 & second=$!; '
            'printf "%s %s" "$first" "$second" > "$1"; wait',
            "bash",
            str(pid_file),
        ],
        start_new_session=True,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if pid_file.exists() and pid_file.read_text(encoding="utf-8").strip():
            first, second = pid_file.read_text(encoding="utf-8").split()
            return shell, (int(first), int(second))
        time.sleep(0.02)
    shell.terminate()
    shell.wait(timeout=5)
    raise AssertionError("test shell did not publish child PIDs")


def _stop_test_shell(shell: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(shell.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    shell.wait(timeout=5)


def test_cli_scope_is_stable_across_processes_and_isolated_between_shells(
    tmp_path: Path,
) -> None:
    first_shell, first_peers = _spawn_shell_with_cli_peers(tmp_path, "first")
    second_shell, second_peers = _spawn_shell_with_cli_peers(tmp_path, "second")
    try:
        first = cli_parent_identity_from_peer(first_peers[0])
        same_shell = cli_parent_identity_from_peer(first_peers[1])
        other_shell = cli_parent_identity_from_peer(second_peers[0])

        assert first.scope_id == same_shell.scope_id
        assert first.owner_pid == same_shell.owner_pid == first_shell.pid
        assert first.owner_start_time == same_shell.owner_start_time
        assert first.scope_id != other_shell.scope_id
        assert other_shell.owner_pid == second_shell.pid
    finally:
        _stop_test_shell(first_shell)
        _stop_test_shell(second_shell)


def test_cli_scope_fails_with_actionable_guidance_without_stable_shell(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "pi_as_mcp.daemon.process_info",
        lambda pid: {
            100: {"pid": 100, "ppid": 50, "start_time": "one", "command": "pi-agent"},
            50: {"pid": 50, "ppid": 1, "start_time": "two", "command": "editor"},
        }.get(pid),
    )

    with pytest.raises(PiRpcError, match="set PI_AGENT_PARENT_ID=<name>"):
        parent_identity_from_peer(100, parent_scope_mode="cli-shell")


def test_process_identity_match_rejects_reused_owner_pid(monkeypatch) -> None:
    monkeypatch.setattr("pi_as_mcp.daemon.pid_exists", lambda pid: True)
    monkeypatch.setattr("pi_as_mcp.daemon.proc_stat", lambda pid: (1, "new-start"))

    assert process_identity_matches(1234, "old-start") is False


def test_exposed_model_aliases_drops_disabled_and_adds_description(tmp_path, monkeypatch) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"defaultProvider": "local", "enabledModels": ["local/alpha", "local/beta"]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "agents": {
                    "models": {
                        "local/alpha": {"disabled": True},
                        "local/beta": {"description": "the good one"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(config))

    rows = exposed_model_aliases()
    refs = {f"{r['provider']}/{r['model']}" for r in rows}
    assert refs == {"local/beta"}  # disabled alpha is hidden
    assert rows[0]["description"] == "the good one"


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


def write_session_policy_config(
    path: Path,
    *,
    persist_sessions: bool,
    idle_eviction_seconds: float,
) -> None:
    path.write_text(
        json.dumps(
            {
                "agents": {
                    "persist_sessions": persist_sessions,
                    "idle_eviction_seconds": idle_eviction_seconds,
                }
            }
        ),
        encoding="utf-8",
    )


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


def write_replyable_fake_pi(tmp_path: Path) -> Path:
    path = tmp_path / "fake-pi-daemon-replyable"
    path.write_text(
        """#!/usr/bin/env python3
import json
import sys
import time

if "--list-models" in sys.argv:
    print("provider   model                    context")
    print("local  example-model  128K")
    raise SystemExit(0)

for line in sys.stdin:
    request = json.loads(line)
    message = request["message"]
    if message == "run":
        time.sleep(0.3)
    print(json.dumps({
        "id": request["id"],
        "type": "response",
        "command": "prompt",
        "success": True,
    }), flush=True)
    print(json.dumps({"type": "agent_start"}), flush=True)
    if message.startswith("idle"):
        answer = {
            "role": "assistant",
            "content": [{"type": "text", "text": "idle now"}],
        }
        print(json.dumps({"type": "message_end", "message": answer}), flush=True)
        print(json.dumps({"type": "agent_end", "messages": [answer]}), flush=True)
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


def test_daemon_reloads_session_policy_for_existing_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pi = write_replyable_fake_pi(tmp_path)
    config = tmp_path / "config.json"
    durable_sessions = tmp_path / "durable-sessions"
    write_session_policy_config(
        config,
        persist_sessions=True,
        idle_eviction_seconds=120,
    )
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(config))
    monkeypatch.setenv("PI_AS_MCP_SESSION_DIR", str(durable_sessions))
    monkeypatch.setenv("PI_AS_MCP_STATS_DIR", str(tmp_path / "stats"))

    state = DaemonState()
    identity = ParentIdentity(scope_id="live-config-scope", owner_pid=None, label="test")
    try:
        manager = state.manager_for(identity)
        manager._runner.pi_bin = str(fake_pi)

        persisted_start = state.start(
            identity,
            prompt="idle persisted",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )
        persisted_id = str(persisted_start["agent_id"])
        persisted_done, timed_out = manager.listen(
            persisted_id,
            after_turn_count=0,
            timeout_seconds=5,
        )
        assert timed_out is False
        assert persisted_done.status == "idle"
        persisted = manager._get(persisted_id)
        assert persisted.session_dir == durable_sessions
        assert persisted.idle_eviction_seconds == 120

        # Simulate an agent that already depends on its durable state, then
        # disable both persistence and eviction in the same long-lived scope.
        assert persisted._evict() is True
        assert persisted._evicted is True
        write_session_policy_config(
            config,
            persist_sessions=False,
            # The stored threshold is irrelevant while persistence is disabled;
            # live policy normalizes it to zero.
            idle_eviction_seconds=120,
        )
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if persisted.idle_eviction_seconds == 0:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("daemon did not apply the saved session policy")
        resumed, _ack, _target = state.reply(
            identity,
            agent_id=persisted_id,
            prompt="idle resumed",
            behavior="auto",
        )
        assert resumed.status in {"starting", "running", "idle"}
        resumed_done, resumed_timeout = manager.listen(
            persisted_id,
            after_turn_count=1,
            timeout_seconds=5,
        )
        assert resumed_timeout is False
        assert resumed_done.status == "idle"
        # The original durable capability remains so the evicted agent was not
        # stranded, but live policy prevents it from being evicted again.
        assert persisted.session_dir == durable_sessions
        assert persisted.idle_eviction_seconds == 0
        assert persisted.process is not None
        assert persisted.process.poll() is None

        ephemeral_start = state.start(
            identity,
            prompt="idle ephemeral",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )
        ephemeral_id = str(ephemeral_start["agent_id"])
        ephemeral_done, ephemeral_timeout = manager.listen(
            ephemeral_id,
            after_turn_count=0,
            timeout_seconds=5,
        )
        assert ephemeral_timeout is False
        assert ephemeral_done.status == "idle"
        ephemeral = manager._get(ephemeral_id)
        assert ephemeral.session_dir is None
        assert ephemeral.idle_eviction_seconds == 0

        # Reverse both toggles. Existing persisted agents adopt the live
        # threshold; existing ephemeral agents cannot safely be retrofitted and
        # stay resident. A newly created agent gets full persistence.
        write_session_policy_config(
            config,
            persist_sessions=True,
            idle_eviction_seconds=120,
        )
        assert state.manager_for(identity) is manager
        assert persisted.session_dir == durable_sessions
        assert persisted.idle_eviction_seconds == 120
        assert ephemeral.session_dir is None
        assert ephemeral.idle_eviction_seconds == 0
        assert ephemeral.process is not None
        assert ephemeral.process.poll() is None

        new_persisted_start = state.start(
            identity,
            prompt="idle new persisted",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )
        new_persisted = manager._get(str(new_persisted_start["agent_id"]))
        assert new_persisted.session_dir == durable_sessions
        assert new_persisted.idle_eviction_seconds == 120
    finally:
        state.close()


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


def call_daemon_handler(
    state: DaemonState,
    request: dict[str, object],
    monkeypatch,
) -> dict[str, object]:
    peer_pid = os.getpid()
    monkeypatch.setattr("pi_as_mcp.daemon.STATE", state)
    handler = Mock()
    handler.peer_pid.return_value = peer_pid
    request = {
        "parent_hint": "mcp:telemetry-test",
        "parent_owner_pid": peer_pid,
        **request,
    }
    return RequestHandler._dispatch_request(handler, request)


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


def test_daemon_rejects_delegation_to_disabled_model(tmp_path: Path, monkeypatch) -> None:
    fake_pi = write_fake_pi(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"agents": {"models": {"local/example-model": {"disabled": True}}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(config))
    state = DaemonState()
    identity = ParentIdentity(scope_id="scope", owner_pid=None, label="caller")
    try:
        state.manager_for(identity)._runner.pi_bin = str(fake_pi)
        with pytest.raises(PiRpcError, match="disabled in pi-as-mcp config"):
            state.start(
                identity,
                prompt="hi",
                cwd=str(tmp_path),
                model="local/example-model",
                provider=None,
                tool_mode="none",
                include_events=False,
            )
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


def test_daemon_concurrent_idle_replies_respect_limit_and_running_reply_is_allowed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_pi = write_replyable_fake_pi(tmp_path)
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(write_config(tmp_path, {"example-model": 1})))
    monkeypatch.setenv("PI_AS_MCP_STATS_DIR", str(tmp_path / "stats"))
    state = DaemonState()
    identity = ParentIdentity(scope_id="reply-limit-scope", owner_pid=None, label="reply-limit")
    try:
        manager = state.manager_for(identity)
        manager._runner.pi_bin = str(fake_pi)

        agent_ids: list[str] = []
        for prompt in ("idle first", "idle second"):
            started = state.start(
                identity,
                prompt=prompt,
                cwd=str(tmp_path),
                model="local/example-model",
                provider=None,
                tool_mode="none",
                include_events=False,
            )
            agent_id = str(started["agent_id"])
            agent_ids.append(agent_id)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if manager.peek(agent_id).status == "idle":
                    break
                time.sleep(0.05)
            else:
                raise AssertionError(f"agent {agent_id} never went idle")

        results: dict[str, tuple[str, object]] = {}
        barrier = threading.Barrier(2)

        def resume(agent_id: str) -> None:
            barrier.wait()
            try:
                snapshot, ack, _target = state.reply(
                    identity,
                    agent_id=agent_id,
                    prompt="run",
                    behavior="auto",
                )
                results[agent_id] = ("ok", (snapshot, ack))
            except PiRpcError as exc:
                results[agent_id] = ("err", str(exc))

        threads = [threading.Thread(target=resume, args=(agent_id,)) for agent_id in agent_ids]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert all(not thread.is_alive() for thread in threads)
        assert sorted(result[0] for result in results.values()) == ["err", "ok"]
        rejection = next(result[1] for result in results.values() if result[0] == "err")
        assert "concurrency limit reached" in str(rejection)

        running_agent_id = next(
            agent_id for agent_id, result in results.items() if result[0] == "ok"
        )
        snapshot, ack, _target = state.reply(
            identity,
            agent_id=running_agent_id,
            prompt="keep going",
            behavior="auto",
        )
        assert snapshot.status == "running"
        assert ack["was_running"] is True
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


def test_start_succeeds_when_post_start_stats_write_fails(tmp_path: Path, monkeypatch) -> None:
    fake_pi = write_fake_pi(tmp_path)
    monkeypatch.setenv("PI_AS_MCP_STATS_DIR", str(tmp_path / "stats"))
    state = DaemonState()
    identity = parent_identity_from_peer(
        os.getpid(),
        parent_hint="mcp:telemetry-test",
        parent_owner_pid=os.getpid(),
    )
    try:
        manager = state.manager_for(identity)
        manager._runner.pi_bin = str(fake_pi)

        def fail_snapshot(*args, **kwargs) -> None:
            raise OSError("stats directory is read-only")

        monkeypatch.setattr(state._stats, "record_agent_snapshot", fail_snapshot)
        started = call_daemon_handler(
            state,
            {
                "command": "start",
                "prompt": "start exactly once",
                "cwd": str(tmp_path),
                "model": "local/example-model",
                "tool_mode": "none",
            },
            monkeypatch,
        )

        agents = manager.summary()
        assert len(agents) == 1
        assert agents[0]["agent_id"] == started["agent_id"]
        assert started["telemetry_warning"]["latest"]["operation"] == "agent snapshot (agent_started)"
        health = state.stats_summary()["telemetry"]
        assert health["healthy"] is False
        assert health["failure_count"] == 1
    finally:
        state.close()


def test_reply_succeeds_when_post_reply_stats_write_fails(tmp_path: Path, monkeypatch) -> None:
    fake_pi = write_replyable_fake_pi(tmp_path)
    monkeypatch.setenv("PI_AS_MCP_STATS_DIR", str(tmp_path / "stats"))
    state = DaemonState()
    identity = parent_identity_from_peer(
        os.getpid(),
        parent_hint="mcp:telemetry-test",
        parent_owner_pid=os.getpid(),
    )
    try:
        manager = state.manager_for(identity)
        manager._runner.pi_bin = str(fake_pi)
        started = state.start(
            identity,
            prompt="idle first",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )
        agent_id = str(started["agent_id"])

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if manager.peek(agent_id, include_events=False).status == "idle":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("agent did not become idle")

        def fail_snapshot(*args, **kwargs) -> None:
            raise OSError("stats directory is read-only")

        monkeypatch.setattr(state._stats, "record_agent_snapshot", fail_snapshot)
        replied = call_daemon_handler(
            state,
            {
                "command": "reply",
                "agent_id": agent_id,
                "prompt": "run",
                "behavior": "auto",
            },
            monkeypatch,
        )

        assert replied["agent_id"] == agent_id
        assert replied["reply_was_running"] is False
        assert len(manager.summary()) == 1
        assert len(manager.peek(agent_id, include_events=False).prompts) == 2
        assert replied["telemetry_warning"]["latest"]["operation"] == "agent snapshot (agent_updated)"
        assert state.stats_summary()["telemetry"]["failure_count"] >= 1
    finally:
        state.close()


def test_transcript_failure_is_visible_without_stopping_agent(tmp_path: Path, monkeypatch) -> None:
    fake_pi = write_fake_pi(tmp_path)
    monkeypatch.setenv("PI_AS_MCP_STATS_DIR", str(tmp_path / "stats"))
    state = DaemonState()
    identity = ParentIdentity(scope_id="transcript-failure", owner_pid=None, label="test")
    try:
        manager = state.manager_for(identity)
        manager._runner.pi_bin = str(fake_pi)

        def fail_transcript(*args, **kwargs) -> None:
            raise OSError("transcript directory is read-only")

        monkeypatch.setattr(state._stats, "append_transcript", fail_transcript)
        started = state.start(
            identity,
            prompt="keep working",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )

        assert len(manager.summary()) == 1
        assert started["telemetry_warning"]["latest"]["operation"] == "transcript append"
        health = state.stats_summary()["telemetry"]
        assert health["healthy"] is False
        assert health["warnings"][0]["error"].endswith("transcript directory is read-only")
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


def test_daemon_score_write_failure_is_not_reported_as_recorded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PI_AS_MCP_STATS_DIR", str(tmp_path / "stats"))
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(write_score_config(tmp_path, enabled=True)))
    state = DaemonState()
    identity = ParentIdentity(scope_id="score-failure", owner_pid=None, label="scorer")
    try:
        def fail_score(*args, **kwargs) -> None:
            raise OSError("score audit is read-only")

        monkeypatch.setattr(state._stats, "record_score", fail_score)
        with pytest.raises(OSError, match="score audit is read-only"):
            state.score_agent(
                identity,
                agent_id="agent-1",
                score=8,
                category="review",
                comment="good result",
            )
    finally:
        state.close()
