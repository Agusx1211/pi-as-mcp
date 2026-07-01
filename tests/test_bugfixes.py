"""Regression tests for a batch of bugs found in a full-codebase audit.

Each test pins the fixed behavior; the comment above it describes the bug.
"""

from __future__ import annotations

import json
import os
import socket
import stat
import threading
import time
from pathlib import Path

import pytest

from pi_as_mcp import cli, daemon_client, paths, server, sessions
from pi_as_mcp.config import parse_app_config
from pi_as_mcp.config_tui import ConfigDraft
from pi_as_mcp.daemon_client import DaemonClient, DaemonClientError
from pi_as_mcp.pi_rpc import CatalogModel, PiRpcError
from pi_as_mcp.sessions import SessionManager, usage_to_json
from pi_as_mcp.tui import format_percent


def write_fake_pi(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "fake-pi"
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


FAKE_PI_ECHO = """#!/usr/bin/env python3
import json
import sys

if "--list-models" in sys.argv:
    print("provider   model                    context")
    print("local  example-model  128K")
    raise SystemExit(0)

for line in sys.stdin:
    request = json.loads(line)
    if request.get("type") == "abort":
        break
    text = request.get("message", "")
    print(json.dumps({"id": request["id"], "type": "response", "command": "prompt", "success": True}), flush=True)
    print(json.dumps({"type": "agent_start"}), flush=True)
    message = {"role": "assistant", "content": [{"type": "text", "text": "echo:" + text}]}
    print(json.dumps({"type": "message_end", "message": message}), flush=True)
    print(json.dumps({"type": "agent_end", "messages": [message]}), flush=True)
"""


# --- sessions: manager.close must leave listeners a terminal status ---------


def test_manager_close_unblocks_listeners_with_terminal_status(tmp_path: Path) -> None:
    # close(reason="parent closed") used to set the *reason string* as the
    # session status; listeners only return on {error,timeout,stopped,exited},
    # so a piw blocked on a closed parent's agent hung until its full timeout.
    fake_pi = write_fake_pi(tmp_path, FAKE_PI_ECHO)
    manager = SessionManager()
    manager._runner.pi_bin = str(fake_pi)

    started = manager.start(
        prompt="one",
        cwd=str(tmp_path),
        model="local/example-model",
        provider=None,
        tool_mode="none",
        include_events=False,
    )
    session = manager._get(started.agent_id)

    results: list = []

    def listen() -> None:
        results.append(session.listen(after_turn_count=5, timeout_seconds=30))

    thread = threading.Thread(target=listen)
    thread.start()
    time.sleep(0.3)
    manager.close(reason="parent closed")
    thread.join(timeout=5)

    assert not thread.is_alive(), "listener still blocked after manager.close"
    snapshot, timed_out = results[0]
    assert timed_out is False
    assert snapshot.status == "stopped"


# --- sessions: failed initial send must not leak the spawned worker ---------


def test_failed_first_prompt_kills_spawned_worker(tmp_path: Path, monkeypatch) -> None:
    # A pi that starts but never speaks RPC used to leak: send() raised out of
    # the constructor before the session was registered anywhere, so nothing
    # could ever terminate the live subprocess or stop its watchdog.
    pid_file = tmp_path / "worker.pid"
    fake_pi = write_fake_pi(
        tmp_path,
        f"""#!/usr/bin/env python3
import os
import sys
import time

if "--list-models" in sys.argv:
    print("provider   model                    context")
    print("local  example-model  128K")
    raise SystemExit(0)

open({str(pid_file)!r}, "w").write(str(os.getpid()))
time.sleep(120)  # never acknowledge the prompt
""",
    )
    monkeypatch.setattr(sessions, "PROMPT_ACK_TIMEOUT_SECONDS", 1)
    manager = SessionManager()
    manager._runner.pi_bin = str(fake_pi)

    with pytest.raises(PiRpcError):
        manager.start(
            prompt="one",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not pid_file.exists():
            time.sleep(0.05)
            continue
        pid = int(pid_file.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return  # worker was cleaned up
        time.sleep(0.05)
    pytest.fail("orphaned Pi worker still alive after failed start")


# --- sessions: prompt ack must fail fast when the worker dies ---------------


def test_prompt_ack_fails_fast_when_worker_exits(tmp_path: Path) -> None:
    # The ack waiter only checked for the response; a worker that died before
    # acknowledging left send() blocked for the full 30s ack timeout with a
    # misleading "timed out" error instead of the real failure.
    fake_pi = write_fake_pi(
        tmp_path,
        """#!/usr/bin/env python3
import sys

if "--list-models" in sys.argv:
    print("provider   model                    context")
    print("local  example-model  128K")
    raise SystemExit(0)

sys.stdin.readline()
print("boom: provider auth failed", file=sys.stderr, flush=True)
raise SystemExit(3)
""",
    )
    manager = SessionManager()
    manager._runner.pi_bin = str(fake_pi)

    started = time.monotonic()
    with pytest.raises(PiRpcError) as excinfo:
        manager.start(
            prompt="one",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )
    assert time.monotonic() - started < 10, "send blocked instead of failing fast"
    assert "terminated" in str(excinfo.value)


# --- sessions: manager closed mid-spawn must not adopt the new session ------


def test_start_on_closed_manager_raises_and_cleans_up(tmp_path: Path) -> None:
    # The daemon reaper can close a manager while a start is spawning its
    # worker; registering into the closed manager leaked an invisible agent.
    fake_pi = write_fake_pi(tmp_path, FAKE_PI_ECHO)
    manager = SessionManager()
    manager._runner.pi_bin = str(fake_pi)
    manager.close()

    with pytest.raises(PiRpcError):
        manager.start(
            prompt="one",
            cwd=str(tmp_path),
            model="local/example-model",
            provider=None,
            tool_mode="none",
            include_events=False,
        )
    assert manager.is_empty()


# --- usage: sub-1% context readings must not be rescaled ---------------------


def test_usage_to_json_keeps_computed_sub_percent_values() -> None:
    # 1311 of 262144 tokens is 0.5%; the fraction heuristic used to rescale the
    # locally computed percentage to 50%.
    data = usage_to_json(
        {"context_used_tokens": 1311, "context_limit_tokens": 262144},
        elapsed_seconds=1.0,
    )
    assert data["context_percent"] == 0.5


def test_usage_to_json_still_normalizes_upstream_fractions() -> None:
    data = usage_to_json({"context_percent": 0.5}, elapsed_seconds=1.0)
    assert data["context_percent"] == 50


def test_format_percent_does_not_rescale_small_values() -> None:
    assert format_percent(0.5) == "0.5%"
    assert format_percent(42) == "42.0%"


# --- daemon client: non-idempotent requests must not be resent --------------


def test_client_does_not_resend_after_post_connect_failure(monkeypatch) -> None:
    # A recv timeout (socket.timeout is an OSError) used to trigger
    # start_daemon() plus a blind re-send, duplicating delegate/reply commands.
    client = DaemonClient()
    sends: list = []

    def fake_send(payload, timeout):
        sends.append(payload)
        raise socket.timeout("timed out")

    monkeypatch.setattr(client, "_send", fake_send)
    monkeypatch.setattr(client, "start_daemon", lambda: pytest.fail("must not spawn a daemon"))

    with pytest.raises(DaemonClientError):
        client.request("delegate", prompt="x")
    assert len(sends) == 1


def test_client_retries_once_on_connect_failure(monkeypatch) -> None:
    client = DaemonClient()
    sends: list = []
    daemon_starts: list = []

    def fake_send(payload, timeout):
        sends.append(payload)
        if len(sends) == 1:
            raise daemon_client._DaemonConnectError("connection refused")
        return [json.dumps({"agent_id": "a1"}).encode("utf-8")]

    monkeypatch.setattr(client, "_send", fake_send)
    monkeypatch.setattr(client, "start_daemon", lambda: daemon_starts.append(True))

    assert client.request("peek", agent_id="a1") == {"agent_id": "a1"}
    assert len(sends) == 2
    assert daemon_starts == [True]


# --- daemon client: agent errors must not read as daemon failures -----------


def test_client_passes_through_snapshot_with_agent_error(monkeypatch) -> None:
    # A snapshot whose agent hit a provider error carries a non-empty "error"
    # field; the client used to raise on it, making errored agents impossible
    # to peek/listen/stop.
    client = DaemonClient()
    snapshot = {"agent_id": "a1", "status": "error", "error": "rate limited"}
    monkeypatch.setattr(
        client, "_send", lambda payload, timeout: [json.dumps(snapshot).encode("utf-8")]
    )
    assert client.request("peek", agent_id="a1") == snapshot


def test_client_raises_on_daemon_error_envelopes(monkeypatch) -> None:
    client = DaemonClient()
    for envelope in (
        {"error": "unknown agent_id: a1", "daemon_error": True},
        {"error": "legacy failure"},  # old daemons: bare single-key envelope
    ):
        monkeypatch.setattr(
            client, "_send", lambda payload, timeout, envelope=envelope: [json.dumps(envelope).encode("utf-8")]
        )
        with pytest.raises(DaemonClientError):
            client.request("peek", agent_id="a1")


# --- daemon client: never unlink the socket of a live-but-busy daemon -------


def test_can_connect_preserves_socket_on_transient_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PI_AS_MCP_RUNTIME_DIR", str(tmp_path))
    sock_path = paths.socket_path()

    # Stale socket (no listener): probe fails with ECONNREFUSED and unlinks.
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(sock_path))
    stale.close()  # bound but never listening -> connect refused
    client = DaemonClient()
    assert client._can_connect() is False
    assert not sock_path.exists(), "stale socket should be removed"

    # Transient failure (timeout under load): the socket must survive.
    sock_path.touch()

    def timeout_connect(self, address):
        raise socket.timeout("timed out")

    monkeypatch.setattr(socket.socket, "connect", timeout_connect)
    assert client._can_connect() is False
    assert sock_path.exists(), "live daemon socket must not be unlinked on a timeout"


# --- cli: negative wait timeout must error, not wait forever ----------------


def test_wait_for_agent_rejects_negative_timeout() -> None:
    with pytest.raises(DaemonClientError):
        cli.wait_for_agent(
            DaemonClient(), agent_id="a1", after_turn_count=0, timeout_seconds=-5
        )


# --- server: the piw shim must always end up executable ---------------------


def test_ensure_wait_shim_heals_non_executable_shim(tmp_path: Path, monkeypatch) -> None:
    # A crash between write_text and chmod used to leave a shim whose content
    # matched forever, so the missing exec bit was never repaired.
    monkeypatch.setattr(server, "runtime_dir", lambda: tmp_path)
    shim = Path(server.ensure_wait_shim())
    assert shim.stat().st_mode & stat.S_IXUSR

    shim.chmod(0o600)  # simulate the crashed half-written state
    shim = Path(server.ensure_wait_shim())
    assert shim.stat().st_mode & stat.S_IXUSR


# --- paths: refuse a runtime dir we do not own -------------------------------


def test_runtime_dir_rejects_symlink(tmp_path: Path, monkeypatch) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    monkeypatch.setenv("PI_AS_MCP_RUNTIME_DIR", str(link))
    with pytest.raises(RuntimeError):
        paths.runtime_dir()


# --- config TUI: bare model names containing "/" are not orphans ------------


def test_slash_bearing_bare_model_key_is_not_an_orphan(tmp_path: Path) -> None:
    # A bare model name like "org/model" used to be misread as a full
    # provider/model ref, flagged as an orphan, and duplicated on save.
    catalog = [CatalogModel("local", "org/model", "128K", "8K")]
    raw = {"agents": {"models": {"org/model": {"limit": 3}}}}
    config = parse_app_config(raw, path=tmp_path / "config.json")
    draft = ConfigDraft.from_sources(
        raw=raw, config=config, catalog=catalog, enabled_refs={"local/org/model"}
    )
    assert draft.orphans == {}
    payload = draft.to_payload()
    assert payload["agents"]["models"] == {"local/org/model": {"limit": 3}}
