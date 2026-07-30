"""Regression tests for a batch of bugs found in a full-codebase audit.

Each test pins the fixed behavior; the comment above it describes the bug.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import socket
import stat
import subprocess
import threading
import time
from pathlib import Path

import pytest

from pi_as_mcp import cli, daemon, daemon_client, paths, server, sessions
from pi_as_mcp.compat import CHECKED_REQUEST_COMMAND
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


def test_client_uses_shell_scope_unless_explicit_parent_id_is_set(
    monkeypatch,
) -> None:
    client = DaemonClient(default_scope_mode="cli-shell")
    sends: list[dict] = []

    def fake_send(payload, timeout):
        sends.append(payload)
        return [b"{}"]

    monkeypatch.setattr(client, "_send", fake_send)
    monkeypatch.delenv("PI_AGENT_PARENT_ID", raising=False)
    client.request("summary")

    monkeypatch.setenv("PI_AGENT_PARENT_ID", "shared-build")
    client.request("summary")

    first_request = sends[0]["request"]
    second_request = sends[1]["request"]
    assert first_request["parent_scope_mode"] == "cli-shell"
    assert "parent_hint" not in first_request
    assert second_request["parent_hint"] == "shared-build"
    assert "parent_scope_mode" not in second_request


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
    assert sends[0]["request"]["command"] == "delegate"


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
    assert [payload["command"] for payload in sends] == [
        "__checked_request__",
        "__checked_request__",
    ]
    assert sum(
        payload.get("request", {}).get("command") == "peek" for payload in sends
    ) == 2
    assert daemon_starts == [True]


def test_client_retries_repeated_connect_failures_with_backoff(monkeypatch) -> None:
    client = DaemonClient()
    sends: list = []
    delays: list[float] = []
    checked_attempts = 0

    def fake_send(payload, timeout):
        nonlocal checked_attempts
        sends.append(payload)
        checked_attempts += 1
        if checked_attempts < 4:
            raise daemon_client._DaemonConnectError("accept queue busy")
        return [json.dumps({"agent_id": "a1"}).encode("utf-8")]

    def fake_backoff(delay, deadline):
        delays.append(delay)
        return delay * 2

    monkeypatch.setattr(client, "_send", fake_send)
    monkeypatch.setattr(client, "start_daemon", lambda: None)
    monkeypatch.setattr(client, "_sleep_with_backoff", fake_backoff)

    assert client.request("peek", agent_id="a1") == {"agent_id": "a1"}
    assert [payload["command"] for payload in sends] == [
        CHECKED_REQUEST_COMMAND,
        CHECKED_REQUEST_COMMAND,
        CHECKED_REQUEST_COMMAND,
        CHECKED_REQUEST_COMMAND,
    ]
    assert delays == [
        daemon_client._INITIAL_RETRY_DELAY_SECONDS,
        daemon_client._INITIAL_RETRY_DELAY_SECONDS * 2,
    ]


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

    # A normal probe must not unlink even a refused socket: only the startup
    # lock holder may do that, or it can race a new listener on the same path.
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(sock_path))
    stale.close()  # bound but never listening -> connect refused
    client = DaemonClient()
    assert client._can_connect() is False
    assert sock_path.exists()
    assert (
        client._socket_state(remove_stale=True)
        is daemon_client._SocketState.ABSENT
    )
    assert not sock_path.exists(), (
        "the startup coordinator should remove a stale socket"
    )

    # Transient failure (timeout under load): the socket must survive.
    sock_path.touch()

    def timeout_connect(self, address):
        raise socket.timeout("timed out")

    monkeypatch.setattr(socket.socket, "connect", timeout_connect)
    assert client._can_connect() is False
    assert sock_path.exists(), "live daemon socket must not be unlinked on a timeout"


def test_start_daemon_does_not_replace_a_busy_listener(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PI_AS_MCP_RUNTIME_DIR", str(tmp_path))
    client = DaemonClient()
    states = iter(
        [
            daemon_client._SocketState.BUSY,
            daemon_client._SocketState.BUSY,
            daemon_client._SocketState.READY,
        ]
    )
    monkeypatch.setattr(client, "_socket_state", lambda **_kwargs: next(states))
    monkeypatch.setattr(
        client,
        "_sleep_with_backoff",
        lambda delay, _deadline: delay,
    )
    monkeypatch.setattr(
        client,
        "_spawn_daemon",
        lambda: pytest.fail("must not replace a live but busy daemon"),
    )

    client.start_daemon()


def _run_fake_coordinated_daemon(
    runtime_path: str,
    stop_event,
) -> None:
    launch_signal = Path(runtime_path) / "test-launch"
    deadline = time.monotonic() + 5
    while not launch_signal.exists():
        if time.monotonic() >= deadline:
            return
        time.sleep(0.01)
    path = os.path.join(runtime_path, "daemon.sock")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(path)
        listener.listen(128)
        listener.settimeout(0.1)
        while not stop_event.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            with connection:
                connection.settimeout(0.5)
                try:
                    payload = connection.recv(65536)
                except socket.timeout:
                    continue
                if payload:
                    request = json.loads(payload.decode("utf-8"))
                    if request["command"] == CHECKED_REQUEST_COMMAND:
                        response = {"models": []}
                    else:
                        response = {
                            "error": f"unknown command: {request['command']}",
                            "daemon_error": True,
                        }
                    connection.sendall((json.dumps(response) + "\n").encode("utf-8"))


class _ColdStartTestClient(DaemonClient):
    """Signals a test server instead of launching the real background daemon."""

    def _spawn_daemon(self) -> None:
        runtime_path = Path(os.environ["PI_AS_MCP_RUNTIME_DIR"])
        launch_log = runtime_path / "test-launches"
        fd = os.open(launch_log, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            os.write(fd, f"{os.getpid()}\n".encode())
        finally:
            os.close(fd)
        (runtime_path / "test-launch").touch()


def _run_cold_start_client(runtime_path: str, start_event, results) -> None:
    os.environ["PI_AS_MCP_RUNTIME_DIR"] = runtime_path
    if not start_event.wait(timeout=5):
        results.put(("error", "start gate timed out"))
        return
    try:
        response = _ColdStartTestClient().request("models")
    except Exception as exc:
        results.put(("error", repr(exc)))
    else:
        results.put(("ok", response))


@pytest.mark.parametrize("with_stale_socket", [False, True])
def test_multiprocess_cold_start_launches_once_and_all_clients_converge(
    tmp_path: Path,
    monkeypatch,
    with_stale_socket: bool,
) -> None:
    try:
        context = multiprocessing.get_context("spawn")
    except ValueError:
        pytest.skip("requires multiprocessing spawn")

    monkeypatch.setenv("PI_AS_MCP_RUNTIME_DIR", str(tmp_path))
    if with_stale_socket:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale:
            stale.bind(str(paths.socket_path()))

    stop_event = context.Event()
    start_event = context.Event()
    results = context.Queue()

    server_process = context.Process(
        target=_run_fake_coordinated_daemon,
        args=(str(tmp_path), stop_event),
    )
    clients = [
        context.Process(
            target=_run_cold_start_client,
            args=(str(tmp_path), start_event, results),
        )
        for _ in range(16)
    ]
    server_process.start()
    for process in clients:
        process.start()
    start_event.set()

    try:
        outcomes = [results.get(timeout=10) for _ in clients]
        for process in clients:
            process.join(timeout=5)
        assert all(not process.is_alive() for process in clients)
        assert outcomes == [("ok", {"models": []})] * len(clients)
        launches = (tmp_path / "test-launches").read_text(encoding="utf-8").splitlines()
        assert len(launches) == 1
        lock_path = paths.daemon_start_lock_path()
        assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
        assert daemon.UnixServer.request_queue_size >= len(clients)
    finally:
        stop_event.set()
        server_process.join(timeout=5)
        for process in clients:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        if server_process.is_alive():
            server_process.terminate()
            server_process.join(timeout=5)


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


def test_wait_shim_keeps_custom_runtime_without_parent_environment(
    tmp_path: Path, monkeypatch
) -> None:
    custom_runtime = tmp_path / "custom runtime's namespace"
    custom_runtime.mkdir()
    monkeypatch.setenv("PI_AS_MCP_RUNTIME_DIR", str(custom_runtime))

    fake_python = tmp_path / "fake python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import json
import os
import socket
import sys

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.connect(os.path.join(os.environ["PI_AS_MCP_RUNTIME_DIR"], "daemon.sock"))
    client.sendall((json.dumps({"argv": sys.argv[1:]}) + "\\n").encode())
    print(client.recv(65536).decode(), end="")
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    monkeypatch.setattr(server.sys, "executable", str(fake_python))

    command = server.wait_command("agent-1", after_turn_count=2)

    decoy_dir = tmp_path / "decoy"
    decoy_dir.mkdir()
    decoy = decoy_dir / "piw"
    decoy.write_text("#!/bin/sh\necho wrong-piw\nexit 99\n", encoding="utf-8")
    decoy.chmod(0o700)

    received: list[dict] = []
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(custom_runtime / "daemon.sock"))
    listener.listen()

    def serve_once() -> None:
        connection, _ = listener.accept()
        with connection:
            payload = b""
            while not payload.endswith(b"\n"):
                payload += connection.recv(65536)
            received.append(json.loads(payload))
            connection.sendall(b'{"namespace":"custom"}')

    thread = threading.Thread(target=serve_once)
    thread.start()
    try:
        child_env = os.environ.copy()
        child_env.pop("PI_AS_MCP_RUNTIME_DIR")
        child_env["PATH"] = f"{decoy_dir}{os.pathsep}{child_env['PATH']}"
        completed = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            env=child_env,
            timeout=5,
        )
    finally:
        thread.join(timeout=5)
        listener.close()

    assert not thread.is_alive()
    assert json.loads(completed.stdout) == {"namespace": "custom"}
    assert received == [
        {"argv": ["-m", "pi_as_mcp.cli", "wait", "agent-1", "-a", "2"]}
    ]


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
