from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from pi_as_mcp.daemon_discovery import daemon_pids


def _fake_process(
    proc_root: Path,
    pid: int,
    argv: list[str],
    environment: dict[str, str] | None = None,
) -> None:
    process_dir = proc_root / str(pid)
    process_dir.mkdir()
    (process_dir / "cmdline").write_bytes(
        b"\0".join(item.encode() for item in argv) + b"\0"
    )
    encoded_environment = b"\0".join(
        f"{key}={value}".encode()
        for key, value in (environment or {}).items()
    )
    (process_dir / "environ").write_bytes(encoded_environment)


def test_linux_discovery_matches_exact_argv_and_runtime_with_spaces(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    target = tmp_path / "custom runtime namespace"
    other = tmp_path / "other runtime"
    target.mkdir()
    other.mkdir()
    _fake_process(
        proc_root,
        101,
        ["/usr/bin/python3", "-m", "pi_as_mcp.daemon"],
        {"PI_AS_MCP_RUNTIME_DIR": str(target)},
    )
    _fake_process(
        proc_root,
        102,
        ["/bin/bash", "-c", "python -m pi_as_mcp.daemon"],
        {"PI_AS_MCP_RUNTIME_DIR": str(target)},
    )
    _fake_process(
        proc_root,
        103,
        ["/venv/bin/pi-agent-daemon"],
        {"PI_AS_MCP_RUNTIME_DIR": str(other)},
    )
    _fake_process(
        proc_root,
        104,
        ["/venv/bin/pi-agent-daemon"],
        {"PI_AS_MCP_RUNTIME_DIR": str(target)},
    )

    assert daemon_pids(
        target,
        target / "daemon.sock",
        platform_name="linux",
        proc_root=proc_root,
    ) == [101, 104]


def test_linux_discovery_preserves_default_runtime_matching(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _fake_process(
        proc_root,
        201,
        [sys.executable, "-m", "pi_as_mcp.daemon"],
    )

    assert daemon_pids(
        Path(f"/tmp/pi-as-mcp-{os.getuid()}"),
        Path("/unused"),
        platform_name="linux",
        proc_root=proc_root,
    ) == [201]


def _darwin_socket(response: object, *, peer_pid: int = 4321) -> Mock:
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.getsockopt.return_value = struct.pack("=i", peer_pid)
    payload = json.dumps(response).encode()
    client.recv.side_effect = [payload, b""]
    return client


def test_darwin_discovery_authenticates_exact_socket_peer(tmp_path: Path) -> None:
    runtime_path = tmp_path / "runtime with spaces"
    socket_path = runtime_path / "daemon.sock"
    client = _darwin_socket({"agents": [], "stats": {}})
    socket_factory = Mock(return_value=client)

    assert daemon_pids(
        runtime_path,
        socket_path,
        platform_name="darwin",
        socket_factory=socket_factory,
    ) == [4321]
    socket_factory.assert_called_once_with(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect.assert_called_once_with(str(socket_path))
    client.sendall.assert_called_once_with(b'{"command":"tui_summary"}\n')


def test_darwin_discovery_fails_closed_without_peer_credentials(
    tmp_path: Path,
) -> None:
    client = _darwin_socket({"agents": []})
    client.getsockopt.side_effect = OSError("LOCAL_PEERPID unavailable")

    assert daemon_pids(
        tmp_path,
        tmp_path / "daemon.sock",
        platform_name="darwin",
        socket_factory=Mock(return_value=client),
    ) == []


@pytest.mark.parametrize(
    "response",
    [
        {"error": "not a daemon", "daemon_error": True},
        {"agents": "not a list"},
        {"agents": [None]},
        "not an object",
    ],
)
def test_darwin_discovery_rejects_non_daemon_listener(
    tmp_path: Path,
    response: object,
) -> None:
    client = _darwin_socket(response)

    assert daemon_pids(
        tmp_path,
        tmp_path / "daemon.sock",
        platform_name="darwin",
        socket_factory=Mock(return_value=client),
    ) == []


def _wait_for_daemon(socket_path: Path, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(0.2)
                client.connect(str(socket_path))
                client.sendall(b'{"command":"tui_summary"}\n')
                if client.recv(65536):
                    return
        except OSError:
            time.sleep(0.05)
    raise AssertionError(f"daemon did not listen on {socket_path}")


def _running_pids(pids: set[int]) -> set[int]:
    running: set[int] = set()
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        running.add(pid)
    return running


def _stop_daemons(pids: list[int]) -> None:
    remaining = set(pids)
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    while remaining and time.monotonic() < deadline:
        remaining = _running_pids(remaining)
        if remaining:
            time.sleep(0.05)
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("uv") is None,
    reason="live refresh exercise requires Linux /proc and uv",
)
def test_refresh_script_restarts_daemon_in_runtime_path_with_spaces(
    tmp_path: Path,
) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    runtime_path = tmp_path / "custom runtime namespace"
    environment = os.environ.copy()
    environment["PI_AS_MCP_RUNTIME_DIR"] = str(runtime_path)
    old_daemon = subprocess.Popen(
        [sys.executable, "-m", "pi_as_mcp.daemon"],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    socket_path = runtime_path / "daemon.sock"
    new_pids: list[int] = []
    try:
        _wait_for_daemon(socket_path)
        reaper = threading.Thread(target=old_daemon.wait, daemon=True)
        reaper.start()

        completed = subprocess.run(
            [
                str(project_dir / "scripts" / "refresh-daemon.sh"),
                "--force",
                "--no-link",
            ],
            cwd=project_dir,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
        )

        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert f"runtime: {runtime_path}" in completed.stdout
        reaper.join(timeout=5)
        assert not reaper.is_alive()
        new_pids = daemon_pids(runtime_path, socket_path)
        assert len(new_pids) == 1
        assert new_pids[0] != old_daemon.pid
    finally:
        if old_daemon.poll() is None:
            old_daemon.terminate()
            old_daemon.wait(timeout=5)
        _stop_daemons(new_pids or daemon_pids(runtime_path, socket_path))
