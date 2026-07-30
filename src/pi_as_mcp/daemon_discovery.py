from __future__ import annotations

import json
import os
import socket
import sys
from collections.abc import Callable
from pathlib import Path

from pi_as_mcp.pi_rpc import PiRpcError
from pi_as_mcp.unix_socket import unix_socket_peer_pid

SocketFactory = Callable[[int, int], socket.socket]


def _is_daemon_argv(argv: list[str]) -> bool:
    """Match the daemon executable without substring-based process matching."""
    if not argv:
        return False
    executable = os.path.basename(argv[0])
    if "python" in executable and "pi_as_mcp.daemon" in argv[1:]:
        return True
    return executable == "pi-agent-daemon"


def _linux_runtime_dir(process_dir: Path) -> Path | None:
    try:
        raw_environment = (process_dir / "environ").read_bytes()
    except OSError:
        raw_environment = b""
    environment: dict[str, str] = {}
    for item in raw_environment.split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        environment[key.decode("utf-8", "replace")] = value.decode(
            "utf-8", "replace"
        )
    override = environment.get("PI_AS_MCP_RUNTIME_DIR")
    if override:
        return Path(override).expanduser()
    try:
        uid = process_dir.stat().st_uid
    except OSError:
        return None
    return Path(f"/tmp/pi-as-mcp-{uid}")


def _linux_daemon_pids(runtime_path: Path, proc_root: Path) -> list[int]:
    target = runtime_path.resolve()
    pids: list[int] = []
    try:
        process_dirs = list(proc_root.iterdir())
    except OSError:
        return pids
    for process_dir in process_dirs:
        if not process_dir.name.isdigit() or process_dir.name == str(os.getpid()):
            continue
        try:
            raw_command = (process_dir / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [
            item.decode("utf-8", "replace")
            for item in raw_command.split(b"\0")
            if item
        ]
        if not _is_daemon_argv(argv):
            continue
        process_runtime = _linux_runtime_dir(process_dir)
        if process_runtime is not None and process_runtime.resolve() == target:
            pids.append(int(process_dir.name))
    return sorted(pids)


def _darwin_daemon_pid(
    socket_path: Path,
    socket_factory: SocketFactory,
) -> int | None:
    """Authenticate the process serving the exact runtime socket.

    Asking the kernel for the connected peer avoids unsafe `pgrep`/`ps`
    substring matching. The passive summary probe additionally ensures the peer
    speaks the daemon protocol before its PID is returned for termination.
    """
    try:
        with socket_factory(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(5)
            client.connect(str(socket_path))
            peer_pid = unix_socket_peer_pid(client, platform_name="darwin")
            client.sendall(b'{"command":"tui_summary"}\n')
            chunks: list[bytes] = []
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
    except (OSError, PiRpcError):
        return None

    try:
        response = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(response, dict)
        or response.get("daemon_error")
        or not isinstance(response.get("agents"), list)
        or not all(isinstance(agent, dict) for agent in response["agents"])
    ):
        return None
    return peer_pid


def daemon_pids(
    runtime_path: Path,
    socket_path: Path,
    *,
    platform_name: str | None = None,
    proc_root: Path = Path("/proc"),
    socket_factory: SocketFactory = socket.socket,
) -> list[int]:
    """Return daemon PIDs scoped to one exact runtime namespace."""
    current_platform = platform_name or sys.platform
    if current_platform.startswith("linux"):
        return _linux_daemon_pids(runtime_path, proc_root)
    if current_platform == "darwin":
        pid = _darwin_daemon_pid(socket_path, socket_factory)
        return [] if pid is None else [pid]
    raise RuntimeError(
        f"refresh daemon discovery is unsupported on {current_platform}"
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: python -m pi_as_mcp.daemon_discovery RUNTIME_DIR SOCKET"
        )
    print(
        " ".join(
            str(pid)
            for pid in daemon_pids(Path(sys.argv[1]), Path(sys.argv[2]))
        )
    )


if __name__ == "__main__":
    main()
