from __future__ import annotations

import os
from pathlib import Path

DEFAULT_RUNTIME_DIR_TEMPLATE = "/tmp/pi-as-mcp-{uid}"


def runtime_dir() -> Path:
    override = os.environ.get("PI_AS_MCP_RUNTIME_DIR")
    path = Path(override).expanduser() if override else Path(DEFAULT_RUNTIME_DIR_TEMPLATE.format(uid=os.getuid()))
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass
    return path


def socket_path() -> Path:
    return runtime_dir() / "daemon.sock"


def session_dir() -> Path:
    """Directory where Pi persists per-agent session logs.

    Lives under the (ephemeral) runtime dir so sessions are scoped to the daemon's
    lifetime and reclaimed on reboot; idle workers can be evicted and resumed from
    these files instead of being kept resident.
    """
    path = runtime_dir() / "sessions"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def log_path() -> Path:
    return runtime_dir() / "daemon.log"
