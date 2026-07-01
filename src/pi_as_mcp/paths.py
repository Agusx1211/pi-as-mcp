from __future__ import annotations

import os
import stat as stat_module
from pathlib import Path

DEFAULT_RUNTIME_DIR_TEMPLATE = "/tmp/pi-as-mcp-{uid}"
# Durable session transcript store. Lives under the stats dir (~/.pi-as-mcp), not
# the ephemeral runtime dir, so full agent transcripts survive reboots / /tmp
# cleanup. Keep in sync with stats.DEFAULT_STATS_DIR ("~/.pi-as-mcp").
DEFAULT_SESSION_DIR = "~/.pi-as-mcp/sessions"
SESSION_DIR_ENV = "PI_AS_MCP_SESSION_DIR"


def runtime_dir() -> Path:
    override = os.environ.get("PI_AS_MCP_RUNTIME_DIR")
    path = Path(override).expanduser() if override else Path(DEFAULT_RUNTIME_DIR_TEMPLATE.format(uid=os.getuid()))
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    # The default lives in world-writable /tmp: refuse a pre-existing entry
    # owned by another user (or a symlink), which could let them substitute or
    # intercept the daemon socket placed inside.
    info = path.lstat()
    if not stat_module.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError(
            f"refusing to use runtime dir {path}: not a directory owned by uid {os.getuid()}"
        )
    try:
        path.chmod(0o700)
    except OSError:
        pass
    return path


def socket_path() -> Path:
    return runtime_dir() / "daemon.sock"


def session_dir() -> Path:
    """Directory where Pi persists per-agent session logs.

    Durable on purpose: lives under the stats dir (``~/.pi-as-mcp/sessions`` by
    default), NOT the ephemeral runtime dir, so full agent transcripts survive
    reboots and ``/tmp`` cleanup. Idle workers are still evicted and resumed from
    these files instead of being kept resident; eviction/resume read from here.

    Override with ``PI_AS_MCP_SESSION_DIR`` (matches the ``PI_AS_MCP_RUNTIME_DIR``
    style). The path is created 0o700 on first use.
    """
    override = os.environ.get(SESSION_DIR_ENV)
    path = Path(override).expanduser() if override else Path(DEFAULT_SESSION_DIR).expanduser()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def log_path() -> Path:
    return runtime_dir() / "daemon.log"
