#!/usr/bin/env bash
#
# refresh-daemon.sh
#
#   1. Update (or install) the local pi-as-mcp installation:
#        - `uv sync` re-installs the editable package and its deps into .venv,
#          creating the venv on a fresh checkout.
#        - (re)links the console scripts into ~/.local/bin so `pi-agent`,
#          `pi-agent-tui`, etc. are on PATH.
#   2. Wait for the running daemon to be free of work (no in-flight turns).
#   3. Stop that daemon and force a fresh one to start on the current code.
#
# The daemon is identified by the runtime dir it serves (honouring
# PI_AS_MCP_RUNTIME_DIR), so isolated/test daemons in other namespaces are left
# alone.
#
# Usage:
#   scripts/refresh-daemon.sh [--force] [--max-wait SECONDS] [--poll SECONDS]
#                             [--bin-dir DIR] [--no-link]
#
#   --force            Do not wait; interrupt any in-flight turns immediately.
#   --max-wait N       Give up waiting after N seconds (default 0 = wait forever).
#                      On timeout the script aborts without killing the daemon.
#   --poll N           Seconds between "is it busy?" checks (default 3).
#   --bin-dir DIR      Where to link console scripts (default ~/.local/bin).
#   --no-link          Skip the ~/.local/bin symlink step.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PY="$PROJECT_DIR/.venv/bin/python"

FORCE=0
MAX_WAIT=0
POLL_INTERVAL=3
BIN_DIR="${HOME}/.local/bin"
DO_LINK=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --max-wait) MAX_WAIT="${2:?--max-wait needs a value}"; shift 2 ;;
        --poll) POLL_INTERVAL="${2:?--poll needs a value}"; shift 2 ;;
        --bin-dir) BIN_DIR="${2:?--bin-dir needs a value}"; shift 2 ;;
        --no-link) DO_LINK=0; shift ;;
        -h|--help) awk 'NR==1{next} /^[^#]/{exit} {sub(/^# ?/,""); print}' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Update / install
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' is required but not on PATH (see https://docs.astral.sh/uv/)" >&2
    exit 1
fi

echo "==> Syncing pi-as-mcp (editable install + dependencies)…"
( cd "$PROJECT_DIR" && uv --native-tls sync )

if [[ ! -x "$VENV_PY" ]]; then
    echo "error: expected venv python at $VENV_PY after 'uv sync'" >&2
    exit 1
fi

if [[ "$DO_LINK" -eq 1 ]]; then
    echo "==> Linking console scripts into $BIN_DIR…"
    mkdir -p "$BIN_DIR"
    for name in pi-agent pi-agent-tui pi-as-mcp piw pi-agent-daemon; do
        target="$PROJECT_DIR/.venv/bin/$name"
        if [[ -x "$target" ]]; then
            ln -sfn "$target" "$BIN_DIR/$name"
            echo "    $name -> $target"
        fi
    done
fi

# Resolve the daemon's runtime dir + socket exactly as the package does.
read -r RUNTIME_DIR SOCKET < <(
    "$VENV_PY" - <<'PY'
from pi_as_mcp.paths import runtime_dir, socket_path
print(runtime_dir(), socket_path())
PY
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Count agents currently running a turn. Connects directly to the socket so it
# never auto-spawns a daemon. Prints "-1" when no daemon is reachable.
active_agent_count() {
    "$VENV_PY" - "$SOCKET" <<'PY'
import json, socket, sys

sock_path = sys.argv[1]
try:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as cl:
        cl.settimeout(5)
        cl.connect(sock_path)
        cl.sendall((json.dumps({"command": "tui_summary"}) + "\n").encode("utf-8"))
        buf = b""
        while True:
            chunk = cl.recv(65536)
            if not chunk:
                break
            buf += chunk
    data = json.loads(buf.decode("utf-8"))
    agents = data.get("agents") or []
    active = sum(
        1 for a in agents if str(a.get("status", "")).lower() in {"starting", "running"}
    )
    print(active)
except (OSError, ValueError):
    print(-1)
PY
}

# Print the PID(s) of the daemon serving our runtime dir (space separated).
find_daemon_pids() {
    "$VENV_PY" - "$RUNTIME_DIR" <<'PY'
import os, sys
from pathlib import Path

target = Path(sys.argv[1]).resolve()


def runtime_dir_of(pid: str) -> Path | None:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        raw = b""
    env = {}
    for kv in raw.split(b"\0"):
        if b"=" in kv:
            key, value = kv.split(b"=", 1)
            env[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    override = env.get("PI_AS_MCP_RUNTIME_DIR")
    if override:
        return Path(override).expanduser()
    try:
        uid = os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return None
    return Path(f"/tmp/pi-as-mcp-{uid}")


def is_daemon(argv: list[str]) -> bool:
    # Match an actual worker process, not anything that merely mentions the
    # module name (this script, pgrep, an editor). Require either
    # `<python> ... pi_as_mcp.daemon` (the auto-started form) or the
    # pi-agent-daemon console script, with pi_as_mcp.daemon as a whole argv
    # element rather than a substring of some larger argument.
    if not argv:
        return False
    base = os.path.basename(argv[0])
    if "python" in base and "pi_as_mcp.daemon" in argv[1:]:
        return True
    return base == "pi-agent-daemon"


pids = []
for pid in os.listdir("/proc"):
    if not pid.isdigit() or pid == str(os.getpid()):
        continue
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        continue
    argv = [part.decode("utf-8", "replace") for part in cmdline.split(b"\0") if part]
    if not is_daemon(argv):
        continue
    rt = runtime_dir_of(pid)
    if rt is not None and rt.resolve() == target:
        pids.append(pid)

print(" ".join(pids))
PY
}

# ---------------------------------------------------------------------------
# 2. Wait for the daemon to be free of work
# ---------------------------------------------------------------------------
echo "==> Checking daemon for in-flight work (runtime: $RUNTIME_DIR)…"
waited=0
while true; do
    n="$(active_agent_count)"
    if [[ "$n" == "-1" ]]; then
        echo "    no reachable daemon — nothing to drain."
        break
    fi
    if [[ "$n" -eq 0 ]]; then
        echo "    daemon is idle (0 active turns)."
        break
    fi
    if [[ "$FORCE" -eq 1 ]]; then
        echo "    --force: interrupting ${n} active turn(s)."
        break
    fi
    if [[ "$MAX_WAIT" -gt 0 && "$waited" -ge "$MAX_WAIT" ]]; then
        echo "error: still ${n} active turn(s) after ${MAX_WAIT}s; aborting." >&2
        echo "       re-run with --force to restart anyway." >&2
        exit 1
    fi
    echo "    ${n} active turn(s) in progress; waiting… (${waited}s elapsed)"
    sleep "$POLL_INTERVAL"
    waited=$((waited + POLL_INTERVAL))
done

# ---------------------------------------------------------------------------
# 3. Stop the daemon and force a fresh one to start
# ---------------------------------------------------------------------------
echo "==> Restarting the daemon…"
PIDS="$(find_daemon_pids)"
if [[ -z "$PIDS" ]]; then
    echo "    no daemon process found for this runtime dir; will start a fresh one."
else
    for pid in $PIDS; do
        echo "    stopping daemon pid $pid (SIGTERM)…"
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in $PIDS; do
        for _ in $(seq 1 20); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.5
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "    pid $pid ignored SIGTERM; sending SIGKILL." >&2
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
fi

# A fresh daemon auto-starts on the next client connection. `summary` is cheap
# and triggers DaemonClient.ensure_daemon(), which spawns it on the current code.
echo "    starting a fresh daemon on the current code…"
( cd "$PROJECT_DIR" && "$VENV_PY" -m pi_as_mcp.cli summary >/dev/null 2>&1 ) || true

sleep 0.5
NEW_PIDS="$(find_daemon_pids)"
if [[ -n "$NEW_PIDS" ]]; then
    echo "==> Done. Daemon is back up (pid: ${NEW_PIDS// /, })."
    echo "    Any open pi-agent-tui reconnects automatically on its next poll."
else
    echo "error: daemon did not come back up; see $RUNTIME_DIR/daemon.log" >&2
    exit 1
fi
