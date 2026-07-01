from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from typing import Any

from pi_as_mcp.paths import log_path, socket_path


class DaemonClientError(RuntimeError):
    pass


class _DaemonConnectError(OSError):
    """Connect-phase failure: the request was never delivered, so it is safe
    to spawn the daemon and re-send. Post-connect failures must NOT be retried
    (the daemon may already be executing a non-idempotent command)."""


class DaemonClient:
    def __init__(self, *, default_parent_hint: str | None = None, parent_owner_pid: int | None = None) -> None:
        self.default_parent_hint = default_parent_hint
        self.parent_owner_pid = parent_owner_pid

    def request(self, command: str, *, request_timeout_seconds: int = 30, **params: Any) -> dict[str, Any]:
        payload = {"command": command, **params}
        parent_hint = os.environ.get("PI_AGENT_PARENT_ID") or self.default_parent_hint
        if parent_hint:
            payload["parent_hint"] = parent_hint
        if self.parent_owner_pid is not None:
            payload["parent_owner_pid"] = self.parent_owner_pid

        # Happy path: try the real connection directly instead of probing with a
        # throwaway socket first. Only spawn+wait for the daemon when the connect
        # itself fails (request never delivered), then retry once. A failure
        # after connect is NOT retried: commands like delegate/reply are not
        # idempotent and the daemon may already be executing the first send.
        try:
            chunks = self._send(payload, request_timeout_seconds)
        except _DaemonConnectError:
            self.start_daemon()
            try:
                chunks = self._send(payload, request_timeout_seconds)
            except OSError as exc:
                raise DaemonClientError(f"daemon request failed: {exc}") from exc
        except socket.timeout as exc:
            raise DaemonClientError(
                f"daemon did not respond within {request_timeout_seconds}s"
            ) from exc
        except OSError as exc:
            raise DaemonClientError(f"daemon request failed: {exc}") from exc

        if not chunks:
            raise DaemonClientError("daemon returned no response")
        response = json.loads(b"".join(chunks).decode("utf-8"))
        if not isinstance(response, dict):
            raise DaemonClientError("daemon returned non-object response")
        # Only a daemon-level failure envelope is an error. A successful
        # snapshot legitimately carries a non-empty "error" field (the agent's
        # own provider error) and must be returned, not raised.
        if response.get("error") and (response.get("daemon_error") or set(response) == {"error"}):
            raise DaemonClientError(str(response["error"]))
        return response

    def _send(self, payload: dict[str, Any], request_timeout_seconds: int) -> list[bytes]:
        path = socket_path()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(request_timeout_seconds)
            try:
                client.connect(str(path))
            except OSError as exc:
                raise _DaemonConnectError(str(exc)) from exc
            client.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            chunks: list[bytes] = []
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        return chunks

    def ensure_daemon(self) -> None:
        if self._can_connect():
            return
        self.start_daemon()

    def start_daemon(self) -> None:
        log_file = log_path().open("ab")
        subprocess.Popen(
            [sys.executable, "-m", "pi_as_mcp.daemon"],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            close_fds=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self._can_connect():
                return
            time.sleep(0.05)
        raise DaemonClientError(f"daemon did not start; see {log_path()}")

    def _can_connect(self) -> bool:
        path = socket_path()
        if not path.exists():
            return False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(0.2)
                client.connect(str(path))
            return True
        except ConnectionRefusedError:
            # No listener behind the file: a stale socket from a dead daemon.
            # Remove it so the next daemon can bind.
            try:
                path.unlink()
            except OSError:
                pass
            return False
        except OSError:
            # Transient failure (connect timeout under load, unlink race): the
            # daemon may well be alive — never delete its socket here, or every
            # live agent it owns becomes unreachable.
            return False
