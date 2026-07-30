from __future__ import annotations

import enum
import fcntl
import json
import os
import random
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pi_as_mcp.compat import (
    CHECKED_REQUEST_COMMAND,
    COMPAT_COMMAND,
    DAEMON_BUILD_ID,
    DAEMON_PROTOCOL_VERSION,
    compatibility_identity,
)
from pi_as_mcp.paths import daemon_start_lock_path, log_path, socket_path


_DAEMON_START_TIMEOUT_SECONDS = 5.0
_CONNECT_RETRY_TIMEOUT_SECONDS = 5.0
_CONNECT_ATTEMPT_TIMEOUT_SECONDS = 0.25
_INITIAL_RETRY_DELAY_SECONDS = 0.01
_MAX_RETRY_DELAY_SECONDS = 0.2


class DaemonClientError(RuntimeError):
    pass


class _DaemonConnectError(OSError):
    """Connect-phase failure: the request was never delivered, so it is safe
    to spawn the daemon and re-send. Post-connect failures must NOT be retried
    (the daemon may already be executing a non-idempotent command)."""


class _SocketState(enum.Enum):
    READY = "ready"
    ABSENT = "absent"
    BUSY = "busy"


class DaemonClient:
    def __init__(
        self,
        *,
        default_parent_hint: str | None = None,
        parent_owner_pid: int | None = None,
        default_scope_mode: str | None = None,
    ) -> None:
        self.default_parent_hint = default_parent_hint
        self.parent_owner_pid = parent_owner_pid
        self.default_scope_mode = default_scope_mode
        self._compatible_socket: tuple[int, int] | None = None

    def request(self, command: str, *, request_timeout_seconds: int = 30, **params: Any) -> dict[str, Any]:
        payload = {"command": command, **params}
        parent_hint = os.environ.get("PI_AGENT_PARENT_ID") or self.default_parent_hint
        if parent_hint:
            payload["parent_hint"] = parent_hint
        elif self.default_scope_mode:
            payload["parent_scope_mode"] = self.default_scope_mode
        if self.parent_owner_pid is not None:
            payload["parent_owner_pid"] = self.parent_owner_pid

        self._ensure_compatible(request_timeout_seconds)
        checked_payload = {
            "command": CHECKED_REQUEST_COMMAND,
            **compatibility_identity(),
            "request": payload,
        }

        # Happy path: try the real connection directly instead of probing with a
        # throwaway socket first. Only spawn+wait for the daemon when the connect
        # itself fails (request never delivered), then re-check the replacement
        # daemon and retry connect with bounded backoff. A failure after connect
        # is NOT retried: commands like delegate/reply are not idempotent and may
        # already be executing.
        try:
            chunks = self._send(checked_payload, request_timeout_seconds)
        except _DaemonConnectError:
            self._compatible_socket = None
            self.start_daemon()
            self._ensure_compatible(request_timeout_seconds)
            try:
                chunks = self._send_with_connect_retries(
                    checked_payload,
                    request_timeout_seconds,
                )
            except _DaemonConnectError as exc:
                raise DaemonClientError(f"daemon request failed: {exc}") from exc
            except socket.timeout as exc:
                raise DaemonClientError(
                    f"daemon did not respond within {request_timeout_seconds}s"
                ) from exc
            except OSError as exc:
                raise DaemonClientError(f"daemon request failed: {exc}") from exc
        except socket.timeout as exc:
            raise DaemonClientError(
                f"daemon did not respond within {request_timeout_seconds}s"
            ) from exc
        except OSError as exc:
            raise DaemonClientError(f"daemon request failed: {exc}") from exc

        response = self._decode_response(chunks)
        if response.get("compatibility_error"):
            raise DaemonClientError(self._compatibility_error(response))
        if self._is_legacy_unknown_command(response, CHECKED_REQUEST_COMMAND):
            raise DaemonClientError(self._legacy_error())
        # Only a daemon-level failure envelope is an error. A successful
        # snapshot legitimately carries a non-empty "error" field (the agent's
        # own provider error) and must be returned, not raised.
        if response.get("error") and (response.get("daemon_error") or set(response) == {"error"}):
            raise DaemonClientError(str(response["error"]))
        return response

    def _ensure_compatible(self, request_timeout_seconds: int) -> None:
        socket_identity = self._socket_identity()
        if socket_identity is not None and socket_identity == self._compatible_socket:
            return

        self._compatible_socket = None
        for _attempt in range(2):
            before = self._socket_identity()
            try:
                chunks = self._send({"command": COMPAT_COMMAND}, request_timeout_seconds)
            except _DaemonConnectError:
                self.start_daemon()
                try:
                    chunks = self._send_with_connect_retries(
                        {"command": COMPAT_COMMAND},
                        request_timeout_seconds,
                    )
                except _DaemonConnectError as exc:
                    raise DaemonClientError(
                        f"daemon compatibility check failed: {exc}"
                    ) from exc
                except socket.timeout as exc:
                    raise DaemonClientError(
                        "daemon compatibility check timed out after "
                        f"{request_timeout_seconds}s"
                    ) from exc
                except OSError as exc:
                    raise DaemonClientError(
                        f"daemon compatibility check failed: {exc}"
                    ) from exc
            except socket.timeout as exc:
                raise DaemonClientError(
                    f"daemon compatibility check timed out after {request_timeout_seconds}s"
                ) from exc
            except OSError as exc:
                raise DaemonClientError(f"daemon compatibility check failed: {exc}") from exc

            response = self._decode_response(chunks)
            if self._is_legacy_unknown_command(response, COMPAT_COMMAND):
                raise DaemonClientError(self._legacy_error())
            if response.get("daemon_error"):
                raise DaemonClientError(
                    f"daemon compatibility check failed: {response.get('error') or 'unknown error'}"
                )
            if (
                response.get("protocol_version") != DAEMON_PROTOCOL_VERSION
                or response.get("build_id") != DAEMON_BUILD_ID
            ):
                raise DaemonClientError(self._compatibility_error(response))

            after = self._socket_identity()
            if after is None or (before is not None and before != after):
                # The socket changed during the probe. Do not cache a result for
                # a daemon that may no longer own the pathname.
                continue
            self._compatible_socket = after
            return
        raise DaemonClientError(
            "daemon socket was replaced during compatibility checks; no request was executed"
        )

    @staticmethod
    def _decode_response(chunks: list[bytes]) -> dict[str, Any]:
        if not chunks:
            raise DaemonClientError("daemon returned no response")
        try:
            response = json.loads(b"".join(chunks).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DaemonClientError("daemon returned an invalid response") from exc
        if not isinstance(response, dict):
            raise DaemonClientError("daemon returned non-object response")
        return response

    @staticmethod
    def _is_legacy_unknown_command(response: dict[str, Any], command: str) -> bool:
        error = str(response.get("error") or "")
        return bool(response.get("daemon_error") or set(response) == {"error"}) and (
            "unknown command" in error and command in error
        )

    @staticmethod
    def _socket_identity(path: Path | None = None) -> tuple[int, int] | None:
        target = path or socket_path()
        try:
            stat = target.stat()
        except OSError:
            return None
        return stat.st_dev, stat.st_ino

    @staticmethod
    def _refresh_hint() -> str:
        checkout_script = Path(__file__).resolve().parents[2] / "scripts" / "refresh-daemon.sh"
        if checkout_script.is_file():
            return str(checkout_script)
        return "scripts/refresh-daemon.sh"

    def _legacy_error(self) -> str:
        return (
            "the running pi-as-mcp daemon predates compatibility checks; no request "
            f"was executed. Refresh it with {self._refresh_hint()} after checking "
            "that existing agents may be restarted"
        )

    def _compatibility_error(self, daemon: dict[str, Any]) -> str:
        daemon_protocol = daemon.get("protocol_version", "unknown")
        daemon_build = daemon.get("build_id", "unknown")
        daemon_version = daemon.get("package_version", "unknown")
        return (
            "pi-as-mcp client/daemon mismatch; no request was executed "
            f"(client protocol={DAEMON_PROTOCOL_VERSION}, build={DAEMON_BUILD_ID}; "
            f"daemon protocol={daemon_protocol}, build={daemon_build}, "
            f"version={daemon_version}). Existing agents were left untouched. "
            f"Refresh the daemon with {self._refresh_hint()}; the script waits for "
            "active turns unless --force is explicitly supplied"
        )

    def _send(self, payload: dict[str, Any], request_timeout_seconds: int) -> list[bytes]:
        path = socket_path()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            # A saturated Unix accept queue can make connect block. Bound each
            # attempt independently, then restore the caller's response timeout
            # after connection. Retrying is safe only before connect succeeds.
            client.settimeout(
                min(request_timeout_seconds, _CONNECT_ATTEMPT_TIMEOUT_SECONDS)
            )
            try:
                client.connect(str(path))
            except OSError as exc:
                raise _DaemonConnectError(str(exc)) from exc
            client.settimeout(request_timeout_seconds)
            client.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            chunks: list[bytes] = []
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        return chunks

    def _send_with_connect_retries(
        self,
        payload: dict[str, Any],
        request_timeout_seconds: int,
    ) -> list[bytes]:
        deadline = time.monotonic() + _CONNECT_RETRY_TIMEOUT_SECONDS
        delay = _INITIAL_RETRY_DELAY_SECONDS
        while True:
            try:
                return self._send(payload, request_timeout_seconds)
            except _DaemonConnectError:
                if time.monotonic() >= deadline:
                    raise
                delay = self._sleep_with_backoff(delay, deadline)

    def ensure_daemon(self) -> None:
        if self._can_connect():
            return
        self.start_daemon()

    def start_daemon(self) -> None:
        deadline = time.monotonic() + _DAEMON_START_TIMEOUT_SECONDS
        lock_fd = self._open_start_lock()
        locked = False
        delay = _INITIAL_RETRY_DELAY_SECONDS
        try:
            while True:
                state = self._socket_state()
                if state is _SocketState.READY:
                    return
                if state is _SocketState.ABSENT:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        locked = True
                        break
                # BUSY means a socket exists but its accept queue is temporarily
                # unavailable. Never launch a replacement that could orphan a
                # live daemon's socket; keep probing within the bounded window.
                if time.monotonic() >= deadline:
                    raise DaemonClientError(
                        f"daemon socket remained unavailable; see {log_path()}"
                    )
                delay = self._sleep_with_backoff(delay, deadline)

            # Another lock holder may have completed startup just before this
            # process acquired the lock. Re-check before touching the socket.
            state = self._socket_state(remove_stale=True)
            if state is _SocketState.READY:
                return
            if state is _SocketState.BUSY:
                if self._wait_until_ready(deadline):
                    return
                raise DaemonClientError(
                    f"daemon socket remained unavailable; see {log_path()}"
                )

            self._spawn_daemon()
            if self._wait_until_ready(deadline):
                return
            raise DaemonClientError(f"daemon did not start; see {log_path()}")
        finally:
            if locked:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def _spawn_daemon(self) -> None:
        with log_path().open("ab") as log_file:
            subprocess.Popen(
                [sys.executable, "-m", "pi_as_mcp.daemon"],
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=log_file,
                close_fds=True,
                start_new_session=True,
            )

    def _wait_until_ready(self, deadline: float) -> bool:
        delay = _INITIAL_RETRY_DELAY_SECONDS
        while time.monotonic() < deadline:
            if self._socket_state() is _SocketState.READY:
                return True
            delay = self._sleep_with_backoff(delay, deadline)
        return False

    def _can_connect(self) -> bool:
        return self._socket_state() is _SocketState.READY

    def _socket_state(self, *, remove_stale: bool = False) -> _SocketState:
        path = socket_path()
        if not path.exists():
            return _SocketState.ABSENT
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(0.2)
                client.connect(str(path))
            return _SocketState.READY
        except ConnectionRefusedError:
            # No listener behind the file: a stale socket from a dead daemon.
            # Only the startup lock holder may remove it; otherwise a contender
            # can unlink a new listener through a stale-probe race.
            if remove_stale:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    raise DaemonClientError(
                        f"could not remove stale daemon socket {path}: {exc}"
                    ) from exc
            return _SocketState.ABSENT
        except FileNotFoundError:
            return _SocketState.ABSENT
        except OSError:
            # Transient failure (connect timeout under load, unlink race): the
            # daemon may well be alive — never delete its socket here, or every
            # live agent it owns becomes unreachable.
            return _SocketState.BUSY

    def _open_start_lock(self) -> int:
        path = daemon_start_lock_path()
        flags = os.O_CREAT | os.O_RDWR
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            lock_fd = os.open(path, flags, 0o600)
        except OSError as exc:
            raise DaemonClientError(
                f"could not open daemon startup lock {path}: {exc}"
            ) from exc
        try:
            info = os.fstat(lock_fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise DaemonClientError(
                    f"refusing daemon startup lock {path}: "
                    f"not a regular file owned by uid {os.getuid()}"
                )
            os.fchmod(lock_fd, 0o600)
            return lock_fd
        except Exception:
            os.close(lock_fd)
            raise

    @staticmethod
    def _sleep_with_backoff(delay: float, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            jittered = delay * random.uniform(0.75, 1.25)
            time.sleep(min(jittered, remaining))
        return min(delay * 2, _MAX_RETRY_DELAY_SECONDS)
