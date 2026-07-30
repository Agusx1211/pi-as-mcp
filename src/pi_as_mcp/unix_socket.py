from __future__ import annotations

import socket
import struct
import sys

from pi_as_mcp.pi_rpc import PiRpcError

_LINUX_UCRED_FORMAT = "=3i"
_DARWIN_PEERPID_FORMAT = "=i"
# CPython does not expose these kernel constants on every supported macOS
# release, so retain their stable XNU values as fallbacks.
_DARWIN_SOL_LOCAL = 0
_DARWIN_LOCAL_PEERPID = 0x002


def _peer_pid_from_sockopt(
    peer_socket: socket.socket,
    *,
    level: int,
    option: int,
    value_format: str,
    source: str,
) -> int:
    try:
        raw = peer_socket.getsockopt(
            level,
            option,
            struct.calcsize(value_format),
        )
        peer_pid = int(struct.unpack(value_format, raw)[0])
    except (OSError, struct.error) as exc:
        raise PiRpcError(f"could not determine Unix socket peer PID via {source}") from exc
    if peer_pid <= 0:
        raise PiRpcError(f"Unix socket returned invalid peer PID {peer_pid} via {source}")
    return peer_pid


def unix_socket_peer_pid(
    peer_socket: socket.socket,
    *,
    platform_name: str | None = None,
) -> int:
    """Return the authenticated PID at the other end of a Unix socket.

    Platforms without a trustworthy PID credential mechanism fail closed:
    guessing would defeat parent ownership validation and CLI scope isolation.
    """
    current_platform = platform_name or sys.platform
    if current_platform.startswith("linux"):
        so_peercred = getattr(socket, "SO_PEERCRED", None)
        if so_peercred is None:
            raise PiRpcError("SO_PEERCRED is unavailable on this Linux Python build")
        return _peer_pid_from_sockopt(
            peer_socket,
            level=socket.SOL_SOCKET,
            option=so_peercred,
            value_format=_LINUX_UCRED_FORMAT,
            source="SO_PEERCRED",
        )
    if current_platform == "darwin":
        return _peer_pid_from_sockopt(
            peer_socket,
            level=getattr(socket, "SOL_LOCAL", _DARWIN_SOL_LOCAL),
            option=getattr(socket, "LOCAL_PEERPID", _DARWIN_LOCAL_PEERPID),
            value_format=_DARWIN_PEERPID_FORMAT,
            source="LOCAL_PEERPID",
        )
    raise PiRpcError(
        f"secure Unix socket peer PID lookup is unsupported on {current_platform}"
    )
