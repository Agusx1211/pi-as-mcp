from __future__ import annotations

import json
import os
from unittest.mock import Mock

import pytest

from pi_as_mcp.compat import (
    CHECKED_REQUEST_COMMAND,
    COMPAT_COMMAND,
    DAEMON_BUILD_ID,
    DAEMON_PROTOCOL_VERSION,
    compatibility_identity,
)
from pi_as_mcp.daemon import RequestHandler
from pi_as_mcp.daemon_client import DaemonClient, DaemonClientError
from pi_as_mcp.pi_rpc import PiRpcError


def encoded(response: dict[str, object]) -> list[bytes]:
    return [json.dumps(response).encode("utf-8")]


def test_every_request_uses_checked_envelope(monkeypatch) -> None:
    client = DaemonClient()
    sent: list[dict[str, object]] = []

    def send(payload: dict[str, object], _timeout: int) -> list[bytes]:
        sent.append(payload)
        return encoded({"ok": True})

    monkeypatch.setattr(client, "_send", send)

    assert client.request("summary") == {"ok": True}
    assert client.request("models") == {"ok": True}
    assert [payload["command"] for payload in sent] == [
        CHECKED_REQUEST_COMMAND,
        CHECKED_REQUEST_COMMAND,
    ]
    assert sent[0]["request"] == {"command": "summary"}
    assert sent[1]["request"] == {"command": "models"}


def test_compat_probe_remains_available_to_older_clients() -> None:
    response = RequestHandler.handle_request(
        Mock(),
        {"command": COMPAT_COMMAND},
    )

    assert response["compatible"] is True
    assert response["protocol_version"] == DAEMON_PROTOCOL_VERSION
    assert response["build_id"] == DAEMON_BUILD_ID


def test_matching_checked_envelope_dispatches_nested_request() -> None:
    class Handler:
        handle_request = RequestHandler.handle_request
        _dispatch_request = RequestHandler._dispatch_request

        @staticmethod
        def peer_pid() -> int:
            return os.getpid()

    response = Handler().handle_request(
        {
            "command": CHECKED_REQUEST_COMMAND,
            **compatibility_identity(),
            "request": {"command": "tui_summary"},
        }
    )

    assert "agents" in response
    assert "stats" in response


@pytest.mark.parametrize(
    "payload",
    [
        {"command": "delegate", "prompt": "must not run"},
        {"command": "reply", "agent_id": "a1", "prompt": "must not run"},
        {"command": "stop", "agent_id": "a1"},
    ],
)
def test_raw_side_effecting_commands_never_dispatch(
    payload: dict[str, object],
) -> None:
    handler = Mock()
    handler._dispatch_request = Mock(
        side_effect=AssertionError("raw operation was dispatched")
    )

    with pytest.raises(PiRpcError, match="compatibility envelope required"):
        RequestHandler.handle_request(handler, payload)

    handler._dispatch_request.assert_not_called()


def test_matching_checked_operation_dispatches() -> None:
    handler = Mock()
    handler._dispatch_request.return_value = {"accepted": True}
    nested = {"command": "delegate", "prompt": "checked"}

    response = RequestHandler.handle_request(
        handler,
        {
            "command": CHECKED_REQUEST_COMMAND,
            **compatibility_identity(),
            "request": nested,
        },
    )

    assert response == {"accepted": True}
    handler._dispatch_request.assert_called_once_with(nested)


def test_raw_tui_summary_remains_available_for_legacy_drain(monkeypatch) -> None:
    agents = [{"agent_id": "a1", "status": "running"}]
    monkeypatch.setattr("pi_as_mcp.daemon.STATE.global_summary", lambda: agents)
    monkeypatch.setattr("pi_as_mcp.daemon.STATE.stats_summary", lambda: {"agents": 1})
    handler = Mock()

    response = RequestHandler.handle_request(handler, {"command": "tui_summary"})

    assert response == {"agents": agents, "stats": {"agents": 1}}
    handler.peer_pid.assert_not_called()


def test_mismatched_daemon_blocks_nested_request(monkeypatch) -> None:
    client = DaemonClient()
    sent: list[dict[str, object]] = []

    def send(payload: dict[str, object], _timeout: int) -> list[bytes]:
        sent.append(payload)
        return encoded(
            {
                "compatible": False,
                "compatibility_error": True,
                "daemon_error": True,
                "protocol_version": DAEMON_PROTOCOL_VERSION,
                "build_id": "older-build",
                "package_version": "0.0.9",
            }
        )

    monkeypatch.setattr(client, "_send", send)

    with pytest.raises(DaemonClientError, match="no request was executed") as exc:
        client.request("delegate", prompt="must not run")

    assert "Existing agents were left untouched" in str(exc.value)
    assert "refresh-daemon.sh" in str(exc.value)
    assert [payload["command"] for payload in sent] == [CHECKED_REQUEST_COMMAND]
    assert sent[0]["request"] == {
        "command": "delegate",
        "prompt": "must not run",
    }


def test_legacy_daemon_blocks_request_with_refresh_hint(monkeypatch) -> None:
    client = DaemonClient()
    sent: list[dict[str, object]] = []

    def send(payload: dict[str, object], _timeout: int) -> list[bytes]:
        sent.append(payload)
        return encoded(
            {
                "error": f"unknown command: {CHECKED_REQUEST_COMMAND}",
                "daemon_error": True,
            }
        )

    monkeypatch.setattr(client, "_send", send)

    with pytest.raises(DaemonClientError, match="predates compatibility checks") as exc:
        client.request("reply", agent_id="a1", prompt="must not run")

    assert "no request was executed" in str(exc.value)
    assert "refresh-daemon.sh" in str(exc.value)
    assert [payload["command"] for payload in sent] == [CHECKED_REQUEST_COMMAND]


def test_checked_envelope_rejects_mismatch_without_dispatch(monkeypatch) -> None:
    handler = Mock()
    start = Mock(side_effect=AssertionError("nested delegate was dispatched"))
    monkeypatch.setattr("pi_as_mcp.daemon.STATE.start", start)

    response = RequestHandler.handle_request(
        handler,
        {
            "command": CHECKED_REQUEST_COMMAND,
            "protocol_version": DAEMON_PROTOCOL_VERSION,
            "build_id": "different-build",
            "request": {
                "command": "delegate",
                "prompt": "must not run",
            },
        },
    )

    assert response["compatibility_error"] is True
    assert response["build_id"] == DAEMON_BUILD_ID
    start.assert_not_called()


def test_checked_envelope_catches_legacy_daemon(monkeypatch) -> None:
    client = DaemonClient()
    sent: list[dict[str, object]] = []

    def legacy_send(payload: dict[str, object], _timeout: int) -> list[bytes]:
        sent.append(payload)
        return encoded(
            {
                "error": f"unknown command: {CHECKED_REQUEST_COMMAND}",
                "daemon_error": True,
            }
        )

    monkeypatch.setattr(client, "_send", legacy_send)

    with pytest.raises(DaemonClientError, match="no request was executed"):
        client.request("delegate", prompt="must not run")

    assert len(sent) == 1
    assert sent[0]["command"] == CHECKED_REQUEST_COMMAND
