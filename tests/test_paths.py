from __future__ import annotations

import os
from pathlib import Path

from pi_as_mcp.paths import runtime_dir, session_dir


def test_runtime_dir_ignores_xdg_runtime_dir(monkeypatch) -> None:
    monkeypatch.delenv("PI_AS_MCP_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/999999")

    assert runtime_dir() == Path(f"/tmp/pi-as-mcp-{os.getuid()}")


def test_runtime_dir_allows_explicit_override(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "custom-runtime"
    monkeypatch.setenv("PI_AS_MCP_RUNTIME_DIR", str(target))
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/999999")

    assert runtime_dir() == target
    assert target.exists()


def test_session_dir_is_durable_under_stats_dir(tmp_path: Path, monkeypatch) -> None:
    # Sessions must NOT live under the ephemeral runtime dir; they default to the
    # durable ~/.pi-as-mcp/sessions store so transcripts survive /tmp cleanup.
    monkeypatch.delenv("PI_AS_MCP_SESSION_DIR", raising=False)
    monkeypatch.setenv("PI_AS_MCP_RUNTIME_DIR", "/tmp/pi-as-mcp-should-not-be-used")
    monkeypatch.setenv("HOME", str(tmp_path))

    path = session_dir()
    assert path == tmp_path / ".pi-as-mcp" / "sessions"
    # Durable store must never be nested under the ephemeral runtime dir.
    assert "pi-as-mcp-should-not-be-used" not in str(path)
    assert runtime_dir() not in path.parents
    assert path.is_dir()


def test_session_dir_allows_explicit_override(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "durable-sessions"
    monkeypatch.setenv("PI_AS_MCP_SESSION_DIR", str(target))

    assert session_dir() == target
    assert target.is_dir()
    assert oct(target.stat().st_mode & 0o777) == "0o700"
