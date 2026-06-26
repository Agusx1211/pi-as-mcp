from __future__ import annotations

import os
from pathlib import Path

from pi_as_mcp.paths import runtime_dir


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
