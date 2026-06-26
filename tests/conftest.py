from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_global_config(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point PI_AS_MCP_CONFIG at an absent file by default.

    There is no per-project config, so an unpinned test would otherwise read the
    developer's real ~/.pi-as-mcp/config.json and behave differently per machine.
    Pointing at a non-existent path makes load_config() return clean defaults.
    Tests that need specific settings call monkeypatch.setenv("PI_AS_MCP_CONFIG", ...)
    in their own body, which runs after this fixture and overrides it.
    """
    absent = tmp_path_factory.mktemp("pi-as-mcp-config") / "config.json"
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(absent))
