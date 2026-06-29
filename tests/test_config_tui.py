from __future__ import annotations

import json

import anyio

from pi_as_mcp import config_tui
from pi_as_mcp.pi_rpc import CatalogModel


class FakeRunner:
    def list_catalog(self, *, timeout_seconds: int = 15):
        return [
            CatalogModel("local", "alpha", "128K", "8K", thinking=True),
            CatalogModel("local", "beta", "256K", "8K", thinking=False),
            CatalogModel("local", "gamma", "64K", "8K", thinking=False),
        ]


def _setup(tmp_path, monkeypatch, config: dict):
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"defaultProvider": "local", "enabledModels": ["local/alpha", "local/beta"]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(cfg))
    return cfg


def test_loads_catalog_marks_pi_and_orphans(tmp_path, monkeypatch) -> None:
    _setup(
        tmp_path,
        monkeypatch,
        {"agents": {"models": {"local/alpha": {"limit": 2, "description": "scout"},
                               "local/ghost": {"limit": 1}}}},
    )

    async def check() -> None:
        app = config_tui.PiConfigTui(runner=FakeRunner())
        async with app.run_test() as pilot:
            await pilot.pause()
            rows = {r.ref: r for r in app.draft.rows}
            assert set(rows) == {"local/alpha", "local/beta", "local/gamma"}
            assert rows["local/alpha"].in_pi is True
            assert rows["local/gamma"].in_pi is False
            assert rows["local/alpha"].limit == 2
            assert rows["local/alpha"].description == "scout"
            # local/ghost is in config but not in the catalog -> flagged as orphan.
            assert "local/ghost" in app.draft.orphans

    anyio.run(check)


def test_toggle_disable_and_globals_then_save(tmp_path, monkeypatch) -> None:
    cfg = _setup(tmp_path, monkeypatch, {})

    async def check() -> None:
        app = config_tui.PiConfigTui(runner=FakeRunner())
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("d")  # disable the first model (local/alpha)
            await pilot.press("u")  # unsafe_read_only on
            await pilot.press("c")  # enable_score on
            assert app.draft.rows[0].disabled is True
            assert app.draft.unsafe_read_only is True
            assert app.draft.enable_score is True
            await pilot.press("w")  # save
            await pilot.pause()
            assert app.draft.dirty is False

    anyio.run(check)

    saved = json.loads(cfg.read_text(encoding="utf-8"))
    assert saved["agents"]["unsafe_read_only"] is True
    assert saved["agents"]["enable_score"] is True
    assert saved["agents"]["models"]["local/alpha"]["disabled"] is True


def test_set_limit_and_description_via_modal(tmp_path, monkeypatch) -> None:
    _setup(tmp_path, monkeypatch, {})

    async def check() -> None:
        app = config_tui.PiConfigTui(runner=FakeRunner())
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("l")
            await pilot.pause()
            await pilot.press("7")
            await pilot.press("enter")
            await pilot.pause()
            assert app.draft.rows[0].limit == 7

            await pilot.press("e")
            await pilot.pause()
            await pilot.press("h", "i")
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert app.draft.rows[0].description == "hi"

    anyio.run(check)


def test_edit_intro_then_save(tmp_path, monkeypatch) -> None:
    cfg = _setup(tmp_path, monkeypatch, {})

    async def check() -> None:
        app = config_tui.PiConfigTui(runner=FakeRunner())
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("t")
            await pilot.pause()
            await pilot.press("y", "o")
            await pilot.press("ctrl+s")
            await pilot.pause()
            await pilot.press("w")
            await pilot.pause()

    anyio.run(check)
    saved = json.loads(cfg.read_text(encoding="utf-8"))
    assert saved["skill"]["intro"].startswith("yo")
