from __future__ import annotations

import json

import pytest

from pi_as_mcp import skill
from pi_as_mcp.config import load_config
from pi_as_mcp.pi_rpc import CatalogModel


def _enable_models(tmp_path, monkeypatch, refs: list[str]) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"defaultProvider": refs[0].split("/")[0], "enabledModels": refs}),
        encoding="utf-8",
    )
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))


def _write_config(tmp_path, monkeypatch, payload: dict) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(config))


CATALOG = [
    CatalogModel("local", "alpha", "128K", "8K", thinking=True),
    CatalogModel("local", "beta", "256K", "8K", thinking=False),
]


def test_roster_excludes_disabled_and_shows_config(tmp_path, monkeypatch) -> None:
    _enable_models(tmp_path, monkeypatch, ["local/alpha", "local/beta"])
    _write_config(
        tmp_path,
        monkeypatch,
        {
            "agents": {
                "models": {
                    "local/alpha": {"disabled": True},
                    "local/beta": {"limit": 3, "description": "Use beta for refactors."},
                }
            }
        },
    )

    body = skill.render_skill_body(load_config(), catalog=CATALOG)

    # alpha is disabled -> hidden; beta shows description, limit and capabilities.
    assert "local/alpha" not in body
    assert "local/beta" in body
    assert "Use beta for refactors." in body
    assert "max 3 concurrent" in body
    assert "256K ctx" in body


def test_custom_intro_overrides_default(tmp_path, monkeypatch) -> None:
    _enable_models(tmp_path, monkeypatch, ["local/alpha"])
    _write_config(tmp_path, monkeypatch, {"skill": {"intro": "MY CUSTOM INTRO"}})

    body = skill.render_skill_body(load_config(), catalog=CATALOG)
    assert body.startswith("MY CUSTOM INTRO")
    assert skill.DEFAULT_INTRO not in body


def test_default_intro_used_when_blank(tmp_path, monkeypatch) -> None:
    _enable_models(tmp_path, monkeypatch, ["local/alpha"])
    _write_config(tmp_path, monkeypatch, {})
    body = skill.render_skill_body(load_config(), catalog=CATALOG)
    assert body.startswith(skill.DEFAULT_INTRO[:20])


def test_server_instructions_point_at_resource(tmp_path, monkeypatch) -> None:
    _enable_models(tmp_path, monkeypatch, ["local/beta"])
    _write_config(tmp_path, monkeypatch, {})
    text = skill.render_server_instructions(load_config(), catalog=CATALOG)
    assert "local/beta" in text
    assert skill.SKILL_RESOURCE_URI in text
