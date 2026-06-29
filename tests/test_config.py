from __future__ import annotations

import json

import pytest

import pi_as_mcp.config as config_module
from pi_as_mcp.config import load_config
from pi_as_mcp.pi_rpc import PiRpcError


def test_load_config_reads_model_concurrency_limits(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "agents": {
                    "enable_score": True,
                    "concurrency_limits": {
                        "models": {
                            "local/example-model": 2,
                            "shared-model": 1,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(config))

    loaded = load_config()

    assert loaded.path == config
    assert loaded.agents.enable_score is True
    full_limit = loaded.agents.concurrency_limit_for_model(provider="local", model="example-model")
    bare_limit = loaded.agents.concurrency_limit_for_model(provider="one", model="shared-model")
    assert full_limit is not None
    assert full_limit.limit == 2
    assert full_limit.match_provider is True
    assert bare_limit is not None
    assert bare_limit.limit == 1
    assert bare_limit.match_provider is False


def test_missing_config_is_empty(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(tmp_path / "missing.json"))

    loaded = load_config()

    assert loaded.path is None
    assert loaded.agents.enable_score is False
    assert loaded.agents.model_concurrency_limits == {}


def test_invalid_model_limit_is_rejected(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"agents": {"concurrency_limits": {"models": {"local/example-model": 0}}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(config))

    with pytest.raises(PiRpcError, match="positive integers"):
        load_config()


def test_invalid_enable_score_is_rejected(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"agents": {"enable_score": "yes"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(config))

    with pytest.raises(PiRpcError, match="enable_score"):
        load_config()


def test_unsafe_read_only_defaults_false_and_parses(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"agents": {"enable_score": True}}), encoding="utf-8")
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(config))
    assert load_config().agents.unsafe_read_only is False

    config.write_text(json.dumps({"agents": {"unsafe_read_only": True}}), encoding="utf-8")
    assert load_config().agents.unsafe_read_only is True


def test_invalid_unsafe_read_only_is_rejected(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"agents": {"unsafe_read_only": "yes"}}), encoding="utf-8")
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(config))

    with pytest.raises(PiRpcError, match="unsafe_read_only"):
        load_config()


def test_session_persistence_settings_default_and_parse(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"agents": {"enable_score": True}}), encoding="utf-8")
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(config))
    agents = load_config().agents
    # Persistence (and idle eviction) is on by default.
    assert agents.persist_sessions is True
    assert agents.idle_eviction_seconds == 120

    config.write_text(
        json.dumps({"agents": {"persist_sessions": False, "idle_eviction_seconds": 30}}),
        encoding="utf-8",
    )
    agents = load_config().agents
    assert agents.persist_sessions is False
    assert agents.idle_eviction_seconds == 30


def test_load_config_caches_until_file_changes(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"agents": {"enable_score": True}}), encoding="utf-8")
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(config))

    parse_calls = 0
    real_loads = config_module.json.loads

    def counting_loads(*args, **kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return real_loads(*args, **kwargs)

    monkeypatch.setattr(config_module.json, "loads", counting_loads)

    assert load_config().agents.enable_score is True
    # Unchanged file: served from cache, no re-parse.
    assert load_config().agents.enable_score is True
    assert parse_calls == 1

    # A real change on disk (different content + size) must be picked up.
    config.write_text(
        json.dumps({"agents": {"enable_score": False, "unsafe_read_only": True}}),
        encoding="utf-8",
    )
    reloaded = load_config().agents
    assert reloaded.enable_score is False
    assert reloaded.unsafe_read_only is True
    assert parse_calls == 2


def test_load_config_caches_missing_then_created(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.json"
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(config))
    assert load_config().path is None
    # Creating the file afterwards is detected (cached miss is re-checked via stat).
    config.write_text(json.dumps({"agents": {"enable_score": True}}), encoding="utf-8")
    assert load_config().agents.enable_score is True


def test_invalid_idle_eviction_seconds_is_rejected(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"agents": {"idle_eviction_seconds": -1}}), encoding="utf-8")
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(config))

    with pytest.raises(PiRpcError, match="idle_eviction_seconds"):
        load_config()
