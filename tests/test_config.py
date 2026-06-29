from __future__ import annotations

import json

import pytest

import pi_as_mcp.config as config_module
from pi_as_mcp.config import load_config, load_raw_config, save_raw_config
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


def test_model_map_parses_limit_disabled_description(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "agents": {
                    "models": {
                        "local/example-model": {"limit": 2, "disabled": True, "description": "scout"},
                        "bare-model": 5,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(config))

    agents = load_config().agents
    assert agents.is_model_disabled(provider="local", model="example-model") is True
    assert agents.description_for_model(provider="local", model="example-model") == "scout"
    full = agents.concurrency_limit_for_model(provider="local", model="example-model")
    assert full is not None and full.limit == 2
    # Integer shorthand is sugar for a concurrency limit on a bare model name.
    bare = agents.concurrency_limit_for_model(provider="any", model="bare-model")
    assert bare is not None and bare.limit == 5 and bare.match_provider is False


def test_legacy_concurrency_limits_still_load(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"agents": {"concurrency_limits": {"models": {"local/example-model": 3}}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(config))
    agents = load_config().agents
    limit = agents.concurrency_limit_for_model(provider="local", model="example-model")
    assert limit is not None and limit.limit == 3
    assert agents.model_concurrency_limits == {"local/example-model": 3}


def test_skill_config_parses(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"skill": {"intro": "hi"}}), encoding="utf-8")
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(config))
    assert load_config().skill.intro == "hi"


def test_invalid_disabled_flag_is_rejected(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"agents": {"models": {"local/x": {"disabled": "yes"}}}}), encoding="utf-8"
    )
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(config))
    with pytest.raises(PiRpcError, match="disabled must be a boolean"):
        load_config()


def test_save_raw_config_round_trips_and_validates(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.json"
    monkeypatch.setenv("PI_AS_MCP_CONFIG", str(config))

    payload = {"agents": {"unsafe_read_only": True, "models": {"local/x": {"limit": 1}}}}
    save_raw_config(payload)
    assert load_raw_config() == payload
    assert load_config().agents.unsafe_read_only is True

    # A payload the daemon would reject is refused at save time.
    with pytest.raises(PiRpcError):
        save_raw_config({"agents": {"models": {"local/x": {"limit": 0}}}})
