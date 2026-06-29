from __future__ import annotations

import subprocess

import pytest

import pi_as_mcp.pi_rpc as pi_rpc
from pi_as_mcp.pi_rpc import (
    READ_ONLY_GUARD,
    TOOL_PROFILES,
    PiRpcError,
    PiRpcRunner,
    assistant_message_text,
    guard_prompt,
    load_pi_settings,
    model_context_tokens,
    model_row_visible,
    parse_context_tokens,
    parse_model_catalog,
    resolve_model,
    validate_tool_mode,
)


def test_parse_model_catalog_reads_rows_and_capabilities() -> None:
    output = """provider   model                     context  max-out  thinking  images
macstudio  qwen3.6-35b-a3b-mtp       262.1K   16.4K    yes       yes
mistral    open-mistral-7b           8K       8K       no        no
"""
    catalog = parse_model_catalog(output)
    assert [c.ref for c in catalog] == ["macstudio/qwen3.6-35b-a3b-mtp", "mistral/open-mistral-7b"]
    qwen = catalog[0]
    assert qwen.context == "262.1K"
    assert qwen.thinking is True
    assert qwen.images is True
    assert catalog[1].thinking is False
    assert catalog[1].images is False


def test_model_row_visible_exact_match() -> None:
    output = """provider   model                    context
local  example-model  128K
openai     gpt-4o-mini              128K
"""
    assert model_row_visible(output, "local", "example-model")
    assert not model_row_visible(output, "local", "example")


def test_model_context_tokens_parses_list_models_context() -> None:
    output = """provider   model                    context
local  example-model  128K
mistral    mistral-medium-3.5       1M
"""
    assert parse_context_tokens("128K") == 128_000
    assert model_context_tokens(output, "mistral", "mistral-medium-3.5") == 1_000_000


def test_assistant_message_text_ignores_user_prompt_echoes() -> None:
    assistant = {"role": "assistant", "content": [{"type": "text", "text": "done"}]}
    user = {"role": "user", "content": [{"type": "text", "text": "submitted prompt"}]}

    assert assistant_message_text(assistant) == "done"
    assert assistant_message_text(user) == ""


def test_validate_tool_mode_rejects_unknown_mode() -> None:
    with pytest.raises(PiRpcError, match="invalid tool_mode"):
        validate_tool_mode("root")


def test_models_are_read_from_pi_settings(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(
        """{
  "defaultProvider": "mistral",
  "defaultModel": "mistral-medium-3.5",
  "enabledModels": [
    "local/example-model",
    "mistral/mistral-medium-3.5"
  ]
}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))
    monkeypatch.setenv("PI_AS_MCP_MODEL", "ignored/model")
    monkeypatch.setenv("PI_AS_MCP_MODELS_JSON", '{"extra":{"provider":"x","model":"y"}}')

    assert PiRpcRunner(pi_bin="pi").model_aliases() == [
        {
            "alias": "example-model",
            "provider": "local",
            "model": "example-model",
        },
        {
            "alias": "mistral-medium-3.5",
            "provider": "mistral",
            "model": "mistral-medium-3.5",
        },
    ]


def test_resolve_model_uses_pi_default_and_configured_aliases(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "settings.json").write_text(
        """{
  "defaultProvider": "mistral",
  "defaultModel": "mistral-medium-3.5",
  "enabledModels": [
    "local/example-model",
    "mistral/mistral-medium-3.5"
  ]
}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))

    assert resolve_model(None, None) == resolve_model("mistral-medium-3.5", None)
    assert resolve_model("local/example-model", None).provider == "local"
    assert resolve_model("example-model", None).model == "example-model"


def test_resolve_model_uses_env_when_pi_settings_are_missing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))
    monkeypatch.setenv("PI_AS_MCP_PROVIDER", "local")
    monkeypatch.setenv("PI_AS_MCP_MODEL", "example-model")

    resolved = resolve_model(None, None)

    assert resolved.provider == "local"
    assert resolved.model == "example-model"
    assert PiRpcRunner(pi_bin="pi").model_aliases() == [
        {"alias": "example-model", "provider": "local", "model": "example-model"}
    ]


def test_explicit_provider_model_does_not_need_default_config(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))
    monkeypatch.delenv("PI_AS_MCP_PROVIDER", raising=False)
    monkeypatch.delenv("PI_AS_MCP_MODEL", raising=False)

    resolved = resolve_model("local/example-model", None)

    assert resolved.provider == "local"
    assert resolved.model == "example-model"
    assert PiRpcRunner(pi_bin="pi").model_aliases() == []


def test_missing_model_config_has_no_personal_fallback(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))
    monkeypatch.delenv("PI_AS_MCP_PROVIDER", raising=False)
    monkeypatch.delenv("PI_AS_MCP_MODEL", raising=False)

    with pytest.raises(PiRpcError, match="No Pi model configured"):
        resolve_model(None, None)


def test_load_pi_settings_caches_until_file_changes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text('{"defaultProvider": "mistral"}', encoding="utf-8")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))

    parse_calls = 0
    real_loads = pi_rpc.json.loads

    def counting_loads(*args, **kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return real_loads(*args, **kwargs)

    monkeypatch.setattr(pi_rpc.json, "loads", counting_loads)

    first = load_pi_settings()
    assert first == {"defaultProvider": "mistral"}
    # Unchanged file: served from cache, no re-parse.
    second = load_pi_settings()
    assert second == {"defaultProvider": "mistral"}
    assert parse_calls == 1

    # A real change on disk (different content + size) must be picked up.
    settings.write_text('{"defaultProvider": "openai", "defaultModel": "gpt"}', encoding="utf-8")
    third = load_pi_settings()
    assert third == {"defaultProvider": "openai", "defaultModel": "gpt"}
    assert parse_calls == 2


def test_load_pi_settings_caches_missing_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))
    assert load_pi_settings() == {}
    # Creating the file afterwards is detected (cached miss is re-checked via stat).
    (tmp_path / "settings.json").write_text('{"defaultProvider": "x"}', encoding="utf-8")
    assert load_pi_settings() == {"defaultProvider": "x"}


def test_build_args_limits_read_only_tools() -> None:
    runner = PiRpcRunner(pi_bin="pi")
    args = runner._build_args("local", "example-model", "read-only")

    assert "--mode" in args
    assert "rpc" in args
    assert "--thinking" not in args
    assert "--no-context-files" in args
    assert "--no-approve" in args
    assert "--tools" in args
    assert "read,grep,find,ls" in args
    assert "bash" not in ",".join(args)


def test_unsafe_read_only_is_not_a_selectable_tool_mode() -> None:
    # The policy lives in config, not as a tool the model can pick.
    assert "unsafe-read-only" not in TOOL_PROFILES


def test_guard_prompt_prepends_active_guard() -> None:
    guarded = guard_prompt(READ_ONLY_GUARD, "inspect the dirty changes")
    assert guarded.startswith(READ_ONLY_GUARD)
    assert guarded.endswith("inspect the dirty changes")

    assert guard_prompt(None, "inspect the dirty changes") == "inspect the dirty changes"


def _fake_list_models(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["pi", "--offline", "--list-models", "example-model"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def test_validate_model_caches_and_skips_second_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = PiRpcRunner(pi_bin="pi")
    listing = """provider   model                    context
local  example-model  128K
"""
    calls = 0

    def fake_list_models(search: str, *, timeout_seconds: int = 15) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return _fake_list_models(listing)

    monkeypatch.setattr(runner, "list_models", fake_list_models)

    first = runner.validate_model("local", "example-model")
    second = runner.validate_model("local", "example-model")

    assert first == 128_000
    assert second == 128_000
    # Second validation of the same (provider, model) is served from cache.
    assert calls == 1


def test_validate_model_does_not_cache_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = PiRpcRunner(pi_bin="pi")
    calls = 0

    def failing_list_models(search: str, *, timeout_seconds: int = 15) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            args=["pi", "--offline", "--list-models", search],
            returncode=1,
            stdout="",
            stderr="boom",
        )

    monkeypatch.setattr(runner, "list_models", failing_list_models)

    with pytest.raises(PiRpcError):
        runner.validate_model("local", "example-model")
    with pytest.raises(PiRpcError):
        runner.validate_model("local", "example-model")

    # Transient failures are retried, not cached.
    assert calls == 2
