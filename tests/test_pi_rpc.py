from __future__ import annotations

import pytest

from pi_as_mcp.pi_rpc import (
    READ_ONLY_GUARD,
    TOOL_PROFILES,
    PiRpcError,
    PiRpcRunner,
    assistant_message_text,
    guard_prompt,
    model_context_tokens,
    model_row_visible,
    parse_context_tokens,
    resolve_model,
    validate_tool_mode,
)


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
