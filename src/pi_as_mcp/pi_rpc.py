from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ToolMode = Literal["none", "read-only", "write", "full"]

DEFAULT_PI_BIN = "pi"
DEFAULT_PI_AGENT_DIR = "~/.pi/agent"
DEFAULT_TURN_TIMEOUT_SECONDS = 300
DEFAULT_MODEL_VALIDATION_TIMEOUT_SECONDS = 15

TOOL_PROFILES: dict[str, list[str]] = {
    "none": [],
    "read-only": ["read", "grep", "find", "ls"],
    "write": ["read", "grep", "find", "ls", "edit", "write"],
    "full": ["read", "grep", "find", "ls", "edit", "write", "bash"],
}

# Prepended to every prompt when the operator enables agents.unsafe_read_only.
# The request runs with the FULL tool set so git/build/test work through bash,
# and the guard instruction is the only thing keeping the run within its
# contract -- a soft, trust-based boundary, not a sandbox.
READ_ONLY_GUARD = (
    "IMPORTANT — READ-ONLY TASK. Before running ANY command, think through its "
    "second-order consequences: if it could create, modify, delete, download, "
    "compile, or otherwise change any file or system state — even as a side "
    "effect — do NOT run it. Investigate only with read-only commands: read "
    "files, grep, and inspect with git/shell read commands (e.g. `git status`, "
    "`git diff`, `git log`, `ls`, `cat`, `find`). You MUST NOT modify, create, "
    "delete, move, or rename files; MUST NOT use edit/write tools or `git "
    "add/commit/checkout/restore/reset/stash/push`; and MUST NOT run builds, "
    "compilers, code generators, installers, migrations, or build/test scripts "
    "(e.g. `make`, `cmake --build`, `npm run build`, `npm test`, `npm install`) "
    "— these write artifacts. If the task would require any such change, do NOT "
    "do it; describe what you would do instead."
)



def guard_prompt(guard: str | None, prompt: str) -> str:
    """Prepend an operator guard instruction when one is active."""
    if guard:
        return f"{guard}\n\n---\n\n{prompt}"
    return prompt


class PiRpcError(RuntimeError):
    """Raised when Pi RPC delegation fails before a normal agent result."""


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    provider: str
    model: str

    def to_json(self) -> dict[str, str]:
        return {
            "alias": self.alias,
            "provider": self.provider,
            "model": self.model,
        }


@dataclass
class ToolCall:
    id: str | None
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    is_error: bool | None = None
    result_preview: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "args": self.args,
            "is_error": self.is_error,
            "result_preview": self.result_preview,
        }


def env_default(name: str, fallback: str) -> str:
    value = os.environ.get(name)
    return value if value else fallback


def env_value(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def pi_agent_dir() -> Path:
    return Path(env_default("PI_CODING_AGENT_DIR", DEFAULT_PI_AGENT_DIR)).expanduser()


def load_pi_settings() -> dict[str, Any]:
    settings_path = pi_agent_dir() / "settings.json"
    if not settings_path.exists():
        return {}

    try:
        parsed = json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PiRpcError(f"Pi settings file is invalid JSON: {settings_path}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise PiRpcError(f"Pi settings file must contain a JSON object: {settings_path}")
    return parsed


def missing_model_config_error() -> PiRpcError:
    return PiRpcError(
        "No Pi model configured. Set defaultModel or enabledModels in "
        f"{pi_agent_dir() / 'settings.json'}, pass --model provider/model, "
        "or set PI_AS_MCP_PROVIDER and PI_AS_MCP_MODEL."
    )


def parse_model_ref(value: str, default_provider: str | None) -> tuple[str, str]:
    if "/" not in value:
        if not default_provider:
            raise PiRpcError(
                f"Pi model reference {value!r} does not include a provider. "
                "Use provider/model or set defaultProvider/PI_AS_MCP_PROVIDER."
            )
        return default_provider, value

    provider, model = value.split("/", 1)
    if not provider or not model:
        raise PiRpcError(f"invalid Pi model reference: {value!r}")
    return provider, model


def pi_configured_provider(settings: dict[str, Any] | None = None) -> str | None:
    settings = settings if settings is not None else load_pi_settings()
    provider = settings.get("defaultProvider")
    if isinstance(provider, str) and provider:
        return provider
    return env_value("PI_AS_MCP_PROVIDER")


def pi_default_provider(settings: dict[str, Any] | None = None) -> str:
    provider = pi_configured_provider(settings)
    if provider:
        return provider
    raise PiRpcError(
        "No Pi provider configured. Set defaultProvider in "
        f"{pi_agent_dir() / 'settings.json'}, pass --model provider/model, "
        "or set PI_AS_MCP_PROVIDER."
    )


def configured_model_specs(*, require: bool = True) -> list[ModelSpec]:
    settings = load_pi_settings()
    default_provider = pi_configured_provider(settings)
    enabled_models = settings.get("enabledModels")

    if isinstance(enabled_models, list):
        specs: list[ModelSpec] = []
        for item in enabled_models:
            if not isinstance(item, str) or not item:
                raise PiRpcError("Pi settings enabledModels entries must be non-empty strings")
            provider, model = parse_model_ref(item, default_provider)
            specs.append(ModelSpec(model, provider, model))
        if specs:
            return specs

    default_model = settings.get("defaultModel")
    if isinstance(default_model, str) and default_model:
        provider, model = parse_model_ref(default_model, default_provider)
        return [ModelSpec(model, provider, model)]

    env_model = env_value("PI_AS_MCP_MODEL")
    if env_model:
        provider, model = parse_model_ref(env_model, default_provider)
        return [ModelSpec(model, provider, model)]

    if require:
        raise missing_model_config_error()
    return []


def default_model_spec() -> ModelSpec:
    settings = load_pi_settings()
    default_provider = pi_configured_provider(settings)
    default_model = settings.get("defaultModel")
    if isinstance(default_model, str) and default_model:
        provider, model = parse_model_ref(default_model, default_provider)
        requested = ModelSpec(model, provider, model)
        for spec in configured_model_specs():
            if spec.provider == requested.provider and spec.model == requested.model:
                return spec
        return requested

    specs = configured_model_specs()
    return specs[0]


def configured_model_aliases(*, require: bool = True) -> dict[str, ModelSpec]:
    aliases: dict[str, ModelSpec] = {}
    ambiguous: set[str] = set()

    def add(alias: str, spec: ModelSpec) -> None:
        if not alias or alias in ambiguous:
            return
        existing = aliases.get(alias)
        if existing is None or existing == spec:
            aliases[alias] = spec
            return
        aliases.pop(alias, None)
        ambiguous.add(alias)

    for spec in configured_model_specs(require=require):
        add(spec.alias, spec)
        add(spec.model, spec)
        add(f"{spec.provider}/{spec.model}", spec)

    return aliases


def resolve_model(model: str | None = None, provider: str | None = None) -> ModelSpec:
    if not model:
        return default_model_spec()

    requested = model.strip()
    if not requested:
        return default_model_spec()

    aliases = configured_model_aliases(require=False)
    if provider is None and requested in aliases:
        return aliases[requested]

    if provider is None and "/" in requested:
        raw_provider, raw_model = parse_model_ref(requested, None)
        return ModelSpec(raw_model, raw_provider, raw_model)

    raw_provider = provider or pi_default_provider()
    return ModelSpec(requested, raw_provider, requested)


def resolve_cwd(cwd: str | None) -> Path:
    path = Path(cwd).expanduser() if cwd else Path.cwd()
    path = path.resolve()
    if not path.exists():
        raise PiRpcError(f"cwd does not exist: {path}")
    if not path.is_dir():
        raise PiRpcError(f"cwd is not a directory: {path}")
    return path


def extract_message_text(message: dict[str, Any] | None) -> str:
    if not message:
        return ""

    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    chunks: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks).strip()


def assistant_message_text(message: dict[str, Any] | None) -> str:
    if not isinstance(message, dict):
        return ""
    role = message.get("role")
    if role is not None and role != "assistant":
        return ""
    return extract_message_text(message)


def last_assistant_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            text = extract_message_text(message)
            if text:
                return text
    return ""


def extract_text_preview(result: dict[str, Any] | None, limit: int = 500) -> str | None:
    if not result:
        return None
    content = result.get("content")
    if not isinstance(content, list):
        return None

    chunks: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                chunks.append(text)
    preview = "".join(chunks).strip()
    if not preview:
        return None
    return preview[:limit] + ("..." if len(preview) > limit else "")


def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {"type": event.get("type")}
    for key in ("command", "success", "error", "toolCallId", "toolName", "isError"):
        if key in event:
            compact[key] = event[key]
    if "args" in event and isinstance(event["args"], dict):
        compact["args"] = event["args"]
    return compact


def validate_tool_mode(tool_mode: str) -> None:
    if tool_mode not in TOOL_PROFILES:
        valid = ", ".join(TOOL_PROFILES)
        raise PiRpcError(f"invalid tool_mode {tool_mode!r}; expected one of: {valid}")


def validate_timeout(timeout_seconds: int) -> None:
    if timeout_seconds < 1 or timeout_seconds > 900:
        raise PiRpcError("timeout_seconds must be between 1 and 900")


def parse_context_tokens(value: str) -> int | None:
    text = value.strip().replace(",", "")
    if not text:
        return None

    multiplier = 1
    suffix = text[-1].lower()
    if suffix in {"k", "m"}:
        multiplier = 1_000 if suffix == "k" else 1_000_000
        text = text[:-1]

    try:
        number = float(text)
    except ValueError:
        return None
    return int(number * multiplier) if number > 0 else None


def model_context_tokens(output: str, provider: str, model: str) -> int | None:
    for line in output.splitlines():
        columns = line.split()
        if len(columns) >= 3 and columns[0] == provider and columns[1] == model:
            return parse_context_tokens(columns[2])
    return None


class PiRpcRunner:
    def __init__(self, pi_bin: str | None = None) -> None:
        self.pi_bin = pi_bin or env_default("PI_AS_MCP_PI_BIN", DEFAULT_PI_BIN)

    def health(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        timeout_seconds: int = 15,
    ) -> dict[str, Any]:
        validate_timeout(timeout_seconds)
        model_spec = resolve_model(model, provider)
        pi_path = shutil.which(self.pi_bin) or self.pi_bin

        version = subprocess.run(
            [self.pi_bin, "--version"],
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        listing = self.list_models(model_spec.model, timeout_seconds=timeout_seconds)
        model_visible = model_row_visible(listing.stdout, model_spec.provider, model_spec.model)

        return {
            "pi_bin": self.pi_bin,
            "pi_path": pi_path,
            "pi_version": version.stdout.strip(),
            "version_stderr": version.stderr.strip(),
            "requested_model": model or model_spec.alias,
            "provider": model_spec.provider,
            "model": model_spec.model,
            "model_visible": model_visible,
            "list_models_stdout": listing.stdout.strip(),
            "list_models_stderr": listing.stderr.strip(),
            "list_models_returncode": listing.returncode,
        }

    def model_aliases(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for spec in configured_model_specs(require=False):
            rows.append(spec.to_json())
        return rows

    def validate_model(self, provider: str, model: str, *, timeout_seconds: int = 15) -> int | None:
        listing = self.list_models(model, timeout_seconds=timeout_seconds)
        if listing.returncode != 0:
            raise PiRpcError(
                f"pi --list-models failed with code {listing.returncode}: "
                f"{listing.stderr.strip() or listing.stdout.strip()}"
            )
        if not model_row_visible(listing.stdout, provider, model):
            raise PiRpcError(
                "Configured Pi model was not found exactly in `pi --list-models` output: "
                f"provider={provider!r} model={model!r}."
            )
        return model_context_tokens(listing.stdout, provider, model)

    def list_models(self, search: str, *, timeout_seconds: int = 15) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PI_OFFLINE"] = "1"
        return subprocess.run(
            [self.pi_bin, "--offline", "--list-models", search],
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )

    def _build_args(
        self,
        provider: str,
        model: str,
        tool_mode: str,
        *,
        session_id: str | None = None,
        session_dir: str | None = None,
    ) -> list[str]:
        args = [
            self.pi_bin,
            "--mode",
            "rpc",
        ]
        if session_id and session_dir:
            # Persist the conversation to disk under our own id. Pi appends each
            # entry synchronously (append-only session log), so an idle worker can
            # be killed to reclaim memory and a later reply respawns a process that
            # resumes the exact session by id, rehydrating full context. Without a
            # session dir we fall back to ephemeral, non-resumable workers.
            args += ["--session-id", session_id, "--session-dir", session_dir]
        else:
            args.append("--no-session")
        args += [
            "--offline",
            "--provider",
            provider,
            "--model",
            model,
            "--no-context-files",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-approve",
        ]
        tools = TOOL_PROFILES[tool_mode]
        if tools:
            args.extend(["--tools", ",".join(tools)])
        else:
            args.append("--no-tools")
        return args


def model_row_visible(output: str, provider: str, model: str) -> bool:
    pattern = re.compile(rf"^{re.escape(provider)}\s+{re.escape(model)}(?:\s|$)")
    return any(pattern.search(line) for line in output.splitlines())
