from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pi_as_mcp.pi_rpc import PiRpcError
from pi_as_mcp.sessions import DEFAULT_IDLE_EVICTION_SECONDS

CONFIG_ENV = "PI_AS_MCP_CONFIG"
DEFAULT_CONFIG_PATH = Path(".pi-as-mcp/config.json")


@dataclass(frozen=True)
class ModelConcurrencyLimit:
    key: str
    limit: int
    provider: str
    model: str
    match_provider: bool

    @property
    def display_model(self) -> str:
        return f"{self.provider}/{self.model}" if self.match_provider else self.model


@dataclass(frozen=True)
class AgentsConfig:
    model_concurrency_limits: dict[str, int]
    enable_score: bool = False
    # When true, read-only agents are launched with the full tool set (so they
    # can run git/build/test via bash) and every prompt is prefixed with a hard
    # read-only instruction. Soft, trust-based guard -- see READ_ONLY_GUARD.
    unsafe_read_only: bool = False
    # When true (default), each agent persists its conversation to disk so an idle
    # worker can be killed to reclaim memory and transparently resumed on a later
    # reply. Set false to fall back to ephemeral, always-resident workers.
    persist_sessions: bool = True
    # Seconds an idle worker may sit before it is evicted (process killed, session
    # kept on disk). 0 disables eviction. Only applies when persist_sessions.
    idle_eviction_seconds: float = DEFAULT_IDLE_EVICTION_SECONDS

    def concurrency_limit_for_model(self, *, provider: str, model: str) -> ModelConcurrencyLimit | None:
        full_key = f"{provider}/{model}"
        if full_key in self.model_concurrency_limits:
            return ModelConcurrencyLimit(
                key=full_key,
                limit=self.model_concurrency_limits[full_key],
                provider=provider,
                model=model,
                match_provider=True,
            )
        if model in self.model_concurrency_limits:
            return ModelConcurrencyLimit(
                key=model,
                limit=self.model_concurrency_limits[model],
                provider=provider,
                model=model,
                match_provider=False,
            )
        return None


@dataclass(frozen=True)
class AppConfig:
    agents: AgentsConfig
    path: Path | None = None


def config_path() -> Path:
    # Config is user-global only: there is no per-project config. The daemon's
    # working directory is incidental, so resolving against cwd would make the
    # active policy depend on where the daemon happened to be launched.
    override = os.environ.get(CONFIG_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / DEFAULT_CONFIG_PATH


def load_config() -> AppConfig:
    path = config_path()
    if not path.exists():
        return AppConfig(agents=AgentsConfig(model_concurrency_limits={}), path=None)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PiRpcError(f"pi-as-mcp config is invalid JSON: {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise PiRpcError(f"pi-as-mcp config must contain a JSON object: {path}")

    return AppConfig(agents=parse_agents_config(payload.get("agents"), path=path), path=path)


def parse_agents_config(value: Any, *, path: Path) -> AgentsConfig:
    if value is None:
        return AgentsConfig(model_concurrency_limits={})
    if not isinstance(value, dict):
        raise PiRpcError(f"pi-as-mcp config agents must be an object: {path}")

    limits = value.get("concurrency_limits", value.get("concurrencyLimits", {}))
    if limits is None:
        limits = {}
    if not isinstance(limits, dict):
        raise PiRpcError(f"pi-as-mcp config agents.concurrency_limits must be an object: {path}")

    models = limits.get("models", {})
    if models is None:
        models = {}
    if not isinstance(models, dict):
        raise PiRpcError(f"pi-as-mcp config agents.concurrency_limits.models must be an object: {path}")

    parsed_models: dict[str, int] = {}
    for model_ref, limit in models.items():
        if not isinstance(model_ref, str) or not model_ref.strip():
            raise PiRpcError(
                f"pi-as-mcp config agents.concurrency_limits.models keys must be non-empty strings: {path}"
            )
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise PiRpcError(
                "pi-as-mcp config agents.concurrency_limits.models values "
                f"must be positive integers: {path}: {model_ref!r}"
            )
        parsed_models[model_ref.strip()] = limit

    enable_score = value.get("enable_score", value.get("enableScore", False))
    if not isinstance(enable_score, bool):
        raise PiRpcError(f"pi-as-mcp config agents.enable_score must be a boolean: {path}")

    unsafe_read_only = value.get("unsafe_read_only", value.get("unsafeReadOnly", False))
    if not isinstance(unsafe_read_only, bool):
        raise PiRpcError(f"pi-as-mcp config agents.unsafe_read_only must be a boolean: {path}")

    persist_sessions = value.get("persist_sessions", value.get("persistSessions", True))
    if not isinstance(persist_sessions, bool):
        raise PiRpcError(f"pi-as-mcp config agents.persist_sessions must be a boolean: {path}")

    idle_eviction_seconds = value.get(
        "idle_eviction_seconds", value.get("idleEvictionSeconds", DEFAULT_IDLE_EVICTION_SECONDS)
    )
    if (
        isinstance(idle_eviction_seconds, bool)
        or not isinstance(idle_eviction_seconds, (int, float))
        or idle_eviction_seconds < 0
    ):
        raise PiRpcError(
            f"pi-as-mcp config agents.idle_eviction_seconds must be a non-negative number: {path}"
        )

    return AgentsConfig(
        model_concurrency_limits=parsed_models,
        enable_score=enable_score,
        unsafe_read_only=unsafe_read_only,
        persist_sessions=persist_sessions,
        idle_eviction_seconds=float(idle_eviction_seconds),
    )
