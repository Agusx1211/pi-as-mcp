from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
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
class ModelConfig:
    """pi-as-mcp's per-model policy, keyed by the reference written in config.

    The reference may be fully qualified (``provider/model``) or a bare model
    name. A bare name matches that model across providers, mirroring how
    concurrency limits resolve.
    """

    # Max concurrent starting/running agents for this model. None means no limit.
    limit: int | None = None
    # When true the model is hidden from the `models` tool/skill and delegation
    # to it is rejected, even if Pi still lists it in enabledModels.
    disabled: bool = False
    # Short human-written guidance about what this model is for and its rules.
    # Surfaced verbatim in the generated skill / AGENTS block.
    description: str = ""


@dataclass(frozen=True)
class AgentsConfig:
    models: dict[str, ModelConfig]
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

    @property
    def model_concurrency_limits(self) -> dict[str, int]:
        """Legacy view: only the refs that carry a positive concurrency limit."""
        return {key: cfg.limit for key, cfg in self.models.items() if cfg.limit is not None}

    def _model_config_for(self, *, provider: str, model: str) -> tuple[str, bool, ModelConfig] | None:
        full_key = f"{provider}/{model}"
        if full_key in self.models:
            return full_key, True, self.models[full_key]
        if model in self.models:
            return model, False, self.models[model]
        return None

    def concurrency_limit_for_model(self, *, provider: str, model: str) -> ModelConcurrencyLimit | None:
        match = self._model_config_for(provider=provider, model=model)
        if match is None:
            return None
        key, match_provider, cfg = match
        if cfg.limit is None:
            return None
        return ModelConcurrencyLimit(
            key=key,
            limit=cfg.limit,
            provider=provider,
            model=model,
            match_provider=match_provider,
        )

    def is_model_disabled(self, *, provider: str, model: str) -> bool:
        match = self._model_config_for(provider=provider, model=model)
        return bool(match and match[2].disabled)

    def description_for_model(self, *, provider: str, model: str) -> str:
        match = self._model_config_for(provider=provider, model=model)
        return match[2].description if match else ""


@dataclass(frozen=True)
class SkillConfig:
    # Custom prose for the top of the generated skill. Empty falls back to the
    # built-in default (see pi_as_mcp.skill.DEFAULT_INTRO).
    intro: str = ""


@dataclass(frozen=True)
class AppConfig:
    agents: AgentsConfig
    skill: SkillConfig = field(default_factory=SkillConfig)
    path: Path | None = None


def config_path() -> Path:
    # Config is user-global only: there is no per-project config. The daemon's
    # working directory is incidental, so resolving against cwd would make the
    # active policy depend on where the daemon happened to be launched.
    override = os.environ.get(CONFIG_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / DEFAULT_CONFIG_PATH


# Cache parsed config keyed by (path, st_mtime_ns, st_size). load_config is hit on
# the daemon hot path (manager_for, start, score_hint, ...); re-reading + re-parsing
# config.json each call is wasted work. A missing file is cached with a `None` stat
# key so we still cheaply re-check existence via os.stat each call.
_CONFIG_LOCK = threading.Lock()
_CONFIG_CACHE: dict[str, tuple[tuple[int, int] | None, AppConfig]] = {}


def _stat_key(path: Path) -> tuple[int, int] | None:
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size)


def load_raw_config() -> dict[str, Any]:
    """Return the config file as a plain dict (or {} when absent).

    Used by the config TUI so it can edit and round-trip the file while
    preserving any keys this version does not understand.
    """
    path = config_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PiRpcError(f"pi-as-mcp config is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PiRpcError(f"pi-as-mcp config must contain a JSON object: {path}")
    return payload


def save_raw_config(payload: dict[str, Any], *, path: Path | None = None) -> Path:
    """Validate then atomically write the raw config dict.

    Validation runs the same parser the daemon uses, so the TUI cannot persist a
    file the daemon would reject at request time.
    """
    target = path or config_path()
    parse_app_config(payload, path=target)  # raises PiRpcError on bad shape
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(target)
    # The TUI just rewrote the file; drop any cached parse so the next
    # load_config re-reads it (mtime/size would catch this too, but be explicit).
    with _CONFIG_LOCK:
        _CONFIG_CACHE.pop(str(target), None)
    return target


def load_config() -> AppConfig:
    path = config_path()
    cache_key = str(path)
    stat_key = _stat_key(path)

    with _CONFIG_LOCK:
        cached = _CONFIG_CACHE.get(cache_key)
        if cached is not None and cached[0] == stat_key:
            return cached[1]

    if stat_key is None:
        result = AppConfig(agents=AgentsConfig(models={}), skill=SkillConfig(), path=None)
        with _CONFIG_LOCK:
            _CONFIG_CACHE[cache_key] = (None, result)
        return result

    payload = load_raw_config()
    result = parse_app_config(payload, path=path)
    with _CONFIG_LOCK:
        _CONFIG_CACHE[cache_key] = (stat_key, result)
    return result


def parse_app_config(payload: dict[str, Any], *, path: Path) -> AppConfig:
    return AppConfig(
        agents=parse_agents_config(payload.get("agents"), path=path),
        skill=parse_skill_config(payload.get("skill"), path=path),
        path=path,
    )


def _parse_model_entry(model_ref: str, value: Any, *, path: Path) -> ModelConfig:
    # Shorthand: "ref": <int> is sugar for a concurrency limit only.
    if isinstance(value, bool):
        raise PiRpcError(
            f"pi-as-mcp config agents.models[{model_ref!r}] must be an object or positive integer: {path}"
        )
    if isinstance(value, int):
        if value < 1:
            raise PiRpcError(
                f"pi-as-mcp config agents.models[{model_ref!r}] integer limit must be positive: {path}"
            )
        return ModelConfig(limit=value)
    if not isinstance(value, dict):
        raise PiRpcError(
            f"pi-as-mcp config agents.models[{model_ref!r}] must be an object or positive integer: {path}"
        )

    limit = value.get("limit")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise PiRpcError(
            f"pi-as-mcp config agents.models[{model_ref!r}].limit must be a positive integer or null: {path}"
        )

    disabled = value.get("disabled", False)
    if not isinstance(disabled, bool):
        raise PiRpcError(
            f"pi-as-mcp config agents.models[{model_ref!r}].disabled must be a boolean: {path}"
        )

    description = value.get("description", "")
    if not isinstance(description, str):
        raise PiRpcError(
            f"pi-as-mcp config agents.models[{model_ref!r}].description must be a string: {path}"
        )

    return ModelConfig(limit=limit, disabled=disabled, description=description.strip())


def parse_agents_config(value: Any, *, path: Path) -> AgentsConfig:
    if value is None:
        return AgentsConfig(models={})
    if not isinstance(value, dict):
        raise PiRpcError(f"pi-as-mcp config agents must be an object: {path}")

    models: dict[str, ModelConfig] = {}

    # Legacy: agents.concurrency_limits.models is a flat {ref: int} map. Seed the
    # richer map from it so old config files keep working; agents.models wins.
    limits = value.get("concurrency_limits", value.get("concurrencyLimits", {}))
    if limits is None:
        limits = {}
    if not isinstance(limits, dict):
        raise PiRpcError(f"pi-as-mcp config agents.concurrency_limits must be an object: {path}")
    legacy_models = limits.get("models", {})
    if legacy_models is None:
        legacy_models = {}
    if not isinstance(legacy_models, dict):
        raise PiRpcError(f"pi-as-mcp config agents.concurrency_limits.models must be an object: {path}")
    for model_ref, limit in legacy_models.items():
        if not isinstance(model_ref, str) or not model_ref.strip():
            raise PiRpcError(
                f"pi-as-mcp config agents.concurrency_limits.models keys must be non-empty strings: {path}"
            )
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise PiRpcError(
                "pi-as-mcp config agents.concurrency_limits.models values "
                f"must be positive integers: {path}: {model_ref!r}"
            )
        models[model_ref.strip()] = ModelConfig(limit=limit)

    raw_models = value.get("models", {})
    if raw_models is None:
        raw_models = {}
    if not isinstance(raw_models, dict):
        raise PiRpcError(f"pi-as-mcp config agents.models must be an object: {path}")
    for model_ref, entry in raw_models.items():
        if not isinstance(model_ref, str) or not model_ref.strip():
            raise PiRpcError(f"pi-as-mcp config agents.models keys must be non-empty strings: {path}")
        models[model_ref.strip()] = _parse_model_entry(model_ref.strip(), entry, path=path)

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
        models=models,
        enable_score=enable_score,
        unsafe_read_only=unsafe_read_only,
        persist_sessions=persist_sessions,
        idle_eviction_seconds=float(idle_eviction_seconds),
    )


def parse_skill_config(value: Any, *, path: Path) -> SkillConfig:
    if value is None:
        return SkillConfig()
    if not isinstance(value, dict):
        raise PiRpcError(f"pi-as-mcp config skill must be an object: {path}")

    intro = value.get("intro", "")
    if not isinstance(intro, str):
        raise PiRpcError(f"pi-as-mcp config skill.intro must be a string: {path}")

    return SkillConfig(intro=intro)
