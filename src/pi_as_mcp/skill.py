"""Generate the cross-tool "cheap sub-agents" skill from live configuration.

The skill is MCP-provided: the running server publishes it as the
``skill://pi-as-mcp/...`` resource and as the server's ``instructions``, so any
MCP host picks it up with no install or on-disk files.

The model roster is auto-generated (never hand-maintained): it is Pi's enabled
models, minus the ones disabled in pi-as-mcp config, annotated with each model's
configured description, concurrency limit, and discovered capabilities. Only the
intro prose is operator-customizable.
"""

from __future__ import annotations

from dataclasses import dataclass

from pi_as_mcp.config import AppConfig, load_config
from pi_as_mcp.pi_rpc import (
    CatalogModel,
    PiRpcError,
    PiRpcRunner,
    configured_model_specs,
)

SKILL_RESOURCE_URI = "skill://pi-as-mcp/cheap-subagents"

DEFAULT_INTRO = """\
You have the `pi-as-mcp` MCP tool available for delegating bounded tasks to
cheaper sub-agent models. You should use it.

These models are free to run and act as full coding agents. They are less
capable than you, but they can do work in parallel while you keep going.

Don't ask them trivial questions you already know the answer to, and don't
delegate important work you should be doing yourself. Use them to work in
parallel, as you work, to explore every nice-to-have you can think of.

None of the models can search the web. If you are going to have them write code,
it is strongly recommended to give them a git worktree to work in."""

USAGE = f"""\
## How to delegate

Call `delegate` with a `prompt`, a `cwd`, a `model`, and a `tool_mode`. It
returns immediately with an `agent_id` and a short `monitor_command` (`piw
<agent_id>`). Run that command in a background shell; it stays quiet while the
worker runs and prints one compact JSON object when the turn finishes — so you
do not need to poll.

Tool modes: `none`, `read-only` (read/grep/find/ls), `write` (adds edit/write),
`full` (adds bash). Prefer `read-only` for scouting.

Example call:

```json
{{
  "cwd": "/path/to/workspace",
  "model": "<model>",
  "tool_mode": "read-only",
  "prompt": "Explore the repo and report how data files are parsed."
}}
```

Then run its `monitor_command` (e.g. `piw <agent_id>`) in the background and
keep working. Use `agent_reply` to send a follow-up, `agent_peek` to check
state without waiting, and `agent_stop` to abort.

Read `{SKILL_RESOURCE_URI}` (or call the `models` tool) for the current
roster at any time."""


@dataclass(frozen=True)
class SkillModel:
    provider: str
    model: str
    description: str
    limit: int | None
    context: str
    thinking: bool

    @property
    def ref(self) -> str:
        return f"{self.provider}/{self.model}"

    def render_line(self) -> str:
        traits: list[str] = []
        if self.context:
            traits.append(f"{self.context} ctx")
        if self.thinking:
            traits.append("thinking")
        if self.limit is not None:
            traits.append(f"max {self.limit} concurrent")
        suffix = f" ({' · '.join(traits)})" if traits else ""
        desc = f" — {self.description}" if self.description else ""
        return f"- `{self.ref}`{suffix}{desc}"


def _catalog_index(catalog: list[CatalogModel] | None) -> dict[str, CatalogModel]:
    return {row.ref: row for row in (catalog or [])}


def collect_skill_models(
    config: AppConfig,
    *,
    catalog: list[CatalogModel] | None = None,
) -> list[SkillModel]:
    """Models pi-as-mcp exposes: Pi-enabled minus pi-as-mcp-disabled."""
    index = _catalog_index(catalog)
    agents = config.agents
    models: list[SkillModel] = []
    for spec in configured_model_specs(require=False):
        if agents.is_model_disabled(provider=spec.provider, model=spec.model):
            continue
        limit_obj = agents.concurrency_limit_for_model(provider=spec.provider, model=spec.model)
        cat = index.get(f"{spec.provider}/{spec.model}")
        models.append(
            SkillModel(
                provider=spec.provider,
                model=spec.model,
                description=agents.description_for_model(provider=spec.provider, model=spec.model),
                limit=limit_obj.limit if limit_obj else None,
                context=cat.context if cat else "",
                thinking=cat.thinking if cat else False,
            )
        )
    return models


def _safe_catalog(runner: PiRpcRunner | None) -> list[CatalogModel] | None:
    """Best-effort catalog fetch; capabilities are decoration, never required."""
    try:
        return (runner or PiRpcRunner()).list_catalog()
    except (PiRpcError, OSError):
        return None


def _intro(config: AppConfig) -> str:
    intro = config.skill.intro.strip()
    return intro or DEFAULT_INTRO


def _models_section(models: list[SkillModel]) -> str:
    if not models:
        return (
            "## Available sub-agent models\n\n"
            "_No models are currently enabled in Pi settings (or all are disabled "
            "in pi-as-mcp config). Enable models in Pi, then reconnect the server._"
        )
    lines = "\n".join(model.render_line() for model in models)
    return f"## Available sub-agent models\n\n{lines}"


def render_skill_body(
    config: AppConfig | None = None,
    *,
    runner: PiRpcRunner | None = None,
    catalog: list[CatalogModel] | None = None,
) -> str:
    """Full skill: intro + auto-generated roster + usage. Served as the resource."""
    config = config or load_config()
    if catalog is None:
        catalog = _safe_catalog(runner)
    models = collect_skill_models(config, catalog=catalog)
    return "\n\n".join([_intro(config), _models_section(models), USAGE])


def render_server_instructions(
    config: AppConfig | None = None,
    *,
    runner: PiRpcRunner | None = None,
    catalog: list[CatalogModel] | None = None,
) -> str:
    """Compact variant for the always-in-context MCP server instructions."""
    config = config or load_config()
    if catalog is None:
        catalog = _safe_catalog(runner)
    models = collect_skill_models(config, catalog=catalog)
    pointer = (
        f"Read the resource `{SKILL_RESOURCE_URI}` (or call the `models` tool) "
        "for usage details and the current roster."
    )
    return "\n\n".join([_intro(config), _models_section(models), pointer])
