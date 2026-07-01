"""Interactive config editor for pi-as-mcp.

Edits the user-global ``~/.pi-as-mcp/config.json`` and regenerates the
cross-tool skill. It surfaces the full Pi model catalog (auto-discovered via
``pi --list-models``) so you can, per model: set a concurrency limit, write a
short human description/rules, or disable it (hide it from pi-as-mcp while
leaving it enabled in Pi). It also flags config entries for models Pi no longer
knows about, and toggles the global agent settings.

The editing model (:class:`ConfigDraft`) is deliberately Textual-free so it can
be unit-tested directly; :class:`PiConfigTui` is a thin widget layer over it.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Input, Label, Static, TextArea

from pi_as_mcp import skill
from pi_as_mcp.config import (
    AppConfig,
    config_path,
    load_raw_config,
    parse_app_config,
    save_raw_config,
)
from pi_as_mcp.pi_rpc import (
    CatalogModel,
    PiRpcError,
    PiRpcRunner,
    configured_model_specs,
)


@dataclass
class ModelRow:
    provider: str
    model: str
    in_pi: bool  # enabled in Pi settings (enabledModels/defaultModel)
    in_catalog: bool  # present in `pi --list-models`
    disabled: bool = False
    limit: int | None = None
    description: str = ""
    context: str = ""
    thinking: bool = False

    @property
    def ref(self) -> str:
        return f"{self.provider}/{self.model}" if self.provider else self.model

    @property
    def has_settings(self) -> bool:
        return self.disabled or self.limit is not None or bool(self.description)


@dataclass
class ConfigDraft:
    """Mutable, Textual-free editing state for the config file."""

    rows: list[ModelRow] = field(default_factory=list)
    enable_score: bool = False
    unsafe_read_only: bool = False
    persist_sessions: bool = True
    idle_eviction_seconds: float = 120.0
    skill_intro: str = ""
    # Config model keys with no corresponding model in the Pi catalog (point 3).
    orphans: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Original parsed file, kept so unknown top-level keys survive a round-trip.
    raw: dict[str, Any] = field(default_factory=dict)
    dirty: bool = False

    @classmethod
    def from_sources(
        cls,
        *,
        raw: dict[str, Any],
        config: AppConfig,
        catalog: list[CatalogModel],
        enabled_refs: set[str],
    ) -> ConfigDraft:
        agents = config.agents
        catalog_refs = {cat.ref for cat in catalog}
        catalog_names = {cat.model for cat in catalog}

        rows: list[ModelRow] = []
        for cat in sorted(catalog, key=lambda c: (c.provider, c.model)):
            limit_obj = agents.concurrency_limit_for_model(provider=cat.provider, model=cat.model)
            rows.append(
                ModelRow(
                    provider=cat.provider,
                    model=cat.model,
                    in_pi=cat.ref in enabled_refs,
                    in_catalog=True,
                    disabled=agents.is_model_disabled(provider=cat.provider, model=cat.model),
                    limit=limit_obj.limit if limit_obj else None,
                    description=agents.description_for_model(provider=cat.provider, model=cat.model),
                    context=cat.context,
                    thinking=cat.thinking,
                )
            )

        # Config keys that match no catalog model -> orphans (model removed from Pi).
        # A key can match either as a full provider/model ref or as a bare model
        # name — bare names may themselves contain "/" (e.g. org/model), so both
        # checks always apply.
        orphans: dict[str, dict[str, Any]] = {}
        for key, cfg in agents.models.items():
            known = key in catalog_refs or key in catalog_names
            if known:
                continue
            entry: dict[str, Any] = {}
            if cfg.limit is not None:
                entry["limit"] = cfg.limit
            if cfg.disabled:
                entry["disabled"] = True
            if cfg.description:
                entry["description"] = cfg.description
            orphans[key] = entry

        return cls(
            rows=rows,
            enable_score=agents.enable_score,
            unsafe_read_only=agents.unsafe_read_only,
            persist_sessions=agents.persist_sessions,
            idle_eviction_seconds=agents.idle_eviction_seconds,
            skill_intro=config.skill.intro,
            orphans=orphans,
            raw=raw,
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize back to config JSON, canonicalizing model keys to full refs.

        Legacy ``agents.concurrency_limits`` and bare model keys that mapped onto
        catalog rows are folded into ``agents.models`` (full ``provider/model``
        refs). Unknown top-level keys and config-only orphan models are preserved.
        """
        payload: dict[str, Any] = {k: v for k, v in self.raw.items() if k not in {"agents", "skill"}}

        models: dict[str, Any] = {}
        for row in self.rows:
            if not row.has_settings:
                continue
            entry: dict[str, Any] = {}
            if row.limit is not None:
                entry["limit"] = row.limit
            if row.disabled:
                entry["disabled"] = True
            if row.description:
                entry["description"] = row.description
            models[row.ref] = entry
        for key, entry in self.orphans.items():
            models[key] = dict(entry)

        agents: dict[str, Any] = {}
        # Preserve any unknown keys under agents (but not the migrated ones).
        raw_agents = self.raw.get("agents")
        if isinstance(raw_agents, dict):
            for k, v in raw_agents.items():
                if k not in {"models", "concurrency_limits", "concurrencyLimits",
                             "enable_score", "enableScore", "unsafe_read_only", "unsafeReadOnly",
                             "persist_sessions", "persistSessions",
                             "idle_eviction_seconds", "idleEvictionSeconds"}:
                    agents[k] = v
        agents["enable_score"] = self.enable_score
        agents["unsafe_read_only"] = self.unsafe_read_only
        agents["persist_sessions"] = self.persist_sessions
        agents["idle_eviction_seconds"] = self.idle_eviction_seconds
        if models:
            agents["models"] = models
        payload["agents"] = agents

        skill_cfg: dict[str, Any] = {}
        raw_skill = self.raw.get("skill")
        if isinstance(raw_skill, dict):
            for k, v in raw_skill.items():
                if k != "intro":
                    skill_cfg[k] = v
        if self.skill_intro.strip():
            skill_cfg["intro"] = self.skill_intro
        if skill_cfg:
            payload["skill"] = skill_cfg

        return payload

    def row_at(self, index: int) -> ModelRow | None:
        if 0 <= index < len(self.rows):
            return self.rows[index]
        return None


def load_draft(*, runner: PiRpcRunner | None = None) -> tuple[ConfigDraft, str | None]:
    """Build a draft from disk + Pi. Returns (draft, catalog_error_or_None)."""
    raw = load_raw_config()
    config = parse_app_config(raw, path=config_path())
    catalog: list[CatalogModel] = []
    catalog_error: str | None = None
    try:
        catalog = (runner or PiRpcRunner()).list_catalog()
    except (PiRpcError, OSError, subprocess.SubprocessError) as exc:
        catalog_error = str(exc)
    enabled_refs = {f"{s.provider}/{s.model}" for s in configured_model_specs(require=False)}
    draft = ConfigDraft.from_sources(
        raw=raw, config=config, catalog=catalog, enabled_refs=enabled_refs
    )
    return draft, catalog_error


class PromptModal(ModalScreen[str | None]):
    """A small input overlay. Returns the entered text, or None on cancel."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    CSS = """
    PromptModal {
        align: center middle;
    }
    #prompt-box {
        width: 80%;
        max-width: 100;
        height: auto;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
    }
    #prompt-box Input, #prompt-box TextArea {
        margin-top: 1;
    }
    #prompt-hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(self, title: str, *, value: str = "", multiline: bool = False) -> None:
        super().__init__()
        self._title = title
        self._value = value
        self._multiline = multiline

    def compose(self) -> ComposeResult:
        with Vertical(id="prompt-box"):
            yield Label(self._title)
            if self._multiline:
                yield TextArea(self._value, id="prompt-text")
                yield Label("Ctrl+S to save · Esc to cancel", id="prompt-hint")
            else:
                yield Input(value=self._value, id="prompt-input")
                yield Label("Enter to save · Esc to cancel", id="prompt-hint")

    def on_mount(self) -> None:
        target = "#prompt-text" if self._multiline else "#prompt-input"
        self.query_one(target).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def on_key(self, event: Any) -> None:
        if self._multiline and event.key == "ctrl+s":
            event.stop()
            self.dismiss(self.query_one("#prompt-text", TextArea).text)

    def action_cancel(self) -> None:
        self.dismiss(None)


class PiConfigTui(App[None]):
    TITLE = "pi-as-mcp config"
    BINDINGS = [
        ("d", "toggle_disabled", "Disable"),
        ("l", "set_limit", "Limit"),
        ("e", "edit_description", "Describe"),
        ("u", "toggle_unsafe", "Unsafe RO"),
        ("c", "toggle_score", "Score"),
        ("p", "toggle_persist", "Persist"),
        ("i", "set_idle", "Idle evict"),
        ("t", "edit_intro", "Skill intro"),
        ("w", "save", "Save"),
        ("r", "reload", "Reload"),
        ("q", "quit", "Quit"),
    ]

    CSS = """
    Screen { layout: vertical; }
    #title { height: 1; text-style: bold; }
    #main { height: 1fr; }
    #models { width: 2fr; height: 1fr; }
    #side { width: 1fr; height: 1fr; padding: 0 1; border-left: solid $foreground 25%; }
    #status { height: 1; color: $text-muted; }
    """

    def __init__(self, *, runner: PiRpcRunner | None = None) -> None:
        super().__init__()
        self._runner = runner
        self.draft: ConfigDraft = ConfigDraft()
        self.catalog_error: str | None = None
        self.status: str = ""

    # --- lifecycle -------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static("pi-as-mcp config", id="title")
        with Horizontal(id="main"):
            yield DataTable(id="models", cursor_type="row", zebra_stripes=True)
            with VerticalScroll(id="side"):
                yield Static(id="side-body")
        yield Static(id="status")

    def on_mount(self) -> None:
        table = self.query_one("#models", DataTable)
        table.add_columns("model", "pi", "mcp", "limit", "description")
        self.reload()
        table.focus()

    # --- data ------------------------------------------------------------

    def reload(self) -> None:
        try:
            self.draft, self.catalog_error = load_draft(runner=self._runner)
        except (PiRpcError, OSError) as exc:
            self.draft = ConfigDraft()
            self.catalog_error = str(exc)
            self.status = f"config error: {exc}"
        else:
            self.status = "loaded" if not self.catalog_error else f"catalog unavailable: {self.catalog_error}"
        self.render_all()

    def render_all(self) -> None:
        self.render_table()
        self.render_side()
        self.render_status()

    def render_table(self) -> None:
        table = self.query_one("#models", DataTable)
        cursor = table.cursor_row
        table.clear()
        for row in self.draft.rows:
            table.add_row(*self._row_cells(row))
        for key in self.draft.orphans:
            table.add_row(
                Text(key, style="red"),
                Text("—", style="red"),
                Text("missing", style="bold red"),
                "",
                Text("not in Pi catalog", style="red"),
            )
        if table.row_count:
            table.move_cursor(row=min(cursor, table.row_count - 1))

    def _row_cells(self, row: ModelRow) -> tuple[Any, ...]:
        pi = Text("on", style="green") if row.in_pi else Text("off", style="dim")
        if row.disabled:
            mcp = Text("disabled", style="bold red")
        elif row.in_pi:
            mcp = Text("exposed", style="green")
        else:
            mcp = Text("—", style="dim")
        limit = "" if row.limit is None else str(row.limit)
        desc = row.description or ""
        if len(desc) > 48:
            desc = desc[:45] + "..."
        return (row.ref, pi, mcp, limit, desc)

    def render_side(self) -> None:
        d = self.draft
        lines = [
            "[b]Global settings[/b]",
            f"  unsafe read-only (u): {self._onoff(d.unsafe_read_only)}",
            f"  enable score (c):     {self._onoff(d.enable_score)}",
            f"  persist sessions (p): {self._onoff(d.persist_sessions)}",
            f"  idle evict secs (i):  {d.idle_eviction_seconds:g}",
            "",
            "[b]Skill (t = edit intro)[/b]",
            f"  intro: {'custom' if d.skill_intro.strip() else 'default'}",
            "  (served live over MCP; no files)",
            "",
            "[b]Per-model keys[/b]",
            "  d disable · l limit · e describe",
            "",
            "[b]Save[/b]",
            "  w save",
        ]
        if d.orphans:
            lines += ["", f"[red]⚠ {len(d.orphans)} config model(s) not in Pi catalog[/red]"]
        if self.catalog_error:
            lines += ["", f"[red]catalog: {self.catalog_error}[/red]"]
        self.query_one("#side-body", Static).update("\n".join(lines))

    def render_status(self) -> None:
        dirty = " · UNSAVED" if self.draft.dirty else ""
        self.query_one("#status", Static).update(f"{self.status}{dirty}")

    @staticmethod
    def _onoff(value: bool) -> str:
        return "[green]on[/green]" if value else "[dim]off[/dim]"

    # --- selection -------------------------------------------------------

    def selected_row(self) -> ModelRow | None:
        table = self.query_one("#models", DataTable)
        return self.draft.row_at(table.cursor_row)

    def mark_dirty(self, message: str) -> None:
        self.draft.dirty = True
        self.status = message
        self.render_all()

    # --- per-model actions ----------------------------------------------

    def action_toggle_disabled(self) -> None:
        row = self.selected_row()
        if row is None:
            return
        row.disabled = not row.disabled
        self.mark_dirty(f"{row.ref}: {'disabled' if row.disabled else 'enabled'}")

    def action_set_limit(self) -> None:
        row = self.selected_row()
        if row is None:
            return
        current = "" if row.limit is None else str(row.limit)

        def done(value: str | None) -> None:
            if value is None:
                return
            value = value.strip()
            if not value:
                row.limit = None
                self.mark_dirty(f"{row.ref}: limit cleared")
                return
            try:
                parsed = int(value)
            except ValueError:
                self.notify("limit must be a positive integer", severity="error")
                return
            if parsed < 1:
                self.notify("limit must be a positive integer", severity="error")
                return
            row.limit = parsed
            self.mark_dirty(f"{row.ref}: limit {parsed}")

        self.push_screen(PromptModal(f"Concurrency limit for {row.ref} (blank = none)", value=current), done)

    def action_edit_description(self) -> None:
        row = self.selected_row()
        if row is None:
            return

        def done(value: str | None) -> None:
            if value is None:
                return
            row.description = value.strip()
            self.mark_dirty(f"{row.ref}: description updated")

        self.push_screen(
            PromptModal(f"Description / rules for {row.ref}", value=row.description, multiline=True),
            done,
        )

    # --- global actions --------------------------------------------------

    def action_toggle_unsafe(self) -> None:
        self.draft.unsafe_read_only = not self.draft.unsafe_read_only
        self.mark_dirty(f"unsafe_read_only: {self.draft.unsafe_read_only}")

    def action_toggle_score(self) -> None:
        self.draft.enable_score = not self.draft.enable_score
        self.mark_dirty(f"enable_score: {self.draft.enable_score}")

    def action_toggle_persist(self) -> None:
        self.draft.persist_sessions = not self.draft.persist_sessions
        self.mark_dirty(f"persist_sessions: {self.draft.persist_sessions}")

    def action_set_idle(self) -> None:
        def done(value: str | None) -> None:
            if value is None:
                return
            try:
                parsed = float(value.strip())
            except ValueError:
                self.notify("idle eviction must be a non-negative number", severity="error")
                return
            if parsed < 0:
                self.notify("idle eviction must be a non-negative number", severity="error")
                return
            self.draft.idle_eviction_seconds = parsed
            self.mark_dirty(f"idle_eviction_seconds: {parsed:g}")

        self.push_screen(
            PromptModal("Idle eviction seconds (0 = never)", value=f"{self.draft.idle_eviction_seconds:g}"),
            done,
        )

    def action_edit_intro(self) -> None:
        def done(value: str | None) -> None:
            if value is None:
                return
            self.draft.skill_intro = value
            self.mark_dirty("skill intro updated")

        current = self.draft.skill_intro or skill.DEFAULT_INTRO
        self.push_screen(PromptModal("Skill intro (blank = built-in default)", value=current, multiline=True), done)

    # --- persistence -----------------------------------------------------

    def _save(self) -> bool:
        try:
            path = save_raw_config(self.draft.to_payload())
        except (PiRpcError, OSError) as exc:
            self.notify(f"save failed: {exc}", severity="error")
            return False
        self.draft.dirty = False
        self.status = f"saved {path}"
        return True

    def action_save(self) -> None:
        if self._save():
            self.render_all()

    def action_reload(self) -> None:
        self.reload()


def run_config_tui() -> None:
    PiConfigTui().run()


def main() -> None:
    run_config_tui()
