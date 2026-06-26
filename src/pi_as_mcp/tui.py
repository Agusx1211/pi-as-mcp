from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from rich import box
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import RichLog, Static, Tree
from textual.widgets.tree import TreeNode

from pi_as_mcp.daemon_client import DaemonClient, DaemonClientError


def short_text(value: Any, *, limit: int = 80) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def requester_display(requester: dict[str, Any] | None) -> str:
    if not requester:
        return ""
    display = str(requester.get("display") or "").strip()
    if display:
        return display
    kind = str(requester.get("kind") or "").strip()
    instance = str(requester.get("instance") or "").strip()
    return " ".join(part for part in (kind, instance) if part)


def render_json(value: Any) -> Syntax:
    return Syntax(json.dumps(value, indent=2, ensure_ascii=False), "json", word_wrap=True)


RUNNING_FRAMES = ("|", "/", "-", "\\")


def status_icon(status: Any, *, frame: int = 0) -> str:
    normalized = str(status or "").lower()
    if normalized in {"starting", "running"}:
        return RUNNING_FRAMES[frame % len(RUNNING_FRAMES)]
    return {
        "idle": "✓",
        "stopped": "■",
        "timeout": "!",
        "error": "x",
        "exited": ".",
    }.get(str(status or "").lower(), "•")


def model_label(data: dict[str, Any] | None) -> str:
    if not data:
        return ""
    provider = str(data.get("provider") or "").strip()
    model = str(data.get("model") or "").strip()
    if provider and model:
        return f"{provider}/{model}"
    return model or str(data.get("model_alias") or "").strip()


def number_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return None
    return None


def format_count(value: Any) -> str:
    number = number_value(value)
    if number is None:
        return "0"
    return f"{int(number):,}"


def format_percent(value: Any) -> str:
    number = number_value(value)
    if number is None:
        return "--"
    if 0 < number <= 1:
        number *= 100
    return f"{number:.1f}%"


def format_rate(value: Any) -> str:
    number = number_value(value)
    if number is None:
        return "--"
    return f"{number:.1f}/s"


def usage_stats_label(detail: dict[str, Any]) -> str:
    usage = detail.get("usage") if isinstance(detail.get("usage"), dict) else {}
    cache_tokens = (number_value(usage.get("cache_read_tokens")) or 0) + (
        number_value(usage.get("cache_write_tokens")) or 0
    )
    return "  ".join(
        [
            f"turns {format_count(detail.get('turn_count') or 0)}",
            f"in {format_count(usage.get('input_tokens'))}",
            f"out {format_count(usage.get('output_tokens'))}",
            f"cache {format_count(cache_tokens)}",
            f"ctx {format_percent(usage.get('context_percent'))}",
            f"t/s {format_rate(usage.get('tokens_per_second'))}",
        ]
    )


def observed_label(stats: dict[str, Any] | None) -> str:
    if not stats:
        return "unknown"
    if stats.get("observed_by_parent"):
        via = str(stats.get("observed_via") or "output")
        count = int(number_value(stats.get("observation_count")) or 0)
        suffix = f" · {count}x" if count > 1 else ""
        return f"yes via {via}{suffix}"
    return "no"


def watching_label(detail: dict[str, Any] | None) -> str:
    """Whether a parent is currently tailing this agent with `piw`."""
    if not detail:
        return "unknown"
    count = int(number_value(detail.get("active_listeners")) or 0)
    if count <= 0:
        return "no"
    suffix = f" · {count}x" if count > 1 else ""
    return f"yes via piw{suffix}"


def score_label(stats: dict[str, Any] | None) -> str:
    if not stats:
        return "--"
    latest = stats.get("latest_score") if isinstance(stats.get("latest_score"), dict) else None
    if not latest:
        return "--"
    score = latest.get("score")
    category = short_text(latest.get("category"), limit=24)
    sentiment = str(latest.get("sentiment") or "")
    parts = [str(score), category, sentiment]
    return " · ".join(part for part in parts if part)


def stats_summary_label(stats: dict[str, Any] | None) -> str:
    if not stats:
        return "agents 0  observed 0  unobserved 0  scores 0"
    avg_score = stats.get("average_score")
    avg = f"{number_value(avg_score):.2f}" if number_value(avg_score) is not None else "--"
    return "  ".join(
        [
            f"agents {format_count(stats.get('total_agents'))}",
            f"observed {format_count(stats.get('observed_agents'))}",
            f"unobserved {format_count(stats.get('unobserved_agents'))}",
            f"scores {format_count(stats.get('scores'))}",
            f"avg score {avg}",
        ]
    )


def agent_spawn_rank(agent: dict[str, Any]) -> float:
    created_at = number_value(agent.get("created_at"))
    if created_at is not None:
        return created_at
    total_seconds = number_value(agent.get("total_seconds"))
    if total_seconds is not None:
        return -total_seconds
    return 0.0


def requester_key(agent: dict[str, Any]) -> str:
    requester = agent.get("requester") if isinstance(agent.get("requester"), dict) else {}
    key = str(requester.get("scope_id") or agent.get("parent_scope_id") or requester_display(requester)).strip()
    return key or "unknown"


def grouped_agents(agents: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels: dict[str, str] = {}
    for agent in agents:
        key = requester_key(agent)
        requester = agent.get("requester") if isinstance(agent.get("requester"), dict) else {}
        labels.setdefault(key, requester_display(requester) or key)
        groups[key].append(agent)

    rows: list[tuple[str, str, list[dict[str, Any]]]] = []
    for key, grouped in groups.items():
        grouped.sort(key=lambda item: (-agent_spawn_rank(item), str(item.get("agent_id") or "")))
        rows.append((key, labels[key], grouped))
    rows.sort(
        key=lambda item: (
            -max((agent_spawn_rank(agent) for agent in item[2]), default=0.0),
            item[1].lower(),
        )
    )
    return rows


class PiAgentTui(App[None]):
    TITLE = "Pi Agents"
    BINDINGS = [
        ("left", "sidebar_left", "Collapse"),
        ("right", "sidebar_right", "Expand"),
        ("r", "refresh", "Refresh"),
        ("q", "quit", "Quit"),
    ]

    CSS = """
    Screen {
        layout: vertical;
    }

    #title {
        height: 1;
        text-style: bold;
    }

    #main {
        height: 1fr;
    }

    #agent-tree {
        width: 44;
        height: 1fr;
        padding: 0 1 0 0;
        border-right: solid $foreground 25%;
        overflow-y: auto;
    }

    #detail {
        width: 1fr;
        height: 1fr;
        padding-left: 1;
    }

    #meta {
        height: 10;
        padding: 0 1;
    }

    #log {
        height: 1fr;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.client = DaemonClient()
        self.agents: list[dict[str, Any]] = []
        self.stats_summary: dict[str, Any] = {}
        self.selected_agent_id: str | None = None
        self.detail: dict[str, Any] | None = None
        self.error: str | None = None
        self.refresh_failed = False
        self.collapsed_parent_ids: set[str] = set()
        self.sidebar_cursor: tuple[str, str] | None = None
        self.status_frame = 0
        self._tree_structure_signature: tuple[tuple[str, tuple[str, ...]], ...] | None = None
        self._last_log_signature: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("Pi Agents", id="title")
        with Horizontal(id="main"):
            yield Tree("agents", id="agent-tree")
            with Vertical(id="detail"):
                yield Static(id="meta")
                yield RichLog(id="log", wrap=True, highlight=True, markup=False, auto_scroll=False)

    def on_mount(self) -> None:
        tree = self.agent_tree
        tree.show_root = False
        tree.auto_expand = False
        tree.show_horizontal_scrollbar = False
        tree.focus()
        self.refresh_data()
        self.set_interval(1.0, self.refresh_data)
        self.set_interval(0.18, self.animate_status)

    @property
    def agent_tree(self) -> Tree[dict[str, Any]]:
        return self.query_one("#agent-tree", Tree)

    def action_refresh(self) -> None:
        self.refresh_data()

    def animate_status(self) -> None:
        if not self.has_animated_status():
            return
        self.status_frame += 1
        try:
            self.render_tree()
            self.query_one("#meta", Static).update(self.meta_renderable())
        except NoMatches:
            return

    def has_animated_status(self) -> bool:
        if any(str(agent.get("status") or "").lower() in {"starting", "running"} for agent in self.agents):
            return True
        if isinstance(self.detail, dict):
            return str(self.detail.get("status") or "").lower() in {"starting", "running"}
        return False

    def action_sidebar_left(self) -> None:
        node = self.agent_tree.cursor_node
        if node is None:
            return
        if node.children and node.is_expanded:
            node.collapse()
            self.remember_parent_state(node)
            return
        parent = node.parent
        if parent is not None and not parent.is_root:
            self.agent_tree.select_node(parent)

    def action_sidebar_right(self) -> None:
        node = self.agent_tree.cursor_node
        if node is None:
            return
        if node.children:
            if node.is_collapsed:
                node.expand()
                self.remember_parent_state(node)
            elif node.children:
                self.agent_tree.select_node(node.children[0])

    def refresh_data(self) -> None:
        try:
            data = self.client.request("tui_summary", request_timeout_seconds=4)
            agents = data.get("agents")
            self.agents = agents if isinstance(agents, list) else []
            stats = data.get("stats")
            self.stats_summary = stats if isinstance(stats, dict) else {}
            self.error = None
            self.refresh_failed = False
            self.reconcile_selection()
            self.load_detail()
        except (DaemonClientError, OSError, json.JSONDecodeError) as exc:
            # A single slow or timed-out poll must not blank the whole dashboard
            # and then repaint it a second later (the "things keep disappearing"
            # flicker). Keep the last-known agents and selection, and only fall
            # back to a hard error state when there is nothing to show yet.
            if self.agents:
                self.refresh_failed = True
            else:
                self.error = str(exc)
                self.stats_summary = {}
                self.detail = None
                self.selected_agent_id = None
        self.render_all()

    def reconcile_selection(self) -> None:
        if not self.agents:
            self.selected_agent_id = None
            return

        ids = [str(agent.get("agent_id") or "") for agent in self.agents]
        if self.selected_agent_id in ids:
            return
        self.selected_agent_id = ids[0]

    def load_detail(self) -> None:
        if not self.selected_agent_id:
            self.detail = None
            return
        try:
            self.detail = self.client.request(
                "inspect",
                request_timeout_seconds=4,
                agent_id=self.selected_agent_id,
                include_events=True,
                verbosity="debug",
                # The dashboard is a passive viewer; polling inspect must not mark
                # the selected agent as observed (that count would climb every tick).
                observe=False,
            )
        except (DaemonClientError, OSError, json.JSONDecodeError) as exc:
            self.detail = {"agent_id": self.selected_agent_id, "error": str(exc)}

    def render_all(self) -> None:
        # "live" is what the tree can actually show (daemon-owned sessions right
        # now); the stats label counts every agent ever recorded. Keeping them
        # separate stops the header from implying the tree is missing agents.
        title = f"Pi Agents | live {len(self.agents)} · {stats_summary_label(self.stats_summary)}"
        if self.refresh_failed:
            title += "  · refresh failed, showing last known"
        self.query_one("#title", Static).update(title)
        self.render_tree()
        self.query_one("#meta", Static).update(self.meta_renderable())
        self.render_log_if_changed()

    def render_tree(self) -> None:
        tree = self.agent_tree
        groups = grouped_agents(self.agents)
        structure_signature = tuple(
            (parent_id, tuple(str(agent.get("agent_id") or "") for agent in parent_agents))
            for parent_id, _parent_label, parent_agents in groups
        )

        if self._tree_structure_signature == structure_signature and tree.root.children and not self.error:
            self.update_tree_labels(groups)
            tree.scroll_x = 0
            return

        self._tree_structure_signature = structure_signature
        cursor_type, cursor_value = self.sidebar_cursor or ("agent", self.selected_agent_id or "")

        tree.root.remove_children()
        tree.root.set_label("agents")

        if self.error:
            self._tree_structure_signature = None
            tree.root.add_leaf(f"❌ daemon error: {short_text(self.error, limit=28)}", data={"type": "error"})
            return

        if not self.agents:
            self._tree_structure_signature = None
            tree.root.add_leaf("no live Pi agents", data={"type": "empty"})
            return

        selected_node: TreeNode[dict[str, Any]] | None = None
        cursor_node: TreeNode[dict[str, Any]] | None = None
        for parent_id, parent_label, parent_agents in groups:
            parent_node = tree.root.add(
                f"{short_text(parent_label, limit=32)} ({len(parent_agents)})",
                data={"type": "parent", "parent_id": parent_id},
                expand=parent_id not in self.collapsed_parent_ids,
            )
            if cursor_type == "parent" and cursor_value == parent_id:
                cursor_node = parent_node
            for agent in parent_agents:
                agent_id = str(agent.get("agent_id") or "")
                label = self.agent_node_label(agent)
                node = parent_node.add_leaf(label, data={"type": "agent", "agent_id": agent_id})
                if cursor_type == "agent" and cursor_value == agent_id:
                    cursor_node = node
                if agent_id == self.selected_agent_id:
                    selected_node = node

        tree.root.expand()
        target_node = cursor_node or selected_node
        if target_node is not None:
            target_data = target_node.data if isinstance(target_node.data, dict) else {}
            if target_data.get("type") == "agent":
                tree.select_node(target_node)
            else:
                tree.move_cursor(target_node, animate=False)
            tree.scroll_x = 0

    def update_tree_labels(self, groups: list[tuple[str, str, list[dict[str, Any]]]]) -> None:
        parent_nodes = [node for node in self.agent_tree.root.children if isinstance(node.data, dict)]
        for parent_node, (_parent_id, parent_label, parent_agents) in zip(parent_nodes, groups, strict=False):
            parent_node.set_label(f"{short_text(parent_label, limit=32)} ({len(parent_agents)})")
            for agent_node, agent in zip(parent_node.children, parent_agents, strict=False):
                agent_node.set_label(self.agent_node_label(agent))

    def agent_node_label(self, agent: dict[str, Any]) -> Text:
        agent_id = str(agent.get("agent_id") or "")[:8]
        model = short_text(str(agent.get("model") or ""), limit=22)
        label = Text()
        label.append(f"{status_icon(agent.get('status'), frame=self.status_frame)} ")
        if model:
            label.append(model)
            label.append("  ")
        label.append(agent_id, style="dim")
        return label

    def remember_parent_state(self, node: TreeNode[dict[str, Any]]) -> None:
        data = node.data if isinstance(node.data, dict) else {}
        parent_id = str(data.get("parent_id") or "")
        if not parent_id:
            return
        if node.is_collapsed:
            self.collapsed_parent_ids.add(parent_id)
        else:
            self.collapsed_parent_ids.discard(parent_id)

    def on_tree_node_collapsed(self, event: Tree.NodeCollapsed[dict[str, Any]]) -> None:
        self.remember_parent_state(event.node)

    def on_tree_node_expanded(self, event: Tree.NodeExpanded[dict[str, Any]]) -> None:
        self.remember_parent_state(event.node)

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[dict[str, Any]]) -> None:
        self.remember_sidebar_cursor(event.node)
        self.select_agent_from_node(event.node)

    def on_tree_node_selected(self, event: Tree.NodeSelected[dict[str, Any]]) -> None:
        self.remember_sidebar_cursor(event.node)
        self.select_agent_from_node(event.node)

    def remember_sidebar_cursor(self, node: TreeNode[dict[str, Any]]) -> None:
        data = node.data if isinstance(node.data, dict) else {}
        node_type = data.get("type")
        if node_type == "agent":
            agent_id = str(data.get("agent_id") or "")
            if agent_id:
                self.sidebar_cursor = ("agent", agent_id)
        elif node_type == "parent":
            parent_id = str(data.get("parent_id") or "")
            if parent_id:
                self.sidebar_cursor = ("parent", parent_id)

    def select_agent_from_node(self, node: TreeNode[dict[str, Any]]) -> None:
        data = node.data if isinstance(node.data, dict) else {}
        if data.get("type") != "agent":
            return
        agent_id = str(data.get("agent_id") or "")
        if not agent_id or agent_id == self.selected_agent_id:
            return
        self.selected_agent_id = agent_id
        self.load_detail()
        self.query_one("#meta", Static).update(self.meta_renderable())
        self.render_log_if_changed(force=True)

    def meta_renderable(self) -> Table | Text:
        if self.error:
            return Text("daemon unavailable", style="bold")

        detail = self.detail
        if not detail:
            return Text(stats_summary_label(self.stats_summary), style="dim")

        requester = detail.get("requester") if isinstance(detail.get("requester"), dict) else {}
        lineage = requester.get("lineage") if isinstance(requester, dict) else []
        parent = ""
        if isinstance(lineage, list) and len(lineage) > 1:
            parent = short_text(lineage[1].get("command"), limit=72)

        table = Table.grid(expand=True)
        table.add_column("field", style="bold")
        table.add_column("value")
        table.add_row("model", model_label(detail) or "unknown")
        table.add_row("agent", str(detail.get("agent_id") or ""))
        table.add_row("state", status_icon(detail.get("status"), frame=self.status_frame))
        table.add_row("requester", requester_display(requester))
        table.add_row("parent process", parent)
        table.add_row("cwd", str(detail.get("cwd") or ""))
        table.add_row("stats", usage_stats_label(detail))
        stats = detail.get("stats") if isinstance(detail.get("stats"), dict) else {}
        table.add_row("observed", observed_label(stats))
        table.add_row("watching", watching_label(detail))
        table.add_row("score", score_label(stats))
        table.add_row("tool mode", str(detail.get("tool_mode") or ""))
        return table

    def log_signature(self) -> str:
        payload = {
            "agent_id": self.selected_agent_id,
            "error": self.error,
            "detail_error": self.detail.get("error") if isinstance(self.detail, dict) else None,
            "prompts": self.detail.get("prompts") if isinstance(self.detail, dict) else None,
            "transcript": self.detail.get("transcript") if isinstance(self.detail, dict) else None,
            "final_text": self.detail.get("final_text") if isinstance(self.detail, dict) else None,
            "stats": self.detail.get("stats") if isinstance(self.detail, dict) else None,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)

    def render_log_if_changed(self, *, force: bool = False) -> None:
        signature = self.log_signature()
        if not force and signature == self._last_log_signature:
            return
        self._last_log_signature = signature
        self.render_log()

    def render_log(self) -> None:
        log = self.query_one("#log", RichLog)
        was_at_bottom = log.is_vertical_scroll_end
        old_scroll_y = log.scroll_y
        log.clear()

        if self.error:
            log.write(Text(self.error, style="bold"))
            self.restore_log_scroll(log, was_at_bottom=was_at_bottom, old_scroll_y=old_scroll_y)
            return
        if not self.detail:
            log.write(Text("No live Pi agents", style="dim"))
            self.restore_log_scroll(log, was_at_bottom=was_at_bottom, old_scroll_y=old_scroll_y)
            return

        detail = self.detail
        error = detail.get("error")
        if error:
            log.write(Panel(str(error), title="Error", box=box.SQUARE))

        stats = detail.get("stats") if isinstance(detail.get("stats"), dict) else {}
        latest_score = stats.get("latest_score") if isinstance(stats.get("latest_score"), dict) else None
        if latest_score:
            score_text = (
                f"{latest_score.get('score')} · {latest_score.get('category') or ''}\n"
                f"{latest_score.get('comment') or ''}"
            )
            log.write(Panel(score_text.strip(), title="Score", box=box.SQUARE))

        if stats:
            log.write(Panel(render_json(stats), title="Stored stats", box=box.SQUARE))

        prompts = detail.get("prompts") if isinstance(detail.get("prompts"), list) else []
        if prompts:
            first = prompts[0]
            if isinstance(first, dict):
                log.write(
                    Panel(
                        Markdown(str(first.get("text") or "")),
                        title=f"Initial request · turn {first.get('turn')}",
                        box=box.SQUARE,
                    )
                )

        transcript = detail.get("transcript") if isinstance(detail.get("transcript"), list) else []
        if transcript:
            for item in transcript:
                if isinstance(item, dict):
                    self.write_transcript_item(log, item)
            self.restore_log_scroll(log, was_at_bottom=was_at_bottom, old_scroll_y=old_scroll_y)
            return

        final_text = str(detail.get("final_text") or "")
        if final_text:
            log.write(Panel(Markdown(final_text), title="Final text", box=box.SQUARE))
        else:
            log.write(Text("No transcript events yet", style="dim"))
        self.restore_log_scroll(log, was_at_bottom=was_at_bottom, old_scroll_y=old_scroll_y)

    def restore_log_scroll(self, log: RichLog, *, was_at_bottom: bool, old_scroll_y: float) -> None:
        if was_at_bottom:
            log.scroll_end(animate=False)
        else:
            log.scroll_to(y=old_scroll_y, animate=False, force=True)

    def write_transcript_item(self, log: RichLog, item: dict[str, Any]) -> None:
        kind = item.get("kind")
        turn = item.get("turn")
        if kind == "prompt":
            behavior = item.get("behavior") or "prompt"
            log.write(
                Panel(
                    Markdown(str(item.get("text") or "")),
                    title=f"Request · turn {turn} · {behavior}",
                    box=box.SQUARE,
                )
            )
            return

        if kind in {"thinking", "thinking_stream"}:
            suffix = " · streaming" if kind == "thinking_stream" else ""
            text = str(item.get("text") or "")
            body = Text(text, style="italic") if text else Text("…", style="dim italic")
            log.write(
                Panel(
                    body,
                    title=f"Thinking · turn {turn}{suffix}",
                    box=box.SQUARE,
                    border_style="dim",
                    style="dim",
                )
            )
            return

        if kind in {"message", "message_stream"}:
            role = str(item.get("role") or "assistant")
            suffix = " · streaming" if kind == "message_stream" else ""
            log.write(
                Panel(
                    Markdown(str(item.get("text") or "")),
                    title=f"{role} · turn {turn}{suffix}",
                    box=box.SQUARE,
                )
            )
            return

        if kind == "tool_call":
            name = str(item.get("tool_name") or "tool")
            args = item.get("tool_args") if isinstance(item.get("tool_args"), dict) else {}
            log.write(
                Panel(
                    render_json(args),
                    title=f"Tool call · {name} · turn {turn}",
                    box=box.SQUARE,
                )
            )
            return

        if kind == "tool_result":
            name = str(item.get("tool_name") or "tool")
            failed = " · error" if bool(item.get("is_error")) else ""
            result = str(item.get("result_preview") or "")
            log.write(
                Panel(
                    result or Text("No text result", style="dim"),
                    title=f"Tool result · {name} · turn {turn}{failed}",
                    box=box.SQUARE,
                )
            )
            return

        log.write(Panel(render_json(item), title=str(kind or "event"), box=box.SQUARE))


def run_tui() -> None:
    PiAgentTui().run()


def main() -> None:
    run_tui()
