from __future__ import annotations

import copy
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

STATS_DIR_ENV = "PI_AS_MCP_STATS_DIR"
DEFAULT_STATS_DIR = "~/.pi-as-mcp"
AGENT_EVENTS_FILE = "agent-events.jsonl"
SCORES_FILE = "scores.jsonl"
TRANSCRIPTS_DIR = "transcripts"
FINISHED_STATUSES = {"idle", "stopped", "timeout", "error", "exited"}


def stats_dir() -> Path:
    override = os.environ.get(STATS_DIR_ENV)
    return Path(override).expanduser() if override else Path(DEFAULT_STATS_DIR).expanduser()


def compact_text(value: Any, *, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def now_event_time() -> tuple[float, str]:
    timestamp = time.time()
    return timestamp, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


class StatsStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or stats_dir()
        self._lock = threading.RLock()
        # Authoritative in-memory aggregates. The daemon is the sole writer of
        # agent-events.jsonl (via _append_jsonl), so we can maintain these
        # incrementally and serve hot-path reads (stats_for_agents / summary)
        # from memory instead of re-parsing the whole (multi-MB, never-rotated)
        # log on every call. We seed once from disk on init so a restarted
        # daemon reconstructs the same state.
        self._agents: dict[str, dict[str, Any]] = {}
        self._observed_count = 0
        self._score_count = 0
        self._score_total = 0
        self._seed_from_disk()

    def _seed_from_disk(self) -> None:
        with self._lock:
            for event in self._read_jsonl(self.agent_events_path):
                self._ingest_event(event)

    def _ingest_event(self, event: dict[str, Any]) -> None:
        """Fold one agent-events record into the in-memory aggregates.

        Mirrors the per-agent row construction of ``stats_for_agents`` and the
        counter logic of ``summary`` so the in-memory state stays byte-for-byte
        equivalent to a full file re-parse. Must be called under ``self._lock``.
        """
        agent_id = str(event.get("agent_id") or "")
        if not agent_id:
            return
        # Copy so the in-memory row never aliases nested objects (prompts,
        # usage, event_counts, ...) held by the caller, matching the prior
        # behaviour where every row was rebuilt from freshly parsed records.
        event = copy.deepcopy(event)
        row = self._agents.get(agent_id)
        if row is None:
            row = empty_agent_stats(agent_id)
            self._agents[agent_id] = row
        before_observed = bool(row.get("observed_by_parent"))
        apply_agent_event(row, event)
        if not before_observed and row.get("observed_by_parent"):
            self._observed_count += 1
        if event.get("type") == "agent_score":
            score = event.get("score")
            if isinstance(score, int):
                self._score_count += 1
                self._score_total += score

    @property
    def agent_events_path(self) -> Path:
        return self.root / AGENT_EVENTS_FILE

    @property
    def scores_path(self) -> Path:
        return self.root / SCORES_FILE

    def transcript_path(self, agent_id: str) -> Path:
        return self.root / TRANSCRIPTS_DIR / f"{agent_id}.jsonl"

    def append_transcript(self, agent_id: str, record: dict[str, Any]) -> None:
        """Append one full-fidelity transcript record for a single agent.

        This is intentionally a per-agent file, not the shared agent-events log:
        transcripts hold every prompt, reasoning stream, tool call, full tool
        result, and message, so they are large and append-fast but never read on
        the hot summary/inspect path. Keeping them out of agent-events.jsonl is
        what stops full transcripts from re-bloating the file that the TUI parses
        on every poll.
        """
        if not agent_id:
            return
        path = self.transcript_path(agent_id)
        with self._lock:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            try:
                path.chmod(0o600)
            except OSError:
                pass

    def record_agent_snapshot(
        self,
        *,
        event_type: str,
        snapshot: dict[str, Any],
        requester: dict[str, Any] | None,
    ) -> dict[str, Any]:
        timestamp, iso = now_event_time()
        event = {
            "type": event_type,
            "timestamp": timestamp,
            "time": iso,
            "agent_id": snapshot.get("agent_id"),
            "status": snapshot.get("status"),
            "created_at": snapshot.get("created_at"),
            "runtime_seconds": runtime_seconds(snapshot),
            "cwd": snapshot.get("cwd"),
            "provider": snapshot.get("provider"),
            "model": snapshot.get("model"),
            "tool_mode": snapshot.get("tool_mode"),
            "turn_count": snapshot.get("turn_count", 0),
            "prompt_count": prompt_count(snapshot),
            "prompts": prompt_records(snapshot),
            "initial_request_preview": initial_request_preview(snapshot),
            "final_text_preview": compact_text(snapshot.get("final_text"), limit=1000),
            "tool_call_count": snapshot.get("tool_call_count", 0),
            "event_counts": snapshot.get("event_counts") if isinstance(snapshot.get("event_counts"), dict) else {},
            "usage": snapshot.get("usage") if isinstance(snapshot.get("usage"), dict) else {},
            "error": snapshot.get("error") or "",
            "requester": requester or {},
        }
        self._append_jsonl(self.agent_events_path, event)
        return event

    def record_observed(
        self,
        *,
        agent_id: str,
        via: str,
        snapshot: dict[str, Any],
        requester: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not agent_output_observed(snapshot):
            return None
        timestamp, iso = now_event_time()
        event = {
            "type": "agent_observed",
            "timestamp": timestamp,
            "time": iso,
            "agent_id": agent_id,
            "via": via,
            "status": snapshot.get("status"),
            "turn_count": snapshot.get("turn_count", 0),
            "final_text_preview": compact_text(snapshot.get("final_text"), limit=1000),
            "error": snapshot.get("error") or "",
            "requester": requester or {},
        }
        self._append_jsonl(self.agent_events_path, event)
        return event

    def record_score(
        self,
        *,
        agent_id: str,
        score: int,
        category: str,
        comment: str,
        requester: dict[str, Any] | None,
    ) -> dict[str, Any]:
        timestamp, iso = now_event_time()
        event = {
            "type": "agent_score",
            "timestamp": timestamp,
            "time": iso,
            "agent_id": agent_id,
            "score": score,
            "category": category,
            "comment": comment,
            "sentiment": score_sentiment(score),
            "requester": requester or {},
        }
        self._append_jsonl(self.scores_path, event)
        self._append_jsonl(self.agent_events_path, event)
        return event

    def agent_stats(self, agent_id: str) -> dict[str, Any]:
        return self.stats_for_agents([agent_id]).get(agent_id, empty_agent_stats(agent_id))

    def stats_for_agents(self, agent_ids: list[str]) -> dict[str, dict[str, Any]]:
        wanted = {agent_id for agent_id in agent_ids if agent_id}
        rows: dict[str, dict[str, Any]] = {}
        if not wanted:
            return rows

        with self._lock:
            for agent_id in wanted:
                row = self._agents.get(agent_id)
                if row is not None:
                    rows[agent_id] = copy.deepcopy(row)
        return rows

    def summary(self) -> dict[str, Any]:
        with self._lock:
            total_agents = len(self._agents)
            observed_count = self._observed_count
            score_count = self._score_count
            score_total = self._score_total
        return {
            "total_agents": total_agents,
            "observed_agents": observed_count,
            "unobserved_agents": max(total_agents - observed_count, 0),
            "scores": score_count,
            "average_score": round(score_total / score_count, 2) if score_count else None,
        }

    def _append_jsonl(self, path: Path, event: dict[str, Any]) -> None:
        with self._lock:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                self.root.chmod(0o700)
            except OSError:
                pass
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            try:
                path.chmod(0o600)
            except OSError:
                pass
            # Keep in-memory aggregates authoritative without re-reading the log.
            # Only agent-events records feed the summary/per-agent stats; the
            # separate scores.jsonl is a write-only audit trail.
            if path == self.agent_events_path:
                self._ingest_event(event)

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self._lock:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict):
                        rows.append(event)
        return rows


def runtime_seconds(snapshot: dict[str, Any]) -> float | None:
    created_at = snapshot.get("created_at")
    if isinstance(created_at, int | float):
        return round(max(time.time() - float(created_at), 0.0), 3)
    return None


def prompt_records(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    prompts = snapshot.get("prompts")
    if not isinstance(prompts, list):
        initial = snapshot.get("initial_request")
        if isinstance(initial, dict):
            prompts = [initial]
        else:
            return []

    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        if not isinstance(prompt, dict):
            continue
        rows.append(
            {
                "turn": prompt.get("turn"),
                "behavior": prompt.get("behavior"),
                "accepted": bool(prompt.get("accepted", False)),
                "text": str(prompt.get("text") or ""),
            }
        )
    return rows


def prompt_count(snapshot: dict[str, Any]) -> int:
    prompts = snapshot.get("prompts")
    if isinstance(prompts, list):
        return len(prompts)
    value = snapshot.get("prompt_count")
    return int(value) if isinstance(value, int) else 0


def initial_request_preview(snapshot: dict[str, Any]) -> str:
    initial = snapshot.get("initial_request")
    if isinstance(initial, dict):
        return compact_text(initial.get("text"), limit=500)
    return compact_text(snapshot.get("initial_request_preview"), limit=500)


def agent_output_observed(snapshot: dict[str, Any]) -> bool:
    if snapshot.get("final_text") or snapshot.get("error"):
        return True
    status = str(snapshot.get("status") or "")
    if status not in FINISHED_STATUSES:
        return False
    turn_count = snapshot.get("turn_count")
    return isinstance(turn_count, int) and turn_count > 0


def empty_agent_stats(agent_id: str) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "observed_by_parent": False,
        "observation_count": 0,
        "scores": [],
    }


def apply_agent_event(row: dict[str, Any], event: dict[str, Any]) -> None:
    event_type = event.get("type")
    row["last_event_at"] = event.get("timestamp")
    row["last_event_time"] = event.get("time")

    if event_type in {"agent_started", "agent_updated", "agent_stopped"}:
        lifecycle_type = str(row.get("last_lifecycle_type") or "")
        stale_start = event_type == "agent_started" and lifecycle_type in {"agent_updated", "agent_stopped"}
        keys = (
            "status",
            "created_at",
            "runtime_seconds",
            "cwd",
            "provider",
            "model",
            "tool_mode",
            "turn_count",
            "prompt_count",
            "initial_request_preview",
            "final_text_preview",
            "tool_call_count",
            "event_counts",
            "usage",
            "error",
            "requester",
        )
        if stale_start:
            keys = (
                "created_at",
                "cwd",
                "provider",
                "model",
                "tool_mode",
                "prompt_count",
                "initial_request_preview",
                "requester",
            )
        for key in keys:
            if key in event:
                row[key] = event[key]
        prompts = event.get("prompts")
        if isinstance(prompts, list) and prompts:
            row["prompts"] = prompts
        if isinstance(event_type, str) and not stale_start:
            row["last_lifecycle_type"] = event_type

    if event_type == "agent_observed":
        row["observed_by_parent"] = True
        row["observed_at"] = event.get("timestamp")
        row["observed_time"] = event.get("time")
        row["observed_via"] = event.get("via")
        row["observation_count"] = int(row.get("observation_count") or 0) + 1

    if event_type == "agent_score":
        score = {
            "score": event.get("score"),
            "category": event.get("category"),
            "comment": event.get("comment"),
            "sentiment": event.get("sentiment"),
            "time": event.get("time"),
            "timestamp": event.get("timestamp"),
            "requester": event.get("requester") if isinstance(event.get("requester"), dict) else {},
        }
        scores = row.setdefault("scores", [])
        if isinstance(scores, list):
            scores.append(score)
            row["latest_score"] = score


def score_sentiment(score: int) -> str:
    if score > 5:
        return "net-positive"
    if score < 5:
        return "net-negative"
    return "neutral"
