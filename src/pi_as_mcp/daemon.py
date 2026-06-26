from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import signal
import socket
import socketserver
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pi_as_mcp.config import AppConfig, ModelConcurrencyLimit, load_config
from pi_as_mcp.pi_rpc import PiRpcError, PiRpcRunner, ToolMode, resolve_model
from pi_as_mcp.paths import session_dir, socket_path
from pi_as_mcp.sessions import SessionManager, validate_response_verbosity
from pi_as_mcp.stats import StatsStore

SO_PEERCRED = 17
SAFE_HINT_RE = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")


def hash_scope(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def proc_stat(pid: int) -> tuple[int | None, str | None]:
    try:
        content = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None, None
    end = content.rfind(")")
    if end == -1:
        return None, None
    fields = content[end + 2 :].split()
    if len(fields) < 20:
        return None, None
    try:
        ppid = int(fields[1])
    except ValueError:
        ppid = None
    return ppid, fields[19]


def pid_exists(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@dataclass(frozen=True)
class ParentIdentity:
    scope_id: str
    owner_pid: int | None
    label: str
    peer_pid: int | None = None


def proc_cmdline(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        raw = b""
    if raw:
        text = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        if text:
            return text

    try:
        return Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()
    except OSError:
        return None


def process_info(pid: int | None) -> dict[str, Any] | None:
    if not pid:
        return None
    ppid, start_time = proc_stat(pid)
    command = proc_cmdline(pid)
    if ppid is None and command is None:
        return None
    return {
        "pid": pid,
        "ppid": ppid,
        "start_time": start_time,
        "command": command or "",
    }


def process_lineage(pid: int | None, *, limit: int = 8) -> list[dict[str, Any]]:
    lineage: list[dict[str, Any]] = []
    seen: set[int] = set()
    current = pid
    while current and current not in seen and len(lineage) < limit:
        seen.add(current)
        info = process_info(current)
        if info is None:
            break
        lineage.append(info)
        next_pid = info.get("ppid")
        current = next_pid if isinstance(next_pid, int) and next_pid > 1 else None
    return lineage


def requester_instance(label: str) -> str:
    if label.startswith("hint:"):
        return label.removeprefix("hint:")
    return label


def classify_requester(label: str, lineage: list[dict[str, Any]]) -> str:
    haystack = " ".join([label, *[str(item.get("command") or "") for item in lineage]]).lower()
    if "claude" in haystack:
        return "Claude"
    if "codex" in haystack:
        return "Codex"
    if label.startswith("hint:mcp:"):
        return "MCP"
    if "pi-agent" in haystack:
        return "pi-agent CLI"
    return "local process"


def requester_info(identity: ParentIdentity) -> dict[str, Any]:
    lineage_root = identity.owner_pid or identity.peer_pid
    lineage = process_lineage(lineage_root)
    kind = classify_requester(identity.label, lineage)
    instance = requester_instance(identity.label)
    return {
        "scope_id": identity.scope_id,
        "label": identity.label,
        "kind": kind,
        "instance": instance,
        "display": f"{kind} {instance}".strip(),
        "owner_pid": identity.owner_pid,
        "peer_pid": identity.peer_pid,
        "peer_process": process_info(identity.peer_pid),
        "owner_process": process_info(identity.owner_pid),
        "lineage": lineage,
    }


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


def agent_spawn_rank(agent: dict[str, Any]) -> float:
    created_at = number_value(agent.get("created_at"))
    if created_at is not None:
        return created_at
    total_seconds = number_value(agent.get("total_seconds"))
    if total_seconds is not None:
        return -total_seconds
    return 0.0


def sibling_summary_text(agent: dict[str, Any], *, limit: int = 100) -> str:
    """A very short description of what a sibling agent is doing."""
    for key in ("current_tool", "initial_request_preview", "last_action"):
        value = agent.get(key)
        if not value:
            continue
        text = " ".join(str(value).split())
        if not text:
            continue
        prefix = "running tool: " if key == "current_tool" else ""
        text = prefix + text
        if len(text) > limit:
            return text[: limit - 3] + "..."
        return text
    return ""


# Surface cleanup candidates first: agents that are done (idle) or dead come
# before ones still doing work.
_SIBLING_STATUS_ORDER = {"idle": 0, "exited": 1, "error": 1, "timeout": 1, "stopped": 1}


def sibling_sort_rank(status: str) -> tuple[int, str]:
    return (_SIBLING_STATUS_ORDER.get(status, 2), status)


def parent_identity_from_peer(
    peer_pid: int,
    parent_hint: str | None = None,
    parent_owner_pid: int | None = None,
) -> ParentIdentity:
    if parent_hint:
        if not SAFE_HINT_RE.match(parent_hint):
            raise PiRpcError("parent_hint contains unsupported characters")
        if parent_owner_pid is not None and parent_owner_pid != peer_pid:
            raise PiRpcError("parent_owner_pid must match the socket peer pid")
        return ParentIdentity(
            scope_id=hash_scope(f"hint:{parent_hint}"),
            owner_pid=parent_owner_pid,
            label=f"hint:{parent_hint}",
            peer_pid=peer_pid,
        )

    _ppid, fallback_start = proc_stat(peer_pid)

    # Strong isolation fallback: scope to the direct client process, not cwd or terminal.
    return ParentIdentity(
        scope_id=hash_scope(f"peer:{peer_pid}:{fallback_start or ''}"),
        owner_pid=None,
        label=f"peer:{peer_pid}",
        peer_pid=peer_pid,
    )


class DaemonState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._managers: dict[str, SessionManager] = {}
        self._identities: dict[str, ParentIdentity] = {}
        self._stats = StatsStore()
        self._closed = False
        self._reaper = threading.Thread(target=self._reap_closed_parents, daemon=True)
        self._reaper.start()

    def manager_for(self, identity: ParentIdentity) -> SessionManager:
        with self._lock:
            manager = self._managers.get(identity.scope_id)
            if manager is None:
                agents_config = load_config().agents
                manager = SessionManager(
                    parent_id=identity.scope_id,
                    owner_pid=identity.owner_pid,
                    observer=lambda event_type, snapshot, identity=identity: self._record_agent_snapshot(
                        event_type=event_type,
                        snapshot=snapshot.to_json(verbosity="normal"),
                        identity=identity,
                    ),
                    transcript_sink=self._stats.append_transcript,
                    session_dir=session_dir() if agents_config.persist_sessions else None,
                    idle_eviction_seconds=agents_config.idle_eviction_seconds,
                )
                self._managers[identity.scope_id] = manager
                self._identities[identity.scope_id] = identity
            return manager

    def start(
        self,
        identity: ParentIdentity,
        *,
        prompt: str,
        cwd: str | None,
        model: str | None,
        provider: str | None,
        tool_mode: ToolMode,
        include_events: bool,
    ) -> dict[str, Any]:
        with self._lock:
            manager = self.manager_for(identity)
            model_spec = resolve_model(model, provider)
            config = load_config()
            self._enforce_concurrency_limits_locked(
                config,
                provider=model_spec.provider,
                model=model_spec.model,
            )
            snapshot = manager.start(
                prompt=prompt,
                cwd=cwd,
                model=model_spec.model,
                provider=model_spec.provider,
                tool_mode=tool_mode,
                include_events=include_events,
                unsafe_read_only=config.agents.unsafe_read_only,
            )
            data = snapshot.to_json(verbosity="normal")
            self._record_agent_snapshot(
                event_type="agent_started",
                snapshot=data,
                identity=identity,
            )
            return snapshot.to_json(verbosity="summary")

    def record_agent_snapshot(
        self,
        *,
        event_type: str,
        snapshot: dict[str, Any],
        identity: ParentIdentity,
    ) -> None:
        self._record_agent_snapshot(event_type=event_type, snapshot=snapshot, identity=identity)

    def record_agent_observed(
        self,
        *,
        via: str,
        snapshot: dict[str, Any],
        identity: ParentIdentity,
    ) -> None:
        agent_id = str(snapshot.get("agent_id") or "")
        if not agent_id:
            return
        self._stats.record_observed(
            agent_id=agent_id,
            via=via,
            snapshot=snapshot,
            requester=requester_info(identity),
        )

    def agent_stats(self, agent_id: str) -> dict[str, Any]:
        return self._stats.agent_stats(agent_id)

    def score_agent(
        self,
        identity: ParentIdentity,
        *,
        agent_id: str,
        score: int,
        category: str,
        comment: str,
    ) -> dict[str, Any]:
        config = load_config()
        if not config.agents.enable_score:
            raise PiRpcError("agent scoring is disabled")

        if isinstance(score, bool) or not isinstance(score, int) or score < 1 or score > 10:
            raise PiRpcError("score must be an integer from 1 to 10")

        category = " ".join(category.split())
        if not category:
            raise PiRpcError("category must not be empty")
        if len(category) > 80:
            raise PiRpcError("category must be 80 characters or fewer")

        comment = " ".join(comment.split())
        if not comment:
            raise PiRpcError("comment must not be empty")
        if len(comment) > 280:
            raise PiRpcError("comment must be 280 characters or fewer")

        event = self._stats.record_score(
            agent_id=agent_id,
            score=score,
            category=category,
            comment=comment,
            requester=requester_info(identity),
        )
        return {
            "agent_id": agent_id,
            "score": score,
            "category": category,
            "comment": comment,
            "sentiment": event["sentiment"],
            "recorded": True,
        }

    def score_hint(self, agent_id: str) -> str | None:
        if not load_config().agents.enable_score:
            return None
        stats = self._stats.agent_stats(agent_id)
        if stats.get("latest_score"):
            return None
        return (
            "Rate this sub-agent with score(agent_id, score=1-10, category, comment). "
            ">5 is net-positive, <5 is net-negative; keep the comment tweet-sized."
        )

    def stats_summary(self) -> dict[str, Any]:
        return self._stats.summary()

    def _enforce_concurrency_limits_locked(
        self,
        config: AppConfig,
        *,
        provider: str,
        model: str,
    ) -> None:
        model_limit = config.agents.concurrency_limit_for_model(provider=provider, model=model)
        if model_limit is None:
            return

        active_count = self._active_model_count_locked(model_limit)
        if active_count >= model_limit.limit:
            raise PiRpcError(
                "Pi agent concurrency limit reached for "
                f"{model_limit.display_model}: {active_count} active, limit {model_limit.limit}"
            )

    def _active_model_count_locked(self, model_limit: ModelConcurrencyLimit) -> int:
        return sum(
            manager.active_model_count(
                provider=model_limit.provider,
                model=model_limit.model,
                match_provider=model_limit.match_provider,
            )
            for manager in self._managers.values()
        )

    def _record_agent_snapshot(
        self,
        *,
        event_type: str,
        snapshot: dict[str, Any],
        identity: ParentIdentity,
    ) -> None:
        self._stats.record_agent_snapshot(
            event_type=event_type,
            snapshot=snapshot,
            requester=requester_info(identity),
        )

    def manager_for_agent(self, agent_id: str) -> SessionManager | None:
        match = self.manager_identity_for_agent(agent_id)
        return match[0] if match else None

    def manager_identity_for_agent(self, agent_id: str) -> tuple[SessionManager, ParentIdentity] | None:
        with self._lock:
            matches = [
                (manager, self._identities[scope_id])
                for scope_id, manager in self._managers.items()
                if manager.has(agent_id)
            ]
        if len(matches) > 1:
            raise PiRpcError(f"agent_id collision across parent scopes: {agent_id}")
        return matches[0] if matches else None

    def sibling_overview(
        self, identity: ParentIdentity, *, exclude_agent_id: str
    ) -> list[dict[str, Any]]:
        """Compact list of the caller's other agents, so a delegating agent is
        reminded of what it already has running and can clean up / consolidate."""
        with self._lock:
            manager = self._managers.get(identity.scope_id)
        if manager is None:
            return []

        siblings: list[dict[str, Any]] = []
        for agent in manager.summary():
            agent_id = str(agent.get("agent_id") or "")
            if not agent_id or agent_id == exclude_agent_id:
                continue
            siblings.append(
                {
                    "agent_id": agent_id,
                    "status": str(agent.get("status") or ""),
                    "model": str(agent.get("model") or ""),
                    "summary": sibling_summary_text(agent),
                }
            )
        siblings.sort(key=lambda item: sibling_sort_rank(str(item["status"])))
        return siblings

    def global_summary(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = [
                (scope_id, manager, self._identities[scope_id])
                for scope_id, manager in self._managers.items()
            ]

        agents: list[dict[str, Any]] = []
        for scope_id, manager, identity in rows:
            requester = requester_info(identity)
            for agent in manager.summary():
                agent["parent_scope_id"] = scope_id
                agent["requester"] = requester
                agents.append(agent)
        agents.sort(key=lambda item: (-agent_spawn_rank(item), str(item.get("agent_id") or "")))
        stats_by_agent = self._stats.stats_for_agents([str(agent.get("agent_id") or "") for agent in agents])
        for agent in agents:
            agent_id = str(agent.get("agent_id") or "")
            if agent_id in stats_by_agent:
                agent["stats"] = stats_by_agent[agent_id]
        return agents

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            managers = list(self._managers.values())
            self._managers.clear()
            self._identities.clear()
        for manager in managers:
            manager.close(reason="daemon closed")

    def _reap_closed_parents(self) -> None:
        while True:
            time.sleep(1.0)
            with self._lock:
                if self._closed:
                    return
                dead_scope_ids = [
                    scope_id
                    for scope_id, identity in self._identities.items()
                    if identity.owner_pid is not None and not pid_exists(identity.owner_pid)
                ]
                dead_managers = [
                    self._managers.pop(scope_id)
                    for scope_id in dead_scope_ids
                    if scope_id in self._managers
                ]
                for scope_id in dead_scope_ids:
                    self._identities.pop(scope_id, None)
            for manager in dead_managers:
                manager.close(reason="parent closed")


STATE = DaemonState()


class RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            line = self.rfile.readline()
            if not line:
                return
            request = json.loads(line.decode("utf-8"))
            response = self.handle_request(request)
        except Exception as exc:
            response = {"error": str(exc)}
        self.wfile.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
        self.wfile.flush()

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if not isinstance(command, str):
            raise PiRpcError("request command must be a string")

        peer_pid = self.peer_pid()
        parent_hint = request.get("parent_hint")
        identity = parent_identity_from_peer(
            peer_pid,
            parent_hint if isinstance(parent_hint, str) else None,
            optional_positive_int(request, "parent_owner_pid"),
        )
        scoped_manager: SessionManager | None = None

        def manager_for_scope() -> SessionManager:
            nonlocal scoped_manager
            if scoped_manager is None:
                scoped_manager = STATE.manager_for(identity)
            return scoped_manager

        verbosity = response_verbosity(request)

        if command in {"start", "delegate"}:
            data = STATE.start(
                identity,
                prompt=required_str(request, "prompt"),
                cwd=optional_str(request, "cwd"),
                model=optional_str(request, "model"),
                provider=None,
                tool_mode=request.get("tool_mode", "read-only"),
                include_events=bool(request.get("include_events", False)),
            )
            if verbosity != "summary":
                agent_id = str(data["agent_id"])
                target_manager = STATE.manager_for_agent(agent_id) or manager_for_scope()
                data = target_manager.peek(
                    agent_id,
                    include_events=bool(request.get("include_events", False)),
                ).to_json(verbosity=verbosity)
            if command == "delegate":
                data["delegate_mode"] = "async"
                data["other_agents"] = STATE.sibling_overview(
                    identity, exclude_agent_id=str(data["agent_id"])
                )
            return data

        if command == "reply":
            agent_id = required_str(request, "agent_id")
            match = STATE.manager_identity_for_agent(agent_id)
            target_manager, target_identity = match if match else (manager_for_scope(), identity)
            snapshot = target_manager.reply(
                agent_id,
                prompt=required_str(request, "prompt"),
                behavior=request.get("behavior", "auto"),
            )
            data = snapshot.to_json(verbosity=verbosity)
            STATE.record_agent_snapshot(event_type="agent_updated", snapshot=data, identity=target_identity)
            return data

        if command == "peek":
            agent_id = required_str(request, "agent_id")
            match = STATE.manager_identity_for_agent(agent_id)
            target_manager, target_identity = match if match else (manager_for_scope(), identity)
            data = target_manager.peek(
                agent_id,
                include_events=bool(request.get("include_events", False)),
            ).to_json(verbosity=verbosity)
            STATE.record_agent_observed(via="peek", snapshot=data, identity=target_identity)
            attach_agent_stats(data)
            return data

        if command == "listen":
            agent_id = required_str(request, "agent_id")
            match = STATE.manager_identity_for_agent(agent_id)
            target_manager, target_identity = match if match else (manager_for_scope(), identity)
            snapshot, timed_out = target_manager.listen(
                agent_id,
                after_turn_count=int(request.get("after_turn_count", 0)),
                timeout_seconds=int(request.get("timeout_seconds", 60)),
                include_events=bool(request.get("include_events", False)),
            )
            data = snapshot.to_json(verbosity=verbosity)
            data["listen_timed_out"] = timed_out
            data["after_turn_count"] = int(request.get("after_turn_count", 0))
            STATE.record_agent_snapshot(event_type="agent_updated", snapshot=data, identity=target_identity)
            if not timed_out:
                STATE.record_agent_observed(via="listen", snapshot=data, identity=target_identity)
                hint = STATE.score_hint(agent_id)
                if hint:
                    data["score_hint"] = hint
            attach_agent_stats(data)
            return data

        if command == "stop":
            agent_id = required_str(request, "agent_id")
            match = STATE.manager_identity_for_agent(agent_id)
            target_manager, target_identity = match if match else (manager_for_scope(), identity)
            data = target_manager.stop(agent_id).to_json(verbosity=verbosity)
            STATE.record_agent_snapshot(event_type="agent_stopped", snapshot=data, identity=target_identity)
            STATE.record_agent_observed(via="stop", snapshot=data, identity=target_identity)
            attach_agent_stats(data)
            return data

        if command == "inspect":
            agent_id = required_str(request, "agent_id")
            # Passive viewers (the TUI dashboard) poll inspect on a timer and must
            # opt out of observation recording. Otherwise every poll appends an
            # agent_observed event, which inflates observation_count without bound,
            # corrupts the observed/unobserved metric, and grows the stats log until
            # reads start tripping client timeouts. Real parents leave observe=True.
            observe = bool(request.get("observe", True))
            match = STATE.manager_identity_for_agent(agent_id)
            if match is None:
                target_manager = manager_for_scope()
                data = target_manager.peek(
                    agent_id,
                    include_events=bool(request.get("include_events", False)),
                ).to_json(verbosity=verbosity)
                data["requester"] = requester_info(identity)
                if observe:
                    STATE.record_agent_observed(via="inspect", snapshot=data, identity=identity)
                attach_agent_stats(data)
                return data

            target_manager, target_identity = match
            data = target_manager.peek(
                agent_id,
                include_events=bool(request.get("include_events", False)),
            ).to_json(verbosity=verbosity)
            data["requester"] = requester_info(target_identity)
            if observe:
                STATE.record_agent_observed(via="inspect", snapshot=data, identity=target_identity)
            attach_agent_stats(data)
            return data

        if command == "tui_summary":
            return {"agents": STATE.global_summary(), "stats": STATE.stats_summary()}

        if command == "summary":
            return {"agents": manager_for_scope().summary()}

        if command == "list":
            return {
                "deprecated": "agent_list is deprecated; use agent_summary. Removal target: 0.2.0.",
                "agents": manager_for_scope().summary(),
            }

        if command == "models":
            return {"models": PiRpcRunner().model_aliases()}

        if command == "health":
            return PiRpcRunner().health(
                model=optional_str(request, "model"),
                timeout_seconds=int(request.get("timeout_seconds", 15)),
            )

        if command == "score":
            return STATE.score_agent(
                identity,
                agent_id=required_str(request, "agent_id"),
                score=required_int(request, "score"),
                category=required_str(request, "category"),
                comment=required_str(request, "comment"),
            )

        raise PiRpcError(f"unknown command: {command}")

    def peer_pid(self) -> int:
        try:
            raw = self.request.getsockopt(socket.SOL_SOCKET, SO_PEERCRED, struct.calcsize("3i"))
            pid, _uid, _gid = struct.unpack("3i", raw)
            return int(pid)
        except OSError:
            return os.getpid()


def required_str(request: dict[str, Any], key: str) -> str:
    value = request.get(key)
    if not isinstance(value, str) or not value:
        raise PiRpcError(f"{key} must be a non-empty string")
    return value


def optional_str(request: dict[str, Any], key: str) -> str | None:
    value = request.get(key)
    return value if isinstance(value, str) and value else None


def optional_positive_int(request: dict[str, Any], key: str) -> int | None:
    value = request.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or value < 1:
        raise PiRpcError(f"{key} must be a positive integer")
    return value


def required_int(request: dict[str, Any], key: str) -> int:
    value = request.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PiRpcError(f"{key} must be an integer")
    return value


def response_verbosity(request: dict[str, Any]) -> str:
    value = request.get("verbosity", "summary")
    if not isinstance(value, str):
        raise PiRpcError("verbosity must be a string")
    return validate_response_verbosity(value)


def attach_agent_stats(data: dict[str, Any]) -> None:
    agent_id = str(data.get("agent_id") or "")
    if agent_id:
        data["stats"] = STATE.agent_stats(agent_id)


class UnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def serve() -> None:
    path = socket_path()
    if path.exists():
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(0.2)
                client.connect(str(path))
            return
        except OSError:
            path.unlink()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    server = UnixServer(str(path), RequestHandler)
    atexit.register(STATE.close)

    def stop(_signum: int, _frame: Any) -> None:
        STATE.close()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        STATE.close()
        try:
            path.unlink()
        except OSError:
            pass


def main() -> None:
    try:
        serve()
    except Exception as exc:
        print(f"pi-agent daemon failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
