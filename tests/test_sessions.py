from __future__ import annotations

import json
import os
import stat
import threading
import time
from pathlib import Path

from pi_as_mcp.sessions import SessionManager, extract_message_update_parts


def test_extract_message_update_parts_assistant_message_event_envelope() -> None:
    # Providers that wrap deltas in an ``assistantMessageEvent`` envelope and
    # carry the characters in ``content`` must still be parsed, otherwise a long
    # pure-reasoning generation streams with zero extracted activity and the
    # inactivity watchdog kills it mid-thought (the qwen-MTP stall bug).
    think = {
        "type": "message_update",
        "assistantMessageEvent": {"type": "thinking_delta", "contentIndex": 0, "content": "reasoning"},
    }
    assert extract_message_update_parts(think) == ("", "reasoning")

    answer = {
        "type": "message_update",
        "assistantMessageEvent": {"type": "text_delta", "contentIndex": 0, "content": "answer"},
    }
    assert extract_message_update_parts(answer) == ("answer", "")

    # Legacy delta shape still works.
    legacy = {"type": "message_update", "delta": {"type": "thinking_delta", "text": "ponder"}}
    assert extract_message_update_parts(legacy) == ("", "ponder")

    # An event with no character payload yields nothing (no false activity).
    empty = {"type": "message_update", "assistantMessageEvent": {"type": "thinking_start", "contentIndex": 0}}
    assert extract_message_update_parts(empty) == ("", "")


def write_fake_pi(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "fake-pi-session"
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_session_start_reply_peek_and_stop(tmp_path: Path) -> None:
    fake_pi = write_fake_pi(
        tmp_path,
        """#!/usr/bin/env python3
import json
import sys

if "--version" in sys.argv:
    print("fake-pi 1.0")
    raise SystemExit(0)
if "--list-models" in sys.argv:
    print("provider   model                    context")
    print("local  example-model  128K")
    raise SystemExit(0)

for line in sys.stdin:
    request = json.loads(line)
    if request.get("type") == "abort":
        break
    text = request.get("message", "")
    print(json.dumps({"id": request["id"], "type": "response", "command": "prompt", "success": True}), flush=True)
    print(json.dumps({"type": "agent_start"}), flush=True)
    print(json.dumps({
        "type": "tool_execution_start",
        "toolCallId": "tool-" + text,
        "toolName": "read",
        "args": {"path": "README.md"},
    }), flush=True)
    print(json.dumps({
        "type": "tool_execution_end",
        "toolCallId": "tool-" + text,
        "toolName": "read",
        "isError": False,
        "result": {"content": [{"type": "text", "text": "file body"}]},
    }), flush=True)
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "echo:" + text}],
        "responseId": "resp-" + text,
        "usage": {"input": 10, "output": 5, "cacheRead": 2, "cacheWrite": 1, "totalTokens": 18},
    }
    print(json.dumps({"type": "message_end", "message": message}), flush=True)
    print(json.dumps({"type": "agent_end", "messages": [message]}), flush=True)
""",
    )

    manager = SessionManager()
    manager._runner.pi_bin = str(fake_pi)

    first = manager.start(
        prompt="one",
        cwd=str(tmp_path),
        model="local/example-model",
        provider=None,
        tool_mode="none",
        include_events=True,
    )

    first_done, first_listen_timeout = manager.listen(
        first.agent_id,
        after_turn_count=0,
        timeout_seconds=5,
    )
    assert first_listen_timeout is False
    assert first_done.final_text == "echo:one"
    first_data = first_done.to_json(verbosity="normal")
    assert first_data["initial_request"]["text"] == "one"
    assert first_data["initial_request"]["accepted"] is True
    assert [item["kind"] for item in first_data["transcript"]] == [
        "prompt",
        "tool_call",
        "tool_result",
        "message",
    ]
    assert first_data["transcript"][0]["behavior"] == "prompt"
    assert first_data["transcript"][1]["tool_name"] == "read"
    assert first_data["created_at"] > 0
    assert first_data["usage"]["input_tokens"] == 10
    assert first_data["usage"]["output_tokens"] == 5
    assert first_data["usage"]["cache_read_tokens"] == 2
    assert first_data["usage"]["cache_write_tokens"] == 1
    assert first_data["usage"]["context_used_tokens"] == 18
    assert first_data["usage"]["context_limit_tokens"] == 128_000
    assert first_done.event_counts.get("message_update", 0) == 0

    manager.reply(first.agent_id, prompt="two", behavior="auto")
    second_done, second_listen_timeout = manager.listen(
        first.agent_id,
        after_turn_count=1,
        timeout_seconds=5,
    )
    assert second_listen_timeout is False
    assert second_done.final_text == "echo:two"
    assert second_done.status == "idle"

    summary = manager.summary()
    assert len(summary) == 1
    assert summary[0]["agent_id"] == first.agent_id
    assert summary[0]["status"] == "idle"
    assert summary[0]["turn_count"] == 2
    assert summary[0]["last_action"] == "idle after turn 2"
    assert summary[0]["final_text_preview"] == "echo:two"
    assert summary[0]["recent_actions"][-2:] == ["received assistant message", "idle after turn 2"]
    assert summary[0]["created_at"] == first_data["created_at"]
    assert summary[0]["usage"]["input_tokens"] == 20
    assert summary[0]["usage"]["output_tokens"] == 10
    assert summary[0]["usage"]["cache_read_tokens"] == 4
    assert summary[0]["usage"]["cache_write_tokens"] == 2
    assert summary[0]["usage"]["total_tokens"] == 36
    assert summary[0]["usage"]["context_used_tokens"] == 18
    assert summary[0]["usage"]["context_percent"] > 0
    assert summary[0]["usage"]["tokens_per_second"] > 0

    stopped = manager.stop(first.agent_id)
    assert stopped.status == "stopped"


def test_provider_rate_limit_error_is_forwarded(tmp_path: Path) -> None:
    # Reproduces a provider 429: empty assistant turns followed by auto_retry
    # events that ultimately fail. The reason must reach the snapshot/summary
    # instead of being swallowed (only counted in event_counts).
    fake_pi = write_fake_pi(
        tmp_path,
        """#!/usr/bin/env python3
import json
import sys

if "--version" in sys.argv:
    print("fake-pi 1.0")
    raise SystemExit(0)
if "--list-models" in sys.argv:
    print("provider   model                    context")
    print("local  example-model  128K")
    raise SystemExit(0)

ERR = "429 Usage limit reached for 5 hour. Your limit will reset at 2026-06-27 16:38:34"

for line in sys.stdin:
    request = json.loads(line)
    if request.get("type") == "abort":
        break
    print(json.dumps({"id": request["id"], "type": "response", "command": "prompt", "success": True}), flush=True)
    # Three empty attempts, each followed by an auto_retry_start, then a
    # terminal auto_retry_end(success=false) carrying the final error.
    for attempt in range(1, 4):
        print(json.dumps({"type": "agent_start"}), flush=True)
        print(json.dumps({"type": "agent_end", "messages": []}), flush=True)
        print(json.dumps({
            "type": "auto_retry_start",
            "attempt": attempt,
            "maxAttempts": 3,
            "delayMs": 1,
            "errorMessage": ERR,
        }), flush=True)
    print(json.dumps({"type": "auto_retry_end", "success": False, "attempt": 3, "finalError": ERR}), flush=True)
""",
    )

    manager = SessionManager()
    manager._runner.pi_bin = str(fake_pi)

    agent = manager.start(
        prompt="do work",
        cwd=str(tmp_path),
        model="local/example-model",
        provider=None,
        tool_mode="none",
        include_events=True,
    )

    # The agent never produces real output; wait for the empty turns to land.
    done, _timed_out = manager.listen(agent.agent_id, after_turn_count=2, timeout_seconds=5)

    expected = "429 Usage limit reached for 5 hour. Your limit will reset at 2026-06-27 16:38:34"
    assert done.error is not None
    assert expected in done.error
    assert done.final_text == ""

    data = done.to_json(verbosity="normal")
    assert expected in data["error"]
    assert data["event_counts"].get("auto_retry_start", 0) == 3
    assert data["event_counts"].get("auto_retry_end", 0) == 1

    summary = manager.summary()
    assert summary[0]["error"] is not None and expected in summary[0]["error"]

    manager.stop(agent.agent_id)


def test_thinking_deltas_record_a_separate_reasoning_item(tmp_path: Path) -> None:
    fake_pi = write_fake_pi(
        tmp_path,
        """#!/usr/bin/env python3
import json
import sys
import time

if "--list-models" in sys.argv:
    print("provider   model                    context")
    print("local  example-model  128K")
    raise SystemExit(0)

line = sys.stdin.readline()
request = json.loads(line)
print(json.dumps({"id": request["id"], "type": "response", "command": "prompt", "success": True}), flush=True)
print(json.dumps({"type": "agent_start"}), flush=True)
print(json.dumps({"type": "message_update", "delta": {"type": "thinking_delta", "text": "let me "}}), flush=True)
print(json.dumps({"type": "message_update", "delta": {"type": "thinking_delta", "text": "ponder"}}), flush=True)
print(json.dumps({"type": "message_update", "delta": {"type": "text_delta", "text": "the "}}), flush=True)
print(json.dumps({"type": "message_update", "delta": {"type": "text_delta", "text": "answer"}}), flush=True)
message = {"role": "assistant", "content": [{"type": "text", "text": "the answer"}]}
print(json.dumps({"type": "message_end", "message": message}), flush=True)
print(json.dumps({"type": "agent_end", "messages": [message]}), flush=True)
while True:
    time.sleep(1)
""",
    )

    manager = SessionManager()
    manager._runner.pi_bin = str(fake_pi)

    started = manager.start(
        prompt="think then answer",
        cwd=str(tmp_path),
        model="local/example-model",
        provider=None,
        tool_mode="none",
        include_events=True,
    )
    done, timed_out = manager.listen(started.agent_id, after_turn_count=0, timeout_seconds=5)
    assert timed_out is False

    transcript = done.to_json(verbosity="normal")["transcript"]
    by_kind = {item["kind"]: item for item in transcript}
    # Reasoning is coalesced into its own item and survives the final message;
    # the spoken answer stays separate.
    assert by_kind["thinking_stream"]["text"] == "let me ponder"
    assert by_kind["message"]["text"] == "the answer"
    assert "thinking_stream" in by_kind and "message" in by_kind
    assert manager.stop(started.agent_id).status == "stopped"


def test_running_reply_keeps_last_assistant_final_text(tmp_path: Path) -> None:
    fake_pi = write_fake_pi(
        tmp_path,
        """#!/usr/bin/env python3
import json
import sys
import time

if "--list-models" in sys.argv:
    print("provider   model                    context")
    print("local  example-model  128K")
    raise SystemExit(0)

turn = 0
for line in sys.stdin:
    request = json.loads(line)
    if request.get("type") == "abort":
        break
    turn += 1
    text = request.get("message", "")
    print(json.dumps({"id": request["id"], "type": "response", "command": "prompt", "success": True}), flush=True)
    print(json.dumps({"type": "agent_start"}), flush=True)
    if turn == 1:
        message = {"role": "assistant", "content": [{"type": "text", "text": "echo:" + text}]}
        print(json.dumps({"type": "message_end", "message": message}), flush=True)
        print(json.dumps({"type": "agent_end", "messages": [message]}), flush=True)
    else:
        user_message = {"role": "user", "content": [{"type": "text", "text": text}]}
        print(json.dumps({"type": "message_end", "message": user_message}), flush=True)
        print(json.dumps({"type": "message_update", "delta": {"type": "text_delta", "text": "partial "}}), flush=True)
        print(json.dumps({"type": "message_update", "delta": {"type": "text_delta", "text": "assistant"}}), flush=True)
        while True:
            time.sleep(1)
""",
    )

    manager = SessionManager()
    manager._runner.pi_bin = str(fake_pi)

    first = manager.start(
        prompt="one",
        cwd=str(tmp_path),
        model="local/example-model",
        provider=None,
        tool_mode="none",
        include_events=True,
    )
    first_done, first_listen_timeout = manager.listen(
        first.agent_id,
        after_turn_count=0,
        timeout_seconds=5,
    )
    assert first_listen_timeout is False
    assert first_done.final_text == "echo:one"

    reply_snapshot = manager.reply(first.agent_id, prompt="two", behavior="auto")
    assert reply_snapshot.status == "running"
    assert reply_snapshot.final_text == "echo:one"

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        snapshot = manager.peek(first.agent_id, include_events=True)
        if snapshot.event_counts.get("message_update", 0) >= 2:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("fake Pi did not emit streaming assistant text")

    assert snapshot.status == "running"
    assert snapshot.final_text == "echo:one"
    transcript = snapshot.to_json(verbosity="normal")["transcript"]
    assert transcript[-1]["kind"] == "message_stream"
    assert transcript[-1]["text"] == "partial assistant"
    assert manager.stop(first.agent_id).status == "stopped"


def test_session_persists_full_transcript(tmp_path: Path) -> None:
    long_result = "X" * 1200  # longer than the 500-char preview limit
    fake_pi = write_fake_pi(
        tmp_path,
        """#!/usr/bin/env python3
import json
import sys
import time

if "--list-models" in sys.argv:
    print("provider   model                    context")
    print("local  example-model  128K")
    raise SystemExit(0)

line = sys.stdin.readline()
request = json.loads(line)
print(json.dumps({"id": request["id"], "type": "response", "command": "prompt", "success": True}), flush=True)
print(json.dumps({"type": "agent_start"}), flush=True)
print(json.dumps({"type": "message_update", "delta": {"type": "thinking_delta", "text": "let me "}}), flush=True)
print(json.dumps({"type": "message_update", "delta": {"type": "text_delta", "text": "think"}}), flush=True)
print(json.dumps({
    "type": "tool_execution_start",
    "toolCallId": "call-1",
    "toolName": "read",
    "args": {"path": "README.md"},
}), flush=True)
print(json.dumps({
    "type": "tool_execution_end",
    "toolCallId": "call-1",
    "toolName": "read",
    "isError": False,
    "result": {"content": [{"type": "text", "text": "%s"}]},
}), flush=True)
message = {"role": "assistant", "content": [{"type": "text", "text": "final answer"}]}
print(json.dumps({"type": "message_end", "message": message}), flush=True)
print(json.dumps({"type": "agent_end", "messages": [message]}), flush=True)
while True:
    time.sleep(1)
"""
        % long_result,
    )

    records: list[tuple[str, dict]] = []

    manager = SessionManager(transcript_sink=lambda agent_id, record: records.append((agent_id, record)))
    manager._runner.pi_bin = str(fake_pi)

    started = manager.start(
        prompt="hello transcript",
        cwd=str(tmp_path),
        model="local/example-model",
        provider=None,
        tool_mode="none",
        include_events=False,
    )
    manager.listen(started.agent_id, after_turn_count=0, timeout_seconds=5)

    # agent_end is persisted after listen wakes; wait for it to land.
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if any(record["type"] == "agent_end" for _agent_id, record in records):
            break
        time.sleep(0.02)
    else:
        raise AssertionError("agent_end transcript record was not written")

    assert all(agent_id == started.agent_id for agent_id, _record in records)
    by_type = [record["type"] for _agent_id, record in records]
    # Full fidelity: lifecycle events, the coalesced reasoning stream, the tool
    # call, the full tool result, the message, and the turn end are all recorded.
    assert by_type[:7] == [
        "prompt",
        "agent_start",
        "stream",
        "tool_execution_start",
        "tool_execution_end",
        "message_end",
        "agent_end",
    ]

    prompt_record = next(r for _a, r in records if r["type"] == "prompt")
    assert prompt_record["data"]["text"] == "hello transcript"

    stream_record = next(r for _a, r in records if r["type"] == "stream")
    assert stream_record["data"]["text"] == "let me think"  # reasoning deltas coalesced

    tool_result = next(r for _a, r in records if r["type"] == "tool_execution_end")
    # Full result, not the truncated preview.
    assert tool_result["data"]["result"]["content"][0]["text"] == long_result

    # The transcript also lands on disk under the per-agent file.
    from pi_as_mcp.stats import StatsStore

    store = StatsStore(root=tmp_path / "stats")
    for agent_id, record in records:
        store.append_transcript(agent_id, record)
    lines = store.transcript_path(started.agent_id).read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records)
    assert json.loads(lines[0])["type"] == "prompt"

    manager.stop(started.agent_id)


def test_idle_worker_is_evicted_and_resumed_on_reply(tmp_path: Path) -> None:
    # A worker that echoes each prompt and records every process launch's argv,
    # so we can assert the persisted-session flags and that a respawn happened.
    launches = tmp_path / "launches.jsonl"
    fake_pi = write_fake_pi(
        tmp_path,
        """#!/usr/bin/env python3
import json
import sys

if "--list-models" in sys.argv:
    print("provider   model                    context")
    print("local  example-model  128K")
    raise SystemExit(0)

with open(%r, "a") as handle:
    handle.write(json.dumps(sys.argv) + "\\n")

for line in sys.stdin:
    request = json.loads(line)
    if request.get("type") == "abort":
        break
    text = request.get("message", "")
    print(json.dumps({"id": request["id"], "type": "response", "command": "prompt", "success": True}), flush=True)
    print(json.dumps({"type": "agent_start"}), flush=True)
    message = {"role": "assistant", "content": [{"type": "text", "text": "echo:" + text}]}
    print(json.dumps({"type": "message_end", "message": message}), flush=True)
    print(json.dumps({"type": "agent_end", "messages": [message]}), flush=True)
"""
        % str(launches),
    )

    # idle_eviction_seconds is large so the watchdog never evicts mid-test; we
    # trigger eviction explicitly to keep the lifecycle deterministic.
    manager = SessionManager(session_dir=tmp_path / "sessions", idle_eviction_seconds=1000)
    manager._runner.pi_bin = str(fake_pi)

    started = manager.start(
        prompt="one",
        cwd=str(tmp_path),
        model="local/example-model",
        provider=None,
        tool_mode="none",
        include_events=False,
    )
    done, timed_out = manager.listen(started.agent_id, after_turn_count=0, timeout_seconds=5)
    assert timed_out is False
    assert done.final_text == "echo:one"

    # Persistence flags are passed so the session lives on disk under our id.
    first_argv = json.loads(launches.read_text(encoding="utf-8").splitlines()[0])
    assert "--session-id" in first_argv
    assert started.agent_id in first_argv
    assert "--session-dir" in first_argv
    assert "--no-session" not in first_argv

    session = manager._get(started.agent_id)
    assert session.process is not None
    first_pid = session.process.pid

    # Evict: the worker process is killed, but the agent stays idle/resumable.
    session._evict()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(first_pid, 0)
        except OSError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("evicted worker process was not killed")

    peeked = manager.peek(started.agent_id)
    assert peeked.status == "idle"
    assert peeked.turn_count == 1

    # A reply transparently respawns a worker (resuming the session) and runs a
    # fresh turn — a second process launch is recorded.
    manager.reply(started.agent_id, prompt="two", behavior="auto")
    resumed, resumed_timeout = manager.listen(started.agent_id, after_turn_count=1, timeout_seconds=5)
    assert resumed_timeout is False
    assert resumed.final_text == "echo:two"
    assert resumed.turn_count == 2

    launch_lines = launches.read_text(encoding="utf-8").splitlines()
    assert len(launch_lines) == 2  # original + respawn
    assert session.process is not None
    assert session.process.pid != first_pid

    stopped = manager.stop(started.agent_id)
    assert stopped.status == "stopped"


def test_unsafe_read_only_flag_opens_full_tools_and_guards_prompt(tmp_path: Path) -> None:
    fake_pi = write_fake_pi(
        tmp_path,
        """#!/usr/bin/env python3
import json
import os
import sys
import time

if "--list-models" in sys.argv:
    print("provider   model                    context")
    print("local  example-model  128K")
    raise SystemExit(0)

line = sys.stdin.readline()
request = json.loads(line)
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "worker-calls.jsonl"), "a") as handle:
    handle.write(json.dumps({"argv": sys.argv, "message": request.get("message")}) + "\\n")
print(json.dumps({"id": request["id"], "type": "response", "command": "prompt", "success": True}), flush=True)
print(json.dumps({"type": "agent_start"}), flush=True)
while True:
    time.sleep(1)
""",
    )

    manager = SessionManager()
    manager._runner.pi_bin = str(fake_pi)
    on = manager.start(
        prompt="alpha task",
        cwd=str(tmp_path),
        model="local/example-model",
        provider=None,
        tool_mode="read-only",
        include_events=False,
        unsafe_read_only=True,
    )
    off = manager.start(
        prompt="bravo task",
        cwd=str(tmp_path),
        model="local/example-model",
        provider=None,
        tool_mode="read-only",
        include_events=False,
        unsafe_read_only=False,
    )

    calls_path = tmp_path / "worker-calls.jsonl"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if calls_path.exists() and len(calls_path.read_text().splitlines()) >= 2:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("workers did not record their launches")

    records = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]
    on_call = next(r for r in records if "alpha task" in r["message"])
    off_call = next(r for r in records if "bravo task" in r["message"])

    def tools_of(call: dict) -> list[str]:
        argv = call["argv"]
        return argv[argv.index("--tools") + 1].split(",")

    # Policy on: full tools (bash for git/build/test) + guarded prompt.
    assert "bash" in tools_of(on_call)
    assert on_call["message"].startswith("IMPORTANT — READ-ONLY")

    # Policy off: locked read-only, no shell, prompt untouched.
    assert "bash" not in tools_of(off_call)
    assert off_call["message"] == "bravo task"

    # Either way the reported tool_mode stays the requested "read-only".
    modes = {s["agent_id"]: s["tool_mode"] for s in manager.summary()}
    assert modes[on.agent_id] == "read-only"
    assert modes[off.agent_id] == "read-only"

    manager.stop(on.agent_id)
    manager.stop(off.agent_id)


def test_session_times_out_only_on_inactivity(tmp_path: Path) -> None:
    # Streams activity for ~2s, then goes silent while still "running".
    streaming_pi = write_fake_pi(
        tmp_path,
        """#!/usr/bin/env python3
import json
import sys
import time

if "--list-models" in sys.argv:
    print("provider   model                    context")
    print("local  example-model  128K")
    raise SystemExit(0)

line = sys.stdin.readline()
request = json.loads(line)
print(json.dumps({"id": request["id"], "type": "response", "command": "prompt", "success": True}), flush=True)
print(json.dumps({"type": "agent_start"}), flush=True)
for _ in range(20):
    print(json.dumps({"type": "message_update", "delta": {"type": "text_delta", "text": "tok "}}), flush=True)
    time.sleep(0.1)
while True:
    time.sleep(1)
""",
    )

    manager = SessionManager()
    manager._runner.pi_bin = str(streaming_pi)
    started = manager.start(
        prompt="stream a while",
        cwd=str(tmp_path),
        model="local/example-model",
        provider=None,
        tool_mode="none",
        include_events=True,
    )
    session = manager._get(started.agent_id)
    session.inactivity_timeout_seconds = 0.5  # shorter than the total stream duration

    # While streaming, repeated activity keeps it alive well past the 0.5s window.
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        snapshot = manager.peek(started.agent_id)
        if snapshot.event_counts.get("message_update", 0) >= 8:  # ~0.8s of stream
            break
        time.sleep(0.05)
    else:
        raise AssertionError("fake Pi did not stream")
    assert snapshot.status == "running"
    assert snapshot.error is None

    # Once the stream goes silent, the inactivity watchdog aborts the turn.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        snapshot = manager.peek(started.agent_id)
        if snapshot.status == "timeout":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("stalled agent was not timed out")
    assert "no activity" in (snapshot.error or "")


def test_manager_close_cleans_up_sessions(tmp_path: Path) -> None:
    fake_pi = write_fake_pi(
        tmp_path,
        """#!/usr/bin/env python3
import json
import sys
import time

if "--list-models" in sys.argv:
    print("provider   model                    context")
    print("local  example-model  128K")
    raise SystemExit(0)

line = sys.stdin.readline()
request = json.loads(line)
print(json.dumps({"id": request["id"], "type": "response", "command": "prompt", "success": True}), flush=True)
print(json.dumps({"type": "agent_start"}), flush=True)
while True:
    time.sleep(1)
""",
    )

    manager = SessionManager(parent_id="cleanup-test")
    manager._runner.pi_bin = str(fake_pi)
    started = manager.start(
        prompt="keep running",
        cwd=str(tmp_path),
        model="local/example-model",
        provider=None,
        tool_mode="none",
        include_events=False,
    )

    assert manager.summary()[0]["status"] == "running"
    pid = manager._get(started.agent_id).process.pid
    assert manager.close(reason="parent closed") == 1
    assert manager.summary() == []
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError(f"worker pid {pid} was not cleaned up")


def test_active_listeners_tracked_while_parent_waits(tmp_path: Path) -> None:
    fake_pi = write_fake_pi(
        tmp_path,
        """#!/usr/bin/env python3
import json
import sys
import time

if "--list-models" in sys.argv:
    print("provider   model                    context")
    print("local  example-model  128K")
    raise SystemExit(0)

line = sys.stdin.readline()
request = json.loads(line)
print(json.dumps({"id": request["id"], "type": "response", "command": "prompt", "success": True}), flush=True)
print(json.dumps({"type": "agent_start"}), flush=True)
while True:
    time.sleep(1)
""",
    )

    manager = SessionManager()
    manager._runner.pi_bin = str(fake_pi)
    started = manager.start(
        prompt="watch me",
        cwd=str(tmp_path),
        model="local/example-model",
        provider=None,
        tool_mode="none",
        include_events=False,
    )

    # No turn ever completes, so a listener blocks until it times out.
    result: dict[str, object] = {}

    def do_listen() -> None:
        snapshot, timed_out = manager.listen(
            started.agent_id, after_turn_count=0, timeout_seconds=2
        )
        result["timed_out"] = timed_out
        result["snapshot"] = snapshot

    assert manager.peek(started.agent_id).active_listeners == 0

    watcher = threading.Thread(target=do_listen)
    watcher.start()
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if manager.peek(started.agent_id).active_listeners >= 1:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("active listener was never observed")

        data = manager.peek(started.agent_id).to_json(verbosity="summary")
        assert data["active_listeners"] == 1
        assert data["observing_with_piw"] is True
    finally:
        watcher.join(timeout=5)

    assert result["timed_out"] is True
    final = manager.peek(started.agent_id)
    assert final.active_listeners == 0
    assert final.to_json(verbosity="summary")["observing_with_piw"] is False
    manager.close()


def test_status_info_matches_snapshot_and_active_count(tmp_path: Path) -> None:
    fake_pi = write_fake_pi(
        tmp_path,
        """#!/usr/bin/env python3
import json
import sys

if "--list-models" in sys.argv:
    print("provider   model                    context")
    print("local  example-model  128K")
    raise SystemExit(0)

for line in sys.stdin:
    request = json.loads(line)
    if request.get("type") == "abort":
        break
    text = request.get("message", "")
    print(json.dumps({"id": request["id"], "type": "response", "command": "prompt", "success": True}), flush=True)
    message = {"role": "assistant", "content": [{"type": "text", "text": "echo:" + text}], "responseId": "r-" + text}
    print(json.dumps({"type": "message_end", "message": message}), flush=True)
    print(json.dumps({"type": "agent_end", "messages": [message]}), flush=True)
""",
    )

    manager = SessionManager()
    manager._runner.pi_bin = str(fake_pi)
    started = manager.start(
        prompt="hi",
        cwd=str(tmp_path),
        model="local/example-model",
        provider=None,
        tool_mode="none",
        include_events=False,
    )
    manager.listen(started.agent_id, after_turn_count=0, timeout_seconds=5)

    session = manager._sessions[started.agent_id]

    # Drive the session through every status-derivation branch and confirm the
    # lightweight accessor stays byte-for-byte identical to the full snapshot.
    def assert_consistent() -> None:
        with session._lock:
            status, model, provider = session._status_info_locked()
            snapshot = session._snapshot_locked(include_events=False)
        assert status == snapshot.status
        assert model == snapshot.model
        assert provider == snapshot.provider
        assert session.status_info() == (snapshot.status, snapshot.model, snapshot.provider)

    # idle (starting/running -> idle)
    assert_consistent()
    assert session.status_info()[0] == "idle"

    # running
    with session._lock:
        session._running = True
    assert_consistent()
    assert session.status_info()[0] == "running"
    with session._lock:
        session._running = False

    # evicted -> idle
    with session._lock:
        session._evicted = True
    assert_consistent()
    assert session.status_info()[0] == "idle"
    with session._lock:
        session._evicted = False

    # closed + stopped -> stopped
    with session._lock:
        session._closed = True
        session._status = "stopped"
    assert_consistent()
    assert session.status_info()[0] == "stopped"
    with session._lock:
        session._closed = False
        session._status = "idle"

    # active_model_count counts only starting/running sessions.
    assert manager.active_model_count(
        provider="local", model="example-model", match_provider=False
    ) == 0
    with session._lock:
        session._running = True
    assert manager.active_model_count(
        provider="local", model="example-model", match_provider=False
    ) == 1
    assert manager.active_model_count(
        provider="local", model="other-model", match_provider=False
    ) == 0
    with session._lock:
        session._running = False

    manager.close()
