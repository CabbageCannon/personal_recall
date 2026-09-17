"""Offline tests for the multi-shard merge (no model, no network, no real chat data).

The brief names two requirements explicitly, and they are the two that would silently corrupt a
history if they were wrong:

* **never assume ``MSG0`` is oldest and ``MSG2`` newest** — chronology comes from ``createTime``, and
  file names/mtimes are not evidence of content order;
* **a contact spanning shards must aggregate to a global ``MAX(CreateTime)``** — which only works if
  the shards are merged *before* session building.

Everything here is synthetic.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from memory import SessionConfig, build_sessions  # noqa: E402
from memory.shards import (  # noqa: E402
    discover_message_shards,
    discover_shard_files,
    load_and_merge,
    merge_shard_events,
)
from memory.weflow import parse_weflow_events  # noqa: E402


def ts(hour: int, minute: int = 0, day: int = 17) -> int:
    return int(datetime(2026, 9, day, hour, minute).timestamp())


def msg(local_id, server_id, when, text, *, is_send=1):
    return {
        "localId": local_id,
        "serverId": server_id,
        "localType": 1,
        "createTime": when,
        "isSend": is_send,
        "senderUsername": "wxid_synthetic",
        "parsedContent": text,
    }


def parse(messages):
    return parse_weflow_events(messages, conversation_id="synthetic")


def _scratch(name: str) -> Path:
    path = BASE_DIR / "tests" / "_scratch_shards" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- ordering: chronology from data, never from the shard name --------------------------------


def test_merge_sorts_across_shards_by_timestamp() -> None:
    early = parse([msg(1, "s1", ts(9), "morning")])
    late = parse([msg(1, "s9", ts(21), "night")])
    events, _ = merge_shard_events([("shard_late", late), ("shard_early", early)])
    assert [e.text for e in events] == ["morning", "night"]


def test_shard_name_does_not_imply_chronology() -> None:
    """'MSG2' must not be assumed newer: here it holds the OLDEST messages."""
    shard0 = parse([msg(1, "s1", ts(18), "newest content, named MSG0")])
    shard2 = parse([msg(1, "s2", ts(6), "oldest content, named MSG2")])
    events, _ = merge_shard_events([("MSG0", shard0), ("MSG2", shard2)])
    assert [e.text for e in events] == ["oldest content, named MSG2", "newest content, named MSG0"]


def test_tie_break_is_deterministic_by_label_then_position() -> None:
    a = parse([msg(1, "sa1", ts(12), "alpha-1"), msg(2, "sa2", ts(12), "alpha-2")])
    b = parse([msg(1, "sb1", ts(12), "beta-1")])
    events, _ = merge_shard_events([("beta", b), ("alpha", a)])
    assert [e.text for e in events] == ["alpha-1", "alpha-2", "beta-1"]
    # and stable when the same input is offered in a different order
    again, _ = merge_shard_events([("alpha", a), ("beta", b)])
    assert [e.text for e in again] == [e.text for e in events]


def test_merging_is_deterministic() -> None:
    shards = [("a", parse([msg(1, "s1", ts(10), "x")])), ("b", parse([msg(2, "s2", ts(9), "y")]))]
    first, _ = merge_shard_events(shards)
    second, _ = merge_shard_events(shards)
    assert [e.id for e in first] == [e.id for e in second]


# --- deduplication ---------------------------------------------------------------------------


def test_duplicate_server_id_across_shards_is_removed_once() -> None:
    first = parse([msg(1, "srv-shared", ts(11), "same message")])
    second = parse([msg(7, "srv-shared", ts(11), "same message")])
    events, report = merge_shard_events([("a", first), ("b", second)])
    assert len(events) == 1
    assert report.duplicates_removed == 1
    assert report.received == 2 and report.kept == 1


def test_messages_without_server_id_are_never_deduplicated() -> None:
    """localId repeats across shards by design, so deduplicating on it would drop real history."""
    first = parse([msg(1, None, ts(11), "one")])
    second = parse([msg(1, None, ts(11), "two")])
    events, report = merge_shard_events([("a", first), ("b", second)])
    assert len(events) == 2, "a message was dropped on a weak identity"
    assert report.undedupeable == 2
    assert any("cannot be deduplicated" in line for line in report.lines())


def test_ids_stay_unique_even_with_colliding_weak_identities() -> None:
    first = parse([msg(1, None, ts(11), "one")])
    second = parse([msg(1, None, ts(11), "two")])
    events, _ = merge_shard_events([("a", first), ("b", second)])
    assert len({e.id for e in events}) == 2


def test_provenance_is_recorded_per_event() -> None:
    events, report = merge_shard_events(
        [("part_a", parse([msg(1, "s1", ts(11), "x")])), ("part_b", parse([msg(2, "s2", ts(12), "y")]))]
    )
    by_text = {e.text: e.metadata["source_shard"] for e in events}
    assert by_text == {"x": "part_a", "y": "part_b"}
    assert report.shards[0].label == "part_a"


# --- sessions aggregate across shards --------------------------------------------------------


def test_a_conversation_spanning_shards_becomes_one_session() -> None:
    """Merging after session building would cut this into two; merging before keeps it whole."""
    early = parse([msg(1, "s1", ts(11, 0), "first half"), msg(2, "s2", ts(11, 30), "still first")])
    late = parse([msg(3, "s3", ts(12, 0), "second half in another shard")])
    events, _ = merge_shard_events([("shardA", early), ("shardB", late)])
    sessions = build_sessions(events, SessionConfig(max_chars=900), conversation_id="weflow")
    assert len(sessions) == 1, f"fragmented into {len(sessions)} sessions"
    assert sessions[0].n_events == 3


def test_session_end_time_is_the_global_max_across_shards() -> None:
    """The brief's 'global MAX(CreateTime)': the latest message wins regardless of which shard holds it."""
    shard_a = parse([msg(1, "s1", ts(11, 0), "early")])
    shard_b = parse([msg(2, "s2", ts(14, 30), "latest, in the other shard")])
    events, _ = merge_shard_events([("a", shard_a), ("b", shard_b)])
    sessions = build_sessions(events, SessionConfig(max_chars=900), conversation_id="weflow")
    assert sessions[0].end_time == datetime(2026, 9, 17, 14, 30)
    assert sessions[0].start_time == datetime(2026, 9, 17, 11, 0)


def test_merged_coverage_is_the_union_of_the_shards() -> None:
    events, report = merge_shard_events(
        [
            ("a", parse([msg(1, "s1", ts(8), "a")])),
            ("b", parse([msg(2, "s2", ts(20), "b")])),
        ]
    )
    assert report.first_timestamp == datetime(2026, 9, 17, 8, 0)
    assert report.last_timestamp == datetime(2026, 9, 17, 20, 0)
    assert any("coverage" in line for line in report.lines())


def test_per_shard_spans_are_reported() -> None:
    _, report = merge_shard_events(
        [
            ("a", parse([msg(1, "s1", ts(8), "a"), msg(2, "s2", ts(9), "b")])),
            ("b", parse([msg(3, "s3", ts(20), "c")])),
        ]
    )
    spans = {info.label: info.span() for info in report.shards}
    assert spans["a"].startswith("2026-09-17 08:00")
    assert spans["b"].startswith("2026-09-17 20:00")


# --- discovery (read-only, names only) -------------------------------------------------------


def _make_multi_dir(name: str) -> Path:
    directory = _scratch(name)
    for shard in ("MSG0.db", "MSG1.db", "MSG2.db"):
        (directory / shard).write_bytes(b"")
    for extra in ("MSG0.db-wal", "MSG0.db-shm", "FTSMSG0.db", "MediaMSG0.db", "config.ini"):
        (directory / extra).write_bytes(b"")
    return directory


def test_message_shard_discovery_lists_all_shards_and_ignores_sidecars() -> None:
    directory = _make_multi_dir("multi")
    try:
        assert discover_message_shards(directory) == ["MSG0.db", "MSG1.db", "MSG2.db"]
    finally:
        shutil.rmtree(directory.parent, ignore_errors=True)


def test_message_shard_discovery_on_a_missing_directory_is_empty() -> None:
    assert discover_message_shards(BASE_DIR / "does_not_exist") == []


def test_shard_export_discovery_takes_json_files_deterministically() -> None:
    directory = _scratch("exports")
    try:
        for name in ("MSG2.json", "MSG0.json", "MSG1.json"):
            (directory / name).write_text("[]", encoding="utf-8")
        (directory / "notes.txt").write_text("ignore me", encoding="utf-8")
        assert [p.name for p in discover_shard_files(directory)] == [
            "MSG0.json",
            "MSG1.json",
            "MSG2.json",
        ]
    finally:
        shutil.rmtree(directory.parent, ignore_errors=True)


# --- loading from disk: loud, not silent -----------------------------------------------------


def test_load_and_merge_reads_several_files_and_labels_them() -> None:
    directory = _scratch("load")
    try:
        (directory / "shard_a.json").write_text(
            json.dumps([msg(1, "s1", ts(11), "from a")]), encoding="utf-8"
        )
        (directory / "shard_b.json").write_text(
            json.dumps([msg(2, "s2", ts(12), "from b")]), encoding="utf-8"
        )
        events, report = load_and_merge(discover_shard_files(directory))
        assert [e.text for e in events] == ["from a", "from b"]
        assert {i.label for i in report.shards} == {"shard_a", "shard_b"}
    finally:
        shutil.rmtree(directory.parent, ignore_errors=True)


def test_a_malformed_shard_raises_rather_than_shrinking_the_history() -> None:
    """A silently skipped shard is a silently smaller history — the failure this phase prevents."""
    directory = _scratch("broken")
    try:
        (directory / "good.json").write_text(json.dumps([msg(1, "s1", ts(11), "ok")]), encoding="utf-8")
        (directory / "broken.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_and_merge(discover_shard_files(directory))
    finally:
        shutil.rmtree(directory.parent, ignore_errors=True)


def test_envelope_shaped_shard_exports_merge_too() -> None:
    directory = _scratch("envelope")
    try:
        for label, when in (("a", ts(11)), ("b", ts(12))):
            (directory / f"{label}.json").write_text(
                json.dumps(
                    {
                        "schema": "weflow-message/v1",
                        "coverage": {"shards": [label]},
                        "messages": [msg(1, f"srv-{label}", when, f"from {label}")],
                    }
                ),
                encoding="utf-8",
            )
        events, report = load_and_merge(discover_shard_files(directory))
        assert [e.text for e in events] == ["from a", "from b"]
        assert len(report.shards) == 2
    finally:
        shutil.rmtree(directory.parent, ignore_errors=True)


def test_empty_shard_list_merges_to_nothing_without_raising() -> None:
    events, report = merge_shard_events([])
    assert events == []
    assert report.kept == 0 and report.first_timestamp is None


# --- CLI wiring: a directory means "merge these shards" ---------------------------------------


def test_recall_treats_a_directory_as_multi_shard() -> None:
    from recall import corpus_files, is_multi_shard

    directory = _scratch("cli_dir")
    try:
        for name in ("MSG0.json", "MSG2.json"):
            (directory / name).write_text("[]", encoding="utf-8")
        assert is_multi_shard(directory) is True
        assert [p.name for p in corpus_files(directory)] == ["MSG0.json", "MSG2.json"]
    finally:
        shutil.rmtree(directory.parent, ignore_errors=True)


def test_recall_treats_a_file_as_a_single_shard() -> None:
    from recall import corpus_files, is_multi_shard

    directory = _scratch("cli_file")
    try:
        path = directory / "MSG2.json"
        path.write_text("[]", encoding="utf-8")
        assert is_multi_shard(path) is False
        assert corpus_files(path) == [path]
    finally:
        shutil.rmtree(directory.parent, ignore_errors=True)
