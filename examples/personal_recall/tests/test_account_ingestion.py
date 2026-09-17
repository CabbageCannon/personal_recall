"""Offline tests for account-wide ingestion: the directory contract and the coverage report.

The hard rule of this phase is a negative one — **a conversation boundary must never be crossed**,
however close in time or similar in wording two conversations are. `test_account_isolation.py` attacks
that rule adversarially; this file covers the plumbing around it:

* what an account export directory is, and how identity is recovered from it;
* detected-versus-exported shards, because "no data loaded" must never be reported as "no memory
  exists" — a missing shard makes the whole account history PARTIAL and the report has to say so;
* the arithmetic of `AccountImportReport` (received / kept / duplicates / per-conversation coverage);
* `crossed_conversation_chunks`, the check that turns the rule into an assertion;
* the `--account` CLI refusing loudly instead of indexing nothing.

No model, no network, no real chat data: everything is synthetic and built in a scratch directory.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import recall  # noqa: E402
from memory import (  # noqa: E402
    ACCOUNT_MANIFEST_FILENAME,
    SESSION_LISTING_FILENAME,
    SessionConfig,
    build_account_sessions,
    build_sessions,
    crossed_conversation_chunks,
    discover_shard_directories,
    import_account,
    load_account_directory,
    read_account_manifest,
    shard_stem,
)

SCRATCH_ROOT = BASE_DIR / "tests" / "_scratch_account"


def ts(hour: int, minute: int = 0) -> int:
    return int(datetime(2026, 9, 20, hour, minute).timestamp())


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


def write_shard(root: Path, shard: str, listing, conversations, *, manifest=None):
    """Materialise one shard's export directory the way the orchestrator writes it."""
    directory = root / shard
    directory.mkdir(parents=True, exist_ok=True)
    (directory / SESSION_LISTING_FILENAME).write_text(
        json.dumps(listing, ensure_ascii=False), encoding="utf-8"
    )
    for talker, messages in conversations.items():
        (directory / f"{talker}_messages.json").write_text(
            json.dumps(messages, ensure_ascii=False), encoding="utf-8"
        )
    if manifest is not None:
        (root / ACCOUNT_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
    return directory


def build_account(root: Path) -> None:
    """Two shards, three conversations, deliberately interleaved in time.

    ``alice`` and ``bob`` share the display name 小王 on purpose: a display name is a label, never an
    identity, so having one must not merge or confuse them. ``alice``'s message in MSG2 repeats one
    from MSG3 with the same ``serverId`` (a shard migration), and one message carries no ``serverId``
    at all and so may never be deduplicated.
    """
    listing = [
        {"username": "alice", "displayName": "小王", "type": 0},
        {"username": "bob", "displayName": "小王", "type": 0},
    ]
    write_shard(
        root,
        "MSG0",
        listing,
        {
            "alice": [msg(1, "a-1", ts(10, 0), "我试了那个库")],
            "bob": [msg(1, "b-1", ts(10, 1), "吃饭吗")],
        },
        manifest={"detected_shards": ["MSG0.db", "MSG1.db", "MSG3.db"], "exported_shards": ["MSG0", "MSG3"]},
    )
    write_shard(
        root,
        "MSG3",
        [{"username": "alice", "displayName": "小王（改过备注）", "type": 0}],
        {
            "alice": [
                msg(2, "a-2", ts(10, 2), "冷启动慢"),
                msg(3, "a-3", ts(10, 2), "最后还是换回去了"),
                msg(4, None, ts(10, 3), "没有 serverId 的一条"),
            ]
        },
    )


def _reset() -> Path:
    shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    return SCRATCH_ROOT


# --------------------------------------------------------------------------------------------
# The directory contract
# --------------------------------------------------------------------------------------------


def test_identity_comes_from_the_listing_not_the_display_name():
    """Two conversations that share one display name stay two conversations."""
    root = _reset()
    try:
        build_account(root)
        layout = load_account_directory(root)
        ids = [descriptor.conversation_id for descriptor in layout.descriptors]
        assert ids == ["alice", "bob"]
        names = {descriptor.conversation_id: descriptor.display_name for descriptor in layout.descriptors}
        assert names == {"alice": "小王", "bob": "小王"}
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_exports_are_discovered_per_shard_and_merged_within_conversation():
    root = _reset()
    try:
        build_account(root)
        layout = load_account_directory(root)
        assert sorted(export.conversation_id for export in layout.exports) == [
            "alice",
            "alice",
            "bob",
        ]
        assert sorted(export.shard for export in layout.exports) == ["MSG0", "MSG0", "MSG3"]
        assert discover_shard_directories(root) == (root / "MSG0", root / "MSG3")
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_a_directory_without_message_exports_is_not_a_shard():
    """A stray folder must not enlarge the account or shift what looks exported."""
    root = _reset()
    try:
        build_account(root)
        (root / "notes").mkdir()
        (root / "notes" / "readme.txt").write_text("not an export", encoding="utf-8")
        assert discover_shard_directories(root) == (root / "MSG0", root / "MSG3")
        assert {export.shard for export in load_account_directory(root).exports} == {"MSG0", "MSG3"}
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_conversation_id_falls_back_to_the_file_name_without_a_listing():
    """Identity may come from the exporter's file name when no listing was recorded."""
    root = _reset()
    try:
        directory = root / "MSG0"
        directory.mkdir(parents=True)
        (directory / "carol_messages.json").write_text(
            json.dumps([msg(1, "c-1", ts(9, 0), "材料发你了")]), encoding="utf-8"
        )
        layout = load_account_directory(root)
        assert [export.conversation_id for export in layout.exports] == ["carol"]
        assert layout.descriptors == ()
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_an_unreadable_listing_is_noted_and_identity_falls_back():
    root = _reset()
    try:
        directory = root / "MSG0"
        directory.mkdir(parents=True)
        (directory / SESSION_LISTING_FILENAME).write_text("{not json", encoding="utf-8")
        (directory / "dave_messages.json").write_text(
            json.dumps([msg(1, "d-1", ts(9, 0), "在吗")]), encoding="utf-8"
        )
        layout = load_account_directory(root)
        assert [export.conversation_id for export in layout.exports] == ["dave"]
        assert any("could not read" in note for note in layout.notes)
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_missing_manifest_is_not_an_error_but_is_admitted():
    """Without a manifest or a real shard dir, completeness cannot be judged — and says so."""
    root = _reset()
    try:
        directory = root / "MSG0"
        directory.mkdir(parents=True)
        (directory / "alice_messages.json").write_text(
            json.dumps([msg(1, "a-1", ts(9, 0), "在吗")]), encoding="utf-8"
        )
        layout = load_account_directory(root)
        assert layout.shards_detected == ("MSG0",)
        assert any("never exported cannot be detected" in note for note in layout.notes)
        assert read_account_manifest(root) == {}
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_the_real_shard_directory_is_authoritative_over_the_manifest():
    """Only the actual Msg/Multi directory can prove a shard is missing."""
    root = _reset()
    try:
        build_account(root)
        multi = root / "_multi"
        multi.mkdir()
        for name in ("MSG0.db", "MSG1.db", "MSG3.db", "FTSMSG0.db"):
            (multi / name).write_bytes(b"")
        layout = load_account_directory(root, shard_dir=multi)
        # The manifest claims MSG0/MSG3 exported and omits MSG2; the real directory is what counts,
        # and FTS* indexes are not message shards.
        assert layout.shards_detected == ("MSG0", "MSG1", "MSG3")
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_a_shard_claimed_exported_but_holding_nothing_is_flagged():
    root = _reset()
    try:
        build_account(root)
        # MSG3 was claimed as exported but now holds no messages at all, so it stops being a shard
        # directory — which must be reported rather than silently shrinking the account.
        (root / "MSG3" / "alice_messages.json").unlink()
        layout = load_account_directory(root)
        assert any("no such export directory" in note for note in layout.notes)
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_shard_stem_normalises_sidecars_and_database_names():
    """``MSG2.db-wal`` and ``MSG2.db`` are the same shard; comparing raw names cries PARTIAL wrongly."""
    assert shard_stem("MSG2.db") == "MSG2"
    assert shard_stem("MSG2.db-wal") == "MSG2"
    assert shard_stem("MSG2.db-shm") == "MSG2"
    assert shard_stem("MSG2") == "MSG2"
    assert shard_stem(r"D:\WeChat\Msg\Multi\MSG2.db") == "MSG2"


# --------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------


def _import(root: Path):
    """Load a directory and import it the way ``recall.import_account_directory`` does."""
    layout = load_account_directory(root)
    return import_account(
        layout.exports,
        shards_detected=layout.shards_detected,
        discovered=layout.descriptors,
        filtered_conversations=layout.filtered_conversations,
    )


def test_report_counts_and_per_conversation_coverage():
    root = _reset()
    try:
        build_account(root)
        events, report = _import(root)
        # MSG0: alice 1 + bob 1; MSG3: alice 3.
        assert report.messages_received == 5
        assert report.messages_kept == 5
        assert report.duplicates_removed == 0
        assert report.undedupeable == 1
        assert report.conversations_discovered == 2
        assert report.conversations_imported == 2
        assert report.conversations_without_messages == ()
        assert report.first_timestamp == datetime(2026, 9, 20, 10, 0)
        assert report.last_timestamp == datetime(2026, 9, 20, 10, 3)
        rows = {row.conversation_id: row for row in report.per_conversation}
        assert rows["alice"].messages == 4
        assert rows["alice"].shards == ("MSG0", "MSG3")
        assert rows["bob"].messages == 1
        assert rows["bob"].shards == ("MSG0",)
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_a_duplicate_server_id_is_dropped_within_its_conversation_and_never_counted_as_kept():
    root = _reset()
    try:
        build_account(root)
        # The same message, migrated into a second shard.
        write_shard(
            root,
            "MSG2",
            [{"username": "alice", "displayName": "小王", "type": 0}],
            {"alice": [msg(9, "a-1", ts(10, 0), "我试了那个库")]},
        )
        events, report = _import(root)
        assert report.messages_received == 6
        assert report.messages_kept == 5
        assert report.duplicates_removed == 1
        assert events["alice"][0].id == "alice#sa-1"
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_an_unexported_shard_makes_the_account_partial_and_is_reported():
    """``No Data Loaded != No Memory Exists``: MSG1 is invisible, and the report must say PARTIAL."""
    root = _reset()
    try:
        build_account(root)
        _, report = _import(root)
        assert report.shards_detected == ("MSG0", "MSG1", "MSG3")
        assert report.shards_exported == ("MSG0", "MSG3")
        assert report.missing_shards == ("MSG1",)
        assert report.partial is True
        rendered = "\n".join(report.lines())
        assert "PARTIAL" in rendered
        assert "MSG1" in rendered
        assert report.as_dict()["partial"] is True
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_a_fully_exported_account_is_not_partial():
    root = _reset()
    try:
        build_account(root)
        (root / ACCOUNT_MANIFEST_FILENAME).write_text(
            json.dumps({"detected_shards": ["MSG0.db", "MSG3.db"], "exported_shards": ["MSG0", "MSG3"]}),
            encoding="utf-8",
        )
        _, report = _import(root)
        assert report.missing_shards == ()
        assert report.partial is False
        assert "PARTIAL" not in "\n".join(report.lines())
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_a_narrowed_export_is_partial_even_when_every_shard_was_exported():
    """A filtered export covers a few conversations but can leave every shard looking exported.

    That is the near-miss the shard check cannot see: ``missing_shards`` is empty, so a report that
    only compares shards calls the tree complete while the rest of the account is absent from it.
    The manifest's filter count is the one piece of evidence, and PARTIAL must follow from it.
    """
    root = _reset()
    try:
        build_account(root)
        (root / ACCOUNT_MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    "detected_shards": ["MSG0.db", "MSG3.db"],
                    "exported_shards": ["MSG0", "MSG3"],
                    "filtered_conversations": 1,
                }
            ),
            encoding="utf-8",
        )
        layout = load_account_directory(root)
        assert layout.filtered_conversations == 1
        assert any("narrowed" in note for note in layout.notes)

        _, report = _import(root)
        assert report.missing_shards == ()
        assert report.filtered_conversations == 1
        assert report.partial is True
        assert report.as_dict()["partial"] is True
        assert report.as_dict()["filtered_conversations"] == 1

        rendered = "\n".join(report.lines())
        assert "PARTIAL" in rendered
        assert "narrowed to 1 conversation(s)" in rendered, "the reason must be named, not implied"
        assert "were not exported" not in rendered, "no shard failed, so that warning is not due"
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_both_reasons_for_being_partial_are_reported_as_two_warnings():
    """A failed shard and a deliberately narrowed export are different problems; never one message."""
    root = _reset()
    try:
        build_account(root)  # MSG1 is detected but was never exported
        (root / ACCOUNT_MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    "detected_shards": ["MSG0.db", "MSG1.db", "MSG3.db"],
                    "exported_shards": ["MSG0", "MSG3"],
                    "filtered_conversations": 1,
                }
            ),
            encoding="utf-8",
        )
        _, report = _import(root)
        warnings = [line for line in report.lines() if line.startswith("WARNING")]
        assert len(warnings) == 2, warnings
        assert any("MSG1" in line and "were not exported" in line for line in warnings)
        assert any("narrowed" in line for line in warnings)
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_a_manifest_without_the_filter_key_reads_as_an_unfiltered_export():
    """An older or hand-written manifest has no filter key: it must load, and read as 0, not error."""
    root = _reset()
    try:
        build_account(root)  # its manifest predates the key
        layout = load_account_directory(root)
        assert layout.filtered_conversations == 0
        assert not any("narrowed" in note for note in layout.notes)

        _, report = _import(root)
        assert report.filtered_conversations == 0
        assert report.as_dict()["filtered_conversations"] == 0
        # MSG1 is still missing, so PARTIAL still holds - for the shard reason, and only that one.
        assert report.partial is True
        rendered = "\n".join(report.lines())
        assert "were not exported" in rendered
        assert "narrowed" not in rendered

        # A key that is present but unusable is also not allowed to break the load.
        (root / ACCOUNT_MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    "detected_shards": ["MSG0.db", "MSG1.db", "MSG3.db"],
                    "exported_shards": ["MSG0", "MSG3"],
                    "filtered_conversations": None,
                }
            ),
            encoding="utf-8",
        )
        assert load_account_directory(root).filtered_conversations == 0
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_a_discovered_conversation_that_yields_nothing_is_reported_not_dropped():
    root = _reset()
    try:
        build_account(root)
        (root / "MSG0" / "alice_messages.json").write_text("[]", encoding="utf-8")
        (root / "MSG3" / "alice_messages.json").write_text("[]", encoding="utf-8")
        _, report = _import(root)
        assert report.conversations_discovered == 2
        assert report.conversations_imported == 1
        assert report.conversations_without_messages == ("alice",)
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


# --------------------------------------------------------------------------------------------
# The hard rule, as an assertion
# --------------------------------------------------------------------------------------------


def test_account_sessions_never_cross_a_conversation_boundary():
    root = _reset()
    try:
        build_account(root)
        events, _ = _import(root)
        chunks = build_account_sessions(events, SessionConfig(max_chars=900))
        assert crossed_conversation_chunks(chunks, events) == ()
        assert {chunk.conversation_id for chunk in chunks} == {"alice", "bob"}
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_the_check_actually_detects_a_crossed_chunk():
    """The invariant check must be able to fail, or it proves nothing."""
    root = _reset()
    try:
        build_account(root)
        events, _ = _import(root)
        fused = build_sessions(
            [*events["alice"], *events["bob"]],
            config=SessionConfig(max_chars=900),
            conversation_id="fused",
        )
        # The builder's own boundary guard already refuses to fuse them, which is the first line of
        # defence. The checker below is the second: build a dirty chunk by hand and insist it is caught.
        assert crossed_conversation_chunks(fused, events) == ()

        dirty = replace(
            fused[0],
            id="fused-session-0001",
            conversation_id="fused",
            event_ids=(events["alice"][0].id, events["bob"][0].id),
        )
        assert crossed_conversation_chunks([dirty], events) == ("fused-session-0001",)
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_a_fused_stream_cannot_cross_even_with_room_to_spare():
    """The backstop: with the gap and budget limits wide open, only the boundary separates them.

    This is the phase's regression scenario — A at 10:00, B at 10:01, A at 10:02, B at 10:03 — with the
    size and silence rules deliberately neutralised, so a cross-conversation chunk could only come from
    the boundary rule being absent.
    """
    root = _reset()
    try:
        build_account(root)
        events, _ = _import(root)
        wide = SessionConfig(max_gap=timedelta(days=3650), max_chars=10_000_000)
        chunks = build_sessions(
            [*events["alice"], *events["bob"]], config=wide, conversation_id="fused"
        )
        # The caller's parameter is what the chunks are labelled with, so the label proves nothing here.
        # What must hold is that no chunk *contains* two conversations — checked from the event ids,
        # which is why the guard is written that way.
        assert crossed_conversation_chunks(chunks, events) == ()
        assert len(chunks) == 2
        assert [chunk.n_events for chunk in chunks] == [4, 1]
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_interleaved_conversations_do_not_shatter_into_one_chunk_per_message():
    """Building per conversation is what keeps both coherence *and* isolation."""
    root = _reset()
    try:
        build_account(root)
        events, _ = _import(root)
        chunks = build_account_sessions(events, SessionConfig(max_chars=900))
        by_conversation = {chunk.conversation_id: chunk for chunk in chunks}
        assert len(by_conversation) == 2
        # alice's four messages are minutes apart and must form one session, not four.
        assert len([c for c in chunks if c.conversation_id == "alice"]) == 1
        assert by_conversation["alice"].n_events == 4
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


# --------------------------------------------------------------------------------------------
# The CLI must refuse loudly rather than index nothing
# --------------------------------------------------------------------------------------------


def test_cli_account_import_is_checked_against_a_crossed_chunk(monkeypatch):
    """The real import path asserts the invariant, so a regression cannot ship silently."""
    root = _reset()
    try:
        build_account(root)
        chunks, report = recall.import_account_directory(root)
        assert report.conversations_imported == 2
        assert {chunk.conversation_id for chunk in chunks} == {"alice", "bob"}

        def fake_crossed(*_args, **_kwargs):
            return ("fused-session-0001",)

        monkeypatch.setattr(recall, "crossed_conversation_chunks", fake_crossed)
        with pytest.raises(AssertionError, match="conversation boundary crossed"):
            recall.import_account_directory(root)
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_cli_account_import_carries_the_filter_count_through_to_the_report():
    """The CLI's own path must not drop the manifest's filter count on the way to the report."""
    root = _reset()
    try:
        build_account(root)
        (root / ACCOUNT_MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    "detected_shards": ["MSG0.db", "MSG3.db"],
                    "exported_shards": ["MSG0", "MSG3"],
                    "filtered_conversations": 2,
                }
            ),
            encoding="utf-8",
        )
        _, report = recall.import_account_directory(root)
        assert report.filtered_conversations == 2
        assert report.partial is True
        assert "narrowed to 2 conversation(s)" in "\n".join(report.lines())
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_cli_errors_on_a_directory_with_no_shard_exports(capsys):
    root = _reset()
    try:
        (root / "empty").mkdir(parents=True, exist_ok=True)
        code = _run_main(["question", "--account", str(root)])
        assert code == 2
        assert "no shard export directories" in capsys.readouterr().err
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_cli_errors_on_a_nonexistent_account_directory(capsys):
    _reset()
    try:
        code = _run_main(["question", "--account", str(SCRATCH_ROOT / "nope")])
        assert code == 2
        assert "is not a directory" in capsys.readouterr().err
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def _run_main(argv):
    """Invoke ``recall.main`` with ``argv`` without letting argparse see pytest's arguments."""
    import sys as _sys

    previous = _sys.argv
    _sys.argv = ["recall.py", *argv]
    try:
        return recall.main()
    finally:
        _sys.argv = previous
