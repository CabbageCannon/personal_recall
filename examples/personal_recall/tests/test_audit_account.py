"""Tests for the full-account audit (Phase 20.5).

The audit exists to answer two questions about a tree holding a *real* account, and both of them are
easy to answer wrongly in a way that looks fine:

* **Is it complete?** A tree whose every shard directory is present but which was exported with a
  conversation filter looks whole. So does one missing a shard, if completeness is judged by the
  directories rather than against the live ``Msg/Multi``.
* **Did it stay separated?** The boundary check is the phase's hard rule, and an audit that reports
  "0 violations" because it never ran the check is worse than no audit at all.

There is a third question this file spends most of its effort on, because the failure is the only
irreversible one: **does the audit leak an identity?** Its whole output is designed to be pasted into
an issue, so an audit that prints a talker or a display name puts a real person's chat identity into
whatever it is pasted into. Every assertion about the report's contents is checked against the
rendered text *and* the serialised JSON, not just against the objects.

Fully synthetic, offline and deterministic: no network, no model, no real WeChat data, no subprocess.
Trees are written as real files because the audit reads from disk; they go to a workspace-local scratch
directory (``pytest``'s ``tmp_path`` cannot be created in this environment).
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from audit_account import (  # noqa: E402
    anon,
    build_report,
    count_message_rows,
    explain_missing_conversations,
    render,
    tree_totals,
)

SCRATCH_ROOT = Path(__file__).resolve().parent / "_scratch_audit"

#: Deliberately unmistakable strings. If any of these reaches the output, the audit has leaked an
#: identity, and a grep for them is the test.
TALKER_A = "wxid_leaky_alpha"
TALKER_B = "wxid_leaky_beta"
GROUP = "987654321@chatroom"
DISPLAY = "真实的备注名"


def ts(hour: int, minute: int = 0, day: int = 17) -> int:
    return int(datetime(2026, 9, day, hour, minute).timestamp())


def msg(text: str, when: int, *, local_id: int = 1, server_id=None) -> dict:
    entry = {
        "localId": local_id,
        "localType": 1,
        "createTime": when,
        "isSend": 1,
        "senderUsername": "wxid_synthetic",
        "parsedContent": text,
    }
    if server_id is not None:
        entry["serverId"] = server_id
    return entry


def write_shard(directory: Path, exports: dict[str, list], *, listing: bool = True) -> None:
    """Write one shard's ``{talker}_messages.json`` files (and its listing) — the exporter's layout."""
    directory.mkdir(parents=True, exist_ok=True)
    if listing:
        (directory / "sessions.json").write_text(
            json.dumps(
                [
                    {
                        "conversation_id": talker,
                        "display_name": DISPLAY,
                        "conversation_type": "group" if "@chatroom" in talker else "direct",
                    }
                    for talker in exports
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    for talker, messages in exports.items():
        (directory / f"{talker}_messages.json").write_text(
            json.dumps(messages, ensure_ascii=False), encoding="utf-8"
        )


@pytest.fixture()
def scratch() -> Path:
    directory = SCRATCH_ROOT / uuid4().hex[:8]
    directory.mkdir(parents=True, exist_ok=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _account(scratch: Path, *, shards=("MSG0", "MSG1"), manifest: dict | None = None) -> Path:
    """A two-shard account: one direct conversation in each shard, one group spanning both."""
    root = scratch / "account"
    write_shard(
        root / shards[0],
        {
            TALKER_A: [msg("第一条", ts(10, 0), server_id="a-1"), msg("第二条", ts(10, 5), server_id="a-2")],
            GROUP: [msg("群里的话", ts(10, 2), server_id="g-1")],
        },
    )
    if len(shards) > 1:
        write_shard(
            root / shards[1],
            {
                TALKER_B: [msg("另一边的第一句", ts(11, 0), server_id="b-1")],
                GROUP: [msg("群里后来又说", ts(11, 30), server_id="g-2")],
            },
        )
    (root / "shard_manifest.json").write_text(
        json.dumps(
            manifest
            if manifest is not None
            else {"detected_shards": list(shards), "exported_shards": list(shards)}
        ),
        encoding="utf-8",
    )
    return root


# ---------------------------------------------------------------------------------------------
# Counting: the tree and the parser are counted by different code on purpose
# ---------------------------------------------------------------------------------------------


def test_report_counts_the_tree(scratch: Path) -> None:
    root = _account(scratch)
    report = build_report(root, None)

    assert report["conversations_discovered"] == 3
    assert report["conversations_imported"] == 3
    assert report["messages_kept"] == 5
    assert report["duplicates_removed"] == 0
    assert report["chunks"] >= 3
    assert report["cross_shard_conversations"] == 1, "only the group spans both shards"
    assert report["crossed_chunks"] == []
    assert report["partial"] is False
    assert report["filtered_conversations"] == 0

    tree = report["tree"]
    # Four files: the group is exported once per shard it appears in, so it is two of them.
    assert tree["conversation_files"] == 4
    assert tree["message_rows_in_files"] == 5
    assert tree["session_listings"] == 2
    assert tree["unreadable_files"] == []


def test_coverage_is_the_union_span_not_a_per_shard_span(scratch: Path) -> None:
    report = build_report(_account(scratch), None)
    assert report["coverage"]["first"].startswith("2026-09-17 10:00")
    assert report["coverage"]["last"].startswith("2026-09-17 11:30")


def test_count_message_rows_handles_every_envelope_the_exporter_can_emit() -> None:
    """A miscount here reports an empty conversation as a fact rather than as a parsing gap."""
    assert count_message_rows([{"a": 1}, {"b": 2}]) == 2
    assert count_message_rows({"messages": [1, 2, 3]}) == 3
    assert count_message_rows({"data": [1]}) == 1
    assert count_message_rows({"items": [1, 1]}) == 2
    assert count_message_rows({"unexpected": "shape"}) == 0
    assert count_message_rows("not a container") == 0


def test_tree_totals_counts_an_unreadable_file_instead_of_failing(scratch: Path) -> None:
    """One corrupt export must not hide the other 448, and must not be silently skipped either."""
    root = _account(scratch)
    (root / "MSG0" / f"{TALKER_A}_messages.json").write_text("{ not json", encoding="utf-8")

    tree = tree_totals(root)
    assert tree["conversation_files"] == 4
    assert tree["message_rows_in_files"] == 3, "five rows, minus the two in the corrupt file"
    assert len(tree["unreadable_files"]) == 1


# ---------------------------------------------------------------------------------------------
# Completeness: two ways to be incomplete, and one of them leaves every directory present
# ---------------------------------------------------------------------------------------------


def test_a_missing_shard_is_reported_as_partial(scratch: Path) -> None:
    """Detected against the live Msg/Multi: MSG1 exists on disk but produced no export."""
    root = _account(scratch, shards=("MSG0",))
    report = build_report(root, None)
    # Without a manifest claim about MSG1, completeness falls back to the directories present.
    assert report["partial"] is False

    # With the manifest recording that MSG1 was detected, the gap becomes visible.
    root2 = _account(scratch / "second", shards=("MSG0",))
    (root2 / "shard_manifest.json").write_text(
        json.dumps({"detected_shards": ["MSG0", "MSG1"], "exported_shards": ["MSG0"]}),
        encoding="utf-8",
    )
    report2 = build_report(root2, None)
    assert report2["missing_shards"] == ["MSG1"]
    assert report2["partial"] is True
    assert "PARTIAL                : True" in render(report2)


def test_a_filtered_export_is_partial_even_with_every_shard_present(scratch: Path) -> None:
    """The shape that hid the bug in Phase 19: no shard missing, most conversations absent."""
    root = _account(
        scratch,
        manifest={
            "detected_shards": ["MSG0", "MSG1"],
            "exported_shards": ["MSG0", "MSG1"],
            "filtered_conversations": 2,
        },
    )
    report = build_report(root, None)
    assert report["missing_shards"] == []
    assert report["filtered_conversations"] == 2
    assert report["partial"] is True
    assert "filtered conversations : 2" in render(report)


# ---------------------------------------------------------------------------------------------
# The hard rule
# ---------------------------------------------------------------------------------------------


def test_a_clean_tree_passes_the_boundary_check(scratch: Path) -> None:
    text = render(build_report(_account(scratch), None))
    assert "PASS - no chunk holds events from two conversations" in text
    assert "FAIL" not in text


def test_a_crossed_chunk_is_rendered_as_a_failure_not_a_caveat() -> None:
    """The renderer's failure path, driven directly so the message can never be softened by accident."""
    text = render(
        {
            "shards_detected": [],
            "shards_exported": [],
            "missing_shards": [],
            "filtered_conversations": 0,
            "partial": False,
            "conversations_discovered": 0,
            "conversations_imported": 0,
            "conversations_without_messages": 0,
            "missing_causes": {},
            "messages_kept": 0,
            "messages_received": 0,
            "duplicates_removed": 0,
            "undedupeable": 0,
            "skipped_messages": 0,
            "coverage": {"first": None, "last": None},
            "cross_shard_conversations": 0,
            "chunks": 1,
            "crossed_chunks": ["alice-session-0001"],
            "largest_conversations": [],
            "largest_share_of_total": 0.0,
            "per_conversation": [],
            "tree": {
                "conversation_files": 0,
                "message_rows_in_files": 0,
                "session_listings": 0,
                "entries_in_listings": 0,
                "unreadable_files": [],
            },
            "notes": [],
        }
    )
    assert "FAIL" in text
    assert "1 chunk(s) mix two conversations" in text


# ---------------------------------------------------------------------------------------------
# The irreversible failure: an identity in the output
# ---------------------------------------------------------------------------------------------


def test_the_report_contains_no_conversation_identity(scratch: Path) -> None:
    """The audit output is built to be pasted into an issue; a talker in it is a real leak."""
    report = build_report(_account(scratch), None)
    rendered = render(report)
    serialised = json.dumps(report, ensure_ascii=False)

    for secret in (TALKER_A, TALKER_B, GROUP, DISPLAY, "leaky", "987654321"):
        assert secret not in rendered, f"{secret!r} reached the rendered audit"
        assert secret not in serialised, f"{secret!r} reached the serialised report"

    # The digests are what is shown instead, so the table must actually be populated.
    assert anon(TALKER_A) in rendered
    assert report["per_conversation"], "the outlier table must not be empty for a non-empty account"


def test_an_unreadable_file_is_named_by_digest(scratch: Path) -> None:
    """Even the failure path must identify the file by digest: it is the line most likely to be quoted."""
    root = _account(scratch)
    (root / "MSG0" / f"{TALKER_A}_messages.json").write_text("{ not json", encoding="utf-8")

    named = tree_totals(root)["unreadable_files"]
    assert named == [f"MSG0/{anon(TALKER_A)}"]
    assert all(TALKER_A not in entry for entry in named)


def test_build_report_refuses_a_corrupt_export_rather_than_reporting_it_as_empty(
    scratch: Path,
) -> None:
    """Pins the current behaviour so a future change to it is deliberate.

    ``import_account`` raises on an unreadable export. That is the *safe* direction — the alternative
    is reporting a conversation as having no messages when it in fact has messages that could not be
    parsed, which is the failure this whole project keeps paying for. What it does not do is say
    *which* file: at 449 exports the bare ``JSONDecodeError`` is the operational gap, not the raise.
    """
    root = _account(scratch)
    (root / "MSG0" / f"{TALKER_A}_messages.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        build_report(root, None)


def test_display_names_from_a_listing_never_reach_the_output(scratch: Path) -> None:
    """A listing is the one file in the tree that carries display names; none may survive."""
    rendered = render(build_report(_account(scratch), None))
    assert DISPLAY not in rendered


# ---------------------------------------------------------------------------------------------
# Why a discovered conversation produced nothing
# ---------------------------------------------------------------------------------------------


def test_missing_conversations_are_classified_by_cause(scratch: Path) -> None:
    """Three different causes with three different next actions must not be summed into one number."""
    root = scratch / "account"
    write_shard(root / "MSG0", {"wxid_present_empty": []})
    (root / "MSG0" / "wxid_rows_all_skipped_messages.json").write_text(
        json.dumps([{"localType": 1}]), encoding="utf-8"
    )  # a row with no createTime/parsedContent: present, but nothing parses

    causes = explain_missing_conversations(
        root, ["wxid_absent_entirely", "wxid_present_empty", "wxid_rows_all_skipped"]
    )
    assert causes.get("no export file at all") == 1
    assert causes.get("file present but holding 0 messages") == 1
    assert causes.get("rows present but all skipped by the parser") == 1


def test_a_conversation_with_no_messages_never_becomes_a_chunk(scratch: Path) -> None:
    root = scratch / "account"
    write_shard(root / "MSG0", {"wxid_silent": [], TALKER_A: [msg("有内容", ts(9, 0), server_id="x-1")]})
    (root / "shard_manifest.json").write_text(
        json.dumps({"detected_shards": ["MSG0"], "exported_shards": ["MSG0"]}), encoding="utf-8"
    )
    report = build_report(root, None)
    assert report["conversations_discovered"] == 2
    assert report["conversations_imported"] == 1
    assert report["conversations_without_messages"] == 1
    assert "no export file at all" not in report["missing_causes"], "the file IS there, just empty"
