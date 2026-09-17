"""Adversarial offline tests for the account-level ingestion layer (Phase 19).

One invariant is under attack here and nothing else:

    **A conversation boundary must never be crossed.**

Two messages belonging to different conversations must never end up in the same ``MemoryChunk``, no
matter how close in time, how similar in text, or whether the two conversations share a display name.
Everything else in this file (dedupe, coverage, determinism, group detection) is exercised because a
bug in it is a way to *reach* a violation: a dedupe key that is not conversation-scoped silently
deletes one conversation's message, a grouping key taken from a display name silently merges two.

Identity is the **talker** (``alice``, ``group_1@chatroom``), never a nickname.

Fully synthetic, offline and deterministic: no network, no model, no real WeChat data, no subprocess.
Shard exports are written as real files on disk because ``import_account`` reads from disk; they go to
a workspace-local scratch directory (``pytest``'s ``tmp_path`` cannot be created in this environment)
which is removed in the fixture teardown.
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

from memory import SessionConfig  # noqa: E402
from memory.account import (  # noqa: E402
    build_account_sessions,
    crossed_conversation_chunks,
    exports_from_directory,
    group_exports_by_conversation,
    import_account,
    shard_stem,
)
from memory.conversations import (  # noqa: E402
    ConversationDescriptor,
    conversation_id_from_export_filename,
    conversation_type_of,
)
from memory.events import MemoryEvent  # noqa: E402

SCRATCH_ROOT = Path(__file__).resolve().parent / "_scratch_isolation"

#: Shard names, chosen so the misreading "MSG1 is newer than MSG0" would be visible.
SHARD_A = "MSG0"
SHARD_B = "MSG1"
SHARD_C = "MSG2"


# --- synthetic data ---------------------------------------------------------------------------


def ts(hour: int, minute: int = 0, day: int = 17) -> int:
    """Unix seconds for a local-time wall clock, matching the adapter's ``createTime`` handling."""
    return int(datetime(2026, 9, day, hour, minute).timestamp())


def msg(
    text: str,
    when: int,
    *,
    local_id: int = 1,
    server_id=None,
    is_send: int = 1,
    display_name: str | None = None,
    sender: str = "wxid_synthetic",
) -> dict:
    """One WeFlow-shaped message. ``serverId`` is absent (not null) when not given, which is the
    shape that must never be deduplicated."""
    entry = {
        "localId": local_id,
        "localType": 1,
        "createTime": when,
        "isSend": is_send,
        "senderUsername": sender,
        "parsedContent": text,
    }
    if server_id is not None:
        entry["serverId"] = server_id
    if display_name is not None:
        # A display name is metadata. The adapter must not read it as an identity.
        entry["displayName"] = display_name
    return entry


def write_shard(directory: Path, exports: dict[str, list]) -> None:
    """Write ``{talker}_messages.json`` files for one shard — the exporter's on-disk layout."""
    directory.mkdir(parents=True, exist_ok=True)
    for talker, messages in exports.items():
        (directory / f"{talker}_messages.json").write_text(
            json.dumps(messages, ensure_ascii=False), encoding="utf-8"
        )


@pytest.fixture()
def scratch() -> Path:
    """A clean workspace-local directory (``tmp_path`` is unavailable in this environment)."""
    directory = SCRATCH_ROOT / f"{uuid4().hex[:8]}"
    shutil.rmtree(directory, ignore_errors=True)  # a leftover from a crashed run must not leak in
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)
        # Leave nothing behind once the last test is done.
        try:
            SCRATCH_ROOT.rmdir()
        except OSError:
            pass


def session_config(**overrides) -> SessionConfig:
    """Default segmentation unless a test wants to relax the gap/budget guards on purpose."""
    return SessionConfig(**overrides) if overrides else SessionConfig()


def assert_no_chunk_crosses_conversations(
    chunks, events_by_conversation: dict[str, list[MemoryEvent]]
) -> None:
    """The invariant, checked the only way that cannot be fooled.

    ``chunk.conversation_id`` is the builder's own claim; ``chunk.event_ids`` is the evidence. Every
    id is mapped back to the conversation that actually owns the event, and a chunk carrying two
    owners is a violation even if its ``conversation_id`` looks clean. The library's own runtime
    checker (``crossed_conversation_chunks``, the one ``recall.import_account_directory`` raises on)
    is asserted alongside it, so the two can never drift apart silently.
    """
    owner_of: dict[str, str] = {
        event.id: conversation_id
        for conversation_id, events in events_by_conversation.items()
        for event in events
    }
    for chunk in chunks:
        owners = {owner_of[event_id] for event_id in chunk.event_ids}
        assert len(owners) == 1, (
            f"chunk {chunk.id!r} mixes conversations {sorted(owners)} "
            f"(declared {chunk.conversation_id!r})"
        )
        # `build_sessions` labels each chunk with the caller's `conversation_id` parameter, and every
        # real caller passes the id its events carry (the account path passes the conversation it
        # parsed with, the text path "txt"/"stress"). So when a chunk claims to be a *known*
        # conversation, it must be the right one. A deliberately neutral label — the mixed-stream
        # backstop below passes "account" for a stream holding several conversations — is not a claim
        # about ownership, and the isolation assertion above plus the library checker are what hold
        # there. Demanding `owners == {chunk.conversation_id}` unconditionally would instead re-assert
        # the parameter-ignoring behaviour that `test_memory_sessions` freezes against.
        if chunk.conversation_id in events_by_conversation:
            assert owners == {chunk.conversation_id}, (
                f"chunk {chunk.id!r} declares {chunk.conversation_id!r} but its events belong to "
                f"{sorted(owners)}"
            )
    assert crossed_conversation_chunks(chunks, events_by_conversation) == (), (
        "the library's own boundary checker disagrees with the assertion above"
    )


def assert_chunks_partition_the_account(
    chunks, events_by_conversation: dict[str, list[MemoryEvent]]
) -> None:
    """Every event appears in exactly one chunk, and no chunk invents an event.

    Isolation is worthless if it is bought by dropping events: a builder that flushed away the other
    conversation's messages would pass the boundary check while silently shrinking the history.
    """
    chunked = [event_id for chunk in chunks for event_id in chunk.event_ids]
    assert len(chunked) == len(set(chunked)), "an event was placed in two chunks"
    expected = [
        event.id for conversation_id in sorted(events_by_conversation)
        for event in events_by_conversation[conversation_id]
    ]
    assert sorted(chunked) == sorted(expected), "chunking lost or invented events"


# --- 1. the brief's exact scenario ------------------------------------------------------------


def test_brief_scenario_two_lookalike_conversations_stay_separate(scratch: Path) -> None:
    """10:00 A: 吃饭吗 / 10:01 B: 吃饭吗 / 10:02 A: 可以 / 10:03 B: 不去了.

    One minute apart, two of the four messages byte-identical. Anything that groups globally and
    then segments by time produces one four-message chunk; the requirement is two coherent chunks.
    """
    write_shard(
        scratch / SHARD_A,
        {
            "alice": [
                msg("吃饭吗", ts(10, 0), local_id=1, server_id="a-1"),
                msg("可以", ts(10, 2), local_id=2, server_id="a-2"),
            ],
            "bob": [
                msg("吃饭吗", ts(10, 1), local_id=9, server_id="b-9"),
                msg("不去了", ts(10, 3), local_id=10, server_id="b-10"),
            ],
        },
    )

    events_by_conversation, report = import_account(
        exports_from_directory(scratch / SHARD_A, shard=SHARD_A), shards_detected=[f"{SHARD_A}.db"]
    )
    assert sorted(events_by_conversation) == ["alice", "bob"]
    assert report.messages_kept == 4 and report.duplicates_removed == 0

    chunks = build_account_sessions(events_by_conversation, session_config())

    assert len(chunks) == 2, f"expected 2 chunks, got {[c.id for c in chunks]}"
    assert [c.n_events for c in chunks] == [2, 2], "a 4-message chunk means the boundary was crossed"
    assert_no_chunk_crosses_conversations(chunks, events_by_conversation)

    by_conversation = {chunk.conversation_id: chunk for chunk in chunks}
    assert by_conversation["alice"].text == "[2026-09-17 10:00] 我: 吃饭吗\n[2026-09-17 10:02] 我: 可以"
    assert by_conversation["bob"].text == "[2026-09-17 10:01] 我: 吃饭吗\n[2026-09-17 10:03] 我: 不去了"


def test_four_message_fusion_cannot_be_bought_by_relaxing_the_config(scratch: Path) -> None:
    """Same data, but every temporal/budget guard switched off.

    With ``max_gap`` and ``max_chars`` effectively infinite and everything on one calendar day, the
    *only* thing still separating the two conversations is the conversation boundary itself.
    """
    write_shard(
        scratch / SHARD_A,
        {
            "alice": [
                msg("吃饭吗", ts(10, 0), local_id=1, server_id="a-1"),
                msg("可以", ts(10, 2), local_id=2, server_id="a-2"),
            ],
            "bob": [
                msg("吃饭吗", ts(10, 1), local_id=9, server_id="b-9"),
                msg("不去了", ts(10, 3), local_id=10, server_id="b-10"),
            ],
        },
    )
    from datetime import timedelta

    events_by_conversation, _ = import_account(
        exports_from_directory(scratch / SHARD_A, shard=SHARD_A), shards_detected=[SHARD_A]
    )
    relaxed = SessionConfig(max_gap=timedelta(days=3650), max_chars=10_000_000)

    chunks = build_account_sessions(events_by_conversation, relaxed)

    assert len(chunks) == 2
    assert sorted(chunk.n_events for chunk in chunks) == [2, 2]
    assert_no_chunk_crosses_conversations(chunks, events_by_conversation)


def test_build_sessions_is_itself_boundary_safe_for_a_mixed_stream(scratch: Path) -> None:
    """Even if a caller hands the raw mixed stream to ``build_sessions``, no chunk may mix.

    This is the second line of defence: ``build_account_sessions`` exists so callers never have to do
    this, but a single flat stream must not be able to produce a cross-conversation chunk either.
    """
    from memory import build_sessions

    write_shard(
        scratch / SHARD_A,
        {
            "alice": [
                msg("吃饭吗", ts(10, 0), local_id=1, server_id="a-1"),
                msg("可以", ts(10, 2), local_id=2, server_id="a-2"),
            ],
            "bob": [msg("不去了", ts(10, 3), local_id=9, server_id="b-9")],
        },
    )
    events_by_conversation, _ = import_account(
        exports_from_directory(scratch / SHARD_A, shard=SHARD_A), shards_detected=[SHARD_A]
    )
    mixed = [event for events in events_by_conversation.values() for event in events]

    chunks = build_sessions(mixed, session_config(), conversation_id="account")

    assert len(chunks) == 2, "a mixed stream must flush on every conversation change"
    assert_no_chunk_crosses_conversations(chunks, events_by_conversation)


# --- 2. identical text, identical speaker name, different conversations -----------------------


def test_identical_text_and_speaker_across_conversations_does_not_merge(scratch: Path) -> None:
    """Byte-identical messages, same ``isSend`` speaker label, same ``localId``, 1s apart, 2 talks."""
    write_shard(
        scratch / SHARD_A,
        {
            "alice": [msg("在吗", ts(9, 0), local_id=1, server_id="a-1", is_send=1)],
            "bob": [msg("在吗", ts(9, 0), local_id=1, server_id="b-1", is_send=1)],
        },
    )

    events_by_conversation, report = import_account(
        exports_from_directory(scratch / SHARD_A, shard=SHARD_A), shards_detected=[SHARD_A]
    )

    assert set(events_by_conversation) == {"alice", "bob"}
    assert report.messages_kept == 2, "identical text is not a duplicate"
    assert report.duplicates_removed == 0
    alice, bob = events_by_conversation["alice"][0], events_by_conversation["bob"][0]
    assert alice.text == bob.text and alice.sender_name == bob.sender_name
    assert alice.id != bob.id, "the same localId in two conversations must not share an event id"

    chunks = build_account_sessions(events_by_conversation, session_config())
    assert len(chunks) == 2
    assert len({chunk.id for chunk in chunks}) == 2
    assert_no_chunk_crosses_conversations(chunks, events_by_conversation)


# --- 3. one display name on two talkers -------------------------------------------------------


def test_same_display_name_on_two_talkers_does_not_merge_them(scratch: Path) -> None:
    """A nickname is editable and not unique. Two talkers both shown as 老王 stay two conversations."""
    write_shard(
        scratch / SHARD_A,
        {
            "wxid_laowang_a": [msg("我到了", ts(10, 0), local_id=1, server_id="x1", display_name="老王")],
            "wxid_laowang_b": [msg("我也到了", ts(10, 0), local_id=1, server_id="x2", display_name="老王")],
        },
    )

    events_by_conversation, report = import_account(
        exports_from_directory(scratch / SHARD_A, shard=SHARD_A), shards_detected=[SHARD_A]
    )

    assert set(events_by_conversation) == {"wxid_laowang_a", "wxid_laowang_b"}
    assert report.conversations_imported == 2
    chunks = build_account_sessions(events_by_conversation, session_config())
    assert len(chunks) == 2, "a shared display name merged two conversations"
    assert_no_chunk_crosses_conversations(chunks, events_by_conversation)


# --- 4. display name change mid-history -------------------------------------------------------


def test_display_name_change_across_shards_stays_one_conversation(scratch: Path) -> None:
    """Groups and remarks are renamed all the time; the talker is what stays the same talker."""
    write_shard(
        scratch / SHARD_A,
        {"alice": [msg("早上好", ts(8, 0), local_id=1, server_id="s1", display_name="小王")]},
    )
    write_shard(
        scratch / SHARD_B,
        {"alice": [msg("晚上好", ts(8, 5), local_id=1, server_id="s2", display_name="王总")]},
    )

    grouped = group_exports_by_conversation(
        tuple(exports_from_directory(scratch / SHARD_A, shard=SHARD_A))
        + tuple(exports_from_directory(scratch / SHARD_B, shard=SHARD_B))
    )
    assert list(grouped) == ["alice"], "a renamed conversation must not become two"

    events_by_conversation, report = import_account(
        tuple(exports_from_directory(scratch / SHARD_A, shard=SHARD_A))
        + tuple(exports_from_directory(scratch / SHARD_B, shard=SHARD_B)),
        shards_detected=[f"{SHARD_A}.db", f"{SHARD_B}.db"],
    )
    assert list(events_by_conversation) == ["alice"]
    assert report.conversations_imported == 1
    assert {event.metadata["source_shard"] for event in events_by_conversation["alice"]} == {
        SHARD_A,
        SHARD_B,
    }

    chunks = build_account_sessions(events_by_conversation, session_config())
    assert len(chunks) == 1 and chunks[0].n_events == 2


# --- 5. cross-shard coherence (what per-conversation building buys) ---------------------------


def test_conversation_spread_over_three_shards_is_one_chunk(scratch: Path) -> None:
    """Minutes apart, three shards, one talker -> ONE chunk carrying all three messages.

    Per-message fragmentation is the other failure mode: it is not a boundary violation, but it
    destroys the retrieval unit, so both properties are asserted together.
    """
    write_shard(scratch / SHARD_A, {"alice": [msg("第一段", ts(11, 0), local_id=1, server_id="s1")]})
    write_shard(scratch / SHARD_B, {"alice": [msg("第二段", ts(11, 4), local_id=1, server_id="s2")]})
    write_shard(scratch / SHARD_C, {"alice": [msg("第三段", ts(11, 9), local_id=1, server_id="s3")]})

    events_by_conversation, report = import_account(
        tuple(exports_from_directory(scratch / SHARD_A, shard=SHARD_A))
        + tuple(exports_from_directory(scratch / SHARD_B, shard=SHARD_B))
        + tuple(exports_from_directory(scratch / SHARD_C, shard=SHARD_C)),
        shards_detected=[f"{SHARD_A}.db", f"{SHARD_B}.db", f"{SHARD_C}.db"],
    )
    assert report.messages_kept == 3
    assert report.partial is False
    assert [event.text for event in events_by_conversation["alice"]] == ["第一段", "第二段", "第三段"]

    chunks = build_account_sessions(events_by_conversation, session_config())

    assert len(chunks) == 1, f"one conversation was fragmented into {len(chunks)} chunks"
    assert chunks[0].n_events == 3
    assert chunks[0].conversation_id == "alice"
    assert chunks[0].start_time == datetime(2026, 9, 17, 11, 0)
    assert chunks[0].end_time == datetime(2026, 9, 17, 11, 9)
    assert [line.split("] ")[1] for line in chunks[0].text.splitlines()] == [
        "我: 第一段",
        "我: 第二段",
        "我: 第三段",
    ]
    assert_no_chunk_crosses_conversations(chunks, events_by_conversation)


# --- 6. interleaved timestamps across conversations -------------------------------------------


def test_interleaved_conversations_produce_no_mixed_chunk(scratch: Path) -> None:
    """Three conversations alternating every minute, several messages each, checked via event_ids."""
    alice = [msg(f"a{n}", ts(10, n), local_id=n, server_id=f"a-{n}") for n in (0, 2, 4, 6)]
    bob = [msg(f"b{n}", ts(10, n), local_id=n, server_id=f"b-{n}") for n in (1, 3)]
    carol = [msg(f"c{n}", ts(10, n), local_id=n, server_id=f"c-{n}") for n in (5, 7, 8)]
    write_shard(scratch / SHARD_A, {"alice": alice, "bob": bob, "carol": carol})

    events_by_conversation, _ = import_account(
        exports_from_directory(scratch / SHARD_A, shard=SHARD_A), shards_detected=[SHARD_A]
    )
    assert {event.text for event in events_by_conversation["carol"]} == {"c5", "c7", "c8"}

    chunks = build_account_sessions(events_by_conversation, session_config())

    assert_no_chunk_crosses_conversations(chunks, events_by_conversation)
    assert sorted(chunk.n_events for chunk in chunks) == [2, 3, 4]
    # A chunk's own timestamps must also stay inside its conversation's span.
    for chunk in chunks:
        owned = [event.timestamp for event in events_by_conversation[chunk.conversation_id]]
        assert chunk.start_time >= min(owned) and chunk.end_time <= max(owned)


# --- 7. dedupe on serverId, scoped to the conversation ----------------------------------------


def test_duplicate_server_id_within_a_conversation_removed_exactly_once(scratch: Path) -> None:
    """A shard migration duplicates one message; the copy is dropped once, not twice or never."""
    write_shard(
        scratch / SHARD_A,
        {"alice": [msg("同一句话", ts(12, 0), local_id=1, server_id="srv-dup")]},
    )
    write_shard(
        scratch / SHARD_B,
        {
            "alice": [
                msg("同一句话", ts(12, 0), local_id=7, server_id="srv-dup"),
                msg("另一句话", ts(12, 1), local_id=8, server_id="srv-new"),
            ]
        },
    )

    events_by_conversation, report = import_account(
        tuple(exports_from_directory(scratch / SHARD_A, shard=SHARD_A))
        + tuple(exports_from_directory(scratch / SHARD_B, shard=SHARD_B)),
        shards_detected=[f"{SHARD_A}.db", f"{SHARD_B}.db"],
    )

    assert [event.text for event in events_by_conversation["alice"]] == ["同一句话", "另一句话"]
    assert report.duplicates_removed == 1, "the identical copy must be removed exactly once"
    assert report.messages_received == 3 and report.messages_kept == 2
    assert report.undedupeable == 0


def test_same_server_id_in_two_conversations_drops_neither(scratch: Path) -> None:
    """The dedupe key must be conversation-scoped.

    A globally keyed ``serverId`` set would delete Bob's message because Alice already had that id —
    silent data loss in a conversation that never duplicated anything.
    """
    write_shard(
        scratch / SHARD_A,
        {
            "alice": [msg("alice 的消息", ts(13, 0), local_id=1, server_id="SHARED-ID")],
            "bob": [msg("bob 的消息", ts(13, 1), local_id=1, server_id="SHARED-ID")],
        },
    )

    events_by_conversation, report = import_account(
        exports_from_directory(scratch / SHARD_A, shard=SHARD_A), shards_detected=[SHARD_A]
    )

    assert set(events_by_conversation) == {"alice", "bob"}
    assert report.duplicates_removed == 0, "a cross-conversation id collision dropped a message"
    assert report.messages_kept == 2
    assert (events_by_conversation["alice"][0].text, events_by_conversation["bob"][0].text) == (
        "alice 的消息",
        "bob 的消息",
    )
    assert events_by_conversation["alice"][0].id != events_by_conversation["bob"][0].id

    chunks = build_account_sessions(events_by_conversation, session_config())
    assert len(chunks) == 2
    assert_no_chunk_crosses_conversations(chunks, events_by_conversation)


# --- 8. localId reused across conversations ---------------------------------------------------


def test_local_id_reused_across_conversations_drops_and_merges_nothing(scratch: Path) -> None:
    """``localId`` is not unique across conversations, so it must not be an identity either."""
    write_shard(
        scratch / SHARD_A,
        {
            "alice": [msg("A1", ts(14, 0), local_id=42), msg("A2", ts(14, 1), local_id=42)],
            "bob": [msg("B1", ts(14, 0), local_id=42), msg("B2", ts(14, 1), local_id=42)],
        },
    )

    events_by_conversation, report = import_account(
        exports_from_directory(scratch / SHARD_A, shard=SHARD_A), shards_detected=[SHARD_A]
    )

    assert [event.text for event in events_by_conversation["alice"]] == ["A1", "A2"]
    assert [event.text for event in events_by_conversation["bob"]] == ["B1", "B2"]
    assert report.messages_kept == 4 and report.duplicates_removed == 0
    assert report.undedupeable == 4, "no serverId anywhere, so nothing was dedupeable"
    all_ids = [event.id for events in events_by_conversation.values() for event in events]
    assert len(set(all_ids)) == 4, "event ids collided across conversations"

    chunks = build_account_sessions(events_by_conversation, session_config())
    assert len(chunks) == 2
    assert sorted(chunk.n_events for chunk in chunks) == [2, 2]
    assert_no_chunk_crosses_conversations(chunks, events_by_conversation)


# --- 9. no serverId at all --------------------------------------------------------------------


def test_messages_without_server_id_are_kept_and_counted(scratch: Path) -> None:
    """Missing ``serverId`` means "cannot dedupe", never "drop it"."""
    write_shard(
        scratch / SHARD_A,
        {"alice": [msg("没有 serverId 1", ts(15, 0), local_id=1), msg("没有 serverId 2", ts(15, 1), local_id=1)]},
    )
    write_shard(
        scratch / SHARD_B,
        {"alice": [msg("没有 serverId 1", ts(15, 0), local_id=1)]},  # an exact textual duplicate
    )

    events_by_conversation, report = import_account(
        tuple(exports_from_directory(scratch / SHARD_A, shard=SHARD_A))
        + tuple(exports_from_directory(scratch / SHARD_B, shard=SHARD_B)),
        shards_detected=[f"{SHARD_A}.db", f"{SHARD_B}.db"],
    )

    assert report.messages_kept == 3, "a message was dropped on a weak identity"
    assert report.duplicates_removed == 0
    assert report.undedupeable == 3
    assert len({event.id for event in events_by_conversation["alice"]}) == 3
    assert any("no serverId" in line for line in report.lines())


# --- 10. PARTIAL: a detected shard that was never exported ------------------------------------


def test_a_detected_but_unexported_shard_makes_the_report_partial(scratch: Path) -> None:
    write_shard(scratch / SHARD_A, {"alice": [msg("只看得到这一条", ts(16, 0), server_id="s1")]})

    _, report = import_account(
        exports_from_directory(scratch / SHARD_A, shard=SHARD_A),
        shards_detected=[f"{SHARD_A}.db", f"{SHARD_B}.db"],
    )

    assert report.partial is True
    assert report.missing_shards == (SHARD_B,)
    assert report.shards_detected == (SHARD_A, SHARD_B)
    assert report.shards_exported == (SHARD_A,)
    assert "PARTIAL" in "\n".join(report.lines())
    assert report.as_dict()["missing_shards"] == [SHARD_B]


def test_a_fully_exported_account_is_not_partial(scratch: Path) -> None:
    """Sidecar file names must not fake a gap, or every complete history would cry PARTIAL."""
    write_shard(scratch / SHARD_A, {"alice": [msg("完整的一天", ts(16, 0), server_id="s1")]})
    write_shard(scratch / SHARD_B, {"bob": [msg("也是完整的", ts(16, 1), server_id="s2")]})

    exports = tuple(exports_from_directory(scratch / SHARD_A, shard=SHARD_A)) + tuple(
        exports_from_directory(scratch / SHARD_B, shard=SHARD_B)
    )
    _, report = import_account(
        exports,
        shards_detected=[f"{SHARD_A}.db", f"{SHARD_B}.db", f"{SHARD_A}.db-wal", f"{SHARD_A}.db-shm"],
    )

    assert report.partial is False
    assert report.missing_shards == ()
    assert report.shards_detected == (SHARD_A, SHARD_B)


def test_shard_stem_normalises_every_reference_to_a_shard(scratch: Path) -> None:
    assert shard_stem(f"{SHARD_A}.db") == shard_stem(f"{SHARD_A}.db-wal") == shard_stem(SHARD_A)
    assert shard_stem(f"{SHARD_A}.db-shm") == SHARD_A
    assert shard_stem(r"D:\WeChat\Msg\Multi\MSG3.db") == "MSG3"
    assert shard_stem("/mnt/c/Msg/MSG4.db-wal") == "MSG4"
    assert shard_stem("MSG5") == "MSG5"


# --- 11. empty and malformed inputs -----------------------------------------------------------


def test_no_exports_at_all_reports_honestly(scratch: Path) -> None:
    empty_dir = scratch / "nothing_here"
    empty_dir.mkdir(parents=True)

    events_by_conversation, report = import_account((), shards_detected=())

    assert events_by_conversation == {}
    assert report.messages_kept == 0 and report.messages_received == 0
    assert report.first_timestamp is None and report.last_timestamp is None
    assert report.partial is False, "no shard detected means nothing is known to be missing"
    assert build_account_sessions(events_by_conversation, session_config()) == []


def test_an_empty_shard_export_yields_no_events_and_no_chunk(scratch: Path) -> None:
    """An export that parses to nothing must not count as an imported, searchable conversation.

    The current contract drops the empty conversation from ``events_by_conversation`` and reports it as
    producing no messages, so "N imported of M discovered" can never overstate what is searchable.
    Either way it must not raise, and it must not invent a chunk.
    """
    write_shard(scratch / SHARD_A, {"alice": []})

    events_by_conversation, report = import_account(
        exports_from_directory(scratch / SHARD_A, shard=SHARD_A), shards_detected=[f"{SHARD_A}.db"]
    )

    assert events_by_conversation == {}, "an empty export became a searchable conversation"
    assert report.messages_received == 0 and report.messages_kept == 0
    assert report.conversations_discovered == 1 and report.conversations_imported == 0
    assert report.conversations_without_messages == ("alice",)
    assert report.per_conversation == ()
    assert report.first_timestamp is None
    assert build_account_sessions(events_by_conversation, session_config()) == []


def test_non_list_payload_and_createTime_less_entries_do_not_raise(scratch: Path) -> None:
    """An exporter that emits a scalar, and messages with no usable time: skipped, never invented."""
    (scratch / "weird").mkdir(parents=True)
    (scratch / "weird" / "alice_messages.json").write_text(
        json.dumps(
            {
                "schema": "weflow-message/v1",
                # `messages` is not a list -> unwrap yields nothing at all.
                "messages": {"oops": True},
            }
        ),
        encoding="utf-8",
    )
    (scratch / "weird" / "bob_messages.json").write_text(
        json.dumps(
            [
                msg("有时间", ts(17, 0), server_id="s1"),
                {"localId": 2, "serverId": "s2", "parsedContent": "没有 createTime"},
                {"localId": 3, "serverId": "s3", "createTime": "not-a-time", "parsedContent": "坏时间"},
                "这不是一个对象",
            ]
        ),
        encoding="utf-8",
    )

    events_by_conversation, report = import_account(
        exports_from_directory(scratch / "weird", shard=SHARD_A), shards_detected=[f"{SHARD_A}.db"]
    )

    assert [event.text for event in events_by_conversation["bob"]] == ["有时间"]
    # `alice`'s file holds an envelope whose `messages` is not a list: nothing parses, so nothing is
    # searchable — and that is said out loud instead of producing an empty, importable conversation.
    assert "alice" not in events_by_conversation
    assert "alice" in report.conversations_without_messages
    assert report.conversations_discovered == 2 and report.conversations_imported == 1
    assert report.skipped_messages == 3, f"expected 3 skipped, got {report.skipped_messages}"
    assert report.messages_received == 1 and report.messages_kept == 1
    assert report.first_timestamp == datetime(2026, 9, 17, 17, 0)


# --- 12. determinism --------------------------------------------------------------------------


def test_two_runs_produce_identical_chunk_ids_and_content(scratch: Path) -> None:
    write_shard(
        scratch / SHARD_A,
        {
            "alice": [msg("甲", ts(18, 0), server_id="s1"), msg("乙", ts(18, 1), server_id="s2")],
            "group_9@chatroom": [msg("群里的话", ts(18, 0), server_id="g1", is_send=0)],
        },
    )
    write_shard(scratch / SHARD_B, {"alice": [msg("丙", ts(18, 2), server_id="s3")]})

    def run() -> tuple[tuple[str, ...], tuple[str, ...]]:
        exports = tuple(exports_from_directory(scratch / SHARD_A, shard=SHARD_A)) + tuple(
            exports_from_directory(scratch / SHARD_B, shard=SHARD_B)
        )
        events_by_conversation, _ = import_account(
            exports, shards_detected=[f"{SHARD_A}.db", f"{SHARD_B}.db"]
        )
        chunks = build_account_sessions(events_by_conversation, session_config())
        return (
            tuple(chunk.id for chunk in chunks),
            tuple(chunk.text for chunk in chunks),
        )

    first, second = run(), run()

    assert first == second, "the same exports produced different chunks"
    assert first[0] == ("alice-session-0001", "group_9@chatroom-session-0001")
    assert len(set(first[0])) == len(first[0]), "chunk ids are not unique"


# --- 13. group detection ----------------------------------------------------------------------


def test_group_talker_is_detected_and_its_id_recovered_from_the_file_name(scratch: Path) -> None:
    write_shard(
        scratch / "groupdir",
        {"group_1@chatroom": [msg("群里第一条", ts(19, 0), is_send=0), msg("群里第二条", ts(19, 1))]},
    )

    assert conversation_type_of("group_1@chatroom") == "group"
    assert conversation_type_of("alice") == "direct"
    assert conversation_id_from_export_filename("group_1@chatroom_messages.json") == "group_1@chatroom"

    exports = exports_from_directory(scratch / "groupdir", shard=SHARD_A)
    assert [export.conversation_id for export in exports] == ["group_1@chatroom"]

    events_by_conversation, report = import_account(exports, shards_detected=[f"{SHARD_A}.db"])
    assert list(events_by_conversation) == ["group_1@chatroom"]
    assert report.conversations_imported == 1

    chunks = build_account_sessions(events_by_conversation, session_config())
    assert len(chunks) == 1 and chunks[0].n_events == 2
    assert chunks[0].conversation_id == "group_1@chatroom"
    # ``participants`` is a sorted set of the literal speaker labels, so "对方" sorts before "我".
    assert chunks[0].participants == ("对方", "我")
    assert_no_chunk_crosses_conversations(chunks, events_by_conversation)


def test_group_and_direct_conversations_never_share_a_chunk(scratch: Path) -> None:
    """A group and a direct chat are different conversations even when a member is in both."""
    write_shard(
        scratch / "mix",
        {
            "group_1@chatroom": [msg("群里说", ts(20, 0), server_id="g1", is_send=0)],
            "alice": [msg("私聊说", ts(20, 1), server_id="a1")],
        },
    )

    events_by_conversation, _ = import_account(
        exports_from_directory(scratch / "mix", shard=SHARD_A), shards_detected=[SHARD_A]
    )
    chunks = build_account_sessions(events_by_conversation, session_config())

    assert len(chunks) == 2
    assert_no_chunk_crosses_conversations(chunks, events_by_conversation)


# --- 14. discovered but never exported --------------------------------------------------------


def test_a_discovered_conversation_without_messages_invents_nothing(scratch: Path) -> None:
    write_shard(scratch / SHARD_A, {"alice": [msg("只有 alice", ts(21, 0), server_id="s1")]})

    events_by_conversation, report = import_account(
        exports_from_directory(scratch / SHARD_A, shard=SHARD_A),
        shards_detected=[f"{SHARD_A}.db"],
        discovered=[
            ConversationDescriptor(conversation_id="alice", display_name="小王"),
            ConversationDescriptor(conversation_id="carol", display_name="Carol"),
        ],
    )

    assert report.conversations_discovered == 2
    assert report.conversations_imported == 1
    assert report.conversations_without_messages == ("carol",)
    assert "carol" not in events_by_conversation

    chunks = build_account_sessions(events_by_conversation, session_config())
    assert len(chunks) == 1 and chunks[0].conversation_id == "alice"
    assert_no_chunk_crosses_conversations(chunks, events_by_conversation)


# --- extra adversarial probes -----------------------------------------------------------------


def test_conversation_ids_that_are_prefixes_of_each_other_do_not_dedupe(scratch: Path) -> None:
    """``alice`` vs ``alice-wife``: a prefix-keyed dedupe or grouping would cross them."""
    write_shard(
        scratch / SHARD_A,
        {
            "alice": [msg("给 alice", ts(22, 0), server_id="same")],
            "alice-wife": [msg("给 alice-wife", ts(22, 1), server_id="same")],
            "alicex": [msg("给 alicex", ts(22, 2), server_id="same")],
        },
    )

    events_by_conversation, report = import_account(
        exports_from_directory(scratch / SHARD_A, shard=SHARD_A), shards_detected=[SHARD_A]
    )

    assert sorted(events_by_conversation) == ["alice", "alice-wife", "alicex"]
    assert report.duplicates_removed == 0
    assert report.messages_kept == 3
    chunks = build_account_sessions(events_by_conversation, session_config())
    assert len(chunks) == 3
    assert_no_chunk_crosses_conversations(chunks, events_by_conversation)


def test_one_shard_directory_holding_every_conversation_still_isolates(scratch: Path) -> None:
    """The orchestrator writes one export per conversation into one shard directory.

    A naive "one file = one corpus, merge the files" reading would fuse them; grouping by
    conversation first is what keeps them apart.
    """
    write_shard(
        scratch / SHARD_A,
        {
            "alice": [msg("A-1", ts(23, 0), server_id="a1"), msg("A-2", ts(23, 1), server_id="a2")],
            "bob": [msg("B-1", ts(23, 0), server_id="b1"), msg("B-2", ts(23, 1), server_id="b2")],
            "carol": [msg("C-1", ts(23, 0), server_id="c1")],
        },
    )

    events_by_conversation, report = import_account(
        exports_from_directory(scratch / SHARD_A, shard=SHARD_A), shards_detected=[f"{SHARD_A}.db"]
    )
    chunks = build_account_sessions(events_by_conversation, session_config())

    assert report.messages_kept == 5
    assert len(chunks) == 3
    assert sorted(chunk.n_events for chunk in chunks) == [1, 2, 2]
    assert_no_chunk_crosses_conversations(chunks, events_by_conversation)
    # Chunks are handed to the index ordered by conversation, then time.
    assert [chunk.conversation_id for chunk in chunks] == ["alice", "bob", "carol"]


def test_chunks_of_different_conversations_never_share_an_id(scratch: Path) -> None:
    """Session indexes restart per conversation — the ids must still be globally unique."""
    write_shard(
        scratch / SHARD_A,
        {
            "alice": [msg("A-1", ts(9, 0), server_id="a1")],
            "bob": [msg("B-1", ts(9, 0), server_id="b1")],
        },
    )
    events_by_conversation, _ = import_account(
        exports_from_directory(scratch / SHARD_A, shard=SHARD_A), shards_detected=[SHARD_A]
    )

    chunks = build_account_sessions(events_by_conversation, session_config())

    assert [chunk.id for chunk in chunks] == ["alice-session-0001", "bob-session-0001"]
    assert len({chunk.id for chunk in chunks}) == 2


def test_day_boundary_inside_one_conversation_does_not_leak_into_another(scratch: Path) -> None:
    """Two conversations, one straddling midnight: the day break must stay inside its own talker."""
    write_shard(
        scratch / SHARD_A,
        {
            "alice": [
                msg("睡前", ts(23, 55, day=17), server_id="a1"),
                msg("第二天", ts(0, 5, day=18), server_id="a2"),
            ],
            "bob": [msg("半夜插一句", ts(23, 58, day=17), server_id="b1")],
        },
    )

    events_by_conversation, _ = import_account(
        exports_from_directory(scratch / SHARD_A, shard=SHARD_A), shards_detected=[SHARD_A]
    )
    chunks = build_account_sessions(events_by_conversation, session_config())

    assert len(chunks) == 3, "alice must break at midnight and bob must stay alone"
    assert sorted(chunk.n_events for chunk in chunks) == [1, 1, 1]
    assert_no_chunk_crosses_conversations(chunks, events_by_conversation)


def test_merging_shards_globally_then_segmenting_would_cross_the_boundary(scratch: Path) -> None:
    """A tripwire on the documented anti-pattern, so a future refactor cannot quietly reintroduce it.

    ``shards.load_and_merge`` is the *pre-Phase-19* entry point: it is handed every conversation's
    export as if they were shards of one corpus and given a single conversation id. This test records
    what that path does — it fuses the account — which is exactly why ``import_account`` groups by
    conversation first. The account path must never route through it.
    """
    from memory.shards import load_and_merge

    write_shard(
        scratch / SHARD_A,
        {
            "alice": [
                msg("吃饭吗", ts(10, 0), local_id=1, server_id="a-1"),
                msg("可以", ts(10, 2), local_id=2, server_id="a-2"),
            ],
            "bob": [
                msg("吃饭吗", ts(10, 1), local_id=9, server_id="b-9"),
                msg("不去了", ts(10, 3), local_id=10, server_id="b-10"),
            ],
        },
    )
    paths = sorted((scratch / SHARD_A).glob("*_messages.json"))
    flattened, _ = load_and_merge(paths, conversation_id="account")

    fused = build_account_sessions({"account": flattened}, session_config())

    assert len(fused) == 1 and fused[0].n_events == 4, (
        "the global-merge path is expected to fuse the account; if this ever stops being true the "
        "module docstring's warning is stale, but the account path must not depend on it"
    )
    assert fused[0].conversation_id == "account", "the flattened corpus lost both talkers' identities"

    # The account path, on the same bytes, must produce the opposite result.
    per_conversation, _ = import_account(
        exports_from_directory(scratch / SHARD_A, shard=SHARD_A), shards_detected=[SHARD_A]
    )
    isolated = build_account_sessions(per_conversation, session_config())
    assert len(isolated) == 2, "grouping by conversation first did not isolate the two talkers"
    assert {chunk.conversation_id for chunk in isolated} == {"alice", "bob"}
    assert_no_chunk_crosses_conversations(isolated, per_conversation)
    assert_chunks_partition_the_account(isolated, per_conversation)


def test_maximally_interleaved_repeated_ids_lose_nothing(scratch: Path) -> None:
    """Three shards, three talkers, same ``localId`` everywhere, one re-used ``serverId``, same second.

    The account must come out isolated *and* complete: the union of the chunked event ids equals the
    union of the imported event ids, so isolation was not bought by dropping a conversation.
    """
    for shard, offset in ((SHARD_A, 0), (SHARD_B, 1), (SHARD_C, 2)):
        write_shard(
            scratch / shard,
            {
                # "reused" is the *same* serverId in all three conversations and all three shards.
                "alice": [
                    msg("A", ts(11, offset), local_id=1, server_id="reused"),
                    msg("A2", ts(11, 30 + offset), local_id=1, server_id=f"a-{offset}"),
                ],
                "bob": [
                    msg("B", ts(11, offset), local_id=1, server_id="reused"),
                    msg("B2", ts(11, 30 + offset), local_id=1),
                ],
                "carol": [
                    msg("C", ts(11, offset), local_id=1, server_id="reused"),
                    msg("C2", ts(11, 30 + offset), local_id=1, server_id="reused"),
                ],
            },
        )

    exports = []
    for shard in (SHARD_A, SHARD_B, SHARD_C):
        exports.extend(exports_from_directory(scratch / shard, shard=shard))
    events_by_conversation, report = import_account(
        exports, shards_detected=[f"{s}.db" for s in (SHARD_A, SHARD_B, SHARD_C)]
    )

    assert sorted(events_by_conversation) == ["alice", "bob", "carol"]
    # Dedupe is conversation-scoped, and "reused" is duplicated *within* a shard as well as across
    # them: alice keeps 1 of 3, bob 1 of 3, carol 1 of 6 -> 9 removed and 9 kept. A globally keyed
    # dedupe would collapse all three conversations' "reused" into one event and shorten every talker.
    assert report.duplicates_removed == 9
    assert report.messages_kept == 9
    assert report.undedupeable == 3  # bob's one serverId-less message per shard, all kept
    assert len({event.id for events in events_by_conversation.values() for event in events}) == 9

    chunks = build_account_sessions(events_by_conversation, session_config())

    assert_no_chunk_crosses_conversations(chunks, events_by_conversation)
    assert_chunks_partition_the_account(chunks, events_by_conversation)
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_the_librarys_own_boundary_checker_actually_catches_a_violation(scratch: Path) -> None:
    """A checker that never fires is not evidence. This one is fed a deliberately dirty chunk.

    ``crossed_conversation_chunks`` is what ``recall.import_account_directory`` raises on for real
    accounts, so it has to be shown to detect a crossing rather than merely to return empty.
    """
    from dataclasses import replace

    write_shard(
        scratch / SHARD_A,
        {
            "alice": [msg("A-1", ts(10, 0), server_id="a1")],
            "bob": [msg("B-1", ts(10, 0), server_id="b1")],
        },
    )
    events_by_conversation, _ = import_account(
        exports_from_directory(scratch / SHARD_A, shard=SHARD_A), shards_detected=[SHARD_A]
    )
    clean = build_account_sessions(events_by_conversation, session_config())
    assert crossed_conversation_chunks(clean, events_by_conversation) == ()

    alice_event, bob_event = events_by_conversation["alice"][0], events_by_conversation["bob"][0]
    dirty = replace(
        clean[0],
        event_ids=(alice_event.id, bob_event.id),
        text=f"{alice_event.line}\n{bob_event.line}",
    )
    assert crossed_conversation_chunks([dirty], events_by_conversation) == (dirty.id,)
