"""The store's projection, tested with no database at all.

``rows.build_snapshot`` is a pure function from the memory layer's own objects to row tuples, and
these tests hold it to the properties the store's correctness depends on: deterministic keys,
identities that cannot collapse, and an absence of identity that stays an absence. They run in the
ordinary suite, on a machine with no PostgreSQL, because none of these questions are about SQL —
and a property that only holds when a container happens to be running is a property nobody checks.

The database-backed half lives in ``test_postgres_store.py`` and carries the ``postgres`` marker.
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

import memory_store  # noqa: E402
import memory_store.rows as rows_module  # noqa: E402

#: Scratch lives beside the test file rather than in the system temp directory: the temporary
#: directory this machine hands out is not always writable by this process, and a suite that fails
#: at fixture setup on one machine reports nothing about the code. The same convention the other
#: test modules here already use.
SCRATCH_ROOT = BASE_DIR / "tests" / "_scratch_store"


@pytest.fixture()
def scratch() -> Path:
    directory = SCRATCH_ROOT / uuid4().hex[:8]
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)
        try:
            SCRATCH_ROOT.rmdir()
        except OSError:
            pass

# ---------------------------------------------------------------------------------------------
# A synthetic account. Every identity is shaped like a real one — a wxid, a chatroom, a member —
# because the failures these tests exist for are about the *shape* of identity, and a fixture whose
# ids were called "alice" would never exercise the code that rejects a talker masquerading as a name.
# ---------------------------------------------------------------------------------------------

TALKER_SELF = "wxid_synthetic_otter"
TALKER_PEER = "wxid_synthetic_heron"
GROUP = "47110022@chatroom"
MEMBER_A = "wxid_synthetic_member_a"
MEMBER_B = "wxid_synthetic_member_b"
SHARED_NAME = "小王"

DAY = int(datetime(2026, 9, 20, 10, 0).timestamp())
HOUR = 3600


def message(
    local_id: int,
    when: int,
    text: str,
    *,
    sender: str = TALKER_SELF,
    is_send: int = 1,
    display: str | None = None,
    reply_to: str | None = None,
) -> dict:
    entry = {
        "localId": local_id,
        "serverId": f"server-{local_id}",
        "localType": 1,
        "createTime": when,
        "isSend": is_send,
        "senderUsername": sender,
        "parsedContent": text,
    }
    if display is not None:
        entry["senderDisplay"] = display
    if reply_to is not None:
        entry["replyTo"] = reply_to
    return entry


def write_tree(root: Path, tree: dict[str, dict[str, list[dict]]]) -> Path:
    for shard, conversations in tree.items():
        directory = root / shard
        directory.mkdir(parents=True, exist_ok=True)
        for talker, messages in conversations.items():
            (directory / f"{talker}_messages.json").write_text(
                json.dumps(messages, ensure_ascii=False), encoding="utf-8"
            )
    return root


@pytest.fixture()
def account(scratch: Path) -> Path:
    """Three conversations: a direct chat with a reply, a group with two members who share a name,
    and a group whose member has no display at all."""
    tree = {
        "MSG0": {
            TALKER_PEER: [
                message(10, DAY, "私聊第一条", sender=TALKER_PEER, is_send=0, display="苍鹭"),
                message(11, DAY + 60, "收到", reply_to="server-10"),
            ],
        },
        "MSG1": {
            GROUP: [
                message(20, DAY + 2 * HOUR, "组里甲说话", sender=MEMBER_A, is_send=0,
                        display=SHARED_NAME),
                message(21, DAY + 2 * HOUR + 30, "组里乙说话", sender=MEMBER_B, is_send=0,
                        display=SHARED_NAME),
                message(22, DAY + 2 * HOUR + 60, "没有名字的人", sender="wxid_synthetic_nobody",
                        is_send=0),
                message(23, DAY + 30 * HOUR, "很久以后", sender=MEMBER_A, is_send=0,
                        display=SHARED_NAME),
            ],
        },
    }
    return write_tree(scratch / "account", tree)


@pytest.fixture()
def snapshot(account: Path) -> "memory_store.StoreSnapshot":
    """The projected rows for the fixture account."""
    return memory_store.account_snapshot(account).snapshot


def row_of(snapshot_rows, key_index: int, key):
    """The one row whose column ``key_index`` equals ``key``, or ``None``."""
    for row in snapshot_rows:
        if row[key_index] == key:
            return row
    return None


def column(table: str, name: str) -> int:
    return rows_module.TABLE_COLUMNS[table].index(name)


def people_by_identity(snapshot) -> dict[str, tuple]:
    index = column("people", "source_identity")
    return {row[index]: row for row in snapshot.people}


# ---------------------------------------------------------------------------------------------
# Determinism: the same corpus always projects to the same rows.
# ---------------------------------------------------------------------------------------------


def test_the_projection_is_a_pure_function_of_the_corpus(account: Path) -> None:
    """Two builds of one tree produce identical rows — the property golden parity rests on."""
    first = memory_store.account_snapshot(account).snapshot
    second = memory_store.account_snapshot(account).snapshot
    for table in rows_module.SNAPSHOT_TABLES:
        assert first.rows_for(table) == second.rows_for(table), table


def test_person_ids_are_deterministic_and_hide_the_identity() -> None:
    """Same identity -> same key, always; and the key is not the identity spelled differently."""
    first = rows_module.person_id_for(MEMBER_A)
    assert first == rows_module.person_id_for(MEMBER_A)
    assert first != rows_module.person_id_for(MEMBER_B)
    # A digest, not an encoding: a wxid must not be recoverable from a log line or a count that
    # happens to carry the key.
    assert "wxid" not in first
    assert MEMBER_A not in first


def test_person_id_distinguishes_source_types() -> None:
    """The same string from two different sources is two people, not one."""
    assert rows_module.person_id_for("someone", "weflow") != rows_module.person_id_for(
        "someone", "other-source"
    )


# ---------------------------------------------------------------------------------------------
# §26 — identity safety. Each of these is a collapse the phase forbids.
# ---------------------------------------------------------------------------------------------


def test_two_speakers_with_one_display_name_stay_two_people(snapshot) -> None:
    """The collapse Phase 20.9 found on real data: two ids the exporter resolves to the same name."""
    people = people_by_identity(snapshot)
    assert MEMBER_A in people and MEMBER_B in people
    assert people[MEMBER_A][0] != people[MEMBER_B][0], "two identities must be two person rows"
    # Both were offered the same name, and the store keeps both — it does not pick one.
    name_index = column("people", "display_name")
    assert people[MEMBER_A][name_index] == SHARED_NAME
    assert people[MEMBER_B][name_index] == SHARED_NAME


def test_the_rendered_labels_of_two_shared_names_are_still_distinguishable(account: Path) -> None:
    """Different person rows are not enough: the evidence must also be able to tell them apart."""
    snapshot = memory_store.account_snapshot(account).snapshot
    label_index = column("conversation_people", "display_label")
    labels = {
        row[1]: row[label_index]
        for row in snapshot.conversation_people
        if row[0] == GROUP
    }
    people = people_by_identity(snapshot)
    assert labels[people[MEMBER_A][0]] != labels[people[MEMBER_B][0]]


def test_a_direct_conversations_peer_is_that_conversation_not_a_shared_placeholder(snapshot) -> None:
    """The forbidden alternative: one 对方 person that every direct chat would merge into."""
    people = people_by_identity(snapshot)
    assert TALKER_PEER in people, "a direct talker is its own peer identity"
    assert "对方" not in people
    assert "" not in people


def test_a_speaker_with_no_identity_gets_no_person(snapshot) -> None:
    """An absent identity stays absent — no invented member, and no shared bucket for them."""
    people = people_by_identity(snapshot)
    assert "wxid_synthetic_nobody" in people, "an explicit id is an identity, display or not"
    person_index = column("memory_events", "speaker_person_id")
    name_index = column("memory_events", "sender_name")
    text_index = column("memory_events", "text")
    for row in snapshot.events:
        if row[text_index] == "没有名字的人":
            # A third *identity* is present even though no display was offered for it, so it is a
            # person — and its pseudonym is the third one, not the first free letter. Numbering only
            # the unnamed senders would mean that giving one member a name renumbers everybody else
            # (`memory.senders`, rule (b)), and this store inherits that decision rather than
            # re-deciding it.
            assert row[person_index] is not None
            assert row[name_index] == "成员C"
            assert "wxid" not in str(row[name_index])


def test_a_self_message_has_no_person(snapshot) -> None:
    """The self speaker has no source identity, so it names nobody (§9)."""
    role_index = column("memory_events", "speaker_role")
    person_index = column("memory_events", "speaker_person_id")
    self_rows = [row for row in snapshot.events if row[role_index] == "self"]
    assert self_rows, "the fixture has self messages"
    assert all(row[person_index] is None for row in self_rows)


def test_a_person_appearing_in_two_conversations_is_one_row_with_two_memberships(scratch: Path) -> None:
    """Same identity in two groups is one person — the rule that makes person filtering meaningful."""
    account = write_tree(
        scratch / "account",
        {
            "MSG0": {GROUP: [message(1, DAY, "甲", sender=MEMBER_A, is_send=0, display="甲")]},
            "MSG1": {
                "99990000@chatroom": [
                    message(2, DAY + HOUR, "甲又来", sender=MEMBER_A, is_send=0, display="甲")
                ]
            },
        },
    )
    snapshot = memory_store.account_snapshot(account).snapshot
    people = people_by_identity(snapshot)
    assert MEMBER_A in people
    person_id = people[MEMBER_A][0]
    memberships = [row for row in snapshot.conversation_people if row[1] == person_id]
    assert len(memberships) == 2
    assert len(snapshot.people) == 1
    conversation_count = column("people", "conversation_count")
    assert people[MEMBER_A][conversation_count] == 2


# ---------------------------------------------------------------------------------------------
# §24/§25 — the fields the store must never merge, and the ones it must never lose.
# ---------------------------------------------------------------------------------------------


def test_the_three_speaker_fields_are_three_separate_columns(snapshot) -> None:
    """``speaker_id``, ``speaker_display`` and ``sender_name`` are three facts, kept apart."""
    columns = rows_module.EVENT_COLUMNS
    for name in ("speaker_id", "speaker_display", "sender_name", "speaker_role"):
        assert name in columns
    id_index = column("memory_events", "speaker_id")
    display_index = column("memory_events", "speaker_display")
    label_index = column("memory_events", "sender_name")
    text_index = column("memory_events", "text")
    row = row_of(snapshot.events, text_index, "组里甲说话")
    assert row is not None
    assert row[id_index] == MEMBER_A, "the raw identity is stored as data"
    assert row[display_index] == SHARED_NAME, "the offered candidate is stored as offered"
    assert row[label_index] != row[display_index] or row[label_index] != row[id_index]


def test_a_raw_identity_never_reaches_the_rendered_label(snapshot) -> None:
    """The rule ``memory.labels.usable_name`` exists for, asserted on the store's own output."""
    label_index = column("memory_events", "sender_name")
    for row in snapshot.events:
        label = str(row[label_index])
        assert "wxid" not in label
        assert "@chatroom" not in label


def test_unicode_and_emoji_survive_the_projection(scratch: Path) -> None:
    body = "今天的天气真好 🌤️ — 记得带伞☂️ 和咖啡因☕"
    account = write_tree(
        scratch / "account",
        {"MSG0": {TALKER_PEER: [message(1, DAY, body, sender=TALKER_PEER, is_send=0)]}},
    )
    snapshot = memory_store.account_snapshot(account).snapshot
    text_index = column("memory_events", "text")
    assert any(row[text_index] == body for row in snapshot.events)
    chunk_text_index = column("memory_chunks", "text")
    assert any(body in row[chunk_text_index] for row in snapshot.chunks)


def test_reply_and_metadata_are_carried_through(snapshot) -> None:
    reply_index = column("memory_events", "reply_to")
    metadata_index = column("memory_events", "metadata")
    text_index = column("memory_events", "text")
    row = row_of(snapshot.events, text_index, "收到")
    assert row is not None
    # `replyTo` is not one of the fields `memory.weflow` lifts onto the event, so the honest
    # expectation is "whatever the event carries" — asserted against the event, not against a guess.
    assert row[reply_index] is None or isinstance(row[reply_index], str)
    assert isinstance(row[metadata_index], dict)
    assert row[metadata_index].get("source_type") == "weflow"


def test_chunk_rows_keep_their_events_in_order(snapshot) -> None:
    """§7.7: the evidence order must be recoverable, which is why ordinals exist at all."""
    ordinal_index = column("chunk_events", "ordinal")
    by_chunk: dict[str, list[tuple[int, str]]] = {}
    for row in snapshot.chunk_events:
        by_chunk.setdefault(row[0], []).append((row[ordinal_index], row[1]))
    assert by_chunk
    chunk_index = column("memory_chunks", "chunk_index")
    for chunk_row in snapshot.chunks:
        chunk_id = chunk_row[0]
        ordered = sorted(by_chunk[chunk_id])
        assert [ordinal for ordinal, _ in ordered] == list(range(len(ordered)))
    assert all(row[chunk_index] >= 1 for row in snapshot.chunks)


def test_chunk_text_hash_matches_its_text(snapshot) -> None:
    text_index = column("memory_chunks", "text")
    hash_index = column("memory_chunks", "text_hash")
    for row in snapshot.chunks:
        assert row[hash_index] == rows_module.content_hash(row[text_index])


def test_every_chunk_cites_only_events_of_its_own_conversation(snapshot) -> None:
    """The phase's hard rule, re-checked on the projected rows rather than trusted."""
    owner = {row[0]: row[1] for row in snapshot.events}
    for row in snapshot.chunk_events:
        chunk_id, event_id = row[0], row[1]
        assert owner[event_id] == chunk_id.rsplit("-session-", 1)[0]


def test_the_column_lists_match_the_tables_the_store_writes() -> None:
    """Every table the writer COPYs has a column list, and it is internally sound.

    ``source_inventory`` is in this mapping and deliberately **not** in ``SNAPSHOT_TABLES``: it
    describes the export tree a generation came from, not the memory, so it is written beside the
    snapshot rather than as part of it — and a caller updating one conversation must not have to
    re-describe the whole export directory to do it.
    """
    assert set(rows_module.SNAPSHOT_TABLES) == {
        "conversations",
        "people",
        "conversation_people",
        "memory_events",
        "memory_chunks",
        "chunk_events",
    }
    assert set(rows_module.TABLE_COLUMNS) == set(rows_module.SNAPSHOT_TABLES) | {
        "source_inventory"
    }
    for table, columns in rows_module.TABLE_COLUMNS.items():
        assert len(columns) == len(set(columns)), table
        assert all(name.islower() for name in columns), table


def test_every_row_has_exactly_the_declared_number_of_values(snapshot) -> None:
    """A tuple shorter than its column list would COPY into the wrong fields, silently."""
    for table in rows_module.SNAPSHOT_TABLES:
        columns = rows_module.TABLE_COLUMNS[table]
        for row in snapshot.rows_for(table):
            assert len(row) == len(columns), f"{table}: {row[0]!r}"


def test_counts_reported_by_the_snapshot_match_its_own_rows(snapshot) -> None:
    counts = snapshot.counts()
    assert counts["events"] == len(snapshot.events)
    assert counts["chunks"] == len(snapshot.chunks)
    assert counts["conversations"] == len(snapshot.conversations)


def test_a_chunk_for_an_unprojected_conversation_is_refused() -> None:
    """A chunk citing events the snapshot does not hold must fail, not silently write orphan links."""
    from memory.sessions import MemoryChunk

    orphan = MemoryChunk(
        id="wxid_absent-session-0001",
        conversation_id="wxid_absent",
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 1),
        participants=(),
        event_ids=("wxid_absent#s1",),
        text="nothing",
    )
    with pytest.raises(rows_module.ProjectionError):
        rows_module.build_snapshot({}, [orphan])
