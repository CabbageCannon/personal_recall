"""Offline tests for group sender identity (no model, no network, no real chat data).

The defect under test, measured on a real account: every non-self speaker rendered as the literal
``对方``, so a group where A, B and C all speak produced ``对方, 对方, 对方``. The evidence could not
tell the members apart and the model could not attribute anything — asked "who did X", the answers
reported that the record only labels everyone ``对方``.

Four properties are load-bearing, and each is the reason for a group of tests below:

* **role, identity and label are three different things** — ``isSend`` still renders ``我``, the raw
  ``senderUsername`` survives as :attr:`MemoryEvent.speaker_id`, and neither of them is what a group
  member is labelled with;
* **the label is per conversation, assigned over the whole stream** — a conversation spanning shards
  must be labelled once, because per-shard labelling gives each shard's own first speaker ``成员A``
  and two people end up with one name;
* **the label is deterministic and never derived from the id** — never a hash, never a random number,
  never the wxid, and the same input always yields the same labels;
* **nothing internal reaches a rendered string** — no wxid, no ``@chatroom``, asserted on the values
  that are actually rendered rather than on field names.

Everything here is synthetic. The ids are shaped like wxids precisely so that a leak is *findable*;
none of them is real. Scratch files go to a workspace-local directory because ``pytest``'s ``tmp_path``
cannot be created in this environment.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from memory import SessionConfig, build_sessions  # noqa: E402
from memory.account import (  # noqa: E402
    build_account_sessions,
    exports_from_directory,
    import_account,
)
from memory.events import OTHER_ROLE, SELF_ROLE, MemoryEvent, parse_txt_events  # noqa: E402
from memory.processor import (  # noqa: E402
    ConversationSessionProcessor,
    WeFlowSessionProcessor,
    session_documents,
)
from memory.senders import SENDER_LABEL_PREFIX, assign_sender_labels, column_label  # noqa: E402
from memory.weflow import OTHER, SELF, parse_weflow_events  # noqa: E402

SCRATCH_ROOT = BASE_DIR / "tests" / "_scratch_senders"

#: Synthetic identity. Shaped like the real thing (a wxid, a chatroom talker) so that a leak of one
#: into a rendered label is detectable by substring; invented, and belonging to nobody.
SELF_ID = "wxid_synthetic_me"
SENDER_A = "wxid_synthetic_member_a"
SENDER_B = "wxid_synthetic_member_b"
GROUP = "11112222@chatroom"
DIRECT = "wxid_synthetic_direct"

#: 2026-09-17 10:00 local, in Unix seconds. Only the *order* of these matters here.
T1000 = 1789000000


def moment(minute: int) -> int:
    """A synthetic ``createTime`` ``minute`` minutes after the base instant."""
    return T1000 + minute * 60


def msg(
    text: str,
    when: int,
    *,
    is_send: int = 0,
    sender: str = SENDER_A,
    local_id: int = 1,
    server_id: str | None = None,
) -> dict:
    """One WeFlow-shaped message."""
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
    return entry


def labels_of(events) -> list[str]:
    """The display label of each event, in stream order — what an evidence line would show."""
    return [event.sender_name for event in events]


def parse_labelled(payload, conversation_id: str):
    """Parse and label, i.e. the two steps every real path performs in that order."""
    return assign_sender_labels(
        parse_weflow_events(payload, conversation_id=conversation_id).events, conversation_id
    )


class _FakeFile:
    """Only ``.path`` is used by ``process_file_inner``."""

    def __init__(self, path: Path) -> None:
        self.path = path


@pytest.fixture()
def scratch() -> Path:
    """A clean workspace-local directory (``tmp_path`` is unavailable in this environment)."""
    directory = SCRATCH_ROOT / "case"
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def write_shard(root: Path, shard: str, exports: dict[str, list]) -> None:
    """Write ``{talker}_messages.json`` files for one shard — the exporter's on-disk layout."""
    directory = root / shard
    directory.mkdir(parents=True, exist_ok=True)
    for talker, messages in exports.items():
        (directory / f"{talker}_messages.json").write_text(
            json.dumps(messages, ensure_ascii=False), encoding="utf-8"
        )


# --- 1. role, identity and label are three separate things -------------------------------------


def test_a_self_message_is_still_the_literal_wo_and_keeps_its_raw_id_as_data() -> None:
    """The invariant the attribution safety logic depends on, and the identity it must not lose."""
    parsed = parse_weflow_events(
        [msg("我的话", moment(0), is_send=1, sender=SELF_ID)], conversation_id=GROUP
    )
    event = parsed.events[0]

    assert event.sender_name == "我"
    assert "] 我: " in event.line
    # The raw id survives as a field — and, as before, in the exporter's own metadata.
    assert event.speaker_role == SELF_ROLE
    assert event.speaker_id == SELF_ID
    assert event.metadata["senderUsername"] == SELF_ID


def test_the_adapters_provisional_label_is_replaced_only_by_role_not_by_identity() -> None:
    """The adapter's job is the role; deciding what a group member is *called* is a later pass."""
    parsed = parse_weflow_events(
        [msg("A 说的", moment(0), sender=SENDER_A)], conversation_id=GROUP
    )
    assert parsed.events[0].sender_name == OTHER, "the adapter no longer emits its provisional label"
    assert parsed.events[0].speaker_id == SENDER_A


def test_a_direct_conversation_still_renders_the_other_speaker_as_duifang() -> None:
    """Role alone distinguishes two speakers: the evaluated direct corpora must not move."""
    events = parse_labelled(
        [
            msg("我的话", moment(0), is_send=1, sender=SELF_ID),
            msg("他的话", moment(1), sender=SENDER_A),
        ],
        DIRECT,
    )
    assert labels_of(events) == ["我", "对方"]


def test_a_direct_conversation_keeps_duifang_even_for_several_other_ids() -> None:
    """A direct chat has one other side by definition; a renamed contact is not a new speaker.

    Nothing here tries to work out *which* wxid is the real peer. ``对方`` is what the record has
    always said for a direct chat, and this phase changes groups only.
    """
    events = parse_labelled(
        [
            msg("一", moment(0), sender=SENDER_A),
            msg("二", moment(1), sender=SENDER_B),
            msg("三", moment(2), sender=SENDER_A),
        ],
        DIRECT,
    )
    assert labels_of(events) == ["对方", "对方", "对方"]


# --- 2. the brief's case: a group tells its members apart ---------------------------------------


def test_a_group_renders_members_distinctly_instead_of_calling_them_all_duifang() -> None:
    """The acceptance-run defect, in the exact shape it was reported: four messages, one label."""
    events = parse_labelled(
        [
            msg("十点整我说话", moment(0), is_send=1, sender=SELF_ID),
            msg("A 说的", moment(1), sender=SENDER_A),
            msg("B 说的", moment(2), sender=SENDER_B),
            msg("A 又说", moment(3), sender=SENDER_A),
        ],
        GROUP,
    )
    assert labels_of(events) == ["我", "成员A", "成员B", "成员A"]
    assert labels_of(events) != ["对方", "对方", "对方", "对方"]


def test_two_members_posting_byte_identical_text_still_get_different_labels() -> None:
    """The case a text-based disambiguation could not solve: the lines are the same, the people are not.

    This is why the labels come from the stream position of an identity and never from the content.
    """
    events = parse_labelled(
        [
            msg("到楼下了", moment(0), sender=SENDER_A, local_id=1),
            msg("到楼下了", moment(1), sender=SENDER_B, local_id=2),
        ],
        GROUP,
    )
    assert events[0].text == events[1].text
    assert labels_of(events) == ["成员A", "成员B"]


def test_one_sender_keeps_one_label_for_the_whole_conversation() -> None:
    events = parse_labelled(
        [
            msg("A 第一条", moment(0), sender=SENDER_A, local_id=1),
            msg("B 第一条", moment(1), sender=SENDER_B, local_id=2),
            msg("A 第二条", moment(2), sender=SENDER_A, local_id=3),
            msg("B 第二条", moment(3), sender=SENDER_B, local_id=4),
        ],
        GROUP,
    )
    by_id = {event.speaker_id: event.sender_name for event in events}
    assert by_id == {SENDER_A: "成员A", SENDER_B: "成员B"}
    assert len(set(labels_of(events))) == 2, "two people ended up sharing a label"


def test_the_label_follows_chronology_not_the_order_the_export_happened_to_use() -> None:
    """A WeFlow export is not guaranteed to be in order; ``parse_weflow_events`` sorts it first."""
    events = parse_labelled(
        [
            msg("B 说的（更早）", moment(1), sender=SENDER_B, local_id=2),
            msg("A 说的（最早）", moment(0), sender=SENDER_A, local_id=1),
        ],
        GROUP,
    )
    assert labels_of(events) == ["成员A", "成员B"], "the first *speaker* was not the first in time"


# --- 3. per conversation, over the whole merged stream -----------------------------------------


def test_a_group_spanning_shards_is_labelled_once_over_the_merged_stream(scratch: Path) -> None:
    """The subtle failure this placement exists to prevent, stated as the alternative.

    ``MSG1`` holds the earlier messages, ``MSG0`` the later ones — and ``MSG0``'s first speaker is
    *not* ``MSG1``'s first speaker. Labelled per shard, each shard would start its own numbering and
    both of those people would be ``成员A``: two members, one name, which reads as a fact rather than
    as an absence and is worse than the ``对方`` it replaced.
    """
    write_shard(
        scratch,
        "MSG1",
        {
            GROUP: [
                msg("A 最早", moment(0), sender=SENDER_A, local_id=1, server_id="s1"),
                msg("B 其次", moment(1), sender=SENDER_B, local_id=2, server_id="s2"),
            ]
        },
    )
    write_shard(
        scratch,
        "MSG0",
        {
            GROUP: [
                msg("B 再其次", moment(2), sender=SENDER_B, local_id=3, server_id="s3"),
                msg("A 最后", moment(3), sender=SENDER_A, local_id=4, server_id="s4"),
            ]
        },
    )

    exports = tuple(exports_from_directory(scratch / "MSG1", shard="MSG1")) + tuple(
        exports_from_directory(scratch / "MSG0", shard="MSG0")
    )
    events_by_conversation, _ = import_account(exports, shards_detected=["MSG0", "MSG1"])
    events = events_by_conversation[GROUP]

    assert [event.text for event in events] == ["A 最早", "B 其次", "B 再其次", "A 最后"]
    assert labels_of(events) == ["成员A", "成员B", "成员B", "成员A"]
    # Per-shard labelling would have produced ["成员A", "成员B"] in MSG1 and ["成员A", "成员B"] again in
    # MSG0 — the same two labels for the same two people here only by luck. The distinguishing case is
    # that MSG0's first speaker is B, and B is 成员B, not 成员A:
    assert labels_of(events)[2] == "成员B"
    assert not (
        labels_of(events)[:2] == ["成员A", "成员B"] and labels_of(events)[2:] == ["成员A", "成员B"]
    ), "each shard was labelled independently"


def test_a_member_present_in_two_shards_keeps_one_label(scratch: Path) -> None:
    """The same person posting in two shards is one person, whichever shard the line came from."""
    write_shard(
        scratch,
        "MSG0",
        {GROUP: [msg("A 在 MSG0", moment(5), sender=SENDER_A, local_id=1, server_id="s1")]},
    )
    write_shard(
        scratch,
        "MSG1",
        {
            GROUP: [
                msg("A 在 MSG1", moment(0), sender=SENDER_A, local_id=2, server_id="s2"),
                msg("B 在 MSG1", moment(1), sender=SENDER_B, local_id=3, server_id="s3"),
            ]
        },
    )

    exports = tuple(exports_from_directory(scratch / "MSG0", shard="MSG0")) + tuple(
        exports_from_directory(scratch / "MSG1", shard="MSG1")
    )
    events_by_conversation, _ = import_account(exports, shards_detected=["MSG0", "MSG1"])
    events = events_by_conversation[GROUP]

    assert labels_of(events) == ["成员A", "成员B", "成员A"]
    assert len({event.sender_name for event in events}) == 2


def test_a_duplicate_message_removed_by_the_merge_does_not_lose_its_speaker_a_label() -> None:
    """Labelling after dedupe: a dropped duplicate must not be what a member was numbered by."""
    shard_a = parse_weflow_events(
        [msg("重复的", moment(0), sender=SENDER_B, local_id=1, server_id="dup")], conversation_id=GROUP
    )
    shard_b = parse_weflow_events(
        [
            msg("重复的", moment(0), sender=SENDER_B, local_id=1, server_id="dup"),
            msg("A 说的", moment(1), sender=SENDER_A, local_id=2, server_id="other"),
        ],
        conversation_id=GROUP,
    )
    from memory.shards import merge_shard_events

    events, report = merge_shard_events([("a", shard_a), ("b", shard_b)], conversation_id=GROUP)

    assert report.duplicates_removed == 1
    assert labels_of(events) == ["成员A", "成员B"], "numbering followed the raw stream, not the kept one"


# --- 4. deterministic, and never derived from the id -------------------------------------------


def test_two_runs_over_the_same_exports_produce_identical_labels(scratch: Path) -> None:
    write_shard(
        scratch,
        "MSG0",
        {
            GROUP: [
                msg("A", moment(0), sender=SENDER_A, local_id=1, server_id="s1"),
                msg("B", moment(1), sender=SENDER_B, local_id=2, server_id="s2"),
                msg("我", moment(2), is_send=1, sender=SELF_ID, local_id=3, server_id="s3"),
            ]
        },
    )

    def run() -> tuple[tuple[str, ...], tuple[str, ...]]:
        exports = tuple(exports_from_directory(scratch / "MSG0", shard="MSG0"))
        events_by_conversation, _ = import_account(exports, shards_detected=["MSG0"])
        chunks = build_account_sessions(events_by_conversation, SessionConfig(max_chars=900))
        return (
            tuple(event.sender_name for event in events_by_conversation[GROUP]),
            tuple(chunk.text for chunk in chunks),
        )

    assert run() == run()


def test_the_same_stream_labelled_twice_is_identical_and_labelling_is_idempotent() -> None:
    parsed = parse_weflow_events(
        [
            msg("A", moment(0), sender=SENDER_A, local_id=1),
            msg("B", moment(1), sender=SENDER_B, local_id=2),
        ],
        conversation_id=GROUP,
    )
    once = assign_sender_labels(parsed.events, GROUP)
    twice = assign_sender_labels(once, GROUP)
    assert labels_of(once) == labels_of(twice) == ["成员A", "成员B"]


def test_no_label_contains_the_raw_id_or_a_derived_token() -> None:
    events = parse_labelled(
        [
            msg("A", moment(0), sender=SENDER_A, local_id=1),
            msg("B", moment(1), sender=SENDER_B, local_id=2),
        ],
        GROUP,
    )
    for event in events:
        assert event.speaker_id not in event.sender_name
        assert "wxid" not in event.sender_name
        assert "chatroom" not in event.sender_name
        assert event.sender_name.startswith(SENDER_LABEL_PREFIX)


# --- 5. the numbering keeps going ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("index", "expected"),
    [(0, "A"), (1, "B"), (25, "Z"), (26, "AA"), (27, "AB"), (51, "AZ"), (52, "BA"), (701, "ZZ")],
)
def test_column_labels_follow_the_excel_column_sequence(index: int, expected: str) -> None:
    assert column_label(index) == expected


def test_a_group_with_more_senders_than_letters_continues_correctly() -> None:
    """28 members: the 27th is past Z, and a scheme that ran out would reuse a name."""
    payload = [
        msg(f"消息 {i}", moment(i), sender=f"wxid_synthetic_member_{i:02d}", local_id=i + 1)
        for i in range(28)
    ]
    events = parse_labelled(payload, GROUP)

    assert labels_of(events)[25] == "成员Z"
    assert labels_of(events)[26] == "成员AA"
    assert labels_of(events)[27] == "成员AB"
    assert len(set(labels_of(events))) == 28, "two members share a label"


# --- 6. the edges: nobody else, and somebody unidentifiable -------------------------------------


def test_an_all_self_group_renders_only_wo() -> None:
    events = parse_labelled(
        [
            msg("一", moment(0), is_send=1, sender=SELF_ID, local_id=1),
            msg("二", moment(1), is_send=1, sender=SELF_ID, local_id=2),
        ],
        GROUP,
    )
    assert labels_of(events) == ["我", "我"]
    assert set(labels_of(events)) == {"我"}


def test_a_group_with_a_single_other_speaker_gets_one_member_label() -> None:
    """Nothing about this phase makes a one-member group render differently from a two-member one."""
    events = parse_labelled(
        [
            msg("我", moment(0), is_send=1, sender=SELF_ID, local_id=1),
            msg("他", moment(1), sender=SENDER_A, local_id=2),
            msg("他又是他", moment(2), sender=SENDER_A, local_id=3),
        ],
        GROUP,
    )
    assert labels_of(events) == ["我", "成员A", "成员A"]


def test_a_group_sender_without_a_username_is_left_as_duifang_rather_than_invented() -> None:
    """Two unidentified messages are not evidence of two people — so neither gets a member label.

    ``对方`` is the honest answer, it is what the record said before this phase, and it can never
    collide with an assigned ``成员X``.
    """
    anonymous = msg("没有 senderUsername", moment(0), sender=SENDER_A, local_id=1)
    del anonymous["senderUsername"]
    events = parse_labelled(
        [anonymous, msg("有身份", moment(1), sender=SENDER_A, local_id=2)], GROUP
    )
    assert labels_of(events) == ["对方", "成员A"]
    assert events[0].speaker_id == ""


def test_a_sender_id_that_is_the_conversation_itself_is_not_an_identity() -> None:
    """The real export's shape, measured: 562 of 589 files name the *conversation* as the sender.

    Every non-self message in a real group carries ``senderUsername`` equal to the group's own talker,
    so treating that field as an identity stamps one ``成员A`` across the whole group — which reads as
    "exactly one other person said all of this", a claim the record does not support and a worse answer
    than the ``对方`` it replaces. The conversation id is never a sender.
    """
    events = parse_labelled(
        [
            msg("第一条", moment(0), sender=GROUP, local_id=1),
            msg("第二条", moment(1), sender=GROUP, local_id=2),
            msg("我说话", moment(2), is_send=1, sender=SELF_ID, local_id=3),
        ],
        GROUP,
    )
    assert labels_of(events) == ["对方", "对方", "我"], (
        "the group's own id is not a member, so every one of its messages stays 对方"
    )
    # and it is still kept as data rather than discarded
    assert events[0].speaker_id == GROUP


def test_a_real_sender_id_still_labels_even_when_another_message_names_the_conversation() -> None:
    """The guard must reject the conversation id without disabling the labelling it exists for."""
    events = parse_labelled(
        [
            msg("来自群本身的记录", moment(0), sender=GROUP, local_id=1),
            msg("张三说的", moment(1), sender=SENDER_A, local_id=2),
            msg("李四说的", moment(2), sender=SENDER_B, local_id=3),
        ],
        GROUP,
    )
    assert labels_of(events) == ["对方", "成员A", "成员B"]


def test_an_empty_stream_labels_to_nothing() -> None:
    assert assign_sender_labels([], GROUP) == []
    assert assign_sender_labels([], DIRECT) == []


def test_labelling_reorders_nothing_and_drops_nothing() -> None:
    parsed = parse_weflow_events(
        [
            msg("A", moment(0), sender=SENDER_A, local_id=1),
            msg("我", moment(1), is_send=1, sender=SELF_ID, local_id=2),
            msg("B", moment(2), sender=SENDER_B, local_id=3),
        ],
        conversation_id=GROUP,
    )
    events = assign_sender_labels(parsed.events, GROUP)
    assert [event.id for event in events] == [event.id for event in parsed.events]
    assert [event.timestamp for event in events] == [event.timestamp for event in parsed.events]
    assert [event.text for event in events] == [event.text for event in parsed.events]


# --- 7. the single-file paths label, and a shard does not ---------------------------------------


def test_the_single_file_processor_labels_a_group_export(scratch: Path) -> None:
    """A whole conversation in one file: the stream *is* the conversation, so labelling it is safe."""
    path = scratch / f"{GROUP}.json"
    path.write_text(
        json.dumps(
            [
                msg("A 说的", moment(0), sender=SENDER_A, local_id=1),
                msg("我的话", moment(1), is_send=1, sender=SELF_ID, local_id=2),
                msg("B 说的", moment(2), sender=SENDER_B, local_id=3),
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    document = asyncio.run(
        WeFlowSessionProcessor(session_config=SessionConfig(max_chars=900)).process_file_inner(
            _FakeFile(path)
        )
    )
    text = document.chunks[0].page_content

    assert "] 成员A: A 说的" in text
    assert "] 我: 我的话" in text
    assert "] 成员B: B 说的" in text
    assert "对方" not in text


def test_the_processor_leaves_a_direct_export_exactly_as_it_was(scratch: Path) -> None:
    path = scratch / f"{DIRECT}.json"
    path.write_text(
        json.dumps(
            [
                msg("他的话", moment(0), sender=SENDER_A, local_id=1),
                msg("我的话", moment(1), is_send=1, sender=SELF_ID, local_id=2),
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    document = asyncio.run(
        WeFlowSessionProcessor(session_config=SessionConfig(max_chars=900)).process_file_inner(
            _FakeFile(path)
        )
    )
    text = document.chunks[0].page_content
    assert "] 对方: 他的话" in text and "] 我: 我的话" in text
    assert "成员" not in text


def test_the_plain_text_path_is_untouched_because_its_speakers_are_already_names() -> None:
    """A txt export names its speakers; a pass that renamed them would destroy the corpus.

    ``assign_sender_labels`` is a strict no-op on a stream whose source declares no role, so this is a
    property of the function and not merely of nobody calling it.
    """
    parsed = parse_txt_events("[2026-09-17 10:00] 张三: 你好\n[2026-09-17 10:01] 我: 你好\n")
    events = assign_sender_labels(parsed.events, GROUP)

    assert labels_of(events) == ["张三", "我"]
    assert all(event.speaker_role == "" for event in events)
    assert events == list(parsed.events), "the plain-text events were modified"


def test_the_plain_text_processor_output_is_unchanged(scratch: Path) -> None:
    body = "[2026-09-17 10:00] 张三: 你好\n[2026-09-17 10:01] 我: 你好\n"
    path = scratch / "chats.txt"
    path.write_text(body, encoding="utf-8")

    document = asyncio.run(
        ConversationSessionProcessor(session_config=SessionConfig(max_chars=900)).process_file_inner(
            _FakeFile(path)
        )
    )
    assert document.chunks[0].page_content == body.rstrip("\n")


# --- 8. nothing internal reaches anything rendered ----------------------------------------------


def _group_events(scratch: Path):
    write_shard(
        scratch,
        "MSG0",
        {
            GROUP: [
                msg("A 说的", moment(0), sender=SENDER_A, local_id=1, server_id="s1"),
                msg("我的话", moment(1), is_send=1, sender=SELF_ID, local_id=2, server_id="s2"),
                msg("B 说的", moment(2), sender=SENDER_B, local_id=3, server_id="s3"),
            ]
        },
    )
    exports = tuple(exports_from_directory(scratch / "MSG0", shard="MSG0"))
    events_by_conversation, _ = import_account(exports, shards_detected=["MSG0"])
    return events_by_conversation[GROUP]


def test_no_raw_identity_reaches_any_rendered_line(scratch: Path) -> None:
    """Asserted on the rendered strings, not on field names: the leak that survived a key-name check."""
    events = _group_events(scratch)
    rendered = "\n".join(event.line for event in events)

    for secret in (SENDER_A, SENDER_B, SELF_ID, "wxid", "@chatroom"):
        assert secret not in rendered, f"{secret!r} reached a rendered evidence line"
    assert "成员A" in rendered and "成员B" in rendered


def test_no_raw_identity_reaches_a_session_chunk_or_its_participants(scratch: Path) -> None:
    """A chunk's text and its participant list are what the LLM and the evidence card read."""
    events = _group_events(scratch)
    chunks = build_sessions(events, SessionConfig(max_chars=900), conversation_id=GROUP)

    assert len(chunks) == 1
    blob = json.dumps(
        {
            "text": chunks[0].text,
            "participants": list(chunks[0].participants),
            "metadata": dict(chunks[0].metadata),
        },
        ensure_ascii=False,
    )
    for secret in (SENDER_A, SENDER_B, SELF_ID, "wxid", "@chatroom"):
        assert secret not in blob, f"{secret!r} reached a retrieval unit"
    assert set(chunks[0].participants) == {"我", "成员A", "成员B"}


def test_no_sender_identity_reaches_the_document_the_prompt_and_the_api_read(scratch: Path) -> None:
    """``session_documents`` is the single definition of what a retrieval unit carries outwards.

    The assertion is on **values**, not on field names, because the leak this project already had rode
    inside a field whose *name* said nothing about identity. One internal token is deliberately not
    asserted against here: the conversation id, which for a group ends in ``@chatroom`` and is part of
    ``conversation_id`` and of ``memory_chunk_id``. That predates this phase — and it is exactly why
    the web projection exists: ``test_web.py`` checks the id never reaches the browser payload.
    """
    events = _group_events(scratch)
    chunks = build_sessions(events, SessionConfig(max_chars=900), conversation_id=GROUP)
    documents = session_documents(chunks, skipped_source_lines=0)

    for secret in (SENDER_A, SENDER_B, SELF_ID, "wxid"):
        assert secret not in documents[0].page_content, f"{secret!r} reached the prompt text"
        assert secret not in json.dumps(documents[0].metadata, ensure_ascii=False), (
            f"{secret!r} reached the served metadata"
        )
    # And the conversation token reaches neither the prompt text nor the participant list.
    assert "@chatroom" not in documents[0].page_content
    assert not any("@chatroom" in name for name in documents[0].metadata["participants"])


def test_an_event_line_never_carries_the_identity_even_for_a_self_message() -> None:
    """``我`` is a role label; the wxid behind it must not ride along beside it."""
    parsed = parse_weflow_events(
        [msg("我的话", moment(0), is_send=1, sender=SELF_ID)], conversation_id=DIRECT
    )
    assert SELF_ID not in parsed.events[0].line


# --- 9. the conversation boundary is still absolute --------------------------------------------


def test_labelling_does_not_let_one_conversation_borrow_anothers_labels(scratch: Path) -> None:
    """Two conversations are labelled independently, even when they share a member."""
    write_shard(
        scratch,
        "MSG0",
        {
            GROUP: [
                msg("群里 A", moment(0), sender=SENDER_A, local_id=1, server_id="g1"),
                msg("群里 B", moment(1), sender=SENDER_B, local_id=2, server_id="g2"),
            ],
            DIRECT: [msg("私聊 A", moment(2), sender=SENDER_A, local_id=3, server_id="d1")],
        },
    )
    exports = tuple(exports_from_directory(scratch / "MSG0", shard="MSG0"))
    events_by_conversation, _ = import_account(exports, shards_detected=["MSG0"])

    assert labels_of(events_by_conversation[GROUP]) == ["成员A", "成员B"]
    # The direct chat is its own conversation: same person, different rendering, and no member label.
    assert labels_of(events_by_conversation[DIRECT]) == ["对方"]

    chunks = build_account_sessions(events_by_conversation, SessionConfig(max_chars=900))
    assert len({chunk.conversation_id for chunk in chunks}) == 2
    for chunk in chunks:
        owners = {
            event.conversation_id
            for event in events_by_conversation[chunk.conversation_id]
            if event.id in chunk.event_ids
        }
        assert owners <= {chunk.conversation_id}, "a chunk mixed two conversations"
