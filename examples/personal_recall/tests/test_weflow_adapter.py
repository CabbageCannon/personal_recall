"""Offline tests for the WeFlow JSON source adapter (no model, no network, no real chat data).

All fixtures here are synthetic. The load-bearing properties are: real fields map to the right
``MemoryEvent`` slots, no internal markup payload can reach the embedding corpus, ids stay unique
without ever dropping a message, and the emitted order is chronological — because ``build_sessions``
consumes sequence order and never re-sorts.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from memory import SessionConfig, build_sessions  # noqa: E402
from memory.weflow import (  # noqa: E402
    NO_TEXT_PLACEHOLDER,
    build_event_id,
    classify,
    normalize_body,
    parse_weflow_events,
    select_body,
    unwrap_payload,
)

# 2026-09-17 11:26 and 12:34 in Unix seconds (synthetic times, synthetic people).
T1126 = 1789000000
T1234 = 1789004000
T0700 = 1788990000

#: A multi-KB internal payload of the kind that must never reach retrieval.
EMOJI_PAYLOAD = (
    '<msg><emoji fromusername="wxid_synthetic" tousername="wxid_synthetic" '
    'type="2" idbuffer="buffer_synthetic" md5="0123456789abcdef" '
    'len="240" productid="com.tencent.xin.emoticon.person.stock" '
    'cdnurl="http://synthetic.invalid/emoji/asset.gif" '
    'thumburl="http://synthetic.invalid/emoji/thumb.gif" '
    'aeskey="00112233445566778899aabbccddeeff" />'
    "<gameext type=\"0\" content=\"0\" /></msg>"
)


def entry(local_id, create_time, *, is_send=1, parsed=None, content=None, raw=None, **extra):
    item = {
        "localId": local_id,
        "serverId": f"srv{local_id}",
        "localType": 1,
        "createTime": create_time,
        "isSend": is_send,
        "senderUsername": "wxid_synthetic_person",
    }
    if parsed is not None:
        item["parsedContent"] = parsed
    if content is not None:
        item["content"] = content
    if raw is not None:
        item["rawContent"] = raw
    item.update(extra)
    return item


# --- envelope handling ---------------------------------------------------------------------


def test_accepts_a_bare_list() -> None:
    result = parse_weflow_events([entry(1, T1126, parsed="我又点了拌粉")])
    assert len(result.events) == 1
    assert result.schema is None


def test_accepts_a_versioned_envelope() -> None:
    payload = {
        "schema": "weflow-message/v1",
        "source": "synthetic",
        "generatedAt": "2026-09-17T20:00:00Z",
        "coverage": {"shards": ["MSG0.db", "MSG2.db"]},
        "messages": [entry(1, T1126, parsed="我又点了拌粉")],
    }
    result = parse_weflow_events(payload)
    assert len(result.events) == 1
    assert result.schema == "weflow-message/v1"
    assert result.events[0].metadata["schema"] == "weflow-message/v1"


def test_unwrap_handles_a_single_message_object() -> None:
    messages, schema = unwrap_payload(entry(1, T1126, parsed="hi"))
    assert len(messages) == 1 and schema is None


def test_unknown_top_level_shape_yields_nothing_rather_than_raising() -> None:
    assert unwrap_payload({"unexpected": True}) == ([], None)
    assert unwrap_payload("not a payload") == ([], None)
    assert parse_weflow_events({"unexpected": True}).events == ()


# --- field mapping -------------------------------------------------------------------------


def test_create_time_becomes_the_timestamp() -> None:
    result = parse_weflow_events([entry(1, T1126, parsed="x")])
    assert result.events[0].timestamp == datetime.fromtimestamp(T1126).replace(microsecond=0)


def test_millisecond_timestamps_are_tolerated() -> None:
    result = parse_weflow_events([entry(1, T1126 * 1000, parsed="x")])
    assert result.events[0].timestamp.year == datetime.fromtimestamp(T1126).year


def test_is_send_decides_the_speaker_and_never_uses_the_username() -> None:
    result = parse_weflow_events(
        [entry(1, T1126, is_send=1, parsed="mine"), entry(2, T1234, is_send=0, parsed="theirs")]
    )
    by_text = {event.text: event.sender_name for event in result.events}
    assert by_text == {"mine": "我", "theirs": "对方"}
    # the real identity is preserved, just not used as the speaker
    assert result.events[0].metadata["senderUsername"] == "wxid_synthetic_person"


def test_body_priority_prefers_parsed_then_content_then_raw() -> None:
    assert select_body(entry(1, T1126, parsed="P", content="C", raw="R"))[0] == "P"
    assert select_body(entry(1, T1126, content="C", raw="R"))[0] == "C"
    assert select_body(entry(1, T1126, raw="R"))[0] == "R"
    assert select_body(entry(1, T1126, parsed="   "))[0] == ""


def test_body_field_used_is_recorded() -> None:
    result = parse_weflow_events([entry(1, T1126, content="C")])
    assert result.events[0].metadata["body_field"] == "content"


def test_metadata_carries_the_export_fields() -> None:
    result = parse_weflow_events([entry(7, T1126, parsed="x")])
    metadata = result.events[0].metadata
    assert metadata["source_type"] == "weflow"
    for field in ("localId", "serverId", "localType", "senderUsername", "isSend", "createTime"):
        assert field in metadata, f"missing {field}"


def test_unknown_fields_do_not_break_parsing() -> None:
    result = parse_weflow_events(
        [entry(1, T1126, parsed="x", brandNewExporterField={"a": 1}, anotherField="y")]
    )
    assert len(result.events) == 1


def test_entries_without_a_usable_time_are_skipped_and_counted() -> None:
    result = parse_weflow_events(
        [entry(1, T1126, parsed="ok"), {"localId": 2, "parsedContent": "no time"}, "junk"]
    )
    assert len(result.events) == 1
    assert result.skipped == 2


# --- ids -----------------------------------------------------------------------------------


def test_id_never_uses_local_id_alone() -> None:
    event_id = build_event_id("conv", "srv9", 123, 0)
    assert "conv" in event_id and "srv9" in event_id and "123" in event_id
    without_server = build_event_id("conv", None, 123, 0)
    assert without_server != "123"
    assert "conv" in without_server


def test_id_falls_back_between_server_and_position() -> None:
    with_server = build_event_id("c", "s1", 5, 0)
    with_local_only = build_event_id("c", None, 5, 0)
    with_neither = build_event_id("c", None, None, 3)
    assert len({with_server, with_local_only, with_neither}) == 3


def test_repeated_local_ids_across_conversations_stay_distinct() -> None:
    payload = [entry(1, T1126, parsed="a"), entry(1, T1234, parsed="b")]
    first = parse_weflow_events(payload, conversation_id="alice")
    second = parse_weflow_events(payload, conversation_id="bob")
    assert {e.id for e in first.events}.isdisjoint({e.id for e in second.events})


def test_id_collisions_are_disambiguated_and_no_message_is_dropped() -> None:
    payload = [entry(4, T1126, parsed="first"), entry(4, T1126, parsed="second")]
    result = parse_weflow_events(payload)
    assert len(result.events) == 2, "a message was dropped to satisfy an id invariant"
    assert len({event.id for event in result.events}) == 2
    assert result.id_collisions == 1


# --- non-text handling ---------------------------------------------------------------------


def test_placeholder_tokens_are_typed_and_kept_short() -> None:
    result = parse_weflow_events(
        [entry(1, T1126, parsed="[图片]"), entry(2, T1234, parsed="[语音]")]
    )
    kinds = {event.text: event.message_type for event in result.events}
    assert kinds == {"[图片]": "image", "[语音]": "voice"}


def test_markup_payload_is_replaced_by_a_placeholder() -> None:
    result = parse_weflow_events([entry(1, T1126, parsed=EMOJI_PAYLOAD)])
    event = result.events[0]
    assert event.text == "[表情]"
    assert event.message_type == "sticker"
    assert result.suppressed_payloads == 1


def test_no_raw_markup_can_reach_the_events() -> None:
    """The property that protects the embedding corpus, asserted over every markup kind."""
    payload = [
        entry(1, T0700, parsed=EMOJI_PAYLOAD),
        entry(2, T1126, parsed='<msg><img aeskey="00112233445566778899aabbccddeeff" /></msg>'),
        entry(3, T1234, parsed='<msg><voip msgtype="1" /></msg>'),
        entry(4, T1234 + 60, parsed="<msg><appmsg><title>shared</title></appmsg></msg>"),
    ]
    result = parse_weflow_events(payload)
    assert result.suppressed_payloads == 4
    for event in result.events:
        assert "<" not in event.text and ">" not in event.text
        assert "aeskey" not in event.text
        assert len(event.text) < 20


def test_a_long_payload_is_never_carried_into_the_event_text() -> None:
    result = parse_weflow_events([entry(1, T1126, parsed=EMOJI_PAYLOAD * 40)])
    assert len(result.events[0].text) < 20


def test_missing_text_becomes_an_explicit_placeholder() -> None:
    result = parse_weflow_events([entry(1, T1126, content="   ")])
    assert result.events[0].text == NO_TEXT_PLACEHOLDER
    assert result.events[0].message_type == "non_text"
    assert result.missing_text == 1


def test_normalize_body_collapses_to_one_line() -> None:
    assert normalize_body("line one\r\nline two\n\nline three  ") == "line one line two line three"


def test_classify_is_deterministic_for_the_same_input() -> None:
    assert classify(EMOJI_PAYLOAD) == classify(EMOJI_PAYLOAD)
    assert classify("plain text") == ("text", "plain text", False)


# --- ordering and integration --------------------------------------------------------------


def test_events_are_sorted_chronologically_even_when_the_export_is_not() -> None:
    payload = [
        entry(3, T1234, parsed="third"),
        entry(1, T0700, parsed="first"),
        entry(2, T1126, parsed="second"),
    ]
    result = parse_weflow_events(payload)
    assert [event.text for event in result.events] == ["first", "second", "third"]


def test_parsing_is_deterministic() -> None:
    payload = [entry(2, T1234, parsed="b"), entry(1, T1126, parsed="a")]
    first = parse_weflow_events(payload)
    second = parse_weflow_events(payload)
    assert [e.id for e in first.events] == [e.id for e in second.events]
    assert [e.line for e in first.events] == [e.line for e in second.events]


def test_chronological_output_keeps_one_conversation_in_one_session() -> None:
    """Regression guard for the Phase M2 failure: unsorted input fragments every message."""
    payload = [entry(i, T1126 + i * 60, parsed=f"message {i}") for i in range(8)][::-1]
    result = parse_weflow_events(payload)
    sessions = build_sessions(result.events, SessionConfig(max_chars=900))
    assert len(sessions) == 1, f"fragmented into {len(sessions)} sessions"
    assert sessions[0].n_events == 8


def test_events_render_the_canonical_evidence_line() -> None:
    result = parse_weflow_events([entry(1, T1126, parsed="我又点了拌粉")])
    line = result.events[0].line
    assert line.startswith("[") and "] 我: " in line and line.endswith("我又点了拌粉")


def _scratch(name: str) -> Path:
    """Workspace-local scratch dir; pytest's tmp_path cannot be created in this environment."""
    path = BASE_DIR / "tests" / "_scratch_weflow" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_load_from_file_uses_the_stem_as_conversation_id() -> None:
    import shutil

    from memory.weflow import load_weflow_file

    directory = _scratch("stem")
    try:
        path = directory / "synthetic_conversation.json"
        path.write_text(
            json.dumps([entry(1, T1126, parsed="hi")], ensure_ascii=False), encoding="utf-8"
        )
        result = load_weflow_file(path)
        assert result.events[0].conversation_id == "synthetic_conversation"
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_file_reader_tolerates_a_utf8_bom() -> None:
    import shutil

    from memory.weflow import load_weflow_file

    directory = _scratch("bom")
    try:
        path = directory / "bom.json"
        path.write_text("\ufeff" + json.dumps([entry(1, T1126, parsed="hi")]), encoding="utf-8")
        assert len(load_weflow_file(path).events) == 1
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_as_parse_result_matches_the_engine_wide_shape() -> None:
    result = parse_weflow_events([entry(1, T1126, parsed="x"), "junk"])
    generic = result.as_parse_result()
    assert generic.events == result.events
    assert generic.skipped_lines == result.skipped


@pytest.mark.parametrize("bad", [None, {}, 12345, [None], [[]]])
def test_malformed_payloads_never_raise(bad) -> None:
    result = parse_weflow_events(bad)
    assert isinstance(result.events, tuple)
