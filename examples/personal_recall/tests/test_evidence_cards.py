"""Offline tests for evidence-card resolution (no model, no network)."""

from __future__ import annotations

import json

from evidence_cards import (
    build_evidence_cards,
    cards_to_dict,
    parse_chat_lines,
    render_cards,
)

SOURCES = [
    {
        "rank": 1,
        "chunk_index": 7,
        "memory_chunk_id": "stress-session-0007",
        "start_time": "2024-05-02 18:26:00",
        "end_time": "2024-05-02 18:36:00",
        "participants": ["张三", "我"],
        "n_events": 4,
        "content": (
            "[2024-05-02 18:26] 张三: 五一后面两天你有空没\n"
            "[2024-05-02 18:27] 我: 想去哪？\n"
            "[2024-05-02 18:28] 张三: 长沙吧，高铁过去也不算远\n"
        ),
    },
    {
        "rank": 2,
        "chunk_index": 9,
        "memory_chunk_id": "stress-session-0009",
        "start_time": "2024-06-15 20:51:00",
        "end_time": "2024-06-15 20:51:00",
        "participants": ["小汪"],
        "n_events": 1,
        "content": "[2024-06-15 20:51] 小汪: 行，我最近天天泡健身房，等你考完喊你一起去\n",
    },
    {
        "rank": 3,
        "chunk_index": 11,
        "memory_chunk_id": "stress-session-0011",
        "start_time": "2024-07-21 15:10:00",
        "end_time": "2024-07-21 15:10:00",
        "participants": ["同学A"],
        "n_events": 1,
        "content": "[2024-07-21 15:10] 同学A: 那你看看 Neon，也是 PostgreSQL\n",
    },
]


def test_parse_chat_lines_ignores_non_message_text() -> None:
    lines = parse_chat_lines("一些标题\n[2024-05-02 18:28] 张三: 长沙吧\n不是消息的一行\n")

    assert len(lines) == 1
    assert lines[0].speaker == "张三" and lines[0].text == "长沙吧"


def test_citations_map_to_cards_zero_based_and_in_citation_order() -> None:
    cards = build_evidence_cards("结论 [来源 2] 和 [来源 0]。", SOURCES)

    assert [c.citation_index for c in cards] == [2, 0]
    assert [c.memory_chunk_id for c in cards] == ["stress-session-0011", "stress-session-0007"]
    assert all(c.cited for c in cards)


def test_repeated_citations_do_not_duplicate_cards() -> None:
    cards = build_evidence_cards("[来源 0][来源 0] 同一来源", SOURCES)

    assert [c.citation_index for c in cards] == [0]


def test_out_of_range_citations_are_ignored() -> None:
    cards = build_evidence_cards("[来源 9] 不存在", SOURCES)

    assert cards == []


def test_card_label_and_lines_expose_provenance() -> None:
    card = build_evidence_cards("[来源 0]", SOURCES)[0]

    assert card.label == "[张三, 我 · 2024-05-02 18:26:00 – 2024-05-02 18:36:00]"
    assert len(card.lines) == 3
    assert "长沙吧" in card.render()


def test_single_timestamp_card_does_not_render_a_range() -> None:
    card = build_evidence_cards("[来源 1]", SOURCES)[0]

    assert card.label == "[小汪 · 2024-06-15 20:51:00]"


def test_uncited_sources_can_be_included_and_are_marked() -> None:
    cards = build_evidence_cards("没有任何引用", SOURCES, include_uncited=True)

    assert [c.cited for c in cards] == [False, False, False]
    assert "the answer cited no source" in render_cards(cards)
    assert "Retrieved but not cited (3)" in render_cards(cards)


def test_render_reports_the_cited_count() -> None:
    text = render_cards(build_evidence_cards("[来源 1]", SOURCES))

    assert "Evidence (1 cited source(s))" in text
    assert "泡健身房" in text


def test_cards_to_dict_is_json_serialisable() -> None:
    payload = cards_to_dict(build_evidence_cards("[来源 0]", SOURCES))

    assert json.loads(json.dumps(payload, ensure_ascii=False))[0]["memory_chunk_id"] == "stress-session-0007"
    assert payload[0]["lines"][2]["text"] == "长沙吧，高铁过去也不算远"
    assert payload[0]["cited"] is True
