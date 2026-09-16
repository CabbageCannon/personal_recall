"""Offline, deterministic tests for the Phase 1 memory layer.

No model, no network, no API. Run from ``examples/personal_recall``::

    python -m pytest tests -q
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from eval_utils import normalize_text
from memory import SessionConfig, build_sessions, parse_txt_events

BASE_DIR = Path(__file__).resolve().parent.parent
STRESS_CORPUS = BASE_DIR / "data" / "stress_chats.txt"
STRESS_QUERIES = BASE_DIR / "data" / "stress_queries.json"


def ts(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------- parse


def test_parse_extracts_events_in_order_with_deterministic_ids() -> None:
    text = "\n".join(
        [
            "[2024-05-02 18:26] 张三: 五一后面两天你有空没",
            "",
            "[2024-05-02 18:27] 我: 想去哪？",
        ]
    )
    result = parse_txt_events(text, conversation_id="chats")

    assert result.skipped_lines == 0
    assert [e.id for e in result.events] == ["chats#00001", "chats#00002"]
    assert [e.sender_name for e in result.events] == ["张三", "我"]
    assert result.events[0].timestamp == ts("2024-05-02 18:26")
    # the line property must round-trip so evidence stays traceable
    assert result.events[0].line == "[2024-05-02 18:26] 张三: 五一后面两天你有空没"


def test_parse_skips_junk_and_handles_crlf() -> None:
    text = "garbage\r\n[2024-05-02 18:27] 我: 想去哪？\r\n\r\nnot a message: x\r\n"
    result = parse_txt_events(text)

    assert len(result.events) == 1
    assert result.skipped_lines == 2


def test_parse_ids_are_stable_across_calls() -> None:
    text = "[2024-01-01 09:00] 我: 早安\n[2024-01-01 09:01] 小王: 早"
    first = parse_txt_events(text).events
    second = parse_txt_events(text).events
    assert [e.id for e in first] == [e.id for e in second]


# --------------------------------------------------------------------------- sessions


def _lines(*entries: tuple[str, str, str]) -> list[str]:
    return [f"[{stamp}] {sender}: {text}" for stamp, sender, text in entries]


def test_close_messages_on_one_day_form_one_session() -> None:
    events = parse_txt_events(
        "\n".join(
            _lines(
                ("2024-05-02 18:26", "张三", "有空没"),
                ("2024-05-02 18:27", "我", "想去哪？"),
                ("2024-05-02 18:28", "张三", "长沙吧"),
            )
        )
    ).events

    sessions = build_sessions(events)

    assert len(sessions) == 1
    session = sessions[0]
    assert session.n_events == 3
    assert session.start_time == ts("2024-05-02 18:26")
    assert session.end_time == ts("2024-05-02 18:28")
    assert session.participants == ("张三", "我")
    assert session.text == "\n".join(_lines(
        ("2024-05-02 18:26", "张三", "有空没"),
        ("2024-05-02 18:27", "我", "想去哪？"),
        ("2024-05-02 18:28", "张三", "长沙吧"),
    ))


@pytest.mark.parametrize(
    "second_stamp, expected_sessions",
    [
        ("2024-05-02 19:00", 1),  # 34 min gap -> same session
        ("2024-05-03 08:00", 2),  # next day -> new session
        ("2024-05-02 23:59", 1),  # same day, 5h33m gap (< 6h) -> same session
    ],
)
def test_gap_and_day_boundaries(second_stamp: str, expected_sessions: int) -> None:
    events = parse_txt_events(
        "\n".join(
            _lines(
                ("2024-05-02 18:26", "张三", "有空没"),
                (second_stamp, "我", "看情况"),
            )
        )
    ).events

    assert len(build_sessions(events)) == expected_sessions


def test_gap_larger_than_threshold_starts_new_session() -> None:
    events = parse_txt_events(
        "\n".join(
            _lines(
                ("2024-05-02 08:00", "张三", "早"),
                ("2024-05-02 20:30", "我", "刚回宿舍"),  # 12.5h later, same day
            )
        )
    ).events

    sessions = build_sessions(events, SessionConfig(max_gap=timedelta(hours=6)))

    assert len(sessions) == 2


def test_max_chars_splits_a_long_burst() -> None:
    body = "字" * 60
    events = parse_txt_events(
        "\n".join(
            _lines(
                ("2024-05-02 18:00", "我", body),
                ("2024-05-02 18:01", "我", body),
                ("2024-05-02 18:02", "我", body),
            )
        )
    ).events

    sessions = build_sessions(events, SessionConfig(max_chars=150))

    assert len(sessions) == 3
    assert all(s.n_chars <= 150 for s in sessions)


def test_budget_counts_the_whole_line_not_just_the_message() -> None:
    events = parse_txt_events(
        "\n".join(_lines(("2024-05-02 18:00", "我", "短" * 10)))
    ).events
    line_len = len(events[0].line)

    # one line fits, two do not
    assert len(build_sessions(events, SessionConfig(max_chars=line_len))) == 1
    both = list(events) + list(events)
    assert len(build_sessions(both, SessionConfig(max_chars=line_len))) == 2


def test_empty_input_yields_no_sessions() -> None:
    assert build_sessions(parse_txt_events("").events) == []


def test_session_ids_are_sequential_and_stable() -> None:
    events = parse_txt_events(
        "\n".join(_lines(("2024-05-02 08:00", "我", "早"), ("2024-05-03 08:00", "我", "早")))
    ).events

    sessions = build_sessions(events, conversation_id="stress")

    assert [s.id for s in sessions] == ["stress-session-0001", "stress-session-0002"]
    assert [s.id for s in build_sessions(events, conversation_id="stress")] == [
        "stress-session-0001",
        "stress-session-0002",
    ]


# --------------------------------------------------------------------------- real corpus

pytestmark_corpus = pytest.mark.skipif(
    not STRESS_CORPUS.exists(), reason="stress corpus not present"
)


@pytestmark_corpus
def test_session_chunking_loses_no_source_line_on_the_real_corpus() -> None:
    """Every message line of the real corpus must survive segmentation verbatim."""
    text = STRESS_CORPUS.read_text(encoding="utf-8")
    parsed = parse_txt_events(text, conversation_id="stress")
    sessions = build_sessions(parsed.events, conversation_id="stress")

    assert parsed.skipped_lines == 0, "corpus contains lines the adapter cannot parse"
    assert len(parsed.events) == 1216

    joined = normalize_text("\n".join(s.text for s in sessions))
    missing = [e.line for e in parsed.events if normalize_text(e.line) not in joined]
    assert missing == []


@pytestmark_corpus
def test_every_gold_evidence_line_is_still_present_in_some_session() -> None:
    """Guards the A/B: session chunks must keep all gold evidence findable."""
    text = STRESS_CORPUS.read_text(encoding="utf-8")
    sessions = build_sessions(
        parse_txt_events(text, conversation_id="stress").events, conversation_id="stress"
    )
    joined = normalize_text("\n".join(s.text for s in sessions))

    queries = json.loads(STRESS_QUERIES.read_text(encoding="utf-8"))
    missing = [
        (q["id"], line)
        for q in queries
        for line in (q.get("relevant_evidence") or [])
        if normalize_text(line) not in joined
    ]

    assert missing == []
