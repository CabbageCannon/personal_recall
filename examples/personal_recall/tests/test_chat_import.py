"""Offline tests for the chat-export importer (no model, no network).

The load-bearing property is the **round trip**: whatever the importer emits must be ingested by the
engine's own adapter with nothing skipped and nothing lost. Every layout test asserts it, because an
importer that produces something the engine cannot read is worse than no importer at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from chat_import import PROFILES, detect_profile, import_text  # noqa: E402
from memory.events import parse_txt_events  # noqa: E402

CANONICAL = (
    "[2024-05-01 11:30] 我: 你那个库最后用的哪个\n"
    "[2024-05-01 11:31] 小王: 还是原来那个\n"
)
TELEGRAM = (
    "[01.05.24 11:30] 我: 你那个库最后用的哪个\n"
    "[01.05.24 11:31] 小王: 还是原来那个\n"
)
ISO = (
    "2024-05-01 11:30:00 我: 你那个库最后用的哪个\n"
    "2024-05-01 11:31:00 小王: 还是原来那个\n"
)
WECHAT = (
    "我  2024-05-01 11:30\n"
    "你那个库最后用的哪个\n"
    "小王  2024-05-01 11:31\n"
    "还是原来那个\n"
)


def _round_trip_ok(output: str, expected_messages: int) -> None:
    parsed = parse_txt_events(output)
    assert len(parsed.events) == expected_messages
    assert parsed.skipped_lines == 0


@pytest.mark.parametrize(
    ("text", "layout"),
    [
        (CANONICAL, "canonical"),
        (TELEGRAM, "bracket_dmy"),
        (ISO, "iso_seconds"),
        (WECHAT, "speaker_first"),
    ],
)
def test_each_layout_is_detected_and_converts_with_a_clean_round_trip(text: str, layout: str) -> None:
    output, report = import_text(text, source="fixture")
    assert report.profile == layout
    assert report.messages == 2
    assert report.verified is True
    assert report.ok
    _round_trip_ok(output, 2)


def test_the_same_conversation_converts_identically_across_layouts() -> None:
    """Different export formats of one conversation must produce the same canonical corpus."""
    outputs = {import_text(text, source=name)[0] for name, text in (
        ("canonical", CANONICAL),
        ("telegram", TELEGRAM),
        ("iso", ISO),
        ("wechat", WECHAT),
    )}
    assert len(outputs) == 1, f"layouts disagree: {outputs}"


def test_speaker_first_joins_a_multi_line_body() -> None:
    text = "我  2024-05-01 11:30\n第一行\n第二行\n小王  2024-05-01 11:31\n好的\n"
    output, report = import_text(text, source="fixture")
    assert report.messages == 2
    assert "[2024-05-01 11:30] 我: 第一行 第二行" in output
    _round_trip_ok(output, 2)


def test_utf8_bom_does_not_swallow_the_first_message() -> None:
    """Regression: a BOM stuck to line 1 and silently cost one message."""
    output, report = import_text("\ufeff" + TELEGRAM, source="fixture")
    assert report.messages == 2, "the BOM ate a message"
    assert report.skipped_lines == 0
    _round_trip_ok(output, 2)


def test_unrecognised_layout_fails_loudly_with_a_diagnosis() -> None:
    """The whole point: never fail silently, and say what was tried."""
    output, report = import_text("2024年5月1日 11:30\n我: 你那个库最后用的哪个\n", source="fixture")
    assert output == ""
    assert not report.ok
    assert report.profile is None
    joined = " ".join(report.notes)
    assert "recognised but not supported yet" in joined
    assert "line-match counts by layout" in joined
    assert "expected one of" in joined


def test_empty_input_is_not_ok() -> None:
    output, report = import_text("", source="fixture")
    assert output == "" and not report.ok


def test_a_colon_in_the_source_name_splits_there_without_corrupting_the_format() -> None:
    """Every profile's speaker group is ``[^:]{1,24}``, so a colon simply ends the name.

    The result is a misparse of an ambiguous export, not corruption: the emitted line is still
    ingestible, which is the property that matters.
    """
    text = "[2024-05-01 11:30] 小:王: 内容\n"
    output, report = import_text(text, source="fixture")
    assert report.messages == 1
    assert output.startswith("[2024-05-01 11:30] 小: ")
    _round_trip_ok(output, 1)


def test_clean_speaker_neutralises_colons_and_truncates() -> None:
    """Defensive: unreachable through the current profiles, but the canonical format depends on it."""
    from chat_import import MAX_SPEAKER_CHARS, _clean_speaker

    notes: list[str] = []
    assert _clean_speaker("小:王", notes) == "小：王"
    assert any("replaced ':'" in note for note in notes)

    notes = []
    long_name = "名" * (MAX_SPEAKER_CHARS + 5)
    assert len(_clean_speaker(long_name, notes)) == MAX_SPEAKER_CHARS
    assert any("truncated" in note for note in notes)


def test_every_profile_declares_a_description_and_formats() -> None:
    for profile in PROFILES:
        assert profile.description and profile.timestamp_formats
        assert profile.layout in {"inline", "header"}


def test_header_layout_at_end_of_file_is_reported_not_dropped_silently() -> None:
    output, report = import_text("我  2024-05-01 11:30\n", source="fixture")
    assert report.messages == 0
    assert any("no message body" in note for note in report.notes)


def test_detect_profile_reports_counts_for_every_layout() -> None:
    profile, counts = detect_profile(WECHAT)
    assert profile is not None and profile.name == "speaker_first"
    assert set(counts) == {p.name for p in PROFILES}
