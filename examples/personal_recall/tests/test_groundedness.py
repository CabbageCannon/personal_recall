"""Offline tests for the gold-free groundedness report (no model, no network).

The report is what a *user* sees, so the tests pin the two properties that make it honest: it must
surface the caveats, and it must never claim to know an answer is wrong — without gold evidence it
cannot.
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from groundedness import assess  # noqa: E402

SOURCES = [
    {
        "rank": 1,
        "content": (
            "[2026-06-10 22:34] 同学A: 那个存储的方案我看了两遍，最后还是没换\n"
            "[2026-06-10 22:39] 我: 现有的够用就别动\n"
        ),
    },
    {"rank": 2, "content": "[2026-05-02 10:26] 我: 本地用文件型的那个\n"},
]


def test_clean_answer_reports_no_caveats() -> None:
    report = assess("你本地用的是文件型的那个 [来源 1]。", SOURCES)
    assert report.citations == (1,)
    assert not report.has_caveats
    assert report.warnings() == []
    assert "no silence claims" in report.summary_line()
    assert "no attribution flags" in report.summary_line()


def test_out_of_range_citation_is_counted_not_silently_dropped() -> None:
    report = assess("你本地用的是文件型的那个 [来源 9]。", SOURCES)
    assert report.invalid_citations == 1
    assert report.citations == ()
    assert any("outside the retrieved sources" in w for w in report.warnings())


def test_uncited_answer_is_flagged() -> None:
    report = assess("你本地用的是文件型的那个。", SOURCES)
    assert report.uncited
    assert any("cites no source" in w for w in report.warnings())


def test_record_silence_claim_is_surfaced_as_a_caveat() -> None:
    report = assess("记录里没有提到你换过存储 [来源 1]。", SOURCES)
    assert report.absence_claims
    assert any("does NOT contain" in w for w in report.warnings())


def test_borrowed_own_situation_is_flagged() -> None:
    """The s032 shape, now visible in the product surface."""
    report = assess("2026-06-10 的记录提到那个存储方案看了两遍，最后还是没换 [来源 0]。", SOURCES)
    assert report.attribution_flags
    assert report.attribution_flags[0]["matched_line_speaker"] == "同学A"
    assert any("another person's own account" in w for w in report.warnings())


def test_report_never_asserts_the_answer_is_wrong() -> None:
    """Without gold it cannot know: the wording must stay a caveat, not a verdict."""
    report = assess("2026-06-10 的记录提到那个存储方案看了两遍 [来源 0]。", SOURCES)
    text = " ".join(report.warnings()).lower()
    for verdict in ("wrong", "incorrect", "false", "fabricat"):
        assert verdict not in text, f"report claims a verdict it cannot support: {verdict}"


def test_as_dict_is_json_serialisable_and_complete() -> None:
    import json

    report = assess("记录里没有提到你换过存储 [来源 0]。", SOURCES)
    payload = report.as_dict()
    json.dumps(payload)  # must not raise
    assert set(payload) == {
        "n_sources",
        "citations",
        "invalid_citations",
        "uncited",
        "absence_claims",
        "attribution_flags",
        "warnings",
    }
    assert payload["n_sources"] == 2


def test_empty_answer_is_handled() -> None:
    report = assess("", SOURCES)
    assert report.uncited
    assert report.citations == ()
