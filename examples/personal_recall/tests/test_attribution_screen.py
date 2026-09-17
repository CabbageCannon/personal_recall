"""Offline tests for the attribution screen (no model, no network).

Two of these tests exist because the screen silently failed without them:

* a sentence written ``...。[来源 5]`` was split at ``。``, orphaning the citation into a fragment
  that matched nothing — which is how the screen missed the `s020` defect on its first run;
* answers attribute with the pronoun 他/她 as often as with a name ("他的系统还没部署"), and flagging
  those was the screen's largest false-positive source.

Both are pinned below with the real sentence shapes.
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from attribution_screen import (  # noqa: E402
    SourceLine,
    best_match,
    bigrams,
    parse_lines,
    screen_answer,
    split_sentences,
)

CHUNK = (
    "[2026-06-10 22:34] 同学A: 那个存储的方案我看了两遍，最后还是没换\n"
    "[2026-06-10 22:39] 我: 现有的够用就别动\n"
)
SOURCES = [{"rank": 1, "content": CHUNK}]


def test_line_classification_separates_user_from_others() -> None:
    assert SourceLine(speaker="我", body="我换回去了").kind == "user"
    assert SourceLine(speaker="小王", body="你那个库用哪个").kind == "user_directed"
    assert SourceLine(speaker="同学A", body="我看了两遍").kind == "own_situation"
    assert SourceLine(speaker="小王", body="行吧").kind == "neutral"


def test_bigrams_handle_chinese_and_ignore_citation_markup() -> None:
    assert bigrams("那个存储") == {"那个", "个存", "存储"}
    assert bigrams("[来源 5]") == set()
    assert bigrams("abc") == {"ab", "bc"}


def test_parse_lines_reads_speaker_and_body() -> None:
    lines = parse_lines(CHUNK)
    assert [line.speaker for line in lines] == ["同学A", "我"]
    assert lines[0].body.startswith("那个存储的方案")


def test_best_match_finds_the_supporting_line() -> None:
    line, score = best_match("那个存储方案看了两遍，最后还是没换", parse_lines(CHUNK))
    assert line is not None and line.speaker == "同学A"
    assert score >= 0.5


def test_flags_borrowed_own_situation_line() -> None:
    """The s032 shape: the answer reports 同学A's own PC as the user's storage."""
    answer = "2026-06-10 的记录里提到，那个存储方案看了两遍，最后还是没换 [来源 0]。"
    findings = screen_answer(answer, SOURCES)
    assert len(findings) == 1
    assert findings[0]["matched_line_speaker"] == "同学A"


def test_citation_after_a_full_stop_stays_attached_to_its_sentence() -> None:
    """Regression: splitting on 。 orphaned [来源 0] and defeated the whole screen."""
    answer = "那个存储方案我看了两遍，最后还是没换。[来源 0]"
    sentences = split_sentences(answer)
    assert len(sentences) == 1, f"citation was orphaned: {sentences}"
    assert screen_answer(answer, SOURCES), "the citation orphan made this defect invisible"


def test_naming_the_speaker_is_not_a_defect() -> None:
    assert not screen_answer("同学A 说那个存储方案他看了两遍，最后还是没换 [来源 0]。", SOURCES)


def test_pronoun_attribution_is_not_a_defect() -> None:
    """Regression: 'he said...' is correct attribution and used to be flagged."""
    assert not screen_answer("他说那个存储方案看了两遍，最后还是没换 [来源 0]。", SOURCES)


def test_pronoun_in_the_previous_sentence_carries_attribution() -> None:
    """Bullets routinely establish who 'he' is on the preceding line."""
    answer = (
        "- 同学A 提到他自己那台机器：\n"
        "- 那个存储方案看了两遍，最后还是没换 [来源 0]。\n"
    )
    assert not screen_answer(answer, SOURCES)


def test_sentences_without_citations_are_ignored() -> None:
    assert not screen_answer("那个存储方案我看了两遍，最后还是没换。", SOURCES)


def test_user_lines_are_never_flagged() -> None:
    """A claim resting on the user's own line is exactly what should be safe."""
    answer = "你说现有的够用就别动 [来源 0]。"
    assert not screen_answer(answer, SOURCES)
