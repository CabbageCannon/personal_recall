"""Offline tests for the citation-consistency checker (no model, no network, no gold data).

The regression fixture is the failure reported from the real WeChat smoke test, rewritten with
synthetic sources: the answer quoted「我又点了拌粉」and cited the wrong chunk. The wrong citation must
be flagged and the right one must not — that pair is the whole point of this module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from citation_consistency import (  # noqa: E402
    MIN_QUOTE_CHARS,
    MISMATCH,
    UNVERIFIED,
    check_answer,
    normalize,
    quotes_in,
)

# Two adjacent sessions, exactly the shape that produced the bug: the quoted message is in the FIRST
# chunk and the answer cited the second.
SOURCES = [
    {
        "rank": 1,
        "content": (
            "[2026-09-17 07:27] 我: 早\n"
            "[2026-09-17 11:26] 我: 我又点了拌粉\n"
            "[2026-09-17 11:53] 我: 快到了"
        ),
    },
    {
        "rank": 2,
        "content": "[2026-09-17 11:53] 我: 到楼下了\n[2026-09-17 12:34] 我: 去拿外卖",
    },
]

WRONG = '11:26 你说“我又点了拌粉” [来源 1]'
RIGHT = '11:26 你说“我又点了拌粉” [来源 0]'


# --- the reported failure ---------------------------------------------------------------------


def test_the_reported_wrong_binding_is_flagged() -> None:
    report = check_answer(WRONG, SOURCES)
    assert len(report.mismatches) == 1
    finding = report.mismatches[0]
    assert finding.cited == (1,)
    assert finding.found_in == (0,)
    assert "我又点了拌粉" in finding.describe()
    assert "Source [0]" in finding.describe()


def test_the_correct_binding_is_not_flagged() -> None:
    report = check_answer(RIGHT, SOURCES)
    assert report.ok
    assert report.warnings() == []
    assert report.quotes_checked == 1


def test_the_warning_is_a_caveat_not_a_verdict() -> None:
    """The text must not claim the answer is wrong: it cannot know that."""
    text = " ".join(check_answer(WRONG, SOURCES).warnings()).lower()
    for verdict in ("wrong", "incorrect", "false", "invalid", "fabricat"):
        assert verdict not in text


# --- quote extraction -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        '你说“我又点了拌粉” [来源 0]',
        '你说"我又点了拌粉" [来源 0]',
        "你说「我又点了拌粉」 [来源 0]",
        "你说『我又点了拌粉』 [来源 0]",
    ],
)
def test_all_quote_styles_are_recognised(sentence: str) -> None:
    assert quotes_in(sentence) == ["我又点了拌粉"]


def test_normalize_strips_punctuation_so_a_quote_can_be_located() -> None:
    assert normalize("[2026-09-17 11:26] 我: 我又点了拌粉") == "202609171126我我又点了拌粉"
    assert normalize("我又点了拌粉！") == "我又点了拌粉"


# --- threshold and scope guards ---------------------------------------------------------------


def test_generic_short_quotes_are_ignored() -> None:
    """Calibrated guard: at 4 chars the generic phrase 「另一个同学」 produced a false mismatch."""
    assert len(normalize("另一个同学")) == 5 and MIN_QUOTE_CHARS == 6
    sources = [
        {"rank": 1, "content": "[2026-05-02 10:21] 我: 不是你，是另一个同学，他当时说也是 PG 的"},
        {"rank": 2, "content": "[2026-02-26 18:36] 同学A: 就 2024 年那会儿我说也是 PostgreSQL 的那个"},
    ]
    assert check_answer("可确认这个“另一个同学”是同学A [来源 1]。", sources).ok


def test_a_quote_found_nowhere_is_unverified_and_not_surfaced() -> None:
    """Calibration showed `unverified` is dominated by the model quoting its own paraphrase."""
    report = check_answer('他说“开始使用/试跑”和“正式项目投入使用” [来源 0]。', SOURCES)
    assert report.findings and all(f.verdict == UNVERIFIED for f in report.findings)
    assert report.warnings() == [], "unverified quotes must not become a warning"
    assert report.as_dict()["unverified"] == len(report.findings)


def test_silence_claims_are_skipped() -> None:
    """The evidence for 'the record does not mention X' is an absence, so no quote can support it."""
    report = check_answer('记录里没有提到“那家店的名字” [来源 0]。', SOURCES)
    assert report.skipped_silence_claims == 1
    assert report.sentences_checked == 0
    assert report.ok


def test_sentences_without_citations_are_ignored() -> None:
    report = check_answer('你说“我又点了拌粉”。', SOURCES)
    assert report.sentences_checked == 0 and report.quotes_checked == 0


def test_an_empty_answer_is_handled() -> None:
    report = check_answer("", SOURCES)
    assert report.ok and report.sentences_checked == 0


# --- multi-source behaviour -------------------------------------------------------------------


def test_a_quote_in_any_ONE_cited_source_is_acceptable() -> None:
    """A sentence may cite several chunks; the quote only has to be in one of them."""
    report = check_answer('你说“去拿外卖” [来源 0][来源 1]。', SOURCES)
    assert report.ok


def test_a_quote_in_a_different_source_still_flags_when_several_are_cited() -> None:
    sources = SOURCES + [{"rank": 3, "content": "[2026-09-17 20:00] 我: 晚饭随便吃了点面条"}]
    report = check_answer('你说“晚饭随便吃了点面条” [来源 0][来源 1]。', sources)
    assert len(report.mismatches) == 1
    assert report.mismatches[0].found_in == (2,)


def test_out_of_range_citations_do_not_crash() -> None:
    report = check_answer('你说“我又点了拌粉” [来源 99]。', SOURCES)
    assert isinstance(report.findings, list)


def test_several_quotes_in_one_sentence_are_each_checked() -> None:
    sources = [
        {"rank": 1, "content": "[2026-09-17 11:26] 我: 我又点了拌粉"},
        {"rank": 2, "content": "[2026-09-17 12:34] 我: 到楼下等一会儿"},
    ]
    report = check_answer('你说“我又点了拌粉”，后来又说“到楼下等一会儿” [来源 0]。', sources)
    assert report.quotes_checked == 2
    assert len(report.mismatches) == 1
    assert report.mismatches[0].quote == "到楼下等一会儿"


def test_the_known_cost_of_the_threshold_short_genuine_quotes_are_ignored() -> None:
    """MIN_QUOTE_CHARS = 6 buys precision at the cost of missing short real quotes.

    「去拿外卖」 is a genuine record line and would be a true mismatch if cited against the wrong
    chunk, but at four characters it is indistinguishable from the model's own phrasing, so it is
    skipped. Recorded as a limitation rather than left implicit.
    """
    sources = [
        {"rank": 1, "content": "[2026-09-17 11:26] 我: 我又点了拌粉"},
        {"rank": 2, "content": "[2026-09-17 12:34] 我: 去拿外卖"},
    ]
    report = check_answer('你说“去拿外卖” [来源 0]。', sources)
    assert report.quotes_checked == 0, "short quotes are below the threshold by design"
    assert report.ok, "and therefore produce no finding at all"


# --- determinism and serialisation ------------------------------------------------------------


def test_checking_is_deterministic() -> None:
    first = check_answer(WRONG, SOURCES).as_dict()
    second = check_answer(WRONG, SOURCES).as_dict()
    assert first == second


def test_as_dict_is_json_serialisable() -> None:
    import json

    payload = check_answer(WRONG, SOURCES).as_dict()
    json.dumps(payload)  # must not raise
    assert set(payload) == {
        "sentences_checked",
        "quotes_checked",
        "skipped_silence_claims",
        "mismatches",
        "unverified",
        "findings",
    }
    assert payload["mismatches"] == 1


def test_no_gold_or_external_input_is_required() -> None:
    """The public entry point takes only an answer and its retrieved sources."""
    import inspect

    signature = inspect.signature(check_answer)
    assert list(signature.parameters) == ["answer", "sources"]
