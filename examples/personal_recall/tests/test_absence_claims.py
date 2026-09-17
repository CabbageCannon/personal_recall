"""Offline tests for the absence-claim screen (no model, no network).

The detector is deliberately narrow: it must catch answers that assert the *record* is silent,
and must NOT fire on ordinary negation about the world. Every positive and negative case below
is a real sentence taken from the graded arms, so the tests fail if the patterns drift away
from the language the models actually produce.
"""

from __future__ import annotations

import sys

import pytest

BASE_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from absence_claims import detect_absence_claims  # noqa: E402

# Real sentences that DO assert the record's silence.
POSITIVES = [
    "所以，已明确的结果是：**一家挂了；另一家的最终结果在提供的记录里没有出现**，最后状态是等九月初消息 [来源 1]。",
    "**记录未说明的部分**：中间是否有中断、暂停或恢复，记录里没有说明。",
    "但记录没有说明候补最后是否成功，也没有说明票最终具体怎么出票或改签。",
    "关于“四月室友让我也跟着弄”时你的回复：记录里没有四月当时的直接对话。",
    "记录中没有 2024 年 10 月的直接对话；相关有日期的记录主要在 2024-11-25 和 2024-12-31。",
    "记录里没有说“小王”和“小汪”是同一人 [来源 13]。",
    "提供的记录中没有出现导师最后给“你”定的毕设题目名称。",
    "之后的记录没有提到健身房或锻炼，所以无法确定今天是否还去。",
    "记录里没有显示你成功跑通或正式采用 WireGuard。",
    "之后只说到“剩下的就看运气”，没有记录最终是否候补成功或如何出票。",
    "因此，无法从提供的上下文确定导师最后定的毕设题目叫什么名字。",
    "现有记录没有说“小汪”就是“王哥/小王”，且称呼也不同[来源 0]。",
]

# Real sentences that are negations about events, or plain statements -- NOT record-silence claims.
PLAIN_NEGATIONS = [
    "所以，你当时没有答应，最终也没有给他一个明确准话，深圳这趟没有去成。",
    "所以结果就是：**没弄成，没有换成阿伟那套，远程访问仍用最早装的那个**。",
    "结论：没有试成，也没有换成，仍用最早装的远程访问方案。",
    "**结论：**从记录看，我最后明确提到锻炼状态是 **2026-02-14**，当时不去健身房，锻炼以跑操场为主。",
    "小汪提到要请客的地方只有这些，且都未给出店名。",
    "那我不去了，卡都快过期了，等开学再说。",
    "我最近还在跑操场，先这样吧。",
]

# A dated record line IS a claim (kept separate so the parametrisation cannot contradict it).
DATED_RECORD_CLAIM = "2026-05-01、2026-08-28：这两天的记录没有提到锻炼或健身房 [来源 6][来源 7]。"


def test_real_record_silence_sentences_are_detected() -> None:
    missed = [s for s in POSITIVES if not detect_absence_claims(s)]
    assert not missed, f"detector missed real absence claims: {missed}"


@pytest.mark.parametrize("sentence", PLAIN_NEGATIONS)
def test_plain_event_negation_is_not_a_record_claim(sentence: str) -> None:
    """'没有答应' / '没有试成' are claims about events, not about the record's contents."""
    assert not detect_absence_claims(sentence), f"false positive on: {sentence}"


def test_dated_record_line_is_a_claim() -> None:
    """'这两天的记录没有提到…' IS a record-silence claim and must be caught."""
    assert detect_absence_claims(DATED_RECORD_CLAIM)


def test_claims_are_deduplicated_per_sentence() -> None:
    answer = "记录没有说明A。记录没有说明A。记录没有说明B。"
    assert len(detect_absence_claims(answer)) == 2


def test_empty_and_clean_answers_yield_nothing() -> None:
    assert detect_absence_claims("") == []
    assert detect_absence_claims("你们组是第三个上去讲的。[来源 0]") == []


def test_claim_carries_its_sentence_for_adjudication() -> None:
    claim = detect_absence_claims(POSITIVES[4])[0]
    assert "2024 年 10 月" in claim.sentence
    assert claim.matched


def _screen_fixture(tmp_path, corpus_text: str):
    """One query whose gold line was NOT retrieved, screened against a given corpus.

    Uses a workspace-local scratch dir rather than pytest's `tmp_path`, which cannot be
    created in this environment (PermissionError on the system temp dir).
    """
    import json
    import shutil

    from absence_claims import screen

    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        gold = "[2024-10-20 15:14] 小王: 可以啊，你们组第几个上去讲的"
        queries = tmp_path / "q.json"
        queries.write_text(
            json.dumps(
                [
                    {
                        "id": "s001",
                        "category": "exact_fact",
                        "answerable": True,
                        "question": "第几个上去讲的？",
                        "relevant_evidence": [gold],
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        results = tmp_path / "r.json"
        results.write_text(
            json.dumps(
                [
                    {
                        "query_id": "s001",
                        "answer": "记录中没有出现答辩顺序的信息。",
                        "evidence_coverage": 0.0,
                        "retrieved_sources": [
                            {"rank": 1, "content": "[2025-01-01 08:00] 我: 无关内容"}
                        ],
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        corpus = tmp_path / "c.txt"
        corpus.write_text(corpus_text, encoding="utf-8")
        return screen(results, queries, corpus)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
        try:  # drop the shared parent too, so the test tree is left clean
            tmp_path.parent.rmdir()
        except OSError:
            pass


SCRATCH = BASE_DIR / "tests" / "_scratch_absence"


def test_risk_requires_the_missing_gold_to_still_be_in_the_corpus() -> None:
    """The defect is only possible if the record still contains it."""
    gold = "[2024-10-20 15:14] 小王: 可以啊，你们组第几个上去讲的"
    with_gold = _screen_fixture(SCRATCH / "a", f"[2024-10-20 15:10] 我: 完事了\n{gold}\n")
    assert with_gold["risk_candidates"] == ["s001"]

    without_gold = _screen_fixture(SCRATCH / "b", "[2024-10-20 15:10] 我: 完事了\n")
    assert without_gold["risk_candidates"] == [], (
        "an ablated corpus must not be flagged: the absence claim is CORRECT there"
    )


def test_corpus_check_is_recorded() -> None:
    report = _screen_fixture(SCRATCH / "c", "irrelevant")
    assert report["corpus_checked"] is True
    assert report["corpus"] is not None
