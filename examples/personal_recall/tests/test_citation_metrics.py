"""Offline tests for the citation parser and metrics."""

from __future__ import annotations

from citation_metrics import evaluate, parse_citations

QUERIES = [
    {
        "id": "q1",
        "answerable": True,
        "relevant_evidence": ["[2024-05-02 18:28] 张三: 长沙吧"],
    },
    {
        "id": "q2",
        "answerable": True,
        "relevant_evidence": ["[2025-01-14 23:20] 张三: 王哥前两天不是还说要去深圳吗"],
    },
    {"id": "q3", "answerable": False, "relevant_evidence": []},
]


def _result(qid: str, answer: str, contents: list[str]) -> dict:
    return {
        "query_id": qid,
        "answer": answer,
        "retrieved_sources": [
            {"rank": i, "chunk_index": i, "content": c} for i, c in enumerate(contents, start=1)
        ],
    }


CHUNKS_Q1 = ["[2024-05-02 18:28] 张三: 长沙吧", "[2024-01-01 09:00] 我: 无关内容"]
CHUNKS_Q2 = ["[2024-01-01 09:00] 我: 无关内容", "[2025-01-14 23:20] 张三: 王哥前两天不是还说要去深圳吗"]


# --------------------------------------------------------------------------- parser


def test_parses_bracket_form_with_and_without_spaces() -> None:
    assert parse_citations("结论 [来源 0]，另一处 [来源2]。", 3)[0] == [0, 2]


def test_parses_bare_form_and_source_keyword() -> None:
    assert parse_citations("来源 2 提到…", 3)[0] == [2]
    assert parse_citations("as [Source 2] says", 3)[0] == [2]


def test_citation_numbering_is_zero_based_like_the_framework_source_labels() -> None:
    # combine_documents renders "Source: {index}" with index = range(len(docs))
    valid, invalid = parse_citations("[来源 0] [来源 2]", 3)

    assert valid == [0, 2]
    assert invalid == 0


def test_out_of_range_citations_are_counted_invalid_not_dropped_silently() -> None:
    valid, invalid = parse_citations("[来源 0] [来源 9] [来源 3]", 3)

    assert valid == [0]
    assert invalid == 2


def test_no_citations_is_empty() -> None:
    assert parse_citations("没有任何引用", 3) == ([], 0)


# --------------------------------------------------------------------------- metrics


def test_full_citation_behaviour_scores_perfectly() -> None:
    metrics = evaluate(
        QUERIES,
        [
            _result("q1", "张三提议去长沙 [来源 0]。", CHUNKS_Q1),
            _result("q2", "张三提到深圳 [来源 1]。", CHUNKS_Q2),
            _result("q3", "记录里没有提到这家店。", ["[2024-01-01 09:00] 我: 无关"]),
        ],
    )

    # 2 of 3 answers cite: the unanswerable one must NOT cite (it asserts nothing)
    assert metrics["citation_rate"] == 2 / 3
    assert metrics["citation_coverage"] == 1.0
    assert metrics["citation_precision_lexical"] == 1.0
    assert metrics["abstention_accuracy"] == 1.0
    assert metrics["invalid_citations"] == 0


def test_uncited_answers_have_zero_citation_coverage_but_full_retrieval_coverage() -> None:
    metrics = evaluate(
        QUERIES,
        [
            _result("q1", "张三提议去长沙。", CHUNKS_Q1),
            _result("q2", "张三提到深圳。", CHUNKS_Q2),
        ],
    )

    assert metrics["citation_rate"] == 0.0
    assert metrics["citation_coverage"] == 0.0
    assert metrics["retrieval_coverage"] == 1.0, "the evidence WAS retrieved, only not cited"


def test_citing_a_chunk_without_gold_evidence_lowers_precision() -> None:
    metrics = evaluate(
        QUERIES,
        [_result("q2", "结论 [来源 0]。", CHUNKS_Q2)],  # chunk 0 has no gold line
    )

    assert metrics["citation_coverage"] == 0.0
    assert metrics["citation_precision_lexical"] == 0.0
    assert metrics["total_citations"] == 1


def test_unanswerable_query_that_asserts_a_fact_fails_abstention() -> None:
    metrics = evaluate(
        QUERIES,
        [_result("q3", "那家店叫老王家烤肉。[来源 0]", ["[2024-01-01 09:00] 我: 无关"])],
    )

    assert metrics["abstention_accuracy"] == 0.0, "an assertion with a citation is not an abstention"
