"""Offline tests for the real-data acceptance harness (no model, no network).

The harness spends API calls, so everything that can be wrong *before* the first call is tested here:
question-set validation, the result shape a human has to label, label preservation across re-runs, and
the per-category summary. The template shipped in the repo must be **refused**, not run.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from real_eval import (  # noqa: E402
    CATEGORIES,
    LABEL_FIELDS,
    PLACEHOLDER_MARKER,
    Question,
    QuestionSetError,
    build_record,
    load_questions,
    render_report,
    summarize,
)

TEMPLATE = BASE_DIR / "eval_questions.template.json"

GROUNDEDNESS = {
    "citations": [0, 1],
    "invalid_citations": 0,
    "absence_claims": [],
    "attribution_flags": [],
    "citation_mismatches": [],
    "warnings": [],
    "uncited": False,
    "n_sources": 3,
}

SOURCES = [
    {"rank": 1, "chunk_index": 3, "memory_chunk_id": "weflow-session-0003", "content": "[2026-09-17 11:26] 我: 我又点了拌粉"},
    {"rank": 2, "chunk_index": 4, "memory_chunk_id": "weflow-session-0004", "content": "[2026-09-17 12:34] 我: 去拿外卖"},
]


def _scratch(name: str) -> Path:
    path = BASE_DIR / "tests" / "_scratch_real_eval" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_questions(directory: Path, questions: list[dict]) -> Path:
    path = directory / "questions.json"
    path.write_text(json.dumps({"questions": questions}, ensure_ascii=False), encoding="utf-8")
    return path


# --- question-set validation -------------------------------------------------------------------


def test_the_six_brief_categories_are_the_supported_set() -> None:
    assert CATEGORIES == (
        "single_fact_recall",
        "timeline_reasoning",
        "speaker_attribution",
        "multi_source_synthesis",
        "abstention",
        "older_memory",
    )


def test_the_shipped_template_is_refused_not_run() -> None:
    """A template that still holds placeholders must fail before spending an API call."""
    with pytest.raises(QuestionSetError) as excinfo:
        load_questions(TEMPLATE)
    message = str(excinfo.value)
    assert PLACEHOLDER_MARKER in message
    assert "still contains the template placeholder" in message


def test_the_template_covers_all_six_categories() -> None:
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    categories = [entry["category"] for entry in payload["questions"]]
    assert set(categories) == set(CATEGORIES)
    assert 15 <= len(payload["questions"]) <= 20, "the brief asks for 15-20 questions"
    for category in CATEGORIES:
        assert categories.count(category) >= 3, f"{category} needs at least three questions"


def test_a_filled_question_set_loads() -> None:
    directory = _scratch("valid")
    try:
        path = _write_questions(
            directory,
            [
                {"id": "q01", "category": "single_fact_recall", "question": "我最后用了哪个库？"},
                {"id": "q02", "category": "abstention", "question": "那家店叫什么名字？", "note": "记录里没有"},
            ],
        )
        questions = load_questions(path)
        assert [q.id for q in questions] == ["q01", "q02"]
        assert questions[1].note == "记录里没有"
    finally:
        shutil.rmtree(directory.parent, ignore_errors=True)


@pytest.mark.parametrize(
    "entry,fragment",
    [
        ({"id": "q01", "category": "nonsense", "question": "我最后用了哪个库？"}, "is not one of"),
        ({"id": "q01", "category": "single_fact_recall", "question": "短"}, "too short"),
        ({"id": "q01", "category": "single_fact_recall", "question": "«未填写»"}, "placeholder"),
    ],
)
def test_bad_entries_are_rejected_with_a_reason(entry: dict, fragment: str) -> None:
    directory = _scratch("bad")
    try:
        path = _write_questions(directory, [entry])
        with pytest.raises(QuestionSetError) as excinfo:
            load_questions(path)
        assert fragment in str(excinfo.value)
    finally:
        shutil.rmtree(directory.parent, ignore_errors=True)


def test_duplicate_ids_are_rejected() -> None:
    directory = _scratch("dupes")
    try:
        path = _write_questions(
            directory,
            [
                {"id": "q01", "category": "single_fact_recall", "question": "我最后用了哪个库？"},
                {"id": "q01", "category": "older_memory", "question": "去年五一去哪了？"},
            ],
        )
        with pytest.raises(QuestionSetError, match="duplicate id"):
            load_questions(path)
    finally:
        shutil.rmtree(directory.parent, ignore_errors=True)


def test_an_empty_question_file_is_rejected() -> None:
    directory = _scratch("empty")
    try:
        path = directory / "questions.json"
        path.write_text(json.dumps({"questions": []}), encoding="utf-8")
        with pytest.raises(QuestionSetError, match="non-empty"):
            load_questions(path)
    finally:
        shutil.rmtree(directory.parent, ignore_errors=True)


# --- the recorded row ------------------------------------------------------------------------


def _question() -> Question:
    return Question(id="q01", category="single_fact_recall", question="我今天中午吃了什么？", note="拌粉")


def test_a_record_carries_everything_a_human_needs_to_judge_it() -> None:
    record = build_record(_question(), "今天中午吃了拌粉 [来源 0]", SOURCES, 1234.5, GROUNDEDNESS)
    for field in (
        "id",
        "category",
        "question",
        "note",
        "answer",
        "retrieved",
        "citations",
        "invalid_citations",
        "groundedness",
        "citation_mismatches",
        "attribution_flags",
        "latency_ms",
        "labels",
    ):
        assert field in record, f"missing {field}"
    assert record["n_retrieved"] == 2
    assert record["retrieved"][0]["memory_chunk_id"] == "weflow-session-0003"
    assert record["latency_ms"] == 1234.5


def test_all_four_labels_start_empty_for_the_human() -> None:
    record = build_record(_question(), "答案", SOURCES, 1.0, GROUNDEDNESS)
    assert record["labels"] == {field: None for field in LABEL_FIELDS}


def test_existing_labels_survive_a_re_run() -> None:
    """Labelling is work; another run must never discard it."""
    previous = {"labels": {"answer_correct": True, "retrieval_correct": False}}
    record = build_record(_question(), "答案", SOURCES, 1.0, GROUNDEDNESS, previous=previous)
    assert record["labels"]["answer_correct"] is True
    assert record["labels"]["retrieval_correct"] is False
    assert record["labels"]["citation_binding_correct"] is None


def test_a_record_is_json_serialisable() -> None:
    json.dumps(build_record(_question(), "答案 [来源 0]", SOURCES, 1.0, GROUNDEDNESS), ensure_ascii=False)


def test_no_gold_or_expected_answer_field_is_required() -> None:
    """The harness must work without an expected answer — the human supplies judgement later."""
    import inspect

    assert "expected" not in " ".join(inspect.signature(build_record).parameters)


# --- summary ---------------------------------------------------------------------------------


def _records() -> list[dict]:
    base = build_record(_question(), "a", SOURCES, 1000.0, GROUNDEDNESS)
    other = build_record(
        Question(id="q02", category="abstention", question="那家店叫什么？"),
        "记录里没有提到店名",
        SOURCES,
        2000.0,
        {**GROUNDEDNESS, "absence_claims": ["记录里没有提到店名"]},
    )
    return [base, other]


def test_summarize_counts_by_category_and_leaves_labels_unlabelled() -> None:
    summary = summarize(_records())
    assert summary["questions"] == 2
    assert summary["unlabelled_questions"] == 2
    assert summary["by_category"]["single_fact_recall"]["questions"] == 1
    assert summary["by_category"]["abstention"]["questions"] == 1
    assert summary["by_category"]["abstention"]["with_warnings"] == 1
    assert summary["labels"]["answer_correct"]["unlabelled"] == 2


def test_summarize_scores_filled_labels() -> None:
    records = _records()
    records[0]["labels"]["answer_correct"] = True
    records[1]["labels"]["answer_correct"] = False
    summary = summarize(records)
    assert summary["labels"]["answer_correct"] == {"true": 1, "false": 1, "unlabelled": 0}
    assert summary["unlabelled_questions"] == 0


def test_summarize_reports_mean_latency_per_category() -> None:
    summary = summarize(_records())
    assert summary["by_category"]["single_fact_recall"]["mean_latency_ms"] == 1000.0
    assert summary["by_category"]["abstention"]["mean_latency_ms"] == 2000.0


def test_render_report_lists_every_category() -> None:
    rendered = render_report(summarize(_records()))
    for category in CATEGORIES:
        assert category in rendered
    for field in LABEL_FIELDS:
        assert field in rendered


def test_summarize_tolerates_an_empty_run() -> None:
    summary = summarize([])
    assert summary["questions"] == 0
    assert render_report(summary)
