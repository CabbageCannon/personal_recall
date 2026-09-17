"""Real-WeChat acceptance eval: a small, honest end-to-end harness over your own chat history.

Deliberately small. The stress corpus answered *"how good is the engine under controlled conditions"*;
this answers *"does it work on my actual history, on questions I actually care about"*. It is a
acceptance check, not a benchmark: 15–20 questions spread over six categories, every field a human
needs to judge an answer recorded next to it.

What it records per question — no gold, no judge, no scoring model:

* ``answer`` — what the engine said;
* ``retrieved`` — every retrieved chunk with its time range, participants and text, so retrieval can
  be judged independently of the answer;
* ``citations`` / ``invalid_citations`` — the citation indices and whether any is out of range;
* ``groundedness`` — the product's caveats, including **citation binding mismatches** and attribution
  flags (Phase 18C / 18A);
* ``latency_ms``.

Four labels are left ``null`` for a human to fill in place: ``answer_correct``, ``retrieval_correct``,
``citation_binding_correct``, ``attribution_correct``. Re-running with the same questions preserves
existing labels, so labelling is never lost by another run.

The six categories come from the project brief:

1. ``single_fact_recall`` — one fact, one place;
2. ``timeline_reasoning`` — several messages, ordered;
3. ``speaker_attribution`` — who said or did it;
4. ``multi_source_synthesis`` — needs two or more chunks joined;
5. ``abstention`` — the record does **not** contain the answer; the engine must decline;
6. ``older_memory`` — the fact is far back in the history.

Usage::

    python real_eval.py --questions eval_questions.json --corpus shards/ --out real_eval_results.json
    python real_eval.py --questions eval_questions.json --corpus shards/ --report-only
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

#: The six categories from the brief, in order. A question must name one of these.
CATEGORIES: tuple[str, ...] = (
    "single_fact_recall",
    "timeline_reasoning",
    "speaker_attribution",
    "multi_source_synthesis",
    "abstention",
    "older_memory",
)

#: The four judgements a human fills in. All start null.
LABEL_FIELDS: tuple[str, ...] = (
    "answer_correct",
    "retrieval_correct",
    "citation_binding_correct",
    "attribution_correct",
)

#: Marker used by the template; a question still containing it has not been filled in.
PLACEHOLDER_MARKER = "«"

#: A question shorter than this is almost certainly a placeholder.
MIN_QUESTION_CHARS = 6


class QuestionSetError(ValueError):
    """The question file is unusable — raised loudly rather than run partially."""


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    note: str = ""


def load_questions(path: Path) -> list[Question]:
    """Read and validate a question set.

    Validation is strict on purpose: a mistyped category or an unfilled template should fail before
    spending a single API call, not produce a results file that silently covers five categories.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    raw = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or not raw:
        raise QuestionSetError(f"{path}: expected a non-empty list of questions")

    questions: list[Question] = []
    seen: set[str] = set()
    problems: list[str] = []
    for index, entry in enumerate(raw, start=1):
        if not isinstance(entry, Mapping):
            problems.append(f"entry {index} is not an object")
            continue
        qid = str(entry.get("id") or f"q{index:02d}")
        category = str(entry.get("category") or "")
        text = str(entry.get("question") or "").strip()

        if qid in seen:
            problems.append(f"{qid}: duplicate id")
        seen.add(qid)
        if category not in CATEGORIES:
            problems.append(f"{qid}: category {category!r} is not one of {list(CATEGORIES)}")
        if len(text) < MIN_QUESTION_CHARS:
            problems.append(f"{qid}: question is too short to be real")
        if PLACEHOLDER_MARKER in text or PLACEHOLDER_MARKER in str(entry.get("note") or ""):
            problems.append(f"{qid}: still contains the template placeholder {PLACEHOLDER_MARKER!r}")
        questions.append(Question(id=qid, category=category, question=text, note=str(entry.get("note") or "")))

    if problems:
        raise QuestionSetError(f"{path} has {len(problems)} problem(s):\n  " + "\n  ".join(problems))
    return questions


def build_record(
    question: Question,
    answer: str,
    serialized_sources: Sequence[Mapping[str, Any]],
    latency_ms: float,
    groundedness: Mapping[str, Any],
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One result row. Pure, so it can be tested without a model.

    Existing human labels are carried over from ``previous`` — re-running must never silently discard
    labelling work.
    """
    citations = list(groundedness.get("citations") or [])
    labels = {field: None for field in LABEL_FIELDS}
    if previous:
        for field in LABEL_FIELDS:
            if isinstance(previous.get("labels"), Mapping) and field in previous["labels"]:
                labels[field] = previous["labels"][field]

    return {
        "id": question.id,
        "category": question.category,
        "question": question.question,
        "note": question.note,
        "answer": answer,
        "retrieved": [dict(source) for source in serialized_sources],
        "n_retrieved": len(serialized_sources),
        "citations": citations,
        "invalid_citations": int(groundedness.get("invalid_citations") or 0),
        "groundedness": dict(groundedness),
        "citation_mismatches": list(groundedness.get("citation_mismatches") or []),
        "attribution_flags": list(groundedness.get("attribution_flags") or []),
        "latency_ms": round(float(latency_ms), 2),
        "labels": labels,
    }


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-category counts plus whatever human labels exist so far."""
    by_category: dict[str, dict[str, Any]] = {
        category: {"questions": 0, "labelled": 0, "with_warnings": 0, "mean_latency_ms": 0.0}
        for category in CATEGORIES
    }
    latencies: dict[str, list[float]] = {category: [] for category in CATEGORIES}
    label_totals = {field: {"true": 0, "false": 0, "unlabelled": 0} for field in LABEL_FIELDS}
    unlabelled = 0

    for record in records:
        category = str(record.get("category"))
        bucket = by_category.setdefault(
            category, {"questions": 0, "labelled": 0, "with_warnings": 0, "mean_latency_ms": 0.0}
        )
        bucket["questions"] += 1
        latencies.setdefault(category, []).append(float(record.get("latency_ms") or 0.0))

        groundedness = record.get("groundedness") or {}
        if (
            record.get("citation_mismatches")
            or record.get("attribution_flags")
            or groundedness.get("absence_claims")
            or record.get("invalid_citations")
        ):
            bucket["with_warnings"] += 1

        labels = record.get("labels") or {}
        any_label = False
        for field in LABEL_FIELDS:
            value = labels.get(field)
            if value is True:
                label_totals[field]["true"] += 1
                any_label = True
            elif value is False:
                label_totals[field]["false"] += 1
                any_label = True
            else:
                label_totals[field]["unlabelled"] += 1
        if any_label:
            bucket["labelled"] += 1
        else:
            unlabelled += 1

    for category, values in latencies.items():
        if values:
            by_category[category]["mean_latency_ms"] = round(sum(values) / len(values), 1)

    return {
        "questions": len(records),
        "unlabelled_questions": unlabelled,
        "by_category": by_category,
        "labels": label_totals,
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        f"questions        : {summary['questions']}",
        f"awaiting labels  : {summary['unlabelled_questions']}",
        "",
        f"{'category':26} {'n':>3} {'labelled':>9} {'caveats':>8} {'mean ms':>9}",
    ]
    for category in CATEGORIES:
        bucket = summary["by_category"].get(category, {})
        lines.append(
            f"{category:26} {bucket.get('questions', 0):3} {bucket.get('labelled', 0):9} "
            f"{bucket.get('with_warnings', 0):8} {bucket.get('mean_latency_ms', 0.0):9}"
        )
    lines.append("")
    lines.append("human labels (fill in the result file's `labels` objects):")
    for field, counts in summary["labels"].items():
        lines.append(
            f"  {field:24} true={counts['true']:3} false={counts['false']:3} "
            f"unlabelled={counts['unlabelled']:3}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, required=True, help="the question set to run")
    ap.add_argument("--corpus", type=Path, required=True, help="a corpus file, or a directory of shards")
    ap.add_argument("--out", type=Path, default=BASE_DIR / "real_eval_results.json")
    ap.add_argument("--report-only", action="store_true", help="summarise --out without asking anything")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    if args.report_only:
        if not args.out.exists():
            print(f"error: {args.out} does not exist", file=sys.stderr)
            return 2
        records = json.loads(args.out.read_text(encoding="utf-8"))
        print(render_report(summarize(records)))
        return 0

    try:
        questions = load_questions(args.questions)
    except (QuestionSetError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.limit:
        questions = questions[: args.limit]

    # Imported here so `--report-only` works without loading the model stack.
    import dotenv

    from groundedness import assess
    from recall import DEFAULT_K, ENV_PATH, build_session
    from uuid import uuid4

    from quivr_core.rag.entities.chat import ChatHistory

    from run_baseline import load_existing_results, save_results, serialize_sources

    # `build_session` expects the environment to be loaded already, exactly as `recall.py` and
    # `verify_product_parity.py` do before calling it.
    dotenv.load_dotenv(ENV_PATH if ENV_PATH.exists() else None)

    existing = {row.get("id"): row for row in load_existing_results(args.out)}
    print(f"questions : {len(questions)}")
    print(f"corpus    : {args.corpus}")
    print(f"k         : {args.k or DEFAULT_K}")
    print(f"output    : {args.out}\n")

    brain, retrieval_config = build_session(args.corpus, **({"k": args.k} if args.k else {}))

    records: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        from time import perf_counter

        started = perf_counter()
        response = brain.ask(
            run_id=uuid4(),
            question=question.question,
            retrieval_config=retrieval_config,
            chat_history=ChatHistory(chat_id=uuid4(), brain_id=brain.id),
        )
        latency_ms = (perf_counter() - started) * 1000

        sources = response.metadata.sources if response.metadata else []
        serialized = serialize_sources(sources)
        answer = response.answer or ""
        groundedness = assess(answer, serialized).as_dict()

        record = build_record(
            question=question,
            answer=answer,
            serialized_sources=serialized,
            latency_ms=latency_ms,
            groundedness=groundedness,
            previous=existing.get(question.id),
        )
        records.append(record)
        save_results(records, args.out)  # after every question: an API failure loses nothing

        flags = []
        if record["citation_mismatches"]:
            flags.append(f"{len(record['citation_mismatches'])} binding mismatch")
        if record["attribution_flags"]:
            flags.append(f"{len(record['attribution_flags'])} attribution")
        if record["invalid_citations"]:
            flags.append(f"{record['invalid_citations']} invalid citation")
        print(
            f"[{index:02d}/{len(questions)}] {question.id} {question.category:24} "
            f"{len(serialized):2} sources {latency_ms / 1000:6.1f}s"
            + (f"  ({', '.join(flags)})" if flags else "")
        )
        print(f"      Q: {question.question}")
        print(f"      A: {answer[:160].replace(chr(10), ' ')}")

    print()
    print(render_report(summarize(records)))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
