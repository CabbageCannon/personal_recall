"""Citation metrics for Personal Recall runs (offline, deterministic, no LLM).

The engine's promise is "Answer -> Evidence -> MemoryEvent -> original chat line". These
metrics check the first link, which nothing measured before:

* **citation rate** — did the answer cite anything at all?
* **valid citation rate** — do the cited numbers exist among the retrieved sources?
* **citation coverage** — of the gold evidence lines, how many sit inside a chunk the
  answer actually cited? (distinguishes "retrieved but not cited" from "never retrieved")
* **citation precision (lexical proxy)** — of the chunks the answer cited, how many contain
  at least one gold evidence line? A conservative lower bound: a cited chunk without gold
  evidence is not necessarily wrong, but it is unverifiable against the eval's gold set.
* **abstention accuracy** — on unanswerable questions, did the answer avoid asserting a
  fact (no citations and/or an explicit decline)?

Usage:
    python citation_metrics.py --queries data/stress_queries.json \
        --results stress_norewrite_results.json --json-out citation_metrics.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from eval_utils import evidence_matches_source

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CITATION_RE = re.compile(r"\[\s*(?:来源|Source)\s*(\d+)\s*\]|(?<!\[)\b来源\s*(\d+)", re.IGNORECASE)
ABSTAIN_RE = re.compile(
    r"(没有找到|没找到|找不到|没有(?:任何)?(?:相关)?(?:记录|证据|提到|提及)|没有提及|未提及|"
    r"无法(?:从|确定|确认)|不能确定|没有明确|没有说|没有提到|记录里没有|聊天记录中?没有|"
    r"没有|不是|并未|未曾|没去|没成功|没定)"
)


def parse_citations(answer: str, n_sources: int) -> tuple[list[int], int]:
    """Return (valid citation numbers in order, count of invalid/out-of-range ones).

    Numbering follows the framework: ``combine_documents`` renders each retrieved chunk as
    ``Source: {index}`` with ``index = range(len(docs))``, i.e. **starting at 0**, and the
    citation prompt tells the model to use that number. So the valid range is
    ``0 .. n_sources - 1`` and ``sources[N]`` is the cited item.
    """
    valid: list[int] = []
    invalid = 0
    for match in CITATION_RE.finditer(answer or ""):
        number = int(match.group(1) or match.group(2))
        if 0 <= number < n_sources:
            valid.append(number)
        else:
            invalid += 1
    return valid, invalid


def evaluate(
    queries: list[dict],
    results: list[dict],
) -> dict:
    by_id = {q["id"]: q for q in queries}
    rows = []
    for result in results:
        query = by_id.get(result["query_id"], {})
        sources = result.get("retrieved_sources") or []
        gold = query.get("relevant_evidence") or []
        answer = result.get("answer") or ""

        cited, invalid = parse_citations(answer, len(sources))
        cited_contents = [sources[n]["content"] for n in cited]

        retrieved_contents = [s["content"] for s in sources]

        if gold:
            retrieved_matched = sum(
                1 for e in gold if any(evidence_matches_source(e.strip(), c) for c in retrieved_contents)
            )
            cited_matched = sum(
                1 for e in gold if any(evidence_matches_source(e.strip(), c) for c in cited_contents)
            )
            cited_with_gold = sum(
                1
                for c in cited_contents
                if any(evidence_matches_source(e.strip(), c) for e in gold)
            )
        else:
            retrieved_matched = cited_matched = cited_with_gold = None

        rows.append(
            {
                "query_id": result["query_id"],
                "answerable": query.get("answerable", True),
                "n_sources": len(sources),
                "n_citations": len(cited),
                "n_invalid_citations": invalid,
                "cited": cited,
                "gold_lines": len(gold),
                "gold_in_retrieved": retrieved_matched,
                "gold_in_cited": cited_matched,
                "cited_chunks_with_gold": cited_with_gold,
                "abstains": bool(ABSTAIN_RE.search(answer)),
            }
        )

    answerable = [r for r in rows if r["answerable"] is not False]
    unanswerable = [r for r in rows if r["answerable"] is False]

    total_citations = sum(r["n_citations"] for r in rows)
    total_invalid = sum(r["n_invalid_citations"] for r in rows)
    cited_chunks = sum(r["n_citations"] for r in rows)
    cited_with_gold = sum(r["cited_chunks_with_gold"] or 0 for r in answerable)

    answerable_gold = sum(r["gold_lines"] for r in answerable)
    gold_retrieved = sum(r["gold_in_retrieved"] or 0 for r in answerable)
    gold_cited = sum(r["gold_in_cited"] or 0 for r in answerable)

    return {
        "queries": len(rows),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "citation_rate": sum(1 for r in rows if r["n_citations"]) / len(rows) if rows else None,
        "total_citations": total_citations,
        "invalid_citations": total_invalid,
        "valid_citation_rate": (
            total_citations / (total_citations + total_invalid)
            if (total_citations + total_invalid)
            else None
        ),
        "citation_coverage": gold_cited / answerable_gold if answerable_gold else None,
        "retrieval_coverage": gold_retrieved / answerable_gold if answerable_gold else None,
        "citation_precision_lexical": cited_with_gold / cited_chunks if cited_chunks else None,
        "abstention_accuracy": (
            sum(1 for r in unanswerable if r["abstains"]) / len(unanswerable)
            if unanswerable
            else None
        ),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Citation metrics for a Personal Recall run.")
    ap.add_argument("--queries", type=Path, default=DATA_DIR / "stress_queries.json")
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    metrics = evaluate(queries, results)

    pct = lambda v: "n/a" if v is None else f"{v * 100:.1f}%"  # noqa: E731
    print(f"results : {args.results.name}")
    print(f"queries : {metrics['queries']} ({metrics['answerable']} answerable, "
          f"{metrics['unanswerable']} unanswerable)")
    print()
    print(f"citation rate              : {pct(metrics['citation_rate'])}  "
          f"({metrics['total_citations']} citations, {metrics['invalid_citations']} invalid)")
    print(f"valid citation rate        : {pct(metrics['valid_citation_rate'])}")
    print(f"citation coverage (gold)   : {pct(metrics['citation_coverage'])}   "
          f"<- gold evidence lines inside a CITED chunk")
    print(f"retrieval coverage (gold)  : {pct(metrics['retrieval_coverage'])}   "
          f"<- gold evidence lines inside any RETRIEVED chunk")
    print(f"citation precision (lexical): {pct(metrics['citation_precision_lexical'])}")
    print(f"abstention accuracy        : {pct(metrics['abstention_accuracy'])}")

    if args.verbose:
        print("\nper query (cited vs gold in retrieved / gold in cited):")
        for row in metrics["rows"]:
            print(f"  {row['query_id']} cited={row['cited']} "
                  f"gold {row['gold_in_retrieved']}/{row['gold_lines']} retrieved, "
                  f"{row['gold_in_cited']}/{row['gold_lines']} cited")

    if args.json_out:
        args.json_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwritten to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
