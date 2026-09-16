import argparse
import json
from collections import defaultdict
from pathlib import Path
from eval_utils import (
    evidence_matches_source,
    strict_evidence_matches_source,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# One per-dataset config is the single source of truth for every
# dataset-specific path. "small" is the default and keeps the original
# file names, so a run without --dataset behaves exactly as before.
DATASET_CONFIG = {
    "small": {
        "queries": DATA_DIR / "queries.json",
        "results": BASE_DIR / "baseline_results.json",
        "summary": BASE_DIR / "baseline_summary.json",
        "labels": BASE_DIR / "manual_labels.json",
    },
    "stress": {
        "queries": DATA_DIR / "stress_queries.json",
        "results": BASE_DIR / "stress_results.json",
        "summary": BASE_DIR / "stress_summary.json",
        "labels": BASE_DIR / "stress_manual_labels.json",
    },
}

DEFAULT_DATASET = "small"


def dataset_paths(dataset: str) -> dict[str, Path]:
    # Resolve every path for one dataset from the config above, so no
    # queries / results / summary / labels path is hard-coded below.
    return DATASET_CONFIG[dataset]

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_saved_retrieval(
    query: dict,
    result: dict,
) -> dict:
    """
    Re-evaluate retrieval using the saved Top-K chunks.

    This lets us update the matching logic without re-running
    embedding, retrieval, rewrite, or LLM generation.
    """
    gold_evidence = query["relevant_evidence"]

    if not gold_evidence:
        return {
            "hit_at_5": None,
            "matched_evidence_count": 0,
            "gold_evidence_count": 0,
            "evidence_coverage": None,
            "strict_hit_at_5": None,
            "strict_matched_evidence_count": 0,
            "strict_evidence_coverage": None,
        }

    source_contents = [
        source["content"]
        for source in result["retrieved_sources"]
    ]

    matched_evidence = []
    strict_matched_evidence = []

    for evidence in gold_evidence:
        if any(
            evidence_matches_source(
                evidence,
                source_content,
            )
            for source_content in source_contents
        ):
            matched_evidence.append(evidence)

        if any(
            strict_evidence_matches_source(
                evidence,
                source_content,
            )
            for source_content in source_contents
        ):
            strict_matched_evidence.append(evidence)

    gold_count = len(gold_evidence)
    matched_count = len(matched_evidence)
    strict_matched_count = len(strict_matched_evidence)

    return {
        "hit_at_5": matched_count > 0,
        "matched_evidence_count": matched_count,
        "gold_evidence_count": gold_count,
        "evidence_coverage": matched_count / gold_count,
        "strict_hit_at_5": strict_matched_count > 0,
        "strict_matched_evidence_count": strict_matched_count,
        "strict_evidence_coverage": strict_matched_count / gold_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize a personal recall retrieval run.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIG),
        default=DEFAULT_DATASET,
        help=(
            "Which dataset to summarize (default: %(default)s). "
            "'small' reads baseline_results.json, "
            "'stress' reads stress_results.json."
        ),
    )
    args = parser.parse_args()

    paths = dataset_paths(args.dataset)

    queries_path = paths["queries"]
    result_path = paths["results"]
    summary_path = paths["summary"]
    labels_path = paths["labels"]

    # Self-documenting run banner
    print(f"Dataset: {args.dataset}")
    print(f"Queries: {queries_path}")
    print(f"Results: {result_path}")
    print(f"Summary: {summary_path}")
    print(f"Labels: {labels_path}")

    queries = load_json(queries_path)
    results = load_json(result_path)
    labels = load_json(labels_path)

    query_by_id = {
        query["id"]: query
        for query in queries
    }
    
    label_by_id = {
        label["query_id"]: label
        for label in labels
    }
    
    missing_labels = [
        result["query_id"]
        for result in results
        if result["query_id"] not in label_by_id
    ]

    if missing_labels:
        raise ValueError(
            f"Missing manual labels for: {missing_labels}"
        )

    if len(results) != len(queries):
        print(
            f"Warning: queries={len(queries)}, "
            f"results={len(results)}"
        )

    evaluated_results = []

    for result in results:
        query_id = result["query_id"]

        if query_id not in query_by_id:
            raise ValueError(
                f"Cannot find query {query_id} in queries.json"
            )

        query = query_by_id[query_id]

        retrieval_eval = evaluate_saved_retrieval(
            query=query,
            result=result,
        )

        manual_label = label_by_id[query_id]

        evaluated_results.append(
            {
                **result,
                **retrieval_eval,
                "manual_label": manual_label["label"],
                "unsupported_claim": manual_label[
                    "unsupported_claim"
                ],
                "manual_note": manual_label.get("note"),
            }
        )

    # =========================================================
    # Overall retrieval metrics
    # =========================================================

    valid_results = [
        result
        for result in evaluated_results
        if result["manual_label"] != "INVALID_QUERY"
    ]

    answerable_results = [
        result
        for result in valid_results
        if result["answerable"]
    ]

    hit_count = sum(
        result["hit_at_5"] is True
        for result in answerable_results
    )

    strict_hit_count = sum(
        result["strict_hit_at_5"] is True
        for result in answerable_results
    )

    answerable_count = len(answerable_results)

    hit_at_5 = (
        hit_count / answerable_count
        if answerable_count
        else 0.0
    )

    strict_hit_at_5 = (
        strict_hit_count / answerable_count
        if answerable_count
        else 0.0
    )

    average_evidence_coverage = (
        sum(
            result["evidence_coverage"]
            for result in answerable_results
            if result["evidence_coverage"] is not None
        )
        / answerable_count
        if answerable_count
        else 0.0
    )

    average_strict_evidence_coverage = (
        sum(
            result["strict_evidence_coverage"]
            for result in answerable_results
            if result["strict_evidence_coverage"] is not None
        )
        / answerable_count
        if answerable_count
        else 0.0
    )

    total_matched_evidence = sum(
        result["matched_evidence_count"]
        for result in answerable_results
    )

    total_gold_evidence = sum(
        result["gold_evidence_count"]
        for result in answerable_results
    )

    weighted_evidence_coverage = (
        total_matched_evidence / total_gold_evidence
        if total_gold_evidence
        else 0.0
    )
    
    # =========================================================
    # Manual answer evaluation
    # =========================================================

    pass_results = [
        result
        for result in valid_results
        if result["manual_label"] == "PASS"
    ]

    partial_results = [
        result
        for result in valid_results
        if result["manual_label"] == "PARTIAL"
    ]

    fail_results = [
        result
        for result in valid_results
        if result["manual_label"] == "FAIL"
    ]

    unsupported_results = [
        result
        for result in valid_results
        if result["unsupported_claim"]
    ]
    
    unanswerable_count = sum(
        not result["answerable"]
        for result in valid_results
    )

    valid_count = len(valid_results)

    pass_rate = (
        len(pass_results) / valid_count
        if valid_count
        else 0.0
    )

    unsupported_rate = (
        len(unsupported_results) / valid_count
        if valid_count
        else 0.0
    )

    invalid_queries = [
        result["query_id"]
        for result in evaluated_results
        if result["manual_label"] == "INVALID_QUERY"
    ]

    # =========================================================
    # Latency
    # =========================================================

    latencies = [
        result["latency_ms"]
        for result in evaluated_results
    ]

    average_latency_ms = (
        sum(latencies) / len(latencies)
        if latencies
        else 0.0
    )

    min_latency_ms = min(latencies) if latencies else 0.0
    max_latency_ms = max(latencies) if latencies else 0.0

    # =========================================================
    # Category metrics
    # =========================================================

    category_results = defaultdict(list)

    for result in evaluated_results:
        category_results[result["category"]].append(result)

    category_summary = {}

    for category, items in category_results.items():
        valid_items = [
            item
            for item in items
            if item["manual_label"] != "INVALID_QUERY"
        ]

        answerable_items = [
            item
            for item in valid_items
            if item["answerable"]
        ]

        # -------------------------
        # Retrieval metrics
        # -------------------------

        if answerable_items:
            category_hits = sum(
                item["hit_at_5"] is True
                for item in answerable_items
            )

            category_hit_at_5 = (
                category_hits / len(answerable_items)
            )

            category_coverage = (
                sum(
                    item["evidence_coverage"]
                    for item in answerable_items
                    if item["evidence_coverage"] is not None
                )
                / len(answerable_items)
            )
        else:
            category_hit_at_5 = None
            category_coverage = None

        # -------------------------
        # Answer evaluation
        # -------------------------

        category_pass_count = sum(
            item["manual_label"] == "PASS"
            for item in valid_items
        )

        category_partial_count = sum(
            item["manual_label"] == "PARTIAL"
            for item in valid_items
        )

        category_fail_count = sum(
            item["manual_label"] == "FAIL"
            for item in valid_items
        )

        category_pass_rate = (
            category_pass_count / len(valid_items)
            if valid_items
            else None
        )

        category_unsupported_count = sum(
            item["unsupported_claim"]
            for item in valid_items
        )

        category_unsupported_rate = (
            category_unsupported_count / len(valid_items)
            if valid_items
            else None
        )

        # -------------------------
        # Latency
        # -------------------------

        category_latency = (
            sum(item["latency_ms"] for item in items)
            / len(items)
        )

        category_summary[category] = {
            "query_count": len(items),
            "valid_query_count": len(valid_items),
            "answerable_count": len(answerable_items),

            "hit_at_5": category_hit_at_5,
            "average_evidence_coverage": category_coverage,

            "pass_count": category_pass_count,
            "partial_count": category_partial_count,
            "fail_count": category_fail_count,
            "pass_rate": category_pass_rate,

            "unsupported_count": category_unsupported_count,
            "unsupported_rate": category_unsupported_rate,

            "average_latency_ms": category_latency,
        }

    # =========================================================
    # Miss / diagnostic cases
    # =========================================================

    retrieval_misses = [
        result["query_id"]
        for result in answerable_results
        if result["hit_at_5"] is False
    ]

    boundary_affected = [
        result["query_id"]
        for result in answerable_results
        if (
            result["hit_at_5"] is True
            and result["strict_evidence_coverage"]
            < result["evidence_coverage"]
        )
    ]

    # =========================================================
    # Save machine-readable summary
    # =========================================================

    summary = {
        "total_queries": len(evaluated_results),
        "valid_queries": valid_count,
        "invalid_queries": len(invalid_queries),
        "answerable_queries": answerable_count,
        "unanswerable_queries": unanswerable_count,
        "retrieval": {
            "hit_at_5": hit_at_5,
            "hit_count": hit_count,
            "strict_hit_at_5": strict_hit_at_5,
            "strict_hit_count": strict_hit_count,
            "average_evidence_coverage": average_evidence_coverage,
            "average_strict_evidence_coverage": (
                average_strict_evidence_coverage
            ),
            "weighted_evidence_coverage": (
                weighted_evidence_coverage
            ),
            "retrieval_misses": retrieval_misses,
            "boundary_affected_queries": boundary_affected,
        },
        "answer_evaluation": {
            "valid_queries": valid_count,
            "pass_count": len(pass_results),
            "partial_count": len(partial_results),
            "fail_count": len(fail_results),
            "pass_rate": pass_rate,
            "unsupported_count": len(unsupported_results),
            "unsupported_rate": unsupported_rate,
            "invalid_queries": invalid_queries,
        },
        "latency": {
            "average_ms": average_latency_ms,
            "min_ms": min_latency_ms,
            "max_ms": max_latency_ms,
        },
        "categories": category_summary,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # =========================================================
    # Console report
    # =========================================================

    print("=" * 80)
    print("Personal Recall Baseline Summary")
    print("=" * 80)

    print(f"Total queries:       {len(evaluated_results)}")
    print(f"Answerable queries:  {answerable_count}")
    print(f"Valid queries:       {valid_count}")
    print(f"Invalid queries:     {len(invalid_queries)}")
    print(f"Unanswerable:        {unanswerable_count}")

    print("\nRetrieval")
    print("-" * 80)
    print(
        f"Hit@5:              "
        f"{hit_count}/{answerable_count} "
        f"({hit_at_5:.2%})"
    )
    print(
        f"Strict Hit@5:       "
        f"{strict_hit_count}/{answerable_count} "
        f"({strict_hit_at_5:.2%})"
    )
    print(
        f"Avg evidence cover: {average_evidence_coverage:.2%}"
    )
    print(
        "Strict avg cover:   "
        f"{average_strict_evidence_coverage:.2%}"
    )
    print(
        f"Weighted coverage:  {weighted_evidence_coverage:.2%}"
    )
    
    print("\nAnswer Evaluation")
    print("-" * 80)

    print(
        f"PASS:              "
        f"{len(pass_results)}/{valid_count} "
        f"({pass_rate:.2%})"
    )

    print(
        f"PARTIAL:           "
        f"{len(partial_results)}"
    )

    print(
        f"FAIL:              "
        f"{len(fail_results)}"
    )

    print(
        f"Unsupported claims:"
        f" {len(unsupported_results)}/{valid_count} "
        f"({unsupported_rate:.2%})"
    )

    print(
        "Invalid queries:   "
        f"{invalid_queries if invalid_queries else 'None'}"
    )
    
    print("\nLatency")
    print("-" * 80)
    print(f"Average: {average_latency_ms:.2f} ms")
    print(f"Minimum: {min_latency_ms:.2f} ms")
    print(f"Maximum: {max_latency_ms:.2f} ms")

    print("\nBy Category")
    print("-" * 80)
    print(
        f"{'Category':<20}"
        f"{'Count':>7}"
        f"{'Hit@5':>12}"
        f"{'Coverage':>12}"
        f"{'PASS':>10}"
        f"{'Latency':>12}"
    )
    print("-" * 80)

    for category, metrics in category_summary.items():
        hit_display = (
            f"{metrics['hit_at_5']:.2%}"
            if metrics["hit_at_5"] is not None
            else "N/A"
        )

        coverage_display = (
            f"{metrics['average_evidence_coverage']:.2%}"
            if metrics["average_evidence_coverage"] is not None
            else "N/A"
        )
        
        pass_display = (
            f"{metrics['pass_rate']:.2%}"
            if metrics["pass_rate"] is not None
            else "N/A"
        )

        print(
            f"{category:<20}"
            f"{metrics['valid_query_count']:>7}"
            f"{hit_display:>12}"
            f"{coverage_display:>12}"
            f"{pass_display:>10}"
            f"{metrics['average_latency_ms']:>10.0f}ms"
        )

    print("\nDiagnostics")
    print("-" * 80)
    print(
        "Retrieval misses: "
        f"{retrieval_misses if retrieval_misses else 'None'}"
    )
    print(
        "Boundary affected: "
        f"{boundary_affected if boundary_affected else 'None'}"
    )

    print("\n" + "=" * 80)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()