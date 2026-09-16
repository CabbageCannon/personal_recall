"""Oracle bound for metadata filtering: is a time/person filter worth building an extractor for?

Both cheaper selectors have now been measured negative on corpus v2 -- a general
cross-encoder re-ranker (-6.62 pts) and sub-question fusion (-7.60 pts) -- even though the
50-candidate pool holds 93.1 % of the gold. The evidence is retrieved; the ranking within the
pool is what loses it.

The one lever not yet tested is restricting candidates by **metadata the chunk already
carries** (``start_time`` / ``end_time`` / ``participants``). Unlike a re-ranker, a filter
removes distractors outright instead of rescoring them.

This probe does NOT build an extractor. It computes the **oracle upper bound**: the window
and the person set are taken from the gold evidence itself, so the number it reports is the
best any perfect extractor could achieve. If the oracle gain is small, the whole direction is
dead and no extractor should be written; if it is large, the extractor becomes worth building.

Free and deterministic (local BGE, repo BM25, no LLM).

Usage:
    python probe_metadata_oracle.py --dataset stress_v2 --verify-results stress_v2_a10_results.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

import dotenv
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from memory import SessionConfig, build_sessions, parse_txt_events  # noqa: E402
from probe_dense_difficulty import DATASET_CONFIG, load_chunks, make_embedder  # noqa: E402
from probe_pool_rerank import WEIGHTS, coverage, rrf_ranking  # noqa: E402
from quivr_core.rag.hybrid import BM25Index  # noqa: E402

TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\s*([^:]+):")


def main() -> int:
    dotenv.load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASET_CONFIG), default="stress_v2")
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--queries", type=Path)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--hybrid-pool", type=int, default=30)
    ap.add_argument("--max-session-chars", type=int, default=900)
    ap.add_argument("--verify-results", type=Path)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    corpus_path = args.corpus or cfg["corpus"]
    queries_path = args.queries or cfg["queries"]

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    texts = load_chunks(corpus_path, chunking="session", max_session_chars=args.max_session_chars)
    sessions = build_sessions(
        parse_txt_events(corpus_path.read_text(encoding="utf-8")).events,
        SessionConfig(max_chars=args.max_session_chars),
    )
    assert len(sessions) == len(texts), (
        f"chunk/session count mismatch: {len(sessions)} vs {len(texts)} - "
        "metadata would not line up with the ranked list"
    )

    embedder = make_embedder()
    chunk_vecs = np.asarray(embedder.embed_documents(texts), dtype=np.float32)
    chunk_vecs /= np.linalg.norm(chunk_vecs, axis=1, keepdims=True) + 1e-12
    question_vecs = np.asarray(
        embedder.embed_documents([q["question"] for q in queries]), dtype=np.float32
    )
    question_vecs /= np.linalg.norm(question_vecs, axis=1, keepdims=True) + 1e-12
    dense_order = np.argsort(-(question_vecs @ chunk_vecs.T), axis=1)
    bm25 = BM25Index(texts)

    fused: list[list[int]] = []
    for qi, query in enumerate(queries):
        dense_top = [int(i) for i in dense_order[qi][: args.hybrid_pool]]
        scores = bm25.scores(query["question"])
        lexical_ranked = sorted(range(len(texts)), key=lambda j: -scores[j])
        lexical_top = [i for i in lexical_ranked if scores[i] > 0][: args.hybrid_pool]
        fused.append(rrf_ranking([dense_top, lexical_top], weights=WEIGHTS))

    if args.verify_results:
        recorded = json.loads(args.verify_results.read_text(encoding="utf-8"))
        exact = total = 0
        for row in recorded:
            qi = next(i for i, q in enumerate(queries) if q["id"] == row["query_id"])
            want = [s["chunk_index"] - 1 for s in row.get("retrieved_sources") or []]
            if not want:
                continue
            total += 1
            if fused[qi][: args.k] == want:
                exact += 1
        print(f"reproduction check vs {args.verify_results.name}: identical {exact}/{total}")
        if exact < total * 0.8:
            print("REFUSING to compare: the offline reproduction does not match the run.")
            return 1

    strategies = ("flat hybrid top-k", "oracle time filter", "oracle person filter", "oracle both")
    results: dict[str, list[float | None]] = {name: [] for name in strategies}
    rows = []

    for qi, query in enumerate(queries):
        gold = query.get("relevant_evidence") or []
        baseline = coverage(gold, texts, fused[qi][: args.k])
        if baseline is None:
            continue

        stamps, people = [], set()
        for line in gold:
            match = TS_RE.match(line.strip())
            if match:
                stamps.append(datetime.strptime(match.group(1), "%Y-%m-%d %H:%M"))
                people.add(match.group(2).strip())

        def filtered(keep) -> list[int]:
            order = [i for i in fused[qi] if keep(i)]
            return order[: args.k]

        if stamps:
            lo, hi = min(stamps), max(stamps)
            time_keep = lambda i: sessions[i].start_time <= hi and sessions[i].end_time >= lo  # noqa: E731
        else:
            time_keep = lambda i: True  # noqa: E731
        person_keep = lambda i: bool(set(sessions[i].participants) & people)  # noqa: E731

        covs = {
            "flat hybrid top-k": baseline,
            "oracle time filter": coverage(gold, texts, filtered(time_keep)),
            "oracle person filter": coverage(gold, texts, filtered(person_keep)),
            "oracle both": coverage(gold, texts, filtered(lambda i: time_keep(i) and person_keep(i))),
        }
        for name, value in covs.items():
            results[name].append(value)
        rows.append(
            {
                "query_id": query["id"],
                "category": query["category"],
                "n_candidates_time": sum(1 for i in range(len(texts)) if time_keep(i)),
                "n_candidates_person": sum(1 for i in range(len(texts)) if person_keep(i)),
                **{f"cov::{k}": v for k, v in covs.items()},
            }
        )

    print(f"\ncorpus {corpus_path.name}: {len(texts)} chunks, "
          f"{len(results['flat hybrid top-k'])} answerable queries")
    print(f"\n{'strategy':26} {'coverage':>10}  {'delta':>8}")
    print("-" * 50)
    base = statistics.mean(v for v in results["flat hybrid top-k"] if v is not None)
    summary = {}
    for name in strategies:
        vals = [v for v in results[name] if v is not None]
        m = statistics.mean(vals)
        summary[name] = m
        print(f"{name:26} {m * 100:9.1f}%  {(m - base) * 100:+7.2f}")

    print(f"\n  mean candidates kept: time "
          f"{statistics.mean(r['n_candidates_time'] for r in rows):.1f}, "
          f"person {statistics.mean(r['n_candidates_person'] for r in rows):.1f} "
          f"of {len(texts)}")

    print(f"\n  {'id':6} {'category':22} {'flat':>6} {'time':>6} {'person':>7} {'both':>6}")
    for r in rows:
        print(f"  {r['query_id']:6} {r['category'][:22]:22} "
              f"{r['cov::flat hybrid top-k']:6.3f} {r['cov::oracle time filter']:6.3f} "
              f"{r['cov::oracle person filter']:7.3f} {r['cov::oracle both']:6.3f}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "corpus": str(corpus_path),
                    "chunks": len(texts),
                    "k": args.k,
                    "oracle_coverage": summary,
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n  written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
