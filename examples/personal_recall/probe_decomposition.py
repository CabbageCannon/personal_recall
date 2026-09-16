"""Offline probe: does per-sub-question retrieval beat one flat query, at EQUAL slots?

This re-tests a Phase 3 negative under the conditions that can actually show a difference.

Phase 3 measured per-slice decomposition as worse, but on corpus v1, where Hit@10 was 100 %
and there was no headroom to win, and with a budget mismatch (the flat reference used 10
slots while the decomposition rows used 1.9-6.1). Corpus v2 has 8 evidence-insufficient
queries whose decisive lines sit at rank 20-50 of the fused ranking, so the mechanism now has
something to recover.

Fairness is the whole point of this probe: **every strategy is cut to the same k slots**.
Sub-question rankings are fused with the same weighted RRF the retrievers use, so the
comparison is "one question, 10 slots" vs "N sub-questions, 10 slots".

Sub-questions come from the frozen `stress_decomposition.json`, so the experiment is
deterministic and free (local BGE, repo BM25, no LLM, no API cost).

Usage:
    python probe_decomposition.py --dataset stress_v2 --verify-results stress_v2_a10_results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import dotenv
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from probe_dense_difficulty import DATASET_CONFIG, load_chunks, make_embedder  # noqa: E402
from probe_pool_rerank import WEIGHTS, coverage, rrf_ranking  # noqa: E402
from quivr_core.rag.hybrid import BM25Index  # noqa: E402

DECOMPOSITION_FILE = BASE_DIR / "stress_decomposition.json"


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
    sub_questions = json.loads(DECOMPOSITION_FILE.read_text(encoding="utf-8"))
    texts = load_chunks(corpus_path, chunking="session", max_session_chars=args.max_session_chars)

    def hybrid_ranking(question: str, dense_row: np.ndarray, bm25_index: BM25Index) -> list[int]:
        dense_top = [int(i) for i in dense_row[: args.hybrid_pool]]
        scores = bm25_index.scores(question)
        lexical_ranked = sorted(range(len(texts)), key=lambda j: -scores[j])
        lexical_top = [i for i in lexical_ranked if scores[i] > 0][: args.hybrid_pool]
        return rrf_ranking([dense_top, lexical_top], weights=WEIGHTS)

    embedder = make_embedder()
    chunk_vecs = np.asarray(embedder.embed_documents(texts), dtype=np.float32)
    chunk_vecs /= np.linalg.norm(chunk_vecs, axis=1, keepdims=True) + 1e-12
    bm25 = BM25Index(texts)

    # Embed the raw questions and every sub-question in one batch.
    raw_questions = [q["question"] for q in queries]
    subs_flat = [s for q in queries for s in sub_questions.get(q["id"], [])]
    all_questions = raw_questions + subs_flat
    vecs = np.asarray(embedder.embed_documents(all_questions), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    sims = vecs @ chunk_vecs.T
    dense_order = np.argsort(-sims, axis=1)

    baseline: list[list[int]] = []
    decomposed: list[list[int]] = []
    best_single: list[list[int]] = []
    n_subs: list[int] = []

    offset = len(raw_questions)
    for qi, query in enumerate(queries):
        baseline.append(hybrid_ranking(query["question"], dense_order[qi], bm25) [: args.k])

        subs = sub_questions.get(query["id"], [])
        n_subs.append(len(subs))
        if not subs:
            decomposed.append(baseline[-1])
            best_single.append(baseline[-1])
            continue

        rankings = [
            hybrid_ranking(sub, dense_order[offset + j], bm25) for j, sub in enumerate(subs)
        ]
        # Equal-slot fusion: treat each sub-question ranking as one more rank list.
        decomposed.append(rrf_ranking(rankings)[: args.k])
        # Control: the best single sub-question alone, given the whole budget.
        candidates = []
        for ranking in rankings:
            candidates.append(
                (coverage(query.get("relevant_evidence") or [], texts, ranking[: args.k]) or 0.0, ranking)
            )
        best_single.append(max(candidates, key=lambda pair: pair[0])[1][: args.k])
        offset += len(subs)

    # --- self-validation ---------------------------------------------------------------
    if args.verify_results:
        recorded = json.loads(args.verify_results.read_text(encoding="utf-8"))
        exact = total = 0
        for row in recorded:
            qi = next(i for i, q in enumerate(queries) if q["id"] == row["query_id"])
            want = [s["chunk_index"] - 1 for s in row.get("retrieved_sources") or []]
            if not want:
                continue
            total += 1
            if baseline[qi] == want:
                exact += 1
        print(f"reproduction check vs {args.verify_results.name}: identical {exact}/{total}")
        if exact < total * 0.8:
            print("REFUSING to compare: the offline reproduction does not match the run.")
            return 1

    answerable = [i for i, q in enumerate(queries) if q["answerable"]]

    def mean_cov(selection: list[list[int]]) -> float:
        return statistics.mean(
            v
            for v in (
                coverage(queries[i].get("relevant_evidence") or [], texts, selection[i])
                for i in answerable
            )
            if v is not None
        )

    strategies = [
        ("flat question, hybrid top-k (A10)", baseline),
        ("sub-question RRF fusion, top-k", decomposed),
        ("best single sub-question, top-k", best_single),
    ]

    print(f"\ncorpus {corpus_path.name}: {len(texts)} chunks, {len(queries)} queries")
    print(f"sub-questions available for {sum(1 for n in n_subs if n)}/{len(queries)} queries "
          f"(mean {statistics.mean(n for n in n_subs if n):.1f})")
    print(f"\n{'strategy':38} {'slots':>6} {'coverage':>10}")
    print("-" * 58)
    summary = {}
    for label, selection in strategies:
        m = mean_cov(selection)
        summary[label] = m
        print(f"{label:38} {args.k:6} {m * 100:9.1f}%")

    base, dec = summary[strategies[0][0]], summary[strategies[1][0]]
    print(f"\n  delta (decomposition - flat) = {(dec - base) * 100:+.2f} pts")

    up = down = same = 0
    print(f"\n  {'id':6} {'category':22} {'flat':>7} {'decomp':>7} {'bestsub':>8}  {'n':>2}")
    for i in answerable:
        a = coverage(queries[i].get("relevant_evidence") or [], texts, baseline[i])
        b = coverage(queries[i].get("relevant_evidence") or [], texts, decomposed[i])
        c = coverage(queries[i].get("relevant_evidence") or [], texts, best_single[i])
        if b > a + 1e-9:
            up += 1
        elif b < a - 1e-9:
            down += 1
        else:
            same += 1
        flag = "" if abs(b - a) < 1e-9 else ("  +" if b > a else "  -")
        print(f"  {queries[i]['id']:6} {queries[i]['category'][:22]:22} {a:7.3f} {b:7.3f} {c:8.3f}  {n_subs[i]:2}{flag}")
    print(f"\n  improved {up} | worsened {down} | unchanged {same}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "corpus": str(corpus_path),
                    "chunks": len(texts),
                    "k": args.k,
                    "coverage": summary,
                    "rows": [
                        {
                            "query_id": queries[i]["id"],
                            "category": queries[i]["category"],
                            "n_sub_questions": n_subs[i],
                            "cov_flat": coverage(queries[i].get("relevant_evidence") or [], texts, baseline[i]),
                            "cov_decomposition": coverage(queries[i].get("relevant_evidence") or [], texts, decomposed[i]),
                            "cov_best_single_sub": coverage(queries[i].get("relevant_evidence") or [], texts, best_single[i]),
                        }
                        for i in answerable
                    ],
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
