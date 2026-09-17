"""Offline probe: does re-ranking a wide candidate pool beat the frozen hybrid Top-k?

Motivation. The recall-vs-k sweep on corpus v2 showed the failures are **ranking-limited,
not representation-limited**: for 7 of the 8 evidence-insufficient queries the missing gold
line sits at dense rank 20-50, well inside reach. Coverage on those 8 goes 47.9 % @10 ->
85.4 % @50. So the question is no longer "can we retrieve it" but "can we promote it into
the 10 slots we actually show the model".

Re-ranking was measured **negative** on corpus v1 (Phase 2: 50 -> 10 gave 80.9 % vs dense
82.4 %) -- but v1 had only ~15 points of headroom between k=10 and k=50, and v2 has ~37 on
the queries that fail. This probe re-tests it where it can actually show a difference.

Everything here is offline and deterministic: local BGE embedder, the repo's own BM25 and
cross-encoder, no LLM and no API cost. The hybrid ranking is reproduced exactly as
``quivr_rag_langgraph`` builds it (each retriever contributes ``--hybrid-pool`` candidates,
weighted RRF with rank starting at 1, dedup by content, stable sort, truncate to k) and is
**verified against a recorded run's chunk indices** before any comparison is trusted.

Usage:
    python probe_pool_rerank.py --dataset stress_v2 \\
        --verify-results stress_v2_a10_results.json \\
        --rerank-model D:\\AIModels\\bge-reranker-base
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Sequence

import dotenv
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from eval_utils import evidence_matches_source  # noqa: E402
from probe_dense_difficulty import DATASET_CONFIG, load_chunks, make_embedder  # noqa: E402
from quivr_core.rag.hybrid import BM25Index  # noqa: E402

RRF_C = 60
#: Weight per retriever, matching ``HybridConfig.weights`` for the dense+lexical pair.
WEIGHTS = (0.5, 0.5)


def rrf_ranking(
    rank_lists: list[list[int]],
    c: int = RRF_C,
    weights: Sequence[float] | None = None,
) -> list[int]:
    """Weighted RRF over pre-ranked index lists, replicating LangChain's fusion.

    Ranks start at 1 and are added as ``weight / (rank + c)``; documents are deduplicated
    by first occurrence across the lists (dense first, then lexical) and the final sort is
    stable, so equal scores keep insertion order -- exactly ``weighted_reciprocal_rank``.

    ``weights`` defaults to equal weights over however many lists are supplied. It must
    never be a fixed-length constant: zipping N lists against 2 weights silently discards
    the remaining lists, which is both wrong and invisible in the output.
    """
    if not rank_lists:
        return []
    if weights is None:
        weights = [1.0 / len(rank_lists)] * len(rank_lists)
    if len(weights) != len(rank_lists):
        raise ValueError(f"{len(rank_lists)} rank lists but {len(weights)} weights")

    scores: dict[int, float] = {}
    for docs, weight in zip(rank_lists, weights):
        for rank, index in enumerate(docs, start=1):
            scores[index] = scores.get(index, 0.0) + weight / (rank + c)
    return [index for index, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def coverage(gold: list[str], texts: list[str], selected: list[int]) -> float | None:
    if not gold:
        return None
    contents = [texts[i] for i in selected]
    hits = sum(
        1 for line in gold if any(evidence_matches_source(line.strip(), c) for c in contents)
    )
    return hits / len(gold)


def main() -> int:
    dotenv.load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASET_CONFIG), default="stress_v2")
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--queries", type=Path)
    ap.add_argument("--k", type=int, default=10, help="slots shown to the model (A10: 10)")
    ap.add_argument("--hybrid-pool", type=int, default=30, help="candidates per retriever")
    ap.add_argument("--pool", type=int, default=50, help="fused candidates fed to the reranker")
    ap.add_argument("--rerank-model", default=None, help="optional; omit to measure hybrid RRF alone")
    ap.add_argument("--max-session-chars", type=int, default=900)
    ap.add_argument("--verify-results", type=Path, help="recorded run to validate the reproduction")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    corpus_path = args.corpus or cfg["corpus"]
    queries_path = args.queries or cfg["queries"]

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    texts = load_chunks(corpus_path, chunking="session", max_session_chars=args.max_session_chars)
    print(f"corpus : {corpus_path}")
    print(f"chunks : {len(texts)}   queries: {len(queries)}")

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
        # Mirrors BM25Index.top_k: stable sort by descending score, drop zero scores.
        lexical_ranked = sorted(range(len(texts)), key=lambda j: -scores[j])
        lexical_top = [i for i in lexical_ranked if scores[i] > 0][: args.hybrid_pool]
        fused.append(rrf_ranking([dense_top, lexical_top], weights=WEIGHTS))

    # --- self-validation against a real run, before trusting anything else -------------
    agreement = None
    if args.verify_results:
        recorded = json.loads(args.verify_results.read_text(encoding="utf-8"))
        exact = partial = 0
        total = 0
        mismatches = []
        for row in recorded:
            qi = next(i for i, q in enumerate(queries) if q["id"] == row["query_id"])
            want = [s["chunk_index"] - 1 for s in row.get("retrieved_sources") or []]
            if not want:
                continue
            got = fused[qi][: len(want)]
            total += 1
            if got == want:
                exact += 1
            elif set(got) == set(want):
                partial += 1
            else:
                mismatches.append((row["query_id"], want, got))
        agreement = (exact, partial, total)
        print(
            f"\nreproduction check vs {args.verify_results.name}: "
            f"identical {exact}/{total}, same-set-different-order {partial}"
        )
        for qid, want, got in mismatches[:5]:
            print(f"  MISMATCH {qid}\n    recorded: {want}\n    offline : {got}")
        if exact < total * 0.8:
            print("\nREFUSING to compare: the offline reproduction does not match the run.")
            return 1

    # --- the comparison ----------------------------------------------------------------
    reranker = None
    if args.rerank_model:
        from langchain_core.documents import Document

        from quivr_core.rag.reranker import LocalCrossEncoderReranker

        reranker = LocalCrossEncoderReranker(model=args.rerank_model, top_n=args.k)

    rows = []
    for qi, query in enumerate(queries):
        gold = query.get("relevant_evidence") or []
        ranked = fused[qi]
        hybrid_k = ranked[: args.k]
        pool = ranked[: args.pool]

        reranked: list[int] = []
        if reranker is not None:
            docs = [
                Document(page_content=texts[i], metadata={"chunk": i}) for i in pool
            ]
            out = reranker.compress_documents(docs, query["question"])
            reranked = [int(d.metadata["chunk"]) for d in out]

        rows.append(
            {
                "query_id": query["id"],
                "category": query["category"],
                "answerable": query["answerable"],
                "cov_hybrid_k": coverage(gold, texts, hybrid_k),
                "cov_rerank_pool": coverage(gold, texts, reranked) if reranked else None,
                "pool_ceiling": coverage(gold, texts, pool),
            }
        )

    answerable = [r for r in rows if r["answerable"]]

    def mean(key: str, subset=answerable) -> float | None:
        vals = [r[key] for r in subset if r[key] is not None]
        return statistics.mean(vals) if vals else None

    print(f"\n{'strategy':34} {'slots':>6} {'coverage':>10}")
    print("-" * 54)
    summary = {}
    for label, key, slots in (
        (f"hybrid RRF top-{args.k} (A10)", "cov_hybrid_k", args.k),
        (f"rerank pool {args.pool} -> top-{args.k}", "cov_rerank_pool", args.k),
        (f"pool {args.pool} ceiling", "pool_ceiling", args.pool),
    ):
        m = mean(key)
        if m is None:
            continue
        summary[label] = m
        print(f"{label:34} {slots:6} {m * 100:9.1f}%")

    base, cand = mean("cov_hybrid_k"), mean("cov_rerank_pool")
    if base is not None and cand is not None:
        print(f"\n  delta (reranked - hybrid) = {(cand - base) * 100:+.2f} pts")
        up = down = same = 0
        print(f"\n  {'id':6} {'category':22} {'hybrid':>8} {'rerank':>8} {'pool cap':>9}")
        for r in answerable:
            a, b, c = r["cov_hybrid_k"], r["cov_rerank_pool"], r["pool_ceiling"]
            if b is None:
                continue
            if b > a + 1e-9:
                up += 1
            elif b < a - 1e-9:
                down += 1
            else:
                same += 1
            flag = "" if abs(b - a) < 1e-9 else ("  +" if b > a else "  -")
            print(
                f"  {r['query_id']:6} {r['category'][:22]:22} {a:8.3f} {b:8.3f} {c:9.3f}{flag}"
            )
        print(f"\n  improved {up} | worsened {down} | unchanged {same}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "corpus": str(corpus_path),
                    "chunks": len(texts),
                    "k": args.k,
                    "hybrid_pool": args.hybrid_pool,
                    "pool": args.pool,
                    "rerank_model": args.rerank_model,
                    "reproduction_check": agreement,
                    "coverage": summary,
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
