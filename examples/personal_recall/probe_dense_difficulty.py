"""Offline dense-retrieval difficulty probe (no LLM, no API cost).

Answers the Phase-0.5 gate question: *is this dataset actually hard for the frozen
baseline retriever?* It reproduces the baseline's retrieval stage exactly:

    recursive_character_splitter(400 / 100)  -> chunks
    BGE-small-zh-v1.5 (local, normalized)    -> embeddings
    cosine Top-K                             -> retrieved evidence
    eval_utils.evidence_matches_source       -> hit / coverage

The only thing it does NOT reproduce is the LLM generation step and any query
rewrite the LangGraph flow performs, so treat it as an upper bound on retrieval
quality: if the raw question already retrieves the gold evidence at rank 1, the
paid baseline will very likely pass too.

Usage:
    python probe_dense_difficulty.py                     # stress dataset defaults
    python probe_dense_difficulty.py --corpus data/chats.txt --queries data/queries.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import dotenv
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

from eval_utils import evidence_matches_source, strict_evidence_matches_source

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

EMBEDDING_MODEL_PATH = Path(r"D:\AIModels\bge-small-zh-v1.5")
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

DATASET_CONFIG = {
    "small": {"corpus": DATA_DIR / "chats.txt", "queries": DATA_DIR / "queries.json"},
    "stress": {
        "corpus": DATA_DIR / "stress_chats.txt",
        "queries": DATA_DIR / "stress_queries.json",
    },
}


def load_chunks(corpus_path: Path) -> list[str]:
    """Chunk the corpus with the EXACT baseline splitter, imported from quivr_core."""
    from langchain_core.documents import Document

    from quivr_core.processor.implementations.simple_txt_processor import (
        recursive_character_splitter,
    )

    text = corpus_path.read_text(encoding="utf-8")
    docs = recursive_character_splitter(
        Document(page_content=text), CHUNK_SIZE, CHUNK_OVERLAP
    )
    return [d.page_content for d in docs]


def make_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=str(EMBEDDING_MODEL_PATH),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def first_gold_rank(
    gold: list[str], ranking: list[int], chunk_texts: list[str], tolerant: bool = True
) -> tuple[int | None, int | None, int]:
    """Position of the first gold-holding chunk in the similarity ranking.

    Returns (rank_in_ranking, chunk_index_in_file, chunks_holding_gold) where both
    positions are 1-based and None when no chunk holds any gold evidence.
    """
    matcher = evidence_matches_source if tolerant else strict_evidence_matches_source
    holders = {
        i
        for i, content in enumerate(chunk_texts, start=1)
        if any(matcher(e.strip(), content) for e in gold)
    }
    if not holders:
        return None, None, 0
    for rank, chunk_idx in enumerate(ranking, start=1):
        if chunk_idx + 1 in holders:
            return rank, chunk_idx + 1, len(holders)
    return None, None, len(holders)


def main() -> int:
    dotenv.load_dotenv()

    ap = argparse.ArgumentParser(description="Offline dense retrieval difficulty probe.")
    ap.add_argument("--dataset", choices=sorted(DATASET_CONFIG), default="stress")
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--queries", type=Path)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    corpus_path = args.corpus or cfg["corpus"]
    queries_path = args.queries or cfg["queries"]
    json_out = args.json_out or BASE_DIR / f"{args.dataset}_dense_probe.json"

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    chunk_texts = load_chunks(corpus_path)
    print(f"corpus  : {corpus_path}")
    print(f"queries : {queries_path}")
    print(f"chunks  : {len(chunk_texts)} (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    embedder = make_embedder()
    chunk_vecs = np.asarray(embedder.embed_documents(chunk_texts), dtype=np.float32)
    question_vecs = np.asarray(
        embedder.embed_documents([q["question"] for q in queries]), dtype=np.float32
    )
    # normalize_embeddings=True already produced unit vectors; be explicit anyway.
    chunk_vecs /= np.linalg.norm(chunk_vecs, axis=1, keepdims=True) + 1e-12
    question_vecs /= np.linalg.norm(question_vecs, axis=1, keepdims=True) + 1e-12

    sims = question_vecs @ chunk_vecs.T
    order = np.argsort(-sims, axis=1)
    top_k = order[:, : args.k]

    rows = []
    for qi, query in enumerate(queries):
        retrieved = [chunk_texts[i] for i in top_k[qi]]
        scores = [float(sims[qi, i]) for i in top_k[qi]]
        gold = query.get("relevant_evidence") or []
        answerable = query.get("answerable", True)

        if not gold:
            hit = coverage = strict_hit = None
            gold_rank = gold_chunk_index = holders = None
        else:
            tolerant_hits = sum(
                1 for e in gold if any(evidence_matches_source(e.strip(), c) for c in retrieved)
            )
            strict_hits = sum(
                1
                for e in gold
                if any(strict_evidence_matches_source(e.strip(), c) for c in retrieved)
            )
            hit = tolerant_hits > 0
            coverage = tolerant_hits / len(gold)
            strict_hit = strict_hits > 0
            gold_rank, gold_chunk_index, holders = first_gold_rank(
                gold, [int(i) for i in order[qi]], chunk_texts
            )

        rows.append(
            {
                "query_id": query["id"],
                "category": query.get("category"),
                "answerable": answerable,
                "question": query["question"],
                "hit_at_k": hit,
                "coverage": coverage,
                "strict_hit_at_k": strict_hit,
                "gold_rank_in_ranking": gold_rank,
                "gold_chunk_index": gold_chunk_index,
                "chunks_holding_gold": holders,
                "top1_score": round(scores[0], 4),
                "top1_chunk": retrieved[0][:160] if retrieved else "",
            }
        )

    answerable_rows = [r for r in rows if r["answerable"] is not False]
    hits = [r for r in answerable_rows if r["hit_at_k"]]
    covs = [r["coverage"] for r in answerable_rows if r["coverage"] is not None]
    ranks = [
        r["gold_rank_in_ranking"]
        for r in answerable_rows
        if r["gold_rank_in_ranking"] is not None
    ]

    print("\n--- overall (answerable only) ---")
    print(f"Hit@{args.k}          : {len(hits)}/{len(answerable_rows)} = "
          f"{(len(hits) / len(answerable_rows) if answerable_rows else 0) * 100:.1f}%")
    print(f"avg coverage  : {(statistics.mean(covs) if covs else 0) * 100:.1f}%")
    if ranks:
        print(f"gold evidence first appears at dense rank min/median/max : "
              f"{min(ranks)} / {statistics.median(ranks)} / {max(ranks)}  (of {len(chunk_texts)} chunks)")
        print(f"queries whose gold is already inside Top-{args.k} : "
              f"{sum(1 for r in ranks if r <= args.k)}/{len(ranks)}")
        print(f"queries whose gold is at dense rank #1 : {sum(1 for r in ranks if r == 1)}/{len(ranks)}")

    print("\n--- per category ---")
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"] or "?"].append(r)
    print(f"{'category':24s} {'n':>3} {'hit@K':>7} {'coverage':>9} {'rank=1':>7}")
    for cat, items in sorted(by_cat.items()):
        ans = [r for r in items if r["answerable"] is not False]
        if not ans:
            print(f"{cat:24s} {len(items):>3} {'n/a':>7} {'n/a':>9} {'n/a':>7}")
            continue
        h = [r for r in ans if r["hit_at_k"]]
        c = [r["coverage"] for r in ans if r["coverage"] is not None]
        r1 = sum(1 for r in ans if r["gold_rank_in_ranking"] == 1)
        print(
            f"{cat:24s} {len(items):>3} "
            f"{(len(h) / len(ans) * 100):>6.1f}% "
            f"{(statistics.mean(c) * 100 if c else 0):>8.1f}% "
            f"{r1:>4}/{len(ans)}"
        )

    misses = [r for r in answerable_rows if not r["hit_at_k"]]
    partial = [r for r in answerable_rows if r["hit_at_k"] and (r["coverage"] or 0) < 1.0]
    print(f"\n--- retrieval MISSES ({len(misses)}) ---")
    for r in misses:
        print(
            f"  {r['query_id']} [{r['category']}] gold at dense rank "
            f"{r['gold_rank_in_ranking']} :: {r['question'][:60]}"
        )
    print(f"\n--- PARTIAL evidence ({len(partial)}) ---")
    for r in partial:
        print(
            f"  {r['query_id']} [{r['category']}] cov={r['coverage']:.2f} "
            f"holders={r['chunks_holding_gold']} :: {r['question'][:60]}"
        )

    trivial = [
        r
        for r in answerable_rows
        if r["gold_rank_in_ranking"] == 1 and (r["coverage"] or 0) == 1.0
    ]
    print(
        f"\n--- VERDICT ---\n  queries already perfectly retrieved by raw dense Top-{args.k}: "
        f"{len(trivial)}/{len(answerable_rows)}"
    )
    if answerable_rows and len(hits) == len(answerable_rows):
        print("  WARNING: dense retrieval already scores 100% - the dataset may be too easy;")
        print("           add stronger distractors / competing states before trusting it.")
    elif answerable_rows and len(hits) / len(answerable_rows) > 0.9:
        print("  NOTE: dense retrieval is still >90% - expect a weak signal; inspect the")
        print("        non-miss failures (partial evidence, wrong state) in the paid run.")
    else:
        print("  OK: dense retrieval already fails on part of the corpus - good stress signal.")

    json_out.write_text(
        json.dumps(
            {
                "corpus": str(corpus_path),
                "queries": str(queries_path),
                "chunks": len(chunk_texts),
                "k": args.k,
                "totals": {
                    "answerable": len(answerable_rows),
                    "hit_at_k": len(hits) / len(answerable_rows) if answerable_rows else None,
                    "avg_coverage": statistics.mean(covs) if covs else None,
                    "gold_rank_1": sum(1 for r in ranks if r == 1),
                    "gold_rank_median": statistics.median(ranks) if ranks else None,
                },
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  probe written to {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
