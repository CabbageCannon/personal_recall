"""Recall-vs-k probe: is the missing gold evidence ranking-limited or representation-limited?

Offline (local BGE, no LLM, no API cost). Sweeps k and reports evidence coverage, so a
failure can be attributed correctly:

* coverage keeps rising with k  -> the evidence IS retrievable, the Top-k window is too
  narrow for multi-episode arcs. A wider window and/or a reranker is the lever, and any
  chunking change must beat that control.
* coverage plateaus low        -> the evidence is not in the dense ranking at all; the
  representation (chunking / memory model / hybrid retrieval) is the lever.

Optionally splits the report by manually-graded evidence sufficiency, which is what the
label diagnostics sidecar records (`evidence_sufficient`).

Usage:
    python probe_recall_at_k.py --dataset stress
    python probe_recall_at_k.py --dataset stress --insufficient <label_diagnostics.json>
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
DATA_DIR = BASE_DIR / "data"
sys.path.insert(0, str(BASE_DIR))

from eval_utils import evidence_matches_source  # noqa: E402
from probe_dense_difficulty import (  # noqa: E402
    DATASET_CONFIG,
    load_chunks,
    make_embedder,
)

DEFAULT_KS = (5, 10, 20, 50, 100)


def load_insufficient(path: Path | None) -> set[str]:
    """Read the ids flagged as evidence-insufficient from a label-diagnostics file."""
    if path is None or not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("queries") or payload.get("rows") or []
    return {r["query_id"] for r in rows if r.get("evidence_sufficient") is False}


def main() -> int:
    dotenv.load_dotenv()

    ap = argparse.ArgumentParser(description="Evidence coverage as a function of k.")
    ap.add_argument("--dataset", choices=sorted(DATASET_CONFIG), default="stress")
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--queries", type=Path)
    ap.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    ap.add_argument(
        "--chunking",
        choices=("fixed", "session"),
        default="fixed",
        help="retrieval unit to sweep (default: %(default)s)",
    )
    ap.add_argument("--max-session-chars", type=int, default=900)
    ap.add_argument("--insufficient", type=Path, help="label diagnostics with evidence_sufficient")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    corpus_path = args.corpus or cfg["corpus"]
    queries_path = args.queries or cfg["queries"]
    json_out = args.json_out or BASE_DIR / f"{args.dataset}_recall_at_k.json"

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    chunks = load_chunks(
        corpus_path, chunking=args.chunking, max_session_chars=args.max_session_chars
    )
    embedder = make_embedder()
    chunk_vecs = np.asarray(embedder.embed_documents(chunks), dtype=np.float32)
    question_vecs = np.asarray(
        embedder.embed_documents([q["question"] for q in queries]), dtype=np.float32
    )
    chunk_vecs /= np.linalg.norm(chunk_vecs, axis=1, keepdims=True) + 1e-12
    question_vecs /= np.linalg.norm(question_vecs, axis=1, keepdims=True) + 1e-12
    order = np.argsort(-(question_vecs @ chunk_vecs.T), axis=1)

    insufficient = load_insufficient(args.insufficient)

    def coverage_at(q_idx: int, k: int) -> float | None:
        gold = queries[q_idx].get("relevant_evidence") or []
        if not gold:
            return None
        retrieved = [chunks[i] for i in order[q_idx][:k]]
        hits = sum(
            1 for e in gold if any(evidence_matches_source(e.strip(), c) for c in retrieved)
        )
        return hits / len(gold)

    print(f"corpus : {corpus_path}")
    print(f"queries: {queries_path}")
    print(f"chunks : {len(chunks)}   queries: {len(queries)}")
    if insufficient:
        print(f"evidence-insufficient ids: {len(insufficient)}")

    groups = {
        "all answerable": [i for i, q in enumerate(queries) if q["answerable"]],
    }
    if insufficient:
        groups["evidence-INsufficient"] = [
            i for i, q in enumerate(queries) if q["id"] in insufficient
        ]
        groups["evidence-sufficient"] = [
            i
            for i, q in enumerate(queries)
            if q["answerable"] and q["id"] not in insufficient
        ]

    header = "group".ljust(24) + "".join(f"cov@{k}".ljust(9) for k in args.ks)
    print("\n" + header)
    print("-" * len(header))
    table: dict[str, dict[int, float | None]] = {}
    for name, idxs in groups.items():
        cells = []
        table[name] = {}
        for k in args.ks:
            vals = [v for v in (coverage_at(i, k) for i in idxs) if v is not None]
            mean = statistics.mean(vals) if vals else None
            table[name][k] = mean
            cells.append(("n/a" if mean is None else f"{mean * 100:.1f}%").ljust(9))
        print(name.ljust(24) + "".join(cells))

    if insufficient:
        print("\nper-query coverage for the evidence-insufficient ids:")
        print("  id     " + "".join(f"cov@{k}".ljust(9) for k in args.ks))
        for i, q in enumerate(queries):
            if q["id"] not in insufficient:
                continue
            cells = []
            for k in args.ks:
                v = coverage_at(i, k)
                cells.append(("n/a" if v is None else f"{v * 100:.0f}%").ljust(9))
            print("  " + q["id"].ljust(7) + "".join(cells))

    base = args.ks[0]
    wide = args.ks[-1]
    head = table["all answerable"]
    print("\n--- VERDICT ---")
    if head.get(base) is not None and head.get(wide) is not None:
        gain = (head[wide] - head[base]) * 100
        print(f"  coverage {head[base] * 100:.1f}% @k={base} -> {head[wide] * 100:.1f}% @k={wide} "
              f"(+{gain:.1f} points)")
        print("  large gain = ranking/window limited; a chunking change must beat the wider-k control"
              if gain > 20
              else "  small gain = representation limited; changing chunking/memory is the lever")

    json_out.write_text(
        json.dumps(
            {
                "corpus": str(corpus_path),
                "queries": str(queries_path),
                "chunks": len(chunks),
                "ks": list(args.ks),
                "coverage": {name: {str(k): v for k, v in row.items()} for name, row in table.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  written to {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
