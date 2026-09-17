"""Does the shipped product retrieve the same evidence the evaluation measured?

Every phase of this project evaluated `run_baseline.py`. The thing a user runs is `recall.py`, which
is a *different code path*: its own brain construction, its own prompt registration, its own corpus
handling. If the two diverge, then the entire evaluation — A12's 31/36, the ablation results, the
groundedness metrics — describes something other than the product.

This harness closes that gap empirically. It drives `recall.build_session()`, the product's own
construction function, over the 36 stress queries on corpus v2 and compares the retrieved chunk
indices against the recorded A12 arm (`stress_v2_attrib_results.json`).

Retrieval runs under `--workflow no-rewrite`, so it is deterministic: the comparison is exact, not
statistical. A mismatch is a real divergence and is reported as such.

Usage::

    python verify_product_parity.py                        # all 36 queries
    python verify_product_parity.py --limit 6              # a fast subset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

import dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
sys.path.insert(0, str(BASE_DIR))

from quivr_core.rag.entities.chat import ChatHistory  # noqa: E402

from groundedness import assess  # noqa: E402
from recall import (  # noqa: E402
    DEFAULT_ANSWER_PROMPT,
    DEFAULT_HYBRID_POOL,
    DEFAULT_K,
    ENV_PATH,
    build_session,
)
from run_baseline import serialize_sources  # noqa: E402

RECORDED_ARM = BASE_DIR / "stress_v2_attrib_results.json"
CORPUS = DATA_DIR / "stress_chats_v2.txt"
QUERIES = DATA_DIR / "stress_queries.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="only the first N queries")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    dotenv.load_dotenv(ENV_PATH if ENV_PATH.exists() else None)

    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    recorded = {r["query_id"]: r for r in json.loads(RECORDED_ARM.read_text(encoding="utf-8"))}
    if args.limit:
        queries = queries[: args.limit]

    print(f"product path : recall.build_session (k={DEFAULT_K}, pool={DEFAULT_HYBRID_POOL}, "
          f"prompt={DEFAULT_ANSWER_PROMPT})")
    print(f"recorded arm : {RECORDED_ARM.name}")
    print(f"corpus       : {CORPUS.name}")
    print(f"queries      : {len(queries)}\n")

    brain, retrieval_config = build_session(
        CORPUS,
        k=DEFAULT_K,
        hybrid_pool=DEFAULT_HYBRID_POOL,
        workflow="no-rewrite",
        answer_prompt=DEFAULT_ANSWER_PROMPT,
    )

    identical = same_set = differing = 0
    answer_identical = 0
    groundedness_flags = []
    rows = []
    for query in queries:
        qid = query["id"]
        response = brain.ask(
            run_id=uuid4(),
            question=query["question"],
            retrieval_config=retrieval_config,
            chat_history=ChatHistory(chat_id=uuid4(), brain_id=brain.id),
        )
        sources = serialize_sources(response.metadata.sources if response.metadata else [])
        got = [s.get("chunk_index") for s in sources]
        want = [s.get("chunk_index") for s in recorded[qid].get("retrieved_sources") or []]

        if got == want:
            identical += 1
            verdict = "identical"
        elif set(got) == set(want):
            same_set += 1
            verdict = "same set, different order"
        else:
            differing += 1
            verdict = "DIVERGED"

        answer = response.answer or ""
        if answer.strip() == (recorded[qid].get("answer") or "").strip():
            answer_identical += 1
        report = assess(answer, sources)
        if report.has_caveats:
            groundedness_flags.append((qid, len(report.attribution_flags), len(report.absence_claims)))

        rows.append(
            {
                "query_id": qid,
                "retrieval": verdict,
                "recorded_chunks": want,
                "product_chunks": got,
                "answer_chars": len(answer),
                "citations": len(report.citations),
            }
        )
        mark = "" if verdict == "identical" else "   <<<"
        print(f"  {qid}  {verdict:24} chunks={len(got):2} citations={len(report.citations):2}{mark}")
        if verdict == "DIVERGED":
            print(f"      recorded: {want}")
            print(f"      product : {got}")

    total = len(rows)
    print(f"\n--- VERDICT ---")
    print(f"  retrieval identical      : {identical}/{total}")
    print(f"  same set, different order: {same_set}/{total}")
    print(f"  DIVERGED                 : {differing}/{total}")
    print(f"  byte-identical answers   : {answer_identical}/{total} (informational: generation is "
          "not guaranteed reproducible)")
    print(f"  answers with caveats     : {len(groundedness_flags)}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {
                    "recorded_arm": str(RECORDED_ARM),
                    "corpus": str(CORPUS),
                    "queries": total,
                    "retrieval_identical": identical,
                    "retrieval_same_set": same_set,
                    "retrieval_diverged": differing,
                    "answers_identical": answer_identical,
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  written to {args.json_out}")

    return 0 if differing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
