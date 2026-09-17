"""Absence-claim screen: answers that assert the *record* is silent.

Why this exists (risk R9, Phase 9). Widening the retrieval window introduced a failure no
existing metric can see. `citation_metrics.py` measures whether a citation supports a claim
about the user; `unsupported_claim` measures fabrication about the user's life. But an answer
can instead claim **the record contains nothing about X** — and when X is sitting in the corpus
just outside the retrieved window, that claim is false while every citation in it remains
valid. It is the one failure a user experiences as the system being *confidently* wrong.

Scope, stated honestly. Detecting the *phrasing* of an absence claim is reliable and is what
this module automates. Deciding whether a given absence claim is **true** requires knowing what
the claim is about, which is not something a regex can do. So this module does not score
correctness. It produces:

* every absence claim an answer makes, with its sentence;
* a **risk screen**: queries where the answer asserts absence *and* the retrieval was
  incomplete for that query (gold evidence exists in the corpus but was not retrieved), i.e.
  exactly the condition under which a false absence becomes possible.

The screen is a filter for adjudication, not a verdict — measured precision is reported in
PROJECT_STATUS.md, and the flagged sentences are adjudicated separately. A high flag count is
not itself a defect: honest scoping ("记录没有说明候补最后是否成功") is desirable behaviour and
is deliberately *not* suppressed.

Usage:
    python absence_claims.py --results stress_v2_a11_k20_results.json
    python absence_claims.py --results stress_v2_a11_k20_results.json --json-out out.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

#: Sentences are the unit of reporting: a claim is only meaningful in its context.
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？\n])")

#: Phrasings that assert the record's silence. Each is anchored either on an explicit
#: record/context noun or on the "absence of a mention" verb pair, so that ordinary negations
#: about the world ("你当时没有答应", "没有试成") do NOT match — those are claims about events,
#: not about the record.
PATTERNS: tuple[re.Pattern[str], ...] = (
    # A record noun immediately followed by an absence marker. This catches the most important
    # shape -- "记录中没有 2024 年 10 月的直接对话" -- where absence trails a noun phrase rather
    # than a mention-verb, and it is what made the first version of this detector miss the very
    # case (s030) the metric exists for. The record noun must be adjacent, so ordinary negation
    # ("你当时没有答应") cannot match.
    re.compile(r"(?:记录|对话|上下文|资料|聊天记录)(?:里|中|内|当中)?(?:都|也|仍|还是|则)?(?:没有|未|不存在|缺少)"),
    # 记录/上下文/资料 + absence + mention-verb, in either order
    re.compile(r"(?:记录|对话|上下文|资料|聊天记录)(?:里|中|内|当中)?[^。！？\n]{0,10}(?:没有|未|不曾|并未)[^。！？\n]{0,10}(?:提到|出现|说明|交代|写明|写|记录|给出|明确|显示|提及|明说)"),
    re.compile(r"(?:没有|未|不曾|并未)(?:提到|出现|说明|交代|提及|明确说明|写明)[^。！？\n]{0,14}(?:记录|上下文|对话)"),
    re.compile(r"(?:提供的|现有|给定|所给|上述|上下文)[^。！？\n]{0,8}(?:记录|对话|上下文|资料)(?:里|中|内)?[^。！？\n]{0,10}(?:没有|未|不存在|缺少)"),
    # "记录未说明的部分" / "记录中没有后续结果" style
    re.compile(r"(?:记录|上下文|资料)(?:里|中|内)?[^。！？\n]{0,8}(?:没有|未|不存在)[^。！？\n]{0,10}(?:结果|信息|内容|说明|后续|更多|任何)"),
    # record-anchored inability to conclude
    re.compile(r"无法(?:从|根据)?[^。！？\n]{0,8}(?:记录|上下文|对话|资料)[^。！？\n]{0,10}(?:确定|判断|确认|得知|查证)"),
    # "没有记录显示/说明" and the bare "没有记录 X" noun usage ("没有记录最终是否候补成功").
    # The negative lookahead keeps the verb usage ("我没有记录了") out.
    re.compile(r"(?:没有|无)(?:任何)?记录(?![了过])"),
    re.compile(r"(?:之前|之后|此后|后面)(?:的)?记录(?:里|中)?[^。！？\n]{0,8}(?:没有|未)[^。！？\n]{0,10}(?:提到|说明|显示|记录|出现)"),
)


@dataclass(frozen=True)
class AbsenceClaim:
    """One sentence in which the answer asserts the record's silence."""

    sentence: str
    matched: str
    start: int


def detect_absence_claims(answer: str) -> list[AbsenceClaim]:
    """Return the absence claims in one answer, one per sentence (deduplicated)."""
    claims: list[AbsenceClaim] = []
    offset = 0
    seen: set[str] = set()
    for sentence in _SENTENCE_SPLIT.split(answer or ""):
        text = sentence.strip()
        if text:
            for pattern in PATTERNS:
                match = pattern.search(text)
                if match and text not in seen:
                    seen.add(text)
                    claims.append(
                        AbsenceClaim(sentence=text, matched=match.group(0), start=offset)
                    )
                    break
        offset += len(sentence)
    return claims


def load_queries(path: Path | None = None) -> dict[str, dict]:
    payload = json.loads((path or DATA_DIR / "stress_queries.json").read_text(encoding="utf-8"))
    return {q["id"]: q for q in payload}


def screen(results_path: Path, queries_path: Path | None = None, corpus_path: Path | None = None) -> dict:
    """Screen one arm's results for absence claims made over incomplete retrieval.

    ``corpus_path`` matters. A false absence is only possible if the thing declared absent is
    actually *in the record* — so without the corpus the screen can only see "gold was missed",
    which over-flags by construction (on an evidence-ablated corpus the gold is genuinely gone
    and every absence claim is correct). Pass the corpus used by the run to have the screen
    confirm the missed gold is really still in the record.
    """
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    queries = load_queries(queries_path)
    corpus_text = None
    if corpus_path is not None:
        corpus_text = corpus_path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")

    per_query = []
    for row in rows:
        qid = row["query_id"]
        query = queries.get(qid, {})
        answer = row.get("answer") or ""
        claims = detect_absence_claims(answer)

        gold = query.get("relevant_evidence") or []
        retrieved = [s.get("content") or "" for s in row.get("retrieved_sources") or []]
        missing_gold = [line for line in gold if not any(line in c for c in retrieved)]
        # The precondition for a false absence: the missing line is still in the record.
        present_in_corpus = (
            [line for line in missing_gold if line in corpus_text]
            if corpus_text is not None
            else list(missing_gold)
        )

        per_query.append(
            {
                "query_id": qid,
                "category": query.get("category"),
                "n_claims": len(claims),
                "claims": [c.sentence for c in claims],
                "evidence_coverage": row.get("evidence_coverage"),
                "missing_gold_lines": missing_gold,
                "missing_gold_present_in_corpus": present_in_corpus,
                # The screen: absence asserted AND the retrieval missed gold that the record
                # still contains.
                "risk": bool(claims) and bool(present_in_corpus),
            }
        )

    with_claims = [r for r in per_query if r["n_claims"]]
    return {
        "results": str(results_path),
        "corpus": str(corpus_path) if corpus_path else None,
        "corpus_checked": corpus_text is not None,
        "queries": len(per_query),
        "answers_with_absence_claim": len(with_claims),
        "total_claims": sum(r["n_claims"] for r in per_query),
        "risk_candidates": [r["query_id"] for r in per_query if r["risk"]],
        "per_query": per_query,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--queries", type=Path)
    ap.add_argument(
        "--corpus",
        type=Path,
        help="the corpus the run used; confirms the missed gold is really still in the record "
        "(strongly recommended — without it the screen over-flags on ablated corpora)",
    )
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--verbose", action="store_true", help="print every claim sentence")
    args = ap.parse_args()

    report = screen(args.results, args.queries, args.corpus)

    print(f"results: {Path(report['results']).name}")
    print(f"corpus checked against: {Path(report['corpus']).name if report['corpus'] else 'NO (risk is an upper bound)'}")
    print(f"answers with an absence claim : {report['answers_with_absence_claim']}/{report['queries']}")
    print(f"total absence claims          : {report['total_claims']}")
    print(f"risk candidates (claim + missed gold still in the corpus): "
          f"{len(report['risk_candidates'])}  {report['risk_candidates']}")

    if args.verbose:
        print("\nclaims by query:")
        for row in report["per_query"]:
            if not row["n_claims"]:
                continue
            flag = "  <== RISK" if row["risk"] else ""
            print(f"\n  {row['query_id']} [{row['category']}] cov={row['evidence_coverage']}{flag}")
            for sentence in row["claims"]:
                print(f"    » {sentence[:150]}")
            for line in row["missing_gold_lines"]:
                print(f"    missing gold: {line[:110]}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwritten to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
