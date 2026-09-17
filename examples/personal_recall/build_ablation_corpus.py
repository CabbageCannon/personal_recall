"""Build an evidence-ablated corpus: the decisive lines are gone, the distractors are not.

Why. The project's product promise is "No Evidence = No Memory Claim", but it is currently tested
by exactly **two** deliberately unanswerable queries. Two data points cannot establish that a
system declines to answer rather than pattern-completing. This builder creates the corpus that
tests it at scale.

What it does. Every episode (blank-line-separated block) that contains any gold evidence line for
any of the 36 queries is removed. What survives is the distractor pack: hundreds of same-topic,
different-fact, gold-shaped sessions about the very entities the questions ask about. A question
like "which database does my current project use" still retrieves twenty sessions discussing
databases — just none of them stating the answer.

That is the false-memory trap. The failure it catches is an answer that sounds right because the
topic is saturated, while nothing in the record supports it.

Usage::

    python build_ablation_corpus.py                 # build into data/
    python build_ablation_corpus.py --check-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SOURCE_NAME = "stress_chats_v2.txt"
QUERIES_NAME = "stress_queries.json"
OUT_NAME = "stress_chats_v2_no_gold.txt"

TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] ([\w\u4e00-\u9fff]+): (.+)$")

sys.path.insert(0, str(BASE_DIR))
from memory.events import parse_txt_events  # noqa: E402
from memory.sessions import SessionConfig, build_sessions  # noqa: E402


def normalise(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=DATA_DIR / SOURCE_NAME)
    ap.add_argument("--queries", type=Path, default=DATA_DIR / QUERIES_NAME)
    ap.add_argument("--out", type=Path, default=DATA_DIR / OUT_NAME)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    failures: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
        if not ok:
            failures.append(label)

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    gold = [line.strip() for q in queries for line in (q.get("relevant_evidence") or [])]
    source = normalise(args.source.read_text(encoding="utf-8"))
    episodes = [b.strip() for b in re.split(r"\n\s*\n", source) if b.strip()]

    kept: list[str] = []
    removed: list[str] = []
    for episode in episodes:
        if any(line in episode for line in gold):
            removed.append(episode)
        else:
            kept.append(episode)

    ablated = "\n\n".join(kept) + "\n"

    print(f"source : {args.source.name}")
    print(f"  episodes {len(episodes)} -> kept {len(kept)}, removed {len(removed)} "
          f"({len(removed) / len(episodes) * 100:.0f}%)")
    print(f"  gold lines removed with them: "
          f"{sum(1 for line in gold if line in source and line not in ablated)}/{len(gold)}")
    print("\nchecks:")

    survivors = [line for line in gold if line in ablated]
    check(not survivors, "no gold evidence line survives", f"{len(survivors)} left")
    check(not any(line in ablated for line in gold), "no gold line is a substring of the output")

    lines = [line for line in ablated.split("\n") if line.strip()]
    check(bool(lines) and all(TS_RE.match(line) for line in lines),
          "every surviving line is a well-formed message")
    stamps = [TS_RE.match(line).group(1) for line in lines]
    check(len(stamps) == len(set(stamps)), "timestamps still unique")
    check(stamps == sorted(stamps), "still chronologically ordered")

    events = parse_txt_events(ablated).events
    chunks = build_sessions(events, SessionConfig())
    per_chunk = len(events) / len(chunks) if chunks else 0
    check(len(chunks) >= 60, "enough surviving chunks to be a real corpus", f"{len(chunks)} chunks")
    check(per_chunk >= 5, "no fragmentation", f"{per_chunk:.1f} msgs/chunk")

    print(f"\n  ablated corpus: {len(ablated)} chars, {len(lines)} messages, "
          f"{len(kept)} episodes, {len(chunks)} chunks")
    print(f"  (source was  {len(source)} chars, "
          f"{sum(1 for line in source.split(chr(10)) if TS_RE.match(line))} messages, "
          f"{len(episodes)} episodes)")
    print(f"  answerable queries whose evidence is now absent: "
          f"{sum(1 for q in queries if q.get('answerable') and q.get('relevant_evidence'))}")

    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    if args.check_only:
        print("\n--check-only: nothing written")
        return 0

    args.out.write_text(ablated, encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
