"""Attribution screen: is a claim about the user actually resting on someone else's own situation?

Why this exists (R11, Phase 12). A12 fixed cross-speaker attribution with a prompt clause, but a
prompt is not an enforcement mechanism: if a future prompt or model change reintroduces the failure,
every existing metric stays green. This is the guard.

The failure it targets (R10, measured in Phase 11). The system reported 张三's own course project as
the user's earliest database, and 同学A's own repurposed PC as the user's local storage. In both cases
the cited line genuinely exists and is quoted correctly — the attribution is what is wrong.

The signal. In this corpus the user is a participant literally named 我. So a line can be classified
mechanically:

* speaker is 我              -> the user's own statement, safe to use as a fact about the user;
* speaker is someone else and the line addresses 你 -> someone speaking to/about the user, safe;
* speaker is someone else, the line says 我 and does not address 你 -> **that person describing their
  own situation**, which is NOT evidence about the user.

That third class is 30.9 % of third-party lines in corpus v2 — common enough to matter, separable
enough to detect. For every sentence in an answer that carries a citation, the screen finds the
cited chunk's best-matching line by character-bigram overlap and flags the sentence when that line
falls in the third class.

Scope, stated honestly. Like `absence_claims.py` this is a **screen**, not a verdict: some flagged
sentences are legitimate (a question can genuinely ask what another person said, which makes their
own-situation line the correct evidence). It reports candidates for adjudication and its precision is
measured, not assumed. Its useful property is recall: it must catch the known R10 defects.

Usage:
    python attribution_screen.py --results stress_v2_attrib_results.json
    python attribution_screen.py --results ... --verbose
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

LINE_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\s*([^:]+):\s*(.+)$")
CITE_RE = re.compile(r"\[来源\s*(\d+)\]")
SENTENCE_SPLIT = re.compile(r"(?<=[。！？\n])")
#: Minimum share of the sentence's bigrams that must appear in a line for it to count as its support.
MATCH_THRESHOLD = 0.25

USER = "我"
#: Third-person references that count as explicit attribution to somebody other than the user.
PRONOUNS = ("他", "她", "他们", "她们")

#: Nested reported speech: the speaker is relaying a *third* party's account ("我表哥说…",
#: "我们组那个同学说…"). The 我 in such a line belongs to the relayed person, not to the speaker,
#: so the line is neither the user's own statement nor cleanly "the speaker's own situation".
RELAY_RE = re.compile(
    r"(表哥|表姐|我们组|我同学|我朋友|我室友|别人|同事|同学|朋友|老师|导师)"
    r"[^。！？\n]{0,10}(说|告诉|推荐|提到|讲|让)"
)

#: When True, relayed lines are excluded from the flagged class. Measured trade-off: this removes
#: every false positive in the current sample but also loses the `s020` hit, so the default keeps
#: them in and relies on adjudication -- a guard should err toward recall.
EXCLUDE_RELAYED = False


def bigrams(text: str) -> set[str]:
    """Character bigrams over CJK plus ASCII words — enough to match a claim to its source line."""
    cleaned = re.sub(r"\[来源\s*\d+\]", "", text)
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "", cleaned)
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)} if len(cleaned) > 1 else set()


@dataclass(frozen=True)
class SourceLine:
    speaker: str
    body: str

    @property
    def kind(self) -> str:
        """``user`` | ``user_directed`` | ``relayed`` | ``own_situation`` | ``neutral``."""
        if self.speaker == USER:
            return "user"
        if "你" in self.body:
            return "user_directed"
        if RELAY_RE.search(self.body):
            return "relayed"
        if "我" in self.body:
            return "own_situation"
        return "neutral"


def parse_lines(chunk_text: str) -> list[SourceLine]:
    lines = []
    for raw in chunk_text.split("\n"):
        match = LINE_RE.match(raw.strip())
        if match:
            lines.append(SourceLine(speaker=match.group(2).strip(), body=match.group(3).strip()))
    return lines


def best_match(sentence: str, lines: list[SourceLine]) -> tuple[SourceLine | None, float]:
    target = bigrams(sentence)
    if not target or not lines:
        return None, 0.0
    best, score = None, 0.0
    for line in lines:
        shared = target & bigrams(line.body)
        ratio = len(shared) / len(target)
        if ratio > score:
            best, score = line, ratio
    return best, score


def split_sentences(answer: str) -> list[str]:
    """Split into claim-bearing units, keeping citation markup attached to its sentence.

    Answers are written as ``...。[来源 5]``, so a naive split on ``。`` orphans the citation into a
    fragment of its own -- which then matches nothing and silently defeats the whole screen (this
    cost the screen the `s020` defect before it was fixed). Fragments that are nothing but citations
    are therefore re-attached to the sentence they belong to.
    """
    sentences: list[str] = []
    for raw in SENTENCE_SPLIT.split(answer or ""):
        text = raw.strip()
        if not text:
            continue
        if sentences and not CITE_RE.sub("", text).strip(" \t-*·`"):
            sentences[-1] = f"{sentences[-1]} {text}"
        else:
            sentences.append(text)
    return sentences


def screen_answer(answer: str, sources: list[dict]) -> list[dict]:
    """Return one result per cited sentence that rests on someone else's own situation.

    A sentence is considered **properly attributed** -- and therefore not flagged, however its
    supporting line reads -- when it, or the sentence right before it, names the speaker or refers to
    them with 他/她/他们. Answers routinely write "他说系统还没部署" under a bullet whose previous line
    established who "he" is; that is correct attribution, and flagging it was this screen's largest
    false-positive source before the context window was added.

    The residual case is a sentence that borrows such a line with *no* attribution at all, which is
    exactly the R10 defect.
    """
    findings = []
    sentences = split_sentences(answer)
    for position, text in enumerate(sentences):
        indices = [int(n) for n in CITE_RE.findall(text)]
        if not indices:
            continue
        context = text if position == 0 else f"{sentences[position - 1]} {text}"
        for index in indices:
            if not 0 <= index < len(sources):
                continue
            lines = parse_lines(sources[index].get("content") or "")
            line, score = best_match(text, lines)
            if line is None or score < MATCH_THRESHOLD:
                continue
            attributed = line.speaker in context or any(p in context for p in PRONOUNS)
            flagged_kind = line.kind == "own_situation" or (
                line.kind == "relayed" and not EXCLUDE_RELAYED
            )
            if flagged_kind and not attributed:
                findings.append(
                    {
                        "sentence": text,
                        "citation": index,
                        "matched_line_speaker": line.speaker,
                        "matched_line": line.body,
                        "match_score": round(score, 3),
                    }
                )
                break
    return findings


def screen(results_path: Path) -> dict:
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    per_query = []
    for row in rows:
        sources = [
            {"rank": s.get("rank"), "content": s.get("content") or "", "chunk_index": s.get("chunk_index")}
            for s in row.get("retrieved_sources") or []
        ]
        findings = screen_answer(row.get("answer") or "", sources)
        per_query.append(
            {
                "query_id": row["query_id"],
                "n_flagged": len(findings),
                "flagged": findings,
                "risk": bool(findings),
            }
        )
    return {
        "results": str(results_path),
        "queries": len(per_query),
        "queries_with_flags": [r["query_id"] for r in per_query if r["risk"]],
        "total_flags": sum(r["n_flagged"] for r in per_query),
        "per_query": per_query,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    report = screen(args.results)
    print(f"results: {Path(report['results']).name}")
    print(f"queries                     : {report['queries']}")
    print(f"queries with an attribution flag: {len(report['queries_with_flags'])}  "
          f"{report['queries_with_flags']}")
    print(f"total flagged sentences     : {report['total_flags']}")

    if args.verbose:
        for row in report["per_query"]:
            if not row["risk"]:
                continue
            print(f"\n  {row['query_id']}")
            for flag in row["flagged"]:
                print(f"    sentence : {flag['sentence'][:120]}")
                print(f"    cite [{flag['citation']}] speaker {flag['matched_line_speaker']} "
                      f"(score {flag['match_score']})")
                print(f"    line     : {flag['matched_line'][:120]}")

    if args.json_out:
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwritten to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
