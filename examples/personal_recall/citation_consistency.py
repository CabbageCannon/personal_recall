"""Citation evidence consistency: does the cited chunk contain the text the answer quotes?

Why this exists. ``citation_metrics.py`` checks that citation *indices* are in range; ``absence_claims``
and ``attribution_screen`` check what an answer claims about the record and about other speakers.
None of them checks the obvious thing: **does the chunk a sentence cites contain the evidence for that
sentence?**

The real failure this targets, from a WeChat smoke test:

    Answer sentence : 11:26 你说“我又点了拌粉” [来源 1]
    Source 0        : 07:27 – 11:53   (contains: 11:26 我: 我又点了拌粉)
    Source 1        : 11:53 – 12:35   (cannot contain 11:26)

Every existing metric passed: the index was valid, the citation was well-formed, the answer was
correct. The *binding* was wrong. Re-running the model produced ``[来源 0]``, confirming a binding
error rather than a retrieval error.

## Why this checks quotes, and not lexical similarity

The first version scored every cited sentence against every source by character-bigram containment.
Measured on six committed arms (216 answers, ~700 sentences) it produced **6 mismatch findings, of
which about 1 held up — ~17 % precision**. Two causes were systematic and are worth recording:

* **multi-fact summary sentences** legitimately cite several sources, so "the best matching single
  source" is a meaningless comparison for them;
* bigram overlap produced **spurious winners**: a claim about 搬家 matched the line 「搬什么家」.

So the check is anchored on something much harder to fake: **a verbatim quote**. When an answer puts
text in quotation marks, that text is a claim about what the record says — and it must appear in the
chunk the answer cites. A quote found in a *different* retrieved source is an unambiguous binding
error; a quote found nowhere is reported separately as unverified, because it may be a light
paraphrase rather than an invention.

What it reports, and what it deliberately does not:

* It is a **groundedness warning, not a truth verdict** — it never says an answer is wrong.
* It is **mechanical and deterministic**, needs no gold data and no LLM judge: only the answer and the
  chunks it was given.
* Sentences asserting the record's *silence* are excluded: the evidence for "the record does not
  mention X" is an absence, so no quoted text can support it and lexical scoring guarantees a false
  positive. That class is judged by ``absence_claims.py`` instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from absence_claims import detect_absence_claims
from attribution_screen import CITE_RE, split_sentences

#: Quotation marks used in Chinese and English answers.
QUOTE_RE = re.compile(r"[“\"「『](?P<text>[^”\"」』\n]{2,})[”\"」』]")

#: A shorter quote than this is too generic to be evidence of anything. Calibrated: at 4 characters a
#: claim quoting the generic phrase 「另一个同学」 was flagged against a source that merely contained
#: that phrase; at 6 the real mismatches (14 and 9 characters) and the reported case (6) all survive.
MIN_QUOTE_CHARS = 6

OK = "ok"
MISMATCH = "mismatch"
UNVERIFIED = "unverified"


def normalize(text: str) -> str:
    """Strip punctuation and whitespace so a quote can be located inside a chat line."""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text or "")


@dataclass(frozen=True)
class BindingFinding:
    """One quoted claim whose citation does not hold up."""

    sentence: str
    quote: str
    cited: tuple[int, ...]
    verdict: str
    found_in: tuple[int, ...] = ()

    def describe(self) -> str:
        if self.verdict == MISMATCH:
            return (
                f"quoted text «{self.quote}» is not in the cited source "
                f"(Source {list(self.cited)}) but appears in Source {list(self.found_in)}"
            )
        return (
            f"quoted text «{self.quote}» is not found verbatim in any retrieved source - "
            "it may be a paraphrase, or not in the record at all"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sentence": self.sentence,
            "quote": self.quote,
            "cited": list(self.cited),
            "verdict": self.verdict,
            "found_in": list(self.found_in),
            "explanation": self.describe(),
        }


@dataclass
class ConsistencyReport:
    quotes_checked: int = 0
    sentences_checked: int = 0
    skipped_silence_claims: int = 0
    findings: list[BindingFinding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    @property
    def mismatches(self) -> list[BindingFinding]:
        return [f for f in self.findings if f.verdict == MISMATCH]

    def warnings(self) -> list[str]:
        """Only the binding mismatch is surfaced.

        ``unverified`` quotes are deliberately NOT a warning: calibration showed they are dominated by
        the model putting its own paraphrase in quotation marks (「两个方向」, 「小王推荐」,
        「开始使用/试跑」) rather than quoting the record, so surfacing them would fire on most answers
        without meaning much. They stay in ``as_dict()`` for analysis.
        """
        out = []
        if self.mismatches:
            out.append(
                f"{len(self.mismatches)} citation binding mismatch(es): quoted text is not in the "
                "cited source but is in another retrieved source"
            )
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "sentences_checked": self.sentences_checked,
            "quotes_checked": self.quotes_checked,
            "skipped_silence_claims": self.skipped_silence_claims,
            "mismatches": len(self.mismatches),
            "unverified": len([f for f in self.findings if f.verdict == UNVERIFIED]),
            "findings": [
                {
                    "sentence": f.sentence,
                    "quote": f.quote,
                    "cited": list(f.cited),
                    "verdict": f.verdict,
                    "found_in": list(f.found_in),
                    "explanation": f.describe(),
                }
                for f in self.findings
            ],
        }


def quotes_in(sentence: str) -> list[str]:
    """Verbatim spans the sentence attributes to the record."""
    return [match.group("text").strip() for match in QUOTE_RE.finditer(sentence or "")]


def check_answer(answer: str, sources: Sequence[Mapping[str, Any]]) -> ConsistencyReport:
    """Check every quoted claim against the source it cites."""
    report = ConsistencyReport()
    contents = [normalize(str(source.get("content") or "")) for source in sources]

    for sentence in split_sentences(answer or ""):
        cited = tuple(sorted({int(n) for n in CITE_RE.findall(sentence)}))
        if not cited:
            continue
        if detect_absence_claims(sentence):
            # A silence claim has no supporting text by construction; judged by absence_claims.py.
            report.skipped_silence_claims += 1
            continue

        quotes = [q for q in quotes_in(sentence) if len(normalize(q)) >= MIN_QUOTE_CHARS]
        if not quotes:
            continue
        report.sentences_checked += 1

        for quote in quotes:
            needle = normalize(quote)
            if not needle:
                continue
            report.quotes_checked += 1
            found_in = tuple(i for i, text in enumerate(contents) if needle in text)
            cited_has_it = any(i in found_in for i in cited)
            if cited_has_it:
                continue
            report.findings.append(
                BindingFinding(
                    sentence=sentence,
                    quote=quote,
                    cited=cited,
                    verdict=MISMATCH if found_in else UNVERIFIED,
                    found_in=found_in,
                )
            )
    return report
