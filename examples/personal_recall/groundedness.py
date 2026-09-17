"""Gold-free groundedness report for a single answer.

Why this exists. The project now has three groundedness instruments, and every one of them was built
for the *evaluation*, where the gold evidence lines are known. A user asking a real question has no
gold, so none of it reaches them: `recall.py` prints evidence cards but never says "this answer
claims the record is silent about something" or "this answer may be resting on someone else's
situation". Those are exactly the two defects the last four phases found (R9, R10).

What can be checked without gold:

* **citation integrity** — how many citations, and whether any index is out of range;
* **record-silence claims** — sentences asserting the record does NOT contain something. Without gold
  the claim cannot be *judged*, only surfaced: an answer that says "the record never mentions X" is
  worth the reader's attention precisely because the reader may know X exists.
* **attribution flags** — whether a claim rests on a line where somebody else describes their own
  situation (R10). Fully checkable from the answer and its retrieved sources alone.

This module therefore reports **caveats, not verdicts**: it never says an answer is wrong, because
without gold it cannot know. The evaluation's job is to measure whether these caveats are worth
showing, and that is measured in PROJECT_STATUS.md Phase 14.

Usage:
    from groundedness import assess
    report = assess(answer, sources)
    print(report.summary_line())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from absence_claims import detect_absence_claims
from attribution_screen import screen_answer
from citation_metrics import parse_citations


@dataclass(frozen=True)
class GroundednessReport:
    """What can honestly be said about one answer without gold evidence."""

    n_sources: int
    citations: tuple[int, ...]
    invalid_citations: int
    absence_claims: tuple[str, ...]
    attribution_flags: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    @property
    def has_caveats(self) -> bool:
        return bool(self.absence_claims or self.attribution_flags or self.invalid_citations)

    @property
    def uncited(self) -> bool:
        return not self.citations

    def warnings(self) -> list[str]:
        """Human-readable caveats, strongest first."""
        out: list[str] = []
        if self.invalid_citations:
            out.append(
                f"{self.invalid_citations} citation(s) point outside the retrieved sources"
            )
        if self.uncited:
            out.append("this answer cites no source at all")
        if self.attribution_flags:
            speakers = sorted({str(f["matched_line_speaker"]) for f in self.attribution_flags})
            out.append(
                f"{len(self.attribution_flags)} statement(s) rest on another person's own account "
                f"({', '.join(speakers)}) - check the answer is not borrowing their situation"
            )
        if self.absence_claims:
            out.append(
                f"{len(self.absence_claims)} statement(s) claim the record does NOT contain "
                "something - the record may simply not have been retrieved"
            )
        return out

    def summary_line(self) -> str:
        parts = [f"{len(self.citations)} citation(s)"]
        if self.invalid_citations:
            parts.append(f"{self.invalid_citations} invalid")
        if self.uncited:
            parts.append("uncited")
        parts.append(
            f"{len(self.absence_claims)} silence claim(s)"
            if self.absence_claims
            else "no silence claims"
        )
        parts.append(
            f"{len(self.attribution_flags)} attribution flag(s)"
            if self.attribution_flags
            else "no attribution flags"
        )
        return " | ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_sources": self.n_sources,
            "citations": list(self.citations),
            "invalid_citations": self.invalid_citations,
            "uncited": self.uncited,
            "absence_claims": list(self.absence_claims),
            "attribution_flags": [dict(f) for f in self.attribution_flags],
            "warnings": self.warnings(),
        }


def assess(answer: str, sources: Sequence[Mapping[str, Any]]) -> GroundednessReport:
    """Build the report for one answer and the sources it was given."""
    answer = answer or ""
    valid, invalid = parse_citations(answer, len(sources))
    claims = detect_absence_claims(answer)
    flags = screen_answer(
        answer,
        [{"rank": s.get("rank"), "content": s.get("content") or ""} for s in sources],
    )
    return GroundednessReport(
        n_sources=len(sources),
        citations=tuple(valid),
        invalid_citations=invalid,
        absence_claims=tuple(c.sentence for c in claims),
        attribution_flags=tuple(flags),
    )
