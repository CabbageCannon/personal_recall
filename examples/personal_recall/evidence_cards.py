"""Evidence cards: turn a cited answer back into readable original chat lines.

The engine's contract is *Answer -> Evidence -> MemoryEvent -> original chat line*. The
eval harness proves the first link with numbers (citation rate / coverage / precision);
this module is the product-facing half: given an answer and the retrieved session chunks,
it resolves the answer's `[来源 N]` citations into evidence cards that show the date range,
the participants, the conversation-session id and the matching source lines.

Kept deliberately free of any model call so it is fully testable offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from citation_metrics import parse_citations

#: One chat line inside a session chunk: "[YYYY-MM-DD HH:MM] speaker: text".
LINE_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] ([^:]{1,16}): (.+)$")


@dataclass
class EvidenceLine:
    """A single MemoryEvent as it appears in the source chat log."""

    timestamp: str
    speaker: str
    text: str

    def render(self) -> str:
        return f"{self.timestamp} {self.speaker}: {self.text}"


@dataclass
class EvidenceCard:
    """One cited retrieval unit, resolved to the chat lines it contains."""

    citation_index: int
    chunk_index: int | None = None
    memory_chunk_id: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    participants: list[str] = field(default_factory=list)
    lines: list[EvidenceLine] = field(default_factory=list)
    cited: bool = True
    n_events: int | None = None

    @property
    def label(self) -> str:
        """The user-facing provenance label, e.g. `[小王, 我 · 2024-10-03 17:43]`."""
        who = ", ".join(self.participants) if self.participants else "未知参与者"
        when = self.start_time or "时间未知"
        if self.end_time and self.end_time != self.start_time:
            when = f"{self.start_time} – {self.end_time}"
        return f"[{who} · {when}]"

    def render(self, indent: str = "    ", max_lines: int = 6) -> str:
        head = (
            f"{indent}[{self.citation_index}] {self.label}"
            + (f" · {self.memory_chunk_id}" if self.memory_chunk_id else "")
            + (f" · 共 {self.n_events} 条消息" if self.n_events else "")
        )
        body = [f"{indent}    {line.render()}" for line in self.lines[:max_lines]]
        if len(self.lines) > max_lines:
            body.append(f"{indent}    …（另有 {len(self.lines) - max_lines} 条）")
        if not body:
            body.append(f"{indent}    （该来源没有可解析的聊天行）")
        return "\n".join([head, *body])


def parse_chat_lines(content: str) -> list[EvidenceLine]:
    """Extract the chat lines from one retrieved chunk."""
    lines: list[EvidenceLine] = []
    for raw in (content or "").splitlines():
        match = LINE_RE.match(raw.strip())
        if match:
            lines.append(
                EvidenceLine(timestamp=match.group(1), speaker=match.group(2), text=match.group(3))
            )
    return lines


def build_evidence_cards(
    answer: str,
    retrieved_sources: Sequence[dict[str, Any]],
    include_uncited: bool = False,
) -> list[EvidenceCard]:
    """Resolve an answer's citations into evidence cards.

    Citations are **0-based** (`[来源 0]` is the first retrieved source), matching the
    framework's ``Source: N`` rendering. With ``include_uncited`` the retrieved-but-uncited
    sources are appended, marked ``cited=False`` — useful when an answer cites nothing and
    the caller still wants to show what was retrieved.
    """
    valid, _invalid = parse_citations(answer or "", len(retrieved_sources))

    ordered: list[int] = []
    for index in valid:
        if index not in ordered:
            ordered.append(index)

    cards = [_card_for(index, retrieved_sources[index], cited=True) for index in ordered]

    if include_uncited:
        for index, source in enumerate(retrieved_sources):
            if index not in ordered:
                cards.append(_card_for(index, source, cited=False))

    return cards


def _card_for(index: int, source: dict[str, Any], cited: bool) -> EvidenceCard:
    return EvidenceCard(
        citation_index=index,
        chunk_index=source.get("chunk_index"),
        memory_chunk_id=source.get("memory_chunk_id"),
        start_time=source.get("start_time"),
        end_time=source.get("end_time"),
        participants=list(source.get("participants") or []),
        lines=parse_chat_lines(source.get("content", "")),
        cited=cited,
        n_events=source.get("n_events"),
    )


def render_cards(cards: Sequence[EvidenceCard]) -> str:
    """Render the evidence section shown under an answer."""
    cited = [c for c in cards if c.cited]
    uncited = [c for c in cards if not c.cited]
    parts: list[str] = []
    if cited:
        parts.append(f"Evidence ({len(cited)} cited source(s)):")
        parts.extend(card.render() for card in cited)
    else:
        parts.append("Evidence: the answer cited no source.")
    if uncited:
        parts.append("")
        parts.append(f"Retrieved but not cited ({len(uncited)}):")
        parts.extend(card.render(max_lines=3) for card in uncited)
    return "\n".join(parts)


def cards_to_dict(cards: Sequence[EvidenceCard]) -> list[dict[str, Any]]:
    return [
        {
            "citation_index": card.citation_index,
            "cited": card.cited,
            "memory_chunk_id": card.memory_chunk_id,
            "chunk_index": card.chunk_index,
            "start_time": card.start_time,
            "end_time": card.end_time,
            "participants": card.participants,
            "n_events": card.n_events,
            "label": card.label,
            "lines": [
                {"timestamp": line.timestamp, "speaker": line.speaker, "text": line.text}
                for line in card.lines
            ],
        }
        for card in cards
    ]
