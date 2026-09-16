"""Conversation-session segmentation: MemoryEvent -> MemoryChunk.

Deterministic rule set (§27) — no NLP, no model:

* **temporal adjacency** — a new session starts when the gap to the previous
  message exceeds ``max_gap`` (a different calendar day always breaks, because a
  gap can only be small within the same day);
* **budget** — a new session starts when adding the next message would exceed
  ``max_chars``, so a single long burst still produces embeddable units;
* ids/timestamps/participants are carried on the chunk so later phases can filter
  by time or person without re-parsing the source.

The chunk text is the verbatim chat lines of the session, so any line in the
session can be cited back to the original export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

from .events import MemoryEvent

#: A new session starts when the silence between two messages exceeds this.
DEFAULT_MAX_GAP = timedelta(hours=6)
#: Soft cap on session size in characters (drives embeddability, not correctness).
DEFAULT_MAX_CHARS = 900


@dataclass(frozen=True)
class SessionConfig:
    """Segmentation knobs. ``max_chars`` is the one worth sweeping in evals."""

    max_gap: timedelta = DEFAULT_MAX_GAP
    max_chars: int = DEFAULT_MAX_CHARS

    def as_dict(self) -> dict[str, Any]:
        return {"max_gap_minutes": self.max_gap.total_seconds() / 60, "max_chars": self.max_chars}


@dataclass(frozen=True)
class MemoryChunk:
    """A conversation session — the retrieval unit."""

    id: str
    conversation_id: str
    start_time: datetime
    end_time: datetime
    participants: tuple[str, ...]
    event_ids: tuple[str, ...]
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_events(self) -> int:
        return len(self.event_ids)

    @property
    def n_chars(self) -> int:
        return len(self.text)


def build_sessions(
    events: Sequence[MemoryEvent],
    config: SessionConfig | None = None,
    conversation_id: str = "txt",
) -> list[MemoryChunk]:
    """Group events into conversation sessions.

    Events are assumed to be in file (chronological) order; the function does not
    re-sort, so a mis-ordered source stays visible instead of being silently fixed.
    """
    cfg = config or SessionConfig()
    chunks: list[MemoryChunk] = []
    current: list[MemoryEvent] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if not current:
            return
        chunks.append(_to_chunk(current, len(chunks) + 1, conversation_id, cfg))
        current = []
        current_chars = 0

    for event in events:
        if current:
            previous = current[-1]
            gap = event.timestamp - previous.timestamp
            same_day = event.timestamp.date() == previous.timestamp.date()
            would_overflow = current_chars + len(event.line) + 1 > cfg.max_chars
            if (not same_day) or gap > cfg.max_gap or would_overflow:
                flush()
        current.append(event)
        current_chars += len(event.line) + 1

    flush()
    return chunks


def _to_chunk(
    events: Sequence[MemoryEvent],
    index: int,
    conversation_id: str,
    config: SessionConfig,
) -> MemoryChunk:
    return MemoryChunk(
        id=f"{conversation_id}-session-{index:04d}",
        conversation_id=conversation_id,
        start_time=events[0].timestamp,
        end_time=events[-1].timestamp,
        participants=tuple(sorted({e.sender_name for e in events})),
        event_ids=tuple(e.id for e in events),
        text="\n".join(e.line for e in events),
        metadata={
            "session_index": index,
            "n_events": len(events),
            "start_time": events[0].timestamp.isoformat(sep=" "),
            "end_time": events[-1].timestamp.isoformat(sep=" "),
            "participants": sorted({e.sender_name for e in events}),
            "session_config": config.as_dict(),
        },
    )
