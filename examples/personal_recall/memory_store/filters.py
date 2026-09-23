"""``RetrievalFilter`` — the structured narrowing that happens *before* semantic recall.

A memory question is almost never "find text like this anywhere in eight years". It is "what did
*this person* say", "in *that* conversation", "around *then*". Each of those is a fact the store
already indexes, and applying it before the nearest-neighbour search is the difference between
ranking 81,672 chunks and ranking the few thousand that could possibly be the answer.

This module is deliberately **pure**: a filter and the SQL predicate it becomes, with no connection
and no import of the store. That is what lets every filter case be tested — including the ones that
must select nothing and the ones that must never merge two people — on a machine with no PostgreSQL.

The clause is written against the alias ``c`` for ``memory_chunks``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

#: The alias the generated predicates are written against.
CHUNK_ALIAS = "c"


@dataclass(frozen=True)
class RetrievalFilter:
    """Which part of the memory a search is allowed to look at. Every field is optional.

    ``person_ids`` means **speakers**, not participants-in-general: a chunk matches when one of the
    named people said something *inside it*. That is the semantics speaker attribution needs —
    "what did 小王 say about the deadline" must not return a chunk where 小王 was merely present
    while somebody else answered — and a conversation-level reading is available separately through
    ``conversation_ids``, which is what "everything from my chat with 小王" wants.

    ``start_time`` / ``end_time`` are an **overlap** window, not a containment one: a session that
    starts before the window and ends inside it contains part of the period being asked about, and
    dropping it would lose exactly the evidence a boundary question is looking for.

    Times are the contract's naive local wall clocks, compared against the same column type, so no
    timezone is invented on either side.
    """

    person_ids: tuple[str, ...] = ()
    conversation_ids: tuple[str, ...] = ()
    start_time: datetime | None = None
    end_time: datetime | None = None

    def __post_init__(self) -> None:
        # Normalised here so two spellings of the same filter are the same filter: the BM25 index is
        # cached by filter, and a list versus a tuple would rebuild a twelve-second index for nothing.
        object.__setattr__(self, "person_ids", tuple(dict.fromkeys(str(p) for p in self.person_ids)))
        object.__setattr__(
            self, "conversation_ids", tuple(dict.fromkeys(str(c) for c in self.conversation_ids))
        )

    @property
    def is_empty(self) -> bool:
        """True when this filter constrains nothing — a global search."""
        return not (
            self.person_ids
            or self.conversation_ids
            or self.start_time is not None
            or self.end_time is not None
        )

    @property
    def time_bounded(self) -> bool:
        return self.start_time is not None or self.end_time is not None

    def key(self) -> tuple[Any, ...]:
        """A hashable identity, for caching a candidate set or a lexical index per filter."""
        return (self.person_ids, self.conversation_ids, self.start_time, self.end_time)

    def describe(self) -> str:
        """Loggable. Names the *shape* of the narrowing and never a person or conversation id."""
        parts: list[str] = []
        if self.person_ids:
            parts.append(f"{len(self.person_ids)} person(s)")
        if self.conversation_ids:
            parts.append(f"{len(self.conversation_ids)} conversation(s)")
        if self.time_bounded:
            parts.append("time window")
        return ", ".join(parts) if parts else "no filter"

    def clause(self, alias: str = CHUNK_ALIAS) -> tuple[str, list[Any]]:
        """``(sql, params)`` for a ``WHERE`` fragment, or ``("", [])`` when nothing is constrained.

        ``RETURNING``-style composition rather than string interpolation of values: every value is a
        bound parameter, so a conversation id can never be read as SQL no matter what an exporter
        one day writes into a talker.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if self.conversation_ids:
            conditions.append(f"{alias}.conversation_id = ANY(%s)")
            params.append(list(self.conversation_ids))

        if self.person_ids:
            # A semi-join against the evidence links, not against membership: this is what makes the
            # filter mean "a chunk this person spoke in" rather than "a chunk from a chat they are in".
            conditions.append(
                "EXISTS (SELECT 1 FROM chunk_events ce JOIN memory_events e ON e.event_id = ce.event_id "
                f"WHERE ce.chunk_id = {alias}.chunk_id AND e.speaker_person_id = ANY(%s))"
            )
            params.append(list(self.person_ids))

        if self.start_time is not None:
            # Overlap, in both directions: `end_time >= start` keeps a session that runs into the
            # window, `start_time <= end` keeps one that begins before it and continues past.
            conditions.append(f"{alias}.end_time >= %s")
            params.append(self.start_time)
        if self.end_time is not None:
            conditions.append(f"{alias}.start_time <= %s")
            params.append(self.end_time)

        return (" AND ".join(conditions), params)


def combine(*filters: RetrievalFilter) -> RetrievalFilter:
    """One filter carrying every constraint of the given ones.

    Intersection by construction: the store applies the fields as a conjunction, so combining two
    "what this person said" filters means both people, which is the reading a planner wants when it
    extracts a person from the question and a person from the conversation scope.
    """
    return RetrievalFilter(
        person_ids=tuple(p for f in filters for p in f.person_ids),
        conversation_ids=tuple(c for f in filters for c in f.conversation_ids),
        start_time=next((f.start_time for f in filters if f.start_time is not None), None),
        end_time=next((f.end_time for f in filters if f.end_time is not None), None),
    )


def from_mapping(payload: Any) -> RetrievalFilter:
    """Build a filter from a loose mapping, ignoring anything unusable.

    Lenient on purpose: this is fed by a query planner whose output is a model's, and a filter that
    cannot be understood must degrade to a *less* narrow search rather than to an error. A wrong
    filter silently excludes the answer; no filter merely costs recall that BM25 and the vector
    search can still recover.
    """
    if not isinstance(payload, dict):
        return RetrievalFilter()
    people = payload.get("person_ids") or payload.get("people") or ()
    conversations = payload.get("conversation_ids") or payload.get("conversations") or ()
    return RetrievalFilter(
        person_ids=tuple(str(p) for p in _sequence(people) if p),
        conversation_ids=tuple(str(c) for c in _sequence(conversations) if c),
        start_time=_datetime(payload.get("start_time")),
        end_time=_datetime(payload.get("end_time")),
    )


def _sequence(value: Any) -> Sequence[Any]:
    if value in (None, ""):
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    return (value,)


def _datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    # The store's chat times are naive local wall clocks; a planner that returns an offset is
    # converted to that local clock rather than compared against a column that has no zone.
    return parsed.astimezone().replace(tzinfo=None) if parsed.tzinfo else parsed
