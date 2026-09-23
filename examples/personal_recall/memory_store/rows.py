"""``MemoryEvent`` / ``MemoryChunk`` -> database rows, as a pure function.

This module is where the store's correctness lives, and it is deliberately the only place that turns
the memory layer into SQL shapes. It has **no database import**: it takes the normalized events and
chunks the importer already produces and returns tuples. Two consequences, both intended:

* The golden parity test (incremental vs fresh bootstrap) can compare *row sets* rather than query
  results, because the same input always produces the same tuples — the digest of a person, the
  ordering rule for a name, the ordinal of an event in a chunk are all decided here, in one place.
* Nothing about "what the database should contain" can drift into `store.py`, which only writes.

Three rules this module follows, each of them a way a naive projection is wrong:

**1. Identity is not a name, and a name is not an identity.** ``speaker_id``, ``speaker_display``
and ``sender_name`` are carried through as three separate columns. Nothing here merges them, and
nothing here re-derives one from another — the memory layer already decided what each means
(``memory.senders``) and a second opinion would be a second contract.

**2. A person is a source identity.** ``person_id`` is a digest of the identity the export stated,
never a surrogate key and never a display name. Two members of one group who both render ``小王``
therefore cannot collide, because they never share a key in the first place — the collapse is
prevented structurally rather than by a dedupe rule that could be forgotten (§8, §26).

**3. The absence of an identity stays an absence.** A self message has no person. A group message
whose export stated no usable sender has no person. Neither gets an invented one, and no two
conversations' unresolved speakers are gathered under a shared placeholder (§9).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from memory.conversations import (
    ConversationDescriptor,
    GROUP,
    conversation_type_of,
)
from memory.events import OTHER_ROLE, SELF_ROLE, MemoryEvent
from memory.labels import ConversationLabel, usable_name
from memory.sessions import MemoryChunk

# The identity rule that decides whether a group message names a speaker at all. Imported from the
# module that owns it rather than reimplemented: ``_identity_of`` rejects the empty id and the id
# that is the conversation spelled back, and a copy of that rule here is precisely how the two would
# drift apart. The private name is the price of not having a second source of truth, and the import
# failing loudly on a rename is the behaviour wanted — a silent fallback would be worse.
from memory.senders import _identity_of

#: What produced these rows. One value today; the column exists so a second source adapter cannot
#: quietly share a person-id space with this one (``person_id`` includes it in the digest).
SOURCE_TYPE = "weflow"

#: Length of a ``person_id`` digest. 128 bits of a sha256, which is far past collision risk at this
#: corpus size and short enough that 1.6M event rows reference it cheaply.
PERSON_ID_LENGTH = 32

#: Column orders, one tuple per table. ``store.py`` COPYs in exactly these orders and a test compares
#: them against ``information_schema`` — a column added to the migration and forgotten here, or the
#: reverse, fails the suite instead of writing rows into the wrong fields.
CONVERSATION_COLUMNS = (
    "conversation_id",
    "source_type",
    "conversation_type",
    "display_label",
    "display_name",
    "event_count",
    "chunk_count",
    "first_event_at",
    "last_event_at",
    "metadata",
)
PEOPLE_COLUMNS = (
    "person_id",
    "source_type",
    "source_identity",
    "person_kind",
    "display_name",
    "first_seen_at",
    "last_seen_at",
    "event_count",
    "conversation_count",
    "metadata",
)
CONVERSATION_PEOPLE_COLUMNS = (
    "conversation_id",
    "person_id",
    "person_kind",
    "display_name",
    "display_label",
    "event_count",
    "first_seen_at",
    "last_seen_at",
)
EVENT_COLUMNS = (
    "event_id",
    "conversation_id",
    "event_time",
    "speaker_role",
    "speaker_person_id",
    "speaker_id",
    "speaker_display",
    "sender_name",
    "text",
    "message_type",
    "reply_to",
    "source_type",
    "source_epoch",
    "source_position",
    "source_server_id",
    "source_local_id",
    "metadata",
)
CHUNK_COLUMNS = (
    "chunk_id",
    "conversation_id",
    "chunk_index",
    "start_time",
    "end_time",
    "n_events",
    "n_chars",
    "text",
    "text_hash",
    "projection_version",
    "metadata",
)
CHUNK_EVENT_COLUMNS = ("chunk_id", "event_id", "ordinal")
#: The export files a generation was built from. Not domain data — provenance, so the store can
#: compute its own delta without borrowing the index's sync checkpoint (§18).
INVENTORY_COLUMNS = ("relative_path", "size", "mtime_ns", "kind", "shard", "conversation_id")

#: Full column order per table, for the COPY statement and the schema-drift test.
TABLE_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "conversations": CONVERSATION_COLUMNS,
    "people": PEOPLE_COLUMNS,
    "conversation_people": CONVERSATION_PEOPLE_COLUMNS,
    "memory_events": EVENT_COLUMNS,
    "memory_chunks": CHUNK_COLUMNS,
    "chunk_events": CHUNK_EVENT_COLUMNS,
    "source_inventory": INVENTORY_COLUMNS,
}

#: Tables written by COPY in a bootstrap. `source_inventory` is written outside `StoreSnapshot`
#: because it describes the *tree*, not the memory — and a caller updating one conversation must not
#: have to re-describe the whole export directory to do it.
SNAPSHOT_TABLES: tuple[str, ...] = (
    "conversations",
    "people",
    "conversation_people",
    "memory_events",
    "memory_chunks",
    "chunk_events",
)

#: The tables a sync truncates and rewrites for the conversations it is replacing, in an order that
#: satisfies the foreign keys on insert and reverses them on delete.
CONVERSATION_SCOPED_TABLES: tuple[str, ...] = (
    "chunk_events",
    "memory_chunks",
    "memory_events",
    "conversation_people",
)


class ProjectionError(ValueError):
    """The events and chunks handed in do not describe one coherent account."""


def person_id_for(source_identity: str, source_type: str = SOURCE_TYPE) -> str:
    """The deterministic key for a source identity.

    A digest rather than the identity itself: the id is what appears in counts, logs and test
    output, and a wxid must not be reachable from any of those (§36). Deterministic rather than
    sequential so that two independent builds of the same export produce the same keys — which is
    what makes the incremental-vs-full parity comparison meaningful.
    """
    payload = f"{source_type}\x00{source_identity}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:PERSON_ID_LENGTH]


def content_hash(text: str) -> str:
    """The chunk-text digest. The same function the incremental index keys vector reuse on."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_epoch(event: MemoryEvent) -> int | None:
    """The exporter's own ``createTime`` in seconds, or ``None`` when it did not state one."""
    raw = event.metadata.get("createTime") if isinstance(event.metadata, Mapping) else None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value // 1000 if value > 10_000_000_000 else value


def _text_field(event: MemoryEvent, key: str) -> str | None:
    raw = event.metadata.get(key) if isinstance(event.metadata, Mapping) else None
    if raw in (None, ""):
        return None
    return str(raw)


@dataclass
class _Membership:
    """One person's participation in one conversation, accumulated before it becomes a row."""

    person_id: str
    person_kind: str
    event_count: int = 0
    first_seen_at: Any = None
    last_seen_at: Any = None
    display_name: str | None = None
    display_label: str = ""
    #: Candidate names, counted so the winner is decided by frequency and then by first appearance —
    #: the same rule ``memory.senders._member_labels`` applies to rendered labels.
    name_votes: dict[str, int] = field(default_factory=dict)
    name_order: dict[str, int] = field(default_factory=dict)


def _pick_name(votes: Mapping[str, int], order: Mapping[str, int]) -> str | None:
    """Most frequent, ties broken by first appearance. ``None`` when nothing was offered."""
    if not votes:
        return None
    return min(votes, key=lambda name: (-votes[name], order.get(name, 0)))


@dataclass(frozen=True)
class StoreSnapshot:
    """Every row one bootstrap or one conversation update would write, plus what it cost to decide."""

    conversations: tuple[tuple[Any, ...], ...] = ()
    people: tuple[tuple[Any, ...], ...] = ()
    conversation_people: tuple[tuple[Any, ...], ...] = ()
    events: tuple[tuple[Any, ...], ...] = ()
    chunks: tuple[tuple[Any, ...], ...] = ()
    chunk_events: tuple[tuple[Any, ...], ...] = ()
    stats: Mapping[str, Any] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        return {
            "conversations": len(self.conversations),
            "people": len(self.people),
            "conversation_people": len(self.conversation_people),
            "events": len(self.events),
            "chunks": len(self.chunks),
            "chunk_events": len(self.chunk_events),
        }

    def rows_for(self, table: str) -> tuple[tuple[Any, ...], ...]:
        return {
            "conversations": self.conversations,
            "people": self.people,
            "conversation_people": self.conversation_people,
            "memory_events": self.events,
            "memory_chunks": self.chunks,
            "chunk_events": self.chunk_events,
        }[table]


def _conversation_metadata(descriptor: ConversationDescriptor | None) -> dict[str, Any]:
    if descriptor is None:
        return {}
    payload: dict[str, Any] = {"type": descriptor.conversation_type}
    if descriptor.aliases:
        payload["aliases"] = list(descriptor.aliases)
    payload.update(dict(descriptor.metadata))
    return payload


def _display_label_for(
    conversation_id: str,
    descriptor: ConversationDescriptor | None,
    label: ConversationLabel | None,
) -> str | None:
    """The human-readable conversation name, or ``None`` when there is not one.

    The sidecar (Phase 20.6's resolved labels) wins when it holds a usable name, because resolving a
    talker to a person's actual name is what that sidecar exists for. Falling back to the exporter's
    ``displayName`` is safe only because :func:`memory.labels.usable_name` is consulted again here: an
    exporter that copies the talker into that field yields ``None``, not a wxid sitting in a column
    that promises a label.
    """
    if label is not None:
        resolved = usable_name(label.label, conversation_id)
        if resolved:
            return resolved
    if descriptor is not None:
        offered = usable_name(descriptor.display_name, conversation_id)
        if offered:
            return offered
    return None


def build_snapshot(
    events_by_conversation: Mapping[str, Sequence[MemoryEvent]],
    chunks: Sequence[MemoryChunk],
    *,
    descriptors: Sequence[ConversationDescriptor] = (),
    labels: Mapping[str, ConversationLabel] | None = None,
    source_type: str = SOURCE_TYPE,
    projection_version: int = 1,
) -> StoreSnapshot:
    """Project a whole account (or any subset of conversations) into rows.

    The unit is a **set of conversations supplied complete**. An incremental update hands in exactly
    the conversations it re-rendered, so everything below is decided from a conversation's whole
    current event stream — never from a delta. That is what makes ``DELETE`` + re-insert a correct
    update rather than a lossy one: a message that changed, a sender renamed retroactively, a session
    boundary that moved are all just the conversation rendering differently (§16, §17).
    """
    descriptor_by_id = {d.conversation_id: d for d in descriptors}
    label_by_id = dict(labels or {})

    events_by_conversation = {
        str(cid): tuple(events)
        for cid, events in events_by_conversation.items()
        if events
    }
    known_conversations = set(events_by_conversation)

    # --- membership and identity, decided per conversation --------------------------------------
    memberships: dict[tuple[str, str], _Membership] = {}
    person_kind: dict[str, str] = {}
    person_identity: dict[str, str] = {}
    orphan_chunks = 0

    for conversation_id in sorted(events_by_conversation):
        events = events_by_conversation[conversation_id]
        is_group = conversation_type_of(conversation_id) == GROUP
        by_person: dict[str, _Membership] = {}

        for position, event in enumerate(events):
            if event.speaker_role != OTHER_ROLE:
                # A self message carries no source identity, so it names no person (§9). Recorded as
                # an absence rather than as a synthetic "me" that would join nothing.
                continue

            if is_group:
                identity = _identity_of(event, conversation_id)
                kind = "group_member"
                if not identity:
                    continue
            else:
                # A direct conversation has exactly one other side by definition, and that side is
                # the conversation itself: its talker is the peer's wxid. Measured on the real
                # account, 134 of 140 direct conversations state it again in the message field and
                # none contradicts it, and 73 of the talkers appear verbatim as a group member —
                # which is what makes this the *same* person in both, not a coincidence of shape.
                #
                # The forbidden alternative is what this replaces: one shared "对方" person for every
                # direct conversation, which would merge strangers (§9).
                identity = conversation_id
                kind = "direct_peer"

            person_id = person_id_for(identity, source_type)
            person_identity.setdefault(person_id, identity)
            # A person who is an explicit group member somewhere keeps that kind: an explicit sender
            # field is stronger evidence than a file name.
            person_kind[person_id] = (
                "group_member"
                if person_kind.get(person_id) == "group_member" or kind == "group_member"
                else "direct_peer"
            )

            member = by_person.get(person_id)
            if member is None:
                member = _Membership(person_id=person_id, person_kind=kind)
                by_person[person_id] = member
            member.event_count += 1
            if member.first_seen_at is None or event.timestamp < member.first_seen_at:
                member.first_seen_at = event.timestamp
            if member.last_seen_at is None or event.timestamp > member.last_seen_at:
                member.last_seen_at = event.timestamp
            member.display_label = event.sender_name or member.display_label

            offered = usable_name(event.speaker_display, conversation_id, identity)
            if offered:
                member.name_votes[offered] = member.name_votes.get(offered, 0) + 1
                member.name_order.setdefault(offered, position)

        for person_id, member in by_person.items():
            member.display_name = _pick_name(member.name_votes, member.name_order)
            memberships[(conversation_id, person_id)] = member

    # --- events ---------------------------------------------------------------------------------
    event_rows: list[tuple[Any, ...]] = []
    conversation_event_count: dict[str, int] = defaultdict(int)
    conversation_first: dict[str, Any] = {}
    conversation_last: dict[str, Any] = {}
    resolved_person: dict[tuple[str, str], str | None] = {}

    for conversation_id in sorted(events_by_conversation):
        is_group = conversation_type_of(conversation_id) == GROUP
        for event in events_by_conversation[conversation_id]:
            person_id: str | None = None
            if event.speaker_role == OTHER_ROLE:
                identity = _identity_of(event, conversation_id) if is_group else conversation_id
                if identity:
                    person_id = person_id_for(identity, source_type)
            event_rows.append(
                (
                    event.id,
                    conversation_id,
                    event.timestamp,
                    event.speaker_role or "",
                    person_id,
                    event.speaker_id or "",
                    event.speaker_display or "",
                    event.sender_name,
                    event.text,
                    event.message_type,
                    event.reply_to,
                    source_type,
                    _source_epoch(event),
                    event.metadata.get("position") if isinstance(event.metadata, Mapping) else None,
                    _text_field(event, "serverId"),
                    _text_field(event, "localId"),
                    dict(event.metadata),
                )
            )
            conversation_event_count[conversation_id] += 1
            if conversation_id not in conversation_first or event.timestamp < conversation_first[conversation_id]:
                conversation_first[conversation_id] = event.timestamp
            if conversation_id not in conversation_last or event.timestamp > conversation_last[conversation_id]:
                conversation_last[conversation_id] = event.timestamp
            resolved_person[(conversation_id, event.id)] = person_id

    # --- chunks and evidence linkage ------------------------------------------------------------
    chunk_rows: list[tuple[Any, ...]] = []
    chunk_event_rows: list[tuple[Any, ...]] = []
    chunk_count: dict[str, int] = defaultdict(int)
    chunk_index: dict[str, int] = defaultdict(int)

    for chunk in chunks:
        if chunk.conversation_id not in known_conversations:
            # A chunk for a conversation that supplied no event cannot be written (the events it
            # cites would be missing) and cannot be silently dropped either.
            raise ProjectionError(
                f"chunk {chunk.id} belongs to a conversation with no events in this snapshot"
            )
        chunk_index[chunk.conversation_id] += 1
        chunk_rows.append(
            (
                chunk.id,
                chunk.conversation_id,
                chunk_index[chunk.conversation_id],
                chunk.start_time,
                chunk.end_time,
                chunk.n_events,
                chunk.n_chars,
                chunk.text,
                content_hash(chunk.text),
                projection_version,
                dict(chunk.metadata),
            )
        )
        chunk_count[chunk.conversation_id] += 1
        for ordinal, event_id in enumerate(chunk.event_ids):
            if (chunk.conversation_id, event_id) not in resolved_person:
                orphan_chunks += 1
                raise ProjectionError(
                    f"chunk {chunk.id} cites an event that is not in this snapshot"
                )
            chunk_event_rows.append((chunk.id, event_id, ordinal))

    # --- conversation rows ----------------------------------------------------------------------
    conversation_rows: list[tuple[Any, ...]] = []
    for conversation_id in sorted(events_by_conversation):
        descriptor = descriptor_by_id.get(conversation_id)
        conversation_rows.append(
            (
                conversation_id,
                source_type,
                conversation_type_of(conversation_id),
                _display_label_for(conversation_id, descriptor, label_by_id.get(conversation_id)),
                descriptor.display_name if descriptor is not None else "",
                conversation_event_count[conversation_id],
                chunk_count.get(conversation_id, 0),
                conversation_first.get(conversation_id),
                conversation_last.get(conversation_id),
                _conversation_metadata(descriptor),
            )
        )

    # --- membership rows, and the person rows derived from them ----------------------------------
    membership_rows: list[tuple[Any, ...]] = []
    person_memberships: dict[str, list[_Membership]] = defaultdict(list)
    for (conversation_id, person_id), member in sorted(memberships.items()):
        membership_rows.append(
            (
                conversation_id,
                person_id,
                member.person_kind,
                member.display_name,
                member.display_label or None,
                member.event_count,
                member.first_seen_at,
                member.last_seen_at,
            )
        )
        person_memberships[person_id].append(member)

    people_rows: list[tuple[Any, ...]] = []
    for person_id in sorted(person_memberships):
        owned = person_memberships[person_id]
        votes: dict[str, int] = defaultdict(int)
        order: dict[str, Any] = {}
        for member in owned:
            if member.display_name:
                votes[member.display_name] += 1
                seen = order.setdefault(member.display_name, member.first_seen_at)
                if member.first_seen_at is not None and (seen is None or member.first_seen_at < seen):
                    order[member.display_name] = member.first_seen_at
        # Ordered by (count desc, earliest sighting asc, value asc). The final tie-break is the name
        # itself because two names offered at the same instant would otherwise be separated by
        # whichever `min` happened to see first — a property of the loop, not of the corpus. The SQL
        # that recomputes this column after an incremental sync uses the identical ordering, and the
        # parity test is what holds the two to it.
        winner = (
            min(votes, key=lambda name: (-votes[name], order.get(name), name)) if votes else None
        )
        people_rows.append(
            (
                person_id,
                source_type,
                person_identity[person_id],
                person_kind[person_id],
                winner,
                min((m.first_seen_at for m in owned if m.first_seen_at is not None), default=None),
                max((m.last_seen_at for m in owned if m.last_seen_at is not None), default=None),
                sum(m.event_count for m in owned),
                len(owned),
                {},
            )
        )

    stats = {
        "events_without_person": sum(1 for row in event_rows if row[4] is None),
        "self_events": sum(1 for row in event_rows if row[3] == SELF_ROLE),
        "group_members": sum(1 for row in people_rows if row[3] == "group_member"),
        "direct_peers": sum(1 for row in people_rows if row[3] == "direct_peer"),
        "named_people": sum(1 for row in people_rows if row[4]),
        "conversations_labelled": sum(1 for row in conversation_rows if row[3]),
        "chunk_event_links": len(chunk_event_rows),
        "orphan_chunks_rejected": orphan_chunks,
    }

    return StoreSnapshot(
        conversations=tuple(conversation_rows),
        people=tuple(people_rows),
        conversation_people=tuple(membership_rows),
        events=tuple(event_rows),
        chunks=tuple(chunk_rows),
        chunk_events=tuple(chunk_event_rows),
        stats=stats,
    )


def _iter_values(snapshot: StoreSnapshot, table: str) -> Iterable[tuple[Any, ...]]:
    return snapshot.rows_for(table)
