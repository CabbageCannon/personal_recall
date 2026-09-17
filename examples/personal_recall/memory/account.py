"""Account-wide ingestion: every conversation of an account, each one kept separate.

The ordering here is the whole point, and it is the opposite of the obvious one:

    (shard, conversation) exports
              ↓
    group BY CONVERSATION          <- first, always
              ↓
    merge shards WITHIN a conversation
              ↓
    one event stream per conversation
              ↓
    sessions are built per conversation

Merging globally first and then trying to recover conversations is what produces the failure this
module exists to prevent: Alice at 10:00, Bob at 10:02 and a group at 10:04 are close in time, and a
global merge plus a time-adjacency session builder will happily fuse them into one chunk. A conversation
boundary is absolute — no amount of temporal or lexical proximity may cross it.

Deduplication keeps Phase 18B's conservative rule: only the exporter's ``serverId`` is a strong enough
identity to drop a message on, and it is applied **within a conversation**, which is where a duplicated
message can actually occur (a shard migration). Messages without a ``serverId`` are never deduplicated
and are counted instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .conversations import (
    ACCOUNT_MANIFEST_FILENAME,
    EXPORT_FILENAME_SUFFIX,
    SESSION_LISTING_FILENAME,
    ConversationDescriptor,
    conversation_id_from_export_filename,
    parse_session_listing,
    shard_stem,
)
from .events import MemoryEvent
from .sessions import MemoryChunk, SessionConfig, build_sessions
from .shards import (
    MergeReport,
    discover_message_shards,
    load_and_merge,
    merge_shard_events,
)
from .weflow import WeFlowParseResult, parse_weflow_events

#: Manifest keys accepted for each list, in preference order. The orchestrator writes the first
#: spelling; the rest are tolerated so an older or hand-written manifest still counts its shards.
_DETECTED_KEYS = ("detected_shards", "shards_detected", "detected", "shards")
_EXPORTED_KEYS = ("exported_shards", "shards_exported", "exported")

#: Manifest key recording that the export was narrowed with a conversation filter, and to how many
#: conversations. Absent (an older or hand-written manifest) reads as 0 = the whole account. It is a
#: count and never an identity: the manifest may not learn which conversations were selected.
_FILTERED_KEYS = ("filtered_conversations", "conversations_filtered")


@dataclass(frozen=True)
class ConversationShardExport:
    """One conversation's export from one shard — the unit the orchestrator produces."""

    conversation_id: str
    shard: str
    path: str

    def as_dict(self) -> dict[str, Any]:
        return {"conversation_id": self.conversation_id, "shard": self.shard, "path": self.path}


@dataclass(frozen=True)
class ConversationCoverage:
    """Per-conversation result, so a gap can be attributed to one person rather than the whole account."""

    conversation_id: str
    shards: tuple[str, ...]
    messages: int
    duplicates_removed: int
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None

    def span(self) -> str:
        if self.first_timestamp is None:
            return "no messages"
        return f"{self.first_timestamp:%Y-%m-%d %H:%M} .. {self.last_timestamp:%Y-%m-%d %H:%M}"


@dataclass
class AccountImportReport:
    """What the account import actually got, including what it did not."""

    shards_detected: tuple[str, ...] = field(default_factory=tuple)
    shards_exported: tuple[str, ...] = field(default_factory=tuple)
    conversations_discovered: int = 0
    conversations_imported: int = 0
    messages_received: int = 0
    messages_kept: int = 0
    duplicates_removed: int = 0
    skipped_messages: int = 0
    undedupeable: int = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    per_conversation: tuple[ConversationCoverage, ...] = field(default_factory=tuple)
    conversations_without_messages: tuple[str, ...] = field(default_factory=tuple)
    #: Shards that exist on disk but produced no export. Their history is invisible, not empty.
    missing_shards: tuple[str, ...] = field(default_factory=tuple)
    #: How many conversations the export was narrowed to before it ever reached this importer; 0 means
    #: the whole account was exported. A *count* only — never the ids, because a report may be copied
    #: and pasted, and the identity of a conversation is not this module's to carry around.
    filtered_conversations: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def partial(self) -> bool:
        """True when the account history is known to be incomplete.

        ``No Data Loaded != No Memory Exists``: if a message shard exists but was never exported, the
        answers simply cannot see it, and the report must say so rather than let a smaller history
        pass for the whole past.

        A shard that is absent is one way to be incomplete, and it is not the only one. An export
        narrowed by a conversation filter is incomplete *by construction* while leaving every shard
        present: if the selected conversations happen to span every shard, nothing is missing by the
        shard test and a hard-coded ``partial = bool(missing_shards)`` would call 8 conversations out
        of 272 a complete account. Both causes count, and neither may mask the other.
        """
        return bool(self.missing_shards) or self.filtered_conversations > 0

    def lines(self) -> list[str]:
        out = [
            f"message shards : {len(self.shards_exported)} exported of "
            f"{len(self.shards_detected) or '?'} detected",
            f"conversations  : {self.conversations_imported} imported of "
            f"{self.conversations_discovered} discovered",
            f"messages       : {self.messages_kept} kept of {self.messages_received} "
            f"({self.duplicates_removed} duplicate(s) removed)",
        ]
        if self.first_timestamp is not None:
            out.append(
                f"coverage       : {self.first_timestamp:%Y-%m-%d %H:%M} .. "
                f"{self.last_timestamp:%Y-%m-%d %H:%M}"
            )
        if self.skipped_messages:
            out.append(f"skipped        : {self.skipped_messages} unparseable message(s)")
        if self.undedupeable:
            out.append(
                f"note           : {self.undedupeable} message(s) have no serverId and were kept "
                "un-deduplicated rather than risk dropping real history"
            )
        if self.conversations_without_messages:
            out.append(
                f"note           : {len(self.conversations_without_messages)} discovered "
                "conversation(s) produced no messages in the exported shards"
            )
        # Two independent ways to be PARTIAL, deliberately reported as two messages: "a shard failed"
        # (something is broken, re-run the export) and "the export was narrowed" (nothing is broken,
        # there is simply more account than was asked for). One merged sentence would leave the reader
        # unable to tell which of the two happened, and the next action differs between them.
        if self.missing_shards:
            out.append(
                f"WARNING        : {len(self.missing_shards)} message shard(s) exist but were not "
                f"exported {list(self.missing_shards)} - this account history is PARTIAL, and "
                "messages in those shards are invisible to every answer"
            )
        if self.filtered_conversations:
            out.append(
                f"WARNING        : this export was narrowed to "
                f"{self.filtered_conversations} conversation(s) on purpose, so this account history "
                "is PARTIAL by construction - every conversation that was not selected is absent "
                "from the export and invisible to every answer"
            )
        for note in self.notes:
            out.append(f"note           : {note}")
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "shards_detected": list(self.shards_detected),
            "shards_exported": list(self.shards_exported),
            "missing_shards": list(self.missing_shards),
            "filtered_conversations": self.filtered_conversations,
            "partial": self.partial,
            "conversations_discovered": self.conversations_discovered,
            "conversations_imported": self.conversations_imported,
            "conversations_without_messages": list(self.conversations_without_messages),
            "messages_received": self.messages_received,
            "messages_kept": self.messages_kept,
            "duplicates_removed": self.duplicates_removed,
            "skipped_messages": self.skipped_messages,
            "undedupeable": self.undedupeable,
            "first_timestamp": self.first_timestamp.isoformat(sep=" ") if self.first_timestamp else None,
            "last_timestamp": self.last_timestamp.isoformat(sep=" ") if self.last_timestamp else None,
            "per_conversation": [
                {
                    "conversation_id": row.conversation_id,
                    "shards": list(row.shards),
                    "messages": row.messages,
                    "duplicates_removed": row.duplicates_removed,
                    "first_timestamp": row.first_timestamp.isoformat(sep=" ") if row.first_timestamp else None,
                    "last_timestamp": row.last_timestamp.isoformat(sep=" ") if row.last_timestamp else None,
                }
                for row in self.per_conversation
            ],
            "notes": list(self.notes),
        }


def group_exports_by_conversation(
    exports: Sequence[ConversationShardExport],
) -> dict[str, list[ConversationShardExport]]:
    """Group exports by conversation id, deterministically.

    This is the first step and it is not optional: merging shards before grouping would interleave
    different conversations in one event stream.
    """
    grouped: dict[str, list[ConversationShardExport]] = {}
    for export in exports:
        grouped.setdefault(export.conversation_id, []).append(export)
    return {
        conversation_id: sorted(rows, key=lambda row: (row.shard, row.path))
        for conversation_id, rows in sorted(grouped.items())
    }


def exports_from_directory(
    directory: Path,
    shard: str,
    conversation_ids: Mapping[str, str] | None = None,
) -> tuple[ConversationShardExport, ...]:
    """Build exports from a directory of ``{talker}_messages.json`` files (one shard's output).

    ``conversation_ids`` maps file names to conversation ids for exports that do not follow the naming
    convention; the orchestrator uses it when a manifest is available, because identity must never be
    guessed from a file name when a recorded one exists.
    """
    from .conversations import EXPORT_FILENAME_SUFFIX

    exports: list[ConversationShardExport] = []
    for path in sorted(Path(directory).glob(f"*{EXPORT_FILENAME_SUFFIX}")):
        conversation_id = None
        if conversation_ids:
            conversation_id = conversation_ids.get(path.name)
        if conversation_id is None:
            conversation_id = conversation_id_from_export_filename(path.name)
        if conversation_id is None:
            continue
        exports.append(
            ConversationShardExport(conversation_id=conversation_id, shard=shard, path=str(path))
        )
    return tuple(exports)


def _first_list(payload: Any, keys: Sequence[str]) -> list[str] | None:
    """The first list among ``keys`` in a mapping payload, as strings."""
    if not isinstance(payload, Mapping):
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    return None


def _as_count(value: Any) -> int:
    """A non-negative ``int`` from whatever a manifest or a caller supplied.

    A number, and a numeric string, are honoured; everything else — ``None``, a bool, a non-numeric
    string, a negative, any other type — reads as ``0``. A count that cannot be read must never be
    able to *fail* the load, because refusing to open an account directory is the one outcome worse
    than reading a manifest imperfectly.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def _first_count(payload: Any, keys: Sequence[str]) -> int:
    """The first usable count among ``keys`` in a mapping payload. Absent keys read as ``0``."""
    if not isinstance(payload, Mapping):
        return 0
    for key in keys:
        if key in payload:
            return _as_count(payload.get(key))
    return 0


@dataclass(frozen=True)
class AccountLayout:
    """What an account export directory actually contains, before anything is parsed."""

    exports: tuple[ConversationShardExport, ...] = ()
    shards_detected: tuple[str, ...] = ()
    descriptors: tuple[ConversationDescriptor, ...] = ()
    #: From the manifest: how many conversations the export was narrowed to (0 = the whole account).
    filtered_conversations: int = 0
    notes: tuple[str, ...] = ()


def discover_shard_directories(root: Path) -> tuple[Path, ...]:
    """The subdirectories of ``root`` that hold one shard's exports.

    A directory qualifies only by containing at least one ``{talker}_messages.json``, so a stray
    folder cannot silently enlarge the account or shift which shards look exported.
    """
    directory = Path(root)
    if not directory.is_dir():
        return ()
    return tuple(
        sorted(
            path
            for path in directory.iterdir()
            if path.is_dir() and any(path.glob(f"*{EXPORT_FILENAME_SUFFIX}"))
        )
    )


def read_account_manifest(root: Path) -> dict[str, Any]:
    """Read the account manifest, or ``{}`` when it is absent or unreadable.

    Absence is normal (a hand-assembled export directory) and must not be an error — the caller then
    falls back to what it can see, and says so.
    """
    path = Path(root) / ACCOUNT_MANIFEST_FILENAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def load_account_directory(root: Path, *, shard_dir: Path | None = None) -> AccountLayout:
    """Discover an account export tree: ``root/<shard>/<talker>_messages.json`` (+ ``sessions.json``).

    Identity is taken from the recorded listing or the exporter's file name, never guessed from
    message content, and the detected-vs-exported comparison prefers the real ``Msg/Multi``
    directory when it is available — that is the only source that can prove a shard is missing.
    """
    directory = Path(root)
    notes: list[str] = []
    shard_dirs = discover_shard_directories(directory)

    exports: list[ConversationShardExport] = []
    descriptors: dict[str, ConversationDescriptor] = {}
    for shard_dir_path in shard_dirs:
        listing_path = shard_dir_path / SESSION_LISTING_FILENAME
        if listing_path.exists():
            try:
                listing = json.loads(listing_path.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, OSError) as exc:
                notes.append(
                    f"{shard_dir_path.name}: could not read {SESSION_LISTING_FILENAME} ({exc}); "
                    "conversation identity fell back to export file names"
                )
            else:
                # Union across shards: a conversation seen in any shard is a known conversation, and
                # its display name is a label, never a key.
                found = parse_session_listing(listing)
                for descriptor in found:
                    descriptors.setdefault(descriptor.conversation_id, descriptor)
                if not found and isinstance(listing, list) and listing:
                    # A listing that is present and non-empty but yields no identity is a schema
                    # mismatch, not an empty shard. Say so: the alternative is an account that looks
                    # perfectly healthy while every conversation in it has gone missing.
                    notes.append(
                        f"{shard_dir_path.name}/{SESSION_LISTING_FILENAME} holds "
                        f"{len(listing)} entr{'y' if len(listing) == 1 else 'ies'} but none of them "
                        "carried a recognisable conversation id, so those conversations are unknown"
                    )
        exports.extend(exports_from_directory(shard_dir_path, shard=shard_dir_path.name))

    manifest = read_account_manifest(directory)
    detected = _first_list(manifest, _DETECTED_KEYS)
    if shard_dir is not None and Path(shard_dir).is_dir():
        # Authoritative: the actual Msg/Multi directory. `MSG2.db-wal` and friends normalise onto the
        # same stem, so a shard is not counted missing merely because a sidecar was listed.
        detected = [shard_stem(name) for name in discover_message_shards(Path(shard_dir))]
    elif detected is not None:
        detected = [shard_stem(name) for name in detected]
    else:
        detected = [path.name for path in shard_dirs]
        notes.append(
            "no shard manifest and no --shard-dir: completeness was checked against the export "
            "directories themselves, so a shard that was never exported cannot be detected here"
        )

    exported = {shard_stem(export.shard) for export in exports}
    manifest_exported = _first_list(manifest, _EXPORTED_KEYS)
    if manifest_exported:
        claimed = {shard_stem(name) for name in manifest_exported}
        silent = sorted(claimed - set(path.name for path in shard_dirs))
        if silent:
            notes.append(
                f"the manifest lists {silent} as exported but no such export directory holds any "
                "messages; those shards are being treated as NOT exported"
            )

    # A conversation-filtered export writes every shard's full listing, so it can mark every shard
    # "exported" and still hold almost none of the account. The manifest's filter count is the only
    # evidence of that, and its absence (an older or hand-written manifest) means unfiltered.
    filtered = _first_count(manifest, _FILTERED_KEYS)
    if filtered:
        notes.append(
            f"the manifest records that this export was narrowed to {filtered} conversation(s): only "
            "those conversations were exported, and every other conversation of this account is not "
            "searchable here"
        )

    return AccountLayout(
        exports=tuple(exports),
        shards_detected=tuple(sorted(set(detected))),
        descriptors=tuple(descriptors[key] for key in sorted(descriptors)),
        filtered_conversations=filtered,
        notes=tuple(notes),
    )


def import_account(
    exports: Sequence[ConversationShardExport],
    *,
    shards_detected: Sequence[str] = (),
    discovered: Sequence[ConversationDescriptor] = (),
    filtered_conversations: int = 0,
) -> tuple[dict[str, list[MemoryEvent]], AccountImportReport]:
    """Import an account: group by conversation, merge shards within it, and report coverage.

    Returns events **per conversation** — never a single flat stream — so the caller cannot
    accidentally build sessions across a conversation boundary.

    ``filtered_conversations`` is how many conversations the export was narrowed to (0 = whole
    account), read from the export manifest by :func:`load_account_directory`. It changes no message
    and no chunk; it only stops a deliberately narrowed export from being reported as complete.
    """
    grouped = group_exports_by_conversation(exports)

    events_by_conversation: dict[str, list[MemoryEvent]] = {}
    coverage: list[ConversationCoverage] = []
    report = AccountImportReport(
        shards_detected=tuple(sorted({shard_stem(name) for name in shards_detected})),
        shards_exported=tuple(sorted({shard_stem(export.shard) for export in exports})),
        filtered_conversations=_as_count(filtered_conversations),
    )

    for conversation_id, rows in grouped.items():
        shards: list[tuple[str, WeFlowParseResult]] = []
        for row in rows:
            payload = json.loads(Path(row.path).read_text(encoding="utf-8-sig"))
            # Parse with the conversation's own id so ids, dedupe keys and provenance are all scoped
            # to this conversation and cannot collide with another's.
            shards.append((row.shard, parse_weflow_events(payload, conversation_id=conversation_id)))

        events, merge = merge_shard_events(shards, conversation_id=conversation_id)
        events_by_conversation[conversation_id] = list(events)

        report.messages_received += merge.received
        report.duplicates_removed += merge.duplicates_removed
        report.undedupeable += merge.undedupeable
        report.skipped_messages += sum(info.skipped for info in merge.shards)

        coverage.append(
            ConversationCoverage(
                conversation_id=conversation_id,
                shards=tuple(sorted(row.shard for row in rows)),
                messages=len(events),
                duplicates_removed=merge.duplicates_removed,
                first_timestamp=merge.first_timestamp,
                last_timestamp=merge.last_timestamp,
            )
        )

    # A conversation is "imported" only if it actually contributed messages. One that was discovered
    # but whose every export is empty is NOT imported — it is reported as producing no messages, so
    # "N imported of M discovered" can never overstate what is searchable.
    discovered_ids = {descriptor.conversation_id for descriptor in discovered} | set(grouped)
    empty = sorted(cid for cid, rows in events_by_conversation.items() if not rows)
    for conversation_id in empty:
        del events_by_conversation[conversation_id]

    report.conversations_imported = len(events_by_conversation)
    report.conversations_discovered = len(discovered_ids)
    report.conversations_without_messages = tuple(sorted(discovered_ids - set(events_by_conversation)))
    report.messages_kept = sum(len(rows) for rows in events_by_conversation.values())
    report.per_conversation = tuple(
        row for row in coverage if row.conversation_id not in set(empty)
    )

    stamps = [event.timestamp for rows in events_by_conversation.values() for event in rows]
    report.first_timestamp = min(stamps) if stamps else None
    report.last_timestamp = max(stamps) if stamps else None

    exported = set(report.shards_exported)
    report.missing_shards = tuple(sorted(s for s in report.shards_detected if s not in exported))

    return events_by_conversation, report


def crossed_conversation_chunks(
    chunks: Sequence[MemoryChunk],
    events_by_conversation: Mapping[str, Sequence[MemoryEvent]],
) -> tuple[str, ...]:
    """Ids of chunks that hold events from more than one conversation. Must always be empty.

    This is the phase's hard rule expressed as a check rather than a promise: a conversation boundary
    may never be crossed, no matter how close in time or how similar two conversations are. It is
    asserted by the regression tests and re-checked on every real account import, so a future change to
    the ordering cannot quietly reintroduce a cross-conversation retrieval unit.
    """
    owner = {
        event.id: conversation_id
        for conversation_id, events in events_by_conversation.items()
        for event in events
    }
    crossed: list[str] = []
    for chunk in chunks:
        owners = {owner.get(event_id) for event_id in chunk.event_ids}
        owners.discard(None)
        if len(owners) > 1:
            crossed.append(chunk.id)
    return tuple(crossed)


def build_account_sessions(
    events_by_conversation: Mapping[str, Sequence[MemoryEvent]],
    config: SessionConfig,
) -> list[MemoryChunk]:
    """Build sessions **one conversation at a time**, then concatenate.

    This is the step the ordering in the module docstring exists for. Feeding a time-interleaved
    account stream to a single ``build_sessions`` call isolates conversations but shatters coherence:
    Alice 10:00, Bob 10:01, Alice 10:02 flush on every message, so a conversation ends up with one
    chunk per message. Building per conversation gives both — coherence inside a conversation, and a
    boundary that cannot be crossed between them.

    Chunks come back ordered by conversation and then by time, which is the order the index receives
    them in.
    """
    chunks: list[MemoryChunk] = []
    for conversation_id in sorted(events_by_conversation):
        chunks.extend(
            build_sessions(
                events_by_conversation[conversation_id],
                config=config,
                conversation_id=conversation_id,
            )
        )
    return chunks
