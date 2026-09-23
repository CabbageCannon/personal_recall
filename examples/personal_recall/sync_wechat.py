"""Incremental WeChat sync: ask the exporter for the delta, never for the history.

Phase 21A made an unchanged source load in seconds. Any *change* still cost the whole history — a
full re-export, a full re-import and every chunk re-embedded — because the only signal available was
one aggregate fingerprint that says *that* something moved and nothing about *what*.

This module answers the finer question, in the only order that stays cheap:

1. **`sessions --json` reports one `lastTimestamp` per conversation.** That is a free change signal:
   comparing it with what the last successful sync recorded costs one listing per shard and tells us
   which conversations moved *before* a single message is exported.
2. **Only those conversations are exported, and only from a window.** ``export <talker> json --from
   <checkpoint − overlap>``, never the whole history.
3. **The delta is merged into the conversation's existing export file** — union with the old file,
   deduplicated by ``serverId``, ordered by ``createTime``. History is never re-exported and never
   rewritten from scratch: the merge is a superset operation on what is already there.

Why an overlap window, and why it is not a bug
----------------------------------------------

``weflow-cli`` documents its own read as ``{"mode": "overlapping-time-window", "stableCursor":
false}``: there is no cursor that means "resume exactly here". A window that starts exactly at the
last known timestamp would drop every message sharing that second, and a boundary can always move
between two runs. So the window deliberately starts *before* the checkpoint, the exporter re-sends
the messages that straddle it, and the merge drops them by ``serverId``. The duplicated reads are
the price of the missing cursor, and they are harmless because the dedupe already exists — it is the
same rule ``memory.shards`` applies across shards, applied to the same message arriving twice in
time instead of twice in space.

"No new messages" is an error, not an empty set
-----------------------------------------------

A window with nothing in it makes the CLI fail (``EXPORT_FAILED``/``未找到消息``). An incremental
sync that reports *that* as a failure would look broken on every quiet day, so it is classified as
an empty window and treated as a no-op — but only when both the code **and** the text say so, because
the same code also covers a database that could not be read. Guessing the other way — treating a real
failure as "nothing new" — would silently stop syncing a conversation forever.

What this module will not do
----------------------------

* **It never writes the user's real ``~/.weflow-cli``.** Every exporter call goes through
  ``exporter``'s scratch-profile mechanism, which is the only thing that switches shards.
* **It never deletes.** A conversation that vanished from WeChat stays in the tree: the tree is
  history, and removing a message the index may have cited is not a sync's decision to make.
* **It never touches the index.** It stages deltas, promotes export files, and returns what it
  learned. The index is advanced by ``incremental_index`` *after* the source is complete, and the
  checkpoint is written after both — see that module for the ordering and why it is that order.
* **It refuses rather than guesses.** An unreadable shard, a config that cannot be read, a listing
  that fails: the sync stops, reports, and changes nothing. It does not export "the shards that
  worked" and call the account complete.

Where the state lives
---------------------

``<index dir>/<account leaf>.sync-state.json`` — beside the index entry, inside the private cache
directory (``data/real/.index_cache`` by default, git-ignored). It holds **conversation identity**:
the per-shard ``conversation_id -> lastTimestamp`` map, and an inventory of the export tree's files
whose relative paths contain talker ids. That is allowed *there* and nowhere else — the directory
already holds the chat text every one of those ids wrote — and it is why the file is named in
``.gitignore``'s coverage, never printed, and never summarised into a log line or the project
status. Everything this module reports is a count.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import exporter
import index_cache
from index_cache import SOURCE_INVENTORY_VERSION, SourceFileEntry
from memory.account import load_account_directory
from memory.conversations import conversation_id_from_export_filename, shard_stem
from memory.shards import discover_message_shards
from memory.weflow import unwrap_payload

__all__ = [
    "DEFAULT_OVERLAP_SECONDS",
    "InventoryDiff",
    "SYNC_NO_CHANGE",
    "SYNC_STATE_SUFFIX",
    "SYNC_STATE_VERSION",
    "SYNC_SYNCED",
    "SYNC_UNSAFE",
    "SyncOutcome",
    "SyncState",
    "SyncStateError",
    "diff_inventories",
    "load_sync_state",
    "merge_export_payloads",
    "sync_account_source",
    "sync_state_path",
    "write_sync_state",
]

#: The layout of the sync state file. Bump when a field is added, removed or re-meant: a state this
#: module cannot read *exactly* is a state it must not advance from.
SYNC_STATE_VERSION = 1

#: Appended to the cache entry's own name, so one account's sync state sits beside its index and can
#: never be confused with another account's. It is not inside the entry because an entry is replaced
#: wholesale by a rename at the end of every promotion.
SYNC_STATE_SUFFIX = ".sync-state.json"

#: How far *before* the checkpoint a delta window starts. Long enough to survive a message whose
#: ``createTime`` ties the boundary, a clock that moved, and an export that ran while the database
#: was being written; short enough that re-reading it costs nothing. Six hours matches the session
#: chunker's ``max_gap``, so the re-read can never reach back past the tail chunk it might rebuild.
DEFAULT_OVERLAP_SECONDS = 6 * 60 * 60

#: Delta staging lives under the index root, which is excluded from the source inventory: a staging
#: directory inside the account tree would change the fingerprint of the tree it is about to update.
SYNC_STAGING_INFIX = ".sync-staging-"

#: The three things a sync can say. ``SYNC_NO_CHANGE`` is the cheap path (one listing per shard, no
#: export, no write); ``SYNCED`` means files were promoted; ``SYNC_UNSAFE`` means nothing was.
SYNC_NO_CHANGE = "NO_CHANGE"
SYNC_SYNCED = "SYNCED"
SYNC_UNSAFE = "UNSAFE_CHANGE"


class SyncStateError(ValueError):
    """The sync state is present but is not something this module may advance from.

    Absent is *not* this error: a tree that was never synced has no state, and the honest response
    there is a full export. Present-and-unreadable is different — it means the base is unknown, and
    an unknown base is the one thing an incremental sync must not invent.
    """


# ---------------------------------------------------------------------------------------------
# The state file
# ---------------------------------------------------------------------------------------------


def sync_state_path(cache_dir: Path) -> Path:
    """Where one account's sync state lives: beside its index entry, never inside it."""
    cache_dir = Path(cache_dir)
    return cache_dir.with_name(f"{cache_dir.name}{SYNC_STATE_SUFFIX}")


@dataclass(frozen=True)
class SyncState:
    """What the last successful sync knew, in counts, timestamps and relative paths.

    Deliberately carries **conversation identity** — the talker ids are the keys of the timestamp map
    and appear inside the inventory's relative paths. This file lives in the private cache directory,
    next to an index that already holds the text those conversations are made of; it is never
    printed, never summarised and never copied anywhere else.

    Deliberately carries no message text, no display name, no ``serverId``, no absolute path, and no
    chunk or session information: a sync decision needs to know *which conversation moved and how
    far*, and nothing about what was said.
    """

    generation: str
    advanced_at: str
    shards: tuple[str, ...]
    conversations: Mapping[str, Mapping[str, int]]
    index_source_fingerprint: str
    inventory: tuple[SourceFileEntry, ...]
    version: int = SYNC_STATE_VERSION
    inventory_version: int = SOURCE_INVENTORY_VERSION

    def last_timestamp(self, shard: str, conversation_id: str) -> int | None:
        """The ``lastTimestamp`` this (shard, conversation) had at the last successful sync."""
        value = (self.conversations.get(shard) or {}).get(conversation_id)
        return int(value) if isinstance(value, int) else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sync_state_version": self.version,
            "source_inventory_version": self.inventory_version,
            "generation": self.generation,
            "advanced_at": self.advanced_at,
            "shards": list(self.shards),
            "conversations": {
                shard: {cid: int(stamp) for cid, stamp in sorted(rows.items())}
                for shard, rows in sorted(self.conversations.items())
            },
            "index_source_fingerprint": self.index_source_fingerprint,
            "inventory": [entry.as_dict() for entry in self.inventory],
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "SyncState":
        """Strictly. Every field below is used to decide whether work may be skipped."""
        if not isinstance(payload, Mapping):
            raise SyncStateError("the sync state is not an object")
        expected = {
            "sync_state_version",
            "source_inventory_version",
            "generation",
            "advanced_at",
            "shards",
            "conversations",
            "index_source_fingerprint",
            "inventory",
        }
        if set(payload) != expected:
            raise SyncStateError("the sync state has unexpected or missing fields")

        version = _int_field(payload, "sync_state_version")
        if version != SYNC_STATE_VERSION:
            raise SyncStateError(
                f"sync state version changed ({version} -> {SYNC_STATE_VERSION})"
            )
        inventory_version = _int_field(payload, "source_inventory_version")
        if inventory_version != SOURCE_INVENTORY_VERSION:
            raise SyncStateError(
                f"source inventory version changed ({inventory_version} -> "
                f"{SOURCE_INVENTORY_VERSION})"
            )
        generation = payload.get("generation")
        if not isinstance(generation, str) or not generation:
            raise SyncStateError("the sync state has no generation")
        advanced_at = payload.get("advanced_at")
        if not isinstance(advanced_at, str) or not advanced_at:
            raise SyncStateError("the sync state has no timestamp")
        try:
            datetime.fromisoformat(advanced_at)
        except ValueError as exc:
            raise SyncStateError("the sync state's timestamp is not ISO-8601") from exc

        fingerprint = payload.get("index_source_fingerprint")
        if not _is_digest(fingerprint):
            raise SyncStateError("the sync state's index fingerprint is not a sha256 digest")

        shards = payload.get("shards")
        if not isinstance(shards, list) or any(not isinstance(s, str) or not s for s in shards):
            raise SyncStateError("the sync state's shard list is not a list of names")
        if len(set(shards)) != len(shards):
            raise SyncStateError("the sync state's shard list repeats a shard")

        conversations = payload.get("conversations")
        if not isinstance(conversations, Mapping):
            raise SyncStateError("the sync state's conversation map is not an object")
        parsed_conversations: dict[str, dict[str, int]] = {}
        for shard, rows in conversations.items():
            if not isinstance(shard, str) or not isinstance(rows, Mapping):
                raise SyncStateError("a conversation row is not a shard -> talker -> timestamp map")
            if shard not in set(shards):
                raise SyncStateError("the conversation map names a shard the shard list does not")
            parsed: dict[str, int] = {}
            for cid, stamp in rows.items():
                if not isinstance(cid, str) or not cid:
                    raise SyncStateError("a conversation row has an empty talker")
                if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp < 0:
                    raise SyncStateError("a conversation row has no usable timestamp")
                parsed[cid] = int(stamp)
            parsed_conversations[shard] = parsed

        inventory = payload.get("inventory")
        if not isinstance(inventory, list):
            raise SyncStateError("the sync state's inventory is not a list")
        entries: list[SourceFileEntry] = []
        try:
            for raw in inventory:
                entries.append(SourceFileEntry.from_dict(raw))
        except ValueError as exc:
            raise SyncStateError(f"the sync state's inventory is unusable ({exc})") from exc
        if len({entry.relative_path for entry in entries}) != len(entries):
            raise SyncStateError("the sync state's inventory repeats a path")

        return cls(
            version=version,
            inventory_version=inventory_version,
            generation=generation,
            advanced_at=advanced_at,
            shards=tuple(shards),
            conversations=parsed_conversations,
            index_source_fingerprint=str(fingerprint),
            inventory=tuple(entries),
        )


def _int_field(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SyncStateError(f"{key} is not a non-negative integer")
    return value


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def load_sync_state(cache_dir: Path) -> SyncState | None:
    """Read the sync state, or ``None`` when there is none. Raises for a state that is unusable.

    ``None`` and an exception are different answers and the caller treats them differently: no state
    means no base, which the *sync* answers with a full export; an unusable state means the base is
    unknown, which nothing may advance from.
    """
    path = sync_state_path(cache_dir)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise SyncStateError(f"the sync state is unreadable ({type(exc).__name__})") from exc
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise SyncStateError("the sync state is not JSON") from exc
    return SyncState.from_dict(payload)


def write_sync_state(cache_dir: Path, state: SyncState) -> Path:
    """Write the state atomically: a full file replaces the old one, never a partial write.

    This is the **last** thing a successful update does (see ``incremental_index``). Written first,
    it would claim work that had not happened yet; written non-atomically, a kill mid-write would
    leave a state that parses as neither the old generation nor the new one.
    """
    path = sync_state_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{uuid4().hex[:8]}")
    temporary.write_text(
        index_cache.canonical_json(state.as_dict()), encoding="utf-8"
    )
    try:
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return path


def cleanup_sync_staging(cache_dir: Path) -> int:
    """Remove leftover delta staging for this account; returns how many directories went.

    Called before a sync starts, so an interrupted one leaves no rubble for the next to reason
    about. Scoped to this account's leaf: a second account's in-flight sync is not disturbed.
    """
    cache_dir = Path(cache_dir)
    root = cache_dir.parent
    if not root.is_dir():
        return 0
    removed = 0
    for path in sorted(root.glob(f"{cache_dir.name}{SYNC_STAGING_INFIX}*")):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed


# ---------------------------------------------------------------------------------------------
# Which files and which conversations moved
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class InventoryDiff:
    """The difference between the tree the index was built from and the tree on disk now.

    ``removed`` is the reason this is a diff and not just a list of changes: a file that existed and
    no longer does is history the index can no longer account for, and no incremental update may
    quietly proceed past it. It is reported, never acted on.
    """

    added: tuple[SourceFileEntry, ...] = ()
    changed: tuple[SourceFileEntry, ...] = ()
    removed: tuple[SourceFileEntry, ...] = ()
    #: Conversation ids whose files changed or appeared — the unit of work, never the file.
    affected: tuple[str, ...] = ()
    #: Export files whose conversation cannot be established (no shard directory, no talker in the
    #: name). Their owner is ambiguous, and an ambiguous owner is refused rather than guessed.
    ambiguous: tuple[str, ...] = ()

    @property
    def touched_files(self) -> int:
        return len(self.added) + len(self.changed)

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.changed or self.removed)


def diff_inventories(
    previous: Sequence[SourceFileEntry], current: Sequence[SourceFileEntry]
) -> InventoryDiff:
    """Diff two inventories by relative path, attributing every change to a conversation.

    A conversation is the unit the caller acts on, not the file: a conversation's messages live in
    one or more shards, so "this file changed" and "this conversation moved" are different statements
    and only the second is safe to re-import from.
    """
    before = {entry.relative_path: entry for entry in previous}
    after = {entry.relative_path: entry for entry in current}

    added: list[SourceFileEntry] = []
    changed: list[SourceFileEntry] = []
    affected: set[str] = set()
    ambiguous: list[str] = []

    for path, entry in after.items():
        old = before.get(path)
        if old is None:
            added.append(entry)
        elif (old.size, old.mtime_ns) != (entry.size, entry.mtime_ns):
            changed.append(entry)
        else:
            continue
        if entry.kind == index_cache.KIND_EXPORT:
            if entry.conversation_id and entry.shard:
                affected.add(entry.conversation_id)
            else:
                ambiguous.append(path)

    removed = [before[path] for path in sorted(set(before) - set(after))]

    # Ambiguity is a property of the whole inventory, not of what moved: an export whose owner
    # cannot be established *now* is a reason to refuse now, whether or not this diff touched it.
    for entry in (*after.values(), *before.values()):
        if entry.kind == index_cache.KIND_EXPORT and not (
            entry.conversation_id and entry.shard
        ):
            ambiguous.append(entry.relative_path)

    return InventoryDiff(
        added=tuple(sorted(added, key=lambda entry: entry.relative_path)),
        changed=tuple(sorted(changed, key=lambda entry: entry.relative_path)),
        removed=tuple(removed),
        affected=tuple(sorted(affected)),
        ambiguous=tuple(sorted(set(ambiguous))),
    )


# ---------------------------------------------------------------------------------------------
# Merging a delta into an export file
# ---------------------------------------------------------------------------------------------


def _create_time_seconds(entry: Mapping[str, Any]) -> int:
    """``createTime`` in seconds, or ``0`` when it is absent or unusable.

    Milliseconds are normalised away because the column is seconds in the database and a tolerant
    reader is not a different chronology. This is a *sort* key only — the reader that turns messages
    into events does its own timestamp handling, and a message whose stamp cannot be read must still
    reach it rather than be dropped here.
    """
    raw = entry.get("createTime")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 0
    return value // 1000 if value > 10_000_000_000 else value


def _dedupe_key(entry: Mapping[str, Any]) -> str:
    """How one message object is recognised as "already in the tree".

    ``serverId`` when there is one — the same conservative identity ``memory.shards`` dedupes on.
    Without one, **exact structural equality**, because the alternative is worse in both
    directions: dropping a keyless message on a weaker key (``localId`` repeats across shards and
    conversations) would delete real history, and *keeping* it would re-add the same message on
    every overlapping-window sync until the file grew without bound. Two byte-identical message
    objects are the same message; two messages that differ anywhere — including in their
    ``localId`` — are both kept.
    """
    server_id = entry.get("serverId")
    if server_id not in (None, "", 0, "0"):
        return "s" + str(server_id)
    return "j" + json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)


_missing = object()


@dataclass(frozen=True)
class MergeOutcome:
    """The result of merging a delta into an export payload."""

    payload: Any
    #: Entries in the merged payload.
    kept: int
    #: Delta entries that were already present, by ``serverId`` or by exact structural equality.
    duplicates: int
    #: How many entries this delta actually contributed. Zero means the file must not be rewritten
    #: at all — see :func:`sync_account_source`.
    added: int


def merge_export_payloads(existing: Any, delta: Any) -> MergeOutcome:
    """Union two export payloads: everything already there, plus what is genuinely new.

    The existing entry wins a tie, so a message already in the tree is not replaced by the
    exporter's copy of it — the tree is the record the index was built from.

    The order is by ``createTime``, with the previous relative order preserved for equal stamps (a
    stable sort), so merging the same delta twice is a no-op on the bytes as well as on the set.

    The envelope is preserved: a payload that carried ``{"schema": …, "messages": […]}"`` is written
    back in that shape, because the parser records the schema on every event and dropping it would
    change metadata for no reason.
    """
    old_entries = [entry for entry in unwrap_payload(existing)[0]]
    new_entries = [entry for entry in unwrap_payload(delta)[0]]

    merged: list[Any] = []
    seen: set[str] = set()
    duplicates = 0
    for entry in old_entries:
        # A non-object entry is not a message. Keep it rather than silently dropping it: the parser
        # counts it as skipped, which is visible, and a merge is not a validator.
        merged.append(entry)
        if isinstance(entry, Mapping):
            seen.add(_dedupe_key(entry))

    added = 0
    for entry in new_entries:
        if isinstance(entry, Mapping):
            key = _dedupe_key(entry)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
        added += 1
        merged.append(entry)

    merged.sort(key=lambda entry: _create_time_seconds(entry) if isinstance(entry, Mapping) else 0)
    envelope = None
    if isinstance(existing, Mapping) and isinstance(existing.get("messages"), list):
        envelope = dict(existing)
        envelope["messages"] = merged
    return MergeOutcome(
        payload=merged if envelope is None else envelope,
        kept=len(merged),
        duplicates=duplicates,
        added=added,
    )


def _read_payload(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _promote_file(target: Path, payload: Any) -> None:
    """Write a payload beside its target and rename it into place.

    A reader of the tree (the importer, another process) sees either the old file or the new one,
    never a half-written one, and a kill mid-write leaves the old file untouched.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp-{uuid4().hex[:8]}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    try:
        os.replace(temporary, target)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------------------------
# The sync
# ---------------------------------------------------------------------------------------------


@dataclass
class SyncOutcome:
    """What one sync did, in counts. Never a talker, a path, a message or a shard name."""

    ok: bool = True
    outcome: str = SYNC_NO_CHANGE
    reason: str = ""
    #: How many (shard, conversation) pairs were exported, and how many came back empty.
    exports_run: int = 0
    empty_windows: int = 0
    #: Export files promoted into the tree (one per conversation that actually gained messages).
    files_promoted: int = 0
    #: New message objects added across those files — a superset of the delta, since the overlap
    #: window re-sends the boundary and the dedupe removes it.
    messages_added: int = 0
    duplicates_dropped: int = 0
    #: Conversations the tree already had that were refreshed, and conversations added to it.
    conversations_refreshed: int = 0
    conversations_added: int = 0
    #: Conversations the live listing knows and the tree does not, skipped because the export was
    #: originally narrowed. Reported so a narrowing is never silently widened.
    conversations_skipped: int = 0
    #: Failures, redacted. Each one leaves its checkpoint entry unadvanced, so the next sync retries.
    failures: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    #: What the checkpoint should record next: ``shard -> talker -> lastTimestamp``. Identity-bearing,
    #: written to the private state file and never printed.
    advanced_timestamps: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    @property
    def wrote_something(self) -> bool:
        return bool(self.files_promoted)

    def lines(self) -> list[str]:
        """What the sync did, in counts. Unprefixed: the caller labels the block."""
        out: list[str] = []
        if self.outcome == SYNC_NO_CHANGE:
            out.append(f"no conversation moved ({self.exports_run} export(s) considered)")
        elif self.outcome == SYNC_UNSAFE:
            out.append(f"REFUSED ({self.reason}); the export tree was not touched")
        else:
            out.append(
                f"{self.exports_run} window(s) exported, {self.empty_windows} with nothing new"
            )
            out.append(
                f"{self.messages_added} new message(s) merged into {self.files_promoted} export "
                f"file(s); {self.duplicates_dropped} re-sent message(s) were already in the tree"
            )
            out.append(
                f"{self.conversations_refreshed} conversation(s) refreshed, "
                f"{self.conversations_added} added"
            )
        if self.conversations_skipped:
            out.append(
                f"NOTE: {self.conversations_skipped} conversation(s) exist in WeChat and not in "
                "this tree; they were NOT exported, because this tree was narrowed when it was built"
            )
        for note in self.notes:
            out.append(f"note: {note}")
        for failure in self.failures:
            out.append(f"failure: {failure}")
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "outcome": self.outcome,
            "reason": self.reason,
            "exports_run": self.exports_run,
            "empty_windows": self.empty_windows,
            "files_promoted": self.files_promoted,
            "messages_added": self.messages_added,
            "duplicates_dropped": self.duplicates_dropped,
            "conversations_refreshed": self.conversations_refreshed,
            "conversations_added": self.conversations_added,
            "conversations_skipped": self.conversations_skipped,
            "failures": list(self.failures),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class _PlannedExport:
    """One (shard, conversation) window the sync intends to ask for."""

    shard: str
    conversation_id: str
    #: ``None`` means "no lower bound": the caller has no base for this pair, so the whole
    #: conversation is exported. Never a guess — the window is chosen per pair, not per account.
    since: datetime | None
    known: int | None


def plan_exports(
    listed: Mapping[str, Mapping[str, int | None]],
    state: SyncState | None,
    *,
    known_conversations: Sequence[str] = (),
    overlap_seconds: int = DEFAULT_OVERLAP_SECONDS,
    include_unknown: bool = True,
) -> tuple[list[_PlannedExport], list[str]]:
    """Which (shard, conversation) windows to ask for, and what could not be decided.

    The cheap signal first: a conversation whose ``lastTimestamp`` equals the one the last sync
    recorded is not exported at all. Everything else gets a window — and a pair with no recorded
    timestamp, or one whose timestamp went *backwards*, gets an unbounded one, because a window
    computed from a base that is missing or wrong would skip exactly the messages it was meant to
    fetch.
    """
    plans: list[_PlannedExport] = []
    notes: list[str] = []
    known = {str(cid) for cid in known_conversations}
    skipped_unknown = 0
    unbounded = 0

    for shard in sorted(listed):
        for conversation_id in sorted(listed[shard]):
            if known and conversation_id not in known and not include_unknown:
                skipped_unknown += 1
                continue
            live = listed[shard][conversation_id]
            saved = None if state is None else state.last_timestamp(shard, conversation_id)
            if saved is not None and live is not None and live == saved:
                continue
            if saved is None or live is None or live < saved:
                unbounded += 1
                plans.append(
                    _PlannedExport(
                        shard=shard, conversation_id=conversation_id, since=None, known=live
                    )
                )
                continue
            plans.append(
                _PlannedExport(
                    shard=shard,
                    conversation_id=conversation_id,
                    since=datetime.fromtimestamp(saved - max(0, int(overlap_seconds))),
                    known=live,
                )
            )

    if unbounded:
        notes.append(
            f"{unbounded} conversation window(s) had no usable checkpoint (a new conversation, a "
            "listing without a timestamp, or a shard whose newest message moved backwards), so "
            "those were exported in full rather than from a window"
        )
    if skipped_unknown:
        notes.append(
            f"{skipped_unknown} conversation(s) the live listing knows were skipped because this "
            "tree was a narrowed export"
        )
    return plans, notes


def list_live_timestamps(
    shards: Sequence[str],
    multi_dir: Path,
    *,
    runner: exporter.Runner | None = None,
    scratch_root: Path | None = None,
) -> dict[str, dict[str, int | None]]:
    """``shard -> talker -> lastTimestamp``, from one live listing per shard.

    Deliberately the live listing and not the tree's ``sessions.json``: the tree's copy is what the
    *export* saw, and the whole point of this call is what the database holds now. One call per
    shard, no message and no database read, which is what makes "nothing changed" cheap.

    A ``None`` value means the listing carried no usable ``lastTimestamp`` for that conversation —
    a fact the caller must treat as "cannot prove it did not move", never as "did not move". Raises
    :class:`exporter.ListingFailed` when a shard could not be asked at all: an unreadable shard
    cannot be proven unchanged either.
    """
    listed: dict[str, dict[str, int | None]] = {}
    for shard in shards:
        descriptors = exporter.list_conversations(
            shard, Path(multi_dir), runner=runner, scratch_root=scratch_root
        )
        row: dict[str, int | None] = {}
        for descriptor in descriptors:
            stamp = descriptor.metadata.get("lastTimestamp")
            row[descriptor.conversation_id] = (
                None if isinstance(stamp, bool) or not isinstance(stamp, int) else int(stamp)
            )
        listed[shard] = row
    return listed


def sync_account_source(
    account_dir: Path,
    *,
    multi_dir: Path,
    index_dir: Path | None = None,
    cache_dir: Path | None = None,
    state: SyncState | None = None,
    state_loaded: bool = False,
    runner: exporter.Runner | None = None,
    scratch_root: Path | None = None,
    overlap_seconds: int = DEFAULT_OVERLAP_SECONDS,
) -> SyncOutcome:
    """Fetch whatever moved since the last successful sync, and merge it into the tree.

    The order is the whole crash-safety story, and it is *not* the obvious one:

    read the state → list the shards (cheap) → export only the moved windows **into staging** →
    validate each delta → merge and promote the export files → return what the checkpoint should
    record.

    The state is **not** advanced here. Nothing this function does is worth a claim it has not
    earned: a kill anywhere leaves the previous generation loadable and the next sync re-deriving
    the same delta from the same unadvanced base, which the ``serverId`` dedupe makes harmless.
    """
    account_dir = Path(account_dir)
    cache = (
        Path(cache_dir)
        if cache_dir is not None
        else index_cache.account_cache_dir(account_dir, index_dir)
    )

    if not state_loaded:
        try:
            state = load_sync_state(cache)
        except SyncStateError as exc:
            return SyncOutcome(
                ok=False,
                outcome=SYNC_UNSAFE,
                reason=f"the sync state is unusable ({exc})",
            )

    shards = tuple(
        shard_stem(name) for name in discover_message_shards(Path(multi_dir))
    )
    if not shards:
        return SyncOutcome(
            ok=False,
            outcome=SYNC_UNSAFE,
            reason="no message shard was found, so there is nothing to sync from",
        )

    try:
        listed = list_live_timestamps(
            shards, Path(multi_dir), runner=runner, scratch_root=scratch_root
        )
    except (exporter.ExporterError, OSError) as exc:
        # A shard that cannot be listed cannot be proven unchanged. Nothing has been written yet,
        # and nothing will be.
        return SyncOutcome(
            ok=False,
            outcome=SYNC_UNSAFE,
            reason=f"a shard could not be listed ({type(exc).__name__})",
        )

    # Which conversations this tree already holds, and whether it was deliberately narrowed.
    layout = load_account_directory(account_dir)
    known_conversations = tuple(
        sorted({export.conversation_id for export in layout.exports})
    )
    narrowed = layout.filtered_conversations > 0

    plans, notes = plan_exports(
        listed,
        state,
        known_conversations=known_conversations,
        overlap_seconds=overlap_seconds,
        include_unknown=not narrowed,
    )
    outcome = SyncOutcome(notes=tuple(notes), conversations_skipped=0)
    if narrowed:
        live_ids = {cid for row in listed.values() for cid in row}
        outcome.conversations_skipped = len(live_ids - set(known_conversations))
    if not plans:
        outcome.outcome = SYNC_NO_CHANGE
        return outcome

    # Every write from here on happens under the index root, never inside the account tree: a
    # staging directory in the tree would change the fingerprint of the very source being updated.
    root = cache.parent
    root.mkdir(parents=True, exist_ok=True)
    cleanup_sync_staging(cache)
    staging = root / f"{cache.name}{SYNC_STAGING_INFIX}{uuid4().hex[:8]}"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        advanced: dict[str, dict[str, int]] = {}
        # Accumulated as a list and frozen onto the outcome at the end: ``SyncOutcome`` is the value
        # the caller reads and reports, and a value that is mutated in place across a long loop is
        # the one thing a partial failure could leave half-written.
        failures: list[str] = []
        for plan in plans:
            result = exporter.export_conversation(
                plan.conversation_id,
                staging / plan.shard,
                shard=plan.shard,
                multi_dir=Path(multi_dir),
                runner=runner,
                scratch_root=scratch_root,
                since=plan.since,
            )
            outcome.exports_run += 1
            if not result.ok and not result.empty_window:
                # Not advanced: the next sync asks for the same window again and retries. The
                # exporter's own message names the conversation, and a failure line is a log line —
                # so the shard and the machine-readable code are what is kept, and nothing else.
                failures.append(
                    f"a window for shard {plan.shard} could not be exported "
                    f"(code={exporter.redact(result.code) or 'unknown'})"
                )
                continue
            if result.empty_window:
                outcome.empty_windows += 1
                _record(advanced, plan, plan.known)
                continue

            delta_path = Path(result.path)
            if not delta_path.is_file():
                delta_path = staging / plan.shard / f"{plan.conversation_id}_messages.json"
            try:
                delta_payload = _read_payload(delta_path)
                entries, _ = unwrap_payload(delta_payload)
                if not entries or not all(isinstance(e, Mapping) for e in entries):
                    raise ValueError("the delta is not a list of message objects")
            except (OSError, ValueError) as exc:
                failures.append(
                    f"a delta for shard {plan.shard} was not a usable export "
                    f"({type(exc).__name__}); its window was not advanced"
                )
                continue

            target = account_dir / plan.shard / f"{plan.conversation_id}_messages.json"
            existing = _missing
            if target.is_file():
                try:
                    existing = _read_payload(target)
                except (OSError, ValueError) as exc:
                    failures.append(
                        f"an existing export for shard {plan.shard} could not be read "
                        f"({type(exc).__name__}); its window was not advanced"
                    )
                    continue
            merge = merge_export_payloads(
                None if existing is _missing else existing, delta_payload
            )
            outcome.duplicates_dropped += merge.duplicates
            if existing is not _missing and not merge.added:
                # The overlap window re-sent messages that are already in the tree — the normal
                # result of a sync on a conversation whose only movement was the listing's clock.
                # The file must NOT be rewritten: a new mtime is a source change, and a source
                # change is what makes the index refuse to load.
                _record(advanced, plan, plan.known)
                continue
            _promote_file(target, merge.payload)
            outcome.files_promoted += 1
            outcome.messages_added += merge.added
            if existing is _missing:
                outcome.conversations_added += 1
            else:
                outcome.conversations_refreshed += 1
            _record(advanced, plan, plan.known)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    outcome.failures = tuple(failures)
    outcome.advanced_timestamps = {
        shard: dict(row) for shard, row in sorted(advanced.items())
    }
    outcome.outcome = SYNC_SYNCED if outcome.files_promoted else SYNC_NO_CHANGE
    if outcome.failures:
        # Not ``SYNC_UNSAFE`` — the tree is consistent and the failed windows were simply not
        # advanced, so the next sync retries them. But not ``ok`` either: a run that could not read
        # part of what it was asked for must never read as one that did.
        outcome.ok = False
        outcome.notes = outcome.notes + (
            f"{len(outcome.failures)} window(s) failed and were left unadvanced, so the next sync "
            "will ask for them again",
        )
    return outcome


def _record(
    advanced: dict[str, dict[str, int]], plan: _PlannedExport, stamp: int | None
) -> None:
    """Record what this (shard, conversation) is now known up to. Nothing is recorded for ``None``."""
    if stamp is None:
        return
    advanced.setdefault(plan.shard, {})[plan.conversation_id] = int(stamp)


# ---------------------------------------------------------------------------------------------
# The state an update should persist
# ---------------------------------------------------------------------------------------------


def advanced_state(
    *,
    previous: SyncState | None,
    inventory: Sequence[SourceFileEntry],
    shards: Sequence[str],
    index_source_fingerprint: str,
    advanced_timestamps: Mapping[str, Mapping[str, int]] | None = None,
    now: datetime | None = None,
) -> SyncState:
    """The state a completed update should write — built, never written, by this function.

    Timestamps are **merged**, never replaced: a sync touches a few conversations, and the rest keep
    the value the last successful sync recorded for them. Replacing the map would forget every
    conversation the run did not happen to visit, and the next sync would re-export them in full.

    ``previous`` is the checkpoint of the generation being *carried forward*, and the caller decides
    whether it may be: an older checkpoint that describes a different tree must not donate its
    timestamps, because "we are up to here" would then be a statement about a history this generation
    never held.
    """
    merged: dict[str, dict[str, int]] = {}
    if previous is not None:
        for shard, rows in previous.conversations.items():
            merged.setdefault(shard, {}).update(rows)
    for shard, rows in (advanced_timestamps or {}).items():
        merged.setdefault(shard, {}).update({cid: int(stamp) for cid, stamp in rows.items()})
    # A shard that is gone is dropped, and one that appeared starts empty: the map describes the
    # shards this account *has*, which is what a later sync compares a live listing against.
    current = {shard_stem(name) for name in shards}
    merged = {shard: rows for shard, rows in merged.items() if shard in current}
    return SyncState(
        generation=uuid4().hex,
        advanced_at=(now or datetime.now(timezone.utc)).replace(microsecond=0).isoformat(),
        shards=tuple(sorted(current)),
        conversations={shard: dict(sorted(rows.items())) for shard, rows in sorted(merged.items())},
        index_source_fingerprint=index_source_fingerprint,
        inventory=tuple(inventory),
    )
