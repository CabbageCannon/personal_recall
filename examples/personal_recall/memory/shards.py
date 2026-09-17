"""Multi-shard merge: several WeFlow shard exports become one corpus.

Why this exists. WeChat 3.x splits one conversation's history across several databases
(``Msg/Multi/MSG0.db``, ``MSG1.db``, ``MSG2.db``, …), and the exporter reads exactly **one** file per
run — ``sqlcipherCore.open()`` holds a single connection and every query runs against it (confirmed in
Phase 18B). Whichever shard is configured therefore *is* the visible history. Pointing it at ``MSG0``
silently hides everything in ``MSG2``, which is the failure this module exists to prevent:
**No Data Loaded ≠ No Memory Exists.**

The merge is deliberately at the ``MemoryEvent`` level, before session building. Merging the *output*
instead (one corpus per shard) would cut the same conversation into separate chunks per shard and make
a contact's "last message" the maximum of whichever shard happened to be read.

Ordering rules:

* **Chronology comes from ``createTime``, never from the file name or its mtime.** The three shards on
  this machine happen to be ordered by update time, which is a coincidence of this profile, not a
  property to rely on.
* **Ties are broken by (shard label, position in shard)**, so the merged order is deterministic and
  reproducible rather than dependent on directory enumeration.

Deduplication is deliberately conservative:

* A message is deduplicated only on a **strong identity** — the exporter's ``serverId``. That survives
  a message appearing in two shards after a migration.
* Without ``serverId`` there is no safe cross-shard identity: WeFlow documents that ``localId`` repeats
  across conversations *and shards*, so deduplicating on it would drop real messages. Those events are
  kept and counted as ``undedupeable``, and the report says so rather than quietly guessing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .events import MemoryEvent
from .weflow import WeFlowParseResult, parse_weflow_events

#: Default conversation id for a merged corpus (all shards of one conversation share it).
DEFAULT_MERGED_CONVERSATION = "weflow"

#: Shard exports are looked up by this suffix when a directory is given.
SHARD_SUFFIX = ".json"


@dataclass(frozen=True)
class ShardInfo:
    """What one shard contributed."""

    label: str
    events: int = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    skipped: int = 0
    suppressed_payloads: int = 0
    duplicates: int = 0

    def span(self) -> str:
        if self.first_timestamp is None:
            return "no messages"
        return f"{self.first_timestamp:%Y-%m-%d %H:%M} .. {self.last_timestamp:%Y-%m-%d %H:%M}"


@dataclass(frozen=True)
class MergeReport:
    """Per-shard provenance plus the union coverage of the merged corpus."""

    shards: tuple[ShardInfo, ...]
    received: int = 0
    kept: int = 0
    duplicates_removed: int = 0
    undedupeable: int = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None

    @property
    def duplicate_shards(self) -> tuple[str, ...]:
        return tuple(info.label for info in self.shards if info.duplicates)

    def lines(self) -> list[str]:
        out = [
            f"shards merged : {len(self.shards)}",
            f"messages      : {self.kept} kept of {self.received} "
            f"({self.duplicates_removed} duplicate(s) removed)",
        ]
        if self.first_timestamp is not None:
            out.append(
                f"coverage      : {self.first_timestamp:%Y-%m-%d %H:%M} .. "
                f"{self.last_timestamp:%Y-%m-%d %H:%M}"
            )
        if self.undedupeable:
            out.append(
                f"note          : {self.undedupeable} message(s) carry no serverId, so they cannot be "
                "deduplicated across shards; they were all kept rather than risk dropping real history"
            )
        for info in self.shards:
            out.append(
                f"  {info.label:28} {info.events:7} msgs  {info.span()}"
                + (f"  ({info.duplicates} duplicate)" if info.duplicates else "")
                + (f"  [{info.suppressed_payloads} payload suppressed]" if info.suppressed_payloads else "")
            )
        return out


def discover_shard_files(directory: Path) -> list[Path]:
    """Every ``*.json`` shard export in a directory, in a deterministic order."""
    return sorted(path for path in Path(directory).glob(f"*{SHARD_SUFFIX}") if path.is_file())


def discover_message_shards(multi_dir: Path) -> list[str]:
    """Names of the ``MSG*.db`` shards present in a WeChat ``Msg/Multi`` directory.

    Read-only and names-only: this exists so the report can show which shards *exist* versus which were
    actually exported, which is the whole point of the phase. It never opens a database and never
    touches key material.
    """
    directory = Path(multi_dir)
    if not directory.is_dir():
        return []
    return sorted(
        path.name
        for path in directory.glob("MSG*.db")
        if path.is_file() and not path.name.startswith("FTS")
    )


def _strong_key(conversation_id: str, server_id: Any) -> str | None:
    if server_id in (None, "", 0, "0"):
        return None
    return f"{conversation_id}#s{server_id}"


def merge_shard_events(
    shards: Sequence[tuple[str, WeFlowParseResult]],
    conversation_id: str = DEFAULT_MERGED_CONVERSATION,
) -> tuple[list[MemoryEvent], MergeReport]:
    """Merge parsed shards into one chronological, deduplicated event list.

    ``shards`` is ``[(label, parse_result), …]``. Labels are the shard's provenance and its tie-break
    key, so they must be distinct.
    """
    merged: list[tuple[datetime, str, int, MemoryEvent]] = []
    seen_strong: set[str] = set()
    used_ids: set[str] = set()
    infos: list[ShardInfo] = []
    received = duplicates = undedupeable = 0

    for label, parsed in sorted(shards, key=lambda item: item[0]):
        kept_here = 0
        dup_here = 0
        shard_first: datetime | None = None
        shard_last: datetime | None = None

        for event in parsed.events:
            received += 1
            server_id = event.metadata.get("serverId")
            local_id = event.metadata.get("localId")
            position = event.metadata.get("position", 0)

            key = _strong_key(conversation_id, server_id)
            if key is None:
                undedupeable += 1
                event_id = (
                    f"{conversation_id}#{label}#l{local_id}"
                    if local_id not in (None, "")
                    else f"{conversation_id}#{label}#p{position:05d}"
                )
            else:
                if key in seen_strong:
                    duplicates += 1
                    dup_here += 1
                    continue
                seen_strong.add(key)
                event_id = key

            # Ids must stay unique even when several weak events share a localId.
            unique_id = event_id
            suffix = 1
            while unique_id in used_ids:
                suffix += 1
                unique_id = f"{event_id}~{suffix}"
            used_ids.add(unique_id)

            merged_event = replace(
                event,
                id=unique_id,
                conversation_id=conversation_id,
                metadata={**event.metadata, "source_shard": label},
            )
            merged.append((event.timestamp, label, int(position), merged_event))
            kept_here += 1
            if shard_first is None or event.timestamp < shard_first:
                shard_first = event.timestamp
            if shard_last is None or event.timestamp > shard_last:
                shard_last = event.timestamp

        infos.append(
            ShardInfo(
                label=label,
                events=kept_here,
                first_timestamp=shard_first,
                last_timestamp=shard_last,
                skipped=parsed.skipped,
                suppressed_payloads=parsed.suppressed_payloads,
                duplicates=dup_here,
            )
        )

    merged.sort(key=lambda item: (item[0], item[1], item[2]))
    events = [item[3] for item in merged]

    report = MergeReport(
        shards=tuple(infos),
        received=received,
        kept=len(events),
        duplicates_removed=duplicates,
        undedupeable=undedupeable,
        first_timestamp=events[0].timestamp if events else None,
        last_timestamp=events[-1].timestamp if events else None,
    )
    return events, report


def load_and_merge(
    paths: Sequence[Path],
    conversation_id: str = DEFAULT_MERGED_CONVERSATION,
) -> tuple[list[MemoryEvent], MergeReport]:
    """Read each shard export and merge it. Labels are the file stems.

    Raises on an unreadable or malformed shard rather than skipping it: a silently dropped shard is a
    silently smaller history, which is exactly the failure this module prevents.
    """
    shards: list[tuple[str, WeFlowParseResult]] = []
    for path in sorted(Path(p) for p in paths):
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        shards.append((Path(path).stem, parse_weflow_events(payload, conversation_id=conversation_id)))
    return merge_shard_events(shards, conversation_id=conversation_id)
