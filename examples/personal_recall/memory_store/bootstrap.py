"""The seam between the normalized account importer and the store (§13).

Nothing here parses a WeFlow payload, resolves a name or builds a session. Every one of those is the
existing memory layer's job, and this module's whole content is calling it and handing the result to
``rows.build_snapshot``. That is deliberate: a second path from raw JSON to events would be a second
opinion about what the account contains, and the first thing the two would disagree about is the
thing nobody checked.

Two entry points, because there are two things a store can be asked to do — describe the whole
account, or describe a set of conversations that moved:

* :func:`account_snapshot` — the whole export tree, for a bootstrap.
* :func:`conversation_snapshot` — exactly the conversations named, re-rendered from every shard they
  appear in, for an incremental sync.

They differ only in which exports they select. Both end in the same importer, the same session
builder and the same projection, which is what makes "the incremental result equals a fresh
bootstrap" a property the code has rather than a hope the tests have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import index_cache
import incremental_index
from memory.account import (
    AccountImportReport,
    build_account_sessions,
    crossed_conversation_chunks,
    import_account,
    load_account_directory,
)
from memory.conversations import ConversationDescriptor
from memory.labels import CONVERSATION_LABEL_FILENAME, ConversationLabel, read_label_sidecar
from memory.processor import DOCUMENT_PROJECTION_VERSION
from memory.sessions import SessionConfig

from . import schema
from .rows import StoreSnapshot, build_snapshot


class SnapshotError(ValueError):
    """The account tree could not be turned into a snapshot."""


@dataclass(frozen=True)
class SnapshotResult:
    """A projection plus what produced it, so a caller can report without re-deriving anything."""

    snapshot: StoreSnapshot
    report: AccountImportReport | None = None
    conversations: tuple[str, ...] = ()
    stats: Mapping[str, Any] = field(default_factory=dict)

    @property
    def counts(self) -> dict[str, int]:
        return self.snapshot.counts()


def read_labels(account_dir: Path) -> dict[str, ConversationLabel]:
    """The resolved conversation-name sidecar, or ``{}`` when the tree has none.

    Absence is normal — a tree built before Phase 20.6 has no sidecar — and means the exporter's own
    ``displayName`` is all there is, which :func:`rows._display_label_for` already handles.
    """
    path = Path(account_dir) / CONVERSATION_LABEL_FILENAME
    return read_label_sidecar(path) if path.exists() else {}


def _project(
    events_by_conversation: Mapping[str, Sequence[Any]],
    chunks: Sequence[Any],
    *,
    descriptors: Sequence[ConversationDescriptor],
    labels: Mapping[str, ConversationLabel],
) -> StoreSnapshot:
    return build_snapshot(
        events_by_conversation,
        chunks,
        descriptors=descriptors,
        labels=labels,
        projection_version=DOCUMENT_PROJECTION_VERSION,
    )


def account_snapshot(
    account_dir: Path,
    *,
    shard_dir: Path | None = None,
    session_config: SessionConfig | None = None,
    descriptors: Sequence[ConversationDescriptor] = (),
    labels: Mapping[str, ConversationLabel] | None = None,
) -> SnapshotResult:
    """Project the whole account export tree.

    The same import and the same sessionisation the retrieval index uses, so the store's counts can
    be compared with the index's rather than merely believed (§33).
    """
    account_dir = Path(account_dir)
    config = session_config or SessionConfig()
    layout = load_account_directory(account_dir, shard_dir=shard_dir)
    if not layout.exports:
        raise SnapshotError(f"no conversation exports found under {account_dir.name}")

    events_by_conversation, report = import_account(
        layout.exports,
        shards_detected=layout.shards_detected,
        discovered=layout.descriptors,
        filtered_conversations=layout.filtered_conversations,
    )
    report.notes = report.notes + layout.notes
    chunks = build_account_sessions(events_by_conversation, config)
    _assert_no_crossing(chunks, events_by_conversation)

    snapshot = _project(
        events_by_conversation,
        chunks,
        descriptors=tuple(descriptors) or layout.descriptors,
        labels=read_labels(account_dir) if labels is None else labels,
    )
    return SnapshotResult(
        snapshot=snapshot,
        report=report,
        conversations=tuple(sorted(events_by_conversation)),
        stats=dict(snapshot.stats),
    )


def conversation_snapshot(
    account_dir: Path,
    conversations: Sequence[str],
    *,
    shard_dir: Path | None = None,
    session_config: SessionConfig | None = None,
    descriptors: Sequence[ConversationDescriptor] = (),
    labels: Mapping[str, ConversationLabel] | None = None,
) -> SnapshotResult:
    """Project exactly ``conversations``, each re-rendered from every shard it appears in.

    The re-render is :func:`incremental_index.reimport_events` — the *same* function Phase 21B uses
    to advance the retrieval index — so the database and the index move to the same render of the
    same conversations in the same run, rather than each re-deriving it (§16).

    A conversation in ``conversations`` that no longer has any export produces no rows. That is not
    an error: it is how a conversation is dropped, and the caller passes the whole affected set as
    ``updating`` precisely so the absence is acted on.
    """
    account_dir = Path(account_dir)
    config = session_config or SessionConfig()
    wanted = tuple(dict.fromkeys(str(cid) for cid in conversations))
    reimported = incremental_index.reimport_events(
        account_dir, wanted, shard_dir=shard_dir, session_config=config
    )
    snapshot = _project(
        reimported.events_by_conversation,
        reimported.chunks,
        descriptors=tuple(descriptors) or reimported.descriptors,
        labels=read_labels(account_dir) if labels is None else labels,
    )
    return SnapshotResult(
        snapshot=snapshot,
        report=None,
        conversations=tuple(sorted(reimported.events_by_conversation)),
        stats=dict(snapshot.stats),
    )


def inventory_rows(entries: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
    """``index_cache.SourceFileEntry``s as ``rows.INVENTORY_COLUMNS`` tuples."""
    return tuple(
        (entry.relative_path, entry.size, entry.mtime_ns, entry.kind, entry.shard,
         entry.conversation_id)
        for entry in entries
    )


def inventory_entries(store: Any) -> tuple[Any, ...]:
    """The store's own source inventory, as ``index_cache.SourceFileEntry``s.

    Read back through the *same* type the walk produces, so a stored inventory and a fresh one are
    compared by :func:`sync_wechat.diff_inventories` without either side being re-typed by hand.
    """
    return tuple(
        index_cache.SourceFileEntry(
            relative_path=str(row[0]),
            size=int(row[1]),
            mtime_ns=int(row[2]),
            kind=str(row[3]),
            shard=str(row[4]),
            conversation_id=str(row[5]),
        )
        for row in store.stored_inventory()
    )


def state_fields(
    account_dir: Path,
    *,
    session_config: SessionConfig | None = None,
    source_fingerprint: str | None = None,
    exclude: Sequence[Path] = (),
) -> dict[str, Any]:
    """What ``memory_store_state`` should record about *which* account snapshot this is.

    ``source_fingerprint`` defaults to the value the Phase 21A/21B index manifest computes over the
    same tree, expressed by the same function — so "is the database at the same generation as the
    index" is a comparison of two recorded values rather than an assumption (§18). ``exclude`` must
    therefore be the same list the index passes; a different one would produce a different digest
    over the same tree and make the comparison above silently always false.
    """
    account_dir = Path(account_dir)
    config = session_config or SessionConfig()
    return {
        "store_schema_version": schema.SCHEMA_VERSION,
        "projection_version": DOCUMENT_PROJECTION_VERSION,
        "chunking_fingerprint": index_cache.chunking_fingerprint(config),
        "source_fingerprint": (
            source_fingerprint
            if source_fingerprint is not None
            else index_cache.source_fingerprint(account_dir, exclude=exclude)
        ),
    }


def _assert_no_crossing(chunks: Sequence[Any], events_by_conversation: Mapping[str, Any]) -> None:
    """No retrieval unit may hold events from two conversations. Re-checked, not trusted."""
    crossed = crossed_conversation_chunks(chunks, events_by_conversation)
    if crossed:
        raise SnapshotError(
            "conversation boundary crossed: "
            f"{len(crossed)} chunk(s) mix events from more than one conversation"
        )
