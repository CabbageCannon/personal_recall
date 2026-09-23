"""Planning a store update from a moved export tree (§16, §18).

Phase 21B already answers the hard question — "which conversations moved, and is this delta safe to
advance in place?" — and it answers it in one place, with rules that were tested on a real account.
This module asks *that* function, with the store's own recorded generation as the base instead of an
index manifest. It does not re-derive the affected set, and it does not have a second opinion about
what "safe" means: a store that disagreed with the index about which conversations moved would be
exactly the silent divergence this phase exists to prevent.

What the store checks for itself, and why: the index refuses a delta produced under different
*chunking* or *embedding* rules because its stored vectors would be wrong. The store holds no
vectors, but it does hold chunks, so a chunking change would leave old conversations segmented one
way and new ones another inside a single store. That is not a stale cache, it is an inconsistent
database, so it is refused — and the refusal names the difference rather than shrugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import incremental_index
import index_cache
from memory.account import load_account_directory
from memory.conversations import shard_stem
from memory.processor import DOCUMENT_PROJECTION_VERSION
from memory.sessions import SessionConfig

from . import schema
from . import bootstrap
from .bootstrap import SnapshotResult, conversation_snapshot, inventory_rows, state_fields
from .store import PostgresMemoryStore

NO_CHANGE = "no_change"
INCREMENTAL = "incremental"
UNSAFE = "unsafe"


@dataclass(frozen=True)
class SyncPlan:
    """What a store sync would do, or why it will not do it."""

    outcome: str
    reason: str = ""
    affected: tuple[str, ...] = ()
    #: Which recorded fingerprints disagree with this run's configuration. Empty on a clean plan.
    stale: tuple[str, ...] = ()
    classification: Any = None
    source_fingerprint: str | None = None

    @property
    def is_incremental(self) -> bool:
        return self.outcome == INCREMENTAL

    @property
    def is_no_change(self) -> bool:
        return self.outcome == NO_CHANGE

    def lines(self) -> list[str]:
        if self.outcome == NO_CHANGE:
            return ["store          : source is identical to the stored generation; nothing to do"]
        if self.outcome == INCREMENTAL:
            return [
                f"store          : source moved; {len(self.affected)} conversation(s) to rewrite"
            ]
        return [f"store          : NOT advanced ({self.reason})"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "reason": self.reason,
            "conversations_affected": len(self.affected),
            "stale": list(self.stale),
        }


def plan_store_sync(
    store: PostgresMemoryStore,
    account_dir: Path,
    *,
    shard_dir: Path | None = None,
    index_dir: Path | None = None,
    cache_dir: Path | None = None,
    session_config: SessionConfig | None = None,
) -> SyncPlan:
    """Decide whether the store may be advanced in place, and over which conversations.

    Never raises for a refusal: every reason to decline is a :data:`UNSAFE` plan carrying the reason,
    because the caller's answer to all of them is the same and it is not an exception — it is
    ``--rebuild-postgres``.
    """
    account_dir = Path(account_dir)
    config = session_config or SessionConfig()
    state = store.store_state()
    if state is None:
        return SyncPlan(UNSAFE, "the store holds no generation; bootstrap it first")

    # The same exclusion the index passes, so the two fingerprints describe the same walk. A digest
    # taken over a different file set would compare unequal for an unchanged tree, every time.
    exclude = (index_cache.account_cache_dir(account_dir, index_dir).parent,)
    wanted = state_fields(account_dir, session_config=config, exclude=exclude)

    stale: list[str] = []
    if int(state["store_schema_version"]) != schema.SCHEMA_VERSION:
        stale.append("schema")
    if int(state["projection_version"]) != DOCUMENT_PROJECTION_VERSION:
        stale.append("document projection")
    if str(state["chunking_fingerprint"]) != wanted["chunking_fingerprint"]:
        stale.append("session chunking")
    if stale:
        return SyncPlan(
            UNSAFE,
            "the stored generation was built under different rules (" + ", ".join(stale) + ")",
            stale=tuple(stale),
        )

    base = str(state["source_fingerprint"])
    if base == wanted["source_fingerprint"]:
        return SyncPlan(
            NO_CHANGE,
            "the export tree is identical to the stored generation",
            source_fingerprint=base,
        )

    try:
        current = index_cache.source_inventory_entries(account_dir, exclude=exclude)
    except OSError:
        current = None

    # The base of the delta is the tree the *store's* generation was built from — kept in the store
    # itself, not borrowed from the Phase 21B checkpoint. The checkpoint describes one generation and
    # is advanced by whichever consumer runs first, so an index run would overwrite the store's base
    # and vice versa. Everything downstream of this line is Phase 21B's own classifier, unchanged.
    previous = bootstrap.inventory_entries(store)

    layout = load_account_directory(account_dir, shard_dir=shard_dir)
    detected = {shard_stem(shard) for shard in layout.shards_detected}
    exported = {export.shard for export in layout.exports}
    previously_exported = {entry.shard for entry in previous if entry.shard}
    newly_missing = sorted(detected - exported - previously_exported)

    classification = incremental_index.classify_source_change(
        # The store's own invalidity is the source and nothing else — a structural difference was
        # refused above, by name, before anything was walked.
        inspection=index_cache.Inspection(
            outcome=index_cache.INVALID, reason=index_cache.SOURCE_CHANGED, manifest=None
        ),
        previous_inventory=previous,
        current_inventory=current,
        state=None,
        missing_shards=newly_missing,
        # The generation being advanced from is the store's, not an index manifest's, and the
        # inventory above came from that same record — so the two agree by construction rather than
        # by a check that could pass while describing different trees.
        base_source_fingerprint=base,
    )

    if not classification.is_incremental:
        return SyncPlan(
            UNSAFE,
            classification.reason or "the change is not safe to advance in place",
            classification=classification,
            source_fingerprint=base,
        )
    return SyncPlan(
        INCREMENTAL,
        classification.reason,
        affected=tuple(classification.affected),
        classification=classification,
        source_fingerprint=base,
    )


@dataclass(frozen=True)
class SyncOutcome:
    """A planned store sync that ran, plus what it wrote."""

    plan: SyncPlan
    result: SnapshotResult | None = None
    report: Any = None
    conversations: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.report is not None

    def lines(self) -> list[str]:
        out = list(self.plan.lines())
        out.extend(self.report.lines() if self.report is not None else ())
        return out


def run_store_sync(
    store: PostgresMemoryStore,
    account_dir: Path,
    *,
    plan: SyncPlan,
    shard_dir: Path | None = None,
    session_config: SessionConfig | None = None,
) -> SyncOutcome:
    """Re-render the planned conversations and replace them in the store, in one transaction.

    The conversations are re-rendered **whole**, from every shard they appear in, by the same
    function the index uses — never appended to. A new message can replace the tail chunk it joined,
    and a member gaining a name renames messages the delta never touched, so "append what is new" is
    not an update; it is a different index (§16, §17).
    """
    if not plan.is_incremental:
        return SyncOutcome(plan=plan)
    config = session_config or SessionConfig()
    result = conversation_snapshot(
        account_dir, plan.affected, shard_dir=shard_dir, session_config=config
    )
    cache_parent = index_cache.account_cache_dir(account_dir, None).parent
    inventory = index_cache.source_inventory_entries(account_dir, exclude=(cache_parent,))
    report = store.sync(
        result.snapshot,
        # The whole affected set, not merely the conversations that still render: one whose export
        # is gone must be *dropped*, and passing it here is what says so.
        updating=list(plan.affected),
        state=state_fields(account_dir, session_config=config, exclude=(cache_parent,)),
        # The inventory as it is now: the next sync's base is this generation's tree, including the
        # files whose conversations this run did not touch.
        inventory=inventory_rows(inventory),
    )
    return SyncOutcome(
        plan=plan, result=result, report=report, conversations=plan.affected
    )
