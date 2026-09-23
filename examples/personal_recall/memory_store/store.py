"""The canonical memory store: the one module that talks to PostgreSQL.

Every SQL statement in this project lives here or in ``migrations/``. Nothing else opens a
connection, so "what does this product do to the database" is a question with a single file for an
answer — no SQL scattered through the CLI, the web app, the eval runner or the sync script (§21).

The store is a **projection of the normalized memory layer**, and the direction of dependency is the
point: ``rows.py`` turns events and chunks into tuples, this module writes them, and nothing here
re-parses an export or re-derives a label (§13). Rebuilding the store means re-running the importer,
which is the same code the retrieval index already runs — so the two cannot disagree about what the
account contains.

What this module deliberately does not do: it is not on the retrieval path. ``build_account_session``
still loads FAISS and BM25 and fuses them with RRF, and a machine with no PostgreSQL at all still
answers questions. The database is an *additional* structured view of the same memory (§20, §22).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import psycopg
from psycopg.types.json import Jsonb

from . import schema
from .config import StoreTarget
from .rows import TABLE_COLUMNS, StoreSnapshot

#: Which columns hold JSONB, so COPY adapts exactly those and nothing else. Adapting every value
#: with an isinstance check would run tens of millions of them for a full bootstrap; this is a
#: precomputed lookup instead.
_JSON_INDEXES: Mapping[str, tuple[int, ...]] = {
    table: tuple(i for i, name in enumerate(columns) if name == "metadata")
    for table, columns in TABLE_COLUMNS.items()
}

#: Stage names handed to :meth:`PostgresMemoryStore._stage`. They exist so a crash test can fail at a
#: named point *between* statements and prove what the transaction left behind (§30), which is the
#: only way to test rollback honestly — a test that raises before `BEGIN` proves nothing.
STAGE_DELETED = "deleted"
STAGE_CONVERSATIONS = "conversations-written"
STAGE_EVENTS = "events-written"
STAGE_CHUNKS = "chunks-written"
STAGE_CHUNK_EVENTS = "chunk-events-written"
STAGE_MEMBERSHIPS = "memberships-written"
STAGE_PEOPLE = "people-written"
STAGE_AGGREGATES = "aggregates-recomputed"
STAGE_STATE = "state-written"

#: Every data table, in an order that truncates cleanly. `people` is last because two tables
#: reference it. `memory_store_state` is deliberately absent: a rebuild resets the *data*, and the
#: generation counter keeps rising so a generation number is never reused for different contents.
_TRUNCATE_ORDER = (
    "conversations",
    "memory_events",
    "memory_chunks",
    "chunk_events",
    "conversation_people",
    "people",
    "source_inventory",
)

_DATA_TABLES = (
    "conversations",
    "people",
    "conversation_people",
    "memory_events",
    "memory_chunks",
    "chunk_events",
    "source_inventory",
)


class StoreError(RuntimeError):
    """The store could not be used as asked."""


class StoreNotEmpty(StoreError):
    """A bootstrap was asked for on a database that already holds a store."""


@dataclass(frozen=True)
class WriteReport:
    """What one write did. Counts, timings and a generation — never a row's content."""

    mode: str
    rows_written: Mapping[str, int] = field(default_factory=dict)
    replaced_conversations: int = 0
    dropped_conversations: int = 0
    people_added: int = 0
    people_removed: int = 0
    store_generation: int = 0
    timings_ms: Mapping[str, int] = field(default_factory=dict)
    total_ms: int = 0
    database_size_bytes: int | None = None
    table_counts: Mapping[str, int] = field(default_factory=dict)

    def lines(self) -> list[str]:
        """Operator-facing lines: aggregates and timings only, safe for a terminal or a log."""
        out = [
            f"store          : {self.mode} wrote "
            + ", ".join(f"{name}={count}" for name, count in sorted(self.rows_written.items())),
            f"store          : {self.replaced_conversations} conversation(s) rewritten"
            + (f", {self.dropped_conversations} dropped" if self.dropped_conversations else "")
            + f"; generation {self.store_generation}",
        ]
        if self.people_added or self.people_removed:
            out.append(
                f"store          : {self.people_added} person row(s) added, "
                f"{self.people_removed} removed"
            )
        if self.database_size_bytes:
            out.append(f"store          : database size {_human_bytes(self.database_size_bytes)}")
        if self.timings_ms:
            out.append(
                "store          : "
                + ", ".join(f"{k}={v}ms" for k, v in sorted(self.timings_ms.items()))
            )
        out.append(f"store          : total {self.total_ms}ms")
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "rows_written": dict(self.rows_written),
            "replaced_conversations": self.replaced_conversations,
            "dropped_conversations": self.dropped_conversations,
            "people_added": self.people_added,
            "people_removed": self.people_removed,
            "store_generation": self.store_generation,
            "timings_ms": dict(self.timings_ms),
            "total_ms": self.total_ms,
            "database_size_bytes": self.database_size_bytes,
            "table_counts": dict(self.table_counts),
        }


@dataclass(frozen=True)
class StoreCounts:
    """How many rows of each kind the database holds right now."""

    conversations: int = 0
    people: int = 0
    conversation_people: int = 0
    events: int = 0
    chunks: int = 0
    chunk_events: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "conversations": self.conversations,
            "people": self.people,
            "conversation_people": self.conversation_people,
            "events": self.events,
            "chunks": self.chunks,
            "chunk_events": self.chunk_events,
        }


def connect(target: StoreTarget) -> psycopg.Connection:
    """Open a connection to the target. Raises ``psycopg.OperationalError`` when unreachable."""
    return psycopg.connect(target.url, autocommit=False)


class PostgresMemoryStore:
    """The canonical structured memory store, and its only writer."""

    def __init__(self, target: StoreTarget, connection: psycopg.Connection | None = None) -> None:
        self.target = target
        self._conn = connection if connection is not None else connect(target)
        self._owns_connection = connection is None

    # -- lifecycle ------------------------------------------------------------------------------

    @classmethod
    def open(cls, target: StoreTarget, *, migrate: bool = True) -> "PostgresMemoryStore":
        store = cls(target)
        if migrate:
            store.migrate()
        return store

    def close(self) -> None:
        if self._owns_connection and not self._conn.closed:
            self._conn.close()

    def __enter__(self) -> "PostgresMemoryStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _stage(self, name: str) -> None:
        """No-op hook, overridden by tests to fail at a named point inside a transaction (§30).

        A method rather than a flag: production code carries no switch that could be set by accident,
        and the crash tests subclass this class rather than reaching into it.
        """

    # -- schema ---------------------------------------------------------------------------------

    def migrate(self) -> tuple[schema.Migration, ...]:
        """Bring the schema up to this build's version, and leave the connection clean.

        The commit is not tidiness. Reading the migration history runs inside a transaction, and a
        connection that migrates and then sits ``idle in transaction`` holds ``ACCESS SHARE`` locks
        on every table it just created — for as long as it lives. A second session that then tries
        to ``DROP`` or ``ALTER`` anything blocks silently on the first one having finished nothing.
        That is invisible in a single-connection test and fatal in a script that cleans up after
        itself.
        """
        try:
            ran = schema.apply_migrations(self._conn)
            schema.require_current(self._conn)
        except Exception:
            self._conn.rollback()
            raise
        self._conn.commit()
        return ran

    def schema_status(self) -> dict[str, object]:
        return schema.migration_status(self._conn)

    # -- reads ----------------------------------------------------------------------------------

    def table_counts(self) -> StoreCounts:
        with self._conn.cursor() as cur:
            values = [
                int(cur.execute(f"SELECT count(*) FROM {table}").fetchone()[0])  # noqa: S608
                for table in _DATA_TABLES
            ]
        self._conn.commit()
        return StoreCounts(
            conversations=values[0],
            people=values[1],
            conversation_people=values[2],
            events=values[3],
            chunks=values[4],
            chunk_events=values[5],
        )

    def stored_inventory(self) -> tuple[tuple[Any, ...], ...]:
        """The export files the stored generation was built from, in :data:`rows.INVENTORY_COLUMNS`.

        Plain tuples rather than the caller's own entry type: this module is the data layer and does
        not import the index cache, and a tuple in a documented column order is a narrower contract
        than an object whose fields may grow.
        """
        columns = ", ".join(TABLE_COLUMNS["source_inventory"])
        with self._conn.cursor() as cur:
            rows = cur.execute(
                f"SELECT {columns} FROM source_inventory ORDER BY relative_path"  # noqa: S608
            ).fetchall()
        self._conn.commit()
        return tuple(tuple(row) for row in rows)

    def store_state(self) -> dict[str, Any] | None:
        """The ``memory_store_state`` row for this store, or ``None`` if it was never written."""
        keys = (
            "store_schema_version",
            "projection_version",
            "chunking_fingerprint",
            "source_fingerprint",
            "store_generation",
            "conversation_count",
            "event_count",
            "chunk_count",
            "people_count",
            "first_event_at",
            "last_event_at",
            "last_sync_at",
        )
        with self._conn.cursor() as cur:
            row = cur.execute(
                f"SELECT {', '.join(keys)} FROM memory_store_state WHERE store_name = %s",
                (self.target.store_name,),
            ).fetchone()
        self._conn.commit()
        return dict(zip(keys, row)) if row is not None else None

    def database_size_bytes(self) -> int | None:
        with self._conn.cursor() as cur:
            value = cur.execute("SELECT pg_database_size(current_database())").fetchone()[0]
        self._conn.commit()
        return int(value) if value is not None else None

    # -- writes ---------------------------------------------------------------------------------

    def bootstrap(
        self,
        snapshot: StoreSnapshot,
        *,
        state: Mapping[str, Any],
        inventory: Sequence[tuple[Any, ...]] | None = None,
        rebuild: bool = False,
    ) -> WriteReport:
        """Write a whole account into the store.

        Without ``rebuild`` this refuses a non-empty store rather than merging into it: a bootstrap
        is a statement about the *whole* account, and running one over an existing store would leave
        whatever the snapshot did not mention. ``rebuild`` says the caller means to replace it, and it
        truncates only this database's tables — the schema version is checked first (§37).
        """
        existing = self.table_counts()
        occupied = bool(existing.events or existing.conversations)
        if occupied and not rebuild:
            raise StoreNotEmpty(
                f"store {self.target.store_name!r} already holds {existing.events} event(s); "
                "pass rebuild=True (CLI: --rebuild-postgres) to replace it"
            )
        return self._write(
            snapshot,
            state=state,
            replacing=None,
            truncate=occupied,
            mode="rebuild" if occupied else "bootstrap",
            inventory=inventory,
        )

    def sync(
        self,
        snapshot: StoreSnapshot,
        *,
        updating: Sequence[str],
        state: Mapping[str, Any],
        inventory: Sequence[tuple[Any, ...]] | None = None,
    ) -> WriteReport:
        """Replace exactly the conversations named in ``updating`` with their current render.

        The whole update is one transaction (§18): either every affected conversation and the store
        state move to the new generation together, or nothing does. A conversation in ``updating``
        that produced no rows in ``snapshot`` is *dropped* — its export is gone, so its events are —
        and that is a deletion rather than a silent omission.
        """
        naming = tuple(dict.fromkeys(str(cid) for cid in updating))
        if not naming:
            raise StoreError("sync needs at least one conversation to update")
        return self._write(
            snapshot,
            state=state,
            replacing=naming,
            truncate=False,
            mode="incremental",
            inventory=inventory,
        )

    # -- the write itself -----------------------------------------------------------------------

    def _write(
        self,
        snapshot: StoreSnapshot,
        *,
        state: Mapping[str, Any],
        replacing: Sequence[str] | None,
        truncate: bool,
        mode: str,
        inventory: Sequence[tuple[Any, ...]] | None = None,
    ) -> WriteReport:
        started = time.perf_counter()
        timings: dict[str, int] = {}
        rows_written: dict[str, int] = {}
        present = {row[0] for row in snapshot.conversations}
        dropped = 0
        people_before = 0

        with self._conn.transaction():
            if truncate:
                mark = time.perf_counter()
                with self._conn.cursor() as cur:
                    cur.execute("TRUNCATE " + ", ".join(_TRUNCATE_ORDER) + " CASCADE")
                timings["truncate"] = _ms(mark)

            if replacing is not None:
                mark = time.perf_counter()
                with self._conn.cursor() as cur:
                    # The cascade removes this conversation's events, chunks, evidence links and
                    # memberships. A conversation that no longer renders is simply not re-inserted.
                    cur.execute(
                        "DELETE FROM conversations WHERE conversation_id = ANY(%s)",
                        (list(replacing),),
                    )
                dropped = len([cid for cid in replacing if cid not in present])
                timings["delete"] = _ms(mark)
            self._stage(STAGE_DELETED)

            mark = time.perf_counter()
            self._copy("conversations", snapshot.conversations)
            timings["conversations"] = _ms(mark)
            rows_written["conversations"] = len(snapshot.conversations)
            self._stage(STAGE_CONVERSATIONS)

            # People come before the events that reference them: `memory_events.speaker_person_id`
            # is a real foreign key, and a COPY that named a person row that does not exist yet is
            # exactly the kind of half-truth a transaction would otherwise have to roll back.
            mark = time.perf_counter()
            person_ids = [row[0] for row in snapshot.people]
            people_before = _scalar(
                self._conn, "SELECT count(*) FROM people WHERE person_id = ANY(%s)", (person_ids,)
            )
            self._upsert_people_identity(snapshot)
            people_after = _scalar(
                self._conn, "SELECT count(*) FROM people WHERE person_id = ANY(%s)", (person_ids,)
            )
            timings["people"] = _ms(mark)
            rows_written["people"] = people_after
            self._stage(STAGE_PEOPLE)

            mark = time.perf_counter()
            self._copy("memory_events", snapshot.events)
            timings["events"] = _ms(mark)
            rows_written["events"] = len(snapshot.events)
            self._stage(STAGE_EVENTS)

            mark = time.perf_counter()
            self._copy("memory_chunks", snapshot.chunks)
            timings["chunks"] = _ms(mark)
            rows_written["chunks"] = len(snapshot.chunks)
            self._stage(STAGE_CHUNKS)

            mark = time.perf_counter()
            self._copy("chunk_events", snapshot.chunk_events)
            timings["chunk_events"] = _ms(mark)
            rows_written["chunk_events"] = len(snapshot.chunk_events)
            self._stage(STAGE_CHUNK_EVENTS)

            mark = time.perf_counter()
            self._copy("conversation_people", snapshot.conversation_people)
            timings["memberships"] = _ms(mark)
            rows_written["conversation_people"] = len(snapshot.conversation_people)
            self._stage(STAGE_MEMBERSHIPS)

            mark = time.perf_counter()
            self._recompute_people(snapshot)
            removed = self._drop_orphan_people()
            timings["aggregates"] = _ms(mark)
            self._stage(STAGE_AGGREGATES)

            if inventory is not None:
                # The whole inventory, replaced rather than merged: it describes the tree *now*, and
                # the delta the next sync computes must be measured against that, not against a
                # union that would make a deleted export look like a file that never existed.
                mark = time.perf_counter()
                self._replace_inventory(inventory)
                timings["inventory"] = _ms(mark)
                rows_written["source_inventory"] = len(inventory)

            mark = time.perf_counter()
            generation = self._write_state(snapshot, state)
            timings["state"] = _ms(mark)
            self._stage(STAGE_STATE)

        counts = self.table_counts()
        size = self.database_size_bytes()
        return WriteReport(
            mode=mode,
            rows_written=rows_written,
            replaced_conversations=len(present) if replacing is None else len(replacing),
            dropped_conversations=dropped,
            people_added=max(0, people_after - people_before),
            people_removed=removed,
            store_generation=generation,
            timings_ms=timings,
            total_ms=int((time.perf_counter() - started) * 1000),
            database_size_bytes=size,
            table_counts=counts.as_dict(),
        )

    def _copy(self, table: str, rows: Sequence[tuple[Any, ...]]) -> None:
        """Bulk-write rows with ``COPY``.

        Never a row-at-a-time ``INSERT``: 1.6M events through a per-row path would make the
        bootstrap take hours for no correctness gained (§6, §14).
        """
        if not rows:
            return
        columns = ", ".join(TABLE_COLUMNS[table])
        json_indexes = _JSON_INDEXES[table]
        statement = f"COPY {table} ({columns}) FROM STDIN"
        with self._conn.cursor() as cur:
            with cur.copy(statement) as copy:
                if not json_indexes:
                    for row in rows:
                        copy.write_row(row)
                    return
                for row in rows:
                    adapted = list(row)
                    for index in json_indexes:
                        adapted[index] = Jsonb(adapted[index])
                    copy.write_row(tuple(adapted))

    def _upsert_people_identity(self, snapshot: StoreSnapshot) -> None:
        """Add any person this snapshot introduced; leave existing rows alone.

        A person's identity columns are a pure function of their ``person_id`` — it is a digest of
        the identity (``rows.person_id_for``) — so an existing row's identity can never need
        revising and ``DO NOTHING`` is exact rather than merely convenient. Every *derived* column is
        recomputed separately, from the memberships that actually exist.
        """
        people = snapshot.people
        if not people:
            return
        columns = TABLE_COLUMNS["people"]
        statement = (
            f"INSERT INTO people ({', '.join(columns)}) "
            f"VALUES ({', '.join(['%s'] * len(columns))}) "
            "ON CONFLICT (person_id) DO NOTHING"
        )
        payload = [
            tuple(Jsonb(value) if columns[i] == "metadata" else value for i, value in enumerate(row))
            for row in people
        ]
        with self._conn.cursor() as cur:
            cur.executemany(statement, payload)

    def _recompute_people(self, snapshot: StoreSnapshot) -> None:
        """Rebuild every derived ``people`` column from the memberships, for the people this write touched.

        This is why `people` holds no cached fact a sync could fail to revise. The name resolution in
        :data:`_RECOMPUTE_PEOPLE_SQL` — most frequent, then earliest sighting, then the name itself —
        is the same order ``rows._pick_name`` applies, and the parity test holds the two to it.
        """
        person_ids = [row[0] for row in snapshot.people]
        if not person_ids:
            return
        with self._conn.cursor() as cur:
            cur.execute(_RECOMPUTE_PEOPLE_SQL, {"ids": person_ids})

    def _replace_inventory(self, rows: Sequence[tuple[Any, ...]]) -> None:
        """Swap the stored source inventory for the one this write was given."""
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM source_inventory")
        self._copy("source_inventory", list(rows))

    def _drop_orphan_people(self) -> int:
        """Remove people with no membership left.

        A fresh bootstrap only ever creates a person who appears in an event, so a person left with
        no membership is a difference between the incremental store and a rebuilt one. Deleting them
        here is what keeps those two provably equal instead of approximately equal.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM people p WHERE NOT EXISTS ("
                "SELECT 1 FROM conversation_people cp WHERE cp.person_id = p.person_id)"
            )
            return cur.rowcount or 0

    def _write_state(self, snapshot: StoreSnapshot, state: Mapping[str, Any]) -> int:
        """Record which generation the store now holds. Always the last write in the transaction.

        The counts come from the tables rather than from the snapshot: an incremental write only
        touches a few conversations, and a state row describing those instead of the store would
        claim the database is far smaller than it is.
        """
        aggregates = self._aggregates()
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT store_generation FROM memory_store_state WHERE store_name = %s",
                (self.target.store_name,),
            )
            row = cur.fetchone()
            generation = (int(row[0]) if row is not None else 0) + 1

            cur.execute(
                """
                INSERT INTO memory_store_state (
                    store_name, store_schema_version, projection_version, chunking_fingerprint,
                    source_fingerprint, store_generation, conversation_count, event_count,
                    chunk_count, people_count, first_event_at, last_event_at, last_sync_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now(), now())
                ON CONFLICT (store_name) DO UPDATE SET
                    store_schema_version = EXCLUDED.store_schema_version,
                    projection_version   = EXCLUDED.projection_version,
                    chunking_fingerprint = EXCLUDED.chunking_fingerprint,
                    source_fingerprint   = EXCLUDED.source_fingerprint,
                    store_generation     = EXCLUDED.store_generation,
                    conversation_count   = EXCLUDED.conversation_count,
                    event_count          = EXCLUDED.event_count,
                    chunk_count          = EXCLUDED.chunk_count,
                    people_count         = EXCLUDED.people_count,
                    first_event_at       = EXCLUDED.first_event_at,
                    last_event_at        = EXCLUDED.last_event_at,
                    last_sync_at         = now(),
                    updated_at           = now()
                """,
                (
                    self.target.store_name,
                    int(state.get("store_schema_version") or schema.SCHEMA_VERSION),
                    int(state.get("projection_version") or 0),
                    str(state.get("chunking_fingerprint") or ""),
                    str(state.get("source_fingerprint") or ""),
                    generation,
                    aggregates["conversations"],
                    aggregates["events"],
                    aggregates["chunks"],
                    aggregates["people"],
                    aggregates["first_event_at"],
                    aggregates["last_event_at"],
                ),
            )
        return generation

    def _aggregates(self) -> dict[str, Any]:
        """Store-wide counts and time bounds, in one round trip."""
        with self._conn.cursor() as cur:
            row = cur.execute(
                "SELECT (SELECT count(*) FROM conversations),"
                "       (SELECT count(*) FROM memory_events),"
                "       (SELECT count(*) FROM memory_chunks),"
                "       (SELECT count(*) FROM people),"
                "       (SELECT min(event_time) FROM memory_events),"
                "       (SELECT max(event_time) FROM memory_events)"
            ).fetchone()
        return {
            "conversations": int(row[0]),
            "events": int(row[1]),
            "chunks": int(row[2]),
            "people": int(row[3]),
            "first_event_at": row[4],
            "last_event_at": row[5],
        }

    # -- acceptance helpers ---------------------------------------------------------------------

    def verify_counts(self, expected: Mapping[str, int]) -> dict[str, Any]:
        """Compare the database's row counts with a snapshot's. The §33 parity check."""
        actual = self.table_counts().as_dict()
        mismatches = {
            name: {"expected": int(value), "actual": actual.get(name)}
            for name, value in expected.items()
            if actual.get(name) != int(value)
        }
        return {"ok": not mismatches, "actual": actual, "mismatches": mismatches}

    def identity_scope(self) -> dict[str, Any]:
        """Aggregates about identity resolution, without naming a person or a conversation."""
        with self._conn.cursor() as cur:
            row = cur.execute(
                "SELECT (SELECT count(DISTINCT conversation_id) FROM memory_events),"
                "       (SELECT count(*) FROM conversations WHERE display_label IS NOT NULL),"
                "       (SELECT count(*) FROM conversations),"
                "       (SELECT count(*) FROM memory_events WHERE speaker_person_id IS NOT NULL),"
                "       (SELECT count(*) FROM memory_events)"
            ).fetchone()
            kinds = dict(
                cur.execute(
                    "SELECT person_kind, count(*) FROM people GROUP BY person_kind"
                ).fetchall()
            )
            multi = int(
                cur.execute(
                    "SELECT count(*) FROM (SELECT 1 FROM conversation_people "
                    "GROUP BY person_id HAVING count(*) > 1) t"
                ).fetchone()[0]
            )
            # Two people who render the same name somewhere must still be two rows. This is the
            # collapse §26 forbids, measured rather than asserted.
            shared = int(
                cur.execute(
                    "SELECT count(*) FROM ("
                    "  SELECT display_name FROM people WHERE display_name IS NOT NULL "
                    "  GROUP BY display_name HAVING count(*) > 1) t"
                ).fetchone()[0]
            )
        return {
            "conversations_with_events": int(row[0]),
            "conversations_labelled": int(row[1]),
            "conversations": int(row[2]),
            "events_with_person": int(row[3]),
            "events": int(row[4]),
            "people_by_kind": {str(k): int(v) for k, v in kinds.items()},
            "people_in_multiple_conversations": multi,
            "names_shared_by_multiple_people": shared,
        }

    def structured_query_smoke(self, *, sample: int = 100) -> dict[str, Any]:
        """The five structured queries of §31, as a timing/count report.

        Read-only, and it prints nothing it reads: every query returns a count and the elapsed
        milliseconds. The point is to prove the database can answer a conversation+time, person+time
        or evidence-order question at all — not to put it on the retrieval path, which this phase
        must not do.
        """
        with self._conn.cursor() as cur:
            conversations = cur.execute(
                "SELECT conversation_id, min(event_time), max(event_time) FROM memory_events "
                "GROUP BY conversation_id ORDER BY count(*) DESC LIMIT %s",
                (sample,),
            ).fetchall()
            people = cur.execute(
                "SELECT speaker_person_id, min(event_time), max(event_time) FROM memory_events "
                "WHERE speaker_person_id IS NOT NULL GROUP BY speaker_person_id "
                "ORDER BY count(*) DESC LIMIT %s",
                (sample,),
            ).fetchall()
            chunks = cur.execute(
                "SELECT chunk_id FROM chunk_events GROUP BY chunk_id "
                "ORDER BY count(*) DESC LIMIT %s",
                (sample,),
            ).fetchall()
        self._conn.commit()

        widest = conversations[0] if conversations else None
        busiest = people[0] if people else None
        results: dict[str, Any] = {}
        if widest is not None:
            results["A_conversation_events"] = self._timed(
                "SELECT count(*) FROM memory_events WHERE conversation_id = %s", (widest[0],)
            )
            results["B_time_range_events"] = self._timed(
                # Inclusive on both ends: a conversation whose whole history is one message has
                # min == max, and a half-open range would report that period as empty.
                "SELECT count(*) FROM memory_events WHERE event_time >= %s AND event_time <= %s",
                (widest[1], widest[2]),
            )
            results["D_conversation_and_time"] = self._timed(
                "SELECT count(*) FROM memory_events WHERE conversation_id = %s AND event_time >= %s",
                (widest[0], widest[1]),
            )
        if busiest is not None:
            results["C_person_events"] = self._timed(
                "SELECT count(*) FROM memory_events WHERE speaker_person_id = %s", (busiest[0],)
            )
        if chunks:
            results["E_chunk_evidence_in_order"] = self._timed(
                "SELECT count(*) FROM chunk_events ce "
                "JOIN memory_events e ON e.event_id = ce.event_id WHERE ce.chunk_id = %s",
                (chunks[0][0],),
            )
        results["sampled"] = {
            "conversations": len(conversations),
            "people": len(people),
            "chunks": len(chunks),
        }
        return results

    def _timed(self, sql: str, params: Sequence[Any]) -> dict[str, Any]:
        mark = time.perf_counter()
        with self._conn.cursor() as cur:
            value = int(cur.execute(sql, tuple(params)).fetchone()[0])
        self._conn.commit()
        return {"count": value, "ms": round((time.perf_counter() - mark) * 1000, 2)}

    def evidence_for_chunk(self, chunk_id: str) -> tuple[tuple[int, str], ...]:
        """``(ordinal, event_id)`` for one chunk, in order — the evidence a citation resolves to."""
        with self._conn.cursor() as cur:
            rows = cur.execute(
                "SELECT ordinal, event_id FROM chunk_events WHERE chunk_id = %s ORDER BY ordinal",
                (chunk_id,),
            ).fetchall()
        self._conn.commit()
        return tuple((int(ordinal), str(event_id)) for ordinal, event_id in rows)


def _scalar(conn: psycopg.Connection, sql: str, params: Sequence[Any]) -> int:
    with conn.cursor() as cur:
        return int(cur.execute(sql, tuple(params)).fetchone()[0])


def _ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TiB"


#: Recompute every derived `people` column from the memberships. Runs over `conversation_people`
#: only — a few thousand rows — which is what makes people maintenance cheap enough to be exact.
#: A person's events are partitioned by conversation, so aggregating memberships *is* aggregating
#: events: same numbers, a scan four orders of magnitude smaller.
_RECOMPUTE_PEOPLE_SQL = """
WITH agg AS (
    SELECT cp.person_id,
           count(*)              AS conversation_count,
           sum(cp.event_count)   AS event_count,
           min(cp.first_seen_at) AS first_seen_at,
           max(cp.last_seen_at)  AS last_seen_at,
           bool_or(cp.person_kind = 'group_member') AS is_group_member
    FROM conversation_people cp
    WHERE cp.person_id = ANY(%(ids)s)
    GROUP BY cp.person_id
), votes AS (
    SELECT cp.person_id, cp.display_name,
           count(*)              AS n,
           min(cp.first_seen_at) AS first_at
    FROM conversation_people cp
    WHERE cp.person_id = ANY(%(ids)s) AND cp.display_name IS NOT NULL
    GROUP BY cp.person_id, cp.display_name
), best AS (
    SELECT DISTINCT ON (person_id) person_id, display_name
    FROM votes
    ORDER BY person_id, n DESC, first_at ASC, display_name ASC
)
UPDATE people p
SET display_name       = best.display_name,
    first_seen_at      = agg.first_seen_at,
    last_seen_at       = agg.last_seen_at,
    event_count        = agg.event_count,
    conversation_count = agg.conversation_count,
    person_kind        = CASE WHEN agg.is_group_member THEN 'group_member' ELSE 'direct_peer' END
FROM agg
LEFT JOIN best ON best.person_id = agg.person_id
WHERE p.person_id = agg.person_id
"""
