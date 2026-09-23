"""The canonical memory store against a real PostgreSQL (§23–§31).

Every test here runs against an actual server, in a **schema of its own** that is created before the
test and dropped after it. Two reasons for that shape rather than mocks or a shared database:

* A mock proves the SQL this module *meant* to send. The questions this phase has to answer — does
  the migration apply from empty, does the cascade actually clean up, does the wall clock survive,
  does an incremental update really equal a fresh bootstrap — are all about what PostgreSQL does
  with the statements, so a fake server would answer a different question.
* A schema is a complete store with a `search_path` of its own: it is isolated, it is dropped
  without touching anything else in the database, and it makes the "these two stores must be
  row-for-row identical" tests cheap enough to run on every case.

``pytest -m postgres`` runs these; ``pytest -m "not postgres"`` runs the rest on a machine with no
server at all. The URL comes from ``PERSONAL_RECALL_TEST_DATABASE_URL`` or
``PERSONAL_RECALL_DATABASE_URL``, and if neither resolves or the server is unreachable the whole
module *skips* — a suite that fails because a container is down reports nothing about the code.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import psycopg
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import memory_store  # noqa: E402
from memory_store import (  # noqa: E402
    PostgresMemoryStore,
    StoreNotEmpty,
    StoreTarget,
    schema as schema_module,
)
from memory.processor import DOCUMENT_PROJECTION_VERSION  # noqa: E402
from memory.sessions import SessionConfig  # noqa: E402
import index_cache  # noqa: E402
from memory_store import rows as rows_module  # noqa: E402
from memory_store import store as store_module  # noqa: E402
from memory_store import sync as store_sync  # noqa: E402

pytestmark = pytest.mark.postgres

#: Where a test database is named. Deliberately a *test* variable first: pointing the suite at the
#: real store should take a deliberate act, not an inherited environment.
TEST_URL_VAR = "PERSONAL_RECALL_TEST_DATABASE_URL"

SCRATCH_ROOT = BASE_DIR / "tests" / "_scratch_postgres"

DAY = int(datetime(2026, 9, 20, 10, 0).timestamp())
HOUR = 3600

#: Synthetic identities, shaped like the real ones. Nothing here is a real wxid or chatroom.
CONV_A = "wxid_synthetic_otter"
CONV_B = "wxid_synthetic_heron"
GROUP = "47110022@chatroom"
MEMBER_A = "wxid_synthetic_member_a"
MEMBER_B = "wxid_synthetic_member_b"
PEER_NAME = "苍鹭"
MEMBER_NAME = "小明"


# =============================================================================================
# Harness
# =============================================================================================


#: The project's own env file, the same one `recall.ENV_PATH` names.
ENV_PATH = BASE_DIR.parent.parent / ".env"


def _configured_url() -> str | None:
    """The test database URL: the environment first, then the project's ``.env``.

    ``dotenv_values`` rather than ``load_dotenv``: reading the file for these two variables must not
    put an API key into ``os.environ`` for the whole suite, which would quietly change what other
    tests are testing.
    """
    names = (TEST_URL_VAR, memory_store.DATABASE_URL_VAR)
    for name in names:
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value
    if not ENV_PATH.exists():
        return None
    try:
        import dotenv

        values = dotenv.dotenv_values(ENV_PATH)
    except Exception:  # noqa: BLE001 - a configuration file we cannot read is simply not configured
        return None
    for name in names:
        value = str(values.get(name) or "").strip()
        if value:
            return value
    return None


@pytest.fixture(scope="session")
def pg_url() -> str:
    url = _configured_url()
    if not url:
        pytest.skip(f"no PostgreSQL URL configured ({TEST_URL_VAR} or {memory_store.DATABASE_URL_VAR})")
    try:
        psycopg.connect(url, connect_timeout=5).close()
    except psycopg.OperationalError as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"PostgreSQL is not reachable ({type(exc).__name__})")
    return url


def _create_schema(pg_url: str) -> str:
    name = f"pr_test_{uuid4().hex[:12]}"
    conn = psycopg.connect(pg_url)
    try:
        conn.execute(f'CREATE SCHEMA "{name}"')
        conn.commit()
    finally:
        conn.close()
    return name


def _drop_schema(pg_url: str, name: str) -> None:
    conn = psycopg.connect(pg_url)
    try:
        conn.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')
        conn.commit()
    finally:
        conn.close()


@contextmanager
def open_store(pg_url: str, store_name: str = "test") -> Iterator[PostgresMemoryStore]:
    """A migrated store in a private schema, dropped when the block ends.

    The schema is supplied as a libpq startup parameter rather than by executing ``SET
    search_path``: a ``SET`` inside a transaction is rolled back with it, so a crash test would
    silently start writing into ``public`` the moment it proved the rollback worked.
    """
    schema = _create_schema(pg_url)
    connection = psycopg.connect(pg_url, options=f"-c search_path={schema}")
    store = PostgresMemoryStore(StoreTarget(url=pg_url, store_name=store_name), connection=connection)
    store.migrate()
    try:
        yield store
    finally:
        connection.close()
        _drop_schema(pg_url, schema)


@pytest.fixture()
def store(pg_url: str) -> Iterator[PostgresMemoryStore]:
    with open_store(pg_url) as opened:
        yield opened


@pytest.fixture()
def scratch() -> Path:
    directory = SCRATCH_ROOT / uuid4().hex[:8]
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)
        try:
            SCRATCH_ROOT.rmdir()
        except OSError:
            pass


# =============================================================================================
# A synthetic account, and the mutations §28 needs
# =============================================================================================


def message(
    local_id: int,
    when: int,
    text: str,
    *,
    sender: str = CONV_A,
    is_send: int = 1,
    display: str | None = None,
) -> dict:
    entry = {
        "localId": local_id,
        "serverId": f"server-{local_id}",
        "localType": 1,
        "createTime": when,
        "isSend": is_send,
        "senderUsername": sender,
        "parsedContent": text,
    }
    if display is not None:
        entry["senderDisplay"] = display
    return entry


def base_tree() -> dict[str, dict[str, list[dict]]]:
    """Two direct conversations and one group, spread over two shards.

    ``CONV_A`` is one message in ``MSG0``; ``CONV_B`` is a single message *from the peer*, so the
    direct-peer identity rule has something to resolve. The group's only member carries **no**
    display, which is what makes the retroactive-relabel case below a real change rather than a
    no-op.
    """
    return {
        "MSG0": {
            CONV_A: [message(1, DAY, "甲的第一条")],
            CONV_B: [message(10, DAY + 60, "乙的回复", sender=CONV_B, is_send=0, display=PEER_NAME)],
        },
        "MSG1": {
            GROUP: [message(20, DAY + 2 * HOUR, "群里的第一句", sender=MEMBER_A, is_send=0)],
        },
    }


def write_tree(root: Path, tree: dict[str, dict[str, list[dict]]]) -> Path:
    for shard, conversations in tree.items():
        directory = root / shard
        directory.mkdir(parents=True, exist_ok=True)
        for talker, messages in conversations.items():
            (directory / f"{talker}_messages.json").write_text(
                json.dumps(messages, ensure_ascii=False), encoding="utf-8"
            )
    return root


def read_messages(account: Path, shard: str, talker: str) -> list[dict]:
    path = account / shard / f"{talker}_messages.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def write_messages(account: Path, shard: str, talker: str, messages: list[dict]) -> None:
    directory = account / shard
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{talker}_messages.json").write_text(
        json.dumps(messages, ensure_ascii=False), encoding="utf-8"
    )


@pytest.fixture()
def account(scratch: Path) -> Path:
    return write_tree(scratch / "account", base_tree())


def current_inventory(account: Path):
    """The export tree's inventory exactly as the store's own planner computes it."""
    return index_cache.source_inventory_entries(
        account, exclude=(index_cache.account_cache_dir(account, None).parent,)
    )


def store_state_fields(account: Path) -> dict:
    return memory_store.state_fields(
        account, exclude=(index_cache.account_cache_dir(account, None).parent,)
    )


def bootstrap(store: PostgresMemoryStore, account: Path, *, rebuild: bool = False):
    result = memory_store.account_snapshot(account)
    report = store.bootstrap(
        result.snapshot,
        state=store_state_fields(account),
        inventory=memory_store.bootstrap.inventory_rows(current_inventory(account)),
        rebuild=rebuild,
    )
    return result, report


def sync(store: PostgresMemoryStore, account: Path, conversations: list[str]):
    result = memory_store.conversation_snapshot(account, conversations)
    report = store.sync(
        result.snapshot,
        updating=conversations,
        state=store_state_fields(account),
        inventory=memory_store.bootstrap.inventory_rows(current_inventory(account)),
    )
    return result, report


# =============================================================================================
# Row-level comparison: what "the same store" means
# =============================================================================================

DUMP_TABLES = (
    "conversations",
    "people",
    "conversation_people",
    "memory_events",
    "memory_chunks",
    "chunk_events",
    # The generation's provenance is part of the store: two generations that describe the same
    # memory but came from different trees are not the same store (§18).
    "source_inventory",
)

#: Columns a store is *expected* to differ on between two writes of the same content — when it was
#: written, not what it holds. §29 names these as the only allowed differences.
OPERATIONAL_COLUMNS = frozenset({"updated_at"})

#: ``memory_store_state`` fields that describe the content. ``store_generation``, ``last_sync_at``
#: and ``updated_at`` are operational and excluded.
STATE_DATA_FIELDS = (
    "store_schema_version",
    "projection_version",
    "chunking_fingerprint",
    "source_fingerprint",
    "conversation_count",
    "event_count",
    "chunk_count",
    "people_count",
    "first_event_at",
    "last_event_at",
)


def _normalize(value: Any) -> Any:
    """A comparable form for a column value: JSONB in a stable key order, everything else as-is."""
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    if isinstance(value, list):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return value


def dump(store: PostgresMemoryStore) -> dict[str, list[tuple]]:
    """Every data row of a store, normalized and ordered. The parity oracle."""
    out: dict[str, list[tuple]] = {}
    with store._conn.cursor() as cur:
        for table in DUMP_TABLES:
            columns = [c for c in rows_module.TABLE_COLUMNS[table] if c not in OPERATIONAL_COLUMNS]
            rows = cur.execute(f"SELECT {', '.join(columns)} FROM {table}").fetchall()  # noqa: S608
            out[table] = sorted((tuple(_normalize(v) for v in row) for row in rows), key=repr)
    store._conn.commit()
    return out


def dump_signature(store: PostgresMemoryStore) -> str:
    """The whole store as one comparable string.

    ``default=str`` covers the timestamp columns: a naive wall clock renders unambiguously and no
    locale can reorder it, so two stores holding the same times produce the same characters.
    """
    return json.dumps(
        {table: rows for table, rows in dump(store).items()},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def state_data(store: PostgresMemoryStore) -> dict[str, Any]:
    state = store.store_state() or {}
    return {key: state.get(key) for key in STATE_DATA_FIELDS}


def find_event(store: PostgresMemoryStore, text: str) -> tuple | None:
    """The event row whose ``text`` is exactly ``text``. Column order is ``rows.EVENT_COLUMNS``."""
    with store._conn.cursor() as cur:
        rows = cur.execute(
            f"SELECT {', '.join(rows_module.EVENT_COLUMNS)} FROM memory_events WHERE text = %s",
            (text,),
        ).fetchall()
    store._conn.commit()
    assert len(rows) <= 1, f"{len(rows)} events carry the same text"
    return rows[0] if rows else None


def event_field(row: tuple, name: str) -> Any:
    return row[rows_module.EVENT_COLUMNS.index(name)]


def count_where(store: PostgresMemoryStore, sql: str, params: tuple = ()) -> int:
    with store._conn.cursor() as cur:
        return int(cur.execute(sql, params).fetchone()[0])


# =============================================================================================
# §23 — schema
# =============================================================================================


def test_a_migrations_create_the_schema_from_empty(pg_url: str) -> None:
    """An empty database becomes a usable store by running the migration, and nothing else."""
    with open_store(pg_url) as store:
        assert schema_module.current_version(store._conn) == schema_module.SCHEMA_VERSION
        assert store.table_counts().as_dict() == {
            "conversations": 0,
            "people": 0,
            "conversation_people": 0,
            "events": 0,
            "chunks": 0,
            "chunk_events": 0,
        }


def test_b_the_expected_tables_constraints_and_indexes_exist(store: PostgresMemoryStore) -> None:
    """Every table, primary key and secondary index the migration promises, checked against catalog."""
    with store._conn.cursor() as cur:
        tables = {
            str(row[0])
            for row in cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = current_schema() AND table_type = 'BASE TABLE'"
            ).fetchall()
        }
        assert set(rows_module.TABLE_COLUMNS) | {
            "memory_store_state",
            schema_module.MIGRATIONS_TABLE,
        } <= tables

        indexes = {
            str(row[0])
            for row in cur.execute(
                "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema()"
            ).fetchall()
        }
        for expected in (
            "memory_events_conversation_time_idx",
            "memory_events_person_time_idx",
            "memory_events_time_idx",
            "memory_chunks_conversation_time_idx",
            "memory_chunks_start_time_idx",
            "conversation_people_person_idx",
            "chunk_events_ordinal_idx",
            "chunk_events_event_idx",
        ):
            assert expected in indexes, expected

        # Every foreign key needs an index on its referencing column, and PostgreSQL does not make
        # one. Without it a cascading delete sequentially scans the whole referencing table once per
        # deleted row — measured on the real account at six minutes for eleven conversations, because
        # `chunk_events` had no index that could answer "which links cite this event".
        unindexed = cur.execute(
            """
            SELECT c.conname, t.relname AS child, a.attname AS column
            FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY (c.conkey)
            WHERE c.contype = 'f' AND n.nspname = current_schema()
              AND NOT EXISTS (
                SELECT 1 FROM pg_index i
                WHERE i.indrelid = t.oid AND a.attnum = i.indkey[0]
              )
            """
        ).fetchall()
        assert unindexed == [], f"foreign keys with no leading-column index: {unindexed}"

        # The writer's column lists must name real columns, and the tables must hold nothing the
        # writer neither writes nor declares as operational. A column added to the migration and
        # forgotten in `rows.py` fails here rather than being silently left NULL forever.
        for table, columns in rows_module.TABLE_COLUMNS.items():
            actual = {
                str(row[0])
                for row in cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = current_schema() AND table_name = %s",
                    (table,),
                ).fetchall()
            }
            assert set(columns) <= actual, f"{table}: missing {sorted(set(columns) - actual)}"
            undeclared = actual - set(columns)
            assert undeclared <= OPERATIONAL_COLUMNS, f"{table}: undeclared {sorted(undeclared)}"
    store._conn.commit()


def test_b_there_is_no_vector_column_and_no_vector_extension(store: PostgresMemoryStore) -> None:
    """§3: this phase does not build any part of Phase 22B in advance."""
    with store._conn.cursor() as cur:
        columns = [
            str(row[0])
            for row in cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema()"
            ).fetchall()
        ]
    assert not any("embedding" in name or "vector" in name for name in columns)
    assert not any("vector" in name for name in columns)
    store._conn.commit()


def test_c_migrating_leaves_no_open_transaction(pg_url: str) -> None:
    """A connection that migrates must not sit ``idle in transaction`` holding locks on its tables.

    Found the hard way: a script that migrated, read its data and then dropped its schema hung
    forever, because the migrating connection still held ``ACCESS SHARE`` on every table it had
    created. Nothing in a one-connection test can notice that; a second session can.
    """
    schema = _create_schema(pg_url)
    connection = psycopg.connect(pg_url, options=f"-c search_path={schema}")
    try:
        store = PostgresMemoryStore(
            StoreTarget(url=pg_url, store_name="tx"), connection=connection
        )
        store.migrate()
        probe = psycopg.connect(pg_url)
        try:
            state = probe.execute(
                "SELECT state FROM pg_stat_activity WHERE pid = %s",
                (connection.info.backend_pid,),
            ).fetchone()[0]
            probe.commit()
        finally:
            probe.close()
        assert state == "idle", f"the migrating connection is {state}"

        # And the proof that matters: another session can take the schema away.
        dropper = psycopg.connect(pg_url)
        try:
            dropper.execute(f'DROP SCHEMA "{schema}" CASCADE')
            dropper.commit()
        finally:
            dropper.close()
        schema = ""  # already gone
    finally:
        connection.close()
        if schema:
            _drop_schema(pg_url, schema)


def test_c_migrating_twice_applies_nothing_the_second_time(store: PostgresMemoryStore) -> None:
    """Re-running the migration step is a no-op, and the version is read, not assumed."""
    assert schema_module.pending_migrations(store._conn) == ()
    ran = schema_module.apply_migrations(store._conn)
    assert ran == ()
    assert schema_module.current_version(store._conn) == schema_module.SCHEMA_VERSION


def test_c_an_empty_database_reports_version_zero_and_the_pending_migration(pg_url: str) -> None:
    """Before any migration the version is 0 and exactly one migration is pending."""
    schema = _create_schema(pg_url)
    try:
        conn = psycopg.connect(pg_url, options=f"-c search_path={schema}")
        try:
            assert schema_module.current_version(conn) == 0
            pending = schema_module.pending_migrations(conn)
            assert [m.version for m in pending] == [1]
        finally:
            conn.close()
    finally:
        _drop_schema(pg_url, schema)


def test_d_deleting_a_conversation_cascades_to_every_derived_row(store: PostgresMemoryStore, account: Path) -> None:
    """The foreign keys are real: one delete clears events, chunks, links and memberships."""
    bootstrap(store, account)
    before = store.table_counts()
    assert before.events > 0 and before.chunks > 0 and before.conversation_people > 0

    # Counted before the delete, because "how many rows should have gone" is the question.
    orphans_before = count_where(
        store, "SELECT count(*) FROM memory_events WHERE conversation_id = %s", (CONV_B,)
    )
    assert orphans_before > 0

    with store._conn.cursor() as cur:
        cur.execute("DELETE FROM conversations WHERE conversation_id = %s", (CONV_B,))
    store._conn.commit()
    after = store.table_counts()
    assert after.conversations == before.conversations - 1
    assert after.events == before.events - orphans_before, "the events went with the conversation"
    assert after.chunks < before.chunks, "and so did its chunks and their evidence links"

    # Nothing may cite a deleted conversation, at any level.
    for table, column in (
        ("memory_events", "conversation_id"),
        ("memory_chunks", "conversation_id"),
        ("conversation_people", "conversation_id"),
    ):
        assert (
            count_where(store, f"SELECT count(*) FROM {table} WHERE {column} = %s", (CONV_B,)) == 0
        ), table
    assert count_where(
        store,
        "SELECT count(*) FROM chunk_events ce WHERE NOT EXISTS "
        "(SELECT 1 FROM memory_chunks c WHERE c.chunk_id = ce.chunk_id)",
    ) == 0


def test_e_a_schema_mismatch_fails_loudly(pg_url: str) -> None:
    """A database from a different schema version is refused rather than queried."""
    schema = _create_schema(pg_url)
    connection = psycopg.connect(pg_url, options=f"-c search_path={schema}")
    store = PostgresMemoryStore(StoreTarget(url=pg_url, store_name="test"), connection=connection)
    try:
        store.migrate()
        with connection.cursor() as cur:
            cur.execute(
                f"UPDATE {schema_module.MIGRATIONS_TABLE} SET version = %s WHERE version = %s",
                (999, schema_module.SCHEMA_VERSION),
            )
        connection.commit()

        with pytest.raises(schema_module.SchemaVersionError):
            schema_module.require_current(connection)
        with pytest.raises(schema_module.SchemaVersionError):
            store.migrate()
    finally:
        connection.close()
        _drop_schema(pg_url, schema)


def test_e_a_migration_that_cannot_apply_leaves_the_version_unmoved(store: PostgresMemoryStore, scratch: Path) -> None:
    """A broken migration is its own transaction: the database stays at the last version that ran."""
    broken = scratch / "migrations"
    broken.mkdir()
    source = schema_module.MIGRATIONS_DIR / "001_initial_memory_store.sql"
    (broken / "001_initial_memory_store.sql").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (broken / "002_deliberately_broken.sql").write_text(
        "CREATE TABLE impossible (a integer, a text);", encoding="utf-8"
    )
    before = schema_module.current_version(store._conn)
    with pytest.raises(psycopg.errors.Error):
        schema_module.apply_migrations(store._conn, directory=broken)
    assert schema_module.current_version(store._conn) == before


# =============================================================================================
# §24/§25 — round trips
# =============================================================================================


@pytest.fixture()
def rich_account(scratch: Path) -> Path:
    """One conversation per speaker case §24 names, plus unicode, an emoji and a long body."""
    long_body = "很长的消息。" * 400
    tree = {
        "MSG0": {
            CONV_B: [
                message(1, DAY, "对方说的话", sender=CONV_B, is_send=0, display=PEER_NAME),
                message(2, DAY + 60, "我的话", sender=CONV_A, is_send=1),
            ],
        },
        "MSG1": {
            GROUP: [
                message(20, DAY + 2 * HOUR, "有名成员说的话", sender=MEMBER_A, is_send=0, display=MEMBER_NAME),
                message(21, DAY + 2 * HOUR + 60, "无名成员说的话", sender=MEMBER_B, is_send=0),
                message(22, DAY + 2 * HOUR + 120, "没有身份的人说的话", sender="", is_send=0),
                message(23, DAY + 3 * HOUR, "今天天气真好 🌤️☕", sender=MEMBER_A, is_send=0, display=MEMBER_NAME),
                message(24, DAY + 4 * HOUR, long_body, sender=MEMBER_A, is_send=0, display=MEMBER_NAME),
            ],
        },
    }
    return write_tree(scratch / "rich", tree)


def test_event_round_trip_preserves_the_three_speaker_fields(store: PostgresMemoryStore, rich_account: Path) -> None:
    """§24: the four speaker facts survive a write and a read, unmerged."""
    bootstrap(store, rich_account)

    named = find_event(store, "有名成员说的话")
    assert named is not None
    assert event_field(named, "speaker_role") == "other"
    assert event_field(named, "speaker_id") == MEMBER_A
    assert event_field(named, "speaker_display") == MEMBER_NAME
    assert event_field(named, "sender_name") == MEMBER_NAME
    assert event_field(named, "speaker_person_id") is not None

    unnamed = find_event(store, "无名成员说的话")
    assert unnamed is not None
    assert event_field(unnamed, "speaker_id") == MEMBER_B
    assert event_field(unnamed, "speaker_display") == ""
    assert event_field(unnamed, "sender_name") == "成员B", "a member with no name keeps a pseudonym"
    assert event_field(unnamed, "speaker_person_id") is not None

    self_row = find_event(store, "我的话")
    assert self_row is not None
    assert event_field(self_row, "speaker_role") == "self"
    assert event_field(self_row, "sender_name") == "我"
    assert event_field(self_row, "speaker_person_id") is None


def test_event_round_trip_for_a_message_with_no_identity(store: PostgresMemoryStore, rich_account: Path) -> None:
    """A group message whose export stated no sender is labelled 对方 and names no person."""
    bootstrap(store, rich_account)
    row = find_event(store, "没有身份的人说的话")
    assert row is not None
    assert event_field(row, "speaker_id") == ""
    assert event_field(row, "sender_name") == "对方"
    assert event_field(row, "speaker_person_id") is None


def test_event_round_trip_for_unicode_and_a_long_body(store: PostgresMemoryStore, rich_account: Path) -> None:
    bootstrap(store, rich_account)
    emoji = find_event(store, "今天天气真好 🌤️☕")
    assert emoji is not None, "the emoji body survived the write"
    long_row = find_event(store, "很长的消息。" * 400)
    assert long_row is not None
    assert len(event_field(long_row, "text")) == len("很长的消息。" * 400)


def test_event_round_trip_keeps_metadata_and_the_source_epoch(store: PostgresMemoryStore, rich_account: Path) -> None:
    bootstrap(store, rich_account)
    row = find_event(store, "有名成员说的话")
    assert row is not None
    metadata = event_field(row, "metadata")
    assert metadata["source_type"] == "weflow"
    assert metadata["senderUsername"] == MEMBER_A
    assert event_field(row, "source_epoch") == DAY + 2 * HOUR
    assert event_field(row, "source_server_id") == "server-20"
    assert event_field(row, "source_position") is not None


def test_no_raw_identity_reaches_a_rendered_column(store: PostgresMemoryStore, rich_account: Path) -> None:
    """The privacy rule, expressed as a query rather than as a promise."""
    bootstrap(store, rich_account)
    leaked = count_where(
        store,
        "SELECT count(*) FROM memory_events "
        "WHERE sender_name LIKE '%%wxid%%' OR sender_name LIKE '%%@chatroom%%'",
    )
    assert leaked == 0
    labelled = count_where(
        store, "SELECT count(*) FROM conversations WHERE display_label LIKE '%%wxid%%'"
    )
    assert labelled == 0


def test_chunk_round_trip_preserves_text_times_and_evidence_order(store: PostgresMemoryStore, rich_account: Path) -> None:
    """§25: a chunk reads back with its identity, its bounds, and its events in order."""
    bootstrap(store, rich_account)
    with store._conn.cursor() as cur:
        rows = cur.execute(
            "SELECT chunk_id, conversation_id, start_time, end_time, n_events, n_chars, text, "
            "text_hash, projection_version, metadata, n_events "
            "FROM memory_chunks ORDER BY chunk_id"
        ).fetchall()
    store._conn.commit()
    assert rows
    for chunk_id, conversation_id, start, end, n_events, n_chars, text, text_hash, version, metadata, _ in rows:
        assert start <= end
        assert n_chars == len(text)
        assert text_hash == rows_module.content_hash(text)
        assert version == DOCUMENT_PROJECTION_VERSION, "read back as the projection that wrote it"
        assert isinstance(metadata, dict)
        evidence = store.evidence_for_chunk(chunk_id)
        assert len(evidence) == n_events, chunk_id
        assert [ordinal for ordinal, _ in evidence] == list(range(n_events))
        ordinals = count_where(
            store, "SELECT count(*) FROM chunk_events WHERE chunk_id = %s", (chunk_id,)
        )
        assert ordinals == n_events


def test_chunk_evidence_resolves_to_events_that_exist_and_are_ordered(store: PostgresMemoryStore, rich_account: Path) -> None:
    """The point of `chunk_events`: a citation resolves to a run of real messages in order."""
    bootstrap(store, rich_account)
    with store._conn.cursor() as cur:
        chunk_id = str(
            cur.execute(
                "SELECT chunk_id FROM memory_chunks ORDER BY n_events DESC LIMIT 1"
            ).fetchone()[0]
        )
    store._conn.commit()
    evidence = store.evidence_for_chunk(chunk_id)
    assert len(evidence) >= 2
    with store._conn.cursor() as cur:
        times = [
            cur.execute(
                "SELECT event_time FROM memory_events WHERE event_id = %s", (event_id,)
            ).fetchone()[0]
            for _, event_id in evidence
        ]
    store._conn.commit()
    assert times == sorted(times), "the ordinals follow the conversation's own chronology"


# =============================================================================================
# §26 — people identity, in the database
# =============================================================================================


def test_two_speakers_sharing_a_display_name_are_two_rows(store: PostgresMemoryStore, scratch: Path) -> None:
    """The collapse §26 forbids, measured on the stored rows rather than on the projection."""
    account = write_tree(
        scratch / "shared",
        {
            "MSG0": {
                GROUP: [
                    message(1, DAY, "甲", sender=MEMBER_A, is_send=0, display="小王"),
                    message(2, DAY + 60, "乙", sender=MEMBER_B, is_send=0, display="小王"),
                ]
            }
        },
    )
    bootstrap(store, account)
    assert count_where(store, "SELECT count(*) FROM people") == 2
    assert count_where(store, "SELECT count(*) FROM people WHERE display_name = '小王'") == 2
    # And the evidence can still tell them apart, which is the reason the qualifier exists.
    with store._conn.cursor() as cur:
        labels = [
            str(row[0])
            for row in cur.execute(
                "SELECT display_label FROM conversation_people ORDER BY display_label"
            ).fetchall()
        ]
    store._conn.commit()
    assert len(set(labels)) == 2
    assert all("wxid" not in label for label in labels)


def test_one_identity_in_two_conversations_is_one_person_with_two_memberships(store: PostgresMemoryStore, scratch: Path) -> None:
    account = write_tree(
        scratch / "shared-person",
        {
            "MSG0": {GROUP: [message(1, DAY, "第一次", sender=MEMBER_A, is_send=0, display="小明")]},
            "MSG1": {
                "88880000@chatroom": [
                    message(2, DAY + HOUR, "第二次", sender=MEMBER_A, is_send=0, display="小明")
                ]
            },
        },
    )
    bootstrap(store, account)
    assert count_where(store, "SELECT count(*) FROM people") == 1
    assert count_where(store, "SELECT count(*) FROM conversation_people") == 2
    assert count_where(
        store, "SELECT count(*) FROM conversation_people cp WHERE ("
        "SELECT count(*) FROM conversation_people x WHERE x.person_id = cp.person_id) = 2"
    ) == 2


def test_each_direct_conversation_gets_its_own_peer(store: PostgresMemoryStore, scratch: Path) -> None:
    """The forbidden merge: one 对方 person shared by every direct chat."""
    account = write_tree(
        scratch / "directs",
        {
            "MSG0": {
                CONV_A: [message(1, DAY, "甲", sender=CONV_A, is_send=0)],
                CONV_B: [message(2, DAY + 60, "乙", sender=CONV_B, is_send=0)],
            },
            "MSG1": {
                "wxid_synthetic_third": [
                    message(3, DAY + 120, "丙", sender="wxid_synthetic_third", is_send=0)
                ]
            },
        },
    )
    bootstrap(store, account)
    assert count_where(store, "SELECT count(*) FROM people") == 3
    assert count_where(store, "SELECT count(*) FROM people WHERE person_kind <> 'direct_peer'") == 0
    names = [
        str(row[0])
        for row in store._conn.execute(
            "SELECT coalesce(display_name, '') FROM people"
        ).fetchall()
    ]
    store._conn.commit()
    assert "对方" not in names


def test_a_direct_talker_that_is_also_a_group_member_is_one_person(store: PostgresMemoryStore, scratch: Path) -> None:
    """Measured on the real account: 73 of 140 direct talkers appear verbatim as group members."""
    account = write_tree(
        scratch / "both",
        {
            "MSG0": {CONV_A: [message(1, DAY, "私聊", sender=CONV_A, is_send=0)]},
            "MSG1": {
                GROUP: [message(2, DAY + 60, "群里", sender=CONV_A, is_send=0, display="小水獭")]
            },
        },
    )
    bootstrap(store, account)
    assert count_where(store, "SELECT count(*) FROM people") == 1
    kind = str(
        store._conn.execute("SELECT person_kind FROM people").fetchone()[0]
    )
    store._conn.commit()
    assert kind == "group_member", "an explicit sender field is stronger evidence than a file name"


# =============================================================================================
# §27 — idempotency
# =============================================================================================


def test_bootstrapping_twice_leaves_the_store_unchanged(store: PostgresMemoryStore, account: Path) -> None:
    """Same corpus, same store: no duplicates, and the second write is refused unless asked for."""
    bootstrap(store, account)
    first_counts = store.table_counts().as_dict()
    first_rows = dump_signature(store)

    with pytest.raises(StoreNotEmpty):
        bootstrap(store, account)

    bootstrap(store, account, rebuild=True)
    assert store.table_counts().as_dict() == first_counts
    assert dump_signature(store) == first_rows


def test_syncing_the_same_delta_twice_changes_nothing_the_second_time(store: PostgresMemoryStore, account: Path) -> None:
    """An incremental update is idempotent: the second run reaches the same state, not a bigger one."""
    bootstrap(store, account)
    messages = read_messages(account, "MSG1", GROUP)
    messages.append(message(30, DAY + 35 * HOUR, "新增的一句", sender=MEMBER_A, is_send=0))
    write_messages(account, "MSG1", GROUP, messages)

    sync(store, account, [GROUP])
    once = store.table_counts().as_dict()
    once_rows = dump_signature(store)

    sync(store, account, [GROUP])
    assert store.table_counts().as_dict() == once
    assert dump_signature(store) == once_rows


def test_syncing_an_unchanged_conversation_is_a_no_op(store: PostgresMemoryStore, account: Path) -> None:
    bootstrap(store, account)
    before = dump_signature(store)
    sync(store, account, [GROUP])
    assert dump_signature(store) == before


# =============================================================================================
# §28 — affected conversations, and what an incremental update must do to them
# =============================================================================================


def assert_incremental_equals_full(
    store: PostgresMemoryStore, pg_url: str, account: Path, conversations: list[str]
) -> None:
    """The heart of the phase: after a sync, the store must equal a fresh bootstrap of the same tree.

    Row for row, in every table, not merely in counts. ``store_generation`` and the operational
    timestamps are the only fields allowed to differ (§29).
    """
    sync(store, account, conversations)
    with open_store(pg_url, "reference") as reference:
        bootstrap(reference, account)
        assert dump_signature(store) == dump_signature(reference)
        assert state_data(store) == state_data(reference)


def test_case_new_message(store: PostgresMemoryStore, pg_url: str, account: Path) -> None:
    bootstrap(store, account)
    messages = read_messages(account, "MSG1", GROUP)
    messages.append(message(30, DAY + 35 * HOUR, "后来又说了一句", sender=MEMBER_A, is_send=0))
    write_messages(account, "MSG1", GROUP, messages)
    assert_incremental_equals_full(store, pg_url, account, [GROUP])


def test_case_tail_merge_into_the_same_chunk(store: PostgresMemoryStore, pg_url: str, account: Path) -> None:
    """A message inside ``max_gap`` joins the last chunk, so that chunk is *replaced*, not appended."""
    bootstrap(store, account)
    before = count_where(
        store, "SELECT count(*) FROM memory_chunks WHERE conversation_id = %s", (GROUP,)
    )
    messages = read_messages(account, "MSG1", GROUP)
    messages.append(message(31, DAY + 2 * HOUR + 300, "紧接着的一句", sender=MEMBER_A, is_send=0))
    write_messages(account, "MSG1", GROUP, messages)

    sync(store, account, [GROUP])
    after = count_where(
        store, "SELECT count(*) FROM memory_chunks WHERE conversation_id = %s", (GROUP,)
    )
    assert after == before, "the new message joined the existing session rather than opening one"
    with open_store(pg_url, "reference") as reference:
        bootstrap(reference, account)
        assert dump_signature(store) == dump_signature(reference)


def test_case_an_old_event_changes(store: PostgresMemoryStore, pg_url: str, account: Path) -> None:
    """Editing a message the delta never touched must still reach the store."""
    bootstrap(store, account)
    messages = read_messages(account, "MSG1", GROUP)
    messages[0]["parsedContent"] = "群里的第一句（已更正）"
    write_messages(account, "MSG1", GROUP, messages)
    assert_incremental_equals_full(store, pg_url, account, [GROUP])


def test_case_a_sender_label_changes_retroactively(store: PostgresMemoryStore, pg_url: str, account: Path) -> None:
    """A member gaining a display renames their *older* messages, which the delta never touches.

    This is the case that makes per-message appending wrong: the new message carries the name, and
    every message that member ever sent now renders with it.
    """
    bootstrap(store, account)
    before = find_event(store, "群里的第一句")
    assert before is not None
    assert event_field(before, "sender_name") == "成员A", "with no display, the member is a pseudonym"

    messages = read_messages(account, "MSG1", GROUP)
    messages.append(
        message(32, DAY + 3 * HOUR, "现在有名字了", sender=MEMBER_A, is_send=0, display=MEMBER_NAME)
    )
    write_messages(account, "MSG1", GROUP, messages)

    sync(store, account, [GROUP])
    after = find_event(store, "群里的第一句")
    assert after is not None
    assert event_field(after, "sender_name") == MEMBER_NAME, "the old message was relabelled"
    assert event_field(after, "speaker_id") == MEMBER_A, "and its identity did not move"

    with open_store(pg_url, "reference") as reference:
        bootstrap(reference, account)
        assert dump_signature(store) == dump_signature(reference)


def test_case_the_chunk_boundaries_move(store: PostgresMemoryStore, pg_url: str, account: Path) -> None:
    """Inserting in the middle renumbers the sessions after it — chunk ids are positional."""
    bootstrap(store, account)
    messages = read_messages(account, "MSG1", GROUP)
    messages.append(message(33, DAY + 30 * HOUR, "很晚的一句", sender=MEMBER_A, is_send=0))
    write_messages(account, "MSG1", GROUP, messages)
    sync(store, account, [GROUP])
    assert count_where(
        store, "SELECT count(*) FROM memory_chunks WHERE conversation_id = %s", (GROUP,)
    ) == 2

    messages = read_messages(account, "MSG1", GROUP)
    messages.append(message(34, DAY + 60 * HOUR, "更晚的一句", sender=MEMBER_A, is_send=0))
    write_messages(account, "MSG1", GROUP, messages)
    assert_incremental_equals_full(store, pg_url, account, [GROUP])
    assert count_where(
        store, "SELECT count(*) FROM memory_chunks WHERE conversation_id = %s", (GROUP,)
    ) == 3


def test_case_a_conversation_gains_a_shard(store: PostgresMemoryStore, pg_url: str, account: Path) -> None:
    """A conversation that was one shard and is now two must be re-rendered from both."""
    bootstrap(store, account)
    write_messages(account, "MSG0", GROUP, [message(40, DAY + 45 * HOUR, "另一个分片里的消息", sender=MEMBER_B, is_send=0)])
    assert_incremental_equals_full(store, pg_url, account, [GROUP])


def scoped_rows(store: PostgresMemoryStore, conversation_id: str) -> tuple:
    """Every row one conversation owns, in a form that can be compared before and after."""
    with store._conn.cursor() as cur:
        rows = (
            cur.execute(
                "SELECT conversation_id, conversation_type, display_label, event_count, chunk_count, "
                "first_event_at, last_event_at FROM conversations WHERE conversation_id = %s",
                (conversation_id,),
            ).fetchall(),
            cur.execute(
                "SELECT event_id, event_time, speaker_role, speaker_person_id, speaker_id, "
                "speaker_display, sender_name, text, message_type FROM memory_events "
                "WHERE conversation_id = %s ORDER BY event_id",
                (conversation_id,),
            ).fetchall(),
            cur.execute(
                "SELECT chunk_id, chunk_index, start_time, end_time, n_events, text_hash "
                "FROM memory_chunks WHERE conversation_id = %s ORDER BY chunk_id",
                (conversation_id,),
            ).fetchall(),
            cur.execute(
                "SELECT person_id, person_kind, display_name, display_label, event_count, "
                "first_seen_at, last_seen_at FROM conversation_people "
                "WHERE conversation_id = %s ORDER BY person_id",
                (conversation_id,),
            ).fetchall(),
            cur.execute(
                "SELECT ce.chunk_id, ce.event_id, ce.ordinal FROM chunk_events ce "
                "JOIN memory_chunks c ON c.chunk_id = ce.chunk_id "
                "WHERE c.conversation_id = %s ORDER BY ce.chunk_id, ce.ordinal",
                (conversation_id,),
            ).fetchall(),
        )
    store._conn.commit()
    return rows


def test_an_unaffected_conversation_is_left_exactly_as_it_was(store: PostgresMemoryStore, account: Path) -> None:
    """The other half of "only what moved": a sync must not rewrite what it was not asked about."""
    bootstrap(store, account)
    before_a = scoped_rows(store, CONV_A)
    before_b = scoped_rows(store, CONV_B)
    assert before_b[1], "the fixture conversation has events to compare"

    messages = read_messages(account, "MSG1", GROUP)
    messages.append(message(35, DAY + 40 * HOUR, "只有群变了", sender=MEMBER_A, is_send=0))
    write_messages(account, "MSG1", GROUP, messages)
    sync(store, account, [GROUP])

    assert scoped_rows(store, CONV_B) == before_b, "an untouched conversation's rows are the same rows"
    assert scoped_rows(store, CONV_A) == before_a


def test_a_conversation_whose_export_vanishes_is_dropped(store: PostgresMemoryStore, account: Path) -> None:
    """A conversation that no longer renders is deleted, not left behind as a stale row."""
    bootstrap(store, account)
    assert count_where(store, "SELECT count(*) FROM conversations WHERE conversation_id = %s", (CONV_B,)) == 1
    (account / "MSG0" / f"{CONV_B}_messages.json").unlink()
    sync(store, account, [CONV_B])
    assert count_where(store, "SELECT count(*) FROM conversations WHERE conversation_id = %s", (CONV_B,)) == 0
    assert count_where(store, "SELECT count(*) FROM memory_events WHERE conversation_id = %s", (CONV_B,)) == 0


def test_a_person_left_with_no_conversation_is_removed(store: PostgresMemoryStore, account: Path) -> None:
    """Orphan cleanup is what keeps an incremental store equal to a rebuilt one."""
    bootstrap(store, account)
    peer_id = str(
        store._conn.execute(
            "SELECT person_id FROM conversation_people WHERE conversation_id = %s", (CONV_B,)
        ).fetchone()[0]
    )
    store._conn.commit()
    (account / "MSG0" / f"{CONV_B}_messages.json").unlink()
    sync(store, account, [CONV_B])
    assert count_where(store, "SELECT count(*) FROM people WHERE person_id = %s", (peer_id,)) == 0


# =============================================================================================
# §29 — golden parity over the whole store
# =============================================================================================


def test_a_full_delta_sequence_ends_equal_to_a_fresh_bootstrap(store: PostgresMemoryStore, pg_url: str, account: Path) -> None:
    """Several incremental steps in a row, then compared with one clean build of the same tree."""
    bootstrap(store, account)

    steps: list[tuple[list[str], Any]] = []
    steps.append(
        (["MSG1", GROUP], lambda: write_messages(
            account, "MSG1", GROUP,
            read_messages(account, "MSG1", GROUP)
            + [message(50, DAY + 3 * HOUR, "第一步", sender=MEMBER_A, is_send=0)],))
    )
    for conversations, mutate in steps:
        mutate()
        sync(store, account, conversations)

    # A second round, this time touching two conversations at once.
    write_messages(
        account, "MSG0", CONV_A,
        read_messages(account, "MSG0", CONV_A) + [message(51, DAY + 4 * HOUR, "第二步")],
    )
    write_messages(
        account, "MSG1", GROUP,
        read_messages(account, "MSG1", GROUP)
        + [message(52, DAY + 5 * HOUR, "第三步", sender=MEMBER_B, is_send=0, display="小红")],
    )
    sync(store, account, [CONV_A, GROUP])

    with open_store(pg_url, "reference") as reference:
        bootstrap(reference, account)
        assert dump(store) == dump(reference)
        assert state_data(store) == state_data(reference)


def test_the_store_state_describes_the_account_it_holds(store: PostgresMemoryStore, account: Path) -> None:
    result, _ = bootstrap(store, account)
    state = store.store_state()
    assert state is not None
    assert state["store_schema_version"] == schema_module.SCHEMA_VERSION
    assert state["projection_version"] == DOCUMENT_PROJECTION_VERSION
    assert state["event_count"] == result.counts["events"]
    assert state["chunk_count"] == result.counts["chunks"]
    assert state["conversation_count"] == result.counts["conversations"]
    assert state["people_count"] == result.counts["people"]
    assert len(state["source_fingerprint"]) == 64
    assert len(state["chunking_fingerprint"]) == 64
    assert state["store_generation"] == 1


def test_the_generation_advances_once_per_write_and_never_repeats(store: PostgresMemoryStore, account: Path) -> None:
    """§18: the store can answer "which generation am I", and a generation is never reused."""
    bootstrap(store, account)
    assert store.store_state()["store_generation"] == 1
    bootstrap(store, account, rebuild=True)
    assert store.store_state()["store_generation"] == 2
    messages = read_messages(account, "MSG1", GROUP)
    messages.append(message(60, DAY + 40 * HOUR, "再一步", sender=MEMBER_A, is_send=0))
    write_messages(account, "MSG1", GROUP, messages)
    sync(store, account, [GROUP])
    assert store.store_state()["store_generation"] == 3


# =============================================================================================
# §30 — transactions and crash safety
# =============================================================================================


class CrashStore(PostgresMemoryStore):
    """A store that can be armed to fail at a named point inside the write transaction.

    Armed explicitly rather than configured at construction: the interesting failure is the *second*
    write, so the test has to get the store to a known good generation first, and a store that
    failed on its own bootstrap would never reach the state under test.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.armed_stage: str = ""

    def arm(self, stage: str) -> None:
        self.armed_stage = stage

    def disarm(self) -> None:
        self.armed_stage = ""

    def _stage(self, name: str) -> None:
        if self.armed_stage and name == self.armed_stage:
            raise RuntimeError(f"injected failure at {name}")


@contextmanager
def crashing_store(pg_url: str, store_name: str) -> Iterator[CrashStore]:
    schema = _create_schema(pg_url)
    connection = psycopg.connect(pg_url, options=f"-c search_path={schema}")
    store = CrashStore(StoreTarget(url=pg_url, store_name=store_name), connection=connection)
    store.migrate()
    try:
        yield store
    finally:
        connection.close()
        _drop_schema(pg_url, schema)


ALL_WRITE_STAGES = (
    store_module.STAGE_CONVERSATIONS,
    store_module.STAGE_PEOPLE,
    store_module.STAGE_EVENTS,
    store_module.STAGE_CHUNKS,
    store_module.STAGE_CHUNK_EVENTS,
    store_module.STAGE_MEMBERSHIPS,
    store_module.STAGE_AGGREGATES,
    store_module.STAGE_STATE,
)


@pytest.mark.parametrize("fail_at", ALL_WRITE_STAGES)
def test_a_failure_anywhere_in_a_bootstrap_leaves_an_empty_store(
    pg_url: str, account: Path, fail_at: str
) -> None:
    """Every stage of the write is inside the transaction: there is no partially-written store."""
    with crashing_store(pg_url, "crash") as store:
        store.arm(fail_at)
        with pytest.raises(RuntimeError, match="injected failure"):
            bootstrap(store, account)
        assert store.table_counts().as_dict() == {
            "conversations": 0,
            "people": 0,
            "conversation_people": 0,
            "events": 0,
            "chunks": 0,
            "chunk_events": 0,
        }, fail_at
        assert store.store_state() is None, fail_at


@pytest.mark.parametrize("fail_at", ALL_WRITE_STAGES)
def test_a_failure_inside_a_sync_leaves_the_old_generation(
    pg_url: str, account: Path, fail_at: str
) -> None:
    """The case a user actually hits: a sync dies midway and the store must still be the old one."""
    with crashing_store(pg_url, "sync-crash") as store:
        bootstrap(store, account)
        before_rows = dump_signature(store)
        generation = store.store_state()["store_generation"]

        messages = read_messages(account, "MSG1", GROUP)
        messages.append(message(70, DAY + 50 * HOUR, "会失败的增量", sender=MEMBER_A, is_send=0))
        write_messages(account, "MSG1", GROUP, messages)

        store.arm(fail_at)
        with pytest.raises(RuntimeError, match="injected failure"):
            sync(store, account, [GROUP])

        assert store.store_state()["store_generation"] == generation, f"no half-generation ({fail_at})"
        assert dump_signature(store) == before_rows, f"the previous generation is intact ({fail_at})"


def test_a_rolled_back_sync_can_be_retried_and_then_succeeds(pg_url: str, account: Path) -> None:
    """A crash is recoverable: the same delta run again reaches the state a clean build would."""
    with crashing_store(pg_url, "retry") as store:
        bootstrap(store, account)

        messages = read_messages(account, "MSG1", GROUP)
        messages.append(message(71, DAY + 51 * HOUR, "重试的增量", sender=MEMBER_A, is_send=0))
        write_messages(account, "MSG1", GROUP, messages)

        store.arm(store_module.STAGE_CHUNKS)
        with pytest.raises(RuntimeError, match="injected failure"):
            sync(store, account, [GROUP])

        store.disarm()
        sync(store, account, [GROUP])
        assert count_where(
            store, "SELECT count(*) FROM memory_events WHERE text = %s", ("重试的增量",)
        ) == 1

        with open_store(pg_url, "reference") as reference:
            bootstrap(reference, account)
            assert dump_signature(store) == dump_signature(reference), (
                "a sync that failed and was retried reaches the same store a clean build does"
            )


# =============================================================================================
# §10 — the timestamp contract
# =============================================================================================


@pytest.mark.parametrize("epoch", [1546300800, 1672531200, 1767225600, 1789301234])
def test_chat_times_round_trip_exactly_under_any_session_timezone(
    pg_url: str, scratch: Path, epoch: int
) -> None:
    """The column type was chosen by measurement, and this holds it to that choice.

    The contract's chat times are naive local wall clocks. ``TIMESTAMPTZ`` was rejected because the
    same stored value read back under ``Asia/Shanghai`` shifts by eight hours — this test reads the
    store back under three different session timezones and requires the wall clock to be identical
    in all of them.
    """
    from memory_store.rows import EVENT_COLUMNS

    account = write_tree(
        scratch / "clock",
        {"MSG0": {CONV_A: [message(1, epoch, "时间点", sender=CONV_A, is_send=1)]}},
    )
    result = memory_store.account_snapshot(account)
    expected = result.snapshot.events[0][EVENT_COLUMNS.index("event_time")]
    assert expected.tzinfo is None, "the contract's timestamps are naive"

    schema = _create_schema(pg_url)
    connection = psycopg.connect(pg_url, options=f"-c search_path={schema}")
    try:
        store = PostgresMemoryStore(StoreTarget(url=pg_url, store_name="clock"), connection=connection)
        store.migrate()
        store.bootstrap(result.snapshot, state=memory_store.state_fields(account))
        for zone in ("UTC", "Asia/Shanghai", "America/New_York"):
            connection.execute(f"SET TIME ZONE '{zone}'")
            with connection.cursor() as cur:
                read_back = cur.execute("SELECT event_time FROM memory_events").fetchone()[0]
            assert read_back == expected, zone
            assert read_back.tzinfo is None, zone
        connection.rollback()
    finally:
        connection.close()
        _drop_schema(pg_url, schema)


def test_the_raw_source_epoch_is_kept_alongside_the_wall_clock(store: PostgresMemoryStore, account: Path) -> None:
    """The wall clock can never be ambiguous because the exporter's own instant is stored too."""
    bootstrap(store, account)
    epochs = [
        int(row[0])
        for row in store._conn.execute(
            "SELECT source_epoch FROM memory_events ORDER BY source_epoch"
        ).fetchall()
    ]
    store._conn.commit()
    assert epochs == sorted(epochs)
    assert DAY in epochs


# =============================================================================================
# §31 — structured queryability
# =============================================================================================


def test_the_five_structured_queries_answer(store: PostgresMemoryStore, account: Path) -> None:
    """Conversation+time, person+time and chunk->events all work, and return counts not content."""
    bootstrap(store, account)
    smoke = store.structured_query_smoke()
    for key in (
        "A_conversation_events",
        "B_time_range_events",
        "C_person_events",
        "D_conversation_and_time",
        "E_chunk_evidence_in_order",
    ):
        assert key in smoke, key
        assert smoke[key]["count"] >= 1, key
        assert isinstance(smoke[key]["ms"], float)
    # The report names no conversation, person, chunk or message — only sample sizes.
    assert set(smoke["sampled"]) == {"conversations", "people", "chunks"}


def test_identity_scope_reports_aggregates_without_naming_anyone(store: PostgresMemoryStore, account: Path) -> None:
    bootstrap(store, account)
    scope = store.identity_scope()
    assert scope["conversations"] == 3
    assert scope["events_with_person"] >= 1
    assert set(scope["people_by_kind"]) <= {"group_member", "direct_peer"}
    assert isinstance(scope["names_shared_by_multiple_people"], int)


# =============================================================================================
# §16/§18 — planning a sync from the store's own record of its generation
# =============================================================================================
#
# The property these tests exist for: a store sync must not need the retrieval index to exist, and
# must not be confused by an index run that advanced past it. The Phase 21B checkpoint describes one
# generation and is advanced by whichever consumer runs, so the store keeps its own record — and
# these tests drive the planner with no index cache anywhere on disk.


def test_a_bootstrap_records_the_export_files_its_generation_came_from(store, account) -> None:
    """A store that cannot say which tree it was built from cannot compute its own delta (§18)."""
    bootstrap(store, account)
    inventory = store.stored_inventory()
    assert inventory, "the generation recorded the files it was built from"
    for row in inventory:
        relative_path, size, mtime_ns, kind, shard, conversation_id = row
        assert relative_path and "/" in relative_path
        assert size >= 0 and mtime_ns > 0
        assert kind in ("export", "listing", "manifest")
        assert shard and conversation_id


def test_planning_over_an_unchanged_tree_is_a_no_op(store, account) -> None:
    bootstrap(store, account)
    plan = store_sync.plan_store_sync(store, account)
    assert plan.is_no_change, plan.reason
    assert plan.affected == ()


def test_planning_over_a_moved_tree_names_exactly_the_conversations_that_moved(store, account) -> None:
    bootstrap(store, account)
    messages = read_messages(account, "MSG1", GROUP)
    messages.append(message(80, DAY + 60 * HOUR, "移动了", sender=MEMBER_A, is_send=0))
    write_messages(account, "MSG1", GROUP, messages)

    plan = store_sync.plan_store_sync(store, account)
    assert plan.is_incremental, plan.reason
    assert plan.affected == (GROUP,), "only the group moved"


def test_planning_without_an_index_cache_at_all_works(store, account) -> None:
    """The store is self-sufficient: no FAISS index, no Phase 21B checkpoint, no cache directory."""
    bootstrap(store, account)
    messages = read_messages(account, "MSG1", GROUP)
    messages.append(message(81, DAY + 61 * HOUR, "没有索引也要能同步", sender=MEMBER_A, is_send=0))
    write_messages(account, "MSG1", GROUP, messages)

    assert not index_cache.account_cache_dir(account, None).exists()
    plan = store_sync.plan_store_sync(store, account)
    assert plan.is_incremental, plan.reason
    store_sync.run_store_sync(store, account, plan=plan)
    assert count_where(
        store, "SELECT count(*) FROM memory_events WHERE text = %s", ("没有索引也要能同步",)
    ) == 1

    with open_store(pg_url_of(store), "reference") as reference:
        bootstrap(reference, account)
        assert dump_signature(store) == dump_signature(reference)


def pg_url_of(store: PostgresMemoryStore) -> str:
    """The URL a store was opened with. Read from the target, never from a global."""
    return store.target.url


def test_a_store_with_no_generation_refuses_to_plan(store, account) -> None:
    plan = store_sync.plan_store_sync(store, account)
    assert plan.outcome == store_sync.UNSAFE
    assert "bootstrap" in plan.reason


def test_planning_refuses_a_chunking_change_rather_than_segmenting_two_ways(store, account) -> None:
    """A chunking change leaves old conversations segmented one way and new ones another."""
    bootstrap(store, account)
    messages = read_messages(account, "MSG1", GROUP)
    messages.append(message(82, DAY + 62 * HOUR, "分块变了", sender=MEMBER_A, is_send=0))
    write_messages(account, "MSG1", GROUP, messages)

    plan = store_sync.plan_store_sync(
        store, account, session_config=SessionConfig(max_chars=1234)
    )
    assert plan.outcome == store_sync.UNSAFE
    assert plan.stale == ("session chunking",)
    assert not plan.affected


def test_planning_refuses_a_removed_export(store, account) -> None:
    """A file the store was built from that is now gone is history it can no longer account for."""
    bootstrap(store, account)
    (account / "MSG0" / f"{CONV_B}_messages.json").unlink()
    plan = store_sync.plan_store_sync(store, account)
    assert plan.outcome == store_sync.UNSAFE
    assert "gone" in plan.reason
    assert not plan.affected


def test_an_unsafe_sync_run_writes_nothing(store, account) -> None:
    bootstrap(store, account)
    before = dump_signature(store)
    (account / "MSG0" / f"{CONV_B}_messages.json").unlink()
    plan = store_sync.plan_store_sync(store, account)
    outcome = store_sync.run_store_sync(store, account, plan=plan)
    assert not outcome.ok
    assert dump_signature(store) == before


def test_a_planned_sync_reaches_the_same_store_as_a_rebuild(pg_url: str, account: Path) -> None:
    """The whole incremental path through its real entry points, compared row for row."""
    with open_store(pg_url, "planning") as store:
        bootstrap(store, account)
        messages = read_messages(account, "MSG1", GROUP)
        messages.append(message(83, DAY + 63 * HOUR, "计划同步", sender=MEMBER_A, is_send=0))
        write_messages(account, "MSG1", GROUP, messages)

        plan = store_sync.plan_store_sync(store, account)
        assert plan.is_incremental, plan.reason
        outcome = store_sync.run_store_sync(store, account, plan=plan)
        assert outcome.ok
        assert outcome.report.mode == "incremental"
        assert count_where(
            store, "SELECT count(*) FROM memory_events WHERE text = %s", ("计划同步",)
        ) == 1

        with open_store(pg_url, "reference") as reference:
            bootstrap(reference, account)
            assert dump(store) == dump(reference)
            assert state_data(store) == state_data(reference)

        # And the generation it just wrote is the base the next plan compares against.
        assert store_sync.plan_store_sync(store, account).is_no_change


def test_a_sync_that_ran_leaves_the_store_advanced_to_the_new_tree(store, account) -> None:
    """A sync that did not move the store's own fingerprint would redo the same work forever."""
    bootstrap(store, account)
    before = store.store_state()["source_fingerprint"]
    messages = read_messages(account, "MSG1", GROUP)
    messages.append(message(84, DAY + 64 * HOUR, "指纹要动", sender=MEMBER_A, is_send=0))
    write_messages(account, "MSG1", GROUP, messages)

    plan = store_sync.plan_store_sync(store, account)
    store_sync.run_store_sync(store, account, plan=plan)
    after = store.store_state()["source_fingerprint"]
    assert after != before
    assert after == store_state_fields(account)["source_fingerprint"]


# =============================================================================================
# §38 — configuration and failure behaviour
# =============================================================================================


def test_no_configured_url_raises_a_named_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(memory_store.DATABASE_URL_VAR, raising=False)
    monkeypatch.delenv(memory_store.STORE_NAME_VAR, raising=False)
    assert memory_store.database_url() is None
    with pytest.raises(memory_store.StoreNotConfigured):
        memory_store.target_for(Path("account"))


def test_an_unparsable_url_is_not_echoed_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A URL this project cannot parse must not be printed in full — it may carry a password."""
    target = memory_store.StoreTarget(url="not-a-url-with-a-secret", store_name="s")
    described = target.describe()
    assert "secret" not in described
    assert "s" in described


def test_a_password_never_reaches_the_loggable_description(store: PostgresMemoryStore) -> None:
    described = store.target.describe()
    assert "personal_recall_local" not in described
    assert ":" in described


def test_an_unreachable_server_raises_rather_than_reporting_success() -> None:
    """§38: an explicit store command fails loudly; nothing claims a sync that did not happen."""
    target = memory_store.StoreTarget(
        url="postgresql://nobody:nobody@127.0.0.1:1/nothing", store_name="unreachable"
    )
    with pytest.raises(psycopg.OperationalError):
        memory_store.connect(target)


def test_the_store_is_optional_for_everything_else() -> None:
    """Importing the product must not require a database, and neither must importing this module."""
    import importlib

    for name in ("memory_store", "memory_store.rows", "memory_store.bootstrap", "memory_store.store"):
        assert importlib.import_module(name) is not None
