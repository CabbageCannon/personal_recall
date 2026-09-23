"""The schema's version history, and the only code allowed to change it.

Migrations are numbered ``.sql`` files beside this module. They are tracked artifacts: an applied
migration is never edited, a change is a new file, and the database records which ones it has run in
``schema_migrations``. Nothing in the application issues a `CREATE TABLE`, so "what shape is this
database" has exactly one answer and it is reproducible from the repository alone (§11).

The runner is deliberately small. Alembic would add a second dependency, a second file format and a
revision graph for a schema with one migration and a linear future; a number, a name and a
``SELECT`` answers the same questions here with nothing to keep in sync (§6 — one clear path).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

#: The schema version this build of the code knows how to read and write. Bump it in the same commit
#: as the migration that changes it; a database that disagrees then fails loudly instead of being
#: queried with the wrong column list (§23 Test E).
#:
#: 1 — the canonical memory store (Phase 22A).
#: 2 — `memory_chunks.embedding` + its HNSW index (Phase 22B).
SCHEMA_VERSION = 2

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

#: ``001_initial_memory_store.sql`` -> version 1, name "initial_memory_store".
MIGRATION_PATTERN = re.compile(r"^(\d{3})_([a-z0-9_]+)\.sql$")

#: The runner's own bookkeeping table. Created by the runner rather than by a migration, because a
#: migration cannot record that it ran before the table that records it exists.
MIGRATIONS_TABLE = "schema_migrations"

CREATE_MIGRATIONS_TABLE = f"""
CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} (
    version    INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


class SchemaError(RuntimeError):
    """The database's schema is not one this code can use."""


class SchemaVersionError(SchemaError):
    """The database is at a different schema version than this build expects (§23 Test E)."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path

    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8")


def discover_migrations(directory: Path | None = None) -> tuple[Migration, ...]:
    """Every migration in the repository, ordered by version.

    Duplicate version numbers are an error rather than a silent last-one-wins: two migrations that
    claim the same slot would apply in file-name order, which is not an order anybody chose.
    """
    root = Path(directory or MIGRATIONS_DIR)
    found: dict[int, Migration] = {}
    for path in sorted(root.glob("*.sql")):
        match = MIGRATION_PATTERN.match(path.name)
        if match is None:
            continue
        version = int(match.group(1))
        if version in found:
            raise SchemaError(
                f"two migrations claim version {version}: "
                f"{found[version].path.name} and {path.name}"
            )
        found[version] = Migration(version=version, name=match.group(2), path=path)
    return tuple(found[key] for key in sorted(found))


def applied_versions(conn) -> dict[int, str]:
    """``version -> name`` for everything this database has already run.

    In a transaction of its own, so that reading the history works on a connection an earlier
    failure left aborted. That is not a hypothetical: the failure a user actually hits is a broken
    migration, and the next thing anybody does is ask what version the database is now at.
    """
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(CREATE_MIGRATIONS_TABLE)
            cur.execute(f"SELECT version, name FROM {MIGRATIONS_TABLE}")
            return {int(version): str(name) for version, name in cur.fetchall()}


def current_version(conn) -> int:
    """The database's schema version — ``0`` for a database with no tables at all."""
    versions = applied_versions(conn)
    return max(versions, default=0)


def pending_migrations(conn, directory: Path | None = None) -> tuple[Migration, ...]:
    """Migrations in the repository that this database has not run yet."""
    applied = applied_versions(conn)
    return tuple(m for m in discover_migrations(directory) if m.version not in applied)


def apply_migrations(
    conn,
    *,
    directory: Path | None = None,
    target_version: int | None = None,
) -> tuple[Migration, ...]:
    """Run every pending migration in order. Returns the ones that ran.

    Each migration is its own transaction: a migration that fails leaves the database at the last
    version that fully applied, never half of two. That matters more than batching, because the
    failure a user actually hits is a typo in a new migration, and the useful outcome is a database
    they can still start from rather than one that would need a rebuild.

    A database **newer** than this code is refused before anything runs. Without that check the
    runner would see the applied set as "not containing 001" — a newer database has a different
    history, not a longer one — and try to apply migrations that are already in force.
    """
    available = discover_migrations(directory)
    known = max((m.version for m in available), default=0)
    current = current_version(conn)
    if current > known:
        raise SchemaVersionError(
            f"this database is at schema version {current}, which is newer than the {known} this "
            "build knows; update the code rather than migrating it backwards"
        )

    applied = applied_versions(conn)
    ran: list[Migration] = []
    for migration in available:
        if migration.version in applied:
            continue
        if target_version is not None and migration.version > target_version:
            break
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(migration.sql())
                cur.execute(
                    f"INSERT INTO {MIGRATIONS_TABLE} (version, name) VALUES (%s, %s)",
                    (migration.version, migration.name),
                )
        ran.append(migration)
    return tuple(ran)


def require_current(conn, *, expected: int = SCHEMA_VERSION) -> int:
    """Return the version, or raise if the database is not the one this code was written for.

    Both directions are errors and both are named, because they have different fixes: an older
    database needs :func:`apply_migrations`, a newer one means the code is behind the schema and
    running it anyway would write rows a newer reader would misread.
    """
    version = current_version(conn)
    if version == expected:
        return version
    known = max((m.version for m in discover_migrations()), default=0)
    if version < expected:
        raise SchemaVersionError(
            f"this database is at schema version {version} and this build needs {expected}; "
            f"run the migration step first ({known} migration(s) exist in the repository)"
        )
    raise SchemaVersionError(
        f"this database is at schema version {version}, which is newer than the {expected} this "
        "build knows; update the code rather than writing to it"
    )


def migration_status(conn) -> dict[str, object]:
    """A small, loggable summary: versions and counts, never any stored data."""
    applied = applied_versions(conn)
    available = discover_migrations()
    conn.commit()
    return {
        "current_version": max(applied, default=0),
        "known_version": SCHEMA_VERSION,
        "available": [m.version for m in available],
        "applied": sorted(applied),
        "pending": [m.version for m in available if m.version not in applied],
    }


def describe_plan(migrations: Sequence[Migration]) -> str:
    """``001_initial_memory_store, 002_…`` — what ``apply_migrations`` is about to do."""
    return ", ".join(f"{m.version:03d}_{m.name}" for m in migrations) or "(nothing)"
