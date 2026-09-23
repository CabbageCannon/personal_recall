"""Manage the canonical structured memory store (Phase 22A).

    python store_admin.py --account data/real/account_full_v3 --bootstrap-postgres
    python store_admin.py --account data/real/account_full_v3 --sync-postgres
    python store_admin.py --account data/real/account_full_v3 --status

Every command here is **explicit**. Nothing on the retrieval path calls any of them, so a machine
with no PostgreSQL still imports, indexes and answers questions exactly as before, and an operator
who never runs this file never needs a database (§22, §38). The reverse is also true and is the point
of the `--` on the destructive flag: ``--rebuild-postgres`` starts from an empty store, and the run
says out loud which database it is about to empty.

Output is aggregate counts, timings and fingerprints. Never a message, a name, a wxid or a talker —
the same rule the sync and index reports follow, for the same reason: this is a terminal, and the
corpus is somebody's private history (§36).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import dotenv  # noqa: E402
import psycopg  # noqa: E402

import index_cache  # noqa: E402
import memory_store  # noqa: E402
from memory.sessions import SessionConfig  # noqa: E402
from memory_store import sync as store_sync  # noqa: E402

#: The project's env file, read the way every other entry point reads it.
ENV_PATH = BASE_DIR.parent.parent / ".env"

#: Distinct exit codes, because "nothing was wrong, the store simply declined to advance" is not the
#: same outcome as "the database was unreachable" and a script that cannot tell them apart will
#: retry the wrong one.
EXIT_OK = 0
EXIT_ERROR = 2
EXIT_NOT_ADVANCED = 3


def _session_config() -> SessionConfig:
    """The product's own segmentation, imported from the module that declares it.

    Not a second copy of the constant: the store's chunking fingerprint is compared against the
    index's, and two constants that happen to agree today would make that comparison meaningless the
    day one of them moved.
    """
    from recall import DEFAULT_MAX_SESSION_CHARS

    return SessionConfig(max_chars=DEFAULT_MAX_SESSION_CHARS)


def _cache_parent(account_dir: Path, index_dir: Path | None) -> Path:
    return index_cache.account_cache_dir(account_dir, index_dir).parent


def _inventory(account_dir: Path, index_dir: Path | None):
    return index_cache.source_inventory_entries(
        account_dir, exclude=(_cache_parent(account_dir, index_dir),)
    )


def _state_fields(account_dir: Path, index_dir: Path | None) -> dict:
    return memory_store.state_fields(
        account_dir,
        session_config=_session_config(),
        exclude=(_cache_parent(account_dir, index_dir),),
    )


def _open(args: argparse.Namespace) -> memory_store.PostgresMemoryStore:
    target = memory_store.target_for(
        args.account, url=args.postgres_url, store_name=args.store_name
    )
    return memory_store.PostgresMemoryStore.open(target)


def _report(store: memory_store.PostgresMemoryStore) -> dict:
    """Everything `--status` knows: versions, counts and identity scope. No identity itself."""
    return {
        "target": store.target.describe(),
        "schema": store.schema_status(),
        "state": {
            key: value
            for key, value in (store.store_state() or {}).items()
            if "fingerprint" not in key
        },
        "counts": store.table_counts().as_dict(),
        "identity": store.identity_scope(),
        "database_size_bytes": store.database_size_bytes(),
    }


def _print_status(store: memory_store.PostgresMemoryStore) -> None:
    payload = _report(store)
    print(f"target   : {payload['target']}")
    print(
        f"schema   : version {payload['schema']['current_version']} "
        f"(code knows {payload['schema']['known_version']}, "
        f"pending {payload['schema']['pending'] or 'none'})"
    )
    if not payload["state"]:
        print("state    : EMPTY - no generation has been written to this store yet")
        return
    state = payload["state"]
    print(
        f"state    : store {store.target.store_name!r}, generation {state['store_generation']}, "
        f"last sync {state['last_sync_at']}"
    )
    print(
        f"state    : {state['conversation_count']} conversation(s), {state['event_count']} event(s), "
        f"{state['chunk_count']} chunk(s), {state['people_count']} person(s)"
    )
    print(
        f"events   : {state['first_event_at']} .. {state['last_event_at']} "
        "(a wall clock; the exporter's own epoch is on every row)"
    )
    counts = payload["counts"]
    print(
        f"rows     : conversations={counts['conversations']} events={counts['events']} "
        f"chunks={counts['chunks']} chunk_events={counts['chunk_events']} "
        f"people={counts['people']} memberships={counts['conversation_people']}"
    )
    identity = payload["identity"]
    print(
        f"identity : {identity['conversations_labelled']}/{identity['conversations']} conversation(s) "
        f"have a human-readable label; {identity['events_with_person']}/{identity['events']} "
        "event(s) have a trustworthy speaker"
    )
    print(
        f"identity : people by kind {identity['people_by_kind']}, "
        f"{identity['people_in_multiple_conversations']} in more than one conversation, "
        f"{identity['names_shared_by_multiple_people']} name(s) shared by several people "
        "(shared names stay separate rows)"
    )
    print(f"database : {payload['database_size_bytes']} bytes")


def main() -> int:
    dotenv.load_dotenv(ENV_PATH if ENV_PATH.exists() else None)

    parser = argparse.ArgumentParser(
        description="Manage the canonical structured memory store.",
        epilog="This is a separate command from `recall.py` on purpose: nothing here runs during a "
        "question, so retrieval never depends on a database.",
    )
    parser.add_argument(
        "--account",
        type=Path,
        required=True,
        help="the account export directory the store describes (also names the store by default)",
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
        help="optional: the WeChat Msg/Multi directory, so the report can say which shards exist "
        "versus which were exported (read-only, names only)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="where the retrieval index lives, if there is one. The store does not read it; it is "
        "excluded from the source fingerprint so a database and an index built over the same tree "
        "record the same generation",
    )
    parser.add_argument(
        "--postgres-url",
        default=None,
        help=f"the database to use (default: ${memory_store.DATABASE_URL_VAR}). Never printed with "
        "credentials; see .env.example",
    )
    parser.add_argument(
        "--store-name",
        default=None,
        help="the store's logical name (default: the account directory's name)",
    )

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--bootstrap-postgres",
        action="store_true",
        help="write the whole account into an empty store. Refuses a store that already holds a "
        "generation - use --rebuild-postgres to say you mean to replace it",
    )
    action.add_argument(
        "--rebuild-postgres",
        action="store_true",
        help="delete this store's data and write the whole account again. Only this database's "
        "tables are touched, and the target is printed before anything is dropped",
    )
    action.add_argument(
        "--sync-postgres",
        action="store_true",
        help="advance the store over the conversations that moved, in one transaction. Falls back to "
        "nothing: an unsafe change is refused and reported, never half-applied",
    )
    action.add_argument("--status", action="store_true", help="what this store holds, in aggregates")
    action.add_argument(
        "--smoke",
        action="store_true",
        help="run the structured query smoke (conversation+time, person+time, chunk->evidence) and "
        "report counts and milliseconds only",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable output instead of the report"
    )
    args = parser.parse_args()

    account_dir = args.account.resolve()
    if not account_dir.is_dir():
        print(f"error: --account {account_dir} is not a directory.", file=sys.stderr)
        return EXIT_ERROR

    try:
        store = _open(args)
    except memory_store.StoreNotConfigured as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except (psycopg.OperationalError, psycopg.Error) as exc:
        # Loud, and never reported as a successful sync (§38). The message carries the class name
        # rather than the full server error, which can quote the connection string back.
        print(f"error: could not open the memory store ({type(exc).__name__})", file=sys.stderr)
        return EXIT_ERROR

    try:
        if args.status:
            if args.json:
                print(json.dumps(_report(store), ensure_ascii=False, indent=2, default=str))
            else:
                _print_status(store)
            return EXIT_OK

        if args.smoke:
            result = store.structured_query_smoke()
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            else:
                for name, payload in sorted(result.items()):
                    if name == "sampled":
                        print(f"smoke    : sampled {payload}")
                    else:
                        print(f"smoke    : {name} -> {payload['count']} row(s) in {payload['ms']}ms")
            return EXIT_OK

        started = perf_counter()
        if args.sync_postgres:
            plan = store_sync.plan_store_sync(
                store, account_dir, shard_dir=args.shard_dir, index_dir=args.index_dir,
                session_config=_session_config(),
            )
            if not args.json:
                print(f"target   : {store.target.describe()}")
                for line in plan.lines():
                    print(f"  {line}")
            if plan.is_no_change:
                return EXIT_OK
            if not plan.is_incremental:
                if args.json:
                    print(json.dumps(plan.as_dict(), ensure_ascii=False, indent=2))
                else:
                    print(
                        "hint     : run --rebuild-postgres to rebuild the store from this tree "
                        "(a full bootstrap; the export tree is unchanged)",
                        file=sys.stderr,
                    )
                return EXIT_NOT_ADVANCED
            outcome = store_sync.run_store_sync(
                store, account_dir, plan=plan, shard_dir=args.shard_dir,
                session_config=_session_config(),
            )
            if args.json:
                print(
                    json.dumps(
                        {"plan": plan.as_dict(), "report": outcome.report.as_dict()},
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    )
                )
            else:
                for line in outcome.lines():
                    print(f"  {line}")
                print(f"  store          : wall clock {(perf_counter() - started):.1f}s")
            return EXIT_OK

        snapshot = memory_store.account_snapshot(
            account_dir, shard_dir=args.shard_dir, session_config=_session_config()
        )
        if not args.json:
            print(f"target   : {store.target.describe()}")
            print(
                f"source   : {snapshot.counts['conversations']} conversation(s), "
                f"{snapshot.counts['events']} event(s), {snapshot.counts['chunks']} chunk(s), "
                f"{snapshot.counts['people']} person(s)"
            )
            if args.rebuild_postgres:
                print("store    : --rebuild-postgres: this store's data will be replaced")
        report = store.bootstrap(
            snapshot.snapshot,
            state=_state_fields(account_dir, args.index_dir),
            inventory=memory_store.bootstrap.inventory_rows(_inventory(account_dir, args.index_dir)),
            rebuild=args.rebuild_postgres,
        )
        if args.json:
            print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, default=str))
        else:
            for line in report.lines():
                print(f"  {line}")
            print(f"  store          : wall clock {(perf_counter() - started):.1f}s")
        return EXIT_OK
    except memory_store.StoreNotEmpty as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except memory_store.SnapshotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
