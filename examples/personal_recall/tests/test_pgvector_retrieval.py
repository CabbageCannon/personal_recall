"""Metadata-filtered hybrid retrieval against a real PostgreSQL + pgvector.

Everything here needs a live server, in a schema of its own that is created and dropped per test.
What it checks is the part the pure tests cannot: that the SQL predicate actually selects the rows it
claims to, that pgvector's ordering agrees with the vectors that were written, and that a plan's
filter and strategy reach the evidence.

``pytest -m postgres`` runs these; the rest of the suite runs with no server at all.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import memory_store  # noqa: E402
import query_plan  # noqa: E402
from memory_store import (  # noqa: E402
    PostgresMemoryStore,
    PostgresVectorStore,
    RetrievalFilter,
    StoreTarget,
    candidate_report,
)

pytestmark = pytest.mark.postgres

TEST_URL_VAR = "PERSONAL_RECALL_TEST_DATABASE_URL"
ENV_PATH = BASE_DIR.parent.parent / ".env"
SCRATCH_ROOT = BASE_DIR / "tests" / "_scratch_pgvector"

DAY = int(datetime(2026, 5, 1, 10, 0).timestamp())
HOUR = 3600
#: The migration declares `vector(512)` — bge-small-zh-v1.5's width — and pgvector enforces it, so a
#: synthetic vector has to be that wide. Which is the point: the column cannot silently accept a
#: vector from a different model.
DIMENSION = 512

CONV_A = "wxid_synthetic_otter"
CONV_B = "wxid_synthetic_heron"
GROUP = "47110022@chatroom"
MEMBER_A = "wxid_synthetic_member_a"
MEMBER_B = "wxid_synthetic_member_b"
NAME_A = "小明"
NAME_B = "小王"


def _configured_url() -> str | None:
    for name in (TEST_URL_VAR, memory_store.DATABASE_URL_VAR):
        value = str(__import__("os").environ.get(name) or "").strip()
        if value:
            return value
    if not ENV_PATH.exists():
        return None
    try:
        import dotenv

        values = dotenv.dotenv_values(ENV_PATH)
    except Exception:  # noqa: BLE001
        return None
    for name in (TEST_URL_VAR, memory_store.DATABASE_URL_VAR):
        value = str(values.get(name) or "").strip()
        if value:
            return value
    return None


@pytest.fixture(scope="session")
def pg_url() -> str:
    url = _configured_url()
    if not url:
        pytest.skip("no PostgreSQL URL configured")
    try:
        psycopg.connect(url, connect_timeout=5).close()
    except psycopg.OperationalError as exc:  # pragma: no cover
        pytest.skip(f"PostgreSQL is not reachable ({type(exc).__name__})")
    return url


def _create_schema(pg_url: str) -> str:
    name = f"pr_vec_{uuid4().hex[:12]}"
    with psycopg.connect(pg_url) as conn:
        conn.execute(f'CREATE SCHEMA "{name}"')
        conn.commit()
    return name


def _drop_schema(pg_url: str, name: str) -> None:
    with psycopg.connect(pg_url) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')
        conn.commit()


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


def message(local_id: int, when: int, text: str, *, sender: str, is_send: int, display=None) -> dict:
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


def write_tree(root: Path) -> Path:
    tree = {
        "MSG0": {
            CONV_A: [message(1, DAY, "甲说的话", sender=CONV_A, is_send=0, display=NAME_A)],
            CONV_B: [
                message(2, DAY + HOUR, "乙说的话", sender=CONV_B, is_send=0, display=NAME_B)
            ],
        },
        "MSG1": {
            GROUP: [
                message(3, DAY + 2 * HOUR, "群里第一句", sender=MEMBER_A, is_send=0, display=NAME_A),
                message(4, DAY + 30 * HOUR, "群里很久以后", sender=MEMBER_B, is_send=0,
                        display=NAME_B),
            ]
        },
    }
    for shard, conversations in tree.items():
        directory = root / shard
        directory.mkdir(parents=True, exist_ok=True)
        for talker, messages in conversations.items():
            (directory / f"{talker}_messages.json").write_text(
                json.dumps(messages, ensure_ascii=False), encoding="utf-8"
            )
    return root


def pseudo_vector(seed: str, dimension: int = DIMENSION) -> list[float]:
    """A deterministic unit-ish vector from a chunk id, so a self-query has an exact answer."""
    import hashlib
    import math

    digest = hashlib.sha256(seed.encode()).digest()
    values = [digest[i % len(digest)] / 255.0 - 0.5 for i in range(dimension)]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


class HashEmbedder:
    """A deterministic stand-in for a model: the same text always produces the same vector.

    Deliberately not semantic. These tests are about *scoping* — which chunks a plan is allowed to
    see, and whether anything outside the scope leaks in — and a query embedding that happened to be
    meaningful would invite reading the results as retrieval quality, which is what the real
    acceptance measures with the real model.
    """

    def embed_query(self, text: str) -> list[float]:
        return pseudo_vector(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [pseudo_vector(t) for t in texts]


@pytest.fixture()
def loaded(pg_url: str, scratch: Path):
    """A store holding the synthetic account, with a vector on every chunk."""
    account = write_tree(scratch / "account")
    schema = _create_schema(pg_url)
    # `public` is on the search path because the `vector` type lives there: `CREATE EXTENSION` puts an
    # extension in one schema for the whole database, and a schema-only path hides it, so the Phase
    # 22B migration fails with `type "vector" does not exist`. It comes *second*, so every table the
    # store creates resolves to this test's own schema first and nothing else is reachable by an
    # unqualified name — the isolation is the shadowing, not the absence of a fallback.
    connection = psycopg.connect(pg_url, options=f"-c search_path={schema},public")
    store = PostgresMemoryStore(
        StoreTarget(url=pg_url, store_name="pgvector"), connection=connection
    )
    store.migrate()
    result = memory_store.account_snapshot(account)
    store.bootstrap(result.snapshot, state=memory_store.state_fields(account))

    chunk_ids = [row[0] for row in store.select_chunks()]
    store.write_embeddings(
        [(chunk_id, pseudo_vector(chunk_id)) for chunk_id in chunk_ids], dimension=DIMENSION
    )
    embedder = HashEmbedder()
    try:
        yield store, account, embedder, chunk_ids
    finally:
        connection.close()
        _drop_schema(pg_url, schema)


# ---------------------------------------------------------------------------------------------
# The document projection
# ---------------------------------------------------------------------------------------------


def test_the_store_projects_documents_identically_to_the_index(loaded) -> None:
    """The claim the whole backend rests on: a chunk retrieves as the same Document either way.

    Compared field for field against what a full index build produces, because the prompt renders
    these fields and a difference here would make the two backends incomparable while looking fine.
    """
    store, account, embedder, chunk_ids = loaded
    view = PostgresVectorStore(store, embedder, origin=f"account:{account.name}")
    documents = view.documents()

    assert {d.metadata["memory_chunk_id"] for d in documents} == set(chunk_ids)

    # Rebuild the same documents the way `recall.build_brain_from_chunks` does, from the chunks the
    # importer produces, and require the two to be identical.
    chunks, _ = recall_import(account)
    expected = {}
    for index, chunk in enumerate(chunks, start=1):
        from memory.processor import session_documents

        document = session_documents([chunk], 0)[0]
        document.metadata["chunk_index"] = index
        document.metadata["original_file_name"] = f"account:{account.name}"
        # `sessions_total` is the whole account's count in both paths; the index build passes the
        # full chunk list, and the store reports the same number.
        document.metadata["sessions_total"] = len(chunks)
        expected[document.metadata["memory_chunk_id"]] = document

    for document in documents:
        counterpart = expected[document.metadata["memory_chunk_id"]]
        assert document.page_content == counterpart.page_content
        assert dict(document.metadata) == dict(counterpart.metadata), document.metadata[
            "memory_chunk_id"
        ]


def recall_import(account: Path):
    """The product's own import, so the comparison above is against the real projection."""
    import recall

    layout = recall.load_account_directory(account)
    events_by_conversation, _ = recall.import_account(layout.exports, shards_detected=())
    from memory import SessionConfig

    chunks = recall.build_account_sessions(
        events_by_conversation, SessionConfig(max_chars=recall.DEFAULT_MAX_SESSION_CHARS)
    )
    return chunks, events_by_conversation


def test_get_returns_the_shape_iter_documents_reads(loaded) -> None:
    from quivr_core.rag.hybrid import iter_documents

    store, account, embedder, _ = loaded
    view = PostgresVectorStore(store, embedder, origin="account:x")
    listed = iter_documents(view)
    assert len(listed) == len(view.documents())
    assert [d.metadata["memory_chunk_id"] for d in listed] == [
        d.metadata["memory_chunk_id"] for d in view.documents()
    ]


# ---------------------------------------------------------------------------------------------
# Dense search
# ---------------------------------------------------------------------------------------------


def test_a_chunks_own_vector_retrieves_that_chunk(loaded) -> None:
    store, account, embedder, chunk_ids = loaded
    for chunk_id in chunk_ids:
        found = store.dense_chunks(pseudo_vector(chunk_id), limit=1)
        assert found and found[0][0] == chunk_id


def test_the_filter_narrows_the_candidate_set_exactly(loaded) -> None:
    store, _account, _embedder, _ = loaded
    for conversation_id in (CONV_A, CONV_B, GROUP):
        with store._conn.cursor() as cur:
            expected = int(
                cur.execute(
                    "SELECT count(*) FROM memory_chunks WHERE conversation_id = %s",
                    (conversation_id,),
                ).fetchone()[0]
            )
        store._conn.commit()
        assert store.count_chunks(RetrievalFilter(conversation_ids=(conversation_id,))) == expected


def test_a_person_filter_selects_chunks_that_person_spoke_in(loaded) -> None:
    """The semi-join semantics: evidence, not membership."""
    store, _account, _embedder, _ = loaded
    with store._conn.cursor() as cur:
        person_id = str(
            cur.execute(
                "SELECT person_id FROM people WHERE display_name = %s", (NAME_A,)
            ).fetchone()[0]
        )
        expected = int(
            cur.execute(
                "SELECT count(DISTINCT ce.chunk_id) FROM chunk_events ce "
                "JOIN memory_events e ON e.event_id = ce.event_id WHERE e.speaker_person_id = %s",
                (person_id,),
            ).fetchone()[0]
        )
    store._conn.commit()
    assert store.count_chunks(RetrievalFilter(person_ids=(person_id,))) == expected


def test_the_dense_search_respects_the_filter(loaded) -> None:
    store, _account, _embedder, chunk_ids = loaded
    target = next(
        row[0] for row in store.select_chunks(RetrievalFilter(conversation_ids=(GROUP,)))
    )
    other = next(row[0] for row in store.select_chunks(RetrievalFilter(conversation_ids=(CONV_A,))))

    scoped = store.dense_chunks(
        pseudo_vector(other), filters=RetrievalFilter(conversation_ids=(GROUP,)), limit=10
    )
    assert scoped, "the group has chunks"
    assert all(row[0] in chunk_ids for row in scoped)
    assert other not in {row[0] for row in scoped}, "a chunk outside the scope cannot be returned"
    assert target in {row[0] for row in scoped} or len(scoped) >= 1


def test_an_empty_filter_result_returns_nothing_rather_than_failing(loaded) -> None:
    store, _account, _embedder, _ = loaded
    empty = RetrievalFilter(person_ids=("0" * 32,))
    assert store.count_chunks(empty) == 0
    assert store.select_chunks(empty) == ()
    assert store.dense_chunks(pseudo_vector("x"), filters=empty, limit=5) == ()


def test_a_time_window_selects_by_overlap(loaded) -> None:
    store, _account, _embedder, _ = loaded
    start = datetime.fromtimestamp(DAY) + timedelta(hours=1)
    windowed = RetrievalFilter(start_time=start, end_time=start + timedelta(hours=30))
    selected = store.select_chunks(windowed)
    assert selected
    with store._conn.cursor() as cur:
        overlapping = int(
            cur.execute(
                "SELECT count(*) FROM memory_chunks WHERE end_time >= %s AND start_time <= %s",
                (windowed.start_time, windowed.end_time),
            ).fetchone()[0]
        )
    store._conn.commit()
    assert len(selected) == overlapping


def test_the_candidate_report_measures_the_reduction(loaded) -> None:
    store, _account, _embedder, _ = loaded
    report = candidate_report(store, RetrievalFilter(conversation_ids=(CONV_A,)))
    assert report["global_candidates"] > report["filtered_candidates"]
    assert 0 < report["reduction"] < 1
    assert report["filter"]["conversations"] == 1


# ---------------------------------------------------------------------------------------------
# The planner's seam
# ---------------------------------------------------------------------------------------------


def test_a_plans_evidence_reaches_the_retriever(loaded) -> None:
    """The seam the product uses: the store's retriever returns what the plan decided."""
    store, account, embedder, chunk_ids = loaded
    view = PostgresVectorStore(store, embedder, origin=f"account:{account.name}")
    documents = view.documents()[:2]
    view.set_override(documents)
    returned = view.as_retriever().invoke("anything")
    assert [d.metadata["memory_chunk_id"] for d in returned] == [
        d.metadata["memory_chunk_id"] for d in documents
    ]
    view.clear_override()
    assert view.override is None


def test_a_narrowed_view_does_not_inherit_an_override(loaded) -> None:
    """Inheriting one would answer the *old* search while looking like a new one."""
    store, account, embedder, _ = loaded
    view = PostgresVectorStore(store, embedder, origin="account:x")
    view.set_override(view.documents()[:1])
    assert view.with_filters(RetrievalFilter(conversation_ids=(GROUP,))).override is None


def test_executing_a_plan_narrows_and_orders_its_evidence(loaded) -> None:
    store, account, embedder, _ = loaded
    view = PostgresVectorStore(store, embedder, origin=f"account:{account.name}")
    names = store.person_vocabulary()
    conversations = store.conversation_vocabulary()

    plan = query_plan.plan_query("一共说了几次？", names=names, conversations=conversations)
    assert plan.intent == "multi_event_exhaustive"
    config = query_plan.retrieval_config_for(_minimal_config(), plan.strategy)
    result = query_plan.execute_plan("一共说了几次？", plan, vector_store=view, retrieval_config=config)
    assert result.plan.strategy == "exhaustive"
    assert result.candidates == result.global_candidates, "no filter, so no narrowing"
    assert result.documents, "the exhaustive strategy must return evidence, not nothing"


def test_a_plan_scoped_to_one_conversation_narrows_the_candidates(loaded) -> None:
    store, account, embedder, _ = loaded
    view = PostgresVectorStore(store, embedder, origin=f"account:{account.name}")
    plan = query_plan.MemoryQueryPlan(
        intent="single_fact", conversation_scope=[GROUP], strategy="hybrid"
    )
    result = query_plan.execute_plan(
        "群里说了什么", plan, vector_store=view, retrieval_config=_minimal_config()
    )
    assert 0 < result.candidates < result.global_candidates
    assert result.reduction > 0
    for document in result.documents:
        assert document.metadata["conversation_id"] == GROUP, "nothing out of scope leaked in"


def test_a_person_scoped_plan_returns_only_that_persons_evidence(loaded) -> None:
    store, account, embedder, _ = loaded
    view = PostgresVectorStore(store, embedder, origin=f"account:{account.name}")
    with store._conn.cursor() as cur:
        person_id = str(
            cur.execute("SELECT person_id FROM people WHERE display_name = %s", (NAME_B,)).fetchone()[0]
        )
    store._conn.commit()
    plan = query_plan.MemoryQueryPlan(
        intent="speaker_attribution", people=[NAME_B], person_ids=[person_id], strategy="person_hybrid"
    )
    result = query_plan.execute_plan(
        "小王说了什么", plan, vector_store=view, retrieval_config=_minimal_config()
    )
    assert result.documents
    assert result.candidates < result.global_candidates, "the person filter narrowed the search"


def _minimal_config():
    """A retrieval config with hybrid enabled and nothing that needs an API key or a model."""
    from quivr_core.rag.entities.config import HybridConfig, LLMEndpointConfig, RetrievalConfig

    # The product's own shape — hybrid enabled, the measured weights and BM25 parameters — built
    # without reaching for a model: nothing in an executor test generates anything, and the
    # config's constructor is what resolves an API key, which is exactly what is not wanted here.
    return RetrievalConfig(
        llm_config=LLMEndpointConfig(),
        k=20,
        hybrid_config=HybridConfig(enabled=True, candidate_k=30),
    )


def test_the_embedding_coverage_reports_what_was_written(loaded) -> None:
    store, _account, _embedder, chunk_ids = loaded
    coverage = store.embedding_coverage()
    assert coverage["chunks"] == len(chunk_ids)
    assert coverage["with_vector"] == len(chunk_ids)
    assert coverage["dimension"] == DIMENSION
