"""Offline tests for the persistent retrieval index (Phase 21A).

The claim under test is narrow and must be exact:

    a warm start reconstructs **the same retrieval** as a cold build, without embedding an
    unchanged chunk and without re-parsing the account — and it says so, in numbers that name
    nobody.

Two instruments make that checkable rather than plausible:

* **an embedder that raises if ``embed_documents`` is called.** "It was fast" is not evidence; a
  warm start that quietly re-embedded would still be fast on a synthetic corpus. The stub turns
  zero-embedding into a pass/fail result. ``embed_query`` stays allowed, because a question really
  does get embedded and pretending otherwise would make the parity test a lie;
* **a deterministic, model-free embedder**, so the cold and warm retrieval orders can be compared
  for *identity* (same sources, same order, same metadata) and not merely for shape.

Everything else is synthetic and offline: no model download, no network, no API call, no real WeChat
data. The account trees are written to a workspace-local scratch directory because ``tmp_path``
cannot be created in this environment; they are removed in the fixture teardown.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent.parent
sys.path.insert(0, str(BASE_DIR))

import index_cache  # noqa: E402
import recall  # noqa: E402
from groundedness import assess  # noqa: E402
from memory import SessionConfig  # noqa: E402
from memory.account import AccountImportReport  # noqa: E402
from memory.conversations import ACCOUNT_MANIFEST_FILENAME  # noqa: E402

SCRATCH_ROOT = BASE_DIR / "tests" / "_scratch_index_cache"

#: Markers that must never reach the manifest or a status line. The talker id is deliberately shaped
#: like a wxid and the group id like a chatroom, because those are the two shapes the privacy rule
#: names — a check that only looks for a message body would pass while the identity rode along.
DIRECT_TALKER = "wxid_synthetic_otter"
GROUP_TALKER = "47110022@chatroom"
FIRST_MESSAGE = "我把数据库换成了 Supabase，记得改配置"
SECOND_MESSAGE = "周六下午三点老地方见"
FORBIDDEN_IN_OUTPUT = (DIRECT_TALKER, GROUP_TALKER, "wxid", "chatroom", FIRST_MESSAGE, SECOND_MESSAGE)

T0 = int(datetime(2026, 9, 20, 10, 0).timestamp())


# ---------------------------------------------------------------------------------------------
# Synthetic account
# ---------------------------------------------------------------------------------------------


def message(local_id: int, when: int, text: str, *, is_send: int = 1) -> dict:
    """One WeFlow-shaped message — the shape ``memory.weflow`` parses."""
    return {
        "localId": local_id,
        "serverId": f"server-{local_id}",
        "localType": 1,
        "createTime": when,
        "isSend": is_send,
        "senderUsername": DIRECT_TALKER,
        "parsedContent": text,
    }


def write_account(root: Path, *, text: str = FIRST_MESSAGE) -> Path:
    """A two-shard, two-conversation export tree — the smallest thing with a real conversation
    boundary, so the parity test compares something that could actually go wrong."""
    (root / "MSG0").mkdir(parents=True, exist_ok=True)
    (root / "MSG1").mkdir(parents=True, exist_ok=True)
    (root / "MSG0" / f"{DIRECT_TALKER}_messages.json").write_text(
        json.dumps(
            [
                message(1, T0, text),
                message(2, T0 + 120, SECOND_MESSAGE, is_send=0),
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (root / "MSG1" / f"{DIRECT_TALKER}_messages.json").write_text(
        json.dumps([message(3, T0 + 3600, "后来我还是决定用 Postgres")], ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "MSG0" / f"{GROUP_TALKER}_messages.json").write_text(
        json.dumps([message(4, T0 + 600, "群里讨论了一下", is_send=0)], ensure_ascii=False),
        encoding="utf-8",
    )
    return root


# ---------------------------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------------------------


class ScriptedEmbedder(Embeddings):
    """Deterministic, model-free, and armed to prove that a warm start embeds nothing.

    A real ``langchain_core.embeddings.Embeddings`` subclass, so the framework's async path
    (``aembed_documents``) reaches ``embed_documents`` exactly as it does for the shipped model — the
    stub must not be a shortcut around the call being measured.

    ``embed_documents`` raises unless it is explicitly allowed, so "the warm path embedded no
    document" is a test result rather than an inference from a stopwatch. ``embed_query`` is always
    allowed: the question is embedded on every run, cache or no cache.
    """

    def __init__(
        self,
        *,
        dimension: int = 8,
        model_name: str = "scripted-bge-small",
        normalize_embeddings: bool = True,
        allow_documents: bool = False,
    ) -> None:
        self.dimension = dimension
        self.model_name = model_name
        self.encode_kwargs = {"normalize_embeddings": normalize_embeddings}
        self.allow_documents = allow_documents
        self.document_calls = 0
        self.query_calls = 0

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [digest[i % len(digest)] / 255.0 - 0.5 for i in range(self.dimension)]
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self.allow_documents:
            raise AssertionError(
                "the embedder was asked to embed documents on a path that must embed none"
            )
        self.document_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._vector(text)


@pytest.fixture(autouse=True)
def _event_loop():
    """`build_brain_from_chunks` resolves its loop with `asyncio.get_event_loop()`.

    The product deliberately does not call `asyncio.run` (it closes the loop the later synchronous
    `Brain.ask` needs), and any earlier test that did clears this thread's loop, after which
    `get_event_loop()` raises. Hand the thread one, and put back what was there.
    """
    try:
        previous = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        previous = None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        asyncio.set_event_loop(previous)
        loop.close()


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    """The product's LLM endpoint is constructed on every build; no test ever calls the model."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test-key")


@pytest.fixture(autouse=True)
def _restore_prompt_registry():
    """Undo the one piece of global state a build touches.

    ``build_account_session`` calls ``register_answer_prompt`` — the product's own behaviour, and the
    reason ``--answer-prompt`` reaches the model at all. The registry it writes into lives in the
    framework and is process-global, so a session built here with a variant would leave that variant
    installed for every later test in the same process, including the ones that assert what the
    *stock* prompt is. One process runs the whole suite, so the tests that change global state are
    the tests that put it back.
    """
    from quivr_core.rag.prompts import custom_prompts

    before = dict(custom_prompts)
    try:
        yield
    finally:
        restore_prompts(before)


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


@pytest.fixture()
def account(scratch: Path) -> Path:
    return write_account(scratch / "account")


@pytest.fixture()
def build_account_session(account: Path, scratch: Path):
    """The seam, pre-wired to a synthetic account and an explicit index directory."""

    def _build(embedder=None, **kwargs):
        return recall.build_account_session(
            kwargs.pop("account_dir", account),
            index_dir=kwargs.pop("index_dir", scratch / "index"),
            embedder=embedder or ScriptedEmbedder(allow_documents=True),
            **kwargs,
        )

    return _build


def cache_dir_of(account: Path, scratch: Path) -> Path:
    return index_cache.account_cache_dir(account, scratch / "index")


def read_manifest(cache: Path) -> dict:
    return json.loads((cache / index_cache.MANIFEST_FILENAME).read_text(encoding="utf-8"))


def write_manifest(cache: Path, payload: dict, *, sync_ready: bool = True) -> None:
    """Rewrite the manifest — optionally keeping the READY marker consistent with it.

    The two are checked separately, so a test that wants to prove the *index* disagrees with the
    manifest must keep the marker in agreement, or it would fail on the marker instead.
    """
    (cache / index_cache.MANIFEST_FILENAME).write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )
    if sync_ready:
        (cache / index_cache.READY_FILENAME).write_text(
            json.dumps(
                {
                    "cache_format_version": payload["cache_format_version"],
                    "chunk_count": payload["chunk_count"],
                    "source_fingerprint": payload["source_fingerprint"],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def retriever_sources(brain, retrieval_config, question: str) -> list[tuple[str, dict]]:
    """What the product's own hybrid retriever returns for a question, as (text, metadata).

    Uses the framework's ``QuivrQARAGLangGraph.get_retriever`` — the exact call ``Brain.ask`` makes
    — so the comparison is of the shipped retrieval path and not of a copy of it.
    """
    from quivr_core.rag.quivr_rag_langgraph import QuivrQARAGLangGraph

    rag = QuivrQARAGLangGraph(
        retrieval_config=retrieval_config, llm=None, vector_store=brain.vector_db
    )
    documents = rag.get_retriever().invoke(question)
    return [(document.page_content, dict(document.metadata)) for document in documents]


# ---------------------------------------------------------------------------------------------
# Cold miss, warm hit
# ---------------------------------------------------------------------------------------------


def test_a_cold_start_builds_the_index_and_writes_a_cache(account, scratch, build_account_session):
    embedder = ScriptedEmbedder(allow_documents=True)
    session = build_account_session(embedder)

    assert session.index.outcome == index_cache.MISS
    assert session.index.chunk_count > 0
    assert session.index.cache_written is True
    assert embedder.document_calls == 1

    cache = cache_dir_of(account, scratch)
    assert {path.name for path in cache.iterdir()} == {
        index_cache.VECTORS_FILENAME,
        index_cache.DOCSTORE_FILENAME,
        index_cache.MANIFEST_FILENAME,
        index_cache.READY_FILENAME,
    }


def test_a_warm_start_loads_it_and_embeds_nothing(account, scratch, build_account_session):
    """The core claim of the phase, measured: the warm path is never asked to embed a document."""
    build_account_session(ScriptedEmbedder(allow_documents=True))

    # Armed: any attempt to embed a document fails the test rather than passing quietly.
    refusing = ScriptedEmbedder(allow_documents=False)
    session = build_account_session(refusing)

    assert session.index.outcome == index_cache.HIT
    assert session.index.embedded is False
    assert refusing.document_calls == 0
    assert session.index.chunk_count > 0


def test_a_warm_start_does_not_re_parse_the_account(account, scratch, build_account_session, monkeypatch):
    """Skipped embedding is the headline; skipped *parsing* is the other half of the win."""
    build_account_session(ScriptedEmbedder(allow_documents=True))

    def refuse(*_args, **_kwargs):
        raise AssertionError("a warm start re-parsed the export tree")

    monkeypatch.setattr(recall, "import_account_directory", refuse)
    session = build_account_session(ScriptedEmbedder(allow_documents=False))
    assert session.index.outcome == index_cache.HIT


def test_the_warm_status_reports_counts_versions_and_timings(account, scratch, build_account_session):
    build_account_session(ScriptedEmbedder(allow_documents=True))
    session = build_account_session(ScriptedEmbedder(allow_documents=False))

    assert session.index.cache_age_seconds is not None
    assert session.index.load_ms >= 0
    assert session.index.total_ms >= session.index.load_ms
    assert session.index.cache_format_version == index_cache.CACHE_FORMAT_VERSION
    assert session.index.schema_version == index_cache.PROJECTION_VERSION
    rendered = "\n".join(session.index.lines())
    assert "INDEX CACHE HIT" in rendered
    assert "0 chunks embedded" in rendered
    assert "INDEX CACHE TIMING" in rendered


def test_the_status_object_is_json_safe(account, scratch, build_account_session):
    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    payload = session.index.as_dict()
    assert json.loads(json.dumps(payload))["outcome"] == index_cache.MISS
    assert payload["embedded"] is True


# ---------------------------------------------------------------------------------------------
# Retrieval parity — identity, not shape
# ---------------------------------------------------------------------------------------------


def test_cold_and_warm_retrieval_return_the_same_sources_in_the_same_order(
    account, scratch, build_account_session
):
    question = "我最后用了哪个数据库"
    cold = build_account_session(ScriptedEmbedder(allow_documents=True))
    cold_sources = retriever_sources(cold.brain, cold.retrieval_config, question)

    warm = build_account_session(ScriptedEmbedder(allow_documents=False))
    warm_sources = retriever_sources(warm.brain, warm.retrieval_config, question)

    assert cold_sources, "the synthetic corpus retrieved nothing, so parity is untested"
    assert [text for text, _ in warm_sources] == [text for text, _ in cold_sources]
    assert [metadata for _, metadata in warm_sources] == [metadata for _, metadata in cold_sources]


def test_the_citation_metadata_survives_the_round_trip(account, scratch, build_account_session):
    """A citation needs the vector back *and* the metadata bound to it — that binding is `index.pkl`."""
    cold = build_account_session(ScriptedEmbedder(allow_documents=True))
    warm = build_account_session(ScriptedEmbedder(allow_documents=False))

    cold_metadata = [metadata for _, metadata in retriever_sources(cold.brain, cold.retrieval_config, "数据库")]
    warm_metadata = [metadata for _, metadata in retriever_sources(warm.brain, warm.retrieval_config, "数据库")]

    assert cold_metadata
    for metadata in warm_metadata:
        for key in ("memory_chunk_id", "conversation_id", "start_time", "end_time", "chunk_index"):
            assert key in metadata, f"the cached docstore lost {key}"
    assert warm_metadata == cold_metadata


def test_the_warm_index_holds_the_same_number_of_documents_as_the_build(account, scratch, build_account_session):
    from quivr_core.rag.hybrid import iter_documents

    cold = build_account_session(ScriptedEmbedder(allow_documents=True))
    warm = build_account_session(ScriptedEmbedder(allow_documents=False))

    cold_docs = iter_documents(cold.brain.vector_db)
    warm_docs = iter_documents(warm.brain.vector_db)
    assert len(warm_docs) == len(cold_docs) == cold.index.chunk_count
    assert [doc.page_content for doc in warm_docs] == [doc.page_content for doc in cold_docs]


# ---------------------------------------------------------------------------------------------
# What invalidates an entry
# ---------------------------------------------------------------------------------------------


def test_a_changed_export_invalidates_the_cache(account, scratch, build_account_session):
    build_account_session(ScriptedEmbedder(allow_documents=True))
    export = account / "MSG0" / f"{DIRECT_TALKER}_messages.json"
    payload = json.loads(export.read_text(encoding="utf-8"))
    payload.append(message(9, T0 + 7200, "后来又加了一条消息"))
    export.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert "source export tree changed" in session.index.reason
    assert session.index.cache_written is True


def test_an_added_export_invalidates_the_cache(account, scratch, build_account_session):
    build_account_session(ScriptedEmbedder(allow_documents=True))
    (account / "MSG1" / f"{GROUP_TALKER}_messages.json").write_text(
        json.dumps([message(5, T0 + 900, "新导出的一个会话", is_send=0)], ensure_ascii=False),
        encoding="utf-8",
    )

    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert "source export tree changed" in session.index.reason


def test_a_touched_but_unchanged_export_is_a_miss_not_a_silent_hit(account, scratch, build_account_session):
    """A false miss costs a rebuild; a false hit answers from a history that is not on disk."""
    build_account_session(ScriptedEmbedder(allow_documents=True))
    export = account / "MSG0" / f"{DIRECT_TALKER}_messages.json"
    export.write_text(export.read_text(encoding="utf-8"), encoding="utf-8")

    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID


def test_a_changed_max_chars_invalidates_the_cache(account, scratch, build_account_session):
    build_account_session(ScriptedEmbedder(allow_documents=True))
    session = build_account_session(
        ScriptedEmbedder(allow_documents=True),
        session_config=SessionConfig(max_chars=SessionConfig().max_chars // 3),
    )
    assert session.index.outcome == index_cache.INVALID
    assert "chunking configuration changed" in session.index.reason


def test_a_changed_chunking_shape_version_invalidates_the_cache(
    account, scratch, build_account_session, monkeypatch
):
    build_account_session(ScriptedEmbedder(allow_documents=True))
    monkeypatch.setattr(index_cache, "CHUNKING_SHAPE_VERSION", index_cache.CHUNKING_SHAPE_VERSION + 1)
    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert "chunking configuration changed" in session.index.reason


def test_a_changed_embedding_model_invalidates_the_cache(account, scratch, build_account_session):
    build_account_session(ScriptedEmbedder(allow_documents=True))
    session = build_account_session(
        ScriptedEmbedder(allow_documents=True, model_name="some-other-model")
    )
    assert session.index.outcome == index_cache.INVALID
    assert "embedding configuration changed" in session.index.reason


def test_turning_off_normalisation_invalidates_the_cache(account, scratch, build_account_session):
    """Normalisation changes what a vector *is*, so the vectors on disk are no longer comparable."""
    build_account_session(ScriptedEmbedder(allow_documents=True))
    session = build_account_session(
        ScriptedEmbedder(allow_documents=True, normalize_embeddings=False)
    )
    assert session.index.outcome == index_cache.INVALID
    assert "embedding configuration changed" in session.index.reason


def test_an_embedder_that_disagrees_with_the_vectors_fails_loudly_not_silently(
    account, scratch, build_account_session
):
    """The limit of what a cache can check without embedding, stated rather than papered over.

    A live embedder whose *configuration* is identical but whose vectors are a different width cannot
    be told apart from the one that built the entry — telling them apart means embedding a document,
    the one thing a warm start must never do. What the cache does instead is record the width in the
    manifest and check the loaded index against it, so the disagreement surfaces as a failing search
    over a store it cannot compare with; it never becomes an answer computed from the wrong vectors.
    """
    build_account_session(ScriptedEmbedder(dimension=8, allow_documents=True))
    warm = build_account_session(ScriptedEmbedder(dimension=16, allow_documents=False))

    assert warm.index.outcome == index_cache.HIT
    assert warm.index.vector_dimension == 8  # what the vectors on disk are
    # FAISS's own guard, raised from the search: loud, immediate, and impossible to mistake for a
    # retrieval result. (It carries no message of its own, which is exactly why the manifest records
    # the width in the first place.)
    with pytest.raises(AssertionError):
        warm.brain.vector_db.similarity_search("数据库")


def test_a_bumped_projection_version_invalidates_the_cache(
    account, scratch, build_account_session, monkeypatch
):
    """The representation changed — that is exactly what Phase 20.9's sender labels were."""
    build_account_session(ScriptedEmbedder(allow_documents=True))
    monkeypatch.setattr(index_cache, "PROJECTION_VERSION", index_cache.PROJECTION_VERSION + 1)
    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert "document projection changed" in session.index.reason


def test_a_bumped_cache_format_version_invalidates_the_cache(
    account, scratch, build_account_session, monkeypatch
):
    build_account_session(ScriptedEmbedder(allow_documents=True))
    monkeypatch.setattr(index_cache, "CACHE_FORMAT_VERSION", index_cache.CACHE_FORMAT_VERSION + 1)
    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert "cache format version changed" in session.index.reason


# ---------------------------------------------------------------------------------------------
# What must NOT invalidate an entry
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "knobs",
    [
        {"k": 15},
        {"k": 5},
        {"hybrid_pool": 12},
        {"workflow": "rag"},
        {"answer_prompt": "cited-narrow"},
        {"temperature": 0.7},
        {"max_output_tokens": 512},
        {"k": 3, "hybrid_pool": 4, "workflow": "rag", "answer_prompt": "cited", "temperature": 1.0},
    ],
)
def test_retrieval_and_generation_knobs_do_not_invalidate(account, scratch, build_account_session, knobs):
    """They select from an index; they never build one. Changing k 20 -> 15 must still be a hit."""
    build_account_session(ScriptedEmbedder(allow_documents=True))
    session = build_account_session(ScriptedEmbedder(allow_documents=False), **knobs)
    assert session.index.outcome == index_cache.HIT, session.index.reason
    assert session.index.embedded is False


def test_the_adopted_reference_is_a_hit_against_an_entry_built_with_it(account, scratch, build_account_session):
    build_account_session(ScriptedEmbedder(allow_documents=True), k=recall.DEFAULT_K)
    session = build_account_session(ScriptedEmbedder(allow_documents=False), k=recall.DEFAULT_K)
    assert session.index.outcome == index_cache.HIT
    assert session.retrieval_config.k == recall.DEFAULT_K


# ---------------------------------------------------------------------------------------------
# Corruption: every failure is a miss with a stated reason, then a clean rebuild
# ---------------------------------------------------------------------------------------------


def test_a_corrupt_manifest_is_a_miss_with_a_reason(account, scratch, build_account_session):
    build_account_session(ScriptedEmbedder(allow_documents=True))
    cache = cache_dir_of(account, scratch)
    (cache / index_cache.MANIFEST_FILENAME).write_bytes(b"{not json at all")

    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert "manifest unreadable" in session.index.reason
    assert session.index.cache_written is True
    # and the entry it wrote is whole: the next run is a hit again
    assert build_account_session(ScriptedEmbedder(allow_documents=False)).index.outcome == index_cache.HIT


def test_a_truncated_manifest_is_a_miss_with_a_reason(account, scratch, build_account_session):
    build_account_session(ScriptedEmbedder(allow_documents=True))
    cache = cache_dir_of(account, scratch)
    whole = (cache / index_cache.MANIFEST_FILENAME).read_text(encoding="utf-8")
    (cache / index_cache.MANIFEST_FILENAME).write_text(whole[: len(whole) // 2], encoding="utf-8")

    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert "manifest unreadable" in session.index.reason


def test_a_manifest_with_an_unexpected_field_is_a_miss_with_a_reason(account, scratch, build_account_session):
    """A field this format version does not define cannot be interpreted, so it is not ignored."""
    build_account_session(ScriptedEmbedder(allow_documents=True))
    cache = cache_dir_of(account, scratch)
    payload = read_manifest(cache)
    payload["account_name"] = "something a future version might add"
    write_manifest(cache, payload)

    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert "manifest rejected" in session.index.reason
    assert "unexpected field" in session.index.reason


def test_a_manifest_missing_a_field_is_a_miss_with_a_reason(account, scratch, build_account_session):
    build_account_session(ScriptedEmbedder(allow_documents=True))
    cache = cache_dir_of(account, scratch)
    payload = read_manifest(cache)
    del payload["chunking_fingerprint"]
    write_manifest(cache, payload)

    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert "manifest rejected" in session.index.reason
    assert "missing field" in session.index.reason


def test_a_missing_manifest_is_a_miss_with_a_reason(account, scratch, build_account_session):
    build_account_session(ScriptedEmbedder(allow_documents=True))
    cache = cache_dir_of(account, scratch)
    (cache / index_cache.MANIFEST_FILENAME).unlink()

    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert "no manifest" in session.index.reason


def test_a_missing_vectors_artifact_is_a_miss_with_a_reason(account, scratch, build_account_session):
    build_account_session(ScriptedEmbedder(allow_documents=True))
    cache = cache_dir_of(account, scratch)
    (cache / index_cache.VECTORS_FILENAME).unlink()

    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert index_cache.VECTORS_FILENAME in session.index.reason


def test_a_missing_documents_artifact_is_a_miss_with_a_reason(account, scratch, build_account_session):
    build_account_session(ScriptedEmbedder(allow_documents=True))
    cache = cache_dir_of(account, scratch)
    (cache / index_cache.DOCSTORE_FILENAME).unlink()

    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert index_cache.DOCSTORE_FILENAME in session.index.reason


def test_an_empty_artifact_is_a_miss_with_a_reason(account, scratch, build_account_session):
    """A truncated copy leaves a file that exists and holds nothing — the worst kind of plausible."""
    build_account_session(ScriptedEmbedder(allow_documents=True))
    cache = cache_dir_of(account, scratch)
    (cache / index_cache.VECTORS_FILENAME).write_bytes(b"")

    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert "empty" in session.index.reason


def test_a_chunk_count_mismatch_is_a_miss_with_a_reason(account, scratch, build_account_session):
    """The manifest is a claim; the index is the measurement. They have to agree."""
    build_account_session(ScriptedEmbedder(allow_documents=True))
    cache = cache_dir_of(account, scratch)
    payload = read_manifest(cache)
    payload["chunk_count"] = payload["chunk_count"] + 7
    write_manifest(cache, payload)

    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert "chunk count mismatch" in session.index.reason
    assert "chunk count mismatch" not in build_account_session(
        ScriptedEmbedder(allow_documents=False)
    ).index.reason


def test_a_vector_dimension_mismatch_is_a_miss_with_a_reason(account, scratch, build_account_session):
    embedder = ScriptedEmbedder(dimension=8, allow_documents=True)
    build_account_session(embedder)
    cache = cache_dir_of(account, scratch)
    payload = read_manifest(cache)
    # Keep the manifest internally consistent, so the dimension check is what fails and not the
    # embedding fingerprint that also covers it.
    payload["vector_dimension"] = 99
    payload["embedding_fingerprint"] = index_cache.embedding_fingerprint(embedder, dimension=99)
    write_manifest(cache, payload)

    session = build_account_session(ScriptedEmbedder(dimension=8, allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert "vector dimension mismatch" in session.index.reason


def test_a_missing_ready_marker_is_a_miss_with_a_reason(account, scratch, build_account_session):
    """The marker is written last. Without it, an interrupted build looks like an index."""
    build_account_session(ScriptedEmbedder(allow_documents=True))
    cache = cache_dir_of(account, scratch)
    (cache / index_cache.READY_FILENAME).unlink()

    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert "READY" in session.index.reason
    assert build_account_session(ScriptedEmbedder(allow_documents=False)).index.outcome == index_cache.HIT


def test_a_ready_marker_that_disagrees_with_the_manifest_is_a_miss(account, scratch, build_account_session):
    """A half-promoted directory: a new manifest beside the old marker's claim."""
    build_account_session(ScriptedEmbedder(allow_documents=True))
    cache = cache_dir_of(account, scratch)
    ready = json.loads((cache / index_cache.READY_FILENAME).read_text(encoding="utf-8"))
    ready["chunk_count"] = ready["chunk_count"] + 1
    (cache / index_cache.READY_FILENAME).write_text(json.dumps(ready), encoding="utf-8")

    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert "READY" in session.index.reason


def test_a_corrupt_ready_marker_is_a_miss_with_a_reason(account, scratch, build_account_session):
    build_account_session(ScriptedEmbedder(allow_documents=True))
    cache = cache_dir_of(account, scratch)
    (cache / index_cache.READY_FILENAME).write_bytes(b"\x00\x01not a marker")

    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert "READY" in session.index.reason


def test_a_corrupt_docstore_is_a_miss_with_a_reason_and_a_clean_rebuild(
    account, scratch, build_account_session
):
    """Unpickling can fail on a truncated file; that must be a miss, never an escaping exception."""
    build_account_session(ScriptedEmbedder(allow_documents=True))
    cache = cache_dir_of(account, scratch)
    (cache / index_cache.DOCSTORE_FILENAME).write_bytes(b"not a pickle")

    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert session.index.cache_written is True
    assert build_account_session(ScriptedEmbedder(allow_documents=False)).index.outcome == index_cache.HIT


def test_a_missing_cache_directory_is_a_plain_miss(account, scratch, build_account_session):
    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.MISS
    assert "no index cache" in session.index.reason


# ---------------------------------------------------------------------------------------------
# Atomicity: staging, promotion, and what an interruption leaves behind
# ---------------------------------------------------------------------------------------------


def test_the_index_is_written_into_a_staging_directory_and_promoted(
    account, scratch, build_account_session, monkeypatch
):
    """Nothing may be written into the live entry: a reader must never see a half-built one."""
    from langchain_community.vectorstores import FAISS

    cache = cache_dir_of(account, scratch)
    seen: dict[str, Path] = {}
    original = FAISS.save_local

    def spy(self, folder_path, index_name="index"):
        seen["folder"] = Path(folder_path)
        seen["cache_existed"] = cache.exists()
        return original(self, folder_path=folder_path, index_name=index_name)

    monkeypatch.setattr(FAISS, "save_local", spy)
    build_account_session(ScriptedEmbedder(allow_documents=True))

    assert seen["folder"].name.startswith(f"{cache.name}{index_cache.STAGING_INFIX}")
    assert seen["folder"].parent == cache.parent
    assert seen["cache_existed"] is False
    # promoted: the staging directory is gone and the entry is complete
    assert not seen["folder"].exists()
    assert {path.name for path in cache.iterdir()} == {
        index_cache.VECTORS_FILENAME,
        index_cache.DOCSTORE_FILENAME,
        index_cache.MANIFEST_FILENAME,
        index_cache.READY_FILENAME,
    }


def test_a_leftover_staging_directory_is_ignored_and_cleaned_up(account, scratch, build_account_session):
    cache = cache_dir_of(account, scratch)
    leftover = cache.with_name(f"{cache.name}{index_cache.STAGING_INFIX}deadbeef")
    leftover.mkdir(parents=True)
    (leftover / "index.faiss").write_bytes(b"half a build from a run that died at minute 70")

    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.MISS
    assert not leftover.exists()
    assert build_account_session(ScriptedEmbedder(allow_documents=False)).index.outcome == index_cache.HIT


def test_a_leftover_staging_directory_does_not_stop_a_hit(account, scratch, build_account_session):
    build_account_session(ScriptedEmbedder(allow_documents=True))
    cache = cache_dir_of(account, scratch)
    leftover = cache.with_name(f"{cache.name}{index_cache.STAGING_INFIX}feedface")
    leftover.mkdir(parents=True)

    session = build_account_session(ScriptedEmbedder(allow_documents=False))
    assert session.index.outcome == index_cache.HIT
    assert not leftover.exists()


def test_a_directory_of_half_written_artifacts_is_not_an_index(account, scratch, build_account_session):
    """Rubble from a build that never got as far as a manifest. It is refused, not read."""
    cache = cache_dir_of(account, scratch)
    cache.mkdir(parents=True)
    (cache / index_cache.VECTORS_FILENAME).write_bytes(b"vectors without a manifest")
    (cache / index_cache.DOCSTORE_FILENAME).write_bytes(b"docstore without a manifest")

    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert session.index.outcome == index_cache.INVALID
    assert "no manifest" in session.index.reason
    assert session.index.cache_written is True
    assert build_account_session(ScriptedEmbedder(allow_documents=False)).index.outcome == index_cache.HIT


def test_a_build_killed_mid_write_promotes_nothing(account, scratch, build_account_session, monkeypatch):
    """The minute-70 case, simulated: partial artifacts exist, and no reader may ever see them."""
    from langchain_community.vectorstores import FAISS

    cache = cache_dir_of(account, scratch)

    def die(self, folder_path, index_name="index"):
        (Path(folder_path) / index_cache.VECTORS_FILENAME).write_bytes(b"half an index")
        raise OSError("killed while saving")

    # A private patch context rather than the test's `monkeypatch`: undoing that one would also undo
    # the fixtures' environment, which is not this test's to dismantle.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(FAISS, "save_local", die)
        session = build_account_session(ScriptedEmbedder(allow_documents=True))

        # The run still answers — the index it built is in memory — but nothing was persisted.
        assert session.index.cache_written is False
        assert "cache write failed" in session.index.reason
        assert session.index.chunk_count > 0
        assert not cache.exists(), "a half-written build was promoted into the live entry"
        assert not list(cache.parent.glob(f"{cache.name}{index_cache.STAGING_INFIX}*"))

    # And the next run is an honest cold start, not a hit against rubble.
    assert build_account_session(ScriptedEmbedder(allow_documents=True)).index.outcome == index_cache.MISS


def test_a_failed_cache_write_still_returns_a_working_index(
    account, scratch, build_account_session, monkeypatch
):
    """A cache that cannot be written is a slower next run, not a failed one."""
    from langchain_community.vectorstores import FAISS

    def refuse(self, folder_path, index_name="index"):
        raise OSError("disk full")

    monkeypatch.setattr(FAISS, "save_local", refuse)
    session = build_account_session(ScriptedEmbedder(allow_documents=True))

    assert session.index.cache_written is False
    assert "cache write failed" in session.index.reason
    assert "OSError" in session.index.reason
    assert session.index.chunk_count > 0
    assert retriever_sources(session.brain, session.retrieval_config, "数据库")
    cache = cache_dir_of(account, scratch)
    assert not cache.exists()
    assert not list(cache.parent.glob(f"{cache.name}{index_cache.STAGING_INFIX}*"))


def test_rebuild_index_ignores_a_valid_cache(account, scratch, build_account_session):
    build_account_session(ScriptedEmbedder(allow_documents=True))
    embedder = ScriptedEmbedder(allow_documents=True)
    session = build_account_session(embedder, rebuild_index=True)

    assert session.index.outcome == index_cache.REBUILD
    assert "explicit" in session.index.reason
    assert embedder.document_calls == 1
    assert build_account_session(ScriptedEmbedder(allow_documents=False)).index.outcome == index_cache.HIT


def test_deleting_the_cache_leaves_a_fully_functional_system(account, scratch, build_account_session):
    """The export tree is the source of truth; the index is an accelerator and nothing else."""
    first = build_account_session(ScriptedEmbedder(allow_documents=True))
    before = retriever_sources(first.brain, first.retrieval_config, "数据库")

    shutil.rmtree(cache_dir_of(account, scratch))
    rebuilt = build_account_session(ScriptedEmbedder(allow_documents=True))
    after = retriever_sources(rebuilt.brain, rebuilt.retrieval_config, "数据库")

    assert rebuilt.index.outcome == index_cache.MISS
    assert after == before


# ---------------------------------------------------------------------------------------------
# Location: one entry per account, override, git-ignored, never an export tree
# ---------------------------------------------------------------------------------------------


def test_the_index_directory_override_is_honoured(account, scratch):
    elsewhere = scratch / "somewhere" / "else"
    session = recall.build_account_session(
        account, index_dir=elsewhere, embedder=ScriptedEmbedder(allow_documents=True)
    )
    expected = index_cache.account_cache_dir(account, elsewhere)
    assert expected.parent == elsewhere
    assert (expected / index_cache.READY_FILENAME).exists()
    assert session.index.outcome == index_cache.MISS
    # Nothing was written under the default root for this account: the override is where the entry
    # lives, not a second copy of it.
    assert not index_cache.account_cache_dir(account).exists()


def test_two_accounts_with_the_same_name_do_not_share_an_entry(scratch):
    """A collision would answer one person's question from another person's history."""
    first = write_account(scratch / "one" / "account")
    second = write_account(scratch / "two" / "account", text="完全不同的另一段历史")

    assert index_cache.cache_leaf(first) != index_cache.cache_leaf(second)

    index_dir = scratch / "index"
    recall.build_account_session(
        first, index_dir=index_dir, embedder=ScriptedEmbedder(allow_documents=True)
    )
    second_session = recall.build_account_session(
        second, index_dir=index_dir, embedder=ScriptedEmbedder(allow_documents=True)
    )
    # The second account is a cold start, not a hit against the first account's vectors.
    assert second_session.index.outcome == index_cache.MISS


def test_the_cache_is_never_mistaken_for_an_export_tree(account, scratch, build_account_session):
    """The importer scans for ``*_messages.json``; an index must not be able to look like an export."""
    build_account_session(ScriptedEmbedder(allow_documents=True))
    cache = cache_dir_of(account, scratch)
    assert not list(cache.rglob("*_messages.json"))
    from memory.account import discover_shard_directories

    # Even with the cache INSIDE the account tree, it is not a shard directory.
    nested = account / ".index_cache"
    recall.build_account_session(
        account, index_dir=nested, embedder=ScriptedEmbedder(allow_documents=True)
    )
    assert (nested / index_cache.cache_leaf(account)) not in discover_shard_directories(account)


def test_the_index_cache_is_git_ignored() -> None:
    import subprocess

    for relative in (
        "examples/personal_recall/data/real/.index_cache/account-0123456789ab/index.faiss",
        "examples/personal_recall/data/real/.index_cache/account-0123456789ab/index.pkl",
        "examples/personal_recall/.index_cache/account-0123456789ab/manifest.json",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", relative],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode == 0, f"{relative} is not git-ignored"


def test_a_cache_nested_in_the_account_tree_does_not_invalidate_itself(account, scratch, build_account_session):
    """Otherwise every build would fingerprint its own output and rebuild forever."""
    nested = account / ".index_cache"
    first = recall.build_account_session(
        account, index_dir=nested, embedder=ScriptedEmbedder(allow_documents=True)
    )
    assert first.index.outcome == index_cache.MISS
    second = recall.build_account_session(
        account, index_dir=nested, embedder=ScriptedEmbedder(allow_documents=False)
    )
    assert second.index.outcome == index_cache.HIT


# ---------------------------------------------------------------------------------------------
# Privacy: the manifest and every status line name nobody
# ---------------------------------------------------------------------------------------------


def test_the_manifest_carries_versions_digests_and_counts_only(account, scratch, build_account_session):
    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    cache = cache_dir_of(account, scratch)
    payload = read_manifest(cache)

    assert set(payload) == index_cache.MANIFEST_KEYS

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)

    for value in strings(payload):
        looks_like_a_digest = len(value) == 64 and all(char in "0123456789abcdef" for char in value)
        looks_like_a_timestamp = value[:2] == "20" and "T" in value
        assert looks_like_a_digest or looks_like_a_timestamp, (
            f"the manifest holds a free-text value {value!r}; only versions, digests, counts and "
            "timestamps may be persisted"
        )
    assert session.index.chunk_count > 0


def test_the_manifest_and_every_status_line_carry_no_identity(account, scratch, build_account_session):
    session = build_account_session(ScriptedEmbedder(allow_documents=True))
    warm = build_account_session(ScriptedEmbedder(allow_documents=False))
    cache = cache_dir_of(account, scratch)

    rendered = (
        (cache / index_cache.MANIFEST_FILENAME).read_text(encoding="utf-8")
        + (cache / index_cache.READY_FILENAME).read_text(encoding="utf-8")
        + json.dumps(session.index.as_dict())
        + json.dumps(warm.index.as_dict())
        + "\n".join(session.index.lines())
        + "\n".join(warm.index.lines())
    )
    for forbidden in FORBIDDEN_IN_OUTPUT:
        assert forbidden not in rendered, f"the index cache leaked {forbidden!r}"


def test_the_cached_documents_hold_the_chat_text_and_the_manifest_does_not(
    account, scratch, build_account_session
):
    """The asymmetry is the design: the docstore is private because it is the index, and the
    manifest is shareable because it is only versions, digests and counts."""
    build_account_session(ScriptedEmbedder(allow_documents=True))
    cache = cache_dir_of(account, scratch)

    docstore = (cache / index_cache.DOCSTORE_FILENAME).read_bytes()
    assert FIRST_MESSAGE.encode("utf-8") in docstore
    assert FIRST_MESSAGE.encode("utf-8") not in (cache / index_cache.MANIFEST_FILENAME).read_bytes()


# ---------------------------------------------------------------------------------------------
# The reconstructed report
# ---------------------------------------------------------------------------------------------


def test_a_warm_start_reports_the_same_aggregates_as_the_cold_one(account, scratch, build_account_session):
    cold = build_account_session(ScriptedEmbedder(allow_documents=True))
    warm = build_account_session(ScriptedEmbedder(allow_documents=False))

    assert warm.report.conversations_imported == cold.report.conversations_imported
    assert warm.report.conversations_discovered == cold.report.conversations_discovered
    assert warm.report.messages_kept == cold.report.messages_kept
    assert warm.report.messages_received == cold.report.messages_received
    assert warm.report.first_timestamp == cold.report.first_timestamp
    assert warm.report.last_timestamp == cold.report.last_timestamp
    assert warm.report.partial == cold.report.partial
    assert warm.report.missing_shards == cold.report.missing_shards
    assert warm.report.shards_exported == cold.report.shards_exported


def test_a_warm_report_says_it_was_reconstructed_and_shows_no_rows(account, scratch, build_account_session):
    build_account_session(ScriptedEmbedder(allow_documents=True))
    warm = build_account_session(ScriptedEmbedder(allow_documents=False))

    assert warm.report.reconstructed_from_cache is True
    assert warm.report.per_conversation == ()
    rendered = "\n".join(warm.report.lines()) + json.dumps(warm.report.as_dict())
    assert "reconstructed from the persistent index cache" in rendered
    assert "was not persisted" in rendered


def test_a_cold_report_is_not_marked_as_reconstructed(account, scratch, build_account_session):
    cold = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert cold.report.reconstructed_from_cache is False
    assert cold.report.per_conversation


def test_a_partial_account_stays_partial_across_the_cache(scratch, build_account_session):
    """No Data Loaded != No Memory Exists, and a warm start must not lose that claim."""
    account = write_account(scratch / "partial")
    (account / ACCOUNT_MANIFEST_FILENAME).write_text(
        json.dumps({"filtered_conversations": 3}), encoding="utf-8"
    )
    cold = build_account_session(account_dir=account, embedder=ScriptedEmbedder(allow_documents=True))
    warm = build_account_session(account_dir=account, embedder=ScriptedEmbedder(allow_documents=False))

    assert cold.report.partial is True
    assert warm.index.outcome == index_cache.HIT
    assert warm.report.partial is True
    assert warm.report.filtered_conversations == 3
    assert "PARTIAL" in "\n".join(warm.report.lines())


def test_warm_report_is_rebuilt_from_the_live_tree(account, scratch, build_account_session):
    """Shard completeness is read now, not remembered: it costs a directory listing."""
    build_account_session(ScriptedEmbedder(allow_documents=True))
    cache = cache_dir_of(account, scratch)
    manifest = index_cache.IndexManifest.from_dict(read_manifest(cache))
    from memory import load_account_directory

    report = index_cache.warm_report(manifest, load_account_directory(account))
    assert report.shards_exported == ("MSG0", "MSG1")
    assert report.missing_shards == ()
    assert report.reconstructed_from_cache is True


def test_the_manifest_round_trips(account, scratch, build_account_session):
    build_account_session(ScriptedEmbedder(allow_documents=True))
    cache = cache_dir_of(account, scratch)
    payload = read_manifest(cache)
    manifest = index_cache.IndexManifest.from_dict(payload)
    assert manifest.as_dict() == payload
    assert manifest.chunk_count > 0
    assert manifest.age_seconds() < 3600


def test_a_manifest_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(index_cache.ManifestError):
        index_cache.IndexManifest.from_dict(["not", "an", "object"])


def test_report_aggregates_hold_no_conversation_id(account, scratch, build_account_session):
    cold = build_account_session(ScriptedEmbedder(allow_documents=True))
    aggregates = index_cache.report_aggregates(cold.report)
    assert set(aggregates) == set(index_cache.REPORT_KEYS)
    assert DIRECT_TALKER not in json.dumps(aggregates)
    # The one report field that is a tuple of conversation ids survives as a number, never as ids.
    assert isinstance(aggregates["conversations_without_messages"], int)


# ---------------------------------------------------------------------------------------------
# Fingerprints, in isolation
# ---------------------------------------------------------------------------------------------


def test_the_source_fingerprint_ignores_files_the_import_never_reads(account, scratch):
    """A label sidecar is written beside the exports and changes no chunk."""
    before = index_cache.source_fingerprint(account)
    (account / "conversation_labels.json").write_text('{"someone": "a name"}', encoding="utf-8")
    (account / "notes.txt").write_text("a scratch note", encoding="utf-8")
    assert index_cache.source_fingerprint(account) == before


def test_the_source_fingerprint_follows_the_files_the_import_reads(account, scratch):
    before = index_cache.source_fingerprint(account)
    listing = account / "MSG0" / "sessions.json"
    listing.write_text("[]", encoding="utf-8")
    assert index_cache.source_fingerprint(account) != before


def test_the_source_fingerprint_is_order_independent(account, scratch):
    """Two walks of the same tree must agree; no filesystem promises a stable directory order."""
    entries = index_cache.source_inventory(account)
    assert entries == sorted(entries)
    assert index_cache.source_fingerprint(account) == index_cache.source_fingerprint(account)


def test_the_source_fingerprint_survives_a_move(account, scratch):
    """It is over the *tree*, not the machine: the same history in another directory is the same
    history, and the entry stays valid."""
    before = index_cache.source_fingerprint(account)
    moved = scratch / "moved"
    shutil.copytree(account, moved)
    assert index_cache.source_fingerprint(moved) == before


def test_the_source_fingerprint_raises_rather_than_skipping_an_unreadable_file(account, scratch, monkeypatch):
    """An inventory with a hole in it cannot establish anything, so it must not answer."""
    import os as os_module

    def refuse(*_args, **_kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(os_module, "walk", refuse)
    with pytest.raises(OSError):
        index_cache.source_fingerprint(account)


def test_the_chunking_fingerprint_tracks_the_config() -> None:
    base = SessionConfig(max_chars=900)
    assert index_cache.chunking_fingerprint(base) == index_cache.chunking_fingerprint(
        SessionConfig(max_chars=900)
    )
    assert index_cache.chunking_fingerprint(base) != index_cache.chunking_fingerprint(
        SessionConfig(max_chars=901)
    )
    assert index_cache.chunking_fingerprint(base) != index_cache.chunking_fingerprint(
        SessionConfig(max_chars=900, max_gap=SessionConfig().max_gap * 2)
    )


def test_the_embedding_fingerprint_tracks_the_identity_and_the_dimension() -> None:
    base = ScriptedEmbedder(dimension=8)
    assert index_cache.embedding_fingerprint(base, dimension=8) == index_cache.embedding_fingerprint(
        ScriptedEmbedder(dimension=8), dimension=8
    )
    assert index_cache.embedding_fingerprint(base, dimension=8) != index_cache.embedding_fingerprint(
        base, dimension=9
    )
    assert index_cache.embedding_fingerprint(base, dimension=8) != index_cache.embedding_fingerprint(
        ScriptedEmbedder(dimension=8, model_name="other"), dimension=8
    )
    assert index_cache.embedding_fingerprint(base, dimension=8) != index_cache.embedding_fingerprint(
        ScriptedEmbedder(dimension=8, normalize_embeddings=False), dimension=8
    )


def test_the_embedding_fingerprint_ignores_the_device_and_the_batch_size() -> None:
    """They change how long an embedding takes, never what it is."""
    plain = ScriptedEmbedder(dimension=8)
    plain.model_kwargs = {"device": "cpu"}
    other = ScriptedEmbedder(dimension=8)
    other.model_kwargs = {"device": "cuda"}
    other.batch_size = 512
    assert index_cache.embedding_fingerprint(plain, dimension=8) == index_cache.embedding_fingerprint(
        other, dimension=8
    )


def test_the_adopted_model_identity_is_what_the_product_uses() -> None:
    assert recall.EMBEDDING_MODEL_PATH == index_cache.EMBEDDING_MODEL_PATH
    assert index_cache.EMBEDDING_NORMALIZE is True
    assert "bge-small-zh" in str(index_cache.EMBEDDING_MODEL_PATH)


# ---------------------------------------------------------------------------------------------
# One seam, reachable from every front end
# ---------------------------------------------------------------------------------------------


def test_the_cli_goes_through_the_seam() -> None:
    source = inspect.getsource(recall.main)
    assert "build_account_session(" in source
    for bypass in ("import_account_directory(", "build_account_brain(", "index_cache.save_index("):
        assert bypass not in source, f"recall.main bypasses the cache-aware seam via {bypass}"


def test_the_webapp_goes_through_the_seam() -> None:
    source = (BASE_DIR / "webapp" / "app.py").read_text(encoding="utf-8")
    assert "recall.build_account_session(" in source
    for bypass in (
        "import recall_account",
        "import index_cache",
        "index_cache.inspect(",
        "index_cache.load_store(",
        "index_cache.save_index(",
        "allow_dangerous_deserialization",
        "FAISS",
    ):
        assert bypass not in source, f"webapp/app.py grew its own index handling: {bypass}"


def test_the_acceptance_runner_goes_through_the_seam() -> None:
    source = inspect.getsource(__import__("real_eval").main)
    assert "build_account_session(" in source
    for bypass in ("import_account_directory", "build_account_brain"):
        assert bypass not in source


def test_the_seam_is_the_only_place_that_decides_the_cache() -> None:
    """Structure, not behaviour: a second cache decision cannot be added without failing here.

    ``recall.py`` holds the seam, so the assertions split: the module is allowed exactly one cache
    decision (inside the seam), and the three front ends are allowed none at all.
    """
    for name in ("webapp/app.py", "web.py", "real_eval.py"):
        source = (BASE_DIR / name).read_text(encoding="utf-8")
        for piece in ("source_fingerprint", "index_cache.inspect(", "index_cache.load_store(",
                      "index_cache.save_index(", "allow_dangerous_deserialization"):
            assert piece not in source, f"{name} made its own index decision: {piece}"

    seam = inspect.getsource(recall.build_account_session)
    assert seam.count("index_cache.inspect(") == 1
    assert seam.count("index_cache.source_fingerprint(") == 1
    assert seam.count("index_cache.load_store(") == 1
    assert seam.count("index_cache.save_index(") == 1
    whole = (BASE_DIR / "recall.py").read_text(encoding="utf-8")
    assert whole.count("index_cache.inspect(") == 1
    assert whole.count("index_cache.load_store(") == 1
    assert whole.count("index_cache.save_index(") == 1
    assert whole.count("allow_dangerous_deserialization") == 0


def test_the_corpus_path_is_untouched_by_the_cache() -> None:
    """The single-corpus path is the evaluated reference and keeps its own behaviour."""
    source = inspect.getsource(recall.build_session)
    for piece in ("register_answer_prompt", "build_llm_config(", "build_retrieval_config(", "build_brain("):
        assert piece in source
    assert "index_cache" not in source


def installed_answer_prompt():
    """The prompt the model would actually be given, straight out of the process-global registry."""
    from quivr_core.rag.prompts import TemplatePromptName, custom_prompts

    return custom_prompts[TemplatePromptName.RAG_ANSWER_PROMPT]


def prompt_text(prompt) -> str:
    return prompt.messages[3].prompt.template


def restore_prompts(before: dict) -> None:
    """Put the registry back the way ``before`` found it — what a fresh process starts from."""
    from quivr_core.rag.prompts import custom_prompts, register_prompt

    for name, prompt in before.items():
        if custom_prompts.get(name) is not prompt:
            register_prompt(name, prompt, override=True)


def test_a_cold_session_installs_the_answer_prompt_exactly_once(account, scratch, build_account_session):
    """``register_answer_prompt`` *adds to the prompt it finds*, so registering twice doubles the rules.

    The seam is where a session starts, so it is the natural place to register — but the cold path
    then hands the work to a builder that registers as well, and one build must install the variant
    once, not twice.
    """
    from run_baseline import build_cited_attributed_answer_prompt

    stock = installed_answer_prompt()
    expected = prompt_text(build_cited_attributed_answer_prompt(stock))

    cold = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert cold.index.outcome == index_cache.MISS
    assert prompt_text(installed_answer_prompt()) == expected


def test_a_warm_session_asks_with_the_same_prompt_as_a_cold_one(account, scratch, build_account_session):
    """Every real run is its own process, so the comparison starts from the registry a cold run sees.

    A warm start skips the parse and the embedding; it must not also skip *installing the prompt*, or
    the same question would be answered under different instructions depending on whether the index
    happened to be on disk.
    """
    from quivr_core.rag.prompts import custom_prompts

    before = dict(custom_prompts)
    cold = build_account_session(ScriptedEmbedder(allow_documents=True))
    assert cold.index.outcome == index_cache.MISS
    installed_by_a_cold_start = prompt_text(installed_answer_prompt())

    restore_prompts(before)
    warm = build_account_session(ScriptedEmbedder(allow_documents=False))
    assert warm.index.outcome == index_cache.HIT
    assert prompt_text(installed_answer_prompt()) == installed_by_a_cold_start


def test_a_build_that_finds_nothing_to_index_does_not_touch_the_registry(scratch, build_account_session):
    """The prompt lives in a process-global registry, so a mistyped tree must not reach it at all."""
    (scratch / "empty").mkdir()
    before = installed_answer_prompt()
    with pytest.raises(FileNotFoundError):
        build_account_session(account_dir=scratch / "empty")
    assert installed_answer_prompt() is before


# ---------------------------------------------------------------------------------------------
# The CLI, end to end
# ---------------------------------------------------------------------------------------------


def canned_answer(brain, retrieval_config, *, question, show_uncited=False):
    """The answer assembly, stubbed: this file is about the index, and no test calls a model."""
    sources: list = []
    report = assess("", sources)
    return {
        "question": question,
        "answer": "",
        "evidence": [],
        "groundedness": report.as_dict(),
        "retrieved": 0,
        "latency_ms": 0,
        "sources": sources,
        "cards": [],
        "report": report,
    }


def index_lines(output: str) -> list[str]:
    """Only the lines the CLI prints for the index cache — the ones this phase is responsible for."""
    return [line for line in output.splitlines() if line.startswith("index    :")]


def run_main(argv: list[str]) -> int:
    import sys as _sys

    previous = _sys.argv
    _sys.argv = ["recall.py", *argv]
    try:
        return recall.main()
    finally:
        _sys.argv = previous


def test_the_cli_builds_once_then_reports_a_hit(account, scratch, monkeypatch, capsys):
    index_dir = scratch / "index"
    monkeypatch.setattr(recall, "answer_question", canned_answer)
    monkeypatch.setattr(
        index_cache, "build_embedder", lambda: ScriptedEmbedder(allow_documents=True)
    )
    assert run_main(["问题", "--account", str(account), "--index-dir", str(index_dir)]) == 0
    first = capsys.readouterr().out
    assert "INDEX CACHE MISS" in first
    assert "cache written" in first

    refusing = ScriptedEmbedder(allow_documents=False)
    monkeypatch.setattr(index_cache, "build_embedder", lambda: refusing)
    assert run_main(["问题", "--account", str(account), "--index-dir", str(index_dir)]) == 0
    second = capsys.readouterr().out
    assert "INDEX CACHE HIT" in second
    assert "0 chunks embedded" in second
    assert refusing.document_calls == 0, "the CLI warm path embedded documents"

    # The index's own lines name nobody, cold or warm — that is the promise this phase adds.
    for forbidden in FORBIDDEN_IN_OUTPUT:
        assert forbidden not in "\n".join(index_lines(first))
        assert forbidden not in "\n".join(index_lines(second))
    # A warm start goes further: it prints no per-conversation rows, so the whole output is clean.
    # (A cold run prints the import table, which has always carried conversation ids: that is the
    # account report, not the index status, and this phase does not change it.)
    for forbidden in FORBIDDEN_IN_OUTPUT:
        assert forbidden not in second


def test_the_cli_rebuild_flag_rebuilds(account, scratch, monkeypatch, capsys):
    index_dir = scratch / "index"
    monkeypatch.setattr(recall, "answer_question", canned_answer)
    monkeypatch.setattr(
        index_cache, "build_embedder", lambda: ScriptedEmbedder(allow_documents=True)
    )
    run_main(["问题", "--account", str(account), "--index-dir", str(index_dir)])
    capsys.readouterr()

    assert (
        run_main(
            ["问题", "--account", str(account), "--index-dir", str(index_dir), "--rebuild-index"]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "INDEX CACHE REBUILD" in printed


def test_the_cli_reports_an_invalid_entry_and_rebuilds(account, scratch, monkeypatch, capsys):
    index_dir = scratch / "index"
    monkeypatch.setattr(recall, "answer_question", canned_answer)
    monkeypatch.setattr(
        index_cache, "build_embedder", lambda: ScriptedEmbedder(allow_documents=True)
    )
    run_main(["问题", "--account", str(account), "--index-dir", str(index_dir)])
    capsys.readouterr()
    cache = index_cache.account_cache_dir(account, index_dir)
    (cache / index_cache.READY_FILENAME).unlink()

    assert run_main(["问题", "--account", str(account), "--index-dir", str(index_dir)]) == 0
    printed = capsys.readouterr().out
    assert "INDEX CACHE INVALID" in printed
    assert "READY" in printed


def test_the_cli_still_refuses_an_account_with_no_exports(scratch, capsys):
    empty = scratch / "not_an_account"
    (empty / "empty").mkdir(parents=True)
    assert run_main(["question", "--account", str(empty)]) == 2
    assert "no shard export directories" in capsys.readouterr().err


def test_the_cli_help_documents_the_index_flags() -> None:
    source = inspect.getsource(recall.main)
    assert '"--index-dir"' in source
    assert '"--rebuild-index"' in source
