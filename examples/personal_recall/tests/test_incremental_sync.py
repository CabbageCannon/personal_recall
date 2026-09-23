"""Offline tests for the incremental WeChat sync and the incremental index (Phase 21B).

Phase 21A made an unchanged source load in seconds — and made *any* change cost the whole history: a
full re-export, a full re-import and every chunk re-embedded. The claims under test here are the ones
that make a small delta cost a small amount of work:

    a conversation that did not move is not re-embedded; a conversation that moved is re-rendered
    from **every** shard it appears in; an unchanged chunk keeps its vector; and the result is
    *exactly* what a clean rebuild of the same tree would have produced.

Three instruments, each aimed at something that could otherwise be believed rather than shown:

* **an embedder that counts, and that refuses.** It records every text it was asked to embed, and a
  test arms it with the texts that must *not* be embedded. "It was fast" is not evidence — a delta
  path that quietly re-embedded everything would still look quick on a synthetic account.
* **a fake ``weflow-cli``, driven through the exporter's own runner seam.** It answers ``sessions
  --json`` and ``export`` from a dict, resolves which shard to read from the scratch profile the
  *exporter* wrote (never the user's real config), and can be told to fail or to die. The whole sync
  path — listings, windows, empty windows, staging, promotion — runs with no WeChat installation,
  which is also the product rule: ``--account`` alone never touches a local database.
* **a clean rebuild as the oracle.** The golden test advances a generation over a delta, builds the
  same tree from scratch, and compares the documents, every metadata value, every vector, the chunk
  count and the Top-K retrieval order — for *identity*, not for "similar answers".

Everything is synthetic, offline and workspace-local (``tmp_path`` cannot be created in this
environment). No real WeChat data, no model download, no network, no API call.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import inspect
import json
import math
import shutil
import sys
import tempfile
import types
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import exporter  # noqa: E402
import incremental_index  # noqa: E402
import index_cache  # noqa: E402
import recall  # noqa: E402
import sync_wechat  # noqa: E402
from memory.conversations import ACCOUNT_MANIFEST_FILENAME  # noqa: E402
from memory.sessions import SessionConfig  # noqa: E402

SCRATCH_ROOT = BASE_DIR / "tests" / "_scratch_incremental"

#: Two direct talkers and one group. The ids are shaped like the real things (a wxid, a chatroom)
#: because those are the two shapes the privacy rule names: a check that only looks for a message
#: body would pass while the identity rode along on a log line.
TALKER_A = "wxid_synthetic_otter"
TALKER_B = "wxid_synthetic_heron"
TALKER_C = "wxid_synthetic_marten"
GROUP = "47110022@chatroom"
MEMBER = "wxid_synthetic_member"
MEMBER_NAME = "小明"

FIRST_MESSAGE = "我把数据库换成了 Supabase，记得改配置"
SECOND_MESSAGE = "周六下午三点老地方见"
THIRD_MESSAGE = "后来我还是决定用 Postgres"
GROUP_MESSAGE = "群里讨论了一下"
LATER_GROUP_MESSAGE = "群里又聊了一次"
DELTA_MESSAGE = "对了，记得把连接串发我"

#: Nothing here may reach a status line, a manifest, a classification line or a benchmark output.
FORBIDDEN_IN_OUTPUT = (
    TALKER_A,
    TALKER_B,
    TALKER_C,
    GROUP,
    MEMBER,
    MEMBER_NAME,
    "wxid",
    "chatroom",
    FIRST_MESSAGE,
    SECOND_MESSAGE,
    THIRD_MESSAGE,
    GROUP_MESSAGE,
    LATER_GROUP_MESSAGE,
    DELTA_MESSAGE,
)

HOUR = 3600


def message(
    local_id: int,
    when: int,
    text: str,
    *,
    sender: str = TALKER_A,
    is_send: int = 1,
    display: str | None = None,
    server_id: str | None = None,
) -> dict:
    """One WeFlow-shaped message — the shape ``memory.weflow`` parses."""
    entry = {
        "localId": local_id,
        "serverId": server_id or f"server-{local_id}",
        "localType": 1,
        "createTime": when,
        "isSend": is_send,
        "senderUsername": sender,
        "parsedContent": text,
    }
    if display is not None:
        entry["senderDisplay"] = display
    return entry


# ---------------------------------------------------------------------------------------------
# The account: two shards, three conversations
# ---------------------------------------------------------------------------------------------

#: A fixed local clock, so every window and every gap in these tests is arithmetic.
DAY = int(datetime(2026, 9, 20, 10, 0).timestamp())


def default_tree() -> dict[str, dict[str, list[dict]]]:
    """The starting account.

    Two things are deliberate. ``TALKER_A`` is **the multi-shard conversation**: one message in
    ``MSG0`` and another 30 hours later in ``MSG1``, so it is one conversation with two chunks whose
    tails live in different databases — everything a shard-local implementation would get wrong.
    ``GROUP`` has a member with no display name on their two messages (28 hours apart, so two
    chunks), because a later display has to relabel history that the delta never touches.
    """
    return {
        "MSG0": {
            TALKER_A: [message(1, DAY, FIRST_MESSAGE)],
            TALKER_B: [
                message(20, DAY + 600, SECOND_MESSAGE, sender=TALKER_B, is_send=0),
                message(21, DAY + 900, SECOND_MESSAGE, sender=TALKER_B),
            ],
        },
        "MSG1": {
            TALKER_A: [message(3, DAY + 30 * HOUR, THIRD_MESSAGE)],
            GROUP: [
                message(30, DAY + 2 * HOUR, GROUP_MESSAGE, sender=MEMBER, is_send=0),
                message(31, DAY + 30 * HOUR, LATER_GROUP_MESSAGE, sender=MEMBER, is_send=0),
            ],
        },
    }


def write_tree(root: Path, tree: dict[str, dict[str, list[dict]]]) -> Path:
    """Write — never delete — the export files named by ``tree``."""
    for shard, conversations in tree.items():
        directory = root / shard
        directory.mkdir(parents=True, exist_ok=True)
        for talker, messages in conversations.items():
            _write_payload(directory / f"{talker}_messages.json", messages)
    return root


def write_multi_dir(directory: Path, shards) -> Path:
    """The ``Msg/Multi`` directory: names only, never opened (``-wal`` sidecars included)."""
    directory.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        (directory / f"{shard}.db").write_bytes(b"")
        (directory / f"{shard}.db-wal").write_bytes(b"")
    return directory


def _write_payload(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------
# The instruments
# ---------------------------------------------------------------------------------------------


class CountingEmbedder(Embeddings):
    """Deterministic, model-free, count-keeping, and armed to refuse.

    ``embed_documents`` records every text it was asked for, so a test can assert *which* chunks were
    embedded rather than how many, and it raises unless it was explicitly allowed to embed at all.
    ``arm(...)`` names texts that must never be embedded: a call with any of them fails at the exact
    place an unchanged chunk was re-embedded.
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
        self.embedded: list[str] = []
        self.forbidden: set[str] = set()

    def arm(self, *texts: str) -> "CountingEmbedder":
        """Name the texts that must never be embedded. Returns self, so it reads in one line."""
        self.forbidden.update(text for text in texts if text)
        return self

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
        for text in texts:
            assert text not in self.forbidden, (
                "an unchanged chunk was re-embedded; its vector should have been reused: "
                f"{text[:40]!r}"
            )
        self.document_calls += 1
        self.embedded.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._vector(text)


class FakeCli:
    """A ``weflow-cli`` stand-in served through the exporter's own runner seam.

    It reads the database path out of the **scratch profile the exporter wrote** and switches shards
    on it — the same mechanism the real CLI uses — so a sync that exported from the wrong shard shows
    up here as another shard's messages, not as a mock returning whatever it was told.
    """

    def __init__(self, multi_dir: Path) -> None:
        self.multi_dir = multi_dir
        self.tree: dict[str, dict[str, list[dict]]] = {}
        self.listing: dict[str, dict[str, object]] = {}
        self.calls: list[list[str]] = []
        #: The shards the exporter's own scratch profile selected, in call order. Not what the test
        #: asked for — what the config the exporter wrote pointed the CLI at.
        self.shards_used: list[str] = []
        self.sessions_calls = 0
        self.fail_for: set[str] = set()
        self.interrupt_on: str | None = None
        #: A listing entry with no usable ``lastTimestamp``: "cannot prove it did not move".
        self.forget: set[tuple[str, str]] = set()

    # --- what the databases hold ---------------------------------------------------------------
    def set(self, shard: str, talker: str, messages: list[dict]) -> "FakeCli":
        """The database for ``(shard, talker)`` now holds exactly these messages."""
        self.tree.setdefault(shard, {})[talker] = list(messages)
        self.listing.setdefault(shard, {})[talker] = (
            max(int(entry["createTime"]) for entry in messages) if messages else 0
        )
        return self

    def listed_only(self, shard: str, talker: str, stamp: int) -> "FakeCli":
        """A conversation the listing knows and the exporter can emit nothing for.

        The listing's clock is fed by every message the account holds; the JSON export skips the ones
        with no user-visible content. A conversation whose only activity is of that kind is therefore
        real, moved, and empty at the same time — the case ``未找到消息`` describes, and the reason
        "no new messages" must not be reported as a failure.
        """
        self.listing.setdefault(shard, {})[talker] = int(stamp)
        return self

    # --- the runner ----------------------------------------------------------------------------
    def __call__(self, argv: list[str], env: dict[str, str]) -> types.SimpleNamespace:
        config = json.loads(
            (Path(env["USERPROFILE"]) / ".weflow-cli" / "config.json").read_text(encoding="utf-8")
        )
        database = Path(config.get("dbPath3x") or config.get("dbPath") or "")
        shard = database.stem
        known = set(self.tree) | set(self.listing)
        assert shard in known, f"the exporter pointed at a shard the fixture does not have: {shard}"
        self.shards_used.append(shard)

        if argv[0] == "sessions":
            self.sessions_calls += 1
            rows = [
                {
                    "username": talker,
                    "displayName": "",
                    "type": 2 if "@chatroom" in talker else 1,
                    "lastTimestamp": (
                        None if (shard, talker) in self.forget else stamp
                    ),
                }
                for talker, stamp in sorted(self.listing.get(shard, {}).items())
            ]
            return types.SimpleNamespace(
                returncode=0, stdout=json.dumps({"success": True, "sessions": rows}), stderr=""
            )

        assert argv[0] == "export", f"unexpected command: {argv}"
        self.calls.append(list(argv))
        talker = argv[1]
        out_dir = Path(argv[argv.index("--output") + 1])
        since = argv[argv.index("--from") + 1] if "--from" in argv else ""
        if self.interrupt_on == f"{shard}/{talker}":
            # A window that was being written when the run died: the file lands in staging and the
            # process never comes back to remove it.
            _write_payload(
                out_dir / f"{talker}_messages.json", self.tree.get(shard, {}).get(talker, [])
            )
            raise KeyboardInterrupt("killed while the export was running")
        if talker in self.fail_for:
            return types.SimpleNamespace(
                returncode=1,
                stdout=json.dumps(
                    {"success": False, "code": "EXPORT_FAILED", "error": f"数据库无法读取（{talker}）"}
                ),
                stderr="",
            )

        messages = list(self.tree.get(shard, {}).get(talker, ()))
        if since:
            cutoff = datetime.fromisoformat(since).timestamp()
            messages = [entry for entry in messages if int(entry["createTime"]) >= cutoff]
        if not messages:
            # What the real CLI does with a window that holds nothing: an error, not an empty set.
            return types.SimpleNamespace(
                returncode=1,
                stdout=json.dumps(
                    {
                        "success": False,
                        "code": "EXPORT_FAILED",
                        "error": f"{exporter.NO_MESSAGES_MARKER}（{talker}）",
                    }
                ),
                stderr="",
            )

        path = _write_payload(out_dir / f"{talker}_messages.json", messages)
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"success": True, "count": len(messages), "path": str(path)}),
            stderr="",
        )


# ---------------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------------

SYNTHETIC_CONFIG = {
    "dbPath": r"C:\synthetic\WeChat Files\wxid_synthetic\Msg\Multi\MSG0.db",
    "wxid": "wxid_synthetic",
    "decryptKey3x": "lock:synthetic-not-a-real-key",
    "dataVersion": "3.x",
    "dbPath3x": r"C:\synthetic\WeChat Files\wxid_synthetic\Msg\Multi\MSG0.db",
}


@pytest.fixture(autouse=True)
def _event_loop():
    """``build_brain_from_chunks`` resolves its loop with ``asyncio.get_event_loop()``."""
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
    """Undo the one piece of global state a build touches (the framework's prompt registry).

    The registry is a read-only view over a private dict, so it is put back through the module's own
    ``register_prompt`` rather than by writing into it — the same way ``test_index_cache`` does it,
    and the only way that cannot half-restore a prompt.
    """
    from quivr_core.rag.prompts import custom_prompts, register_prompt

    before = dict(custom_prompts)
    try:
        yield
    finally:
        for name, prompt in before.items():
            if custom_prompts.get(name) is not prompt:
                register_prompt(name, prompt, override=True)


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
def fake_home(scratch: Path, monkeypatch) -> Path:
    """An isolated HOME/USERPROFILE holding a synthetic ``weflow-cli`` config. Never ``~``.

    The exporter resolves its config through ``homedir()``, so this is what lets it run at all — and
    it is simultaneously the guarantee that no test can reach the user's real profile: the path is a
    scratch directory this fixture owns, and the runner never sees anything else.
    """
    home = scratch / "home"
    (home / ".weflow-cli").mkdir(parents=True)
    (home / ".weflow-cli" / "config.json").write_text(
        json.dumps(SYNTHETIC_CONFIG, indent=2), encoding="utf-8"
    )
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture()
def tree() -> dict[str, dict[str, list[dict]]]:
    return default_tree()


@pytest.fixture()
def account(scratch: Path, tree) -> Path:
    return write_tree(scratch / "account", tree)


@pytest.fixture()
def cli(scratch: Path, tree, fake_home) -> FakeCli:
    """The fake databases start out holding exactly what the export tree holds."""
    multi_dir = write_multi_dir(scratch / "multi", sorted(tree))
    fake = FakeCli(multi_dir)
    for shard, conversations in tree.items():
        for talker, messages in conversations.items():
            fake.set(shard, talker, messages)
    return fake


INDEX_DIR = "index"


def _session(account: Path, scratch: Path, embedder=None, **kwargs):
    """The seam, pre-wired to a synthetic account and an explicit index directory.

    ``index_dir`` is never left to the module default: the product's default index root is inside
    ``data/real/``, and a test that forgot to override it would write a synthetic account into the
    user's real index area. Every path here is inside the scratch directory this test owns.
    """
    return recall.build_account_session(
        kwargs.pop("account_dir", account),
        index_dir=kwargs.pop("index_dir", None) or scratch / INDEX_DIR,
        embedder=embedder if embedder is not None else CountingEmbedder(),
        **kwargs,
    )


def _synced(account: Path, scratch: Path, cli: FakeCli, embedder=None, **kwargs):
    """The seam, with the WeChat sync driven by the fake CLI."""
    return _session(
        account,
        scratch,
        embedder,
        sync=True,
        multi_dir=cli.multi_dir,
        sync_runner=cli,
        **kwargs,
    )


def cache_dir_of(account: Path, scratch: Path, index_dir: Path | None = None) -> Path:
    return index_cache.account_cache_dir(account, index_dir or scratch / INDEX_DIR)


def checkpoint_of(account: Path, scratch: Path) -> Path:
    return sync_wechat.sync_state_path(cache_dir_of(account, scratch))


def build_once(account: Path, scratch: Path, *, index_dir: Path | None = None):
    """A cold build with everything allowed: the generation every other session starts from."""
    override = {} if index_dir is None else {"index_dir": index_dir}
    return _session(account, scratch, CountingEmbedder(allow_documents=True), **override)


def bootstrap(account: Path, scratch: Path, cli: FakeCli):
    """A cold build followed by one sync: the state a real incremental run starts from.

    The second session is not padding. A cold build cannot know any ``lastTimestamp`` — nothing has
    asked WeChat — so the first sync exports every conversation once, adds nothing to any file, and
    records where each of them stood. That bootstrap is part of the phase, not a test detail: it is
    what makes a later sync cost one listing per shard, and it exercises the promise that a window
    re-sending known messages rewrites nothing (a new mtime would invalidate the whole index).
    """
    built = build_once(account, scratch)
    synced = _synced(account, scratch, cli, CountingEmbedder(allow_documents=True))
    return built, synced


# ---------------------------------------------------------------------------------------------
# Reading a generation back
# ---------------------------------------------------------------------------------------------


def chunks_of(session) -> list[tuple[str, dict]]:
    """The stored documents in index order as ``(text, metadata)`` — the sequence a rebuild makes."""
    documents = incremental_index.stored_documents(session.brain.vector_db)
    return [(document.page_content, dict(document.metadata)) for document in documents]


def vectors_of(session) -> dict[str, list[float]]:
    """``(conversation, chunk id) -> vector``, read back one by one through ``reconstruct_n``."""
    store = session.brain.vector_db
    out: dict[str, list[float]] = {}
    for position, document in enumerate(incremental_index.stored_documents(store)):
        key = f"{document.metadata['conversation_id']}\x00{document.metadata['memory_chunk_id']}"
        out[key] = [round(float(value), 6) for value in store.index.reconstruct_n(position, 1)[0]]
    return out


def texts_of(session, conversation_id: str) -> list[str]:
    return [
        text
        for text, metadata in chunks_of(session)
        if metadata["conversation_id"] == conversation_id
    ]


def vector_rows(session, conversation_id: str) -> list[list[float]]:
    return [
        vector
        for key, vector in vectors_of(session).items()
        if key.startswith(f"{conversation_id}\x00")
    ]


def retrieval(session, question: str) -> list[tuple[str, dict]]:
    """What the product's own hybrid retriever returns — the call ``Brain.ask`` makes."""
    from quivr_core.rag.quivr_rag_langgraph import QuivrQARAGLangGraph

    rag = QuivrQARAGLangGraph(
        retrieval_config=session.retrieval_config, llm=None, vector_store=session.brain.vector_db
    )
    return [
        (document.page_content, dict(document.metadata))
        for document in rag.get_retriever().invoke(question)
    ]


def fingerprint_of(account: Path, scratch: Path) -> str:
    return index_cache.source_fingerprint(account, exclude=(scratch / INDEX_DIR,))


def assert_no_identity(text: str) -> None:
    for marker in FORBIDDEN_IN_OUTPUT:
        assert marker not in text, f"an identity marker reached a status line: {marker}"


# ---------------------------------------------------------------------------------------------
# A — a no-op costs nothing
# ---------------------------------------------------------------------------------------------


def test_a_sync_with_nothing_new_is_a_cache_hit_and_embeds_nothing(account, scratch, cli):
    _, first = bootstrap(account, scratch, cli)

    # The first sync exported every conversation (nothing had ever asked WeChat) and rewrote no
    # file: a window re-sending what the tree already holds must not change a single mtime, because
    # a changed mtime is a changed source, and a changed source is a full rebuild.
    assert first.sync is not None and first.sync.ok
    assert first.sync.outcome == sync_wechat.SYNC_NO_CHANGE
    assert first.sync.files_promoted == 0
    assert first.sync.messages_added == 0
    assert first.sync.duplicates_dropped > 0, "the bootstrap window re-sent known messages"
    assert first.index.outcome == index_cache.HIT

    state = sync_wechat.load_sync_state(cache_dir_of(account, scratch))
    assert state is not None
    assert sum(len(rows) for rows in state.conversations.values()) == 4, (
        "the sync learned where each conversation stood and did not record it"
    )

    fingerprint_before = fingerprint_of(account, scratch)
    exports_before = len(cli.calls)
    listings_before = cli.sessions_calls

    embedder = CountingEmbedder()
    second = _synced(
        account,
        scratch,
        cli,
        embedder.arm(*(text for text, _ in chunks_of(first))),
        incremental_update=True,
    )

    assert second.index.outcome == index_cache.HIT
    assert second.sync is not None and second.sync.outcome == sync_wechat.SYNC_NO_CHANGE
    assert second.sync.exports_run == 0, "a conversation that did not move was exported anyway"
    assert len(cli.calls) == exports_before, f"the no-op sync ran exports: {cli.calls}"
    assert cli.sessions_calls > listings_before, "the listing is the free signal this path uses"
    assert embedder.document_calls == 0
    assert fingerprint_of(account, scratch) == fingerprint_before


def test_a_sync_without_the_database_directory_refuses_rather_than_guessing(account, scratch):
    with pytest.raises(ValueError) as error:
        _session(account, scratch, sync=True)
    assert "multi-dir" in str(error.value)


def test_a_plain_account_run_never_asks_wechat(account, scratch, cli, fake_home):
    """``--account`` alone is the offline, evaluated path: it must not touch a local database."""
    session = build_once(account, scratch)

    assert cli.sessions_calls == 0 and cli.calls == []
    assert session.sync is None
    state = sync_wechat.load_sync_state(cache_dir_of(account, scratch))
    assert state is not None, "a generation must leave a checkpoint for a later delta to start from"
    assert state.conversations == {}, "a run that asked WeChat nothing recorded no timestamps"
    # And the profile the exporter would have copied keys out of is exactly as the fixture wrote it.
    assert json.loads((fake_home / ".weflow-cli" / "config.json").read_text(encoding="utf-8")) == (
        SYNTHETIC_CONFIG
    )


# ---------------------------------------------------------------------------------------------
# B — one message in one conversation re-imports that conversation and nothing else
# ---------------------------------------------------------------------------------------------


def test_an_appended_message_reimports_only_the_conversation_that_moved(account, scratch, cli):
    _, first = bootstrap(account, scratch, cli)
    before = {cid: texts_of(first, cid) for cid in (TALKER_A, TALKER_B, GROUP)}
    # Two chunks for the conversation that spans two shards, two for the group whose member speaks
    # 28 hours apart, one for the direct talker whose two messages are five minutes apart.
    assert [len(rows) for rows in before.values()] == [2, 1, 2]
    vectors_before = vectors_of(first)
    exports_before = len(cli.calls)

    cli.set(
        "MSG0", TALKER_A, [message(1, DAY, FIRST_MESSAGE), message(40, DAY + 120, DELTA_MESSAGE)]
    )
    embedder = CountingEmbedder(allow_documents=True).arm(
        *(text for rows in before.values() for text in rows[1:])
    )
    session = _synced(account, scratch, cli, embedder, incremental_update=True)

    assert session.index.outcome == index_cache.INCREMENTAL
    assert session.index.conversations_reimported == 1
    assert session.sync is not None and session.sync.conversations_refreshed == 1
    # One conversation, in one shard, over a bounded window. The file is not the unit of work, but
    # this is what the delta itself cost.
    ran = cli.calls[exports_before:]
    assert len(ran) == 1 and ran[0][1] == TALKER_A and "--from" in ran[0]
    assert session.index.chunks_embedded == 1, "an unaffected chunk was re-embedded"
    assert session.index.chunk_count == first.index.chunk_count
    assert DELTA_MESSAGE in "\n".join(texts_of(session, TALKER_A))
    # Every other conversation is untouched: same text, and byte-identical vectors.
    assert texts_of(session, TALKER_B) == before[TALKER_B]
    assert texts_of(session, GROUP) == before[GROUP]
    after = vectors_of(session)
    for key, vector in vectors_before.items():
        # Chunk ids are 1-based (`-session-0001` is the first session of the conversation), so the
        # first chunk of this conversation is the one the delta rewrote.
        if key != f"{TALKER_A}\x00{TALKER_A}-session-0001":
            assert after[key] == vector, f"an untouched chunk's vector changed: {key}"


def test_the_delta_is_merged_into_the_tree_without_re_exporting_history(account, scratch, cli):
    bootstrap(account, scratch, cli)
    path = account / "MSG0" / f"{TALKER_A}_messages.json"
    before = json.loads(path.read_text(encoding="utf-8"))

    cli.set("MSG0", TALKER_A, [*before, message(41, DAY + 300, DELTA_MESSAGE)])
    _synced(account, scratch, cli, CountingEmbedder(allow_documents=True), incremental_update=True)

    merged = json.loads(path.read_text(encoding="utf-8"))
    assert [entry["serverId"] for entry in merged] == [
        entry["serverId"] for entry in before
    ] + ["server-41"], "the merge rewrote or reordered history instead of appending to it"


# ---------------------------------------------------------------------------------------------
# C/D — the tail chunk is decided by the whole conversation, not by the new message
# ---------------------------------------------------------------------------------------------


def test_a_message_inside_the_gap_replaces_the_tail_chunk_without_duplicating_it(
    account, scratch, cli
):
    """A gap is a *session* boundary, not a chunk boundary: the tail absorbs the new message."""
    _, first = bootstrap(account, scratch, cli)
    tail = texts_of(first, GROUP)[-1]
    assert LATER_GROUP_MESSAGE in tail

    cli.set(
        "MSG1",
        GROUP,
        [
            message(30, DAY + 2 * HOUR, GROUP_MESSAGE, sender=MEMBER, is_send=0),
            message(31, DAY + 30 * HOUR, LATER_GROUP_MESSAGE, sender=MEMBER, is_send=0),
            message(32, DAY + 30 * HOUR + 60, DELTA_MESSAGE, sender=MEMBER, is_send=0),
        ],
    )
    session = _synced(
        account, scratch, cli, CountingEmbedder(allow_documents=True), incremental_update=True
    )

    after = texts_of(session, GROUP)
    assert len(after) == 2, "the new message split the tail instead of joining it"
    assert after.count(tail) == 0, "the old tail survived beside the new one"
    assert DELTA_MESSAGE in after[-1] and LATER_GROUP_MESSAGE in after[-1]
    assert session.index.chunks_embedded == 1
    assert session.index.chunks_reused == session.index.chunk_count - 1


def test_a_message_beyond_the_gap_leaves_the_tail_alone_and_embeds_only_the_new_chunk(
    account, scratch, cli
):
    _, first = bootstrap(account, scratch, cli)
    before = texts_of(first, GROUP)

    cli.set(
        "MSG1",
        GROUP,
        [
            message(30, DAY + 2 * HOUR, GROUP_MESSAGE, sender=MEMBER, is_send=0),
            message(31, DAY + 30 * HOUR, LATER_GROUP_MESSAGE, sender=MEMBER, is_send=0),
            message(33, DAY + 40 * HOUR, DELTA_MESSAGE, sender=MEMBER, is_send=0),
        ],
    )
    embedder = CountingEmbedder(allow_documents=True).arm(*before)
    session = _synced(account, scratch, cli, embedder, incremental_update=True)

    after = texts_of(session, GROUP)
    assert after[:2] == before, "an untouched chunk changed"
    assert DELTA_MESSAGE in after[-1]
    assert session.index.chunks_embedded == 1
    assert embedder.embedded == [after[-1]]


# ---------------------------------------------------------------------------------------------
# E — a new conversation leaves the existing ones alone
# ---------------------------------------------------------------------------------------------


def test_a_new_conversation_is_added_without_touching_the_others(account, scratch, cli):
    _, first = bootstrap(account, scratch, cli)
    before = {cid: texts_of(first, cid) for cid in (TALKER_A, TALKER_B, GROUP)}
    vectors_before = vectors_of(first)

    # The conversation exists in WeChat and nowhere in the tree: it is discovered by the listing.
    fresh = [message(50, DAY + 5 * HOUR, DELTA_MESSAGE, sender=TALKER_C)]
    cli.set("MSG1", TALKER_C, fresh)

    embedder = CountingEmbedder(allow_documents=True).arm(
        *(text for rows in before.values() for text in rows)
    )
    session = _synced(account, scratch, cli, embedder, incremental_update=True)

    assert session.index.outcome == index_cache.INCREMENTAL
    assert session.sync is not None and session.sync.conversations_added == 1
    assert (account / "MSG1" / f"{TALKER_C}_messages.json").is_file()
    assert DELTA_MESSAGE in "\n".join(texts_of(session, TALKER_C))
    assert {cid: texts_of(session, cid) for cid in before} == before
    after = vectors_of(session)
    for key, vector in vectors_before.items():
        assert after[key] == vector, f"an untouched conversation's vector changed: {key}"


# ---------------------------------------------------------------------------------------------
# F — a conversation in two shards is rebuilt whole
# ---------------------------------------------------------------------------------------------


def test_a_conversation_in_two_shards_is_rebuilt_from_both(account, scratch, cli):
    """The hard requirement: the unit of work is the conversation, never the changed file.

    ``TALKER_A``'s tail chunk was built from a message that lives in ``MSG1``. The delta arrives in
    ``MSG0``, close enough to that tail to be absorbed by it — so the correct chunk can only be made
    by re-reading *both* shards. A shard-local implementation would render ``MSG0`` alone and lose
    the ``MSG1`` message: a different index, not a cheaper one.
    """
    _, first = bootstrap(account, scratch, cli)
    assert len(texts_of(first, TALKER_A)) == 2

    cli.set(
        "MSG0",
        TALKER_A,
        [message(1, DAY, FIRST_MESSAGE), message(44, DAY + 30 * HOUR + 120, DELTA_MESSAGE)],
    )
    session = _synced(
        account, scratch, cli, CountingEmbedder(allow_documents=True), incremental_update=True
    )

    assert session.index.outcome == index_cache.INCREMENTAL
    assert cli.shards_used[-1] == "MSG0", "the export did not come from the shard that moved"
    after = texts_of(session, TALKER_A)
    assert len(after) == 2
    assert FIRST_MESSAGE in after[0] and DELTA_MESSAGE not in after[0]
    assert THIRD_MESSAGE in after[1] and DELTA_MESSAGE in after[1], (
        "the tail was rendered without the message that lives in the other shard"
    )
    assert session.index.chunks_embedded == 1

    clean = build_once(account, scratch, index_dir=scratch / "clean")
    assert chunks_of(session) == chunks_of(clean)


def test_a_duplicate_server_id_across_shards_is_still_deduped_after_a_delta(account, scratch, cli):
    """The cross-shard dedupe belongs to the conversation, so a delta must not lose it."""
    bootstrap(account, scratch, cli)
    twin = message(1, DAY, FIRST_MESSAGE)
    write_tree(
        account, {"MSG1": {TALKER_A: [message(3, DAY + 30 * HOUR, THIRD_MESSAGE), twin]}}
    )

    session = _session(
        account, scratch, CountingEmbedder(allow_documents=True), incremental_update=True
    )
    clean = build_once(account, scratch, index_dir=scratch / "clean")

    assert session.index.outcome == index_cache.INCREMENTAL
    assert chunks_of(session) == chunks_of(clean)
    assert texts_of(session, TALKER_A) == texts_of(clean, TALKER_A)
    assert len(texts_of(session, TALKER_A)) == 2, "the twin was counted as a second message"


# ---------------------------------------------------------------------------------------------
# G — a new shard is discovered
# ---------------------------------------------------------------------------------------------


def test_a_new_shard_is_discovered_and_ingested(account, scratch, cli):
    _, first = bootstrap(account, scratch, cli)
    before = texts_of(first, TALKER_A)

    fresh = [message(60, DAY + 6 * HOUR, DELTA_MESSAGE, sender=TALKER_C)]
    cli.set("MSG2", TALKER_C, fresh)
    write_multi_dir(cli.multi_dir, ["MSG2"])

    session = _synced(
        account,
        scratch,
        cli,
        CountingEmbedder(allow_documents=True).arm(*before),
        incremental_update=True,
    )

    assert session.index.outcome == index_cache.INCREMENTAL
    assert DELTA_MESSAGE in "\n".join(texts_of(session, TALKER_C))
    assert texts_of(session, TALKER_A) == before
    state = sync_wechat.load_sync_state(cache_dir_of(account, scratch))
    assert state is not None and "MSG2" in state.shards


# ---------------------------------------------------------------------------------------------
# H — a delta that repeats a known message changes nothing at all
# ---------------------------------------------------------------------------------------------


def test_a_delta_that_re_sends_known_messages_adds_only_what_is_new(account, scratch, cli):
    """The overlap window exists because there is no cursor. The dedupe is what makes that safe.

    The window starts six hours before the checkpoint, so the exporter hands back messages the tree
    already holds — by design. They must be recognised and dropped, not appended: a duplicated event
    would give the conversation a second copy of a message, which is a different chunk text, a
    different chunk count and a different answer, and no rebuild would ever produce it.
    """
    build, _ = bootstrap(account, scratch, cli)
    cli.set(
        "MSG0",
        TALKER_B,
        [
            message(20, DAY + 600, SECOND_MESSAGE, sender=TALKER_B, is_send=0),
            message(21, DAY + 900, SECOND_MESSAGE, sender=TALKER_B),
            message(22, DAY + 1200, DELTA_MESSAGE, sender=TALKER_B, is_send=0),
        ],
    )
    session = _synced(
        account,
        scratch,
        cli,
        CountingEmbedder(allow_documents=True).arm(*(text for text, _ in chunks_of(build))),
        incremental_update=True,
    )

    assert session.sync is not None
    assert session.sync.duplicates_dropped == 2, "the re-sent messages were not recognised"
    assert session.sync.messages_added == 1
    assert session.sync.files_promoted == 1

    stored = json.loads((account / "MSG0" / f"{TALKER_B}_messages.json").read_text(encoding="utf-8"))
    server_ids = [entry["serverId"] for entry in stored]
    assert len(server_ids) == len(set(server_ids)), "a repeated message was stored twice"
    assert len(stored) == 3
    after = texts_of(session, TALKER_B)
    assert len(after) == 1 and after[0].count("\n") == 2, "an event was duplicated into the chunk"
    clean = build_once(account, scratch, index_dir=scratch / "clean")
    assert chunks_of(session) == chunks_of(clean)
    assert session.index.chunk_count == clean.index.chunk_count


# ---------------------------------------------------------------------------------------------
# I — an edited message replaces its chunk and its vector
# ---------------------------------------------------------------------------------------------


def test_an_edited_message_replaces_its_chunk_and_drops_the_stale_vector(account, scratch, cli):
    build, _ = bootstrap(account, scratch, cli)
    stale_text = texts_of(build, TALKER_B)[0]
    stale_vectors = vector_rows(build, TALKER_B)

    # An edit arrives as a full re-export of the file (a merge is a union: it would keep the old
    # text, which is the documented rule), so the tree is rewritten with the new wording.
    write_tree(
        account,
        {
            "MSG0": {
                TALKER_B: [
                    message(20, DAY + 600, DELTA_MESSAGE, sender=TALKER_B, is_send=0),
                    message(21, DAY + 900, SECOND_MESSAGE, sender=TALKER_B),
                ]
            }
        },
    )
    session = _session(
        account, scratch, CountingEmbedder(allow_documents=True), incremental_update=True
    )

    assert session.index.outcome == index_cache.INCREMENTAL
    after = texts_of(session, TALKER_B)
    assert len(after) == 1 and after[0] != stale_text and DELTA_MESSAGE in after[0]
    assert stale_text not in after, "the stale chunk survived the edit"
    # The conversation's one chunk was re-embedded; everything else in the account kept its vector.
    assert session.index.chunks_embedded == 1
    assert session.index.chunks_reused == session.index.chunk_count - 1
    for vector in vector_rows(session, TALKER_B):
        assert vector not in stale_vectors, "the stale vector is still in the index"

    clean = build_once(account, scratch, index_dir=scratch / "clean")
    assert chunks_of(session) == chunks_of(clean)


# ---------------------------------------------------------------------------------------------
# J — a display name on a new message relabels the speaker's earlier chunks
# ---------------------------------------------------------------------------------------------


def test_a_new_display_name_relabels_the_speakers_earlier_chunks(account, scratch, cli):
    """The whole-conversation rule: a delta can change the text of chunks it does not touch.

    The member's first chunk is 28 hours before the delta and shares no message with it. Nothing in
    the delta mentions it — but the speaker is one person, and once the export says their name, every
    line they spoke renders with it. Reading the delta alone would leave that chunk under a
    pseudonym, which is a different index from the one a rebuild produces.
    """
    _, first = bootstrap(account, scratch, cli)
    before = texts_of(first, GROUP)
    assert len(before) == 2 and "成员A" in before[0]

    cli.set(
        "MSG1",
        GROUP,
        [
            message(30, DAY + 2 * HOUR, GROUP_MESSAGE, sender=MEMBER, is_send=0),
            message(31, DAY + 30 * HOUR, LATER_GROUP_MESSAGE, sender=MEMBER, is_send=0),
            message(
                32, DAY + 30 * HOUR + 60, DELTA_MESSAGE, sender=MEMBER, is_send=0, display=MEMBER_NAME
            ),
        ],
    )
    session = _synced(
        account, scratch, cli, CountingEmbedder(allow_documents=True), incremental_update=True
    )

    after = texts_of(session, GROUP)
    assert MEMBER_NAME in "\n".join(after)
    assert before[0] not in after, "an earlier chunk kept a pseudonym the conversation no longer uses"
    assert "成员A" not in "\n".join(after)
    assert session.index.chunks_embedded == 2, "the relabelled chunk was not re-embedded"

    clean = build_once(account, scratch, index_dir=scratch / "clean")
    assert texts_of(session, GROUP) == texts_of(clean, GROUP)


# ---------------------------------------------------------------------------------------------
# K–P — what may be advanced in place, and what must rebuild
# ---------------------------------------------------------------------------------------------


def test_the_classifier_calls_a_moved_tree_an_incremental_candidate(account, scratch, cli):
    """K: the same rules over a newer tree is the one change this path may advance."""
    bootstrap(account, scratch, cli)
    cache = cache_dir_of(account, scratch)
    write_tree(
        account,
        {
            "MSG0": {
                TALKER_A: [message(1, DAY, FIRST_MESSAGE), message(70, DAY + 120, DELTA_MESSAGE)]
            }
        },
    )

    inspection = index_cache.inspect(
        cache,
        expected=index_cache.ExpectedIndex(
            source_fingerprint=index_cache.source_fingerprint(account, exclude=(cache.parent,)),
            chunking_fingerprint=index_cache.chunking_fingerprint(SessionConfig()),
            schema_version=index_cache.PROJECTION_VERSION,
            embedder=CountingEmbedder(),
        ),
    )
    assert inspection.outcome == index_cache.INVALID
    assert inspection.reason == index_cache.SOURCE_CHANGED
    assert inspection.manifest is not None

    state = sync_wechat.load_sync_state(cache)
    assert state is not None
    classification = incremental_index.classify_source_change(
        inspection=inspection,
        # The *before* side is the checkpoint's own inventory — the tree the index was built from —
        # not a second walk of the tree as it is now, which would diff the change against itself.
        previous_inventory=state.inventory,
        current_inventory=index_cache.source_inventory_entries(account, exclude=(cache.parent,)),
        state=state,
    )
    assert classification.outcome == incremental_index.SAFE_INCREMENTAL_CHANGE
    assert classification.affected == (TALKER_A,)
    assert classification.diff is not None and classification.diff.touched_files == 1
    assert_no_identity(" ".join(classification.lines()))
    assert_no_identity(json.dumps(classification.as_dict()))


@pytest.mark.parametrize(
    "change, marker",
    [
        ("max_chars", "chunking"),
        ("max_gap", "chunking"),
        ("model", "embedding"),
        ("normalize", "embedding"),
        ("projection", "projection"),
        ("cache_format", "cache format"),
    ],
)
def test_a_structural_change_forces_a_full_rebuild(
    account, scratch, cli, monkeypatch, change, marker
):
    """L–P: a moved *rule* is not a delta — the stored vectors were made under the old one."""
    build_once(account, scratch)
    write_tree(
        account,
        {
            "MSG0": {
                TALKER_A: [message(1, DAY, FIRST_MESSAGE), message(71, DAY + 120, DELTA_MESSAGE)]
            }
        },
    )

    # A rebuild embeds everything, so the embedder must be allowed to; the parametrised cases below
    # replace it only when the change *is* the embedder.
    embedder = CountingEmbedder(allow_documents=True)
    kwargs: dict = {}
    if change == "max_chars":
        kwargs["session_config"] = SessionConfig(max_chars=200)
    elif change == "max_gap":
        kwargs["session_config"] = SessionConfig(max_gap=timedelta(minutes=30))
    elif change == "model":
        embedder = CountingEmbedder(model_name="scripted-other-model", allow_documents=True)
    elif change == "normalize":
        embedder = CountingEmbedder(normalize_embeddings=False, allow_documents=True)
    elif change == "projection":
        monkeypatch.setattr(index_cache, "PROJECTION_VERSION", index_cache.PROJECTION_VERSION + 1)
    elif change == "cache_format":
        monkeypatch.setattr(
            index_cache, "CACHE_FORMAT_VERSION", index_cache.CACHE_FORMAT_VERSION + 1
        )

    session = _session(account, scratch, embedder, incremental_update=True, **kwargs)

    assert session.index.outcome != index_cache.INCREMENTAL
    assert session.index.embedded is True
    assert session.index.chunks_reused == 0
    assert session.index.chunk_count > 0
    assert marker in session.index.reason, session.index.reason
    assert_no_identity(session.index.reason)


@pytest.mark.parametrize("knobs", [{"k": 5}, {"hybrid_pool": 12}, {"workflow": "rag"}])
def test_retrieval_knobs_leave_the_vectors_reusable(account, scratch, cli, knobs):
    """``k`` selects from an index; it never builds one. It must not cost an embedding."""
    bootstrap(account, scratch, cli)
    embedder = CountingEmbedder()
    session = _session(account, scratch, embedder, incremental_update=True, **knobs)

    assert session.index.outcome == index_cache.HIT
    assert session.index.embedded is False
    assert embedder.document_calls == 0
    assert session.index.chunk_count > 0


# ---------------------------------------------------------------------------------------------
# Q–T — crash safety, and refusing rather than guessing
# ---------------------------------------------------------------------------------------------


def test_a_sync_killed_mid_export_leaves_the_previous_generation_usable(account, scratch, cli):
    _, first = bootstrap(account, scratch, cli)
    fingerprint_before = fingerprint_of(account, scratch)
    checkpoint = checkpoint_of(account, scratch)
    checkpoint_before = checkpoint.read_bytes()
    cache = cache_dir_of(account, scratch)

    cli.set(
        "MSG0", TALKER_A, [message(1, DAY, FIRST_MESSAGE), message(90, DAY + 200, DELTA_MESSAGE)]
    )
    cli.interrupt_on = f"MSG0/{TALKER_A}"
    with pytest.raises(KeyboardInterrupt):
        _synced(account, scratch, cli, CountingEmbedder())

    # Nothing moved: the tree, the checkpoint and the index are the generation they were.
    assert fingerprint_of(account, scratch) == fingerprint_before
    assert checkpoint.read_bytes() == checkpoint_before
    assert not list(cache.parent.glob(f"{cache.name}{sync_wechat.SYNC_STAGING_INFIX}*")), (
        "a killed sync left its delta staging behind"
    )
    follow_up = _session(account, scratch, CountingEmbedder())
    assert follow_up.index.outcome == index_cache.HIT
    assert follow_up.index.chunk_count == first.index.chunk_count


def test_a_leftover_sync_staging_directory_is_removed_by_the_next_sync(account, scratch, cli):
    bootstrap(account, scratch, cli)
    cache = cache_dir_of(account, scratch)
    rubble = cache.parent / f"{cache.name}{sync_wechat.SYNC_STAGING_INFIX}deadbeef"
    (rubble / "MSG0").mkdir(parents=True)
    (rubble / "MSG0" / "leftover_messages.json").write_text("[]", encoding="utf-8")
    cli.set(
        "MSG0", TALKER_A, [message(1, DAY, FIRST_MESSAGE), message(80, DAY + 400, DELTA_MESSAGE)]
    )

    session = _synced(
        account, scratch, cli, CountingEmbedder(allow_documents=True), incremental_update=True
    )

    assert not rubble.exists(), "an interrupted sync left a staging directory behind"
    assert session.sync is not None and session.sync.files_promoted == 1


def test_an_index_write_that_fails_leaves_the_checkpoint_and_the_entry_alone(
    account, scratch, cli
):
    """The one state that could lose messages: a checkpoint ahead of the index it describes."""
    from langchain_community.vectorstores import FAISS

    bootstrap(account, scratch, cli)
    cache = cache_dir_of(account, scratch)
    checkpoint = checkpoint_of(account, scratch)
    checkpoint_before = checkpoint.read_bytes()
    manifest_before = (cache / index_cache.MANIFEST_FILENAME).read_bytes()

    cli.set(
        "MSG0", TALKER_A, [message(1, DAY, FIRST_MESSAGE), message(91, DAY + 240, DELTA_MESSAGE)]
    )

    def refuse(self, folder_path, index_name="index"):
        raise OSError("disk full")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(FAISS, "save_local", refuse)
        session = _synced(
            account, scratch, cli, CountingEmbedder(allow_documents=True), incremental_update=True
        )

        # The delta could not be written, so nothing may claim that it was.
        assert checkpoint.read_bytes() == checkpoint_before
        assert (cache / index_cache.MANIFEST_FILENAME).read_bytes() == manifest_before
        assert session.index.cache_written is False
        assert session.index.chunk_count > 0
        assert DELTA_MESSAGE in "\n".join(texts_of(session, TALKER_A))

    # The tree holds the delta and the index does not: the next run rebuilds, which is the
    # conservative direction. A checkpoint that had moved would have claimed the messages were in.
    assert sync_wechat.load_sync_state(cache) is not None
    follow_up = _session(account, scratch, CountingEmbedder(allow_documents=True))
    assert DELTA_MESSAGE in "\n".join(texts_of(follow_up, TALKER_A))


def test_a_half_written_index_entry_is_never_promoted(account, scratch, cli):
    """A staging directory that never got a READY marker is rubble, not a generation."""
    from langchain_community.vectorstores import FAISS

    bootstrap(account, scratch, cli)
    cache = cache_dir_of(account, scratch)
    manifest_before = (cache / index_cache.MANIFEST_FILENAME).read_bytes()

    def die(self, folder_path, index_name="index"):
        (Path(folder_path) / index_cache.VECTORS_FILENAME).write_bytes(b"half an index")
        raise OSError("killed while saving")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(FAISS, "save_local", die)
        session = _session(
            account,
            scratch,
            CountingEmbedder(allow_documents=True),
            rebuild_index=True,
        )

        for infix in (index_cache.STAGING_INFIX, index_cache.TRASH_INFIX):
            assert not list(cache.parent.glob(f"{cache.name}{infix}*")), (
                f"a half-written entry was left behind as {infix} rubble"
            )
    # The entry that was working a moment ago is still the entry, and the run still answered.
    assert (cache / index_cache.MANIFEST_FILENAME).read_bytes() == manifest_before
    assert session.index.cache_written is False
    assert session.index.outcome == index_cache.REBUILD
    assert session.index.chunk_count > 0


def test_a_corrupt_checkpoint_is_refused_rather_than_guessed(account, scratch, cli):
    bootstrap(account, scratch, cli)
    checkpoint_of(account, scratch).write_text("{not json", encoding="utf-8")
    write_tree(
        account,
        {
            "MSG0": {
                TALKER_A: [message(1, DAY, FIRST_MESSAGE), message(92, DAY + 260, DELTA_MESSAGE)]
            }
        },
    )

    session = _session(
        account, scratch, CountingEmbedder(allow_documents=True), incremental_update=True
    )

    assert session.index.outcome != index_cache.INCREMENTAL
    assert session.index.chunks_reused == 0
    assert "checkpoint" in session.index.reason, session.index.reason
    # And it was replaced by a usable one, so the next run is not stuck refusing forever.
    assert sync_wechat.load_sync_state(cache_dir_of(account, scratch)) is not None


def test_a_checkpoint_from_another_generation_is_refused(account, scratch, cli):
    bootstrap(account, scratch, cli)
    cache = cache_dir_of(account, scratch)
    state = sync_wechat.load_sync_state(cache)
    assert state is not None
    sync_wechat.write_sync_state(
        cache, dataclasses.replace(state, index_source_fingerprint="0" * 64)
    )
    write_tree(
        account,
        {
            "MSG0": {
                TALKER_A: [message(1, DAY, FIRST_MESSAGE), message(93, DAY + 300, DELTA_MESSAGE)]
            }
        },
    )

    session = _session(
        account, scratch, CountingEmbedder(allow_documents=True), incremental_update=True
    )

    assert session.index.outcome != index_cache.INCREMENTAL
    assert session.index.chunks_reused == 0
    assert "checkpoint" in session.index.reason, session.index.reason


def test_an_export_file_that_disappeared_is_refused(account, scratch, cli):
    """History the index can account for and the tree no longer holds is not a delta."""
    bootstrap(account, scratch, cli)
    (account / "MSG0" / f"{TALKER_B}_messages.json").unlink()

    session = _session(
        account, scratch, CountingEmbedder(allow_documents=True), incremental_update=True
    )

    assert session.index.outcome != index_cache.INCREMENTAL
    assert session.index.chunks_reused == 0
    assert "gone" in session.index.reason, session.index.reason


def test_a_shard_with_no_export_is_refused(account, scratch, cli):
    """A shard that exists and has no export makes the history PARTIAL. Never advance it.

    The tree also moved, so this is not a refusal to do nothing: the incremental path was reachable
    and it declined, because advancing over a tree that is missing a shard's messages would certify
    partial history as the generation.
    """
    bootstrap(account, scratch, cli)
    write_tree(
        account,
        {
            "MSG0": {
                TALKER_A: [message(1, DAY, FIRST_MESSAGE), message(81, DAY + 420, DELTA_MESSAGE)]
            }
        },
    )
    write_multi_dir(cli.multi_dir, ["MSG7"])

    session = _session(
        account,
        scratch,
        CountingEmbedder(allow_documents=True),
        incremental_update=True,
        shard_dir=cli.multi_dir,
    )

    assert session.index.outcome != index_cache.INCREMENTAL
    assert session.index.chunks_reused == 0
    assert "shard" in session.index.reason, session.index.reason


def test_an_empty_window_is_a_no_op_rather_than_a_failure(account, scratch, cli):
    """The exporter fails with ``未找到消息``; that is "nothing new", never an error.

    ``EXPORT_FAILED`` also covers a database that could not be read, which is why the empty window is
    recognised by the code *and* the text together. Getting this wrong in the other direction — a
    failure read as "nothing new" — would silently stop syncing a conversation forever, so both
    halves are exercised here: this one must be a quiet no-op, and the one below must not be.
    """
    _, first = bootstrap(account, scratch, cli)
    fingerprint_before = fingerprint_of(account, scratch)
    # A conversation the listing knows and the exporter has nothing to emit for: real, moved, empty.
    cli.listed_only("MSG1", TALKER_C, DAY + 40 * HOUR)

    session = _synced(account, scratch, cli, CountingEmbedder(), incremental_update=True)

    assert session.sync is not None
    assert session.sync.exports_run == 1 and session.sync.empty_windows == 1
    assert session.sync.failures == ()
    assert session.sync.ok is True
    assert session.sync.files_promoted == 0
    assert session.sync.outcome == sync_wechat.SYNC_NO_CHANGE
    assert session.index.outcome == index_cache.HIT
    assert session.index.chunk_count == first.index.chunk_count
    assert fingerprint_of(account, scratch) == fingerprint_before
    assert not (account / "MSG1" / f"{TALKER_C}_messages.json").exists(), (
        "an empty window produced an export file"
    )

    # And it was recorded, so the next sync does not ask for the same empty window again.
    state = sync_wechat.load_sync_state(cache_dir_of(account, scratch))
    assert state is not None and state.last_timestamp("MSG1", TALKER_C) == DAY + 40 * HOUR


def test_an_unreadable_shard_is_refused_by_the_sync(account, scratch, cli):
    """A shard that cannot be listed cannot be proven unchanged, so nothing may be written."""
    bootstrap(account, scratch, cli)
    fingerprint_before = fingerprint_of(account, scratch)
    checkpoint_before = checkpoint_of(account, scratch).read_bytes()

    def explode(argv, env):
        raise FileNotFoundError("no such shard")

    failed = _session(
        account,
        scratch,
        CountingEmbedder(),
        sync=True,
        multi_dir=cli.multi_dir,
        sync_runner=explode,
    )

    assert failed.sync is not None and not failed.sync.ok
    assert failed.sync.outcome == sync_wechat.SYNC_UNSAFE
    assert "listed" in failed.sync.reason
    assert fingerprint_of(account, scratch) == fingerprint_before
    assert checkpoint_of(account, scratch).read_bytes() == checkpoint_before
    assert failed.index.outcome == index_cache.HIT
    assert_no_identity(" ".join(failed.sync.lines()))
    assert_no_identity(json.dumps(failed.sync.as_dict()))


def test_a_window_that_cannot_be_read_is_reported_not_hidden(account, scratch, cli):
    """One window that could not be read must not read as success, and must not break the run."""
    _, first = bootstrap(account, scratch, cli)
    cli.fail_for = {TALKER_A}
    cli.set(
        "MSG0", TALKER_A, [message(1, DAY, FIRST_MESSAGE), message(94, DAY + 320, DELTA_MESSAGE)]
    )

    session = _synced(account, scratch, cli, CountingEmbedder(), incremental_update=True)

    assert session.sync is not None and session.sync.failures
    assert session.sync.ok is False
    assert "failure:" in "\n".join(session.sync.lines())
    # Nothing was promoted for it, so the generation is exactly the one that was already there.
    assert session.index.outcome == index_cache.HIT
    assert session.index.chunk_count == first.index.chunk_count
    assert session.brain.vector_db.index.ntotal > 0
    assert_no_identity(" ".join(session.sync.lines()))


def test_a_narrowed_tree_is_not_silently_widened(account, scratch, cli):
    """A tree exported with a filter holds a subset on purpose. A sync may never widen it.

    The manifest is written *before* the account is indexed, so nothing about it moves later: what is
    under test is the sync's decision, not a rebuild. A full export widens a narrowed tree; a sync
    must not, because pulling in conversations the user excluded would change what the account holds
    and index history the export deliberately left out.
    """
    (account / ACCOUNT_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema": "privrecall-account-export/v1",
                "detected_shards": ["MSG0", "MSG1"],
                "exported_shards": ["MSG0", "MSG1"],
                "filtered_conversations": 1,
            }
        ),
        encoding="utf-8",
    )
    build, _ = bootstrap(account, scratch, cli)
    # A conversation that exists in WeChat and was deliberately left out of the tree.
    cli.set("MSG1", TALKER_C, [message(51, DAY + 7 * HOUR, DELTA_MESSAGE, sender=TALKER_C)])

    session = _synced(account, scratch, cli)

    assert session.sync is not None and session.sync.conversations_skipped == 1
    assert not (account / "MSG1" / f"{TALKER_C}_messages.json").exists(), (
        "a narrowed tree was silently widened by a sync"
    )
    assert all(call[1] != TALKER_C for call in cli.calls), "an excluded conversation was exported"
    assert session.index.outcome == index_cache.HIT
    assert session.index.chunk_count == build.index.chunk_count


# ---------------------------------------------------------------------------------------------
# The golden test: A + delta must equal B
# ---------------------------------------------------------------------------------------------


def test_the_incremental_result_is_exactly_a_clean_rebuild(account, scratch, cli):
    """The phase in one comparison, against the only oracle that cannot be argued with.

    A snapshot is indexed and synced, a delta is applied to the tree, and the generation is
    *advanced*. The same tree is then built from scratch. If the two differ in one document, one
    metadata value, one vector, the chunk count, or a single retrieval order, then "incremental" is
    a different index that merely answers similarly — which is exactly what this phase must not ship.
    """
    _, first = bootstrap(account, scratch, cli)
    snapshot_chunks = sum(len(texts_of(first, cid)) for cid in (TALKER_A, TALKER_B, GROUP))
    assert snapshot_chunks == 5

    # The delta, in one sync: a message inside a tail's gap on the multi-shard conversation (which
    # *changes* a chunk), a relabelled speaker and a message beyond the gap (which makes one appear),
    # a brand-new conversation, and a brand-new shard.
    cli.set(
        "MSG0",
        TALKER_A,
        [message(1, DAY, FIRST_MESSAGE), message(95, DAY + 30 * HOUR + 120, DELTA_MESSAGE)],
    )
    cli.set(
        "MSG1",
        GROUP,
        [
            message(30, DAY + 2 * HOUR, GROUP_MESSAGE, sender=MEMBER, is_send=0),
            message(31, DAY + 30 * HOUR, LATER_GROUP_MESSAGE, sender=MEMBER, is_send=0),
            message(
                32, DAY + 30 * HOUR + 60, DELTA_MESSAGE, sender=MEMBER, is_send=0, display=MEMBER_NAME
            ),
            message(33, DAY + 40 * HOUR, THIRD_MESSAGE, sender=MEMBER, is_send=0, display=MEMBER_NAME),
        ],
    )
    cli.set("MSG2", TALKER_C, [message(50, DAY + 5 * HOUR, DELTA_MESSAGE, sender=TALKER_C)])
    write_multi_dir(cli.multi_dir, ["MSG2"])

    advanced = _synced(
        account, scratch, cli, CountingEmbedder(allow_documents=True), incremental_update=True
    )

    assert advanced.index.outcome == index_cache.INCREMENTAL
    assert advanced.index.chunks_reused > 0 and advanced.index.chunks_embedded > 0
    assert advanced.index.chunks_embedded < advanced.index.chunk_count, (
        "the delta re-embedded the whole account"
    )

    clean = build_once(account, scratch, index_dir=scratch / "clean")

    # events and chunks: count, text, metadata, order. The count is pinned so the comparison below
    # cannot pass by comparing two empty sequences.
    assert advanced.index.chunk_count == clean.index.chunk_count == 7
    assert chunks_of(advanced) == chunks_of(clean), "a chunk, its text or its metadata differs"
    # vectors: read back from the index one by one, not merely counted
    assert vectors_of(advanced) == vectors_of(clean), "an embedded vector differs"
    assert len(vectors_of(advanced)) == 7
    # retrieval: order and citation metadata, through the product's own retriever
    for question in ("数据库最后用了哪个", "周六下午三点", DELTA_MESSAGE, MEMBER_NAME, GROUP_MESSAGE):
        shared = retrieval(advanced, question)
        assert shared and shared == retrieval(clean, question), (
            f"retrieval differs for {question!r}"
        )


def test_the_golden_comparison_notices_a_shard_local_rebuild(account, scratch, cli):
    """A guard on the guard: the oracle is sensitive to the implementation it exists to catch.

    If re-rendering only the shard whose file moved produced the same index, the golden test above
    would prove nothing. So that wrong index is built here on purpose — by indexing a tree that is
    missing the conversation's other shard, which is exactly what a shard-local pass would see — and
    the comparison is shown to fail.
    """
    bootstrap(account, scratch, cli)
    cli.set(
        "MSG0",
        TALKER_A,
        [message(1, DAY, FIRST_MESSAGE), message(96, DAY + 30 * HOUR + 120, DELTA_MESSAGE)],
    )
    write_tree(account, {"MSG0": {TALKER_A: cli.tree["MSG0"][TALKER_A]}})

    clean = build_once(account, scratch, index_dir=scratch / "clean")
    assert len(texts_of(clean, TALKER_A)) == 2
    assert DELTA_MESSAGE in texts_of(clean, TALKER_A)[-1]

    shard_local_root = scratch / "shard-local"
    shutil.copytree(account, shard_local_root)
    (shard_local_root / "MSG1" / f"{TALKER_A}_messages.json").unlink()
    shard_local = _session(
        shard_local_root,
        scratch,
        CountingEmbedder(allow_documents=True),
        index_dir=scratch / "shard-local-index",
    )

    assert texts_of(shard_local, TALKER_A) != texts_of(clean, TALKER_A), (
        "the fixture cannot tell a whole-conversation rebuild from a shard-local one"
    )
    assert chunks_of(shard_local) != chunks_of(clean)


# ---------------------------------------------------------------------------------------------
# Where the checkpoint lives, and what may never be printed
# ---------------------------------------------------------------------------------------------


def test_the_checkpoint_sits_beside_the_index_and_never_inside_the_tree(account, scratch, cli):
    bootstrap(account, scratch, cli)
    cache = cache_dir_of(account, scratch)
    checkpoint = checkpoint_of(account, scratch)

    assert checkpoint.parent == cache.parent
    assert checkpoint.name.endswith(sync_wechat.SYNC_STATE_SUFFIX)
    assert not list(account.rglob(f"*{sync_wechat.SYNC_STATE_SUFFIX}")), (
        "the checkpoint was written into the export tree"
    )
    assert cache.is_dir(), "the index entry and its checkpoint live together"


def test_the_privacy_promise_holds_around_the_new_paths(account, scratch, cli, fake_home):
    """Counts, hash prefixes and timings only: the state file may hold identity, the log may not."""
    temp_root = Path(tempfile.gettempdir())
    profiles_before = set(temp_root.glob("privrecall_export_*"))
    bootstrap(account, scratch, cli)
    cli.set(
        "MSG1",
        GROUP,
        [
            message(30, DAY + 2 * HOUR, GROUP_MESSAGE, sender=MEMBER, is_send=0),
            message(31, DAY + 30 * HOUR, LATER_GROUP_MESSAGE, sender=MEMBER, is_send=0),
            message(
                32, DAY + 30 * HOUR + 60, DELTA_MESSAGE, sender=MEMBER, is_send=0, display=MEMBER_NAME
            ),
        ],
    )
    session = _synced(
        account, scratch, cli, CountingEmbedder(allow_documents=True), incremental_update=True
    )

    assert session.sync is not None
    assert_no_identity("\n".join(session.sync.lines()))
    assert_no_identity(json.dumps(session.sync.as_dict(), ensure_ascii=False))
    assert_no_identity(session.index.reason)
    assert_no_identity(
        " ".join(
            f"{name}={value}"
            for name, value in session.index.__dict__.items()
            if isinstance(value, (str, int))
        )
    )

    # The user's real config was not touched: the scratch profile is the only config that was
    # written, and it was deleted by the same call that made it — no key material left on disk.
    assert json.loads((fake_home / ".weflow-cli" / "config.json").read_text(encoding="utf-8")) == (
        SYNTHETIC_CONFIG
    )
    assert not list(SCRATCH_ROOT.rglob("privrecall_export_*"))
    assert set(temp_root.glob("privrecall_export_*")) == profiles_before


def test_the_new_seam_is_still_the_only_place_that_decides_the_cache():
    """Structure, not behaviour: the incremental path may not grow a second decision."""
    whole = (BASE_DIR / "recall.py").read_text(encoding="utf-8")
    assert whole.count("index_cache.inspect(") == 1
    assert whole.count("index_cache.load_store(") == 1
    assert whole.count("index_cache.save_index(") == 1
    seam = inspect.getsource(recall.build_account_session)
    for piece in (
        "sync_wechat.sync_account_source(",
        "_classify_change(",
        "incremental_index.update_account_index(",
    ):
        assert seam.count(piece) == 1, f"the seam does not hold exactly one {piece}"
    # The verdict itself has exactly one call site in the whole module, and it gathers facts only —
    # the rule lives in `incremental_index`, where it can be read (and tested) without an import.
    assert whole.count("incremental_index.classify_source_change(") == 1
    classifier = inspect.getsource(recall._classify_change)
    assert "classify_source_change(" in classifier

    # And the sync is opt-in: the offline front ends never reach for a local database.
    for name in ("webapp/app.py", "web.py", "real_eval.py", "audit_account.py"):
        source = (BASE_DIR / name).read_text(encoding="utf-8")
        for piece in ("sync_wechat", "multi_dir", "sync_runner"):
            assert piece not in source, f"{name} grew a WeChat dependency: {piece}"
