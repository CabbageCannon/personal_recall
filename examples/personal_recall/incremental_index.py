"""Advance an index by a delta: re-render what changed, keep every vector that did not.

Phase 21A's cache answers one question — "is this the *same* tree?" — and its answer to anything else
is a full rebuild. This module answers the question in between: the tree moved, *which conversation
moved*, and *which of that conversation's chunks are actually different*.

Three things make that answerable, and each of them is a rule rather than an optimisation:

1. **A conversation is the unit of work, never a file.** A conversation's messages live in one or
   more of the ``MSG*.db`` shards, and an export only ever reads the shard its config names, so a
   file is a *piece* of a conversation. Re-rendering one shard's piece and leaving the others alone
   would produce a merge over a subset, which is not the merge a clean rebuild produces — the
   ``serverId`` dedupe would see fewer copies, the sort would have fewer rows, and the sender
   resolution would label from less evidence. So an affected conversation is re-loaded from **every**
   shard it appears in, then merged, sorted, deduped, labelled and sessionised exactly as a full
   import would. The golden test compares the incremental result against a clean rebuild of the same
   tree, and anything less than whole-conversation re-rendering fails it.

2. **Whole-conversation re-rendering is also the only way to know what changed.** A chunk is not
   "the new message appended". A message that lands inside ``max_gap`` of a session's tail makes the
   tail chunk *different* rather than making a new chunk; a new ``senderDisplay`` retroactively
   relabels that speaker's earlier messages, so chunks far behind the delta change text; a
   ``max_chars`` boundary can shift every following chunk of that conversation. None of that is
   visible in the delta, which is why the decision to reuse a vector is made by comparing re-rendered
   chunks, never by looking at what arrived.

3. **A chunk is unchanged when its identity and its projected content agree.** Identity is the
   ``memory_chunk_id`` (conversation + session ordinal) — a chunk keeps its vector only if the chunk
   *and* its text are the same. Its **metadata is not part of that test and is recomputed for every
   document**: ``sessions_total`` and ``chunk_index`` are account-global numbers, so one new chunk in
   one conversation shifts ``chunk_index`` for every chunk after it. Metadata is cheap; the embedding
   is the expensive part, and it is the only part this module reuses.

What is reused, and how
-----------------------

The stored generation is a FAISS store: ``index_to_docstore_id[i]`` names the document at vector
``i``, and ``index.reconstruct_n(i, 1)`` returns that vector. Reading the store back, the old
document sequence and its vectors give a map from ``(memory_chunk_id, conversation_id, sha256(text))``
to the vector that was produced for it. Rebuilding the store for the new document list then costs one
``embed_documents`` call over the documents whose key is *not* in that map, and nothing else:
``FAISS.from_embeddings`` takes precomputed vectors and never calls the embedder. That is the whole
claim, and it is measured rather than asserted — the tests drive a counting embedder that refuses to
be called with unchanged text.

Why the metadata still has to be rebuilt
---------------------------------------

Because parity is the requirement. A rebuild of the same tree must produce the same documents, in
the same order, with the same metadata and the same vectors — otherwise "incremental" is a different
index that happens to answer similarly, and the golden test compares retrieval order, not prose.

Where the crash-safety ordering lives
-------------------------------------

This module is the *second* half of it, and it is invoked only after the source has been promoted by
``sync_wechat``. Its own order is: import the affected conversations (read-only) → assemble the new
document list (read-only) → embed → build the store → verify it against the manifest → write it to a
staging entry and promote it with a rename. Nothing is written anywhere until the store is complete,
and the checkpoint is written **after** that, by the caller. A kill before the promotion leaves the
previous entry untouched and loadable; a kill after it leaves an index that is *ahead* of the
checkpoint, which the next run repairs by rebuilding the diff from the older base — never the other
way round, which would claim work that had not happened.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import index_cache
import sync_wechat
from index_cache import IndexManifest
from memory.account import (
    build_account_sessions,
    crossed_conversation_chunks,
    import_account,
    load_account_directory,
)
from memory.sessions import SessionConfig
from sync_wechat import InventoryDiff, SyncState

__all__ = [
    "NO_CHANGE",
    "SAFE_INCREMENTAL_CHANGE",
    "UNSAFE_CHANGE",
    "Classification",
    "IncrementalResult",
    "classify_source_change",
    "stored_documents",
    "update_account_index",
]

#: The three things a source change can be. ``NO_CHANGE`` is Phase 21A's cache hit; the middle one
#: is the only case that may advance in place; the third is everything that must not, and it always
#: ends in a full rebuild (or in the caller's explicit one) with the reason spelled out.
NO_CHANGE = "NO_CHANGE"
SAFE_INCREMENTAL_CHANGE = "SAFE_INCREMENTAL_CHANGE"
UNSAFE_CHANGE = "UNSAFE_CHANGE"

#: The metadata keys ``memory.processor.session_documents`` writes. An affected conversation's
#: documents are re-projected from its *chunks*, but a conversation that was not affected has no
#: chunks to re-project — its stored documents are carried forward as they are. That carry-forward is
#: only sound while the stored projection is the one this module knows how to complete, so the key set
#: is checked on every stored document rather than assumed from the schema version alone.
_PROJECTION_KEYS = frozenset(
    {
        "memory_chunk_id",
        "conversation_id",
        "start_time",
        "end_time",
        "participants",
        "n_events",
        "sessions_total",
        "skipped_source_lines",
    }
)

#: Added by ``recall.build_brain_from_chunks`` on top of the projection. ``original_file_name`` is
#: what the framework's document prompt needs; ``chunk_index`` numbers the whole account.
_DOCUMENT_KEYS = _PROJECTION_KEYS | {"chunk_index", "original_file_name"}


@dataclass(frozen=True)
class Classification:
    """Whether a source change may be advanced in place, and on what evidence."""

    outcome: str
    reason: str = ""
    #: Conversations whose export files moved or appeared — the unit of work.
    affected: tuple[str, ...] = ()
    diff: InventoryDiff | None = None

    @property
    def is_incremental(self) -> bool:
        return self.outcome == SAFE_INCREMENTAL_CHANGE

    @property
    def is_no_change(self) -> bool:
        return self.outcome == NO_CHANGE

    def lines(self) -> list[str]:
        if self.outcome == SAFE_INCREMENTAL_CHANGE:
            return [
                f"source         : changed; {len(self.affected)} conversation(s) moved, "
                f"{self.diff.touched_files if self.diff else 0} export file(s) added or rewritten"
            ]
        if self.outcome == NO_CHANGE:
            return ["source         : identical to the generation in the index"]
        return [f"source         : change is NOT safe to advance in place ({self.reason})"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "reason": self.reason,
            "conversations_affected": len(self.affected),
            "files_added": len(self.diff.added) if self.diff else 0,
            "files_changed": len(self.diff.changed) if self.diff else 0,
            "files_removed": len(self.diff.removed) if self.diff else 0,
        }


def classify_source_change(
    *,
    inspection: index_cache.Inspection,
    previous_inventory: Sequence[index_cache.SourceFileEntry] | None = None,
    current_inventory: Sequence[index_cache.SourceFileEntry] | None = None,
    state: SyncState | None = None,
    state_error: str = "",
    missing_shards: Sequence[str] = (),
    unsafe_reasons: Sequence[str] = (),
) -> Classification:
    """Decide whether a moved source may be advanced in place. Never raises for a refusal.

    The order is from the most decisive to the most expensive, and every step refuses rather than
    guesses:

    1. **No stored generation** → nothing to advance. A miss, a rebuild request or a load failure is
       the cold path's business, not this function's.
    2. **The reason for invalidity must be the source and nothing else.** ``inspect`` returns exactly
       one reason for a moved tree (``index_cache.SOURCE_CHANGED``) and a *different* one for every
       structural change — a chunking or embedding configuration change, a moved document projection,
       a cache format bump, a damaged artifact, a READY marker that disagrees with its manifest. Those
       are not deltas: the stored vectors were produced under different rules, so there is nothing to
       reuse and a full rebuild is the only correct answer.
    3. **The checkpoint must describe the generation the index was built from.** A checkpoint that
       belongs to a different generation than the stored index means the base of the diff is unknown,
       and an unknown base is precisely what this path must not invent. (An *older* checkpoint would
       still be conservative — it can only over-report what moved — but "conservative" is an argument
       for the rebuild, not a licence to guess.)
    4. **The tree must be describable and complete.** A walk that failed, an export whose conversation
       cannot be established, a file that disappeared: each is a reason to refuse. A file that
       disappeared is the sharpest of them, because the index can no longer account for history the
       user had.
    5. **The account must not become partial.** A shard that exists and has no export in the tree means
       the history is incomplete; an incremental update over an incomplete tree would rebuild part of
       it and certify the result. ``missing_shards`` is that list, and ``unsafe_reasons`` is where a
       caller puts anything else it learned (a shard that could not be read, a checkpoint that would
       not load) so one function owns the verdict rather than each caller inventing one.
    """
    if inspection.outcome == index_cache.HIT:
        return Classification(NO_CHANGE, "the source is identical to the indexed generation")

    if not inspection.is_hit and inspection.outcome != index_cache.INVALID:
        return Classification(
            UNSAFE_CHANGE,
            f"there is no stored generation to advance ({inspection.outcome.lower()})",
        )

    if inspection.reason != index_cache.SOURCE_CHANGED:
        # Every other invalid reason is a *structural* change: the vectors in the entry were made
        # under different rules and reusing them would produce an index no rebuild would ever make.
        return Classification(UNSAFE_CHANGE, inspection.reason or "the stored entry is unusable")

    if inspection.manifest is None:
        return Classification(UNSAFE_CHANGE, "the stored generation's manifest is unavailable")

    if state_error:
        return Classification(UNSAFE_CHANGE, f"the checkpoint is unusable ({state_error})")
    if state is None:
        return Classification(
            UNSAFE_CHANGE,
            "no checkpoint: this tree has never been synced, so the base of the delta is unknown",
        )
    if state.index_source_fingerprint != inspection.manifest.source_fingerprint:
        return Classification(
            UNSAFE_CHANGE,
            "the checkpoint describes a different generation than the index was built from",
        )

    if previous_inventory is None or current_inventory is None:
        return Classification(UNSAFE_CHANGE, "the export tree could not be walked")

    for reason in unsafe_reasons:
        return Classification(UNSAFE_CHANGE, reason)

    diff = sync_wechat.diff_inventories(previous_inventory, current_inventory)
    if diff.ambiguous:
        return Classification(
            UNSAFE_CHANGE,
            f"{len(diff.ambiguous)} export file(s) cannot be attributed to a conversation",
            diff=diff,
        )
    if diff.removed:
        return Classification(
            UNSAFE_CHANGE,
            f"{len(diff.removed)} export file(s) that the index was built from are gone",
            diff=diff,
        )
    if missing_shards:
        return Classification(
            UNSAFE_CHANGE,
            f"{len(missing_shards)} message shard(s) exist and have no export in this tree",
            diff=diff,
        )
    if not diff.affected:
        # Nothing this path can act on: files moved but no conversation did (a rewritten listing, a
        # touched manifest). Reported rather than silently rebuilt, because it is still a change.
        return Classification(
            UNSAFE_CHANGE, "the change is not attributable to a conversation", diff=diff
        )

    return Classification(
        SAFE_INCREMENTAL_CHANGE,
        f"{len(diff.affected)} conversation(s) moved",
        affected=diff.affected,
        diff=diff,
    )


# ---------------------------------------------------------------------------------------------
# What the stored generation holds
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredGeneration:
    """The stored documents in index order, with the vector that belongs to each one."""

    documents: tuple[Any, ...]
    vectors: Mapping[str, Any]
    #: ``(memory_chunk_id, conversation_id) -> digest of the text that was embedded`` — the identity
    #: half of the reuse key, kept separately so a same-id-different-text chunk can be detected
    #: without trusting the digest map's own keys.
    digests: Mapping[tuple[str, str], str] = field(default_factory=dict)


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stored_documents(vector_store: Any) -> tuple[Any, ...]:
    """The stored documents, in the order the vectors are in. Read-only.

    ``index_to_docstore_id`` is the authority on the order — the docstore is a bag keyed by a random
    id, so iterating it would compare a document sequence against a *shuffled* vector sequence and
    quietly pair every chunk with somebody else's embedding. The ids themselves are deliberately
    never used as an identity: they are ``uuid4`` and mean nothing.
    """
    mapping = getattr(vector_store, "index_to_docstore_id", None)
    docstore = getattr(vector_store, "docstore", None)
    count = int(getattr(getattr(vector_store, "index", None), "ntotal", -1))
    if not isinstance(mapping, Mapping) or count < 0 or len(mapping) != count:
        raise ValueError("the stored index does not expose a complete document mapping")
    documents = []
    for position in range(count):
        key = mapping.get(position)
        if key is None:
            raise ValueError(f"the stored index has no document at position {position}")
        document = _search(docstore, key)
        if document is None:
            raise ValueError(f"the stored index has no document for position {position}")
        documents.append(document)
    return tuple(documents)


def _search(docstore: Any, key: Any) -> Any:
    try:
        return docstore.search(key)
    except KeyError:
        return None


def stored_vectors(vector_store: Any) -> dict[str, Any]:
    """``(memory_chunk_id, conversation_id, text digest) -> the vector that was embedded for it``.

    The one place a vector is read back out of the store. ``reconstruct_n(i, 1)`` returns exactly the
    vector at position ``i`` — this is a ``IndexFlatL2``, which stores its vectors, so the value is
    the original float32 that was added and not a re-computation from anything.
    """
    index = getattr(vector_store, "index", None)
    if index is None or not hasattr(index, "reconstruct_n"):
        raise ValueError("the stored index cannot be read back vector by vector")
    reuse: dict[str, Any] = {}
    for position, document in enumerate(stored_documents(vector_store)):
        metadata = document.metadata or {}
        chunk_id = str(metadata.get("memory_chunk_id") or "")
        conversation_id = str(metadata.get("conversation_id") or "")
        if not chunk_id:
            raise ValueError("a stored document has no chunk identity")
        reuse[_reuse_key(chunk_id, conversation_id, document.page_content)] = index.reconstruct_n(
            position, 1
        )[0]
    return reuse


def _reuse_key(chunk_id: str, conversation_id: str, text: str) -> str:
    """The identity of an embedded chunk: its id, its owner and its exact projected text.

    All three, because all three decide whether the vector still means what it meant. An id alone
    would reuse a vector for a chunk that was re-rendered into different text; the text alone would
    reuse it across two conversations that happen to share a sentence, which is exactly the
    separation this product is built on.
    """
    return f"{chunk_id}\x00{conversation_id}\x00{text_digest(text)}"


# ---------------------------------------------------------------------------------------------
# The update
# ---------------------------------------------------------------------------------------------


@dataclass
class IncrementalResult:
    """What one incremental update did, or why it refused to do it."""

    ok: bool = True
    outcome: str = SAFE_INCREMENTAL_CHANGE
    reason: str = ""
    vector_store: Any = None
    manifest: IndexManifest | None = None
    chunk_count: int = 0
    vector_dimension: int = 0
    chunks_reused: int = 0
    chunks_embedded: int = 0
    conversations_reimported: int = 0
    conversations_dropped: int = 0
    cache_written: bool = False
    import_ms: int = 0
    embed_ms: int = 0
    save_ms: int = 0
    total_ms: int = 0

    @property
    def reused_for_nothing(self) -> bool:
        """True when the delta turned out to be a no-op for the index."""
        return self.ok and self.chunks_embedded == 0

    def lines(self) -> list[str]:
        if not self.ok:
            return [f"index          : NOT advanced incrementally ({self.reason})"]
        return [
            f"index          : {self.chunk_count} chunks ({self.chunks_reused} reused, "
            f"{self.chunks_embedded} embedded)",
            f"conversations  : {self.conversations_reimported} re-imported"
            + (f", {self.conversations_dropped} dropped" if self.conversations_dropped else ""),
            f"cache          : {'written' if self.cache_written else 'NOT written'}",
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "outcome": self.outcome,
            "reason": self.reason,
            "chunk_count": self.chunk_count,
            "vector_dimension": self.vector_dimension,
            "chunks_reused": self.chunks_reused,
            "chunks_embedded": self.chunks_embedded,
            "conversations_reimported": self.conversations_reimported,
            "conversations_dropped": self.conversations_dropped,
            "cache_written": self.cache_written,
            "import_ms": self.import_ms,
            "embed_ms": self.embed_ms,
            "save_ms": self.save_ms,
            "total_ms": self.total_ms,
        }


def reimport_conversations(
    account_dir: Path,
    conversation_ids: Sequence[str],
    *,
    shard_dir: Path | None = None,
    session_config: SessionConfig,
) -> list[Any]:
    """Re-render whole conversations from **every** shard they appear in.

    The hard requirement, in code: the exports are selected by conversation (not by the shard whose
    file moved), passed to the same importer the full build uses, and sessionised by the same
    session builder. Nothing here is a shortcut — the only thing that changed is *which*
    conversations the pipeline sees.

    A conversation with no export file left is not an error here: it simply produces no chunk, and
    the caller drops it, exactly as a full rebuild would.
    """
    layout = load_account_directory(account_dir, shard_dir=shard_dir)
    wanted = {str(cid) for cid in conversation_ids}
    selected = [export for export in layout.exports if export.conversation_id in wanted]
    events_by_conversation, _ = import_account(selected, shards_detected=())
    chunks = build_account_sessions(dict(events_by_conversation), session_config)
    crossed = crossed_conversation_chunks(chunks, events_by_conversation)
    if crossed:
        raise AssertionError(
            "conversation boundary crossed: chunk(s) "
            f"{list(crossed)} mix events from more than one conversation"
        )
    return chunks


def update_account_index(
    account_dir: Path,
    *,
    cache_dir: Path,
    stored_manifest: IndexManifest,
    vector_store: Any,
    classification: Classification,
    session_config: SessionConfig,
    embedder: Any,
    shard_dir: Path | None = None,
    origin: str = "",
) -> IncrementalResult:
    """Advance the stored index by the classified delta. Never raises for a refusal.

    Order, and why: import (read-only) → assemble (read-only) → embed → build → verify → promote.
    The store is written to a staging entry and renamed into place by
    :func:`index_cache.save_index`, so a failure at any point before that rename leaves the previous
    entry exactly as it was. The caller writes the checkpoint afterwards.

    ``origin`` must be the same ``original_file_name`` the full build stamps on every document
    (``account:<name>``), because a fresh document carrying a different one would be a document no
    rebuild would ever produce.
    """
    started = time.perf_counter()
    affected = tuple(classification.affected)
    if not classification.is_incremental or not affected:
        return IncrementalResult(
            ok=False,
            outcome=UNSAFE_CHANGE,
            reason="no conversation was classified as moved",
        )
    origin = origin or f"account:{Path(account_dir).name}"

    try:
        previous = stored_documents(vector_store)
    except ValueError as exc:
        return IncrementalResult(ok=False, outcome=UNSAFE_CHANGE, reason=str(exc))
    broken = _projection_problem(previous)
    if broken:
        return IncrementalResult(ok=False, outcome=UNSAFE_CHANGE, reason=broken)

    # --- re-render the affected conversations, from every shard they appear in ------------------
    import_started = time.perf_counter()
    try:
        chunks = reimport_conversations(
            account_dir, affected, shard_dir=shard_dir, session_config=session_config
        )
    except (OSError, ValueError) as exc:
        return IncrementalResult(
            ok=False,
            outcome=UNSAFE_CHANGE,
            reason=f"an affected conversation could not be re-imported ({type(exc).__name__})",
        )
    fresh_documents = _project_documents(chunks, origin=origin)
    fresh_by_conversation = _group_by_conversation(fresh_documents)
    import_ms = int((time.perf_counter() - import_started) * 1000)

    # --- assemble the next generation's document list in the canonical order --------------------
    carried, dropped = _carry_forward(previous, set(fresh_by_conversation), set(affected))
    documents = _assemble(carried, fresh_by_conversation)
    total = len(documents)
    if not total:
        return IncrementalResult(
            ok=False, outcome=UNSAFE_CHANGE, reason="the advanced generation would hold no chunk"
        )
    for position, document in enumerate(documents, start=1):
        document.metadata["sessions_total"] = total
        document.metadata["chunk_index"] = position

    # --- reuse every vector whose chunk did not change; embed the rest ---------------------------
    try:
        reuse = stored_vectors(vector_store)
    except ValueError as exc:
        return IncrementalResult(ok=False, outcome=UNSAFE_CHANGE, reason=str(exc))

    keys = [
        _reuse_key(
            str(document.metadata["memory_chunk_id"]),
            str(document.metadata["conversation_id"]),
            document.page_content,
        )
        for document in documents
    ]
    missing = [position for position, key in enumerate(keys) if key not in reuse]
    embed_started = time.perf_counter()
    fresh_vectors: list[Any] = []
    if missing:
        texts = [documents[position].page_content for position in missing]
        try:
            fresh_vectors = list(embedder.embed_documents(texts))
        except Exception as exc:  # noqa: BLE001 - the embedder is a model; any failure is a refusal
            return IncrementalResult(
                ok=False,
                outcome=UNSAFE_CHANGE,
                reason=f"the changed chunks could not be embedded ({type(exc).__name__})",
            )
        if len(fresh_vectors) != len(missing):
            return IncrementalResult(
                ok=False,
                outcome=UNSAFE_CHANGE,
                reason="the embedder returned a different number of vectors than chunks",
            )
    embed_ms = int((time.perf_counter() - embed_started) * 1000)

    vectors: list[Any] = []
    fresh_position = 0
    for position, key in enumerate(keys):
        if key in reuse:
            vectors.append(reuse[key])
        else:
            vectors.append(fresh_vectors[fresh_position])
            fresh_position += 1

    # --- build, verify against the manifest, then promote ---------------------------------------
    try:
        from langchain_community.vectorstores import FAISS

        store = FAISS.from_embeddings(
            list(zip([document.page_content for document in documents], vectors)),
            embedder,
            metadatas=[dict(document.metadata) for document in documents],
        )
    except Exception as exc:  # noqa: BLE001 - a store that cannot be assembled is a refusal
        return IncrementalResult(
            ok=False,
            outcome=UNSAFE_CHANGE,
            reason=f"the advanced index could not be assembled ({type(exc).__name__})",
        )

    dimension = int(getattr(getattr(store, "index", None), "d", 0))
    manifest = IndexManifest(
        cache_format_version=index_cache.CACHE_FORMAT_VERSION,
        schema_version=index_cache.PROJECTION_VERSION,
        source_fingerprint=index_cache.source_fingerprint(
            account_dir, exclude=(Path(cache_dir).parent,)
        ),
        chunking_fingerprint=index_cache.chunking_fingerprint(session_config),
        embedding_fingerprint=index_cache.embedding_fingerprint(
            embedder, dimension=stored_manifest.vector_dimension or dimension
        ),
        chunk_count=total,
        vector_dimension=dimension,
        created_at=index_cache.utc_now(),
        report=_merged_aggregate(stored_manifest, documents),
    )
    problem = index_cache.verify_store(store, manifest)
    if problem:
        return IncrementalResult(
            ok=False,
            outcome=UNSAFE_CHANGE,
            reason=f"the advanced index did not verify ({problem})",
        )

    result = IncrementalResult(
        chunk_count=total,
        vector_dimension=dimension,
        chunks_reused=total - len(missing),
        chunks_embedded=len(missing),
        conversations_reimported=len(fresh_by_conversation),
        conversations_dropped=dropped,
        manifest=manifest,
        vector_store=store,
    )
    save_started = time.perf_counter()
    try:
        index_cache.save_index(Path(cache_dir), vector_store=store, manifest=manifest)
    except (OSError, ValueError) as exc:
        # The index was built and is answerable; losing the cache must not cost the user their
        # answer, and must not leave a half-written entry. Reported, not swallowed.
        result.ok = False
        result.outcome = UNSAFE_CHANGE
        result.reason = f"the advanced index could not be written ({type(exc).__name__})"
        result.cache_written = False
    else:
        result.cache_written = True
    result.save_ms = int((time.perf_counter() - save_started) * 1000)
    result.import_ms = import_ms
    result.embed_ms = embed_ms
    result.total_ms = int((time.perf_counter() - started) * 1000)
    return result


# ---------------------------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------------------------


def _projection_problem(documents: Sequence[Any]) -> str:
    """Verify the stored documents carry the projection this module knows how to complete.

    The schema version already says they should; this says they do. A document missing a metadata
    key would be carried forward with metadata a rebuild would not produce, and the parity the
    golden test demands is exactly the property that would break.
    """
    for document in documents:
        metadata = getattr(document, "metadata", None)
        if not isinstance(metadata, Mapping):
            return "a stored document has no metadata"
        keys = set(metadata)
        if keys != _DOCUMENT_KEYS:
            missing = sorted(_DOCUMENT_KEYS - keys)
            extra = sorted(keys - _DOCUMENT_KEYS)
            detail = []
            if missing:
                detail.append(f"missing {', '.join(missing)}")
            if extra:
                detail.append(f"unexpected {', '.join(extra)}")
            return f"a stored document's metadata does not match the projection ({'; '.join(detail)})"
    return ""


def _project_documents(chunks: Sequence[Any], origin: str = "") -> list[Any]:
    """Project fresh chunks the way ``recall.build_brain_from_chunks`` does.

    Deliberately the same two steps in the same order: ``session_documents`` for the projection, then
    ``chunk_index`` and ``original_file_name``. ``sessions_total`` and ``chunk_index`` are overwritten
    afterwards with account-global values — the chunk list here is one conversation's.
    """
    from memory.processor import session_documents

    documents = session_documents(list(chunks), skipped_source_lines=0)
    for document in documents:
        document.metadata["original_file_name"] = origin
    return documents


def _group_by_conversation(documents: Sequence[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for document in documents:
        grouped.setdefault(str(document.metadata["conversation_id"]), []).append(document)
    return grouped


def _carry_forward(
    previous: Sequence[Any], fresh_ids: set[str], affected: set[str]
) -> tuple[dict[str, list[Any]], int]:
    """The stored documents of the conversations this update did not touch, in their stored order.

    A conversation is carried forward when it was **not** re-imported. An affected conversation that
    produced no chunk this time is dropped, exactly as a full rebuild drops it (``import_account``
    removes conversations that contributed no messages), and the count is reported.
    """
    carried: dict[str, list[Any]] = {}
    dropped = 0
    for document in previous:
        conversation_id = str(document.metadata["conversation_id"])
        if conversation_id in affected or conversation_id in fresh_ids:
            continue
        carried.setdefault(conversation_id, []).append(document)
    for conversation_id in sorted(affected - fresh_ids):
        dropped += 1
    return carried, dropped


def _assemble(
    carried: Mapping[str, Sequence[Any]], fresh: Mapping[str, Sequence[Any]]
) -> list[Any]:
    """The next generation's documents, in the order a clean rebuild would put them in.

    ``build_account_sessions`` orders chunks by conversation (sorted) and the index takes them in
    that order, so the assembled list is the union of both maps walked in sorted conversation order.
    Order is not cosmetic here: it decides ``chunk_index``, the document sequence a rebuild would
    produce, and how ties in the vector search are broken.
    """
    documents: list[Any] = []
    for conversation_id in sorted(set(carried) | set(fresh)):
        rows = fresh.get(conversation_id)
        documents.extend(rows if rows is not None else carried[conversation_id])
    return documents


def _merged_aggregate(
    stored_manifest: IndexManifest, documents: Sequence[Any]
) -> dict[str, Any]:
    """The next generation's aggregate report, for the manifest.

    Two kinds of number live in there, and they are treated differently on purpose:

    * **What the documents themselves say** is recomputed exactly — the conversations present, the
      events they hold (``n_events`` is per chunk, and the sessions of a conversation partition its
      events), and the coverage span (a session's start and end *are* its first and last event).
      Those come out identical to what a full rebuild would report.
    * **What only the import saw** — messages received and dropped by the parser, skipped lines,
      conversations discovered but empty — is **carried forward from the stored generation**, because
      the stored documents do not record it and the delta re-imports a handful of conversations whose
      previous contribution is not recoverable. Adding the delta's counters instead would double-count
      every re-imported conversation, which is worse than a stale diagnostic: the load-bearing number
      here is ``messages_kept``, and that one is exact.

    A full rebuild re-derives all of it; this is the price of not doing one, and it is confined to
    diagnostics. The keys are exactly the ones the manifest validator accepts, so the entry stays
    loadable — a report with an unexpected key makes the whole cache INVALID.
    """
    payload = dict(stored_manifest.report or {})

    def number(key: str) -> int:
        value = payload.get(key)
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0

    merged: dict[str, Any] = {key: number(key) for key in index_cache.REPORT_COUNT_KEYS}
    for key in index_cache.REPORT_TUPLE_COUNT_KEYS:
        merged[key] = number(key)

    merged["conversations_imported"] = len(
        {str(document.metadata["conversation_id"]) for document in documents}
    )
    merged["messages_kept"] = sum(
        int(document.metadata.get("n_events") or 0) for document in documents
    )
    starts = [_document_time(document, "start_time") for document in documents]
    ends = [_document_time(document, "end_time") for document in documents]
    starts = [value for value in starts if value is not None]
    ends = [value for value in ends if value is not None]
    merged["first_timestamp"] = min(starts).isoformat() if starts else None
    merged["last_timestamp"] = max(ends).isoformat() if ends else None
    return merged


def _document_time(document: Any, key: str) -> Any:
    from datetime import datetime

    value = (document.metadata or {}).get(key)
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
