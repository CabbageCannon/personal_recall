"""Retrieval over the canonical store: metadata narrowing, then vectors and BM25, then RRF.

The whole point of this module is how little of it there is. `recall.get_retriever` already knows how
to build a hybrid retriever from *any* object exposing `as_retriever` and a document list, and
`hybrid.HybridRRFRetriever` already implements the weighted fusion. So this file does not contain a
second fusion, a second BM25, or a second notion of what a Document looks like — it contains a store
that quacks like the FAISS one, and the product's existing retrieval path runs on top of it unchanged.

That is what makes the comparison in the acceptance meaningful: the only difference between the two
backends is where the dense candidates come from and which subset they are drawn from.

The order of operations is the point of the phase:

1. **Narrow** — `RetrievalFilter` becomes a SQL predicate over the structured indexes.
2. **Dense** — pgvector ranks the survivors by cosine distance.
3. **Lexical** — BM25 ranks the *same* survivors, so the two sides of the fusion see one corpus.
4. **Fuse** — the existing weighted RRF, with the existing weights and the existing constant.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .filters import RetrievalFilter
from .store import DEFAULT_EF_SEARCH
from .rows import content_hash

class PostgresVectorStore:
    """A `VectorStore`-shaped view of the canonical memory store, restricted by a filter.

    Deliberately duck-typed rather than a LangChain `VectorStore` subclass: the two methods the
    product's retrieval path actually calls are ``as_retriever`` and ``get``, and inheriting the base
    class would mean inheriting an embedding-and-insert surface this store must never expose — vectors
    arrive by import, never by adding documents through a retrieval object.
    """

    def __init__(
        self,
        store: Any,
        embedder: Any,
        *,
        origin: str,
        filters: RetrievalFilter | None = None,
        ef_search: int = DEFAULT_EF_SEARCH,
    ) -> None:
        self.store = store
        self.embedder = embedder
        #: The `original_file_name` every document carries, matching what a full index build stamps.
        #: Distinct from the FAISS path's value only if a caller says so, which is why the caller does.
        self.origin = origin
        self.filters = filters or RetrievalFilter()
        self.ef_search = int(ef_search)
        self._documents: list[Document] | None = None
        self._account_order: dict[str, int] | None = None
        self._sessions_total: int | None = None
        #: Evidence a query plan has already decided on. Not carried by `with_filters`: a narrowed
        #: view is a different search, and inheriting an override would silently answer the old one.
        self._override: list[Document] | None = None

    # -- shape ----------------------------------------------------------------------------------

    def with_filters(self, filters: RetrievalFilter | None) -> "PostgresVectorStore":
        """The same store narrowed differently. Shares the connection; drops the cached candidates.

        A new object rather than a mutated one: the candidate list and the BM25 index built over it
        are cached, and a filter change is exactly the event that invalidates both. Returning a fresh
        view makes that impossible to forget.
        """
        return PostgresVectorStore(
            self.store,
            self.embedder,
            origin=self.origin,
            filters=filters or RetrievalFilter(),
            ef_search=self.ef_search,
        )

    @property
    def account_order(self) -> dict[str, int]:
        """``chunk_id -> account-wide position``, computed once per view and cached."""
        if self._account_order is None:
            self._account_order = self.store.account_chunk_order()
        return self._account_order

    @property
    def sessions_total(self) -> int:
        """The account's chunk count — what every document reports as `sessions_total`.

        The whole account, not the filtered subset: the field is part of the document projection a
        full build produces, and a filtered view is a *view* of that account, not a smaller one.
        Writing the candidate count here would make the same chunk project differently depending on
        which question retrieved it.
        """
        if self._sessions_total is None:
            self._sessions_total = int(self.store.table_counts().chunks)
        return self._sessions_total

    def document_for(self, row: Sequence[Any]) -> Document:
        """One chunk row as the Document a full index build would have produced.

        Field for field the same as `memory.processor.session_documents` plus the two keys
        `recall.build_brain_from_chunks` stamps on top. That equality is the whole claim of this
        module, and a test compares the two projections chunk by chunk rather than trusting it.
        """
        chunk_id, conversation_id, start_time, end_time, n_events, text, metadata = row
        chunk_id = str(chunk_id)
        payload = metadata if isinstance(metadata, dict) else {}
        return Document(
            page_content=str(text),
            metadata={
                "memory_chunk_id": chunk_id,
                "conversation_id": str(conversation_id),
                "start_time": start_time.isoformat(sep=" "),
                "end_time": end_time.isoformat(sep=" "),
                "participants": list(payload.get("participants") or []),
                "n_events": int(n_events),
                "sessions_total": self.sessions_total,
                "skipped_source_lines": 0,
                "chunk_index": int(self.account_order.get(chunk_id, 0)),
                "original_file_name": self.origin,
            },
        )

    # -- the two methods the product's retrieval path calls ---------------------------------------

    def get(self) -> dict[str, Any]:
        """Every candidate document, in account order.

        This is what `hybrid.iter_documents` reads to build the BM25 index, so the lexical side sees
        exactly the set the dense side is allowed to return. When a filter narrows the search, BM25 is
        built over the narrowed corpus — which is the phase's second half: the metadata filter shrinks
        what the *lexical* retriever can rank, not only the vector one.
        """
        documents = self.documents()
        return {
            "documents": [document.page_content for document in documents],
            "metadatas": [dict(document.metadata) for document in documents],
        }

    def documents(self) -> list[Document]:
        """The candidate documents, built once and cached for this view."""
        if self._documents is None:
            rows = self.store.select_chunks(self.filters)
            self._documents = [self.document_for(row) for row in rows]
        return self._documents

    def as_retriever(self, search_kwargs: dict[str, Any] | None = None, **kwargs: Any) -> BaseRetriever:
        """A dense retriever over this view, honouring the `search_kwargs` shape the caller uses.

        When an override is set — by a query plan that has already decided what the evidence is —
        that list is returned instead. The seam exists so a plan's evidence reaches the existing
        generation path *through the existing workflow*: `get_retriever` asks the store for a
        retriever, and this one already knows the answer. Nothing downstream branches on whether a
        plan was involved.
        """
        if self._override is not None:
            return StaticRetriever(documents=list(self._override))
        options = dict(search_kwargs or {})
        options.update(kwargs)
        return PgVectorRetriever(source=self, k=int(options.get("k") or 30))

    def set_override(self, documents: Sequence[Document] | None) -> None:
        """Pin the evidence for the next retrieval. ``None`` clears it."""
        self._override = list(documents) if documents is not None else None

    def clear_override(self) -> None:
        self._override = None

    @property
    def override(self) -> list[Document] | None:
        return self._override

    # -- direct use -------------------------------------------------------------------------------

    def similarity_search(self, query: str, k: int = 10) -> list[Document]:
        return self.similarity_search_with_score(query, k=k)[0]

    def similarity_search_with_score(self, query: str, k: int = 10):
        embedding = self.embedder.embed_query(query)
        rows = self.store.dense_chunks(embedding, filters=self.filters, limit=k)
        return [self.document_for(row[:-1]) for row in rows], [float(row[-1]) for row in rows]

    def count(self) -> int:
        return self.store.count_chunks(self.filters)


class StaticRetriever(BaseRetriever):
    """Returns a fixed document list, ignoring the query.

    The seam that lets planned evidence reach the existing generation path without a second answering
    pipeline: the workflow asks a vector store for a retriever, and this one already knows the answer.
    Nothing downstream branches on whether a plan was involved.
    """

    documents: list[Document] = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[Callbacks] = None, **kwargs: Any
    ) -> list[Document]:
        return list(self.documents)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Optional[Callbacks] = None, **kwargs: Any
    ) -> list[Document]:
        return list(self.documents)


class PgVectorRetriever(BaseRetriever):
    """Dense retrieval inside the store, narrowed by the source's filter.

    A `BaseRetriever` rather than a `VectorStoreRetriever` because the fused ranking needs the dense
    side to return exactly ``k`` candidates from the *filtered* set, which is a property this object
    owns rather than one a per-call search kwarg can express.
    """

    source: Any
    k: int = 30

    def _search(self, query: str) -> list[Document]:
        embedding = self.source.embedder.embed_query(query)
        rows = self.source.store.dense_chunks(
            embedding, filters=self.source.filters, limit=self.k
        )
        return [self.source.document_for(row[:-1]) for row in rows]

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[Callbacks] = None, **kwargs: Any
    ) -> list[Document]:
        return self._search(query)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Optional[Callbacks] = None, **kwargs: Any
    ) -> list[Document]:
        return self._search(query)


def describe_filters(filters: RetrievalFilter | None) -> dict[str, Any]:
    """A loggable summary of a narrowing: counts and bounds, never an identity."""
    active = filters or RetrievalFilter()
    return {
        "people": len(active.person_ids),
        "conversations": len(active.conversation_ids),
        "start_time": str(active.start_time) if active.start_time else None,
        "end_time": str(active.end_time) if active.end_time else None,
    }


def candidate_report(store: Any, filters: RetrievalFilter | None) -> dict[str, Any]:
    """Global candidate count vs filtered, and the reduction. The number this phase is measured by."""
    global_count = store.count_chunks()
    filtered = store.count_chunks(filters)
    return {
        "global_candidates": global_count,
        "filtered_candidates": filtered,
        "reduction": (1 - filtered / global_count) if global_count else 0.0,
        "filter": describe_filters(filters),
    }


def content_digest(document: Document) -> str:
    """The chunk-text digest of a retrieved document, for comparing two backends' outputs."""
    return content_hash(document.page_content)
