"""Offline, deterministic tests for hybrid retrieval (BM25 + RRF fusion).

No model, no network. The fusion ordering and the budget cut are tested with stub
retrievers, so the assertions are exact. Run from ``examples/personal_recall``::

    python -m pytest tests -q
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from quivr_core.rag.entities.config import HybridConfig, RetrievalConfig
from quivr_core.rag.hybrid import (
    BM25Index,
    BM25Retriever,
    HybridRRFRetriever,
    iter_documents,
    tokenize,
)

DOCS = [
    Document(page_content="我准备把正式项目迁到 Railway 上的 PostgreSQL", metadata={"i": 0}),
    Document(page_content="小王: 可以试试 Supabase", metadata={"i": 1}),
    Document(page_content="今天天气不错，出去走了走", metadata={"i": 2}),
    Document(page_content="Neon 免费层会休眠，冷启动要等好几秒", metadata={"i": 3}),
]


class _StubRetriever(BaseRetriever):
    """Returns a fixed ranking; records that it was queried."""

    documents: list[Document]

    def _get_relevant_documents(self, query, *, run_manager=None, **kwargs):
        return list(self.documents)

    async def _aget_relevant_documents(self, query, *, run_manager=None, **kwargs):
        return list(self.documents)


# ------------------------------------------------------------------------ tokenize


def test_tokenize_keeps_ascii_words_and_adds_cjk_bigrams() -> None:
    tokens = tokenize("Railway 上的 PostgreSQL")

    assert "railway" in tokens
    assert "postgresql" in tokens
    assert "上的" in tokens, "CJK bigrams must be produced (no segmenter available)"
    assert "上" in tokens and "的" in tokens


def test_tokenize_is_case_insensitive_for_ascii() -> None:
    assert "neon" in tokenize("NEON 免费层")


# -------------------------------------------------------------------------- bm25


def test_bm25_index_ranks_the_matching_document_first() -> None:
    index = BM25Index([d.page_content for d in DOCS])

    assert index.top_k("Railway PostgreSQL 迁库", 1) == [0]


def test_bm25_index_returns_nothing_for_a_query_with_no_shared_terms() -> None:
    index = BM25Index([d.page_content for d in DOCS])

    assert index.top_k("zzzzz", 3) == []
    assert index.scores("zzzzz") == [0.0, 0.0, 0.0, 0.0]


def test_bm25_retriever_respects_k_and_drops_zero_scores() -> None:
    retriever = BM25Retriever(documents=DOCS, k=2)

    assert [d.metadata["i"] for d in retriever.invoke("Neon 休眠 冷启动")] == [3]


def test_bm25_retriever_async_matches_sync() -> None:
    retriever = BM25Retriever(documents=DOCS, k=3)

    sync = [d.metadata["i"] for d in retriever.invoke("Railway PostgreSQL Supabase")]
    async_result = [d.metadata["i"] for d in asyncio.run(retriever.ainvoke("Railway PostgreSQL Supabase"))]

    assert async_result == sync


def test_bm25_index_is_built_lazily() -> None:
    retriever = BM25Retriever(documents=DOCS, k=1)

    assert retriever._index is None
    retriever.invoke("Supabase")
    assert retriever._index is not None


# ---------------------------------------------------------------------- fusion


def test_fusion_puts_a_document_ranked_by_both_retrievers_first() -> None:
    first = _StubRetriever(documents=[DOCS[0], DOCS[1]])
    second = _StubRetriever(documents=[DOCS[1], DOCS[2]])
    fused = HybridRRFRetriever(retrievers=[first, second], weights=[0.5, 0.5], k=3)

    order = [d.metadata["i"] for d in fused.invoke("q")]

    assert order[0] == 1, "the document both retrievers returned must win the fusion"
    assert sorted(order) == [0, 1, 2]


def test_fusion_truncates_to_k() -> None:
    both = _StubRetriever(documents=DOCS)
    fused = HybridRRFRetriever(retrievers=[both, both], weights=[0.5, 0.5], k=2)

    assert len(fused.invoke("q")) == 2


def test_fusion_deduplicates_documents_returned_by_both_retrievers() -> None:
    both = _StubRetriever(documents=DOCS)
    fused = HybridRRFRetriever(retrievers=[both, both], weights=[0.5, 0.5], k=10)

    assert [d.metadata["i"] for d in fused.invoke("q")] == [0, 1, 2, 3]


def test_fusion_async_matches_sync() -> None:
    first = _StubRetriever(documents=[DOCS[0], DOCS[1]])
    second = _StubRetriever(documents=[DOCS[1], DOCS[3]])
    fused = HybridRRFRetriever(retrievers=[first, second], weights=[0.5, 0.5], k=3)

    sync = [d.metadata["i"] for d in fused.invoke("q")]
    async_result = [d.metadata["i"] for d in asyncio.run(fused.ainvoke("q"))]

    assert async_result == sync


# ------------------------------------------------------------ document enumeration


def test_iter_documents_uses_the_public_get_api_when_present() -> None:
    store = SimpleNamespace(
        get=lambda: {"documents": ["a", "b"], "metadatas": [{"i": 0}, {"i": 1}]}
    )

    docs = iter_documents(store)

    assert [d.page_content for d in docs] == ["a", "b"]
    assert docs[1].metadata == {"i": 1}


def test_iter_documents_falls_back_to_the_docstore() -> None:
    store = SimpleNamespace(docstore=SimpleNamespace(_dict={"x": DOCS[0]}))

    assert [d.metadata["i"] for d in iter_documents(store)] == [0]


def test_iter_documents_reports_an_unenumerable_store_clearly() -> None:
    with pytest.raises(ValueError, match="Cannot enumerate documents"):
        iter_documents(SimpleNamespace())


# ----------------------------------------------------------------------- wiring


def test_hybrid_is_disabled_by_default() -> None:
    assert HybridConfig().enabled is False
    assert RetrievalConfig().hybrid_config.enabled is False


def test_get_retriever_returns_dense_only_when_hybrid_is_disabled() -> None:
    from quivr_core.rag.quivr_rag_langgraph import QuivrQARAGLangGraph

    dense = _StubRetriever(documents=DOCS)
    store = SimpleNamespace(
        as_retriever=lambda **kwargs: dense,
        get=lambda: {"documents": [d.page_content for d in DOCS], "metadatas": [d.metadata for d in DOCS]},
    )
    stub = SimpleNamespace(
        vector_store=store, retrieval_config=RetrievalConfig(hybrid_config=HybridConfig())
    )

    assert QuivrQARAGLangGraph.get_retriever(stub, search_kwargs={"k": 10}) is dense


def test_get_retriever_fuses_and_cuts_when_hybrid_is_enabled() -> None:
    from quivr_core.rag.quivr_rag_langgraph import QuivrQARAGLangGraph

    captured: dict[str, object] = {}

    class _Dense(_StubRetriever):
        def __init__(self):
            super().__init__(documents=DOCS)

    def _as_retriever(**kwargs):
        captured.update(kwargs)
        return _Dense()

    store = SimpleNamespace(
        as_retriever=_as_retriever,
        get=lambda: {
            "documents": [d.page_content for d in DOCS],
            "metadatas": [d.metadata for d in DOCS],
        },
    )
    config = RetrievalConfig(k=2, hybrid_config=HybridConfig(enabled=True, candidate_k=30))
    stub = SimpleNamespace(vector_store=store, retrieval_config=config)

    retriever = QuivrQARAGLangGraph.get_retriever(stub, search_kwargs={"k": 2})

    assert isinstance(retriever, HybridRRFRetriever)
    assert retriever.k == 2
    assert len(retriever.retrievers) == 2
    assert isinstance(retriever.retrievers[1], BM25Retriever)
    assert retriever.retrievers[1].documents and len(retriever.retrievers[1].documents) == len(DOCS)
    # the dense side must be widened to the same pool size as the lexical side
    assert captured["search_kwargs"] == {"k": 30}
    assert len(retriever.invoke("Railway")) == 2
