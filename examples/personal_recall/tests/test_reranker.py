"""Offline, deterministic tests for the local cross-encoder re-ranker.

No model download, no network: a stub encoder stands in for the CrossEncoder, so the
scoring/ordering/threshold logic is tested exactly and fast. Run from
``examples/personal_recall``::

    python -m pytest tests -q
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from quivr_core.rag.entities.config import DefaultRerankers, RerankerConfig
from quivr_core.rag.reranker import LocalCrossEncoderReranker

MODEL = "stub-cross-encoder"


class _StubEncoder:
    """Scores a pair by document length, so ordering is known in advance."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def predict(self, pairs, batch_size: int = 32):
        self.calls.append((len(pairs), batch_size))
        return [float(len(text)) for _, text in pairs]


def _reranker(**kwargs) -> LocalCrossEncoderReranker:
    reranker = LocalCrossEncoderReranker(model=MODEL, **kwargs)
    reranker._encoder = _StubEncoder()
    return reranker


# --------------------------------------------------------------------------- config


def test_local_supplier_requires_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCAL_API_KEY", raising=False)

    config = RerankerConfig(
        supplier=DefaultRerankers.LOCAL, model=MODEL, top_n=3
    )

    assert config.supplier is DefaultRerankers.LOCAL
    assert config.model == MODEL
    assert config.top_n == 3
    assert config.api_key is None
    assert DefaultRerankers.LOCAL.requires_api_key is False


def test_hosted_suppliers_still_require_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="COHERE_API_KEY"):
        RerankerConfig(supplier=DefaultRerankers.COHERE)


def test_local_supplier_without_a_model_fails_loudly() -> None:
    with pytest.raises(ValueError, match="explicit model"):
        RerankerConfig(supplier=DefaultRerankers.LOCAL)


def test_hosted_supplier_default_model_is_unchanged() -> None:
    assert DefaultRerankers.COHERE.default_model == "rerank-v3.5"


# ---------------------------------------------------------------- compress_documents


def test_keeps_the_best_top_n_in_score_order() -> None:
    docs = [Document(page_content="a" * 5), Document(page_content="b" * 20), Document(page_content="c" * 9)]

    out = _reranker(top_n=2).compress_documents(docs, query="q")

    assert [d.page_content[0] for d in out] == ["b", "c"]
    assert [d.metadata["rerank_score"] for d in out] == [20.0, 9.0]


def test_input_documents_are_not_mutated() -> None:
    docs = [Document(page_content="a" * 5, metadata={"chunk_index": 7})]

    out = _reranker(top_n=5).compress_documents(docs, query="q")

    assert "rerank_score" not in docs[0].metadata
    assert out[0].metadata["rerank_score"] == 5.0
    assert out[0].metadata["chunk_index"] == 7, "existing metadata must survive"


def test_relevance_threshold_drops_weak_documents() -> None:
    docs = [Document(page_content="a" * 5), Document(page_content="b" * 20)]

    out = _reranker(top_n=5, relevance_score_threshold=10.0).compress_documents(docs, "q")

    assert [d.page_content[0] for d in out] == ["b"]


def test_threshold_is_applied_before_top_n_cut() -> None:
    docs = [Document(page_content="x" * n) for n in (30, 20, 3)]

    out = _reranker(top_n=2, relevance_score_threshold=10.0).compress_documents(docs, "q")

    assert len(out) == 2
    assert [d.metadata["rerank_score"] for d in out] == [30.0, 20.0]


def test_empty_input_short_circuits() -> None:
    reranker = _reranker(top_n=5)

    assert reranker.compress_documents([], query="q") == []
    assert reranker._encoder.calls == [], "no model call for an empty candidate list"


def test_batch_size_is_forwarded() -> None:
    docs = [Document(page_content="a" * n) for n in range(4)]

    reranker = _reranker(top_n=4, batch_size=2)
    reranker.compress_documents(docs, query="q")

    assert reranker._encoder.calls == [(4, 2)]


def test_custom_score_key_is_respected() -> None:
    docs = [Document(page_content="a" * 3)]

    out = _reranker(top_n=1, score_key="cross_encoder_score").compress_documents(docs, "q")

    assert out[0].metadata["cross_encoder_score"] == 3.0


# ------------------------------------------------------------------- framework wiring


def test_framework_selects_the_local_reranker_for_the_local_supplier() -> None:
    """`QuivrRAG.get_reranker` must build our compressor from the config alone."""
    from types import SimpleNamespace

    from quivr_core.rag.quivr_rag_langgraph import QuivrQARAGLangGraph

    stub = SimpleNamespace(
        retrieval_config=SimpleNamespace(
            reranker_config=RerankerConfig(
                supplier=DefaultRerankers.LOCAL, model=MODEL, top_n=4
            )
        )
    )

    reranker = QuivrQARAGLangGraph.get_reranker(stub)

    assert isinstance(reranker, LocalCrossEncoderReranker)
    assert reranker.top_n == 4
    assert reranker.model == MODEL


def test_framework_falls_back_to_the_no_op_compressor_without_a_supplier() -> None:
    from types import SimpleNamespace

    from quivr_core.rag.quivr_rag_langgraph import (
        IdempotentCompressor,
        QuivrQARAGLangGraph,
    )

    stub = SimpleNamespace(
        retrieval_config=SimpleNamespace(reranker_config=RerankerConfig())
    )

    assert isinstance(QuivrQARAGLangGraph.get_reranker(stub), IdempotentCompressor)
