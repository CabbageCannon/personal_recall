"""Local cross-encoder re-ranking.

The built-in re-ranker suppliers (`cohere`, `jina`) are hosted services: they need an
API key and they receive the query together with the retrieved chunks. Personal Recall
re-ranks a user's private conversation memory, so the re-rank stage must also be able to
run locally, with no network and no third party.

`LocalCrossEncoderReranker` plugs a local ``sentence-transformers`` CrossEncoder into the
re-rank stage that :meth:`QuivrQARAGLangGraph.retrieve` already performs via
``ContextualCompressionRetriever``, so no other part of the pipeline changes: the
retriever still returns ``RetrievalConfig.k`` candidates and the reranker decides which
``RerankerConfig.top_n`` of them are passed on.

The model is loaded lazily on first use, so constructing a config never triggers a
download or a disk read.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import PrivateAttr

logger = logging.getLogger("quivr_core")


class LocalCrossEncoderReranker(BaseDocumentCompressor):
    """Re-rank documents with a locally hosted cross-encoder model.

    Attributes:
        model: local path or hub id of the cross-encoder.
        top_n: how many documents to keep after re-ranking.
        batch_size: pairs scored per forward pass.
        max_length: token truncation length for the (query, document) pair.
        device: torch device string, e.g. ``cpu``.
        relevance_score_threshold: optional cut-off; scores below it are dropped.
        score_key: metadata key receiving the cross-encoder score, so the retrieval
            trace can explain why a chunk was kept.

    Every returned document is a copy carrying the score in its metadata; the input
    documents are never mutated.
    """

    model: str
    top_n: int = 5
    batch_size: int = 32
    max_length: int = 512
    device: str = "cpu"
    relevance_score_threshold: float | None = None
    score_key: str = "rerank_score"

    _encoder: Any = PrivateAttr(default=None)

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            from sentence_transformers import CrossEncoder

            logger.debug(
                "loading local cross-encoder %s on %s", self.model, self.device
            )
            self._encoder = CrossEncoder(
                self.model, max_length=self.max_length, device=self.device
            )
        return self._encoder

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Score every (query, document) pair and keep the best ``top_n``."""
        if not documents:
            return []

        encoder = self._get_encoder()
        scores = encoder.predict(
            [(query, document.page_content) for document in documents],
            batch_size=self.batch_size,
        )

        ranked = sorted(
            zip(documents, scores), key=lambda pair: float(pair[1]), reverse=True
        )

        selected: list[Document] = []
        for document, score in ranked:
            value = float(score)
            if (
                self.relevance_score_threshold is not None
                and value < self.relevance_score_threshold
            ):
                continue
            selected.append(
                Document(
                    page_content=document.page_content,
                    metadata={**document.metadata, self.score_key: value},
                )
            )
            if len(selected) >= self.top_n:
                break

        return selected
