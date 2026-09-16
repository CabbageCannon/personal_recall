"""Hybrid retrieval: dense vectors fused with BM25 by reciprocal rank fusion.

Dense embeddings retrieve by meaning; BM25 retrieves by exact terms (proper nouns,
product names, numbers) that embeddings sometimes rank low. Fusing the two ranked
lists with RRF recovers evidence neither retriever ranks highly alone.

Measured on the Personal Recall stress corpus (session chunks, 10 slots):

    dense only            82.4 % evidence coverage
    BM25 only             78.2 %
    RRF(dense, BM25)      84.8 %   <- the only mechanism that beat dense at equal slots

Design notes:

* ``tokenize`` uses ASCII/digit words plus CJK unigrams **and bigrams**. Whitespace
  tokenisation is meaningless for Chinese (a whole sentence becomes one token), and a
  segmenter would add a dependency for no measured benefit.
* BM25 is implemented here (~50 lines) rather than via ``langchain_community``'s
  ``BM25Retriever``, which requires the ``rank_bm25`` package. The formula below is the
  exact one used for the offline measurement, so measurement and deployment agree.
* ``HybridRRFRetriever`` reuses the framework's fusion (``EnsembleRetriever.rank_fusion``,
  weighted RRF with ``c=60``) and then **truncates to ``k``**, because the plain
  ensemble returns the union of its retrievers' lists — which would quietly hand the
  model more context than the configuration claims.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import PrivateAttr

#: ASCII/digit words, CJK characters (bigrams are added on top).
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\+\#\.\-]*|\d+|[\u4e00-\u9fff]")

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75
DEFAULT_RRF_C = 60


def tokenize(text: str) -> list[str]:
    """Tokenise mixed Chinese/English chat text without a segmenter."""
    tokens: list[str] = []
    cjk: list[str] = []
    for token in TOKEN_RE.findall(text):
        if "\u4e00" <= token[0] <= "\u9fff":
            cjk.append(token)
            tokens.append(token)
        else:
            tokens.append(token.lower())
    tokens.extend(first + second for first, second in zip(cjk, cjk[1:]))
    return tokens


class BM25Index:
    """Okapi BM25 over a fixed document set (rank_bm25-free)."""

    def __init__(self, texts: Sequence[str], k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(text) for text in texts]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        self._df = document_frequency
        self._n = len(self._tokens)

    def scores(self, query: str) -> list[float]:
        """BM25 score for every document, in corpus order."""
        scores = [0.0] * self._n
        if not self._n or not self._avg_length:
            return scores
        for term in set(tokenize(query)):
            frequency = self._df.get(term, 0)
            if frequency == 0:
                continue
            idf = math.log(1 + (self._n - frequency + 0.5) / (frequency + 0.5))
            for i, tokens in enumerate(self._tokens):
                term_frequency = tokens.count(term)
                if term_frequency:
                    denominator = term_frequency + self.k1 * (
                        1 - self.b + self.b * self._lengths[i] / self._avg_length
                    )
                    scores[i] += idf * term_frequency * (self.k1 + 1) / denominator
        return scores

    def top_k(self, query: str, k: int) -> list[int]:
        """Indices of the ``k`` best documents, best first."""
        scored = sorted(enumerate(self.scores(query)), key=lambda pair: -pair[1])
        return [index for index, score in scored[:k] if score > 0]


class BM25Retriever(BaseRetriever):
    """Lexical retriever over a fixed document list, usable inside an ensemble."""

    documents: list[Document]
    k: int = 10
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B

    _index: Optional[BM25Index] = PrivateAttr(default=None)

    def _get_index(self) -> BM25Index:
        if self._index is None:
            self._index = BM25Index(
                [document.page_content for document in self.documents], k1=self.k1, b=self.b
            )
        return self._index

    def _search(self, query: str) -> list[Document]:
        index = self._get_index()
        return [self.documents[i] for i in index.top_k(query, self.k)]

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[Callbacks] = None, **kwargs: Any
    ) -> list[Document]:
        return self._search(query)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Optional[Callbacks] = None, **kwargs: Any
    ) -> list[Document]:
        return self._search(query)


class HybridRRFRetriever(BaseRetriever):
    """Weighted-RRF fusion of several retrievers, truncated to ``k`` documents."""

    retrievers: list[BaseRetriever]
    weights: list[float] = [0.5, 0.5]
    k: int = 10
    c: int = DEFAULT_RRF_C

    _ensemble: Any = PrivateAttr(default=None)

    def _get_ensemble(self) -> Any:
        if self._ensemble is None:
            from langchain.retrievers import EnsembleRetriever

            self._ensemble = EnsembleRetriever(
                retrievers=self.retrievers, weights=self.weights, c=self.c
            )
        return self._ensemble

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[Callbacks] = None, **kwargs: Any
    ) -> list[Document]:
        fused = self._get_ensemble().rank_fusion(query, run_manager)
        return list(fused)[: self.k]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Optional[Callbacks] = None, **kwargs: Any
    ) -> list[Document]:
        fused = await self._get_ensemble().arank_fusion(query, run_manager)
        return list(fused)[: self.k]


def iter_documents(vector_store: VectorStore) -> list[Document]:
    """Enumerate every document in a vector store, across store flavours.

    LangChain vector stores expose their contents differently (``get()`` for Chroma,
    an in-memory docstore for FAISS); BM25 needs the whole corpus up front, so try the
    public API first and fall back to the docstore.
    """
    getter = getattr(vector_store, "get", None)
    if callable(getter):
        try:
            payload = getter()
            contents = payload.get("documents") if isinstance(payload, dict) else None
            metadatas = payload.get("metadatas") if isinstance(payload, dict) else None
            if contents:
                return [
                    Document(page_content=text, metadata=metadata or {})
                    for text, metadata in zip(contents, metadatas or [{}] * len(contents))
                ]
        except Exception:  # noqa: BLE001 - fall through to the docstore path
            pass

    docstore = getattr(vector_store, "docstore", None)
    store = getattr(docstore, "_dict", None)
    if isinstance(store, dict) and store:
        return list(store.values())

    raise ValueError(
        f"Cannot enumerate documents from {type(vector_store).__name__} for BM25; "
        "it exposes neither get() nor an in-memory docstore."
    )
