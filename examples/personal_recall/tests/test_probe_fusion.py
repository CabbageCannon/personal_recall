"""Offline tests for the offline retrieval probes (no model, no network).

`rrf_ranking` in `probe_pool_rerank.py` is load-bearing: every probe result rests on the
claim that the offline reproduction of the runner's hybrid retrieval is exact. The probes
assert that agreement against a recorded run, but that check can only fail loudly if the
fusion is wrong in a way that changes the Top-k. These tests pin the fusion itself against
the framework implementation it is reproducing.
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from probe_pool_rerank import rrf_ranking  # noqa: E402


def _docs(n: int) -> list[Document]:
    return [Document(page_content=f"d{i}") for i in range(n)]


def _framework_fusion(rank_lists: list[list[Document]]) -> list[str]:
    """Call LangChain's own weighted_reciprocal_rank, the code the runner actually uses."""
    ensemble = EnsembleRetriever(retrievers=[], weights=[0.5] * len(rank_lists), c=60)
    return [doc.page_content for doc in ensemble.weighted_reciprocal_rank(rank_lists)]


def test_matches_framework_on_disjoint_lists() -> None:
    docs = _docs(6)
    lists = [[docs[0], docs[1], docs[2]], [docs[5], docs[4], docs[3]]]
    expected = _framework_fusion(lists)
    got = [f"d{i}" for i in rrf_ranking([[0, 1, 2], [5, 4, 3]])]
    assert got == expected


def test_matches_framework_with_overlap_and_ties() -> None:
    """Overlapping documents collapse and accumulate; ties keep insertion order."""
    docs = _docs(6)
    lists = [[docs[0], docs[1], docs[2]], [docs[2], docs[0], docs[3]], [docs[3], docs[0], docs[1]]]
    expected = _framework_fusion(lists)
    got = [f"d{i}" for i in rrf_ranking([[0, 1, 2], [2, 0, 3], [3, 0, 1]])]
    assert got == expected


def test_matches_framework_on_identical_lists() -> None:
    docs = _docs(4)
    lists = [[docs[0], docs[1], docs[2], docs[3]], [docs[0], docs[1], docs[2], docs[3]]]
    expected = _framework_fusion(lists)
    got = [f"d{i}" for i in rrf_ranking([[0, 1, 2, 3], [0, 1, 2, 3]])]
    assert got == expected


def test_single_list_preserves_order() -> None:
    assert rrf_ranking([[3, 1, 2]]) == [3, 1, 2]


def test_empty_rank_list_is_ignored() -> None:
    """A retriever contributing nothing (e.g. BM25 with no term matches) must not break fusion."""
    docs = _docs(3)
    expected = _framework_fusion([[docs[2], docs[0], docs[1]], []])
    got = [f"d{i}" for i in rrf_ranking([[2, 0, 1], []])]
    assert got == expected


def test_rank_is_one_based_so_first_beats_second() -> None:
    """A document ranked 1st by both lists must outrank one ranked 2nd by both."""
    got = rrf_ranking([[0, 1], [0, 1]])
    assert got == [0, 1]
