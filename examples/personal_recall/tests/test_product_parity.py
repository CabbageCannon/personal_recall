"""Tests that the shipped product stays the system the evaluation measured.

The evaluation drives `run_baseline.py`; the user drives `recall.py`. Phase 16 verified empirically
that both retrieve identically (36/36), but an empirical result does not stop a future edit from
re-introducing the divergence. These tests pin the *structure* that makes the two comparable —
a single construction path and adopted-reference defaults — plus the recorded parity artifact.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import recall  # noqa: E402

PARITY_ARTIFACT = BASE_DIR / "product_parity.json"


def test_main_has_no_second_construction_path() -> None:
    """All config construction must live in `build_session`, or the two paths can drift apart."""
    source = inspect.getsource(recall.main)
    assert "build_session(" in source
    for inline in ("LLMEndpointConfig(", "RetrievalConfig(", "HybridConfig("):
        assert inline not in source, f"main() builds {inline} inline again; parity is no longer guaranteed"


def test_build_session_is_the_shared_constructor() -> None:
    source = inspect.getsource(recall.build_session)
    for piece in ("register_answer_prompt", "LLMEndpointConfig(", "RetrievalConfig(", "build_brain("):
        assert piece in source, f"build_session lost {piece}"


def test_defaults_match_the_adopted_reference() -> None:
    """A12 on corpus v2: session chunks, hybrid pool 30, k=20, no-rewrite, attribution clause."""
    assert recall.DEFAULT_K == 20
    assert recall.DEFAULT_HYBRID_POOL == 30
    assert recall.DEFAULT_ANSWER_PROMPT == "cited-attributed"
    assert recall.DEFAULT_MAX_SESSION_CHARS == 900

    signature = inspect.signature(recall.build_session)
    assert signature.parameters["workflow"].default == "no-rewrite"
    assert signature.parameters["answer_prompt"].default == "cited-attributed"
    assert signature.parameters["k"].default == 20


def test_the_cli_offers_the_adopted_prompt_as_its_default() -> None:
    source = inspect.getsource(recall.main)
    assert "default=DEFAULT_ANSWER_PROMPT" in source
    assert "cited-attributed" in source


def test_recorded_parity_shows_no_divergence() -> None:
    """Pins the measured result: the product retrieves exactly what the A12 arm retrieved."""
    assert PARITY_ARTIFACT.exists(), f"missing {PARITY_ARTIFACT.name}; run verify_product_parity.py"
    payload = json.loads(PARITY_ARTIFACT.read_text(encoding="utf-8"))
    assert payload["queries"] == 36
    assert payload["retrieval_diverged"] == 0
    assert payload["retrieval_identical"] == 36
    assert len(payload["rows"]) == 36
    for row in payload["rows"]:
        assert row["recorded_chunks"] == row["product_chunks"], row["query_id"]
