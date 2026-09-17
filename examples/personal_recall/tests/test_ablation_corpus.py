"""Regression tests for the evidence-ablated corpus (offline: no model, no network).

The ablated corpus is the instrument behind the false-memory eval. If a gold line ever survives
in it, every "the system abstained" result becomes meaningless — the question would still be
answerable. These tests fail loudly in that case, and also guard the real corpus against being
ablated by accident.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import build_ablation_corpus as build  # noqa: E402

SOURCE = build.DATA_DIR / build.SOURCE_NAME
ABLATED = build.DATA_DIR / build.OUT_NAME
QUERIES = build.DATA_DIR / build.QUERIES_NAME


def _gold() -> list[str]:
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    return [line.strip() for q in queries for line in (q.get("relevant_evidence") or [])]


def test_ablated_corpus_exists() -> None:
    assert ABLATED.exists(), f"missing {ABLATED}; run build_ablation_corpus.py"


def test_no_gold_line_survives_in_the_ablated_corpus() -> None:
    text = build.normalise(ABLATED.read_text(encoding="utf-8"))
    survivors = [line for line in _gold() if line in text]
    assert not survivors, f"{len(survivors)} gold line(s) survived: {survivors[:2]}"


def test_real_corpus_still_contains_every_gold_line() -> None:
    """Guards against accidentally ablating (or overwriting) the standing benchmark."""
    text = build.normalise(SOURCE.read_text(encoding="utf-8"))
    missing = [line for line in _gold() if line not in text]
    assert not missing, f"corpus v2 lost {len(missing)} gold line(s)"


def test_ablated_corpus_is_smaller_but_still_substantial() -> None:
    source_lines = build.normalise(SOURCE.read_text(encoding="utf-8")).split("\n")
    ablated_lines = build.normalise(ABLATED.read_text(encoding="utf-8")).split("\n")
    src_msgs = [line for line in source_lines if build.TS_RE.match(line)]
    abl_msgs = [line for line in ablated_lines if build.TS_RE.match(line)]
    assert len(abl_msgs) < len(src_msgs), "ablation removed nothing"
    assert len(abl_msgs) > 1000, f"ablation removed too much: {len(abl_msgs)} messages left"


def test_ablated_corpus_is_well_formed_and_ordered() -> None:
    lines = [line for line in build.normalise(ABLATED.read_text(encoding="utf-8")).split("\n") if line.strip()]
    assert all(build.TS_RE.match(line) for line in lines)
    stamps = [build.TS_RE.match(line).group(1) for line in lines]
    assert len(stamps) == len(set(stamps)), "duplicate timestamps"
    assert stamps == sorted(stamps), "not chronological; build_sessions reads file order"


def test_ablated_corpus_keeps_the_distractor_pack() -> None:
    """The eval is only meaningful if same-topic content survives the ablation."""
    events = build.parse_txt_events(build.normalise(ABLATED.read_text(encoding="utf-8"))).events
    chunks = build.build_sessions(events, build.SessionConfig())
    assert len(chunks) >= 60, f"only {len(chunks)} chunks left; the trap is gone"


def test_build_is_idempotent_and_matches_disk() -> None:
    import subprocess

    out = ABLATED.parent / "_tmp_ablation_check.txt"
    try:
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "build_ablation_corpus.py"), "--out", str(out)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "[FAIL]" not in result.stdout
        assert out.read_text(encoding="utf-8") == ABLATED.read_text(encoding="utf-8")
    finally:
        out.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
