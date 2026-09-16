"""Regression tests for the stress corpus v2 build (offline: no model, no network).

The corpus is a measurement instrument: if the distractor pack ever starts
restating a gold fact, or the merge stops being chronological, every v2 arm
silently measures something other than ambiguity. These tests pin the two
invariants that matter and keep the checked-in corpus in sync with its sources.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import build_stress_v2 as build  # noqa: E402

PARTS_DIR = build.DEFAULT_PARTS_DIR
CORPUS = build.DATA_DIR / build.OUT_NAME
QUERIES = build.DATA_DIR / build.QUERIES_NAME


def _gold() -> list[str]:
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    return [line.strip() for q in queries for line in (q.get("relevant_evidence") or [])]


def test_pack_parts_are_checked_in() -> None:
    for name in build.PART_NAMES:
        path = PARTS_DIR / name
        assert path.exists(), f"missing distractor pack part: {path}"


def test_corpus_exists_and_is_chronological() -> None:
    text = build.normalise(CORPUS.read_text(encoding="utf-8"))
    lines = [line for line in text.split("\n") if line.strip()]
    stamps = [build.TS_RE.match(line).group(1) for line in lines]
    assert len(stamps) == len(set(stamps)), "timestamps must be globally unique"
    assert stamps == sorted(stamps), "build_sessions consumes file order, so it must be sorted"


def test_corpus_matches_its_sources_byte_for_byte() -> None:
    """Editing stress_chats_v2.txt by hand must fail here, not silently ship."""
    episodes: list[tuple[str, str]] = []
    for path in [build.DATA_DIR / build.V1_NAME] + [PARTS_DIR / n for n in build.PART_NAMES]:
        for block in build.split_episodes(path.read_text(encoding="utf-8")):
            match = build.TS_RE.match(block.split("\n", 1)[0])
            assert match is not None, f"{path.name}: malformed episode"
            episodes.append((match.group(1), block))
    episodes.sort(key=lambda item: item[0])
    rebuilt = "\n\n".join(block for _, block in episodes) + "\n"
    assert rebuilt == build.normalise(CORPUS.read_text(encoding="utf-8"))


def test_no_gold_line_is_restated_by_the_pack() -> None:
    gold = _gold()
    assert gold, "the query set must carry gold evidence"
    pack_text = "\n\n".join(
        (PARTS_DIR / name).read_text(encoding="utf-8") for name in build.PART_NAMES
    )
    pack_contents = {
        build.TS_RE.match(line).group(3).strip()
        for line in build.normalise(pack_text).split("\n")
        if build.TS_RE.match(line)
    }
    assert not (pack_contents & build.gold_contents(gold))


def test_every_gold_line_survives_exactly_once() -> None:
    text = build.normalise(CORPUS.read_text(encoding="utf-8"))
    for line in _gold():
        assert text.count(line) == 1, f"gold line missing or duplicated: {line[:40]!r}"


def test_corpus_does_not_fragment_under_session_chunking() -> None:
    """A mis-ordered corpus collapses to one message per chunk; guard the profile."""
    text = build.normalise(CORPUS.read_text(encoding="utf-8"))
    events = build.parse_txt_events(text).events
    chunks = build.build_sessions(events, build.SessionConfig())
    per_chunk = len(events) / len(chunks)
    assert per_chunk >= 5, f"fragmented: {len(chunks)} chunks for {len(events)} messages"


def test_build_is_idempotent_and_passes_its_own_checks() -> None:
    """Run the real build into a temp path: every check must pass and match on disk."""
    import subprocess

    out = CORPUS.parent / "_tmp_build_check.txt"
    try:
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "build_stress_v2.py"), "--out", str(out)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "[FAIL]" not in result.stdout
        assert out.read_text(encoding="utf-8") == CORPUS.read_text(encoding="utf-8")
    finally:
        out.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
