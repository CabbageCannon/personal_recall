"""Prove the WeFlow adapter did not change the existing text path.

Adding a second source adapter must not move a single byte of the path the evaluation measured. This
test rebuilds the corpus-v2 retrieval units through the **real processor** and compares every chunk's
text and id against the contents the recorded A12 arm stored for the same corpus. If the text path
had drifted — even by a character, even in one chunk — the recorded evidence would no longer match.

It is offline: the arm artifact is committed, so no API call is needed.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from memory.processor import ConversationSessionProcessor  # noqa: E402
from memory.sessions import SessionConfig  # noqa: E402

CORPUS = BASE_DIR / "data" / "stress_chats_v2.txt"
RECORDED_ARM = BASE_DIR / "stress_v2_attrib_results.json"
MAX_SESSION_CHARS = 900


class _FakeFile:
    def __init__(self, path: Path) -> None:
        self.path = path


@pytest.fixture(scope="module")
def rebuilt_chunks() -> dict[str, str]:
    """chunk_index -> chunk text, as the processor produces it today."""
    document = asyncio.run(
        ConversationSessionProcessor(
            session_config=SessionConfig(max_chars=MAX_SESSION_CHARS)
        ).process_file_inner(_FakeFile(CORPUS))
    )
    return {chunk.metadata["memory_chunk_id"]: chunk.page_content for chunk in document.chunks}


def test_rebuilt_chunk_count_matches_the_recorded_arm(rebuilt_chunks: dict[str, str]) -> None:
    recorded = json.loads(RECORDED_ARM.read_text(encoding="utf-8"))
    recorded_ids = {
        source["memory_chunk_id"]
        for row in recorded
        for source in row.get("retrieved_sources") or []
    }
    # 36 queries x 20 slots, 199 distinct sessions of the 234 in the corpus.
    assert len(recorded_ids) >= 190, "the recorded arm changed; re-derive the expectation"
    missing = recorded_ids - set(rebuilt_chunks)
    assert not missing, f"{len(missing)} chunk id(s) the arm cited no longer exist: {sorted(missing)[:3]}"


def test_every_recorded_chunk_content_is_reproduced_exactly(
    rebuilt_chunks: dict[str, str]
) -> None:
    """The strongest available offline statement that the text path is unchanged."""
    recorded = json.loads(RECORDED_ARM.read_text(encoding="utf-8"))
    total_sources = sum(len(row.get("retrieved_sources") or []) for row in recorded)
    compared = 0
    mismatches = []
    for row in recorded:
        for source in row.get("retrieved_sources") or []:
            chunk_id = source["memory_chunk_id"]
            rebuilt = rebuilt_chunks.get(chunk_id)
            if rebuilt is None:
                continue
            compared += 1
            if rebuilt != source["content"]:
                mismatches.append(chunk_id)
    assert total_sources >= 700, f"only {total_sources} recorded sources; the check is not meaningful"
    assert compared == total_sources, (
        f"only {compared} of {total_sources} recorded sources could be compared"
    )
    assert not mismatches, (
        f"{len(mismatches)} chunk(s) differ from the recorded arm: {sorted(set(mismatches))[:3]}"
    )


def test_chunk_ids_use_the_txt_conversation_id(rebuilt_chunks: dict[str, str]) -> None:
    """The text path keeps conversation_id 'txt', exactly as every recorded arm assumes."""
    assert all(chunk_id.startswith("txt-session-") for chunk_id in rebuilt_chunks)
