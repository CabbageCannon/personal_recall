"""Wiring tests for the WeFlow source adapter, plus the privacy guard.

Two things are verified here that the pure adapter tests cannot cover:

* the framework's processor registry actually resolves a ``.json`` corpus to the WeFlow processor
  and still resolves ``.txt`` to the session processor — the registration is real, not assumed;
* a JSON corpus and a TXT corpus produce **the same document shape**, so the retrieval path cannot
  tell them apart. That is what keeps the evaluation valid for a real export.

The privacy tests use ``git check-ignore``, so "real chat data must never be committed" is a checked
property rather than a comment.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent.parent
sys.path.insert(0, str(BASE_DIR))

from memory.processor import ConversationSessionProcessor, WeFlowSessionProcessor  # noqa: E402
from memory.sessions import SessionConfig  # noqa: E402
from recall import (  # noqa: E402
    JSON_EXTENSION,
    SOURCE_FORMATS,
    count_corpus_events,
    detect_source_format,
    register_source_processors,
)

T1126 = 1789000000


def _weflow_payload(n: int = 6, step: int = 120) -> list[dict]:
    return [
        {
            "localId": i + 1,
            "serverId": f"srv{i + 1}",
            "localType": 1,
            "createTime": T1126 + i * step,
            "isSend": i % 2,
            "senderUsername": "wxid_synthetic_person",
            "parsedContent": f"synthetic message {i}",
        }
        for i in range(n)
    ]


CANONICAL_TXT = "\n".join(
    f"[2026-09-17 {11 + i // 60:02d}:{(26 + i) % 60:02d}] 我: synthetic message {i}" for i in range(6)
) + "\n"


class _FakeFile:
    """Only `.path` is used by `process_file_inner`."""

    def __init__(self, path: Path) -> None:
        self.path = path


def _scratch(name: str) -> Path:
    path = BASE_DIR / "tests" / "_scratch_wiring" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture()
def json_corpus():
    import shutil

    directory = _scratch("json")
    path = directory / "synthetic_chat.json"
    path.write_text(json.dumps(_weflow_payload(), ensure_ascii=False), encoding="utf-8")
    yield path
    shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture()
def txt_corpus():
    import shutil

    directory = _scratch("txt")
    path = directory / "synthetic_chat.txt"
    path.write_text(CANONICAL_TXT, encoding="utf-8")
    yield path
    shutil.rmtree(directory, ignore_errors=True)


# --- registry wiring -------------------------------------------------------------------------


def test_registry_resolves_each_extension_to_its_processor() -> None:
    import warnings

    from quivr_core.files.file import get_file_extension
    from quivr_core.processor.registry import get_processor_class

    register_source_processors()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert get_processor_class(get_file_extension(Path("a.txt"))) is ConversationSessionProcessor
        assert get_processor_class(get_file_extension(Path("a.json"))) is WeFlowSessionProcessor


def test_a_weflow_run_does_not_replace_the_text_processor() -> None:
    """Registering the JSON adapter must not change what a .txt corpus resolves to."""
    import warnings

    from quivr_core.files.file import get_file_extension
    from quivr_core.processor.registry import get_processor_class

    register_source_processors()
    register_source_processors()  # idempotent
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert get_processor_class(get_file_extension(Path("a.txt"))) is ConversationSessionProcessor


def test_source_format_detection_and_choices() -> None:
    assert detect_source_format(Path("x.json")) == "weflow"
    assert detect_source_format(Path("x.JSON")) == "weflow"
    assert detect_source_format(Path("x.txt")) == "text"
    assert detect_source_format(Path("x.unknown")) == "text"
    assert set(SOURCE_FORMATS) == {"text", "weflow"}
    assert JSON_EXTENSION == ".json"


# --- processor behaviour ---------------------------------------------------------------------


def test_weflow_processor_ingests_json_into_sessions(json_corpus: Path) -> None:
    processor = WeFlowSessionProcessor(session_config=SessionConfig(max_chars=900))
    document = asyncio.run(processor.process_file_inner(_FakeFile(json_corpus)))
    assert document.chunks, "the adapter produced no chunks"
    # all six messages are within one session window, so exactly one retrieval unit
    assert len(document.chunks) == 1
    assert document.chunks[0].metadata["n_events"] == 6
    assert document.chunks[0].metadata["conversation_id"] == "synthetic_chat"
    assert document.chunks[0].metadata["skipped_source_lines"] == 0


def test_json_and_txt_corpora_produce_the_same_document_shape(
    json_corpus: Path, txt_corpus: Path
) -> None:
    """The retrieval path must not be able to tell a JSON corpus from a TXT corpus."""
    json_doc = asyncio.run(
        WeFlowSessionProcessor(session_config=SessionConfig(max_chars=900)).process_file_inner(
            _FakeFile(json_corpus)
        )
    )
    txt_doc = asyncio.run(
        ConversationSessionProcessor(session_config=SessionConfig(max_chars=900)).process_file_inner(
            _FakeFile(txt_corpus)
        )
    )
    assert set(json_doc.chunks[0].metadata) == set(txt_doc.chunks[0].metadata)


def test_weflow_chunk_text_is_the_canonical_evidence_form(json_corpus: Path) -> None:
    """Session text must stay citable chat lines, whatever the source format."""
    document = asyncio.run(
        WeFlowSessionProcessor(session_config=SessionConfig(max_chars=900)).process_file_inner(
            _FakeFile(json_corpus)
        )
    )
    for line in document.chunks[0].page_content.split("\n"):
        assert line.startswith("[2026-") and "] 我: " in line or "] 对方: " in line


def test_processor_conversation_id_defaults_to_the_file_stem(json_corpus: Path) -> None:
    document = asyncio.run(
        WeFlowSessionProcessor(session_config=SessionConfig(max_chars=900)).process_file_inner(
            _FakeFile(json_corpus)
        )
    )
    assert document.chunks[0].metadata["memory_chunk_id"].startswith("synthetic_chat-session-")


def test_processor_metadata_declares_the_source_format() -> None:
    assert WeFlowSessionProcessor().processor_metadata["source_format"] == "weflow"
    assert "source_format" not in ConversationSessionProcessor().processor_metadata


# --- corpus counting (what the CLI guard uses) ------------------------------------------------


def test_count_corpus_events_for_both_adapters(json_corpus: Path, txt_corpus: Path) -> None:
    json_count, json_skipped, detail = count_corpus_events(json_corpus, "weflow")
    txt_count, txt_skipped, _ = count_corpus_events(txt_corpus, "text")
    assert (json_count, json_skipped) == (6, 0)
    assert (txt_count, txt_skipped) == (6, 0)
    assert "messages" in detail


def test_count_corpus_events_reports_unmatched_input_as_zero() -> None:
    import shutil

    directory = _scratch("empty")
    try:
        path = directory / "empty.json"
        path.write_text(json.dumps([{"localId": 1}]), encoding="utf-8")
        count, skipped, _ = count_corpus_events(path, "weflow")
        assert count == 0 and skipped == 1
    finally:
        shutil.rmtree(directory, ignore_errors=True)


# --- privacy: real data must never be committable ---------------------------------------------


def _ignored(relative: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", relative],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.returncode == 0


SENSITIVE_PATHS = [
    "examples/personal_recall/data/real/alice.json",
    "examples/personal_recall/data/real/alice.txt",
    "examples/personal_recall/data/my_chat.txt",
    "examples/personal_recall/data/real_export.json",
    "examples/personal_recall/data/weflow_2026.json",
    "examples/personal_recall/MSG2.db",
    "examples/personal_recall/MSG0.db-wal",
    "examples/personal_recall/wechat_key.txt",
    "examples/personal_recall/decrypt_key.bin",
    "examples/personal_recall/.env",
    # These two names are not hypothetical: a real smoke test left both files sitting untracked and
    # NOT ignored in this working tree, one `git add -A` away from committing real chat lines.
    "examples/personal_recall/data/wechat_smoke.txt",
    "examples/personal_recall/recall_debug.json",
    # root-level scratch that `git add -A` would otherwise sweep up
    ".tmp_stress2/probe.json",
    ".tmp_grade1/labels.json",
]

SYNTHETIC_PATHS = [
    "examples/personal_recall/data/chats.txt",
    "examples/personal_recall/data/stress_chats.txt",
    "examples/personal_recall/data/stress_chats_v2.txt",
    "examples/personal_recall/data/stress_chats_v2_no_gold.txt",
    "examples/personal_recall/data/stress_queries.json",
    "examples/personal_recall/data/stress_v2_parts/part_a_2024.txt",
]


@pytest.mark.parametrize("path", SENSITIVE_PATHS)
def test_real_chat_data_paths_are_git_ignored(path: str) -> None:
    assert _ignored(path), f"{path} is NOT ignored; real chat data could be committed"


@pytest.mark.parametrize("path", SYNTHETIC_PATHS)
def test_synthetic_corpora_stay_trackable(path: str) -> None:
    assert not _ignored(path), f"{path} is a synthetic corpus and must remain tracked"


def test_gitignore_is_not_vacuous() -> None:
    """A .gitignore that matches nothing would make the checks above pass for the wrong reason."""
    assert (BASE_DIR / ".gitignore").exists()
    ignored = [path for path in SENSITIVE_PATHS if _ignored(path)]
    assert len(ignored) == len(SENSITIVE_PATHS)
