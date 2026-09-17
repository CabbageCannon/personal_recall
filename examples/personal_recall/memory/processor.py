"""Retrieval-unit processor: a chat export becomes one Document per session.

Registered for ``.txt`` instead of ``SimpleTxtProcessor`` to swap the *retrieval
unit* (fixed-character slice -> conversation session) while keeping everything
else in the pipeline identical: same embedder, same FAISS index, same Top-K, same
prompt. That makes the Phase 1 A/B a single-variable experiment.

``WeFlowSessionProcessor`` reuses the same session construction and the same document
shape for WeFlow JSON exports (``.json``), so the only thing that differs between a
text corpus and a JSON corpus is where the ``MemoryEvent``s come from.
"""

from __future__ import annotations

import json
from typing import Any

import aiofiles
from langchain_core.documents import Document

from quivr_core.files.file import QuivrFile
from quivr_core.processor.processor_base import ProcessedDocument, ProcessorBase
from quivr_core.processor.registry import FileExtension

from .events import parse_txt_events
from .sessions import MemoryChunk, SessionConfig, build_sessions
from .weflow import parse_weflow_events


def session_documents(
    sessions: list[MemoryChunk],
    skipped_source_lines: int,
) -> list[Document]:
    """Build one Document per session — the single definition shared by every adapter.

    NOTE: the inner metadata deliberately does NOT carry `original_file_name`.
    `ProcessorBase.process_file` prefixes the page content with
    "Filename: ... Content: ..." when it is present, and the fixed-character
    baseline never triggers that — adding it here would change the embedded
    text and turn the chunking A/B into a two-variable experiment.
    """
    return [
        Document(
            page_content=session.text,
            metadata={
                "memory_chunk_id": session.id,
                "conversation_id": session.conversation_id,
                "start_time": session.start_time.isoformat(sep=" "),
                "end_time": session.end_time.isoformat(sep=" "),
                "participants": list(session.participants),
                "n_events": session.n_events,
                "sessions_total": len(sessions),
                "skipped_source_lines": skipped_source_lines,
            },
        )
        for session in sessions
    ]


class ConversationSessionProcessor(ProcessorBase[str]):
    """Emit one chunk per conversation session (see ``memory/sessions.py``)."""

    supported_extensions = [FileExtension.txt]

    def __init__(
        self,
        session_config: SessionConfig = SessionConfig(),
        conversation_id: str = "txt",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.session_config = session_config
        self.conversation_id = conversation_id

    @property
    def processor_metadata(self) -> dict[str, Any]:
        return {
            "processor_cls": "ConversationSessionProcessor",
            "session": self.session_config.as_dict(),
        }

    async def process_file_inner(self, file: QuivrFile) -> ProcessedDocument[str]:
        async with aiofiles.open(file.path, mode="r") as f:
            content = await f.read()

        parsed = parse_txt_events(content, conversation_id=self.conversation_id)
        sessions = build_sessions(
            parsed.events,
            config=self.session_config,
            conversation_id=self.conversation_id,
        )
        documents = session_documents(sessions, parsed.skipped_lines)

        return ProcessedDocument(
            chunks=documents,
            processor_cls="ConversationSessionProcessor",
            processor_response=content,
        )


class WeFlowSessionProcessor(ConversationSessionProcessor):
    """Emit one chunk per conversation session from a WeFlow JSON export.

    Deliberately a subclass: it reuses the session construction and document shape unchanged, so a
    JSON corpus and a TXT corpus produce the same kind of retrieval unit. Only the event source
    differs — the JSON is parsed straight into ``MemoryEvent``s with no canonical-TXT intermediate.
    """

    supported_extensions = [".json"]

    def __init__(
        self,
        session_config: SessionConfig = SessionConfig(),
        conversation_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(session_config=session_config, conversation_id=conversation_id or "", **kwargs)
        #: ``None`` means "use the export's file stem", which keeps two exports of different
        #: conversations from sharing an id space.
        self.conversation_id = conversation_id or ""

    @property
    def processor_metadata(self) -> dict[str, Any]:
        return {
            "processor_cls": "WeFlowSessionProcessor",
            "session": self.session_config.as_dict(),
            "source_format": "weflow",
        }

    async def process_file_inner(self, file: QuivrFile) -> ProcessedDocument[str]:
        async with aiofiles.open(file.path, mode="r", encoding="utf-8-sig") as f:
            content = await f.read()

        conversation_id = self.conversation_id or file.path.stem
        parsed = parse_weflow_events(json.loads(content), conversation_id=conversation_id)
        sessions = build_sessions(
            parsed.events,
            config=self.session_config,
            conversation_id=conversation_id,
        )
        documents = session_documents(sessions, parsed.skipped)

        return ProcessedDocument(
            chunks=documents,
            processor_cls="WeFlowSessionProcessor",
            processor_response=content,
        )
