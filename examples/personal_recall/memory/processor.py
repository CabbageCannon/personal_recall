"""Retrieval-unit processor: a chat export becomes one Document per session.

Registered for ``.txt`` instead of ``SimpleTxtProcessor`` to swap the *retrieval
unit* (fixed-character slice -> conversation session) while keeping everything
else in the pipeline identical: same embedder, same FAISS index, same Top-K, same
prompt. That makes the Phase 1 A/B a single-variable experiment.
"""

from __future__ import annotations

from typing import Any

import aiofiles
from langchain_core.documents import Document

from quivr_core.files.file import QuivrFile
from quivr_core.processor.processor_base import ProcessedDocument, ProcessorBase
from quivr_core.processor.registry import FileExtension

from .events import parse_txt_events
from .sessions import SessionConfig, build_sessions


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

        # NOTE: the inner metadata deliberately does NOT carry `original_file_name`.
        # `ProcessorBase.process_file` prefixes the page content with
        # "Filename: ... Content: ..." when it is present, and the fixed-character
        # baseline never triggers that — adding it here would change the embedded
        # text and turn the chunking A/B into a two-variable experiment.
        documents = [
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
                    "skipped_source_lines": parsed.skipped_lines,
                },
            )
            for session in sessions
        ]

        return ProcessedDocument(
            chunks=documents,
            processor_cls="ConversationSessionProcessor",
            processor_response=content,
        )
