"""The web UI's backend: one page, two endpoints, and the product's own recall path.

What this module deliberately is **not**: a second retrieval implementation. ``POST /api/recall``
calls :func:`recall.answer_question`, the same function the CLI prints from, so a web answer and a
CLI answer are the same answer produced the same way. There is nothing here that could drift.

Three constraints shape the rest of the file:

* **Local only.** The server binds ``127.0.0.1`` and there is no argument that changes it. This
  serves real private chat; a wildcard bind is not a smaller version of this tool, it is a
  different one.
* **The index is built once.** Loading an account means parsing every export and embedding every
  session, which is minutes, not milliseconds. :class:`RecallState` holds the brain for the life of
  the process and every request only reads it.
* **A failed startup is not a crash.** If indexing fails, the server still starts and
  ``GET /api/status`` reports ``ready: false`` with a short reason: a page that explains what is
  wrong beats a process that refuses to exist. That reason is a *status field* — the console log of
  a failed start carries counts and an exception class, never a message body, because log lines
  outlive terminals in a way chat content must not.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

#: The only address this server ever binds. Not a default and not a fallback: a constant.
HOST = "127.0.0.1"
DEFAULT_PORT = 8000

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

#: Longest reason string ``/api/status`` will carry. A reason is a status field, not a log file.
REASON_LIMIT = 200


def short_reason(exc: BaseException, *, limit: int = REASON_LIMIT) -> str:
    """A one-line reason for a failed index build: exception class plus a clipped message.

    Clip and flatten, because this string is served to the page and an unbounded message would be a
    stack trace in a status field. The class name is kept for the same reason the exporter keeps
    one: a failure that reports *nothing* is the failure mode this project keeps paying for.
    """
    message = " ".join(str(exc).split())
    reason = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    if len(reason) > limit:
        reason = reason[: limit - 1] + "…"
    return reason


def read_conversation_labels(account_dir: Path) -> dict[str, str]:
    """``conversation_id -> display name`` for the evidence card headers.

    A card says "我, 对方 · 2025-05-12" (in a group: "我, 成员A, 成员B"), which is unambiguous inside
    one conversation and ambiguous across an account of forty. The name is decoration and never a key
    (see
    ``memory.conversations``), so this is best-effort by design: an export tree with no
    ``sessions.json`` and no sidecar still answers questions, just without the header.

    Conversations with no real name are left out rather than labelled with their talker: a wxid is an
    internal identifier, and the UI has no business printing one.

    Two sources, in order. The **sidecar** (``<account>/conversation_labels.json``, written by
    ``sync_conversation_labels.py``) is authoritative when it holds anything, because it is the one
    that asked ``weflow-cli`` for names; when it is absent, empty or unreadable the tree's own
    ``sessions.json`` listing is used, exactly as before Phase 20.6. Both paths go through the same
    rule — ``memory.labels.usable_name`` — and neither ever yields a talker id: on a real account the
    exporter's listing gives every conversation a ``displayName`` equal to its ``username``, so a
    truthiness guard would pass for all of them and the page would print a group id where a name
    belongs.

    Merging the two sources per conversation was considered and rejected: it would make a card's
    header depend on which source happened to know that conversation, so the same page would explain
    itself two different ways depending on a file's history. One source wins, and it is the one that
    was written to be authoritative.
    """
    from memory import load_account_directory
    from memory.labels import (
        CONVERSATION_LABEL_FILENAME,
        read_label_sidecar,
        resolve_conversation_labels,
    )

    sidecar = read_label_sidecar(Path(account_dir) / CONVERSATION_LABEL_FILENAME)
    if sidecar:
        return {talker: label.label for talker, label in sidecar.items()}

    try:
        layout = load_account_directory(account_dir)
    except (OSError, ValueError):
        return {}
    # The descriptors resolve through the same precedence as the sidecar did; with no contact records
    # the only offer is each conversation's own display name.
    return {
        talker: label.label
        for talker, label in resolve_conversation_labels(layout.descriptors, ()).items()
    }


@dataclass
class RecallState:
    """The account index for this process: built once, then read by every request."""

    account_dir: Path
    #: Where the persistent index lives, and whether to ignore a valid one. Both are handed straight to
    #: ``build_account_session``, so the server makes no cache decision of its own — it has the same two
    #: controls as the CLI, over the same single cache, which is what keeps a warm web start and a warm
    #: CLI start the same start.
    index_dir: Path | None = None
    rebuild_index: bool = False
    brain: Any = None
    retrieval_config: Any = None
    report: Any = None
    conversation_labels: dict[str, str] = field(default_factory=dict)
    detail: str = ""
    #: What the persistent index did at startup (``recall.index_cache.IndexStatus``), for the log
    #: line `web.py` prints. Aggregate numbers only — never a message, a name or an id.
    index_status: Any = None

    @property
    def ready(self) -> bool:
        return self.brain is not None

    def load(self) -> None:
        """Open the account index — from the persistent index when it is valid. Never raises.

        A warm start is the normal case: the index is loaded rather than rebuilt, and ``web.py``
        prints what happened. The cache decision itself is not made here — ``build_account_session``
        makes it, once, for the CLI, this server and the acceptance runner alike.
        """
        import dotenv
        import recall

        # `build_account_session` expects the environment to be loaded already, exactly as
        # `recall.main` arranges before calling it.
        dotenv.load_dotenv(recall.ENV_PATH if recall.ENV_PATH.exists() else None)

        try:
            session = recall.build_account_session(
                self.account_dir,
                index_dir=self.index_dir,
                rebuild_index=self.rebuild_index,
            )
        except Exception as exc:  # noqa: BLE001 - any failure must still leave a server that answers
            self.detail = short_reason(exc)
            return

        self.brain = session.brain
        self.retrieval_config = session.retrieval_config
        self.report = session.report
        self.index_status = session.index
        self.conversation_labels = read_conversation_labels(self.account_dir)

    def answer(self, question: str) -> dict[str, Any]:
        """Ask one question through the shared product path.

        ``Brain.ask`` is the framework's synchronous wrapper around an async pipeline and resolves
        its loop with ``asyncio.get_event_loop()``. FastAPI runs a sync endpoint in a worker thread,
        and a worker thread has no current loop, so the call would fail with "There is no current
        event loop in thread 'AnyIO worker thread'". A fresh loop per call keeps two questions from
        ever sharing one.
        """
        import recall

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return recall.answer_question(self.brain, self.retrieval_config, question=question)
        finally:
            asyncio.set_event_loop(None)
            loop.close()


class RecallRequest(BaseModel):
    """The request body: a question, and nothing else."""

    question: str = ""


#: Exactly what ``webapp/static/app.js`` renders. The API returns the page's fields and nothing else.
#:
#: ``memory_chunk_id`` is the reason this list exists rather than a pass-through: a chunk id is
#: ``f"{conversation_id}-session-NNNN"``, so shipping it hands the browser the talker — a group id or
#: a wxid — in a response that deliberately keeps the conversation label out. Chunk ids, retrieval
#: scores and ``n_events`` are useful to the CLI's ``--json``, which reads ``recall.answer_question``
#: directly and is unaffected by this projection; they are not useful to a page that must not print
#: an internal identifier. Found by asserting on the payload's *values*, not on its key names: the
#: first check looked for a ``conversation_id`` key, found none, and passed while the id rode along
#: inside another field.
PAGE_CARD_FIELDS = ("citation_index", "start_time", "end_time", "participants", "lines")


def evidence_for_page(
    evidence: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    labels: dict[str, str],
) -> list[dict[str, Any]]:
    """Project the shared evidence cards down to the page's fields, adding the conversation name.

    ``citation_index`` is the card's position in the retrieved sources — that is what the framework's
    ``Source: N`` numbering means — so the conversation it came from is one lookup away. Only the
    display name is copied out; the conversation id stays where it is, and every other field is
    dropped rather than forwarded.
    """
    projected: list[dict[str, Any]] = []
    for card in evidence:
        page_card = {key: card[key] for key in PAGE_CARD_FIELDS if key in card}

        index = card.get("citation_index")
        if isinstance(index, int) and 0 <= index < len(sources):
            name = labels.get(sources[index].get("conversation_id") or "")
            if name:
                page_card["conversation"] = name
        projected.append(page_card)
    return projected


def coverage_of(report: Any) -> dict[str, Any]:
    """The coverage block of ``/api/status``: counts and a span, never a conversation id.

    ``partial`` is carried because it is the one status a user must act on: history that exists but
    was never exported is invisible to every answer, and a smaller history must never be mistaken
    for a smaller past.
    """
    if report is None:
        return {}
    return {
        "first_timestamp": (
            report.first_timestamp.isoformat(sep=" ") if report.first_timestamp else None
        ),
        "last_timestamp": (
            report.last_timestamp.isoformat(sep=" ") if report.last_timestamp else None
        ),
        "partial": bool(report.partial),
    }


def create_app(state: RecallState) -> FastAPI:
    """Build the app around one account index (loaded, or failed to load — both are servable)."""
    # No /docs, /redoc or /openapi.json: three more endpoints is three more things served from a
    # process holding private chat, for a UI nobody debugs through Swagger.
    app = FastAPI(title="Personal Recall", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/")
    def index() -> FileResponse:
        """The single page."""
        return FileResponse(INDEX_HTML)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        """Readiness and coverage. Counts only — never a message, a question or a display name."""
        if not state.ready:
            return {
                "ready": False,
                "conversation_count": 0,
                "message_count": 0,
                "coverage": {},
                "detail": state.detail or "索引尚未就绪。",
            }
        report = state.report
        return {
            "ready": True,
            "conversation_count": report.conversations_imported,
            "message_count": report.messages_kept,
            "coverage": coverage_of(report),
        }

    @app.post("/api/recall")
    def recall_question(payload: RecallRequest) -> dict[str, Any]:
        """Answer one question, with its evidence and its caveats."""
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空。")
        if not state.ready:
            raise HTTPException(status_code=503, detail=state.detail or "索引尚未就绪。")

        try:
            result = state.answer(question)
        except Exception as exc:  # noqa: BLE001 - see below
            # Deliberately not a 500: an escaping exception is logged by the server with its
            # message, and an upstream API error quotes the request — which here is the question.
            # Question text must not reach a console log, so the failure is reported by class name
            # and the page's own error state carries the rest.
            print(f"recall failed: {type(exc).__name__}", file=sys.stderr)
            raise HTTPException(status_code=502, detail="检索失败，请重试。") from None

        return {
            "answer": result["answer"],
            "evidence": evidence_for_page(
                result["evidence"], result["sources"], state.conversation_labels
            ),
            "groundedness": result["groundedness"],
            "latency_ms": result["latency_ms"],
        }

    return app
