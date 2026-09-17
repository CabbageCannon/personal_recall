"""The local web UI of Personal Recall: one page, one question box, the answer with its evidence.

The frontend is plain HTML/CSS/JS served by FastAPI — there is no build step, no bundler and no
npm toolchain in this repository, and adding one would make "start the tool" a longer command than
"ask a question". The backend is a thin adapter over ``recall.answer_question``: the web UI is a
second *view* of the product, never a second implementation of it.
"""

from .app import HOST, DEFAULT_PORT, RecallState, create_app, read_conversation_labels, short_reason

__all__ = [
    "DEFAULT_PORT",
    "HOST",
    "RecallState",
    "create_app",
    "read_conversation_labels",
    "short_reason",
]
