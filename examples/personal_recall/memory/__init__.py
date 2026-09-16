"""Personal Recall memory model: chat history -> MemoryEvent -> MemoryChunk.

Two levels, deliberately kept distinct (they answer different questions):

* ``MemoryEvent``  — one atomic chat message. The *fact* / citation unit.
* ``MemoryChunk``  — a conversation session. The *retrieval* unit.

Phase 1 introduces this layer so retrieval operates on coherent conversation
sessions instead of arbitrary fixed-character slices, and so later phases
(temporal filtering, entity resolution, evidence/citation) have stable ids and
time ranges to attach to.
"""

from .events import MemoryEvent, parse_txt_events
from .sessions import MemoryChunk, SessionConfig, build_sessions

__all__ = [
    "MemoryEvent",
    "MemoryChunk",
    "SessionConfig",
    "build_sessions",
    "parse_txt_events",
]
