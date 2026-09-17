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
from .shards import (
    MergeReport,
    ShardInfo,
    discover_message_shards,
    discover_shard_files,
    load_and_merge,
    merge_shard_events,
)
from .weflow import WeFlowParseResult, parse_weflow_events

__all__ = [
    "MemoryEvent",
    "MemoryChunk",
    "MergeReport",
    "SessionConfig",
    "ShardInfo",
    "WeFlowParseResult",
    "build_sessions",
    "discover_message_shards",
    "discover_shard_files",
    "load_and_merge",
    "merge_shard_events",
    "parse_txt_events",
    "parse_weflow_events",
]
