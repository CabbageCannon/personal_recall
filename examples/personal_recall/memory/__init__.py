"""Personal Recall memory model: chat history -> MemoryEvent -> MemoryChunk.

Two levels, deliberately kept distinct (they answer different questions):

* ``MemoryEvent``  — one atomic chat message. The *fact* / citation unit.
* ``MemoryChunk``  — a conversation session. The *retrieval* unit.

Phase 1 introduces this layer so retrieval operates on coherent conversation
sessions instead of arbitrary fixed-character slices, and so later phases
(temporal filtering, entity resolution, evidence/citation) have stable ids and
time ranges to attach to.
"""

from .account import (
    ACCOUNT_MANIFEST_FILENAME,
    SESSION_LISTING_FILENAME,
    AccountImportReport,
    AccountLayout,
    ConversationCoverage,
    ConversationShardExport,
    build_account_sessions,
    crossed_conversation_chunks,
    discover_shard_directories,
    exports_from_directory,
    group_exports_by_conversation,
    import_account,
    load_account_directory,
    read_account_manifest,
)
from .conversations import (
    EXPORT_FILENAME_SUFFIX,
    ConversationDescriptor,
    account_label,
    conversation_id_from_export_filename,
    conversation_type_of,
    descriptor_from_session,
    parse_session_listing,
    shard_stem,
)
from .events import OTHER_ROLE, SELF_ROLE, MemoryEvent, parse_txt_events
from .senders import assign_sender_labels, column_label
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
    "ACCOUNT_MANIFEST_FILENAME",
    "EXPORT_FILENAME_SUFFIX",
    "SESSION_LISTING_FILENAME",
    "AccountImportReport",
    "AccountLayout",
    "ConversationCoverage",
    "ConversationDescriptor",
    "ConversationShardExport",
    "MemoryEvent",
    "MemoryChunk",
    "MergeReport",
    "OTHER_ROLE",
    "SELF_ROLE",
    "SessionConfig",
    "ShardInfo",
    "WeFlowParseResult",
    "account_label",
    "assign_sender_labels",
    "build_account_sessions",
    "build_sessions",
    "column_label",
    "conversation_id_from_export_filename",
    "conversation_type_of",
    "crossed_conversation_chunks",
    "descriptor_from_session",
    "discover_message_shards",
    "discover_shard_directories",
    "discover_shard_files",
    "exports_from_directory",
    "group_exports_by_conversation",
    "import_account",
    "load_account_directory",
    "load_and_merge",
    "merge_shard_events",
    "parse_session_listing",
    "parse_txt_events",
    "parse_weflow_events",
    "read_account_manifest",
    "shard_stem",
]
