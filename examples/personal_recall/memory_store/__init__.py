"""Phase 22A — the PostgreSQL canonical memory store.

Two layers of source of truth now exist, and the order between them matters:

* **Raw source** — the WeChat databases and the WeFlow export tree. This is the history, it is never
  deleted, and it is what everything is rebuilt from.
* **Structured memory** — this PostgreSQL store: conversations, people, events, chunks and the
  evidence links between them, in the form Personal Recall already normalized them into.

The store is a *derived* view. It can be dropped and rebuilt from the exports at any time, and
``memory_store`` never becomes the only place something is written down. What it adds is the ability
to ask structured questions — this person, this period, this conversation, in this order — without
parsing JSON or scanning a vector index.

It is **not** a retrieval backend. FAISS, BM25 and the weighted RRF fusion are exactly as Phase 21B
left them, ``build_account_session`` still answers from the vector index, and a machine with no
PostgreSQL is still a working Personal Recall (§20, §22). Phase 22B is where a metadata filter and
pgvector would enter, and nothing here anticipates it: there is no vector column, no extension and
no migration stub.
"""

from __future__ import annotations

from .bootstrap import (
    SnapshotError,
    SnapshotResult,
    account_snapshot,
    conversation_snapshot,
    read_labels,
    state_fields,
)
from .filters import RetrievalFilter, combine
from .vectors import (
    IndexVector,
    VectorImportError,
    VectorImportReport,
    import_vectors,
    read_index_vectors,
)
from .config import (
    DATABASE_URL_VAR,
    STORE_NAME_VAR,
    StoreNotConfigured,
    StoreTarget,
    database_url,
    store_name_for,
    target_for,
)
from .rows import (
    TABLE_COLUMNS,
    ProjectionError,
    StoreSnapshot,
    build_snapshot,
    content_hash,
    person_id_for,
)
from .schema import (
    SCHEMA_VERSION,
    Migration,
    SchemaError,
    SchemaVersionError,
    apply_migrations,
    current_version,
    discover_migrations,
    require_current,
)
from .retrieval import (
    DEFAULT_EF_SEARCH,
    PgVectorRetriever,
    PostgresVectorStore,
    candidate_report,
    describe_filters,
)
from .store import (
    PostgresMemoryStore,
    StoreCounts,
    StoreError,
    StoreNotEmpty,
    WriteReport,
    connect,
)

__all__ = [
    "DATABASE_URL_VAR",
    "Migration",
    "PgVectorRetriever",
    "PostgresMemoryStore",
    "PostgresVectorStore",
    "ProjectionError",
    "IndexVector",
    "RetrievalFilter",
    "SCHEMA_VERSION",
    "STORE_NAME_VAR",
    "SchemaError",
    "SchemaVersionError",
    "SnapshotError",
    "SnapshotResult",
    "StoreCounts",
    "StoreError",
    "StoreNotConfigured",
    "StoreNotEmpty",
    "StoreSnapshot",
    "StoreTarget",
    "TABLE_COLUMNS",
    "VectorImportError",
    "VectorImportReport",
    "WriteReport",
    "account_snapshot",
    "apply_migrations",
    "build_snapshot",
    "DEFAULT_EF_SEARCH",
    "candidate_report",
    "connect",
    "combine",
    "content_hash",
    "conversation_snapshot",
    "current_version",
    "database_url",
    "describe_filters",
    "discover_migrations",
    "import_vectors",
    "person_id_for",
    "read_index_vectors",
    "read_labels",
    "require_current",
    "state_fields",
    "store_name_for",
    "target_for",
]
