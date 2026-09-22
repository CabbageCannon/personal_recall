"""The persistent retrieval index: a cold build is paid once, a warm start loads it.

Indexing an account means parsing every export and embedding every session. On a real account that
is ~1 636 000 messages collapsing into ~81 576 chunks and roughly 80 minutes before the first
question can be asked. Almost all of that time is spent producing something that does not change
between runs: the same chunks, embedded with the same model into the same vectors.

This module persists exactly that, and nothing else:

    <index dir>/<account leaf>/
        index.faiss      the vectors            } written by `FAISS.save_local`, which is also
        index.pkl        the docstore binding   } what Quivr itself uses (vector <-> Document
                                                 } <-> metadata, which a citation needs)
        manifest.json    identity-free identity of the build (see :class:`IndexManifest`)
        READY            the completeness marker, written **last**

Three rules shape every decision here.

**1. A false HIT is far worse than a false MISS.** A wrong cache entry silently answers questions
from history that is not the history on disk — the one failure this tool cannot recover from,
because the answer looks exactly like a correct one. A false miss costs 80 minutes and is always
recoverable. So when a fingerprint cannot be established, or an artifact is short a byte, the entry
is a miss with a *stated reason* and the run rebuilds.

**2. The JSON export tree stays the source of truth.** This is a pure accelerator: ``rm -rf`` the
index directory and the next run rebuilds from the exports, with no other consequence. Nothing here
can become the only copy of anything.

**3. The index may not learn who the user talks to.** The cached *documents* hold private chat text
— that is the point of a local retrieval index, and it never leaves the machine. The **manifest**,
every status line and every log line carry only versions, hex digests, counts and timestamps: no
message text, no speaker id, no wxid, no ``@chatroom`` id, no API key. A cache entry is identified
by a digest of the export tree, never by a path or a name.

What invalidates an entry (and what deliberately does not):

===========================  ==========================================================
invalidates                  why
===========================  ==========================================================
the export tree changed      path + size + mtime_ns of every file the import reads
chunking configuration       ``SessionConfig`` (``max_gap``, ``max_chars``)
chunking code shape          ``CHUNKING_SHAPE_VERSION`` — a new rule, not a new knob
document projection          ``DOCUMENT_PROJECTION_VERSION`` — the session text format
embedding identity           model path, ``normalize_embeddings``, vector dimension
cache format                 ``CACHE_FORMAT_VERSION`` — this module's own layout
===========================  ==========================================================

===========================  ==========================================================
does **not** invalidate      why
===========================  ==========================================================
``k``, ``hybrid_pool``       retrieval knobs: they select from the index, never build it
``workflow``, ``answer_prompt``, LLM model, ``temperature``, ``max_output_tokens``
--------------------------------------------------------------------------------------
===========================  ==========================================================

The fingerprint is over the files the import actually *reads* — ``*_messages.json`` exports,
``sessions.json`` listings and the account's ``shard_manifest.json`` — and not over every file in
the tree. A listing rewritten by ``sync_conversation_labels.py`` sits in the same directory and
changes no chunk; hashing it would buy an 80-minute rebuild for nothing.

BM25 is deliberately **not** persisted. It is built lazily on the first query from the documents
enumerated out of the FAISS docstore, so a warm start rebuilds it in memory. That is a rebuild, and
the timings reported by :class:`IndexStatus` say so; it is never claimed as a load.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from memory.account import AccountImportReport, AccountLayout
from memory.conversations import (
    ACCOUNT_MANIFEST_FILENAME,
    EXPORT_FILENAME_SUFFIX,
    SESSION_LISTING_FILENAME,
    shard_stem,
)
from memory.processor import DOCUMENT_PROJECTION_VERSION
from memory.sessions import SessionConfig

# ---------------------------------------------------------------------------------------------
# Versions. Every one of these is an explicit statement that a cache entry is only valid for the
# code and configuration it was written by; bump the relevant constant when its meaning changes.
# ---------------------------------------------------------------------------------------------

#: This module's own layout: the artifact names, the manifest fields, the marker's contents.
CACHE_FORMAT_VERSION = 1

#: The shape of the segmentation **code**, as opposed to its knobs (which live in ``SessionConfig``
#: and are hashed directly). Bump when a rule is added or a boundary changes — a change that leaves
#: ``max_gap``/``max_chars`` identical while producing different chunks.
CHUNKING_SHAPE_VERSION = 1

#: Re-exported from the module that builds the documents, so the projection fingerprint and the code
#: that renders a session into a Document cannot drift apart.
PROJECTION_VERSION = DOCUMENT_PROJECTION_VERSION

#: File names inside a cache entry. ``index.faiss`` / ``index.pkl`` are chosen by
#: ``FAISS.save_local``; they are named here because validation has to look for them.
MANIFEST_FILENAME = "manifest.json"
READY_FILENAME = "READY"
VECTORS_FILENAME = "index.faiss"
DOCSTORE_FILENAME = "index.pkl"

#: A build writes into ``<leaf>.staging-<token>`` and is promoted by a rename, so a run interrupted
#: at minute 70 leaves a directory that no reader will ever mistake for an index.
STAGING_INFIX = ".staging-"
#: Where a promoted entry's predecessor is moved before the new one takes its place.
TRASH_INFIX = ".trash-"

#: The account data area. ``recall.DATA_DIR/real/.index_cache`` — spelled out here rather than
#: imported so this module has no dependency on the product module that imports it.
DEFAULT_INDEX_ROOT = Path(__file__).resolve().parent / "data" / "real" / ".index_cache"

#: The adopted product embedding model. The model path and ``normalize_embeddings`` are part of a
#: cache entry's identity; the device and the batch size are deliberately not, because they change
#: how long an embedding takes, never what it is. One definition, so the corpus path, the account
#: path and the fingerprint cannot disagree about which model produced the vectors.
EMBEDDING_MODEL_PATH = Path(r"D:\AIModels\bge-small-zh-v1.5")
EMBEDDING_NORMALIZE = True


def build_embedder() -> Any:
    """The product embedder. Built lazily so importing this module stays cheap."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=str(EMBEDDING_MODEL_PATH),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": EMBEDDING_NORMALIZE},
    )


# ---------------------------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------------------------


def canonical_json(payload: Mapping[str, Any]) -> str:
    """One serialisation for every digest and every file this module writes."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def chunking_fingerprint(config: SessionConfig) -> str:
    """``SessionConfig`` plus the shape of the chunking code that consumes it.

    The gap is hashed in seconds, not as the float minutes ``SessionConfig.as_dict`` reports: a
    digest must not depend on how a number happens to be spelled.
    """
    return _digest(
        {
            "shape_version": CHUNKING_SHAPE_VERSION,
            "max_gap_seconds": int(config.max_gap.total_seconds()),
            "max_chars": int(config.max_chars),
        }
    )


def embedding_identity(embedder: Any) -> dict[str, Any]:
    """Model identity/path and ``normalize_embeddings`` — the two things that change a vector.

    Device, batch size and the class's other construction arguments are absent on purpose: they
    change throughput, not output, and hashing them would invalidate an 80-minute build because
    somebody moved work to a GPU.
    """
    model = getattr(embedder, "model_name", None) or getattr(embedder, "model", None)
    if not model:
        model = type(embedder).__name__
    encode_kwargs = getattr(embedder, "encode_kwargs", None) or {}
    return {
        "kind": type(embedder).__name__,
        "model": str(model),
        "normalize_embeddings": bool(encode_kwargs.get("normalize_embeddings", False)),
    }


def embedding_fingerprint(embedder: Any, *, dimension: int) -> str:
    """The embedding identity **including the resulting vector dimension**.

    A caller validating a stored entry passes the dimension recorded there (which the loaded index
    is separately checked against), so the comparison needs no embedding call to resolve.
    """
    return _digest({**embedding_identity(embedder), "dimension": int(dimension)})


#: The files an account import actually reads. Everything else in the tree — label sidecars, editor
#: backups, a stray note — is not an input to the index, and a fingerprint over it would rebuild 81
#: thousand chunks because a file the index never opens was touched.
CONSUMED_FILENAMES = (SESSION_LISTING_FILENAME, ACCOUNT_MANIFEST_FILENAME)


def _is_consumed(name: str) -> bool:
    return name.endswith(EXPORT_FILENAME_SUFFIX) or name in CONSUMED_FILENAMES


def _inside(path: Path, roots: Sequence[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def source_inventory(account_dir: Path, *, exclude: Sequence[Path] = ()) -> list[str]:
    """``relative path \\0 size \\0 mtime_ns`` for every file the import reads, sorted.

    Relative paths keep an entry valid when the tree is moved, and keep absolute machine paths out
    of the digest. Sorting makes the walk's order — which no filesystem promises — irrelevant.

    A file that cannot be stat'd raises rather than being skipped: an inventory with a hole in it
    cannot establish that the source is unchanged, and that question must be answered, not guessed.
    """
    root = Path(account_dir)

    def _raise(error: OSError) -> None:
        raise error

    excluded = [Path(path).resolve() for path in exclude]
    entries: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=_raise):
        here = Path(dirpath)
        if _inside(here.resolve(), excluded):
            dirnames[:] = []
            continue
        # A checkout's metadata is not chat history, and it churns constantly.
        dirnames[:] = sorted(name for name in dirnames if name != ".git")
        for name in sorted(filenames):
            if not _is_consumed(name):
                continue
            stat = (here / name).stat()
            relative = (here / name).relative_to(root).as_posix()
            entries.append(f"{relative}\x00{stat.st_size}\x00{stat.st_mtime_ns}")
    entries.sort()
    return entries


def source_fingerprint(account_dir: Path, *, exclude: Sequence[Path] = ()) -> str:
    """A digest over what the import will read. Raises ``OSError`` if the tree cannot be walked."""
    return hashlib.sha256(
        "\n".join(source_inventory(account_dir, exclude=exclude)).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------------------------
# The manifest — identity-free by construction
# ---------------------------------------------------------------------------------------------

#: Counts, and nothing else, from the import report. The per-conversation rows, the shard names and
#: the free-text notes are deliberately NOT persisted: a reconstructed report may say how much was
#: imported, never who it was.
REPORT_COUNT_KEYS = (
    "conversations_discovered",
    "conversations_imported",
    "messages_received",
    "messages_kept",
    "duplicates_removed",
    "skipped_messages",
    "undedupeable",
)

#: Report attributes that are *tuples of conversation ids*, persisted as their length and nothing
#: else. ``conversations_without_messages`` is the shape that matters here: a report may be pasted
#: into a message, and the manifest is the one file in the cache that is meant to be shareable, so a
#: conversation id must not survive the trip. The count is what the warm report can honestly say.
REPORT_TUPLE_COUNT_KEYS = ("conversations_without_messages",)

REPORT_TIMESTAMP_KEYS = ("first_timestamp", "last_timestamp")

#: Every key a persisted report may hold. Validation rejects anything else, so a field added to the
#: report cannot start riding into the manifest by accident.
REPORT_KEYS = (
    frozenset(REPORT_COUNT_KEYS)
    | frozenset(REPORT_TUPLE_COUNT_KEYS)
    | frozenset(REPORT_TIMESTAMP_KEYS)
)

MANIFEST_KEYS = frozenset(
    {
        "cache_format_version",
        "schema_version",
        "source_fingerprint",
        "chunking_fingerprint",
        "embedding_fingerprint",
        "chunk_count",
        "vector_dimension",
        "created_at",
        "report",
    }
)


class ManifestError(ValueError):
    """The manifest is absent, unreadable or not the shape this format version defines."""


def _int_field(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestError(f"{key} is not a non-negative integer")
    return value


def _digest_field(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ManifestError(f"{key} is not a sha256 digest")
    return value


def _timestamp_field(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{key} is not a timestamp")
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ManifestError(f"{key} is not an ISO timestamp") from exc
    return value


def utc_now() -> str:
    """The one clock this module reads. Seconds resolution, UTC, so a manifest is comparable."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def report_aggregates(report: AccountImportReport) -> dict[str, Any]:
    """The identity-free aggregate of an import report, for the manifest."""
    payload: dict[str, Any] = {key: int(getattr(report, key)) for key in REPORT_COUNT_KEYS}
    for key in REPORT_TUPLE_COUNT_KEYS:
        payload[key] = len(getattr(report, key) or ())
    for key in REPORT_TIMESTAMP_KEYS:
        value = getattr(report, key)
        # ISO-8601 with the ``T``: a persisted value must not depend on how a report happens to
        # print itself, and a digest over a separator is a digest over nothing.
        payload[key] = value.isoformat() if value is not None else None
    return payload


def _validate_report(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ManifestError("report is not an object")
    if set(payload) != REPORT_KEYS:
        raise ManifestError("report has unexpected keys")
    validated: dict[str, Any] = {
        key: _int_field(payload, key) for key in (*REPORT_COUNT_KEYS, *REPORT_TUPLE_COUNT_KEYS)
    }
    for key in REPORT_TIMESTAMP_KEYS:
        value = payload.get(key)
        if value is None:
            validated[key] = None
        else:
            validated[key] = _timestamp_field(payload, key)
    return validated


@dataclass(frozen=True)
class IndexManifest:
    """What a cache entry is, in versions, digests and counts.

    This object is the *only* thing written about an account besides the vectors themselves, and it
    is asserted by tests to carry nothing that names a person, a conversation or a message.
    """

    cache_format_version: int
    schema_version: int
    source_fingerprint: str
    chunking_fingerprint: str
    embedding_fingerprint: str
    chunk_count: int
    vector_dimension: int
    created_at: str
    report: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cache_format_version": self.cache_format_version,
            "schema_version": self.schema_version,
            "source_fingerprint": self.source_fingerprint,
            "chunking_fingerprint": self.chunking_fingerprint,
            "embedding_fingerprint": self.embedding_fingerprint,
            "chunk_count": self.chunk_count,
            "vector_dimension": self.vector_dimension,
            "created_at": self.created_at,
            "report": dict(self.report),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "IndexManifest":
        """Strictly. A manifest this function cannot fully read is a reason to rebuild, not a
        reason to guess — every field below is either used to decide validity or shown to a user."""
        if not isinstance(payload, Mapping):
            raise ManifestError("not an object")
        missing = MANIFEST_KEYS - set(payload)
        if missing:
            raise ManifestError(f"missing field(s): {', '.join(sorted(missing))}")
        extra = set(payload) - MANIFEST_KEYS
        if extra:
            raise ManifestError(f"unexpected field(s): {', '.join(sorted(extra))}")
        return cls(
            cache_format_version=_int_field(payload, "cache_format_version"),
            schema_version=_int_field(payload, "schema_version"),
            source_fingerprint=_digest_field(payload, "source_fingerprint"),
            chunking_fingerprint=_digest_field(payload, "chunking_fingerprint"),
            embedding_fingerprint=_digest_field(payload, "embedding_fingerprint"),
            chunk_count=_int_field(payload, "chunk_count"),
            vector_dimension=_int_field(payload, "vector_dimension"),
            created_at=_timestamp_field(payload, "created_at"),
            report=_validate_report(payload.get("report")),
        )

    def age_seconds(self, *, now: datetime | None = None) -> float:
        """How long ago the entry was written. Never used to *invalidate*, only to report."""
        created = datetime.fromisoformat(self.created_at)
        current = now or datetime.now(timezone.utc)
        return max(0.0, (current - created).total_seconds())


READY_KEYS = frozenset({"cache_format_version", "chunk_count", "source_fingerprint"})


@dataclass(frozen=True)
class ReadyMarker:
    """The completeness statement, written last and only when the entry is whole.

    It repeats the source fingerprint and the chunk count so a half-promoted directory — a new
    manifest beside old vectors, say — is refused rather than loaded with a plausible-looking
    manifest and somebody else's vectors.
    """

    cache_format_version: int
    chunk_count: int
    source_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "cache_format_version": self.cache_format_version,
            "chunk_count": self.chunk_count,
            "source_fingerprint": self.source_fingerprint,
        }

    @classmethod
    def for_manifest(cls, manifest: IndexManifest) -> "ReadyMarker":
        return cls(
            cache_format_version=manifest.cache_format_version,
            chunk_count=manifest.chunk_count,
            source_fingerprint=manifest.source_fingerprint,
        )

    @classmethod
    def from_dict(cls, payload: Any) -> "ReadyMarker":
        if not isinstance(payload, Mapping):
            raise ManifestError("not an object")
        if set(payload) != READY_KEYS:
            raise ManifestError("does not hold the fields a marker must hold")
        return cls(
            cache_format_version=_int_field(payload, "cache_format_version"),
            chunk_count=_int_field(payload, "chunk_count"),
            source_fingerprint=_digest_field(payload, "source_fingerprint"),
        )


# ---------------------------------------------------------------------------------------------
# Where an entry lives
# ---------------------------------------------------------------------------------------------


def _sanitize(name: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in name)


def cache_leaf(account_dir: Path) -> str:
    """The directory name for one account: its name, plus a digest of its resolved path.

    The digest is what keeps two accounts called ``account`` in different directories apart — a
    collision would serve one account's history for another's question. ``normcase`` makes the
    identity compare like the filesystem does, so a differently-cased path is the same account.
    """
    resolved = os.path.normcase(str(Path(account_dir).resolve()))
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
    return f"{_sanitize(Path(account_dir).name) or 'account'}-{digest}"


def account_cache_dir(account_dir: Path, index_dir: Path | None = None) -> Path:
    """Where this account's entry lives. ``index_dir`` overrides the account data area."""
    root = Path(index_dir) if index_dir is not None else DEFAULT_INDEX_ROOT
    return root / cache_leaf(account_dir)


def cleanup_staging(cache_dir: Path) -> int:
    """Delete leftover staging and trash directories for this account; returns how many.

    Called before anything else reads or writes: a run interrupted mid-build must not leave rubble
    that a later run has to reason about. Scoped to this account's leaf so a second account's
    in-flight build is not disturbed.
    """
    root = Path(cache_dir).parent
    if not root.is_dir():
        return 0
    removed = 0
    for infix in (STAGING_INFIX, TRASH_INFIX):
        for path in sorted(root.glob(f"{cache_dir.name}{infix}*")):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
    return removed


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------

#: The four things a startup can say. ``REBUILD`` is a deliberate request (``--rebuild-index``),
#: ``INVALID`` an entry that exists but cannot be trusted — the two are separate because the reader
#: of a log line needs to know which happened.
HIT = "HIT"
MISS = "MISS"
INVALID = "INVALID"
REBUILD = "REBUILD"


@dataclass(frozen=True)
class ExpectedIndex:
    """What a stored entry must match to be usable, computed from *this* run's configuration."""

    source_fingerprint: str | None
    chunking_fingerprint: str
    schema_version: int
    embedder: Any


@dataclass(frozen=True)
class Inspection:
    """The verdict on a stored entry, with a reason when it is not a hit."""

    outcome: str
    reason: str = ""
    manifest: IndexManifest | None = None

    @property
    def is_hit(self) -> bool:
        return self.outcome == HIT and self.manifest is not None


def _read_manifest(cache_dir: Path) -> IndexManifest:
    """Read and validate the manifest, or say exactly why it cannot be used.

    The reasons distinguish *absent* from *unreadable* from *rejected*: one means an interrupted
    build, one means a damaged file, one means the entry was written by a different shape of this
    module. All three end in a rebuild, but a log line that cannot tell them apart cannot be acted on.
    """
    path = Path(cache_dir) / MANIFEST_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError("no manifest") from exc
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise ManifestError("manifest unreadable") from exc
    try:
        return IndexManifest.from_dict(payload)
    except ManifestError as exc:
        raise ManifestError(f"manifest rejected ({exc})") from exc


def _read_ready(cache_dir: Path) -> ReadyMarker:
    path = Path(cache_dir) / READY_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError("missing READY marker") from exc
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise ManifestError("READY marker unreadable") from exc
    try:
        return ReadyMarker.from_dict(payload)
    except ManifestError as exc:
        raise ManifestError(f"READY marker rejected ({exc})") from exc


def _artifact_problem(cache_dir: Path) -> str:
    """The first thing wrong with the two artifacts an entry cannot exist without."""
    for name in (VECTORS_FILENAME, DOCSTORE_FILENAME):
        path = Path(cache_dir) / name
        try:
            size = path.stat().st_size
        except OSError:
            return f"{name} is missing"
        if size <= 0:
            return f"{name} is empty"
    return ""


def inspect(cache_dir: Path, *, expected: ExpectedIndex, rebuild: bool = False) -> Inspection:
    """Decide whether a stored entry may be loaded. Never raises, never loads half of anything.

    The order is deliberate: the cheapest and most decisive checks first, the source-tree
    fingerprint last, because that is the only one that touches the filesystem beyond the entry.
    """
    if rebuild:
        return Inspection(REBUILD, "an explicit rebuild was requested")
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return Inspection(MISS, "no index cache for this account")
    try:
        manifest = _read_manifest(cache_dir)
    except ManifestError as exc:
        return Inspection(INVALID, str(exc))
    if manifest.cache_format_version != CACHE_FORMAT_VERSION:
        return Inspection(
            INVALID,
            f"cache format version changed ({manifest.cache_format_version} -> {CACHE_FORMAT_VERSION})",
        )
    if manifest.schema_version != expected.schema_version:
        return Inspection(
            INVALID,
            f"document projection changed ({manifest.schema_version} -> {expected.schema_version})",
        )
    try:
        ready = _read_ready(cache_dir)
    except ManifestError as exc:
        return Inspection(INVALID, str(exc))
    if ready != ReadyMarker.for_manifest(manifest):
        return Inspection(INVALID, "READY marker does not match the manifest")
    problem = _artifact_problem(cache_dir)
    if problem:
        return Inspection(INVALID, problem)
    if manifest.chunking_fingerprint != expected.chunking_fingerprint:
        return Inspection(INVALID, "chunking configuration changed")
    if manifest.embedding_fingerprint != embedding_fingerprint(
        expected.embedder, dimension=manifest.vector_dimension
    ):
        return Inspection(INVALID, "embedding configuration changed")
    if expected.source_fingerprint is None:
        return Inspection(INVALID, "source export tree unreadable")
    if manifest.source_fingerprint != expected.source_fingerprint:
        return Inspection(INVALID, "source export tree changed")
    return Inspection(HIT, "", manifest)


def verify_store(vector_store: Any, manifest: IndexManifest) -> str:
    """Check a *loaded* store against the manifest. Returns ``""`` when they agree.

    The manifest is a claim; this is the measurement. A count or a dimension that disagrees means
    the entry is not what it says it is — a truncated copy, a mixed promotion — and answering from
    it would cite chunks that are not there.
    """
    index = getattr(vector_store, "index", None)
    count = int(getattr(index, "ntotal", -1))
    if count != manifest.chunk_count:
        return f"chunk count mismatch (index holds {count}, manifest claims {manifest.chunk_count})"
    dimension = int(getattr(index, "d", -1))
    if dimension != manifest.vector_dimension:
        return (
            f"vector dimension mismatch (index is {dimension}, manifest claims "
            f"{manifest.vector_dimension})"
        )
    mapping = getattr(vector_store, "index_to_docstore_id", None)
    if isinstance(mapping, Mapping) and len(mapping) != count:
        return f"documents artifact incomplete ({len(mapping)} of {count} chunks bound)"
    return ""


def load_store(cache_dir: Path, embedder: Any) -> Any:
    """Load a persisted FAISS store. The one place the pickle is unpickled.

    ``allow_dangerous_deserialization=True`` is required and is not taken lightly: the file is
    unpickled, and an attacker who can write into the index directory could execute code through it.
    That directory is local, written by this tool, and holds the same private history the tool is
    already trusted with — but it is a real trust boundary, and it is why the index directory must
    never be shared, synced from elsewhere, or pointed at a directory someone else can write.
    """
    from langchain_community.vectorstores import FAISS

    return FAISS.load_local(
        folder_path=str(cache_dir),
        embeddings=embedder,
        allow_dangerous_deserialization=True,
    )


# ---------------------------------------------------------------------------------------------
# Writing: staging, then one rename
# ---------------------------------------------------------------------------------------------


def save_index(cache_dir: Path, *, vector_store: Any, manifest: IndexManifest) -> Path:
    """Write a complete entry beside ``cache_dir`` and promote it with a rename.

    Nothing is ever written into the final directory: the build lands in a staging directory that no
    reader looks at, the READY marker is written **last** (its presence is the statement that the
    entry is whole), and only then does a rename make it the entry. A run that dies at any point
    before that rename leaves either nothing or a directory without a READY marker, both of which
    the next run refuses and cleans up.
    """
    cache_dir = Path(cache_dir)
    root = cache_dir.parent
    root.mkdir(parents=True, exist_ok=True)
    cleanup_staging(cache_dir)
    staging = root / f"{cache_dir.name}{STAGING_INFIX}{uuid4().hex[:8]}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        vector_store.save_local(folder_path=str(staging))
        problem = _artifact_problem(staging)
        if problem:
            raise OSError(f"a written index is incomplete: {problem}")
        (staging / MANIFEST_FILENAME).write_text(
            canonical_json(manifest.as_dict()), encoding="utf-8"
        )
        # Last: everything above must be on disk before this file exists.
        (staging / READY_FILENAME).write_text(
            canonical_json(ReadyMarker.for_manifest(manifest).as_dict()), encoding="utf-8"
        )
        _promote(staging, cache_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return cache_dir


#: A rename can fail *transiently* on Windows while something (an indexer, a search service, a virus
#: scanner) still holds a handle on a file written a millisecond ago. The wait is bounded and tiny;
#: what is being protected against is a cache write that gives up and costs a rebuild on every run.
RENAME_ATTEMPTS = 5
RENAME_DELAY_SECONDS = 0.04


def _replace(src: Path, dst: Path) -> None:
    """``os.replace``, retried briefly against a transient sharing violation."""
    last: OSError | None = None
    for attempt in range(RENAME_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except OSError as exc:
            last = exc
            if attempt + 1 < RENAME_ATTEMPTS:
                time.sleep(RENAME_DELAY_SECONDS)
    assert last is not None
    raise last


def _promote(staging: Path, cache_dir: Path) -> None:
    """Move a finished staging directory into place, keeping the old entry until the last moment.

    The old entry survives until the new one has landed: it is renamed aside, not deleted, and is
    only removed once the promotion has succeeded — so a failure here costs a rebuild, never the
    index that was working a moment ago.
    """
    trash = cache_dir.with_name(f"{cache_dir.name}{TRASH_INFIX}{uuid4().hex[:8]}")
    moved_old = False
    if cache_dir.exists():
        shutil.rmtree(trash, ignore_errors=True)
        _replace(cache_dir, trash)
        moved_old = True
    try:
        _replace(staging, cache_dir)
    except OSError:
        # Put the previous entry back rather than leaving the account with no index at all.
        if moved_old and not cache_dir.exists():
            try:
                _replace(trash, cache_dir)
            except OSError:
                pass
        raise
    if moved_old:
        shutil.rmtree(trash, ignore_errors=True)


# ---------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------


def _seconds(milliseconds: int) -> str:
    return f"{milliseconds / 1000:.2f} s"


def human_age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} d"


@dataclass(frozen=True)
class IndexStatus:
    """What the index cache did this run, in aggregate numbers only.

    This is the object every log line about the cache is rendered from, which is why it holds no
    path, no account name and nothing from the history: a status line outlives the terminal it was
    printed in.
    """

    outcome: str
    reason: str = ""
    chunk_count: int = 0
    vector_dimension: int = 0
    cache_age_seconds: float | None = None
    cache_format_version: int = CACHE_FORMAT_VERSION
    schema_version: int = PROJECTION_VERSION
    cache_written: bool = False
    #: True only when this run embedded the chunks itself. A ``HIT`` is a run that embedded nothing.
    embedded: bool = False
    fingerprint_ms: int = 0
    load_ms: int = 0
    import_ms: int = 0
    embed_ms: int = 0
    save_ms: int = 0
    total_ms: int = 0

    @property
    def is_hit(self) -> bool:
        return self.outcome == HIT

    def lines(self) -> list[str]:
        """Two lines: the verdict, then where the time went.

        The second line exists because a warm start is not instant either — the fingerprint walk,
        the FAISS load and the lazy BM25 rebuild all cost something, and a later phase that wants to
        make a warm start faster needs to know which of them to attack.
        """
        details: list[str] = []
        if self.outcome == HIT:
            details.append(f"{self.chunk_count} chunks")
            details.append("0 chunks embedded")
            details.append(f"cache v{self.cache_format_version}")
            details.append(f"projection v{self.schema_version}")
            if self.cache_age_seconds is not None:
                details.append(f"age {human_age(self.cache_age_seconds)}")
            details.append(f"loaded in {_seconds(self.load_ms)}")
        else:
            details.append(f"{self.chunk_count} chunks")
            details.append(f"{self.chunk_count} chunks embedded")
            details.append(f"projection v{self.schema_version}")
            details.append("cache written" if self.cache_written else "cache NOT written")
        head = f"INDEX CACHE {self.outcome}"
        if self.reason:
            head += f" ({self.reason})"
        timings = [f"fingerprint {self.fingerprint_ms} ms"]
        if self.load_ms:
            timings.append(f"load {self.load_ms} ms")
        if self.import_ms:
            timings.append(f"import {self.import_ms} ms")
        if self.embed_ms:
            timings.append(f"embed {self.embed_ms} ms")
        if self.save_ms:
            timings.append(f"save {self.save_ms} ms")
        timings.append(f"total {self.total_ms} ms")
        return [f"{head} ({', '.join(details)})", "INDEX CACHE TIMING (" + ", ".join(timings) + ")"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "reason": self.reason,
            "chunk_count": self.chunk_count,
            "vector_dimension": self.vector_dimension,
            "cache_age_seconds": self.cache_age_seconds,
            "cache_format_version": self.cache_format_version,
            "schema_version": self.schema_version,
            "cache_written": self.cache_written,
            "embedded": self.embedded,
            "timings_ms": {
                "fingerprint": self.fingerprint_ms,
                "load": self.load_ms,
                "import": self.import_ms,
                "embed": self.embed_ms,
                "save": self.save_ms,
                "total": self.total_ms,
            },
        }


def warm_report(manifest: IndexManifest, layout: AccountLayout) -> AccountImportReport:
    """The aggregate part of an import report, rebuilt from the manifest — and nothing invented.

    A warm start does not parse the exports, so it cannot know what each conversation contained.
    It rebuilds what *was* persisted (counts, coverage, the two independent ways to be PARTIAL) and
    flags itself, so a caller prints "this detail was not kept" instead of an empty table that reads
    like an account with no conversations in it.

    Shard completeness is taken from the **live** tree rather than from the manifest: enumerating
    shard directories and reading listings costs milliseconds, and it means a shard that appeared or
    vanished since the build is reported as it is now, not as it was.
    """
    payload = manifest.report
    notes = list(layout.notes)
    # The count survives the cache; the ids do not, and must not be invented to fill the field. The
    # note carries the number so an empty id list can never be read as "no conversation was empty".
    without_messages = int(payload.get("conversations_without_messages", 0))
    if without_messages:
        notes.append(
            f"{without_messages} discovered conversation(s) produced no messages in the exported "
            "shards (counted from the index cache; the ids were not persisted)"
        )
    report = AccountImportReport(
        shards_detected=tuple(layout.shards_detected),
        shards_exported=tuple(sorted({shard_stem(export.shard) for export in layout.exports})),
        conversations_discovered=int(payload.get("conversations_discovered", 0)),
        conversations_imported=int(payload.get("conversations_imported", 0)),
        messages_received=int(payload.get("messages_received", 0)),
        messages_kept=int(payload.get("messages_kept", 0)),
        duplicates_removed=int(payload.get("duplicates_removed", 0)),
        skipped_messages=int(payload.get("skipped_messages", 0)),
        undedupeable=int(payload.get("undedupeable", 0)),
        first_timestamp=_parse_timestamp(payload.get("first_timestamp")),
        last_timestamp=_parse_timestamp(payload.get("last_timestamp")),
        filtered_conversations=int(layout.filtered_conversations),
        notes=tuple(notes),
        reconstructed_from_cache=True,
    )
    exported = set(report.shards_exported)
    report.missing_shards = tuple(sorted(s for s in report.shards_detected if s not in exported))
    return report


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
