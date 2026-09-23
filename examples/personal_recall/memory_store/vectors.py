"""Loading the retrieval index's vectors into the store — without re-embedding anything.

Eighty-one thousand chunks took most of an hour to embed once. They are already on disk in the
persistent index, and they are the *same vectors for the same text* — the store and the index are two
projections of one memory layer. Recomputing them would be an hour of work that produces values equal
to the ones already stored, so this module's entire job is to move them across and prove that is what
happened.

The proof is not "the counts match". A count is satisfied by pairing every chunk with somebody else's
vector. The identity that actually holds the two layers together is the **text digest**: the index
records a digest of the exact string it embedded, the store records a digest of the exact string it
stores, and both are sha256 of the chunk text. A vector is transferred only when those two digests are
equal, and every mismatch is counted and reported instead of being written.

A chunk with no matching vector is *not* an error and not a guess: it is a chunk the index has never
seen, and it is embedded only if a caller supplies an embedder for it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: How many vectors to hand to one COPY. Big enough that the round trips disappear, small enough that
#: the biggest intermediate list stays modest next to a 512-float payload per row.
DEFAULT_BATCH = 4096


@dataclass(frozen=True)
class IndexVector:
    """One vector, with the identity of the text it was embedded for."""

    chunk_id: str
    conversation_id: str
    text_digest: str
    vector: tuple[float, ...]

    @property
    def dimension(self) -> int:
        return len(self.vector)


@dataclass(frozen=True)
class VectorImportReport:
    """What a vector import found and wrote. Counts and a dimension — never a chunk's text."""

    index_vectors: int = 0
    store_chunks: int = 0
    matched: int = 0
    #: Vectors whose chunk exists but whose stored text is not the text that was embedded. Never
    #: written: a vector for a different string is worse than no vector at all, because it looks like
    #: an answer.
    text_mismatch: int = 0
    #: Vectors for chunks the store does not hold — the index and the store are at different
    #: generations, which is a state to report rather than to import into.
    unknown_chunks: int = 0
    #: Store chunks the index has never seen. Embedded only when an embedder was supplied.
    missing: int = 0
    embedded: int = 0
    written: int = 0
    dimension: int = 0
    index_dimension: int = 0
    timings_ms: Mapping[str, int] = field(default_factory=dict)
    total_ms: int = 0

    @property
    def aligned(self) -> bool:
        """True when the store and the index describe the same chunks and the same vectors."""
        return (
            self.text_mismatch == 0
            and self.unknown_chunks == 0
            and self.missing == 0
            and self.dimension == self.index_dimension
        )

    @property
    def reembedded(self) -> int:
        """Chunks this run had to embed. The number the fast track exists to keep at zero."""
        return self.embedded

    def lines(self) -> list[str]:
        out = [
            f"vectors        : {self.index_vectors} in the index, {self.store_chunks} chunk(s) in "
            f"the store, {self.matched} matched by text digest",
            f"vectors        : {self.written} written, {self.embedded} embedded here, "
            f"dimension {self.dimension}",
        ]
        if self.text_mismatch or self.unknown_chunks or self.missing:
            out.append(
                f"vectors        : {self.text_mismatch} text mismatch(es), "
                f"{self.unknown_chunks} unknown chunk(s), {self.missing} chunk(s) with no vector"
            )
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "index_vectors": self.index_vectors,
            "store_chunks": self.store_chunks,
            "matched": self.matched,
            "text_mismatch": self.text_mismatch,
            "unknown_chunks": self.unknown_chunks,
            "missing": self.missing,
            "embedded": self.embedded,
            "written": self.written,
            "dimension": self.dimension,
            "index_dimension": self.index_dimension,
            "aligned": self.aligned,
            "timings_ms": dict(self.timings_ms),
            "total_ms": self.total_ms,
        }


class VectorImportError(RuntimeError):
    """The index and the store cannot be aligned, so importing would write wrong vectors."""


def read_index_vectors(vector_store: Any) -> tuple[IndexVector, ...]:
    """Every ``(chunk, vector)`` pair the retrieval index holds, in the index's own order.

    ``stored_documents`` is the authority on order and ``reconstruct_n`` returns the vector at a
    position — the two must be read together, because the docstore is a bag keyed by a random uuid and
    iterating it would pair every chunk with somebody else's embedding. Reusing Phase 21B's function
    rather than re-deriving it keeps that pairing rule in one place.
    """
    import incremental_index

    documents = incremental_index.stored_documents(vector_store)
    index = getattr(vector_store, "index", None)
    if index is None or not hasattr(index, "reconstruct_n"):
        raise VectorImportError("the stored index cannot be read back vector by vector")

    vectors: list[IndexVector] = []
    for position, document in enumerate(documents):
        metadata = getattr(document, "metadata", None) or {}
        chunk_id = str(metadata.get("memory_chunk_id") or "")
        if not chunk_id:
            raise VectorImportError("a stored document has no chunk identity")
        raw = index.reconstruct_n(position, 1)[0]
        vectors.append(
            IndexVector(
                chunk_id=chunk_id,
                conversation_id=str(metadata.get("conversation_id") or ""),
                text_digest=incremental_index.text_digest(document.page_content),
                vector=tuple(float(value) for value in raw),
            )
        )
    return tuple(vectors)


def import_vectors(
    store: Any,
    *,
    vectors: Sequence[IndexVector] | None = None,
    vector_store: Any = None,
    embedder: Any = None,
) -> VectorImportReport:
    """Move the index's vectors into the store, embedding only what the index does not have.

    ``vectors`` is injectable so a test can drive the whole reconciliation — including the mismatch
    and unknown-chunk branches, which are the ones that must never write — with no index on disk.
    """
    started = time.perf_counter()
    timings: dict[str, int] = {}

    mark = time.perf_counter()
    if vectors is None:
        if vector_store is None:
            raise VectorImportError("either vectors or a vector store must be given")
        vectors = read_index_vectors(vector_store)
    timings["read_index"] = _ms(mark)

    index_dimension = vectors[0].dimension if vectors else 0
    for vector in vectors:
        if vector.dimension != index_dimension:
            raise VectorImportError(
                "the stored index holds vectors of more than one dimension; refusing to import"
            )

    mark = time.perf_counter()
    hashes = store.chunk_text_hashes()
    timings["read_store"] = _ms(mark)

    matched: list[tuple[str, tuple[float, ...]]] = []
    text_mismatch = 0
    unknown = 0
    seen: set[str] = set()
    for vector in vectors:
        stored_hash = hashes.get(vector.chunk_id)
        if stored_hash is None:
            unknown += 1
            continue
        if stored_hash != vector.text_digest:
            text_mismatch += 1
            continue
        matched.append((vector.chunk_id, vector.vector))
        seen.add(vector.chunk_id)

    missing = [chunk_id for chunk_id in hashes if chunk_id not in seen]

    embedded = 0
    if missing and embedder is not None:
        mark = time.perf_counter()
        texts = store.chunk_texts(sorted(missing))
        ordered = [chunk_id for chunk_id in sorted(missing) if chunk_id in texts]
        fresh = list(embedder.embed_documents([texts[chunk_id] for chunk_id in ordered]))
        matched.extend(zip(ordered, (tuple(float(v) for v in row) for row in fresh)))
        embedded = len(ordered)
        timings["embed_missing"] = _ms(mark)

    mark = time.perf_counter()
    written = store.write_embeddings(matched, dimension=index_dimension) if matched else 0
    timings["write"] = _ms(mark)

    return VectorImportReport(
        index_vectors=len(vectors),
        store_chunks=len(hashes),
        matched=len(matched) - embedded,
        text_mismatch=text_mismatch,
        unknown_chunks=unknown,
        missing=len(missing),
        embedded=embedded,
        written=written,
        dimension=written and index_dimension or index_dimension,
        index_dimension=index_dimension,
        timings_ms=timings,
        total_ms=int((time.perf_counter() - started) * 1000),
    )


def _ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)
