-- Phase 22B migration 002: dense retrieval inside the canonical store.
--
-- A new file rather than an edit to 001. An applied migration is a historical record — the database
-- that ran it, and the ones that will run 001 and then this, must end up in the same shape, and
-- rewriting 001 would make "which statements produced this database" unanswerable.
--
-- This is the phase where the store stops being only a structured view and starts answering
-- retrieval. It does not change what is stored about the *memory*: no column here describes a
-- message, a speaker or a conversation. It adds one vector and one index over it.

-- `vector` is provided by the pgvector distribution of the PostgreSQL image (see the compose file);
-- it is not in the stock image. Recorded here because "which extension is this schema built on" is
-- part of the schema, not part of a container's configuration.
CREATE EXTENSION IF NOT EXISTS vector;

-- 512 is bge-small-zh-v1.5's output width, which is what every vector in the existing FAISS index
-- was produced with. A migration cannot compute it, so it is written down — and the import refuses
-- to run when the index it is reading disagrees, which is the check that makes a hardcoded number
-- safe rather than merely convenient. Changing the embedding model is a rebuild of both layers.
ALTER TABLE memory_chunks
    ADD COLUMN embedding vector(512);

COMMENT ON COLUMN memory_chunks.embedding IS
    'The chunk''s dense vector, imported from the retrieval index rather than recomputed. NULL until '
    'the import runs, and NULL again for any chunk whose text no longer matches what was embedded.';

-- HNSW, because it is the index that stays good as rows are added: Phase 21B replaces whole
-- conversations, and IVFFlat's list assignment degrades under that kind of churn until it is
-- retrained. No benchmark was run to choose between them — that is a decision to defend with a
-- measurement if it ever matters, not to spend a sprint on.
--
-- Cosine distance, while the FAISS index is `IndexFlatL2`: the embedder normalizes, and for unit
-- vectors L2² = 2 − 2·cos, so the two orderings are identical. The acceptance measures that rather
-- than trusting it.
--
-- Created empty and maintained as the vectors arrive. Building it afterwards would be faster, and
-- would also make the schema depend on the order in which a migration and an import happened to run.
CREATE INDEX memory_chunks_embedding_idx ON memory_chunks
    USING hnsw (embedding vector_cosine_ops);

-- The store's own record of which schema its generation was written under moves with the schema.
-- Without this, `plan_store_sync` would see a store "built under different rules", refuse every
-- incremental update and ask for a full rebuild — for a migration that added one nullable column
-- and changed no stored fact. A migration brings the database *and* its recorded version forward.
UPDATE memory_store_state SET store_schema_version = 2;

