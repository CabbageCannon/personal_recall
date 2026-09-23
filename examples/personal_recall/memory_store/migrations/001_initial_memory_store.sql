-- Phase 22A migration 001: the canonical structured memory store.
--
-- This file is the **only** definition of the schema. Nothing at runtime issues a `CREATE TABLE`;
-- the application opens a database, asks `schema_migrations` what version it is, and refuses to run
-- against a version it does not know (§11, §24 Test E). A future Phase 22B adds `002_…pgvector.sql`
-- beside this one — it does not edit this file.
--
-- Two things this schema deliberately is NOT:
--
--   * **Not a second data model.** Every column is a field `MemoryEvent` / `MemoryChunk` already
--     carries. Where this schema could have "improved" something — normalising a label, deriving a
--     better timestamp, resolving a speaker — it does not, because the Python layer is the contract
--     and a second opinion here would make the two disagree (§7).
--   * **Not the retrieval index.** There is no embedding and no vector column; FAISS and BM25 are
--     untouched by this phase and the store is not on the retrieval path (§3, §20).
--
-- Timestamps: the contract's chat times are `TIMESTAMP WITHOUT TIME ZONE`, holding the **local wall
-- clock** that `memory.weflow._timestamp` produced (epoch -> local -> naive). `TIMESTAMPTZ` was
-- measured and rejected: the same stored value read back under `Asia/Shanghai` moves by 8 hours,
-- because the naive contract value has no zone to be interpreted in. Under `TIMESTAMP` the wall clock
-- survives every session timezone unchanged. The original instant is not lost — `source_epoch` keeps
-- the exporter's own `createTime` verbatim. Operational times (`updated_at`, `applied_at`,
-- `last_sync_at`) *are* real instants and are `TIMESTAMPTZ`.

-- ---------------------------------------------------------------------------------------------
-- Conversations: one row per stable talker identity (§7.2).
-- ---------------------------------------------------------------------------------------------
CREATE TABLE conversations (
    -- The talker (a wxid, or `…@chatroom`). The project's existing stable identity, reused verbatim
    -- rather than replaced by a surrogate key, so a citation, a FAISS document and a database row all
    -- name the same conversation the same way (§7.2).
    conversation_id   TEXT PRIMARY KEY,
    source_type       TEXT NOT NULL,
    conversation_type TEXT NOT NULL CHECK (conversation_type IN ('direct', 'group')),
    -- The human-readable name, or NULL when the export offered none that is actually a name
    -- (`memory.labels.usable_name` — an exporter that copies the talker into `displayName` yields
    -- NULL here, not a wxid in a column that promises a label).
    display_label     TEXT,
    -- What the exporter offered, verbatim, whether usable or not. Kept because "the export said this"
    -- and "this is safe to show" are different facts and collapsing them loses the first.
    display_name      TEXT NOT NULL DEFAULT '',
    event_count       INTEGER NOT NULL DEFAULT 0,
    chunk_count       INTEGER NOT NULL DEFAULT 0,
    first_event_at    TIMESTAMP,
    last_event_at     TIMESTAMP,
    metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------------------------
-- People: one row per trustworthy source identity (§7.3).
--
-- `person_id` is a deterministic digest of the source identity, never a sequence, so a rebuild of
-- this store from the same raw export produces byte-identical keys and the golden parity test
-- (incremental vs fresh bootstrap) is a comparison of real values rather than of renumberings.
--
-- `source_identity` is the raw wxid. It is stored **here and nowhere else**: this database is a
-- local private store (§36), and dropping the identity to be safe would make `people` unjoinable to
-- the exports that produced it. It must never be copied into a log, a README, PROJECT_STATUS, a test
-- snapshot or a commit.
-- ---------------------------------------------------------------------------------------------
CREATE TABLE people (
    person_id       TEXT PRIMARY KEY,
    source_type     TEXT NOT NULL,
    source_identity TEXT NOT NULL,
    -- 'group_member' when an explicit per-message sender field stated this identity;
    -- 'direct_peer' when it is a direct conversation's own talker, which the export states by naming
    -- the file and which 134 of 140 real direct conversations corroborate from the message field too.
    person_kind     TEXT NOT NULL CHECK (person_kind IN ('group_member', 'direct_peer')),
    display_name    TEXT,
    first_seen_at   TIMESTAMP,
    last_seen_at    TIMESTAMP,
    event_count     INTEGER NOT NULL DEFAULT 0,
    conversation_count INTEGER NOT NULL DEFAULT 0,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (source_type, source_identity)
);

-- ---------------------------------------------------------------------------------------------
-- Membership (§7.4). No `role` column: membership *means* "spoke here as a non-self participant",
-- and the self speaker has no source identity to be a person with (§9) — a column holding the same
-- constant on every row would document nothing and could drift from `memory_events.speaker_role`.
--
-- Two label columns, because they are two different facts and Phase 20.9 is the reason:
--
--   * `display_name` — the person's own name *within this conversation*, or NULL when the export
--     offered none that is a name. Account-wide this is ambiguous (one person renders differently in
--     two conversations), so it is recorded where it is unambiguous, and `people.display_name`
--     is later resolved from these rows.
--   * `display_label` — the label the evidence actually renders for them here, i.e. what
--     `MemoryEvent.sender_name` says. It differs from `display_name` exactly when two members of one
--     conversation share a name, and that is the case the qualifier exists for: `小王（成员A）`.
--
-- This table is also what makes `people` maintainable. Every aggregate a person has — first seen,
-- last seen, how many events, how many conversations — is an aggregate over their memberships, so
-- recomputing them after a sync touches a few thousand rows instead of re-aggregating 1.6M events.
-- That is not a shortcut: memberships partition a person's events by conversation, so the aggregate
-- over memberships *is* the aggregate over events.
-- ---------------------------------------------------------------------------------------------
CREATE TABLE conversation_people (
    conversation_id TEXT NOT NULL REFERENCES conversations (conversation_id) ON DELETE CASCADE,
    person_id       TEXT NOT NULL REFERENCES people (person_id) ON DELETE CASCADE,
    -- Why this person counts as a participant *here*, which is what makes `people.person_kind`
    -- recomputable rather than remembered: a person is a `group_member` exactly while some
    -- membership says so. Keeping the kind only on `people` would make it a cached fact that an
    -- incremental sync could fail to revise — and the case that breaks is a member who leaves the
    -- group that named them, which is precisely the case a sync exists to handle.
    person_kind     TEXT NOT NULL CHECK (person_kind IN ('group_member', 'direct_peer')),
    display_name    TEXT,
    display_label   TEXT,
    event_count     INTEGER NOT NULL DEFAULT 0,
    first_seen_at   TIMESTAMP,
    last_seen_at    TIMESTAMP,
    PRIMARY KEY (conversation_id, person_id)
);

-- ---------------------------------------------------------------------------------------------
-- Events (§7.5): a faithful projection of `MemoryEvent`. `event_id` is the project's own id, kept
-- as primary key so a citation resolves to exactly one row.
-- ---------------------------------------------------------------------------------------------
CREATE TABLE memory_events (
    event_id          TEXT PRIMARY KEY,
    conversation_id   TEXT NOT NULL REFERENCES conversations (conversation_id) ON DELETE CASCADE,
    event_time        TIMESTAMP NOT NULL,
    speaker_role      TEXT NOT NULL DEFAULT '',
    -- NULL for a self message and for a message whose export stated no identity — a real absence, not
    -- an invented person (§8, §9).
    speaker_person_id TEXT REFERENCES people (person_id) ON DELETE SET NULL,
    -- Raw fields, never merged: `speaker_id` is data, `speaker_display` is a candidate, `sender_name`
    -- is the rendered label. Three facts, three columns (§24).
    speaker_id        TEXT NOT NULL DEFAULT '',
    speaker_display   TEXT NOT NULL DEFAULT '',
    sender_name       TEXT NOT NULL,
    text              TEXT NOT NULL,
    message_type      TEXT NOT NULL DEFAULT 'text',
    reply_to          TEXT,
    source_type       TEXT NOT NULL,
    -- The exporter's own `createTime`, in seconds, exactly as received. The wall clock above is what
    -- the contract renders; this is what the source actually said, and it is the reason the wall clock
    -- can never be ambiguous.
    source_epoch      BIGINT,
    source_position   INTEGER,
    source_server_id  TEXT,
    source_local_id   TEXT,
    metadata          JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- ---------------------------------------------------------------------------------------------
-- Chunks (§7.6). No embedding vector — that is Phase 22B and is not created here in advance.
-- ---------------------------------------------------------------------------------------------
CREATE TABLE memory_chunks (
    chunk_id           TEXT PRIMARY KEY,
    conversation_id    TEXT NOT NULL REFERENCES conversations (conversation_id) ON DELETE CASCADE,
    -- Position within the conversation's render. The project's chunk ids are positional
    -- (`<conversation>-session-0007`), so the index is part of the identity and a re-render that
    -- shifts it produces a different chunk — which is exactly what makes whole-conversation
    -- replacement the correct update unit (§16).
    chunk_index        INTEGER NOT NULL,
    start_time         TIMESTAMP NOT NULL,
    end_time           TIMESTAMP NOT NULL,
    n_events           INTEGER NOT NULL,
    n_chars            INTEGER NOT NULL,
    text               TEXT NOT NULL,
    -- sha256 of `text`: the digest the incremental index already keys vector reuse on, so the store
    -- and the index agree about what "this chunk did not change" means.
    text_hash          TEXT NOT NULL,
    projection_version INTEGER NOT NULL,
    metadata           JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- ---------------------------------------------------------------------------------------------
-- Evidence linkage (§7.7). The chunk already carries `event_ids` in memory; making it a relation
-- is what lets a future temporal or evidence query be a join instead of a scan of every chunk's
-- embedded id list.
-- ---------------------------------------------------------------------------------------------
CREATE TABLE chunk_events (
    chunk_id TEXT NOT NULL REFERENCES memory_chunks (chunk_id) ON DELETE CASCADE,
    event_id TEXT NOT NULL REFERENCES memory_events (event_id) ON DELETE CASCADE,
    ordinal  INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, event_id),
    -- Two events cannot hold one position within a chunk: a chunk whose ordinals repeat has no
    -- recoverable order, and "the evidence in order" is the whole point of this table.
    UNIQUE (chunk_id, ordinal)
);

-- ---------------------------------------------------------------------------------------------
-- Store state (§18, §19): the row that answers "which account snapshot does this database hold?".
--
-- One row. `store_name` exists so a store identifies *itself* and a `--rebuild` can prove it is
-- pointed at the intended target rather than at whatever database the URL happened to name (§37).
-- It holds counts and fingerprints only — never a message, a name or a wxid.
-- ---------------------------------------------------------------------------------------------
CREATE TABLE memory_store_state (
    store_name           TEXT PRIMARY KEY,
    store_schema_version INTEGER NOT NULL,
    projection_version   INTEGER NOT NULL,
    chunking_fingerprint TEXT NOT NULL,
    -- The same value the Phase 21A/21B index manifest records for the same export tree, so the
    -- FAISS generation and the database generation can be compared instead of assumed equal (§18).
    source_fingerprint   TEXT NOT NULL,
    store_generation     BIGINT NOT NULL DEFAULT 0,
    conversation_count   INTEGER NOT NULL DEFAULT 0,
    event_count          BIGINT NOT NULL DEFAULT 0,
    chunk_count          INTEGER NOT NULL DEFAULT 0,
    people_count         INTEGER NOT NULL DEFAULT 0,
    first_event_at       TIMESTAMP,
    last_event_at        TIMESTAMP,
    last_sync_at         TIMESTAMPTZ,
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------------------------
-- Source inventory: which export files this generation was built from (§18).
--
-- The store has to be able to answer "what changed since *my* generation", and the Phase 21B sync
-- checkpoint cannot answer that: it describes one generation, it is advanced by whichever consumer
-- runs, and an index run and a store sync would overwrite each other's base. So the store keeps its
-- own, and `plan_store_sync` diffs this table against the tree as it is now.
--
-- A digest of this table is `memory_store_state.source_fingerprint`, computed by the same function
-- the index manifest uses, so "is the database at the same generation as the index" stays a
-- comparison of two recorded values rather than an assumption.
--
-- It carries a conversation id inside `relative_path`, which is why it is a table of its own and not
-- a field on `memory_store_state`: that row is summary metadata, and the identity belongs with the
-- rest of it — in the private database, and never in a log or a report.
-- ---------------------------------------------------------------------------------------------
CREATE TABLE source_inventory (
    relative_path   TEXT PRIMARY KEY,
    size            BIGINT NOT NULL,
    mtime_ns        BIGINT NOT NULL,
    kind            TEXT NOT NULL,
    shard           TEXT NOT NULL DEFAULT '',
    conversation_id TEXT NOT NULL DEFAULT ''
);

-- ---------------------------------------------------------------------------------------------
-- Indexes (§12). Exactly the ones a person / time / conversation filter needs, and no index over
-- any JSONB: an unqueried index is a write cost paid on all 1.6M rows for nothing.
-- ---------------------------------------------------------------------------------------------
CREATE INDEX memory_events_conversation_time_idx ON memory_events (conversation_id, event_time);
CREATE INDEX memory_events_person_time_idx       ON memory_events (speaker_person_id, event_time)
    WHERE speaker_person_id IS NOT NULL;
CREATE INDEX memory_events_time_idx              ON memory_events (event_time);
CREATE INDEX memory_chunks_conversation_time_idx ON memory_chunks (conversation_id, start_time);
CREATE INDEX memory_chunks_start_time_idx        ON memory_chunks (start_time);
CREATE INDEX conversation_people_person_idx      ON conversation_people (person_id, conversation_id);
CREATE INDEX chunk_events_ordinal_idx            ON chunk_events (chunk_id, ordinal);
CREATE INDEX memory_events_speaker_id_idx        ON memory_events (conversation_id, speaker_id);

-- Every foreign key needs an index on the referencing column, and PostgreSQL does not make one.
-- This one is not for queries — it is what makes the cascade fast when a conversation is replaced.
-- Found on the real account, not in a fixture: without it, `DELETE FROM conversations`, which
-- cascades to `memory_events`, made PostgreSQL sequentially scan the whole 1.6M-row `chunk_events`
-- table once *per deleted event*, because the primary key is `(chunk_id, event_id)` and cannot
-- answer "which links cite this event". Six minutes in, on eleven conversations.
CREATE INDEX chunk_events_event_idx              ON chunk_events (event_id);
