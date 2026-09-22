"""Personal Recall — ask a question about your chat history and get answers with evidence.

This is the product-facing entry point: it answers with the configuration the project's
evaluation adopted (A11 = session chunking, hybrid RRF, no-rewrite, cited-narrow, k=20) and
prints the **evidence cards** behind the answer, so every claim can be traced back to the
original chat lines:

    python recall.py "我之前说的那个数据库最后到底用了没？"
    python recall.py "小王什么时候推荐我用 Supabase 的？" --json

Pipeline (identical to the evaluated reference):
    session chunking -> BGE + FAISS + BM25 (weighted RRF, pool 30 -> top k)
    -> no-rewrite workflow -> cited answer prompt -> answer + [来源 N] citations

Config notes: `--workflow rag` restores the stock LLM query rewrite. It is useful when a
question leans on earlier turns of a real conversation, but it makes retrieval
non-deterministic, which is why the evaluation runs without it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import dotenv
import index_cache
from quivr_core import Brain
from quivr_core.files.file import FileExtension
from quivr_core.llm import LLMEndpoint
from quivr_core.processor.registry import register_processor
from quivr_core.rag.entities.chat import ChatHistory
from quivr_core.rag.entities.config import (
    DefaultModelSuppliers,
    HybridConfig,
    LLMEndpointConfig,
    RetrievalConfig,
)

from evidence_cards import build_evidence_cards, cards_to_dict, render_cards
from groundedness import assess
from memory import (
    EXPORT_FILENAME_SUFFIX,
    AccountImportReport,
    MemoryChunk,
    SessionConfig,
    build_account_sessions,
    crossed_conversation_chunks,
    import_account,
    load_account_directory,
)
from memory.events import parse_txt_events
from memory.processor import (
    ConversationSessionProcessor,
    WeFlowSessionProcessor,
    session_documents,
)
from memory.senders import assign_sender_labels
from memory.sessions import build_sessions
from memory.shards import (
    DEFAULT_MERGED_CONVERSATION,
    discover_message_shards,
    discover_shard_files,
    load_and_merge,
)
from memory.weflow import parse_weflow_events
from run_baseline import register_answer_prompt, serialize_sources

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
#: The adopted embedding model, defined where the persistent index binds to it (see
#: ``index_cache.EMBEDDING_MODEL_PATH``). Kept as a module attribute because it is the product's
#: documented model identity, not merely an implementation detail.
EMBEDDING_MODEL_PATH = index_cache.EMBEDDING_MODEL_PATH

#: The repo-level `.env` (python-dotenv discovers it relative to the *calling file* by
#: default, which is fragile for an installed/relocated CLI), so resolve it explicitly.
ENV_PATH = BASE_DIR.parent.parent / ".env"

DEFAULT_CORPUS = DATA_DIR / "stress_chats.txt"
#: A11, the adopted product reference: widening the window from 10 to 20 slots measured
#: PASS 28 -> 31 on corpus v2 with unsupported claims still 0 and citation-grounding
#: failures 1 -> 0, for a 3.6 % latency cost. See PROJECT_STATUS.md, Phase 9.
DEFAULT_K = 20
DEFAULT_HYBRID_POOL = 30
DEFAULT_MAX_SESSION_CHARS = 900
#: A12 is the adopted product reference: the A11 prompt plus the speaker-attribution clause that
#: removed the cross-speaker fabrications (PROJECT_STATUS.md, Phase 12).
DEFAULT_ANSWER_PROMPT = "cited-attributed"

#: `FileExtension` has no json member, and `get_file_extension()` falls back to the raw suffix string;
#: the processor registry accepts string keys, so this is what a WeFlow corpus is registered under.
JSON_EXTENSION = ".json"

#: The adapter that reads a corpus, by name.
SOURCE_FORMATS = ("text", "weflow")


def render_groundedness(report) -> str:
    """Render the gold-free caveats. Called only in human-readable mode.

    Deliberately phrased as caveats, not verdicts: without the gold evidence lines the tool cannot
    know whether an answer is wrong, only whether it makes statements worth checking.
    """
    lines = [f"Groundedness: {report.summary_line()}"]
    warnings = report.warnings()
    if warnings:
        lines.append("")
        for warning in warnings:
            lines.append(f"  ! {warning}")
    for mismatch in report.citation_mismatches:
        lines.append("")
        lines.append(f"  citation: {mismatch['explanation']}")
        lines.append(f"    in sentence: {mismatch['sentence'][:110]}")
    for flag in report.attribution_flags:
        lines.append("")
        lines.append(f"  attribution: {flag['sentence'][:110]}")
        lines.append(
            f"    rests on {flag['matched_line_speaker']}: {flag['matched_line'][:100]}"
        )
    return "\n".join(lines)


def register_source_processors() -> None:
    """Register the retrieval-unit processor for every supported corpus format.

    The framework resolves a processor by the file's extension, so each corpus format needs its own
    registration. Both are registered unconditionally (a `.txt` corpus never touches the `.json`
    entry and vice versa), which keeps a run's behaviour determined by the file it is given.
    """
    register_processor(FileExtension.txt, ConversationSessionProcessor, override=True)
    # `get_file_extension()` returns the raw string ".json" because FileExtension has no json member;
    # the registry accepts plain strings, so no Core change is needed.
    register_processor(JSON_EXTENSION, WeFlowSessionProcessor, override=True)


def detect_source_format(corpus: Path) -> str:
    """Infer the corpus format from its extension."""
    if corpus.suffix.lower() == JSON_EXTENSION:
        return "weflow"
    return "text"


def count_corpus_events(corpus: Path, source_format: str) -> tuple[int, int, str]:
    """Return ``(n_events, n_skipped, detail)`` for a corpus, using the matching adapter."""
    raw = corpus.read_text(encoding="utf-8-sig", errors="replace")
    if source_format == "weflow":
        # A single export file is one conversation's messages, so this stream *is* the whole
        # conversation and labelling it here cannot split one across shards — which is exactly why the
        # same pass is wrong per shard in ``merge_shard_events`` (see ``memory.senders``). Nothing
        # counted below depends on the speaker; the labelling keeps the representation identical to
        # what ``WeFlowSessionProcessor`` renders for the same file, so the guard and the index agree.
        parsed = parse_weflow_events(json.loads(raw), conversation_id=corpus.stem)
        events = assign_sender_labels(parsed.events, corpus.stem)
        detail = (
            f"{len(events)} messages "
            f"(types: {dict(sorted(parsed.message_types.items()))}, "
            f"suppressed payloads: {parsed.suppressed_payloads})"
        )
        return len(events), parsed.skipped, detail
    parsed_txt = parse_txt_events(raw)
    return len(parsed_txt.events), parsed_txt.skipped_lines, f"{len(parsed_txt.events)} messages"


def corpus_files(corpus: Path) -> list[Path]:
    """The shard exports a corpus path refers to.

    A directory means "every shard export in here"; a file means that one shard. Keeping both shapes
    lets a single-shard run behave exactly as before.
    """
    if corpus.is_dir():
        return discover_shard_files(corpus)
    return [corpus]


def is_multi_shard(corpus: Path) -> bool:
    return corpus.is_dir()


def build_brain_from_chunks(
    chunks, llm_config: LLMEndpointConfig, name: str, origin: str, *, embedder: Any = None
) -> Brain:
    """Build a brain from already-built session chunks.

    Shared by both non-processor paths — a single conversation merged from several shards, and an
    entire account of many conversations — so document shape and metadata can only be defined once.

    Building from documents bypasses ``ProcessorBase.process_file``, so the metadata that step would
    have added has to be supplied here: without ``original_file_name`` the framework's document prompt
    refuses to render (`combine_documents` adds ``index`` itself, but not this). The
    ``"Filename: … Content: …"`` prefix is deliberately NOT applied — ``process_file`` only adds it when
    the inner metadata already carries the field, which is exactly why the processor path's chunk text
    has no prefix.

    ``embedder`` is injectable so a test can hand in a deterministic stub and prove that a warm start
    never calls it — the embedder is the expensive, model-backed part, and "fast" is not the claim
    being made about it.
    """
    documents = session_documents(chunks, skipped_source_lines=0)
    for index, document in enumerate(documents, start=1):
        document.metadata["chunk_index"] = index
        document.metadata["original_file_name"] = origin
    embedder = embedder or index_cache.build_embedder()
    # Mirror `Brain.from_files` exactly: obtain the loop and run on it WITHOUT closing it.
    # `asyncio.run()` closes the loop it creates, which leaves the main thread with no current loop
    # and makes the later synchronous `brain.ask()` fail with "no current event loop".
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        Brain.afrom_langchain_documents(
            name=name,
            langchain_documents=documents,
            llm=LLMEndpoint.from_config(llm_config),
            embedder=embedder,
        )
    )


def build_brain_from_vector_store(
    vector_store, llm_config: LLMEndpointConfig, name: str, embedder
) -> Brain:
    """Build a brain over an **already-built** vector store — the warm-start path.

    Deliberately not ``Brain.afrom_langchain_documents``: that calls ``aadd_documents``, which embeds
    every chunk again, which is the 80 minutes this whole phase exists to avoid. Assembling the brain
    around a loaded store touches no document and no vector.
    """
    return Brain(
        name=name,
        id=uuid4(),
        llm=LLMEndpoint.from_config(llm_config),
        embedder=embedder,
        vector_db=vector_store,
    )


def build_brain_from_events(
    events, llm_config: LLMEndpointConfig, name: str, origin: str, *, embedder: Any = None
) -> Brain:
    """Build a brain from one conversation's merged events (the Phase 18B path).

    Sessions are built over the *union* of the shards, so a conversation spanning two databases becomes
    one retrieval unit with the global latest timestamp rather than one unit per shard.
    """
    sessions = build_sessions(
        events,
        config=SessionConfig(max_chars=DEFAULT_MAX_SESSION_CHARS),
        conversation_id=DEFAULT_MERGED_CONVERSATION,
    )
    return build_brain_from_chunks(sessions, llm_config, name, origin, embedder=embedder)


def build_brain(corpus: Path, llm_config: LLMEndpointConfig) -> Brain:
    """Ingest one chat log as conversation sessions."""
    register_source_processors()
    with warnings.catch_warnings():
        # `FileExtension` has no json member, so the framework warns "extension isn't recognized.
        # Make sure you have registered a parser for .json" while building the file record — even
        # though the processor IS registered (under the string key the same function returns). The
        # warning is a false alarm for a JSON corpus and would otherwise tell a user their working
        # setup is broken, so it is suppressed for exactly that message and no other.
        warnings.filterwarnings(
            "ignore",
            message=r".*extension isn't recognized.*",
            category=UserWarning,
        )
        return Brain.from_files(
            name=f"personal_recall_{corpus.stem}",
            file_paths=[corpus],
            llm=LLMEndpoint.from_config(llm_config),
            embedder=index_cache.build_embedder(),
            processor_kwargs={
                "session_config": SessionConfig(max_chars=DEFAULT_MAX_SESSION_CHARS)
            },
        )


def build_llm_config(*, max_output_tokens: int, temperature: float) -> LLMEndpointConfig:
    """The product's LLM endpoint. Shared by the corpus path and the account path."""
    return LLMEndpointConfig(
        supplier=DefaultModelSuppliers.OPENAI,
        model="deepseek-v4-flash",
        llm_base_url="https://api.deepseek.com",
        env_variable_name="DEEPSEEK_API_KEY",
        max_context_tokens=20000,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )


def build_retrieval_config(
    llm_config: LLMEndpointConfig,
    *,
    k: int = DEFAULT_K,
    hybrid_pool: int = DEFAULT_HYBRID_POOL,
    workflow: str = "no-rewrite",
) -> RetrievalConfig:
    """The adopted retrieval configuration (A11/A12): hybrid RRF over a pool, narrowed to *k*.

    NOTE (framework ordering quirk): ``LLMEndpointConfig`` only resolves its API key when a
    ``RetrievalConfig`` is constructed (its ``__init__`` calls ``llm_config.set_api_key(force_reset=True)``).
    Build the retrieval config BEFORE the brain, or ``LLMEndpoint.from_config`` gets ``api_key=None``.
    """
    from run_baseline import build_workflow_config

    return RetrievalConfig(
        llm_config=llm_config,
        k=k,
        hybrid_config=HybridConfig(enabled=True, candidate_k=hybrid_pool),
        workflow_config=build_workflow_config(workflow),
    )


def build_session(
    corpus: Path,
    *,
    k: int = DEFAULT_K,
    hybrid_pool: int = DEFAULT_HYBRID_POOL,
    workflow: str = "no-rewrite",
    answer_prompt: str = DEFAULT_ANSWER_PROMPT,
    max_output_tokens: int = 8192,
    temperature: float = 0.0,
) -> tuple[Brain, RetrievalConfig]:
    """Register the prompt, build the LLM/retrieval config and ingest the corpus.

    Shared by the CLI and by `verify_product_parity.py`, so the parity check exercises the
    product's own construction path instead of a copy of it.
    """
    register_answer_prompt(answer_prompt)

    llm_config = build_llm_config(max_output_tokens=max_output_tokens, temperature=temperature)
    retrieval_config = build_retrieval_config(
        llm_config, k=k, hybrid_pool=hybrid_pool, workflow=workflow
    )

    if is_multi_shard(corpus):
        # Multi-shard: merge every shard's events BEFORE session building, so one conversation is one
        # retrieval unit with the global latest timestamp rather than one unit per shard.
        events, merge_report = load_and_merge(corpus_files(corpus))
        origin = "merged:" + "+".join(info.label for info in merge_report.shards)
        return (
            build_brain_from_events(events, llm_config, f"personal_recall_{corpus.name}", origin),
            retrieval_config,
        )

    return build_brain(corpus, llm_config), retrieval_config


def account_export_error(account_dir: Path) -> FileNotFoundError:
    """The one wording for "this export tree holds nothing to index".

    A function rather than a literal because it is raised from two places — the seam, before the
    embedding model is loaded, and the importer itself — and two spellings of the same failure would
    make a user's two attempts to diagnose it look like two different problems.
    """
    return FileNotFoundError(
        f"{account_dir} contains no shard export directories holding "
        f"'{{talker}}{EXPORT_FILENAME_SUFFIX}' files"
    )


def import_account_directory(
    account_dir: Path,
    *,
    shard_dir: Path | None = None,
    session_config: SessionConfig | None = None,
) -> tuple[list[MemoryChunk], AccountImportReport]:
    """Discover an account export tree and import every conversation in it, kept separate.

    Returns the built session chunks (not events) because a caller must never receive one flat stream
    it could session-split across a conversation boundary.

    ``session_config`` is the segmentation the caller is indexing with. It is a parameter rather than
    a constant because the persistent index binds to it: a caller that changes ``max_chars`` must be
    able to say so here, and the cache entry built from the old value must not be reused.
    """
    layout = load_account_directory(account_dir, shard_dir=shard_dir)
    if not layout.exports:
        raise account_export_error(account_dir)
    events_by_conversation, report = import_account(
        layout.exports,
        shards_detected=layout.shards_detected,
        discovered=layout.descriptors,
        # No Data Loaded != No Memory Exists, for the account as a whole: an export narrowed with
        # `--only` covers fewer conversations than the account holds, and the report must be able to
        # say so even when every shard was exported. The count comes from the manifest; without it a
        # filtered tree is indistinguishable from a complete one.
        filtered_conversations=layout.filtered_conversations,
    )
    report.notes = report.notes + layout.notes
    chunks = build_account_sessions(
        events_by_conversation,
        session_config or SessionConfig(max_chars=DEFAULT_MAX_SESSION_CHARS),
    )

    # The phase's hard rule, re-checked on real data rather than trusted: no retrieval unit may hold
    # events from two conversations. A violation is a bug, not a caveat, so it fails loudly.
    crossed = crossed_conversation_chunks(chunks, events_by_conversation)
    if crossed:
        raise AssertionError(
            "conversation boundary crossed: chunk(s) "
            f"{list(crossed)} mix events from more than one conversation"
        )
    return chunks, report


def build_account_brain(
    chunks: list[MemoryChunk],
    account_name: str,
    *,
    k: int = DEFAULT_K,
    hybrid_pool: int = DEFAULT_HYBRID_POOL,
    workflow: str = "no-rewrite",
    answer_prompt: str = DEFAULT_ANSWER_PROMPT,
    max_output_tokens: int = 8192,
    temperature: float = 0.0,
    embedder: Any = None,
) -> tuple[Brain, RetrievalConfig]:
    """Index the account's sessions. Split from the import so a caller can report before embedding."""
    register_answer_prompt(answer_prompt)
    llm_config = build_llm_config(max_output_tokens=max_output_tokens, temperature=temperature)
    retrieval_config = build_retrieval_config(
        llm_config, k=k, hybrid_pool=hybrid_pool, workflow=workflow
    )
    brain = build_brain_from_chunks(
        chunks,
        llm_config,
        f"personal_recall_account_{account_name}",
        origin=f"account:{account_name}",
        embedder=embedder,
    )
    return brain, retrieval_config


@dataclass
class AccountSession:
    """One account-wide retrieval session, plus what the persistent index did for it.

    Returned by :func:`build_account_session` — the **one** cache-aware seam. The CLI, the web app
    and the acceptance runner all go through it, so an index can only be built, loaded, invalidated
    or reported in one place; a second implementation of that decision is how one front end starts
    answering from a different index than another.
    """

    brain: Brain
    retrieval_config: RetrievalConfig
    report: AccountImportReport
    index: index_cache.IndexStatus


def build_account_session(
    account_dir: Path,
    *,
    shard_dir: Path | None = None,
    index_dir: Path | None = None,
    rebuild_index: bool = False,
    embedder: Any = None,
    session_config: SessionConfig | None = None,
    k: int = DEFAULT_K,
    hybrid_pool: int = DEFAULT_HYBRID_POOL,
    workflow: str = "no-rewrite",
    answer_prompt: str = DEFAULT_ANSWER_PROMPT,
    max_output_tokens: int = 8192,
    temperature: float = 0.0,
) -> AccountSession:
    """Build the product brain over an entire account — from the persistent index when it is valid.

    The whole phase in one function. A warm start loads vectors that were produced by *exactly* this
    configuration over *exactly* this export tree and skips both the parse and the embedding; anything
    else rebuilds, and says why. ``k``, ``hybrid_pool``, ``workflow``, ``answer_prompt``, the LLM model,
    ``temperature`` and ``max_output_tokens`` are **not** part of that decision — they select from an
    index, never build one, so changing k from 20 to 15 is still a hit.

    Every failure mode degrades to the cold path: an unreadable entry, a fingerprint that cannot be
    established, even a cache directory that cannot be written. The export tree remains the source of
    truth, and this function always returns a working session.
    """
    config = session_config or SessionConfig(max_chars=DEFAULT_MAX_SESSION_CHARS)
    cache_dir = index_cache.account_cache_dir(account_dir, index_dir)
    # Leftover staging from an interrupted build is rubble, not history: remove it before anything
    # reads or writes, so no later run has to reason about it.
    index_cache.cleanup_staging(cache_dir)

    # The export tree is resolved before the embedder is built, and once for both paths: a directory
    # listing costs milliseconds and the model costs tens of seconds, so a mistyped path must fail on
    # the walk rather than after loading a hundred megabytes of weights. The layout is also what the
    # warm path's report needs, so the same walk serves the hit and the miss.
    layout = load_account_directory(account_dir, shard_dir=shard_dir)
    if not layout.exports:
        raise account_export_error(account_dir)
    # The answer prompt is installed by whichever path actually produces a session — here on a hit,
    # and in `build_account_brain` on a miss — never by both, because `register_answer_prompt` *adds
    # to the prompt it finds* rather than replacing it: registering twice would double the rules.
    # Deliberately after the check above, too: a directory that holds nothing to index must not
    # mutate a process-global registry on its way to an error.
    embedder = embedder or index_cache.build_embedder()

    started = perf_counter()
    fingerprint_started = perf_counter()
    try:
        source_fp: str | None = index_cache.source_fingerprint(
            account_dir, exclude=(cache_dir.parent,)
        )
        fingerprint_failure: str | None = None
    except OSError as exc:
        # A tree that cannot be walked cannot be *proven* unchanged, so nothing may be loaded from
        # the cache or written into it. The import below is allowed to fail on its own terms.
        source_fp = None
        fingerprint_failure = f"source export tree unreadable ({type(exc).__name__})"
    fingerprint_ms = int((perf_counter() - fingerprint_started) * 1000)

    inspection = index_cache.inspect(
        cache_dir,
        expected=index_cache.ExpectedIndex(
            source_fingerprint=source_fp,
            chunking_fingerprint=index_cache.chunking_fingerprint(config),
            schema_version=index_cache.PROJECTION_VERSION,
            embedder=embedder,
        ),
        rebuild=rebuild_index,
    )

    # --- warm start: load the index and reconstruct what was imported --------------------------
    if inspection.is_hit:
        load_started = perf_counter()
        vector_store = None
        failure = ""
        try:
            vector_store = index_cache.load_store(cache_dir, embedder)
        except Exception as exc:  # noqa: BLE001 - any load failure is a miss, never a crash
            failure = f"index artifacts unreadable ({type(exc).__name__})"
        else:
            failure = index_cache.verify_store(vector_store, inspection.manifest)
        load_ms = int((perf_counter() - load_started) * 1000)
        if not failure:
            manifest = inspection.manifest
            register_answer_prompt(answer_prompt)
            # The framework resolves the API key inside `RetrievalConfig.__init__` and
            # `LLMEndpoint.from_config` reads it, so the retrieval config is built first — the same
            # ordering `build_account_brain` depends on.
            llm_config = build_llm_config(
                max_output_tokens=max_output_tokens, temperature=temperature
            )
            retrieval_config = build_retrieval_config(
                llm_config, k=k, hybrid_pool=hybrid_pool, workflow=workflow
            )
            return AccountSession(
                brain=build_brain_from_vector_store(
                    vector_store,
                    llm_config,
                    f"personal_recall_account_{account_dir.name}",
                    embedder,
                ),
                retrieval_config=retrieval_config,
                report=index_cache.warm_report(manifest, layout),
                index=index_cache.IndexStatus(
                    outcome=index_cache.HIT,
                    chunk_count=manifest.chunk_count,
                    vector_dimension=manifest.vector_dimension,
                    cache_age_seconds=manifest.age_seconds(),
                    cache_format_version=manifest.cache_format_version,
                    schema_version=manifest.schema_version,
                    embedded=False,
                    fingerprint_ms=fingerprint_ms,
                    load_ms=load_ms,
                    total_ms=int((perf_counter() - started) * 1000),
                ),
            )
        inspection = index_cache.Inspection(index_cache.INVALID, failure)

    # --- cold start: import, embed, then persist ------------------------------------------------
    import_started = perf_counter()
    chunks, report = import_account_directory(
        account_dir, shard_dir=shard_dir, session_config=config
    )
    import_ms = int((perf_counter() - import_started) * 1000)

    embed_started = perf_counter()
    brain, retrieval_config = build_account_brain(
        chunks,
        account_dir.name,
        k=k,
        hybrid_pool=hybrid_pool,
        workflow=workflow,
        answer_prompt=answer_prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        embedder=embedder,
    )
    embed_ms = int((perf_counter() - embed_started) * 1000)

    vector_store = brain.vector_db
    chunk_count = int(getattr(getattr(vector_store, "index", None), "ntotal", 0))
    vector_dimension = int(getattr(getattr(vector_store, "index", None), "d", 0))
    reason = inspection.reason
    cache_written = False
    save_ms = 0
    if source_fp is None:
        reason = fingerprint_failure or "source export tree unreadable"
    else:
        manifest = index_cache.IndexManifest(
            cache_format_version=index_cache.CACHE_FORMAT_VERSION,
            schema_version=index_cache.PROJECTION_VERSION,
            source_fingerprint=source_fp,
            chunking_fingerprint=index_cache.chunking_fingerprint(config),
            embedding_fingerprint=index_cache.embedding_fingerprint(
                embedder, dimension=vector_dimension
            ),
            chunk_count=chunk_count,
            vector_dimension=vector_dimension,
            created_at=index_cache.utc_now(),
            report=index_cache.report_aggregates(report),
        )
        save_started = perf_counter()
        try:
            index_cache.save_index(cache_dir, vector_store=vector_store, manifest=manifest)
        except (OSError, ValueError) as exc:
            # The index is built and answerable; the cache is an accelerator, and losing it must not
            # cost the user their answer. Reported rather than swallowed, because a cache that never
            # writes is a silent 80-minute tax on every run.
            reason = f"{reason}; cache write failed ({type(exc).__name__})"
        else:
            cache_written = True
        save_ms = int((perf_counter() - save_started) * 1000)

    return AccountSession(
        brain=brain,
        retrieval_config=retrieval_config,
        report=report,
        index=index_cache.IndexStatus(
            outcome=inspection.outcome,
            reason=reason,
            chunk_count=chunk_count,
            vector_dimension=vector_dimension,
            cache_format_version=index_cache.CACHE_FORMAT_VERSION,
            schema_version=index_cache.PROJECTION_VERSION,
            cache_written=cache_written,
            embedded=True,
            fingerprint_ms=fingerprint_ms,
            import_ms=import_ms,
            embed_ms=embed_ms,
            save_ms=save_ms,
            total_ms=int((perf_counter() - started) * 1000),
        ),
    )


def main() -> int:
    dotenv.load_dotenv(ENV_PATH if ENV_PATH.exists() else None)

    parser = argparse.ArgumentParser(description="Ask your chat history, with evidence.")
    parser.add_argument("question", help="the memory question to ask")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="a single corpus file, or a directory of WeFlow shard exports to merge",
    )
    parser.add_argument(
        "--account",
        type=Path,
        default=None,
        help="an account export directory: '<shard>/<talker>_messages.json' subdirectories, one per "
        "WeChat MSG*.db, imported as separate conversations (mutually exclusive with --corpus)",
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
        help="optional: the WeChat Msg/Multi directory, so the report can say which MSG*.db shards "
        "exist versus how many were actually exported (read-only, names only)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="account mode only: where the persistent index lives (default: "
        "data/real/.index_cache/<account>). It is derived from real chat, so keep it local, keep it "
        "out of git, and delete it whenever you like - the next run rebuilds from the exports",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="account mode only: ignore a valid persistent index and rebuild it from the exports",
    )
    parser.add_argument(
        "--source-format",
        choices=("auto",) + SOURCE_FORMATS,
        default="auto",
        help="corpus adapter: 'text' (canonical TXT) or 'weflow' (WeFlow JSON export). "
        "'auto' (default) infers it from the file extension.",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--hybrid-pool", type=int, default=DEFAULT_HYBRID_POOL)
    parser.add_argument(
        "--workflow",
        choices=("no-rewrite", "rag"),
        default="no-rewrite",
        help="'no-rewrite' (default) retrieves on the raw question deterministically.",
    )
    parser.add_argument(
        "--answer-prompt",
        choices=("cited-attributed", "cited-narrow", "cited", "timeline", "default"),
        default=DEFAULT_ANSWER_PROMPT,
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--show-uncited", action="store_true", help="also list retrieved sources the answer did not cite")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()

    # --- account-wide mode: every conversation of the account, each kept separate --------------
    if args.account is not None:
        if not args.account.is_dir():
            print(f"error: --account {args.account} is not a directory.", file=sys.stderr)
            return 2
        if not args.json:
            print(f"account  : {args.account}  (source=weflow, account-wide)")
            print(f"index dir: {index_cache.account_cache_dir(args.account, args.index_dir)}")
            print(f"config   : k={args.k}, hybrid pool={args.hybrid_pool}, workflow={args.workflow}, "
                  f"prompt={args.answer_prompt}")
            print(f"question : {args.question}\n")
            print("index    : loading the persistent index, or building it if it is not valid")
        try:
            session = build_account_session(
                args.account,
                shard_dir=args.shard_dir,
                index_dir=args.index_dir,
                rebuild_index=args.rebuild_index,
                k=args.k,
                hybrid_pool=args.hybrid_pool,
                workflow=args.workflow,
                answer_prompt=args.answer_prompt,
                max_output_tokens=args.max_output_tokens,
                temperature=args.temperature,
            )
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except (json.JSONDecodeError, OSError) as exc:
            print(f"error: could not read a shard export: {exc}", file=sys.stderr)
            return 2

        if not session.index.chunk_count:
            print(
                f"error: {args.account} produced 0 conversation sessions, so nothing was indexed. "
                "Check that the export directories hold '<talker>_messages.json' files with messages.",
                file=sys.stderr,
            )
            return 2
        if not args.json:
            # What the index cache did, in aggregate numbers only — never a message, a name or an id.
            for line in session.index.lines():
                print(f"index    : {line}")
            # One import, two consumers: the report below and the brain above describe the same
            # chunks, so the printed coverage can never drift from what was actually indexed.
            for line in session.report.lines():
                print(f"  {line}")
            for row in session.report.per_conversation:
                print(
                    f"  {row.conversation_id:28} {row.messages:6} msgs  {row.span()}"
                    f"  (shards: {', '.join(row.shards)})"
                )
            if session.report.reconstructed_from_cache and not session.report.per_conversation:
                # Say it rather than printing an empty table as if it were data.
                print(
                    "  (no per-conversation rows: a warm start does not re-parse the exports, and "
                    "per-conversation detail is not persisted — the counts above are exact)"
                )
            print()

        return ask_and_render(
            session.brain,
            session.retrieval_config,
            args,
            account_report=session.report,
            index_status=session.index,
        )

    source_format = (
        detect_source_format(args.corpus) if args.source_format == "auto" else args.source_format
    )
    multi_shard = is_multi_shard(args.corpus)

    if not args.json:
        label = "source=weflow, multi-shard" if multi_shard else f"source={source_format}"
        print(f"corpus   : {args.corpus}  ({label})")
        print(f"config   : k={args.k}, hybrid pool={args.hybrid_pool}, workflow={args.workflow}, "
              f"prompt={args.answer_prompt}")
        print(f"question : {args.question}\n")

    # --- multi-shard: merge every shard export, then report the union coverage -----------------
    if multi_shard:
        shard_files = corpus_files(args.corpus)
        if not shard_files:
            print(
                f"error: {args.corpus} contains no '*{'.json'}' shard exports to merge.",
                file=sys.stderr,
            )
            return 2
        try:
            merged_events, merge_report = load_and_merge(shard_files)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"error: could not read a shard export: {exc}", file=sys.stderr)
            return 2
        if not args.json:
            for line in merge_report.lines():
                print(f"  {line}")
            if args.shard_dir:
                present = discover_message_shards(args.shard_dir)
                exported = len(merge_report.shards)
                print(
                    f"  shard check   : {len(present)} MSG*.db present {present}; "
                    f"{exported} exported"
                )
                if present and exported < len(present):
                    # No Data Loaded != No Memory Exists: say the history is partial, do not imply
                    # the missing shards are empty.
                    print(
                        f"  WARNING       : only {exported} of {len(present)} message shards were "
                        "exported, so this history is PARTIAL - older or newer messages in the "
                        "other shards are invisible to every answer."
                    )
            print()
        if not merged_events:
            print(
                f"error: merging {len(shard_files)} shard export(s) produced 0 messages.",
                file=sys.stderr,
            )
            return 2

    # --- single file: the original guard, behaviour unchanged ---------------------------------
    # Guard against the silent-ingestion failure: an adapter that matches nothing indexes zero
    # messages and every question then answers "the record does not show it" without any error.
    # Fail loudly and say what to run instead.
    if not multi_shard and args.corpus.exists():
        try:
            n_events, n_skipped, detail = count_corpus_events(args.corpus, source_format)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"error: could not read {args.corpus} as {source_format}: {exc}", file=sys.stderr)
            return 2
        if not args.json:
            print(f"ingested : {detail}")
        if not n_events:
            if source_format == "weflow":
                hint = (
                    "A WeFlow export should be a JSON list of messages, or an object with a\n"
                    "'messages' list. Check that the file is the exporter's JSON, not the .db."
                )
            else:
                hint = (
                    "The text adapter reads one message per line as "
                    "'[YYYY-MM-DD HH:MM] speaker: text'.\n"
                    "Convert the export first:\n"
                    f"  python chat_import.py --input {args.corpus} --out data/my_chat.txt\n"
                    "  python chat_import.py --list-formats"
                )
            print(
                f"error: {args.corpus} yielded 0 messages "
                f"({n_skipped} entr{'y' if n_skipped == 1 else 'ies'} skipped) via the "
                f"'{source_format}' adapter.\n{hint}",
                file=sys.stderr,
            )
            return 2

    brain, retrieval_config = build_session(
        args.corpus,
        k=args.k,
        hybrid_pool=args.hybrid_pool,
        workflow=args.workflow,
        answer_prompt=args.answer_prompt,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
    )
    return ask_and_render(brain, retrieval_config, args)


def answer_question(
    brain: Brain,
    retrieval_config: RetrievalConfig,
    *,
    question: str,
    show_uncited: bool = False,
) -> dict[str, Any]:
    """Ask one question and assemble everything the product shows: the one recall path.

    The CLI, the local web UI and the acceptance runner must be the *same system*, so none of them
    may own a copy of this sequence — ``brain.ask`` -> ``serialize_sources`` -> evidence cards ->
    groundedness. ``ask_and_render`` prints what this returns; ``webapp`` serves it; an eval harness
    can score it. A second copy is how the product silently stops being the thing that was measured.

    The returned dict carries the publishable shapes (``answer``, ``evidence``, ``groundedness``,
    ``retrieved``, ``latency_ms``) and, for a rendering caller, the in-process objects the
    human-readable CLI output needs (``cards``, ``report``) plus the ``sources`` the cards were
    resolved from. That last key exists for one reason: a card knows the chunk it came from but not
    the conversation's name, and ``sources[card["citation_index"]]["conversation_id"]`` is the only
    honest way to join them without re-deriving the citation mapping.
    """
    started = perf_counter()
    response = brain.ask(
        run_id=uuid4(),
        question=question,
        retrieval_config=retrieval_config,
        chat_history=ChatHistory(chat_id=uuid4(), brain_id=brain.id),
    )
    latency_ms = int((perf_counter() - started) * 1000)

    sources = response.metadata.sources if response.metadata else []
    serialized = serialize_sources(sources)
    answer = response.answer or ""
    cards = build_evidence_cards(answer, serialized, include_uncited=show_uncited)
    groundedness = assess(answer, serialized)

    return {
        "question": question,
        "answer": answer,
        "evidence": cards_to_dict(cards),
        "groundedness": groundedness.as_dict(),
        "retrieved": len(serialized),
        "latency_ms": latency_ms,
        "sources": serialized,
        "cards": cards,
        "report": groundedness,
    }


def ask_and_render(
    brain: Brain,
    retrieval_config: RetrievalConfig,
    args,
    *,
    account_report: AccountImportReport | None = None,
    index_status: index_cache.IndexStatus | None = None,
) -> int:
    """Ask the question and print the answer with its evidence. The one place output is formatted.

    Thin on purpose: the recall work happens in :func:`answer_question`, so this function decides
    only *how* the result is written out.
    """
    result = answer_question(
        brain, retrieval_config, question=args.question, show_uncited=args.show_uncited
    )

    if args.json:
        print(
            json.dumps(
                {
                    "question": result["question"],
                    "answer": result["answer"],
                    "evidence": result["evidence"],
                    "groundedness": result["groundedness"],
                    "retrieved": result["retrieved"],
                    "account": account_report.as_dict() if account_report is not None else None,
                    # Counts, versions and timings only — the same identity-free rule as the status
                    # lines, because `--json` output gets pasted into issues.
                    "index": index_status.as_dict() if index_status is not None else None,
                    "config": {
                        "k": args.k,
                        "hybrid_pool": args.hybrid_pool,
                        "workflow": args.workflow,
                        "answer_prompt": args.answer_prompt,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print("Answer:\n")
    print(result["answer"])
    print()
    print(render_cards(result["cards"]))
    print()
    print(render_groundedness(result["report"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
