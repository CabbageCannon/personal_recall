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
from pathlib import Path
from uuid import uuid4

import dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
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
from memory import SessionConfig
from memory.events import parse_txt_events
from memory.processor import (
    ConversationSessionProcessor,
    WeFlowSessionProcessor,
    session_documents,
)
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
EMBEDDING_MODEL_PATH = Path(r"D:\AIModels\bge-small-zh-v1.5")

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
        parsed = parse_weflow_events(json.loads(raw))
        detail = (
            f"{len(parsed.events)} messages "
            f"(types: {dict(sorted(parsed.message_types.items()))}, "
            f"suppressed payloads: {parsed.suppressed_payloads})"
        )
        return len(parsed.events), parsed.skipped, detail
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


def build_brain_from_events(events, llm_config: LLMEndpointConfig, name: str, origin: str) -> Brain:
    """Build a brain from already-merged events — used by the multi-shard path.

    Sessions are built over the *union* of the shards, so a conversation spanning two databases becomes
    one retrieval unit with the global latest timestamp rather than one unit per shard.

    Building from documents bypasses ``ProcessorBase.process_file``, so the metadata that step would
    have added has to be supplied here: without ``original_file_name`` the framework's document prompt
    refuses to render (`combine_documents` adds ``index`` itself, but not this). The
    ``"Filename: … Content: …"`` prefix is deliberately NOT applied — ``process_file`` only adds it when
    the inner metadata already carries the field, which is exactly why the processor path's chunk text
    has no prefix.
    """
    sessions = build_sessions(
        events,
        config=SessionConfig(max_chars=DEFAULT_MAX_SESSION_CHARS),
        conversation_id=DEFAULT_MERGED_CONVERSATION,
    )
    documents = session_documents(sessions, skipped_source_lines=0)
    for index, document in enumerate(documents, start=1):
        document.metadata["chunk_index"] = index
        document.metadata["original_file_name"] = origin
    embedder = HuggingFaceEmbeddings(
        model_name=str(EMBEDDING_MODEL_PATH),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
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
            embedder=HuggingFaceEmbeddings(
                model_name=str(EMBEDDING_MODEL_PATH),
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            ),
            processor_kwargs={
                "session_config": SessionConfig(max_chars=DEFAULT_MAX_SESSION_CHARS)
            },
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
    from run_baseline import build_workflow_config

    register_answer_prompt(answer_prompt)

    llm_config = LLMEndpointConfig(
        supplier=DefaultModelSuppliers.OPENAI,
        model="deepseek-v4-flash",
        llm_base_url="https://api.deepseek.com",
        env_variable_name="DEEPSEEK_API_KEY",
        max_context_tokens=20000,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )

    # NOTE (framework ordering quirk): `LLMEndpointConfig` only resolves its API key when a
    # `RetrievalConfig` is constructed (its __init__ calls `llm_config.set_api_key(force_reset=True)`).
    # Build the retrieval config BEFORE the brain, or `LLMEndpoint.from_config` gets api_key=None.
    retrieval_config = RetrievalConfig(
        llm_config=llm_config,
        k=k,
        hybrid_config=HybridConfig(enabled=True, candidate_k=hybrid_pool),
        workflow_config=build_workflow_config(workflow),
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
        "--shard-dir",
        type=Path,
        default=None,
        help="optional: the WeChat Msg/Multi directory, so the report can say which MSG*.db shards "
        "exist versus how many were actually exported (read-only, names only)",
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
    response = brain.ask(
        run_id=uuid4(),
        question=args.question,
        retrieval_config=retrieval_config,
        chat_history=ChatHistory(chat_id=uuid4(), brain_id=brain.id),
    )

    sources = response.metadata.sources if response.metadata else []
    serialized = serialize_sources(sources)
    answer = response.answer or ""
    cards = build_evidence_cards(answer, serialized, include_uncited=args.show_uncited)
    groundedness = assess(answer, serialized)

    if args.json:
        print(
            json.dumps(
                {
                    "question": args.question,
                    "answer": answer,
                    "evidence": cards_to_dict(cards),
                    "groundedness": groundedness.as_dict(),
                    "retrieved": len(serialized),
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
    print(answer)
    print()
    print(render_cards(cards))
    print()
    print(render_groundedness(groundedness))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
