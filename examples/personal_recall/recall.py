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
import json
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
from memory.processor import ConversationSessionProcessor
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


def build_brain(corpus: Path, llm_config: LLMEndpointConfig) -> Brain:
    """Ingest one chat log as conversation sessions."""
    register_processor(FileExtension.txt, ConversationSessionProcessor, override=True)
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


def main() -> int:
    dotenv.load_dotenv(ENV_PATH if ENV_PATH.exists() else None)

    parser = argparse.ArgumentParser(description="Ask your chat history, with evidence.")
    parser.add_argument("question", help="the memory question to ask")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
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
        # A12 is the adopted product reference: the A11 prompt plus the speaker-attribution
        # clause that removed the cross-speaker fabrications (PROJECT_STATUS.md, Phase 12).
        default="cited-attributed",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--show-uncited", action="store_true", help="also list retrieved sources the answer did not cite")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()

    register_answer_prompt(args.answer_prompt)

    llm_config = LLMEndpointConfig(
        supplier=DefaultModelSuppliers.OPENAI,
        model="deepseek-v4-flash",
        llm_base_url="https://api.deepseek.com",
        env_variable_name="DEEPSEEK_API_KEY",
        max_context_tokens=20000,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
    )

    from run_baseline import build_workflow_config


    # NOTE (framework ordering quirk): `LLMEndpointConfig` only resolves its API key when a
    # `RetrievalConfig` is constructed (its __init__ calls `llm_config.set_api_key(force_reset=True)`).
    # Build the retrieval config BEFORE the brain, or `LLMEndpoint.from_config` gets api_key=None.
    retrieval_config = RetrievalConfig(
        llm_config=llm_config,
        k=args.k,
        hybrid_config=HybridConfig(enabled=True, candidate_k=args.hybrid_pool),
        workflow_config=build_workflow_config(args.workflow),
    )

    if not args.json:
        print(f"corpus   : {args.corpus}")
        print(f"config   : k={args.k}, hybrid pool={args.hybrid_pool}, workflow={args.workflow}, "
              f"prompt={args.answer_prompt}")
        print(f"question : {args.question}\n")

    brain = build_brain(args.corpus, llm_config)
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
