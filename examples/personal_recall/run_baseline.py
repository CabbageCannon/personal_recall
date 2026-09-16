import argparse
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import dotenv
from langgraph.graph import END, START
from langchain_community.embeddings import HuggingFaceEmbeddings

from quivr_core import Brain
from quivr_core.files.file import FileExtension
from quivr_core.llm import LLMEndpoint
from quivr_core.processor.implementations.simple_txt_processor import (
    SimpleTxtProcessor,
)
from quivr_core.processor.registry import register_processor
from quivr_core.processor.splitter import SplitterConfig
from quivr_core.rag.entities.config import (
    DefaultModelSuppliers,
    DefaultRerankers,
    DefaultWorkflow,
    HybridConfig,
    LLMEndpointConfig,
    NodeConfig,
    RerankerConfig,
    RetrievalConfig,
    WorkflowConfig,
)
from quivr_core.rag.entities.chat import ChatHistory
from quivr_core.rag.prompts import TemplatePromptName, custom_prompts, register_prompt

from memory import SessionConfig
from memory.processor import ConversationSessionProcessor
from eval_utils import (
    evidence_matches_source,
    strict_evidence_matches_source,
)

# =========================
# Paths
# =========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# One per-dataset config is the single source of truth for every
# dataset-specific path. "small" is the default and keeps the original
# file names, so a run without --dataset behaves exactly as before.
DATASET_CONFIG = {
    "small": {
        "corpus": DATA_DIR / "chats.txt",
        "queries": DATA_DIR / "queries.json",
        "results": BASE_DIR / "baseline_results.json",
    },
    "stress": {
        "corpus": DATA_DIR / "stress_chats.txt",
        "queries": DATA_DIR / "stress_queries.json",
        "results": BASE_DIR / "stress_results.json",
    },
}

DEFAULT_DATASET = "small"


def dataset_paths(dataset: str) -> dict[str, Path]:
    # Resolve every path for one dataset from the config above, so no
    # corpus / queries / results path is hard-coded further down.
    return DATASET_CONFIG[dataset]

EMBEDDING_MODEL_PATH = Path(r"D:\AIModels\bge-small-zh-v1.5")

# =========================
# Baseline configuration
# =========================

RETRIEVAL_K = 5
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

#: The frozen baseline retrieval unit. Anything else is an experiment and must be
#: tagged so it cannot overwrite the baseline results.
DEFAULT_CHUNKING = "fixed"
DEFAULT_MAX_SESSION_CHARS = 900
#: Candidate pool fed to a reranker; the reranker then keeps `--k` of them.
DEFAULT_CANDIDATE_K = 50
#: Candidates per retriever before RRF fusion (measured plateau).
DEFAULT_HYBRID_POOL = 30

#: Frozen LLM output budget. The 'timeline' answer prompt makes the model reason much
#: longer (measured: up to 4096 reasoning tokens), so it needs more headroom.
MAX_OUTPUT_TOKENS = 4096

#: Frozen LLM temperature. Non-zero makes even retrieval vary between runs, because the
#: query-rewrite step is an LLM call (measured: identical retrieval configs shared only
#: 4/36 retrieved evidence sets between runs).
LLM_TEMPERATURE = 0.3

#: Frozen workflow. 'no-rewrite' drops the LLM query-condensation node; measured at
#: temperature 0 it still does not make runs reproducible (6/36 shared evidence sets),
#: so determinism must come from removing the LLM from the retrieval path entirely.
DEFAULT_WORKFLOW = "rag"

#: Appended to the RAG answer prompt by `--answer-prompt timeline`.
#:
#: Targets the two generation-side failures measured on the stress run:
#:   * over-abstention: the stock prompt says "if you cannot provide an answer ... just
#:     answer that you don't have the answer", with no counterweight telling the model to
#:     commit when the dated lines *do* settle the question (`s025`);
#:   * evidence misreading: the stock prompt says to "state" contradictory information,
#:     which makes the model report both readings instead of resolving them by date (`s007`).
TIMELINE_PROMPT_ADDENDUM = """
Time-line reasoning rules (apply in addition to the rules above):
- First order the retrieved records by date and read what happened in each one.
- If the dated records TOGETHER determine the asked fact, you MUST give that conclusion.
  Do not answer that you cannot determine it merely because one line is vague.
- Only if NO record touches the asked fact may you say the records do not show it.
- If records look contradictory (one is a later recollection, another is a contemporaneous
  statement), trust the CONTEMPORANEOUS, DATED statement and resolve the conflict yourself
  instead of handing both readings back.
- Distinguish "plan / intend / suggestion" from "what actually happened"; when asked about
  an outcome, answer from what actually happened.
- Keep the original dates when you quote a record.
"""


#: Appended on top of the timeline rules by `--answer-prompt cited`.
#:
#: The stock prompt actively forbids this ("Don't cite the source id in the answer
#: objects"), so the rule below deliberately overrides that line: without a machine-readable
#: link from a claim to the source it came from, the answer's provenance cannot be verified
#: against the original chat lines (Phase 7).
CITATION_PROMPT_ADDENDUM = """
Citation rules (these override the earlier instruction not to cite source ids):
- End every factual statement with the source it came from, written exactly as [来源 N],
  where N is the number shown after "Source: " in the context above.
- If one statement uses several sources, list them together, e.g. [来源 2][来源 5].
- Only cite a source that actually supports that statement. Never guess a number; if you
  are unsure, cite nothing for that statement.
- Every date you write must come from the source you cite for that statement.
- If nothing in the context supports the answer, say so instead of citing anything.
"""


#: Appended on top of the citation rules by `--answer-prompt cited-committed`.
#:
#: Measured problem it targets: requiring a source for every claim pushed the model back into
#: over-abstention on questions whose answer must be *joined across* sources — `s025`
#: ("无法确认…上下文没有说明…是同一个人") and `s032` ("无法确定是否同一个") both regressed from
#: PASS to PARTIAL in A8 while citing the supporting facts correctly. The hedges, not the
#: citations, cost those two queries.
COMMITMENT_PROMPT_ADDENDUM = """
Commitment rules (these take priority over the citation rules when they seem to conflict):
- Citing is not hedging. Give the conclusion that the cited sources support; do not change your
  answer to "cannot determine" merely because you must cite sources for it.
- When a conclusion only follows from SEVERAL sources together - for example deciding whether two
  people mentioned separately are the same person, or whether a plan was later replaced by another
  one - draw that conclusion yourself and cite all the sources it rests on.
- Only answer that the records do not show something when no combination of the cited sources
  supports any conclusion.
"""


def _rebuild_answer_prompt(base: object, extra: str) -> object:
    """Copy ``base``'s messages verbatim and append ``extra`` to the final human message."""
    from langchain_core.prompts import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        MessagesPlaceholder,
        SystemMessagePromptTemplate,
    )

    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(base.messages[0].prompt.template),
            MessagesPlaceholder(variable_name="chat_history"),
            SystemMessagePromptTemplate.from_template(base.messages[2].prompt.template),
            HumanMessagePromptTemplate.from_template(
                base.messages[3].prompt.template + extra
            ),
        ]
    )


def build_timeline_answer_prompt(base: object) -> object:
    """Return the augmented answer prompt built from ``base`` (pure, no global mutation)."""
    return _rebuild_answer_prompt(base, TIMELINE_PROMPT_ADDENDUM)


def build_cited_answer_prompt(base: object) -> object:
    """Timeline rules plus mandatory [来源 N] citations on every factual statement."""
    return _rebuild_answer_prompt(
        base, TIMELINE_PROMPT_ADDENDUM + CITATION_PROMPT_ADDENDUM
    )


def build_cited_committed_answer_prompt(base: object) -> object:
    """Citations plus the commitment clause (cite, don't hedge)."""
    return _rebuild_answer_prompt(
        base,
        TIMELINE_PROMPT_ADDENDUM
        + CITATION_PROMPT_ADDENDUM
        + COMMITMENT_PROMPT_ADDENDUM,
    )


def register_timeline_answer_prompt() -> None:
    """Install the augmented answer prompt through the framework's registration API.

    The stock messages are copied verbatim and the addendum is appended to the final
    human message, so this stays a single-variable change.
    """
    register_prompt(
        TemplatePromptName.RAG_ANSWER_PROMPT,
        build_timeline_answer_prompt(custom_prompts[TemplatePromptName.RAG_ANSWER_PROMPT]),
        override=True,
    )


def register_cited_answer_prompt() -> None:
    """Install the timeline prompt plus mandatory machine-readable citations."""
    register_prompt(
        TemplatePromptName.RAG_ANSWER_PROMPT,
        build_cited_answer_prompt(custom_prompts[TemplatePromptName.RAG_ANSWER_PROMPT]),
        override=True,
    )


def register_cited_committed_answer_prompt() -> None:
    """Install the citation prompt plus the commitment clause."""
    register_prompt(
        TemplatePromptName.RAG_ANSWER_PROMPT,
        build_cited_committed_answer_prompt(
            custom_prompts[TemplatePromptName.RAG_ANSWER_PROMPT]
        ),
        override=True,
    )


def register_answer_prompt(variant: str) -> None:
    """Dispatch on the ``--answer-prompt`` value."""
    if variant == "timeline":
        register_timeline_answer_prompt()
    elif variant == "cited":
        register_cited_answer_prompt()
    elif variant == "cited-committed":
        register_cited_committed_answer_prompt()
    elif variant != "default":
        raise ValueError(f"unknown answer prompt variant: {variant}")


def load_queries(queries_path: Path) -> list[dict]:
    with queries_path.open("r", encoding="utf-8") as f:
        queries = json.load(f)

    return queries
  

def evaluate_retrieval(
    query: dict, # 我自己设置的问答内容
    sources: list, # 检索出来的内容
) -> dict:
    # 这里拿到的是一条条对话,粒度较低
    gold_evidence = query["relevant_evidence"]

    # 如果问题本身就不足以被回答,也就没有正确的evidence
    if not gold_evidence:
        return {
            "hit_at_5": None,
            "matched_evidence": [],
            "matched_evidence_count": 0,
            "gold_evidence_count": 0,
            "evidence_coverage": None,
            
            "strict_hit_at_5": None,
            "strict_matched_evidence_count": 0,
            "strict_evidence_coverage": None,
        }

    # 与正确答案匹配的证据
    matched_evidence = []
    # 与正确答案严格匹配的证据
    strict_matched_evidence = []

    for evidence in gold_evidence:
        evidence = evidence.strip()

        if any(
            evidence_matches_source(
                evidence,
                source.page_content
            )
            for source in sources
        ):
            matched_evidence.append(evidence)
            
        if any(
            strict_evidence_matches_source(
                evidence,
                source.page_content,
            )
            for source in sources
        ):
            strict_matched_evidence.append(evidence)

    matched_count = len(matched_evidence)
    gold_count = len(gold_evidence)

    return {
        "hit_at_5": matched_count > 0,
        "matched_evidence": matched_evidence,
        "matched_evidence_count": matched_count,
        "gold_evidence_count": gold_count,
        "evidence_coverage": matched_count / gold_count,
        
        "strict_hit_at_5": len(strict_matched_evidence) > 0,
        "strict_matched_evidence_count": len(strict_matched_evidence),
        "strict_evidence_coverage": (
            len(strict_matched_evidence) / gold_count
        ),
    }
  
# 把document转成可以写进json的dict
#: Session fields carried from the retrieval unit into the result file, so a retrieved
#: chunk is a full EvidenceItem: which conversation session it came from, when it
#: happened, and who was in it. Without these a citation cannot be traced back to the
#: original chat lines (Phase 7: Answer -> Evidence -> MemoryEvent -> source chat).
EVIDENCE_METADATA_FIELDS = (
    "memory_chunk_id",
    "conversation_id",
    "start_time",
    "end_time",
    "participants",
    "n_events",
    "rerank_score",
)


def serialize_sources(sources: list) -> list[dict]:
    serialized = []

    for rank, source in enumerate(sources, start=1):
        entry = {
            "rank": rank,
            "chunk_index": source.metadata.get("chunk_index"),
            "content": source.page_content,
            "original_file_name": source.metadata.get(
                "original_file_name"
            ),
        }
        for field in EVIDENCE_METADATA_FIELDS:
            if field in source.metadata:
                entry[field] = source.metadata[field]
        serialized.append(entry)

    return serialized  

def save_results(results: list[dict], result_path: Path) -> None:
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_existing_results(result_path: Path) -> list[dict]:
    if not result_path.exists():
        return []

    with result_path.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the personal recall retrieval baseline.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIG),
        default=DEFAULT_DATASET,
        help=(
            "Which dataset to run (default: %(default)s). "
            "'small' reads data/chats.txt, "
            "'stress' reads data/stress_chats.txt."
        ),
    )
    parser.add_argument(
        "--chunking",
        choices=("fixed", "session"),
        default=DEFAULT_CHUNKING,
        help=(
            "Retrieval unit (default: %(default)s). 'fixed' is the frozen baseline "
            "(400/100 character slices); 'session' uses conversation sessions."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=RETRIEVAL_K,
        help="Top-K retrieved for the answer (default: %(default)s, the frozen baseline).",
    )
    parser.add_argument(
        "--max-session-chars",
        type=int,
        default=DEFAULT_MAX_SESSION_CHARS,
        help="Session size budget when --chunking session (default: %(default)s).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help=(
            "Experiment tag; results are written to <dataset>_<tag>_results.json "
            "instead of the frozen baseline file."
        ),
    )
    parser.add_argument(
        "--rerank-model",
        default=None,
        help=(
            "Local cross-encoder re-ranking the candidate pool before generation "
            "(e.g. a sentence-transformers model path). Off by default."
        ),
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help=(
            "Retriever candidates fed to the re-ranker (default: %(default)s; "
            "only used together with --rerank-model)."
        ),
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help=(
            "Fuse the dense retriever with BM25 by weighted RRF (measured: 86.3%% vs "
            "82.4%% evidence coverage at 10 slots). Off by default."
        ),
    )
    parser.add_argument(
        "--hybrid-pool",
        type=int,
        default=DEFAULT_HYBRID_POOL,
        help="Candidates per retriever before fusion (default: %(default)s).",
    )
    parser.add_argument(
        "--answer-prompt",
        choices=("default", "timeline", "cited", "cited-committed"),
        default="default",
        help=(
            "Answer prompt variant (default: %(default)s). 'timeline' appends explicit "
            "time-line reasoning rules targeting over-abstention and evidence misreading; "
            "'cited' adds mandatory [来源 N] citations; 'cited-committed' adds the "
            "commitment clause (cite, don't hedge) on top."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=MAX_OUTPUT_TOKENS,
        help=(
            "LLM output budget (default: %(default)s, the frozen baseline). The "
            "'timeline' prompt reasons much longer, so it needs more headroom or the "
            "reasoning consumes the whole budget and the answer comes back empty."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=LLM_TEMPERATURE,
        help=(
            "LLM temperature (default: %(default)s, the frozen baseline value). The "
            "pipeline's query rewrite is an LLM call, so a non-zero temperature makes "
            "even the retrieved evidence differ between runs."
        ),
    )
    parser.add_argument(
        "--workflow",
        choices=("rag", "no-rewrite"),
        default=DEFAULT_WORKFLOW,
        help=(
            "Workflow variant (default: %(default)s). 'no-rewrite' drops the LLM "
            "query-condensation node so retrieval runs on the raw question, which makes "
            "the retrieval stage deterministic and therefore measurable."
        ),
    )
    args = parser.parse_args()

    paths = dataset_paths(args.dataset)

    # Guardrail: a non-baseline configuration must never overwrite the frozen
    # baseline results, so it has to be named explicitly.
    is_baseline_config = (
        args.chunking == DEFAULT_CHUNKING
        and args.k == RETRIEVAL_K
        and args.rerank_model is None
        and not args.hybrid
        and args.answer_prompt == "default"
        and args.max_output_tokens == MAX_OUTPUT_TOKENS
        and args.temperature == LLM_TEMPERATURE
        and args.workflow == DEFAULT_WORKFLOW
    )
    if not is_baseline_config and not args.tag:
        parser.error(
            "--tag is required when --chunking/--k/--rerank-model/--hybrid/--answer-prompt/"
            "--max-output-tokens/--temperature/--workflow differ from the frozen baseline "
            f"({DEFAULT_CHUNKING} chunking, k={RETRIEVAL_K}, no reranker, dense only, "
            f"default prompt, {MAX_OUTPUT_TOKENS} output tokens, temperature "
            f"{LLM_TEMPERATURE}, {DEFAULT_WORKFLOW} workflow); this keeps "
            f"{paths['results'].name} untouched."
        )

    if args.workflow == "no-rewrite":
        workflow_config = WorkflowConfig(
            nodes=[
                NodeConfig(name=START, edges=["filter_history"]),
                NodeConfig(name="filter_history", edges=["retrieve"]),
                NodeConfig(name="retrieve", edges=["generate_rag"]),
                NodeConfig(name="generate_rag", edges=[END]),
            ]
        )
    else:
        workflow_config = WorkflowConfig(nodes=DefaultWorkflow.RAG.nodes)

    register_answer_prompt(args.answer_prompt)
    if args.tag:
        results_path = BASE_DIR / f"{args.dataset}_{args.tag}_results.json"
    else:
        results_path = paths["results"]

    # Dataset-specific brain name: an index built from one corpus must
    # never be confused with an index built from another corpus (or from a
    # different chunking of it).
    brain_name = f"personal_recall_{args.dataset}"
    if args.tag:
        brain_name = f"{brain_name}_{args.tag}"

    dotenv.load_dotenv()

    # 0. Self-documenting run banner
    print(f"Dataset: {args.dataset}")
    print(f"Corpus: {paths['corpus']}")
    print(f"Queries: {paths['queries']}")
    print(f"Results: {results_path}")
    print(f"Chunking: {args.chunking}")
    print(f"Retrieval k: {args.k}")
    if args.rerank_model:
        print(f"Reranker: local cross-encoder {args.rerank_model}")
        print(f"Candidate k: {args.candidate_k} -> top {args.k}")
    print(f"Hybrid (dense+BM25 RRF): {args.hybrid}")
    if args.hybrid:
        print(f"Hybrid pool: {args.hybrid_pool} per retriever -> top {args.k}")
    print(f"Answer prompt: {args.answer_prompt}")
    print(f"Max output tokens: {args.max_output_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Brain name: {brain_name}")

    # 1. Load gold queries
    queries = load_queries(paths["queries"])

    # 2. Build LLM
    llm_config = LLMEndpointConfig(
        supplier=DefaultModelSuppliers.OPENAI,
        model="deepseek-v4-flash",
        llm_base_url="https://api.deepseek.com",
        env_variable_name="DEEPSEEK_API_KEY",
        max_context_tokens=20000,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
    )
    
    llm = LLMEndpoint.from_config(llm_config)
    
    # 3. Build local embedding model
    embedder = HuggingFaceEmbeddings(
        model_name=str(EMBEDDING_MODEL_PATH),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    # 4. Register the TXT processor for the requested retrieval unit
    if args.chunking == "session":
        # Phase 1 experiment: one retrieval unit per conversation session.
        # Everything downstream (embedder, index, k, prompt) stays identical.
        register_processor(
            FileExtension.txt,
            ConversationSessionProcessor,
            override=True,
        )
        processor_kwargs = {
            "session_config": SessionConfig(max_chars=args.max_session_chars)
        }
    else:
        register_processor(
            FileExtension.txt,
            SimpleTxtProcessor,
            override=True,
        )
        processor_kwargs = {
            "splitter_config": SplitterConfig(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        }
    
    # 5. Build Brain once
    brain = Brain.from_files(
        name=brain_name,
        file_paths=[paths["corpus"]],
        llm=llm,
        embedder=embedder,
        processor_kwargs=processor_kwargs,
    )
    
    # 6. Fix baseline retrieval configuration.
    #    Without a reranker, `k` is the number of chunks handed to the model (the
    #    frozen behaviour). With one, the retriever returns `candidate_k` candidates
    #    and the reranker keeps the best `k` of them — the framework's own
    #    retrieve-then-rerank stage.
    if args.rerank_model:
        reranker_config = RerankerConfig(
            supplier=DefaultRerankers.LOCAL,
            model=args.rerank_model,
            top_n=args.k,
        )
        retrieval_k = args.candidate_k
    else:
        reranker_config = RerankerConfig()
        retrieval_k = args.k

    retrieval_config = RetrievalConfig(
        llm_config=llm_config,
        k=retrieval_k,
        workflow_config=workflow_config,
        reranker_config=reranker_config,
        hybrid_config=HybridConfig(
            enabled=args.hybrid,
            candidate_k=args.hybrid_pool,
        ),
    )
    
    print(f"Loaded queries: {len(queries)}")
    print("Brain ready")
    print(f"Retrieval k: {retrieval_config.k}")
    if args.chunking == "session":
        print(f"Session max chars: {args.max_session_chars}")
    else:
        print(f"Chunk size: {CHUNK_SIZE}")
        print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(f"Embedding model: {EMBEDDING_MODEL_PATH}")
    
    # 7. Run baseline evaluation (incremental / resume)
    existing_results = load_existing_results(results_path)

    result_by_id = {
        result["query_id"]: result
        for result in existing_results
    }

    skipped_count = 0
    executed_count = 0

    print(f"Existing results: {len(existing_results)}")

    for index, query in enumerate(queries, start=1):
        query_id = query["id"]

        print(
            f"\n[{index:02d}/{len(queries)}] "
            f"{query_id} - {query['question']}"
        )

        existing_result = result_by_id.get(query_id)

        # Resume is only safe when query_id AND question both match.
        # q022 keeps its id but changed its question, so it must be re-run.
        if (
            existing_result is not None
            and existing_result.get("question") == query["question"]
        ):
            print("Status: SKIP (existing result matches current question)")
            skipped_count += 1
            continue

        if existing_result is not None:
            print("Status: RE-RUN (question changed)")
        else:
            print("Status: RUN (new query)")

        # Every gold query gets an independent chat history.
        eval_chat = ChatHistory(
            chat_id=uuid4(),
            brain_id=brain.id,
        )

        start_time = perf_counter()

        response = brain.ask(
            run_id=uuid4(),
            question=query["question"],
            retrieval_config=retrieval_config,
            chat_history=eval_chat,
        )

        latency_ms = (perf_counter() - start_time) * 1000

        sources = (
            response.metadata.sources
            if response.metadata
            else []
        )

        retrieval_eval = evaluate_retrieval(
            query=query,
            sources=sources,
        )

        result = {
            "query_id": query["id"],
            "category": query["category"],
            "answerable": query["answerable"],
            "question": query["question"],
            "expected_answer": query["expected_answer"],
            "answer": response.answer,
            "latency_ms": round(latency_ms, 2),
            "retrieved_sources": serialize_sources(sources),
            **retrieval_eval,
        }

        # Re-running a query overwrites its old result; new queries are added.
        result_by_id[query_id] = result
        executed_count += 1

        # Save after every query so a later API failure
        # does not lose completed results.
        # Rebuild in queries.json order so a re-run query (e.g. q022)
        # stays in place instead of moving to the end of the file.
        ordered_results = [
            result_by_id[current_query["id"]]
            for current_query in queries
            if current_query["id"] in result_by_id
        ]

        save_results(ordered_results, results_path)

        print(f"Answer: {response.answer}")
        print(f"Latency: {latency_ms:.2f} ms")
        print(f"Sources: {len(sources)}")

        if retrieval_eval["hit_at_5"] is None:
            print("Hit@5: N/A")
        else:
            print(
                "Hit@5: "
                f"{retrieval_eval['hit_at_5']} "
                f"("
                f"{retrieval_eval['matched_evidence_count']}/"
                f"{retrieval_eval['gold_evidence_count']} evidence"
                f")"
            )

    print("\n" + "=" * 80)
    print(f"Total queries: {len(queries)}")
    print(f"Executed: {executed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Results saved to: {results_path}")