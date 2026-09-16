import argparse
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import dotenv
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
    LLMEndpointConfig,
    RerankerConfig,
    RetrievalConfig,
)
from quivr_core.rag.entities.chat import ChatHistory

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
def serialize_sources(sources: list) -> list[dict]:
    serialized = []

    for rank, source in enumerate(sources, start=1):
        serialized.append(
            {
                "rank": rank,
                "chunk_index": source.metadata.get("chunk_index"),
                "content": source.page_content,
                "original_file_name": source.metadata.get(
                    "original_file_name"
                ),
            }
        )

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
    args = parser.parse_args()

    paths = dataset_paths(args.dataset)

    # Guardrail: a non-baseline configuration must never overwrite the frozen
    # baseline results, so it has to be named explicitly.
    is_baseline_config = (
        args.chunking == DEFAULT_CHUNKING
        and args.k == RETRIEVAL_K
        and args.rerank_model is None
    )
    if not is_baseline_config and not args.tag:
        parser.error(
            "--tag is required when --chunking/--k/--rerank-model differ from the "
            f"frozen baseline ({DEFAULT_CHUNKING} chunking, k={RETRIEVAL_K}, no "
            f"reranker); this keeps {paths['results'].name} untouched."
        )
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
        max_output_tokens=4096,
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
        reranker_config=reranker_config,
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