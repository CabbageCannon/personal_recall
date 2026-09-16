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
    LLMEndpointConfig,
    RetrievalConfig,
)
from quivr_core.rag.entities.chat import ChatHistory
from eval_utils import (
    evidence_matches_source,
    strict_evidence_matches_source,
)

# =========================
# Paths
# =========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CHAT_PATH = DATA_DIR / "chats.txt"
QUERIES_PATH = DATA_DIR / "queries.json"
RESULT_PATH = BASE_DIR / "baseline_results.json"

EMBEDDING_MODEL_PATH = Path(r"D:\AIModels\bge-small-zh-v1.5")

# =========================
# Baseline configuration
# =========================

RETRIEVAL_K = 5
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

def load_queries() -> list[dict]:
    with QUERIES_PATH.open("r", encoding="utf-8") as f:
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

def save_results(results: list[dict]) -> None:
    with RESULT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_existing_results() -> list[dict]:
    if not RESULT_PATH.exists():
        return []

    with RESULT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    dotenv.load_dotenv()

    # 1. Load gold queries
    queries = load_queries()

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
    
    # 4. Register TXT processor
    register_processor(
        FileExtension.txt,
        SimpleTxtProcessor,
        override=True,
    )
    
    # 5. Build Brain once
    brain = Brain.from_files(
        name="personal_recall_baseline",
        file_paths=[CHAT_PATH],
        llm=llm,
        embedder=embedder,
        processor_kwargs={
            "splitter_config": SplitterConfig(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        },
    )
    
    # 6. Fix baseline retrieval configuration
    retrieval_config = RetrievalConfig(
        llm_config=llm_config,
        k=RETRIEVAL_K,
    )
    
    print(f"Loaded queries: {len(queries)}")
    print("Brain ready")
    print(f"Retrieval k: {retrieval_config.k}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(f"Embedding model: {EMBEDDING_MODEL_PATH}")
    
    # 7. Run baseline evaluation (incremental / resume)
    existing_results = load_existing_results()

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

        save_results(ordered_results)

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
    print(f"Results saved to: {RESULT_PATH}")