from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

import quivr_core.rag.quivr_rag_langgraph as rag_module
from quivr_core.rag.entities.chat import ChatHistory
from quivr_core.rag.entities.models import RAGResponseMetadata
from quivr_core.rag.quivr_rag_langgraph import (
    QuivrQARAGLangGraph,
    SplittedInput,
    UserTasks,
)


def make_rag() -> QuivrQARAGLangGraph:
    """
    Build a minimal QuivrQARAGLangGraph without creating a real LLM
    or vector store.
    """
    rag = object.__new__(QuivrQARAGLangGraph)

    rag.retrieval_config = SimpleNamespace(
        prompt=None,
        max_files=20,
        workflow_config=SimpleNamespace(nodes=[]),
    )

    rag.llm_endpoint = SimpleNamespace()
    rag.final_nodes = []

    return rag


def make_state(messages, original_query: str):
    return {
        "messages": messages,
        "original_query": original_query,
        "chat_history": ChatHistory(
            chat_id=uuid4(),
            brain_id=uuid4(),
        ),
        "files": "",
        "tasks": UserTasks(["rewritten task"]),
    }


@pytest.mark.parametrize(
    "messages",
    [
        [
            HumanMessage(content="What is Quivr?"),
        ],
        [
            SystemMessage(content="THIS IS NOT THE USER QUERY"),
            HumanMessage(content="What is Quivr?"),
        ],
    ],
)
def test_build_rag_prompt_inputs_uses_original_query(messages):
    """
    The RAG generation prompt must always use original_query as the
    current user task, regardless of where HumanMessage appears in messages.
    """
    rag = make_rag()

    state = make_state(
        messages=messages,
        original_query="What is Quivr?",
    )

    inputs = rag._build_rag_prompt_inputs(state, docs=[])

    assert inputs["task"] == "What is Quivr?"
    assert inputs["task"] != "THIS IS NOT THE USER QUERY"


@pytest.mark.parametrize(
    ("system_prompt", "expected_messages"),
    [
        (
            None,
            [
                ("user", "What is Quivr?"),
            ],
        ),
        (
            "THIS IS NOT THE USER QUERY",
            [
                ("system", "THIS IS NOT THE USER QUERY"),
                ("user", "What is Quivr?"),
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_answer_astream_stores_original_query(
    monkeypatch,
    system_prompt,
    expected_messages,
):
    """
    answer_astream must place the original user question into graph state
    explicitly, independently of message ordering.
    """

    class FakeChain:
        def __init__(self):
            self.received_state = None

        async def astream_events(self, state, **kwargs):
            self.received_state = state

            # Keep this function an async generator without producing
            # any graph/model events.
            if False:
                yield {}

    rag = make_rag()
    fake_chain = FakeChain()

    rag.build_chain = lambda: fake_chain

    # Avoid unrelated metadata/file formatting logic in this focused unit test.
    monkeypatch.setattr(
        rag_module,
        "format_file_list",
        lambda files, max_files: "",
    )
    monkeypatch.setattr(
        rag_module,
        "get_chunk_metadata",
        lambda message, docs: RAGResponseMetadata(),
    )

    history = ChatHistory(
        chat_id=uuid4(),
        brain_id=uuid4(),
    )

    responses = [
        response
        async for response in rag.answer_astream(
            run_id=uuid4(),
            question="What is Quivr?",
            system_prompt=system_prompt,
            history=history,
            list_files=[],
        )
    ]

    assert fake_chain.received_state is not None

    assert fake_chain.received_state["original_query"] == "What is Quivr?"
    assert fake_chain.received_state["messages"] == expected_messages

    # Make sure the async generator still reaches its final metadata chunk.
    assert responses[-1].last_chunk is True


def test_routing_split_uses_original_query():
    """
    routing_split must send original_query into the split prompt instead of
    blindly using messages[0].
    """
    rag = make_rag()

    state = make_state(
        messages=[
            SystemMessage(content="SYSTEM_PROMPT_SENTINEL"),
            HumanMessage(content="HUMAN_MESSAGE_SENTINEL"),
        ],
        original_query="ORIGINAL_QUERY_SENTINEL",
    )

    captured = {}

    def fake_invoke_structured_output(prompt, output_class):
        captured["prompt"] = prompt
        captured["output_class"] = output_class

        return SplittedInput(
            instructions=None,
            task_list=["test task"],
        )

    rag.invoke_structured_output = fake_invoke_structured_output

    rag.routing_split(state)

    assert captured["output_class"] is SplittedInput

    # The split prompt must contain the explicit original query.
    assert "ORIGINAL_QUERY_SENTINEL" in captured["prompt"]

    # The system message must not accidentally become user_input.
    assert "SYSTEM_PROMPT_SENTINEL" not in captured["prompt"]