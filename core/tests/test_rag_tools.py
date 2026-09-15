import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from quivr_core.llm_tools.entity import ToolWrapper
from quivr_core.llm_tools.llm_tools import LLMToolFactory
from quivr_core.rag.quivr_rag_langgraph import QuivrQARAGLangGraph, UserTasks


@pytest.mark.asyncio
async def test_run_tool_preserves_formatter_for_each_concurrent_tool(monkeypatch):
    """
    Each concurrent tool response must be parsed by the formatter belonging
    to the same tool/task.

    Tool B is deliberately forced to finish before Tool A so that the test
    also verifies that completion order does not break the association.
    """

    tool_b_finished = asyncio.Event()
    completion_order = []

    class FakeToolA:
        async def ainvoke(self, formatted_input):
            # Wrapper A must have formatted Task A's input.
            assert formatted_input == {"query_a": "task A"}

            # Force A to finish after B.
            await tool_b_finished.wait()

            completion_order.append("A")
            return {
                "tool": "A",
                "value": "raw-response-A",
            }

    class FakeToolB:
        async def ainvoke(self, formatted_input):
            # Wrapper B deliberately uses a different input format.
            assert formatted_input == {"query_b": "task B"}

            completion_order.append("B")
            tool_b_finished.set()

            return {
                "tool": "B",
                "value": "raw-response-B",
            }

    def format_input_a(task):
        return {"query_a": task}

    def format_output_a(response):
        # If Wrapper B/A association is broken, this assertion exposes it.
        assert response["tool"] == "A"

        return [
            Document(
                page_content="parsed-by-A",
                metadata={"formatter": "A"},
            )
        ]

    def format_input_b(task):
        return {"query_b": task}

    def format_output_b(response):
        assert response["tool"] == "B"

        return [
            Document(
                page_content="parsed-by-B",
                metadata={"formatter": "B"},
            )
        ]

    wrapper_a = ToolWrapper(
        tool=FakeToolA(),
        format_input=format_input_a,
        format_output=format_output_a,
    )

    wrapper_b = ToolWrapper(
        tool=FakeToolB(),
        format_input=format_input_b,
        format_output=format_output_b,
    )

    def fake_create_tool(tool_name, config):
        if tool_name == "tool_a":
            return wrapper_a

        if tool_name == "tool_b":
            return wrapper_b

        raise AssertionError(f"Unexpected tool requested: {tool_name}")

    monkeypatch.setattr(
        LLMToolFactory,
        "create_tool",
        fake_create_tool,
    )

    # run_tool only needs reranker_config here because it calls
    # filter_chunks_by_relevance() after formatting tool output.
    rag = object.__new__(QuivrQARAGLangGraph)
    rag.retrieval_config = SimpleNamespace(
        reranker_config=SimpleNamespace(
            relevance_score_threshold=None,
            relevance_score_key="relevance_score",
        )
    )

    tasks = UserTasks(
        [
            "task A",
            "task B",
        ]
    )

    task_a_id, task_b_id = tasks.ids

    tasks.set_tool(task_a_id, "tool_a")
    tasks.set_tool(task_b_id, "tool_b")

    result = await rag.run_tool(
        {
            "tasks": tasks,
        }
    )

    result_tasks = result["tasks"]

    # Prove that B actually completed first.
    assert completion_order == ["B", "A"]

    # Despite the reversed completion order, each task must be parsed
    # by its own formatter.
    assert result_tasks(task_a_id).docs[0].page_content == "parsed-by-A"
    assert result_tasks(task_a_id).docs[0].metadata["formatter"] == "A"

    assert result_tasks(task_b_id).docs[0].page_content == "parsed-by-B"
    assert result_tasks(task_b_id).docs[0].metadata["formatter"] == "B"