"""Offline tests for the prompt registry API and the timeline answer prompt."""

from __future__ import annotations

import pytest
from langchain_core.prompts import ChatPromptTemplate

from quivr_core.rag.prompts import (
    TemplatePromptName,
    _templ_registry,
    custom_prompts,
    register_prompt,
)

from run_baseline import (
    CITATION_PROMPT_ADDENDUM,
    COMMITMENT_PROMPT_ADDENDUM,
    NARROW_COMMITMENT_PROMPT_ADDENDUM,
    TIMELINE_PROMPT_ADDENDUM,
    build_cited_answer_prompt,
    build_cited_committed_answer_prompt,
    build_cited_narrow_answer_prompt,
    build_timeline_answer_prompt,
)


@pytest.fixture()
def restore_registry():
    """Snapshot and restore the global prompt registry around a test."""
    saved = dict(_templ_registry)
    yield
    _templ_registry.clear()
    _templ_registry.update(saved)


def test_register_prompt_refuses_to_silently_replace(restore_registry) -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_prompt(
            TemplatePromptName.RAG_ANSWER_PROMPT,
            custom_prompts[TemplatePromptName.RAG_ANSWER_PROMPT],
        )


def test_register_prompt_with_override_is_visible_through_the_read_only_view(restore_registry) -> None:
    replacement = ChatPromptTemplate.from_messages([("human", "{task}")])

    register_prompt(TemplatePromptName.RAG_ANSWER_PROMPT, replacement, override=True)

    assert custom_prompts[TemplatePromptName.RAG_ANSWER_PROMPT] is replacement


def test_timeline_prompt_keeps_the_stock_structure_and_appends_the_rules() -> None:
    stock = custom_prompts[TemplatePromptName.RAG_ANSWER_PROMPT]

    timeline = build_timeline_answer_prompt(stock)

    assert [type(m).__name__ for m in timeline.messages] == [
        type(m).__name__ for m in stock.messages
    ]
    assert timeline.messages[0].prompt.template == stock.messages[0].prompt.template
    assert timeline.messages[2].prompt.template == stock.messages[2].prompt.template
    assert timeline.messages[3].prompt.template == (
        stock.messages[3].prompt.template + TIMELINE_PROMPT_ADDENDUM
    )
    assert sum(1 for m in timeline.messages if type(m).__name__.startswith("Human")) == 1


def test_timeline_prompt_renders_the_same_variables() -> None:
    stock = custom_prompts[TemplatePromptName.RAG_ANSWER_PROMPT]
    timeline = build_timeline_answer_prompt(stock)

    rendered = timeline.format_messages(
        chat_history=[],
        context="ctx",
        custom_instructions="None",
        files="None",
        task="t",
        rephrased_task="r",
    )

    assert rendered[-1].content.rstrip().endswith(TIMELINE_PROMPT_ADDENDUM.strip())
    # the empty chat-history placeholder renders no message, so just assert the context landed
    assert any("ctx" in message.content for message in rendered)


def test_variants_are_strictly_additive_and_independent() -> None:
    stock = custom_prompts[TemplatePromptName.RAG_ANSWER_PROMPT]

    timeline = build_timeline_answer_prompt(stock).messages[3].prompt.template
    cited = build_cited_answer_prompt(stock).messages[3].prompt.template
    committed = build_cited_committed_answer_prompt(stock).messages[3].prompt.template

    assert TIMELINE_PROMPT_ADDENDUM in timeline
    assert CITATION_PROMPT_ADDENDUM not in timeline

    assert TIMELINE_PROMPT_ADDENDUM in cited and CITATION_PROMPT_ADDENDUM in cited
    assert COMMITMENT_PROMPT_ADDENDUM not in cited, "the plain cited arm must stay reproducible"

    assert CITATION_PROMPT_ADDENDUM in committed
    assert COMMITMENT_PROMPT_ADDENDUM in committed
    assert len(committed) > len(cited) > len(timeline)


def test_narrow_variant_forbids_invented_temporal_structure() -> None:
    stock = custom_prompts[TemplatePromptName.RAG_ANSWER_PROMPT]

    narrow = build_cited_narrow_answer_prompt(stock).messages[3].prompt.template
    committed = build_cited_committed_answer_prompt(stock).messages[3].prompt.template

    assert NARROW_COMMITMENT_PROMPT_ADDENDUM in narrow
    assert CITATION_PROMPT_ADDENDUM in narrow and TIMELINE_PROMPT_ADDENDUM in narrow
    # the broad clause's "only say the records do not show it when ... no conclusion" escape hatch
    # is gone; the narrow one explicitly forbids inferring state changes
    assert COMMITMENT_PROMPT_ADDENDUM not in narrow
    assert "do NOT assert that" in narrow and "stopped" in narrow and "resumed" in narrow
    assert "SAME person" in narrow
    assert narrow != committed
