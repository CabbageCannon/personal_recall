"""The query planner, tested with no database, no model and no key.

The planner is the one layer whose mistakes are silent. A wrong retrieval produces a worse answer,
which a reader can notice; a wrong *filter* produces a confident answer built from evidence that was
never looked at, and nothing downstream can tell. So everything here is about what the plan claims,
checked against questions whose right answer is written down in the test.

The vocabulary is injected rather than read from a store, which is what makes that possible: planning
is a pure function of the question plus the names the account actually holds.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import query_plan  # noqa: E402

#: A synthetic account vocabulary. Two people share 小王 — the collision Phase 20.9 found in real
#: data and the case every identity rule in this project exists for.
NAMES = (
    ("小明", "person-xiaoming"),
    ("小王", "person-wang-a"),
    ("小王", "person-wang-b"),
    ("苍鹭", "person-heron"),
)
CONVERSATIONS = (("项目讨论组", "11111111@chatroom"), ("家庭群", "22222222@chatroom"))

TODAY = datetime(2026, 9, 23, 15, 0)


def plan(question: str) -> query_plan.MemoryQueryPlan:
    return query_plan.plan_query(question, names=NAMES, conversations=CONVERSATIONS, today=TODAY)


# ---------------------------------------------------------------------------------------------
# §B7 — intent classification
# ---------------------------------------------------------------------------------------------

SINGLE_FACT = (
    "服务器用什么配置",
    "会议定在哪一天",
    "他推荐的那个数据库是什么",
)

SPEAKER_ATTRIBUTION = (
    "谁推荐我用 Supabase 的？",
    "这个方案是谁提出来的",
    "谁决定换掉原来的服务",
)

TEMPORAL_STATE = (
    "后来数据库到底用了什么？",
    "最后我们决定改成哪个方案",
    "现在用的是什么版本",
)

MULTI_EVENT_EXHAUSTIVE = (
    "一共改过几次方案？",
    "我们总共讨论过哪些话题",
    "把提过的数据库都列一下",
)

ABSENCE_CLAIM = (
    "小王有没有提过服务器迁移？",
    "我们是否讨论过备份策略",
    "他提没提过上线时间",
)


@pytest.mark.parametrize("question", SINGLE_FACT)
def test_single_fact_questions_route_to_the_default_path(question: str) -> None:
    assert plan(question).intent == "single_fact"
    assert plan(question).strategy == "hybrid"


@pytest.mark.parametrize("question", SPEAKER_ATTRIBUTION)
def test_attribution_questions_are_recognised(question: str) -> None:
    assert plan(question).intent == "speaker_attribution"


@pytest.mark.parametrize("question", TEMPORAL_STATE)
def test_temporal_questions_are_recognised(question: str) -> None:
    assert plan(question).intent == "temporal_state"
    assert plan(question).strategy == "timeline"


@pytest.mark.parametrize("question", MULTI_EVENT_EXHAUSTIVE)
def test_exhaustive_questions_are_recognised(question: str) -> None:
    result = plan(question)
    assert result.intent == "multi_event_exhaustive"
    assert result.strategy == "exhaustive"
    assert result.requires_exhaustive_recall is True


@pytest.mark.parametrize("question", ABSENCE_CLAIM)
def test_absence_questions_are_recognised(question: str) -> None:
    assert plan(question).intent == "absence_claim"


def test_an_unrecognised_question_is_a_single_fact_not_an_error() -> None:
    """The fallback has to be the *widest* path, not an exception: a plan that fails is worse."""
    result = plan("嗯")
    assert result.intent == "single_fact"
    assert result.strategy == "hybrid"
    assert result.person_ids == []


def test_the_more_specific_intent_wins_when_two_match() -> None:
    """"谁最后改的" is both an attribution and a temporal question. The person is the harder half."""
    assert plan("谁最后改的方案").intent == "speaker_attribution"


# ---------------------------------------------------------------------------------------------
# §B7 — filter extraction
# ---------------------------------------------------------------------------------------------


def test_a_named_person_is_resolved_from_the_accounts_own_display_names() -> None:
    result = plan("小明上次说的那个方案")
    assert result.people == ["小明"]
    assert result.person_ids == ["person-xiaoming"]
    assert result.strategy == "person_hybrid"


def test_a_question_naming_no_person_extracts_no_person() -> None:
    """The rule that stops a language model's guess from becoming a filter: only real names match."""
    result = plan("Alice 说过什么")
    assert result.people == []
    assert result.person_ids == []


def test_a_shared_display_name_resolves_to_both_people_and_says_so() -> None:
    """Two people called 小王 must never collapse to one — and the filter is a union, not a guess."""
    result = plan("小王有没有提过服务器迁移？")
    assert result.people == ["小王"]
    assert sorted(result.person_ids) == ["person-wang-a", "person-wang-b"]
    assert result.filter().person_ids == ("person-wang-a", "person-wang-b")


def test_resolving_a_person_routes_to_the_narrowed_shape() -> None:
    """Narrowing is orthogonal to intent, and naming it keeps two very different searches apart."""
    result = plan("苍鹭说的是哪个方案")
    assert result.intent == "single_fact"
    assert result.strategy == "person_hybrid"
    assert result.person_ids == ["person-heron"]


def test_an_attribution_question_that_names_nobody_stays_a_wide_search() -> None:
    """The person is what the answer must *supply* here, so the search cannot be narrowed to them."""
    result = plan("谁推荐我用 Supabase 的？")
    assert result.intent == "speaker_attribution"
    assert result.strategy == "hybrid"
    assert result.person_ids == []


def test_a_named_conversation_scopes_the_search() -> None:
    result = plan("项目讨论组里说的那个时间")
    assert result.conversation_scope == ["11111111@chatroom"]
    assert result.filter().conversation_ids == ("11111111@chatroom",)


def test_an_absolute_date_becomes_a_closed_window() -> None:
    result = plan("2023年5月我们决定了什么")
    assert result.time_range is not None
    assert result.time_range.start == datetime(2023, 5, 1)
    assert result.time_range.end == datetime(2023, 5, 31, 23, 59, 59)


def test_a_relative_expression_is_resolved_against_the_given_day() -> None:
    """``today`` is injected, so the test does not change meaning as the calendar moves."""
    result = plan("上周讨论的那个方案")
    assert result.time_range is not None
    assert result.time_range.end == TODAY.replace(hour=23, minute=59, second=59)
    assert result.time_range.start.day == 16


def test_no_time_expression_means_no_time_filter() -> None:
    assert plan("服务器用什么配置").time_range is None


def test_the_plan_becomes_a_retrieval_filter_with_every_field() -> None:
    result = query_plan.plan_query(
        "小明在2023年5月于项目讨论组说了什么",
        names=NAMES,
        conversations=CONVERSATIONS,
        today=TODAY,
    )
    active = result.filter()
    assert active.person_ids == ("person-xiaoming",)
    assert active.conversation_ids == ("11111111@chatroom",)
    assert active.start_time == datetime(2023, 5, 1)
    assert not active.is_empty


def test_a_plan_that_narrows_nothing_produces_an_empty_filter() -> None:
    assert plan("服务器用什么配置").filter().is_empty


def test_describe_never_names_a_person_or_a_conversation() -> None:
    """The plan is logged. A log line is not a place for an identity."""
    described = str(plan("小明在项目讨论组里说了什么").describe())
    assert "小明" not in described
    assert "@chatroom" not in described


# ---------------------------------------------------------------------------------------------
# §B2 — the LLM path refines; it never becomes the only plan
# ---------------------------------------------------------------------------------------------


class BrokenLLM:
    def with_structured_output(self, schema):  # noqa: ANN001, ANN201
        raise RuntimeError("the model is unavailable")


class LyingLLM:
    """Returns a plan that invents a person. The grounded fields must survive it."""

    def with_structured_output(self, schema):  # noqa: ANN001, ANN201
        return self

    def invoke(self, prompt: str):  # noqa: ANN201
        return query_plan.MemoryQueryPlan(
            intent="multi_event_exhaustive",
            people=["不存在的人"],
            person_ids=["invented-person-id"],
            requires_exhaustive_recall=True,
        )


def test_a_model_that_fails_leaves_the_rule_plan_untouched() -> None:
    base = plan("一共改过几次方案？")
    refined = query_plan.plan_with_llm("一共改过几次方案？", BrokenLLM(), base=base)
    assert refined is base
    assert refined.source == "rules"


def test_a_model_cannot_widen_the_taxonomy_or_invent_a_person() -> None:
    """An invented person id narrows the search to nothing, and no later stage could notice."""
    base = plan("一共改过几次方案？")
    refined = query_plan.plan_with_llm("一共改过几次方案？", LyingLLM(), base=base)
    assert refined.source == "llm"
    assert refined.person_ids == base.person_ids == []
    assert refined.people == []
    # and the strategy is recomputed from the *grounded* fields rather than trusted
    assert refined.strategy == "exhaustive"


def test_no_model_means_the_rule_plan() -> None:
    base = plan("后来怎么样了")
    assert query_plan.plan_with_llm("后来怎么样了", None, base=base) is base


# ---------------------------------------------------------------------------------------------
# §B5/§B6 — the evidence shaping each strategy asks for
# ---------------------------------------------------------------------------------------------


def document(chunk_id: str, start: str, conversation: str) -> object:
    from langchain_core.documents import Document

    return Document(
        page_content=f"evidence {chunk_id}",
        metadata={
            "memory_chunk_id": chunk_id,
            "start_time": start,
            "conversation_id": conversation,
        },
    )


def test_duplicate_evidence_is_collapsed_before_it_is_counted() -> None:
    """An exhaustive question is answered by counting. Counting a list with repeats over-counts."""
    documents = [
        document("c1", "2024-01-01 10:00:00", "conv-a"),
        document("c1", "2024-01-01 10:00:00", "conv-a"),
        document("c2", "2024-01-02 10:00:00", "conv-a"),
    ]
    assert len(query_plan.dedupe_by_event(documents)) == 2


def test_chronological_order_is_by_the_time_the_session_started() -> None:
    documents = [
        document("c2", "2024-03-01 09:00:00", "conv-a"),
        document("c1", "2021-01-01 09:00:00", "conv-a"),
        document("c3", "2022-06-01 09:00:00", "conv-a"),
    ]
    assert [d.metadata["memory_chunk_id"] for d in query_plan.chronological(documents)] == [
        "c1",
        "c3",
        "c2",
    ]
    newest = query_plan.chronological(documents, newest_first=True)
    assert newest[0].metadata["memory_chunk_id"] == "c2"


def test_grouping_keeps_each_conversation_chronological() -> None:
    documents = [
        document("a2", "2024-02-01 09:00:00", "conv-a"),
        document("b1", "2024-01-15 09:00:00", "conv-b"),
        document("a1", "2024-01-01 09:00:00", "conv-a"),
    ]
    grouped = query_plan.group_by_conversation(documents)
    assert [d.metadata["memory_chunk_id"] for d in grouped["conv-a"]] == ["a1", "a2"]
    assert [d.metadata["memory_chunk_id"] for d in grouped["conv-b"]] == ["b1"]


def test_the_timeline_strategy_puts_the_latest_evidence_first() -> None:
    """Otherwise the model answers with the earliest thing it read, which is the wrong decade."""
    result = query_plan.PlanResult(
        plan=plan("后来怎么样了"),
        documents=[
            document("old", "2020-01-01 09:00:00", "conv-a"),
            document("new", "2025-01-01 09:00:00", "conv-a"),
        ],
    )
    ordered = query_plan.apply_strategy(result)
    assert ordered.documents[0].metadata["memory_chunk_id"] == "new"
    assert any("newest first" in note for note in ordered.notes)


def test_the_exhaustive_strategy_keeps_more_than_a_top_twenty() -> None:
    """The structural fix §B6 names: twenty is not a sample of "how many times"."""
    result = query_plan.PlanResult(
        plan=plan("一共改过几次方案？"),
        documents=[document(f"c{i:03d}", f"2024-01-01 09:{i:02d}:00", "conv-a") for i in range(120)],
    )
    kept = query_plan.apply_strategy(result)
    assert len(kept.documents) > query_plan.STRATEGY_K["hybrid"]
    assert len(kept.documents) == query_plan.STRATEGY_K["exhaustive"]
    assert any("wider than the answer window" in note for note in kept.notes)


def test_every_strategy_widens_the_pool_it_asks_for() -> None:
    assert query_plan.STRATEGY_POOL["exhaustive"] > query_plan.STRATEGY_POOL["timeline"]
    assert query_plan.STRATEGY_POOL["timeline"] > query_plan.STRATEGY_POOL["hybrid"]
    assert query_plan.STRATEGY_K["exhaustive"] > query_plan.STRATEGY_K["timeline"]


def test_the_strategy_config_changes_only_the_pool_and_the_cut() -> None:
    """The evaluated numbers — weights, BM25 parameters, RRF constant — are not the planner's."""
    from quivr_core.rag.entities.config import HybridConfig, LLMEndpointConfig, RetrievalConfig

    base = RetrievalConfig(
        llm_config=LLMEndpointConfig(),
        k=20,
        hybrid_config=HybridConfig(enabled=True, candidate_k=30, k1=1.2, b=0.75),
    )
    widened = query_plan.retrieval_config_for(base, "exhaustive")
    assert widened.k == query_plan.STRATEGY_K["exhaustive"]
    assert widened.hybrid_config.candidate_k == query_plan.STRATEGY_POOL["exhaustive"]
    assert widened.hybrid_config.weights == base.hybrid_config.weights
    assert widened.hybrid_config.k1 == base.hybrid_config.k1
    assert widened.hybrid_config.b == base.hybrid_config.b
    assert widened.hybrid_config.rrf_c == base.hybrid_config.rrf_c


def test_an_unknown_strategy_is_rejected_rather_than_guessed() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        query_plan.MemoryQueryPlan(strategy="something-else")  # type: ignore[arg-type]
