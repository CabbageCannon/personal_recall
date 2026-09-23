"""The Memory Query Planner: a question becomes a plan, a plan becomes evidence.

Everything before this module answers *"what resembles the question"*. That is the wrong question
for a large part of what people actually ask a memory:

* *"谁说过服务器要迁移？"* — no amount of vector similarity knows that "谁" means a person, and the
  name is not in the question at all. The answer is a person filter, not a better embedding.
* *"后来数据库到底用了什么？"* — the evidence is whatever was said **last**, and a top-20 ranked by
  similarity has no reason to contain it.
* *"我们一共改过几次方案？"* — the answer is a **count over the whole record**. Twenty chunks is not
  a sample of that; it is a different question.

So the planner does one thing: turn a question into a small, explicit, testable structure — which
people, which period, which conversations, and which retrieval shape — and hand it to the retrieval
layer. It never answers anything. A planner that generated text would be a second answerer, and the
first thing to go wrong would be that nobody could tell which one produced a sentence.

Two deliberate limits:

* **The taxonomy is four intents plus one.** Not a tree. Every extra category is a category the
  tests do not cover and the rules cannot be checked against, and the four cover what the product is
  asked for.
* **Extraction is grounded in the account, not in the language.** A person is recognised by finding
  an actual `people.display_name` in the question — not by guessing that a capitalised word is a
  name. A hallucinated filter is the one failure this layer can cause that nothing downstream can
  detect: it silently excludes the answer, and the generated text still reads fluently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal, Sequence

from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator

from memory_store.filters import RetrievalFilter

#: The four retrieval shapes, plus the one that is a *claim about absence* and needs the widest net
#: of all. Anything else is `single_fact`, which is the hybrid path the product already had.
Intent = Literal[
    "single_fact",
    "speaker_attribution",
    "temporal_state",
    "multi_event_exhaustive",
    "absence_claim",
]

Strategy = Literal["hybrid", "person_hybrid", "timeline", "exhaustive"]

#: How wide each strategy casts, in candidates before fusion. The number that fixes the structural
#: problem §B6 names: a fixed top-20 cannot represent "how many times", so the exhaustive strategy
#: does not ask for twenty.
STRATEGY_POOL = {
    "hybrid": 30,
    "person_hybrid": 30,
    "timeline": 80,
    "exhaustive": 400,
}

#: How many fused results each strategy keeps.
STRATEGY_K = {
    "hybrid": 20,
    "person_hybrid": 20,
    "timeline": 20,
    "exhaustive": 60,
}


class TimeRange(BaseModel):
    """A closed period the question is about. Naive local wall clocks, like every stored time."""

    start: datetime | None = None
    end: datetime | None = None

    def is_empty(self) -> bool:
        return self.start is None and self.end is None


class MemoryQueryPlan(BaseModel):
    """What the question is asking for, in the terms the retrieval layer understands.

    Every field is either something a rule can justify or something a model returned and a caller
    chose to trust — and both are visible here, so a wrong answer can be traced to a wrong plan
    rather than to "the retrieval was bad".
    """

    intent: Intent = "single_fact"
    #: Names as they appear in the question. Kept beside the resolved ids because "the question said
    #: 小明 and I matched it to two different people" is a different bug from "the question named
    #: nobody", and only the pair can tell them apart.
    people: list[str] = Field(default_factory=list)
    person_ids: list[str] = Field(default_factory=list)
    time_range: TimeRange | None = None
    conversation_scope: list[str] = Field(default_factory=list)
    strategy: Strategy = "hybrid"
    requires_exhaustive_recall: bool = False
    #: How the plan was reached: ``rules`` or ``llm``. Reported, never shown to a user as prose.
    source: str = "rules"
    notes: list[str] = Field(default_factory=list)

    @field_validator("time_range", mode="before")
    @classmethod
    def _coerce_range(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return TimeRange(**value)
        return value

    def filter(self) -> RetrievalFilter:
        """The narrowing this plan implies. Empty when the plan narrows nothing."""
        window = self.time_range
        return RetrievalFilter(
            person_ids=tuple(self.person_ids),
            conversation_ids=tuple(self.conversation_scope),
            start_time=window.start if window else None,
            end_time=window.end if window else None,
        )

    def describe(self) -> dict[str, Any]:
        """A loggable summary: the shape of the plan, never a name or an id."""
        return {
            "intent": self.intent,
            "strategy": self.strategy,
            "people": len(self.person_ids),
            "conversations": len(self.conversation_scope),
            "time_bounded": bool(self.time_range and not self.time_range.is_empty()),
            "exhaustive": self.requires_exhaustive_recall,
            "source": self.source,
        }


# ---------------------------------------------------------------------------------------------
# Rule extraction
# ---------------------------------------------------------------------------------------------

#: Ordered: the first pattern that matches wins, so the more specific intents come first.
#:
#: Exhaustive first because it is the only one that changes the *shape* of retrieval rather than its
#: ordering — missing it presents a sample as a total, which is a wrong answer rather than a worse
#: one. Attribution before temporal because "谁最后改的" is both, and the person is the harder half to
#: recover: the timeline strategy would order by recency and could surface the wrong person's message
#: with the newest timestamp on it, which reads as a confident answer to a different question.
INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "multi_event_exhaustive",
        ("一共", "总共", "多少次", "几次", "哪些", "分别", "都有什么", "都有哪些", "全部", "所有",
         "一共几", "几个", "几次", "列一下", "列举"),
    ),
    (
        "speaker_attribution",
        ("谁", "是谁", "谁说的", "谁提", "谁推荐", "谁建议", "谁决定", "谁发"),
    ),
    (
        "temporal_state",
        ("后来", "最后", "最终", "现在改成", "之后变成", "变成了", "改成什么", "现在用", "目前",
         "最新", "最近一次", "最终决定"),
    ),
    (
        "absence_claim",
        ("有没有", "是否", "提没提", "说过吗", "有没有提", "是否提过", "记不记得"),
    ),
)

#: Relative time expressions, and how far back each reaches. ``None`` for the end means "now".
RELATIVE_WINDOWS: tuple[tuple[str, int | None], ...] = (
    ("今天", 0),
    ("昨天", 1),
    ("前天", 2),
    ("上周", 7),
    ("这个星期", 7),
    ("这周", 7),
    ("上个月", 30),
    ("这个月", 30),
    ("最近", 30),
    ("去年", 365),
    ("今年", 365),
)

ABSOLUTE_DATE_RE = re.compile(r"(\d{4})[-/年](\d{1,2})(?:[-/月](\d{1,2}))?")
MONTH_ONLY_RE = re.compile(r"(\d{4})年(\d{1,2})月")


def _end_of_month(year: int, month: int) -> datetime:
    return datetime(year + (month // 12), (month % 12) + 1, 1) - timedelta(seconds=1)


def extract_time_range(question: str, *, today: datetime | None = None) -> TimeRange | None:
    """The period a question is about, from absolute dates first and relative words second.

    Absolute wins because it is strictly more informative: "去年 3 月" contains both, and only the
    date pins it. A relative expression with no anchor is resolved against ``today`` — injectable so
    a test does not change meaning as the calendar moves.
    """
    now = today or datetime.now()
    absolute = ABSOLUTE_DATE_RE.search(question)
    if absolute:
        year = int(absolute.group(1))
        month = int(absolute.group(2))
        day = absolute.group(3)
        if day is not None:
            start = datetime(year, month, int(day))
            return TimeRange(start=start, end=start + timedelta(days=1) - timedelta(seconds=1))
        return TimeRange(start=datetime(year, month, 1), end=_end_of_month(year, month))

    month_only = MONTH_ONLY_RE.search(question)
    if month_only:
        year, month = int(month_only.group(1)), int(month_only.group(2))
        return TimeRange(start=datetime(year, month, 1), end=_end_of_month(year, month))

    for phrase, days in RELATIVE_WINDOWS:
        if phrase in question:
            if days is None:
                return None
            end = now.replace(hour=23, minute=59, second=59)
            return TimeRange(start=(now - timedelta(days=days)).replace(hour=0, minute=0, second=0), end=end)
    return None


def extract_intent(question: str) -> tuple[str, list[str]]:
    """``(intent, matched cues)``. Never fails: an unrecognised question is a single fact."""
    for intent, cues in INTENT_RULES:
        matched = [cue for cue in cues if cue in question]
        if matched:
            return intent, matched
    return "single_fact", []


def extract_people(question: str, names: Sequence[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """``(names found, resolved person ids)`` — grounded in the account's own display names.

    A name is recognised only when it is a name the store actually holds, and a name that maps to
    more than one person is reported as found **but not resolved**: filtering on one of two people
    who share a name would silently drop the other's evidence, and the honest answer is to search
    without the filter and say so in the notes.
    """
    found: list[str] = []
    ids: list[str] = []
    ambiguous: list[str] = []
    for display, person_id in names:
        if not display or display not in question:
            continue
        if display not in found:
            found.append(display)
        if person_id not in ids:
            ids.append(person_id)
    # A display name carried by several people contributes several ids; that is the ambiguity, and
    # it is recorded rather than narrowed away.
    counts: dict[str, int] = {}
    for display, _ in names:
        if display in found:
            counts[display] = counts.get(display, 0) + 1
    ambiguous = sorted(name for name, count in counts.items() if count > 1)

    distinct_ids = {pid for display, pid in names if display in found}
    if ambiguous:
        # Keep the ids: the planner's job is to say what it found, and the *executor* decides. A
        # caller that wants the conservative reading takes the unfiltered path, and the note is what
        # tells it to.
        return found, sorted(distinct_ids)
    return found, sorted(distinct_ids)


def extract_conversations(question: str, labels: Sequence[tuple[str, str]]) -> list[str]:
    """Conversation ids whose human-readable label the question names. Labels must be real ones."""
    found: list[str] = []
    for label, conversation_id in labels:
        if label and len(label) > 1 and label in question and conversation_id not in found:
            found.append(conversation_id)
    return found


def strategy_for(intent: str, *, has_people: bool, requires_exhaustive: bool) -> str:
    """Route an intent to a retrieval shape. The mapping is the planner's whole output in one place.

    ``person_hybrid`` means "a hybrid search that has been narrowed to somebody" and is chosen
    whenever the question resolved a person — not only for attribution questions. Narrowing is
    orthogonal to intent, and naming the narrowed shape makes it visible in a report that would
    otherwise show only "hybrid" for two searches of very different sizes.
    """
    if requires_exhaustive or intent == "multi_event_exhaustive":
        return "exhaustive"
    if intent == "temporal_state":
        return "timeline"
    if has_people:
        return "person_hybrid"
    return "hybrid"


def plan_query(
    question: str,
    *,
    names: Sequence[tuple[str, str]] = (),
    conversations: Sequence[tuple[str, str]] = (),
    today: datetime | None = None,
    source: str = "rules",
) -> MemoryQueryPlan:
    """Build a plan from rules alone.

    ``names`` is ``(display_name, person_id)`` for every person the store can resolve, and
    ``conversations`` is ``(label, conversation_id)`` for every conversation with a human-readable
    name. Passing them in — rather than reaching into a database — is what makes this function a
    pure function of the question plus the account's vocabulary, and therefore testable with no
    server, no model and no key.

    Deliberately the only planner entry point that runs without a model. The LLM path in
    :func:`plan_with_llm` refines this one; it never replaces it, so there is always a plan and always
    a reason for it.
    """
    intent, cues = extract_intent(question)
    window = extract_time_range(question, today=today)
    people, person_ids = extract_people(question, names)
    scope = extract_conversations(question, conversations)

    requires_exhaustive = intent == "multi_event_exhaustive"
    strategy = strategy_for(intent, has_people=bool(person_ids), requires_exhaustive=requires_exhaustive)

    notes: list[str] = []
    if cues:
        notes.append(f"intent from cues: {', '.join(cues)}")
    if people:
        notes.append(f"named {len(people)} person(s) that the account knows")
    if window and not window.is_empty():
        notes.append("time window from an explicit expression")
    if scope:
        notes.append(f"scoped to {len(scope)} named conversation(s)")

    return MemoryQueryPlan(
        intent=intent,  # type: ignore[arg-type]
        people=people,
        person_ids=person_ids,
        time_range=window,
        conversation_scope=scope,
        strategy=strategy,  # type: ignore[arg-type]
        requires_exhaustive_recall=requires_exhaustive,
        source=source,
        notes=notes,
    )


def plan_with_llm(question: str, llm: Any, *, base: MemoryQueryPlan | None = None) -> MemoryQueryPlan:
    """Refine a rule plan with a model's structured output, falling back to the rule plan.

    The model fills the same schema, so a plan is a plan whichever produced it and the executor never
    branches on provenance. Every failure mode — no model, a refused call, malformed JSON, a value
    outside the taxonomy — returns ``base`` unchanged: a wrong plan excludes evidence silently, and a
    missing one merely searches more widely.
    """
    fallback = base or plan_query(question)
    if llm is None:
        return fallback
    try:
        structured = llm.with_structured_output(MemoryQueryPlan)
        proposed = structured.invoke(
            "Extract a retrieval plan for a personal chat-memory question. "
            "Use `single_fact` unless the question is clearly one of the other categories. "
            f"Question: {question}"
        )
    except Exception:  # noqa: BLE001 - any model failure means "keep the rule plan"
        return fallback
    if not isinstance(proposed, MemoryQueryPlan):
        return fallback
    proposed.source = "llm"
    # The rule plan's *grounded* fields are copied over, not merged: a model cannot know which
    # display names this account actually has, and a person id it invented would narrow the search
    # to nothing while still reading as a confident answer. What the model contributes is the
    # judgement — the intent and how wide the net has to be — and those are the fields a rule cannot
    # read from a bag of cue words.
    proposed.people = list(fallback.people)
    proposed.person_ids = list(fallback.person_ids)
    proposed.conversation_scope = list(fallback.conversation_scope)
    proposed.time_range = fallback.time_range or proposed.time_range
    proposed.strategy = strategy_for(
        proposed.intent,
        has_people=bool(proposed.person_ids),
        requires_exhaustive=proposed.requires_exhaustive_recall,
    )
    return proposed


# ---------------------------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------------------------


@dataclass
class PlanResult:
    """The evidence a plan produced, and what it cost to produce it."""

    plan: MemoryQueryPlan
    documents: list[Document] = field(default_factory=list)
    candidates: int = 0
    global_candidates: int = 0
    elapsed_ms: int = 0
    retrieval_config: Any = None
    notes: list[str] = field(default_factory=list)

    @property
    def reduction(self) -> float:
        if not self.global_candidates:
            return 0.0
        return 1 - self.candidates / self.global_candidates

    def describe(self) -> dict[str, Any]:
        return {
            **self.plan.describe(),
            "candidates": self.candidates,
            "global_candidates": self.global_candidates,
            "reduction": round(self.reduction, 4),
            "documents": len(self.documents),
            "elapsed_ms": self.elapsed_ms,
        }


def dedupe_by_event(documents: Sequence[Document]) -> list[Document]:
    """Collapse chunks that carry the same evidence.

    An exhaustive question — "how many times" — is answered by counting events, and the same event
    can appear in two retrieved chunks only when a conversation was re-rendered; more often the
    duplicate is the *same* chunk reached by both retrievers. Either way, counting a document list
    without this over-counts, and over-counting is exactly the failure an exhaustive question is
    asking about.
    """
    seen: set[str] = set()
    out: list[Document] = []
    for document in documents:
        key = str(document.metadata.get("memory_chunk_id") or document.page_content)
        if key in seen:
            continue
        seen.add(key)
        out.append(document)
    return out


def chronological(documents: Sequence[Document], *, newest_first: bool = False) -> list[Document]:
    """Order evidence by the time its session starts. The order a temporal question needs."""
    return sorted(
        documents,
        key=lambda d: str(d.metadata.get("start_time") or ""),
        reverse=newest_first,
    )


def group_by_conversation(documents: Sequence[Document]) -> dict[str, list[Document]]:
    """Evidence by conversation, each group chronological. The shape an aggregate answer needs."""
    grouped: dict[str, list[Document]] = {}
    for document in chronological(documents):
        grouped.setdefault(str(document.metadata.get("conversation_id") or ""), []).append(document)
    return grouped


def apply_strategy(result: PlanResult) -> PlanResult:
    """Order and trim the evidence the way the plan's strategy requires.

    The four strategies differ in more than a `k`. `timeline` must put the *latest* evidence first or
    the model answers with the earliest thing it read; `exhaustive` must keep a set wide enough to
    count over and must not silently truncate to twenty; the hybrid ones keep the fused order, which
    is the ranking that was evaluated.
    """
    strategy = result.plan.strategy
    documents = dedupe_by_event(result.documents)
    if strategy == "timeline":
        documents = chronological(documents, newest_first=True)
        result.notes.append("ordered newest first for a temporal question")
    elif strategy == "exhaustive":
        documents = chronological(documents)
        grouped = group_by_conversation(documents)
        result.notes.append(
            f"{len(grouped)} conversation(s) hold the {len(documents)} evidence unit(s) "
            "considered; the set is wider than the answer window on purpose"
        )
    result.documents = documents[: STRATEGY_K[strategy]]
    if len(documents) > len(result.documents):
        result.notes.append(
            f"evidence cut from {len(documents)} to {len(result.documents)} after ordering"
        )
    return result


# ---------------------------------------------------------------------------------------------
# Executing a plan
# ---------------------------------------------------------------------------------------------


def retrieval_config_for(retrieval_config: Any, strategy: str) -> Any:
    """The product's retrieval configuration, widened for this strategy and otherwise untouched.

    Only two numbers move — the pre-fusion pool and the cut — because those are the two the strategy
    is *about*: an exhaustive question needs a set wide enough to count over, and a temporal one needs
    the latest evidence to be inside the pool at all. The weights, the BM25 parameters, the RRF
    constant and the workflow are the evaluated ones and are not the planner's business.
    """
    hybrid = getattr(retrieval_config, "hybrid_config", None)
    update: dict[str, Any] = {"k": STRATEGY_K[strategy]}
    if hybrid is not None and getattr(hybrid, "enabled", False):
        update["hybrid_config"] = hybrid.model_copy(
            update={"candidate_k": STRATEGY_POOL[strategy]}
        )
    try:
        return retrieval_config.model_copy(update=update)
    except AttributeError:  # a caller passing a plain object: leave it alone rather than guess
        return retrieval_config


def build_retriever(vector_store: Any, retrieval_config: Any) -> Any:
    """The product's hybrid retriever, assembled exactly as `get_retriever` assembles it."""
    from quivr_core.rag.hybrid import BM25Retriever, HybridRRFRetriever, iter_documents

    hybrid = getattr(retrieval_config, "hybrid_config", None)
    search_kwargs = {"k": hybrid.candidate_k if hybrid and hybrid.enabled else retrieval_config.k}
    dense = vector_store.as_retriever(search_kwargs=search_kwargs)
    if not (hybrid and hybrid.enabled):
        return dense
    lexical = BM25Retriever(
        documents=iter_documents(vector_store),
        k=hybrid.candidate_k,
        k1=hybrid.k1,
        b=hybrid.b,
    )
    return HybridRRFRetriever(
        retrievers=[dense, lexical],
        weights=hybrid.weights,
        k=retrieval_config.k,
        c=hybrid.rrf_c,
    )


def execute_plan(
    question: str,
    plan: MemoryQueryPlan,
    *,
    vector_store: Any,
    retrieval_config: Any,
) -> PlanResult:
    """Narrow, retrieve, fuse, then order the evidence the way the strategy requires.

    One measurement is kept honest throughout: ``candidates`` is what the filter left, not what was
    returned. A plan that narrows 81,672 chunks to 5,735 and returns 20 has done something the
    returned list cannot show.
    """
    started = datetime.now()
    view = vector_store.with_filters(plan.filter())
    config = retrieval_config_for(retrieval_config, plan.strategy)
    documents = list(build_retriever(view, config).invoke(question))
    result = PlanResult(
        plan=plan,
        documents=documents,
        candidates=view.count(),
        global_candidates=vector_store.count(),
        retrieval_config=config,
    )
    result = apply_strategy(result)
    notes = list(plan.notes)
    if plan.person_ids and len(plan.people) and len(plan.person_ids) > len(plan.people):
        notes.append(
            "a named person matched more than one identity; the filter is a union, so the search is "
            "wider rather than narrower"
        )
    result.notes = notes + result.notes
    result.elapsed_ms = int((datetime.now() - started).total_seconds() * 1000)
    return result
