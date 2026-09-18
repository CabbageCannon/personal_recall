"""Offline tests for the local web UI (no model, no network, no real chat, no browser).

The web layer is the easiest place in this project to grow a second, subtly different product: an
endpoint that retrieves for itself, a page that shows a score nobody should see, a warning that
floods the reader. So the tests here are mostly *negative*:

* the API answers through ``recall.answer_question`` — the function the CLI prints from — and the
  test drives the real one against a fake brain, because "it calls the shared path" is the whole
  reason this layer is allowed to exist;
* nothing is rebuilt per request (an account index takes minutes to build, and a question must not);
* a failed index build leaves a server that still starts and says why;
* a failed question never echoes the question into the response or the console;
* the page renders the conversation, the time range, the participants and the chat lines — and none
  of the internal ids, scores or metadata the API also carries;
* a caveat appears only when the groundedness report has one.

The fake brain deliberately keeps the framework's blocking contract (``Brain.ask`` resolves a loop
with ``asyncio.get_event_loop()`` and fails in a thread that has none), so these tests exercise the
thread/loop wiring that makes the real product answerable from FastAPI at all.

Everything is synthetic. No test starts a server, and no test reads a real export.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from memory import AccountImportReport  # noqa: E402
from webapp import HOST, RecallState, create_app, read_conversation_labels, short_reason  # noqa: E402

SCRATCH_ROOT = BASE_DIR / "tests" / "_scratch_web"

STATIC_DIR = BASE_DIR / "webapp" / "static"

#: A synthetic account: two conversations, one of them in a shard that was never exported. The
#: display names are invented and the talker ids are strings, not wxids.
ALICE = "wxid_synthetic_alice"
BOB = "wxid_synthetic_bob"
LABELS = {ALICE: "小王", BOB: "小李"}

SUPABASE_LINE = "[2025-05-12 17:43] 我: 我最后换成 Supabase 了"
LUNCH_LINE = "[2025-05-13 09:10] 我: 我中午吃了拌粉"
#: Someone describing *their own* situation, which is the R10 defect the screen exists for.
OTHER_SITUATION_LINE = "[2025-05-13 09:10] 对方: 我最近换到了 Supabase"


class FakeSource:
    """One retrieved chunk, shaped like the framework's ``Document``."""

    def __init__(self, content: str, **metadata) -> None:
        self.page_content = content
        self.metadata = metadata


class FakeBrain:
    """A brain that answers from a script and does not touch a model, a network or a disk.

    ``ask`` is the framework's synchronous wrapper over an async pipeline, and it resolves its loop
    with ``asyncio.get_event_loop()`` — which raises in a thread that has none. Reproducing that here
    means a missing loop is a test failure rather than a surprise on the user's first question.
    """

    id = "fake-brain"

    def __init__(self, answer: str, sources: list[FakeSource], *, error: Exception | None = None):
        self.answer_text = answer
        self.sources = sources
        self.error = error
        self.calls: list[str] = []

    async def _aask(self):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            answer=self.answer_text, metadata=SimpleNamespace(sources=self.sources)
        )

    def ask(self, *, run_id, question, retrieval_config, chat_history):
        self.calls.append(question)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._aask())


def source(content: str, *, conversation_id: str = ALICE, index: int = 0) -> FakeSource:
    return FakeSource(
        content,
        chunk_index=index,
        memory_chunk_id=f"{conversation_id}-session-{index:04d}",
        conversation_id=conversation_id,
        start_time=content[1:17],
        end_time=content[1:17],
        participants=["我", "对方"],
        n_events=len(content.splitlines()),
    )


def account_report(*, partial: bool = False) -> AccountImportReport:
    return AccountImportReport(
        shards_detected=("MSG0", "MSG1"),
        shards_exported=("MSG0",) if partial else ("MSG0", "MSG1"),
        conversations_discovered=2,
        conversations_imported=2,
        messages_received=7,
        messages_kept=7,
        first_timestamp=datetime(2025, 5, 12, 17, 43),
        last_timestamp=datetime(2025, 5, 13, 9, 10),
        missing_shards=("MSG1",) if partial else (),
    )


def loaded_state(answer: str, sources: list[FakeSource], **kwargs) -> RecallState:
    """A state whose index is already built — the shape every request sees after startup."""
    state = RecallState(account_dir=SCRATCH_ROOT)
    state.brain = FakeBrain(answer, sources, error=kwargs.get("error"))
    state.retrieval_config = object()
    state.report = kwargs.get("report") or account_report()
    state.conversation_labels = dict(kwargs.get("labels", LABELS))
    return state


def client_for(state: RecallState) -> TestClient:
    return TestClient(create_app(state))


# --- index lifecycle -------------------------------------------------------------------------


def test_a_failed_index_build_leaves_a_server_that_still_starts() -> None:
    """Indexing fails -> `ready: false` + a reason. Never an exception, never a dead process."""
    state = RecallState(account_dir=SCRATCH_ROOT / "does_not_exist")
    state.load()  # a real build over a real (empty) directory: it fails before any model loads

    assert state.ready is False
    assert "FileNotFoundError" in state.detail
    assert len(state.detail) <= 200

    payload = client_for(state).get("/api/status").json()
    assert payload["ready"] is False
    assert payload["conversation_count"] == 0
    assert payload["message_count"] == 0
    assert payload["coverage"] == {}
    assert "no shard export directories" in payload["detail"]


def test_a_long_failure_message_is_clipped_to_one_line() -> None:
    reason = short_reason(RuntimeError("line one\nline two " + "x" * 500))
    assert "\n" not in reason
    assert len(reason) <= 200
    assert reason.startswith("RuntimeError: line one line two")


def test_the_api_answers_through_the_real_shared_assembly() -> None:
    """The endpoint runs the real `recall.answer_question` over a fake brain.

    No monkeypatching here: the response has to carry what only the shared assembly produces — the
    citation resolved into a card whose lines were parsed out of the cited chunk, and a groundedness
    report over the same sources. A private copy of that sequence in the web layer would show up as
    a card that cannot be parsed back to a chat line.
    """
    question = "我最后用了哪个数据库？"
    state = loaded_state(f"你最后换成了 Supabase。[来源 0]", [source(SUPABASE_LINE)])
    payload = client_for(state).post("/api/recall", json={"question": question}).json()

    assert state.brain.calls == [question]
    assert payload["answer"] == "你最后换成了 Supabase。[来源 0]"

    card = payload["evidence"][0]
    # These parsed lines are the evidence that the card came from the shared assembly: turning a
    # cited chunk into timestamped speaker lines is what `build_evidence_cards` does, and the web
    # layer has no code that could produce them. The chunk id that used to be asserted here is now
    # deliberately withheld — it embeds the talker — so the proof rests on the lines instead.
    assert card["lines"] == [
        {"timestamp": "2025-05-12 17:43", "speaker": "我", "text": "我最后换成 Supabase 了"}
    ]
    assert "memory_chunk_id" not in card
    assert set(payload["groundedness"]) == {
        "n_sources", "citations", "invalid_citations", "uncited", "absence_claims",
        "attribution_flags", "citation_mismatches", "warnings",
    }


def test_the_backend_has_no_recall_logic_of_its_own() -> None:
    """Structure, not behaviour: a second retrieval path cannot be added without failing here."""
    source_text = (BASE_DIR / "webapp" / "app.py").read_text(encoding="utf-8")
    assert "recall.answer_question" in source_text, "the API must call the shared recall path"
    for reimplementation in (
        "brain.ask",
        "serialize_sources",
        "build_evidence_cards",
        "from groundedness import",
        "from run_baseline import",
    ):
        assert reimplementation not in source_text, (
            f"webapp/app.py grew its own {reimplementation!r}; the web UI must not re-implement "
            "the product's recall path"
        )


def test_the_api_calls_the_shared_function(monkeypatch) -> None:
    """One call, with the question, and the response is that function's result — nothing else.

    "Nothing else" now includes nothing the page does not render: the cards are projected onto
    ``PAGE_CARD_FIELDS``, so a field the shared assembly produced and the page ignores (``label``,
    ``chunk_index``, the chunk id) is dropped rather than forwarded. The endpoint still adds nothing
    of its own — the equality below is exact, which is what makes that checkable.
    """
    import recall

    seen = {}

    def spy(brain, retrieval_config, *, question, show_uncited=False):
        seen["question"] = question
        seen["show_uncited"] = show_uncited
        return {
            "answer": "你最后换成了 Supabase。[来源 0]",
            "evidence": [{"citation_index": 0, "label": "[我, 对方 · 2025-05-12 17:43]", "lines": []}],
            "groundedness": {"warnings": [], "citation_mismatches": [], "attribution_flags": [],
                             "absence_claims": [], "invalid_citations": 0, "uncited": False},
            "latency_ms": 1234,
            "sources": [{"conversation_id": ALICE}],
        }

    monkeypatch.setattr(recall, "answer_question", spy)
    state = loaded_state("unused", [])

    payload = client_for(state).post("/api/recall", json={"question": "我最后用了哪个数据库？"}).json()

    assert seen == {"question": "我最后用了哪个数据库？", "show_uncited": False}
    assert payload == {
        "answer": "你最后换成了 Supabase。[来源 0]",
        "evidence": [
            {
                "citation_index": 0,
                "lines": [],
                "conversation": "小王",
            }
        ],
        "groundedness": {"warnings": [], "citation_mismatches": [], "attribution_flags": [],
                         "absence_claims": [], "invalid_citations": 0, "uncited": False},
        "latency_ms": 1234,
    }


def test_serving_questions_never_rebuilds_the_index(monkeypatch) -> None:
    """An index takes minutes to build. If a request ever triggered one, that would be the bug."""
    import recall

    def refuse(*_args, **_kwargs):
        raise AssertionError("a request tried to rebuild the account index")

    monkeypatch.setattr(recall, "build_account_session", refuse)
    state = loaded_state(f"我中午吃了拌粉。[来源 0]", [source(LUNCH_LINE)])
    client = client_for(state)

    for _ in range(2):
        assert client.post("/api/recall", json={"question": "我中午吃了什么？"}).status_code == 200
    assert len(state.brain.calls) == 2


# --- the API contract ------------------------------------------------------------------------


def test_recall_returns_the_documented_schema() -> None:
    state = loaded_state(f"你最后换成了 Supabase。[来源 0]", [source(SUPABASE_LINE), source(LUNCH_LINE, index=1)])
    payload = client_for(state).post("/api/recall", json={"question": "我最后用了哪个数据库？"}).json()

    assert set(payload) == {"answer", "evidence", "groundedness", "latency_ms"}
    assert payload["answer"] == "你最后换成了 Supabase。[来源 0]"
    assert isinstance(payload["evidence"], list)
    assert isinstance(payload["groundedness"], dict)
    assert isinstance(payload["latency_ms"], int)
    assert payload["latency_ms"] >= 0


def test_status_reports_counts_and_coverage_only() -> None:
    state = loaded_state("unused", [], report=account_report(partial=True))
    payload = client_for(state).get("/api/status").json()

    assert set(payload) == {"ready", "conversation_count", "message_count", "coverage"}
    assert payload["ready"] is True
    assert payload["conversation_count"] == 2
    assert payload["message_count"] == 7
    assert payload["coverage"] == {
        "first_timestamp": "2025-05-12 17:43:00",
        "last_timestamp": "2025-05-13 09:10:00",
        "partial": True,
    }


def test_status_carries_no_conversation_id_and_no_display_name() -> None:
    """Status is a count surface. Identity does not belong on it, not even to a local page."""
    state = loaded_state("unused", [], report=account_report())
    body = json.dumps(client_for(state).get("/api/status").json(), ensure_ascii=False)

    assert ALICE not in body and BOB not in body
    assert "小王" not in body and "小李" not in body


@pytest.mark.parametrize("question", ["", "   ", "\n\t "])
def test_an_empty_question_is_refused_with_a_message(question: str) -> None:
    state = loaded_state("unused", [])
    response = client_for(state).post("/api/recall", json={"question": question})

    assert response.status_code == 400
    assert response.json()["detail"]
    assert state.brain.calls == [], "nothing may reach retrieval on an empty question"


def test_a_missing_question_field_is_refused_too() -> None:
    response = client_for(loaded_state("unused", [])).post("/api/recall", json={})
    assert response.status_code == 400


def test_the_page_refuses_an_empty_question_before_sending_it() -> None:
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "input.value.trim()" in script
    assert "if (!question)" in script


def test_a_question_is_refused_while_the_index_is_not_ready() -> None:
    state = RecallState(account_dir=SCRATCH_ROOT / "does_not_exist")
    state.detail = "FileNotFoundError: nothing to index"
    response = client_for(state).post("/api/recall", json={"question": "我中午吃了什么？"})

    assert response.status_code == 503
    assert "nothing to index" in response.json()["detail"]


# --- evidence --------------------------------------------------------------------------------


def test_a_cited_source_becomes_a_card_with_the_number_the_answer_used() -> None:
    """`[来源 1]` must produce the card labelled source 1 — the index is 0-based, so no offset."""
    state = loaded_state(
        f"你去拿了外卖。[来源 1]", [source(SUPABASE_LINE), source(LUNCH_LINE, index=1)]
    )
    payload = client_for(state).post("/api/recall", json={"question": "后来呢？"}).json()

    assert [card["citation_index"] for card in payload["evidence"]] == [1]
    card = payload["evidence"][0]
    assert card["participants"] == ["我", "对方"]
    assert card["start_time"] == "2025-05-13 09:10"
    assert [line["text"] for line in card["lines"]] == ["我中午吃了拌粉"]


def test_the_card_carries_the_conversation_display_name() -> None:
    state = loaded_state(f"我中午吃了拌粉。[来源 0]", [source(LUNCH_LINE)])
    card = client_for(state).post("/api/recall", json={"question": "我中午吃了什么？"}).json()["evidence"][0]

    assert card["conversation"] == "小王"


def test_a_conversation_with_no_recorded_name_is_left_unnamed() -> None:
    """No display name on record means no header — a talker id is an identifier, not a label."""
    state = loaded_state(f"我中午吃了拌粉。[来源 0]", [source(LUNCH_LINE)], labels={})
    response = client_for(state).post("/api/recall", json={"question": "我中午吃了什么？"})

    assert "conversation" not in response.json()["evidence"][0]


def test_read_conversation_labels_reads_the_recorded_listing() -> None:
    """A display name comes from the export tree's own listing, or the card goes unnamed."""
    root = SCRATCH_ROOT / "labels"
    try:
        shard = root / "MSG0"
        shard.mkdir(parents=True, exist_ok=True)
        (shard / "sessions.json").write_text(
            json.dumps(
                [{"username": ALICE, "displayName": "小王"}, {"username": BOB}], ensure_ascii=False
            ),
            encoding="utf-8",
        )
        for talker in (ALICE, BOB):
            (shard / f"{talker}_messages.json").write_text("[]", encoding="utf-8")

        assert read_conversation_labels(root) == {ALICE: "小王"}
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_read_conversation_labels_survives_a_missing_directory() -> None:
    assert read_conversation_labels(SCRATCH_ROOT / "nowhere") == {}


def test_a_listing_that_names_a_conversation_after_itself_yields_no_label() -> None:
    """The real-account shape, found by looking at the page in a browser (Phase 20.5).

    ``weflow-cli sessions --json`` returns ``displayName`` **equal to** ``username`` when it has no
    remark or nickname to resolve. Every one of a real account's 272 conversations came back that way,
    so a guard of "is the display name non-empty" passed for all of them and every evidence card
    printed a raw wxid or group id where a conversation name belongs. No synthetic fixture had caught
    it, because a fixture always gives the two values different string constants.
    """
    root = SCRATCH_ROOT / "labels_self"
    talkers = (ALICE, BOB, "synthetic_group@chatroom")
    try:
        shard = root / "MSG0"
        shard.mkdir(parents=True, exist_ok=True)
        (shard / "sessions.json").write_text(
            json.dumps(
                [{"username": t, "displayName": t} for t in talkers], ensure_ascii=False
            ),
            encoding="utf-8",
        )
        for talker in talkers:
            (shard / f"{talker}_messages.json").write_text("[]", encoding="utf-8")

        assert read_conversation_labels(root) == {}, (
            "a talker id printed as a name is the leak this guard exists to prevent"
        )
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_a_group_talker_is_never_a_display_name() -> None:
    """Even a differently-spelled group id is an identifier: the suffix gives it away."""
    root = SCRATCH_ROOT / "labels_group"
    try:
        shard = root / "MSG0"
        shard.mkdir(parents=True, exist_ok=True)
        (shard / "sessions.json").write_text(
            json.dumps(
                [
                    {"username": "synthetic_group@chatroom", "displayName": "other_synthetic@chatroom"},
                    {"username": ALICE, "displayName": "小王"},
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        for talker in ("synthetic_group@chatroom", ALICE):
            (shard / f"{talker}_messages.json").write_text("[]", encoding="utf-8")

        assert read_conversation_labels(root) == {ALICE: "小王"}
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_the_page_shows_no_internal_id_score_or_metadata() -> None:
    """The API carries chunk ids and scores for the CLI's `--json`; the page must not print them."""
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    for internal in (
        "memory_chunk_id",
        "chunk_index",
        "rerank_score",
        "original_file_name",
        "n_events",
        "latency_ms",
    ):
        assert internal not in script, f"the page renders {internal!r}, which is debug metadata"

    # The card renders exactly the fields a reader needs, and nothing that came along with them.
    for field in ("citation_index", "conversation", "start_time", "end_time", "participants", "lines"):
        assert field in script, f"the evidence card lost {field}"


def test_the_api_ships_no_chunk_id_and_no_talker_to_the_browser() -> None:
    """The DOM rule applies to the payload too: a response to a browser is still shown to a browser.

    Found on a real account (Phase 20.5), and found *only* because the check moved from the response's
    key names to its values. A chunk id is ``f"{conversation_id}-session-NNNN"``, so ``memory_chunk_id``
    carries the talker — a group id or a wxid — inside a response that had already been stripped of the
    conversation label. The first version of this check looked for a ``conversation_id`` key, found
    none, and reported a clean payload while the id rode along in another field.
    """
    state = loaded_state(f"你最后换成了 Supabase。[来源 0]", [source(SUPABASE_LINE)])
    payload = client_for(state).post("/api/recall", json={"question": "我最后用了哪个数据库？"}).json()

    card = payload["evidence"][0]
    assert card["citation_index"] == 0
    assert set(card) <= {
        "citation_index",
        "conversation",
        "start_time",
        "end_time",
        "participants",
        "lines",
    }, f"the page card grew a field nobody renders: {sorted(card)}"

    blob = json.dumps(payload, ensure_ascii=False)
    assert ALICE not in blob, "the talker id reached the browser"
    assert "session-" not in blob, "a chunk id reached the browser"
    assert "memory_chunk_id" not in blob
    assert "chunk_index" not in blob


def test_a_named_conversation_still_reaches_the_card_after_the_projection() -> None:
    """Projecting must drop debug fields, not the one field the projection exists to add."""
    state = loaded_state(f"你最后换成了 Supabase。[来源 0]", [source(SUPABASE_LINE)])
    card = client_for(state).post("/api/recall", json={"question": "我最后用了哪个数据库？"}).json()[
        "evidence"
    ][0]
    assert card["conversation"] == "小王"
    assert card["lines"], "the original chat lines are the point of the card"


# --- groundedness ----------------------------------------------------------------------------


def test_a_clean_answer_reports_no_caveats() -> None:
    state = loaded_state(f"你最后换成了 Supabase。[来源 0]", [source(SUPABASE_LINE)])
    report = client_for(state).post("/api/recall", json={"question": "我最后用了哪个数据库？"}).json()[
        "groundedness"
    ]

    assert report["warnings"] == []
    assert report["citation_mismatches"] == []
    assert report["attribution_flags"] == []
    assert report["absence_claims"] == []
    assert report["uncited"] is False


def test_a_citation_binding_mismatch_is_reported() -> None:
    """A quote that is not in the source it cites, but is in another retrieved one, is an alarm."""
    state = loaded_state(
        f'你说“我最后换成 Supabase 了”。[来源 1]',
        [source(SUPABASE_LINE), source(LUNCH_LINE, conversation_id=BOB, index=1)],
    )
    report = client_for(state).post("/api/recall", json={"question": "我最后用了哪个数据库？"}).json()[
        "groundedness"
    ]

    assert report["citation_mismatches"], "the quoted line is in source 0, not in the cited source 1"
    assert report["citation_mismatches"][0]["cited"] == [1]
    assert report["citation_mismatches"][0]["found_in"] == [0]


def test_someone_elses_situation_borrowed_without_attribution_is_reported() -> None:
    """The R10 defect: a claim rests on the other person's own account and never says so."""
    state = loaded_state(
        f"我最近换到了 Supabase。[来源 0]", [source(OTHER_SITUATION_LINE)]
    )
    report = client_for(state).post("/api/recall", json={"question": "我最近怎么样？"}).json()[
        "groundedness"
    ]

    assert report["attribution_flags"], "the answer restates 对方's own situation as the user's"
    assert report["attribution_flags"][0]["matched_line_speaker"] == "对方"


def test_a_silence_claim_is_reported_as_a_reminder_not_an_alarm() -> None:
    state = loaded_state("记录里没有提到这家店。[来源 0]", [source(LUNCH_LINE)])
    report = client_for(state).post("/api/recall", json={"question": "那家店叫什么？"}).json()[
        "groundedness"
    ]

    assert report["absence_claims"]
    assert not report["citation_mismatches"] and not report["attribution_flags"]


def test_the_page_shows_nothing_when_the_report_has_no_caveats() -> None:
    """Both caveat elements ship hidden and are filled from the report — never from a template."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="warning" hidden' in html
    assert 'id="note" hidden' in html
    assert "warningLine.hidden = !alarms.length" in script
    assert "noteLine.hidden = !notes.length" in script
    assert "alarms.length ? " in script, "an empty alarm list must render nothing at all"


def test_the_page_surfaces_exactly_the_actionable_caveats() -> None:
    """Binding mismatch and attribution flags alarm; a silence claim is light text; no more."""
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    alarms = script.split("function renderCaveats")[1].split("warningLine.textContent")[0]
    for field in ("citation_mismatches", "invalid_citations", "attribution_flags"):
        assert field in alarms, f"{field} is an alarm a reader must act on"
    assert "absence_claims" not in alarms, "a silence claim is a reminder, not an alarm"
    assert "记录可能不完整，请结合原始聊天确认。" in script


# --- privacy ---------------------------------------------------------------------------------


def test_a_failed_question_echoes_neither_the_question_nor_the_error(capsys) -> None:
    question = "我最后换成了什么数据库？"
    state = loaded_state(
        "unused", [], error=RuntimeError(f"upstream rejected the request: {question}")
    )
    response = client_for(state).post("/api/recall", json={"question": question})

    assert response.status_code == 502
    assert question not in response.text
    assert "upstream rejected" not in response.text

    printed = capsys.readouterr()
    assert question not in printed.out + printed.err
    assert "RuntimeError" in printed.err, "the failure is still diagnosable, by class name"


def test_the_page_keeps_nothing_in_the_browser_and_logs_nothing() -> None:
    for name in ("app.js", "index.html"):
        text = (STATIC_DIR / name).read_text(encoding="utf-8")
        for storage in ("localStorage", "sessionStorage", "document.cookie", "indexedDB"):
            assert storage not in text, f"{name} persists state via {storage}"

    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert not re.search(r"console\s*\.", script), "the console is one copy away from a support thread"


def test_a_question_is_never_written_to_a_log_by_the_recall_endpoint() -> None:
    """`/api/recall` has no logging call at all; the only print is the failure class name."""
    source_text = (BASE_DIR / "webapp" / "app.py").read_text(encoding="utf-8")
    assert "print(" in source_text  # the one, sanitized failure line
    for line in source_text.splitlines():
        if "print(" in line:
            assert "question" not in line and "{exc}" not in line


# --- the server ------------------------------------------------------------------------------


def test_the_server_binds_localhost_only() -> None:
    """A constant, not a default: this process serves real private chat."""
    import web

    assert HOST == "127.0.0.1"
    assert web.HOST == "127.0.0.1"
    assert web.server_options(8000)["host"] == "127.0.0.1"

    source_text = (BASE_DIR / "web.py").read_text(encoding="utf-8")
    assert "0.0.0.0" not in source_text
    assert "--host" not in source_text, "no flag may make this reachable from the network"


def test_no_address_but_localhost_is_bound_anywhere_in_the_web_layer() -> None:
    for name in ("web.py", "webapp/app.py"):
        text = (BASE_DIR / name).read_text(encoding="utf-8")
        assert "0.0.0.0" not in text
        assert '"::"' not in text


def test_the_page_and_its_assets_are_served() -> None:
    client = client_for(loaded_state("unused", []))

    page = client.get("/")
    assert page.status_code == 200
    assert "Personal Recall" in page.text
    assert "问问过去发生过什么..." in page.text

    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_there_are_no_other_endpoints() -> None:
    """Four routes: the page, its assets, the question and the status. Nothing else exists."""
    app = create_app(loaded_state("unused", []))
    paths = {getattr(route, "path", None) for route in app.routes}

    assert paths == {"/", "/static", "/api/status", "/api/recall"}


def test_the_readme_documents_the_web_ui_as_it_is() -> None:
    """`tests/test_readme.py` cannot probe web.py (it is not a script that spawns help), so the
    drift check for the four things a reader needs lives here."""
    readme = (BASE_DIR / "README.md").read_text(encoding="utf-8")
    assert "Ask from a browser" in readme, "the README lost the web UI section"
    assert "python web.py" in readme, "how to start it"
    assert "http://127.0.0.1:8000" in readme, "where to open it"
    assert "export_account.py" in readme, "the page reads that export tree, so say how to make one"
    assert "data/real/account" in readme, "where the data lives"

    section = readme.split("Ask from a browser", 1)[1].split("\n## ", 1)[0]
    prose = re.sub(r"```.*?```", "", section, flags=re.DOTALL)
    declared = set(
        re.findall(r'add_argument\(\s*"(--[\w-]+)"', (BASE_DIR / "web.py").read_text(encoding="utf-8"))
    )
    documented = set(re.findall(r"--[\w-]+", prose))
    assert documented, "the section must say how to point the server at another export tree"
    assert documented <= declared, (
        f"the README documents {sorted(documented - declared)}, which web.py does not declare"
    )


def test_the_page_has_no_feature_it_was_not_asked_for() -> None:
    """The forbidden list, checked where it would actually appear: the page and the API."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8").lower()
    for forbidden in ("login", "sign in", "signup", "upload", "sidebar", "settings", "history"):
        assert forbidden not in html

    routes = {getattr(route, "path", None) for route in create_app(loaded_state("unused", [])).routes}
    assert not any("upload" in path or "history" in path for path in routes)
