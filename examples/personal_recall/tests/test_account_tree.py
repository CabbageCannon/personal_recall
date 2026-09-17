"""Offline tests for the account export tree: the writer and the reader must agree.

Phase 19 has two halves that meet on disk — `exporter.export_account_tree` writes
``<out>/<shard>/<talker>_messages.json`` plus ``sessions.json`` and ``shard_manifest.json``, and
``memory.account.load_account_directory`` reads exactly that. Neither half is much use alone, and a
mismatch between them (a different file name, a different key) would look like an empty account rather
than an error, so the round trip is tested here rather than assumed.

No real account, no real exporter: a fake runner answers ``sessions``/``export`` by reading the shard
out of the *scratch* config the module just wrote, which also proves the shard-switching mechanism
actually selects the shard it claims to. Nothing touches the developer's own ``~/.weflow-cli``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import exporter  # noqa: E402
from memory import (  # noqa: E402
    SessionConfig,
    build_account_sessions,
    crossed_conversation_chunks,
    import_account,
    load_account_directory,
    shard_stem,
)

SCRATCH_ROOT = BASE_DIR / "tests" / "_scratch_account_tree"


def ts(hour: int, minute: int = 0) -> int:
    return int(datetime(2026, 10, 1, hour, minute).timestamp())


def msg(local_id, server_id, when, text, *, is_send=1):
    return {
        "localId": local_id,
        "serverId": server_id,
        "localType": 1,
        "createTime": when,
        "isSend": is_send,
        "senderUsername": "wxid_synthetic",
        "parsedContent": text,
    }


class FakeExporter:
    """A stand-in for ``weflow-cli`` that reads the shard from the scratch config it was handed.

    That is the mechanism under test as much as the tree layout: if the scratch profile did not switch
    the database, this fake would see the wrong shard and the assertions below would fail.
    """

    def __init__(self, listings, messages, unlistable=(), failing_exports=()):
        self.listings = listings
        self.messages = messages
        self.unlistable = set(unlistable)
        self.failing_exports = set(failing_exports)
        #: Injected verbatim into a failed listing's stderr, so redaction can be tested.
        self.injected_stderr = ""
        self.argv_seen = []
        self.shards_seen = []
        self.scratch_alive_during_call = []

    def __call__(self, argv, env):
        self.argv_seen.append(list(argv))
        scratch = Path(env["USERPROFILE"])
        config_file = scratch / ".weflow-cli" / "config.json"
        self.scratch_alive_during_call.append(config_file.is_file())
        config = json.loads(config_file.read_text(encoding="utf-8"))
        shard = shard_stem(Path(config["dbPath3x"]).name)
        self.shards_seen.append(shard)

        if argv[0] == "sessions":
            if shard in self.unlistable:
                return subprocess.CompletedProcess(
                    argv, 1, stdout="", stderr=f"listing exploded {self.injected_stderr}".strip()
                )
            listing = [
                {"username": talker, "displayName": f"label-{talker}", "type": 0}
                for talker in self.listings.get(shard, ())
            ]
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(listing), stderr="")

        talker = argv[1]
        out_dir = Path(argv[argv.index("--output") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        if (shard, talker) in self.failing_exports:
            return subprocess.CompletedProcess(
                argv, 3, stdout=json.dumps({"success": False, "error": "export failed"}), stderr=""
            )
        payload = self.messages.get((shard, talker), [])
        path = out_dir / f"{talker}_messages.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps({"success": True, "path": str(path), "count": len(payload)}),
            stderr="",
        )


def _make_account(unlistable=(), db_names=("MSG0.db", "MSG1.db", "MSG2.db", "FTSMSG0.db")):
    """A synthetic account: a real-config stand-in, a Msg/Multi dir, and a fake exporter.

    ``db_names`` selects which databases exist on disk, and therefore which shards are *detected*.
    Dropping ``MSG1.db`` yields an account where every detected shard can be exported, which is the
    only shape in which a conversation filter can leave ``missing_shards`` empty.
    """
    shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)

    real_config = SCRATCH_ROOT / "real_config" / ".weflow-cli" / "config.json"
    real_config.parent.mkdir(parents=True, exist_ok=True)
    real_config.write_text(
        json.dumps({"dbPath3x": str(SCRATCH_ROOT / "multi" / "MSG0.db"), "decryptKey3x": "ab" * 32}),
        encoding="utf-8",
    )
    multi = SCRATCH_ROOT / "multi"
    multi.mkdir(parents=True, exist_ok=True)
    for name in db_names:
        (multi / name).write_bytes(b"")

    listings = {"MSG0": ["alice", "bob"], "MSG2": ["alice", "carol@chatroom"]}
    messages = {
        ("MSG0", "alice"): [msg(1, "a-1", ts(10, 0), "我试了那个库")],
        ("MSG0", "bob"): [msg(1, "b-1", ts(10, 1), "吃饭吗")],
        ("MSG2", "alice"): [msg(2, "a-2", ts(10, 2), "冷启动慢")],
        ("MSG2", "carol@chatroom"): [msg(1, "c-1", ts(10, 3), "材料发你了")],
    }
    return {
        "multi": multi,
        "fake": FakeExporter(listings, messages, unlistable=unlistable),
        "real_config": real_config,
        "listings": listings,
        "messages": messages,
        "out": SCRATCH_ROOT / "out",
    }


@pytest.fixture()
def account():
    state = _make_account()
    original = exporter.real_config_path
    exporter.real_config_path = lambda env=None: state["real_config"]  # type: ignore[assignment]
    try:
        yield state
    finally:
        exporter.real_config_path = original  # type: ignore[assignment]
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


@pytest.fixture()
def complete_account():
    """The same account with ``MSG1.db`` absent, so no detected shard can ever be missing.

    That is the shape the filter bug hides in: a narrowed export whose conversations span every
    shard leaves the shard comparison with nothing to complain about.
    """
    state = _make_account(db_names=("MSG0.db", "MSG2.db", "FTSMSG0.db"))
    original = exporter.real_config_path
    exporter.real_config_path = lambda env=None: state["real_config"]  # type: ignore[assignment]
    try:
        yield state
    finally:
        exporter.real_config_path = original  # type: ignore[assignment]
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def _export(state, **kwargs):
    return exporter.export_account_tree(
        state["out"],
        multi_dir=state["multi"],
        runner=state["fake"],
        scratch_root=SCRATCH_ROOT / "scratch",
        **kwargs,
    )


# --------------------------------------------------------------------------------------------
# The layout, and the round trip back through the importer
# --------------------------------------------------------------------------------------------


def test_tree_layout_is_what_the_importer_reads(account):
    report = _export(account)
    out = account["out"]
    assert (out / "shard_manifest.json").is_file()
    assert (out / "MSG0" / "sessions.json").is_file()
    assert (out / "MSG0" / "alice_messages.json").is_file()
    assert (out / "MSG0" / "bob_messages.json").is_file()
    assert (out / "MSG2" / "alice_messages.json").is_file()
    assert (out / "MSG2" / "carol@chatroom_messages.json").is_file()
    # MSG1 exists on disk but was not exported: the report must say PARTIAL, not complete.
    assert report.detected_shards == ("MSG0", "MSG1", "MSG2")
    assert report.exported_shards == ("MSG0", "MSG2")
    assert report.missing_shards == ("MSG1",)
    assert report.partial is True
    assert "PARTIAL" in "\n".join(report.lines())


def test_freshness_indexes_are_not_mistaken_for_message_shards(account):
    """``FTSMSG0.db`` is a search index; counting it would invent a shard that can never be exported."""
    report = _export(account)
    assert "FTSMSG0" not in report.detected_shards
    assert "MSG0" in report.detected_shards


def test_the_written_tree_round_trips_through_the_importer(account):
    """The whole point: what the exporter writes, the account importer reads, and it stays isolated."""
    _export(account)
    layout = load_account_directory(account["out"], shard_dir=account["multi"])
    assert [d.conversation_id for d in layout.descriptors] == ["alice", "bob", "carol@chatroom"]
    assert layout.shards_detected == ("MSG0", "MSG1", "MSG2")

    events_by_conversation, report = import_account(
        layout.exports, shards_detected=layout.shards_detected, discovered=layout.descriptors
    )
    assert report.messages_kept == 4
    assert report.missing_shards == ("MSG1",)
    assert report.partial is True

    chunks = build_account_sessions(events_by_conversation, SessionConfig(max_chars=900))
    assert crossed_conversation_chunks(chunks, events_by_conversation) == ()
    assert sorted(chunk.conversation_id for chunk in chunks) == [
        "alice",
        "bob",
        "carol@chatroom",
    ]
    # alice appears in both exported shards and must still be ONE conversation.
    assert sum(1 for chunk in chunks if chunk.conversation_id == "alice") == 1


def test_alice_is_one_conversation_across_two_shards(account):
    _export(account)
    layout = load_account_directory(account["out"])
    alice = [export for export in layout.exports if export.conversation_id == "alice"]
    assert sorted(export.shard for export in alice) == ["MSG0", "MSG2"]


# --------------------------------------------------------------------------------------------
# The shard-switching mechanism
# --------------------------------------------------------------------------------------------


def test_the_scratch_profile_switched_the_shard_on_every_call(account):
    _export(account)
    fake = account["fake"]
    # Every *detected* shard is listed, MSG1 included and FTSMSG0 still excluded: whether a shard
    # holds conversations can only be learned by asking it, so each one gets its own scratch profile.
    assert sorted(set(fake.shards_seen)) == ["MSG0", "MSG1", "MSG2"]
    assert all(fake.scratch_alive_during_call), "the config must exist while the exporter runs"
    scratch_root = SCRATCH_ROOT / "scratch"
    leftover = list(scratch_root.iterdir()) if scratch_root.is_dir() else []
    assert leftover == [], f"scratch profiles must be gone afterwards, found {leftover}"


def test_argv_is_the_export_contract(account):
    _export(account, only=["alice"])
    exports = [argv for argv in account["fake"].argv_seen if argv[0] == "export"]
    assert exports, "expected at least one export invocation"
    for argv in exports:
        assert argv[1] == "alice"
        assert argv[2] == "json"
        assert "--non-interactive" in argv
        assert argv[argv.index("--limit") + 1] == "0"
        assert "--json" in argv


def test_the_real_config_is_never_written(account):
    before = hashlib.sha256(account["real_config"].read_bytes()).hexdigest()
    _export(account)
    after = hashlib.sha256(account["real_config"].read_bytes()).hexdigest()
    assert before == after


# --------------------------------------------------------------------------------------------
# Failure reporting: a shard may be lost, but never silently
# --------------------------------------------------------------------------------------------


def test_a_shard_that_cannot_be_listed_is_reported_and_the_rest_are_exported():
    state = _make_account(unlistable={"MSG0"})
    original = exporter.real_config_path
    exporter.real_config_path = lambda env=None: state["real_config"]  # type: ignore[assignment]
    try:
        report = exporter.export_account_tree(
            state["out"],
            multi_dir=state["multi"],
            runner=state["fake"],
            scratch_root=SCRATCH_ROOT / "scratch",
        )
        assert report.exported_shards == ("MSG2",)
        assert report.missing_shards == ("MSG0", "MSG1")
        assert report.partial is True
        assert any("MSG0" in failure for failure in report.failures)
        # MSG2's data still made it, so one broken shard does not cost the account its history.
        assert (state["out"] / "MSG2" / "alice_messages.json").is_file()
        assert not (state["out"] / "MSG0" / "alice_messages.json").exists()
    finally:
        exporter.real_config_path = original  # type: ignore[assignment]
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_a_failed_conversation_does_not_stop_its_shard(account):
    """One unreadable conversation costs that conversation, not the shard and not the account."""
    account["fake"].failing_exports = {("MSG0", "bob")}
    report = _export(account)
    assert report.exported_shards == ("MSG0", "MSG2")
    assert report.files_written == 3
    assert any("bob" in failure for failure in report.failures)
    assert (account["out"] / "MSG0" / "alice_messages.json").is_file()
    assert not (account["out"] / "MSG0" / "bob_messages.json").exists()


def test_only_filters_by_exact_id_and_a_display_name_selects_nothing(account):
    _export(account, only=["alice"])
    fake = account["fake"]
    exported = {argv[1] for argv in fake.argv_seen if argv[0] == "export"}
    assert exported == {"alice"}

    account["fake"].argv_seen.clear()
    state2_out = SCRATCH_ROOT / "out2"
    exporter.export_account_tree(
        state2_out,
        multi_dir=account["multi"],
        runner=account["fake"],
        scratch_root=SCRATCH_ROOT / "scratch",
        only=["label-alice"],
    )
    assert not [argv for argv in account["fake"].argv_seen if argv[0] == "export"]


def test_the_manifest_records_no_conversation_identity(account):
    """The manifest is the file most likely to be copied around; it must carry counts, not people."""
    _export(account)
    blob = (account["out"] / "shard_manifest.json").read_text(encoding="utf-8")
    assert "alice" not in blob
    assert "bob" not in blob
    assert "carol" not in blob
    assert "label-" not in blob
    manifest = json.loads(blob)
    assert manifest["schema"] == "privrecall-account-export/v1"
    assert manifest["detected_shards"] == ["MSG0", "MSG1", "MSG2"]
    assert manifest["exported_shards"] == ["MSG0", "MSG2"]


def test_a_subset_of_shards_can_be_exported_and_still_reports_the_whole_account(account):
    report = _export(account, shards=["MSG2"])
    assert report.detected_shards == ("MSG0", "MSG1", "MSG2")
    assert report.exported_shards == ("MSG2",)
    assert report.missing_shards == ("MSG0", "MSG1")
    assert not (account["out"] / "MSG0").exists() or not list(
        (account["out"] / "MSG0").glob("*_messages.json")
    )


def test_exporting_nothing_at_all_is_an_error_not_an_empty_account(account):
    with pytest.raises(exporter.ExporterError):
        exporter.export_account_tree(
            account["out"], multi_dir=SCRATCH_ROOT / "no_such_dir", runner=account["fake"]
        )


def test_errors_are_redacted_before_they_are_reported(account):
    """A shard listing that leaks a key in stderr must not put it in the report."""
    key = "cd" * 32
    account["fake"].unlistable = {"MSG0"}
    account["fake"].injected_stderr = f"decryptKey3x={key}"
    report = _export(account)
    rendered = "\n".join(report.lines()) + json.dumps(report.as_dict())
    assert key not in rendered
    assert "[REDACTED]" in rendered


# --------------------------------------------------------------------------------------------
# A narrowed export is PARTIAL even when every shard is present
# --------------------------------------------------------------------------------------------


def test_a_filtered_export_spanning_every_shard_is_still_partial(complete_account):
    """The regression: ``--only`` narrows what is exported but still writes every shard's listing.

    A filter whose conversations happen to cover every shard therefore leaves nothing "missing" by
    the shard test — and a report that stops there calls 8 exported conversations out of 272 a
    complete account. Every shard was exported; the tree is still PARTIAL, and must say so.
    """
    report = _export(complete_account, only=["alice"])

    # Nothing here is broken: every detected shard produced an export, so a shard-only completeness
    # check sees a healthy account. The filter is the only evidence that it is not the whole account.
    assert report.detected_shards == ("MSG0", "MSG2")
    assert report.exported_shards == ("MSG0", "MSG2")
    assert report.missing_shards == ()
    assert report.filtered_conversations == 1
    assert "narrowed" in "\n".join(report.lines())

    layout = load_account_directory(complete_account["out"], shard_dir=complete_account["multi"])
    assert layout.shards_detected == ("MSG0", "MSG2")
    assert layout.filtered_conversations == 1
    assert any("narrowed" in note for note in layout.notes)

    _, imported = import_account(
        layout.exports,
        shards_detected=layout.shards_detected,
        discovered=layout.descriptors,
        filtered_conversations=layout.filtered_conversations,
    )
    assert imported.missing_shards == (), "no shard was lost - the filter is the whole story"
    assert imported.conversations_discovered == 3, "alice, bob and carol were all discovered"
    assert imported.conversations_imported == 1, "only alice was in the export"
    assert imported.partial is True, "a narrowed export is PARTIAL even with every shard present"

    rendered = "\n".join(imported.lines())
    assert "PARTIAL" in rendered
    assert "narrowed to 1 conversation(s)" in rendered
    assert "were not exported" not in rendered, "the reason is the filter, not a failed shard"
    assert imported.as_dict()["partial"] is True
    assert imported.as_dict()["filtered_conversations"] == 1


def test_an_unfiltered_export_of_a_complete_account_is_not_partial(complete_account):
    """The guard against over-warning: no filter and no missing shard must stay a clean bill."""
    report = _export(complete_account)
    assert report.detected_shards == ("MSG0", "MSG2")
    assert report.missing_shards == ()
    assert report.filtered_conversations == 0
    assert report.partial is False
    assert "PARTIAL" not in "\n".join(report.lines())

    layout = load_account_directory(complete_account["out"], shard_dir=complete_account["multi"])
    assert layout.filtered_conversations == 0
    assert not any("narrowed" in note for note in layout.notes)

    _, imported = import_account(
        layout.exports,
        shards_detected=layout.shards_detected,
        discovered=layout.descriptors,
        filtered_conversations=layout.filtered_conversations,
    )
    assert imported.missing_shards == ()
    assert imported.filtered_conversations == 0
    assert imported.partial is False
    assert "PARTIAL" not in "\n".join(imported.lines())


def test_a_filtered_manifest_records_the_count_and_still_no_conversation_identity(account):
    """The filtered manifest is the one most tempted to record *who* was selected. It must not."""
    report = _export(account, only=["alice"])
    blob = (account["out"] / "shard_manifest.json").read_text(encoding="utf-8")
    assert "alice" not in blob
    assert "bob" not in blob
    assert "carol" not in blob
    assert "label-" not in blob
    manifest = json.loads(blob)
    assert manifest["filtered_conversations"] == 1
    assert report.filtered_conversations == 1

    # An unfiltered run writes 0 rather than omitting the key: "no filter" has to be stated, or a
    # reader cannot tell it apart from a manifest written before this key existed.
    _export(account)
    assert json.loads(
        (account["out"] / "shard_manifest.json").read_text(encoding="utf-8")
    )["filtered_conversations"] == 0
