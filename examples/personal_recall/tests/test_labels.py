"""Offline tests for human-readable conversation names (Phase 20.6).

Three things are under test, and each has a failure mode that is invisible until a real account hits
it:

* **the rule** — ``memory.labels.usable_name``. On the tested ``weflow-cli`` every ``displayName``
  comes back equal to its ``username``, so the only guard that works is "this is not the identity
  spelled again". A truthiness check passes for all 272 conversations of a real account and prints a
  group id where a name belongs.
* **the resolution** — which offered name wins, by field kind. Two contacts can share a nickname, a
  group's own id ends in ``@chatroom``, and a listing that offers nothing must leave a conversation
  *absent* rather than labelled with its talker.
* **the sidecar** — the one file this project writes that holds both names and talkers. It is written
  by ``sync_conversation_labels.py``, read by the web UI, and must never reach
  ``shard_manifest.json``, the browser, or the user's real ``~/.weflow-cli``.

Everything is synthetic and offline: no model, no network, no real chat, no real ``weflow-cli``. The
fake CLI answers ``contacts --json`` by reading the *scratch* config it was handed, so the tests also
prove the exporter's scratch-profile mechanism is the one being used — the same proof
``tests/test_account_tree.py`` makes for ``sessions``.

Scratch directories live under ``tests/_scratch_labels`` (``pytest``'s ``tmp_path`` cannot be created
in this environment) and are removed in the fixture teardown.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import exporter  # noqa: E402
import memory.labels as labels  # noqa: E402
import sync_conversation_labels as sync_cli  # noqa: E402
from memory import load_account_directory  # noqa: E402
from memory.conversations import (  # noqa: E402
    ConversationDescriptor,
    parse_session_listing,
)
from memory.labels import (  # noqa: E402
    CONVERSATION_LABEL_FILENAME,
    LABEL_SCHEMA,
    ContactRecord,
    ConversationLabel,
    NameOffer,
    parse_contact_listing,
    read_label_sidecar,
    resolve_conversation_labels,
    usable_name,
    write_label_sidecar,
)
from webapp import RecallState, create_app, read_conversation_labels  # noqa: E402

SCRATCH_ROOT = BASE_DIR / "tests" / "_scratch_labels"

#: Synthetic identity. The talker ids are strings, not wxids; the names are invented.
ALICE = "wxid_synthetic_alice"
BOB = "wxid_synthetic_bob"
CAROL = "wxid_synthetic_carol"
GROUP = "11112222@chatroom"
OTHER_GROUP = "33334444@chatroom"

NAME_ALICE = "小王"
NAME_BOB = "小李"
NAME_GROUP = "周末爬山群"


# --- scratch and fixtures ----------------------------------------------------------------------


@pytest.fixture()
def scratch() -> Path:
    """A clean workspace-local directory (``tmp_path`` is unusable here) removed afterwards."""
    directory = SCRATCH_ROOT / uuid4().hex[:8]
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)
        try:
            SCRATCH_ROOT.rmdir()
        except OSError:
            pass


def write_account_tree(
    root: Path,
    listings: dict[str, list[dict]],
    *,
    shards: tuple[str, ...] = ("MSG0",),
    manifest: dict | None = None,
) -> Path:
    """Write the export tree shape the importer reads: a listing and an export per shard."""
    root.mkdir(parents=True, exist_ok=True)
    for shard, entries in listings.items():
        shard_dir = root / shard
        shard_dir.mkdir(parents=True, exist_ok=True)
        (shard_dir / "sessions.json").write_text(
            json.dumps(entries, ensure_ascii=False), encoding="utf-8"
        )
        for entry in entries:
            talker = entry.get("username") or entry.get("conversation_id") or ""
            if talker:
                (shard_dir / f"{talker}_messages.json").write_text("[]", encoding="utf-8")
    payload = manifest if manifest is not None else {
        "schema": "privrecall-account-export/v1",
        "detected_shards": list(shards),
        "exported_shards": list(shards),
        "conversations_listed": sum(len(v) for v in listings.values()),
        "files_written": sum(len(v) for v in listings.values()),
        "total_messages": 0,
        "filtered_conversations": 0,
        "failures": [],
    }
    (root / "shard_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root


class FakeCli:
    """A stand-in for ``weflow-cli`` that proves which config it was handed.

    It reads the scratch profile's config file, so a run that did not go through
    ``exporter._prepare_scratch_profile`` shows up here as a missing file rather than as a silent
    difference in behaviour.
    """

    def __init__(self, contacts: list[dict] | None = None, *, ok: bool = True, error: str = ""):
        self.contacts = contacts if contacts is not None else []
        self.ok = ok
        self.error = error
        self.argv_seen: list[list[str]] = []
        self.scratch_alive_during_call: list[bool] = []
        self.config_seen: list[dict] = []

    def __call__(self, argv, env):
        self.argv_seen.append(list(argv))
        scratch = Path(env["USERPROFILE"])
        config_file = scratch / ".weflow-cli" / "config.json"
        self.scratch_alive_during_call.append(config_file.is_file())
        if config_file.is_file():
            self.config_seen.append(json.loads(config_file.read_text(encoding="utf-8")))
        if not self.ok:
            return subprocess.CompletedProcess(
                argv, 1, stdout=json.dumps({"success": False, "error": self.error}), stderr=""
            )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps({"success": True, "contacts": self.contacts}, ensure_ascii=False),
            stderr="",
        )


def real_config_stand_in(scratch: Path) -> Path:
    """A stand-in for the user's ``~/.weflow-cli/config.json`` — read, never written."""
    path = scratch / "real_config" / ".weflow-cli" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"dbPath3x": str(scratch / "MSG0.db"), "decryptKey3x": "ab" * 32}),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def scratch_root(scratch: Path) -> Path:
    root = scratch / "scratch"
    root.mkdir(parents=True, exist_ok=True)
    return root


# --- the rule ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["", "   ", "\n\t ", None, 0])
def test_usable_name_rejects_a_name_that_is_not_there(name) -> None:
    assert usable_name(name, ALICE) == ""


@pytest.mark.parametrize("name", [ALICE, f"  {ALICE}  ", GROUP, OTHER_GROUP])
def test_usable_name_rejects_the_identity_spelled_again(name: str) -> None:
    """The measured real-account shape: ``displayName`` equal to ``username`` is not a name."""
    conversation_id = GROUP if "@chatroom" in name else ALICE
    assert usable_name(name, conversation_id) == ""


def test_usable_name_rejects_the_group_suffix_even_for_another_conversation() -> None:
    """A chatroom id is an identifier whoever it names — the suffix gives it away."""
    assert usable_name(OTHER_GROUP, ALICE) == ""
    assert usable_name(f"群里的人 {GROUP}", ALICE) == ""


@pytest.mark.parametrize("name", [NAME_ALICE, f"  {NAME_ALICE}  ", "Group named by a person"])
def test_usable_name_returns_a_real_name(name: str) -> None:
    assert usable_name(name, ALICE)


def test_has_real_name_delegates_to_usable_name(monkeypatch) -> None:
    """One rule, one implementation: the property must call ``usable_name``, not copy it."""
    calls: list[tuple] = []

    def spy(name, conversation_id):
        calls.append((name, conversation_id))
        return "SENTINEL" if str(name).strip() else ""

    monkeypatch.setattr(labels, "usable_name", spy)
    descriptor = ConversationDescriptor(conversation_id=ALICE, display_name=NAME_ALICE)

    assert descriptor.has_real_name is True
    assert calls == [(NAME_ALICE, ALICE)], "has_real_name resolved a name without asking the rule"
    empty = ConversationDescriptor(conversation_id=ALICE, display_name="")
    assert empty.has_real_name is False


def test_the_identity_rule_is_not_written_twice() -> None:
    """Structure, not behaviour: a second copy of the rule is how the two would drift apart."""
    import inspect

    source = inspect.getsource(ConversationDescriptor.has_real_name.fget)
    assert "usable_name" in source
    assert "@chatroom" not in source, "the group-suffix rule was copied into the descriptor"

    assert "def usable_name(" in (BASE_DIR / "memory" / "labels.py").read_text(encoding="utf-8")
    for consumer in ("webapp/app.py", "sync_conversation_labels.py", "memory/account.py"):
        text = (BASE_DIR / consumer).read_text(encoding="utf-8")
        assert "@chatroom" not in text, f"{consumer} grew its own copy of the group-suffix rule"


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "",
        42,
        {},
        [],
        {"success": True},
        {"contacts": None},
        {"contacts": "not-a-list"},
        {"contacts": [None, 1, "x", [], {"no_talker": True}]},
        {"contacts": [{"username": 12345, "displayName": NAME_ALICE}]},
        {"contacts": [{"username": ALICE, "displayName": {"nested": NAME_ALICE}}]},
        {"contacts": [{"username": ALICE, "aliases": [None, 7, {"a": 1}]}]},
        {"data": {"items": []}},
    ],
)
def test_parse_contact_listing_never_raises_on_junk(payload) -> None:
    """A listing is decoration: no shape of it may stop the product from answering."""
    assert isinstance(parse_contact_listing(payload), tuple)


def test_parse_contact_listing_reads_the_measured_contacts_shape() -> None:
    payload = {
        "success": True,
        "contacts": [
            {"username": ALICE, "displayName": NAME_ALICE},
            {"username": BOB, "displayName": BOB},
        ],
    }
    records = parse_contact_listing(payload)
    assert [r.conversation_id for r in records] == [ALICE, BOB]
    assert records[0].offers == (NameOffer(kind="displayName", value=NAME_ALICE),)


@pytest.mark.parametrize("container", ["contacts", "sessions", "data", "items"])
def test_parse_contact_listing_accepts_every_container_and_a_bare_list(container: str) -> None:
    entry = {"username": ALICE, "remark": NAME_ALICE}
    assert parse_contact_listing({container: [entry]}) == parse_contact_listing([entry])


def test_parse_contact_listing_tags_each_offered_name_with_its_field_kind() -> None:
    record = parse_contact_listing(
        [
            {
                "username": ALICE,
                "remark": NAME_ALICE,
                "nickname": "昵称",
                "displayName": "显示名",
                "alias": "别名",
                "aliases": ["别名二"],
                "type": 0,
                "lastTimestamp": 1758000000,
                "unreadCount": 3,
            }
        ]
    )[0]
    assert sorted((offer.kind, offer.value) for offer in record.offers) == sorted(
        [
            ("remark", NAME_ALICE),
            ("nickname", "昵称"),
            ("displayName", "显示名"),
            ("other", "别名"),
            ("other", "别名二"),
        ]
    ), "a field that is not a name (type, timestamp) must not become an offer"


def test_parse_contact_listing_merges_two_records_for_one_talker() -> None:
    """The same talker listed by two shards stays one record, with both offers kept."""
    records = parse_contact_listing(
        [{"username": ALICE, "remark": NAME_ALICE}, {"username": ALICE, "nickname": "昵称"}]
    )
    assert len(records) == 1
    assert [offer.kind for offer in records[0].offers] == ["remark", "nickname"]


# --- resolution --------------------------------------------------------------------------------


def descriptor(talker: str, display_name: str = "") -> ConversationDescriptor:
    return parse_session_listing([{"username": talker, "displayName": display_name}])[0]


def test_a_remark_wins_over_a_nickname() -> None:
    resolved = resolve_conversation_labels(
        [descriptor(ALICE)],
        [ContactRecord(ALICE, (NameOffer("nickname", "昵称"), NameOffer("remark", NAME_ALICE)))],
    )
    assert resolved[ALICE].label == NAME_ALICE
    assert resolved[ALICE].source == "remark"


def test_a_nickname_is_used_when_there_is_no_remark() -> None:
    resolved = resolve_conversation_labels(
        [descriptor(ALICE)],
        [ContactRecord(ALICE, (NameOffer("nickname", "昵称"), NameOffer("displayName", "显示名")))],
    )
    assert resolved[ALICE].label == "昵称"


def test_a_display_name_equal_to_the_talker_is_not_a_name() -> None:
    """The real-account shape, end to end: 272 conversations of ``displayName == username``."""
    resolved = resolve_conversation_labels(
        [descriptor(ALICE, ALICE), descriptor(GROUP, GROUP)],
        parse_contact_listing(
            [{"username": ALICE, "displayName": ALICE}, {"username": GROUP, "displayName": GROUP}]
        ),
    )
    assert resolved == {}


def test_a_name_containing_the_group_suffix_is_not_a_name() -> None:
    resolved = resolve_conversation_labels(
        [descriptor(GROUP)],
        [ContactRecord(GROUP, (NameOffer("displayName", OTHER_GROUP),))],
    )
    assert resolved == {}


def test_a_conversation_with_no_real_name_is_absent_not_none_and_not_the_id() -> None:
    resolved = resolve_conversation_labels(
        [descriptor(ALICE, ALICE), descriptor(CAROL, NAME_ALICE)],
        [ContactRecord(ALICE, (NameOffer("nickname", ALICE),))],
    )
    assert ALICE not in resolved
    assert None not in resolved.values()
    assert resolved[CAROL].label == NAME_ALICE


def test_two_conversations_sharing_one_display_name_stay_two_entries() -> None:
    """A name is editable and not unique; the talker is the key."""
    resolved = resolve_conversation_labels(
        [descriptor(ALICE), descriptor(BOB)],
        [
            ContactRecord(ALICE, (NameOffer("nickname", "老王"),)),
            ContactRecord(BOB, (NameOffer("nickname", "老王"),)),
        ],
    )
    assert set(resolved) == {ALICE, BOB}
    assert resolved[ALICE].label == resolved[BOB].label == "老王"
    assert resolved[ALICE].conversation_id != resolved[BOB].conversation_id


def test_precedence_is_by_field_kind_not_by_the_order_of_the_fields() -> None:
    """The same entry, two key orders, one label — a dict's order is not a precedence."""
    first = parse_contact_listing([{"username": ALICE, "remark": NAME_ALICE, "nickname": "昵称"}])
    second = parse_contact_listing([{"nickname": "昵称", "remark": NAME_ALICE, "username": ALICE}])
    assert (
        resolve_conversation_labels([descriptor(ALICE)], first)[ALICE].label
        == resolve_conversation_labels([descriptor(ALICE)], second)[ALICE].label
        == NAME_ALICE
    )


def test_a_tie_inside_one_kind_resolves_the_same_way_whatever_the_listing_order() -> None:
    """Two listings, same names, opposite order: one label, or a card header would flip."""
    forwards = resolve_conversation_labels(
        [descriptor(ALICE)],
        [ContactRecord(ALICE, (NameOffer("nickname", "阿一"), NameOffer("nickname", "阿二")))],
    )
    backwards = resolve_conversation_labels(
        [descriptor(ALICE)],
        [ContactRecord(ALICE, (NameOffer("nickname", "阿二"), NameOffer("nickname", "阿一")))],
    )
    assert forwards[ALICE].label == backwards[ALICE].label


def test_a_group_is_named_by_its_group_name_or_not_at_all() -> None:
    """A group has no remark tier: anything else about a group that is a string is its id."""
    resolved = resolve_conversation_labels(
        [descriptor(GROUP), descriptor(OTHER_GROUP)],
        [
            ContactRecord(GROUP, (NameOffer("remark", "我的群备注"), NameOffer("nickname", "群昵称"))),
            ContactRecord(OTHER_GROUP, (NameOffer("displayName", NAME_GROUP),)),
        ],
    )
    assert GROUP not in resolved
    assert resolved[OTHER_GROUP].label == NAME_GROUP
    assert resolved[OTHER_GROUP].kind == "group"


def test_the_export_trees_own_listing_still_resolves_without_contact_records() -> None:
    """The fallback path: no sidecar, no contacts, only ``sessions.json`` to go on."""
    resolved = resolve_conversation_labels(
        [descriptor(ALICE, NAME_ALICE), descriptor(BOB, BOB), descriptor(GROUP, NAME_GROUP)], ()
    )
    assert {talker: label.label for talker, label in resolved.items()} == {
        ALICE: NAME_ALICE,
        GROUP: NAME_GROUP,
    }


# --- the sidecar -------------------------------------------------------------------------------


def label(talker: str, name: str, kind: str = "direct", source: str = "remark") -> ConversationLabel:
    return ConversationLabel(conversation_id=talker, label=name, kind=kind, source=source)


def test_the_sidecar_round_trips(scratch: Path) -> None:
    path = scratch / CONVERSATION_LABEL_FILENAME
    written = {ALICE: label(ALICE, NAME_ALICE), GROUP: label(GROUP, NAME_GROUP, "group")}

    write_label_sidecar(path, written)
    assert read_label_sidecar(path) == written

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema"] == LABEL_SCHEMA
    assert [entry["conversation_id"] for entry in document["labels"]] == sorted(written)


def test_writing_the_sidecar_is_idempotent(scratch: Path) -> None:
    path = scratch / CONVERSATION_LABEL_FILENAME
    written = {BOB: label(BOB, NAME_BOB), ALICE: label(ALICE, NAME_ALICE)}
    first = write_label_sidecar(path, written).read_bytes()
    second = write_label_sidecar(path, written).read_bytes()
    assert first == second, "a re-run rewrote the file differently"


@pytest.mark.parametrize(
    "content",
    [
        None,  # missing file
        "",
        "   ",
        "not json at all",
        "[]",
        '"a string"',
        "{}",
        json.dumps({"schema": "something-else/v1", "labels": []}),
        json.dumps({"schema": LABEL_SCHEMA}),
        json.dumps({"schema": LABEL_SCHEMA, "labels": {"not": "a list"}}),
        json.dumps({"schema": LABEL_SCHEMA, "labels": ["not-an-object", 3, None]}),
        json.dumps({"schema": LABEL_SCHEMA, "labels": [{"label": NAME_ALICE}]}),
        json.dumps({"schema": LABEL_SCHEMA, "labels": [{"conversation_id": ALICE}]}),
    ],
)
def test_a_missing_corrupt_or_wrong_schema_sidecar_reads_as_empty(scratch: Path, content) -> None:
    path = scratch / CONVERSATION_LABEL_FILENAME
    if content is not None:
        path.write_text(content, encoding="utf-8")
    assert read_label_sidecar(path) == {}
    assert read_label_sidecar(scratch / "nowhere" / "nothing.json") == {}
    assert read_label_sidecar(scratch) == {}, "a directory in the sidecar's place is not a sidecar"


def test_a_hand_edited_sidecar_cannot_smuggle_an_id_in_as_a_label(scratch: Path) -> None:
    """Defence in depth: the rule is applied on the way out as well as on the way in."""
    path = scratch / CONVERSATION_LABEL_FILENAME
    path.write_text(
        json.dumps(
            {
                "schema": LABEL_SCHEMA,
                "labels": [
                    {"conversation_id": ALICE, "label": ALICE, "kind": "direct", "source": "x"},
                    {"conversation_id": GROUP, "label": OTHER_GROUP, "kind": "group"},
                    {"conversation_id": BOB, "label": NAME_BOB, "kind": "nonsense"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    read = read_label_sidecar(path)
    assert set(read) == {BOB}, "a talker id was read back as a name"
    assert read[BOB].kind == "direct", "an unknown kind falls back to the talker's own type"


def test_the_writer_drops_a_label_that_is_not_a_name(scratch: Path) -> None:
    path = scratch / CONVERSATION_LABEL_FILENAME
    write_label_sidecar(path, {ALICE: label(ALICE, ALICE), BOB: label(BOB, NAME_BOB)})
    assert set(read_label_sidecar(path)) == {BOB}


# --- the sync CLI ------------------------------------------------------------------------------


def contacts_payload() -> list[dict]:
    """The measured shape: every entry carries a display name, and half of them are the entry's id."""
    return [
        {"username": ALICE, "displayName": NAME_ALICE},
        {"username": BOB, "displayName": BOB},
        {"username": GROUP, "displayName": NAME_GROUP},
        {"username": CAROL, "displayName": CAROL},
    ]


def test_sync_writes_the_sidecar_and_reports_coverage(scratch: Path, scratch_root: Path) -> None:
    account = write_account_tree(
        scratch / "account",
        {
            "MSG0": [
                {"username": ALICE, "displayName": ALICE},
                {"username": BOB, "displayName": BOB},
                {"username": GROUP, "displayName": GROUP},
                {"username": CAROL, "displayName": CAROL},
            ]
        },
    )
    fake = FakeCli(contacts_payload())
    original = exporter.real_config_path
    exporter.real_config_path = lambda env=None: real_config_stand_in(scratch)  # type: ignore[assignment]
    try:
        report = sync_cli.sync_labels(account, runner=fake, scratch_root=scratch_root)
    finally:
        exporter.real_config_path = original  # type: ignore[assignment]

    assert report.error == ""
    assert report.written is True
    assert (report.conversations, report.direct, report.group) == (4, 3, 1)
    assert (report.resolved, report.resolved_direct, report.resolved_group) == (2, 1, 1)
    assert report.unresolved == 2

    sidecar = read_label_sidecar(account / CONVERSATION_LABEL_FILENAME)
    # BOB and CAROL are offered their own ids back, which is the real installation's shape: two of
    # the four conversations have a name, the other two have nothing at all and are simply absent.
    assert {talker: entry.label for talker, entry in sidecar.items()} == {
        ALICE: NAME_ALICE,
        GROUP: NAME_GROUP,
    }
    assert sidecar[ALICE].source == "displayName"
    assert sidecar[GROUP].kind == "group", "the group's own name is a group label, not a direct one"

    rendered = "\n".join(report.lines())
    assert "direct       : 1 resolved / 3" in rendered
    assert "group        : 1 resolved / 1" in rendered


def test_sync_asks_contacts_json_through_the_scratch_profile(
    scratch: Path, scratch_root: Path
) -> None:
    account = write_account_tree(
        scratch / "account", {"MSG0": [{"username": ALICE, "displayName": ALICE}]}
    )
    fake = FakeCli(contacts_payload())
    original = exporter.real_config_path
    real_config = real_config_stand_in(scratch)
    exporter.real_config_path = lambda env=None: real_config  # type: ignore[assignment]
    try:
        sync_cli.sync_labels(account, runner=fake, scratch_root=scratch_root)
    finally:
        exporter.real_config_path = original  # type: ignore[assignment]

    assert fake.argv_seen == [["contacts", "--json"]]
    assert all(fake.scratch_alive_during_call), "the copied config must exist while the CLI runs"
    assert all(config.get("dbPath3x") for config in fake.config_seen)
    assert list(scratch_root.iterdir()) == [], "a scratch profile was left on disk"


def test_sync_leaves_the_real_config_untouched(scratch: Path, scratch_root: Path) -> None:
    """The user's own ``~/.weflow-cli`` is read once and never written."""
    account = write_account_tree(
        scratch / "account", {"MSG0": [{"username": ALICE, "displayName": ALICE}]}
    )
    real_config = real_config_stand_in(scratch)
    before = hashlib.sha256(real_config.read_bytes()).hexdigest()

    original = exporter.real_config_path
    exporter.real_config_path = lambda env=None: real_config  # type: ignore[assignment]
    try:
        sync_cli.sync_labels(
            account, runner=FakeCli(contacts_payload()), scratch_root=scratch_root
        )
    finally:
        exporter.real_config_path = original  # type: ignore[assignment]

    assert hashlib.sha256(real_config.read_bytes()).hexdigest() == before


def test_sync_is_re_runnable_and_idempotent(scratch: Path, scratch_root: Path) -> None:
    account = write_account_tree(
        scratch / "account",
        {"MSG0": [{"username": ALICE, "displayName": ALICE}, {"username": BOB, "displayName": BOB}]},
    )
    original = exporter.real_config_path
    exporter.real_config_path = lambda env=None: real_config_stand_in(scratch)  # type: ignore[assignment]
    try:
        first = sync_cli.sync_labels(
            account, runner=FakeCli(contacts_payload()), scratch_root=scratch_root
        )
        bytes_after_first = (account / CONVERSATION_LABEL_FILENAME).read_bytes()
        second = sync_cli.sync_labels(
            account, runner=FakeCli(contacts_payload()), scratch_root=scratch_root
        )
    finally:
        exporter.real_config_path = original  # type: ignore[assignment]

    assert first.as_dict() == second.as_dict()
    assert (account / CONVERSATION_LABEL_FILENAME).read_bytes() == bytes_after_first


def test_sync_prints_coverage_and_never_a_name(
    scratch: Path, scratch_root: Path, monkeypatch, capsys
) -> None:
    account = write_account_tree(
        scratch / "account",
        {
            "MSG0": [
                {"username": ALICE, "displayName": ALICE},
                {"username": BOB, "displayName": BOB},
                {"username": GROUP, "displayName": GROUP},
            ]
        },
    )
    monkeypatch.setattr(
        sync_cli,
        "list_contacts",
        lambda **_kwargs: parse_contact_listing(contacts_payload()),
    )
    assert sync_cli.main(["--account", str(account)]) == 0

    printed = capsys.readouterr()
    output = printed.out + printed.err
    for secret in (ALICE, BOB, CAROL, GROUP, NAME_ALICE, NAME_BOB, NAME_GROUP, "chatroom", "wxid"):
        assert secret not in output, f"the coverage report printed {secret!r}"
    assert "labels resolved: 2 of 3" in output
    assert "direct       : 1 resolved / 2" in output
    assert "group        : 1 resolved / 1" in output


def test_sync_writes_nothing_when_the_exporter_fails(
    scratch: Path, scratch_root: Path, capsys
) -> None:
    """A failed run must not replace a good sidecar with an empty one."""
    account = write_account_tree(
        scratch / "account", {"MSG0": [{"username": ALICE, "displayName": ALICE}]}
    )
    write_label_sidecar(account / CONVERSATION_LABEL_FILENAME, {ALICE: label(ALICE, NAME_ALICE)})
    before = (account / CONVERSATION_LABEL_FILENAME).read_bytes()

    key = "cd" * 32
    original = exporter.real_config_path
    exporter.real_config_path = lambda env=None: real_config_stand_in(scratch)  # type: ignore[assignment]
    try:
        report = sync_cli.sync_labels(
            account,
            runner=FakeCli(ok=False, error=f"decryptKey3x={key}"),
            scratch_root=scratch_root,
        )
    finally:
        exporter.real_config_path = original  # type: ignore[assignment]

    assert report.error and report.written is False
    assert key not in report.error and "[REDACTED]" in report.error
    assert (account / CONVERSATION_LABEL_FILENAME).read_bytes() == before


def test_sync_refuses_an_account_with_no_conversations(scratch: Path, scratch_root: Path) -> None:
    empty = scratch / "not_an_account"
    empty.mkdir()
    report = sync_cli.sync_labels(empty, runner=FakeCli(), scratch_root=scratch_root)
    assert report.error and "no conversations found" in report.error
    assert report.written is False
    assert not (empty / CONVERSATION_LABEL_FILENAME).exists()


def test_the_sidecar_never_reaches_the_shard_manifest(scratch: Path, scratch_root: Path) -> None:
    """The manifest is the file most likely to be copied around; it stays counts-only."""
    account = write_account_tree(
        scratch / "account",
        {"MSG0": [{"username": ALICE, "displayName": ALICE}, {"username": BOB, "displayName": BOB}]},
    )
    original = exporter.real_config_path
    exporter.real_config_path = lambda env=None: real_config_stand_in(scratch)  # type: ignore[assignment]
    try:
        sync_cli.sync_labels(account, runner=FakeCli(contacts_payload()), scratch_root=scratch_root)
    finally:
        exporter.real_config_path = original  # type: ignore[assignment]

    blob = (account / "shard_manifest.json").read_text(encoding="utf-8")
    for secret in (ALICE, BOB, CAROL, NAME_ALICE, NAME_BOB, "label"):
        assert secret not in blob, f"the manifest learned {secret!r}"
    manifest = json.loads(blob)
    assert manifest["schema"] == "privrecall-account-export/v1"
    assert set(manifest) == {
        "schema",
        "detected_shards",
        "exported_shards",
        "conversations_listed",
        "files_written",
        "total_messages",
        "filtered_conversations",
        "failures",
    }, "the manifest grew a field; counts only, and nothing that names a conversation"

    # And the sidecar must not confuse the tree reader: the account still reads as it did.
    layout = load_account_directory(account)
    assert [d.conversation_id for d in layout.descriptors] == [ALICE, BOB]
    assert layout.shards_detected == ("MSG0",)


# --- the web UI --------------------------------------------------------------------------------


def test_read_conversation_labels_prefers_the_sidecar_over_the_listing(scratch: Path) -> None:
    """The listing says nothing on a real account; the sidecar is where the name actually is."""
    account = write_account_tree(
        scratch / "account",
        {"MSG0": [{"username": ALICE, "displayName": ALICE}, {"username": GROUP, "displayName": GROUP}]},
    )
    assert read_conversation_labels(account) == {}, "the listing alone has no name to give"

    write_label_sidecar(account / CONVERSATION_LABEL_FILENAME, {ALICE: label(ALICE, NAME_ALICE)})
    assert read_conversation_labels(account) == {ALICE: NAME_ALICE}


def test_an_empty_sidecar_falls_back_to_the_listing(scratch: Path) -> None:
    """A sync that resolved nothing must not hide a name the tree itself does have."""
    account = write_account_tree(
        scratch / "account", {"MSG0": [{"username": ALICE, "displayName": NAME_ALICE}]}
    )
    write_label_sidecar(account / CONVERSATION_LABEL_FILENAME, {})
    assert read_conversation_labels(account) == {ALICE: NAME_ALICE}


def test_an_account_export_with_no_sidecar_behaves_exactly_as_before(scratch: Path) -> None:
    """Phase 20.5's behaviour, unchanged: the listing, filtered by the usable-name rule."""
    account = write_account_tree(
        scratch / "account",
        {
            "MSG0": [
                {"username": ALICE, "displayName": NAME_ALICE},
                {"username": BOB, "displayName": BOB},
                {"username": GROUP, "displayName": OTHER_GROUP},
                {"username": CAROL},
            ]
        },
    )
    assert not (account / CONVERSATION_LABEL_FILENAME).exists()
    assert read_conversation_labels(account) == {ALICE: NAME_ALICE}
    assert read_conversation_labels(scratch / "nowhere") == {}


def evidence_payload(conversation_id: str) -> dict:
    """The shape ``recall.answer_question`` returns, for one cited source."""
    return {
        "answer": "你最后换成了 Supabase。[来源 0]",
        "evidence": [
            {
                "citation_index": 0,
                "label": "[我, 对方 · 2025-05-12 17:43]",
                "lines": [
                    {"timestamp": "2025-05-12 17:43", "speaker": "我", "text": "我最后换成 Supabase 了"}
                ],
            }
        ],
        "groundedness": {
            "warnings": [],
            "citation_mismatches": [],
            "attribution_flags": [],
            "absence_claims": [],
            "invalid_citations": 0,
            "uncited": False,
        },
        "latency_ms": 1234,
        "sources": [{"conversation_id": conversation_id}],
    }


def app_client(monkeypatch, labels_by_talker: dict[str, str], conversation_id: str) -> TestClient:
    import recall

    payload = evidence_payload(conversation_id)
    monkeypatch.setattr(recall, "answer_question", lambda *args, **kwargs: payload)
    state = RecallState(account_dir=SCRATCH_ROOT / "unused")
    state.brain = object()
    state.retrieval_config = object()
    state.report = SimpleNamespace(
        conversations_imported=2,
        messages_kept=4,
        first_timestamp=None,
        last_timestamp=None,
        partial=False,
    )
    state.conversation_labels = dict(labels_by_talker)
    return TestClient(create_app(state))


def all_values(payload) -> list:
    """Every value in a JSON structure, however deeply nested — the check that found the 20.5 leak."""
    values = []
    stack = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
        else:
            values.append(current)
    return values


def test_the_evidence_card_shows_the_human_label(scratch: Path, monkeypatch) -> None:
    account = write_account_tree(
        scratch / "account",
        {"MSG0": [{"username": ALICE, "displayName": ALICE}, {"username": GROUP, "displayName": GROUP}]},
    )
    write_label_sidecar(
        account / CONVERSATION_LABEL_FILENAME,
        {ALICE: label(ALICE, NAME_ALICE), GROUP: label(GROUP, NAME_GROUP, "group")},
    )
    resolved = read_conversation_labels(account)

    direct = app_client(monkeypatch, resolved, ALICE).post(
        "/api/recall", json={"question": "我最后用了哪个数据库？"}
    ).json()
    assert direct["evidence"][0]["conversation"] == NAME_ALICE

    group = app_client(monkeypatch, resolved, GROUP).post(
        "/api/recall", json={"question": "我最后用了哪个数据库？"}
    ).json()
    assert group["evidence"][0]["conversation"] == NAME_GROUP


def test_the_evidence_card_still_renders_when_the_conversation_has_no_name(scratch: Path, monkeypatch
) -> None:
    """No name on record means no header — and the card keeps everything it is for."""
    account = write_account_tree(
        scratch / "account", {"MSG0": [{"username": GROUP, "displayName": GROUP}]}
    )
    resolved = read_conversation_labels(account)
    assert resolved == {}

    card = app_client(monkeypatch, resolved, GROUP).post(
        "/api/recall", json={"question": "我最后用了哪个数据库？"}
    ).json()["evidence"][0]

    assert "conversation" not in card
    assert card["citation_index"] == 0
    assert [line["text"] for line in card["lines"]] == ["我最后换成 Supabase 了"]


@pytest.mark.parametrize("talker", [ALICE, GROUP])
def test_no_talker_wxid_or_chatroom_token_reaches_the_browser(
    scratch: Path, monkeypatch, talker: str
) -> None:
    """Asserted on the response's *values*: a key-name check is how the 20.5 leak was missed."""
    account = write_account_tree(
        scratch / "account",
        {"MSG0": [{"username": ALICE, "displayName": ALICE}, {"username": GROUP, "displayName": GROUP}]},
    )
    write_label_sidecar(
        account / CONVERSATION_LABEL_FILENAME,
        {talker: label(talker, NAME_ALICE if talker == ALICE else NAME_GROUP, "direct")},
    )
    client = app_client(monkeypatch, read_conversation_labels(account), talker)

    response = client.post("/api/recall", json={"question": "我最后用了哪个数据库？"}).json()
    values = [str(value) for value in all_values(response)]
    for value in values:
        assert ALICE not in value and BOB not in value and CAROL not in value
        assert "chatroom" not in value
        assert "wxid" not in value
        assert "session-" not in value, "a chunk id carries the talker"

    blob = json.dumps(response, ensure_ascii=False)
    assert GROUP not in blob and "chatroom" not in blob and "wxid" not in blob

    status_blob = json.dumps(client.get("/api/status").json(), ensure_ascii=False)
    assert GROUP not in status_blob and "wxid" not in status_blob
