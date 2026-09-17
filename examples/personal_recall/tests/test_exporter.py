"""Offline tests for the account exporter boundary (no exporter run, no real config, no real data).

What these tests are actually protecting, in order of how bad the failure would be:

* the user's real ``~/.weflow-cli`` is never written to, and the scratch profile is **always** gone
  afterwards — including when the runner raises, because that scratch holds a copy of the decrypt key;
* no secret reaches an error string (``redact`` is checked directly against a 64-hex key);
* a missing config fails loudly instead of silently running against whatever database happens to be
  configured — that failure mode is "No Data Loaded ≠ No Memory Exists" at the account level;
* one failing conversation does not abort the rest of the shard.

Everything is synthetic. The runner is a fake that records its invocations, so no test here starts
``weflow-cli`` or reads a single real message. File-based tests use a workspace-local scratch dir
because ``pytest``'s ``tmp_path`` cannot be created in this environment (PermissionError).
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from exporter import (  # noqa: E402
    ExportResult,
    ExporterError,
    export_account,
    export_conversation,
    export_shard,
    list_conversations,
    redact,
)
from memory.conversations import ConversationDescriptor, parse_session_listing  # noqa: E402

SCRATCH_ROOT = Path(__file__).resolve().parent / "_scratch_exporter"

#: A synthetic "real" config. It is never the user's: it carries a fake wrapped key so a leak would be
#: visible, and a dbPath3x pointing at a file that does not exist.
FAKE_DECRYPT_KEY = "a" * 64
FAKE_WRAPPED_KEY = "lock:" + "QWxhZGRpbjpvcGVuIHNlc2FtZQ" * 2
SYNTHETIC_CONFIG = {
    "dbPath": r"C:\synthetic\WeChat Files\wxid_synthetic",
    "wxid": "wxid_synthetic",
    "decryptKey": "",
    "decryptKey3x": FAKE_WRAPPED_KEY,
    "dataVersion": "3.x",
    "dbPath3x": r"C:\synthetic\WeChat Files\wxid_synthetic\Msg\Multi\MSG2.db",
    "ntKey": "",
    "whitelist": [],
}


# --- fixtures and fake runner -----------------------------------------------------------------


def _scratch(name: str) -> Path:
    """A clean workspace-local directory (``tmp_path`` is unavailable in this environment)."""
    path = SCRATCH_ROOT / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_synthetic_config(home: Path) -> Path:
    """Create a fake ``homedir()/.weflow-cli/config.json`` and return its path."""
    profile = home / ".weflow-cli"
    profile.mkdir(parents=True, exist_ok=True)
    config = profile / "config.json"
    config.write_text(json.dumps(SYNTHETIC_CONFIG, indent=2), encoding="utf-8")
    return config


@pytest.fixture()
def fake_home():
    """An isolated HOME/USERPROFILE holding a synthetic config. Never ``~``."""
    home = _scratch("home")
    _write_synthetic_config(home)
    return home


def _leftover_configs() -> list[str]:
    """Config files anywhere under the scratch root.

    The fixture's synthetic home lives there too, so it is excluded: what these tests care about is
    whether an *export* left a config profile behind.
    """
    return [
        str(path)
        for path in SCRATCH_ROOT.rglob("config.json")
        if "home" not in path.relative_to(SCRATCH_ROOT).parts
    ]


def _completed(payload=None, *, stdout=None, stderr="", returncode=0):
    text = stdout if stdout is not None else ("" if payload is None else json.dumps(payload))
    return subprocess.CompletedProcess(args=["weflow-cli"], returncode=returncode, stdout=text, stderr=stderr)


def session_entry(username: str, display: str = "", **extra):
    entry = {"username": username, "displayName": display, "type": 1, "lastTimestamp": 1700000000}
    entry.update(extra)
    return entry


SESSIONS = [
    session_entry("wxid_alice", "Alice 备注"),
    session_entry("wxid_bob", "Bob"),
    session_entry("12345678@chatroom", "Synthetic Group", type=2),
]

TALKERS = ("12345678@chatroom", "wxid_alice", "wxid_bob")


class FakeRunner:
    """Records every invocation and answers with what ``weflow-cli --json`` would print.

    It also probes the scratch profile *while the call is in flight*, which is the only moment the
    scratch directory is supposed to exist.
    """

    def __init__(self, *, sessions=SESSIONS, fail=(), fail_sessions=False, raise_for=(), raw=None):
        self.sessions = sessions
        self.fail = set(fail)
        self.fail_sessions = fail_sessions
        self.raise_for = set(raise_for)
        self.raw = raw
        self.calls: list[dict] = []
        #: ``(env, config_exists, config_payload)`` captured during the call.
        self.seen: list[tuple[dict, bool, dict | None]] = []

    def __call__(self, argv, env):
        self.calls.append({"argv": list(argv), "env": dict(env)})
        home = Path(env["HOME"])
        config = home / ".weflow-cli" / "config.json"
        payload = None
        if config.is_file():
            payload = json.loads(config.read_text(encoding="utf-8"))
        self.seen.append((dict(env), config.is_file(), payload))

        if self.raw is not None:
            return _completed(stdout=self.raw, returncode=1)

        if argv[0] == "sessions":
            if self.fail_sessions:
                return _completed({"success": False, "error": "未完成初始化"}, returncode=1)
            return _completed({"success": True, "sessions": self.sessions})

        talker = argv[1]
        if talker in self.raise_for:
            raise FileNotFoundError(f"synthetic runner failure for {talker}")
        if talker in self.fail:
            return _completed(
                {"success": False, "code": "EXPORT_FAILED", "error": "synthetic export failure"},
                returncode=1,
            )
        return _completed(
            {
                "success": True,
                "format": "json",
                "contract": "raw",
                "path": str(Path(argv[argv.index("--output") + 1]) / f"{talker}_messages.json"),
                "count": 7,
            }
        )


@pytest.fixture(autouse=True)
def _cleanup_scratch():
    yield
    shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


# --- 1. listing -------------------------------------------------------------------------------


def test_list_conversations_parses_sessions_payload(monkeypatch, fake_home) -> None:
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    runner = FakeRunner()

    descriptors = list_conversations("MSG0", runner=runner, scratch_root=_scratch("listing"))

    assert [d.conversation_id for d in descriptors] == list(TALKERS)
    assert descriptors == parse_session_listing({"success": True, "sessions": SESSIONS})
    assert runner.calls[0]["argv"][:2] == ["sessions", "--json"]


def test_display_name_is_never_the_id(monkeypatch, fake_home) -> None:
    """The whole identity rule: ``displayName`` is an editable alias, ``username`` is the key."""
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))

    descriptors = list_conversations("MSG0", runner=FakeRunner(), scratch_root=_scratch("listing"))
    by_id = {d.conversation_id: d for d in descriptors}

    assert "wxid_alice" in by_id
    assert "Alice 备注" not in by_id
    alice = by_id["wxid_alice"]
    assert alice.display_name == "Alice 备注"
    assert alice.aliases == ("Alice 备注",)
    assert alice.label == "Alice 备注"
    assert by_id["12345678@chatroom"].conversation_type == "group"
    assert by_id["wxid_alice"].conversation_type == "direct"


def test_listing_failure_is_loud(monkeypatch, fake_home) -> None:
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    with pytest.raises(ExporterError):
        list_conversations(
            "MSG0", runner=FakeRunner(fail_sessions=True), scratch_root=_scratch("listing")
        )


def test_listing_with_unparseable_output_is_loud(monkeypatch, fake_home) -> None:
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    with pytest.raises(ExporterError) as excinfo:
        list_conversations(
            "MSG0", runner=FakeRunner(raw="not json at all"), scratch_root=_scratch("listing")
        )
    assert "no JSON" in str(excinfo.value)


# --- 2. argv and path -------------------------------------------------------------------------


def test_export_conversation_builds_expected_argv(monkeypatch, fake_home) -> None:
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    out_dir = _scratch("out") / "MSG0"
    runner = FakeRunner()

    result = export_conversation(
        "wxid_alice", out_dir, shard="MSG0", runner=runner, scratch_root=_scratch("argv")
    )

    argv = runner.calls[0]["argv"]
    assert argv == [
        "export",
        "wxid_alice",
        "json",
        "--output",
        str(out_dir),
        "--non-interactive",
        "--limit",
        "0",
        "--json",
    ]
    assert result == ExportResult(
        shard="MSG0",
        conversation_id="wxid_alice",
        path=str(out_dir / "wxid_alice_messages.json"),
        messages=7,
        ok=True,
        error="",
    )
    assert result.path.endswith("wxid_alice_messages.json")


def test_export_conversation_falls_back_to_convention_path(monkeypatch, fake_home) -> None:
    """When the CLI omits ``path``, identity still comes from ``{talker}_messages.json``."""
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    out_dir = _scratch("out") / "MSG1"

    class NoPathRunner(FakeRunner):
        def __call__(self, argv, env):
            completed = super().__call__(argv, env)
            if argv[0] == "export":
                completed.stdout = json.dumps({"success": True, "count": None})
            return completed

    result = export_conversation(
        "wxid_bob", out_dir, shard="MSG1", runner=NoPathRunner(), scratch_root=_scratch("nopath")
    )
    assert result.path == str(out_dir / "wxid_bob_messages.json")
    assert result.messages == 0
    assert result.ok is True


# --- 3. scratch profile during and after the call ---------------------------------------------


def test_scratch_profile_exists_during_and_is_gone_after(monkeypatch, fake_home) -> None:
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    scratch_root = _scratch("lifecycle")
    out_dir = _scratch("out") / "MSG0"
    runner = FakeRunner()

    export_conversation(
        "wxid_alice", out_dir, shard="MSG0", runner=runner, scratch_root=scratch_root
    )

    env, config_exists, payload = runner.seen[0]
    scratch = Path(env["HOME"])
    assert env["USERPROFILE"] == str(scratch)
    assert env["HOME"] == str(scratch)
    assert scratch.parent == scratch_root
    assert scratch.name.startswith("privrecall_export_")
    assert config_exists is True, "the exporter must see a config while it runs"
    assert payload is not None
    assert payload["dbPath3x"] == SYNTHETIC_CONFIG["dbPath3x"]

    assert not scratch.exists(), "the scratch profile must be removed after the call"
    assert list(scratch_root.iterdir()) == []
    assert out_dir.is_dir(), "the export directory is the caller's, and stays"


def test_scratch_profile_is_removed_when_the_runner_raises(monkeypatch, fake_home) -> None:
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    scratch_root = _scratch("raises")
    runner = FakeRunner(raise_for={"wxid_alice"})

    result = export_conversation(
        "wxid_alice", _scratch("out2"), shard="MSG0", runner=runner, scratch_root=scratch_root
    )

    assert result.ok is False
    assert "FileNotFoundError" in result.error
    seen_scratch = Path(runner.seen[0][0]["HOME"])
    assert runner.seen[0][1] is True
    assert not seen_scratch.exists()
    assert list(scratch_root.iterdir()) == []


# --- 4. the real config is never written ------------------------------------------------------


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_real_config_is_never_written(monkeypatch, fake_home) -> None:
    """Point the "real" config at a temp file and prove nothing under it changes."""
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    before = _hash_tree(fake_home)

    export_conversation(
        "wxid_alice",
        _scratch("out3"),
        shard="MSG0",
        runner=FakeRunner(),
        scratch_root=_scratch("untouched"),
    )
    list_conversations("MSG0", runner=FakeRunner(), scratch_root=_scratch("untouched2"))

    assert _hash_tree(fake_home) == before
    assert set(before) == {str(Path(".weflow-cli") / "config.json")}
    # The only config written anywhere is the one inside the scratch profile, and it is gone.
    assert _leftover_configs() == []


def test_no_config_or_key_is_written_outside_the_scratch(monkeypatch, fake_home) -> None:
    """Nothing resembling a config appears anywhere except the scratch profile, mid-call."""
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    scratch_root = _scratch("isolated")
    runner = FakeRunner()

    export_conversation(
        "wxid_alice", _scratch("out4"), shard="MSG0", runner=runner, scratch_root=scratch_root
    )

    # Exactly one config file existed, it was inside the scratch root, and it is gone now.
    assert runner.seen[0][1] is True
    assert Path(runner.seen[0][0]["HOME"]).parent == scratch_root
    assert _leftover_configs() == []
    # Nothing under the scratch root carries real key material: the fixture's synthetic home is the
    # only file left, and it holds a fake key, never the real one.
    for path in [p for p in SCRATCH_ROOT.rglob("*") if p.is_file()]:
        assert path == fake_home / ".weflow-cli" / "config.json"


# --- 5. missing config ------------------------------------------------------------------------


def test_missing_real_config_is_a_clear_error(monkeypatch) -> None:
    empty_home = _scratch("empty_home")
    monkeypatch.setenv("USERPROFILE", str(empty_home))
    monkeypatch.setenv("HOME", str(empty_home))
    scratch_root = _scratch("missing_config")
    runner = FakeRunner()

    with pytest.raises(ExporterError) as excinfo:
        export_conversation(
            "wxid_alice", _scratch("out5"), shard="MSG0", runner=runner, scratch_root=scratch_root
        )

    message = str(excinfo.value)
    assert "config not found" in message
    assert "weflow-cli init" in message
    assert str(empty_home / ".weflow-cli" / "config.json") in message
    assert runner.calls == [], "the exporter must not be invoked without a config"
    assert list(scratch_root.iterdir()) == [], "no scratch profile is left behind"


def test_config_without_a_database_key_is_rejected(monkeypatch) -> None:
    """A config the module cannot understand fails loudly rather than inventing a db path."""
    home = _scratch("half_home")
    (home / ".weflow-cli").mkdir(parents=True, exist_ok=True)
    (home / ".weflow-cli" / "config.json").write_text(
        json.dumps({"wxid": "wxid_synthetic", "decryptKey3x": FAKE_WRAPPED_KEY}), encoding="utf-8"
    )
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    multi_dir = _scratch("multi_nodb")
    (multi_dir / "MSG0.db").write_bytes(b"synthetic")
    scratch_root = _scratch("nodbp")
    runner = FakeRunner()

    with pytest.raises(ExporterError) as excinfo:
        export_conversation(
            "wxid_alice",
            _scratch("out6"),
            shard="MSG0",
            multi_dir=multi_dir,
            runner=runner,
            scratch_root=scratch_root,
        )
    assert "dbPath" in str(excinfo.value)
    assert FAKE_WRAPPED_KEY not in str(excinfo.value)
    assert runner.calls == []
    assert list(scratch_root.iterdir()) == []


def test_listing_without_a_multi_dir_needs_a_config_that_selects_a_database(monkeypatch) -> None:
    """Preferring the configured shard is fine; running with *no* shard at all is not."""
    home = _scratch("nodb_list_home")
    (home / ".weflow-cli").mkdir(parents=True, exist_ok=True)
    (home / ".weflow-cli" / "config.json").write_text(
        json.dumps({"wxid": "wxid_synthetic"}), encoding="utf-8"
    )
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    runner = FakeRunner()

    with pytest.raises(ExporterError) as excinfo:
        list_conversations("MSG0", runner=runner, scratch_root=_scratch("nodb_list"))

    assert "selects no database" in str(excinfo.value)
    assert runner.calls == []


def test_every_db_path_key_in_the_real_config_is_repointed(monkeypatch) -> None:
    """Both ``dbPath`` and ``dbPath3x`` are rewritten, and nothing else in the file is touched."""
    home = _scratch("both_home")
    (home / ".weflow-cli").mkdir(parents=True, exist_ok=True)
    (home / ".weflow-cli" / "config.json").write_text(
        json.dumps({**SYNTHETIC_CONFIG, "dbPath": r"C:\synthetic\account"}), encoding="utf-8"
    )
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    multi_dir = _scratch("multi")
    (multi_dir / "MSG0.db").write_bytes(b"synthetic")
    runner = FakeRunner()

    export_conversation(
        "wxid_alice",
        _scratch("out13"),
        shard="MSG0",
        multi_dir=multi_dir,
        runner=runner,
        scratch_root=_scratch("both"),
    )

    payload = runner.seen[0][2]
    expected = str(multi_dir / "MSG0.db")
    assert payload["dbPath"] == expected
    assert payload["dbPath3x"] == expected
    # Every other key is carried over verbatim, including the ones this module never looks at.
    for key, value in SYNTHETIC_CONFIG.items():
        if key not in {"dbPath", "dbPath3x"}:
            assert payload[key] == value
    assert payload["decryptKey3x"] == FAKE_WRAPPED_KEY


def test_unknown_shard_is_a_clear_error(monkeypatch, fake_home) -> None:
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    multi_dir = _scratch("multi_unknown")
    (multi_dir / "MSG0.db").write_bytes(b"synthetic")
    runner = FakeRunner()

    with pytest.raises(ExporterError) as excinfo:
        export_conversation(
            "wxid_alice",
            _scratch("out14"),
            shard="MSG9",
            multi_dir=multi_dir,
            runner=runner,
            scratch_root=_scratch("unknown"),
        )
    assert "MSG9" in str(excinfo.value)
    assert "MSG0" in str(excinfo.value)
    assert runner.calls == []


# --- 6. redaction -----------------------------------------------------------------------------


def test_redact_removes_hex_and_base64_runs() -> None:
    # The path is synthetic like everything else here: a test fixture is a tracked file, and a real
    # account path would put a real wxid in the repository the moment this is committed.
    message = (
        f"decryptKey3x={FAKE_WRAPPED_KEY} key={FAKE_DECRYPT_KEY} "
        "path=C:\\Users\\me\\WeChat Files\\wxid_synthetic\\Msg\\Multi\\MSG2.db"
    )

    cleaned = redact(message)

    assert FAKE_DECRYPT_KEY not in cleaned
    assert FAKE_WRAPPED_KEY not in cleaned
    assert "[REDACTED]" in cleaned
    assert "decryptKey3x=" in cleaned
    assert "MSG2.db" in cleaned


def test_redact_leaves_ordinary_text_alone() -> None:
    assert redact("") == ""
    assert redact("export failed: talker not found") == "export failed: talker not found"
    # A short display fragment is not a key and must survive error reporting.
    assert redact("Alice 备注") == "Alice 备注"


def test_failed_export_error_never_contains_the_key(monkeypatch, fake_home) -> None:
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))

    class LeakyRunner(FakeRunner):
        def __call__(self, argv, env):
            if argv[0] == "export":
                self.calls.append({"argv": list(argv), "env": dict(env)})
                self.seen.append((dict(env), True, None))
                return _completed(
                    stdout="",
                    stderr=f"sqlcipher open failed with key {FAKE_DECRYPT_KEY} for {FAKE_WRAPPED_KEY}",
                    returncode=1,
                )
            return super().__call__(argv, env)

    result = export_conversation(
        "wxid_alice", _scratch("out7"), shard="MSG0", runner=LeakyRunner(), scratch_root=_scratch("leak")
    )

    assert result.ok is False
    assert FAKE_DECRYPT_KEY not in result.error
    assert FAKE_WRAPPED_KEY not in result.error
    assert "[REDACTED]" in result.error
    assert "sqlcipher open failed" in result.error


# --- 7. shard and account iteration -----------------------------------------------------------


def test_export_shard_iterates_conversations_and_keeps_going(monkeypatch, fake_home) -> None:
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    out_dir = _scratch("out8") / "MSG0"
    runner = FakeRunner(fail={"wxid_alice", "12345678@chatroom"})

    results = export_shard("MSG0", out_dir, runner=runner, scratch_root=_scratch("shard"))

    assert [r.conversation_id for r in results] == list(TALKERS)
    by_id = {r.conversation_id: r for r in results}
    assert by_id["wxid_bob"].ok is True
    assert by_id["wxid_bob"].messages == 7
    assert by_id["wxid_alice"].ok is False
    assert "synthetic export failure" in by_id["wxid_alice"].error
    assert by_id["12345678@chatroom"].ok is False
    # One failure did not abort the batch: all three were attempted.
    exports = [c for c in runner.calls if c["argv"][0] == "export"]
    assert [c["argv"][1] for c in exports] == list(TALKERS)


def test_only_filters_by_conversation_id(monkeypatch, fake_home) -> None:
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    runner = FakeRunner()

    results = export_shard(
        "MSG0",
        _scratch("out9"),
        only=["12345678@chatroom", "wxid_alice"],
        runner=runner,
        scratch_root=_scratch("only"),
    )

    assert [r.conversation_id for r in results] == ["12345678@chatroom", "wxid_alice"]
    assert all(r.ok for r in results)
    assert [c["argv"][1] for c in runner.calls if c["argv"][0] == "export"] == [
        "12345678@chatroom",
        "wxid_alice",
    ]


def test_only_accepts_a_display_name_only_as_an_id_and_finds_nothing(monkeypatch, fake_home) -> None:
    """Filtering by a display name must not silently select a conversation."""
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    runner = FakeRunner()

    results = export_shard(
        "MSG0",
        _scratch("out10"),
        only=["Alice 备注"],
        runner=runner,
        scratch_root=_scratch("onlyname"),
    )

    assert results == []
    assert [c for c in runner.calls if c["argv"][0] == "export"] == []


def test_export_account_iterates_shards(monkeypatch, fake_home) -> None:
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    runner = FakeRunner(fail={"wxid_alice"})

    results = export_account(
        ["MSG0", "MSG2"], _scratch("out11"), runner=runner, scratch_root=_scratch("account")
    )

    assert [r.shard for r in results] == ["MSG0"] * 3 + ["MSG2"] * 3
    assert [r.conversation_id for r in results] == list(TALKERS) * 2
    assert sum(1 for r in results if not r.ok) == 2
    listings = [c for c in runner.calls if c["argv"][0] == "sessions"]
    assert len(listings) == 2, "one listing per shard"
    assert list(SCRATCH_ROOT.rglob("privrecall_export_*")) == []


def test_export_account_records_a_listing_failure_instead_of_losing_the_account(
    monkeypatch, fake_home
) -> None:
    """A shard that cannot even be listed is not quietly reported as an empty shard.

    ``export_account`` is the account-wide path, so raising for one unlistable database would cost
    every *other* shard its history as well — and the ones already exported are indistinguishable
    from the ones never reached. ``export_shard`` therefore records the failure against the shard:
    one failed result naming it, nothing exported for it, and the account export carries on. What
    the old contract protected still holds, because "cannot list this shard" is visible as a failure
    rather than as an empty result the caller might read as "no conversations here".
    """
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    out_dir = _scratch("out12")
    runner = FakeRunner(fail_sessions=True)

    results = export_account(
        ["MSG0"],
        out_dir,
        runner=runner,
        scratch_root=_scratch("account_fail"),
    )

    failed = [r for r in results if not r.ok]
    assert len(failed) == 1, "an unlistable shard is exactly one recorded failure"
    assert failed[0].shard == "MSG0"
    assert "MSG0" in failed[0].error, "the failure must name the shard it belongs to"
    assert failed[0].conversation_id == "", "no conversation was reachable, so none is claimed"
    # The failure is recorded, never disguised: the results are neither empty nor a success.
    assert results, "a shard that could not be listed is not silently absent from the results"
    assert not any(r.ok for r in results), "and it is certainly not reported as an empty success"
    # Nothing was attempted, and so nothing was written, for the shard that could not be listed.
    assert [c for c in runner.calls if c["argv"][0] == "export"] == []
    assert list(out_dir.rglob("*_messages.json")) == [], "no export file for the unlistable shard"


# --- 8. boundary: no parsing, no sessions, no model -------------------------------------------


def _exporter_module():
    import exporter

    # A stray ``exporter`` module on ``sys.path`` would make every assertion below meaningless.
    assert Path(exporter.__file__).resolve() == BASE_DIR / "exporter.py"
    return exporter


def _imported_names(module) -> set[str]:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


FORBIDDEN_MODULES = ("memory.sessions", "memory.weflow", "memory.account", "memory.events", "memory.processor")
FORBIDDEN_SUBSTRINGS = ("embed", "openai", "anthropic", "langchain", "llm")


def test_module_does_not_import_the_parser_or_the_session_builder() -> None:
    module = _exporter_module()
    imported = _imported_names(module)

    assert "memory.conversations" in imported, "descriptors come from the existing module"
    assert "memory.shards" in imported
    for forbidden in FORBIDDEN_MODULES:
        assert forbidden not in imported, f"{forbidden} must not be imported by the exporter boundary"
    assert not any(
        # ``memory.*`` only: a stdlib name is not a parsing dependency.
        any(part in name for part in FORBIDDEN_SUBSTRINGS)
        for name in imported
        if name.startswith("memory")
    )
    assert not hasattr(module, "parse_weflow_events")
    assert not hasattr(module, "build_sessions")
    assert not hasattr(module, "MemoryEvent")
    assert not hasattr(module, "import_account")
    assert not hasattr(module, "MemoryChunk")


def test_module_deletes_only_through_the_scratch_cleanup() -> None:
    """The only deletion in the module is the scratch profile's own ``rmtree``."""
    source = Path(_exporter_module().__file__).read_text(encoding="utf-8")
    assert "unlink(" not in source
    assert "os.remove" not in source
    assert "write_bytes" not in source
    # Every deletion goes through the one helper that is documented as cleanup-safe.
    assert source.count("shutil.rmtree(") == 1


def test_export_result_carries_no_content_field() -> None:
    """The boundary's output is paths and counts. There is no field for a message body."""
    fields = set(ExportResult.__dataclass_fields__)
    assert fields == {"shard", "conversation_id", "path", "messages", "ok", "error"}


def test_export_environment_repoints_both_home_variables() -> None:
    """``USERPROFILE`` is what ``os.homedir()`` reads on Windows and ``HOME`` on POSIX; set both."""
    import exporter

    scratch = _scratch("env")
    env = exporter._export_environment(scratch, {"USERPROFILE": "C:\\real", "HOME": "/real", "KEEP": "yes"})

    assert env["USERPROFILE"] == str(scratch)
    assert env["HOME"] == str(scratch)
    assert env["KEEP"] == "yes", "the rest of the environment is inherited, not replaced"
    # The real values are not left anywhere for the child to fall back to.
    assert "C:\\real" not in env.values()
    assert "/real" not in env.values()


@pytest.mark.skipif(
    shutil.which("weflow-cli") is None, reason="weflow-cli is not installed on PATH"
)
def test_real_runner_launches_without_touching_the_account() -> None:
    """Exercise the real subprocess seam with ``--version``: no database, no config, no export.

    This is the one code path the fake runner cannot cover (the Windows ``.cmd`` shell invocation), so
    it is worth one process. ``run_subprocess`` invokes ``weflow-cli`` by name; a version banner reads
    no chats and writes nothing, and the assertions below deliberately check only that the process ran.
    """
    import exporter

    completed = exporter.run_subprocess(["--version"], dict(os.environ))

    assert completed.returncode == 0
    printed = str(completed.stdout).strip()
    assert printed, "the CLI printed nothing, so the shell invocation did not reach it"
    assert all(part.isdigit() for part in printed.split(".") if part), (
        f"unexpected --version output: {printed!r}"
    )


def test_scratch_default_root_is_the_system_temp_dir(monkeypatch, fake_home) -> None:
    """``scratch_root=None`` uses the OS temp dir, and still removes what it made."""
    import tempfile

    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    runner = FakeRunner()

    export_conversation("wxid_bob", _scratch("out15"), shard="MSG0", runner=runner)

    scratch = Path(runner.seen[0][0]["HOME"])
    assert scratch.parent == Path(tempfile.gettempdir())
    assert scratch.name.startswith("privrecall_export_")
    assert not scratch.exists()


def test_parse_session_listing_is_reused_not_reimplemented() -> None:
    """Identity parsing lives in ``memory.conversations``; the exporter must not fork it."""
    source = Path(_exporter_module().__file__).read_text(encoding="utf-8")
    assert "parse_session_listing" in source
    assert "displayName" not in source, "the sessions entry shape belongs to memory.conversations"
    assert 'entry.get("username")' not in source


def test_conversation_descriptor_identity_round_trips() -> None:
    """A descriptor's id is exactly what the exporter is invoked with and names its file after."""
    descriptors = parse_session_listing({"sessions": SESSIONS})
    assert all(isinstance(d, ConversationDescriptor) for d in descriptors)
    for descriptor in descriptors:
        assert f"{descriptor.conversation_id}_messages.json".endswith("_messages.json")
    assert {d.conversation_id for d in descriptors} == set(TALKERS)


def test_api_signatures_are_the_ones_callers_expect() -> None:
    """The brief's four entry points, with the parameter names and kinds callers already use."""
    import exporter

    listing = inspect.signature(exporter.list_conversations)
    assert list(listing.parameters) == [
        "shard",
        "multi_dir",
        "runner",
        "scratch_root",
    ], "multi_dir is the explicit shard -> database mapping; there is no global state"
    assert listing.parameters["shard"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert listing.parameters["runner"].kind is inspect.Parameter.KEYWORD_ONLY
    assert listing.parameters["runner"].default is None
    assert listing.parameters["scratch_root"].default is None

    one = inspect.signature(exporter.export_conversation)
    assert list(one.parameters) == ["talker", "out_dir", "shard", "multi_dir", "runner", "scratch_root"]
    assert one.parameters["talker"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert one.parameters["out_dir"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert one.parameters["shard"].kind is inspect.Parameter.KEYWORD_ONLY
    assert one.parameters["shard"].default is inspect.Parameter.empty, "shard is required, never guessed"

    shard = inspect.signature(exporter.export_shard)
    assert list(shard.parameters) == [
        "shard",
        "out_dir",
        "only",
        "conversations",
        "multi_dir",
        "runner",
        "scratch_root",
    ]
    assert shard.parameters["only"].kind is inspect.Parameter.KEYWORD_ONLY
    assert shard.parameters["only"].default is None
    assert shard.parameters["conversations"].kind is inspect.Parameter.KEYWORD_ONLY
    # Optional: a caller that already listed the shard reuses that listing, everyone else pays for it.
    assert shard.parameters["conversations"].default is None

    account = inspect.signature(exporter.export_account)
    assert list(account.parameters) == [
        "shards",
        "out_dir",
        "only",
        "multi_dir",
        "runner",
        "scratch_root",
    ]
    assert account.parameters["only"].kind is inspect.Parameter.KEYWORD_ONLY

    fields = exporter.ExportResult.__dataclass_fields__
    assert list(fields) == ["shard", "conversation_id", "path", "messages", "ok", "error"]
    assert fields["messages"].default == 0
    assert fields["ok"].default is True
    assert fields["error"].default == ""
