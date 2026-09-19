"""The account exporter boundary: run ``weflow-cli``, get files, touch nothing else.

Scope, and it is deliberately the *whole* module: **subprocess orchestration plus privacy**. This
module turns (shard, conversation) pairs into JSON files on disk and reports where they are. It does
not parse a single message, does not build a ``MemoryEvent``, does not build sessions, does not embed
and does not call a model. Parsing starts at ``memory.weflow`` and is driven by ``memory.account``;
identity comes from ``memory.conversations``. Keeping the sensitive path this narrow is the point: a
module that only knows how to invoke an external program is a module that cannot leak a message.

One command does not fit the (shard, conversation) shape — ``contacts --json``, which answers with
names rather than talkers — and it lives here anyway, because **no other module may run
``weflow-cli``**: :func:`list_contacts`. It touches no chat content either; it asks the exporter what
it knows, and ``memory.labels`` decides which of those answers is a name.

How a caller drives it
----------------------

::

    from memory.shards import discover_message_shards
    from exporter import export_account, list_conversations

    multi_dir = Path(r"C:\\Users\\me\\WeChat Files\\wxid_x\\Msg\\Multi")
    shards = [stem(name) for name in discover_message_shards(multi_dir)]   # ["MSG0", "MSG2", ...]

    descriptors = list_conversations("MSG0", multi_dir=multi_dir)
    results = export_account(shards, out_dir, multi_dir=multi_dir, only=["wxid_alice"])

``multi_dir`` is the account's ``Msg/Multi`` directory and is passed explicitly on every call — there
is no global state and no ambient account. A ``shard`` is a stem (``MSG0``) or a file name
(``MSG0.db``); it is also the label that comes back in :attr:`ExportResult.shard`, so the caller can
attribute a file to the database it came from. ``only`` filters by **conversation id**, never by a
display name.

The JSON files that come out still contain no conversation field (see ``memory.conversations``);
``ExportResult.conversation_id`` and the ``{talker}_messages.json`` file name are the identity, which
is why the caller is expected to record a manifest rather than re-derive it from a directory listing.

Why a "scratch profile" exists at all
-------------------------------------

WeChat 3.x splits one account's history across ``Msg/Multi/MSG0.db``, ``MSG1.db``, ``MSG2.db``, … and
the exporter reads exactly **one** database per run. Which one is decided by ``dbPath3x`` in its
config, and that config lives at::

    join(homedir(), '.weflow-cli')            # weflow-cli 1.6.0, configService.ts

``CONFIG_DIR`` is a module-level constant derived from ``homedir()``. There is **no environment
variable override** for it and **no CLI flag** that accepts a database path (verified against
``dist/src/services/configService.js`` and ``dist/bin/weflow-cli.js`` of the installed 1.6.0). So the
only supported way to point the exporter at a different shard is to change what ``homedir()``
resolves to for that child process:

1. create a scratch directory;
2. write ``<scratch>/.weflow-cli/config.json`` — a copy of the user's real config with **only**
   ``dbPath3x`` (and ``dbPath`` when the real config has one) repointed at the shard's database;
3. run the exporter with ``USERPROFILE`` and ``HOME`` set to ``<scratch>``;
4. delete the scratch directory in a ``finally:``, always.

**This depends entirely on the exporter's current ``homedir()``-based config resolution.** If a
future weflow-cli reads its config from anywhere else — an env var, an XDG path, a flag, OS keychain
or a shared per-user service — this mechanism silently stops switching shards, and the failure mode
is the dangerous one: every shard would be read from whichever database the *real* config names, so
the merged corpus would look complete while missing entire shards. If that day comes, replace
``EXPORT_PROFILE_DIRNAME`` / ``_export_environment`` with whatever the new resolution honours. If a
``--db``/``--db-path`` flag or a ``WEFLOW_CONFIG_DIR``-style override is ever added, prefer it over
this and delete the copy.

Privacy rules this module enforces
----------------------------------

* The user's real ``~/.weflow-cli`` is **never written to, never modified and never renamed**. The
  scratch copy is the only thing edited, and it is deleted afterwards.
* Key material the exporter needs (``decryptKey3x``, ``ntKey``, ``contactKey``, …) is copied
  *verbatim and silently* so the exporter can decrypt; it is never logged, printed, returned, or
  included in an error message. The scratch directory is created inside the OS temp dir, so nothing
  is copied into a location the user's own config did not already have it in — and it is removed in
  the same call.
* Nothing in this module ever returns or prints config *contents*; only the **names** of the keys it
  adjusted are recorded, and never inside ``ExportResult.error``.
* All surfaced subprocess stderr goes through :func:`redact`, which strips 32+ hex-char runs and
  32+ char base64-ish runs. A decrypted 3.x key is a 64-hex string and must never survive an
  exception message.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from memory.conversations import (
    ACCOUNT_MANIFEST_FILENAME,
    SESSION_LISTING_FILENAME,
    ConversationDescriptor,
    parse_session_listing,
    shard_stem,
)
from memory.labels import ContactRecord, parse_contact_listing
from memory.shards import discover_message_shards

__all__ = [
    "AccountExportReport",
    "ExportResult",
    "ExporterError",
    "export_account",
    "export_account_tree",
    "export_conversation",
    "export_shard",
    "list_contacts",
    "list_conversations",
    "list_conversations_across_shards",
    "list_shard_database_paths",
    "real_config_path",
    "redact",
    "run_subprocess",
]

#: The exporter's CLI name, resolved through ``PATH``.
CLI_NAME = "weflow-cli"

#: Config directory the exporter builds from ``homedir()``. Mirrored, never imported.
EXPORT_PROFILE_DIRNAME = ".weflow-cli"

#: Config file inside that directory.
EXPORT_CONFIG_FILENAME = "config.json"

#: Prefix for the per-call scratch profile directories.
SCRATCH_PREFIX = "privrecall_export_"

#: Database-path keys the scratch profile may rewrite. Everything else is copied untouched.
SHARD_DB_KEYS: tuple[str, ...] = ("dbPath", "dbPath3x")

#: Default timeout for one exporter invocation. The CLI does not cancel cleanly on Windows, so this
#: bounds a hung decrypt rather than stopping it gracefully.
DEFAULT_TIMEOUT_SECONDS = 600.0

Runner = Callable[[list[str], dict[str, str]], "subprocess.CompletedProcess[str]"]

_HEX_OR_BASE64_RUN = re.compile(r"\b[0-9a-fA-F]{32,}\b|\b[A-Za-z0-9+/]{32,}={0,2}")
_REDACTED = "[REDACTED]"


class ExporterError(RuntimeError):
    """A precondition failed before the exporter could be invoked.

    Raised for a missing real config and for an unresolvable shard — the cases where continuing would
    mean exporting *some other* database and reporting success. The message never contains key
    material (see :func:`redact`).
    """


class ListingFailed(ExporterError):
    """The exporter ran for a shard but produced no usable listing.

    Deliberately a subclass of :class:`ExporterError` so a caller may treat it as the fatal thing it is
    to a single-shard run, while :func:`export_shard` — which is exporting a whole account — records it
    against that shard and keeps going. "Cannot list this shard" must never be reported as "this shard
    is empty", and must never cost the other shards their data either.
    """


@dataclass(frozen=True)
class ExportResult:
    """What one (shard, conversation) export produced. Paths and counts only — never content."""

    shard: str
    conversation_id: str
    path: str
    messages: int = 0  # from the CLI's --json result, 0 if unknown
    ok: bool = True
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "shard": self.shard,
            "conversation_id": self.conversation_id,
            "path": self.path,
            "messages": self.messages,
            "ok": self.ok,
            "error": self.error,
        }


# --- the runner seam --------------------------------------------------------------------------


def run_subprocess(argv: list[str], env: dict[str, str]) -> "subprocess.CompletedProcess[str]":
    """The real runner: one ``weflow-cli`` invocation, captured and decoded as UTF-8.

    ``weflow-cli`` is a Node CLI installed as a Windows ``.cmd`` shim, so on Windows it is invoked
    through the shell. The command line is built with ``subprocess.list2cmdline`` rather than joined
    by hand, because a talker id or an output path containing a space would otherwise be split into
    two arguments.
    """
    executable = shutil.which(CLI_NAME) or CLI_NAME
    if os.name == "nt":
        return subprocess.run(
            subprocess.list2cmdline([executable, *argv]),
            env=env,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    return subprocess.run(
        [executable, *argv],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )


# --- redaction --------------------------------------------------------------------------------


def redact(text: str) -> str:
    """Strip hex keys and long base64 blobs from anything that might be surfaced.

    The 3.x decrypt key is 64 hex characters; salts and wrapped keys are base64. Both shapes are
    covered by one pattern (hex is a subset of the base64 alphabet), alternated so that a run without
    word boundaries — ``key=deadbeef…`` — is still caught by the hex branch. The replacement
    ``[REDACTED]`` does not match either branch, so a single pass is complete.
    """
    if not text:
        return ""
    return _HEX_OR_BASE64_RUN.sub(_REDACTED, str(text))


# --- shard -> database ------------------------------------------------------------------------


def _shard_database_path(multi_dir: Path, shard: str) -> Path:
    """Map a shard stem (``MSG0``) to the database file inside ``multi_dir``.

    Accepts the stem the detector produces (``discover_message_shards`` → ``MSG0.db`` →
    ``memory.account.shard_stem`` → ``MSG0``) as well as a raw file name. Read-only: names only, no
    database is opened and no key is needed to resolve a path.

    Raises rather than falling back to "whatever the config says", because that fallback is exactly
    the silent wrong-shard export this module exists to prevent.
    """
    directory = Path(multi_dir)
    if not directory.is_dir():
        raise ExporterError(
            f"message shard directory not found: {directory} - point multi_dir at the account's "
            "Msg/Multi directory (the one holding MSG0.db, MSG1.db, ...)"
        )
    name = str(shard).strip()
    stem = name[: -len(".db")] if name.endswith(".db") else name
    for candidate in (name, f"{stem}.db"):
        path = directory / candidate
        if path.is_file():
            return path
    available = ", ".join(discover_message_shards(directory)) or "none"
    raise ExporterError(f"shard {shard!r} has no database in {directory} (found: {available})")


# --- scratch profile --------------------------------------------------------------------------


def real_config_path(env: Mapping[str, str] | None = None) -> Path:
    """Where the exporter's real config lives for *this* process.

    Mirrors weflow-cli's ``join(homedir(), '.weflow-cli', 'config.json')``. Only ever read, and only
    for two things: to prove it exists, and to copy it into a scratch profile.
    """
    source = os.environ if env is None else env
    home = source.get("USERPROFILE") or source.get("HOME") or str(Path.home())
    return Path(home) / EXPORT_PROFILE_DIRNAME / EXPORT_CONFIG_FILENAME


def _read_config_mapping(path: Path) -> dict[str, Any]:
    """Read a config file into a plain ``dict`` without interpreting or logging its contents."""
    if not path.is_file():
        raise ExporterError(
            f"weflow-cli config not found at {path} - run `weflow-cli init` first, or point the "
            "exporter at the account that is configured. Refusing to run with an empty config: it "
            "would export the wrong database (or none) and still report success."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        # A JSONDecodeError message quotes the offending document, which is the config itself.
        raise ExporterError(
            f"weflow-cli config at {path} is not readable JSON ({type(exc).__name__}); "
            "refusing to run with a config that cannot be understood"
        ) from None
    if not isinstance(raw, dict):
        raise ExporterError(f"weflow-cli config at {path} is not a JSON object")
    return raw


def _patched_config(config: Mapping[str, Any], database: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Copy a config, repointing only the fields that select the database.

    Returns ``(patched, adjusted_key_names)``. Everything else — including key material — is copied
    byte-for-byte and never inspected.
    """
    patched = dict(config)
    adjusted: list[str] = []
    for key in SHARD_DB_KEYS:
        if key not in patched:
            continue
        if patched.get(key):
            patched[key] = str(database)
            adjusted.append(key)
    if not adjusted:
        # A 3.x shard is selected by dbPath3x; a config carrying neither key cannot be a 3.x account
        # config, and guessing a key to add would be writing a config we do not understand.
        raise ExporterError(
            "the weflow-cli config has neither dbPath nor dbPath3x set, so it does not select a "
            "database this module can repoint; add the 3.x database path with `weflow-cli init` first"
        )
    return patched, tuple(adjusted)


def _write_scratch_profile(config: Mapping[str, Any], scratch_root: Path | None) -> Path:
    """Create a scratch profile holding a copy of the config. Returns its root directory.

    The directory is created inside ``scratch_root`` when given and inside the OS temp dir otherwise
    (``scratch_root=None`` → ``tempfile.mkdtemp(prefix="privrecall_export_")``). The config is the
    only file written; nothing is written to the real ``~/.weflow-cli``.

    Either the whole profile exists when this returns, or nothing does: a partially written profile
    would be the one failure this module cannot leave on disk, because it carries the copied key.
    """
    if scratch_root is None:
        scratch = Path(tempfile.mkdtemp(prefix=SCRATCH_PREFIX))
    else:
        root = Path(scratch_root)
        root.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix=SCRATCH_PREFIX, dir=str(root)))
    try:
        profile = scratch / EXPORT_PROFILE_DIRNAME
        profile.mkdir(parents=True, exist_ok=True)
        (profile / EXPORT_CONFIG_FILENAME).write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        _discard_scratch_profile(scratch)
        raise
    return scratch


def _prepare_scratch_profile(
    database: Path | None,
    scratch_root: Path | None,
) -> tuple[Path, tuple[str, ...]]:
    """Build the scratch profile for one exporter run.

    ``database=None`` means "list against whatever database the real config already names" — the only
    case is :func:`list_conversations` without a ``multi_dir``. Even then the config must select a
    database: a config with no ``dbPath``/``dbPath3x`` would run the exporter against nothing and
    report an empty account, which is the failure mode this module exists to make loud.

    Returns the scratch root and the names of the keys that were repointed (names only; contents never
    leave this function).
    """
    config = _read_config_mapping(real_config_path())
    if database is None:
        if not any(config.get(key) for key in SHARD_DB_KEYS):
            raise ExporterError(
                "the weflow-cli config selects no database (neither dbPath nor dbPath3x is set); run "
                "`weflow-cli init`, or pass multi_dir so the shard's database can be resolved"
            )
        return _write_scratch_profile(config, scratch_root), ()
    patched, adjusted = _patched_config(config, Path(database))
    return _write_scratch_profile(patched, scratch_root), adjusted


def _discard_scratch_profile(scratch: Path | None) -> None:
    """Remove a scratch profile and everything in it. Never raises out of cleanup.

    Called from a ``finally:`` block, including on the exception path: a scratch profile holds the
    copied key material, so leaving one behind is a privacy failure, not a tidiness one.
    """
    if scratch is None:
        return
    try:
        shutil.rmtree(scratch, ignore_errors=True)
    except Exception:  # pragma: no cover - rmtree already swallows OSErrors
        pass


def _export_environment(scratch: Path, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for one exporter run: the process env with ``homedir()`` repointed.

    ``USERPROFILE`` is what ``os.homedir()`` reads on Windows and ``HOME`` is what it reads on POSIX,
    so both are set: one of them is the documented override on each platform and the other is
    harmless context for the Node child.
    """
    env = dict(os.environ if base_env is None else base_env)
    home = str(scratch)
    env["USERPROFILE"] = home
    env["HOME"] = home
    return env


# --- exporter output --------------------------------------------------------------------------


def _stdout_json(completed: "subprocess.CompletedProcess[str]") -> Any:
    """The CLI's ``--json`` payload, or ``None`` when it emitted something unparseable."""
    text = (getattr(completed, "stdout", "") or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _cli_message_count(payload: Any) -> int:
    """``count`` from the CLI's ``--json`` result; ``0`` when it is absent or not a number."""
    if not isinstance(payload, Mapping):
        return 0
    value = payload.get("count")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _describe_failure(
    completed: "subprocess.CompletedProcess[str]",
    payload: Any,
) -> str:
    """A short, redacted reason for a failed run. Never includes config contents."""
    parts: list[str] = []
    if isinstance(payload, Mapping):
        code = payload.get("code") or payload.get("error") or payload.get("message")
        if code:
            parts.append(f"code={redact(str(code))}")
        if payload.get("success") is not True:
            extra = payload.get("error") or payload.get("message")
            if extra and str(extra) != str(code):
                parts.append(redact(str(extra)))
    stderr = redact(getattr(completed, "stderr", "") or "").strip()
    if stderr:
        parts.append(f"stderr={stderr[:400]}")
    if not parts:
        stdout = redact(getattr(completed, "stdout", "") or "").strip()
        parts.append(f"stdout={stdout[:400]}" if stdout else "no output from weflow-cli")
    return " | ".join(parts)


# --- public API -------------------------------------------------------------------------------


def list_conversations(
    shard: str,
    multi_dir: Path | None = None,
    *,
    runner: Runner | None = None,
    scratch_root: Path | None = None,
) -> tuple[ConversationDescriptor, ...]:
    """Every conversation the exporter can see in one shard, keyed by talker id.

    ``weflow-cli sessions --json`` is scoped to whatever database the config names, so this switches
    to the requested shard exactly like :func:`export_conversation` does — a listing from the wrong
    shard is how a conversation silently disappears from the account.

    The returned descriptors come from :func:`memory.conversations.parse_session_listing`; a display
    name is carried as an alias and is never an id.
    """
    runner = runner or run_subprocess
    database = None if multi_dir is None else _shard_database_path(Path(multi_dir), shard)

    scratch: Path | None = None
    try:
        scratch, adjusted = _prepare_scratch_profile(database, scratch_root)
        completed = runner(["sessions", "--json", "--limit", "1000"], _export_environment(scratch))
        payload = _stdout_json(completed)
        if payload is None:
            raise ListingFailed(
                f"weflow-cli sessions --json produced no JSON for shard {shard!r} (exit "
                f"{getattr(completed, 'returncode', '?')}); {_describe_failure(completed, None)}"
            )
        if isinstance(payload, Mapping) and payload.get("success") is False:
            raise ListingFailed(
                f"weflow-cli sessions failed for shard {shard!r} ({', '.join(adjusted)} repointed); "
                f"{_describe_failure(completed, payload)}"
            )
        return parse_session_listing(payload)
    finally:
        _discard_scratch_profile(scratch)


def list_contacts(
    *,
    runner: Runner | None = None,
    scratch_root: Path | None = None,
) -> tuple[ContactRecord, ...]:
    """Every name the exporter can resolve for this account's contacts.

    ``weflow-cli contacts --json`` is the only command that answers with *names* rather than talkers,
    and it is the input a human-readable conversation label is resolved from. It runs inside the same
    scratch profile as every other call here — the same ``USERPROFILE`` switch, the same cleanup, the
    same redaction — even though contacts do not depend on which shard is selected: a second way to
    invoke the CLI is exactly the thing this module exists to prevent.

    Two consequences of reusing that mechanism, both deliberate: the config must select a database
    (``_prepare_scratch_profile`` refuses a config that names none, because a run against nothing
    would look like an account with no names), and each call pays one config copy, which is the price
    of never writing the user's real ``~/.weflow-cli``.

    Raises :class:`ListingFailed` when the CLI produced no usable listing — the caller must be able to
    tell "there are no names" from "we could not ask". A successful run with no resolved names is a
    legitimate, common result on the tested version: it echoes each entry's own id back as that
    entry's display name, which :func:`memory.labels.parse_contact_listing` records as an offer and
    :func:`memory.labels.usable_name` then rejects. The shape of a listing entry is parsed in exactly
    one place, and it is not this module.
    """
    runner = runner or run_subprocess

    scratch: Path | None = None
    try:
        scratch, adjusted = _prepare_scratch_profile(None, scratch_root)
        completed = runner(["contacts", "--json"], _export_environment(scratch))
        payload = _stdout_json(completed)
        if payload is None:
            raise ListingFailed(
                "weflow-cli contacts --json produced no JSON (exit "
                f"{getattr(completed, 'returncode', '?')}); {_describe_failure(completed, None)}"
            )
        if isinstance(payload, Mapping) and payload.get("success") is False:
            raise ListingFailed(
                f"weflow-cli contacts failed ({', '.join(adjusted) or 'no key repointed'}); "
                f"{_describe_failure(completed, payload)}"
            )
        return parse_contact_listing(payload)
    finally:
        _discard_scratch_profile(scratch)


def export_conversation(
    talker: str,
    out_dir: Path,
    *,
    shard: str,
    multi_dir: Path | None = None,
    runner: Runner | None = None,
    scratch_root: Path | None = None,
) -> ExportResult:
    """Export one conversation from one shard. Never raises for an exporter failure.

    A failed run is reported as ``ok=False`` with a redacted ``error``: the caller is exporting a
    whole account, and one unreadable conversation must not abort the others.
    """
    runner = runner or run_subprocess
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    expected = directory / f"{talker}_messages.json"

    scratch: Path | None = None
    try:
        database = None if multi_dir is None else _shard_database_path(Path(multi_dir), shard)
        scratch, adjusted = _prepare_scratch_profile(database, scratch_root)
        argv = [
            "export",
            str(talker),
            "json",
            "--output",
            str(directory),
            "--non-interactive",
            "--limit",
            "0",
            "--json",
        ]
        try:
            completed = runner(argv, _export_environment(scratch))
        except Exception as exc:  # noqa: BLE001 - one conversation may not abort the shard
            return ExportResult(
                shard=shard,
                conversation_id=str(talker),
                path=str(expected),
                ok=False,
                error=f"weflow-cli could not be run ({type(exc).__name__}): {redact(str(exc))}",
            )

        payload = _stdout_json(completed)
        succeeded = isinstance(payload, Mapping) and payload.get("success") is True
        if not succeeded:
            return ExportResult(
                shard=shard,
                conversation_id=str(talker),
                path=str(expected),
                ok=False,
                error=(
                    f"weflow-cli export failed for {talker} in shard {shard} (exit "
                    f"{getattr(completed, 'returncode', '?')}, {'+'.join(adjusted)} repointed): "
                    f"{_describe_failure(completed, payload)}"
                ),
            )

        reported = payload.get("path")
        path = str(reported) if isinstance(reported, str) and reported else str(expected)
        return ExportResult(
            shard=shard,
            conversation_id=str(talker),
            path=path,
            messages=_cli_message_count(payload),
        )
    finally:
        _discard_scratch_profile(scratch)


def export_shard(
    shard: str,
    out_dir: Path,
    *,
    only: Sequence[str] | None = None,
    conversations: Sequence[ConversationDescriptor] | None = None,
    multi_dir: Path | None = None,
    runner: Runner | None = None,
    scratch_root: Path | None = None,
) -> list[ExportResult]:
    """Export every conversation of one shard. ``only`` filters by exact conversation id.

    ``conversations`` lets a caller that has *already* listed the shard reuse that listing instead of
    paying for a second exporter run — the account tree writer does exactly that.

    Failures are recorded, not raised: the results list always has one entry per attempted
    conversation, in listing order. A shard that cannot be *listed* becomes one failed entry —
    ``conversation_id=""`` — so an account-wide export loses that shard's messages and says so, rather
    than losing every other shard's as well.
    """
    if conversations is None:
        try:
            conversations = list_conversations(
                shard, multi_dir, runner=runner, scratch_root=scratch_root
            )
        except ListingFailed as exc:
            return [
                ExportResult(
                    shard=shard,
                    conversation_id="",
                    path="",
                    ok=False,
                    error=f"could not list shard {shard}: {exc}",
                )
            ]
    wanted = None if only is None else {str(item) for item in only}
    results: list[ExportResult] = []
    for descriptor in conversations:
        if wanted is not None and descriptor.conversation_id not in wanted:
            continue
        results.append(
            export_conversation(
                descriptor.conversation_id,
                out_dir,
                shard=shard,
                multi_dir=multi_dir,
                runner=runner,
                scratch_root=scratch_root,
            )
        )
    return results


def export_account(
    shards: Sequence[str],
    out_dir: Path,
    *,
    only: Sequence[str] | None = None,
    multi_dir: Path | None = None,
    runner: Runner | None = None,
    scratch_root: Path | None = None,
) -> list[ExportResult]:
    """Export every conversation of every shard, in the order the shards are given.

    One entry per attempt, across all shards, so the caller can merge them per conversation and report
    a shard that produced nothing rather than assuming it was empty.
    """
    results: list[ExportResult] = []
    for shard in shards:
        results.extend(
            export_shard(
                shard,
                out_dir,
                only=only,
                multi_dir=multi_dir,
                runner=runner,
                scratch_root=scratch_root,
            )
        )
    return results


def list_conversations_across_shards(
    shards: Sequence[str],
    *,
    multi_dir: Path | None = None,
    runner: Runner | None = None,
    scratch_root: Path | None = None,
) -> tuple[ConversationDescriptor, ...]:
    """Union the listings of several shards, keeping per-shard provenance in ``metadata``."""
    merged: dict[str, ConversationDescriptor] = {}
    for shard in shards:
        for descriptor in list_conversations(
            shard, multi_dir, runner=runner, scratch_root=scratch_root
        ):
            previous = merged.get(descriptor.conversation_id)
            previous_metadata = dict(previous.metadata) if previous else {}
            shards_seen = tuple(dict.fromkeys(tuple(previous_metadata.get("shards", ())) + (shard,)))
            merged[descriptor.conversation_id] = ConversationDescriptor(
                conversation_id=descriptor.conversation_id,
                display_name=descriptor.display_name or (previous.display_name if previous else ""),
                conversation_type=descriptor.conversation_type,
                aliases=tuple(
                    dict.fromkeys((previous.aliases if previous else ()) + descriptor.aliases)
                ),
                metadata={**previous_metadata, **descriptor.metadata, "shards": shards_seen},
            )
    return tuple(merged[key] for key in sorted(merged))


def list_shard_database_paths(multi_dir: Path, shards: Sequence[str] | None = None) -> dict[str, str]:
    """``shard stem -> database path`` for the shards of one account. File names only, no key access."""
    directory = Path(multi_dir)
    names = list(shards) if shards is not None else [
        name[: -len(".db")] if name.endswith(".db") else name
        for name in discover_message_shards(directory)
    ]
    return {name: str(_shard_database_path(directory, name)) for name in names}


# --- the account export tree ------------------------------------------------------------------


@dataclass(frozen=True)
class AccountExportReport:
    """What an account export produced. Counts, shard names and redacted errors — never content."""

    detected_shards: tuple[str, ...] = ()
    exported_shards: tuple[str, ...] = ()
    conversations_listed: int = 0
    files_written: int = 0
    total_messages: int = 0
    failures: tuple[str, ...] = ()
    output_dir: str = ""
    #: How many conversations a ``only`` filter narrowed this export to; 0 means the whole account.
    #: The import side needs this to know that a tree with every shard present is still not the whole
    #: account: "all shards exported" must never be read as "all conversations searchable".
    filtered_conversations: int = 0

    @property
    def missing_shards(self) -> tuple[str, ...]:
        """Shards whose messages were NOT written, so their history is invisible downstream."""
        return tuple(sorted(set(self.detected_shards) - set(self.exported_shards)))

    @property
    def partial(self) -> bool:
        return bool(self.missing_shards)

    def lines(self) -> list[str]:
        out = [
            f"output dir     : {self.output_dir}",
            f"message shards : {len(self.exported_shards)} exported of "
            f"{len(self.detected_shards)} detected {list(self.detected_shards)}",
            f"conversations  : {self.conversations_listed} listed",
            f"files written  : {self.files_written} ({self.total_messages} messages reported by the "
            "exporter)",
        ]
        if self.filtered_conversations:
            out.append(
                f"note           : this export was deliberately narrowed to "
                f"{self.filtered_conversations} conversation(s) - every other conversation of this "
                "account is absent from the tree and is therefore not searchable"
            )
        if self.partial:
            out.append(
                f"WARNING        : {len(self.missing_shards)} message shard(s) were not exported "
                f"{list(self.missing_shards)} - this account history is PARTIAL, and messages in "
                "those shards are invisible to every answer"
            )
        for failure in self.failures:
            out.append(f"failure        : {failure}")
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "detected_shards": list(self.detected_shards),
            "exported_shards": list(self.exported_shards),
            "missing_shards": list(self.missing_shards),
            "partial": self.partial,
            "conversations_listed": self.conversations_listed,
            "files_written": self.files_written,
            "total_messages": self.total_messages,
            "failures": list(self.failures),
            "output_dir": self.output_dir,
            "filtered_conversations": self.filtered_conversations,
        }


def export_account_tree(
    out_dir: Path,
    *,
    multi_dir: Path | None = None,
    shards: Sequence[str] | None = None,
    only: Sequence[str] | None = None,
    runner: Runner | None = None,
    scratch_root: Path | None = None,
) -> AccountExportReport:
    """Export an entire account into the tree the account importer reads.

    ::

        <out_dir>/shard_manifest.json          detected vs exported shards (no talkers, no content)
        <out_dir>/MSG0/sessions.json           this shard's conversation listing (talkers + labels)
        <out_dir>/MSG0/{talker}_messages.json  the messages themselves

    This is the only writer of that layout. Writing the per-shard listing next to the exports is what
    lets ``memory.account`` recover conversation identity without guessing from a file name, and the
    manifest is what lets it report a shard that exists but was never exported. Both directories hold
    real chat data and must live somewhere ignored by git (``data/real/`` by convention).

    A shard that fails is recorded in ``failures`` and simply absent from the tree: the remaining
    shards are still exported, because a partial history that says "PARTIAL" is more useful than no
    history at all — and the import report will refuse to call it complete.

    ``only`` narrows the export to those conversation ids, and each shard still gets its **full**
    ``sessions.json`` listing — so a filter whose conversations happen to span every shard leaves no
    shard missing and the tree looks complete while most of the account is absent from it. The
    manifest therefore records ``filtered_conversations`` (the number of conversations the export was
    narrowed to, 0 when unfiltered) as a **count**; the importer turns that into PARTIAL. Identity is
    still never written to the manifest: no id, name or talker goes in, only how many were selected.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    detected: tuple[str, ...] = ()
    if multi_dir is not None:
        detected = tuple(shard_stem(name) for name in discover_message_shards(Path(multi_dir)))
    if not detected:
        detected = tuple(shard_stem(name) for name in (shards or ()))
    if not detected:
        raise ExporterError(
            "no message shards to export: pass multi_dir (the account's Msg/Multi directory) or an "
            "explicit list of shard stems"
        )
    targets = tuple(shard_stem(name) for name in shards) if shards else detected

    # How many conversations this export was narrowed to. A filter is recorded as a *count* only —
    # the manifest must never learn which conversations they were. A filter that matches nothing is
    # still a filter: only an absent ``only`` means the whole account was exported.
    filtered = len({str(item) for item in only}) if only else 0

    failures: list[str] = []
    exported: list[str] = []
    listed = written = messages = 0

    for shard in targets:
        shard_dir = directory / shard
        shard_dir.mkdir(parents=True, exist_ok=True)
        try:
            conversations = list_conversations(
                shard, multi_dir, runner=runner, scratch_root=scratch_root
            )
        except ListingFailed as exc:
            failures.append(redact(str(exc)))
            continue

        (shard_dir / SESSION_LISTING_FILENAME).write_text(
            json.dumps([descriptor.as_dict() for descriptor in conversations], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        listed += len(conversations)

        results = export_shard(
            shard,
            shard_dir,
            only=only,
            conversations=conversations,
            multi_dir=multi_dir,
            runner=runner,
            scratch_root=scratch_root,
        )
        for result in results:
            if result.ok:
                written += 1
                messages += result.messages
            else:
                failures.append(redact(result.error))
        if any(result.ok for result in results):
            exported.append(shard)

    manifest = {
        "schema": "privrecall-account-export/v1",
        "detected_shards": list(detected),
        "exported_shards": sorted(exported),
        "conversations_listed": listed,
        "files_written": written,
        "total_messages": messages,
        "filtered_conversations": filtered,
        "failures": failures,
    }
    (directory / ACCOUNT_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return AccountExportReport(
        detected_shards=tuple(sorted(detected)),
        exported_shards=tuple(sorted(exported)),
        conversations_listed=listed,
        files_written=written,
        total_messages=messages,
        failures=tuple(failures),
        output_dir=str(directory),
        filtered_conversations=filtered,
    )
