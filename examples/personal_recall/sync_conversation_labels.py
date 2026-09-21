"""Resolve human-readable conversation names into ``<account>/conversation_labels.json``.

    python sync_conversation_labels.py --account data/real/account_full

The evidence card's header ("我, 对方 · 2025-05-12") is unambiguous inside one conversation and
ambiguous across an account of 272. The name that fixes that is *not* in the export tree: WeFlow
exports carry no conversation field at all, and the talker id is the only identity there is. The one
command that answers with names is ``weflow-cli contacts --json``, so this command asks it once, joins
its answer against the conversations the tree actually holds, and records the result next to them.

It writes the sidecar and nothing else. No message, no chunk, no embedding, no index: names are
evidence *readability*, never retrieval. A name is not injected into any chunk's text or into what
gets embedded — see ``memory.labels``.

Why a sidecar rather than a re-resolution at render time
-------------------------------------------------------

Resolving at render time would mean the web process invoking ``weflow-cli`` — a subprocess in the
request path of a server that holds private chat, and a second place that knows how to run the
exporter. Instead the CLI does the asking, the answer is recorded once, and the UI only reads. The
sidecar also survives the case the listing cannot: on the tested version every ``displayName`` equals
its ``username``, so ``sessions.json`` has no name to give — the sidecar is where a name will appear
the moment one exists.

Privacy rules this command follows (the mechanism is ``exporter.py``'s, reused unchanged)
----------------------------------------------------------------------------------------

* The user's real ``~/.weflow-cli/config.json`` is **never written**. It is read once, copied into a
  scratch profile in the temp directory, and the scratch profile is deleted in a ``finally:``.
  ``tests/test_labels.py`` hashes the config before and after a run, the way ``tests/test_account_tree``
  does.
* **The output is coverage, never names**: counts of conversations, resolved labels and the
  direct/group split. No label, no talker, no wxid is printed, and a failure is reported through
  ``exporter.redact``.
* The sidecar itself *does* hold names and talkers — it is real data. Its default location is inside
  the account export tree, which lives in ``data/real/`` and is excluded by ``.gitignore``
  (``data/real/`` is ignored as a whole directory, confirmed there). A ``--out`` elsewhere, or an
  ``--account`` outside ``data/real/``, can land the file in a tracked path: the command warns when
  git says the destination is not ignored, because one ``git add -A`` would then commit real names
  and the ids behind them.

Re-running is safe and idempotent: resolution is deterministic, the sidecar is written sorted by
conversation id, and the same inputs produce byte-identical output. A run that cannot reach the
exporter writes nothing at all, so an interrupted run never replaces a good sidecar with an empty one.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from exporter import (  # noqa: E402
    ExporterError,
    list_contacts,
    redact,
    run_subprocess,
)
from memory import load_account_directory  # noqa: E402
from memory.labels import (  # noqa: E402
    CONVERSATION_LABEL_FILENAME,
    GROUP,
    resolve_conversation_labels,
    write_label_sidecar,
)

#: Where the export tree lives by default; the sidecar is written inside it. Under ``data/real/``,
#: which ``.gitignore`` excludes — the sidecar carries real names and talkers.
DEFAULT_ACCOUNT_DIR = BASE_DIR / "data" / "real" / "account_full"

#: Said out loud instead of leaving a zero in a count column. A run that resolves nothing on this
#: installation is the *measured* result, not a bug, and a reader who does not know that will read
#: the empty sidecar as a broken feature.
#:
#: The wording states what was measured and nothing more. An earlier version of this note named
#: `weflow-cli init` in an interactive terminal as "the known unlock" — that was a hypothesis, and it
#: has since been tested and disproven: `init` was run in a terminal, and `contacts --json` afterwards
#: still returned every `displayName` equal to its own `username` (120 of 120 contacts, 0 real names).
#: So the measured fact is that the tested CLI version exposes no human-readable name through any
#: non-interactive command, whatever its configuration. That is a limit of the installed tool, not a
#: gap in this project: the resolution layer below is complete and will use a name the moment a
#: version offers one. No fix is claimed here, because none was found.
NO_NAMES_NOTE = (
    "no conversation name resolved - this is the measured result on the tested weflow-cli, which "
    "exposes no human-readable name through any non-interactive command: `contacts --json` returns "
    "every displayName equal to its username (120 of 120 contacts, 0 real names), including after "
    "`weflow-cli init` was run in an interactive terminal. The resolution layer is complete and will "
    "use a name as soon as a version offers one; until then the UI shows no conversation header "
    "rather than an internal id."
)


@dataclass(frozen=True)
class LabelSyncReport:
    """What one sync produced: counts, paths and a redacted error. Never a name, talker or wxid."""

    account: str = ""
    sidecar: str = ""
    conversations: int = 0
    direct: int = 0
    group: int = 0
    records: int = 0
    resolved_direct: int = 0
    resolved_group: int = 0
    written: bool = False
    error: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def resolved(self) -> int:
        return self.resolved_direct + self.resolved_group

    @property
    def unresolved(self) -> int:
        return max(self.conversations - self.resolved, 0)

    def lines(self) -> list[str]:
        out = [
            f"account        : {self.account}",
            f"sidecar        : {self.sidecar}"
            + ("" if self.written else "  (not written)"),
            f"conversations  : {self.conversations} (direct {self.direct}, group {self.group})",
            f"contact records: {self.records}",
            f"labels resolved: {self.resolved} of {self.conversations}",
            f"  direct       : {self.resolved_direct} resolved / {self.direct}",
            f"  group        : {self.resolved_group} resolved / {self.group}",
            f"unresolved     : {self.unresolved}",
        ]
        for note in self.notes:
            out.append(f"note           : {note}")
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "account": self.account,
            "sidecar": self.sidecar,
            "conversations": self.conversations,
            "direct": self.direct,
            "group": self.group,
            "contact_records": self.records,
            "resolved": self.resolved,
            "resolved_direct": self.resolved_direct,
            "resolved_group": self.resolved_group,
            "unresolved": self.unresolved,
            "written": self.written,
            "error": self.error,
            "notes": list(self.notes),
        }


def sidecar_path_for(account_dir: Path) -> Path:
    """``<account>/conversation_labels.json`` — inside the export tree, next to the manifest."""
    return Path(account_dir) / CONVERSATION_LABEL_FILENAME


def sync_labels(
    account_dir: Path,
    *,
    out: Path | None = None,
    runner: Any = None,
    scratch_root: Path | None = None,
) -> LabelSyncReport:
    """Resolve and write the label sidecar. Never raises; a failure is a report with ``error`` set.

    ``runner`` and ``scratch_root`` are the exporter's own seams, threaded through so the tests can
    drive a fake ``weflow-cli`` and keep every scratch directory inside the workspace. Production
    callers pass neither.
    """
    account = Path(account_dir)
    target = Path(out) if out is not None else sidecar_path_for(account)
    report = LabelSyncReport(account=str(account), sidecar=str(target))

    try:
        layout = load_account_directory(account)
    except (OSError, ValueError) as exc:
        return _failed(report, f"could not read {account} as an export tree ({type(exc).__name__})")

    descriptors = tuple(layout.descriptors)
    direct = sum(1 for d in descriptors if d.conversation_type != GROUP)
    report = replace(
        report,
        conversations=len(descriptors),
        direct=direct,
        group=len(descriptors) - direct,
    )
    if not descriptors:
        # An empty tree resolves to an empty sidecar that looks exactly like "no names exist here".
        # Refuse instead: the likelier cause is a wrong --account, and saying so costs one line.
        return _failed(
            report,
            f"no conversations found in {account} - is that an account export tree written by "
            "export_account.py (a directory per shard, one <talker>_messages.json per conversation)?",
        )

    try:
        records = list_contacts(runner=runner or run_subprocess, scratch_root=scratch_root)
    except ExporterError as exc:
        return _failed(report, f"weflow-cli contacts --json could not be read: {redact(str(exc))}")

    labels = resolve_conversation_labels(descriptors, records)
    report = replace(
        report,
        records=len(records),
        resolved_direct=sum(1 for label in labels.values() if label.kind != GROUP),
        resolved_group=sum(1 for label in labels.values() if label.kind == GROUP),
        notes=() if labels else (NO_NAMES_NOTE,),
    )
    try:
        write_label_sidecar(target, labels)
    except OSError as exc:
        return _failed(report, f"could not write {target} ({type(exc).__name__})")
    return replace(report, written=True)


def _failed(report: LabelSyncReport, error: str) -> LabelSyncReport:
    """The same report with a redacted error and nothing written."""
    return replace(report, written=False, error=error)


def _committable_warnings(target: Path) -> list[str]:
    """Warn when the sidecar would land in a git worktree that does not ignore it.

    The sidecar is the one file this project writes that holds both names *and* talkers, so the
    consequence of a mistyped ``--out`` is a commit of real identity. The check is
    ``export_account.py``'s, reused rather than reimplemented so the two cannot disagree about when
    an export is committable.
    """
    try:
        from export_account import _warn_if_exports_are_committable
    except ImportError:  # pragma: no cover - the module is a sibling script, always importable here
        return []
    try:
        return _warn_if_exports_are_committable(target)
    except Exception:  # noqa: BLE001 - a warning must never fail the run it is warning about
        return []


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve human-readable conversation names for an account export tree and write "
            f"{CONVERSATION_LABEL_FILENAME} next to it. Prints coverage, never names."
        )
    )
    parser.add_argument(
        "--account",
        type=Path,
        default=DEFAULT_ACCOUNT_DIR,
        help=f"the account export tree written by export_account.py (default: {DEFAULT_ACCOUNT_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "where to write the sidecar (default: <account>/" + CONVERSATION_LABEL_FILENAME + "). "
            "It holds real names and talkers, so keep it inside a git-ignored directory."
        ),
    )
    args = parser.parse_args(argv)

    target = args.out if args.out is not None else sidecar_path_for(args.account)
    for line in _committable_warnings(target):
        print(line, file=sys.stderr)

    report = sync_labels(args.account, out=args.out)
    for line in report.lines():
        print(f"  {line}")

    if report.error:
        print(f"\nerror: {report.error}", file=sys.stderr)
        return 2
    if not report.written:
        return 2

    print()
    if report.resolved:
        print(f"next: python web.py --account {args.account}")
        print("      the evidence cards will carry the conversation name")
    else:
        print("note: the sidecar is written and empty of names; the UI shows no conversation header")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
