"""Export an entire WeChat account, one conversation per shard, into the tree recall reads.

This is the **only** step that touches the real account, and it is deliberately the smallest one it can
be. It discovers the account's message shards, asks ``weflow-cli`` for each shard's conversation list,
exports every conversation of every shard into ``<out>/<shard>/<talker>_messages.json``, and records
what it did in ``<out>/shard_manifest.json``. It does not decrypt anything itself, does not patch the
exporter, and does not parse a single message — parsing starts at ``memory.weflow``.

::

    python export_account.py --multi-dir "C:\\Users\\me\\WeChat Files\\wxid_x\\Msg\\Multi" --dry-run
    python export_account.py --multi-dir "C:\\Users\\me\\WeChat Files\\wxid_x\\Msg\\Multi" \\
        --out data/real/account
    python recall.py "我之前说的那个数据库最后到底用了没？" --account data/real/account

Privacy rules this command follows (see ``exporter.py`` for the mechanism):

* The user's real ``~/.weflow-cli`` is **read once and never written**. A scratch profile is created in
  the temp directory, used, and deleted on every exit path, including a crash inside the exporter.
* ``--dry-run`` writes nothing at all and is the recommended first step.
* **The log contains counts, shard names and redacted errors — never a wxid, a display name, a message
  body or key material.** The exported files do contain them, which is why ``--out`` must be somewhere
  git ignores (``data/real/`` by convention) and why this command warns when it is not.

One caveat this command cannot fix, and therefore states instead of hiding: WeChat writes recent
messages to a shard's write-ahead log (``MSG*.db-wal``) before merging it into the database, and
``weflow-cli`` reads the database. Messages still only in the WAL are *not* exported, so a freshly
received conversation can be missing from a complete-looking export. The run is therefore a reliable
view of committed history, not a guarantee of the newest messages.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from exporter import (  # noqa: E402
    AccountExportReport,
    ExporterError,
    export_account_tree,
    list_conversations,
)
from memory.conversations import shard_stem  # noqa: E402
from memory.shards import discover_message_shards  # noqa: E402

#: Where the export tree goes by default. Inside ``data/real/``, which ``.gitignore`` excludes — real
#: chat must never be one ``git add -A`` away from the repository.
DEFAULT_OUT = BASE_DIR / "data" / "real" / "account"

WAL_CAVEAT = (
    "recent messages still sitting in a shard's write-ahead log (MSG*.db-wal) are not visible to the "
    "exporter, so this export is committed history, not a guarantee of the newest messages"
)


def _repo_root() -> Path | None:
    """The enclosing git worktree, or ``None`` when this is not a checkout (or git is absent)."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    root = (completed.stdout or "").strip()
    return Path(root) if root else None


def _is_git_ignored(path: Path) -> bool | None:
    """``True``/``False`` for a path inside a worktree, ``None`` when git cannot answer."""
    try:
        completed = subprocess.run(
            ["git", "check-ignore", "-q", "--", str(path)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    return None


def _warn_if_exports_are_committable(out_dir: Path) -> list[str]:
    """Warn when the export tree sits inside the repository and git does not ignore it.

    A missing warning is not a safety guarantee — the check is best-effort, and it can only see a git
    worktree. It exists because the failure it prevents is irreversible: one ``git add -A`` on a
    directory full of real conversations.
    """
    root = _repo_root()
    if root is None:
        return []
    try:
        resolved = out_dir.resolve()
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return []  # outside the worktree: not our business, and git would not add it anyway
    ignored = _is_git_ignored(out_dir)
    if ignored:
        return []
    return [
        f"WARNING: {out_dir} is inside the git worktree and is NOT ignored, so `git add -A` would "
        f"commit real chat. Use --out {DEFAULT_OUT} (ignored via data/real/) or add the directory "
        "to .gitignore."
    ]


def _dry_run(multi_dir: Path, shards: tuple[str, ...], only: tuple[str, ...]) -> int:
    """Discovery only: how many shards and conversations exist. Writes nothing.

    Counts are printed, ids are not. A dry run that printed wxids would put real identifiers in a
    terminal log, which is exactly what the privacy rules forbid.
    """
    detected = tuple(shard_stem(name) for name in discover_message_shards(multi_dir))
    if not detected:
        print(
            f"error: no MSG*.db message shards found in {multi_dir}.\n"
            "Point --multi-dir at the account's Msg/Multi directory, for example\n"
            r'  "C:\Users\<you>\WeChat Files\<wxid>\Msg\Multi"',
            file=sys.stderr,
        )
        return 2

    print(f"multi-dir      : {multi_dir}")
    print(f"message shards : {len(detected)} detected {list(detected)}")
    print(f"target shards  : {list(shards) if shards else list(detected)}")
    if only:
        print(f"conversation filter: {len(only)} id(s) given (ids not echoed)")
    print()

    total = 0
    failures = 0
    for shard in shards or detected:
        if shard not in detected:
            print(f"  {shard:8} not present in the multi-dir")
            failures += 1
            continue
        try:
            conversations = list_conversations(shard, multi_dir)
        except ExporterError as exc:
            print(f"  {shard:8} could not be listed: {exc}")
            failures += 1
            continue
        wanted = [d for d in conversations if not only or d.conversation_id in set(only)]
        total += len(wanted)
        print(f"  {shard:8} {len(wanted)} conversation(s) to export")
    print()
    print(f"dry run        : nothing written; {total} file(s) would be produced")
    if failures:
        print(f"                 {failures} shard(s) could not be listed")
    print(f"note           : {WAL_CAVEAT}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export every conversation of a WeChat account for Personal Recall."
    )
    parser.add_argument(
        "--multi-dir",
        type=Path,
        required=True,
        help="the account's Msg/Multi directory (holds MSG0.db, MSG1.db, ...); read-only, never decrypted here",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"where to write the export tree (default: {DEFAULT_OUT}; real chat, keep it git-ignored)",
    )
    parser.add_argument(
        "--shards",
        nargs="*",
        default=None,
        help="optional subset of shard stems to export (default: every MSG*.db detected)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="export only this conversation id (talker); repeatable. Ids are matched exactly and are "
        "never treated as display names.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="discover shards and count conversations, write nothing",
    )
    args = parser.parse_args()

    multi_dir = args.multi_dir
    if not multi_dir.is_dir():
        print(f"error: --multi-dir {multi_dir} is not a directory.", file=sys.stderr)
        return 2

    shards = tuple(shard_stem(name) for name in args.shards) if args.shards else ()
    only = tuple(args.only or ())

    if args.dry_run:
        return _dry_run(multi_dir, shards, only)

    for line in _warn_if_exports_are_committable(args.out):
        print(line, file=sys.stderr)

    try:
        report: AccountExportReport = export_account_tree(
            args.out, multi_dir=multi_dir, shards=shards or None, only=only or None
        )
    except ExporterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for line in report.lines():
        print(f"  {line}")
    print()
    print(f"note           : {WAL_CAVEAT}")
    print()
    print("next:")
    print(f'  python recall.py "你的问题" --account {args.out}')

    if not report.files_written:
        print(
            "\nerror: 0 conversation exports were written, so there is nothing to recall. "
            "Check the failures above.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
