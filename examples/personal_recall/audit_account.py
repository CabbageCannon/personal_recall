"""Audit a full account export tree: what is actually in it, and does it stay separated.

Phase 19 proved the *rules* on fixtures and on a bounded real sample (8 conversations). This is the
instrument for the full-account question: with the whole account exported, how many conversations and
messages are really there, does every detected shard have an export, and — the phase's hard rule — does
any retrieval unit hold events from two conversations.

::

    python audit_account.py --account data/real/account_full --shard-dir "<Msg/Multi>"

It reads an existing tree and **writes nothing** except an optional redacted JSON report. It does not
export, does not embed, and does not build an index — those are separate steps, so a failure here is
unambiguously an ingestion failure rather than a model or memory problem.

## What it prints, and what it must never print

Counts, shard names, date ranges and short digests. **Never** a wxid, a display name, a group name or a
line of message text. A per-conversation table is useful for spotting an outlier (one conversation
holding a third of the account), so it is included — keyed by an anonymous digest and showing only
message counts and a span. The audit log is the artifact most likely to be pasted into an issue, so it
is built to be safe to paste.

## Two independent questions

1. **Is the export complete?** Detected shards versus exported shards, and whether a conversation
   filter was used. A shard that exists on disk but produced no export is invisible history, and so is
   a conversation that was never selected — `partial` reports both, with the reason.
2. **Did ingestion stay separated?** `crossed_conversation_chunks` recomputes chunk membership from
   event ids. Any non-zero result is a blocking bug, not a caveat: it means one retrieval unit mixes
   two people's conversations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from memory import (  # noqa: E402
    EXPORT_FILENAME_SUFFIX,
    SESSION_LISTING_FILENAME,
    SessionConfig,
    build_account_sessions,
    conversation_id_from_export_filename,
    crossed_conversation_chunks,
    import_account,
    load_account_directory,
)

DEFAULT_ACCOUNT = BASE_DIR / "data" / "real" / "account_full"

#: Short digest length. Stable enough to follow one conversation across two tables, short enough that
#: it is not an identifier.
DIGEST_CHARS = 10

#: How many conversations the outlier table lists. The tail is not informative at a glance.
TOP_CONVERSATIONS = 10


def anon(value: str) -> str:
    """A stable short digest of an identifier: safe to print, useless as an identifier."""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:DIGEST_CHARS]


def count_message_rows(payload: Any) -> int:
    """How many message objects a WeFlow export holds, across the shapes it can take.

    The exporter has more than one envelope — a bare list, or an object carrying ``messages`` — and an
    audit that silently counted zero for one of them would report an empty conversation as a fact
    rather than as a parsing gap.
    """
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("messages", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
    return 0


def tree_totals(account_dir: Path) -> dict[str, Any]:
    """File-level facts about the tree, independent of the parser.

    Kept separate from the import on purpose: if the parser and the tree disagree, the disagreement is
    the finding, and it is only visible when the two are counted by different code.
    """
    files = 0
    rows = 0
    listings = 0
    listed_entries = 0
    malformed: list[str] = []

    for shard_dir in sorted(p for p in Path(account_dir).iterdir() if p.is_dir()):
        listing_path = shard_dir / SESSION_LISTING_FILENAME
        if listing_path.is_file():
            listings += 1
            try:
                payload = json.loads(listing_path.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, OSError):
                malformed.append(f"{shard_dir.name}/{SESSION_LISTING_FILENAME}")
            else:
                listed_entries += len(payload) if isinstance(payload, list) else 0

        for export_path in sorted(shard_dir.glob(f"*{EXPORT_FILENAME_SUFFIX}")):
            files += 1
            try:
                payload = json.loads(export_path.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, OSError):
                # Digest the *conversation*, not the file name: an unreadable export must be
                # recognisable as the same conversation the outlier table lists, which is the whole
                # point of a stable digest. Hashing the stem would include the "_messages" suffix and
                # give one conversation two different labels.
                talker = conversation_id_from_export_filename(export_path.name) or export_path.stem
                malformed.append(f"{shard_dir.name}/{anon(talker)}")
                continue
            rows += count_message_rows(payload)

    return {
        "conversation_files": files,
        "message_rows_in_files": rows,
        "session_listings": listings,
        "entries_in_listings": listed_entries,
        "unreadable_files": malformed,
    }


def explain_missing_conversations(
    account_dir: Path,
    conversation_ids: Sequence[str],
) -> dict[str, int]:
    """Why discovered conversations produced no messages, by cause.

    "N discovered conversations produced no messages" is a true statement that explains nothing. The
    causes have different next actions — a file that is missing is an export failure, a file that is
    present and empty is a real empty conversation, and a file with rows that all parsed away is a
    schema problem — so they are counted separately rather than summed into one number.
    """
    causes: Counter[str] = Counter()
    for conversation_id in conversation_ids:
        # Iterate rather than glob: a conversation id is untrusted input here, and a glob would treat
        # any bracket in it as a character class.
        found = [
            path
            for shard_dir in sorted(p for p in Path(account_dir).iterdir() if p.is_dir())
            for path in [shard_dir / f"{conversation_id}{EXPORT_FILENAME_SUFFIX}"]
            if path.is_file()
        ]
        if not found:
            causes["no export file at all"] += 1
            continue

        rows = 0
        unreadable = False
        for path in found:
            try:
                rows += count_message_rows(json.loads(path.read_text(encoding="utf-8-sig")))
            except (json.JSONDecodeError, OSError):
                unreadable = True
                break
        if unreadable:
            causes["export file unreadable"] += 1
        elif rows == 0:
            causes["file present but holding 0 messages"] += 1
        else:
            causes["rows present but all skipped by the parser"] += 1
    return dict(causes)


def build_report(account_dir: Path, shard_dir: Path | None) -> dict[str, Any]:
    """Everything the audit knows, in one redacted structure."""
    layout = load_account_directory(account_dir, shard_dir=shard_dir)
    events_by_conversation, report = import_account(
        layout.exports,
        shards_detected=layout.shards_detected,
        discovered=layout.descriptors,
        filtered_conversations=layout.filtered_conversations,
    )
    report.notes = report.notes + layout.notes

    chunks = build_account_sessions(events_by_conversation, SessionConfig(max_chars=900))
    crossed = crossed_conversation_chunks(chunks, events_by_conversation)

    sizes = sorted((len(rows) for rows in events_by_conversation.values()), reverse=True)
    per_conversation = sorted(
        (
            {
                "digest": anon(row.conversation_id),
                "messages": row.messages,
                "shards": len(row.shards),
                "first": row.first_timestamp.isoformat(sep=" ") if row.first_timestamp else None,
                "last": row.last_timestamp.isoformat(sep=" ") if row.last_timestamp else None,
            }
            for row in report.per_conversation
        ),
        key=lambda row: row["messages"],
        reverse=True,
    )

    return {
        "account_dir": str(account_dir),
        "shards_detected": list(report.shards_detected),
        "shards_exported": list(report.shards_exported),
        "missing_shards": list(report.missing_shards),
        "filtered_conversations": report.filtered_conversations,
        "partial": report.partial,
        "conversations_discovered": report.conversations_discovered,
        "conversations_imported": report.conversations_imported,
        "conversations_without_messages": len(report.conversations_without_messages),
        "missing_causes": explain_missing_conversations(
            account_dir, report.conversations_without_messages[:50]
        ),
        "messages_kept": report.messages_kept,
        "messages_received": report.messages_received,
        "duplicates_removed": report.duplicates_removed,
        "undedupeable": report.undedupeable,
        "skipped_messages": report.skipped_messages,
        "coverage": {
            "first": report.first_timestamp.isoformat(sep=" ") if report.first_timestamp else None,
            "last": report.last_timestamp.isoformat(sep=" ") if report.last_timestamp else None,
        },
        "cross_shard_conversations": sum(1 for row in report.per_conversation if len(row.shards) > 1),
        "chunks": len(chunks),
        "crossed_chunks": list(crossed),
        "largest_conversations": sizes[:TOP_CONVERSATIONS],
        "largest_share_of_total": (sizes[0] / report.messages_kept) if sizes and report.messages_kept else 0.0,
        "per_conversation": per_conversation[:TOP_CONVERSATIONS],
        "tree": tree_totals(account_dir),
        "notes": list(report.notes),
    }


def render(report: dict[str, Any]) -> str:
    """The human-readable audit. Counts and digests only — never an identity."""
    out: list[str] = []
    add = out.append

    add("== export completeness ==")
    add(f"  shards detected        : {len(report['shards_detected'])} {report['shards_detected']}")
    add(f"  shards exported        : {len(report['shards_exported'])} {report['shards_exported']}")
    add(f"  missing shards         : {report['missing_shards'] or 'none'}")
    add(f"  filtered conversations : {report['filtered_conversations'] or 0}")
    add(f"  PARTIAL                : {report['partial']}")

    add("== tree (counted from files, not from the parser) ==")
    tree = report["tree"]
    add(f"  conversation files     : {tree['conversation_files']}")
    add(f"  message rows in files  : {tree['message_rows_in_files']}")
    add(f"  session listings       : {tree['session_listings']} "
        f"({tree['entries_in_listings']} listing entries)")
    if tree["unreadable_files"]:
        add(f"  UNREADABLE FILES       : {len(tree['unreadable_files'])} {tree['unreadable_files'][:5]}")

    add("== import ==")
    add(f"  conversations found    : {report['conversations_discovered']}")
    add(f"  conversations imported : {report['conversations_imported']}")
    add(f"  no messages from       : {report['conversations_without_messages']}")
    for cause, count in sorted(report["missing_causes"].items()):
        add(f"      {count:5}  {cause}")
    add(f"  messages               : {report['messages_kept']} kept of {report['messages_received']}")
    add(f"  duplicates removed     : {report['duplicates_removed']}")
    add(f"  without a serverId     : {report['undedupeable']} (kept, not deduplicated)")
    add(f"  skipped (unparseable)  : {report['skipped_messages']}")
    add(f"  coverage               : {report['coverage']['first']} .. {report['coverage']['last']}")

    add("== sessions ==")
    add(f"  chunks                 : {report['chunks']}")
    add(f"  conversations spanning more than one shard: {report['cross_shard_conversations']}")
    add(f"  largest conversation   : {report['largest_conversations'][0] if report['largest_conversations'] else 0} "
        f"messages ({report['largest_share_of_total'] * 100:.1f}% of the account)")

    add("== conversation boundary ==")
    if report["crossed_chunks"]:
        add(f"  FAIL - {len(report['crossed_chunks'])} chunk(s) mix two conversations: "
            f"{report['crossed_chunks'][:5]}")
    else:
        add("  PASS - no chunk holds events from two conversations")

    if report["notes"]:
        add("== notes ==")
        for note in report["notes"]:
            add(f"  {note}")

    add("== largest conversations (anonymous digests) ==")
    for row in report["per_conversation"]:
        add(f"  {row['digest']}  {row['messages']:7} msgs  shards={row['shards']}  "
            f"{row['first']} .. {row['last']}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a full account export tree. Prints counts and digests, never chat."
    )
    parser.add_argument("--account", type=Path, default=DEFAULT_ACCOUNT,
                        help=f"the export tree to audit (default: {DEFAULT_ACCOUNT})")
    parser.add_argument("--shard-dir", type=Path, default=None,
                        help="the account's Msg/Multi directory, so a missing shard can be detected")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the redacted report here (counts and digests only)")
    args = parser.parse_args()

    if not args.account.is_dir():
        print(f"error: --account {args.account} is not a directory.", file=sys.stderr)
        return 2

    shard_dir = args.shard_dir
    if shard_dir is None:
        print(
            "note: no --shard-dir, so completeness is checked against the export directories "
            "themselves. A shard that was never exported cannot be detected that way - which is the "
            "one failure this check exists to find. Pass the account's Msg/Multi directory.",
            file=sys.stderr,
        )

    report = build_report(args.account, shard_dir)
    print(render(report))

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nredacted report written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
