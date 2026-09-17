"""Real-account smoke: does an entire WeChat account reach one index, still separated?

Synthetic tests prove the *rules*. They cannot prove that a real ``Msg/Multi`` directory holds what
the rules assume — that shards really do overlap, that a talker really is stable across them, that the
export contract still matches. This command answers that, on this machine, read-only.

::

    python smoke_account.py --multi-dir "C:\\Users\\me\\WeChat Files\\wxid_x\\Msg\\Multi" --limit 12

It exports a **bounded** subset (a whole account is hundreds of conversations and gigabytes of
history), imports it, builds the real index, and checks the phase's hard rule on real data:
no retrieval unit may hold events from two conversations.

## What it prints, and what it must never print

The output is counts, shard names, date ranges and short digests. **Never** a wxid, a display name, a
group name, or a character of message text — not even on failure, which is why every reportable
identity goes through :func:`anon` first and errors go through ``exporter.redact``. A smoke log is the
artifact most likely to be pasted into a chat window, so it is built to be safe to paste.

The export itself *is* real chat and lands in ``data/real/smoke`` (git-ignored). ``--keep`` leaves it
there for a later ``recall.py --account``; without it the tree is removed on the way out, including
after a failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from exporter import ExporterError, export_account_tree, list_conversations  # noqa: E402
from memory import (  # noqa: E402
    ConversationDescriptor,
    SessionConfig,
    build_account_sessions,
    crossed_conversation_chunks,
    import_account,
    load_account_directory,
    shard_stem,
)
from memory.shards import discover_message_shards  # noqa: E402

DEFAULT_OUT = BASE_DIR / "data" / "real" / "smoke"

#: Deliberately short. Smoke output is meant to be read by a human at a glance, not diffed.
DIGEST_CHARS = 10


def anon(value: str) -> str:
    """A stable short digest of an identifier, safe to print and useless as an identifier.

    Stable so two lines about the same conversation can be recognised as the same one; short so it
    cannot be brute-forced back to a wxid.
    """
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:DIGEST_CHARS]


def pick_conversations(
    shards: list[str],
    multi_dir: Path,
    *,
    limit: int,
) -> tuple[list[ConversationDescriptor], list[tuple[str, ConversationDescriptor]]]:
    """List every shard and choose a bounded subset, preferring conversations seen in several shards.

    A multi-shard conversation is the interesting case — it is the one that proves the per-conversation
    merge works on real data rather than only on a fixture — so those are taken first, and the rest of
    the budget is filled with single-shard conversations so the run also sees conversations that are
    NOT merged.
    """
    per_shard: dict[str, list[ConversationDescriptor]] = {}
    for shard in shards:
        per_shard[shard] = list(list_conversations(shard, multi_dir))

    seen_in: dict[str, list[str]] = {}
    labels: dict[str, ConversationDescriptor] = {}
    for shard in shards:
        for descriptor in per_shard[shard]:
            seen_in.setdefault(descriptor.conversation_id, []).append(shard)
            labels.setdefault(descriptor.conversation_id, descriptor)

    multi = sorted(cid for cid, where in seen_in.items() if len(where) > 1)
    single = sorted(cid for cid, where in seen_in.items() if len(where) == 1)
    direct = sorted(cid for cid in seen_in if labels[cid].conversation_type == "direct")

    # Take multi-shard conversations first, but hold one slot for a single-shard conversation and one
    # for a direct chat when they exist: a sample of nothing but merged groups would leave the other
    # two shapes — the un-merged conversation and the one-to-one chat — unexercised on real data.
    reserved: list[str] = []
    for candidate in (single[0] if single else None, next((c for c in direct if c not in multi), None)):
        if candidate is not None and candidate not in reserved:
            reserved.append(candidate)
    budget = max(limit - len(reserved), 0)
    chosen_ids = [cid for cid in multi[:budget]] + [
        cid for cid in reserved if cid not in multi[:budget]
    ]
    chosen_ids = list(dict.fromkeys(chosen_ids))[:limit]

    chosen = [(cid, labels[cid]) for cid in chosen_ids]
    return [labels[cid] for cid, _ in chosen], chosen


def _report_identity(chosen: list[tuple[str, ConversationDescriptor]]) -> None:
    """One line per selected conversation: an anonymous digest and its type. Nothing else."""
    for conversation_id, descriptor in chosen:
        print(f"    {anon(conversation_id):>{DIGEST_CHARS}}  {descriptor.conversation_type}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only account-wide smoke test. Prints counts and digests, never chat."
    )
    parser.add_argument(
        "--multi-dir",
        type=Path,
        required=True,
        help="the account's Msg/Multi directory (holds MSG0.db, MSG1.db, ...); never modified",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where the bounded export goes")
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="how many conversations to export (default: 12); a whole account is far too slow",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="leave the export tree in place for a later `recall.py --account` (it holds real chat)",
    )
    args = parser.parse_args()

    multi_dir = args.multi_dir
    if not multi_dir.is_dir():
        print(f"error: --multi-dir {multi_dir} is not a directory.", file=sys.stderr)
        return 2

    shards = sorted({shard_stem(name) for name in discover_message_shards(multi_dir)})
    if not shards:
        print(f"error: no MSG*.db message shards found in {multi_dir}", file=sys.stderr)
        return 2

    print("== discovery ==")
    print(f"  message shards detected : {len(shards)} {shards}")

    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True, exist_ok=True)
    ok = False
    try:
        ok = _run(multi_dir, shards, args)
    finally:
        if not args.keep or not ok:
            shutil.rmtree(args.out, ignore_errors=True)
    return 0 if ok else 1


def _run(multi_dir: Path, shards: list[str], args) -> bool:
    try:
        chosen_descriptors, chosen = pick_conversations(shards, multi_dir, limit=args.limit)
    except ExporterError as exc:
        print(f"  listing failed: {exc}", file=sys.stderr)
        return False

    if not chosen:
        print("  no conversations were listed in any shard", file=sys.stderr)
        return False

    print(f"  conversations discovered: {len(chosen)} selected of a bounded sample")
    _report_identity(chosen)

    only = [cid for cid, _ in chosen]
    print("== export ==")
    try:
        export = export_account_tree(args.out, multi_dir=multi_dir, only=only)
    except ExporterError as exc:
        print(f"  export failed: {exc}", file=sys.stderr)
        return False
    for line in export.lines():
        print(f"  {line}")
    if not export.files_written:
        print("  no exports were written, so nothing can be indexed", file=sys.stderr)
        return False

    print("== import (per conversation, then one index) ==")
    layout = load_account_directory(args.out, shard_dir=multi_dir)
    events_by_conversation, report = import_account(
        layout.exports,
        shards_detected=layout.shards_detected,
        discovered=layout.descriptors,
        # This smoke test exports a bounded sample, so the tree is narrowed on purpose; say so rather
        # than let the report call the sample the whole account.
        filtered_conversations=layout.filtered_conversations,
    )
    for line in report.lines():
        print(f"  {line}")

    chunks = build_account_sessions(events_by_conversation, SessionConfig(max_chars=900))
    crossed = crossed_conversation_chunks(chunks, events_by_conversation)
    conversations_in_chunks = sorted({chunk.conversation_id for chunk in chunks})
    print(f"  sessions               : {len(chunks)} chunk(s)")
    print(f"  conversations indexed  : {len(conversations_in_chunks)}")
    print(
        "  boundary check         : "
        + ("OK - no chunk mixes two conversations" if not crossed else f"VIOLATED {crossed}")
    )

    by_conversation = _group(layout.exports)
    multi_shard = sorted(cid for cid, rows in by_conversation.items() if len(rows) > 1)
    print(f"  merged conversations   : {len(multi_shard)} span more than one shard")
    for conversation_id in multi_shard:
        print(f"    {anon(conversation_id)}  shards={len(by_conversation[conversation_id])}")

    if crossed:
        print("\nFAILED: a conversation boundary was crossed on real data.", file=sys.stderr)
        return False

    with open(args.out / "smoke_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "shards_detected": list(report.shards_detected),
                "shards_exported": list(report.shards_exported),
                "partial": report.partial,
                "missing_shards": list(report.missing_shards),
                "conversations_discovered": report.conversations_discovered,
                "conversations_imported": report.conversations_imported,
                "messages_kept": report.messages_kept,
                "duplicates_removed": report.duplicates_removed,
                "conversations_spanning_shards": len(multi_shard),
                "chunks": len(chunks),
                "boundary_violations": list(crossed),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("\nOK: real conversations from multiple shards reached one index, still separated.")
    print("    Anonymised summary written to smoke_summary.json (no ids, no text).")
    return True


def _group(exports) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for export in exports:
        grouped.setdefault(export.conversation_id, []).append(export)
    return grouped


if __name__ == "__main__":
    raise SystemExit(main())
