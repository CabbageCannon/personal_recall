"""Build ``data/stress_chats_v2.txt`` from corpus v1 plus the distractor pack.

Corpus v2 keeps every one of v1's gold evidence lines byte-identical and surrounds
them with same-topic, different-fact sessions, so the *same 36 queries* become hard
again (v1 saturates the retriever at Hit@10 = 100 %).

The merge is episode-level: v1 and the pack occupy disjoint days, so sorting episodes
by their first timestamp yields one globally chronological file with no interleaving
inside an episode. Order matters because ``build_sessions`` consumes file order
without re-sorting — an out-of-order corpus fragments into one message per chunk.

Verification is strict; nothing is written unless every check passes:

* every line matches ``[YYYY-MM-DD HH:MM] speaker: text``
* timestamps are globally unique and strictly ascending
* all gold evidence lines survive the merge, each exactly once
* no pack message restates a gold evidence line
* session chunking produces a sane chunk profile (no fragmentation)

Usage::

    python build_stress_v2.py                     # build into data/
    python build_stress_v2.py --check-only        # verify without writing
    python build_stress_v2.py --parts-dir ...     # alternate pack location
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

V1_NAME = "stress_chats.txt"
QUERIES_NAME = "stress_queries.json"
OUT_NAME = "stress_chats_v2.txt"
DEFAULT_PARTS_DIR = DATA_DIR / "stress_v2_parts"

PART_NAMES = ("part_a_2024.txt", "part_b_2025.txt", "part_c_2026.txt")

TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] ([\w\u4e00-\u9fff]+): (.+)$")

sys.path.insert(0, str(BASE_DIR))
from memory.events import parse_txt_events  # noqa: E402
from memory.sessions import SessionConfig, build_sessions  # noqa: E402


def normalise(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def split_episodes(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", normalise(text)) if block.strip()]


def load_gold() -> list[str]:
    queries = json.loads((DATA_DIR / QUERIES_NAME).read_text(encoding="utf-8"))
    gold = [line.strip() for q in queries for line in (q.get("relevant_evidence") or [])]
    return gold


def gold_contents(gold: list[str]) -> set[str]:
    out: set[str] = set()
    for line in gold:
        match = TS_RE.match(line)
        out.add(match.group(3).strip() if match else line)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS_DIR)
    parser.add_argument("--out", type=Path, default=DATA_DIR / OUT_NAME)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
        if not ok:
            failures.append(label)

    sources = [DATA_DIR / V1_NAME] + [args.parts_dir / name for name in PART_NAMES]
    missing = [str(p) for p in sources if not p.exists()]
    if missing:
        print("missing source files:")
        for path in missing:
            print(f"  {path}")
        return 1

    print("sources:")
    episodes: list[tuple[str, str, bool]] = []
    v1_episode_count = 0
    for index, path in enumerate(sources):
        blocks = split_episodes(path.read_text(encoding="utf-8"))
        if index == 0:
            v1_episode_count = len(blocks)
        for block in blocks:
            first = block.split("\n", 1)[0]
            match = TS_RE.match(first)
            if match is None:
                failures.append(f"{path.name}: episode starts with a malformed line: {first[:60]!r}")
                continue
            episodes.append((match.group(1), block, index > 0))
        print(f"  {path.name:22s} episodes={len(blocks):4d}")

    if failures:
        print("\nFAILED to parse sources")
        return 1

    episodes.sort(key=lambda item: item[0])
    merged = "\n\n".join(block for _, block, _ in episodes) + "\n"

    v1_text = normalise((DATA_DIR / V1_NAME).read_text(encoding="utf-8"))
    v1_line_count = sum(1 for line in v1_text.split("\n") if TS_RE.match(line))
    pack_text = "\n\n".join(block for _, block, is_pack in episodes if is_pack)

    print("\nmerged corpus:")
    lines = [line for line in merged.split("\n") if line.strip()]
    stamps = [TS_RE.match(line).group(1) for line in lines]

    check(len(lines) > v1_line_count, "messages added", f"{v1_line_count} -> {len(lines)}")
    check(len(stamps) == len(set(stamps)), "timestamps globally unique")
    check(stamps == sorted(stamps), "file is chronologically ordered")
    check(
        all(TS_RE.match(line) for line in lines),
        "every line matches the message format",
    )

    gold = load_gold()
    absent = [line for line in gold if line not in merged]
    check(not absent, "all gold evidence lines survive", f"{len(gold) - len(absent)}/{len(gold)}")
    check(
        all(merged.count(line) == 1 for line in gold),
        "every gold line appears exactly once",
    )

    pack_contents = {
        TS_RE.match(line).group(3).strip()
        for line in pack_text.split("\n")
        if TS_RE.match(line)
    }
    restated = pack_contents & gold_contents(gold)
    check(not restated, "pack never restates a gold line", f"{len(restated)} collisions")

    events = parse_txt_events(merged).events
    chunks = build_sessions(events, SessionConfig())
    lens = sorted(chunk.n_chars for chunk in chunks)
    median = lens[len(lens) // 2]
    per_chunk = len(events) / len(chunks) if chunks else 0
    check(
        per_chunk >= 5 and median >= 200,
        "session chunking not fragmented",
        f"{len(chunks)} chunks, {per_chunk:.1f} msgs/chunk, median {median} chars",
    )

    reachable = sum(1 for line in gold if any(line in chunk.text for chunk in chunks))
    check(reachable == len(gold), "every gold line lands in a chunk", f"{reachable}/{len(gold)}")

    print("\n  corpus v2 : "
          f"{len(merged)} chars, {len(lines)} messages, {len(episodes)} episodes, {len(chunks)} chunks")
    print("  corpus v1 : "
          f"{len(v1_text)} chars, {v1_line_count} messages, {v1_episode_count} episodes")

    if failures:
        print(f"\nFAILED: {len(failures)} check(s): {failures}")
        return 1

    if args.check_only:
        print("\n--check-only: nothing written")
        return 0

    args.out.write_text(merged, encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
