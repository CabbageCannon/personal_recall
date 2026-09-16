#!/usr/bin/env python3
"""Offline, deterministic validator for the stress chat-history retrieval dataset.

Validates ``data/stress_chats.txt`` (corpus) and ``data/stress_queries.json``
(queries) with eleven hard checks plus non-fatal difficulty diagnostics.

Guarantees:

* pure Python - no model, no network, no API call;
* deterministic - no randomness, no wall-clock value in the report, so the same
  inputs always produce byte-identical JSON;
* fast - load + checks + diagnostics take ~0.09 s on a 98 KB corpus and ~0.26 s
  on a 296 KB one; the project venv's own interpreter start-up (~1.3 s) is the
  only other cost, far inside the 5 s budget.

Usage::

    python validate_stress_dataset.py [--corpus PATH] [--queries PATH] [--json-out PATH]

Every default path resolves against this file's directory, never the CWD.
Exit code is ``0`` only when all hard checks pass, ``1`` otherwise (``2`` for a
broken environment, e.g. a missing ``eval_utils`` helper next to this file).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

# ---------------------------------------------------------------------------
# Paths and thresholds
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = SCRIPT_DIR / "data" / "stress_chats.txt"
DEFAULT_QUERIES_PATH = SCRIPT_DIR / "data" / "stress_queries.json"
DEFAULT_JSON_OUT_PATH = SCRIPT_DIR / "stress_validation.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:  # same-directory helper - reused, never copied
    from eval_utils import normalize_text
except ImportError as exc:  # pragma: no cover - broken checkout only
    sys.stderr.write(
        f"FATAL: cannot import normalize_text from {SCRIPT_DIR / 'eval_utils.py'}: {exc}\n"
    )
    raise SystemExit(2)

REQUIRED_QUERY_KEYS = frozenset(
    {"id", "question", "expected_answer", "relevant_evidence", "category", "answerable"}
)

# Real splitter configuration (core/quivr_core/processor/.../simple_txt_processor.py)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

MIN_QUERIES = 30
MAX_QUERIES = 45
MIN_CORPUS_CHARS = 40_000
MIN_CORPUS_MESSAGES = 700
MIN_DISTINCT_CATEGORIES = 10
MIN_CHUNKS = 140

# Difficulty thresholds (advisory only - never fail the run)
OVERLAP_TOO_HIGH = 0.5
RARE_ANCHOR_TOO_UNIQUE = 1
DISTRACTOR_NGRAM = 4
DISTRACTOR_MIN_SHARED = 3

MAX_OFFENDERS_SHOWN = 10
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M"
TIMESTAMP_PREFIX_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\] ")
MESSAGE_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] "
    r"(?P<speaker>[^:\s][^:]{0,19}): (?P<content>\S.*)$"
)


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def shorten(text: object, limit: int = 70) -> str:
    """Collapse whitespace and cap length so messages stay one line."""
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "\u2026"


def format_offenders(problems: Sequence[str], limit: int = MAX_OFFENDERS_SHOWN) -> str:
    """Join problem strings for display, truncating the tail of long lists."""
    shown = "; ".join(problems[:limit])
    if len(problems) > limit:
        shown += f"; ... and {len(problems) - limit} more"
    return shown


def strip_timestamp(text: str) -> str:
    """Drop a leading ``[YYYY-MM-DD HH:MM] `` prefix, keeping speaker + content."""
    return TIMESTAMP_PREFIX_RE.sub("", text, count=1)


def char_ngrams(text: str, n: int) -> set[str]:
    """Return the set of character ``n``-grams of ``text`` (empty if too short)."""
    if n <= 0 or len(text) < n:
        return set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def query_label(entry: Any, index: int) -> str:
    """Human-readable identifier for a query entry (falls back to its index)."""
    if isinstance(entry, Mapping):
        raw_id = entry.get("id")
        if isinstance(raw_id, str) and raw_id.strip():
            return raw_id
    return f"index {index}"


def iter_query_mappings(queries: Sequence[Any]) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Yield ``(index, entry)`` for every query entry that is a JSON object."""
    for index, entry in enumerate(queries):
        if isinstance(entry, Mapping):
            yield index, entry


# ---------------------------------------------------------------------------
# Splitter replication (same recursion as the real processor)
# ---------------------------------------------------------------------------


def build_chunks(
    text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> tuple[str, ...]:
    """Materialise the chunks ``recursive_character_splitter`` would produce.

    This replays the recursion of
    ``core/quivr_core/processor/implementations/simple_txt_processor.py``
    literally - a document that fits in ``chunk_size`` is one chunk, otherwise
    the first ``chunk_size`` characters are emitted and the recursion continues
    on ``text[chunk_size - chunk_overlap:]`` - using an explicit stack instead
    of a hand-derived closed formula, so it stays correct if the numbers change.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks: list[str] = []
    pending: list[str] = [text]
    while pending:
        current = pending.pop()
        if len(current) <= chunk_size:
            chunks.append(current)
        else:
            chunks.append(current[:chunk_size])
            pending.append(current[chunk_size - chunk_overlap :])
    return tuple(chunks)


def count_chunks_like_splitter(
    text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> int:
    """Count splitter chunks for ``text`` (single source of truth: build_chunks)."""
    return len(build_chunks(text, chunk_size, chunk_overlap))


# ---------------------------------------------------------------------------
# Corpus / query loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusData:
    """Pre-processed corpus, derived once and reused by every check."""

    path: Path
    text: str
    normalized: str
    n_bytes: int
    lines: tuple[str, ...]
    message_lines: tuple[str, ...]
    episodes: tuple[tuple[str, ...], ...]
    chunks: tuple[str, ...]


@dataclass(frozen=True)
class TimestampReport:
    """Result of validating message format and timestamp ordering."""

    first_timestamp: str | None
    last_timestamp: str | None
    parsed_count: int
    format_offenders: tuple[str, ...]
    calendar_offenders: tuple[str, ...]
    order_offenders: tuple[str, ...]
    duplicate_offenders: tuple[str, ...]


def split_episodes(lines: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Split file lines into episodes (blank-line separated blocks)."""
    episodes: list[tuple[str, ...]] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line)
        elif current:
            episodes.append(tuple(current))
            current = []
    if current:
        episodes.append(tuple(current))
    return tuple(episodes)


def build_corpus_data(path: Path, text: str, n_bytes: int) -> CorpusData:
    """Derive every corpus view (lines, episodes, splitter chunks) once."""
    lines = tuple(text.splitlines())
    message_lines = tuple(line for line in lines if line.strip())
    return CorpusData(
        path=path,
        text=text,
        normalized=normalize_text(text),
        n_bytes=n_bytes,
        lines=lines,
        message_lines=message_lines,
        episodes=split_episodes(lines),
        chunks=build_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP),
    )


@dataclass(frozen=True)
class TextFile:
    """Decoded UTF-8 text file plus its on-disk size and decoding notes."""

    text: str
    n_bytes: int
    warnings: tuple[str, ...] = ()


def read_utf8_text(path: Path, label: str) -> tuple[TextFile | None, str | None]:
    """Read a file as UTF-8 text; return ``(file, error_message)``.

    A leading UTF-8 BOM is tolerated (stripped, reported as a warning) because
    the dataset is often written by Windows tooling that adds one.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, f"{label} unreadable: {path} ({exc})"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, f"{label} is not valid UTF-8: {path} ({exc})"
    warnings: list[str] = []
    if text.startswith("\ufeff"):
        text = text[1:]
        warnings.append(f"{label} started with a UTF-8 BOM; it was stripped for parsing")
    if "\r" in text:
        # The baseline processor opens the file in TEXT mode (universal newlines),
        # so it never sees a carriage return. Normalise here as well, otherwise a
        # CRLF file is measured 1 char per line too long and the chunk estimate
        # (and therefore the chunk_count check) drifts from the real chunker.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        warnings.append(
            f"{label} used CR/CRLF line endings; normalised to LF, matching "
            f"quivr_core's text-mode reader"
        )
    return TextFile(text=text, n_bytes=len(raw), warnings=tuple(warnings)), None


def load_corpus(path: Path) -> tuple[CorpusData | None, str | None, list[str]]:
    """Read the corpus; return ``(data, error_message, warnings)``."""
    if not path.is_file():
        return None, f"corpus file not found: {path}", []
    decoded, error = read_utf8_text(path, "corpus")
    if decoded is None or error is not None:
        return None, error, []
    if not decoded.text.strip():
        return None, f"corpus contains no text: {path}", []
    data = build_corpus_data(path, decoded.text, decoded.n_bytes)
    return data, None, list(decoded.warnings)


def load_queries(path: Path) -> tuple[list[Any] | None, str | None, list[str]]:
    """Read the query file; return ``(entries, error_message, warnings)``."""
    if not path.is_file():
        return None, f"queries file not found: {path}", []
    decoded, error = read_utf8_text(path, "queries")
    if decoded is None or error is not None:
        return None, error, []
    try:
        payload = json.loads(decoded.text)
    except json.JSONDecodeError as exc:
        return None, f"queries file is not valid JSON: {path} ({exc})", list(decoded.warnings)
    if not isinstance(payload, list):
        return (
            None,
            f"queries JSON must be an array, got {type(payload).__name__}: {path}",
            list(decoded.warnings),
        )
    return payload, None, list(decoded.warnings)


def inspect_timestamps(corpus: CorpusData) -> TimestampReport:
    """Validate message format, calendar validity and timestamp ordering."""
    format_offenders: list[str] = []
    calendar_offenders: list[str] = []
    parsed: list[tuple[int, str, datetime]] = []
    for line_no, line in enumerate(corpus.lines, start=1):
        if not line.strip():
            continue
        match = MESSAGE_RE.match(line)
        if match is None:
            format_offenders.append(f"line {line_no}: {shorten(line)!r}")
            continue
        stamp = match.group("ts")
        try:
            moment = datetime.strptime(stamp, TIMESTAMP_FORMAT)
        except ValueError:
            calendar_offenders.append(f"line {line_no}: [{stamp}] is not a real calendar time")
            continue
        parsed.append((line_no, stamp, moment))

    order_offenders: list[str] = []
    previous: datetime | None = None
    for line_no, stamp, moment in parsed:
        if previous is not None and moment < previous:
            order_offenders.append(f"line {line_no}: [{stamp}] goes backwards in file order")
        previous = moment

    counts = Counter(stamp for _, stamp, _ in parsed)
    duplicate_offenders = [
        f"[{stamp}] appears {n} times" for stamp, n in counts.items() if n > 1
    ]

    return TimestampReport(
        first_timestamp=parsed[0][1] if parsed else None,
        last_timestamp=parsed[-1][1] if parsed else None,
        parsed_count=len(parsed),
        format_offenders=tuple(format_offenders),
        calendar_offenders=tuple(calendar_offenders),
        order_offenders=tuple(order_offenders),
        duplicate_offenders=tuple(duplicate_offenders),
    )


def summarize_queries(queries: Sequence[Any]) -> dict[str, Any]:
    """Count queries, the answerable split, and the category distribution."""
    answerable = 0
    unanswerable = 0
    categories: Counter[str] = Counter()
    category_defects: list[str] = []
    for index, entry in iter_query_mappings(queries):
        label = query_label(entry, index)
        flag = entry.get("answerable")
        if flag is True:
            answerable += 1
        elif flag is False:
            unanswerable += 1
        category = entry.get("category")
        if isinstance(category, str) and category.strip():
            categories[category] += 1
        else:
            category_defects.append(f"{label}: category={category!r} is not a non-empty string")
    ordered = dict(sorted(categories.items(), key=lambda item: (-item[1], item[0])))
    return {
        "count": len(queries),
        "answerable": answerable,
        "unanswerable": unanswerable,
        "categories": ordered,
        "category_defects": category_defects,
    }


# ---------------------------------------------------------------------------
# Hard checks
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Outcome of one hard check."""

    name: str
    passed: bool
    message: str
    details: tuple[str, ...] = ()
    warning: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialise to ``{"status", "message", ...}`` for the JSON report."""
        payload: dict[str, Any] = {
            "status": "pass" if self.passed else "fail",
            "message": self.message,
        }
        if self.details:
            payload["details"] = list(self.details)
        if self.warning:
            payload["warning"] = self.warning
        return payload


def skipped(check_name: str, reason: str) -> CheckResult:
    """Check that could not run because an input failed to load."""
    return CheckResult(check_name, False, f"not evaluated - {reason}")


def run_or_skip(
    check_name: str, check: Callable[..., CheckResult], gap: str | None, *args: Any
) -> CheckResult:
    """Run ``check`` unless an input is missing (``gap``), in which case skip it."""
    if gap is not None:
        return skipped(check_name, gap)
    return check(*args)


def check_files_exist_and_parse(
    corpus_error: str | None,
    queries_error: str | None,
    corpus_path: Path,
    queries_path: Path,
) -> CheckResult:
    """Check 1: both files exist and parse."""
    problems = [p for p in (corpus_error, queries_error) if p]
    if problems:
        return CheckResult(
            "files_exist_and_parse", False, format_offenders(problems), tuple(problems)
        )
    return CheckResult(
        "files_exist_and_parse",
        True,
        f"corpus {corpus_path.name} parsed as UTF-8 text; "
        f"queries {queries_path.name} parsed as a JSON array",
    )


def check_query_ids(queries: Sequence[Any]) -> CheckResult:
    """Check 2: query ids are unique and non-empty."""
    problems: list[str] = []
    ids: list[str] = []
    for index, entry in iter_query_mappings(queries):
        raw_id = entry.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            problems.append(f"index {index}: id={raw_id!r} is not a non-empty string")
            continue
        ids.append(raw_id)
    for value, count in Counter(ids).items():
        if count > 1:
            problems.append(f"duplicate id {value!r} appears {count} times")
    if problems:
        return CheckResult(
            "query_ids_unique_non_empty",
            False,
            f"{len(problems)} id problem(s) among {len(queries)} entries: {format_offenders(problems)}",
            tuple(problems),
        )
    return CheckResult(
        "query_ids_unique_non_empty",
        True,
        f"{len(ids)} unique non-empty ids",
    )


def check_query_keys(queries: Sequence[Any]) -> CheckResult:
    """Check 3: every query has exactly the six required keys."""
    problems: list[str] = []
    for index, entry in enumerate(queries):
        if not isinstance(entry, Mapping):
            problems.append(f"index {index}: entry is a {type(entry).__name__}, not a JSON object")
            continue
        label = query_label(entry, index)
        keys = set(entry.keys())
        missing = sorted(REQUIRED_QUERY_KEYS - keys)
        extra = sorted(keys - REQUIRED_QUERY_KEYS)
        if missing:
            problems.append(f"{label}: missing key(s) {', '.join(missing)}")
        if extra:
            problems.append(f"{label}: unexpected extra key(s) {', '.join(extra)}")
    if problems:
        return CheckResult(
            "query_keys_exact",
            False,
            f"{len(problems)} key problem(s): {format_offenders(problems)}",
            tuple(problems),
        )
    return CheckResult(
        "query_keys_exact",
        True,
        f"all {len(queries)} queries have exactly the 6 required keys",
    )


def check_answerable_constraints(queries: Sequence[Any]) -> CheckResult:
    """Check 4: answerable flag cross-checked against evidence and answer."""
    problems: list[str] = []
    for index, entry in iter_query_mappings(queries):
        label = query_label(entry, index)
        answerable = entry.get("answerable")
        evidence = entry.get("relevant_evidence")
        expected = entry.get("expected_answer")
        if not isinstance(answerable, bool):
            problems.append(f"{label}: answerable={answerable!r} is not a JSON boolean")
            continue
        if answerable:
            if not isinstance(evidence, list) or not evidence:
                problems.append(
                    f"{label}: answerable=true requires a non-empty relevant_evidence list"
                )
            else:
                bad = [
                    str(pos)
                    for pos, item in enumerate(evidence)
                    if not (isinstance(item, str) and item.strip())
                ]
                if bad:
                    problems.append(
                        f"{label}: answerable=true has empty/non-string relevant_evidence at "
                        f"index {', '.join(bad)}"
                    )
            if not (isinstance(expected, str) and expected.strip()):
                problems.append(
                    f"{label}: answerable=true requires a non-empty string expected_answer "
                    f"(got {shorten(expected, 40)!r})"
                )
        else:
            if evidence != []:
                detail = f"{len(evidence)} item(s)" if isinstance(evidence, list) else repr(evidence)
                problems.append(
                    f"{label}: answerable=false requires relevant_evidence == [] (got {detail})"
                )
            if expected is not None:
                problems.append(
                    f"{label}: answerable=false requires expected_answer == null "
                    f"(got {shorten(expected, 40)!r})"
                )
    if problems:
        return CheckResult(
            "answerable_constraints",
            False,
            f"{len(problems)} violation(s): {format_offenders(problems)}",
            tuple(problems),
        )
    return CheckResult(
        "answerable_constraints",
        True,
        "every query satisfies its answerable/unanswerable constraints (both directions)",
    )


def check_gold_evidence_grounding(queries: Sequence[Any], corpus: CorpusData) -> CheckResult:
    """Check 5: every gold evidence string is a substring of the corpus text."""
    misses: list[str] = []
    checked = 0
    for index, entry in iter_query_mappings(queries):
        label = query_label(entry, index)
        evidence = entry.get("relevant_evidence")
        if not isinstance(evidence, list):
            continue
        for position, item in enumerate(evidence):
            if not isinstance(item, str) or not item.strip():
                continue
            checked += 1
            if normalize_text(item) not in corpus.normalized:
                misses.append(f"{label}: evidence[{position}] not found in corpus: {shorten(item)!r}")
    if misses:
        return CheckResult(
            "gold_evidence_grounding",
            False,
            f"{len(misses)} of {checked} evidence line(s) are NOT in the corpus "
            f"(fabricated citation?): {format_offenders(misses)}",
            tuple(misses),
        )
    return CheckResult(
        "gold_evidence_grounding",
        True,
        f"all {checked} evidence line(s) from {len(queries)} queries found in the corpus "
        f"(normalize_text substring test)",
    )


def check_evidence_line_format(queries: Sequence[Any]) -> CheckResult:
    """Check 6: each evidence string looks like a corpus message line."""
    problems: list[str] = []
    checked = 0
    for index, entry in iter_query_mappings(queries):
        label = query_label(entry, index)
        evidence = entry.get("relevant_evidence")
        if not isinstance(evidence, list):
            continue
        for position, item in enumerate(evidence):
            if not isinstance(item, str) or not item.strip():
                continue
            checked += 1
            if not TIMESTAMP_PREFIX_RE.match(item):
                problems.append(
                    f"{label}: evidence[{position}] does not start with '[YYYY-MM-DD HH:MM] ': "
                    f"{shorten(item)!r}"
                )
                continue
            stamp = item[1:17]
            try:
                datetime.strptime(stamp, TIMESTAMP_FORMAT)
            except ValueError:
                problems.append(
                    f"{label}: evidence[{position}] has an impossible timestamp [{stamp}]"
                )
    if problems:
        return CheckResult(
            "evidence_line_format",
            False,
            f"{len(problems)} malformed evidence line(s) of {checked}: {format_offenders(problems)}",
            tuple(problems),
        )
    return CheckResult(
        "evidence_line_format",
        True,
        f"all {checked} evidence line(s) start with a '[YYYY-MM-DD HH:MM] ' timestamp",
    )


def check_query_count(queries: Sequence[Any]) -> CheckResult:
    """Check 7: query count is inside the required range (warn on the edges)."""
    count = len(queries)
    if count < MIN_QUERIES or count > MAX_QUERIES:
        return CheckResult(
            "query_count_range",
            False,
            f"{count} queries is outside the required {MIN_QUERIES}-{MAX_QUERIES} range",
        )
    warning = None
    if count in (MIN_QUERIES, MAX_QUERIES):
        warning = (
            f"query count {count} sits exactly on the {MIN_QUERIES}-{MAX_QUERIES} boundary"
        )
    message = f"{count} queries - inside the required {MIN_QUERIES}-{MAX_QUERIES} range"
    if warning:
        message += " (WARN: on the boundary)"
    return CheckResult("query_count_range", True, message, warning=warning)


def check_corpus_size(corpus: CorpusData) -> CheckResult:
    """Check 8: corpus is large enough in characters and messages."""
    chars = len(corpus.text)
    messages = len(corpus.message_lines)
    episodes = len(corpus.episodes)
    problems: list[str] = []
    if chars < MIN_CORPUS_CHARS:
        problems.append(f"chars={chars:,} < required {MIN_CORPUS_CHARS:,}")
    if messages < MIN_CORPUS_MESSAGES:
        problems.append(f"messages={messages:,} < required {MIN_CORPUS_MESSAGES:,}")
    stats = (
        f"bytes={corpus.n_bytes:,}, chars={chars:,}, messages={messages:,}, episodes={episodes:,}"
    )
    if problems:
        return CheckResult(
            "corpus_size", False, f"{stats} - {'; '.join(problems)}", tuple(problems)
        )
    return CheckResult("corpus_size", True, f"{stats} - above the minimums")


def check_category_distribution(summary: Mapping[str, Any]) -> CheckResult:
    """Check 9: at least ten distinct categories are present."""
    categories: Mapping[str, int] = summary["categories"]
    defects: Sequence[str] = summary["category_defects"]
    distinct = len(categories)
    problems = list(defects)
    if distinct < MIN_DISTINCT_CATEGORIES:
        problems.append(f"only {distinct} distinct categories (need >= {MIN_DISTINCT_CATEGORIES})")
    listed = ", ".join(f"{name}={count}" for name, count in categories.items())
    if problems:
        return CheckResult(
            "category_distribution",
            False,
            f"{distinct} distinct categories: {listed} | {format_offenders(problems)}",
            tuple(problems),
        )
    return CheckResult(
        "category_distribution",
        True,
        f"{distinct} distinct categories (>= {MIN_DISTINCT_CATEGORIES}): {listed}",
    )


def check_chunk_count(corpus: CorpusData) -> CheckResult:
    """Check 10: the mirrored splitter produces enough chunks."""
    chunks = len(corpus.chunks)
    detail = (
        f"{chunks} chunks for {len(corpus.text):,} chars via "
        f"recursive_character_splitter(chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})"
    )
    if chunks < MIN_CHUNKS:
        return CheckResult(
            "chunk_count", False, f"{detail} - below the required minimum of {MIN_CHUNKS}"
        )
    return CheckResult("chunk_count", True, f"{detail} - >= {MIN_CHUNKS}")


def check_timestamp_sanity(corpus: CorpusData, report: TimestampReport) -> CheckResult:
    """Check 11: message format, timestamp ordering and uniqueness."""
    problems = (
        [f"format: {item}" for item in report.format_offenders]
        + [f"calendar: {item}" for item in report.calendar_offenders]
        + [f"order: {item}" for item in report.order_offenders]
        + [f"duplicate: {item}" for item in report.duplicate_offenders]
    )
    span = f"first={report.first_timestamp}, last={report.last_timestamp}"
    if problems:
        return CheckResult(
            "corpus_timestamps",
            False,
            f"{len(problems)} timestamp/format problem(s) among {len(corpus.message_lines)} "
            f"messages ({span}): {format_offenders(problems)}",
            tuple(problems),
        )
    return CheckResult(
        "corpus_timestamps",
        True,
        f"all {len(corpus.message_lines)} messages match '[YYYY-MM-DD HH:MM] 说话人: 内容'; "
        f"timestamps non-decreasing and unique ({span})",
    )


# ---------------------------------------------------------------------------
# Difficulty diagnostics (character n-grams only - advisory, never fatal)
# ---------------------------------------------------------------------------


def longest_common_substring_length(left: str, right: str) -> int:
    """Length of the longest common substring of two strings (binary search)."""
    if not left or not right:
        return 0
    low, high = 0, min(len(left), len(right))
    while low < high:
        middle = (low + high + 1) // 2
        if _shares_substring(left, right, middle):
            low = middle
        else:
            high = middle - 1
    return low


def _shares_substring(left: str, right: str, length: int) -> bool:
    """True when some ``length``-character window of ``left`` occurs in ``right``."""
    for start in range(len(left) - length + 1):
        if left[start : start + length] in right:
            return True
    return False


def select_anchor_ngrams(question: str, gold_text: str) -> tuple[set[str], int | None]:
    """Longest shared character 2-3 grams between question and gold evidence."""
    for size in (3, 2):
        shared = char_ngrams(question, size) & char_ngrams(gold_text, size)
        if shared:
            return shared, size
    return set(), None


def chunk_frequency(ngram: str, chunks: Sequence[str], cache: dict[str, int]) -> int:
    """Number of corpus chunks containing ``ngram`` (memoised per run)."""
    cached = cache.get(ngram)
    if cached is None:
        cached = sum(1 for chunk in chunks if ngram in chunk)
        cache[ngram] = cached
    return cached


def analyse_query(
    entry: Mapping[str, Any],
    index: int,
    corpus: CorpusData,
    line_records: Sequence[tuple[str, frozenset[str]]],
    frequency_cache: dict[str, int],
) -> dict[str, Any]:
    """Measure one query's retrieval difficulty with character n-grams only."""
    label = query_label(entry, index)
    question = entry.get("question")
    question_text = normalize_text(question) if isinstance(question, str) else ""
    evidence = entry.get("relevant_evidence")
    evidence_lines = (
        [item for item in evidence if isinstance(item, str) and item.strip()]
        if isinstance(evidence, list)
        else []
    )
    # Timestamps are dropped for n-gram work: two messages sharing a minute would
    # otherwise look lexically similar for reasons that are not content.
    gold_text = " ".join(strip_timestamp(normalize_text(item)) for item in evidence_lines)

    overlap: float | None = None
    lcs_gold: int | None = None
    rare_chunks: int | None = None
    anchor_size: int | None = None
    anchor_total: int | None = None
    distractors: int | None = None
    flags: list[str] = []

    if gold_text:
        question_bigrams = char_ngrams(question_text, 2)
        if question_bigrams:
            hits = sum(1 for gram in question_bigrams if gram in gold_text)
            overlap = round(hits / len(question_bigrams), 4)
        lcs_gold = longest_common_substring_length(question_text, gold_text)

        anchors, anchor_size = select_anchor_ngrams(question_text, gold_text)
        anchor_total = len(anchors)
        if anchors:
            rare_chunks = min(
                chunk_frequency(gram, corpus.chunks, frequency_cache) for gram in anchors
            )

        gold_fourgrams = char_ngrams(gold_text, DISTRACTOR_NGRAM)
        evidence_set = {strip_timestamp(normalize_text(item)) for item in evidence_lines}
        distractors = 0
        if gold_fourgrams:
            for body, grams in line_records:
                if body in evidence_set:
                    continue
                if len(grams & gold_fourgrams) >= DISTRACTOR_MIN_SHARED:
                    distractors += 1

        if overlap is not None and overlap > OVERLAP_TOO_HIGH:
            flags.append(f"overlap>{OVERLAP_TOO_HIGH:g}")
        if rare_chunks == RARE_ANCHOR_TOO_UNIQUE:
            flags.append(f"rare_chunks=={RARE_ANCHOR_TOO_UNIQUE}")
        if distractors == 0:
            flags.append("distractors==0")
        if anchor_total == 0:
            flags.append("no_anchor_ngram")

    return {
        "id": label,
        "category": entry.get("category") if isinstance(entry.get("category"), str) else None,
        "answerable": entry.get("answerable") is True,
        "evidence_lines": len(evidence_lines),
        "gold_chars": len(gold_text),
        "question_gold_overlap": overlap,
        "lcs_question_gold": lcs_gold,
        "lcs_question_corpus": longest_common_substring_length(question_text, corpus.normalized),
        "rare_term_chunks": rare_chunks,
        "anchor_ngram_size": anchor_size,
        "anchor_ngram_count": anchor_total,
        "near_duplicate_distractors": distractors,
        "flags": flags,
    }


def summarise_difficulty(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Bucket the per-query diagnostics into advisory summary counts."""
    with_gold = [r for r in records if r["question_gold_overlap"] is not None]
    without_gold = [r for r in records if r["question_gold_overlap"] is None]

    def bucket(predicate: Any) -> dict[str, Any]:
        ids = [str(r["id"]) for r in with_gold if predicate(r)]
        return {"count": len(ids), "ids": ids}

    overlaps = [float(r["question_gold_overlap"]) for r in with_gold]
    rare_values = [int(r["rare_term_chunks"]) for r in with_gold if r["rare_term_chunks"] is not None]
    corpus_lcs = [int(r["lcs_question_corpus"]) for r in records if r["lcs_question_corpus"]]
    summary: dict[str, Any] = {
        "metrics_evaluated": len(with_gold),
        "excluded_no_gold": len(without_gold),
        "excluded_no_gold_ids": [str(r["id"]) for r in without_gold],
        "question_gold_overlap_mean": round(sum(overlaps) / len(overlaps), 4) if overlaps else None,
        "question_gold_overlap_max": max(overlaps) if overlaps else None,
        "rare_term_chunks_min": min(rare_values) if rare_values else None,
        "lcs_question_corpus_max": max(corpus_lcs) if corpus_lcs else None,
        f"overlap_gt_{OVERLAP_TOO_HIGH:g}": bucket(
            lambda r: float(r["question_gold_overlap"]) > OVERLAP_TOO_HIGH
        ),
        f"rare_term_chunks_eq_{RARE_ANCHOR_TOO_UNIQUE}": bucket(
            lambda r: r["rare_term_chunks"] == RARE_ANCHOR_TOO_UNIQUE
        ),
        "near_duplicate_distractors_eq_0": bucket(
            lambda r: r["near_duplicate_distractors"] == 0
        ),
        "thresholds": {
            "question_gold_overlap_high": OVERLAP_TOO_HIGH,
            "rare_term_chunks_too_unique": RARE_ANCHOR_TOO_UNIQUE,
            "near_duplicate_distractors_none": 0,
            "distractor_min_shared_4grams": DISTRACTOR_MIN_SHARED,
        },
    }
    return summary


def analyse_difficulty(queries: Sequence[Any], corpus: CorpusData) -> dict[str, Any]:
    """Compute per-query difficulty records plus the summary buckets."""
    line_records = tuple(
        (
            strip_timestamp(normalize_text(line)),
            frozenset(char_ngrams(strip_timestamp(normalize_text(line)), DISTRACTOR_NGRAM)),
        )
        for line in corpus.message_lines
    )
    frequency_cache: dict[str, int] = {}
    records = [
        analyse_query(entry, index, corpus, line_records, frequency_cache)
        for index, entry in iter_query_mappings(queries)
    ]
    return {"per_query": records, "summary": summarise_difficulty(records)}


# ---------------------------------------------------------------------------
# Report assembly and rendering
# ---------------------------------------------------------------------------


@dataclass
class Report:
    """Everything the validator knows, ready for stdout and for JSON."""

    corpus_path: Path
    queries_path: Path
    json_out: Path
    corpus: CorpusData | None
    queries: Sequence[Any] | None
    timestamps: TimestampReport | None
    query_summary: Mapping[str, Any]
    checks: dict[str, CheckResult] = field(default_factory=dict)
    difficulty: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True only when every hard check passed."""
        return bool(self.checks) and all(result.passed for result in self.checks.values())

    @property
    def failed_names(self) -> list[str]:
        """Names of the hard checks that failed, in report order."""
        return [name for name, result in self.checks.items() if not result.passed]


def build_report(args: argparse.Namespace) -> Report:
    """Load both inputs, run all hard checks and the difficulty diagnostics."""
    corpus, corpus_error, corpus_warnings = load_corpus(args.corpus)
    queries, queries_error, queries_warnings = load_queries(args.queries)
    timestamps = inspect_timestamps(corpus) if corpus is not None else None
    query_summary = summarize_queries(queries) if queries is not None else {
        "count": 0,
        "answerable": 0,
        "unanswerable": 0,
        "categories": {},
        "category_defects": [],
    }

    report = Report(
        corpus_path=args.corpus,
        queries_path=args.queries,
        json_out=args.json_out,
        corpus=corpus,
        queries=queries,
        timestamps=timestamps,
        query_summary=query_summary,
    )
    if corpus is not None:
        report.warnings.extend(corpus_warnings)
    report.warnings.extend(queries_warnings)

    # Checks are registered in the same order as the numbered hard checks.
    have_queries = queries is not None
    have_corpus = corpus is not None
    queries_gap = None if have_queries else (queries_error or "queries unavailable")
    corpus_gap = None if have_corpus else (corpus_error or "corpus unavailable")
    both_gap = (
        None
        if have_queries and have_corpus
        else (queries_error or corpus_error or "input unavailable")
    )

    report.checks["files_exist_and_parse"] = check_files_exist_and_parse(
        corpus_error, queries_error, args.corpus, args.queries
    )
    report.checks["query_ids_unique_non_empty"] = run_or_skip(
        "query_ids_unique_non_empty", check_query_ids, queries_gap, queries
    )
    report.checks["query_keys_exact"] = run_or_skip(
        "query_keys_exact", check_query_keys, queries_gap, queries
    )
    report.checks["answerable_constraints"] = run_or_skip(
        "answerable_constraints", check_answerable_constraints, queries_gap, queries
    )
    report.checks["gold_evidence_grounding"] = run_or_skip(
        "gold_evidence_grounding", check_gold_evidence_grounding, both_gap, queries, corpus
    )
    report.checks["evidence_line_format"] = run_or_skip(
        "evidence_line_format", check_evidence_line_format, queries_gap, queries
    )
    report.checks["query_count_range"] = run_or_skip(
        "query_count_range", check_query_count, queries_gap, queries
    )
    report.checks["corpus_size"] = run_or_skip(
        "corpus_size", check_corpus_size, corpus_gap, corpus
    )
    report.checks["category_distribution"] = check_category_distribution(query_summary)
    report.checks["chunk_count"] = run_or_skip(
        "chunk_count", check_chunk_count, corpus_gap, corpus
    )
    report.checks["corpus_timestamps"] = run_or_skip(
        "corpus_timestamps",
        check_timestamp_sanity,
        None if (have_corpus and timestamps is not None) else corpus_gap,
        corpus,
        timestamps,
    )

    if corpus is not None and queries is not None:
        report.difficulty = analyse_difficulty(queries, corpus)
    else:
        report.difficulty = {
            "per_query": [],
            "summary": {
                "metrics_evaluated": 0,
                "excluded_no_gold": 0,
                "excluded_no_gold_ids": [],
                "note": "difficulty diagnostics not evaluated: input unavailable",
            },
        }

    for result in report.checks.values():
        if result.warning:
            report.warnings.append(f"{result.name}: {result.warning}")
    return report


def _metric(value: Any, width: int, spec: str = "") -> str:
    """Right-align a metric cell, rendering ``None`` as ``n/a``."""
    if value is None:
        return f"{'n/a':>{width}}"
    if spec:
        return f"{format(value, spec):>{width}}"
    return f"{str(value):>{width}}"


def render_difficulty_table(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Render the per-query diagnostics table."""
    header = (
        f"  {'id':<10}{'ans':<5}{'q2&gold':>8}{'LCS-g':>7}{'LCS-c':>7}"
        f"{'rare':>6}{'dist':>6}  flags"
    )
    lines = [header, "  " + "-" * (len(header) - 2)]
    for record in records:
        flags = ",".join(record["flags"]) if record["flags"] else "-"
        lines.append(
            f"  {str(record['id']):<10}"
            f"{'y' if record['answerable'] else 'n':<5}"
            f"{_metric(record['question_gold_overlap'], 8, '.2f')}"
            f"{_metric(record['lcs_question_gold'], 7)}"
            f"{_metric(record['lcs_question_corpus'], 7)}"
            f"{_metric(record['rare_term_chunks'], 6)}"
            f"{_metric(record['near_duplicate_distractors'], 6)}  {flags}"
        )
    return lines


def render_text(report: Report) -> str:
    """Render the human-readable report printed to stdout."""
    out: list[str] = []
    add = out.append
    add("=" * 78)
    add("Stress dataset validation (offline, deterministic, no model)")
    add("=" * 78)
    add(f"corpus  : {report.corpus_path}")
    add(f"queries : {report.queries_path}")
    add(f"json out: {report.json_out}")
    add("")

    add("-- Corpus " + "-" * 68)
    if report.corpus is None:
        add("  unavailable - see check 'files_exist_and_parse'")
    else:
        corpus = report.corpus
        add(f"  bytes    : {corpus.n_bytes:,}")
        add(f"  chars    : {len(corpus.text):,}   (minimum {MIN_CORPUS_CHARS:,})")
        add(f"  messages : {len(corpus.message_lines):,}   (minimum {MIN_CORPUS_MESSAGES:,})")
        add(f"  episodes : {len(corpus.episodes):,}")
        add(
            f"  chunks   : {len(corpus.chunks):,}   "
            f"(chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}, minimum {MIN_CHUNKS})"
        )
        if report.timestamps is not None:
            add(
                f"  window   : {report.timestamps.first_timestamp} -> "
                f"{report.timestamps.last_timestamp}"
            )
    add("")

    add("-- Queries " + "-" * 67)
    if report.queries is None:
        add("  unavailable - see check 'files_exist_and_parse'")
    else:
        summary = report.query_summary
        add(f"  entries     : {summary['count']}   (required {MIN_QUERIES}-{MAX_QUERIES})")
        add(f"  answerable  : {summary['answerable']}")
        add(f"  unanswerable: {summary['unanswerable']}")
        categories: Mapping[str, int] = summary["categories"]
        add(f"  categories  : {len(categories)} distinct (minimum {MIN_DISTINCT_CATEGORIES})")
        for name, count in categories.items():
            add(f"      {name:<28}{count:>4}")
    add("")

    add("-- Hard checks " + "-" * 63)
    for name, result in report.checks.items():
        add(f"  [{'PASS' if result.passed else 'FAIL'}] {name}")
        add(f"         {result.message}")
    add("")

    if report.warnings:
        add("-- Warnings " + "-" * 66)
        for warning in report.warnings:
            add(f"  WARN {warning}")
        add("")

    add("-- Difficulty diagnostics " + "-" * 52)
    records: Sequence[Mapping[str, Any]] = report.difficulty.get("per_query", [])
    if not records:
        add("  no per-query records (difficulty not evaluated)")
    else:
        add(
            "  q2&gold = share of the question's char 2-grams present in its gold evidence"
        )
        add("  LCS-g/-c = longest common substring vs gold evidence / vs whole corpus")
        add("  rare = min chunk count over the longest shared 2-3 gram (1 = single chunk)")
        add("  dist = messages sharing >= 3 char 4-grams with the gold text (timestamps ignored)")
        add("")
        out.extend(render_difficulty_table(records))
        add("")
        summary = report.difficulty.get("summary", {})
        add("-- Difficulty summary (advisory - never fails the run) " + "-" * 25)
        add(
            f"  metrics evaluated           : {summary.get('metrics_evaluated')} of "
            f"{len(records)} queries "
            f"({summary.get('excluded_no_gold')} without gold evidence excluded)"
        )
        add(f"  question_gold_overlap mean  : {summary.get('question_gold_overlap_mean')}")
        add(f"  question_gold_overlap max   : {summary.get('question_gold_overlap_max')}")
        add(f"  rarest anchor chunk count   : {summary.get('rare_term_chunks_min')}")
        add(f"  LCS(question, corpus) max   : {summary.get('lcs_question_corpus_max')}")
        for key in (
            f"overlap_gt_{OVERLAP_TOO_HIGH:g}",
            f"rare_term_chunks_eq_{RARE_ANCHOR_TOO_UNIQUE}",
            "near_duplicate_distractors_eq_0",
        ):
            bucket = summary.get(key, {"count": 0, "ids": []})
            ids = ", ".join(bucket["ids"]) if bucket["ids"] else "-"
            add(f"  {key:<32}: {bucket['count']:<3} {ids}")
    add("")

    add("-- Verdict " + "-" * 67)
    if report.passed:
        add(
            f"  VERDICT: PASS - all {len(report.checks)} hard checks passed "
            f"(difficulty diagnostics are advisory)"
        )
    else:
        add(
            f"  VERDICT: FAIL - {len(report.failed_names)}/{len(report.checks)} hard checks failed: "
            f"{', '.join(report.failed_names)}"
        )
    return "\n".join(out)


def to_json(report: Report) -> dict[str, Any]:
    """Build the machine-readable report payload."""
    corpus_payload: dict[str, Any] = {"available": report.corpus is not None}
    if report.corpus is not None:
        corpus = report.corpus
        corpus_payload.update(
            {
                "path": str(corpus.path),
                "bytes": corpus.n_bytes,
                "chars": len(corpus.text),
                "messages": len(corpus.message_lines),
                "episodes": len(corpus.episodes),
                "chunks": len(corpus.chunks),
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "first_timestamp": (
                    report.timestamps.first_timestamp if report.timestamps else None
                ),
                "last_timestamp": (
                    report.timestamps.last_timestamp if report.timestamps else None
                ),
            }
        )

    queries_payload: dict[str, Any] = {
        "path": str(report.queries_path),
        "available": report.queries is not None,
        "count": report.query_summary["count"],
        "answerable": report.query_summary["answerable"],
        "unanswerable": report.query_summary["unanswerable"],
        "categories": report.query_summary["categories"],
    }

    return {
        "passed": report.passed,
        "inputs": {
            "corpus": str(report.corpus_path),
            "queries": str(report.queries_path),
            "json_out": str(report.json_out),
        },
        "corpus": corpus_payload,
        "queries": queries_payload,
        "checks": {name: result.to_json() for name, result in report.checks.items()},
        "difficulty": {
            "per_query": report.difficulty.get("per_query", []),
            "summary": report.difficulty.get("summary", {}),
        },
        "warnings": list(report.warnings),
    }


def write_json_output(path: Path, payload: Mapping[str, Any]) -> None:
    """Write the JSON report, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def configure_stdio() -> None:
    """Force UTF-8 on stdout/stderr so Chinese diagnostics never crash a run."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - exotic streams only
            pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the CLI, with defaults resolved against this file's directory."""
    parser = argparse.ArgumentParser(
        prog="validate_stress_dataset.py",
        description=(
            "Offline, deterministic validator for the stress chat-history retrieval "
            "dataset (no model, no network). Exit code 0 only when every hard check passes."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"corpus text file (default: {DEFAULT_CORPUS_PATH})",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help=f"query JSON file (default: {DEFAULT_QUERIES_PATH})",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_JSON_OUT_PATH,
        help=f"machine-readable report path (default: {DEFAULT_JSON_OUT_PATH})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the validator; return the process exit code (0 = all checks passed)."""
    configure_stdio()
    args = parse_args(argv)
    report = build_report(args)
    print(render_text(report))
    write_json_output(args.json_out, to_json(report))
    print("")
    print(f"JSON report written to {args.json_out}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
