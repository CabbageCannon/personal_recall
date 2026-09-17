"""Import a real chat export into the engine's canonical line format.

The engine's adapter accepts exactly one shape (``memory/events.py``)::

    [YYYY-MM-DD HH:MM] 说话人: 内容

Real exports do not look like that, and the failure is **silent** — pointing `recall.py` at a
Telegram- or WeChat-style export indexes zero messages and reports no error, so the engine simply
answers "the record does not show it" for every question. Measured before this module existed:

    canonical                                     2 messages   0 skipped   OK
    telegram-ish [DD.MM.YY HH:MM]                 0 messages   2 skipped   SILENTLY EMPTY
    loose ISO with seconds                        0 messages   2 skipped   SILENTLY EMPTY
    wechat-ish speaker-first header + body line   0 messages   4 skipped   SILENTLY EMPTY

What this module does: detects which of a small set of well-specified layouts a file uses, converts it
to the canonical form, and **verifies the result by re-parsing it through the engine's own adapter**
with zero skipped lines. Nothing is written unless that round-trip succeeds.

Deliberate limits, stated rather than implied:

* Only text messages are supported. Attachments, stickers, recalls and system notices are skipped and
  counted, never guessed at.
* A message body that spans several lines is joined with a space, because the canonical format is one
  message per line — the newlines are a formatting artefact of the export, not content.
* Supported layouts are a fixed list, not a format-guessing framework. An unrecognised file gets a
  diagnostic naming what was tried and showing the lines that failed, instead of silence.

Usage::

    python chat_import.py --input my_export.txt --out data/my_chat.txt
    python chat_import.py --input my_export.txt --check-only
    python chat_import.py --list-formats
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from memory.events import parse_txt_events  # noqa: E402

CANONICAL_TIMESTAMP = "%Y-%m-%d %H:%M"
MAX_SPEAKER_CHARS = 16


@dataclass(frozen=True)
class Profile:
    """One recognised export layout."""

    name: str
    description: str
    pattern: re.Pattern[str]
    timestamp_formats: tuple[str, ...]
    #: "inline" = timestamp, speaker and text on one line; the pattern carries all three groups.
    #: "header" = the line opens a message and following non-matching lines are its body.
    layout: str = "inline"


PROFILES: tuple[Profile, ...] = (
    Profile(
        name="canonical",
        description="[YYYY-MM-DD HH:MM] speaker: text (already ingestible)",
        pattern=re.compile(
            r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\s*(?P<speaker>[^:]{1,24}):\s*(?P<text>.+)$"
        ),
        timestamp_formats=(CANONICAL_TIMESTAMP,),
    ),
    Profile(
        name="bracket_dmy",
        description="[DD.MM.YY HH:MM] speaker: text (Telegram-style export)",
        pattern=re.compile(
            r"^\[(?P<ts>\d{1,2}\.\d{1,2}\.\d{2,4} \d{1,2}:\d{2})\]\s*"
            r"(?P<speaker>[^:]{1,24}):\s*(?P<text>.+)$"
        ),
        timestamp_formats=("%d.%m.%y %H:%M", "%d.%m.%Y %H:%M"),
    ),
    Profile(
        name="iso_seconds",
        description="YYYY-MM-DD HH:MM:SS speaker: text (plain log-style export)",
        pattern=re.compile(
            r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"
            r"(?P<speaker>[^:]{1,24}):\s*(?P<text>.+)$"
        ),
        timestamp_formats=("%Y-%m-%d %H:%M:%S",),
    ),
    Profile(
        name="speaker_first",
        description=(
            "speaker<spaces>YYYY-MM-DD HH:MM, message on the following line(s) "
            "(WeChat-style export)"
        ),
        pattern=re.compile(
            r"^(?P<speaker>\S[^:]{0,23}?)\s{2,}(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s*$"
        ),
        timestamp_formats=(CANONICAL_TIMESTAMP,),
        layout="header",
    ),
)

#: Recognised but deliberately unsupported layouts, so the diagnostic can say something useful.
KNOWN_UNSUPPORTED = (
    (
        re.compile(r"^\d{4}年\d{1,2}月\d{1,2}日\s*\d{1,2}:\d{2}$"),
        "a date on its own line followed by 'speaker: text' lines",
    ),
)


@dataclass
class ImportReport:
    source: str
    profile: str | None = None
    detected_by: str = "auto"
    messages: int = 0
    skipped_lines: int = 0
    speakers: tuple[str, ...] = ()
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    skipped_samples: tuple[tuple[int, str], ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)
    verified: bool = False

    @property
    def ok(self) -> bool:
        return self.messages > 0 and self.verified

    def lines(self) -> list[str]:
        out = [f"source        : {self.source}"]
        out.append(f"layout        : {self.profile or 'NOT RECOGNISED'} ({self.detected_by})")
        out.append(f"messages      : {self.messages}")
        out.append(f"skipped lines : {self.skipped_lines}")
        out.append(f"speakers      : {len(self.speakers)} {list(self.speakers[:8])}")
        if self.first_timestamp:
            out.append(f"time span     : {self.first_timestamp} .. {self.last_timestamp}")
        out.append(
            "round-trip    : "
            + ("verified through the engine adapter" if self.verified else "FAILED")
        )
        for note in self.notes:
            out.append(f"note          : {note}")
        for number, line in self.skipped_samples:
            out.append(f"  skipped L{number}: {line[:100]}")
        return out


def _parse_timestamp(value: str, formats: tuple[str, ...]) -> datetime | None:
    for fmt in formats:
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def _clean_speaker(speaker: str, notes: list[str]) -> str:
    cleaned = speaker.strip().lstrip("[").strip()
    # The canonical format delimits the speaker with ':', so a colon in the name would corrupt it.
    if ":" in cleaned:
        cleaned = cleaned.replace(":", "：")
        notes.append("replaced ':' inside a speaker name with '：' to keep the canonical format")
    if len(cleaned) > MAX_SPEAKER_CHARS:
        original = cleaned
        cleaned = cleaned[:MAX_SPEAKER_CHARS]
        notes.append(f"truncated a speaker name to {MAX_SPEAKER_CHARS} chars: {original!r}")
    return cleaned


def detect_profile(text: str) -> tuple[Profile | None, dict[str, int]]:
    """Return the best-matching profile and the per-profile match counts."""
    counts: dict[str, int] = {}
    for profile in PROFILES:
        counts[profile.name] = sum(
            1 for line in text.split("\n") if profile.pattern.match(line.strip())
        )
    best = max(PROFILES, key=lambda p: counts[p.name])
    return (best if counts[best.name] else None), counts


def convert(
    text: str, profile: Profile
) -> tuple[list[tuple[datetime, str, str]], int, list[tuple[int, str]], list[str]]:
    """Convert text using ``profile`` into ``(messages, skipped, skipped_samples, notes)``."""
    messages: list[tuple[datetime, str, str]] = []
    skipped = 0
    samples: list[tuple[int, str]] = []
    notes: list[str] = []
    body: list[str] = []
    # A header opens a message whose body arrives on the following lines. Tracking openness
    # separately from the body text matters: a header with no body yet must still swallow the next
    # line, and forgetting that skipped every body line of the speaker-first layout.
    state: dict[str, object] = {"open": False, "ts": None, "speaker": ""}

    def flush() -> None:
        if state["open"] and body:
            messages.append((state["ts"], state["speaker"], " ".join(body).strip()))  # type: ignore[arg-type]
        body.clear()
        state["open"] = False

    for number, raw in enumerate(text.split("\n"), start=1):
        line = raw.strip()
        if not line:
            continue
        match = profile.pattern.match(line)
        if match is None:
            if profile.layout == "header" and state["open"]:
                body.append(line)  # a wrapped body line belongs to the open message
            else:
                skipped += 1
                if len(samples) < 5:
                    samples.append((number, line))
            continue

        timestamp = _parse_timestamp(match.group("ts"), profile.timestamp_formats)
        if timestamp is None:
            skipped += 1
            if len(samples) < 5:
                samples.append((number, line))
            continue

        if profile.layout == "header":
            header_without_body = bool(state["open"]) and not body
            flush()
            if header_without_body:
                notes.append("a header was followed immediately by another header")
            state["ts"] = timestamp
            state["speaker"] = _clean_speaker(match.group("speaker"), notes)
            state["open"] = True
            continue

        speaker = _clean_speaker(match.group("speaker"), notes)
        cleaned_body = " ".join(match.group("text").split())
        if not speaker or not cleaned_body:
            skipped += 1
            if len(samples) < 5:
                samples.append((number, line))
            continue
        messages.append((timestamp, speaker, cleaned_body))

    if profile.layout == "header":
        opened = bool(state["open"])
        had_body = bool(body)
        flush()
        if opened and not had_body:
            notes.append("the export ended on a header with no message body")

    return [m for m in messages if m[2]], skipped, samples, notes


def canonical_text(messages: list[tuple[datetime, str, str]]) -> str:
    ordered = sorted(messages, key=lambda m: m[0])
    return (
        "\n".join(
            f"[{ts.strftime(CANONICAL_TIMESTAMP)}] {speaker}: {text}"
            for ts, speaker, text in ordered
        )
        + "\n"
    )


def import_text(
    text: str, source: str = "<text>", profile_name: str = "auto"
) -> tuple[str, ImportReport]:
    """Convert ``text`` to canonical form and verify it through the engine's own adapter."""
    # Real exports often start with a UTF-8 BOM. Left in place it sticks to the first line and makes
    # exactly one message unmatchable — which is how the first smoke test lost a message.
    text = text.lstrip("\ufeff")
    report = ImportReport(source=source)

    if profile_name == "auto":
        profile, counts = detect_profile(text)
        if profile is None:
            unsupported = [
                why
                for pattern, why in KNOWN_UNSUPPORTED
                if any(pattern.match(line.strip()) for line in text.split("\n") if line.strip())
            ]
            report.notes = report.notes + (
                (
                    f"the file looks like {unsupported[0]}, which is recognised but not supported yet",
                )
                if unsupported
                else ("no supported layout matched any line",)
            )
            report.notes = report.notes + (
                f"line-match counts by layout: {counts}",
                "expected one of: " + "; ".join(f"{p.name} = {p.description}" for p in PROFILES),
            )
            return "", report
        report.profile = profile.name
    else:
        profile = next((p for p in PROFILES if p.name == profile_name), None)
        if profile is None:
            raise ValueError(
                f"unknown profile {profile_name!r}; choose from {[p.name for p in PROFILES]}"
            )
        report.profile = profile.name
        report.detected_by = "explicit"

    messages, skipped, samples, notes = convert(text, profile)
    report.messages = len(messages)
    report.skipped_lines = skipped
    report.skipped_samples = tuple(samples)
    report.notes = tuple(notes)
    if messages:
        report.speakers = tuple(sorted({speaker for _, speaker, _ in messages}))
        stamps = sorted(ts for ts, _, _ in messages)
        report.first_timestamp = stamps[0].strftime(CANONICAL_TIMESTAMP)
        report.last_timestamp = stamps[-1].strftime(CANONICAL_TIMESTAMP)

    output = canonical_text(messages) if messages else ""

    # Verification: the output must be ingested by the engine's real adapter with nothing skipped.
    if output:
        parsed = parse_txt_events(output)
        report.verified = len(parsed.events) == len(messages) and parsed.skipped_lines == 0
        if not report.verified:
            report.notes = report.notes + (
                f"round-trip mismatch: adapter read {len(parsed.events)} of {len(messages)} "
                f"messages with {parsed.skipped_lines} skipped",
            )
    return output, report


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, help="the chat export to convert")
    ap.add_argument("--out", type=Path, help="where to write the canonical corpus")
    ap.add_argument("--format", default="auto", help="'auto' or a layout name")
    ap.add_argument("--list-formats", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    if args.list_formats:
        for profile in PROFILES:
            print(f"  {profile.name:16} {profile.description}")
        return 0

    if args.input is None:
        ap.error("--input is required (or use --list-formats)")

    text = args.input.read_text(encoding="utf-8-sig", errors="replace")
    output, report = import_text(text, source=str(args.input), profile_name=args.format)

    print("import report:")
    for line in report.lines():
        print(f"  {line}")

    if not report.ok:
        print("\nFAILED: nothing was written. Fix the format or convert the export first.")
        return 1

    if args.check_only:
        print("\n--check-only: nothing written")
        return 0

    if args.out is None:
        print("\nno --out given; nothing written")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(output, encoding="utf-8")
    print(f"\nwrote {args.out} ({report.messages} messages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
