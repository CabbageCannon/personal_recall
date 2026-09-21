"""Source Adapter v1: plain-text chat export -> MemoryEvent.

The engine must not depend on any single chat-export format (§29). This adapter
is the generic TXT one; WeChat/QQ/JSON/CSV adapters can be added later without
touching the retrieval or eval layers, because everything downstream consumes
``MemoryEvent`` rather than raw text.

Expected line format (one message per line)::

    [YYYY-MM-DD HH:MM] 说话人: 内容

Blank lines separate conversations/episodes and are ignored. Unparseable lines are
skipped and counted, never silently dropped without a trace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

MESSAGE_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] (?P<sender>[^:]{1,16}): (?P<text>.+)$"
)
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M"

#: The two speaker roles. ``isSend == 1`` is :data:`SELF_ROLE`; every other WeFlow message is
#: :data:`OTHER_ROLE`. A **role is not a name and a name is not a role**: the role is a fact the
#: export states, and it is what decides the display label (``memory.senders``), while the identity
#: behind an ``other`` speaker is a separate piece of data (:attr:`MemoryEvent.speaker_id`).
SELF_ROLE = "self"
OTHER_ROLE = "other"


@dataclass(frozen=True)
class MemoryEvent:
    """One atomic chat message — the citation unit of the recall engine.

    Three speaker fields, deliberately separate, because one string cannot carry all three facts:

    * :attr:`speaker_role` — ``"self"``/``"other"``. What the export *says* about who spoke.
    * :attr:`speaker_id` — the stable identity (WeFlow's ``senderUsername``). **Data, never a
      label**: it must not reach a prompt, a response, a log or a rendered line.
    * :attr:`sender_name` — the display label that :attr:`line` renders, i.e. everything the LLM and
      the evidence card ever see. The only one of the three that is allowed to be a string a person
      reads, and the only one ``memory.senders`` rewrites.
    """

    id: str
    conversation_id: str
    timestamp: datetime
    sender_name: str
    text: str
    message_type: str = "text"
    reply_to: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    #: ``"self"``/``"other"`` for a source that states it (the WeFlow adapter reads ``isSend``).
    #: Empty for a source that does not — the plain-text adapter names its speakers directly and has
    #: no role to declare, so it declares none rather than asserting one it cannot know.
    speaker_role: str = ""
    #: The speaker's stable identity, when the source provides one. Kept out of every rendered string
    #: — a label derived from this would be the identity spelled again (see ``memory.senders``).
    speaker_id: str = ""

    @property
    def line(self) -> str:
        """The original chat line, so evidence stays traceable to the source."""
        return f"[{self.timestamp.strftime(TIMESTAMP_FORMAT)}] {self.sender_name}: {self.text}"


@dataclass(frozen=True)
class ParseResult:
    """Events plus a count of lines that did not look like messages."""

    events: tuple[MemoryEvent, ...]
    skipped_lines: int


def parse_txt_events(
    text: str,
    conversation_id: str = "txt",
) -> ParseResult:
    """Parse a text chat export into MemoryEvents, preserving file order.

    Ids are deterministic (``<conversation_id>#00001``) so the same input always
    yields the same ids — required for reproducible citations and tests.
    """
    events: list[MemoryEvent] = []
    skipped = 0

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        match = MESSAGE_RE.match(line)
        if match is None:
            skipped += 1
            continue

        timestamp = datetime.strptime(match.group("ts"), TIMESTAMP_FORMAT)
        events.append(
            MemoryEvent(
                id=f"{conversation_id}#{len(events) + 1:05d}",
                conversation_id=conversation_id,
                timestamp=timestamp,
                sender_name=match.group("sender").strip(),
                text=match.group("text").strip(),
                metadata={"source_type": "txt", "source_line": len(events) + 1},
            )
        )

    return ParseResult(events=tuple(events), skipped_lines=skipped)
