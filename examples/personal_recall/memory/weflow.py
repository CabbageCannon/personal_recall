"""WeFlow JSON source adapter: WeFlow export JSON -> ``MemoryEvent``.

The second source adapter (``memory/events.py`` is the plain-text one). Its only job is the mapping
from the exporter's schema to the engine's atomic unit; it contains no retrieval, embedding,
chunking or LLM logic, so the retrieval path stays exactly as evaluated.

Why this exists rather than a conversion script: the real-data smoke test went
``WeFlow JSON -> hand-written script -> canonical TXT -> MemoryEvent``, which works but makes a
scratch script part of the architecture and loses every field the exporter provides. This adapter
consumes the JSON directly and keeps those fields as metadata.

Contract, verified against the raw export shape::

    [
      {"localId": 123, "serverId": "...", "localType": 1, "createTime": 1789xxxxxx,
       "isSend": 1, "senderUsername": "...", "content": "...",
       "rawContent": "...", "parsedContent": "..."},
      ...
    ]

and the versioned envelope a future exporter may emit::

    {"schema": "weflow-message/v1", "source": "...", "generatedAt": "...",
     "coverage": {...}, "messages": [ ... ]}

Both are accepted. Unknown fields are preserved in ``metadata`` rather than rejected, so an exporter
that adds a field does not break ingestion.

Two decisions worth stating explicitly:

* **``isSend`` decides the speaker, not the identity.** ``isSend == 1`` becomes ``"我"`` and
  ``isSend == 0`` becomes ``"对方"``. The speaker-attribution safety logic in the prompt and in
  ``attribution_screen.py`` depends on the literal string ``"我"``; substituting a wxid or a display
  name would silently disable it. The real ``senderUsername`` is kept in metadata instead.
* **Body text is never guessed.** ``parsedContent`` is preferred over ``content`` over ``rawContent``
  because the exporter's own parse is the most likely to be clean. A body that is raw markup (multi-KB
  ``<msg><emoji .../></msg>`` payloads are real) is **replaced by a short placeholder** so internal
  payload never reaches the embedding corpus, while the message still occupies its place on the
  timeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .events import MemoryEvent, ParseResult

#: Envelope key holding the message list when the payload is versioned rather than a bare list.
ENVELOPE_MESSAGES_KEY = "messages"
ENVELOPE_SCHEMA_KEY = "schema"

SELF = "我"
OTHER = "对方"

#: Body fields in preference order (the exporter's parse first, the raw payload last).
BODY_FIELDS: tuple[str, ...] = ("parsedContent", "content", "rawContent")

#: Placeholder tokens WeChat writes for non-text messages, mapped to a message type.
PLACEHOLDER_TYPES: Mapping[str, str] = {
    "[图片]": "image",
    "[照片]": "image",
    "[语音]": "voice",
    "[视频]": "video",
    "[动画表情]": "sticker",
    "[表情]": "sticker",
    "[文件]": "file",
    "[链接]": "link",
    "[位置]": "location",
    "[转账]": "transfer",
    "[红包]": "red_packet",
    "[语音通话]": "call",
    "[视频通话]": "call",
    "[聊天记录]": "chat_record",
    "[小程序]": "miniprogram",
    "[系统消息]": "system",
}

#: Markup detection is two-tier because ``<msg>`` is only the envelope wrapper: the element inside it
#: carries the actual kind. Matching the wrapper first reported a stock emoji as "non-text", which is
#: why the specific tags are searched for first.
SPECIFIC_MARKUP_RE = re.compile(
    r"<\s*(emoji|img|voip|videomsg|location|appmsg|sysmsg)\b", re.IGNORECASE
)
GENERIC_MARKUP_RE = re.compile(r"<\s*msg\b", re.IGNORECASE)

#: What to emit instead of a suppressed payload, keyed by the tag that triggered it.
MARKUP_PLACEHOLDER: Mapping[str, str] = {
    "img": "[图片]",
    "emoji": "[表情]",
    "voip": "[语音通话]",
    "videomsg": "[视频]",
    "location": "[位置]",
    "appmsg": "[链接]",
    "sysmsg": "[系统消息]",
}

#: Emitted when a message carries no readable text at all.
NO_TEXT_PLACEHOLDER = "[非文本消息]"

#: Metadata copied through verbatim when present.
METADATA_FIELDS: tuple[str, ...] = (
    "localId",
    "serverId",
    "localType",
    "messageType",
    "senderUsername",
    "isSend",
    "createTime",
)


@dataclass(frozen=True)
class WeFlowParseResult:
    """Events plus what the adapter had to do to produce them.

    ``skipped`` counts entries that were not messages at all (missing/!unparseable time).
    ``suppressed_payloads`` counts markup bodies replaced by a placeholder — the number that would
    otherwise have polluted the embedding corpus.
    """

    events: tuple[MemoryEvent, ...]
    skipped: int = 0
    suppressed_payloads: int = 0
    missing_text: int = 0
    id_collisions: int = 0
    schema: str | None = None
    message_types: Mapping[str, int] = field(default_factory=dict)

    def as_parse_result(self) -> ParseResult:
        """The engine-wide result shape, for callers that only need events and skips."""
        return ParseResult(events=self.events, skipped_lines=self.skipped)


def unwrap_payload(payload: Any) -> tuple[list[Any], str | None]:
    """Return ``(messages, schema)`` from either a bare list or a versioned envelope."""
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        messages = payload.get(ENVELOPE_MESSAGES_KEY)
        if isinstance(messages, list):
            return messages, payload.get(ENVELOPE_SCHEMA_KEY)
        # A single message object is accepted too: one message is still a valid export.
        if any(field in payload for field in BODY_FIELDS):
            return [payload], payload.get(ENVELOPE_SCHEMA_KEY)
    return [], None


def normalize_body(raw: str) -> str:
    """Collapse a message body to one line, because the canonical form is one message per line."""
    return " ".join(str(raw).replace("\r\n", "\n").replace("\r", "\n").split())


def select_body(entry: Mapping[str, Any]) -> tuple[str, str | None]:
    """Pick the first non-empty body field; return ``(text, field_used)``."""
    for field_name in BODY_FIELDS:
        value = entry.get(field_name)
        if value is None:
            continue
        text = normalize_body(value)
        if text:
            return text, field_name
    return "", None


def classify(text: str) -> tuple[str, str, bool]:
    """Return ``(message_type, text, suppressed)`` for a non-empty body.

    A markup payload is replaced by a placeholder; a body that is exactly a placeholder token keeps
    that token (it is short, human-readable and occupies the right place in the timeline).
    """
    specific = SPECIFIC_MARKUP_RE.search(text)
    if specific:
        placeholder = MARKUP_PLACEHOLDER.get(specific.group(1).lower(), NO_TEXT_PLACEHOLDER)
        return PLACEHOLDER_TYPES.get(placeholder, "non_text"), placeholder, True
    if GENERIC_MARKUP_RE.search(text):
        return "non_text", NO_TEXT_PLACEHOLDER, True

    stripped = text.strip()
    for token, message_type in PLACEHOLDER_TYPES.items():
        if stripped.startswith(token):
            return message_type, stripped, False
    return "text", stripped, False


def build_event_id(
    conversation_id: str,
    server_id: Any,
    local_id: Any,
    position: int,
) -> str:
    """Deterministic id combining conversation + server + local identity.

    ``localId`` alone is explicitly **not** unique — WeFlow documents that it repeats across
    conversations and shards — so it is never used on its own. ``serverId`` is preferred when
    present; when it is missing, ``localId`` scoped by ``conversation_id`` is the stable fallback,
    and a positional marker is the last resort.
    """
    parts = [conversation_id]
    if server_id not in (None, "", 0, "0"):
        parts.append(f"s{server_id}")
    if local_id not in (None, ""):
        parts.append(f"l{local_id}")
    if len(parts) == 1:
        parts.append(f"p{position:05d}")
    return "#".join(str(part) for part in parts)


def _timestamp(create_time: Any) -> datetime | None:
    """``createTime`` is Unix seconds. Milliseconds are tolerated; anything else is rejected."""
    try:
        value = int(create_time)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if value > 10_000_000_000:  # milliseconds
        value //= 1000
    return datetime.fromtimestamp(value, tz=timezone.utc).astimezone().replace(tzinfo=None)


def parse_weflow_events(
    payload: Any,
    conversation_id: str = "weflow",
) -> WeFlowParseResult:
    """Convert a WeFlow export payload into chronologically ordered ``MemoryEvent``s.

    Events are **sorted by timestamp** because ``build_sessions`` consumes sequence order and never
    re-sorts: emitting export order would fragment one conversation into one chunk per message, the
    failure documented in Phase M2. Ties keep exporter order, which makes the sort stable and
    reproducible.
    """
    entries, schema = unwrap_payload(payload)
    events: list[MemoryEvent] = []
    skipped = 0
    suppressed = 0
    missing_text = 0
    seen_ids: dict[str, int] = {}
    collisions = 0
    type_counts: dict[str, int] = {}

    for position, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            skipped += 1
            continue

        timestamp = _timestamp(entry.get("createTime"))
        if timestamp is None:
            skipped += 1
            continue

        text, body_field = select_body(entry)
        if text:
            message_type, text, was_suppressed = classify(text)
            suppressed += int(was_suppressed)
        else:
            message_type, text = "non_text", NO_TEXT_PLACEHOLDER
            missing_text += 1

        is_send = entry.get("isSend")
        sender_name = SELF if str(is_send) == "1" else OTHER

        event_id = build_event_id(
            conversation_id, entry.get("serverId"), entry.get("localId"), position
        )
        if event_id in seen_ids:
            # Never drop a message to satisfy an id invariant: disambiguate and count instead.
            collisions += 1
            seen_ids[event_id] += 1
            event_id = f"{event_id}~{seen_ids[event_id]}"
        else:
            seen_ids[event_id] = 0

        metadata: dict[str, Any] = {"source_type": "weflow", "position": position}
        if body_field:
            metadata["body_field"] = body_field
        if schema:
            metadata["schema"] = schema
        for field_name in METADATA_FIELDS:
            if field_name in entry:
                metadata[field_name] = entry[field_name]

        type_counts[message_type] = type_counts.get(message_type, 0) + 1
        events.append(
            MemoryEvent(
                id=event_id,
                conversation_id=conversation_id,
                timestamp=timestamp,
                sender_name=sender_name,
                text=text,
                message_type=message_type,
                metadata=metadata,
            )
        )

    events.sort(key=lambda event: event.timestamp)
    return WeFlowParseResult(
        events=tuple(events),
        skipped=skipped,
        suppressed_payloads=suppressed,
        missing_text=missing_text,
        id_collisions=collisions,
        schema=schema,
        message_types=type_counts,
    )


def load_weflow_file(path: Any, conversation_id: str | None = None) -> WeFlowParseResult:
    """Read a WeFlow JSON file and parse it. ``conversation_id`` defaults to the file stem."""
    import json
    from pathlib import Path

    file_path = Path(path)
    payload = json.loads(file_path.read_text(encoding="utf-8-sig"))
    return parse_weflow_events(payload, conversation_id or file_path.stem)
