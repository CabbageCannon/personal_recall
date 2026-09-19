"""Conversation identity: what makes one WeChat conversation distinguishable from another.

Why identity is the hard part. A WeFlow export contains **no conversation field at all** — the
``Message`` shape is ``localId/serverId/localType/createTime/isSend/senderUsername/content/...`` and
neither the message nor the ``weflow-message/v1`` envelope records which chat it belongs to. Identity
therefore comes from *outside* the file:

* the export invocation (``weflow-cli export <talker> …``), and
* the exporter's own file name, ``{talker}_messages.json``.

Everything here is built on the **talker** — WeChat's ``StrTalker``, surfaced by ``sessions --json`` as
``ChatSession.username``. A talker is a wxid, or ``…@chatroom`` for a group: stable, unique, and
unaffected by anything the user can edit.

**A display name is never an identity.** Nicknames, remarks and group names are all editable, and two
different contacts can share one. They are recorded here as ``aliases`` for the UI and for search, and
are deliberately not usable as a key: two sessions whose only difference is the display name are still
two different conversations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: Suffix the exporter appends to a JSON export: ``{talker}_messages.json``.
EXPORT_FILENAME_SUFFIX = "_messages.json"

#: The orchestrator records the ``sessions --json`` listing of each shard next to that shard's exports,
#: so identity survives a renamed file and the account report can say which conversations exist.
#: Defined here — the module that has no dependencies at all — because both the exporter (which writes
#: them without importing any parsing code) and the account importer (which reads them) need the names.
SESSION_LISTING_FILENAME = "sessions.json"

#: The account-level manifest: which message shards were *detected* versus actually *exported*.
#: Without it, an account whose largest shard failed to export looks complete, and every answer
#: silently speaks for a smaller past than the user has.
ACCOUNT_MANIFEST_FILENAME = "shard_manifest.json"

#: WeChat marks a group chat by this suffix on the talker.
GROUP_SUFFIX = "@chatroom"

DIRECT = "direct"
GROUP = "group"


def conversation_type_of(talker: str) -> str:
    """``group`` for a chatroom talker, ``direct`` otherwise."""
    return GROUP if GROUP_SUFFIX in (talker or "") else DIRECT


@dataclass(frozen=True)
class ConversationDescriptor:
    """One conversation, keyed by its stable talker id."""

    conversation_id: str
    display_name: str = ""
    conversation_type: str = DIRECT
    aliases: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """What to show a human — never used as a key."""
        return self.display_name or self.conversation_id

    @property
    def has_real_name(self) -> bool:
        """True only when ``display_name`` is a *name*, not the identity spelled again.

        ``label`` deliberately falls back to the talker, which is the right thing for a terminal the
        owner is reading (it is the handle they pass to ``--only``). It is the wrong thing for a
        surface that promises a human-readable label, because a wxid is an internal identifier.

        The failure this exists for was found on a real account, not in a fixture: ``weflow-cli
        sessions --json`` on the tested version returns ``displayName`` **equal to** ``username`` when
        it has no remark or nickname to resolve, so every one of the account's 272 conversations
        carried a "display name" that was the raw talker. A guard of ``if display_name`` therefore
        passed for all of them and the web UI printed a group id where a name belongs. Synthetic
        fixtures never caught it because they always gave the two values different string constants.

        The rule itself lives in one place, :func:`memory.labels.usable_name` — the exporter's contact
        resolution and the labels sidecar apply exactly the same test, and a second copy of it here
        is how the two would drift apart.
        """
        # Imported inside the property rather than at module scope: ``memory.labels`` imports
        # ``GROUP_SUFFIX`` from this module, so a top-level import back would be a cycle between two
        # modules that are both needed while the package loads. Deferring it keeps the dependency
        # one-way at import time and costs a dict lookup per call on a path that is not hot.
        from memory.labels import usable_name

        return bool(usable_name(self.display_name, self.conversation_id))

    def as_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "display_name": self.display_name,
            "conversation_type": self.conversation_type,
            "aliases": list(self.aliases),
            "metadata": dict(self.metadata),
        }


def descriptor_from_session(entry: Mapping[str, Any]) -> ConversationDescriptor | None:
    """Build a descriptor from one conversation-listing entry.

    Two spellings are accepted, because two producers exist and both must round-trip:

    * the exporter's own ``sessions --json`` shape — ``username`` / ``displayName`` / ``type``;
    * :meth:`ConversationDescriptor.as_dict`, which the orchestrator writes into ``sessions.json`` —
      ``conversation_id`` / ``display_name`` / ``conversation_type``.

    Accepting only the first would make a listing this project wrote itself parse to *zero*
    conversations, which is the worst possible failure: an account that looks empty rather than broken.

    ``username``/``conversation_id`` is the talker; any display name is kept as an alias only.
    """
    talker = str(
        entry.get("username") or entry.get("conversation_id") or entry.get("talker") or ""
    ).strip()
    if not talker:
        return None
    display = str(
        entry.get("displayName") or entry.get("display_name") or entry.get("name") or ""
    ).strip()
    aliases: list[str] = []
    if display and display != talker:
        aliases.append(display)
    for key in ("nickname", "remark", "alias", "aliases"):
        value = entry.get(key)
        candidates = value if isinstance(value, (list, tuple)) else [value]
        for candidate in candidates:
            text = str(candidate or "").strip()
            if text and text not in aliases and text != talker:
                aliases.append(text)
    declared_type = entry.get("conversation_type")
    return ConversationDescriptor(
        conversation_id=talker,
        display_name=display,
        conversation_type=(
            declared_type if declared_type in (DIRECT, GROUP) else conversation_type_of(talker)
        ),
        aliases=tuple(aliases),
        metadata={
            key: entry[key]
            for key in ("type", "lastTimestamp", "sortTimestamp", "unreadCount")
            if key in entry
        },
    )


def parse_session_listing(payload: Any) -> tuple[ConversationDescriptor, ...]:
    """Read a ``sessions --json`` payload into descriptors.

    Accepts the shapes the CLI emits (a bare list, or an object carrying ``sessions``) and is
    duplicate-tolerant: the same talker listed twice collapses to one descriptor with its aliases
    unioned, because a conversation is identified by the talker and nothing else.
    """
    if isinstance(payload, Mapping):
        for key in ("sessions", "data", "items"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        return ()

    merged: dict[str, ConversationDescriptor] = {}
    for entry in payload:
        if not isinstance(entry, Mapping):
            continue
        descriptor = descriptor_from_session(entry)
        if descriptor is None:
            continue
        existing = merged.get(descriptor.conversation_id)
        if existing is None:
            merged[descriptor.conversation_id] = descriptor
            continue
        aliases = tuple(dict.fromkeys(existing.aliases + descriptor.aliases))
        merged[descriptor.conversation_id] = ConversationDescriptor(
            conversation_id=existing.conversation_id,
            display_name=existing.display_name or descriptor.display_name,
            conversation_type=existing.conversation_type,
            aliases=aliases,
            metadata={**descriptor.metadata, **existing.metadata},
        )
    return tuple(merged[key] for key in sorted(merged))


def conversation_id_from_export_filename(name: str) -> str | None:
    """Recover the talker from an exporter file name, or ``None`` if it is not one.

    This is the *only* place identity can come from inside an export directory, which is why the
    orchestrator also writes a manifest: a renamed file would otherwise lose its conversation.
    """
    base = str(name).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if not base.endswith(EXPORT_FILENAME_SUFFIX):
        return None
    talker = base[: -len(EXPORT_FILENAME_SUFFIX)].strip()
    return talker or None


def shard_stem(name: str) -> str:
    """Normalise a shard reference to a comparable label.

    The shard detector reports database file names (``MSG0.db``, and its ``-wal``/``-shm`` sidecars)
    while an export is labelled with the stem (``MSG0``). Comparing the raw strings makes every shard
    look missing, which would cry PARTIAL on a complete history. Lives here — with the other naming
    rules, in the module that has no dependencies — so the writer and the reader of an export tree
    cannot disagree about what a shard is called.
    """
    base = str(name).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    for suffix in (".db-wal", ".db-shm", ".db"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def account_label(descriptors: Sequence[ConversationDescriptor]) -> dict[str, str]:
    """``conversation_id -> display label`` for the UI, falling back to the talker."""
    return {d.conversation_id: d.label for d in descriptors}
