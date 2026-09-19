"""Conversation labels: the name a reader sees, the rule that says when there is none, and the
sidecar that carries it from the exporter's answer to the page.

Three things live here and nothing else:

* :func:`usable_name` — the single rule separating "a name" from "the identity spelled again";
* :func:`resolve_conversation_labels` — which offered name wins for which conversation;
* the sidecar format (:func:`read_label_sidecar` / :func:`write_label_sidecar`).

No subprocess, no ``weflow-cli``, no network, no reading of chat content. The module is deliberately
a leaf with one dependency (``memory.conversations``, for :data:`~memory.conversations.GROUP_SUFFIX`
and the direct/group constants), because three callers depend on it — ``memory.conversations`` itself,
the exporter's contact listing, and the web UI — and a module three callers depend on must not be able
to do anything surprising.

Why this module exists at all
-----------------------------

A talker (``wxid_…``, ``…@chatroom``) is the only thing that identifies a conversation, and it is the
last thing a reader should be shown. The obvious shortcut — "use the display name the listing gives
you" — was measured on a real account and is wrong: ``weflow-cli sessions --json`` returns
``displayName`` **equal to** ``username`` when it has no remark or nickname to resolve, for all 272
conversations of the tested account, and ``contacts --json`` did the same for all 107 of its entries.
A truthiness guard therefore passes for every conversation and the UI prints a wxid where a name
belongs. So the rule is not "is the field non-empty" but :func:`usable_name`, and a conversation with
no real name is *absent* from the result rather than labelled with its id.

That is also why this layer resolves names instead of retrieving by them. A name is decoration: it is
editable, it is not unique (two contacts can share one), and nothing here may reach a chunk's text,
its embedding, or the retrieval configuration. Phase 20.6 is evidence *readability*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from memory.conversations import (
    DIRECT,
    GROUP,
    GROUP_SUFFIX,
    ConversationDescriptor,
    conversation_type_of,
)

__all__ = [
    "CONVERSATION_LABEL_FILENAME",
    "DIRECT",
    "DIRECT_TIERS",
    "DISPLAY_NAME",
    "GROUP",
    "GROUP_TIERS",
    "LABEL_SCHEMA",
    "NICKNAME",
    "OTHER",
    "REMARK",
    "ContactRecord",
    "ConversationLabel",
    "NameOffer",
    "parse_contact_listing",
    "read_label_sidecar",
    "resolve_conversation_labels",
    "usable_name",
    "write_label_sidecar",
]

#: The sidecar's file name inside an account export tree: ``<account>/conversation_labels.json``.
#: It sits next to ``shard_manifest.json``, and like the export tree it is real data (ids and names),
#: so it belongs wherever the tree belongs — ``data/real/`` by convention, which ``.gitignore``
#: excludes. Unlike the manifest it carries identity, which is why it is a separate file: the manifest
#: is the one that gets copied and pasted around, and it must stay counts-only.
CONVERSATION_LABEL_FILENAME = "conversation_labels.json"

#: The sidecar's schema marker. A file without it is read as "no labels" rather than guessed at, so a
#: future format can never be half-understood by this version.
LABEL_SCHEMA = "privrecall-conversation-labels/v1"

# --- the field kinds a name can come from ------------------------------------------------------

#: A name the user typed for this contact themselves. The most specific thing there is.
REMARK = "remark"

#: The contact's own chosen name.
NICKNAME = "nickname"

#: Whatever the exporter resolved for display — the generic fallback field.
DISPLAY_NAME = "displayName"

#: Any other genuinely human-readable name field. Deliberately a short list of known spellings rather
#: than "any string on the entry": a session entry's strings also include type codes and timestamps,
#: and a fallback that accepts any string would happily label a conversation "1758000000".
OTHER = "other"

#: Which fields are read as which kind. Order is irrelevant here — precedence is by *kind*, in the
#: tier lists below, never by the order a dict happened to hand the fields over.
_KEY_KINDS: tuple[tuple[str, str], ...] = (
    ("remark", REMARK),
    ("remarks", REMARK),
    ("remarkName", REMARK),
    ("remark_name", REMARK),
    ("nickname", NICKNAME),
    ("nickName", NICKNAME),
    ("nick_name", NICKNAME),
    ("nick", NICKNAME),
    ("displayName", DISPLAY_NAME),
    ("display_name", DISPLAY_NAME),
    ("name", DISPLAY_NAME),
    ("alias", OTHER),
    ("aliases", OTHER),
)

#: The talker spellings accepted in a contact listing, mirroring ``memory.conversations``. ``username``
#: is what ``weflow-cli contacts --json`` and ``sessions --json`` emit; the other two are this
#: project's own spellings, so a listing it wrote itself also parses.
_TALKER_KEYS: tuple[str, ...] = ("username", "conversation_id", "talker")

#: Containers a payload may wrap its entries in.
_CONTAINER_KEYS: tuple[str, ...] = ("contacts", "sessions", "data", "items")

#: Precedence for a **direct** chat, most specific first. Within a tier, the kinds are tried in the
#: order written (so a nickname beats a generic display name), and ties inside one kind are broken by
#: sorted value — see :func:`_pick_name`.
DIRECT_TIERS: tuple[tuple[str, ...], ...] = (
    (REMARK,),
    (NICKNAME, DISPLAY_NAME),
    (OTHER,),
)

#: Precedence for a **group** chat: the group's own name, or nothing. A group has no "remark tier" —
#: the name a group shows is the name its members gave it, and anything else about a group that is a
#: string here is its id.
GROUP_TIERS: tuple[tuple[str, ...], ...] = ((DISPLAY_NAME,),)


# --- the one rule ------------------------------------------------------------------------------


def usable_name(name: Any, conversation_id: Any) -> str:
    """``name`` when it is a label a person may read, otherwise ``""``.

    The single canonical rule, used by three things that must agree: ``ConversationDescriptor.
    has_real_name``, the resolver below, and the sidecar reader. A name is rejected when it is

    * empty or only whitespace — there is nothing to show;
    * equal to the conversation id — the identity spelled again, which is what the tested
      ``weflow-cli`` emits for every conversation it cannot resolve a remark or nickname for;
    * anything containing ``@chatroom`` — a group id is an identifier even when it is spelled
      differently from the conversation it labels, and a group a person named never ends in it.

    A rejected name returns ``""``, never ``None`` and never the id: "no label" has to be one value
    that every caller can test with ``if``.
    """
    text = str(name or "").strip()
    if not text:
        return ""
    identity = str(conversation_id or "").strip()
    if identity and text == identity:
        return ""
    if GROUP_SUFFIX in text:
        return ""
    return text


# --- what a listing offers, and what it resolves to ---------------------------------------------


@dataclass(frozen=True)
class NameOffer:
    """One name a listing offered, and the field kind it came from (``remark``, ``nickname``, …)."""

    kind: str
    value: str


@dataclass(frozen=True)
class ContactRecord:
    """A talker and every name some listing offered for it, each tagged with its field kind.

    A record is *not* an identity and not a conversation: it is what one listing knew about one
    talker. Two records for the same talker are merged by :func:`parse_contact_listing`, and only
    :func:`resolve_conversation_labels` decides which of the offered names is the one to show.
    """

    conversation_id: str
    offers: tuple[NameOffer, ...] = field(default_factory=tuple)

    def offered(self, kinds: Sequence[str]) -> tuple[NameOffer, ...]:
        """The offers whose kind is one of ``kinds``, in the order they were received."""
        wanted = set(kinds)
        return tuple(offer for offer in self.offers if offer.kind in wanted)

    def merged_with(self, other: "ContactRecord") -> "ContactRecord":
        """Union the offers of two records for the same talker, first-seen order preserved."""
        seen = {(offer.kind, offer.value) for offer in self.offers}
        extra = tuple(offer for offer in other.offers if (offer.kind, offer.value) not in seen)
        return ContactRecord(conversation_id=self.conversation_id, offers=self.offers + extra)

    def as_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "offers": [{"kind": offer.kind, "value": offer.value} for offer in self.offers],
        }


@dataclass(frozen=True)
class ConversationLabel:
    """A conversation and the name to show for it — decided, not guessed."""

    conversation_id: str
    label: str
    kind: str = DIRECT
    #: The field kind that supplied the label (``remark``, ``nickname``, ``displayName``, ``other``).
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "label": self.label,
            "kind": self.kind,
            "source": self.source,
        }


# --- reading a listing -------------------------------------------------------------------------


def _entries_of(payload: Any) -> list[Any]:
    """The entry list inside a payload, or ``[]`` for anything unrecognisable.

    Accepts a bare list, an object wrapping one under ``contacts``/``sessions``/``data``/``items``,
    and — because a listing that parses to *zero* conversations looks exactly like an account with no
    conversations — a single entry object that carries a talker key of its own.
    """
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, Mapping):
        return []
    for key in _CONTAINER_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    if any(isinstance(payload.get(key), str) and payload.get(key).strip() for key in _TALKER_KEYS):
        return [payload]
    return []


def _talker_of(entry: Mapping[str, Any]) -> str:
    for key in _TALKER_KEYS:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _texts_of(value: Any) -> tuple[str, ...]:
    """Every non-empty string in ``value``, whether it is one string or a list of them."""
    candidates = value if isinstance(value, (list, tuple)) else [value]
    texts: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            texts.append(candidate.strip())
    return tuple(texts)


def _offers_of(entry: Mapping[str, Any]) -> tuple[NameOffer, ...]:
    """Every name the entry offers, tagged by kind. Unknown fields are ignored, not guessed at."""
    offers: list[NameOffer] = []
    seen: set[tuple[str, str]] = set()
    for key, kind in _KEY_KINDS:
        if key not in entry:
            continue
        for text in _texts_of(entry[key]):
            if (kind, text) in seen:
                continue
            seen.add((kind, text))
            offers.append(NameOffer(kind=kind, value=text))
    return tuple(offers)


def parse_contact_listing(payload: Any) -> tuple[ContactRecord, ...]:
    """Read a ``contacts --json`` (or ``sessions --json``) payload into contact records.

    Never raises: a payload that is not a listing, an entry that is not an object, a talker that is a
    number, a name field that is a dict — each is skipped rather than fatal. A listing is decoration
    for the UI, and no shape of it may cost the account its ability to answer.

    Duplicate talkers merge, with the first-seen order of their offers preserved, so the same listing
    always produces the same records regardless of how the exporter ordered its fields.
    """
    merged: dict[str, ContactRecord] = {}
    for entry in _entries_of(payload):
        if not isinstance(entry, Mapping):
            continue
        talker = _talker_of(entry)
        if not talker:
            continue
        record = ContactRecord(conversation_id=talker, offers=_offers_of(entry))
        existing = merged.get(talker)
        merged[talker] = record if existing is None else existing.merged_with(record)
    return tuple(merged[talker] for talker in sorted(merged))


# --- resolving names ---------------------------------------------------------------------------


def _pick_name(
    offers: Sequence[NameOffer],
    tiers: Sequence[Sequence[str]],
    conversation_id: str,
) -> NameOffer | None:
    """The winning offer for one conversation, or ``None`` when nothing is a name.

    By *kind*, in the tier order given — never by the order the fields arrived in. Inside one kind,
    the candidates are sorted, so two listings that agree on the set of names resolve to the same
    label even when they hand them over in a different order. Without that, a shard ordering change
    would silently rewrite a card's header.
    """
    for tier in tiers:
        for kind in tier:
            values = sorted(
                {
                    usable_name(offer.value, conversation_id)
                    for offer in offers
                    if offer.kind == kind
                }
                - {""}
            )
            if values:
                return NameOffer(kind=kind, value=values[0])
    return None


def _tiers_for(kind: str) -> Sequence[Sequence[str]]:
    return GROUP_TIERS if kind == GROUP else DIRECT_TIERS


def resolve_conversation_labels(
    descriptors: Iterable[ConversationDescriptor],
    records: Iterable[ContactRecord],
) -> dict[str, ConversationLabel]:
    """``conversation_id -> ConversationLabel`` for the conversations that have a real name.

    Precedence, by field kind and nothing else:

    * **direct chat** — ``remark`` → ``nickname`` → ``displayName`` → any other human-readable name
      field → nothing;
    * **group chat** — the group's display name → nothing.

    A conversation is **absent** from the result when no offered name is usable. It is never mapped
    to ``None``, never to its own id, and a talker id or a ``…@chatroom`` string can never become a
    label even if a listing offers one. Two conversations that share a display name stay two entries,
    because the result is keyed by talker, not by name.

    The conversation's own ``display_name`` counts as a ``displayName``-kind offer, so a tree whose
    ``sessions.json`` is the only listing still resolves — that is what the web UI falls back to when
    there is no sidecar.
    """
    index: dict[str, ContactRecord] = {}
    for record in records:
        talker = str(getattr(record, "conversation_id", "") or "").strip()
        if not talker:
            continue
        offers = tuple(
            offer
            for offer in (getattr(record, "offers", ()) or ())
            if isinstance(offer, NameOffer)
        )
        existing = index.get(talker)
        if existing is None:
            index[talker] = ContactRecord(talker, offers)
        else:
            index[talker] = existing.merged_with(ContactRecord(talker, offers))

    resolved: dict[str, ConversationLabel] = {}
    for descriptor in descriptors:
        talker = str(getattr(descriptor, "conversation_id", "") or "").strip()
        if not talker:
            continue
        declared = getattr(descriptor, "conversation_type", "")
        kind = declared if declared in (DIRECT, GROUP) else conversation_type_of(talker)
        offers = list(index[talker].offers) if talker in index else []
        display = getattr(descriptor, "display_name", "")
        if isinstance(display, str) and display.strip():
            offers.append(NameOffer(kind=DISPLAY_NAME, value=display.strip()))
        chosen = _pick_name(offers, _tiers_for(kind), talker)
        if chosen is not None:
            resolved[talker] = ConversationLabel(
                conversation_id=talker, label=chosen.value, kind=kind, source=chosen.kind
            )
    return resolved


# --- the sidecar --------------------------------------------------------------------------------


def read_label_sidecar(path: Path) -> dict[str, ConversationLabel]:
    """Read a labels sidecar, or ``{}`` for anything that is not a readable one of ours.

    Missing, unreadable (permissions, a directory in its place), empty, malformed JSON, a JSON
    document that is not an object, a missing or different ``schema`` marker, a ``labels`` entry that
    is not a list — every one of those means "no labels", because a decoration must never be able to
    stop the product from answering. An absent sidecar and a broken one are the same thing to a
    caller, and both are normal.

    Each label is re-checked with :func:`usable_name` on the way out, so a hand-edited or truncated
    file cannot smuggle a talker id into a card as a "name".
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, Mapping) or raw.get("schema") != LABEL_SCHEMA:
        return {}
    entries = raw.get("labels")
    if not isinstance(entries, list):
        return {}

    labels: dict[str, ConversationLabel] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        talker = str(entry.get("conversation_id") or "").strip()
        usable = usable_name(entry.get("label"), talker)
        if not talker or not usable:
            continue
        declared = entry.get("kind")
        labels[talker] = ConversationLabel(
            conversation_id=talker,
            label=usable,
            kind=declared if declared in (DIRECT, GROUP) else conversation_type_of(talker),
            source=str(entry.get("source") or ""),
        )
    return labels


def write_label_sidecar(path: Path, labels: Mapping[str, ConversationLabel]) -> Path:
    """Write ``labels`` to ``path`` and return it. Sorted, so re-running rewrites the same bytes.

    Every label is passed through :func:`usable_name` first: a label that is not a name is dropped
    rather than written, so the file can never disagree with the rule that produced it and a reader
    downstream does not have to trust this writer to have applied it.

    The document holds real identity — talkers and names — so it belongs in the account export tree,
    which is git-ignored. Its schema marker is written even when there are no labels: "we asked and
    there was no name" is a result, and it must not look like a file this version cannot read.
    """
    target = Path(path)
    entries: list[dict[str, Any]] = []
    for talker in sorted(labels):
        label = labels[talker]
        identity = str(getattr(label, "conversation_id", "") or talker).strip()
        usable = usable_name(getattr(label, "label", ""), identity)
        if not usable:
            continue
        kind = getattr(label, "kind", "")
        entries.append(
            ConversationLabel(
                conversation_id=identity,
                label=usable,
                kind=kind if kind in (DIRECT, GROUP) else conversation_type_of(identity),
                source=str(getattr(label, "source", "") or ""),
            ).as_dict()
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"schema": LABEL_SCHEMA, "labels": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target
