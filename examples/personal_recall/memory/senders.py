"""Group sender identity: telling one group member apart from another in the evidence.

The defect this exists for, found on a real account. ``memory/weflow.py`` decides the speaker from
``isSend`` alone — ``1`` becomes ``"我"`` and everything else becomes ``"对方"``. That is exactly right
for a direct chat, where there are only ever two speakers and the role *is* the identity. In a group it
collapses every member into one string: A, B and C all speaking produces ``对方, 对方, 对方``, so the
evidence cannot say who said what and a question of the form "who did X" has no answer in the record.
The acceptance run confirmed it — asked who did something, the answers reported that the record only
labels everyone "对方".

Four things are separated here, and only the last is a display string (see ``memory.events``):

* **role** — ``speaker_role``, ``"self"`` or ``"other"``. What ``isSend`` actually means.
* **identity** — ``speaker_id``, the exporter's ``senderUsername``. Stable, kept as data, never shown.
* **display** — ``speaker_display``, the exporter's optional ``senderDisplay``. A *candidate* name.
* **label** — ``sender_name``, what ``MemoryEvent.line`` renders. The only thing this module changes.

The rule, in order:

* **self** -> ``我``, always, in both kinds of conversation. The speaker-attribution safety logic in the
  prompt and in ``attribution_screen.py`` matches this exact string, so substituting anything else —
  a wxid, a display name, a pseudonym — would silently disable it.
* **direct conversation, other** -> ``对方``, unchanged. Role alone distinguishes the two speakers, and
  keeping the string identical is what makes this phase a strict no-op for every already-evaluated
  direct corpus. A direct chat's *conversation* name is a different concern (``memory.conversations``,
  ``memory.labels``) and is not decided here.
* **group conversation, other** -> the member's own label, resolved in this order:

  1. a **usable** ``speaker_display`` — the human name the exporter offered;
  2. otherwise a deterministic pseudonym, ``成员A``, ``成员B``, … (Excel-column style, continuing
     ``…Y, Z, AA, AB, …``);
  3. and ``对方`` when the message carries no usable **identity** at all, because then the export does
     not say who spoke and no label may be invented for it.

  "Usable" is not ``if senderDisplay:``. It is :func:`memory.labels.usable_name`, the project's single
  canonical "is this a name or the identity spelled again" rule, which rejects empty/whitespace, a
  value equal to the conversation id, a value equal to the speaker's own id, and anything containing a
  chatroom suffix. That rule lives **in** ``memory.labels`` and is only *called* from here; this module
  contributes the precedence around it and nothing else. The consequence is the property that matters:
  a raw internal identity can never reach the model just because the field carrying it is called
  "Display".

Three things this pass has to get right, each of them a way the naive version is wrong
------------------------------------------------------------------------------------------

**(a) Resolve across the whole conversation, never per event.** One member may carry a display on some
messages and none on others. Resolved per event, that member would render as ``张三`` on one line and
``成员A`` on the next — one person wearing two labels inside one conversation, which is the collapse
this phase removes, re-introduced one message at a time. So the pass first builds one
``speaker_id -> label`` map for the entire stream, then applies it to every event.

**(b) Pseudonyms must not drift.** The ``成员A/B/C…`` sequence is assigned by first appearance over
**all** group non-self senders, whether or not they end up using a display. Numbering only the senders
that lack a display would mean that adding a name to one member renumbers everybody else — the same
corpus rendering differently between two runs, and a citation that cannot be reproduced.

**(c) Duplicate displays must stay distinguishable.** Two different ids can both resolve to the same
human name; a nickname is not unique and the exporter resolves it per contact. Rendering ``小王`` twice
is an identity collapse — the exact defect this phase exists to remove — so when one conversation has
two members under one candidate name, each is qualified with the pseudonym it already had:
``小王（成员A）``, ``小王（成员B）``. The qualifier is a pseudonym and never a wxid, a hash or a chatroom
id, so the disambiguation is deterministic and adds no identity of its own.

And a sender that somehow carries several displays is resolved deterministically too: the most frequent
one wins, ties broken by first appearance. The measured export has zero such conflicts, but a rule that
depends on that staying true is not a rule.

Where it is applied, and why that placement is the subtle part
-------------------------------------------------------------

Over the **whole conversation's event stream**, never per shard. Applied from inside
``parse_weflow_events`` this would be wrong for exactly the histories ``memory.shards`` exists for: a
conversation the exporter split across ``MSG0``/``MSG1``/``MSG2`` would get shard A's first speaker as
``成员A`` and shard B's *different* first speaker as ``成员A`` too — two people with one label, which is
worse than the defect it replaces because it reads as a fact rather than as an absence.

So it is applied at the three points where a conversation's stream is already final:

* ``memory.shards.merge_shard_events`` — after the merge, so the pass sees every shard at once. This is
  the shard-directory path and, through it, ``memory.account.import_account``.
* ``memory.processor.WeFlowSessionProcessor`` and ``recall.count_corpus_events`` — the two single-file
  paths. Here the whole conversation is in the one file being read, so the stream *is* the conversation
  and labelling it cannot split anything; that is why a per-file pass is correct here and wrong there.

Nothing in this module touches retrieval. It renames a speaker inside text that has already been
selected, so the embedder, FAISS, hybrid search, BM25, RRF, ``k``, the hybrid pool, session chunking,
the reranker, the workflow and the prompt are all exactly as evaluated.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from .conversations import GROUP, conversation_type_of
from .events import OTHER_ROLE, SELF_ROLE, MemoryEvent
from .labels import usable_name
from .weflow import OTHER, SELF

__all__ = [
    "SENDER_LABEL_PREFIX",
    "assign_sender_labels",
    "column_label",
]

#: Prefix of an assigned pseudonym. ``成员A`` reads as "member A" and carries nothing else: a label
#: derived from the raw sender id would be that id spelled out, and a hash or a random number would be
#: neither stable to read nor reproducible to cite.
SENDER_LABEL_PREFIX = "成员"


def column_label(index: int) -> str:
    """``0 -> "A"``, ``25 -> "Z"``, ``26 -> "AA"``, ``27 -> "AB"`` — Excel-column style.

    Chosen over a fixed alphabet because the sequence has to keep going: a group with more members than
    letters is unusual but real, and a scheme that ran out would either reuse a label (two people, one
    name) or fall back to something derived from the id.
    """
    if index < 0:
        raise ValueError("a column label is indexed from 0")
    letters = ""
    value = index
    while True:
        letters = chr(ord("A") + value % 26) + letters
        value = value // 26 - 1
        if value < 0:
            return letters


def _identity_of(event: MemoryEvent, conversation_id: str) -> str:
    """The sender's identity, or ``""`` when the export did not state one.

    Two things are not an identity:

    * an empty ``senderUsername`` — the export does not say who spoke, and two such messages cannot be
      shown to be the same person or different people, so giving them member labels would either invent
      a distinction or assert an identity the export does not contain;
    * the **conversation's own id**. Measured on this project's older export: 562 of 589 files named the
      conversation as the sender and not one named a sender, so treating that field as an identity
      stamps a single ``成员A`` across a whole group — "exactly one other person said all of this",
      which reads as a fact rather than as an absence. The patched exporter no longer does this, and the
      guard stays: it costs one comparison and it is the difference between an honest ``对方`` and an
      invented member the day some exporter writes the talker into the sender field again.

    Same family of rule as ``labels.usable_name`` — the identity is never a name — but this one is
    about the *identity field*, so it lives here next to the precedence it feeds.
    """
    identity = str(event.speaker_id or "").strip()
    if not identity or identity == conversation_id:
        return ""
    return identity


def _member_labels(events: Sequence[MemoryEvent], conversation_id: str) -> dict[str, str]:
    """``speaker_id -> rendered label`` for one group conversation's members.

    Built over the **whole stream** in one pass, which is the point (see the module docstring, (a)):
    the map is keyed by identity, so a member who carries a display on one message and none on another
    is one person with one label everywhere. Nothing here is decided per event.

    Events without a usable identity are not members and are absent from the result; the caller renders
    those ``对方``. The candidate name is judged by ``labels.usable_name``, passed the speaker's own id
    as the identity it must not merely repeat.
    """
    order: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    first_seen: dict[str, dict[str, int]] = {}

    for position, event in enumerate(events):
        if event.speaker_role != OTHER_ROLE:
            continue
        identity = _identity_of(event, conversation_id)
        if not identity:
            continue
        if identity not in counts:
            # First appearance over *every* member, named or not: pseudonyms are assigned here and
            # consumed later, so a member gaining a display never renumbers the others (b).
            order.append(identity)
            counts[identity] = {}
            first_seen[identity] = {}
        display = usable_name(event.speaker_display, conversation_id, identity)
        if display:
            counts[identity][display] = counts[identity].get(display, 0) + 1
            first_seen[identity].setdefault(display, position)

    pseudonyms = {
        identity: f"{SENDER_LABEL_PREFIX}{column_label(index)}" for index, identity in enumerate(order)
    }
    chosen: dict[str, str] = {}
    for identity in order:
        offered = counts[identity]
        if not offered:
            chosen[identity] = pseudonyms[identity]
            continue
        # Most frequent, ties broken by first appearance — deterministic on the stream, never on dict
        # iteration order, and reproducible from the export alone.
        chosen[identity] = min(offered, key=lambda name: (-offered[name], first_seen[identity][name]))

    return _disambiguated(chosen, pseudonyms)


def _disambiguated(
    labels: Mapping[str, str],
    pseudonyms: Mapping[str, str],
) -> dict[str, str]:
    """Qualify any label two members of one conversation would otherwise share (c).

    A name is not unique and a group is not a namespace: the exporter resolves ``senderDisplay``
    per contact, so two different ids can both offer ``小王``. Rendering it for both is exactly the
    collapse this phase exists to remove, and picking one of them to keep would be worse — it would
    silently attribute one person's words to another. So both keep the name and gain the pseudonym they
    already had: ``小王（成员A）`` and ``小王（成员B）``.

    Both halves are already in the record: the name came from the export, the qualifier from the
    conversation's own member order. No wxid, no hash, no chatroom id — a disambiguator that carried an
    identity would defeat the field it is being appended to.

    The rule is stated over the resulting labels rather than over displays, so it covers every way two
    members could end up rendering the same string, not only the two-displays-one-name case.
    """
    shared: dict[str, int] = {}
    for label in labels.values():
        shared[label] = shared.get(label, 0) + 1
    return {
        identity: f"{label}（{pseudonyms[identity]}）" if shared[label] > 1 else label
        for identity, label in labels.items()
    }


def assign_sender_labels(
    events: Sequence[MemoryEvent],
    conversation_id: str,
) -> list[MemoryEvent]:
    """Return ``events`` with each ``sender_name`` set to the label its conversation should show.

    A pure function of ``(events, conversation_id)``: the same stream always produces the same labels,
    which is what lets a citation be reproduced from the export alone. It reads ``speaker_role``,
    ``speaker_id`` and ``speaker_display`` and writes only ``sender_name``, so applying it twice is the
    same as applying it once.

    ``events`` is assumed to be in the conversation's chronological order and is **not** re-sorted —
    "first appearance" means first in the order given, and every caller feeds a stream that
    ``parse_weflow_events`` or ``merge_shard_events`` has already ordered. Re-sorting here would be a
    second opinion about chronology, and the one thing the merge guarantees is that there is only one.

    The precedence itself is described at the top of this module; what is worth restating is what is
    *not* renamed. A stream whose source declares no role at all — the plain-text adapter names its
    speakers directly — is a strict no-op, so this function cannot corrupt a corpus it was never meant
    for even if a future caller wires it in by mistake.
    """
    is_group = conversation_type_of(conversation_id) == GROUP
    # Resolved once for the whole conversation, before a single event is renamed (a).
    members = _member_labels(events, conversation_id) if is_group else {}
    labelled: list[MemoryEvent] = []

    for event in events:
        if event.speaker_role == SELF_ROLE:
            # Never anything else, in either kind of conversation: the prompt's attribution clause and
            # ``attribution_screen.py`` both key off this literal string.
            name = SELF
        elif is_group and event.speaker_role == OTHER_ROLE:
            # ``对方`` when the export states no identity: absent from the map means there was nothing
            # to key a name on, not that the member was forgotten.
            name = members.get(_identity_of(event, conversation_id), OTHER)
        elif event.speaker_role == OTHER_ROLE:
            # A direct conversation has one other side by definition, whatever ids its messages carry.
            name = OTHER
        else:
            name = event.sender_name

        labelled.append(event if event.sender_name == name else replace(event, sender_name=name))

    return labelled
