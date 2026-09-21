"""Group sender identity: telling one group member apart from another in the evidence.

The defect this exists for, found on a real account. ``memory/weflow.py`` decides the speaker from
``isSend`` alone — ``1`` becomes ``"我"`` and everything else becomes ``"对方"``. That is exactly right
for a direct chat, where there are only ever two speakers and the role *is* the identity. In a group it
collapses every member into one string: A, B and C all speaking produces ``对方, 对方, 对方``, so the
evidence cannot say who said what and a question of the form "who did X" has no answer in the record.
The acceptance run confirmed it — asked who did something, the answers reported that the record only
labels everyone "对方".

Three things are separated here, and only the third is a display string (see ``memory.events``):

* **role** — ``speaker_role``, ``"self"`` or ``"other"``. What ``isSend`` actually means.
* **identity** — ``speaker_id``, the exporter's ``senderUsername``. Stable, kept as data, never shown.
* **label** — ``sender_name``, what ``MemoryEvent.line`` renders. The only thing this module changes.

The rule:

* **self** -> ``我``, always, in both kinds of conversation. The speaker-attribution safety logic in the
  prompt and in ``attribution_screen.py`` matches this exact string, so substituting anything else —
  a wxid, a display name, a pseudonym — would silently disable it.
* **direct conversation, other** -> ``对方``, unchanged. Role alone distinguishes the two speakers, and
  keeping the string identical is what makes this phase a strict no-op for every already-evaluated
  direct corpus.
* **group conversation, other** -> a deterministic pseudonym, assigned per conversation, so members are
  distinguishable: ``成员A``, ``成员B``, … by **first appearance in that conversation's chronological
  event order**, continuing ``…Y, Z, AA, AB, …`` (Excel-column style). Never a hash, never a random
  number, never the raw id: a label derived from the id would be the identity spelled again, and a label
  that changed between two runs would make the evidence unreproducible.

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
from typing import Sequence

from .conversations import GROUP, conversation_type_of
from .events import OTHER_ROLE, SELF_ROLE, MemoryEvent
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


def assign_sender_labels(
    events: Sequence[MemoryEvent],
    conversation_id: str,
) -> list[MemoryEvent]:
    """Return ``events`` with each ``sender_name`` set to the label its conversation should show.

    A pure function of ``(events, conversation_id)``: the same stream always produces the same labels,
    which is what lets a citation be reproduced from the export alone.

    ``events`` is assumed to be in the conversation's chronological order and is **not** re-sorted —
    "first appearance" means first in the order given, and every caller feeds a stream that
    ``parse_weflow_events`` or ``merge_shard_events`` has already ordered. Re-sorting here would be a
    second opinion about chronology, and the one thing the merge guarantees is that there is only one.

    Only two things are renamed: a declared ``self`` speaker, and an ``other`` speaker **in a group**.
    Everything else is returned untouched, which has three consequences worth stating:

    * a **direct** conversation keeps ``对方`` exactly as it renders today — the role is enough to
      distinguish its two speakers, and the evaluated corpora must keep rendering byte for byte what
      they rendered;
    * a stream whose source declares **no** role at all — the plain-text adapter names its speakers
      directly — is a strict no-op, so this function cannot corrupt a corpus it was never meant for
      even if a future caller wires it in by mistake;
    * in a group, an ``other``-speaker whose export carried **no** ``senderUsername`` — **or** whose
      ``senderUsername`` is the conversation's own id — keeps ``对方``. Two such messages cannot be
      shown to be the same person or different people, so giving them member labels would either
      invent a distinction or assert an identity the export does not contain. ``对方`` is the honest
      answer, and it can never collide with an assigned ``成员X`` label.

      The second half of that rule is not hypothetical; it is the state of this project's own data.
      Every export in a real 589-file account was swept: **562 carry a ``senderUsername`` equal to the
      talker**, none carries a value that differs, and none has more than one distinct non-self
      sender — the exporter writes the *conversation* into the sender field. Without this guard every
      group in a real account would render as one ``成员A``, asserting that a single other person said
      everything. So on this exporter the fix preserves the honest ``对方`` rather than gaining
      attribution, and the model above is what makes that a one-line change the day a real sender id
      arrives.
    """
    is_group = conversation_type_of(conversation_id) == GROUP
    assigned: dict[str, str] = {}
    labelled: list[MemoryEvent] = []

    for event in events:
        if event.speaker_role == SELF_ROLE:
            # Never anything else, in either kind of conversation: the prompt's attribution clause and
            # ``attribution_screen.py`` both key off this literal string.
            name = SELF
        elif is_group and event.speaker_role == OTHER_ROLE:
            identity = str(event.speaker_id or "").strip()
            if not identity or identity == conversation_id:
                # The conversation's own id is not a sender. Measured on this project's real export:
                # 562 of 589 exports carry ``senderUsername == talker`` and **not one** carries a
                # distinct sender id, so the exporter is writing the conversation into the sender
                # field. Treating that as an identity would stamp every member of every group with the
                # same ``成员A`` — an assertion that exactly one other person spoke, which is worse
                # than the ``对方`` it replaces because it reads as a fact rather than as an absence.
                # Same family of rule as ``labels.usable_name``: the identity is never a name.
                name = OTHER
            else:
                existing = assigned.get(identity)
                if existing is None:
                    # Assigned on first appearance, and only ever here, so one id maps to one label
                    # for the whole conversation and no two ids can share one.
                    existing = f"{SENDER_LABEL_PREFIX}{column_label(len(assigned))}"
                    assigned[identity] = existing
                name = existing
        else:
            name = event.sender_name

        labelled.append(event if event.sender_name == name else replace(event, sender_name=name))

    return labelled
