# Personal Recall Engine

Ask questions about your own chat history and get answers **with the evidence** — every claim traced
back to the original chat lines, plus an honest statement of what the record does *not* support.

This is the product surface of the recall project. The measurement record — every arm, every negative
result, every defect found — lives in [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

All commands below run from this directory (`examples/personal_recall`).

## Setup

The engine calls DeepSeek for generation only; retrieval is fully local.

```bash
# repo-root .env must contain DEEPSEEK_API_KEY
# local embedding model is expected at D:\AIModels\bge-small-zh-v1.5
python -m pytest tests -q          # offline: no API key needed
```

## 1. Bring your own chat log

The engine reads one message per line, as `[YYYY-MM-DD HH:MM] speaker: text`. Real exports rarely look
like that, and feeding one in directly used to index **zero messages with no error**. Convert first:

```bash
python chat_import.py --list-formats
python chat_import.py --input my_export.txt --out data/my_chat.txt
```

The importer recognises four layouts, converts to the canonical form, and **verifies the result by
re-parsing it through the engine's own adapter** — nothing is reported OK unless the round trip is
clean. An unrecognised file gets a diagnosis (per-layout match counts and the lines that failed)
instead of silence.

### WeFlow JSON exports (WeChat 3.x)

A WeFlow JSON export is read **directly**, with no canonical-TXT step in between:

```bash
python recall.py "我今天中午吃了什么？" --corpus my_export.json
python recall.py "我最后用了哪个库？" --corpus my_export.json --source-format weflow
```

`--source-format` defaults to `auto`, which infers the adapter from the file extension (`.json` →
WeFlow, anything else → text). The adapter accepts either a bare JSON list of messages or a versioned
envelope with a `messages` list, so an exporter that starts emitting a schema wrapper keeps working.

Two properties matter for correctness:

* **`isSend` decides the speaker**: `1` becomes `我`. The speaker-attribution safety logic depends on
  the literal `"我"`, so a wxid or display name is *never* substituted for it. In a direct chat the
  other side is `对方`; in a **group** each member gets a deterministic pseudonym instead — `成员A`,
  `成员B`, … by first appearance — because one `对方` for everybody made a group's evidence
  unattributable. The real `senderUsername` is kept as data (`speaker_id`, and in metadata) and is
  never rendered.
* **Internal payloads never reach retrieval.** Real exports carry multi-KB `<msg><emoji …/></msg>`
  blobs; these are replaced by a short placeholder (`[表情]`, `[图片]`, …) so the message keeps its
  place on the timeline without polluting the embedding corpus.

Messages are re-sorted chronologically before session building, because sessions are cut from sequence
order: an out-of-order export would otherwise fragment into one chunk per message.

### More than one shard (WeChat 3.x)

WeChat 3.x splits one conversation across `Msg/Multi/MSG0.db`, `MSG1.db`, `MSG2.db`, … The exporter
reads **one shard per run**, so whichever shard is configured *is* the visible history — point it at
`MSG0` and everything in `MSG2` is invisible. Export each shard and point the engine at the directory
to merge them:

```bash
python recall.py "我今天中午吃了什么？" --corpus shards/ --shard-dir "C:\...\Msg\Multi"
```

Shards are merged **before** session building, so one conversation spanning two databases stays one
retrieval unit with the global latest timestamp. Ordering comes from `createTime` — never from the
file name or its modification time — and ties break deterministically by shard label and position.
Messages are deduplicated only on the exporter's `serverId`, which survives a message appearing in two
shards; without a `serverId` there is no safe cross-shard identity (`localId` repeats by design), so
those messages are all kept and counted in the report rather than guessed at.

`--shard-dir` is optional and read-only. When given, the report compares the `MSG*.db` files that
**exist** against the number actually exported, and says so loudly when the history is partial:

```
  shard check   : 3 MSG*.db present ['MSG0.db', 'MSG1.db', 'MSG2.db']; 2 exported
  WARNING       : only 2 of 3 message shards were exported, so this history is PARTIAL - ...
```

That warning exists because **No Data Loaded ≠ No Memory Exists**: a smaller history must never be
mistaken for a smaller past.

## 2. Ask a question

```bash
python recall.py "我之前说的那个数据库最后到底用了没？"
python recall.py "小王什么时候推荐我用 Supabase 的？" --corpus data/my_chat.txt --json
```

Output has three parts:

* **Answer** — the conclusion, with `[来源 N]` on every factual statement.
* **Evidence** — one card per cited source, quoting the original chat lines with timestamps,
  participants and the chunk id, so a claim can be checked against the record.
* **Groundedness** — caveats a reader should know:

  ```
  Groundedness: 12 citation(s) | 1 binding mismatch(es) | 2 silence claim(s) | 1 attribution flag(s)
    ! 1 citation binding mismatch(es): quoted text is not in the source it cites, but is in
      another retrieved source
    ! this statement claims the record does NOT contain something - the record may simply not
      have been retrieved
    ! 1 statement rests on another person's own account (同学A) - check the answer is not
      borrowing their situation

  citation: quoted text «我又点了拌粉» is not in the cited source (Source [1]) but appears in Source [0]
    in sentence: 11:26 你说“我又点了拌粉” [来源 1]
  ```

  These are **caveats, not verdicts**: without knowing the intended answer the tool cannot say an
  answer is wrong, only that a statement is worth checking. Their measured base rates differ a lot,
  which is how to read them:

  | caveat | base rate | how to read it |
  | ------ | --------- | -------------- |
  | citation binding mismatch | 0–1 per 36 answers | **alarm** — the answer quotes text that is not in the chunk it cites |
  | attribution flag | 0–1 per 36 answers | **alarm** — a statement rests on someone else's own situation |
  | silence claim | ~30 % of answers | **reminder** — the record may simply not have been retrieved |
  | unverified quote | ~2–6 per 36 answers | *not surfaced* — usually the model's own paraphrase in quotes |

  The binding check is anchored on **verbatim quotes only**, because a first version that scored every
  sentence by lexical similarity measured ~17 % precision: multi-fact summary sentences legitimately
  cite several chunks, and bigram overlap produced spurious winners.

## 3. Ask from a browser

The same engine behind one local page: question box, answer, evidence cards, caveats. It reads the
account export tree, so initialise that first (once, from the real WeChat data directory):

```bash
python export_account.py --multi-dir "C:\Users\<you>\WeChat Files\<wxid>\Msg\Multi" --out data/real/account
python web.py
```

Then open **<http://127.0.0.1:8000>**.

* **The data lives in `data/real/account`** — a subdirectory per message shard, one
  `<talker>_messages.json` per conversation, plus the `sessions.json` listings that give the evidence
  cards their conversation names. It is real chat, so it stays where `.gitignore` already covers it.
* **Start it with `python web.py`.** `--account <dir>` points at another export tree and
  `PERSONAL_RECALL_ACCOUNT` does the same for a shell that always uses one; `--port` changes the port.
* **The index is built once, at startup** — parse and embed the whole account, then every question
  reuses it. If that fails, the server still starts and the page says why rather than answering from
  nothing.
* **There is no upload, no login and no history.** Ingestion stays a CLI step, the page keeps no
  record of what you asked, and the server binds `127.0.0.1` only — there is no flag for anything
  else, because this process serves private chat.

Answers come from `recall.answer_question`, the same path the CLI prints from, so the page and the
terminal cannot disagree about what was retrieved or cited.

### Conversation names on the evidence cards

A card header reads "我, 对方 · 2025-05-12" in a direct chat, or "我, 成员A, 成员B · 2025-05-12" in a
group. Unambiguous inside one conversation, and not across an account of 272. The name that fixes that is **not** in the export — WeFlow records no
conversation field at all — and on a fresh `weflow-cli` install the only name it can offer is the
talker id again, so the page shows no header rather than printing a wxid. Ask the exporter once and
record what it knows:

```bash
python sync_conversation_labels.py --account data/real/account_full
```

It writes `conversation_labels.json` inside the account tree and prints **coverage** — how many
conversations resolved, split direct/group — never a name. The page prefers that file and falls back
to the tree's own listing when it is missing or empty, so nothing changes for an export that has no
sidecar. Names are decoration only: they are not put into chunk text, not embedded, and not used to
retrieve.

## 4. Audit a full export before trusting it

Embedding a whole account is the slow step (272 conversations took ~15 minutes here), so check the
tree first — the audit parses and segments it without touching the model, which is also what makes a
failure unambiguous: ingestion, not memory.

```bash
python audit_account.py --account data/real/account_full \
    --shard-dir "C:\Users\<you>\WeChat Files\<wxid>\Msg\Multi"
```

It prints how many shards were detected versus exported, whether the export is `PARTIAL` (a shard that
was never exported, or a conversation filter), the conversation/message/chunk counts, the coverage
span, and whether any chunk mixes two conversations — the one result that must always be `PASS`. It
writes nothing unless `--json <path>` is given, and its output carries counts and short digests only:
never a wxid, a display name or a line of chat, so it is safe to paste into an issue.

## 5. Acceptance check on your own history

The stress corpus answers *"how good is the engine under controlled conditions"*. This answers *"does
it work on my history, on questions I care about"* — 15–20 questions across six categories, with
everything a human needs to judge each answer recorded beside it.

```bash
cp eval_questions.template.json eval_questions.json   # then fill in the questions
python real_eval.py --questions eval_questions.json --account data/real/account --out real_eval_results.json
python real_eval.py --questions eval_questions.json --out real_eval_results.json --report-only
```

The runner **refuses to start while the template still holds placeholders**, so a half-filled question
set cannot quietly spend API calls. Each result row carries the answer, every retrieved chunk (time
range, participants, text), the citations, the groundedness caveats, and the latency — plus four
`labels` left `null` for you: `answer_correct`, `retrieval_correct`, `citation_binding_correct`,
`attribution_correct`. Re-running preserves labels already filled in.

The default real-data path is account-wide: it calls the same `build_account_session()` and
`answer_question()` used by the product. `--corpus` remains available for the older single-conversation
or merged-shard acceptance runs.

## 6. Reproduce the measurements
Everything in `PROJECT_STATUS.md` is regenerable. Retrieval is deterministic under `--workflow
no-rewrite`, which is what makes these comparisons exact rather than statistical.

```bash
# the corpora
python build_stress_v2.py --check-only          # rebuild corpus v2 from its checked-in parts
python validate_stress_dataset.py --corpus data/stress_chats_v2.txt

# an arm (paid: one generation per query)
python run_baseline.py --dataset stress_v2 --chunking session --k 20 --hybrid --hybrid-pool 30 \
    --workflow no-rewrite --answer-prompt cited-attributed --temperature 0 \
    --max-output-tokens 8192 --tag myarm

# the metrics
python citation_metrics.py --results stress_v2_myarm_results.json
python absence_claims.py --results stress_v2_myarm_results.json --corpus data/stress_chats_v2.txt
python attribution_screen.py --results stress_v2_myarm_results.json

# does the product still match what was measured? (fast subset)
python verify_product_parity.py --limit 6
```

## What the engine promises, and where it can still fail

The promise is **no evidence, no memory claim**. Measured on the hardest corpus (36 queries, 2.3×
distractor pressure), the adopted configuration scores **31/36 with zero fabricated memories** and
100 % citation validity — and under deliberate evidence ablation it declines rather than inventing,
2 fabrications → 0 after the speaker-attribution clause.

Known limits, all recorded with numbers in `PROJECT_STATUS.md`:

* **Five queries are unreachable** by any retrieval configuration tested (the decisive line sits
  outside the window). More context buys at most two of them for 2.5× the cost.
* **Text only.** Attachments, images and voice notes are not ingested.
* **Four export layouts.** Anything else needs a conversion rule.
* Answers are **not textually reproducible** run to run; retrieval is.

## Layout

| path | role |
| ---- | ---- |
| `recall.py` | product CLI: question → answer + evidence + caveats |
| `web.py`, `webapp/` | the local web UI: the same answer, in a browser |
| `export_account.py` | WeChat account → the export tree both front ends read |
| `sync_conversation_labels.py` | exporter's contact names → the evidence-card headers |
| `audit_account.py` | export tree → completeness and boundary report, no model needed |
| `chat_import.py` | real export → canonical corpus |
| `groundedness.py` | gold-free caveats shown by the product |
| `run_baseline.py` | the evaluated runner (all arms) |
| `absence_claims.py`, `attribution_screen.py`, `citation_metrics.py` | groundedness instruments |
| `probe_*.py` | offline, free retrieval experiments |
| `build_*.py` | reproducible corpus construction |
| `PROJECT_STATUS.md` | the living decision log |
