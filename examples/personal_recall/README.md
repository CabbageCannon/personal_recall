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

* **`isSend` decides the speaker**: `1` becomes `我`, `0` becomes `对方`. The speaker-attribution
  safety logic depends on the literal `"我"`, so a wxid or display name is *never* substituted for it;
  the real `senderUsername` is kept in metadata instead.
* **Internal payloads never reach retrieval.** Real exports carry multi-KB `<msg><emoji …/></msg>`
  blobs; these are replaced by a short placeholder (`[表情]`, `[图片]`, …) so the message keeps its
  place on the timeline without polluting the embedding corpus.

Messages are re-sorted chronologically before session building, because sessions are cut from sequence
order: an out-of-order export would otherwise fragment into one chunk per message.

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
  Groundedness: 12 citation(s) | 2 silence claim(s) | 1 attribution flag(s)
    ! this statement claims the record does NOT contain something - the record may simply not
      have been retrieved
    ! 1 statement rests on another person's own account (同学A) - check the answer is not
      borrowing their situation
  ```

  These are **caveats, not verdicts**: without knowing the intended answer the tool cannot say an
  answer is wrong, only that a statement is worth checking. The attribution flag is a strong signal
  (it fires on ~0–1 queries per run); the silence-claim notice is common (~30 % of answers) and is a
  reminder rather than an alarm.

## 3. Reproduce the measurements

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
| `chat_import.py` | real export → canonical corpus |
| `groundedness.py` | gold-free caveats shown by the product |
| `run_baseline.py` | the evaluated runner (all arms) |
| `absence_claims.py`, `attribution_screen.py`, `citation_metrics.py` | groundedness instruments |
| `probe_*.py` | offline, free retrieval experiments |
| `build_*.py` | reproducible corpus construction |
| `PROJECT_STATUS.md` | the living decision log |
