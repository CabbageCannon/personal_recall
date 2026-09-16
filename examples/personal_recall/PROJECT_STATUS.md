# Personal Recall Engine — Project Status

Living decision log. Facts only from real eval output; no estimated numbers.
Branch: `personal-recall` · Base: `CabbageCannon/quivr`

---

## Phase map

| Phase | Scope | Status |
|---|---|---|
| 0 | Small-corpus baseline (unmodified Quivr Dense RAG) | ✅ done |
| 0.5 | Recall Stress Corpus + Stress Baseline + dataset audit | ✅ done — baseline visibly fails, failure attributed to retrieval |
| 1 | MemoryEvent / Source Adapter + conversation-aware chunking | ✅ done — kept: additive on top of the window control |
| 2 | Retrieval trace / Evidence representation on structured memory | ⏸ (reranking evaluated in round 2: negative, not adopted) |
| 3 | Hybrid retrieval (BM25 + dense + RRF) | ✅ done — adopted as A5 (+1 PASS, +0.98 coverage); reranker measured negative |
| 4 | Temporal retrieval | ⏸ |
| 5 | Entity-aware recall (Person / Alias) | ⏸ |
| 6 | Multi-evidence / state evolution | ⏸ |
| 7 | Grounded generation (EvidenceItem, citation, abstention) | 🔄 justified: first generation-side failures measured (over-abstention s025, misreading s007) |
| 8 | Persistence (PostgreSQL + pgvector) | ⏸ |
| 9 | Product UI | ⏸ |
| 10 | Multimodal recall | ⏸ |
| 11 | Optional skills / agent layer | ⏸ |

Phase order after 0.5 is **evidence-driven**: the stress baseline decides the first real change.

---

## Baseline invariants (must not drift while A/B testing)

```
TXT → SimpleTxtProcessor → fixed-character chunks (400 / 100)
→ local BGE-small-zh-v1.5 (D:\AIModels\bge-small-zh-v1.5)
→ FAISS → RetrievalConfig(k=5)
→ Quivr LangGraph RAG → DeepSeek (model id as configured in run_baseline.py)
→ Answer + sources
```

Every A/B keeps `k=5`, the embedder, chunking, prompt and retrieval path frozen unless that
exact component is the experiment.

## Phase 0 result (real, from `baseline_summary.json`, 34 queries)

| Metric | Value |
|---|---|
| Answerable / unanswerable | 30 / 4 |
| Retrieval Hit@5 | 30/30 = **1.000** |
| Strict Hit@5 | 29/30 = 0.967 |
| Avg evidence coverage | 0.965 |
| Avg strict coverage | 0.937 |
| Weighted coverage | 0.961 |
| Answer PASS / PARTIAL / FAIL | 34 / 0 / 0 |
| Unsupported claim rate | 0/34 = **0.0** |
| Latency avg / min / max (ms) | 7507.6 / 3618.7 / 15242.1 |
| Chunk-boundary-affected queries | 3 (`q011`, `q027`, `q031`) |

**Phase 0 conclusion (decision):** the small corpus cannot discriminate methods. 6,919 chars
→ 23 chunks, so Top-5 covers **21.7 %** of the entire memory space; entity tokens
(Neon / Tailscale / WireGuard / Cloudflare Pages) act as unique anchors. Dense+FAISS+Top-5
therefore always returns the gold evidence, and generation then looks perfect.
⇒ Do **not** use the small corpus to justify any algorithm change. Keep it as a regression
guard only.

## Phase 0.5 decisions (this phase)

| # | Decision | Why |
|---|---|---|
| D1 | Long-lived branch `personal-recall`, pushed to origin | §git workflow; no short-lived branch sprawl |
| D2 | Stress data lives beside the small data and is fully isolated (`stress_*` files, `--dataset stress`) | must never overwrite the 34-query baseline |
| D3 | Corpus size target = **≥140 chunks**, i.e. ≈45k Chinese chars (≈135 KB) | the dataset card's "50–100 KB" assumed 1 byte/char; Chinese is 3 bytes/char. The chunk count is what governs retrieval selectivity (Top-5 = 3.3 % of space vs 21.7 % today), so the chunk target wins |
| D4 | Facts are frozen in `data/STRESS_CORPUS_SPEC.md` (§3 timelines + §4 verbatim anchors) and the text is written by 4 parallel writers | cross-year coherence of state evolution is the whole point; parallel authorship otherwise breaks it |
| D5 | `stress_queries.json` keeps the **exact** 6-key schema of `queries.json` | no schema drift; difficulty is *measured* by the validator, not self-declared in the data |
| D6 | 36 queries = 12 regression (old categories) + 24 hard (10 stress categories) | separates "did we break the easy stuff" from "did we actually improve recall" |
| D7 | Difficulty gate before spending API budget: offline validation + measured diagnostics (lexical overlap, unique-anchor chunk count, near-duplicate distractor count) + independent audit | §15: never run 36 paid queries against an unaudited dataset |

## Hypotheses the stress baseline must test

| # | Hypothesis | Falsified if |
|---|---|---|
| H1 | With ~150 chunks, dense Top-5 stops covering the gold evidence for state-change questions | Hit@5 stays ≈1.0 |
| H2 | Recurring-entity distractor messages cause *partial* evidence (right entity, wrong date/state) | coverage stays ≈1.0 |
| H3 | "latest state" questions get answered with an outdated state when several states are retrieved | answer PASS stays 100 % |
| H4 | Near-duplicate wording (sleep/idle for both Render and Neon) causes entity confusion | no wrong-entity retrievals |
| H5 | Unanswerable questions about plausible-but-absent facts trigger unsupported claims | unsupported rate stays 0 |

## Risks / watch items

- R1 Corpus reads repetitive/artificial → retrieval difficulty comes from wording artifacts rather than genuine ambiguity. Mitigation: the audit must show the difficulty comes from *competing states*, not from typos.
- R2 Gold evidence is written by paraphrase → validator fails; mitigation: evidence must be verbatim corpus lines.
- R3 Unanswerable questions may be accidentally answerable → audit greps every distinctive token.
- R4 `core/quivr_core/*` currently carries the user's own uncommitted study annotations; this branch commits only `examples/personal_recall/**` and leaves those files untouched.
- R5 DeepSeek cost: the stress run is paid; gated behind the offline audit (D7).

## Next step

Integrate the 4 corpus parts → structural check → author queries → offline validate →
independent difficulty audit → **then** run the unchanged stress baseline → failure analysis.

---

# Phase 0.5 — measured facts

## Dataset (from `stress_validation.json`, real output)

| | small | stress |
|---|---|---|
| File | `data/chats.txt` | `data/stress_chats.txt` |
| Bytes / chars | 11,789 / 6,919 | **94,262 / 49,457** |
| Messages / episodes | 200 / 26 | **1,216 / 100** |
| Chunks @ 400/100 | 23 | **165** |
| Top-5 share of memory space | 21.7 % | **3.0 %** |
| Window | 2024-03 → 2026-05 | 2024-01-08 → 2026-08-28 |
| Queries | 34 (30 answerable) | **36 (34 answerable)**, 15 categories |

Dataset card: `data/STRESS_CORPUS_SPEC.md` (roster, 6 entity-state timelines, 44 verbatim anchor
lines, distractor rules, query rules).
Isolation: `--dataset {small,stress}` on both runner and summarizer; brain name per dataset;
proved by stub harness (index built only from that dataset's corpus, writes only that dataset's
files) and by a no-arg run reproducing `baseline_summary.json` byte-for-byte.

## Offline gate 1 — validator (`validate_stress_dataset.py`)

11/11 hard checks PASS. Difficulty diagnostics: max question↔gold character-bigram overlap
**0.457** (0 queries above the 0.5 lexical-shortcut threshold); 8 queries have a unique rare
anchor (`rare_chunks == 1`); 2 queries have no near-duplicate distractor.
Splitter fidelity: the chunk counter is cross-checked against `quivr_core`'s real
`recursive_character_splitter` over 6,015 length/config cases — 0 mismatches.

## Offline gate 2 — dense retrieval dry run (`probe_dense_difficulty.py`)

Real BGE-small-zh-v1.5 + the real chunker + cosine Top-5, **no LLM, no API cost**.

| | small | stress (pre-repair) |
|---|---|---|
| Hit@5 (answerable) | 30/30 = 100 % | **27/34 = 79.4 %** |
| Avg evidence coverage | 96.5 % | **56.6 %** |
| Gold first appears at dense rank (median / max) | 1 / 3 | 1 / **12** |
| Retrieval misses | 0 | **7** |
| Partial evidence | 3 | **16** |

The probe reproduces the small corpus's official numbers exactly (100 % / 96.5 %), which is what
licenses using it as a pre-flight gate. Reading: the stress corpus fails mostly by **partial
evidence** — the retriever finds the entity early but misses the other state-bearing lines.

## Offline gate 3 — independent adversarial audit (`audit_report.md`)

**Verdict: PASS-WITH-FIXES.** 31/36 gold answers verified TRUE, 108/108 evidence lines verbatim
grounded, zero meta-text leaks, zero duplicate messages, both `s035`/`s036` verified genuinely
unanswerable. Found and acted on:

| # | Finding | Action |
|---|---|---|
| A1 | **3 poisoned gold answers** (s007, s019, s016) — a corpus-faithful model would be graded wrong (internship actually started late April, not May; 小汪's gym arc starts before 2025-02) | repaired (blocking) |
| A2 | 9 of 24 "hard" queries answerable from a single 400-char chunk; only 7 are genuine multi-hop | question-side rewrites that force ≥2 chunks |
| A3 | Both alias traps self-decoding (`王哥（就是小王）`, `阿伟（我室友）` appeared in the only line mentioning each) | glosses removed from the corpus; aliases now resolve from context only (adjacency / event identity) |
| A4 | `stress_validation.json` claimed 169 chunks; the real chunker yields **165** (validator decoded raw bytes, so CRLF counted an extra `\r` per line) | validator normalises newlines like `quivr_core`'s text-mode reader |
| A5 | Corpus contradicts spec §5.7: 7 lines state a final state outright, and the 2026-08-20 recap states four final states in ~2 chunks | **not** fixed: several are anchor lines (A44) and the recap is what makes `latest_state` answerable at all. Recorded as risk R6; the difficulty claim is restated honestly below |

**Honest difficulty claim (post-audit):** the measured difficulty of this dataset is
**retrieval** difficulty (miss + partial evidence), not state-assembly difficulty. Only 7 of the
24 hard queries require genuine multi-evidence assembly. State assembly becomes a measurable
axis only after Phase 1/6 work; do not over-claim it from Phase 0.5.

## Added decisions

| # | Decision | Why |
|---|---|---|
| D8 | The paid baseline is gated behind offline validation **and** an independent adversarial audit | §15; the audit caught 3 poisoned golds that no structural check can see |
| D9 | Poisoned golds and EASY-query difficulty are fixed **question-side**, not by rewriting the corpus | corpus edits would invalidate the validator/probe evidence and risk breaking anchor lines |
| D10 | `expected_answer` truth is an audited property, not an assumption | a wrong gold inverts the metric silently |
| D11 | Freeze the corpus hash in the eval record once the paid run starts | makes the result reproducible and detects later drift |

## Risks

- **R6 (new)** Corpus still contains single-line final-state statements + a recap episode, so a
  lucky single-chunk retrieval can shortcut a state question. Mitigated question-side; revisit if
  the stress baseline still scores too high.
- **R7 (new)** Corpus §3.5 (spec) and the corpus text disagree on the internship month; the corpus
  is the source of truth and the golds now follow the corpus. The spec text was left as the
  original authoring intent — noted so it is not mistaken for a query error again.
- **R8 (new)** The 2026-08-20 recap episode makes four final states reachable in ~2 chunks; it is
  load-bearing for `latest_state` answerability, so removing it would create false unanswerables.

---

# Phase 0.5 — RESULT (real paid run, 36 queries, unchanged baseline)

Frozen dataset hashes at run time:
`stress_chats.txt C98B5E76…FEE59`, `stress_queries.json A8693C0F…EB6D7C`.

## Headline comparison — same frozen pipeline, only the corpus differs

| Metric | Phase 0 small | Phase 0.5 stress |
|---|---|---|
| Queries | 34 (30 answerable) | 36 (34 answerable) |
| Chunks | 23 | 165 |
| Retrieval Hit@5 | 30/30 = **100 %** | 28/34 = **82.35 %** |
| Avg evidence coverage | 96.5 % | **53.19 %** |
| Weighted coverage | 96.1 % | **48.25 %** |
| Answer PASS | 34/34 = **100 %** | 20/36 = **55.56 %** |
| PARTIAL / FAIL | 0 / 0 | **9 / 7** |
| Unsupported claim rate | 0 % | **0 %** |
| Latency avg / max (ms) | 7,508 / 15,242 | 13,525 / 64,521 |
| Retrieval misses | 0 | 6 (`s006 s013 s020 s021 s028 s030`) |

Worst categories by coverage: `latest_state` 16.7 %, `temporal_state_change` 27.1 %,
`negative_evidence` / `multi_evidence` 37.5 %, `implicit_reference` 45.8 %.
Regression categories held up: `exact_fact` / `exact_keyword` / `time_recall` = 100 % coverage, 100 % PASS.

## Failure attribution (manual grading, 36/36, `stress_manual_labels.json`)

| | count |
|---|---|
| Queries whose gold facts were **absent from the retrieved chunks** (`evidence_sufficient=false`) | **16** |
| Failures where the evidence WAS sufficient (pure generation failures) | **0** |
| PASS among the 20 sufficient queries | **20/20** |
| Unsupported claims | **0/36** |
| Suspected gold errors after the audit repair | **0** |

The correlation is perfect: every non-PASS answer is a query where retrieval did not deliver the
gold lines, and the generator was never wrong when it had them. Observed behaviour under
insufficient evidence is **grounded abstention or under-answering**, not hallucination — including
both `unanswerable` queries (`s035` thesis title, `s036` restaurant name), which declined cleanly
and invented nothing.

## How much of the gap is just the Top-5 window? (`probe_recall_at_k.py`)

| group | cov@5 | cov@10 | cov@20 | cov@50 | cov@100 |
|---|---|---|---|---|---|
| all answerable | 53.4 % | 74.5 % | 82.1 % | 94.9 % | 99.3 % |
| the 16 insufficient queries | 37.5 % | 58.9 % | 71.9 % | 90.6 % | 98.4 % |
| the 20 sufficient queries | 67.6 % | 88.4 % | 91.2 % | 98.6 % | 100 % |

**The gold evidence is in the ranking — it sits just outside Top-5.** +21 points of coverage are
available from window/ranking alone (k=5 → 10). Offline, free, reproducible.

## The largest real failure mode (Phase 0.5 conclusion)

> **Retrieval incompleteness on multi-episode state arcs.** The retriever reliably finds the
> *topic* (Hit@5 82 %) but not the *complete set of state-bearing lines* for an arc that spans
> several years: coverage drops 96.5 % → 53.2 %, and 16/36 answers are starved of evidence.

Concrete mechanisms, observed in the retrieved sources:

1. **Naming lines missed** — `s020` (all 4 gold lines absent; the retrieved chunks were the
   migration *process* discussion without the DB names) → the model abstained instead of guessing.
2. **Temporal conflation of same-shaped arcs** — `s021` blends the 2025 Tailscale success
   (2025-06-30) with the 2026 WireGuard failure (2026-06-11) and concludes success.
3. **Current state retrieved, trajectory missed** — `s022` gets the 2026 state right from one
   chunk but misses the 2025-11 / 2025-12 lines that the gold requires.
4. **Not a generation problem** — 0 generation failures, 0 unsupported claims; the 400-char
   fragment boundary frequently cuts the anchoring line out of the retrieved window.

## Next phase decision

Failure is retrieval-side, and the missing evidence is retrievable — so the next change must
attack **evidence completeness per retrieval slot**, not generation.

| Option | Expected effect | Verdict |
|---|---|---|
| **Phase 1: session/conversation-aware chunking + MemoryEvent/MemoryChunk** | one slot = one coherent session (participants + time range) instead of a 400-char fragment; the anchor line arrives with its context, and 5 slots cover 5 sessions rather than 5 fragments of ~2 | **do this first** |
| Control: same corpus at k=10/20 (no structural change) | +21 pts coverage for free | **run as the control arm** — Phase 1 must beat it |
| Phase 3 hybrid BM25+RRF | targets exact-token recall, but the failing questions deliberately avoid entity names and dense already finds the topic | defer; test later, possibly combined |
| Phase 7 abstention / false-memory control | already 0 % unsupported and 2/2 correct abstentions | **deprioritised** — no measured need yet |

**Phase 1 A/B plan:** arm A = session-aware chunking at k=5 (apples-to-apples with this frozen
baseline); arm B = current 400/100 chunks at k=10 and k=20; same 36 stress queries, same embedder,
same model. Success = coverage@5 strictly above arm B's coverage@5, with PASS rate as the headline.
Keep arm A only if it beats the window control; otherwise the honest conclusion is that the window
was the bottleneck and chunking bought nothing.

---

# Phase 1 — RESULT (conversation-aware chunking; 4 timed arms, same 36 queries)

Deliverable: `memory/` (Source Adapter TXT→`MemoryEvent`, `SessionConfig`+`build_sessions`→
`MemoryChunk`, `ConversationSessionProcessor`), 14 offline tests, `--chunking/--k/--tag` experiment
knobs on the runner+summarizer. The frozen baseline path is untouched: with no arguments both
scripts still reproduce their committed outputs byte-for-byte, and a non-baseline configuration
**cannot** write baseline files (the runner refuses without `--tag`).

Segmentation rule (deterministic, no NLP): new session on a new calendar day, on a silence gap
> 6 h, or when the next message would exceed the size budget (900 chars). Corpus → **101 session
chunks** (avg 487 chars, median 13 chat lines) vs 165 fixed 400/100 slices.

## Arm matrix (all four arms: same corpus, queries, BGE embedder, DeepSeek model, prompt)

| arm | retrieval unit | k | chunks | Hit@k | avg coverage | PASS | PARTIAL | FAIL | score /36 | unsupported |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 frozen baseline | fixed 400/100 | 5 | 165 | 82.35 % | 53.19 % | 20 (55.6 %) | 9 | 7 | 24.5 | 0 % |
| A2 Phase 1 | session | 5 | 101 | 88.24 % | 62.75 % | 24 (66.7 %) | 9 | 3 | 28.5 | 0 % |
| A3 window control | fixed 400/100 | **10** | 165 | 88.24 % | 70.10 % | 26 (72.2 %) | 7 | 3 | 29.5 | 0 % |
| **A4 session + window** | session | **10** | 101 | **97.06 %** | **84.07 %** | **29 (80.6 %)** | 6 | **1** | **32.0** | 0 % |

Offline probe grid (free, real BGE, no LLM) — the same conclusion on the retrieval side:

| unit | cov@5 | cov@10 | cov@20 |
|---|---|---|---|
| fixed 400/100 | 53.4 % | 74.5 % | 82.1 % |
| session | 67.2 % | 82.4 % | 91.7 % |

## Decision: **KEEP** conversation-aware chunking

* Session chunking wins at **every matched k** — end-to-end (k5: 24 vs 20 PASS; k10: 29 vs 26 PASS)
  and offline (k5/k10/k20) — so it is **additive on top of the window control**, not a substitute
  for it. The pre-registered rule ("A must beat the window control") is satisfied in the combined
  configuration.
* It also removes chunk-boundary evidence loss: strict coverage == tolerant coverage (62.75 %)
  under session units, vs 51.47 % strict < 53.19 % tolerant under fixed slices.
* Character efficiency: session@k10 (~5,000 chars) reaches coverage that fixed only reaches at
  k=20 (~8,000 chars).
* New reference configuration for subsequent phases: **session units + k=10** →
  PASS 80.6 %, coverage 84.07 %, FAIL 1/36, unsupported 0 %.
* The frozen **small-corpus** baseline stays fixed/k=5 and is still the regression guard.

## Honest caveats (recorded, not hidden)

* **Two queries regressed** because a whole-session context can amplify a wrong frame: `s007`
  (PASS→PARTIAL: the 2026-05-17 session carries 学长's "别拖到五月才动手", which pulled the answer
  back to a May framing) and `s016` (PARTIAL→FAIL: two gym-adjacent 2024 sessions were retrieved
  without the decisive 2024-06-15 line, producing a confident "2024 = talk only" timeline).
  Net effect is still strongly positive (+9 PASS vs A1), but long units are not free.
* `s007` is a grading judgment call: if its closing "五月才明显推进投递" is treated as out-of-scope
  editorialising, A4 is 30/5/1 with **zero** losses vs the fragment arms.
* `k=10` is an experiment knob, not a silent change: the A4 numbers are reported as a *new
  reference*, never as the frozen baseline.
* Cost: 144 paid queries this phase (4 arms × 36).

## Phase 1 failure analysis — the remaining bottleneck moved

Five queries never reached PASS in any arm: `s016 s020 s023 s028 s030`. Every one is a
multi-slice state question missing **one specific time slice** of a multi-year arc, e.g.
`s020` needs the 2025-06-08/06-21 Railway naming lines, `s023` needs the 2024-07-21 Neon
recommendation, `s028` needs the 2024-11-19 account line, `s030` needs the 2024-10/11 lines,
`s016` needs the 2024-06-15 gym line.

`probe_recall_at_k.py --chunking session` shows those lines **are** in the ranking, just deep:

| group | cov@5 | cov@10 | cov@20 | cov@50 | cov@101 |
|---|---|---|---|---|---|
| all answerable | 67.2 % | 82.4 % | 91.7 % | 97.1 % | 100 % |
| the 5 never-PASS queries | 35 % | 50 % | 65 % | 85 % | 100 % |

⇒ The bottleneck is no longer the retrieval **unit** (Phase 1 fixed that) but the **selection**:
the decisive line is ranked outside the top 10 in a 101-unit pool. `s007`/`s019` are separate
generation slips — all their gold lines were retrieved.

## Next phase decision

| Option | Expected effect | Verdict |
|---|---|---|
| **Phase 3-lite: widen candidates (k≈50) + rerank to 5** | the sweep proves the hard tail sits at rank 20–50; a ranker that selects 5 of 50 attacks exactly that | **do this next** |
| Phase 3 hybrid BM25 + RRF (no reranker) | the failing questions deliberately avoid entity tokens, so lexical matching cannot recover them on its own | defer / combine later |
| Phase 4 temporal parsing | would help time-scoped questions, but the hard tail fails on *entity-implied* slices, not on date arithmetic | defer |
| Phase 6 multi-evidence aggregation | plausible, but it presupposes the same re-ranking ability | after the reranker |

**Phase 2 A/B plan:** reference = A4 (session + k=10). Test session + k=50 + rerank→5 with the same
36 queries; keep the reranker only if PASS/coverage improve over A4, and report its latency cost.
A rerank step that cannot beat simply widening k must not be merged.

---

# Phase 2 — candidate widening + local cross-encoder re-rank: **NEGATIVE RESULT (not adopted)**

Hypothesis (from the Phase 1 failure analysis): the decisive line sits at dense rank 20–50, so a
re-ranker that selects 5–10 of 50 candidates should lift evidence coverage.

Built (and kept — see "What was kept"): local re-ranking support in the framework, a
`--rerank-model/--candidate-k` experiment knob, offline probe support, 13 new tests.
Model: `BAAI/bge-reranker-base` (1.1 GB, local, CPU, ~25 pairs/s).

## Measured (offline probes, real BGE + real cross-encoder, no API cost)

Retrieval units = session chunks (101), candidate pool = dense top-50.

| strategy | slots | Hit@k | avg coverage |
|---|---|---|---|
| flat dense top-5 (A2-equivalent) | 5 | 94.1 % | 67.2 % |
| **re-rank 50 → keep 5** | 5 | 94.1 % | **67.2 %** (identical) |
| flat dense top-10 (A4-equivalent) | 10 | 97.1 % | **82.4 %** |
| **re-rank 50 → keep 10** | 10 | 97.1 % | **80.9 %** (worse) |

Per-query at 5 slots: **8 improved, 9 worsened, 19 unchanged** — e.g. `s028`'s gold chunk moved
dense rank 10 → 1 (coverage 0.00 → 1.00) but `s006`'s moved 4 → 17 (1.00 → 0.00). The re-ranker is
genuinely reordering (scores span 0.00–0.99 and it does promote gold chunks), it just is not
net-positive: at equal slots it ties, and at the deployed 10 slots it is slightly *negative*.

Second hypothesis tested in the same pass — **temporal fan-out** (one dense query per year,
unioned), on the theory that the missing line lives in an under-represented year slice:

| strategy | slots | Hit | coverage |
|---|---|---|---|
| flat dense top-10 | 10 | 100 % | 82.4 % |
| year fan-out top-3/year | 9 | 100 % | 75.2 % (worse) |
| year fan-out top-4/year | 12 | 100 % | 83.8 % (+1.4 pts for +2 slots — not a mechanism win) |

**Conclusion:** the residual coverage gap is **not** a selection/ranking problem and **not** a
temporal-allocation problem. Reordering the same candidate pool cannot add evidence, and the gold
lines that are missing are the ones *semantically distant* from the question (the 2024 "可以试试
Supabase" recommendation is part of the production-database arc, but it is about a course project).
Reaching them needs the **question** to be decomposed into per-slice sub-questions, not a better
sort of one similarity pool.

**Caveat recorded:** `bge-reranker-base` truncates at 512 tokens while session chunks are ~490
characters (≈700+ tokens), so part of every long session is never scored by the cross-encoder —
including, often, the decisive line. A future re-test should re-rank at **event** granularity
(or with a longer window) before concluding re-ranking is useless in general.

## What was kept vs. rejected

* **Rejected (not adopted):** re-ranking as part of the reference configuration. The reference
  stays **A4 = session units + k=10** (PASS 29/36, coverage 84.07 %).
* **Kept (infrastructure, off by default, tested):** `core/quivr_core/rag/reranker.py`
  (`LocalCrossEncoderReranker`), `DefaultRerankers.LOCAL`, and the `--rerank-model/--candidate-k`
  knob. Justification: the framework's re-rank stage previously accepted **only** hosted suppliers
  (Cohere/Jina), and `RerankerConfig.validate_model()` demanded `<SUPPLIER>_API_KEY` for *any*
  supplier — which made a private, offline re-ranker impossible to configure at all. That is a real
  bug, fixed and covered by tests (`LOCAL` needs no key; hosted suppliers still require one).
* Zero behaviour change when the knob is unused: both frozen baselines still reproduce their
  committed outputs byte-for-byte.

## Next phase decision (refined by two measured failures)

| Option | Evidence for / against | Verdict |
|---|---|---|
| **Phase 4/6-lite: decompose trajectory questions into per-slice sub-queries, retrieve each, union the evidence** | the missing lines are semantically distant but *individually answerable* — a sub-question like "课程项目最开始用的哪个数据库" matches the 2024-03 line directly | **do this next** |
| Re-rank at event granularity | untested; plausible given the truncation caveat | second candidate |
| Hybrid BM25 + RRF | still untested, and the failing questions deliberately avoid entity tokens | defer |
| Wider k alone (k=20+) | works (coverage 91.7 % offline at k=20) but spends context and does not explain *why* evidence is missed | fallback, not a capability |

---

# Phase 3 — retrieval-mechanism sweep: three negatives, one small win, and a plateau

Four candidate mechanisms, all measured **offline** (real BGE + real session chunks, no API cost)
against the same reference: session units, flat dense top-10 = 82.4 % coverage / 100 % hit.

| mechanism | slots | Hit@k | avg coverage | verdict |
|---|---|---|---|---|
| flat dense top-10 | 10 | 100 % | 82.4 % | reference |
| cross-encoder re-rank (50 → 10) | 10 | 97.1 % | 80.9 % | ❌ worse |
| temporal fan-out (top-3 per year, unioned) | 9 | 100 % | 75.2 % | ❌ worse |
| decompose → union (top-3 / sub-question) | 4.9 | 94.1 % | 70.1 % | ❌ worse |
| decompose → union (top-4 / sub-question) | 6.1 | 94.1 % | 75.2 % | ❌ worse |
| BM25 alone (CJK bigram + ASCII terms) | 10 | 94.1 % | 78.2 % | ❌ worse alone |
| **RRF(dense, BM25)** | 10 | 100 % | **84.8 %** | ✅ **+2.4 pts** |
| flat dense top-20 | 20 | 100 % | 91.7 % | (budget, not a mechanism) |
| **RRF(dense, BM25)** | 20 | 100 % | **93.9 %** | ✅ **+2.2 pts** |

## What the sweep establishes

1. **The one measured win is hybrid fusion.** BM25 alone is worse than dense, but RRF fusion beats
   dense at *both* budgets (+2.4 @10, +2.2 @20) — the first mechanism in this project to beat the
   baseline at matched slots. Small, but consistent, and it costs one extra retriever.
2. **Partial evidence is not a ranking / selection / temporal-allocation problem.** Re-ranking the
   same pool, forcing year diversity, and decomposing into per-slice sub-queries are all ≤ dense.
   Decomposition actively *hurts* because session chunks are already multi-fact: narrowing the query
   dilutes coverage per slot. (36 decompositions were generated by the LLM without seeing the gold
   evidence, so this is not an oracle result.)
3. **Coverage is dominated by the candidate budget**, monotonically:
   `67.2 % @5 → 82.4 % @10 → 91.7 % @20 → 97.1 % @50 → 100 % @101`.
4. **The stress corpus is approaching saturation**: at k=20 both dense and hybrid are ≥91 %, so it
   can no longer resolve mechanism differences beyond ~2 points. More retrieval-mechanism work on
   *this* corpus would be measuring noise.

## Incidental finding (operational)

`deepseek-v4-flash` is a **reasoning** model: a direct call with `max_tokens=400` returns
`finish_reason=length`, `reasoning_tokens=400`, and **empty content**. Quivr's config allows 4096
output tokens, which is why the baseline works — and it explains the latency profile (A4 avg 13.5 s,
max 64.5 s). Anyone calling this model directly must budget for reasoning tokens.

## Decision

* **Adopt next round (opt-in, then measured):** hybrid RRF — the only mechanism with positive
  evidence. Plan: session BM25 retriever + `EnsembleRetriever` behind a `--hybrid` flag, then one
  paid arm against A4 (expect ~+1–3 PASS if the +2.4 coverage translates).
* **Do not merge** re-ranking / fan-out / decomposition: measured non-wins, recorded above.
* **Prerequisite for any further retrieval work: a harder corpus.** Corpus v2 (more episodes, denser
  distractors, longer arcs) is now on the critical path, because at k=20 this corpus is ≥91 % covered
  and mechanism A/Bs can no longer separate.
* Reference configuration unchanged: **A4 = session units + k=10** (PASS 29/36, coverage 84.07 %).

---

# Phase 4 — hybrid retrieval (dense + BM25, weighted RRF): **ADOPTED as the new reference A5**

Implementation: `core/quivr_core/rag/hybrid.py` — CJK-bigram tokenizer, `BM25Index`/`BM25Retriever`
(no `rank_bm25` dependency: the formula is the one that was measured), `HybridRRFRetriever`
(reuses the framework's `EnsembleRetriever.rank_fusion`, weighted RRF `c=60`, then **truncates to
`k`** — the plain ensemble returns the union of its retrievers' lists, which would quietly exceed the
configured context budget), plus a guarded `iter_documents` vector-store enumerator.
`HybridConfig` is **disabled by default**; `--hybrid/--hybrid-pool` are tag-guarded knobs.
17 new tests (44 total).

## Implementation validated against the measurement *before* spending API budget

Driving `get_retriever` over a real FAISS index of the 101 session chunks:

| configuration | slots | Hit@k | coverage | vs offline probe |
|---|---|---|---|---|
| dense only (hybrid disabled) | 10 | 100 % | 82.4 % | ✅ identical |
| hybrid, pool 10 → cut 10 | 10 | 100 % | 84.8 % | ✅ identical |
| hybrid, pool 20 → cut 10 | 10 | 100 % | 84.8 % | — |
| **hybrid, pool 30 → cut 10** | 10 | 100 % | **86.3 %** | measured plateau (50/80 identical) |

Dense-only and pool-10 reproduce the probe exactly, so the deployed mechanism is the measured one;
`candidate_k=30` is the measured plateau and is now the default. The async path (`ainvoke`, which the
pipeline actually uses) and FAISS `docstore._dict` enumeration were verified as well.

## End-to-end paid arm

| arm | unit | k | hybrid | Hit@k | coverage | PASS | PARTIAL | FAIL | unsupported | latency |
|---|---|---|---|---|---|---|---|---|---|---|
| A4 reference | session | 10 | no | 97.1 % | 84.07 % | 29 (80.6 %) | 6 | 1 | 0 % | 12,763 ms |
| **A5 hybrid** | session | 10 | **yes** | 97.1 % | **85.05 %** | **30 (83.3 %)** | 5 | 1 | 0 % | **11,243 ms** |

All 36 queries retrieved a different evidence set than A4 (mechanism verified active). Retrieval
misses fell to one query (`s006`). Arm ladder of PASS: `20 → 24 → 26 → 29 → 30 / 36` for
`frozen → session-k5 → fixed-k10 → A4 → A5`.

## Honest reading: a small, mixed win

* Net **+1 PASS** with **3 gains and 2 regressions** — within ±1 query of noise on 36 queries. The
  reason to adopt is the *direction* (hybrid beat dense at every measured slot budget: offline +2.4
  @10 and +2.2 @20, real +0.9 @10), not the size.
* Gains were retrieval-driven and real: `s016` FAIL→PASS (2024-06-15 gym line retrieved), `s020`
  PARTIAL→PASS (destination "Railway" finally retrieved), `s028` PARTIAL→PASS (2024-11-19 account
  line retrieved).
* Regressions: `s013` PASS→PARTIAL (hybrid *lost* the 2025-10-11 Brave Search line that dense had —
  fusion can demote as well as promote), and `s025` PASS→FAIL.
* The probe **over-predicted** (+3.9 offline vs +0.9 real) because the real dense baseline (84.07 %)
  already exceeded the probe's 82.4 %. Probe numbers are predictions, not results.
* Latency improved (11.2 s vs 12.8 s) but this is not attributed to the mechanism — no controlled
  comparison, model-side variance dominates.

## New finding: the failure profile is no longer purely retrieval-side

`stress_hybrid_k10_label_diagnostics.json` splits the 6 imperfect queries:

| cause | ids |
|---|---|
| retrieval (decisive line missing) | `s013`, `s019`, `s023`, `s030` |
| **generation** (evidence complete, answer still wrong) | **`s007`** (misreads the retrieved 2026-05-17 exchange and asserts a May start), **`s025`** (all 4 gold lines retrieved, yet it refuses to disambiguate 王哥=小王) |

First measured justification for **Phase 7 (grounded generation / abstention control)**: not
hallucination (unsupported claims remain 0/36 in every arm) but *over-abstention and misreading of
retrieved evidence*.

## Decision

* **Adopt A5 (session units + k=10 + hybrid RRF) as the new reference**, with
  `HybridConfig.enabled` still defaulting to **False** so the frozen baselines stay byte-identical.
* Keep the Phase 3 saturation warning: the next *retrieval* experiment needs a harder corpus
  (Corpus v2) to be measurable at this plateau.
* **Next phase (now evidence-backed):** grounded-generation work targeting over-abstention and
  evidence misreading (`s025`, `s007`) — the retrieval side cannot fix those.
