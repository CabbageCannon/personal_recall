# Personal Recall Engine — Project Status

Living decision log. Facts only from real eval output; no estimated numbers.
Branch: `personal-recall` · Base: `CabbageCannon/quivr`

---

## Phase map

| Phase | Scope                                                          | Status                                                                                                                      |
| ----- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 0     | Small-corpus baseline (unmodified Quivr Dense RAG)             | ✅ done                                                                                                                      |
| 0.5   | Recall Stress Corpus + Stress Baseline + dataset audit         | ✅ done — baseline visibly fails, failure attributed to retrieval                                                            |
| 1     | MemoryEvent / Source Adapter + conversation-aware chunking     | ✅ done — kept: additive on top of the window control                                                                        |
| 2     | Retrieval trace / Evidence representation on structured memory | ⏸ (reranking evaluated in round 2: negative, not adopted)                                                                   |
| 3     | Hybrid retrieval (BM25 + dense + RRF)                          | ✅ done — adopted as A5 (+1 PASS, +0.98 coverage); reranker measured negative                                                |
| M1    | **Methodology: reproducibility protocol**                      | ✅ done — temperature 0 is *not* reproducible; removing the LLM rewrite makes retrieval deterministic and offline-exact (A7) |
| 4     | Temporal retrieval                                             | ⏸                                                                                                                           |
| 5     | Entity-aware recall (Person / Alias)                           | ⏸                                                                                                                           |
| 6     | Multi-evidence / state evolution                               | ⏸                                                                                                                           |
| 7     | Grounded generation (EvidenceItem, citation, abstention)       | ✅ done — A10 adopted: 100%% citation rate, 0 misleading, 0 unsupported, PASS 91.7%%                                         |
| M2    | **Corpus v2: distractor pack for measurement headroom**        | ✅ done — v1 saturated (Hit@10 100%%); v2 costs −6.62 coverage pts at 2.3× the space, still 0 retrieval misses, 0 false memories |
| 8     | Persistence (PostgreSQL + pgvector)                            | ⏸                                                                                                                           |
| 9     | Product UI                                                     | ⏸                                                                                                                           |
| 10    | Multimodal recall                                              | ⏸                                                                                                                           |
| 11    | Optional skills / agent layer                                  | ⏸                                                                                                                           |

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

| Metric                          | Value                      |
| ------------------------------- | -------------------------- |
| Answerable / unanswerable       | 30 / 4                     |
| Retrieval Hit@5                 | 30/30 = **1.000**          |
| Strict Hit@5                    | 29/30 = 0.967              |
| Avg evidence coverage           | 0.965                      |
| Avg strict coverage             | 0.937                      |
| Weighted coverage               | 0.961                      |
| Answer PASS / PARTIAL / FAIL    | 34 / 0 / 0                 |
| Unsupported claim rate          | 0/34 = **0.0**             |
| Latency avg / min / max (ms)    | 7507.6 / 3618.7 / 15242.1  |
| Chunk-boundary-affected queries | 3 (`q011`, `q027`, `q031`) |

**Phase 0 conclusion (decision):** the small corpus cannot discriminate methods. 6,919 chars
→ 23 chunks, so Top-5 covers **21.7 %** of the entire memory space; entity tokens
(Neon / Tailscale / WireGuard / Cloudflare Pages) act as unique anchors. Dense+FAISS+Top-5
therefore always returns the gold evidence, and generation then looks perfect.
⇒ Do **not** use the small corpus to justify any algorithm change. Keep it as a regression
guard only.

## Phase 0.5 decisions (this phase)

| #   | Decision                                                                                                                                                                                | Why                                                                                                                                                                                                   |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | Long-lived branch `personal-recall`, pushed to origin                                                                                                                                   | §git workflow; no short-lived branch sprawl                                                                                                                                                           |
| D2  | Stress data lives beside the small data and is fully isolated (`stress_*` files, `--dataset stress`)                                                                                    | must never overwrite the 34-query baseline                                                                                                                                                            |
| D3  | Corpus size target = **≥140 chunks**, i.e. ≈45k Chinese chars (≈135 KB)                                                                                                                 | the dataset card's "50–100 KB" assumed 1 byte/char; Chinese is 3 bytes/char. The chunk count is what governs retrieval selectivity (Top-5 = 3.3 % of space vs 21.7 % today), so the chunk target wins |
| D4  | Facts are frozen in `data/STRESS_CORPUS_SPEC.md` (§3 timelines + §4 verbatim anchors) and the text is written by 4 parallel writers                                                     | cross-year coherence of state evolution is the whole point; parallel authorship otherwise breaks it                                                                                                   |
| D5  | `stress_queries.json` keeps the **exact** 6-key schema of `queries.json`                                                                                                                | no schema drift; difficulty is *measured* by the validator, not self-declared in the data                                                                                                             |
| D6  | 36 queries = 12 regression (old categories) + 24 hard (10 stress categories)                                                                                                            | separates "did we break the easy stuff" from "did we actually improve recall"                                                                                                                         |
| D7  | Difficulty gate before spending API budget: offline validation + measured diagnostics (lexical overlap, unique-anchor chunk count, near-duplicate distractor count) + independent audit | §15: never run 36 paid queries against an unaudited dataset                                                                                                                                           |

## Hypotheses the stress baseline must test

| #   | Hypothesis                                                                                     | Falsified if               |
| --- | ---------------------------------------------------------------------------------------------- | -------------------------- |
| H1  | With ~150 chunks, dense Top-5 stops covering the gold evidence for state-change questions      | Hit@5 stays ≈1.0           |
| H2  | Recurring-entity distractor messages cause *partial* evidence (right entity, wrong date/state) | coverage stays ≈1.0        |
| H3  | "latest state" questions get answered with an outdated state when several states are retrieved | answer PASS stays 100 %    |
| H4  | Near-duplicate wording (sleep/idle for both Render and Neon) causes entity confusion           | no wrong-entity retrievals |
| H5  | Unanswerable questions about plausible-but-absent facts trigger unsupported claims             | unsupported rate stays 0   |

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

|                             | small              | stress                                |
| --------------------------- | ------------------ | ------------------------------------- |
| File                        | `data/chats.txt`   | `data/stress_chats.txt`               |
| Bytes / chars               | 11,789 / 6,919     | **94,262 / 49,457**                   |
| Messages / episodes         | 200 / 26           | **1,216 / 100**                       |
| Chunks @ 400/100            | 23                 | **165**                               |
| Top-5 share of memory space | 21.7 %             | **3.0 %**                             |
| Window                      | 2024-03 → 2026-05  | 2024-01-08 → 2026-08-28               |
| Queries                     | 34 (30 answerable) | **36 (34 answerable)**, 15 categories |

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

|                                                 | small         | stress (pre-repair) |
| ----------------------------------------------- | ------------- | ------------------- |
| Hit@5 (answerable)                              | 30/30 = 100 % | **27/34 = 79.4 %**  |
| Avg evidence coverage                           | 96.5 %        | **56.6 %**          |
| Gold first appears at dense rank (median / max) | 1 / 3         | 1 / **12**          |
| Retrieval misses                                | 0             | **7**               |
| Partial evidence                                | 3             | **16**              |

The probe reproduces the small corpus's official numbers exactly (100 % / 96.5 %), which is what
licenses using it as a pre-flight gate. Reading: the stress corpus fails mostly by **partial
evidence** — the retriever finds the entity early but misses the other state-bearing lines.

## Offline gate 3 — independent adversarial audit (`audit_report.md`)

**Verdict: PASS-WITH-FIXES.** 31/36 gold answers verified TRUE, 108/108 evidence lines verbatim
grounded, zero meta-text leaks, zero duplicate messages, both `s035`/`s036` verified genuinely
unanswerable. Found and acted on:

| #   | Finding                                                                                                                                                                                | Action                                                                                                                                                                           |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A1  | **3 poisoned gold answers** (s007, s019, s016) — a corpus-faithful model would be graded wrong (internship actually started late April, not May; 小汪's gym arc starts before 2025-02) | repaired (blocking)                                                                                                                                                              |
| A2  | 9 of 24 "hard" queries answerable from a single 400-char chunk; only 7 are genuine multi-hop                                                                                           | question-side rewrites that force ≥2 chunks                                                                                                                                      |
| A3  | Both alias traps self-decoding (`王哥（就是小王）`, `阿伟（我室友）` appeared in the only line mentioning each)                                                                        | glosses removed from the corpus; aliases now resolve from context only (adjacency / event identity)                                                                              |
| A4  | `stress_validation.json` claimed 169 chunks; the real chunker yields **165** (validator decoded raw bytes, so CRLF counted an extra `\r` per line)                                     | validator normalises newlines like `quivr_core`'s text-mode reader                                                                                                               |
| A5  | Corpus contradicts spec §5.7: 7 lines state a final state outright, and the 2026-08-20 recap states four final states in ~2 chunks                                                     | **not** fixed: several are anchor lines (A44) and the recap is what makes `latest_state` answerable at all. Recorded as risk R6; the difficulty claim is restated honestly below |

**Honest difficulty claim (post-audit):** the measured difficulty of this dataset is
**retrieval** difficulty (miss + partial evidence), not state-assembly difficulty. Only 7 of the
24 hard queries require genuine multi-evidence assembly. State assembly becomes a measurable
axis only after Phase 1/6 work; do not over-claim it from Phase 0.5.

## Added decisions

| #   | Decision                                                                                          | Why                                                                                       |
| --- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| D8  | The paid baseline is gated behind offline validation **and** an independent adversarial audit     | §15; the audit caught 3 poisoned golds that no structural check can see                   |
| D9  | Poisoned golds and EASY-query difficulty are fixed **question-side**, not by rewriting the corpus | corpus edits would invalidate the validator/probe evidence and risk breaking anchor lines |
| D10 | `expected_answer` truth is an audited property, not an assumption                                 | a wrong gold inverts the metric silently                                                  |
| D11 | Freeze the corpus hash in the eval record once the paid run starts                                | makes the result reproducible and detects later drift                                     |

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

| Metric                 | Phase 0 small      | Phase 0.5 stress                    |
| ---------------------- | ------------------ | ----------------------------------- |
| Queries                | 34 (30 answerable) | 36 (34 answerable)                  |
| Chunks                 | 23                 | 165                                 |
| Retrieval Hit@5        | 30/30 = **100 %**  | 28/34 = **82.35 %**                 |
| Avg evidence coverage  | 96.5 %             | **53.19 %**                         |
| Weighted coverage      | 96.1 %             | **48.25 %**                         |
| Answer PASS            | 34/34 = **100 %**  | 20/36 = **55.56 %**                 |
| PARTIAL / FAIL         | 0 / 0              | **9 / 7**                           |
| Unsupported claim rate | 0 %                | **0 %**                             |
| Latency avg / max (ms) | 7,508 / 15,242     | 13,525 / 64,521                     |
| Retrieval misses       | 0                  | 6 (`s006 s013 s020 s021 s028 s030`) |

Worst categories by coverage: `latest_state` 16.7 %, `temporal_state_change` 27.1 %,
`negative_evidence` / `multi_evidence` 37.5 %, `implicit_reference` 45.8 %.
Regression categories held up: `exact_fact` / `exact_keyword` / `time_recall` = 100 % coverage, 100 % PASS.

## Failure attribution (manual grading, 36/36, `stress_manual_labels.json`)

|                                                                                                  | count     |
| ------------------------------------------------------------------------------------------------ | --------- |
| Queries whose gold facts were **absent from the retrieved chunks** (`evidence_sufficient=false`) | **16**    |
| Failures where the evidence WAS sufficient (pure generation failures)                            | **0**     |
| PASS among the 20 sufficient queries                                                             | **20/20** |
| Unsupported claims                                                                               | **0/36**  |
| Suspected gold errors after the audit repair                                                     | **0**     |

The correlation is perfect: every non-PASS answer is a query where retrieval did not deliver the
gold lines, and the generator was never wrong when it had them. Observed behaviour under
insufficient evidence is **grounded abstention or under-answering**, not hallucination — including
both `unanswerable` queries (`s035` thesis title, `s036` restaurant name), which declined cleanly
and invented nothing.

## How much of the gap is just the Top-5 window? (`probe_recall_at_k.py`)

| group                       | cov@5  | cov@10 | cov@20 | cov@50 | cov@100 |
| --------------------------- | ------ | ------ | ------ | ------ | ------- |
| all answerable              | 53.4 % | 74.5 % | 82.1 % | 94.9 % | 99.3 %  |
| the 16 insufficient queries | 37.5 % | 58.9 % | 71.9 % | 90.6 % | 98.4 %  |
| the 20 sufficient queries   | 67.6 % | 88.4 % | 91.2 % | 98.6 % | 100 %   |

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

| Option                                                                     | Expected effect                                                                                                                                                                                  | Verdict                                           |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------- |
| **Phase 1: session/conversation-aware chunking + MemoryEvent/MemoryChunk** | one slot = one coherent session (participants + time range) instead of a 400-char fragment; the anchor line arrives with its context, and 5 slots cover 5 sessions rather than 5 fragments of ~2 | **do this first**                                 |
| Control: same corpus at k=10/20 (no structural change)                     | +21 pts coverage for free                                                                                                                                                                        | **run as the control arm** — Phase 1 must beat it |
| Phase 3 hybrid BM25+RRF                                                    | targets exact-token recall, but the failing questions deliberately avoid entity names and dense already finds the topic                                                                          | defer; test later, possibly combined              |
| Phase 7 abstention / false-memory control                                  | already 0 % unsupported and 2/2 correct abstentions                                                                                                                                              | **deprioritised** — no measured need yet          |

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

| arm                     | retrieval unit | k      | chunks | Hit@k       | avg coverage | PASS            | PARTIAL | FAIL  | score /36 | unsupported |
| ----------------------- | -------------- | ------ | ------ | ----------- | ------------ | --------------- | ------- | ----- | --------- | ----------- |
| A1 frozen baseline      | fixed 400/100  | 5      | 165    | 82.35 %     | 53.19 %      | 20 (55.6 %)     | 9       | 7     | 24.5      | 0 %         |
| A2 Phase 1              | session        | 5      | 101    | 88.24 %     | 62.75 %      | 24 (66.7 %)     | 9       | 3     | 28.5      | 0 %         |
| A3 window control       | fixed 400/100  | **10** | 165    | 88.24 %     | 70.10 %      | 26 (72.2 %)     | 7       | 3     | 29.5      | 0 %         |
| **A4 session + window** | session        | **10** | 101    | **97.06 %** | **84.07 %**  | **29 (80.6 %)** | 6       | **1** | **32.0**  | 0 %         |

Offline probe grid (free, real BGE, no LLM) — the same conclusion on the retrieval side:

| unit          | cov@5  | cov@10 | cov@20 |
| ------------- | ------ | ------ | ------ |
| fixed 400/100 | 53.4 % | 74.5 % | 82.1 % |
| session       | 67.2 % | 82.4 % | 91.7 % |

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

| group                    | cov@5  | cov@10 | cov@20 | cov@50 | cov@101 |
| ------------------------ | ------ | ------ | ------ | ------ | ------- |
| all answerable           | 67.2 % | 82.4 % | 91.7 % | 97.1 % | 100 %   |
| the 5 never-PASS queries | 35 %   | 50 %   | 65 %   | 85 %   | 100 %   |

⇒ The bottleneck is no longer the retrieval **unit** (Phase 1 fixed that) but the **selection**:
the decisive line is ranked outside the top 10 in a 101-unit pool. `s007`/`s019` are separate
generation slips — all their gold lines were retrieved.

## Next phase decision

| Option                                                  | Expected effect                                                                                              | Verdict               |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------- |
| **Phase 3-lite: widen candidates (k≈50) + rerank to 5** | the sweep proves the hard tail sits at rank 20–50; a ranker that selects 5 of 50 attacks exactly that        | **do this next**      |
| Phase 3 hybrid BM25 + RRF (no reranker)                 | the failing questions deliberately avoid entity tokens, so lexical matching cannot recover them on its own   | defer / combine later |
| Phase 4 temporal parsing                                | would help time-scoped questions, but the hard tail fails on *entity-implied* slices, not on date arithmetic | defer                 |
| Phase 6 multi-evidence aggregation                      | plausible, but it presupposes the same re-ranking ability                                                    | after the reranker    |

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

| strategy                          | slots | Hit@k  | avg coverage           |
| --------------------------------- | ----- | ------ | ---------------------- |
| flat dense top-5 (A2-equivalent)  | 5     | 94.1 % | 67.2 %                 |
| **re-rank 50 → keep 5**           | 5     | 94.1 % | **67.2 %** (identical) |
| flat dense top-10 (A4-equivalent) | 10    | 97.1 % | **82.4 %**             |
| **re-rank 50 → keep 10**          | 10    | 97.1 % | **80.9 %** (worse)     |

Per-query at 5 slots: **8 improved, 9 worsened, 19 unchanged** — e.g. `s028`'s gold chunk moved
dense rank 10 → 1 (coverage 0.00 → 1.00) but `s006`'s moved 4 → 17 (1.00 → 0.00). The re-ranker is
genuinely reordering (scores span 0.00–0.99 and it does promote gold chunks), it just is not
net-positive: at equal slots it ties, and at the deployed 10 slots it is slightly *negative*.

Second hypothesis tested in the same pass — **temporal fan-out** (one dense query per year,
unioned), on the theory that the missing line lives in an under-represented year slice:

| strategy                | slots | Hit   | coverage                                             |
| ----------------------- | ----- | ----- | ---------------------------------------------------- |
| flat dense top-10       | 10    | 100 % | 82.4 %                                               |
| year fan-out top-3/year | 9     | 100 % | 75.2 % (worse)                                       |
| year fan-out top-4/year | 12    | 100 % | 83.8 % (+1.4 pts for +2 slots — not a mechanism win) |

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

| Option                                                                                                           | Evidence for / against                                                                                                                                          | Verdict                    |
| ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| **Phase 4/6-lite: decompose trajectory questions into per-slice sub-queries, retrieve each, union the evidence** | the missing lines are semantically distant but *individually answerable* — a sub-question like "课程项目最开始用的哪个数据库" matches the 2024-03 line directly | **do this next**           |
| Re-rank at event granularity                                                                                     | untested; plausible given the truncation caveat                                                                                                                 | second candidate           |
| Hybrid BM25 + RRF                                                                                                | still untested, and the failing questions deliberately avoid entity tokens                                                                                      | defer                      |
| Wider k alone (k=20+)                                                                                            | works (coverage 91.7 % offline at k=20) but spends context and does not explain *why* evidence is missed                                                        | fallback, not a capability |

---

# Phase 3 — retrieval-mechanism sweep: three negatives, one small win, and a plateau

Four candidate mechanisms, all measured **offline** (real BGE + real session chunks, no API cost)
against the same reference: session units, flat dense top-10 = 82.4 % coverage / 100 % hit.

| mechanism                                  | slots | Hit@k  | avg coverage | verdict                   |
| ------------------------------------------ | ----- | ------ | ------------ | ------------------------- |
| flat dense top-10                          | 10    | 100 %  | 82.4 %       | reference                 |
| cross-encoder re-rank (50 → 10)            | 10    | 97.1 % | 80.9 %       | ❌ worse                   |
| temporal fan-out (top-3 per year, unioned) | 9     | 100 %  | 75.2 %       | ❌ worse                   |
| decompose → union (top-3 / sub-question)   | 4.9   | 94.1 % | 70.1 %       | ❌ worse                   |
| decompose → union (top-4 / sub-question)   | 6.1   | 94.1 % | 75.2 %       | ❌ worse                   |
| BM25 alone (CJK bigram + ASCII terms)      | 10    | 94.1 % | 78.2 %       | ❌ worse alone             |
| **RRF(dense, BM25)**                       | 10    | 100 %  | **84.8 %**   | ✅ **+2.4 pts**            |
| flat dense top-20                          | 20    | 100 %  | 91.7 %       | (budget, not a mechanism) |
| **RRF(dense, BM25)**                       | 20    | 100 %  | **93.9 %**   | ✅ **+2.2 pts**            |

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

| configuration                | slots | Hit@k | coverage   | vs offline probe                   |
| ---------------------------- | ----- | ----- | ---------- | ---------------------------------- |
| dense only (hybrid disabled) | 10    | 100 % | 82.4 %     | ✅ identical                        |
| hybrid, pool 10 → cut 10     | 10    | 100 % | 84.8 %     | ✅ identical                        |
| hybrid, pool 20 → cut 10     | 10    | 100 % | 84.8 %     | —                                  |
| **hybrid, pool 30 → cut 10** | 10    | 100 % | **86.3 %** | measured plateau (50/80 identical) |

Dense-only and pool-10 reproduce the probe exactly, so the deployed mechanism is the measured one;
`candidate_k=30` is the measured plateau and is now the default. The async path (`ainvoke`, which the
pipeline actually uses) and FAISS `docstore._dict` enumeration were verified as well.

## End-to-end paid arm

| arm           | unit    | k   | hybrid  | Hit@k  | coverage    | PASS            | PARTIAL | FAIL | unsupported | latency       |
| ------------- | ------- | --- | ------- | ------ | ----------- | --------------- | ------- | ---- | ----------- | ------------- |
| A4 reference  | session | 10  | no      | 97.1 % | 84.07 %     | 29 (80.6 %)     | 6       | 1    | 0 %         | 12,763 ms     |
| **A5 hybrid** | session | 10  | **yes** | 97.1 % | **85.05 %** | **30 (83.3 %)** | 5       | 1    | 0 %         | **11,243 ms** |

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

| cause                                                  | ids                                                                                                                                                                |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| retrieval (decisive line missing)                      | `s013`, `s019`, `s023`, `s030`                                                                                                                                     |
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

---

# Phase 5 — time-line answer prompt: targeted fixes landed, plus a noise-floor discovery

Hypothesis: the two generation-side failures map onto two stock prompt instructions — *"if you
cannot provide an answer … just answer that you don't have the answer"* (drives `s025`'s
over-abstention) and *"if the provided context contains contradictory … information, state so"*
(drives `s007` reporting both readings instead of resolving them).

Change: `--answer-prompt timeline` appends explicit rules (order the dated records first; commit
when they jointly settle the fact; only say "no record" when none touches it; prefer the
contemporaneous dated statement when records conflict; distinguish plan from what happened).
Core gap exposed and fixed: `custom_prompts` is a **read-only `mappingproxy` with no registration
API**, so overriding a prompt was impossible without touching a private dict — added
`register_prompt(name, prompt, override=False)`, mirroring `register_processor`.
Also added `--max-output-tokens` and `--temperature` knobs (both default to the frozen values).

## Result (A6 vs A5 — identical retrieval config, only the prompt differs)

| arm                    | PASS            | PARTIAL | FAIL       | unsupported | coverage | latency   |
| ---------------------- | --------------- | ------- | ---------- | ----------- | -------- | --------- |
| A5 stock prompt        | 30 (83.3 %)     | 5       | 1 (`s025`) | 0 %         | 85.05 %  | 11,243 ms |
| **A6 timeline prompt** | **31 (86.1 %)** | 5       | **0**      | **0 %**     | 83.58 %  | 13,539 ms |

Only **3 of 36** verdicts changed, and the attribution matters more than the count:

| id     | change         | attribution                                                                                                                                                    |
| ------ | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `s019` | PARTIAL → PASS | **prompt**: picks the dated contemporaneous line over the later "别拖到五月才动手" remark, concludes late April ≈ 04-26, "没有拖到 5 月" — reproduces the gold |
| `s025` | FAIL → PASS    | **prompt**: commits to "不是同一个人" using the retrieved 王哥→小王 reply and 小汪's MySQL line, where A5 refused despite complete evidence                    |
| `s021` | PASS → PARTIAL | **retrieval variance**, not the prompt: the decisive 2026-04-06 line was simply not retrieved this run                                                         |

`s007` is half-fixed: the affirmative "到五月才投了十来家" error is gone and every delay stage is
now dated, but the closing line still hedges "四月下旬/五月" instead of resolving to late April.
No new unsupported claims anywhere (0/36 in both arms; the stronger commitment instruction did not
induce fabrication — every date and number traces to the arm's own retrieved chunks).

## The bigger finding: a measured noise floor that invalidates small single-run deltas

A5 and A6 have an **identical retrieval configuration** (session units, k=10, hybrid RRF) — only
the *answer* prompt differs, which cannot affect retrieval. Yet:

| between two runs of the same retrieval config    | value           |
| ------------------------------------------------ | --------------- |
| queries with an identical retrieved evidence set | **4 / 36**      |
| mean Jaccard overlap of retrieved sets           | 0.740           |
| mean per-query coverage change                   | **2.78 points** |
| max per-query coverage change                    | 50 points       |
| aggregate coverage change                        | 1.38 points     |

Cause: the pipeline **rewrites the query with an LLM call** (`CONDENSE_TASK_PROMPT`) at the frozen
temperature 0.3, and retrieval runs on that rewritten query. So run-to-run variation in *evidence*
is real and large. Implications:

* Any single-run delta below ~2–3 coverage points, or ~±1–2 PASS on 36 queries, **cannot be
  attributed to a mechanism** — which is exactly the size of both the hybrid win (+0.98 coverage,
  +1 PASS) and this prompt win (+1 PASS).
* **A5's adoption is therefore re-framed**: it rests on the *deterministic* offline grid
  (hybrid beat dense at every matched budget, reproducible), not on the +0.98 real-eval delta.
* **A6's adoption rests on mechanism attribution, not the aggregate**: two specific designed fixes
  landed on the two specific failures the change targeted, with groundedness intact.
* New knobs `--temperature` (0.3 → 0.0 makes the rewrite deterministic) and `--max-output-tokens`
  are now available for a reproducibility protocol.

## Decision

* **Adopt A6 (A5 + timeline answer prompt) as the new reference** — on attribution, not on the
  +1 aggregate.
* **New evaluation protocol requirement (from this round):** every future A/B must either
  (a) be measured on the deterministic offline probes where possible, (b) run at `--temperature 0`,
  or (c) be reported with an explicit repeat-run noise band. Small single-run claims are retired.
* Remaining non-PASS in A6: `s007` (half-fixed hedge), `s013`, `s021`, `s023`, `s030` — all
  retrieval-caused except `s007`'s residual hedge.

---

# Phase 6 — determinism: temperature 0 does **not** fix it; removing the LLM from retrieval does

## Experiment 1: two identical runs at temperature 0

Reference config (A6), run twice with `--temperature 0`:

| pair                        | identical evidence sets | identical answers | mean per-query coverage \|Δ\| | aggregate coverage |
| --------------------------- | ----------------------- | ----------------- | ----------------------------- | ------------------ |
| rep1 vs rep2 (**both t=0**) | **6 / 36**              | **0 / 36**        | **3.70 pts** (max 33)         | 85.54 % vs 83.58 % |
| A6 (t=0.3) vs rep1          | 9 / 36                  | 0 / 36            | 5.09 pts                      | —                  |
| A6 (t=0.3) vs rep2          | 5 / 36                  | 0 / 36            | 4.17 pts                      | —                  |

**Conclusion: `temperature=0` does not make this pipeline reproducible.** The serving stack
(reasoning model, batching/routing) is nondeterministic, and the pipeline's `rewrite` node feeds an
**LLM-condensed question** into retrieval, so the *evidence itself* changes between runs. That is
why every small delta measured so far (hybrid +0.98 coverage, prompt +1 PASS) sat inside the noise.

## Experiment 2: take the LLM out of the retrieval path (`--workflow no-rewrite`)

The framework's workflow is config-driven (`WorkflowConfig.nodes`, dispatched by node name), so the
`rewrite` node can be dropped without touching `quivr_core`: `START → filter_history → retrieve →
generate_rag → END`. `retrieve` already builds its own task from the raw user message, so this is a
pure configuration change.

Verification that retrieval became deterministic:

| check                                                                                              | result                                      |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| pipeline retrieval vs the offline deterministic reference (raw question, BGE + FAISS + BM25 + RRF) | **36 / 36 identical, including rank order** |
| offline reference re-run (byte comparison)                                                         | **identical hash**                          |

## Effect on quality (A7 vs the rewrite arms)

| arm                                | Hit@k        | coverage    | PASS            | PARTIAL | FAIL  | unsupported | latency      |
| ---------------------------------- | ------------ | ----------- | --------------- | ------- | ----- | ----------- | ------------ |
| A6 (rewrite, t=0.3)                | 97.06 %      | 83.58 %     | 31              | 5       | 0     | 0 %         | 13,539 ms    |
| rep1 (rewrite, t=0)                | 97.06 %      | 85.54 %     | —               | —       | —     | 0 %         | 13,608 ms    |
| rep2 (rewrite, t=0)                | 97.06 %      | 83.58 %     | —               | —       | —     | 0 %         | 13,279 ms    |
| **A7 (no-rewrite, deterministic)** | **100.00 %** | **86.27 %** | **33 (91.7 %)** | 3       | **0** | **0 %**     | **9,462 ms** |

Removing the rewrite improves the retrieval **and** the answer quality, and cuts latency ~30 %:
zero retrieval misses, +2.7 coverage points over A6, +2 PASS, no failures left.

### Where the +2 net comes from (it is not a clean sweep)

| id     | change vs A6       | note                                                                                                                                                                                                                                                                                            |
| ------ | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `s007` | PARTIAL → PASS     | the "四月下旬/五月" hedge is gone entirely; the answer quotes the 04-20 and 05-10 anchors instead. **Borderline call** — it conveys late April via the quoted anchors rather than asserting it; a stricter grader would leave it PARTIAL (32/4/0)                                               |
| `s021` | PARTIAL → PASS     | the April line *was* retrieved this time (chunk 83); both halves now answered verbatim                                                                                                                                                                                                          |
| `s023` | PARTIAL → PASS     | the exact 2024-07-21 anchor is still missing, but other anchors let it bound the gap to "3–5 个月" (gold: 大概四个月) instead of declining                                                                                                                                                      |
| `s020` | PASS → **PARTIAL** | **the measured cost of dropping the rewrite**: the answer fuses two events and dates the switch "2025 年底", contradicting the dated 2025-06 records in its own context. The trap is a corpus-internal recollection line ("去年年底…数据库也跟着放一块了"); the rewrite arm did not fall for it |

Also `s030`'s cause flipped from retrieval to answer-side (its naming facts were retrieved this arm),
and `s013` is now the **only** retrieval-caused non-PASS (Brave Search's 2025-10-11 line is absent in
A5/A6/A7 alike; A4 passed it purely because its retrieval happened to contain that line).

## What this changes methodologically

1. **Retrieval A/Bs are now free and exact.** The offline probe is no longer a *prediction* of the
   pipeline — with `no-rewrite` it reproduces it rank-for-rank. Any future retrieval change can be
   screened offline at zero API cost and trusted.
2. **Generation A/Bs still need repeats** (answers are not reproducible: 0/36 identical), so PASS
   deltas must be reported with a repeat-run band.
3. **Earlier adoptions are re-grounded on deterministic evidence.** Hybrid's effect at matched slots
   is now a *deterministic* measurement: dense 82.4 % → hybrid 86.3 % coverage — far outside the
   noise band. A5's adoption no longer depends on a noisy single run.
4. **Product caveat recorded:** with a real chat history the condense step resolves pronouns
   ("那个库" → the entity) and remains useful; the stress questions are self-contained and each
   query runs with an empty history, so condensing only paraphrases and injects noise. The eval
   harness therefore measures the `no-rewrite` mode, and that is stated rather than implied.

## Decision

* **Adopt A7 as the evaluation reference**: session units + k=10 + hybrid RRF + timeline answer
  prompt + no-rewrite. **PASS 33/36 (91.7 %), coverage 86.27 %, Hit@10 100 %, 0 unsupported, FAIL 0,
  9.5 s average latency.**
* Keep all knobs defaulting to the frozen baseline values, so the original baselines stay
  byte-identical and every experiment remains reproducible from the tag in its filename.
* Remaining non-PASS in A7: `s013` (retrieval: Brave Search line), `s020`, `s030` (both
  answer-side over/under-specification). The corpus is now close to its ceiling — further retrieval
  work needs **Corpus V2** to have anything left to measure.

---

# Phase 7 — evidence model + citations + citation metrics (A8)

The project's promise is *Answer → Evidence → MemoryEvent → original chat line*, and nothing in the
harness measured that link: **0/36 answers in every previous arm carried any machine-checkable
citation**. This phase builds the first half of it.

## What was added

1. **Evidence metadata persisted** (`EVIDENCE_METADATA_FIELDS` in `serialize_sources`): every
   retrieved source now carries `memory_chunk_id`, `conversation_id`, `start_time`, `end_time`,
   `participants`, `n_events`. A retrieved chunk is now a full EvidenceItem — e.g.
   `txt-session-0017`, 2024-10-20 15:10 → 15:58, participants [小王, 我], 14 events — so a citation
   can be traced back to specific chat lines instead of an opaque chunk index.
2. **`--answer-prompt cited`**: timeline rules + mandatory `[来源 N]` on every factual statement.
   It deliberately **overrides** the stock instruction "Don't cite the source id in the answer
   objects", which is why the rule text says so explicitly.
3. **`citation_metrics.py`** (+9 tests, 57 total): citation rate, valid citation rate,
   **citation coverage** (gold lines inside a *cited* chunk), **retrieval coverage** (gold lines
   inside any *retrieved* chunk — the control), **lexical citation precision**, abstention accuracy.

## Bug caught in the metric (not in the model)

The first measurement reported 36 "invalid" citations, mostly `[来源 0]`. Reading
`combine_documents` showed the framework renders each chunk as `Source: {index}` with
`index = range(len(docs))` — i.e. **0-based** — so `[来源 0]` was *correct* and my 1-based mapping
was wrong. Fixed, with a regression test pinning the convention. (Same lesson as the earlier CRLF
chunk-count bug: check the convention against the framework's code, not against intuition.)

## Citation metrics (A8)

| metric                                       | A8 (cited)  | A7 (uncited) |
| -------------------------------------------- | ----------- | ------------ |
| citation rate                                | **100 %**   | 0 %          |
| citations / invalid                          | 205 / **0** | 0 / 0        |
| **citation coverage** (gold in cited chunk)  | **77.2 %**  | 0 %          |
| retrieval coverage (gold in retrieved chunk) | 84.2 %      | 84.2 %       |
| citation precision (lexical lower bound)     | 61.0 %      | n/a          |
| abstention accuracy                          | 100 %       | 100 %        |
| unsupported claims                           | 0 %         | 0 %          |

`citation coverage 77.2 % vs retrieval coverage 84.2 %` is the honest reading: the model cites
almost everything it uses, and the 7-point gap is evidence it had but did not cite.

## Clean single-variable A/B (a direct payoff of Phase 6)

Retrieval is **byte-identical between A7 and A8 (36/36, order included)** because `no-rewrite`
removed the LLM from the retrieval path — both arms match the offline deterministic reference. So
the citation instruction is the *only* difference, and every verdict change is attributable to it.

| arm                      | PASS   | PARTIAL | FAIL | unsupported | misleading citations |
| ------------------------ | ------ | ------- | ---- | ----------- | -------------------- |
| A7 (no citations)        | **33** | 3       | 0    | 0 %         | n/a                  |
| A8 (mandatory citations) | **31** | 5       | 0    | 0 %         | **0 / 36**           |

## The trade-off, diagnosed

Only 2 of 36 verdicts changed, and both losses are **hedges, not citations**:

* `s025` PASS→PARTIAL — the answer assembles the two disambiguating facts **and cites them
  correctly**, but leads with "无法确认…上下文没有说明…是同一个人" instead of committing.
* `s032` PASS→PARTIAL — same shape: "无法确定是否同一个" where the gold says 基本是同一个.

Citation honesty is perfect: every `[来源 N]` is in range and non-decorative; expanding each
citation to `retrieved_sources[N]` confirms the claim is in the cited chunk (36/36
`citation_supports_claim`). Answers got shorter on 11 queries, but only by dropping A7's
"依据记录：…" evidence restatement — all of those still PASS.

**Diagnosis:** requiring a source for every claim nudged the model back toward *over-abstention* on
questions that need a conclusion drawn **across** sources (identity/equivalence). The citation rule
did not cause the loss; the missing "commit anyway" counterweight did.

## Decision

* **Keep the capability** (evidence fields, `--answer-prompt cited`, metric tooling): verifiable
  provenance is the product requirement, and it costs no groundedness (0 unsupported, 0 misleading).
* **The default reference stays A7 for now** (33/36) — the citation variant is not adopted as the
  reference until the hedging is fixed.
* **Next experiment (specified by this round's evidence):** extend the citation rule with an
  explicit commitment clause — "cite, don't hedge: state the conclusion the cited sources support;
  a conclusion that joins several sources cites all of them" — then re-run A8's config and check
  whether the 2 hedge losses come back without spending citation honesty. Retrieval is now a
  controlled constant, so the delta will be attributable.

---

# Phase 7b — the commitment clause: **highest PASS of the project, REJECTED for a groundedness violation**

Ran exactly the experiment Phase 7 specified: A8's config plus a commitment clause
(`--answer-prompt cited-committed` — "citing is not hedging; draw the conclusion the cited sources
support, including conclusions that only follow from several sources together"). A8's own prompt was
left untouched so it stays reproducible, and a test now pins that the variants are strictly additive.

## Retrieval was again a controlled constant

All three no-rewrite arms (`A7`, `A8`, `A9`) retrieve **identically, 36/36 including rank order**, and
all three match the offline deterministic reference. Answer length: 235 → 267 → 273 chars.

## Results

| arm                        | PASS            | PARTIAL | FAIL | unsupported claims | citation rate | misleading citations |
| -------------------------- | --------------- | ------- | ---- | ------------------ | ------------- | -------------------- |
| A7 timeline (no citations) | 33 (91.7 %)     | 3       | 0    | **0 %**            | 0 %           | n/a                  |
| A8 cited                   | 31 (86.1 %)     | 5       | 0    | **0 %**            | 100 %         | 0 / 36               |
| **A9 cited + committed**   | **34 (94.4 %)** | 2       | 0    | **2.78 % (1/36)**  | 100 %         | 0 / 36               |

The clause did exactly what it was designed to do — the two hedge losses came back, with correct
citations intact:

* `s025`: `无法确认。…上下文没有说明…是同一个人` → **`不是同一个人（按现有记录看，二者没有被关联起来）`**
* `s032`: `无法确定是否同一个。` → **`结论：根据记录看，是同一个——都用 SQLite。`**

## Why it is still rejected

The arm produced the **first unsupported claim in the entire project** (`s016`; unsupported claims
were 0 % in all eleven previous arms). Asked how 小汪's gym habit evolved, A9's summary concludes:

> "…→ 中断一段时间后，**2026-02 因长胖六七斤重新办年卡减肥**" and "可见小汪的健身是
> **"练一段—停一段—因体重反弹再重启"的循环**，且一直有教练指导"

while the audited gold for that query ends with **"（中间那段时间他停没停过，语料里没有交代。）"**.
All ten of s016's own retrieved chunks were re-checked: two show active periods, one is only an
invitation, one a trial, one a restart — **none states that he stopped**, and the only stop-line in
the arm's own context (`我: 不去了，卡快过期了`) is about *me*, which the answer's own footnote
admits. So the commitment instruction made the model **interpolate a state change on exactly the
window the gold says is unestablished**, plus two smaller overreaches in the same sentence
("重新办年卡" implies an earlier annual card; "一直有教练指导" generalises from two mentions).

Precisely: the flag is **groundedness, not correctness** — no positive gold fact is contradicted, which
is why the query still labels PASS. But an invented "stopped, then restarted" cycle about a person is
the textbook false-memory shape, and §36 (`No Evidence = No Memory Claim`) is this project's headline
guarantee. **+1 PASS is not worth shipping the first false memory. A9 is recorded as a negative result
and is not adopted.**

## Decision

* **Reference for quality**: A7 (33/36, 0 unsupported) — unchanged.
* **Product reference**: **A8** (31/36, 100 % citation rate, 0 misleading, 0 unsupported) — citations
  are a product requirement, and its documented cost is 2 PASS, not any loss of groundedness.
* **Rejected**: A9's commitment clause as written. Kept in the codebase as a named, reproducible
  variant (`--answer-prompt cited-committed`) so the negative result can be re-checked.
* For the record, A9 vs A8 was **3 recoveries and 0 regressions** (`s020`, `s025`, `s032` → PASS) and
  vs A7 only `s020` moved: the clause is *effective*, it is simply not *safe*. That distinction is the
  whole reason this phase is a rejection rather than a win.
* **Next experiment (precisely specified by this failure):** narrow the clause — allow cross-source
  *identity / equivalence* conclusions (the three recoveries it earned) while forbidding invented
  temporal structure, e.g. "only conclude what the cited sources state; do not assert that something
  stopped, continued or resumed unless a record says so". Same config, same deterministic retrieval,
  so the delta will again be fully attributable.

## Process fix from this round

The grader's diagnostics sidecar arrived as **malformed JSON** (two unescaped `"` inside a quoted
span; the labels file was fine). It was repaired by re-serialising through `json.dump`, and the gate
list now includes a **JSON-validity sweep over every `*labels.json` / `*diagnostics.json`** before a
commit — 19 files checked, 0 malformed.

---

# Phase 7c — the narrowed clause: citations **and** a clean groundedness record (A10, adopted)

Pre-registered criterion before the run: **PASS ≥ 33 AND unsupported = 0 AND citation honesty
intact.** A9's broad clause proved the mechanism works but is unsafe; this variant keeps the licence
it earned and removes the one it abused (`--answer-prompt cited-narrow`):

* allowed: commit when the sources jointly determine the answer, **including cross-source
  identity/equivalence conclusions**;
* forbidden: asserting that something "stopped / was interrupted / resumed / continued" unless a
  record says so — "where the records are silent, say that the records do not say; never fill the gap
  with an inference".

## Four-arm prompt experiment — retrieval held constant

All four arms (`A7`–`A10`) retrieve **identically, 36/36 pairwise and against the offline
deterministic reference**, so every difference below is attributable to the prompt alone.

| arm                        | PASS            | PARTIAL | FAIL | unsupported  | citation rate | misleading citations | Hit@k |
| -------------------------- | --------------- | ------- | ---- | ------------ | ------------- | -------------------- | ----- |
| A7 timeline (no citations) | 33 (91.7 %)     | 3       | 0    | 0 %          | 0 %           | n/a                  | 100 % |
| A8 cited                   | 31 (86.1 %)     | 5       | 0    | 0 %          | 100 %         | 0 / 36               | 100 % |
| A9 cited-committed         | 34 (94.4 %)     | 2       | 0    | **2.78 %** ✗ | 100 %         | 0 / 36               | 100 % |
| **A10 cited-narrow**       | **33 (91.7 %)** | 3       | 0    | **0 %** ✓    | **100 %**     | **0 / 36**           | 100 % |

Citation metrics (A10): 205 citations, **0 invalid**, gold-in-cited coverage 76.3 % (vs 84.2 %
retrieved), lexical precision 61.5 %, abstention accuracy 100 %.

## What the narrow clause did and did not fix

* **`s016` fixed** (the A9 unsupported claim). A10 now closes with: *"中间是否中断、是否重新开始，
  **记录没有说明，不能断言一直坚持或停过**"* — which is the gold's own caveat, and it correctly notes
  that the 2025-12-27 line is about me rather than 小汪.
* **`s025` still commits** ("不是同一个人"), so the identity licence survived.
* **`s032` reverted to hedging** ("无法仅凭现有记录判断是不是同一个"). A9 committed there; A10 does not.
  Net vs A9: −1 PASS, but groundedness restored (0 unsupported vs 1).

## Decision: **adopt A10 as the product reference**

A10 strictly dominates on the axis this project exists for, and improves on the alternatives:

* vs **A7** (old quality reference): same PASS (33), same groundedness (0 unsupported), **plus**
  verifiable provenance (100 % citation rate, 0 misleading) — a capability A7 did not have at all.
* vs **A8** (previous product reference): **+2 PASS**, same groundedness, same citation honesty.
* vs **A9** (best PASS): −1 PASS, but A9 shipped the project's first false memory; A10 does not.

Non-PASS in A10 (3): `s013` (retrieval — the Brave Search line is absent from every arm),
`s030` (answer-side: the DB is never named although the naming facts were retrieved), `s032`
(equivalence hedge).

**Next experiment (specified by the one remaining prompt-level gap):** add a *default-persistence*
rule for equivalence questions — "if the question asks whether something is the same as before and no
record shows a change, conclude that it is unchanged" — which targets `s032` without reopening the
`s016` hole (that rule concerns staying the same, not stopping/resuming). Same config, same
deterministic retrieval.

---

# Phase 8-lite — product surface: `recall.py` with evidence cards

Until now the evidence model had **no consumer**: citations existed only inside eval JSON. The
engine could prove its numbers but a user could not see *why* an answer was given. This phase adds
the first product-facing surface.

## What was added

* **`evidence_cards.py`** — resolves an answer's `[来源 N]` citations (0-based, matching
  `Source: N`) into `EvidenceCard`s: conversation-session id, start/end timestamps, participants,
  event count, and **the original chat lines themselves**. `render_cards()` prints them;
  `cards_to_dict()` emits JSON. Deliberately model-free, so it is fully testable offline (9 tests).
* **`recall.py`** — the CLI:
  ```powershell
  python recall.py "小王什么时候推荐我用 Supabase 的？"
  python recall.py "我之前说的那个数据库最后到底用了没？" --json
  ```
  It answers with the **evaluated A10 configuration** (session chunking, hybrid RRF, no-rewrite
  workflow, cited answer prompt) and prints the evidence cards under the answer. `--show-uncited`
  also lists retrieved-but-uncited sources when an answer cites nothing.
* **`recall_demo_output.txt`** — a real transcript (synthetic corpus) showing the answer plus the
  cards, e.g. for *"小王什么时候推荐我用 Supabase 的？"* the answer cites `[来源 9]` and the card
  resolves it to `txt-session-0003`, 2024-03-16 20:05–20:44, participants [小王, 我], with the
  original line *"2024-03-16 20:13 小王: 可以试试 Supabase"*.

This closes the loop the project promises: **Answer → Evidence → MemoryEvent → original chat line**,
with the first link measured by `citation_metrics.py` and the last two now visible to a user.

## Real bug found and fixed while building it (Core)

`RetrievalConfig.__init__` calls `llm_config.set_api_key(force_reset=True)`, and that flag made
`set_api_key` **discard an explicitly configured `env_variable_name`** — replacing it with
`OPENAI_API_KEY` and wiping the key. Any setup pointing an OPENAI-supplier endpoint at another
provider (this project's DeepSeek configuration) therefore only worked **by accident of ordering**:
`run_baseline.py` builds the LLM endpoint *before* constructing the `RetrievalConfig`, so the key was
already read; building them in the natural order (config first) produced
`openai.OpenAIError: Missing credentials`.

Fix: derive the default variable name only when none was configured, so `force_reset` still re-reads
the value but no longer clobbers a custom name. Covered by `tests/test_llm_config.py` (4 tests,
including one that asserts the key survives `RetrievalConfig` construction).

Second, smaller finding: `dotenv.load_dotenv()` resolves `.env` **relative to the calling file**, not
the CWD, so a relocated CLI can silently load nothing; `recall.py` now loads the repo `.env` by
explicit path derived from `__file__`.

## Verification

* `72 tests pass` (9 evidence-card + 4 LLM-config new).
* Neither frozen baseline moved: `baseline_summary.json` and `stress_summary.json` still reproduce
  **byte-identically** after the Core fix and the `build_workflow_config` extraction.
* Both demo questions answered with a committed conclusion and traceable cards; `--json` emits
  `question / answer / evidence[] / retrieved / config`.

---

# Phase M2 — Corpus v2: a distractor pack that un-saturates the measurement (adopted as the standing harder benchmark)

## Why: v1 ran out of headroom

Corpus v1 saturates the retriever — **Hit@10 = 100 %**, 0 retrieval misses, evidence coverage 86.27 %.
Every mechanism A/B measured there was fighting over the last few points, and the measured noise floor
(Phase 5: identical configs share only 4–6/36 evidence sets) was the same size as the effects. A harder
corpus that *keeps the gold facts untouched* is the prerequisite for any further honest retrieval work.

## What was built

`data/stress_chats_v2.txt` = corpus v1 **+** a distractor pack (`data/stress_v2_parts/part_{a,b,c}_*.txt`),
merged reproducibly by `build_stress_v2.py`. The 36 gold queries are unchanged and every one of the
114 gold evidence lines survives byte-identically.

| corpus | messages | episodes | session chunks | chars |
| ------ | -------- | -------- | -------------- | ----- |
| v1     | 1,216    | 100      | 101            | 49,457 |
| **v2** | **2,954** | **233** | **234** | **131,347** |

The pack raises retrieval ambiguity **2.3×** without touching a tracked fact: same-entity mentions in
other people's projects, gold-shaped sentences about different things, pronoun/omission lines, and
near-duplicate wording across days. `validate_stress_dataset.py` passes **all 11 hard checks** on v2,
and difficulty diagnostics show the intended pressure (e.g. `s023` now has **102** messages sharing
4-grams with its gold text, `s027` **105**, vs single digits on v1).

## Two defects caught by verification, not by reports

1. **Out-of-order episodes silently destroy the corpus.** `build_sessions` consumes **file order** and
   never re-sorts, and every calendar-day change forces a flush. One writer's draft was not
   chronological, which fragmented its part into **293 chunks of ~43 chars** instead of 40 coherent
   sessions. Fixed at the source and pinned by a test; the build script now fails loudly if the merged
   file is not strictly ascending.
2. **The pack must never restate a gold line.** A first version of the overlap check compared v1
   against *itself* (it sliced "the pack" off the **sorted** list, re-admitting v1 episodes), reporting
   63 phantom collisions. Re-tagged per source; real result is **0 collisions**.

## The corpus change, measured at the retrieval stage the arms actually use

`--workflow no-rewrite` makes retrieval deterministic, so the following is exact, not sampled:

| arm | Hit@5 | strict Hit@5 | evidence coverage | zero-coverage queries |
| --- | ----- | ------------ | ----------------- | --------------------- |
| v1 (A10 recheck) | 34/34 | 34/34 | **0.8627** | 0 |
| v2 (corpus v2)   | 34/34 | 34/34 | **0.7966** | 0 |

**−6.62 pts coverage, and still zero retrieval misses** — the pack costs evidence *quality/precision*
without making any question unanswerable from top-10. Paired: **6 queries worse, 1 better, 27 unchanged**.

An earlier dense-only offline probe predicted a larger **−9.31 pts** (and one rank-24 miss). That probe
is *directional* for the hybrid configuration, not exact: A10 fuses BM25, which recovers several lines
dense ranking dropped. Recorded so the probe is not over-trusted later.

## End-to-end paid A/B (identical configuration A10, only the corpus differs)

| arm | PASS | PARTIAL | FAIL | PASS % | unsupported claims | insufficient evidence | citation rate | invalid citations | abstention | latency |
| --- | ---- | ------- | ---- | ------ | ------------------ | --------------------- | ------------- | ----------------- | ---------- | ------- |
| v1 (A10 recheck) | **33** | 3 | 0 | 91.7 % | **0/36** | 2 | 100 % | 0 | 100 % | 18.6 s |
| v2 (corpus v2)   | **28** | 8 | 0 | 77.8 % | **0/36** | 8 | 100 % | 0 | 100 % | 19.5 s |

**The v1 recheck first validates the baseline**: with the current code it reproduced the recorded A10
result exactly — same PASS count (33), same coverage (0.8627), and **byte-identical retrieval on 36/36
queries**. The `set_api_key` fix changed nothing behaviourally, so the v2 comparison is not confounded
by it.

Net effect of the harder corpus: **−5 PASS (33 → 28), 0 FAIL, and 0 false memories.**
Flips: 6 losses (`s016 s021 s023 s025 s029 s030`) and 1 gain (`s020 PARTIAL → PASS`).

## The finding that matters: distractor pressure costs *recall*, not *truthfulness*

Every single loss is retrieval-caused — `evidence_sufficient=false` for exactly the 8 v2 PARTIALs vs 2
in v1 — and in every one of them the model **declined or enumerated incompletely rather than inventing**:

* `s025` (entity disambiguation, coverage 1.000 → **0.250**) — answers "无法判断是否为同一人" where gold
  says 不是; the decisive 王哥→小王 link was displaced from top-10.
* `s023` (earliest state, 0.750 → **0.250**) — correctly identifies Neon as 同学A's recommendation, then
  refuses the ordering question; the 2024-07-21 anchor was displaced.
* `s021` (0.500), `s029` (0.750), `s016` (0.750), `s030` — the same shape: one decisive low-frequency
  line (a date, a one-off remark, a single reply) pushed out by topically identical distractors, and the
  answer honestly reports the record's silence instead of fabricating.

This is the corpus doing its job **and** the groundedness discipline holding under 2.3× pressure: on the
metric this project exists for — **no evidence, no memory claim** — A10 scored 0 fabrications on both
corpora, with 100 % abstention accuracy on the two deliberately unanswerable queries. The cost of a
harder corpus is measured in PARTIALs, not in false memories.

## Honest caveats

* `s020` flipped **PARTIAL → PASS** in v2 while its coverage was unchanged (0.250) — reciprocal flips
  exist, so a single-run ±1 PASS is still inside noise. The −5 is far outside it, but per-query flips are
  not each individually trustworthy.
* `s030` had **identical coverage** (0.250) in both arms yet flipped PASS → PARTIAL: equal coverage
  masked a different *composition* of retrieved lines. Coverage is a scalar and does not capture which
  decisive line arrived.
* One citation-grounding error in v2 (`s025` cites a 同学A chunk for a claim about 张三's records) — the
  only `citation_supports_claim=false` across both arms. It is a precision defect, not a false memory.
* The pack was written by subagents under a spec and is **verified** (format, allocations, gold overlap,
  tracked-fact scan) but is synthetic text, not real chat history.

## Decision: **keep corpus v2 as the standing harder benchmark**; A10 unchanged as product reference

A10 is not modified by this phase. v2 becomes the corpus future retrieval work is judged on, because it
is the only instrument in the repo with measurable headroom left. Both corpora stay frozen:
`stress_chats.txt` (v1, historical arms) and `stress_chats_v2.txt` (v2), with `build_stress_v2.py` able to
regenerate v2 byte-identically from the checked-in pack.

## Next step (motivated by the measured failure mode)

The dominant failure is now **decisive-line displacement among near-duplicate distractors** — not a lack
of recall, and not generation. The next mechanism to test is therefore retrieval-side **question
decomposition / targeted anchors**: Phase 3 measured per-slice decomposition as *worse at matched
budget*, but that negative was taken on a corpus with Hit@10 = 100 %, where the mechanism had no headroom
to demonstrate anything. On v2 there are 8 insufficient-evidence queries to win, so it is worth re-testing
offline (free, deterministic) before any paid arm — with the pre-registered target of converting the 6
flips back to PASS **without** introducing a single unsupported claim.
