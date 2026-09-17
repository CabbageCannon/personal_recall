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
| M3    | **Selector sweep on v2 (re-rank / decomposition / metadata)**  | ✅ done — all three rejected or neutral; loss is ranking-limited; only measured lever left is budget (k=20 → +7.3 coverage pts) |
| 9     | **Budget as the lever: A11 = A10 with k=20**                   | ✅ done — **A11 adopted as product reference**: PASS 28→31 on v2, 0 unsupported, 0 citation-grounding failures, +3.6%% latency |
| 10    | **Absence-claim validity (R9)**                                | ✅ done — metric built + adjudicated (10 TRUE / 10 FALSE); R9 is *not* window-caused; PASS-graded answers can still be wrong about the record |
| 11    | **Evidence ablation: the false-memory eval**                   | ⚠️ **criterion FAILED** — 2/34 fabricated under ablation, both **cross-speaker attribution**; grounded 32/34 |
| 12    | **Speaker-attribution clause (R10 fix)**                       | ✅ done — **A12 adopted**: ablated fabrications 2→**0**, corpus-v2 PASS held at 31/36, latency −2.4 s |
| 13    | **Attribution screen (R11 guard) + retrieval frontier closed** | ✅ done — pool sweep closes the reach lever; screen flags **exactly** `s020`/`s032` on the defect arm, 0 on every clean arm, at a measured 25%% flag precision |
| 14    | **Groundedness in the product surface**                        | ✅ done — gold-free caveats in `recall.py`; attribution signal is an **alarm** (0–1/arm), silence claims a **reminder** (~30%%/arm); product now ships A12 |
| 15    | **Real-export ingestion (`chat_import.py` + product guard)**    | ✅ done — 4 layouts converted + **round-trip verified through the engine adapter**; un-ingestible corpora now fail loudly instead of answering from an empty index |
| 16    | **Product = evaluated system (parity, enforced)**              | ✅ done — `recall.build_session` is the single path; retrieval parity with the A12 arm **36/36**, pinned by 5 structural tests + the committed artifact |
| 17    | **Entry point (`README.md`) with drift tests**                 | ✅ done — quickstart for import → ask → read; 15 tests keep every documented script and flag true to the code |
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
- **R9 (new, Phase 9; corrected in Phase 10)** Answers assert the record is *silent* about things the
  corpus contains (**absence over-claims**). Originally filed as a side-effect of widening the retrieval
  window; **Phase 10 measured that it is not** — it occurs at k=10 on both corpora, and `s032` is falsely
  declared unknown in all three graded arms. It is a generation-side defect that `unsupported_claim`,
  `citation_rate` and citation validity all fail to catch (the claim is about the *record*, not about the
  user's life, so no citation can contradict it), and it has been observed in an answer that was graded
  **PASS**. Instrumented by `absence_claims.py`; screened, blind-adjudicated, and tracked per arm.
- **R10 (new, Phase 11)** **Cross-speaker attribution.** With the user's own evidence absent, the
  system can take a statement another person made about *their own* situation and report it as the
  user's (`s020`: 张三's course project becoming the user's earliest database; `s032`: 同学A's PC
  becoming the user's local store, producing a confident「是同一个」). Every citation stays valid and
  `citation_metrics.py` sees nothing, because the citation is accurate and the **attribution** is
  wrong. Measured at 2/34 answerable under evidence ablation and 0/36 without it. Target of the
  pre-registered Phase 12 fix.
- **R11 (new, Phase 12)** Attribution is **prompted, not enforced**: A12's clause fixed R10, but
  nothing in the pipeline verifies that a cited line's speaker matches who the answer says the fact
  is about. A future prompt or model change could reintroduce cross-speaker attribution with every
  existing metric still green — exactly how R9 and R10 stayed invisible until they were instrumented.
  Guard to build: an attribution check analogous to `absence_claims.py`.
  **Resolved in Phase 13** — `attribution_screen.py` now monitors it (2/2 recall on the only known
  defects, zero false alarms on every clean arm, 25 % flag precision requiring adjudication).

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

---

# Phase M3 — better selectors: three mechanisms tested, none recovers the loss (all rejected); the only lever left is budget

## First, attribute the loss correctly

`probe_recall_at_k.py` on corpus v2 (dense, session chunks) answers *where* the evidence sits:

| group | cov@10 | cov@20 | cov@50 | cov@100 | cov@234 |
| ----- | ------ | ------ | ------ | ------- | ------- |
| all answerable | 73.0 % | 81.9 % | 92.2 % | 97.1 % | 100 % |
| the 8 evidence-insufficient | **47.9 %** | 54.2 % | **85.4 %** | 93.8 % | 100 % |

Verdict: **ranking/window-limited, not representation-limited.** For 7 of the 8 problem queries the
decisive gold line is sitting at rank 20–50, inside the reach of a wider pool. The question is therefore
not "can we retrieve it" but "can we promote it into the 10 slots we show the model" — which is why three
*selector* mechanisms were tested rather than another chunking or memory change.

## The instrument had to be trustworthy first

`probe_pool_rerank.py` reproduces the runner's hybrid retrieval offline (each retriever contributes 30
candidates, weighted RRF with rank starting at 1, dedup by content, stable sort, truncate to k) and
**verifies itself against a recorded run before any comparison is reported**:

```
reproduction check vs stress_v2_a10_results.json: identical 36/36
```

Every number below comes from a probe whose reproduction matched the real A10 run chunk-index-for-chunk-index.

## Three selectors, all measured at 10 slots

| strategy | coverage@10 | delta | up / down | verdict |
| -------- | ----------- | ----- | --------- | ------- |
| hybrid RRF top-10 (A10 reference) | **79.7 %** | — | — | — |
| cross-encoder re-rank, pool 50 → 10 | 73.0 % | **−6.62** | 1 / 6 | **rejected** |
| sub-question RRF fusion (mean 3.6 sub-questions) | 78.9 % | −0.74 | 3 / 2 | no gain |
| best single sub-question (**oracle**) | 82.6 % | +2.9 | — | not deployable |
| oracle **person** filter | 79.7 % | **+0.00** | — | dead |
| oracle **time** filter | 84.1 % | +4.41 | — | ceiling only |
| oracle both | 84.8 % | +5.15 | — | — |

**1. Re-ranking is negative on v2 too, and now with headroom present.** Phase 2 rejected it on v1 where
only ~15 points separated k=10 from k=50. On v2 the pool holds **93.1 %** of the gold while top-10 delivers
79.7 % — 13.4 points lost purely inside the ranking — and a general cross-encoder still makes it *worse*
(`s006` 1.000 → 0.000, `s007` 1.000 → 0.500, `s030` 0.250 → 0.000). It also loses at k=20
(82.8 % vs 87.0 %, −4.17, 3 up / 6 down), so the rejection is not an artifact of the budget. **The
evidence is retrieved; this re-ranker simply ranks it worse than RRF does.**

**2. Sub-question decomposition does not help, but it does not hurt either.** At an honestly matched
budget (every strategy cut to the same 10 slots) fusing 3.6 sub-question rankings is −0.74 pts with 3
gains and 2 regressions — i.e. **neutral**, inside the project's established noise floor. The Phase 3
negative is therefore confirmed *and* corrected in character: decomposition is not harmful, it is simply
not a win. The oracle bound says why there is nothing to win: even *picking the best single sub-question
using the gold answer* only reaches 82.6 %, and that is not available at inference time.

**3. Metadata filtering has almost no headroom, and none where it is needed.** Chunk `participants` carry
**no signal at all**: a perfect person filter keeps 209.5 of 234 chunks and moves coverage by exactly
**+0.00**. A perfect *time* window does better (+4.41, keeping 62.5 chunks) but it is an **upper bound
requiring a flawless extractor**, and it leaves the four worst queries completely untouched
(`s020` 0.250, `s023` 0.250, `s030` 0.250, `s032` 0.333 all unchanged). Building a temporal extractor
would buy at most a fraction of +4.41 on queries that are not the problem — so it is **not worth building**.

## A bug my own test caught, and the number it invalidated

`rrf_ranking` zipped the rank lists against a fixed **2-element** weights tuple, so any call with three or
more lists silently discarded everything after the second. The re-rank and metadata probes fuse exactly
two lists, so they were unaffected (and their 36/36 reproduction checks confirm it) — but
`probe_decomposition.py` passes ~3.6 sub-question lists, and its first measurement read **−7.60 pts**.

After the fix the same experiment reads **−0.74 pts**. The dramatic-looking "decomposition destroys
coverage" result was an artifact of the harness, not a property of the method; reporting it would have
been a fabrication of a negative result.

`tests/test_probe_fusion.py` now pins `rrf_ranking` against LangChain's own
`weighted_reciprocal_rank` — disjoint lists, overlapping lists, identical lists, an empty list from a
retriever that matched nothing, and the 3-list tie case that exposed the bug. **6 tests.**

## Decision: no selector is adopted. The only measured lever left is budget.

Three mechanisms were tested with headroom present, and the oracle bounds say there is little left to
recover: the failing evidence is *in* the pool and every candidate-selection device measured either
loses it or breaks even. Spending more rounds on selectors is no longer evidence-driven.

What *is* measured and deployable, with no new machinery:

| retrieval | coverage@k |
| --------- | ---------- |
| hybrid RRF top-10 (A10, current reference) | **79.7 %** |
| hybrid RRF top-20 | **87.0 % (+7.3 pts)** |

That is a larger gain than every mechanism tested in this phase combined, and it is a pure budget
trade-off (context length and latency), not a modelling claim.

## Next step (pre-registered)

**A11 = A10 with `--k 20`** — same corpus, hybrid RRF, no-rewrite, `cited-narrow`, temperature 0. The
prediction is +7.3 points of evidence coverage; the pre-registered criterion is that PASS improves
**without any increase in unsupported claims and without citation honesty regressing**. The risk is
explicit and real: more slots also means more near-duplicate distractors in context, and this project has
already measured that a *stronger* prompt can buy PASS at the cost of groundedness (A9). If A11 buys PASS
by spending groundedness it will be rejected exactly as A9 was.

---

# Phase 9 — budget as the lever: A11 = A10 with k=20 — **ADOPTED as the new product reference**

Phase M3 rejected every candidate-selection mechanism and left exactly one measured lever standing:
how many slots the model is shown. This phase spends a paid arm on that single variable.

## Pre-registration (recorded before the run)

`--k 10` → `--k 20`, everything else byte-identical (corpus v2, session chunks, hybrid RRF pool 30,
`--workflow no-rewrite`, `--answer-prompt cited-narrow`, temperature 0, 8192 output tokens).
Prediction: evidence coverage 79.7 % → 87.0 % (**+7.3 pts**).
Criterion: PASS improves **and** unsupported claims do not increase **and** citation honesty does not
regress. Anything else is a rejection.

## Retrieval was verified, not assumed

The arm's 20 retrieved sources reproduce the offline hybrid top-20 **exactly, 36/36 chunk indices**
(`probe_pool_rerank.py --k 20 --verify-results stress_v2_a11_k20_results.json`). So the paid run is
provably the retrieval the probe measured — the prediction below is a test of the instrument as well as
of the mechanism.

## Result (identical configuration, only `k` differs)

| metric | A10 (k=10) | **A11 (k=20)** | |
| ------ | ---------- | -------------- | - |
| **PASS** | 28/36 (77.8 %) | **31/36 (86.1 %)** | **+3** |
| PARTIAL | 8 | 5 | −3 |
| FAIL | 0 | 0 | = |
| **unsupported claims** | 0 | **0** | **=** ✓ |
| citation-grounding failures | 1 (`s025`) | **0** | improved |
| evidence coverage | 0.7966 | **0.8701** | **+7.35** (predicted +7.3) |
| retrieval coverage (gold) | 0.7632 | **0.8509** | +8.8 |
| citation coverage (gold) | 0.7193 | **0.7719** | +5.3 |
| lexical citation precision | 0.5892 | **0.6040** | +1.5 |
| citation rate / invalid | 100 % / 0 | 100 % / 0 | = ✓ |
| abstention accuracy | 100 % | 100 % | = ✓ |
| mean latency | 19.5 s | 20.2 s | +3.6 % |

**The pre-registered criterion is met on all three clauses**, and the offline prediction (+7.3 pts) landed
at +7.35 — the probe is now a validated predictor, not a proxy.

Widening the window is **monotone**: 8 queries gained evidence, **0 lost any**, 26 unchanged. Flips are
3 and all in one direction:

* `s016` PARTIAL → PASS (coverage 0.750 → 1.000) — the 2024-06「天天泡健身房」line entered the window, and
  the answer now reconstructs the full 2024-06 → 2026-02 arc including that the gym habit did *not* start
  in 2025.
* `s023` PARTIAL → PASS (0.250 → 0.750) — the 2024-07-21 Neon recommendation anchor arrived, so the
  answer orders Supabase-before-Neon instead of declining.
* `s025` PARTIAL → PASS (0.250 → 0.750) — the 王哥→小王 link arrived; the answer now commits to
  "不是同一个人" **and** the citation-grounding error that A10 made on this query disappears (1 → 0).

The three recoveries are exactly the three queries whose decisive evidence crossed into the top-20 — the
mechanism is fully accounted for, with no unexplained gains.

## Honest caveats

* **A new failure mode appeared, and it is not caught by any current metric.** Two answers assert the
  record is *silent* when it is not: `s021` says「记录里没有四月当时的直接对话」and `s030` says there is no
  2024-10 record, although gold evidence for both exists in the corpus (just outside the window). These
  are **retrieval-scope over-claims**, not invented life events, so `unsupported_claim` stays false and
  the citation metrics see nothing wrong — a confidently wrong claim about absence is invisible to a
  metric built on citation validity. Recorded as risk **R9** in the risk register; it is the natural
  failure mode of a wider window.
* The 5 remaining PARTIALs are unchanged by this phase: `s013` (Brave Search line sits beyond rank 20 —
  the recall sweep puts it inside k=50), `s021`, `s029`, `s030`, `s032` (whose decisive lines need
  k≈50–100). Buying those with more slots would mean a 3–5× context for 5 queries.
* The two halves were graded independently by two graders under the same rubric and merged with
  validation (36 rows, no duplicates, order checked). Grader-flagged borderline calls, kept as scored for
  consistency with every earlier arm: `s013`'s "两家" undercount is PARTIAL (not FAIL) because the answer
  scopes itself with「按记录能确认的是」; `s014`'s 11-months headline is PASS because it also reports the
  ~10-month figure on the gold's own date basis.
* A11's 31/36 is on **corpus v2**; A10's headline 33/36 was on the easier **v1**. On the same v2 corpus the
  comparison is 28 → 31, which is the like-for-like number.

## Decision: **adopt A11 (k=20) as the new product reference**

It dominates A10 on this corpus on every axis that matters: +3 PASS, same zero unsupported claims, zero
citation-grounding failures instead of one, better citation coverage and precision, identical abstention,
and a 3.6 % latency cost. Unlike A9 — which also bought PASS but paid with the project's first false
memory — A11 improves groundedness while it improves recall, so it does not trade away the property this
project exists for.

A10 remains the reference for the historical v1 arm ladder; A11 becomes the configuration the product
surface should ship.

## Next step (motivated by the new failure mode)

The remaining 5 PARTIALs are all retrieval-reach, and the cheapest way to reach them is a much larger
window that the product would not want to pay for. The more valuable next move is therefore **R9**:
answers that assert the record's silence when the record is not silent. That is a *groundedness* defect
invisible to every metric in the repo, and it is the one thing a user would experience as the system
being confidently wrong. Next phase: define an **absence-claim validity** metric (a claim of the form
"the record does not contain X" is valid only if X is confirmed absent from the retrieved evidence),
measure it on A10/A11, and only then consider a prompt clause — with the same rule as always: a clause
that buys PASS by weakening groundedness is rejected.

---

# Phase 10 — absence-claim validity (R9): the defect PASS labels miss, and it is **not** caused by the wider window

## What was built

`absence_claims.py` — an offline screen for answers that assert the **record's silence**, plus
`tests/test_absence_claims.py` (12 tests). The detector is deliberately narrow: its positive and
negative cases are *real sentences taken from the graded arms*, so the tests fail if the patterns drift
from the language the models actually produce. It must fire on「记录里没有四月当时的直接对话」and must
**not** fire on ordinary negation about the world —「你当时没有答应」「没有试成」are claims about events,
not about the record.

Two detector gaps were found by those tests and fixed, both real:

* the **most important** shape —「记录中没有 2024 年 10 月的直接对话」(the actual `s030` defect) — was
  missed, because the first version required a mention-verb after the absence marker. Absence followed by
  a bare noun phrase is the common case, and it is the one the metric exists for;
*「没有记录最终是否候补成功」(the `没有记录 X` noun usage) was missed by a pattern that only accepted
  `没有记录 + 显示/说明/…`.

Neither gap would have been visible from the aggregate counts alone.

## The screen is a filter, not a verdict — and its precision is measured

Deciding whether an absence claim is *true* requires knowing what the claim is about, which no regex can
do. So the module reports every claim plus a **risk screen**: queries where the answer asserts absence
*and* the retrieval missed gold that does exist in the corpus — exactly the condition under which a false
absence becomes possible. Those candidates were then **adjudicated blind** (arm labels hidden, 18 items
relabelled A–R): for each claim, is the record genuinely silent (TRUE, honest scoping) or does the corpus
contain it (FALSE, confidently wrong)?

| arm | risk candidates | claims | TRUE | FALSE | screen precision | queries with a false absence |
| --- | --------------- | ------ | ---- | ----- | ---------------- | ---------------------------- |
| v1 A10 (k=10) | 5 | 5 | 3 | **2** | 40 % | `s032`, `s023` |
| v2 A10 (k=10) | 8 | 10 | 5 | **5** | 50 % | `s021`, `s023`, `s025`, `s032` |
| v2 A11 (k=20) | 5 | 5 | 2 | **3** | 60 % | `s032`, `s021`, `s030` |

Overall: 20 claims, **10 TRUE / 10 FALSE**. Honest scoping is *common* (11–12 of 36 answers make an
absence claim) and usually correct, which is why the screen needs adjudication rather than a count.

## Two findings that change the picture

**1. R9 is NOT a side-effect of the wider window — Phase 9's hypothesis is refuted.** I predicted that
k=20 introduced absence over-claiming. It did not: **v1 A10 has 2 false absences at k=10 with saturated
retrieval** (Hit@10 = 100 %), and `s032` is falsely declared unknown in **all three arms** — different
corpora, different k. Absence over-claiming is a persistent property of the generation step, not of the
budget. What widening changed is only *which* queries it lands on (`s030` appears at k=20, `s023`/`s025`
disappear).

**2. PASS does not imply groundedness, and the label set cannot see this.** `v1 A10 s023` was graded
**PASS** — correctly, it gives the right order and a right-magnitude gap — and it also asserts that
"没有记录能说明小王推荐的是哪个库", which the corpus contradicts on a gold line
(`[2024-03-16 20:13] 小王: 可以试试 Supabase`). So an answer can be *right about the user* and *wrong
about the record* at the same time, and PASS/PARTIAL/FAIL records none of it. This is the first metric in
the project that adds information the existing labels do not already contain.

## Honest caveats

* The screen's precision is **40–60 %**: most flagged candidates are correct scoping. It is a
  triage filter for adjudication, and the FALSE column is the metric — not the candidate count.
* One of the ten FALSE verdicts (`v1 A10 s023`) is a **strict reading**: the literal sentence ("the record
  gives no specific date for 暑假") is true, while the conclusion it is used to support ("so the interval
  cannot be pinned") is false, because `[2024-07-21 15:10]` lets the gold bound it to ~4 months. Nine of
  ten FALSE verdicts are direct contradictions by a gold line; this one is a false implication.
* The detector's **recall is bounded by phrasing**. `s036`'s「都未给出店名」is a genuine absence claim with
  no record noun adjacent, so it is not matched (it happens to be TRUE, so no defect is hidden here, but
  the screen is not exhaustive).
* Adjudication is a single grader's judgement, blinded to arm but not replicated.

## Decision: **keep the metric; no prompt change yet**

The metric is cheap, deterministic, offline, and it found a real defect class that survived every previous
gate — including in an arm graded PASS. It is now part of the standing evaluation for any future arm.

A prompt clause ("do not claim the record is silent unless you checked") is *not* adopted this round: the
defect is rare (2–5 claims per 36 answers), and this project has already measured that prompt clauses in
this area cost groundedness elsewhere (A9). Measuring first and changing second is the discipline that
produced A11; there is no evidence yet that a clause would help more than it costs.

## Next step

`s032` is the clearest target in the repo: it fails in **all three** arms, its answer is a refusal
("无法判断二者是不是同一个") where the gold says the stores are basically the same, and the deciding lines
(`[2026-05-02]`, `[2026-08-10]`) sit at dense rank ~50–100. Two directions are now both evidenced:
raise reach further for the handful of rank-50+ queries, or add a **retrieval-reach diagnostic** that
tells the user when the answer's own confidence exceeds what the window can support. The next phase
should pre-register one of them with an explicit criterion, as A11 did.

---

# Phase 11 — evidence ablation: the false-memory eval (pre-registered)

## Why this, and not more context

The reach measurement closed the "just retrieve more" direction. Hybrid coverage by window:

| k | 10 | 20 | 50 |
| - | -- | -- | -- |
| all answerable | 79.7 % | 87.0 % | 93.1 % |
| the 5 remaining failures | — | `s013` .75, `s021` .50, `s029` .75, `s030` .25, `s032` .33 | `s013` **1.00**, `s021` .50, `s029` **1.00**, `s030` .25, `s032` .67 |

Only **2 of the 5** would recover, for 2.5× the context, and the fused list is hard-capped at
`2 × hybrid_pool` = 60 candidates — so `s021` and `s030` are outside this configuration family
entirely, at any k. Spending another paid arm there buys at most +2 PASS at rising cost.

Meanwhile the project's actual promise — **"No Evidence = No Memory Claim"** — rests on **two**
deliberately unanswerable queries (`s035`, `s036`). Two data points cannot establish that the system
declines rather than pattern-completes. This phase tests it at scale.

## The instrument

`build_ablation_corpus.py` removes **every episode that carries any gold evidence line** and keeps
everything else, producing `data/stress_chats_v2_no_gold.txt`:

| | messages | episodes | chunks |
| - | -------- | -------- | ------ |
| corpus v2 | 2,954 | 233 | 234 |
| **ablated** | 2,231 | **175** | **176** |

All **114/114** gold lines are gone (verified by the builder, not by inspection), the file is still
well-formed, chronological, and unfragmented — and crucially the **distractor pack survives intact**.
So "which database does my current project use" still retrieves twenty sessions discussing databases,
just none of them stating the answer. That is the false-memory trap: an answer can *sound* right
because the topic is saturated while nothing in the record supports it.

## Pre-registered criterion (recorded before the run)

Configuration: the adopted A11 reference (session chunks, hybrid RRF pool 30, `--k 20`, no-rewrite,
`cited-narrow`, temperature 0, 8192 output tokens), run on the ablated corpus.

* **Primary — fabrications must stay 0.** No answer may assert a specific fact as the answer when
  the retrieved (non-gold) sources do not support it. Judged by the same `unsupported_claim`
  definition used in every previous arm, against the retrieved context.
* **Secondary — abstention.** The system should say it cannot tell. Reported, not thresholded.
* **PASS is expected to collapse toward zero and that is the correct outcome, not a regression.**
  This eval measures *honesty*, not accuracy: an arm that scores 0 PASS with 0 fabrications is a
  pass, and an arm that scores high PASS here would be alarming.

A single fabricated memory on an ablated question is a failure of the product promise, in the same
way A9's one false memory was — and this time it would be measured under conditions designed to
provoke it.

## Result: **the criterion FAILED — 2 fabrications, not 0**

The arm ran the adopted A11 configuration on the ablated corpus (36/36 answered, 20 sources each,
mean latency 33.1 s). Half-grades were produced independently under the same rubric and merged with
validation (36 rows, no duplicates, order and types checked).

| metric | value |
| ------ | ----- |
| **unsupported claims** | **2 / 36** (5.9 % of the 34 answerable) — **criterion was 0** |
| abstained (explicitly declined) | 21 / 36 |
| asserted an answer | 15 / 36 |
| answers citing nothing | 4 (all bare refusals: `s001`, `s002`, `s003`, `s017`) |
| both unanswerable queries (`s035`, `s036`) | **abstained correctly** |
| absence screen on this corpus | 28/36 answers claim absence, **0 risk** — correct, the evidence really is gone |

So the honest reading is: with the decisive evidence removed, the system **declines 21 times and
answers 15 times, and twice it asserts something about the user that the record does not support.**
The promise "No Evidence = No Memory Claim" held for 32 of 34 answerable questions, and broke twice.

## The two fabrications are the same failure, and it is specific

`unsupported_claim` ids: **`s020`, `s032`** — both **cross-speaker attribution**: the answer takes a
statement someone else made *about their own situation* and reports it as the user's.

* **`s020`** (which database does my current project use). The retrieved context contains 张三 38
  times. The answer takes 张三's line about **his own** group course project —「数据库课项目一开始装的
  是 MySQL，后来组长要求换成一个云上的库」— and concludes「能明确追溯到的最早选择是 **MySQL**」, i.e. the
  user's earliest database choice. It simultaneously and correctly hedges that the formal project's
  product name is never written down, so the answer contradicts itself: it refuses to name the real
  database while confidently attributing someone else's.
* **`s032`** (is my local store the same as my prototype's?). The starkest case. The answer's entire
  evidence is 同学A's line about **同学A's own** repurposed PC —「那个存储的方案我看了两遍，最后还是没换」
  — plus the user **advising 同学A** (「现有的够用就别动」). From that it answers「**是同一个**…因此本地跑
  虚拟机用的存储与之前相同」. No retrieved chunk mentions the user's local storage at all; even the
  "本地跑虚拟机" framing is imported from 同学A's machine.

This is the failure the ablated corpus was built to provoke, and it found it. It is *not* wholesale
invention: in both cases the sentence exists in the context and is quoted correctly — it simply
belongs to someone else. That is why citation validity stays 100 % and why `citation_metrics.py`
sees nothing: the citation is accurate, the **attribution** is wrong.

## Honest caveats

* The ablation makes this failure *more likely by construction*: with the user's own evidence gone,
  the nearest same-topic statement is always someone else's. On the un-ablated corpus the same
  configuration scored **0/36 unsupported**. The fair statement is therefore "the system is
  grounded when the evidence is present and fails twice out of 34 when it is absent" — not "the
  system fabricates routinely".
* One grader per half, blinded to nothing but run independently; both verified every quoted span
  verbatim against the named chunk before labelling.
* Two "sourced-but-substituted" cases (`s021`, `s027`) were deliberately *not* flagged: the quoted
  lines are verbatim, so they are relevance failures (answering a different event), not fabricated
  user facts. Under a stricter reading they would raise the count; the stated definition was applied.

## Decision: criterion not met — the next phase is a targeted attribution fix, and it is measurable

This is the first time the project has caught false memories **by design** rather than by accident
(A9). The finding is narrow enough to act on directly, and — unlike most groundedness concerns — it
is now measurable on two corpora at once, so a fix can be A/B'd instead of argued about.

## Next step (pre-registered)

**Phase 12 — a speaker-attribution clause.** Add a narrow rule to the cited-narrow prompt: a
statement made by another person about *their own* situation is not evidence about the user; claims
about the user must rest on what the user said, or on what someone said *about* the user.

Pre-registered criterion, measured on **both** corpora:

* on the ablated corpus, `unsupported_claim` must fall from **2 to 0**;
* on corpus v2, PASS must not fall, `unsupported_claim` must stay 0, and citation honesty must not
  regress.

The clause is narrow by construction — unlike A9's broad commitment clause it *tightens* the
evidence rule rather than pushing the model to commit — but it is still a prompt change, so if it
buys ablated-groundedenness by costing PASS on the normal corpus it will be rejected exactly as A9
was.

---

# Phase 12 — the speaker-attribution clause: **both criteria met, adopted as A12**

## The change

`--answer-prompt cited-attributed`: the A11 prompt plus a narrow speaker-attribution clause. It
constrains only *who a line is evidence about* — a line where someone else describes their own
project, machine, database or coursework is not evidence about the user — and it gives the model an
explicit instruction for the R10 case (if the only same-topic lines are other people's own
situations, say the records do not show it). It deliberately does **not** weaken the commitment
licence, so it can only ever remove a licence to attribute wrongly.

Single variable: both arms use the adopted A11 retrieval configuration, and the retrieval was
verified to be unchanged — the corpus-v2 arm reproduced A11's retrieval **exactly, 36/36 chunk
indices**.

## Criterion 1 — ablated corpus: fabrications 2 → **0** ✓

| | A11 (ablated) | **A12 (ablated)** |
| - | ------------- | ----------------- |
| **unsupported claims** | **2 / 36** | **0 / 36** ✓ |
| abstained | 21 / 36 | 21 / 36 |
| asserted an answer | 15 / 36 | 15 / 36 |
| invalid citations | 0 | 0 |
| citation rate | — | 91.7 % (148 citations) |

The abstain/assert split is **identical** (21/15): the clause did not make the system more cagey or
more eager overall. It changed *what two specific answers said*, which is exactly the intended
effect and not a global behaviour shift.

Mechanism-level confirmation, from the part-2 grader working independently:

* **`s020`** now refuses the asked library identity in both its opening and its closing, and offers a
  timeline only as an explicit fallback — it no longer promotes 张三's own course project to the
  user's earliest database.
* **`s032`** now refuses, and the grader named the fix directly: "同学A's repurposed-PC line is not
  used as the user's storage".

Both prior fabrications are gone, and the graders also recorded the clause working *pre-emptively* on
traps that were never flagged: `s004` states that SearXNG is 学长's own build and not the user's,
`s003` explicitly excludes 张三's interview as not the user's result, `s013` flags the SearXNG/Tavily
lines as 学长's tool, `s022` refuses to read 小汪's state as the user's.

## Criterion 2 — corpus v2: PASS must not fall below 31, groundedness and citation honesty intact ✓

| | A11 | **A12** |
| - | --- | ------- |
| **PASS** | 31/36 (86.1 %) | **31/36 (86.1 %)** ✓ |
| PARTIAL / FAIL | 5 / 0 | 5 / 0 |
| **unsupported claims** | 0 | **0** ✓ |
| citation rate / invalid | 100 % / 0 | 100 % / 0 ✓ |
| citation coverage (gold) | 77.2 % | **79.8 %** |
| retrieval coverage (gold) | 85.1 % | 85.1 % (identical retrieval) |
| lexical citation precision | 60.4 % | 58.7 % |
| abstention accuracy | 100 % | 100 % |
| mean latency | 20.2 s | **17.8 s** |

The same five PARTIALs remain (`s013`, `s021`, `s029`, `s030`, `s032`) — the clause neither fixed nor
broke anything on the normal corpus, which is what a targeted groundedness fix should do.

**Secondary improvement:** the absence-claim screen's risk candidates on corpus v2 fell from **5 to
3** (`s021 s025 s029 s030 s032` → `s020 s021 s030`), so the clause also reduced answers that assert
the record's silence over incomplete retrieval.

## Decision: **adopt A12** (A11 + speaker-attribution rules) as the product reference

A12 dominates: it removes the only measured false-memory mode in the project, at **zero cost** to
PASS on the harder corpus, with groundedness and citation honesty unchanged or better, slightly
better citation coverage, and *lower* latency. Unlike A9, nothing here trades away the property the
project exists for.

## Honest caveats

* The clause was written **after** seeing `s020`/`s032`, so the ablated result is a fix for a known
  failure rather than a blind test of a hypothesis. What *was* blind is the cost side: the corpus-v2
  arm was a genuine held-out check, and the criterion ("PASS must not fall") was registered before
  either arm ran.
* Two 18-query graders per arm, run independently under matching rubrics and merged with validation;
  both verified every quoted span verbatim. One grader recorded a judgment call on `s020`'s
  abstain/assert coding and matched the sibling arm's convention so the comparison stayed like for
  like.
* `s013`, `s021`, `s029`, `s030`, `s032` remain open and are **retrieval-reach** problems, not
  attribution or generation problems — Phase M3 established that no selector recovers them, and the
  reach table shows `s021`/`s030` are outside this configuration family at any k.
* Attribution is now *prompted*, not *enforced*. Nothing in the pipeline checks who a cited line is
  about, so a future prompt regression could reintroduce R10 silently. A detector analogous to
  `absence_claims.py` is the obvious guard and does not exist yet.

## Next step

The retrieval-reach residual (`s013 s021 s029 s030 s032`) is well understood and expensive to chase:
`probe_decomposition`, `probe_pool_rerank` and `probe_metadata_oracle` all measured no win. The
higher-value work now is on the **groundedness side**, where each phase has found something real:
R9 (absence over-claims) and R10 (cross-speaker attribution) were both invisible to every metric the
project had before they were instrumented. The natural next instrument is an **attribution check** —
verifying that each cited line's speaker actually matches who the answer says the fact is about —
plus wiring the existing groundedness signals into the product surface so a user can see them.

---

# Phase 13 — the attribution screen (R11 guard): 2/2 recall on the only known defects, at a measured precision that requires adjudication

## First, the retrieval lever was closed properly

Phase 9 measured `k` but always with `hybrid-pool 30`, where the fused list is **capped at 2×pool = 60
candidates** — so "k=50" was never really 50 candidates wide. Widening the pool settles it:

| hybrid-pool | 30 | 60 | 100 | 200 |
| ----------- | -- | -- | --- | --- |
| coverage @k=20 | 87.0 % | 85.5 % | 87.0 % | 87.7 % |

Flat and non-monotone at the adopted k=20: the ceiling is a genuine **ranking** limit, not a pool
artefact. (At k=50 the pool does matter — 93.1 % → 94.1 % → 96.3 % — but that is 50 chunks in context
to recover ~3 points and at most 2 of the 5 remaining failures.) Combined with Phase M3's rejected
selectors, the retrieval space is now fully characterised: **k (10/20/50) × pool (30/60/100/200) ×
reranker × decomposition × metadata filtering** all measured, one small win (k=20, already adopted)
and a large documented negative space.

## What was built

`attribution_screen.py` + 11 tests. The signal is mechanical and rests on a measurable property of
this corpus: the user is a participant literally named 我, so every line is one of

* speaker is 我 → the user's own statement (**safe**);
* speaker is someone else and the line addresses 你 → about the user (**safe**);
* speaker is someone else, the line says 我 and does not address 你 → **that person's own situation**;
* …and a subclass: **relayed** speech, where the 我 belongs to a third party the speaker is quoting
  ("我表哥说…", "我们组那个同学说…").

The middle class is **30.9 %** of third-party lines in corpus v2 — common enough to matter, separable
enough to detect. For each cited sentence the screen finds the cited chunk's best-matching line by
character-bigram overlap and flags the sentence when that line is someone else's own situation and the
sentence does not attribute it.

## It catches the only defects that exist — after two bugs

**Recall is 2/2**: on the one arm that provably contains fabrications (ablated A11), the screen flags
`s020` and `s032` and nothing else. Both bugs that hid this were found by measurement, not inspection:

1. **The citation was being orphaned.** Answers write `…。[来源 5]`, so splitting on `。` put the
   citation in a fragment of its own, which matched nothing — and `s020` was missed entirely. This is
   the same class of harness bug as Phase M3's `rrf_ranking` weights tuple: silent, and it would have
   produced a confident "the screen doesn't catch it" conclusion.
2. **Pronoun attribution was flagged.** Answers attribute as often with 他/她 ("他的系统还没部署") as
   with a name; `s031` was flagged in four different arms because of it. Fixed with a one-sentence
   context window.

## Flags across every arm — and a measured precision, not an assumed one

| arm | flagged queries |
| --- | --------------- |
| v1 A10 (k=10) | — |
| v2 A10 (k=10) | `s026` |
| v2 A11 (k=20) | `s026` |
| **v2 A12** (adopted) | — |
| **ablated A11** (the 2 known defects) | **`s020`, `s032`** |
| **ablated A12** (adopted) | — |

All 4 flags were adjudicated blind, and the result is deliberately unflattering: **1 TRUE_POSITIVE
(`s032` — the real defect) / 3 FALSE_POSITIVE → 25 % flag precision** (50 % at query level). The
adjudicator also identified *why* the false ones fire: their matched lines are **nested reported
speech** ("我表哥…", "我们组那个同学…"), which is neither the user's statement nor cleanly the
speaker's own situation.

Making that a first-class class (`relayed`) yields two honest operating points:

| | recall on known defects | precision (flag / query) | false alarms on the 4 clean arms |
| - | --- | --- | --- |
| **A — include relayed (default)** | **2/2** | 25 % / 50 % | 0 |
| B — exclude relayed | 1/2 (misses `s020`) | 100 % / 100 % | 0 |

The default is **A**: a guard that stays silent is worse than one that is noisy, because the whole
point is to hear about a regression. Note the reassuring property both share — **zero false alarms on
every arm that has no known defect**, including both adopted A12 arms.

## Decision: keep it as a **regression alarm**, explicitly not a verdict

It is not accurate enough to gate anything on a single flag, and it is not claimed to be. What it does
is turn R11 from "attribution is unenforced" into "attribution is *monitored*": if a future prompt or
model change reintroduces cross-speaker attribution, the arm that had 2 defects still lights up 2
queries while every clean arm stays dark.

## Honest caveats

* **The precision sample is tiny (4 flags).** 25 % is a measurement on this sample, not a stable
  estimate; the sample cannot shrink further because the screen only fires on 4 sentences across six
  36-query arms.
* The screen is **lexical**. It matches claims to lines by character-bigram overlap and never
  understands a sentence; `best_match` can therefore point at the wrong line, and its `s020` hit is at
  the right *query* via a sentence the adjudicator judged legitimate — i.e. correct at query level for
  partly the wrong reason.
* The `relayed` class is a regex over a handful of patterns, not an analysis of reported speech.
* Recall is only known against **two** defects. A screen validated on two examples is a screen, not a
  proof.

## Next step

The groundedness surface now has three instruments (`citation_metrics`, `absence_claims`,
`attribution_screen`) and two measured, fixed-or-monitored risks (R9, R10). None of them is visible to
a user: `recall.py` prints evidence cards but not whether an answer over-claims absence or borrows
someone else's situation. Wiring these signals into the product surface — so a recalled answer carries
its own groundedness caveats — is the natural next phase and is directly aligned with the project's
"no evidence, no memory claim" promise.

---

# Phase 14 — groundedness in the product surface: caveats a user can act on, with measured base rates

## The design constraint that shapes everything

Every instrument so far was built for the **evaluation**, where the gold evidence lines are known.
A user asking a real question has no gold. That rules out the strongest signals — the absence screen's
`risk` flag is defined as "absence claimed AND gold missed that still exists in the record", which is
unknowable in production.

So `groundedness.py` reports only what survives without gold:

| signal | needs gold? | what it can say |
| ------ | ----------- | --------------- |
| citation integrity | no | how many citations, whether any index is out of range |
| record-silence claims | no (to *surface*) | this answer asserts the record does NOT contain something |
| attribution flags | **no** | a claim rests on a line where somebody else describes their own situation |

It reports **caveats, not verdicts**, and a test enforces that: the warning text may not contain
"wrong", "incorrect", "false" or "fabricated", because without gold the tool cannot know.

## Wired into `recall.py` — and the product now ships A12

`recall.py` prints the panel after the evidence cards and includes it under `--json` as
`groundedness`. Verified end-to-end on a real question:

```
Groundedness: 1 citation(s) | no silence claims | no attribution flags
```

While wiring it I found the surface was still shipping **A11's** prompt: `--answer-prompt` defaulted
to `cited-narrow`. Corrected to `cited-attributed`, so `recall.py` now defaults to the adopted
reference on both axes (`k=20`, attribution clause) — the same kind of drift Phase 9 fixed for
`DEFAULT_K`.

## Does the panel actually fire where it should? Measured across all six arms

This is the acceptance test: the signal must light up on the arms with known defects and stay quiet on
the clean ones.

| arm | attribution flags | answers with a silence claim | uncited answers |
| --- | ----------------- | ---------------------------- | --------------- |
| v1 A10 | — | 11 | 0 |
| v2 A10 | `s026` | 12 | 0 |
| v2 A11 | `s026` | 11 | 0 |
| **v2 A12** (adopted) | **—** | 11 | 0 |
| **ablated A11** (the 2 known fabrications) | **`s020`, `s032`** ✓ | 28 | 4 |
| **ablated A12** (adopted) | **—** | 28 | 3 |

The attribution signal behaves as an **alarm**: zero flags on both adopted arms, and on the one arm
that provably contains fabrications it names exactly those two queries.

## The two signals have very different base rates, and that changes how they should be read

* **Attribution flags: low base rate (0–1 queries per arm).** A flag is worth stopping on.
* **Silence claims: high base rate — 11–12 of 36 answers (≈30 %) in normal operation, 28 of 36 under
  ablation.** An answer that says "the record never mentions X" is extremely common and usually
  correct. The panel therefore functions as a *reminder* ("check whether the record really is silent"),
  not an alarm — and the module docstring says so rather than implying a strong signal.

That asymmetry is the honest headline of this phase: **one of the three instruments transfers to the
product as an alarm, one transfers as a reminder, and the strongest of them (the absence `risk` flag)
does not transfer at all** because it is defined in terms of gold.

## Decision: ship the panel; keep the evaluation instruments as the gate

The caveats are advisory by construction and cannot be wrong in a way that costs the user anything —
the worst case is a reader checking a true statement. Nothing here gates an answer.

## Honest caveats

* The panel's absence caveat is **much weaker than the evaluation's**: the eval can say "absence claimed
  over incomplete retrieval" (measured at 40–60 % precision after adjudication); the product can only
  say "this answer claims silence". Presenting them as the same thing would be misleading.
* The attribution flag inherits every limitation recorded in Phase 13 — lexical matching, a 4-flag
  precision sample, recall known against only two defects.
* The smoke test was a single real question; the panel's rendering under a *flagged* answer was
  verified offline against the stored arms, not in a live flagged run.

## Next step

The evaluation and the product now agree on what "grounded" means, which makes the remaining
retrieval residual (`s013 s021 s029 s030 s032`) the only substantial open item — and it is a
*coverage* problem with a fully characterised solution space that has already returned one win (k=20)
and five measured negatives. Worth one more attempt only if a genuinely new mechanism appears; the
higher-value work is now making the engine usable end-to-end on a real corpus (`recall.py` runs on a
36-query synthetic stress corpus, not on the user's own chat export), which is what "core version
complete" should mean for this project.

---

# Phase 15 — real-export ingestion: the engine stops failing silently on a user's own chat log

## The failure, observed before anything was built

The adapter accepts exactly one line shape. Pointing the product at a real export in any other layout
indexed **zero messages and reported no error** — the engine would then answer "the record does not
show it" to every question, looking confident while being blind:

| export layout | messages | skipped | outcome |
| ------------- | -------- | ------- | ------- |
| canonical `[YYYY-MM-DD HH:MM] speaker: text` | 2 | 0 | OK |
| Telegram-ish `[DD.MM.YY HH:MM]` | 0 | 2 | **silently empty** |
| loose ISO with seconds | 0 | 2 | **silently empty** |
| WeChat-ish speaker-first header + body line | 0 | 4 | **silently empty** |
| date-on-its-own-line | 0 | 3 | **silently empty** |

## What was built

`chat_import.py` + 14 tests. It detects which of four well-specified layouts a file uses, converts it
to canonical form, and **verifies the result by re-parsing it through the engine's own adapter** —
nothing is reported as OK unless the round trip yields the same message count with zero skipped lines.
Detected layouts now cover all four realistic shapes:

| layout | shape |
| ------ | ----- |
| `canonical` | `[YYYY-MM-DD HH:MM] speaker: text` |
| `bracket_dmy` | `[DD.MM.YY HH:MM] speaker: text` |
| `iso_seconds` | `YYYY-MM-DD HH:MM:SS speaker: text` |
| `speaker_first` | `speaker<spaces>YYYY-MM-DD HH:MM`, body on the following line(s) |

An unrecognised file gets a **diagnosis**, not silence: it names what was tried, prints the per-layout
match counts, shows the first five lines that failed with line numbers, and — for the one layout that
is recognised but unsupported (date-on-its-own-line) — says so explicitly.

`recall.py` now refuses a corpus it cannot ingest, exiting non-zero with the conversion command to run
instead of answering questions about an empty index.

## Three bugs, all found by running the thing rather than reading it

1. **The header layout skipped every message body.** A header set the pending timestamp but never
   marked the message as *open*, so the "is a body expected?" test was always false and each body line
   was counted as skipped. Fixed by tracking openness explicitly; the `speaker_first` layout went from
   0 to 2 messages on the fixture.
2. **A UTF-8 BOM ate the first message.** Real exports carry one; stuck to line 1 it defeats the
   pattern for exactly one message. Found by the first end-to-end CLI smoke test, not by the unit
   tests, which had no BOM case.
3. **My own test's premise was wrong.** I wrote a test asserting that a colon inside a speaker name
   gets sanitised to `：`. It cannot: every profile's speaker group is `[^:]{1,24}`, so the name simply
   ends at the first colon. The sanitiser is defensive and unreachable through the current profiles, so
   the test now covers the real behaviour (a split, which still round-trips cleanly) and the sanitiser
   is unit-tested directly.

## A process failure worth recording

While patching two files I used a PowerShell `Get-Content -Raw | Set-Content` round trip, which
re-encoded them in the ANSI codepage and **destroyed both files' Chinese text irrecoverably** —
`chat_import.py` (untracked, rewritten from scratch) and `recall.py` (restored from git and re-edited).
This is the same family as the long-standing "PowerShell mangles inline Python" hazard already noted in
this log, now with a sharper rule: **file edits go through the file tools, never through PowerShell
text round-trips.** Both files were verified to decode as UTF-8 afterwards.

## Honest limits

* A **fixed list of four layouts**, not a format-guessing framework. Most real exports are not in any
  of them and will get the diagnostic rather than a conversion.
* **Text messages only.** Attachments, stickers, recalls and system notices are skipped and counted;
  nothing is inferred.
* **Multi-line bodies are joined with a single space**, because the canonical format is one message per
  line. Newlines inside a message are a formatting artefact of the export; if a user's history carries
  meaning in those breaks, it is lost.
* The importer was validated on **fixtures I wrote**, not on a real user export. Its formats are
  motivated by common exporters, but no real WeChat/Telegram file has been through it.
* Sorting is by timestamp, so messages that share a minute keep their file order but an export whose
  ordering is meaningful beyond timestamps may be reordered.

## Decision: the engine is now usable end-to-end on a real chat log

Question in, answer with traceable evidence cards and groundedness caveats out — on the user's own
export rather than only on the synthetic stress corpus. That is the last piece "core version complete"
was missing, and it is the shape the project was for.

## Next step

The remaining retrieval residual is characterised and expensive; the corpus needs no further work. The
open items that would matter to a user are the ones the log already names: the product still indexes
**only text**, retrieval is tuned and measured on a synthetic corpus rather than a real one, and any
real export outside the four supported layouts still needs a conversion rule. The natural next phase is
a **second source adapter** (a real, widely-used export layout) driven by an actual sample rather than
by my assumptions — which requires the user to supply one, and is therefore the right thing to ask for
at this gate instead of guessing.

---

# Phase 16 — the evaluation describes the product (measured, 36/36), and that is now enforced

## The gap this closes

Fifteen phases evaluated `run_baseline.py`. The thing a user runs is `recall.py` — a **different code
path**: its own brain construction, prompt registration and corpus handling. If the two diverge, then
A12's 31/36, the ablation results, and every groundedness metric describe something other than the
shipped product, and nothing in the repo would have noticed.

## The fix, in two parts

**One construction path.** The product's LLM/retrieval/brain construction lived inline inside
`main()`, where no test could drive it. It is now `recall.build_session()`, called by both the CLI and
the parity harness — so the check exercises the product's own code rather than a copy of it.

**An empirical parity check.** `verify_product_parity.py` drives `recall.build_session()` over the 36
stress queries on corpus v2 and compares the retrieved chunk indices, query by query, against the
recorded A12 arm. Retrieval runs under `--workflow no-rewrite`, so this is deterministic: **the
comparison is exact, not statistical.**

| result | count |
| ------ | ----- |
| retrieval **identical**, chunk-index-for-chunk-index | **36 / 36** |
| same set, different order | 0 |
| **DIVERGED** | **0** |

The evaluation transfers to the product. Everything measured across Phases 0.5–14 — the arm ladder,
the arm's 31/36, the 2→0 fabrication fix, the groundedness base rates — describes what a user actually
runs.

## Two informational findings

* **Byte-identical answers: 0/36.** The product does not reproduce the recorded answers textually,
  which is the project's long-standing measured result (Phase 6: temperature 0 does not make this
  pipeline reproducible, and only removing the LLM from retrieval does). It is the reason every
  quality claim in this log is attached to a *configuration* measured once, not to a promise that a
  re-run yields the same text. Retrieval, by contrast, is bit-reproducible — which is exactly why the
  parity check is stated in terms of retrieval.
* **Answers carrying a groundedness caveat: 9/36**, consistent with Phase 14's separately measured
  base rates (attribution flags 0–1 per arm, silence claims ~11 per arm).

## How it is enforced from now on

`tests/test_product_parity.py` (5 tests) pins the structure that makes the two paths comparable, so
the divergence cannot be re-introduced silently:

* `main()` must contain **no** `LLMEndpointConfig(` / `RetrievalConfig(` / `HybridConfig(` — all
  construction must go through `build_session`;
* `build_session` must still register the prompt, build both configs and ingest the corpus;
* the defaults must equal the adopted reference (k=20, hybrid pool 30, `no-rewrite`,
  `cited-attributed`, 900-char sessions), and the CLI must advertise that prompt as its default;
* the committed `product_parity.json` must show 0 divergences across 36 rows.

## Honest caveats

* Parity is measured on **corpus v2 with the stress queries** — the only corpus with a recorded arm to
  compare against. A real user export exercises the same retrieval path but a corpus the evaluation
  never saw; parity there is argued from shared code, not measured.
* Retrieval parity is necessary but not sufficient for behavioural parity: the same 20 chunks with the
  same prompt can still yield different text, which is the 0/36 above and is inherent to the model.
* The structural tests assert *where* code lives, not that it is correct. They would not catch a
  change that altered both paths identically — which is the point, since such a change is a
  configuration change that needs re-measuring, not a divergence.

## Decision: the core version is complete and self-consistent

A user can bring their own chat log (`chat_import.py`), ask a question (`recall.py`), and get an answer
with traceable evidence cards and groundedness caveats — from the same configuration the evaluation
measured, at 36/36 retrieval parity, with the promise that the numbers in this log describe the tool
they are holding.

## Next step

The one open item that cannot be advanced without the user is the **second source adapter**: the
importer's four layouts are motivated by common exporters but no real export has been through it, and
guessing further formats would be speculation dressed as work. This gate therefore asks for a real
sample (even the first 20 lines, or just the shape) rather than inventing one. Everything else the log
lists as ⏸ — persistence (Phase 8), a UI (Phase 9), multimodal (Phase 10) — is additive scope beyond
the recall core, not a gap in it.

---

# Phase 17 — an entry point a new user can actually follow, kept honest by tests

## Why this was the remaining gap

Seventeen phases produced 16 instruments, 162 tests and a 1,200-line decision log — and no document
that tells someone how to *use* the thing. `PROJECT_STATUS.md` is a measurement record, not a
quickstart: it explains what was rejected and why, which is the right content for a decision log and
the wrong content for a first run.

`README.md` now covers the three things a user needs: bring your own export, ask a question, read the
answer (including what the groundedness caveats mean and how much to trust each one). It also lists
the commands that regenerate the measurements, the layout of the directory, and the known limits.

## Documentation drifts, so the README is tested

A README that names a flag renamed two phases ago fails at the first thing a new user tries. So
`tests/test_readme.py` (15 tests) parses the fenced commands out of `README.md` and checks them against
the code:

* every documented script exists;
* every documented `--flag` is **declared** by that script's argparse;
* the quickstart still shows `--corpus` and `--json` for the product CLI;
* the README links the decision log, and still states the limitations.

The flag check reads `add_argument("--…")` statically rather than spawning `--help`. That was tried
first and cost a full RAG-stack import per script — nine subprocesses, over two minutes for one test
module. Static parsing is immediate and catches the drift that matters (a renamed flag no longer
matches); the authoritative end-to-end probe is kept for `recall.py`, the command a user runs first.

One of the tests exists purely to keep the document honest about its own subject: the README must still
state that five queries are unreachable, that ingestion is text-only, and that answers are not
textually reproducible. A README that only sells the happy path is not the document this project
keeps.

## Note on the process hazard from Phase 15

Phase 15 recorded that a PowerShell `Get-Content -Raw | Set-Content` round trip destroyed two files by
re-encoding them. This phase hit the same pattern again while removing two unused imports, and this
time the file survived — because `-Encoding UTF8` was passed explicitly. The failure needs the encoding
to be omitted; with it, the round trip is safe for UTF-8 content. The standing rule is unchanged (use
the file tools), and the file was verified to decode as UTF-8 with zero replacement characters
immediately afterwards rather than assumed to be fine.

## Decision

The core version is complete, self-consistent, and now documented for someone who is not me.
