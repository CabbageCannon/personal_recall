# Personal Recall Engine — Project Status

Living decision log. Facts only from real eval output; no estimated numbers.
Branch: `personal-recall` · Repo: `CabbageCannon/personal_recall` (renamed from
`CabbageCannon/quivr`, which still redirects) · Built on Quivr (`QuivrHQ/quivr`)

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
| 18A   | **WeFlow JSON source adapter (real WeChat, direct)**           | ✅ done — JSON → `MemoryEvent` with no TXT relay; text path proven **byte-identical (720/720)**; real-data leak found and closed |
| 18B   | WeChat 3.x multi-shard completeness (`MSG*.db`)                 | ✅ done (B1) — event-level merge before sessioning; dedupe on `serverId` only; **PARTIAL-history warning** vs the real `Msg/Multi`; single-file path still 720/720 |
| 18C   | Citation evidence consistency                                   | ✅ done — quote-anchored binding check (a lexical first version measured **17%% precision and was discarded**); **2 mismatches / 216 answers**, surfaced in the product panel |
| 18D   | Real-data acceptance harness                                    | ✅ done — `real_eval.py` + template; refuses placeholders, preserves labels; proven end-to-end on all 6 categories; **awaiting your questions** |
| 20.1  | Account-wide real acceptance eval parity                       | ✅ `--account` uses `build_account_session`; every question uses the shared `answer_question` path |
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

---

# Phase 18A — WeFlow JSON source adapter (real WeChat data, direct)

## The gap

A real WeChat export had been answered successfully, but only through
``WeFlow JSON → hand-written script → canonical TXT → MemoryEvent``. That works and proves the recall
path, but it makes a scratch script part of the architecture and discards every field the exporter
provides. This phase makes the JSON a first-class source.

## Files changed

| file | change |
| ---- | ------ |
| `memory/weflow.py` | **new** — the adapter: WeFlow schema → `MemoryEvent`. No retrieval/embedding/chunking/LLM logic. |
| `memory/processor.py` | `WeFlowSessionProcessor` (registered for `.json`) + the document builder extracted into one shared `_session_documents()` |
| `memory/__init__.py` | exports the adapter |
| `recall.py` | `register_source_processors()`, `--source-format`, a format-aware ingestion guard, and a narrow suppression of a misleading framework warning |
| `.gitignore` (this dir, **new**) + root `.gitignore` | real-data protection |
| `tests/test_weflow_adapter.py`, `tests/test_weflow_wiring.py`, `tests/test_text_path_unchanged.py` | **new** — 69 tests |

## Key design decisions

* **No Core changes.** `FileExtension` has no `json` member, but `get_file_extension()` falls back to
  the raw string `'.json'` and the processor registry accepts string keys — so the adapter registers
  under that string. `core/quivr_core/**` was not touched (it also holds your uncommitted annotations).
* **`isSend` decides the speaker, never the identity.** `1 → "我"`, `0 → "对方"`. The
  speaker-attribution safety logic depends on the literal `"我"`; substituting a wxid or display name
  would silently disable it. The real `senderUsername` is preserved in metadata.
* **Body priority** `parsedContent → content → rawContent`, first non-empty.
* **Internal payloads never reach retrieval.** Real exports carry multi-KB `<msg><emoji …/></msg>`
  blobs. Detection is **two-tier**, because `<msg>` is only the wrapper — the informative element is
  inside it; matching the wrapper first reported a stock emoji as generic non-text (a bug the tests
  caught). Payloads become a short placeholder (`[表情]`/`[图片]`/…) so the message keeps its timeline
  position without polluting the embedding corpus. A message with no readable text becomes
  `[非文本消息]` rather than vanishing.
* **Ids are deterministic and never `localId` alone**: `conversation_id` + (`serverId` when present) +
  `localId`, with a positional fallback. A collision is disambiguated and counted — **never resolved by
  dropping a message**.
* **Events are sorted chronologically before session building.** `build_sessions` consumes sequence
  order and never re-sorts, so an out-of-order export would fragment into one chunk per message (the
  Phase M2 failure). Ties keep exporter order, so the sort is stable.
* **Both payload shapes accepted**: a bare list, and a versioned envelope with `messages`. Unknown
  fields are preserved in metadata, not rejected.

## Verification

**End-to-end, through the product CLI**, on a synthetic WeFlow export (20 messages, 3 days):

```
corpus   : synthetic_weflow.json  (source=weflow)
ingested : 20 messages (types: {'image': 1, 'sticker': 1, 'text': 18}, suppressed payloads: 1)
Answer   : 我试 Neon 时发现：能连上，但免费层会休眠（…冷启动），所以我先不用它，回到了原来的方案 [来源 0]
Evidence : synthetic_weflow-session-0002 …
Groundedness: 1 citation(s) | no silence claims | no attribution flags
```

**Product = Eval path: UNCHANGED, and proven so.** `tests/test_text_path_unchanged.py` rebuilds the
corpus-v2 retrieval units through the real processor and compares every chunk against the contents the
recorded A12 arm stored: **720/720 sources reproduced exactly, 199/199 cited chunk ids intact**, and
`conversation_id` is still `"txt"`. Nothing in the measured path moved.

**Tests:** 231 pass (was 162; +69). JSON artifacts valid; frozen baselines byte-identical.

## A real privacy incident, found and fixed

The privacy check was not hypothetical. Two files from the real smoke test were sitting **untracked and
NOT ignored** in the working tree — `data/wechat_smoke.txt` (real chat lines) and `recall_debug.json`
(45 KB: the real question plus retrieved sources). A single `git add -A` would have committed real
WeChat content. Neither was committed here.

Fixes: this directory now has a `.gitignore` covering real WeFlow JSON, converted real corpora,
`MSG*.db` and its `-wal`/`-shm` sides, key material, and the actual leaked filenames; the root
`.gitignore` covers the `.tmp_*/` scratch trees that `git add -A` would also have swept up. All of it is
**tested** with `git check-ignore` — 14 sensitive paths must be ignored and 6 synthetic corpora must
stay trackable, so a `.gitignore` that matches nothing fails the suite.

Also restored: `stress_validation.json` had been overwritten with a *v2* validation, destroying the
frozen Phase 0.5 v1 record; it was reverted to HEAD rather than committed.

## Scope discipline

Retrieval, chunking, prompt, k, hybrid pool and workflow defaults were **not** touched. No new
retrieval experiments. The only product-path change is that a second *source* is now readable.

## Next phase (not started)

**18B — WeChat 3.x multi-shard completeness.** Reading only `MSG0.db` when `MSG1`/`MSG2` exist silently
loses the newest history, which is the `No Data Loaded ≠ No Memory Exists` failure this project refuses
to ship. That requires reading the exporter's 3.x DB access code and deciding between patching
`weflow-cli` and building a DevPilot-side merge layer — a decision I will present before writing code,
as requested.

Stopping here for review, as instructed.

---

# Phase 18B — investigation and the patch-vs-layer decision (no code written yet)

## What the exporter actually does

Read from the installed package (`weflow-cli` 1.6.0, `%APPDATA%\npm\node_modules\weflow-cli`).

**The single-file assumption is structural, not a per-query bug.** `sqlcipherCore.open()` is the whole
3.x access layer:

* signature is `open(dbPath: string, keyHex: string, wxid: string)` — **one** path (`sqlcipherCore.ts:336`);
* it decrypts that one file to a temp path and opens **one** `DatabaseSync` (`:358-360`);
* every query method (`getSessions`, `getMessages`, search, export) runs against `this.db`, that single
  connection — there is no shard concept anywhere in the file (780 lines).

Its own comment states the layout it assumes (`sqlcipherCore.ts:349`):

```
// dbPath: {wxDirRoot}/{wxid}/Msg/Multi/MSG0.db
```

and the 3.x connect path takes exactly one configured path (`chatService.ts:157`):
`configService.get('dbPath3x')`.

**So the reported symptom follows directly**: whichever single file is configured *is* the entire
visible history. Configure `MSG0.db` and everything in `MSG1`/`MSG2` is invisible. The manual workaround
— repointing `dbPath3x` at `MSG2.db` — is the only lever the current design offers.

**The session-level defect is in the same place.** `getSessions()` (`:371-405`) computes

```sql
SELECT m.StrTalker, MAX(m.CreateTime) ... FROM MSG m GROUP BY m.StrTalker
```

against one shard, so a contact's "last message" is that shard's maximum. This is exactly the
`global MAX(CreateTime)` requirement in the brief: aggregating across shards is not something a
per-shard query can be patched into locally, it has to happen above the access layer.

## Two facts that decide the options

1. **No command can be pointed at a database.** Every command's options were enumerated; `sessions` takes
   only `-k/-n/--json`, `chat` only `--top-k/--talker/--api-key/--dry-run/--yes/--json`. Paths and keys
   come from persisted config, never from the command line.
2. **The config location cannot be redirected by an environment variable.** It is hardcoded:
   `CONFIG_DIR = join(homedir(), '.weflow-cli')`, `CONFIG_FILE = config.json`
   (`configService.ts:89-90`). So per-shard runs would require either mutating the user's global config
   between invocations, or redirecting `HOME`/`USERPROFILE` for the child process.

## Option A — fork / patch `weflow-cli`

Turn the 3.x layer from one connection into N: `openAll(dbPaths[])`, then fan out **every** query and
merge above it (sort, stable tie-break, dedupe, global `MAX(CreateTime)`).

* **Good**: correct at the source, and every consumer benefits at once (sessions, chat, MCP, export).
* **Bad**: the fan-out has to be threaded through every query method, not just `getSessions`; the
  package ships compiled `dist/` so a patch lives in a global npm directory that a reinstall or upgrade
  erases; and DevPilot would end up maintaining a fork of a fast-moving third-party codebase. That is a
  lot of surface for a project whose subject is recall quality, not exporter upkeep.

## Option B — DevPilot-side shard merge layer (recommended)

Keep the boundary the brief asks for — `WeFlow → stable JSON`, `DevPilot → MemoryEvent → Recall` — and
put the merge where the adapter and the eval already live.

Two layers, deliberately separable:

* **B1 (the part that matters, testable offline, needs no database):** ingest **N** WeFlow JSON files as
  one corpus. Merge by `createTime`, stable tie-break, dedupe; compute session `MAX(CreateTime)`
  globally; and report which shard each message came from plus the covered time range. If a shard is
  missing or unreadable, **fail loudly** rather than returning a smaller history — `No Data Loaded ≠ No
  Memory Exists` is the whole point of the phase.
* **B2 (automation on top, optional):** discover `Msg/Multi/MSG*.db`, export each shard by redirecting
  `USERPROFILE` to a scratch home containing a copy of the config, then hand the JSON files to B1. No
  global config mutation, and the decrypted temp filenames are already distinct per shard
  (`basename(dbPath) + '.decrypted.db'`), so shards cannot collide.

**Why B**: the merge logic is the part that can be wrong in interesting ways, and B puts it in the repo
that owns the adapter, the tests and the eval — fully testable with synthetic per-shard JSON today,
without the WeChat database. B1 also works if the user exports shards by hand, which is what they
already did.

**Honest costs of B**: B2 depends on a `USERPROFILE` redirect the exporter does not document, so it
could break on a version bump; it means copying the user's config — including key material — into a
scratch directory (DevPilot never *derives* a key, but the key would pass through a file DevPilot
writes); it is N sequential invocations, so slower; and per-shard ordering is only as good as the
`createTime` values, which is why the tie-break is explicit.

## Incidental privacy finding (worth knowing regardless of the decision)

While creating the connection, the exporter writes a **decrypted plaintext copy** of the real message
database to the system temp directory — `join(os.tmpdir(), 'weflow_3x_decrypt')` +
`basename(dbPath) + '.decrypted.db'` (`sqlcipherCore.ts:85,91`) — and deletes it on close (`:773`). A
crash or a kill leaves plaintext chat history in `%TEMP%\weflow_3x_decrypt\`. That directory is outside
this repo, so it is not a commit risk, but it is real data at rest outside the WeChat data directory
and the user should know it exists.

## Decision required

A or B — and if B, whether to build only **B1** now (offline, testable, works with hand-exported shards)
and treat **B2** as a later convenience. My recommendation is **B, starting with B1**.

---

# Phase 18B — multi-shard merge (option B1, as chosen)

**Decision taken by the user: B, B1 only.** No exporter patch, no auto-discovery/auto-export (B2). The
boundary stays `WeFlow → stable JSON` / `DevPilot → MemoryEvent → Recall`.

## Files changed

| file | change |
| ---- | ------ |
| `memory/shards.py` | **new** — discover shard exports, merge events, dedupe, per-shard provenance and union coverage |
| `memory/__init__.py` | exports the shard API |
| `memory/processor.py` | `_session_documents` → public `session_documents` (same definition, reused by the merged path) |
| `recall.py` | `--corpus` accepts a **directory** of shard exports; `--shard-dir` for the completeness check; `build_brain_from_events` |
| `README.md` | multi-shard usage |
| `tests/test_shards.py` | **new** — 21 tests |

## Key design decisions

* **The merge happens at the `MemoryEvent` level, before session building.** Built via
  `Brain.afrom_langchain_documents` over the merged sessions. Merging *output* instead would cut one
  conversation into a chunk per shard and make a contact's "last message" the maximum of whichever
  shard happened to be read.
* **Chronology comes from `createTime`, never from the file name or its mtime.** The three shards on
  this machine happen to be ordered by update time — a coincidence of this profile, not a property to
  rely on. Ties break by `(shard label, position in shard)`, so the merged order is deterministic.
* **Deduplication only on a strong identity — the exporter's `serverId`.** Without it there is no safe
  cross-shard identity: WeFlow documents that `localId` repeats across conversations *and shards*, so
  deduplicating on it would drop real messages. Those events are kept and counted as `undedupeable`,
  and the report says so. `test_messages_without_server_id_are_never_deduplicated` pins this.
* **A missing or malformed shard fails loudly.** `load_and_merge` raises rather than skipping;
  a silently dropped shard is a silently smaller history.
* **The completeness check.** `--shard-dir` (read-only, names only) reports the `MSG*.db` files that
  exist versus how many were exported, and prints a `PARTIAL` warning when they differ. This is the
  phase's whole point: **No Data Loaded ≠ No Memory Exists.**

## Two framework requirements found the hard way

1. **`asyncio.run()` breaks the later sync call.** It closes the loop it creates, so the subsequent
   `brain.ask()` raises `no current event loop`. `Brain.from_files` uses `get_event_loop()` +
   `run_until_complete()` and never closes; the merged path now mirrors that exactly.
2. **Building from documents must supply `original_file_name`.** Doing so bypasses
   `ProcessorBase.process_file`, which is what normally adds it. Without it the framework refuses to
   render the document prompt (`combine_documents` injects `index` itself, but not this field). The
   `"Filename: … Content: …"` prefix is deliberately *not* applied — `process_file` only adds it when
   the inner metadata already carries the field, which is why the processor path's chunk text has no
   prefix.

## Verification

**End-to-end through the product CLI**, on two synthetic shard exports (named `MSG0.json`/`MSG2.json`,
deliberately sharing one `serverId`), with `--shard-dir` pointed at the real
`...\Msg\Multi` on this machine (names read only — no database opened, no key material touched):

```
corpus   : synthetic_shards  (source=weflow, multi-shard)
  shards merged : 2
  messages      : 9 kept of 10 (1 duplicate(s) removed)
  coverage      : 2026-09-14 09:30 .. 2026-09-17 13:02
    MSG0   4 msgs  2026-09-14 09:30 .. 2026-09-15 21:15
    MSG2   5 msgs  2026-09-15 21:20 .. 2026-09-17 13:02  (1 duplicate)
  shard check   : 3 MSG*.db present ['MSG0.db', 'MSG1.db', 'MSG2.db']; 2 exported
  WARNING       : only 2 of 3 message shards were exported, so this history is PARTIAL - ...
Answer   : 你今天中午吃的是**拌粉**（外卖）[来源 0]
Evidence : weflow-session-0003 · 2026-09-17 11:26 - 13:02 · 4 messages
Groundedness: 2 citation(s) | no silence claims | no attribution flags
```

The answer and its evidence come from a message that lives in the **second** shard, which the previous
single-shard design could not see.

**Product = Eval path: unchanged.** The single-file path is untouched — `tests/test_text_path_unchanged.py`
still reproduces **720/720** recorded A12 sources exactly, and `tests/test_product_parity.py` still
passes. The multi-shard path is additive: it is taken only when `--corpus` is a directory.

**Tests:** 252 pass (was 231; +21). All Python files UTF-8.

## What the real shard directory showed

The real `Msg/Multi` contains **three families** of sharded databases, not one:

| family | files |
| ------ | ----- |
| messages | `MSG0.db` (188 MB), `MSG1.db` (251 MB), `MSG2.db` (220 MB) |
| full-text index | `FTSMSG0.db`, `FTSMSG1.db`, `FTSMSG2.db` |
| media | `MediaMSG0.db`, `MediaMSG1.db`, `MediaMSG2.db` |

`MSG1.db` is the **largest** message shard, so exporting only `MSG0` (the exporter's default) discards
the biggest single piece of history. File mtimes are `MSG0` 2026-06-26, `MSG1` 2026-09-06, `MSG2`
2026-09-17 — ordered here, but treated as a coincidence and never used as logic.

**A new completeness risk, flagged not fixed:** `MSG2.db-wal` is **8.2 MB** (`MediaMSG2.db-wal`
1.8 MB, `FTSMSG2.db-wal` 4.2 MB). The exporter decrypts `MSG2.db` alone
(`sqlcipherCore.decryptToTemp`), so recent writes still sitting in the write-ahead log may not be
visible to an export even from the newest shard. This is worth verifying against a known recent
message before trusting recency, and it is the kind of gap this phase exists to surface.

## Next phase (not started)

**18C — citation evidence consistency.** Entirely offline and independent of the shard work: a
mechanical check that a cited sentence is actually supported by the chunk it cites, reported as a
groundedness warning (no gold, no LLM judge), with the real `拌粉` mis-binding case as a synthetic
regression fixture. Stopping here for review as instructed.

---

# Phase 18C — citation evidence consistency

## The failure this targets

Reported from the real WeChat smoke test, and invisible to every existing metric:

```
Answer sentence : 11:26 你说“我又点了拌粉” [来源 1]
Source 0        : 07:27 – 11:53   (contains: 11:26 我: 我又点了拌粉)
Source 1        : 11:53 – 12:35   (cannot contain 11:26)
```

The index was valid, the citation was well-formed, the answer was correct — the **binding** was wrong.
Re-running the model produced `[来源 0]`, confirming a binding error rather than a retrieval error.

## Files changed

| file | change |
| ---- | ------ |
| `citation_consistency.py` | **new** — the checker |
| `groundedness.py` | surfaces binding mismatches in the product panel |
| `recall.py` | `render_groundedness` prints the offending quote and sentence |
| `README.md` | caveat table with measured base rates |
| `tests/test_citation_consistency.py`, `tests/test_groundedness.py` | **new** / extended — 23 tests |

## A first version that measured badly, and what replaced it

The first implementation scored **every** cited sentence against every source by character-bigram
containment. Measured across six committed arms (216 answers, ~700 sentences): **6 mismatch findings,
about 1 of which held up — ~17 % precision.** Two causes were systematic:

* **multi-fact summary sentences** legitimately cite several chunks, so "the best matching single
  source" is a meaningless comparison for them;
* bigram overlap produced **spurious winners** — a claim about 搬家 matched the line 「搬什么家」.

That version was discarded rather than shipped. The replacement is anchored on something much harder
to fake: **a verbatim quote**. When an answer puts text in quotation marks it is making a claim about
what the record says, and that text must appear in the chunk it cites.

Three further calibration decisions, each measured:

* **Record-silence claims are excluded.** The evidence for "the record does not mention X" is an
  *absence*, so no quoted text can support it and lexical scoring guarantees a false positive. That
  class belongs to `absence_claims.py`. Removing it cut the noise substantially.
* **`MIN_QUOTE_CHARS = 6`.** At 4 characters, a claim quoting the generic phrase 「另一个同学」 was
  flagged against a source that merely contained that phrase. At 6, both genuine mismatches (14 and 9
  characters) and the reported case (6) survive. The cost is recorded as a test: the genuine but short
  quote 「去拿外卖」 is now deliberately ignored.
* **`unverified` quotes are not surfaced.** They outnumber mismatches 8:1 (16 vs 2) and inspection
  showed they are dominated by the model putting its **own paraphrase** in quotes — 「两个方向」,
  「小王推荐」, 「开始使用/试跑」. They stay in the JSON for analysis; only mismatches become warnings.

## Result after calibration

| | fixture (wrong cite) | fixture (right cite) |
| - | -------------------- | -------------------- |
| findings | **1 mismatch**, naming Source 0 | **0** |

Across the six committed arms: **2 mismatches in 216 answers** (both inspected and judged genuine —
`v2 A11 s034` and `v2 A12 s027`), 16 unverified, and at most one flagged query per arm. That is an
alarm-grade base rate, not a reminder.

What a user now sees (rendered offline from the fixture, no API call):

```
Groundedness: 1 citation(s) | 1 binding mismatch(es) | no silence claims | no attribution flags

  ! 1 citation binding mismatch(es): quoted text is not in the source it cites, but is in another retrieved source

  citation: quoted text «我又点了拌粉» is not in the cited source (Source [1]) but appears in Source [0]
    in sentence: 11:26 你说“我又点了拌粉” [来源 1]
```

## Honest limits

* **Recall on the one independently labelled defect is zero, and by design.** `stress_v2_a10 s025` was
  graded as citing a 同学A chunk for a claim about 张三's records — that is an *attribution* error on a
  *silence* claim, which this module now skips because it is a silence claim. It is `attribution_screen`'s
  territory. This module is not a general citation auditor and is not claimed to be one.
* **Quotes only.** A sentence with no quoted span is not checked at all, so a paraphrase cannot be
  caught. That is the price of the precision.
* **`unverified` is unresolved, not validated.** It was judged noisy by inspection of 16 findings, not
  adjudicated blind.
* The mismatch sample is **2 findings**; precision is judged by reading them, not measured.

## Scope discipline

Retrieval, chunking, prompt, k, hybrid pool and workflow defaults were **not** touched, and no new LLM
judge was introduced. The check is mechanical and deterministic.

**Tests:** 275 pass (was 252; +23). All Python files UTF-8.

## Next phase (not started)

**18D — real-WeChat acceptance eval harness**, 15–20 questions across the six categories, recording
answers, retrieved sources, groundedness and citation-consistency warnings for manual labelling.
Stopping here for review as instructed.

---

# Phase 18D — real-data acceptance harness

## Files changed

| file | change |
| ---- | ------ |
| `real_eval.py` | **new** — the harness |
| `eval_questions.template.json` | **new** — 18 slots, three per category, with per-category guidance |
| `README.md` | acceptance-check usage (drift-tested) |
| `tests/test_real_eval.py` | **new** — 19 tests |

## Design

* **Small by construction**, as the brief asks: 15–20 questions over the six categories
  (`single_fact_recall`, `timeline_reasoning`, `speaker_attribution`, `multi_source_synthesis`,
  `abstention`, `older_memory`). No judge system, no scoring model.
* **Records everything a human needs to judge an answer**, per question: the answer, every retrieved
  chunk (time range, participants, text, chunk id), the citation indices, the groundedness object
  including **citation binding mismatches** and attribution flags, and the latency.
* **Four labels start `null`** for a human to fill in place: `answer_correct`, `retrieval_correct`,
  `citation_binding_correct`, `attribution_correct`. Re-running **preserves existing labels** —
  labelling is work and another run must never discard it.
* **Reuses the product path**: `recall.build_session`, so the harness cannot drift from the shipped
  configuration. Retrieval defaults were not touched.
* **Fails before spending anything.** Question-set validation rejects a mistyped category, a duplicate
  id, a too-short question, and **any entry still holding the template placeholder** — so a half-filled
  question set cannot quietly burn API calls and produce a results file covering five categories.
* **Resumable**: results are saved after every question (the lesson from Phase 11's transient
  `RemoteProtocolError`), and `--report-only` summarises an existing run without loading the model.

## Verification

**End-to-end on a six-question demo set** (one per category) against the synthetic shard corpus:

```
[05/6] q05 abstention             3 sources    1.7s
      Q: 我常去的那家店叫什么名字？
      A: 提供的记录里没有提到你常去的那家店的名字。

category                     n  labelled  caveats   mean ms
single_fact_recall           1         0        0    2330.7
timeline_reasoning           1         0        1    6289.1
speaker_attribution          1         0        0    1986.0
multi_source_synthesis       1         0        0    1849.3
abstention                   1         0        1    1658.3
older_memory                 1         0        0    1626.9
```

All six categories produced a record with the full field set, the abstention question was correctly
declined, and the report renders per-category counts, caveat counts, mean latency and the four
labelling fields. `--report-only` reproduces the table without loading the model stack.

**One bug found by running it**: the harness called `build_session` without loading `.env`, which
`recall.py` and `verify_product_parity.py` each do for themselves — the first run died with
`Missing credentials`. Fixed in the harness (the product path was not changed).

**Tests:** 296 pass (was 276; +20). All Python files UTF-8.

## What is still needed from the user

The harness is complete and proven; the **questions are yours**. `eval_questions.template.json` has
18 slots with guidance per category, and the runner refuses to run until they are filled. Questions
should be ones you already know the answer to — that is what makes the manual labels meaningful.

## Phase 18A–18D: status

| phase | state |
| ----- | ----- |
| 18A WeFlow JSON source adapter | ✅ JSON → `MemoryEvent`, no TXT relay; text path proven byte-identical (720/720); real-data `.gitignore` gap found and closed |
| 18B multi-shard completeness | ✅ option B1: event-level merge, `serverId`-only dedupe, PARTIAL-history warning against the real `Msg/Multi` |
| 18C citation evidence consistency | ✅ quote-anchored binding check; a lexical first version measured 17 % precision and was discarded; 2 mismatches / 216 answers |
| 18D real-data acceptance harness | ✅ complete and proven end-to-end; **awaiting your questions** |

The adopted product path (retrieval, chunking, prompt, k, hybrid pool, workflow) was **not changed** in
any of the four phases.

---

# Phase 19 — account-wide multi-conversation ingestion

## The problem, stated precisely

Phase 18B made *one* conversation survive being split across `MSG0/MSG1/MSG2`. Phase 19 is the step
that turns that into an account: hundreds of conversations, each of them possibly split across those
same shards, all of them reaching one recall index.

The naive extension is the trap. Exporting everything into one directory and running the Phase 18B
merge over it produces an event stream where Alice at 10:00, Bob at 10:01 and a group at 10:02 are
adjacent — and a session builder whose boundary rule is temporal adjacency will fuse them into one
retrieval unit. The messages are unrelated; only their timestamps are close.

## The ordering rule

```
(shard, conversation) exports
          ↓
group BY CONVERSATION        <- always first
          ↓
merge shards WITHIN a conversation
          ↓
sessions built per conversation
          ↓
one global index
```

`memory/account.py` enforces this by **returning `dict[conversation_id, list[MemoryEvent]]`**, never a
flat stream. A caller that wants to build sessions must go through `build_account_sessions`, which
loops conversations and calls `build_sessions` once per conversation. There is deliberately no API
that hands out a merged account-wide event list, because that is the object from which the bug is
built.

## Files changed

| file | change |
| ---- | ------ |
| `memory/conversations.py` | **new** — conversation identity: talker ids, descriptors, listing parsing, shard-name normalisation |
| `memory/account.py` | **new** — grouping, per-conversation merge, import report, boundary checker |
| `exporter.py` | **new** — the only code that touches the real account: shard switching, listing, per-conversation export, the export tree + manifest |
| `export_account.py` | **new** — CLI over `exporter.py`, with `--dry-run` |
| `smoke_account.py` | **new** — anonymised real-account smoke |
| `recall.py` | `--account` mode, `import_account_directory`, `build_account_brain`, `build_account_session`, and the shared `answer_question` recall path |
| `memory/sessions.py` | conversation boundary in `build_sessions`; `conversation_id` on `MemoryChunk` |
| `memory/__init__.py` | package exports |
| `tests/test_account_isolation.py`, `test_account_ingestion.py`, `test_account_tree.py`, `test_exporter.py` | **new** — 120+ tests |

## Key decisions

* **Identity is the talker (`StrTalker` / `sessions --json` `username`), never a display name.**
  Nicknames, remarks and group names are all editable, and two contacts can share one. They are
  carried as `aliases` for the UI and are not usable as a key. A WeFlow export contains **no
  conversation field at all**, so identity comes from outside the file: the export invocation and the
  file name `{talker}_messages.json`. The orchestrator writes each shard's listing to `sessions.json`
  so a renamed file does not silently lose its conversation.

* **Dedup stays conservative**: `serverId` only, applied *within* a conversation. `localId` repeats
  across conversations and shards, so deduplicating on it drops real messages. Messages with no
  `serverId` are kept and counted (`undedupeable`) rather than guessed at.

* **The conversation boundary is checked, not promised.** `crossed_conversation_chunks()` recomputes
  chunk membership from event ids and `recall.import_account_directory` raises on a violation. The
  phase's hard rule is a runtime assertion on real data, not only a test fixture.

* **Shard switching without touching the user's config.** `weflow-cli` opens exactly one database per
  run, so an account export must point it at each shard in turn. `exporter.py` reads the real
  `~/.weflow-cli/config.json` **once**, writes a patched copy into a temp scratch profile, and deletes
  it on every exit path including a crash. The real config is never written; a test hashes it before
  and after to prove it. **The scratch profile contains a copy of the decrypt key**, which is why its
  cleanup is asserted directly and why the mechanism is documented as depending on the current
  `weflow-cli` config layout.

* **The manifest holds counts, never people.** `shard_manifest.json` records detected vs exported
  shards so a shard that exists but was never exported is detectable. It is the file most likely to be
  copied around, so a test asserts it contains no talker, no display name and no message text.

## Completeness: two independent ways to be PARTIAL

`AccountImportReport.partial` is true when **either**

1. a message shard exists on disk but produced no export (`missing_shards`), or
2. the export was narrowed with a conversation filter (`filtered_conversations > 0`).

The second case was **found by running the real smoke**, not by a test. `export_account.py --only`
writes each shard's *full* listing but exports only the named conversations; when the selected
conversations happen to span every shard, every shard is "exported", `missing_shards` is empty, and the
report called 8 conversations out of 272 a complete account. The near-miss is what hid it: when a
filtered conversation is absent from some shard, that shard produces no export and *does* land in
`missing_shards`. The bug appears exactly when the report looks most trustworthy.

The two causes are reported as two separate warnings with different next actions — "a shard failed,
re-run the export" versus "nothing is broken, there is simply more account than you asked for".

## Verification

**Synthetic account, 3 shards / 4 conversations** (A spans 3 shards, B spans 2, C is shard-local,
group D spans shard 0 and 2), with duplicate `serverId`s, reused `localId`s, lookalike contacts,
display-name changes and time-interleaved messages: conversation isolation, shard merge, ordering,
dedupe, stable ids, coverage, PARTIAL warnings, indexing and recall all pass.

The requirement's own scenario is a named test —
`test_brief_scenario_two_lookalike_conversations_stay_separate`:

```
10:00 A: 吃饭吗   10:01 B: 吃饭吗   10:02 A: 可以   10:03 B: 不去了
```

One minute apart, two messages byte-identical, all four on one calendar day. The requirement is two
two-message chunks; a global merge plus temporal segmentation gives one four-message chunk. A companion
test re-runs it with `max_gap` and `max_chars` effectively infinite, so the *only* thing separating the
conversations is the boundary itself. A third test hands a deliberately mixed stream straight to
`build_sessions` to prove the second line of defence, and a fourth asserts the library's own boundary
checker actually fails when a violation is injected — otherwise every "no violation" assertion above
would be vacuous.

**Real account (read-only, anonymised).** `smoke_account.py` on this machine's `Msg/Multi`:

```
message shards detected : 3 ['MSG0', 'MSG1', 'MSG2']      (FTSMSG*/MediaMSG* excluded)
conversations discovered: 8 selected of a bounded sample   (7 group, 1 direct)
message shards exported : 3 of 3 detected
messages                : 21154 kept of 21154 (0 duplicates removed; 12 without serverId)
coverage                : 2025-05-21 21:10 .. 2026-09-17 19:05
sessions                : 1222 chunk(s)
conversations indexed   : 8
boundary check          : OK - no chunk mixes two conversations
merged conversations    : 6 of 8 span more than one shard (2-3 shards each)
```

Real shards do overlap heavily — 449 listings union to 272 distinct conversations — so cross-shard
merging is the normal case, not an edge case. The smoke prints counts, shard names, date ranges and
10-character digests only: no wxid, no display name, no message text. Its export lands in
`data/real/`, which `.gitignore` excludes.

**Tests:** see the closing status table for the current count. The adopted product path — retrieval,
chunking, prompt, `k`, hybrid pool, workflow — was **not changed** in this phase, and
`test_product_parity.py` now enforces structurally that the adopted config is constructed in exactly
one place, because Phase 19 added a second *entry* point that could have grown its own copy.

## Known limits

* **WAL.** `MSG*.db-wal` can hold the newest messages and `weflow-cli` reads the database, not the log.
  An export is committed history, not a guarantee of the newest messages. `export_account.py` prints
  this caveat on every run; it is not fixed, only stated.
* **The filter count lives only in `shard_manifest.json`.** A narrowed tree whose manifest was deleted,
  or written by an older exporter, still looks complete when every shard directory is present — the
  per-shard listings are deliberately full, so nothing else on disk reveals the narrowing.
* **Identity depends on the export file name.** `{talker}_messages.json` is the only place a
  conversation id exists in the tree. `exports_from_directory` accepts a `conversation_ids` map for
  callers that have a recorded mapping, but the orchestrator does not currently write one.
* **`exporter.AccountExportReport.partial` tracks missing shards only.** The export-side report says
  "3 exported of 3 detected" plus a note for a narrowed export; the *import* side is the one that must
  not overclaim, and it is the one that reports PARTIAL.

## Phase 19: status

| requirement | state |
| ----------- | ----- |
| discover every conversation of the account | ✅ `list_conversations` per shard, unioned (`sessions --json`) |
| stable conversation identity | ✅ talker id; display name is an alias, never a key |
| merge a conversation across shards | ✅ event-level merge, `serverId` dedupe, `createTime` ordering |
| conversations never share a session | ✅ runtime checker + a test whose injected violation must trip it |
| one index for every conversation | ✅ `build_account_brain` over per-conversation chunks |
| deterministic / testable | ✅ two runs produce identical chunk ids and content |
| synthetic account end-to-end | ✅ 3 shards / 4 conversations, all shapes |
| real smoke | ✅ 3 real shards, 8 conversations, 21154 messages, boundary OK |
| partial-history completeness report | ✅ two independent causes, reported separately |
| real-data protection | ✅ `data/real/` ignored; smoke prints digests only; manifest carries no identity |
| old single-conversation path | ✅ unchanged; text path proven byte-identical |
| product parity regression | ✅ `test_product_parity.py` extended, passing |

---

# Phase 20 — minimal local web interface

## What was built

One page, one question box, and the three things worth showing: the answer, the evidence cards behind
it, and the caveats a reader should act on. Nothing else.

```
Personal Recall
[ 问问过去发生过什么...                    ] [→]

Answer
------------------------------------------------
你当时最后换成了 Supabase。[来源 1]

Evidence
+----------------------------------------------+
| 小王 · 来源 1 · 2025-05-12                    |
| 我、对方                                       |
| ...原始聊天...                                 |
+----------------------------------------------+

[!] citation / attribution warning   (only when there is one)
记录可能不完整，请结合原始聊天确认。   (only when there is one)
```

| file | role |
| ---- | ---- |
| `web.py` | entrypoint: `python web.py` → `http://127.0.0.1:8000` |
| `webapp/app.py` | `RecallState` (the index, built once), `create_app`, two endpoints |
| `webapp/static/` | one HTML page, one stylesheet, one script — 490 lines total |
| `tests/test_web.py` | 36 tests |

## The rule this phase had to obey: one recall path

The phase brief forbids a second retrieval implementation, and the temptation is real — a web backend
is exactly where a `serialize_sources` + `build_evidence_cards` + `assess` sequence gets written a
second time because the CLI's version prints instead of returning.

The fix is structural rather than disciplinary. `recall.answer_question(brain, retrieval_config, *,
question, show_uncited=False) -> dict` now owns the whole assembly —

```
brain.ask → serialize_sources → build_evidence_cards → assess
```

— and `ask_and_render` (CLI) and `RecallState.answer()` (`webapp/app.py`) both call it. The CLI
function is now only a printer. `webapp/app.py` contains no `brain.ask`, no `serialize_sources`, no
`build_evidence_cards` and no `assess`, and a test pins that. `test_product_parity.py` separately
enforces that the adopted retrieval config is still constructed in exactly one place, which matters
because Phase 19 added a second *entry* point (an account of many conversations) that could have grown
its own copy.

The extraction is behaviour-preserving: the pre-refactor `ask_and_render` was reimplemented verbatim
and stdout diffed over 16 invocations (clean / citation-mismatch + attribution / uncited / empty answer
× `--json` / human × `--show-uncited` on and off) — **0 byte differences**.

## API

Two endpoints, plus the page and its static assets. `/docs`, `/redoc` and `/openapi.json` are
disabled, so the process holding private chat serves exactly four routes.

```
POST /api/recall   {"question": "..."}
                -> {"answer", "evidence", "groundedness", "latency_ms"}
GET  /api/status -> {"ready", "conversation_count", "message_count", "coverage"}
```

* An empty or whitespace-only question is refused twice: the page will not submit one, and the API
  returns 400 rather than spending an embedding on it.
* A failed index build is **not a crash**. The server still starts, `/api/status` reports
  `ready: false` with a short reason, and questions get 503. A page that explains what is wrong beats
  a process that refuses to exist.
* The index is built **once**, at startup. Indexing parses every export and embeds every session —
  minutes, not milliseconds — so per-question work would make the page unusable. There is no index
  persistence yet; that is a deliberate deferral, not an oversight.

## What the page shows, and what it refuses to

Evidence cards carry the conversation display name, the time range, the participants and the original
chat lines: the four things §29 asks for, and nothing more. **Embedding scores, RRF scores, chunk ids
and internal UUIDs are not rendered** — the API carries some of them for `--json` parity with the CLI,
and a test asserts the page renders none of them. The one use of `citation_index` is legitimate: it is
what makes `[来源 2]` in the answer find the card labelled `来源 2`.

Groundedness is graded rather than dumped. Only **citation binding mismatches** and **speaker
attribution flags** become warnings; a silence claim — which cannot be judged without knowing the
intended answer — is a soft line of text (`记录可能不完整，请结合原始聊天确认。`) rather than an alarm.
When there is nothing to say, nothing renders.

## Privacy

* **`127.0.0.1` only.** `HOST` is a module constant in `webapp/app.py`, not a parameter, and `web.py`
  has no flag that reaches it. This process serves real private chat to whoever can reach the port, so
  a wildcard bind is a different product, not a configuration.
* **No browser persistence.** No `localStorage`, `sessionStorage`, `IndexedDB` or cookies anywhere in
  the frontend; no query history. Verified by grep, not by intent.
* **Question text never reaches a log.** The recall endpoint catches every exception and prints only
  the exception *class name*, because an upstream API error quotes its request — which here is the
  question. The trade-off is deliberate and worth stating: a genuine bug in the assembly surfaces as a
  502 plus a class name on stderr rather than a traceback.
* **Conversation ids are not printed.** A card header shows the display name when the export recorded
  one, and omits the header entirely when it did not — a wxid is an internal identifier and the UI has
  no business showing one.
* **No upload.** Ingestion stays a CLI step (`export_account.py`), so the web process never handles a
  file it did not already have, and the browser needs no filesystem permission.

## Verification

**Live, against a real account export** (3 conversations, 4238 messages, 3 real message shards):

```
index    : 3 conversation(s), 4238 message(s) in 66.4s
GET  /api/status        -> ready, conversation_count=3, message_count=4238,
                           coverage 2025-06-02 19:14 .. 2026-09-17 16:47, partial=true
POST /api/recall        -> 200, keys exactly {answer, evidence, groundedness, latency_ms},
                           6 evidence cards, 6 with a conversation label,
                           0 exposing a conversation id or a score
POST /api/recall "   "  -> 400
GET  /openapi.json      -> 404  (docs disabled)
```

The binding was checked rather than assumed: the machine's LAN address
(`169.254.83.107:8123`) is **unreachable** while `127.0.0.1:8123` answers.

`partial=true` above is the Phase 19 completeness rule arriving in the UI, and it exposed one
integration gap that was fixed here: `web.py`'s startup warning restated only the missing-shard cause,
so a **narrowed** export — every shard present, 269 of 272 conversations absent — announced
"0 message shard(s) were not exported" over an incomplete history. It now reprints the report's own
warning lines, so the two causes cannot drift apart again.

**Tests:** see the closing status table. The web tests are offline and deterministic — they stub the
brain, so no model, no network and no real data are needed.

## Known limits

* **The page was never seen rendered.** There is no browser and no jsdom in this environment, and no
  npm toolchain was added to get one. `node --check` passes, the HTML/CSS/JS are served 200, and the
  CSS is written to the brief (light background, one centred 880px column, CJK font stack, hairline
  borders, a `max-width: 560px` breakpoint), but **phone-width appearance is unconfirmed**.
* **Real-account scale was not measured end to end for the web process.** Indexing 4238 messages took
  66.4s; an account of 449 conversations is a different order of magnitude, and index persistence does
  not exist yet.
* **Concurrency is untested.** Whether `Brain.ask` is safe under two simultaneous calls on one brain
  was not established. The page blocks a double submit; nothing server-side serialises.
* **One recall path is enforced, not proven exhaustive.** A test asserts `webapp/app.py` never calls
  the assembly steps directly, which closes the way this could realistically drift — not every way.

## Phase 20: status

| requirement | state |
| ----------- | ----- |
| starts on localhost with one command | ✅ `python web.py` → `http://127.0.0.1:8000` |
| account-wide index loads | ✅ 3 shards / 3 conversations / 4238 messages indexed once at startup |
| can ask a real question | ✅ a real DeepSeek call returned answer + evidence + caveats |
| answer renders | ✅ |
| evidence cards render | ✅ conversation name, time range, participants, original lines |
| citations map to the right card | ✅ `[来源 N]` ↔ the card labelled `来源 N` |
| groundedness warns correctly | ✅ mismatches and attribution alarms; silence claims softened; silent when clean |
| retrieval logic not duplicated | ✅ one `answer_question`; pinned by a test |
| UI stays minimal | ✅ 490 lines of frontend; no sidebar, tabs, charts, banners or settings |
| usable at phone width | ⚠️ written to the brief and served correctly, **not visually confirmed** |
| real data never enters the repo | ✅ `data/real/` ignored; no browser persistence; ids never printed |
| tests pass | ✅ see the closing table |

---

# Phase 20.1 — account-wide real acceptance eval parity

`real_eval.py` had remained on the Phase 18D shape: `--corpus` built one corpus with
`build_session()`, then the harness repeated `brain.ask → serialize_sources → assess` itself. The
product had since moved to an account-wide index and a shared answer assembly path.

The runner now accepts `--account` and builds that index with `build_account_session()`. Each question
goes through `answer_question()`, the same function used by the CLI and web UI; `--corpus` remains as
the legacy acceptance input. `--report-only` needs no source because it reads only the existing result
file. A structural test rejects reintroducing direct `brain.ask`, source serialization, or groundedness
assembly in the harness.

**Verification:** 437 offline tests pass; no model call and no real-data read were needed.

---

# Phase 20.5 — Full account operational acceptance

Phase 19 proved the rules on fixtures and Phase 20 built a UI on a bounded sample of 8 conversations.
This phase ran the whole thing on the real account, at real scale, and it is the first time any of
these numbers existed.

Everything below is an **aggregate**: shard names, counts, date ranges and durations. No wxid, no
display name, no group name, no message text and no question appears anywhere in this document or in
the commands that produced it.

## The account, located without being asked

Read from the live `~/.weflow-cli/config.json`: `dbPath3x` → `…\Msg\Multi\MSG2.db`, whose parent
directory is the account. The machine holds exactly one directory with a `Msg\Multi`, and it is the
one the config points at, so there was no ambiguity to resolve and no reason to prompt.

| shard | size | shard WAL |
| ----- | ---- | --------- |
| `MSG0.db` | 180 MB | 0 MB |
| `MSG1.db` | 240 MB | 0 MB |
| `MSG2.db` | 210 MB | 1.1 MB |

`FTSMSG*.db` (search indexes) and `MediaMSG*.db` are excluded by `discover_message_shards`, which the
dry run confirmed: 3 shards detected, not 9.

**WAL caveat, recorded not fixed.** `MSG2.db-wal` holds 1.1 MB that the exporter cannot see, and
WeChat writes the newest messages to the log before merging it into the database. The newest messages
in the account may therefore be missing from everything below. This is the caveat Phase 19 already
documented; it is restated here because a full-account number makes it easy to forget.

## Export

Dry run first, and it was clean: all three shards listed without an error, no lock, no key problem.
`196 + 146 + 107 = 449` conversation exports to produce, nothing written.

Then the full export — **no `--only`, no `--shards`**:

```
message shards : 3 exported of 3 detected ['MSG0', 'MSG1', 'MSG2']
conversations  : 449 listed
files written  : 449 (412416 messages reported by the exporter)
failures       : 0
filtered_conversations: 0
wall time      : 39.5 min
```

Every completeness condition the phase sets is met: `missing_shards == 0`, `filtered_conversations ==
0`, all detected shards exported, zero failures, and nothing skipped silently.

**The user's real config was not touched.** Its SHA-256 was recorded before the export and compared
after: identical. The exporter reads it once and runs `weflow-cli` against a temp scratch profile it
deletes on every exit path.

**The databases were opened read-only.** No `MSG*.db` was modified, moved, renamed or checkpointed.

## Audit — offline, no model, no embedding

`audit_account.py` (new this phase) parses and segments the whole tree without loading BGE, so a
failure here is unambiguously ingestion rather than memory.

```
conversation files     : 449
message rows in files  : 412416      <- counted from files
messages kept          : 412416      <- counted by the parser
conversations found    : 272
conversations imported : 272
conversations with no messages : 0
duplicates removed     : 0
without a serverId     : 279 (kept, never deduplicated)
skipped (unparseable)  : 0
coverage               : 2025-05-20 00:32:12 .. 2026-09-19 00:08:11
chunks                 : 22751
conversations spanning >1 shard : 118  (43 %)
largest conversation   : 176396 messages  (42.8 % of the whole account)
PARTIAL                : False
crossed chunks         : 0   PASS
```

Three things worth reading twice:

* **The two counts agree.** 449 files and 412 416 rows, counted from the filesystem; 272
  conversations and 412 416 messages, counted by the parser. A disagreement between them would have
  been the finding.
* **`discovered == imported` (272/272).** Phase 19's report distinguishes conversations that were
  found from conversations that contributed messages, and at this scale the difference is zero: there
  is no half-exported conversation hiding in the account, and nothing to explain away.
* **118 of 272 conversations (43 %) span more than one shard.** Cross-shard merging is not an edge
  case on real data; without Phase 18B, two thirds of the account's *conversations* would be split.
* **The boundary held.** `crossed_conversation_chunks` recomputed chunk membership from event ids
  across all 22 751 chunks and found none mixing two conversations. This is the phase's hard rule,
  checked on the real account rather than promised.

## Index build

The real product path, unchanged: BGE-small-zh-v1.5 on CPU, hybrid retrieval, `k=20`, hybrid pool 30,
`no-rewrite`, the cited-attributed prompt.

```
index : 272 conversation(s), 412416 message(s) in 870.6s   (14.5 min)
```

Peak process RSS observed during the build: **≈1.5 GB**. No OOM, no FAISS error, no Windows path
problem, no event-loop error. The index is built **once at startup** and reused; there is still no
persistence, so 14.5 minutes is the price of every restart at this scale.

## Web, in a real browser

`GET /api/status`:

```
ready: true, conversation_count: 272, message_count: 412416,
coverage: 2025-05-20 00:32:12 .. 2026-09-19 00:08:11, partial: false
```

Phase 20 shipped with a known gap — nobody had ever seen the page render. Closed here by driving real
headless Chrome over CDP, typing into the real input and submitting the real form, so the path under
test is the one a person takes:

| viewport | horizontal overflow | column | cards | card overflow | answer / evidence / warning |
| -------- | ------------------- | ------ | ----- | ------------- | --------------------------- |
| 1280×900 | 0 px | 880 px | 7 | 0 | rendered / rendered / rendered |
| 390×844 | 0 px | 390 px | 2 | 0 | rendered / rendered / rendered |

CJK font stack resolves, background is the intended light `#fcfcfb`, the input is 304×48 and the
button 46×48 (comfortable at phone width). The card count differs between the two runs because the
cards follow what the *answer* cited, and answer generation is not reproducible run to run — a
documented property of the product, not a defect.

A real question was asked end to end: HTTP 200, answer, evidence, groundedness, `latency_ms` 43.7 s
(18.3 s on an earlier question) at `k=20` over 22 751 chunks.

## Two real bugs — both found only by looking at real data

Neither was reachable from a synthetic fixture, and both were identifier leaks.

**1. Every evidence card printed a raw talker id where a conversation name belongs.**
`weflow-cli sessions --json` on this version returns `displayName` **equal to** `username` when it has
no remark or nickname to resolve. All 272 of the account's conversations came back that way, so
`read_conversation_labels`'s guard of "is the display name non-empty" passed for every one of them and
the card header read `来源 10` followed by the group's raw talker id. The web backend's own docstring promised the
opposite — *"a wxid is an internal identifier, and the UI has no business printing one"* — and the
guard tested the wrong thing.

Fixed by moving the rule to where identity rules live: `ConversationDescriptor.has_real_name` is false
when the display name is absent, equals the talker, or carries the group suffix. `read_conversation_labels`
now returns **0** labels for this account instead of 272 identifiers, and the card simply shows
`来源 10` with the time range and participants.

Fixtures never caught it because a fixture always gives the two values different string constants.

**2. The API payload carried the talker inside `memory_chunk_id`.** A chunk id is
`f"{conversation_id}-session-NNNN"`, so the response handed the browser a group id in a field
alongside the conversation label that had just been removed. The page never rendered it — but a
response to a browser is still shown to a browser.

This one is worth recording for *how* it was missed: the first version of the check asked whether the
payload contained a `conversation_id` **key**, found none, and passed. Asserting on the payload's
**values** found it immediately. `webapp/app.py` now projects each card onto `PAGE_CARD_FIELDS` — the
six fields `app.js` actually renders — so debug metadata is dropped rather than forwarded, and a test
pins the field set exactly.

Both fixes have regression tests that fail without them.

## Tests

**455 pass** (437 before this phase: +14 for `audit_account.py`, +4 for the two privacy fixes).

## Known limits and what to do next

* **No conversation names exist anywhere in this pipeline.** The weflow-cli version tested returns
  `displayName == username` for every conversation, so after the fix the cards carry no label at all.
  A reader can tell *when* and *between whom*, but not *which chat*. Getting real names means reading
  `weflow-cli contacts` (remarks, nicknames, group names) and joining on the talker — **this is the
  single biggest remaining gap**, and it is a small feature rather than a redesign.
* **Real acceptance questions are still pending.** `eval_questions.json` does not exist; the 18-slot
  template is untouched. Questions must be ones whose answers the user already knows, which is what
  makes the four manual labels meaningful, so they were not invented here. Full account ready; real
  acceptance eval pending user.
* **14.5 minutes per start.** No index persistence, no embedding cache, no incremental indexing. Not
  optimised by design — this phase measured first.
* **One conversation is 42.8 % of the account**, which will dominate any latency figure that mixes
  conversations together.
* **WAL**, as above: committed history, not a guarantee of the newest messages.
* **A conversation-filtered export still cannot be detected if its manifest is deleted** (a Phase 19
  limit). The full export here is unfiltered, so it does not apply to these numbers.

## Phase 20.5: status

| requirement | state |
| ----------- | ----- |
| the full account is discovered | ✅ located automatically from the live weflow config, one candidate |
| every MSG shard was processed | ✅ 3 of 3; `FTS*`/`Media*` correctly excluded |
| no `--only`, no `--shards`, `filtered_conversations = 0` | ✅ |
| export tree built | ✅ 449 files, 412 416 messages, 0 failures, 39.5 min |
| conversations unioned correctly | ✅ 272 discovered, 272 imported |
| multi-shard conversations merged | ✅ 118 of 272 span more than one shard |
| `crossed conversation chunks = 0` | ✅ PASS over 22 751 chunks |
| `AccountImportReport.partial = false` | ✅ (`missing_shards = 0`, `filtered_conversations = 0`) |
| full-account session build | ✅ 22 751 chunks, 0 skipped |
| full-account BGE + FAISS index | ✅ built, no OOM, no error |
| real build time recorded | ✅ 870.6 s, ≈1.5 GB peak RSS |
| web ready, `/api/status` correct | ✅ 272 / 412 416 / `partial: false` |
| a real query end to end | ✅ HTTP 200 with answer, evidence, groundedness, latency |
| the page actually looked at | ✅ desktop and 390 px, 0 overflow, cards readable |
| real data still git-ignored | ✅ `data/real/` covers the export, logs, audit and screenshots |
| real acceptance eval | ⏸ **pending user questions** |

---

# Phase 20.6 — Human-readable conversation names

Phase 20.5 found that evidence cards had no conversation header at all. The cause was that
`weflow-cli` returns `displayName` equal to `username` for every conversation, and the fix there
stopped the page printing an internal id in its place. This phase set out to supply a real name.

**Outcome: the resolution layer is built, tested and wired in, and it resolves 0 of the account's 272
conversations — because this installation of `weflow-cli` exposes no human-readable name through any
non-interactive command.** The cause is measured and stated below rather than papered over with a
fallback.

## What was investigated

Every read command the CLI advertises was probed through the project's own runner (same scratch
profile, same config handling), and the shape of each response recorded:

| command | result |
| ------- | ------ |
| `sessions --json` | 272 conversations, **every** `displayName == username` |
| `contacts --json` | 107 contacts, **every** `displayName == username` |
| `messages <talker> --json` | `{success, talker, messages}` — no name field at all |
| `capabilities` | documents `contacts --json` as the only contact source; no name-bearing alternative |
| `config show` / raw config | `contactDbPath`, `contactKey`, `contactSalt` all unset |
| `whitelist list` / `blacklist list` / `sns users` / `fav list` | no name-bearing listing |
| `init --dry-run --json` | `interactiveRequired: true` — no per-database detail, and it reports it will write configuration |

Two further experiments, both negative:

* **Repointing `contactDbPath` at the on-disk contact databases** (`MicroMsg.db`, 30 MB;
  `ChatRoomUser.db`) in a scratch profile changed nothing: still 107 contacts, still zero real names.
  The `contacts` command does not appear to read that key on this version.
* **The export envelope carries no name either.** A `{talker}_messages.json` is a bare array of
  `content / createTime / isSend / localId / localType / parsedContent / rawContent / senderUsername /
  serverId`. `senderUsername` is a wxid.

The names exist on disk. `weflow-cli` does not surface them with this configuration, and the command
that would configure it (`init`) requires an interactive terminal and writes the user's config — which
this project never does. Parsing the databases directly was considered and rejected: it is out of
bounds for this project by explicit rule, and it would be a second, unmaintained exporter.

**So this is a configuration gap in the installed tool, not a design problem in the product.**

The one hypothesis left at the time — that `init` was the unlock and had simply never been run because
it needs a terminal — has since been **tested and disproven** (Phase 20.8B). The user ran `init` in an
interactive terminal; `contacts --json` afterwards still returned every `displayName` equal to its own
`username` (120 of 120 contacts, 0 real names). So the measured statement is the narrower one: **the
tested CLI version exposes no human-readable name through any non-interactive command, whatever its
configuration.** The resolution layer is complete and will use a name the moment a version offers one;
no fix is claimed, because none was found.

## What was built

Decoupled end to end, so name enrichment never requires re-exporting messages (39.5 minutes) and never
touches the 412 416-message tree:

```
weflow-cli contacts --json
        ↓  exporter.list_contacts()          (the only module allowed to run weflow-cli)
        ↓  memory.labels.parse_contact_listing()
        ↓  resolve_conversation_labels()     precedence, per conversation kind
        ↓  conversation_labels.json          local sidecar, in the git-ignored account tree
        ↓  webapp.read_conversation_labels()
        ↓  evidence card header
```

| file | role |
| ---- | ---- |
| `memory/labels.py` | **new** — the usable-name rule, the precedence tiers, the sidecar format |
| `sync_conversation_labels.py` | **new** — contacts → sidecar, prints coverage and never a name |
| `exporter.py` | `list_contacts()`, reusing the existing scratch-profile mechanism |
| `webapp/app.py` | the sidecar is authoritative when it holds anything; the tree's own listing is the fallback |
| `tests/test_labels.py` | **new** — 77 tests |

### One rule, one place

`usable_name(name, conversation_id)` returns the name when a person may read it, else `""`. It rejects
empty/whitespace, a name equal to the conversation id, and anything containing `@chatroom`.
`ConversationDescriptor.has_real_name` now **delegates to it** rather than holding a second copy of
the same test — the two drifting apart is exactly how Phase 20.5's leak happened. It is applied on the
way in (resolution) and on the way out (the sidecar reader), so a hand-edited sidecar cannot smuggle
an id into a card.

### Precedence

```
direct : remark → nickname / displayName → alias → (nothing)
group  : the group's own name → (nothing)
```

Ordering is by **field kind**, not by the order a payload happens to list its keys; ties inside a kind
break deterministically. A group is deliberately given no remark tier — the name a group shows is the
name its members gave it. **A talker id is never a fallback in any tier**, for either kind.

### Coverage, measured on the real account

```
conversations  : 272 (direct 136, group 136)
contact records: 107
labels resolved: 0 of 272
  direct       : 0 resolved / 136
  group        : 0 resolved / 136
```

The CLI prints exactly this, and a note naming the cause and the unlock. It prints no label, no talker
and no wxid, so the output is safe to paste.

## Verification

* **The rule, against the real data shapes**: empty → `""`; whitespace → `""`; `displayName` equal to
  the talker → `""`; a group id → `""`; a *different* group id → `""`; a real name → the name.
* **The whole chain with synthetic names**, since no real one exists: a synthetic sidecar over a
  synthetic tree resolves through the real `read_conversation_labels` and yields names, with no value
  equal to its own key and no `@chatroom` in any label.
* **Degradation**: with no sidecar, the reader returns `{}` — byte-for-byte the pre-20.6 behaviour, so
  every existing export keeps working.
* **The user's real `~/.weflow-cli/config.json` is untouched**: SHA-256 `9b61ad15b461ef09` before and
  after the sync run.
* **The sidecar is git-ignored** (`data/real/`), and `shard_manifest.json` still carries counts only.
* **No identifier reaches the browser**: a test walks every nested **value** in the API response (key
  names alone are how Phase 20.5's leak survived its first check).

**Tests: 532 pass** (455 before this phase, +77). No retrieval code was touched: no change to
embedding, FAISS, hybrid search, RRF, `k`, hybrid pool, chunking, prompts or workflow, and no name is
injected into chunk text.

## Known limits

* **No real name has ever resolved on this installation.** The resolution path is proven against
  synthetic payloads shaped like the measured responses, not against a live success.
* **The `remark` / `nickname` key spellings are defensive, not measured.** `username` and `displayName`
  are the measured keys; the others are spellings already used elsewhere in this project. If the
  unlock emits a different spelling, adding it is a one-line change to `_KEY_KINDS`, and an unknown
  field is ignored rather than fatal.
* **A hand-edited sidecar can disagree with `contacts`.** The sidecar is authoritative by design —
  the alternative was a card whose header depends on which source happened to know that conversation.
* **Duplicate names are allowed and not disambiguated.** Two conversations may both read "小王": a
  label is not an identity, and the frontend is never shown a stable id to tell them apart.

---

# Phase 20.7 — Full-account real acceptance eval

The first time the engine has been asked questions whose answers a human actually knows, against the
whole account. Everything below is an aggregate: counts, timings and anonymous digests. The questions,
the answers, the evidence and the labels stay in `data/real/`, which `.gitignore` excludes.

## What was run

```
real_eval.py --questions data/real/eval_question.json \
             --account data/real/account_full \
             --shard-dir "<the account's Msg/Multi>" \
             --out data/real/real_eval_results.json
```

The product defaults were used exactly as shipped and were not re-tuned for this run: session
chunking, BGE-small-zh-v1.5, hybrid retrieval, `k=20`, hybrid pool 30, `no-rewrite`, cited-attributed
prompt. This is acceptance, not another experiment arm.

The index was built **once** for this phase (≈14.5 minutes, the Phase 20.5 figure), not once per
phase.

## The question set

18 questions, three in each of the six categories:

| category | n | what it is for |
| -------- | - | -------------- |
| `single_fact_recall` | 3 | one fact, usually from a single episode |
| `timeline_reasoning` | 3 | something that changed over time; needs several points ordered |
| `speaker_attribution` | 3 | who recommended / said / did it — the guard against borrowing someone else's experience |
| `multi_source_synthesis` | 3 | answer must combine evidence from more than one time or conversation |
| `abstention` | 3 | the user is confident the record does not contain it |
| `older_memory` | 3 | genuinely old material, against recency bias over 412 416 messages |

Validated **offline before anything was spent**, through the harness's own `load_questions()`: 18
questions, ids unique across `q01`–`q18`, all six categories populated, every question ≥ 6 characters
(minimum measured 13), every question carrying a note (minimum 21 characters), and the placeholder
marker absent from every `question` and every `note`.

Two details worth recording:

* **`note` never reaches the model.** It is the user's own gold memory — expected answer, rough date,
  rough contact, known state changes — and it is written into the result record for the human labeller
  only. `real_eval.py` passes `question.question` to `answer_question`; the note is never part of the
  prompt, the retrieval query or anything sent to DeepSeek.
* **The file carries a leftover `instructions` block** holding the template's placeholder marker. The
  loader ignores that field entirely, and it reaches neither the model nor the results, so it is
  cosmetic — but it is the kind of thing that would fail a stricter validator, and it is recorded here
  rather than silently ignored.

## Latency

| category | mean ms | median ms | mean sources |
| -------- | ------- | --------- | ------------ |
| `single_fact_recall` | 31 478 | 25 153 | 20.0 |
| `timeline_reasoning` | 54 471 | 33 448 | 20.0 |
| `speaker_attribution` | 49 108 | 40 169 | 20.0 |
| `multi_source_synthesis` | 44 899 | 46 357 | 20.0 |
| `abstention` | 31 824 | 31 831 | 20.0 |
| `older_memory` | 26 454 | 28 026 | 20.0 |

**Overall mean 39 706 ms, median 32 640 ms.** Every question retrieved the full `k=20`, and the mean is
well above the 14.5-minute index build amortised per query — this is a local-CPU embedding plus a
network generation call over a 22 751-chunk index, not a tuned latency.

## Automated signals

| signal | count |
| ------ | ----- |
| invalid citations | **0** |
| citation binding mismatches | **0** |
| attribution flags | 3 (all inside one `speaker_attribution` question) |
| silence claims | 14 |
| answers carrying any warning | 13 / 18 |

Zero invalid citations and zero binding mismatches across 360 retrieved sources is the strongest
result here: every citation the answers made resolved to a real source, and no answer quoted text that
was not in the chunk it cited.

The 14 silence claims are **reminders, not alarms** — the documented base rate for that signal is
around 30 % of answers, and the tool cannot distinguish "the record does not show it" from "it was not
retrieved". The three attribution flags sit exactly where the category was designed to put them.

## Conversation diversity, and whether the outlier dominates

Phase 20.5 measured one conversation holding **42.8 %** of the account's messages. The obvious worry
was that it would also swallow the evidence for questions that have nothing to do with it. It does not:

```
questions whose evidence came from exactly one conversation :  0 / 18
mean distinct conversations per question                     : 12.3
max distinct conversations for one question                  : 16

retrieved slots                                              : 360
distinct conversations appearing in evidence                 :  86
most-retrieved conversation's share of all slots             : 14.7 %
corpus baseline: the largest conversation, by message count  : 42.8 %
```

The largest conversation is **under**-represented relative to its size, not over. Retrieval is not
proportional to corpus mass, and no single conversation crowds the others out — 86 distinct
conversations appear across 360 slots.

This is recorded as an observation, not acted on. Nothing about retrieval was changed in this phase.

## What is deliberately not concluded

**All 18 `labels` objects are still `null`.** `answer_correct`, `retrieval_correct`,
`citation_binding_correct` and `attribution_correct` are human judgements; no judge model was
introduced and nothing was inferred from the answers. The four fields are the deliverable's hinge, and
the failure-case table the phase is really after —

```
Retrieval Miss          Wrong Conversation     Temporal Confusion
Speaker Attribution     Multi-evidence Failure Generation Failure
Citation Binding        Unsupported Claim      Correct / False Abstention
Large-conversation Dominance (here: not observed)
```

— cannot be filled in from latency and warning counts. The evidence needed to fill it is saved in
`data/real/real_eval_results.json`, and the next step is a human reading it alongside the notes.

## Phase 20.7: status

| requirement | state |
| ----------- | ----- |
| question set validated offline before spending | ✅ 18 / 6 categories / ids unique / no placeholder in any question or note |
| full account, product defaults, `k=20` | ✅ unchanged; one index build shared with the phase |
| all questions executed | ✅ 18 / 18 |
| automatic aggregates recorded | ✅ latency, sources, invalid citations, mismatches, attribution, silence claims |
| conversation diversity measured | ✅ 12.3 distinct conversations per question, 0 single-conversation answers |
| outlier dominance checked | ✅ 14.7 % of slots against a 42.8 % corpus share — not dominant |
| manual labels | ⏸ **all 18 pending the user** |
| failure-case table | ⏸ awaiting those labels |
| real eval data kept out of the repo | ✅ questions, results and the analysis script all under `data/real/` |

---

# Phase 20.8B — Group sender identity

The acceptance eval in Phase 20.7 surfaced a defect that no synthetic corpus had: **in a group chat,
every non-self speaker rendered as the literal `对方`.** The adapter decided the speaker from `isSend`
alone — `1` became `我`, everything else became `对方` — which is exactly right for a direct chat, where
there are only two speakers and the role *is* the identity, and collapses entirely in a group: A, B and
C all speaking produced `对方, 对方, 对方`. Asked "who did X", the answers correctly reported that the
record only labels them `对方` — the evidence could not tell the members apart, so nothing could be
attributed.

## The model: role, identity, label

One display string was carrying three different facts. They are now separate:

| concept | field | what it is |
| ------- | ----- | ---------- |
| role | `MemoryEvent.speaker_role` | `"self"` / `"other"` — what `isSend` actually means |
| identity | `MemoryEvent.speaker_id` | the exporter's `senderUsername`. Data, never a label |
| label | `MemoryEvent.sender_name` | what `MemoryEvent.line` renders — all the LLM and the card see |

Both new fields have defaults, so the plain-text adapter, the synthetic corpora and every existing
constructor keep working untouched. `sender_name` is the only one anything rewrites.

## The rule

* **self** → `我`, always, in both kinds of conversation. The attribution safety clause in the prompt
  and `attribution_screen.py` match this exact string, so substituting anything else would silently
  disable them.
* **direct, other** → `对方`, unchanged. Role alone distinguishes two speakers, and keeping the string
  identical makes this phase a strict no-op for every already-evaluated direct corpus.
* **group, other** → a deterministic pseudonym, `成员A`, `成员B`, …, assigned by **first appearance in
  that conversation's chronological order** and continuing `…Y, Z, AA, AB, …`. Never a hash, never a
  random number, never the wxid: a label derived from the id is the identity spelled again.

Two edges are answered rather than guessed. A member whose export carried no `senderUsername` keeps
`对方` — two anonymous messages cannot be shown to be the same person or different people, and `对方`
can never collide with an assigned `成员X`. A stream whose source declares no role at all (the
plain-text adapter names its speakers directly) is left completely alone.

## Where it is applied, and why that is the subtle part

Over the **whole conversation's stream**, never per shard. Applied from inside `parse_weflow_events`,
a conversation spanning `MSG0`/`MSG1` would get shard A's first speaker as `成员A` and shard B's
*different* first speaker as `成员A` too — two people with one label, which is worse than the defect it
replaces because it reads as a fact rather than as an absence.

| call site | why | why it cannot split a conversation |
| --------- | --- | ---------------------------------- |
| `memory.shards.merge_shard_events` | the stream is final here; covers the shard directory and `import_account` | the merge has already combined every shard |
| `memory.processor.WeFlowSessionProcessor` | a single `{talker}_messages.json` | the whole conversation is in the one file being read |
| `recall.count_corpus_events` | the same single file, counted rather than indexed | the same |

## What is deliberately unchanged

Change **only** the sender representation. The embedding model, FAISS, hybrid search, BM25, RRF, `k`,
the hybrid pool, session chunking, the prompt, the reranker and the workflow are all identical, and no
name is injected into chunk text beyond the speaker label itself. A later controlled eval can attribute
a change in results to this commit and nothing else.

## Privacy

No raw `senderUsername`, wxid or `@chatroom` token appears in a rendered line, a session chunk, a
participant list, or the document metadata a prompt and the API read. Checked on values, not on key
names: the leak this project already had survived a check that only looked at names. The conversation
id itself — which for a group carries the chatroom suffix — is still carried by `conversation_id` and
`memory_chunk_id`; that predates this phase, and the web projection that strips it from the browser
payload is unchanged.

## Tests

`tests/test_senders.py` — 37 tests, all offline and synthetic. The brief's case is asserted literally
(`我 / 成员A / 成员B / 成员A`), alongside: two members posting byte-identical text still getting
different labels; one id keeping one label across shards; a group spanning `MSG0`/`MSG1` labelled once
over the merged stream; determinism across runs; 28 members continuing past `Z` to `AA`; an all-self
group and a one-member group; an id-less sender staying `对方`; and no wxid or chatroom token in any
rendered value. `tests/test_account_isolation.py` gained one changed expectation — a group's
`participants` is now `("成员A", "我")` instead of `("对方", "我")`, which is the point of the phase.

## What the real export actually carries — measured, and it changes the conclusion

Everything above is verified against synthetic exports. The real account was then swept, and it does
not carry the data this fix needs:

| checked across all 589 exports of the migrated account | result |
| ------------------------------------------------------ | ------ |
| `senderUsername` equal to the conversation's own talker | **562** |
| `senderUsername` differing from the talker | **0** |
| exports with more than one distinct non-self sender | **0** |
| exports with no non-self message at all (not applicable) | 27 |

**The exporter writes the *conversation* into the sender field.** Every group message in the account
carries the chatroom's own id as its `senderUsername`, so there is no per-member identity anywhere in
the exported data to preserve.

Without a guard, the pseudonym rule would therefore stamp every real group with a single `成员A` —
asserting that exactly one other person said all of it. That is a worse answer than `对方`, because it
reads as a fact rather than as an absence. So the rule carries a second edge, applied and tested:

* a sender id that **is the conversation's own id** is not an identity, and those messages keep `对方`
  — the same answer as before, for the same reason (`labels.usable_name` already refuses a name that
  is the identity spelled again).

**The honest conclusion for this phase is therefore negative on real data**: the model is now correct
and the collapse is fixed *where a sender id exists*, but on this exporter **no group gains member
attribution** — every one still renders `对方`. The value delivered is the separation itself plus the
guard: the day an exporter supplies a real sender id, the labels appear with no further change, and in
the meantime the engine does not manufacture a member who was never in the record.

This also means the controlled re-eval (Phase 20.8C) is expected to show **no movement** from A to B:
the sender representation of the real corpus is byte-identical before and after, because there was
never an identity to render. That is a prediction with a mechanism behind it, and it is worth
measuring rather than assuming.

## Phase 20.8B: status

| requirement | state |
| ----------- | ----- |
| self still the literal `我` | ✅ asserted, and the attribution tests are unchanged |
| direct conversation still `对方` | ✅ asserted, including with several other ids |
| group members distinguishable **when the export carries a sender id** | ✅ `我 / 成员A / 成员B / 成员A`, synthetic |
| group members distinguishable **on this project's real export** | ❌ **not possible** — 562/589 exports name the conversation, 0 name a sender. Groups still render `对方` |
| the conversation's own id never becomes a member | ✅ guarded and tested |
| one id, one label, whole conversation | ✅ including across shards |
| labelled once over the merged stream | ✅ a multi-shard group cannot get two `成员A`s |
| deterministic, never derived from the id | ✅ two runs byte-identical; no wxid in any label |
| > 26 senders | ✅ `…Z, AA, AB` (Excel-column, no reuse) |
| no internal id in any rendered value | ✅ prompt text, chunk text, participants, served metadata |
| plain-text path untouched | ✅ `assign_sender_labels` is a no-op on a role-less stream |
| retrieval logic untouched | ✅ no change to embedder, index, search, chunking or prompt |
| `init` claim corrected | ✅ sync layer states the measured fact, not the hypothesis |

---

# Phase 20.8A — Full-history refresh after the phone migration

The user migrated their complete phone chat history into the PC WeChat. Everything measured in
Phase 20.5 and evaluated in Phase 20.7 was therefore a *partial* view of their past, which put a
standing question mark over every retrieval miss and silence claim in those results. This phase
established how much history arrived, refreshed the baseline, and ran the same acceptance eval again
on **completely unchanged code** so that any movement could be attributed to the data alone.

Aggregates only throughout: counts, shard names, date ranges and durations. No question, answer,
contact, group name or identifier appears below.

## First: did the migration actually land?

The sanity check the phase opens with, and it mattered — a full export costs two hours:

| | Phase 20.5 | Phase 20.8A |
| --- | --- | --- |
| shards | 3 | **11** (eight new: `MSG3`–`MSG10`) |
| `MSG0` / `MSG1` / `MSG2` size | 180 / 240 / 210 MB | **360 / 690 / 270 MB** |

The active database had plainly changed, so the export proceeded. The dry run listed every shard
without an error: 588 conversation-exports to produce against 449 before.

## Export

No `--only`, no `--shards`. The old tree was left untouched and the new one written beside it.

```
message shards : 11 exported of 11 detected
conversations  : 589 listed
files written  : 589 (1625160 messages reported by the exporter)
failures       : 0
filtered_conversations: 0
wall time      : 126.4 min   (Phase 20.5: 39.5 min)
```

## What the migration actually added

Both columns are from `audit_account.py`, the same instrument, on the two trees:

| metric | OLD (20.5) | NEW (20.8A) | delta |
| ------ | ---------- | ----------- | ----- |
| shards | 3 | 11 | +8 |
| conversation exports | 449 | 589 | +140 (+31 %) |
| messages | 412 416 | **1 625 160** | **+1 212 744 (+294 %)** |
| conversations | 272 | **281** | +9 (+3 %) |
| chunks | 22 751 | **78 726** | +55 975 (+246 %) |
| cross-shard conversations | 118 | 124 | +6 |
| duplicates removed | 0 | 0 | — |
| without a serverId | 279 | 279 | — |
| skipped | 0 | 0 | — |
| `PARTIAL` | False | **False** | — |
| crossed chunks | 0 | **0 PASS** | — |
| discovered / imported | 272 / 272 | **281 / 281** | — |

**Coverage is the headline.** The earliest message moved from **2025-05-20 to 2019-08-26** — five and
a half more years of history — while the latest moved forward a day. The largest conversation grew from
176 396 to 526 711 messages but its *share* fell from 42.8 % to 32.4 %: the new material is spread
across the account rather than piled onto the conversations that were already biggest.

One conversation now spans **all eleven shards** and reaches back to 2019-08-26, which is the clearest
single illustration of why the per-conversation shard merge from Phase 19 was necessary rather than
nice to have.

## Control eval (A) — same 18 questions, unchanged code

The point of this run is attribution: **new corpus, old code**. The question set was verified
byte-identical to the one Phase 20.7 ran (ids, question text, notes, categories) before anything was
spent, so a difference in the results cannot be a difference in the questions.

Validity was then confirmed after the fact: the run's 18 records contain **zero** `成员X` labels, i.e.
they were produced by the pre-Phase-20.8B speaker rendering. (The run had been launched just before
that work landed, and its modules were imported before the first file was written — the absence of an
`ImportError` from the new module is itself the proof.)

### OLD vs A

```
answers that changed textually : 18 / 18
answers whose evidence changed : 18 / 18   (7–19 of 20 sources replaced, every question)

signal                  OLD(20.7)   A(20.8A)
silence claims                 14         11     <- 2 resolved, 0 newly introduced
citation mismatches             0          0
attribution flags               3          3
invalid citations               0          0
mean latency ms             39 706     91 830
mean distinct conversations    12.3       11.3
```

Every answer changed and essentially the whole evidence set turned over. That is what a 3.9× corpus
with 5.7 more years in it should do, and it is why the raw "did the answer change" count is not
evidence of improvement on its own.

The signal worth reading is the asymmetric one: **two silence claims disappeared and none appeared**.
A silence claim is the engine saying the record does not contain something; the most likely reason for
a false one is that the evidence was never in the corpus, which is exactly what the migration
corrected. Three failures were *not* resolved by the extra history, which is itself useful — it says
those three are not simply missing data.

Latency roughly doubled (39.7 s → 91.8 s mean) on a 3.5× larger index. That is the cost of the
history, recorded rather than optimised.

## The measurement that replaces a three-hour re-run

Phase 20.8C asks whether preserving sender identity moves the results. Before spending another
index build on it, the mechanism was measured directly: the whole corpus was rendered by the
**pre-20.8B code** (a git worktree at the previous commit) and by the **current code**, and the
rendered lines hashed.

```
25 largest real conversations (13 of them groups), 628 996 rendered lines
  pre-20.8B : b97be9cf798b7dc9e72570eb305827f9
  20.8B     : b97be9cf798b7dc9e72570eb305827f9     <- identical
```

**The sender-identity change is a byte-level no-op on this corpus.** That is not a surprise in
hindsight — Phase 20.8B had already measured that 562 of 589 exports name the conversation rather than
a sender, so there was no identity to render — but it converts a prediction into a fact, and it means
the A→B comparison can only ever measure LLM sampling noise. The targeted re-run is still being
executed, because "measured end to end" beats "argued from a hash", but the expectation is now
explicit and falsifiable rather than vague.

## Phase 20.8A: status

| requirement | state |
| ----------- | ----- |
| migration verified before spending two hours | ✅ 3 → 11 shards, sizes up 2–3× |
| full export, no filter, old baseline preserved | ✅ 589 files, 1 625 160 messages, 0 failures, `account_full` untouched |
| audit: complete and separated | ✅ `PARTIAL` False, 0 crossed chunks, 281/281 discovered == imported, 0 skipped |
| OLD vs NEW comparison | ✅ +1 212 744 messages, coverage 2025-05 → **2019-08** |
| control eval on unchanged code | ✅ 18/18, validity confirmed by the absence of `成员X` |
| evidence-level diff | ✅ all evidence turned over; silence claims 14 → 11, citations still perfect |
| question set unchanged | ✅ verified byte-identical to Phase 20.7's |
| real data kept out of the repo | ✅ questions, results, diffs and the analysis scripts all under `data/real/` |

---

# Phase 20.8C — Controlled re-eval of the sender change

The controlled design is three runs over the same 18 questions:

| run | corpus | speaker rendering |
| --- | ------ | ----------------- |
| **OLD** | PC-history (272 conversations) | pre-20.8B |
| **A** | migrated (281 conversations) | pre-20.8B |
| **B** | migrated | post-20.8B |

OLD → A isolates the **data-completeness** effect (recorded in Phase 20.8A). A → B isolates the
**sender-identity** effect, and this is the record of it.

## Targeted speaker subset first (q07–q09)

The three `speaker_attribution` questions are where a sender-collapse defect would show, so they run
before anything else, on their own index build.

```
questions                  A (old code)   B (new sender)
answers changed textually           —          3 / 3
evidence changed                    —          0 / 3      <- zero sources gained, zero lost
distinct conversations           9.0           9.0
silence claims                      2             1
attribution flags                   3             0
citation mismatches                 0             0
invalid citations                   0             0
mean latency ms                 89 133        61 974
```

**Not one retrieved source differs.** For all three questions the evidence is identical — the same
sources in the same ranks, the same number of distinct conversations. The answers differ, and so do
the caveat counts, but those sit on top of byte-identical evidence: they are generation
nondeterminism, which this project has measured before and never claims otherwise.

That was predicted, and the prediction has a mechanism: Phase 20.8B swept all 589 exports and found
562 naming the conversation rather than a sender and **zero** naming a sender, so the guard keeps every
real group rendering `对方` — unchanged from before. The same conclusion was reached independently and
more cheaply by rendering 628 996 lines of the corpus through both code versions and comparing digests:
identical.

The value of this run is that it turned "argued from a hash" into "measured end to end through the
product path". A change that touches how every message is rendered, and moves nothing, is worth
confirming rather than assuming — and the falsifiable form of the claim is in the record now: **if a
single source had differed, the invariance claim would have been wrong.**

## What this means for the phase's own goal

Phase 20.8B set out to stop group members collapsing into one `对方`. It did the part that is possible
— the model is now correct, the identity is preserved as data, and the conversation's own id can never
be mistaken for a member — but **on this exporter no group gains attribution**, because the exported
data does not contain a sender identity to begin with. That is the honest result, and the A→B
comparison is what rules out the possibility that it was quietly working anyway.

## Full 18-question B run

The targeted subset suggested the invariance; the full set confirms it.

```
questions                      A (old code)   B (new sender)
answers changed textually             —            18 / 18
evidence changed                      —             0 / 18      <- zero sources gained, zero lost
distinct conversations              11.3            11.3
silence claims                        11              10
attribution flags                      3               0
citation mismatches                    0               0
invalid citations                      0               0
mean latency ms                   91 830          80 155
```

**Not one of the 360 retrieved sources differs, on any of the 18 questions.** Every `conversations`
column reads `N -> N` with the same N. The answers differ in every case and the caveat counts move
slightly in both directions (three silence claims resolved, one introduced; three attribution flags to
zero) — all of it generation nondeterminism on byte-identical evidence, which is why this project does
not treat a changed answer as a result on its own.

## The controlled comparison, complete

| comparison | isolates | outcome |
| ---------- | -------- | ------- |
| OLD → A | data completeness | every answer and every evidence set changed; silence claims 14 → 11; coverage extended 5.7 years |
| A → B | sender identity | **nothing moved** — 0/18 evidence changes, in both directions |

So the phase's two questions have measured answers. The migration was worth two hours of export and
four index builds: it resolved two silence claims outright and replaced the evidence behind every
question with evidence drawn from a corpus 3.9× larger and 5.7 years longer. The sender-identity
change, by contrast, moves nothing on this exporter — and now that is a measurement rather than an
expectation.

Both results are negative-leaning and both are useful. The first says the earlier acceptance run was
partly measuring a missing corpus. The second says the remaining failures are *not* a rendering
artifact: q07's "the record only labels them 对方" is not something a better label can fix, because the
exported data does not contain a member identity in the first place. Whatever those failures are made
of, they survive both corrections — which narrows the search for Phase 21 to retrieval and generation
rather than data plumbing.

---

# Phase 20.9 — Patched exporter integration and real sender regression

Phase 20.8 ended with a measured dead end: the exporter put the *conversation* into the sender field,
so 562 of 589 exports named the chatroom rather than the person and no group could gain member
attribution. The exporter has since been patched, and this phase integrates that contract, verifies it
on the real account, and re-runs the acceptance set against it.

Aggregates only: counts, ratios, shard names and durations. No person's name, no wxid, no chatroom id,
no question and no answer appears in this record.

## The patched exporter

Tested against a local, unpushed build (`213458c`, `master`, version 1.7.0 — note the upstream npm
install is 1.5.1 and its version number *is not* a reliable indicator of which build runs).

Exposed to this project non-destructively: a shim directory holding a `weflow-cli.cmd` that invokes the
local `cli.cjs`, prepended to `PATH` for the duration of the work. Nothing global was installed and no
npm package was overwritten. Verified through the path the product actually takes —
`exporter.run_subprocess` — rather than by reading a version string.

**A trap worth recording**: `where.exe` and Python's `shutil.which` both resolve the shim first and
correctly, but Git Bash's own command resolution prefers the extension-less npm shim, so a bare
`weflow-cli --version` in that shell reports the *old* build. The Python path is what matters and it
was confirmed to be 1.7.0.

## The raw contract gate

Run before touching any Personal Recall code, over a bounded export of the 24 conversations behind the
speaker-attribution questions, across all 12 shards — 937 548 group non-self messages:

| check | before the patch | after |
| ----- | ---------------- | ----- |
| `senderUsername` equal to the conversation id | 100 % | **0** |
| group exports with more than one distinct sender | 0 / 76 | **75 / 76** |
| distinct senders in the largest group | 1 | **471** |
| `senderUsername` empty | — | 2.0 % |
| `senderDisplay` usable | field did not exist | **97.5 %** |
| `senderDisplay` equal to an id, a conversation id, or containing a chatroom suffix | — | **0 / 0 / 0** |
| one sender, two different displays | — | **0** |

The gate the phase set — the conversation id must never appear as a sender — passed, so the
integration proceeded.

## The integration

Four things are now separate on `MemoryEvent`, and only the last is rendered:

| concept | field | what it is |
| ------- | ----- | ---------- |
| role | `speaker_role` | `self` / `other` |
| identity | `speaker_id` | the raw sender. Data, never rendered |
| display | `speaker_display` | the exporter's optional candidate name |
| label | `sender_name` | what `MemoryEvent.line` renders |

```
self            -> 我, always, both conversation kinds
direct, other   -> 对方, unchanged even when a display is present
group, other    -> a usable speaker_display
                   else a deterministic 成员A/B/C pseudonym
                   else 对方
```

"Usable" is `memory.labels.usable_name` — the project's single "is this a name or the identity spelled
again" rule — extended with the speaker's own id, so a value equal to the sender id, equal to the
conversation id, empty, or carrying a chatroom suffix is refused. A raw identity cannot reach the model
because the field carrying it happens to be called "Display".

Three things the naive version gets wrong, each handled and tested:

* **Resolution is over the whole conversation.** A display arriving on a later message labels that
  member's earlier messages too; resolving per event would make one person wear two labels inside one
  conversation.
* **Pseudonyms are numbered over every member**, named or not — numbering only the unnamed ones would
  renumber everybody the moment somebody acquires a name, so the same corpus would render differently
  between runs.
* **Two members resolving to one human name stay distinguishable**, qualified with the pseudonym each
  already had. This is not hypothetical: the real corpus produces **20 259 such labels**, every one of
  which would otherwise have been an identity collapse.

One ambiguity in the brief — a message carrying a display but no sender id — was measured rather than
argued: it occurs **0 times** in the real export, so the chosen reading changes nothing.

## Measured on the real bounded corpus

24 conversations, 1 269 939 messages, 58 724 chunks:

| rendered label | messages |
| -------------- | -------- |
| direct: `我` / `对方` | unchanged, byte for byte |
| group: a **human-readable name** | **914 498** |
| group: a pseudonym `成员X` | 4 463 |
| group: `对方` (no sender in the export) | 18 577 |

Crossed conversation chunks: **0**. Across 16 cross-shard group conversations, **0 speakers were given
more than one label**. A privacy sweep found **0 labels equal to a raw id** and **0 containing a
chatroom token**; the only two places a raw id appears inside rendered text are inside message
*content*, where a person pasted one.

## Full account v3

Not assumed to be 11 shards: rediscovered as **12** (`MSG11` is new). The export resumed across an
interrupted run rather than repeating it — the product writer re-exports everything on every run, so a
small driver that skips conversations already on disk was used to produce the identical tree. `v2` and
`v1` were left untouched.

```
shards           : 12 of 12        files written : 612        failures : 0
the export was NOT filtered (filtered_conversations = 0)
PARTIAL          : False           missing shards: none       crossed chunks: 0
```

**v2 → v3** (the snapshot advanced; this is *not* attributable to the sender patch):

| metric | v2 | v3 | delta |
| ------ | -- | -- | ----- |
| shards | 11 | 12 | +1 |
| conversation files | 589 | 612 | +23 |
| messages | 1 625 160 | **1 630 452** | +5 292 |
| conversations | 281 | 281 | 0 |
| chunks | 78 726 | **81 576** | +2 850 |
| coverage start | 2019-08-26 | **2019-08-13** | 13 days earlier |

## Sender coverage, full account

| | value |
| --- | ----- |
| group non-self messages | 1 243 041 |
| sender identity resolved | **1 210 494 (97.38 %)** |
| sender identity unknown | 32 547 (2.62 %) |
| **sender == conversation id** | **0** |
| human-readable display resolved | **1 195 322 (96.16 %)** |
| display equal to an id / conversation id / containing a chatroom suffix | **0 / 0 / 0** |
| distinct real senders | **5 017** (4 687 of them, 93.4 %, have a display) |

## Conversation labels

`sync_conversation_labels.py` against v3 with the patched exporter:

```
conversations   : 281 (direct 140, group 141)
labels resolved : 131 of 281      (Phase 20.6 measured 0 of 281)
  direct        :   1 resolved / 140
  group         : 130 resolved / 141
```

A large improvement, and an uneven one: group conversation names now resolve almost completely (92 %)
while direct chats do not (0.7 %). The cause is measured rather than guessed — `contacts --json` returns
at most **500 records** (a higher `--limit` returns no more, and an over-large one returns nothing), and
of those 500 only one matches a direct conversation while all 141 groups do. The listing is
predominantly chatrooms and group members, not the user's one-to-one contacts. That is an exporter-side
limitation, recorded here for whoever picks it up; no workaround was attempted.

## Targeted speaker re-eval (q07–q09) on v3

The full 18-question run was deliberately skipped at the user's direction: the three
`speaker_attribution` questions are this phase's actual regression target, and Phase 20.8C already
measured that a sender-representation change moves no evidence at all on the other fifteen — so a
second 100-minute index build would buy a confirmation, not information.

### Representation, measured on the retrieved evidence

Every speaker label appearing in the retrieved evidence lines for the three questions, counted by kind:

| question | v2: `对方` / name / `成员X` | v3: `对方` / name / `成员X` |
| -------- | -------------------------- | -------------------------- |
| q07 | 360 / **0** / 0 | 19 / **347** / 5 |
| q08 | 227 / **0** / 0 | 175 / 50 / 0 |
| q09 | 157 / **0** / 0 | 21 / **117** / 5 |

**The group evidence names who spoke.** q07 went from 0 named senders in its evidence to 347 named
lines; q09 from 0 to 117. q08 moves less (50 of 225) because most of its evidence is one-to-one chat,
where the label is `对方` by design and this phase deliberately did not change it.

### Did the answers follow?

A structural check — not a correctness judgement, which stays with the human labeller:

| question | v2 | v3 |
| -------- | -- | -- |
| q07 | 0 named senders in evidence; answer says `对方` | **58 distinct named senders; the answer names one of them; it no longer says `对方`** |
| q08 | 0; answer says `对方` | 11 named; the answer still says `对方` |
| q09 | 0; answer says `对方` | **45 distinct named senders; the answer names one of them; it no longer says `对方`** |

q07 and q09 previously answered that the record only labels everyone `对方` — which was true, and was
the defect. They now name a person. **Whether it is the right person is not something this project
asserts from an automated check**; that is the human label, and it is still `null`.

### Signals and the confound

```
                        v2        v3
citation mismatches      0         0
invalid citations        0         0
attribution flags        0         0
silence claims           1         2      (q07 1->0, q08 0->2)
evidence changed       —          3 / 3
distinct conversations   9.0      10.0
mean latency ms        61 974    65 593
```

Every question's evidence changed, and per the brief that change is **not** attributed wholesale to
the sender fix: between the two exports the account also advanced (+5 292 messages, a coverage start 13
days earlier, one new shard). What *is* directly attributable is the representation itself — the same
kind of line that read `对方` in v2 reads a human name in v3, and there is no other candidate cause for
that. Separating the two effects event-by-event was not attempted.

---

# Phase 21A — Persistent retrieval index

Every process start re-embedded all 81 576 chunks with BGE and rebuilt FAISS — about **80 minutes**
before the first question on the real account. This phase makes a cold build persist and a warm start
load it, with **zero embedding of unchanged chunks**. The index is a pure accelerator: the JSON export
tree remains the source of truth, and deleting the cache leaves a fully working system that rebuilds.

## What already existed, and what was added

Quivr's own `Brain.save` / `Brain.load` were investigated first, because reinventing FAISS
serialization is the obvious way to get this wrong. They persist a FAISS store plus a
`BrainSerialized` config — **but they only support `OpenAIEmbeddings`** and raise
`"can't serialize embedder other than openai for now"` for anything else, which includes our
`HuggingFaceEmbeddings`. So the seam is not reusable as a whole.

What *is* reused is what Quivr itself calls underneath it: `FAISS.save_local` / `FAISS.load_local`,
which write `index.faiss` (vectors) and `index.pkl` (docstore + `index_to_docstore_id`) — the
vector↔Document↔metadata binding a citation depends on. Nothing about FAISS serialization was
reinvented.

**BM25 needs no persistence at all.** It is built lazily on first query (`rag/hybrid.py`) from
documents enumerated out of the FAISS docstore, so it is a pure function of what was already
persisted. It is **warm-rebuilt, not persisted**, and this record says so rather than claiming a
persistence that was not done. Its rebuild cost is measured below.

| artifact | how it was treated |
| -------- | ------------------ |
| dense vectors | **persisted** (`index.faiss`) |
| documents + metadata | **persisted** (`index.pkl`) |
| BM25 lexical index | **rebuilt** on first query from the loaded documents |
| account report | aggregates **persisted**; per-conversation rows deliberately not |

## One seam

`recall.build_account_session(...)` is the only place that decides whether to load or build. The CLI
(`recall.py`), the web app (`webapp/app.py`), and the acceptance runner (`real_eval.py`) all call it
and none of them inspects the cache; structural tests fail if a second decision point appears. The
warm branch constructs the `Brain` around the loaded vector store, which embeds nothing, and
`k` / `hybrid_pool` / `workflow` / `answer_prompt` / LLM settings are *not* part of the decision.

## The manifest, and what invalidates a cache

`manifest.json` holds versions, hex digests and counts only — no message text, no speaker id, no wxid,
no chatroom id, asserted by test. Validation runs cheapest-and-most-decisive first, with the source
tree last because it is the only check that walks the filesystem:

```
--rebuild-index  ->  directory exists  ->  manifest readable & strict  ->  cache format
  ->  document projection  ->  READY marker agrees  ->  artifacts present & non-empty
  ->  chunking  ->  embedding  ->  source  ->  HIT
```

| fingerprint | what it covers | what it deliberately ignores |
| ----------- | -------------- | ---------------------------- |
| **source** | relative path + size + `mtime_ns` for every file the import reads, sorted; a moved tree stays a hit | file contents (never read for this) |
| **chunking** | `max_gap`, `max_chars`, and a chunking-shape version | — |
| **projection** | an explicit `DOCUMENT_PROJECTION_VERSION` — the sender-label work of Phase 20.9 is exactly the kind of change it exists for | — |
| **embedding** | embedder class, model identity, `normalize_embeddings`, vector dimension | device, batch size |
| **not in any of them** | — | `k`, `hybrid_pool`, `workflow`, `answer_prompt`, LLM model, `temperature`, `max_output_tokens` |

An unstattable file **raises** rather than being skipped: an inventory with a hole cannot establish
that the source is unchanged, and a false cache hit is far worse than a false miss. A load that passes
inspection is still measured against the manifest (`verify_store`: chunk count, dimension, docstore
completeness), because the manifest is a claim and that is the check.

## Atomic build

`save_index` never writes into the live entry. It builds into `<leaf>.staging-<hex>`, verifies the
artifacts, writes `manifest.json`, writes `READY` **last**, then promotes: the old entry is renamed
aside (not deleted), the staging directory is renamed into place, and the old one is removed only
after success — with the old entry restored if the rename fails. An interrupted build therefore leaves
nothing loadable, and a leftover staging directory is ignored and cleaned up rather than mistaken for
an index.

## How "zero embedding" is proved

Not with a stopwatch. The tests pass an `Embeddings` subclass whose `embed_documents` **raises**
(`embed_query` is still allowed), at three levels: the seam, the CLI end to end, and the cold path as
a control. A warm start that embedded anything would fail rather than look fast. The reported warm
figure is `0 chunks embedded`, and it is that instrument's output, not an inference from elapsed time.

## Tests

88 new tests in `tests/test_index_cache.py`, covering cold/warm, retrieval and citation-metadata
parity, every invalidation dimension listed above (and the knobs that must *not* invalidate),
corruption and partial-build recovery, staging/promotion, privacy, and the structural claim that the
three front ends share one seam. Full suite: **686 passed**, from a 598 baseline.

## Cold build and warm start, measured on account_full_v3

The benchmark the phase exists for. The account is the same snapshot Phase 20.9 exported: 281
conversations, 1 630 452 messages, **81 576 chunks** — and the cache's own `chunk_count` is 81 576, so
nothing was dropped between the export and the index.

### Cold build

```
INDEX CACHE MISS (81576 chunks, 81576 chunks embedded, cache written)
INDEX CACHE TIMING (fingerprint 100 ms, import 52222 ms, embed 4251641 ms, save 743 ms, total 4304711 ms)
```

| stage | time | share |
| ----- | ---- | ----- |
| fingerprint the source tree | **0.10 s** | 0.002 % |
| import — parse 1 630 452 messages into 81 576 chunks | **52.2 s** | 1.2 % |
| **embed** | **4 251.6 s = 70.9 min** | **98.8 %** |
| save (FAISS + docstore + manifest + promote) | **0.74 s** | 0.02 % |
| **total** | **4 304.7 s = 71.7 min** | |

Embedding is essentially the entire cost, and everything this phase added — fingerprinting, the
manifest, the atomic save — together accounts for **less than one second**. A second cold build on the
same data, run while other work was competing for the CPU, took 73 minutes wall-clock, and an earlier
uncontended one about 42; the spread is machine load, not variance in the work.

**Cache on disk: 264 MB** (`index.faiss` 167 MB of vectors, `index.pkl` 108 MB of documents, plus the
manifest and READY marker).

### Warm start — a fresh Python process

```
INDEX CACHE HIT (81576 chunks, 0 chunks embedded, cache v1, projection v1, age 23 min, loaded in 2.77 s)
INDEX CACHE TIMING (fingerprint 315 ms, load 2772 ms, total 9018 ms)
```

| | cold | warm |
| --- | ---- | ---- |
| chunks embedded | 81 576 | **0** |
| chunks loaded | — | 81 576 |
| fingerprint | 0.10 s | 0.32 s |
| import / load | 52.2 s | **2.77 s** |
| embedding | 70.9 min | **0** |
| total to a usable session | 71.7 min | **3.1 s** (9.0 s including the query's own model load) |

**0 is not an elapsed-time inference.** The warm path is exercised in the tests with an embedder whose
`embed_documents` raises, at three levels — the seam, the CLI end to end, and the cold path as a
control — so a warm start that embedded anything would fail rather than merely look fast. The figure
above is that instrument's output on the real account.

### BM25 is rebuilt, not persisted

Measured on the real index: docstore load 0.96 s, `iter_documents` immediate (81 576 documents), and
**BM25 index construction plus its first query 11.95 s**. That is 0.3 % of the cold build and is paid
once per process on first query, not per query. It is recorded as rebuilt. Persisting it would save
about twelve seconds on a start that already takes three, so it was not done — if it ever becomes the
head of the profile, that is a one-artifact change.

## Retrieval parity — cold build vs warm load

Six frozen questions (q07, q09 and one from each of the single-fact, temporal, multi-source and
older-memory families), asked through the product's own retriever — the dense retriever over the FAISS
store, the `BM25Retriever` over the same documents, and the `HybridRRFRetriever` that fuses them with
the configured weights — reconstructed from whatever store the session produced. No model is called, so
generation noise is structurally absent rather than merely ignored.

```
documents in index : cold 81576   warm 81576

query     sources  same order  same set  metadata  content
q07            20        True      True      True     True
q09            20        True      True      True     True
q01            20        True      True      True     True
q04            20        True      True      True     True
q10            20        True      True      True     True
q17            20        True      True      True     True

EXACT PARITY (same sources, same order, same metadata): True
```

Not set equality: the **order** is identical, and so are each source's chunk id, conversation digest,
start time, content digest and length. A citation that pointed at the right evidence before points at
the same evidence now.

## Invalidation, as a real smoke rather than only as a test

On a bounded copy of a real export (never on the v3 snapshot):

| action | result |
| ------ | ------ |
| first run | `INDEX CACHE MISS` → built, 1222 chunks embedded, cache written |
| run again, unchanged | `INDEX CACHE HIT`, **0 chunks embedded** |
| **one message edited in one export file** | `INDEX CACHE INVALID (source export tree changed)` → rebuilt |
| **`k` 20 → 15** | `INDEX CACHE HIT`, **0 chunks embedded** |
| **`--answer-prompt` changed** | `INDEX CACHE HIT`, **0 chunks embedded** |
| **`--temperature` changed** | `INDEX CACHE HIT`, **0 chunks embedded** |

The two directions the phase cares about, both confirmed on real data: a changed corpus is never
mistaken for an unchanged one, and a changed query-time setting never costs seventy minutes of
embedding.

## Product integration

`recall.build_account_session` is the single seam, and a structural test fails if a second decision
point appears. All three front ends reach it and none owns cache logic:

| front end | how it reaches the seam |
| --------- | ----------------------- |
| `recall.py` | calls it; `--index-dir` and `--rebuild-index` |
| `webapp/app.py` | `RecallState.load` calls it; `web.py` exposes the same two flags |
| `real_eval.py` | calls it with `index_dir` / `rebuild_index` |

Verified on the real path: a CLI warm start and a web warm start both report `INDEX CACHE HIT … 0
chunks embedded` against the same entry, so a warm page is not quietly rebuilding what a warm CLI
already loaded.

## Phase 21A: status

| requirement | state |
| ----------- | ----- |
| a cold build persists | ✅ 264 MB, atomic promotion, READY written last |
| a new process warm-starts from it | ✅ **2.77 s load, 0 chunks embedded** (from 71.7 min) |
| cold vs warm retrieval parity | ✅ exact, including order and citation metadata |
| source / chunking / embedding / schema changes invalidate | ✅ tested per dimension, source change confirmed on real data |
| `k` / prompt / generation config do **not** invalidate | ✅ confirmed on real data |
| partial or corrupt caches are never loaded | ✅ staging never promoted; every failure is a stated reason |
| CLI / web / eval share one cache | ✅ one seam, structural test, warm web start verified |
| regression | ✅ 686 passed (598 before, +88) |
| real cold + warm benchmark | ✅ above |
| retrieval behaviour unchanged | ✅ parity; no change to BGE, FAISS parameters, hybrid search, BM25, RRF, `k`, pool, chunking, prompt, reranker or workflow |

## Known limitations

* **A changed source still means a full rebuild.** Incremental sync is Phase 21B by explicit decision;
  this phase makes the common case (nothing changed) fast and leaves the rare case correct.
* **BM25 is rebuilt on first query**, not persisted — 11.95 s, measured.
* **The fingerprint cost grows with the tree, not with the corpus.** It stats every file the import
  reads: 0.10 s cold, 0.32 s warm on 612 exports. It does not read file contents, which is what keeps
  it cheap — and why a `mtime_ns` touch is treated as a change rather than being verified away.
* **`allow_dangerous_deserialization=True` is a real trust boundary.** `FAISS.load_local` unpickles
  `index.pkl`, which holds the chat text. The index directory is therefore as private as the export
  tree and must stay local and git-ignored; nothing loads an index from a path this project did not
  write and validate.
* **The manifest records the vector width but cannot detect a live embedder that keeps the same
  configured identity while producing different vectors.** That case surfaces as FAISS's own assertion
  at load time — loud, never a silent wrong answer.
* **One entry per account, keyed by a digest of the account path**, so two accounts cannot share one,
  and a renamed account gets a fresh entry rather than a stale hit.

---

# Phase 21B — Incremental WeChat sync and incremental index update

Phase 21A made an unchanged source load in 3.1 s with zero embeddings. Any change to the source, though,
invalidated everything: a full re-import and **81 576 chunks re-embedded, 71.7 minutes**. This phase
makes a small delta cost a small amount of work, and proves the result is the same one a clean rebuild
would have produced.

## The repository

`CabbageCannon/quivr` was renamed to **`CabbageCannon/personal_recall`** with the GitHub rename API, so
it is the same repository — commits, branches, issues, history — under a new name, and the old URL still
redirects. The local `origin` was repointed at the new URL rather than relying on that redirect. Only
the one line naming the repository changed; Quivr attribution and the `quivr_core` package name are
untouched, because Personal Recall is still built on Quivr and that is not what the rename was about.

## What the exporter can actually do

Measured against the patched CLI (`213458c`), before designing anything:

| | result |
| --- | --- |
| `export <talker> json --from <date> --to <date>` | **works** — a windowed export of an active conversation returned 5 557 messages |
| `sessions --json` | carries `lastTimestamp` per conversation — a **free change signal**, no message body needed |
| exports must repoint the shard | yes: a conversation lives in one or more of the 12 `MSG*.db` files and one export reads one shard |
| an empty window | returns `EXPORT_FAILED 未找到消息` — **an error, not an empty set**, so "nothing new" must be handled as a no-op |
| `capabilities` advertises coverage fields (`mayHaveMore`, `oldestCreateTime`, …) | **the CLI does not emit them** (`{contract, count, format, path, success}` only) — nothing was built on them |
| a stable cursor | explicitly absent: `"incrementalRead": {"mode": "overlapping-time-window", "stableCursor": false}` |

**Chosen strategy: conversation-level refresh, driven by a message-level time window.** Detection is
cheap (`lastTimestamp` versus the checkpoint); only conversations that moved are exported, and only for
the window since the checkpoint plus a deliberate overlap — the exporter documents an overlapping window
with no stable cursor, so boundary duplicates are expected and the existing `serverId` dedupe is what
makes them harmless. History is never re-exported: the delta is merged into the existing export file.
This is **not** a row-level cursor, and the record says so rather than implying one.

## Architecture

```
WeChat MSG*.db
   -> discover shards (every run; 11 became 12 once already)
   -> sessions --json per shard -> lastTimestamp per conversation
   -> diff against the checkpoint -> affected conversations
   -> export only those windows, into staging
   -> merge into the export tree (union by serverId, ordered by createTime)
   -> re-render the WHOLE affected conversation (all its shards, re-merged, re-sessionized)
   -> diff chunks -> reuse unchanged vectors -> embed only changed/new
   -> FAISS.from_embeddings -> verify -> atomic promote -> checkpoint LAST
```

One seam, layered: `recall.build_account_session` to `sync_wechat.sync_account_source` to
`incremental_index.update_account_index`. The CLI, the web app and the acceptance runner all reach it;
a plain `--account` never touches the WeChat database, so offline corpora and CI keep working. Sync is
opt-in (`--sync-wechat --multi-dir ... --incremental-index`).

**A conversation is the unit of work, never a shard.** A conversation changed in `MSG0` may also live in
`MSG1`; rebuilding only the changed shard would sort fewer rows, let `serverId` dedupe see fewer copies,
and let the sender-labelling pass see less evidence — a *different* index, not a cheaper one. So all of
the conversation's files are reloaded and the whole conversation re-rendered, which also handles the two
effects that make "append the new message" wrong: a message inside `max_gap` **replaces the tail chunk**
rather than extending it, and a new `senderDisplay` can retroactively relabel a speaker's earlier
messages.

**Vector reuse.** A chunk is unchanged when its identity and exact projected text agree
(`chunk_id | conversation_id | sha256(page_content)`). Unchanged chunks take their vectors from the old
index (`reconstruct_n` in `index_to_docstore_id` order — verified on the real cache that position `i` is
the document's own vector) and `FAISS.from_embeddings` rebuilds the structure around them, which does
**not** call `embed_documents`. Only the rest are embedded. The reuse key deliberately excludes
`chunk_index` / `sessions_total`, which are account-global and recomputed.

## Real account: what actually happened

| run | outcome |
| --- | --- |
| **1st sync** (existing 21A index, never synced) | exported the deltas, then **refused to advance incrementally** — *"no checkpoint: this tree has never been synced, so the base of the delta is unknown"* — and did a full rebuild: **81 633 chunks, 43.2 min** |
| **2nd sync**, minutes later | **`INDEX CACHE INCREMENTAL (12 conversation(s) moved)` — 81 656 chunks, 81 629 reused, 27 embedded, total 38.0 s** (embed 1.07 s) |
| **3rd sync**, minutes later | **`INDEX CACHE HIT`, 0 chunks embedded, 1.02 s load** |

The first run is the phase's most useful negative result and it is **correct behaviour, not a bug**: an
index with no checkpoint has no known base, and advancing it incrementally would mean inventing one. The
cost is that adopting a Phase 21A index pays exactly one full rebuild before incremental begins —
recorded as a limitation below rather than hidden.

The corpus moved forward by **1 168 messages** (1 630 452 to 1 631 620) in the day between the Phase 20.9
export and the first sync.

**A real crash test, unplanned.** The first sync was killed mid-export by the environment. The staging
directory was left behind and the promoted generation was **completely untouched** — the next warm start
reported `INDEX CACHE HIT ... 0 chunks embedded` against the pre-sync index. That is Test Q happening
for real rather than in a fixture.

## Incremental == clean rebuild, on real data

The strongest available check was run rather than argued: a clean full rebuild of the *same* source into
a separate index directory (43 min), compared against the incrementally-updated index.

```
                incremental     full-rebuild
documents             81656            81656

query     sources  same order  same set  metadata  content
q07            20        True      True      True     True
q09            20        True      True      True     True
q01            20        True      True      True     True
q04            20        True      True      True     True
q10            20        True      True      True     True
q17            20        True      True      True     True

EXACT PARITY: True
```

An index assembled from 81 629 reused vectors and 27 new ones retrieves **identically** — same documents
in the same order, same citation metadata, same content — to one that embedded all 81 656.

## Performance

| | old behaviour (any source change) | incremental |
| --- | --- | --- |
| affected conversations | — | 12 |
| chunks | 81 576 re-embedded | 81 656 total, **81 629 reused, 27 embedded** |
| embedding | 70.9 min (Phase 21A) / 43.2 min (21B full fallback) | **1.07 s** |
| total | 71.7 min | **38.0 s** |
| no-op sync | 3.1 s (cache hit) | **1.02 s load, 0 embedded** |

**27 of 81 656 chunks — 0.03 % of the corpus — were embedded for that delta.**

## Classification and fail-closed

`NO_CHANGE` means cache hit. `SAFE_INCREMENTAL_CHANGE` means incremental. Everything else is
`UNSAFE_CHANGE`, which never advances: a missing manifest, an unusable or other-generation checkpoint,
an unwalkable tree, a file whose conversation cannot be established, a file that disappeared, a shard
with no export, a change attributable to no conversation. A structural change — chunking config,
embedding model or normalisation, projection version, cache format — forces a full rebuild, while `k`,
`hybrid_pool`, `workflow`, the prompt and every LLM setting deliberately do not.

## Tests

41 new tests in `tests/test_incremental_sync.py`, covering the no-op, append, tail-merge and new-chunk
cases, a new conversation, a conversation spanning two shards, a new shard, re-sent `serverId`s, an
edited message, retroactive display labels, the config matrix both ways, crash safety, corruption and
fail-closed refusals, and the golden incremental-vs-clean-rebuild comparison. Full suite **727 passed**,
from a 686 baseline.

## Known limitations

* **Adopting an un-synced index costs one full rebuild.** The first sync after a Phase 21A build has no
  checkpoint, so it cannot advance incrementally — correct, but the cost should be expected. A future
  phase could baseline a checkpoint from a verified `HIT`.
* **Sync granularity is conversation-level, not row-level.** The window bounds what is *fetched*; the
  unit of *re-import* is the conversation, because a conversation's chunks depend on its whole history
  and its whole-history sender labels.
* **An edited old message inside the window is picked up; one outside it is not.** The window is the
  contract, and the exporter offers no change feed to do better.
* **A deleted export file is refused, not handled.** It is classified `UNSAFE` and falls back to a full
  rebuild rather than silently dropping history.
* **`FAISS.load_local` still unpickles `index.pkl`.** The trust boundary is unchanged from Phase 21A:
  the index directory holds the chat text and must stay local and git-ignored.
* **BM25 is still rebuilt on first query** (about 12 s), from the updated docstore.
* **The patched exporter is a local, unpushed build** (`213458c`). Nothing here depends on it being
  published — the sync goes through the same `exporter` seam and its tests use a fake CLI — but the real
  acceptance above ran against that build.
* **PostgreSQL and pgvector have not been started**, by explicit decision.

---

# Phase 22A — PostgreSQL canonical memory store

**Scope.** Persist the memory layer Personal Recall already produces — `MemoryEvent`, `MemoryChunk`,
conversation and participant identity — as a structured, incrementally updatable, transactional
PostgreSQL store. Nothing here touches retrieval: no pgvector, no vector column, no metadata filter,
no query planner. FAISS, BM25, weighted RRF, `k`, the pool, chunking, prompts, sender labels and the
persistent index are exactly as Phase 21B left them, and a machine with no PostgreSQL still answers
questions.

## Status

| requirement | state |
| ----------- | ----- |
| schema migration creates the store from an empty database | ✅ `001_initial_memory_store.sql`, tracked, applied once, version-checked in both directions |
| `MemoryEvent` / `MemoryChunk` round-trip losslessly | ✅ §24/§25 cases incl. unicode, emoji, 2 KB bodies, replies, absence of identity |
| full-account counts match the normalized importer exactly | ✅ 281 / 1,632,344 / 81,672 — the same numbers the persistent index reports |
| person identity cannot collapse on a shared display name | ✅ identity-keyed digest; **169** shared names on the real account stay 169 separate people |
| a direct conversation's peer never merges across conversations | ✅ 65 `direct_peer` rows, one per conversation, never a shared `对方` |
| chunk → event evidence order fully recoverable | ✅ 1,632,344 links, ordinals unique, 0 events without a chunk |
| same snapshot bootstraps idempotently | ✅ refused unless `--rebuild-postgres`; rebuild reproduces identical rows |
| Phase 21B affected conversations sync transactionally | ✅ one transaction per sync; the store derives its own delta with 21B's classifier |
| incremental result equals a fresh bootstrap | ✅ bounded real data **and** real full-account data |
| crash rollback leaves no half-generation | ✅ 8 injection points × bootstrap and sync, all rolled back |
| PostgreSQL failure does not break retrieval | ✅ nothing on the retrieval path opens a connection |
| current 6-query retrieval exact parity | ✅ 6/6 vs the Phase 21B reference generation |
| regression | ✅ see below |
| real PostgreSQL integration tests | ✅ 73 tests against a live server, not mocks |
| full real account acceptance | ✅ below |

## Why PostgreSQL, and what it is not

Two layers of source of truth now exist and the order between them matters. The **raw** layer is the
WeChat databases and the WeFlow export tree — the history, never deleted, the thing everything is
rebuilt from. The **structured** layer is this store: the same memory, normalized once, written into
tables that can answer *what did this person say, in this conversation, between these dates, in this
order*.

The store is **derived**. Drop it and rebuild it from the exports at any time; nothing is written
only there. It is also **not a retrieval backend**: `build_account_session` still loads FAISS and BM25
and fuses them with RRF, and the store is not consulted for a single answer.

## Runtime

| | |
| --- | --- |
| server | PostgreSQL 16.15 (Debian), `docker-compose.postgres.yml`, loopback `127.0.0.1:55432` |
| storage | a **bind mount on `D:`** (`D:/personal_recall_pgdata`), not Docker's data root on `C:` |
| driver | `psycopg` 3.3.6 — one DB access path, no ORM, no Alembic |
| migrations | numbered `.sql` files beside `memory_store/schema.py`, recorded in `schema_migrations` |
| bulk path | `COPY … FROM STDIN`; never a row-at-a-time `INSERT` |

A container rather than an installed server: the store has to be creatable, droppable and reproducible
without touching a PostgreSQL the user may already run for something else. It publishes on loopback
only, installs no service and edits no `PATH`.

## Schema

`conversations` · `people` · `conversation_people` · `memory_events` · `memory_chunks` · `chunk_events`
· `source_inventory` · `memory_store_state` · `schema_migrations`.

`memory_events` keeps `speaker_id`, `speaker_display` and `sender_name` as three separate columns —
the same three facts `memory.events` separates, kept separate all the way down. `chunk_events` is a
real relation rather than an embedded id list, so the evidence behind a retrieval unit is a join.

**Timestamps are `TIMESTAMP WITHOUT TIME ZONE`.** The contract's chat times are naive local wall
clocks, and `TIMESTAMPTZ` was measured and rejected: the same stored value read back under
`Asia/Shanghai` moves by eight hours, because a naive value has no zone to be interpreted in. Under
`TIMESTAMP` the wall clock survives every session timezone; a test holds the store to that under
three of them. The exporter's own `createTime` is kept verbatim alongside, so the true instant is
never reconstructed from a wall clock.

## Identity

`person_id` is a digest of the source identity, never a surrogate key and never a display name, which
is why a name collision cannot merge two people — structurally, not by a rule someone could forget.

| | real account |
| --- | --- |
| conversations with a human-readable label | 131 / 281 |
| events with a trustworthy speaker | 1,350,570 / 1,632,344 (82.8%) |
| `group_member` people | 5,019 |
| `direct_peer` people | 65 |
| people in more than one conversation | 384 |
| **names shared by several people** | **169** — and 169 separate rows |

A direct conversation's peer is its own talker. Measured, not assumed: 131 of 140 direct talkers are
`wxid_*`, 134 of 140 state it again in the message field with **zero** contradictions, and 73 appear
verbatim as a group member's identity — so a direct chat and a group chat about the same person resolve
to the same row. The forbidden alternative this replaces is one shared `对方` person that would merge
every direct conversation into one stranger.

## Full-account acceptance

Built from the export tree as it stood after a real WeChat sync.

| | |
| --- | --- |
| conversations | 281 |
| events | 1,632,344 |
| chunks | 81,672 |
| people / memberships | 5,084 / 5,740 |
| inventory entries | 637 |
| event range | 2019-08-13 … 2026-09-23 |
| database size | 2.2 GiB after bootstrap |

`COPY` split: events **764 s**, chunk_events **524 s**, chunks **102 s**, people+memberships+state
**7 s** — **1,416,740 ms total**, 25.7 min wall clock including the export parse.

Referential integrity, checked on the real store rather than asserted: 0 links to missing events,
0 links to missing chunks, **0 events with no chunk**, ordinals unique within every chunk, 0 orphan
people, 0 character-count mismatches, 0 chunk-metadata mismatches.

## Incremental update

The unit is the **conversation**, never the message. A new message can replace the tail chunk it
joined; a member gaining a display name renames messages the delta never touched; a session boundary
that moves renumbers the chunks after it. Appending what is new is therefore not an update — it is a
different store.

The store keeps **its own record of the export files its generation was built from** (`source_inventory`),
so it computes its delta with Phase 21B's own classifier, from its own base, without borrowing the
sync checkpoint. The checkpoint describes one generation and is advanced by whichever consumer runs
first; an index run would overwrite the store's base and vice versa. Independent on real data:
after a real WeChat sync moved 11 conversations, `--sync-postgres` reported **the same 11
conversations** the index had.

| real full-account incremental | |
| --- | --- |
| conversations rewritten | 1 (from a real mtime move: 127,411 events, 4,551 chunks) |
| delete phase | **3.6 s** |
| events `COPY` | 42.9 s |
| chunk_events `COPY` | 20.4 s |
| **transaction total** | **73.5 s**, generation 1 → 2 |
| no-op sync | `source is identical to the stored generation; nothing to do` |

Idempotence on real data: touching a real export and syncing **twice** reproduced byte-identical rows
for that conversation both times, with the store's counts unchanged and the generation advancing.
Re-rendering a real 127k-event conversation is deterministic.

Bounded real acceptance (§32) over four real conversation shapes — direct, small group, multi-shard
group, 8+ member group — with a synthetic delta: 34,100 events, 2,295 chunks, **0 crossed-conversation
chunks**, 2,295/2,295 chunks with complete ordered evidence, and `incremental == fresh bootstrap` row
for row across all six tables.

## Structured query smoke

Read-only, on the full store, counts and milliseconds only — no text, no name, no identifier.

| query | rows | ms |
| --- | --- | --- |
| conversation → events | 526,966 | 311 |
| time range → events | 908,572 | 2,802 |
| **person → events** | 93,904 | **15** |
| conversation + time → events | 526,966 | 64 |
| chunk → evidence, in order | 37 | 32 |

The time-range query is a full scan because no filter bounds it; that is a property of the query, not
of the store. The indexed paths — person, conversation, chunk evidence — are the ones a metadata
filter would use.

## Two defects real data found that fixtures did not

**`chunk_events.event_id` had no index.** PostgreSQL does not create one for a foreign key, and the
primary key is `(chunk_id, event_id)`, which cannot answer *which links cite this event*. Replacing
11 conversations made the cascade from `memory_events` sequentially scan the whole 1.6M-row table
once per deleted event: **25 minutes and still running** on the real account. With the index the same
operation is **3.6 s**. A test now asserts that every foreign key has an index on its referencing
column — the general form of what went wrong.

**A container's `/dev/shm` defaults to 64 MB.** A whole-account integrity check failed with
`could not resize shared memory segment … No space left on device` — a disk-space error for something
that is not a disk. `shm_size: 1gb`.

## Crash safety

One transaction per write. Failure injected at eight named points inside a bootstrap and inside a
sync, plus a real one nobody arranged: a sync over the full account was killed mid-delete, and the
store came back at its previous generation with its counts intact. After the kill PostgreSQL **kept
executing the abandoned statement** for 25 minutes, holding locks that blocked a schema drop — a
reminder that a client dying does not stop a server-side delete.

## Tests

| | before | after |
| --- | --- | --- |
| whole suite | 727 | **821** |
| ordinary suite (`pytest -m "not postgres"`) | 727 | **747** |
| PostgreSQL integration (`pytest -m postgres`) | — | **74** |

The 74 run against a **live server**, each in a schema it creates and drops, and the rest of the
suite runs on a machine with no PostgreSQL at all — the module *skips* rather than fails when the
server is unreachable, because a suite that fails because a container is down reports nothing about
the code. The projection tests are deliberately unmarked: the properties they check must hold
everywhere, not only where a container happens to be running. Splitting it this way is what makes
"the store works" and "the product works without a store" two separate claims rather than one.

One gap worth recording: the three failures this phase's own tests caught were found by running the
**whole** suite, not by running the two new files. Twenty projection tests and seventy-four database
tests were green while a change to `TABLE_COLUMNS` had broken an assertion in a file neither command
touched.

## Privacy

The database holds real chat text and real identities, and is a local private store, like the index
cache. What must never leave it is narrower and enforced: connection strings come only from
`PERSONAL_RECALL_DATABASE_URL` (`.env`, git-ignored; `.env.example` holds placeholders), the loggable
form of a target carries host/port/database and never credentials, an unparsable URL is reported as
unparsable rather than echoed, every operator command prints counts and fingerprints only, and
`PROJECT_STATUS.md` carries aggregates. The tests use synthetic identities shaped like real ones —
`wxid_synthetic_*`, `NNNN@chatroom` — and a scan confirmed no real conversation id, wxid or chatroom
appears in any file this phase adds.

## Retrieval unchanged

The frozen six-query probe, run after Phase 22A against the same index generation Phase 21B finished
on (81,656 documents): **6/6 exact parity** — same order, same set, same citation metadata, same
content digests. Phase 22A changed no retrieval code, and the measurement says so rather than the
claim.
