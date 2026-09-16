# Personal Recall Engine — Project Status

Living decision log. Facts only from real eval output; no estimated numbers.
Branch: `personal-recall` · Base: `CabbageCannon/quivr`

---

## Phase map

| Phase | Scope | Status |
|---|---|---|
| 0 | Small-corpus baseline (unmodified Quivr Dense RAG) | ✅ done |
| 0.5 | Recall Stress Corpus + Stress Baseline + dataset audit | ✅ done — baseline visibly fails, failure attributed to retrieval |
| 1 | MemoryEvent / Source Adapter + conversation-aware chunking | ⏸ gated on 0.5 failure analysis |
| 2 | Retrieval trace / Evidence representation on structured memory | ⏸ |
| 3 | Hybrid retrieval (BM25 + dense + RRF, reranker only if needed) | ⏸ |
| 4 | Temporal retrieval | ⏸ |
| 5 | Entity-aware recall (Person / Alias) | ⏸ |
| 6 | Multi-evidence / state evolution | ⏸ |
| 7 | Grounded generation (EvidenceItem, citation, abstention) | ⏸ |
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
