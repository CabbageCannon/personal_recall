# Adversarial audit — `stress_chats.txt` / `stress_queries.json`

Auditor: independent (did not author any artifact). Date of audit: 2026-09-16.
Scope: corpus `data/stress_chats.txt` (1216 msgs / 100 episodes), queries `data/stress_queries.json`
(36), contract `data/STRESS_CORPUS_SPEC.md`, machine reports `stress_validation.json`,
`stress_dense_probe.json`. **No file was modified except this report.** Throwaway scripts were
written under `%TEMP%\stress_audit\`.

**Line-number convention.** `L###` = physical line number in `data/stress_chats.txt` (1-based).
**Chunk convention.** `c###` = chunk index of the *frozen baseline chunker*
(`quivr_core … recursive_character_splitter(chunk_size=400, chunk_overlap=100)`), re-executed
during this audit. The real chunker yields **165 chunks** (164 × 400 chars + 1 × 268);
`stress_validation.json` claims **169** — see R11.

---

## 1. Verdict summary

| Metric | Count |
|---|---|
| Queries audited | 36 (34 answerable + 2 unanswerable) |
| `expected_answer` fully TRUE and supported | **31** |
| `expected_answer` WRONG / materially over-specified (must change) | **3** — s007, s016, s019 |
| `expected_answer` PARTLY supported / terminology conflict (should change) | **2** — s026, s022 |
| `expected_answer` TRUE but ambiguous against a corpus line (low) | **1** — s030 |
| Evidence lines not found verbatim in the corpus | **0** (108/108 grounded) |
| Category mislabels (§7 rules) | **1 clear** (s003) + 2 soft (s009, s030) |
| §7.3.1 violation caused by the *corpus*, not the query | 1 (s020 — a single line states the final answer) |
| **Hard queries s013–s036 = 24** → HARD | **7** — s013, s016, s019, s022, s030, s031, s034 |
| → BORDERLINE | **6** — s014, s015, s021, s027, s029, s033 |
| → EASY (answerable from ONE chunk or one line) | **9** — s017, s018, s020, s023, s024, s025, s026, s028, s032 |
| → n/a (unanswerable, verified absent) | 2 — s035, s036 |
| s035 / s036 genuinely unanswerable | **yes / yes** |
| Name traps | 小王↔小汪 **works**; 王哥=小王 **defused** (self-decoding, 1 mention); 阿伟=室友 **defused** (self-decoding, 1 mention) |
| Meta-text leaks (评测/query/evidence/gold/category/…) | **0** |

**Headline falsifications**

1. **s007 + s019 gold is wrong on the single number the query is built around.** Both golds say
   the internship applications started in **May**. The corpus says otherwise: L1091 (2026-04-15,
   "你实习投了没" → "还没"), L1104 (2026-04-20, "我这周投几家试试"), L1150 (2026-05-10,
   "我投出去快两周了") ⇒ first submissions ≈ **2026-04-26, i.e. April**. This contradicts
   spec §3.5 ("2026-05-17 实际：5 月才开始投"). No corpus line states "5 月才开始投";
   the only May-ish line (L1163) is 学长's *admonition* ("别拖到五月才动手"), not a fact.
   A model answering correctly from the corpus gets marked wrong.
2. **s016's gold misstates the start of the timeline it asks about.** Gold: "2025 年 2 月他刚开始
   去健身房". Corpus: 小汪 was already gym-going well before — L26 (2024-02-10 "聊聊健身的事"),
   **L119 (2024-06-15 "我最近天天泡健身房")**, L316 (2024-12-28 "回来一起去健身房练两天").
   Also "中间断了" is *inferred*, never stated, and the 2025-02→2025-06 "满一个月" pair already
   implies an unmentioned earlier lapse.
3. **9 of the 24 "hard" queries are answerable from a single 400-char chunk** (list in §3),
   because the corpus contains explicit one-line final-state statements (L1139, L1003, L1297,
   L1181, L910, L1288, L1292) — a direct violation of spec §5.7 ("Do not let any single line state
   the final answer… the state must have to be assembled from ≥2 lines").
4. **Both alias traps are self-decoding.** `王哥` occurs exactly once (L370) and the line itself
   says "王哥（就是小王）"; `阿伟` occurs exactly once (L1071) as "阿伟（我室友）".
   The disambiguation queries s025/s029 therefore test nothing.
5. The 2026-08-20 episode (L1287–L1300, ≈2 chunks) is a **recap episode** that states the FINAL
   state of the frontend, backend, database, search and remote-access lines in 14 messages.

---

## 2. All 36 queries — expected-answer truth table

Proof column = corpus line(s) that carry the fact. Verdicts: TRUE / TRUE(minor) / PARTLY / WRONG.

| id | category | verdict | proof lines | note |
|---|---|---|---|---|
| s001 | exact_fact | TRUE | L228–L229 ("第三个") | single line L229 |
| s002 | exact_fact | TRUE | L818 (FIFO、LRU、Clock 三个) | single line |
| s003 | exact_fact | TRUE | L1232/L1239 (约的两家) + L1246 (一家挂了，另一家等二面) + L1303 | 2-hop across 2026-06-24 → 2026-07-05; category label too weak (see §4) |
| s004 | exact_keyword | TRUE | L947 (SearXNG) + L944/L945 (月初/聚合搜索) | — |
| s005 | exact_keyword | TRUE | L463/L465 (Flex, 生成扫描器) | — |
| s006 | semantic_recall | TRUE | L187 (Vercel/国内不稳/算了) + L189 (白屏) + L194 | — |
| s007 | semantic_recall | **WRONG** | L1042 ✓, L1164 ✓, L1169 ✓ — but **L1091, L1104, L1150 falsify "拖到五月才投"** | must fix |
| s008 | semantic_recall | TRUE | L765 (学校给的 MySQL) + L1116 (学院统一配的) + L766 ("省得你自己去挑") | — |
| s009 | person_recall | TRUE | L1025/L1027 | question time-frame ambiguous vs the 2025-12-20/2025-12-22 sessions (L859, L871) |
| s010 | person_recall | TRUE | L806 (检索或多模态) + L772/L792/L912/L936/L1014/L986 (题目一直没给) | — |
| s011 | time_recall | TRUE | L86/L91 (2024-05 长沙) + L454 ("去年五一去的长沙") | — |
| s012 | time_recall | TRUE | L1188 (六月十八号上午) + L1197 + L1216 (当天已答辩) | 小汪's 二十号 (L1196) is a real distractor |
| s013 | temporal_state_change | TRUE | L515 Tavily / L728 Brave / L947 SearXNG / L1181 back to Tavily + L1176/L1178 (why) | — |
| s014 | temporal_state_change | TRUE | L187 Vercel / L618 Pages / L641 / L1288 | "为什么定在现在这家" is only loosely answered by the corpus (L647 "比之前那个快不少", L1288 "慢是慢但没再动") |
| s015 | temporal_state_change | TRUE | L33 Render / L834 冷启动十几秒 / L839 迁到 Railway / L1130 + L849 (一秒内) | — |
| s016 | temporal_state_change | **PARTLY→WRONG** | L408/L579/L967/L969 ✓ but start is falsified by **L119, L26, L316**; "中间断了" inferred only | must fix |
| s017 | plan_vs_actual | TRUE | L272/L273 (plan) + L371/L372 (2025-01) + L695/L697 (recap) | — |
| s018 | plan_vs_actual | TRUE | L689 (plan) + L704 (候补) + L714/L716 (actual) | gold's "只抢到候补" is an over-reading: L704 says he was *on* the waitlist, the corpus never says the ticket came from it |
| s019 | plan_vs_actual | **WRONG** | L1042/L1043 ✓ + L1169 ✓ — but "实际拖到 5 月才开始投，晚了大约一个月" falsified by L1091/L1104/L1150 | must fix |
| s020 | latest_state | TRUE | L553 + L577 + L1003 + L1139 (Railway; Neon 一直没上) | gold correct; but corpus gives a one-line answer (L1139) — §7.3.1 violated by corpus |
| s021 | latest_state | TRUE | L431 Tailscale / L1075 / L1212 / L1297 | — |
| s022 | latest_state | TRUE(minor) | L778 (早不去了) / L902 (卡快过期) / L972 / L974 (跑操场) | premise "之前办的那张卡" rests on L902 alone (the corpus never says 我 办过卡); "偶尔跑跑操场" over-specifies L974 ("还在跑操场") |
| s023 | earliest_state | TRUE | L34 (小王: Supabase) + L37 (今晚注册) + L161 + L169 | — |
| s024 | earliest_state | TRUE | L33 (后端我放 Render) + L258 + L618 | — |
| s025 | entity_disambiguation | TRUE | L370 ("王哥（就是小王）") + L34 + L765 | answer is a single self-decoding line |
| s026 | entity_disambiguation | PARTLY | L641/L643 support "不是一回事" | gold glosses Tunnel as "让设备之间互相访问", but L435 says the opposite ("一个是把服务挂到公网上给所有人看，一个是你自己的设备互相连") — terminology conflict |
| s027 | entity_disambiguation | TRUE | L127 (同学A: Neon) + L34 (小王: Supabase) + L1140/L1141 | caveat: L261–L262 reads as if 小王 had recommended the Neon-ish 方案 |
| s028 | negative_evidence | TRUE | L245 / L1001 / L1087 (+L1086) | — |
| s029 | negative_evidence | TRUE | L1071 / L1075–L1076 / L1212 (+L1299) | — |
| s030 | multi_evidence | TRUE(low) | L169 / L227 / L245 / L258 | "收尾" collides with L377 (2025-01-20 "我还在收尾", a different course project); time frame of the question is fuzzy |
| s031 | multi_evidence | TRUE | L765 / L897 / L1118 / L1313 | — |
| s032 | implicit_reference | TRUE | L667 (SQLite, 不上云) + L1279 (+L1143) | — |
| s033 | implicit_reference | TRUE | L806/L807/L814 + L1014 | — |
| s034 | cross_time | TRUE | L34 / L358 / L553 / L1139 | — |
| s035 | unanswerable | ✓ correct | none (see §5) | — |
| s036 | unanswerable | ✓ correct | none (see §5) | — |

All 108 evidence lines are byte-identical copies of corpus lines (verified independently, not
only via the validator).

---

## 3. Difficulty falsification — all 24 hard queries (s013–s036)

Rubric used: **D** = minimum number of distinct 400-char chunks needed to produce the gold answer.
HARD = D≥3 across ≥3 episodes and no single line states the outcome; BORDERLINE = D=2 (or D=1 for
the literal sub-questions while the gold demands more); EASY = D=1 (one chunk contains the answer).

| id | D | best single-chunk candidate | distractor present? | verdict | rewrite if EASY/BORDERLINE |
|---|---|---|---|---|---|
| s013 | 4 | c064 (L509–519) Tavily only; c090 (L722–733) Brave only; c117 (L942–953) SearXNG only; c148 (L1178–1188) "搜索还是回到 Tavily 了" | yes — L1176/L1183 (2025-05-23 "养不动…换回来") + L1295 (2026-08-20 "用回最早接的那个") duplicate the outcome in other years | **HARD** | — |
| s014 | 2 | **c077 (L614–624)** contains Vercel ("试过，去年就试了，国内访问不太稳") + Pages ("前端我准备改 Cloudflare Pages") + the reason ("之前那个前端在国内打开要等好几秒") | yes — Vercel vs Pages vs Tunnel (2025-03 L433, 2025-08 L640) | **BORDERLINE** | make it cross-chunk: "前端托管从头换到尾一共隔了多久？换完以后国内访问的问题彻底解决了吗？" (needs L187 2024-09 + L618 2025-07 + L654/L653 2025-08 "偶尔不稳") |
| s015 | 2 | c104 (L836–846) "后端也从 Render 迁到 Railway 了"; c103 (L827–838) "冷启动太慢，第一个请求要等十几秒". Single-line spoiler: **L910 "后端也挪了一次，冷启动那个毛病算是解决了"** | yes — L33/L618 Render mentions in 2024/2025 vs L1290 "搬到另一家" | **BORDERLINE** | "后端那次搬家，前后第一个请求的等待时间差多少？" (needs L834 十几秒 + L849 一秒内) |
| s016 | 3–4 | c051 (2025-02 体验卡), c072 (2025-06 满一个月), c119/c120 (2026-02 胖了+年卡) | yes — 我's own gym lines (L778/L902/L972) and 小汪's 2024 lines | **HARD** | fix the gold (see §1) — but note the corpus itself supplies a 2024 state the gold ignores |
| s017 | **1** | **c086 (L689–699)**: L695 "说去深圳结果没定下来，最后哪也没去" + L697 "后来课程作业一堆，就黄了" | yes — L281 ("我这边随时能走"), L295/L298 (张三's own 元旦 talk) | **EASY** | "小王提议的那个元旦行程，我是当场答应的、当场拒绝的，还是拖到最后也没给他准话？" (needs L273 "先别定" + L274 "月底前给我个准话" + L371 "后来不是没定嘛" — the recap line does not answer it) |
| s018 | **1** | **c088 (L705–717)**: L714 "重庆回来了，人真的多" + L716 "洪崖洞…挤得走不动，拍照都得排队" (+L710 假期一开始就走) | yes — 2024-10 国庆 "哪也没去" (L215), 2025-04 五一 没出去 (L476) | **EASY** | "国庆去重庆这趟，出发前是先订的住宿还是先买到的票？票最后是怎么解决的？" (needs L689 票还没买 + L704 候补 + L708 先订住宿) |
| s019 | 3 | c129 (plan) / c137–c138 (2026-04-20 "这周投几家试试") / c145–c146 (actual) | yes — the baseline month itself is unstable: L885 "明年三月要投" vs L917/L1019/L1042 (四月/别等到五月) | **HARD** (assembly) but **gold wrong** | fix gold to "4 月下旬才开始投（4/20 说这周投几家，5/10 说投出去快两周了），比学长说的四月稍晚"; also fix the question's "跟当初说好的时间" (two candidate baselines: 三月 L885, 四月 L917/L1042) |
| s020 | **1** | **c142 (L1132–1142)**: L1139 "正式的项目都在 Railway 上，Neon 就一直没上". Second path: c123/c124 L1003 "后来从 Supabase 迁走那个，一直用到现在" | yes — Neon reappears L1086 (2026-04), SQLite L667, MySQL L765 | **EASY** (content) — retrieval still MISSES (probe coverage 0.00) | "从最早课程项目那个库，到现在正式这套，中间一共换过几轮？现在这套是哪个月定下来的？" (needs L34/L358/L553 + L529) |
| s021 | 2 | c163 (L1293–1304) L1297 "还是最早装的那个，四月想换过一次，没弄成"; c152 (L1208–1219) L1212 "配了一晚上没配通，还是回去用 Tailscale 了" | yes — L1299 "他自己换的那套组网，跟我不一样" (室友's WireGuard) | **BORDERLINE** | "四月那次我想换的是哪套？卡在哪一步？" (needs c133 what + c152 how it failed) |
| s022 | 3 | three separate states: c096 (L778), c111 (L902), c120 (L972/L974) — no single chunk holds two | yes — 小汪's far more prominent gym thread (L408/L579/L969) | **HARD** | (gold: drop "偶尔") |
| s023 | **1** | **c005 (L32–41)**: L34 "可以试试 Supabase" + L37 "那我今晚注册一个，先建两张表试试" | yes — Neon/Railway/SQLite/MySQL later | **EASY** | "我最先用上的那个库，和同学A后来推荐的那个，哪个在前、中间隔了多久？" (needs L34 2024-03 + L127 2024-07 + arithmetic) |
| s024 | **1** | **c005 (L32–41)**: L33 "后端我放 Render 了" (the project's first episode) | yes — Render vs Railway/Vercel | **EASY** | "最开始后端放的那个平台，一直用到哪个月才换掉？" (needs L33 + L618 + L839) |
| s025 | **1** | **c046 (L365–375)**: L370 "王哥（就是小王）" | yes — 小汪 (L765) is a real look-alike distractor | **EASY** | "跟我聊数据库和部署的那个同学，和弄毕设、数据库用学校 MySQL 的那个，会不会其实是同一个人？" (drops the self-decoding alias). Real remedy is corpus-side: delete "（就是小王）" from L370 |
| s026 | **1** | **c080 (L638–648)**: L641 "Pages，这次不是内网穿透" + L643 "一个是部署网页，一个是机器之间互相访问" + L644 (NAS) | yes — Tunnel appears 2025-03 (L433) and 2025-08 (L640) | **EASY** | "我前端一开始试的那家、后来换的这家，和室友说的 Cloudflare Tunnel，哪两个是一类的？" (needs c024 Vercel 2024-09 + c054 distinction 2025-03 + c080 2025-08) |
| s027 | 2 | c017 (L125–134) L127 "同学A: 那你看看 Neon，也是 PostgreSQL"; c142/c143 L1141 "不是你，是另一个同学" | yes — 小王's own false memory L1140 + L261 | **BORDERLINE** | "小王为什么会把这件事记成是他推的？我当时是怎么纠正的？" (forces L1140+L1141 plus L34/L127) |
| s028 | **1** | **c135 (L1081–1090)**: L1082 "那个 PG 的…免费层会睡，冷启动很慢" + L1083 "就是当时嫌这个才放下的" + L1085–L1087 "要改的地方太多，连接串、迁移脚本、备份都得跟着动" + L1086 "最后还是没换" | yes — L157 "Neon 连上了" reads like a positive answer | **EASY** | "当年试那个库的时候我具体做到哪一步就收手了？账号后来还留着吗？" (needs c021 2024-08-04 + c031 2024-11-19) |
| s029 | 2 | c133 L1071 (阿伟→WireGuard) for the alias; c134 L1075–L1076 / c152 L1212 / c163 L1299 for the refusal | yes — L1203 "在试你上次说的那套组网方案" reads like a switch | **BORDERLINE** | "室友那套新方案我到底试过没有？试的结果是什么？我现在用的是哪一套？" (needs c133 + c152 + c163 ⇒ D=3) |
| s030 | 4 | c022 (Supabase env), c029 (答辩), c031 (Neon pause), c033 (Render) — no single chunk | yes — 2025 DB/deploy lines | **HARD** | — |
| s031 | 4 | c094 MySQL / c111 论文 / c139 不用上线 / c165 交完就完事 | yes — my own "上线" thread (L1188/L1194) and L1198 小汪 没部署 | **HARD** | — |
| s032 | **1** | **c083 (L663–674)**: L667 "本地原型我就直接用 SQLite，不上云" + L671 "正式那套还在线上跑着呢" | yes — L1279 "本地用文件型的那个" in 2026 | **EASY** | "我现在本地用的那个存储，和 2025 年那会儿本地原型用的是不是同一个？" (needs c083 + c160/c161 and the identity judgement) |
| s033 | 2 | c100 (L803–814) L806/L807/L814 ("检索我比较熟，多模态我基本没碰过" / "先把检索这条线走通"); c125 L1014 for the 2026-03 restatement | yes — 多模态 is offered then dropped (L927 "多模态也可以一并看看") | **BORDERLINE** | "导师给的两个方向里被我排除的是哪一个、为什么？到 2026 年 3 月我有没有回头动过它？" |
| s034 | 4 | c005 (Supabase), c044/c045 (换回), c069 (Railway), c142 (final) | yes — L135 2026-04 Neon reconsidered; L1143 "本地写着玩的时候用过轻量的那个" | **HARD** | — |
| s035 | — | unanswerable (verified §5) | — | **n/a** | — |
| s036 | — | unanswerable (verified §5) | — | **n/a** | — |

Additional §7.3 checks for the hard set:

* `plan_vs_actual` plan/actual pairs are both present and ≥2 weeks apart in all three cases:
  s017 L272-273 (2024-12-07) → L371 (2025-01-14) ✓ different verbs;
  s018 L689 (2025-09-22) → L714 (2025-10-06) = **exactly 14 days** (borderline pass);
  s019 L1042-1043 (2026-03-18) → L1169 (2026-05-17) ✓ different verbs.
* `negative_evidence` positive-looking bait exists: s028 → L157 "Neon 连上了…"；s029 → L1203
  "在试你上次说的那套组网方案".
* `implicit_reference` / `latest_state` / `negative_evidence` questions contain **no latin token**
  (checked all 7: s020, s021, s022, s028, s029, s032, s033 — zero `[A-Za-z0-9]` matches).
  §7.3.1/§7.3.2 satisfied on the query side.
* §7.3.8 lexical shortcut: max question↔single-line bigram overlap over all 36 queries is
  **0.375** (s034 vs L1081) — no query exceeds the 0.5 threshold. Pass.
* §7.3.5 `earliest_state`: ≥3 later mentions exist for both DB (Neon/Railway/SQLite/MySQL) and
  backend (Render 2024-03 → Railway 2025-12) ✓ — but see the EASY verdicts, because the *first*
  mention is also the answer.

---

## 4. Unanswerable verification (s035, s036)

**s035 — "导师最后给我定的毕设题目叫什么名字？"** → genuinely absent. Evidence:
* Every one of 导师's 16 lines (L806, L808, L810, L812, L814, L925, L927, L978, L980, L982, L984,
  L986, L1188, L1190, L1192, L1194) gives a *direction* or process advice — never a title or name.
  L986 is explicit: "题目的事我还是那个意思，方向定了再定名".
* 我's own lines repeatedly state the title is undecided: L772 "题目还没定下来", L792 "还没给我
  具体题目", L912 "毕设到现在题目还没定", L936 "具体题目等他拍板", L1014 "导师那边题目还没定".
* No latin token anywhere names a topic, and the only candidate nouns are the two *directions*
  (检索 / 多模态, L806, L927) — a model that answers "检索" is wrong, but the question is not so
  vague that unrelated lines look like an answer. Reverse risk: **low**.

**s036 — "小汪说等我过去要请我吃的那家店，叫什么名字？"** → genuinely absent, premise present.
* Premise holds: L26 "开学前我请你吃饭", L975 "顺便吃个饭", L1200 "过了我请你吃饭",
  L1270 "那你八月来找我玩啊，我请你吃烧烤".
* No restaurant/store name exists anywhere in the corpus. Every 店 reference is unnamed:
  L99 "那家网红店", L90 "我看了个青旅", L720 "有一家排了两个小时", L1228 "就学校后门那家".
* Reverse risk: **moderate and healthy** — the food *type* is present (烧烤 L1270), and L589 is an
  inverted-direction mention ("我请你吃那家烤肉", 我 treating 小汪). A model can answer "那家烤肉" or
  "烧烤", which is exactly the wrong-answer behaviour the probe should catch. The question is not
  vague enough for unrelated lines to be mistaken for an answer.

---

## 5. Corpus quality risks

**R1 — Single-line final states defeat spec §5.7 (highest corpus-side risk).**
Lines that state a FINAL state outright: `L1139` "正式的项目都在 Railway 上，Neon 就一直没上";
`L1003` "后来从 Supabase 迁走那个，一直用到现在"; `L1292` "真正跑正式项目的一直是现在这个";
`L1297` "还是最早装的那个，四月想换过一次，没弄成"; `L1181` "搜索还是回到 Tavily 了";
`L910` "后端也挪了一次，冷启动那个毛病算是解决了"; `L1288` "前端最早那家国内访问不行，后来换了个
托管的，慢是慢但没再动". §5.7 requires the state to be assembled from ≥2 lines — the corpus
provides 7 escape hatches.

**R2 — The 2026-08-20 episode (L1287–L1300) is a compressed recap.** c162+c163 answer the *shape*
of four separate lines at once ("前端最早那家…换了个托管的…没再动" / "后端…搬到另一家" / "数据库…
真正跑正式项目的一直是现在这个" / "搜索…用回最早接的那个" / "远程…还是最早装的那个"). Names are
withheld, so the queries survive, but any model that retrieves these two chunks learns the answer
topology for s013/s014/s015/s020/s021/s034 in one shot.

**R3 — Alias traps are self-decoding (spec §2, §5.5–5.6, §7.3.3).**
`王哥` = 1 occurrence (L370) and the line itself says "王哥（就是小王）".
`阿伟` = 1 occurrence (L1071) and the line itself says "阿伟（我室友）".
Consequently s025 and s029 are single-line lookups (see §3), and two of the three
`entity_disambiguation` queries test nothing.
The **小王 / 小汪** trap does work: 小王 145 mentions vs 小汪 68; the topic partition is clean —
no 小王 line mentions 毕设/MySQL, and the only 小汪 line containing 部署 (L1198) is about his own
毕设 demo. Both names exceed the ≥6 requirement.

**R4 — Corpus contradicts its own spec on the internship month (§3.5).** See §1 item 1.
Secondary defect in the same thread: the *plan* month is itself unstable — L885 (2025-12-25)
"明年**三月**要投实习" vs L917 (2025-12-31) "学长说**四月**就能开始投了" and L1042
(2026-03-18) "四月开始就可以陆续投了". s019 asks "跟当初说好的时间差了多久", so the
comparison baseline is ambiguous before the answer month is even considered.

**R5 — s016's timeline is contradicted inside the corpus.** L26 (2024-02), **L119 (2024-06-15
"我最近天天泡健身房")**, L316 (2024-12) precede the gold's claimed start (2025-02).

**R6 — Tunnel terminology conflict.** L435 (室友): "一个是把服务挂到公网上给所有人看，一个是你
自己的设备互相连" (Tunnel = public exposure) vs L643 (我): "一个是部署网页，一个是机器之间互相
访问" (Tunnel = machine-to-machine). Two mutually exclusive definitions of the same contrast.
Also note §3.2 frames the 2025-03 Tunnel line as the NAS/intranet story, but L433's Tunnel
suggestion is actually about *my* lab machine.

**R7 — L261–L262 (2024-11-25) muddy the Neon provenance.** 小王: "我上次跟你说的那个方案你试了
没" → 我: "暑假试过一下，后来觉得麻烦就放下了". The only thing "试过一下然后放下" is Neon, so this
line reads as 小王 having recommended Neon — contradicting §3.1 and L1141. It functions as an
intentional distractor (echoed by 小王's own confusion at L1140) but it makes "小王推的" a defensible
answer for s027/s034.

**R8 — "收尾" keyword collision.** L377 (2025-01-20 "课程项目这周要交了，我还在收尾") collides with
s030's "2024 年那个课程项目收尾的时候".

**R9 — No mechanical-template or repetition problem.**
0 exactly duplicated messages; only **2** near-duplicate pairs at bigram-Jaccard ≥0.55
(L449/L600 "…现在一条命令就进去了" j=0.63; L1071/L1073 the 阿伟/WireGuard paraphrase j=0.59).
No repeated sentence-openers beyond human-normal ("我也是这么想的" ×3). Message lengths 4–49
(median 17), episodes 7–14 messages, timestamps strictly increasing and unique, 0 multi-date
episodes, one blank line per episode boundary (100 blanks / 100 episodes) — all conform to §1.

**R10 — No meta-text leaks.** Scanning for 评测/评估/query/evidence/gold/category/benchmark/数据集/
标注/测试集/语料/答案/正确/错误/基线/召回/打分/anchor/distractor/prompt yields only in-story hits:
L935 "数据集和测试这块" (about the character's own project), L81 "对一下答案" (exam answers),
L466 "必须自己写状态机". Zero benchmark leakage.

**R11 — Machine-report and doc drift.**
* `stress_validation.json` reports `"chunks": 169`; the frozen baseline chunker yields **165**
  (matches `stress_dense_probe.json`). The validator's hand-replayed splitter is not equivalent to
  `quivr_core`'s; the 169 number should not be quoted.
* Spec §6 targets 98 episodes (E001–E098; 2025H2 = 24) but the corpus has **100** (2025H2 = 26),
  and "≈45,000 chars → ≈150 chunks" vs actual 49,468 chars → 165 chunks. Documentation drift only.
* The merged corpus equals the four `part_*.txt` files **message-for-message** (1216/1216 identical
  sequence); the only difference is 3 added blank lines at the part boundaries, which are correct
  episode separators. The staged `queries_hard.json` + `queries_regression.json` are identical to
  the shipped `stress_queries.json` (36/36, zero field differences).

---

## 6. Recommendation

### **PASS-WITH-FIXES** — the corpus is structurally sound and retrieval-hard, but 3 gold answers are
poisoned and 9 of the 24 "hard" queries do not test what the contract claims.

**Must fix before the paid baseline run (gold answers are wrong/contradicted):**

| id | reason (one line) |
|---|---|
| **s007** | gold says applications started "五月"; corpus L1091/L1104/L1150 put the first submission ≈ 2026-04-26 |
| **s019** | same defect, and it is the *only* number the query asks for ("拖到几月…差了多久"); the "当初说好的时间" baseline is itself ambiguous (三月 L885 vs 四月 L917/L1042) |
| **s016** | gold asserts 小汪 "2025 年 2 月刚开始去健身房"; L26/L119/L316 show earlier gym activity, and "中间断了" is never stated |

**Should fix (correctness/robustness):**

| id | reason |
|---|---|
| **s026** | gold glosses Cloudflare Tunnel as "让设备之间互相访问", contradicting L435 (and matching only L643) — a model quoting L435 would be marked wrong |
| **s022** | "偶尔跑跑操场" over-specifies L974 ("还在跑操场"); the "之前办的那张卡" premise rests on the single line L902 |
| **s030** | "收尾" collides with L377 (a different course project) — narrow the time frame (e.g. "2024 年 10 月到年底") |
| **s003** | category `exact_fact` but the answer needs a 2-hop join across 2026-06-24 → 2026-07-05; relabel as `multi_evidence` or split |
| **s009** | "学长帮我看简历的时候" is ambiguous across three review sessions (L859, L871, L1022) |

**Not blocking, but the difficulty claim needs to be restated honestly.** 9 of 24 hard queries
(s017, s018, s020, s023, s024, s025, s026, s028, s032) are single-chunk answerable; 6 more
(s014, s015, s021, s027, s029, s033) need only 2 chunks. The *measured* difficulty of the dataset
(probe: Hit@5 79.4 %, coverage 56.6 %, 7 misses) is therefore mostly **retrieval** difficulty, not
state-assembly difficulty. The genuine multi-hop set is **7 queries: s013, s016, s019, s022, s030,
s031, s034**. Rewrites for the EASY/BORDERLINE items are given in §3.

**Do not run the paid baseline against the current `stress_queries.json`** unless s007, s016 and
s019 are corrected first — with those three golds as-is, a model that reads the corpus correctly is
scored wrong, and the headline metric will understate real capability by an unknown amount.
