# Stress Corpus Spec (Phase 0.5)

Dataset card + authoring contract for `stress_chats.txt` / `stress_queries.json`.

Purpose: this corpus must be **substantially harder for dense retrieval than
`chats.txt`**. Difficulty comes from *retrieval ambiguity*, not from length:

- the same entity recurs across 2024/2025/2026 with a **changing state**
- plan vs actual outcome for the same topic
- near-duplicate distractor messages (same entity, similar wording, different time/outcome)
- pronouns / omissions ("那个数据库", "之前那个", "后来换的那个") instead of entity names
- two different people whose names look alike (小王 vs 小汪)

Isolation: `stress_*` files are a **separate dataset**. The 34-query small baseline
(`chats.txt`, `baseline_results.json`) must never be touched or overwritten.

Everything here is **synthetic**. No real names, accounts, phone numbers, or private chat.

---

## 1. Hard format rules

- One message per line: `[YYYY-MM-DD HH:MM] 说话人: 内容`
- `我` = the owner of the chat history. Others: `小王` `小汪` `张三` `室友` `同学A` `学长` `导师`
- Blank line **between episodes** (each episode = one continuous conversation burst)
- Timestamps strictly increasing inside one part; no duplicate timestamps anywhere
- Timestamps never go backwards across parts (2024 → 2025 → 2026)
- Realistic short chat lines: 4–60 Chinese chars per message, 6–14 messages per episode
- Plain text only. No markdown, no headings, no comments, no numbering
- Never write the words "gold evidence" / "query" / "category" inside the corpus

## 2. People roster (fixed identities — do not mix them up)

| Person | Identity | Topics | Notes |
|---|---|---|---|
| 我 | chat owner | everything | first person |
| 小王 | 大学同学 | 数据库 / 部署 / 旅行 / 实习 | also called **王哥** (same person) |
| 小汪 | **高中**同学 | 毕设(MySQL) / 健身 / 游戏 | **different person from 小王** — main name trap |
| 张三 | 大学同学 | 旅行 / 编译原理 / 操作系统 / 计网 | invites 长沙 |
| 室友 | dormmate | NAS / 内网穿透 / 实验室机器 | also called **阿伟** (same person) |
| 同学A | classmate | 部署 / 搜索 provider / Agent 项目 | |
| 学长 | senior | 简历 / 实习 / 职业规划 | |
| 导师 | advisor | 毕设选题 | never gives a concrete thesis title |

## 3. Entity state timelines (the backbone — facts are FIXED here)

### 3.1 Database line
| Time | Event |
|---|---|
| 2024-03 | 小王 recommends **Supabase**; 我用 Supabase 做课程项目 |
| 2024-07 | 同学A recommends **Neon** (PostgreSQL-flavored) |
| 2024-07-21 | 我 plans: 今晚把 Render + Neon 接起来试试 (**plan**) |
| 2024-08 | 实际：Neon 连上了，但只跑通连接，没做正式项目 (**actual**) |
| 2024-11 | Neon 免费层会休眠、冷启动慢 → 先不用 |
| 2025-01 | 换回 **Supabase** |
| 2025-06 | Supabase 免费额度不够 → 迁到 **Railway PostgreSQL**，并跑起来 |
| 2025-09 | 本地原型直接用 **SQLite** |
| 2026-02 | 我回顾：Neon 只测试过，没正式用 |
| 2026-04 | 又看了一次 Neon，最后还是没换 |
| **FINAL** | 正式使用 = **Railway PostgreSQL**；**Neon 从未正式采用，只测试过** |

### 3.2 Deployment line
| Time | Event |
|---|---|
| 2024-03 | 后端 **Render** |
| 2024-09 | 前端试 **Vercel** → 国内访问不稳 → 放弃 |
| 2025-07 | 前端准备改 **Cloudflare Pages** |
| 2025-08 | Cloudflare Pages 部署成功，国内访问偶尔不稳；室友问是 Pages 还是 Tunnel → **Pages** |
| 2025-03 | 室友聊 **Cloudflare Tunnel**（NAS 内网穿透）→ **不同场景**，是 distractor |
| 2025-12 | 后端 Render → **Railway** |
| **FINAL** | 前端 **Cloudflare Pages**，后端 **Railway** |

### 3.3 Remote-access line
| Time | Event |
|---|---|
| 2025-03 | 室友推荐 **Tailscale**，我装上 |
| 2025-08 | 还在用 Tailscale |
| 2026-04 | 室友把家里 NAS 换成 **WireGuard**；我说懒得折腾 |
| 2026-06 | 我试配 WireGuard，一晚没配通 → 回到 **Tailscale** |
| **FINAL** | 我 = **Tailscale**；WireGuard 只有室友在用 |

### 3.4 Search-provider line
| Time | Event |
|---|---|
| 2025-06 | 接的是 **Tavily** |
| 2025-10 | 试 **Brave Search** → 中文结果不行 → 放一边 |
| 2026-01 | 自建 **SearXNG** 两周 → 维护麻烦 → 放弃 |
| 2026-05 | 回到 **Tavily** |
| **FINAL** | **Tavily** |

### 3.5 Internship line (plan vs actual)
| Time | Event |
|---|---|
| 2026-03-18 | 学长: 四月开始就可以陆续投；我计划两周内整理完项目 (**plan**) |
| 2026-05-17 | 实际：5 月才开始投，回复不多 (**actual, late**) |
| 2026-06 | 两家约面试，都是 AI 应用方向 |
| **FINAL** | 计划 4 月初投 → 实际 5 月才投 |

### 3.6 Travel line
| Time | Event |
|---|---|
| 2024-05 | 五一 长沙（去了） |
| 2024-12 | 元旦 深圳 → 提出但没定（"先别定"）→ **没去** |
| 2025-04 | 五一 没出去（人太多） |
| 2025-09 | 国庆 计划去重庆，票还没买 (**plan**) |
| 2025-10 | 实际：重庆去了，人很多 (**actual**) |
| 2026-05 | 五一 在学校待着 |
| **FINAL** | 2025 国庆 = 重庆（去了）；2024 元旦深圳 = 没去 |

### 3.7 毕设 / 其他
- 导师从 2025-11 起多次提"选题方向"（多模态 / 检索），但**从未给过具体题目**
- 小汪 毕设用学校给的 MySQL（与 Neon/Supabase 无关）
- 课程线：2024-04 计网实验(VLAN/单臂路由)、2025-01 编译原理(scanner/Flex)、2025-12 操作系统(FIFO/LRU/Clock)

---

## 4. Anchor lines (MUST appear verbatim, at exactly these timestamps)

These are the facts the eval will depend on. Copy them **character-for-character**.

| ID | Line |
|---|---|
| A01 | `[2024-03-16 20:13] 小王: 可以试试 Supabase` |
| A02 | `[2024-07-21 15:10] 同学A: 那你看看 Neon，也是 PostgreSQL` |
| A03 | `[2024-07-21 15:17] 我: 好，我今晚把 Render 和 Neon 接起来试试` |
| A04 | `[2024-08-04 21:30] 我: Neon 连上了，不过只是跑通连接，没拿它做正式项目` |
| A05 | `[2024-11-19 22:41] 我: Neon 免费层会休眠，冷启动要等好几秒，先不用了` |
| A06 | `[2025-01-12 20:08] 我: 数据库我还是换回 Supabase 了` |
| A07 | `[2025-06-08 19:52] 我: Supabase 免费额度不太够，我准备把正式项目迁到 Railway 上的 PostgreSQL` |
| A08 | `[2025-06-21 21:14] 我: Railway 的 PostgreSQL 已经建好了，项目跑起来了` |
| A09 | `[2025-09-03 22:05] 我: 本地原型我就直接用 SQLite，不上云` |
| A10 | `[2026-02-26 18:39] 同学A: Neon 后来用了没？` |
| A11 | `[2026-02-26 18:40] 我: 之前测试过连接，但没专门拿它做正式项目` |
| A12 | `[2026-04-15 21:22] 我: 又看了一下 Neon，最后还是没换` |
| A13 | `[2024-03-16 20:12] 我: 还没，后端我放 Render 了，但数据库还在看` |
| A14 | `[2024-09-08 16:20] 我: 前端我先试了 Vercel，国内访问不太稳，算了` |
| A15 | `[2025-07-21 15:07] 我: 后端还是 Render，前端我准备改 Cloudflare Pages` |
| A16 | `[2025-08-12 20:08] 我: Pages，这次不是内网穿透` |
| A17 | `[2025-03-09 22:13] 室友: 实在不行你再看看 Cloudflare Tunnel，不过那个场景不太一样` |
| A18 | `[2025-12-14 20:31] 我: 后端也从 Render 迁到 Railway 了` |
| A19 | `[2025-03-09 22:05] 室友: 你要不装个 Tailscale` |
| A20 | `[2025-08-12 20:11] 我: 对，一个是部署网页，一个是机器之间互相访问` |
| A21 | `[2026-04-06 21:34] 室友: 我最近把家里 NAS 换成 WireGuard 了` |
| A22 | `[2026-04-06 21:38] 我: 我还是懒得折腾` |
| A23 | `[2026-06-11 22:47] 我: WireGuard 我配了一晚上没配通，还是回去用 Tailscale 了` |
| A24 | `[2025-06-03 14:42] 我: 现在接的是 Tavily` |
| A25 | `[2025-10-11 21:02] 我: Brave Search 的中文结果不太行，先放一边` |
| A26 | `[2026-01-19 22:18] 我: 自己搭了个 SearXNG，试了两周，维护太麻烦` |
| A27 | `[2026-05-23 20:44] 我: 搜索还是回到 Tavily 了` |
| A28 | `[2026-03-18 20:51] 学长: 对，四月开始就可以陆续投了，别等到五月` |
| A29 | `[2026-03-18 20:53] 我: 行，那我这两周先把两个主要项目整理完` |
| A30 | `[2026-05-17 16:09] 我: 已经开始投了，不过回复不算多` |
| A31 | `[2026-06-24 21:36] 我: 有两家约了面试，都是 AI 应用方向的` |
| A32 | `[2024-12-07 16:13] 我: 先别定，我还不知道元旦有没有课程作业` |
| A33 | `[2025-09-22 20:15] 我: 国庆我准备去重庆，票还没买` |
| A34 | `[2025-10-06 21:40] 我: 重庆回来了，人真的多` |
| A35 | `[2026-05-01 18:22] 我: 五一就在学校待着，哪也没去` |
| A36 | `[2025-01-14 23:20] 张三: 王哥前两天不是还说要去深圳吗` |
| A37 | `[2025-01-14 23:21] 我: 嗯，小王提过，后来不是没定嘛` |
| A38 | `[2026-04-06 21:31] 我: 阿伟说他把家里 NAS 换成 WireGuard 了` |
| A39 | `[2025-11-08 19:40] 小汪: 我最近在弄毕设，数据库就用学校给的 MySQL` |
| A40 | `[2026-02-14 20:12] 小汪: 健身房年卡我办了，你要不要一起` |
| A41 | `[2025-11-26 20:30] 导师: 选题方向你可以先往检索或者多模态那边想` |
| A42 | `[2026-03-05 21:10] 我: 导师那边题目还没定，只说往检索方向想` |
| A43 | `[2026-05-02 10:15] 小王: 你之前那个数据库最后到底用的哪个` |
| A44 | `[2026-05-02 10:16] 我: 正式的项目都在 Railway 上，Neon 就一直没上` |

## 5. Distractor rules (this is what makes the corpus "stress")

Every hard topic needs **near-duplicate messages that are wrong for the query**:

1. **Same entity, different time/outcome**: at least 3 mentions of Neon *before* the
   FINAL state, each with a different status (recommend → plan → tested → paused → reconsidered).
2. **Plan vs actual pair**: the plan line and the actual line must be ≥2 weeks apart and
   must NOT use the same verbs.
3. **Omission/pronoun lines**: at least 12 lines total that reference an entity without
   naming it, e.g. `我: 那个数据库还是算了`, `我: 后来换的那个顺手多了`,
   `我: 之前那个免费层会睡`. These must be resolvable from neighbouring lines.
4. **Tunnel vs Pages**: Cloudflare Tunnel (intranet) and Cloudflare Pages (web deploy)
   must both appear multiple times, in different years.
5. **小王 vs 小汪**: both names appear ≥ 6 times, in unrelated topics.
6. **Wrong-tool distractors**: `Vercel` and `Railway` both appear as deployment candidates;
   `SQLite`/`MySQL` appear as *other people's* or local-only choices.
7. Do **not** let any single line state the final answer in a way that makes the
   state-evolution questions trivial — the state must have to be assembled from ≥2 lines.

## 6. Part assignment + size targets

| Part | File | Range | Episodes | Anchor lines | Min chars |
|---|---|---|---|---|---|
| 2024 | `part_2024.txt` | 2024-01-01 → 2024-12-31 | 24 | A01–A05, A13, A14, A32 | 11,000 |
| 2025H1 | `part_2025h1.txt` | 2025-01-01 → 2025-06-30 | 22 | A06, A07, A08, A17, A19, A24, A36, A37 | 9,000 |
| 2025H2 | `part_2025h2.txt` | 2025-07-01 → 2025-12-31 | 26 | A09, A15, A16, A18, A20, A25, A33, A34, A39, A41 | 10,000 |
| 2026 | `part_2026.txt` | 2026-01-01 → 2026-08-31 | 28 | A10, A11, A12, A21, A22, A23, A26, A27, A28, A29, A30, A31, A35, A38, A40, A42, A43, A44 | 15,000 |

The per-part `Min chars` column is the authoring budget. The figures below are the **measured**
ones (authoritative: `stress_validation.json`) — the pre-authoring estimate in an earlier revision
of this file (45,000 chars / ≈150 chunks / 98 episodes) was superseded:

* **49,457 chars / 94,262 bytes / 1,216 messages / 100 episodes**
* **165 chunks** at chunk_size=400 / overlap=100 → Top-5 covers **3.0 %** of the memory space
  (vs 21.7 % for the 23-chunk small corpus)
* 44/44 anchor lines present verbatim; timestamps strictly increasing and unique

The corpus was authored as four chronological parts and merged; two self-decoding alias glosses
(`王哥（就是小王）`, `阿伟（我室友）`) were removed after the independent audit showed they defeated
the entity-disambiguation queries, so A36/A38 in §4 carry the post-audit wording.

Episode topic guidance per part is given in the writer prompts. Facts come from §3, wording
is the writer's, anchors are verbatim.

---

## 7. Query authoring rules (`stress_queries.json`)

Schema is **identical** to `data/queries.json` — exactly 6 keys, no extras:
`id`, `question`, `expected_answer`, `relevant_evidence` (list), `category` (str), `answerable` (bool).

`id` = `s001` … `s036`, ordered so that all regression queries come first.

### 7.1 Composition (36 queries)

| Bucket | Category | Count |
|---|---|---|
| regression (12) | `exact_fact` | 3 |
| | `exact_keyword` | 2 |
| | `semantic_recall` | 3 |
| | `person_recall` | 2 |
| | `time_recall` | 2 |
| hard (24) | `temporal_state_change` | 4 |
| | `plan_vs_actual` | 3 |
| | `latest_state` | 3 |
| | `earliest_state` | 2 |
| | `entity_disambiguation` | 3 |
| | `negative_evidence` | 2 |
| | `multi_evidence` | 2 |
| | `implicit_reference` | 2 |
| | `cross_time` | 1 |
| | `unanswerable` | 2 |

`regression` = same *kinds* of questions as the small baseline, so a score drop on them would
mean the new corpus broke something that used to work. They may be easier.

### 7.2 Evidence rules

- `relevant_evidence` entries are **whole corpus lines copied verbatim**, including the
  `[YYYY-MM-DD HH:MM] 说话人: ` prefix. 1–4 lines per query. Never paraphrase.
- Every evidence line must exist in `stress_chats.txt` (the validator enforces this).
- For state-change queries the evidence **must span at least 2 different dates**, otherwise
  the query does not test state evolution.

### 7.3 Anti-shortcut rules (what makes the hard queries hard)

1. `latest_state` / `negative_evidence` / `implicit_reference`: the question must **not** contain
   the entity's distinctive token. Use "那个数据库", "后来换的那个", "现在到底用哪个" instead of
   "Neon"/"Railway"/"Tavily". The answer must require assembling ≥2 lines.
2. `implicit_reference`: the question is pronoun/omission-only. Zero entity names allowed,
   except a person name if needed to disambiguate the conversation.
3. `entity_disambiguation`: exploit 小王 vs 小汪, or 王哥=小王, or 阿伟=室友, or
   Pages vs Tunnel. The correct answer must be contradicted by a similar-looking distractor line.
4. `plan_vs_actual`: ask about the *outcome* ("最后做了吗/到底去没去/后来投了没"), not the plan.
   Evidence must include both the plan line and the actual line.
5. `earliest_state`: "第一次/最早" — the corpus must contain ≥3 later mentions so the model
   has to order them; `expected_answer` states the first occurrence and its date.
6. `cross_time`: one question spanning ≥2 years; evidence from ≥2 years.
7. `unanswerable`: pick a plausible-sounding fact that is genuinely **absent** from the corpus
   (e.g. the advisor's concrete thesis title, a book recommendation, a wedding). `answerable`
   must be `false`, `expected_answer` must be `null`, `relevant_evidence` must be `[]`.
   Before finalising, grep the corpus for every distinctive token of the question to prove
   the fact is really absent.
8. Do not reuse a question that can be answered by a single line whose text overlaps the
   question's wording by more than ~50% of its character bigrams — that is a lexical shortcut.

### 7.4 Answer format

- `expected_answer`: for regression, a short string ("Cloudflare Pages"). For state-change
  questions, one or two sentences naming the final state **and** the trajectory.
- `question` language: natural spoken Chinese, the way the owner would actually ask
  ("我之前说的那个数据库最后到底用了没"). No keywords lists, no quotes around entity names.
