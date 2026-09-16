# Stress Corpus v2 — Distractor Pack Spec

`stress_chats_v2.txt` = corpus v1 (1,216 messages, 165 chunks) **plus** this pack. The 36 gold
queries are unchanged: their evidence lines stay the only place the tracked facts are stated.

**Purpose:** raise retrieval ambiguity ~2.5x without touching a single tracked fact. v1 saturates the
retriever (Hit@10 = 100 %, coverage 86 %), so mechanism A/Bs there measure noise. The pack makes the
same questions hard again by surrounding the gold arcs with many *topically identical* sessions that
say *different things*.

## 1. Forbidden content (never state, restate, deny or hint at these)

These are the tracked facts behind the 36 queries. A distractor pack that repeats them would raise
coverage artificially; one that contradicts them would break answerability.

* **Neon** — 同学A 推荐 / 我只跑通连接、从未正式采用 / 免费层休眠
* **Supabase** — 2024-03 起课程项目一直用它 / 2025-01 换回 / 2025-06 迁走
* **Railway PostgreSQL** — 2025-06-08 计划、06-21 建好、正式项目一直用它
* **后端** Render → Railway（2025-12-14）；**前端** Vercel（2024-09 放弃）→ Cloudflare Pages（2025-07/08）
* **远程访问** — Tailscale（2025-03 起）/ WireGuard（室友 2026-04 换、我 2026-06 试失败回退）
* **搜索** — Tavily → Brave Search（2025-10 放弃）→ SearXNG（2026-01 放弃）→ 回 Tavily（2026-05）
* **实习** — 计划四月、实际 4 月下旬；两家面试（一家挂了）
* **出行** — 2024 五一长沙（去了）、2024 元旦深圳（没定）、2025 国庆重庆（去了）、2026 五一在学校
* **小汪健身** 的四个阶段；**导师从未给过毕设题目**；**小汪毕设数据库 = 学校给的 MySQL**
* **绝不能出现任何餐厅/店铺的名字**，也**绝不能出现任何毕设题目名字** —— 这两点会直接破坏两个
  `unanswerable` 查询（s035 题目名、s036 店名）。

Mentioning the *entities* (Neon, Supabase, Railway, Tailscale, Tavily, SQLite, MySQL, PostgreSQL,
Render, Vercel, Cloudflare, FAISS, BGE ...) is **required** — just never about the tracked arcs above.

## 2. Required texture (what actually creates the ambiguity)

1. **Same entity, other contexts** — 别人的项目、课程作业、教程里看到的、二手转述、工具对比、吐槽。
2. **Gold-shaped sentences about different things** — 大量 "我准备试试… / 后来换成… / 最后还是没上… /
   迁移完了 / 免费额度不够 / 冷启动太慢 / 又看了一遍还是没换"。
3. **Omission and pronouns** — 至少 20 行只说 "那个库 / 之前那个 / 后来换的那个 / 老方案 / 新那套"。
4. **Cross-speaker disagreement** — 同一个人/同一个工具在不同人嘴里结论不同（但不得触及第 1 节的事实）。
5. **Near-duplicate wording** — 与其它 episode 高度相似、只差一两个词的句子（这是最有效的 distractor）。

## 3. Format (identical to v1)

* `[YYYY-MM-DD HH:MM] 说话人: 内容` — one space after `]`, one after `:`.
* Speakers only: `我` `小王` `小汪` `张三` `室友` `同学A` `学长` `导师`
* 7–14 messages per episode, 4–60 Chinese chars per message, **blank line between episodes**
* Timestamps strictly increasing within your file; **only the days allocated to you** (below)
* Plain text only. No headings, no numbering, no markdown, no comments.
* Never write the words 评测/query/category/evidence/测试数据/synthetic.

## 4. Allocation (~20,000 Chinese chars per writer, 32–36 episodes)

### Writer A — `part_a_2024.txt`

Window: 2024-01-01 .. 2024-11-27. **Use only these 40 days** (several episodes may share a day; times must be unique):

2024-01-01, 2024-01-10, 2024-01-18, 2024-01-26, 2024-02-03, 2024-02-12, 2024-02-20, 2024-02-28, 2024-03-07, 2024-03-15, 2024-03-25, 2024-04-02, 2024-04-12, 2024-04-20, 2024-04-28, 2024-05-08, 2024-05-16, 2024-05-24, 2024-06-01, 2024-06-09, 2024-06-18, 2024-06-26, 2024-07-04, 2024-07-12, 2024-07-20, 2024-07-30, 2024-08-09, 2024-08-17, 2024-08-25, 2024-09-02, 2024-09-11, 2024-09-20, 2024-09-28, 2024-10-07, 2024-10-15, 2024-10-24, 2024-11-01, 2024-11-09, 2024-11-17, 2024-11-27

### Writer B — `part_b_2025.txt`

Window: 2025-01-01 .. 2025-11-05. **Use only these 40 days** (several episodes may share a day; times must be unique):

2025-01-01, 2025-01-09, 2025-01-18, 2025-01-26, 2025-02-03, 2025-02-10, 2025-02-18, 2025-02-26, 2025-03-05, 2025-03-14, 2025-03-21, 2025-03-28, 2025-04-05, 2025-04-12, 2025-04-19, 2025-04-27, 2025-05-05, 2025-05-13, 2025-05-21, 2025-05-28, 2025-06-05, 2025-06-13, 2025-06-22, 2025-07-03, 2025-07-11, 2025-07-18, 2025-07-27, 2025-08-03, 2025-08-10, 2025-08-18, 2025-08-26, 2025-09-02, 2025-09-11, 2025-09-18, 2025-09-26, 2025-10-04, 2025-10-13, 2025-10-21, 2025-10-29, 2025-11-05

### Writer C — `part_c_2026.txt`

Window: 2026-01-01 .. 2026-08-09. **Use only these 40 days** (several episodes may share a day; times must be unique):

2026-01-01, 2026-01-07, 2026-01-12, 2026-01-17, 2026-01-23, 2026-01-29, 2026-02-03, 2026-02-08, 2026-02-13, 2026-02-19, 2026-02-25, 2026-03-03, 2026-03-09, 2026-03-15, 2026-03-21, 2026-03-27, 2026-04-01, 2026-04-07, 2026-04-12, 2026-04-18, 2026-04-24, 2026-04-30, 2026-05-07, 2026-05-13, 2026-05-19, 2026-05-25, 2026-05-30, 2026-06-05, 2026-06-10, 2026-06-16, 2026-06-22, 2026-06-28, 2026-07-03, 2026-07-09, 2026-07-14, 2026-07-19, 2026-07-25, 2026-07-30, 2026-08-04, 2026-08-09


## 5. Writer deliverable

`D:\DevPilot\.tmp_stress2\<your file>` — plus a verification run reporting: chars, message count,
episode count, day usage (must be a subset of your allocated days), speaker counts, and a scan
proving no forbidden phrase from §1 appears.
