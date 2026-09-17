"use strict";

/* One page, one flow: question -> answer -> evidence -> caveats.
 *
 * Two rules this file keeps:
 *   - Nothing is stored. No browser-side storage of any kind, no cookie, no history entry: a search
 *     over private chat must leave no trace after the tab is closed.
 *   - Nothing is logged. A failed request prints nothing to the console either, because a console
 *     line is one "copy" away from a support thread.
 *
 * The backend has already decided what the answer and the evidence are; this file only decides how
 * they look, and it renders with textContent, never innerHTML — model output is not markup.
 */

const form = document.getElementById("ask");
const input = document.getElementById("question");
const submit = document.getElementById("submit");
const stateLine = document.getElementById("state");
const answerBlock = document.getElementById("answer-block");
const answerText = document.getElementById("answer");
const evidenceBlock = document.getElementById("evidence-block");
const evidenceList = document.getElementById("evidence");
const warningLine = document.getElementById("warning");
const noteLine = document.getElementById("note");

let ready = true;   // set by the status probe; a failed index says so instead of failing per question
let busy = false;

/* --- tiny DOM helpers ---------------------------------------------------------------------- */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null && text !== "") node.textContent = text;
  return node;
}

function setState(text) {
  stateLine.textContent = text || "";
  stateLine.hidden = !text;
}

function clearResult() {
  answerText.textContent = "";
  answerBlock.hidden = true;
  evidenceList.replaceChildren();
  evidenceBlock.hidden = true;
  warningLine.textContent = "";
  warningLine.hidden = true;
  noteLine.textContent = "";
  noteLine.hidden = true;
}

/* --- evidence ------------------------------------------------------------------------------ */

function timeRange(card) {
  const start = card.start_time || "";
  const end = card.end_time || "";
  if (start && end && end !== start) return start + " – " + end;
  return start || end || "时间未知";
}

function renderLine(line) {
  const row = el("p", "line");
  row.append(el("span", "line-time", line.timestamp));
  row.append(el("span", "line-speaker", line.speaker));
  row.append(el("span", "line-text", line.text));
  return row;
}

function renderCard(card) {
  const node = el("article", "card");

  const head = el("div", "card-head");
  // The answer cites `[来源 N]` with N exactly `citation_index` (the framework numbers retrieved
  // sources from 0), so this label must not be adjusted by one: it is how a reader finds the card
  // a sentence is talking about.
  head.append(el("span", "card-source", "来源 " + card.citation_index));
  head.append(el("span", "card-conversation", card.conversation));
  node.append(head);

  const meta = el("div", "card-meta");
  meta.append(el("span", null, timeRange(card)));
  if (card.participants && card.participants.length) {
    meta.append(el("span", null, card.participants.join("、")));
  }
  node.append(meta);

  const lines = el("div", "lines");
  if (card.lines && card.lines.length) {
    for (const line of card.lines) lines.append(renderLine(line));
  } else {
    lines.append(el("p", "lines-empty", "该来源没有可解析的聊天行。"));
  }
  node.append(lines);

  return node;
}

/* --- caveats ------------------------------------------------------------------------------- */

function renderCaveats(report) {
  const alarms = [];

  const mismatches = report.citation_mismatches || [];
  if (mismatches.length) {
    let text = mismatches.length + " 处引用与来源对不上：答案引用的原文不在所标注的来源里";
    const first = mismatches[0];
    if (mismatches.length === 1 && first.cited && first.cited.length && first.found_in && first.found_in.length) {
      text += "（来源 " + first.cited[0] + " 的原文实际出现在来源 " + first.found_in[0] + "）";
    }
    alarms.push(text + "。");
  }

  if (report.invalid_citations) {
    alarms.push(report.invalid_citations + " 处引用超出了检索到的来源范围。");
  }

  const flags = report.attribution_flags || [];
  if (flags.length) {
    const speakers = Array.from(
      new Set(flags.map((flag) => flag.matched_line_speaker).filter(Boolean))
    ).join("、");
    const whose = speakers ? speakers + "自己的描述" : "他人自己的描述";
    alarms.push(flags.length + " 处陈述依赖" + whose + "，请确认答案没有把对方的情况当成你的。");
  }

  if (report.uncited) {
    alarms.push("该答案没有标注来源，无法核对。");
  }

  warningLine.textContent = alarms.length ? "[!] " + alarms.join(" ") : "";
  warningLine.hidden = !alarms.length;

  // A silence claim cannot be judged without knowing the intended answer, so it is a reminder to
  // check the record, not an alarm. Nothing is shown when there is nothing to say.
  const notes = [];
  if ((report.absence_claims || []).length) {
    notes.push("记录可能不完整，请结合原始聊天确认。");
  }
  noteLine.textContent = notes.join(" ");
  noteLine.hidden = !notes.length;
}

/* --- the flow ------------------------------------------------------------------------------ */

function render(result) {
  answerText.textContent = result.answer || "";
  answerBlock.hidden = false;

  const cards = result.evidence || [];
  evidenceList.replaceChildren(...cards.map(renderCard));
  evidenceBlock.hidden = !cards.length;

  renderCaveats(result.groundedness || {});
}

async function ask(question) {
  busy = true;
  submit.disabled = true;
  clearResult();
  setState("正在检索…");

  try {
    const response = await fetch("/api/recall", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question }),
    });
    if (!response.ok) throw new Error("recall failed");
    render(await response.json());
    setState("");
  } catch (error) {
    setState("查询失败，请重试。");
  } finally {
    busy = false;
    submit.disabled = !ready;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) {
    input.focus();
    return;   // an empty question is refused here as well as at the API
  }
  if (busy || !ready) return;
  ask(question);
});

async function probeStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) return;
    const status = await response.json();
    if (!status.ready) {
      ready = false;
      submit.disabled = true;
      setState(status.detail || "索引尚未就绪。");
    }
  } catch (error) {
    // A status probe that fails says nothing useful; the first question reports the truth.
  }
}

probeStatus();
