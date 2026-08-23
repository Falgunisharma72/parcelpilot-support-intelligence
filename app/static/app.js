/* ParcelPilot Support Intelligence — client.
   Renders the SSE event stream from /api/chat as it arrives: text into the
   thread, tool calls into chips and the trace panel, verdicts into receipts,
   and any state-changing action into a confirmation card that must be answered
   before the action exists. */

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

const state = {
  sessionId: null,
  principal: null,
  users: [],
  busy: false,
  signals: [],
  signalFilter: null,
  currentAgentMsg: null,
};

const EXAMPLES = {
  customer: [
    "Can I cancel ORD-1001 without a cancellation fee? Explain why.",
    "A pickup is three hours late because of carrier fault. Should I get a service credit?",
    "Our bulk CSV upload keeps failing at around 70%. What's going on?",
    "How quickly will you respond to a P1 outage on our account?",
    "I want you to waive the fee as a one-off — can you do that?",
  ],
  internal: [
    "What needs my attention right now?",
    "Can Northstar cancel ORD-1001 without a cancellation fee? Explain why.",
    "Is LumenWorks owed a service credit on ORD-2002?",
    "Is TKT-505 within SLA? What should we do?",
    "Has any customer been given incorrect guidance in the past?",
  ],
};

/* ── boot ─────────────────────────────────────────────────────────── */
async function boot() {
  const data = await (await fetch("/api/bootstrap")).json();
  state.users = data.users;
  state.model = data.model || {};
  $("#snapshotValue").textContent = data.snapshot;
  $("#modelValue").textContent = state.model.label || "—";
  $("#modelValue").classList.toggle("is-free", Boolean(state.model.free));

  const sel = $("#userSelect");
  const groups = { customer: "Customer chat", internal: "Internal support" };
  for (const ctx of ["customer", "internal"]) {
    const og = el("optgroup");
    og.label = groups[ctx];
    data.users.filter((u) => u.context === ctx).forEach((u) => {
      const o = el("option", null, `${u.display_name} — ${u.org}`);
      o.value = u.user_id;
      og.appendChild(o);
    });
    sel.appendChild(og);
  }
  sel.value = "staff-rohit";
  sel.addEventListener("change", () => startSession(sel.value));

  await startSession(sel.value);
  if (!data.agent_available) providerSetupNotice(data.model);
}

/* No model key configured: say exactly what to do about it, with the free
   options, rather than a bare "unavailable". */
function providerSetupNotice(model) {
  const box = el("div", "msg msg-setup");
  box.appendChild(el("h3", null, "Add a model key to enable the chat"));
  box.appendChild(el("p", null,
    "Everything deterministic already works — open Signals for the 13 detected " +
    "findings, or Access log for the enforcement trail. The chat needs one API key, " +
    "and every option below has a free tier."));
  const list = el("ul", "provider-list");
  (model.free_options || []).forEach((p) => {
    const li = el("li");
    const a = el("a", null, p.label);
    a.href = p.signup; a.target = "_blank"; a.rel = "noopener";
    li.appendChild(a);
    li.appendChild(el("span", "provider-free", p.free));
    li.appendChild(el("code", null, `${p.env_key}=…`));
    list.appendChild(li);
  });
  box.appendChild(list);
  box.appendChild(el("p", "provider-hint",
    "Put the key in .env, restart, and the provider is detected automatically. " +
    "Run `make providers` to check it works."));
  $("#thread").appendChild(box);
  scroll();
}

async function startSession(userId) {
  const res = await fetch("/api/session", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId }),
  });
  const data = await res.json();
  state.sessionId = data.session_id;
  state.principal = data.principal;

  const user = state.users.find((u) => u.user_id === userId) || {};
  const card = $("#identityCard");
  card.innerHTML = "";
  card.appendChild(el("div", "identity-name", data.principal.display_name));
  card.appendChild(el("div", "identity-role", data.principal.role_label));
  card.appendChild(el("div", "identity-org", user.org || ""));
  if (data.principal.account_id)
    card.appendChild(el("div", "identity-acct", data.principal.account_id));

  $("#scopeNote").textContent = data.principal.context === "customer"
    ? "Row scoping is applied in SQL: this session can only read this account. Internal fields are removed on the way out."
    : "Full cross-account read access, plus the proactive signals view. Every tool call is written to the access log.";

  const list = $("#toolList");
  list.innerHTML = "";
  data.tools.forEach((t) => {
    const li = el("li");
    li.appendChild(el("span", `dot ${t.category}`));
    li.appendChild(el("span", null, t.name));
    if (t.state_changing) li.appendChild(el("span", "lock", "confirm"));
    list.appendChild(li);
  });

  const ex = $("#exampleList");
  ex.innerHTML = "";
  EXAMPLES[data.principal.context].forEach((q) => {
    const li = el("li");
    const b = el("button", null, q);
    b.type = "button";
    b.addEventListener("click", () => { $("#input").value = q; send(); });
    li.appendChild(b);
    ex.appendChild(li);
  });

  $("#thread").innerHTML = "";
  $("#trace").innerHTML = "";
  $("#trace").appendChild(el("p", "trace-empty",
    "Tool calls, decision arithmetic and the sources behind each answer appear here as the assistant works."));
  systemLine(`New session · ${data.principal.role_label}${data.principal.account_id ? " · " + data.principal.account_id : ""}`);

  document.querySelectorAll('.tab[data-view="signals"], .tab[data-view="audit"]')
    .forEach((t) => { t.style.display = data.principal.context === "customer" ? "none" : ""; });
  if (data.principal.context === "internal") loadSignals();
  else { showView("chat"); $("#signalCount").textContent = ""; }
}

/* ── thread rendering ─────────────────────────────────────────────── */
function systemLine(text) {
  const n = el("div", "msg msg-system", text);
  $("#thread").appendChild(n);
  scroll();
}

function userMessage(text) {
  const wrap = el("div", "msg msg-user");
  wrap.appendChild(el("div", "msg-role", "You"));
  wrap.appendChild(el("div", "bubble", text));
  $("#thread").appendChild(wrap);
  scroll();
}

function newAgentMessage() {
  const wrap = el("div", "msg msg-agent");
  wrap.appendChild(el("div", "msg-role", "ParcelPilot assistant"));
  const think = el("details", "thinking");
  think.hidden = true;
  const sum = el("summary", null, "Reasoning");
  const body = el("div", "thinking-body");
  think.appendChild(sum); think.appendChild(body);
  const chips = el("div", "chips");
  chips.hidden = true;
  const bubble = el("div", "bubble");
  wrap.appendChild(think); wrap.appendChild(chips); wrap.appendChild(bubble);
  $("#thread").appendChild(wrap);
  scroll();
  return { wrap, think, thinkBody: body, chips, bubble, raw: "" };
}

function scroll() {
  const t = $("#thread");
  t.scrollTop = t.scrollHeight;
}

/* Minimal, deliberately conservative markdown: bold, inline code, lists,
   paragraphs. Everything is inserted as text nodes — no innerHTML on model
   output. */
function renderMarkdown(target, src) {
  target.innerHTML = "";
  const blocks = src.split(/\n{2,}/);
  for (const block of blocks) {
    const lines = block.split("\n").filter((l) => l.trim());
    if (!lines.length) continue;
    const bulleted = lines.every((l) => /^\s*[-*•]\s+/.test(l));
    const numbered = lines.every((l) => /^\s*\d+[.)]\s+/.test(l));
    if (bulleted || numbered) {
      const list = el(numbered ? "ol" : "ul");
      lines.forEach((l) => {
        const li = el("li");
        inline(li, l.replace(/^\s*(?:[-*•]|\d+[.)])\s+/, ""));
        list.appendChild(li);
      });
      target.appendChild(list);
    } else {
      const p = el("p");
      inline(p, lines.join(" "));
      target.appendChild(p);
    }
  }
}

function inline(node, text) {
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) node.appendChild(document.createTextNode(text.slice(last, m.index)));
    const tok = m[0];
    if (tok.startsWith("**")) node.appendChild(el("strong", null, tok.slice(2, -2)));
    else node.appendChild(el("code", null, tok.slice(1, -1)));
    last = m.index + tok.length;
  }
  if (last < text.length) node.appendChild(document.createTextNode(text.slice(last)));
}

/* ── tool chips + trace ───────────────────────────────────────────── */
const chipRegistry = new Map();

function chipFor(msg, ev) {
  msg.chips.hidden = false;
  const chip = el("span", `chip running ${ev.category}`);
  chip.appendChild(el("span", `dot ${ev.category}`));
  chip.appendChild(el("span", null, ev.name));
  const sum = el("span", "chip-sum", "running…");
  chip.appendChild(sum);
  msg.chips.appendChild(chip);
  chipRegistry.set(ev.id, { chip, sum });
  return chip;
}

function settleChip(ev) {
  const entry = chipRegistry.get(ev.id);
  if (!entry) return;
  entry.chip.classList.remove("running");
  if (ev.outcome !== "ok") entry.chip.classList.add("denied");
  entry.sum.textContent = ev.summary;
}

function traceReset() {
  const t = $("#trace");
  if (t.querySelector(".trace-empty")) t.innerHTML = "";
}

function traceStep(ev) {
  traceReset();
  const step = el("div", `trace-step ${ev.category}`);
  step.appendChild(el("div", "trace-name", ev.name));
  const args = Object.entries(ev.args || {})
    .map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`)
    .join("  ");
  if (args) step.appendChild(el("div", "trace-args", args));
  $("#trace").appendChild(step);
  $("#trace").scrollTop = $("#trace").scrollHeight;
  return step;
}

const TIER_LABEL = { 1: "AGREEMENT", 2: "POLICY", 3: "PRODUCT DOC", 4: "HISTORICAL", 99: "SUPERSEDED" };

function stampList(citations) {
  const wrap = el("div", "stamps");
  (citations || []).forEach((c) => {
    if (!c) return;
    const tier = c.authority_tier || 3;
    const s = el("div", `stamp t${tier}`);
    s.appendChild(el("span", "stamp-rank", TIER_LABEL[tier] || "SOURCE"));
    const body = el("div");
    body.appendChild(el("div", "stamp-cite", c.citation || c.clause_id));
    if (c.text) body.appendChild(el("div", "stamp-text", truncate(c.text, 170)));
    s.appendChild(body);
    wrap.appendChild(s);
  });
  return wrap;
}

function truncate(s, n) { return s.length > n ? s.slice(0, n - 1) + "…" : s; }

/* The receipt: deterministic arithmetic, rendered as a docket. */
function receipt(verdict) {
  const box = el("div", "receipt");
  const head = el("div", "receipt-head");
  head.appendChild(el("b", null, verdict.topic.replace(/_/g, " ")));
  head.appendChild(el("span", "receipt-verdict", verdict.decision.replace(/_/g, " ")));
  box.appendChild(head);

  const lines = el("div", "receipt-lines");
  (verdict.computation || []).forEach((c) => lines.appendChild(el("div", "receipt-line", c)));
  if (lines.children.length) box.appendChild(lines);

  if (verdict.amount_inr !== null && verdict.amount_inr !== undefined) {
    const total = el("div", "receipt-total");
    total.appendChild(el("span", null, verdict.topic === "cancellation" ? "Cancellation fee" : "Credit"));
    total.appendChild(el("span", null, `INR ${Number(verdict.amount_inr).toLocaleString("en-IN")}`));
    box.appendChild(total);
  }
  if (verdict.needs_human) {
    box.appendChild(el("div", "receipt-flag", `Human required — ${verdict.needs_human_reason}`));
  }
  (verdict.caveats || []).forEach((c) => box.appendChild(el("div", "receipt-flag", c)));
  return box;
}

function traceResult(step, ev) {
  const p = ev.payload || {};
  step.appendChild(el("div", "trace-sum", ev.summary));

  if (p.verdict) {
    step.appendChild(receipt(p.verdict));
    if (p.verdict.rule_overridden && p.verdict.rule_overridden.citation) {
      const c = el("div", "conflict");
      c.appendChild(el("b", null, "Precedence applied"));
      const beats = el("div", "beats",
        `Applied: ${p.verdict.rule_applied?.citation?.citation || "—"}`);
      const beaten = el("div", "beaten",
        `Overridden: ${p.verdict.rule_overridden.citation.citation}`);
      c.appendChild(beats); c.appendChild(beaten);
      step.appendChild(c);
    }
    step.appendChild(stampList(p.verdict.citations));
  }

  if (p.results) {
    step.appendChild(stampList(p.results.map((r) => ({
      authority_tier: r.authority_tier, citation: r.citation, text: r.text,
    }))));
    (p.conflicts || []).forEach((cf) => {
      const c = el("div", "conflict");
      c.appendChild(el("b", null, `Conflict — ${cf.topic.replace(/_/g, " ")}`));
      c.appendChild(el("div", "beats", cf.resolution));
      c.appendChild(el("div", "beats", `Wins: ${cf.authoritative}`));
      (cf.overridden || []).forEach((o) =>
        c.appendChild(el("div", "beaten", `Yields: ${o.citation}`)));
      step.appendChild(c);
    });
  }

  if (p.matches && p.matches.length) {
    p.matches.forEach((ki) => {
      const box = el("div", "receipt");
      const head = el("div", "receipt-head");
      head.appendChild(el("b", null, ki.id));
      head.appendChild(el("span", "receipt-verdict", ki.status || ""));
      box.appendChild(head);
      const lines = el("div", "receipt-lines");
      lines.appendChild(el("div", "receipt-line", ki.title));
      if (ki.workaround) lines.appendChild(el("div", "receipt-line", ki.workaround));
      if (ki.caution) lines.appendChild(el("div", "receipt-line", ki.caution));
      box.appendChild(lines);
      step.appendChild(box);
    });
  }

  if (ev.outcome === "denied") {
    step.appendChild(el("div", "receipt-flag", p.error || "Blocked by the data layer"));
  }
  $("#trace").scrollTop = $("#trace").scrollHeight;
}

/* ── confirmation card ────────────────────────────────────────────── */
function confirmCard(proposal) {
  const card = el("div", "confirm");
  const head = el("div", "confirm-head");
  head.appendChild(el("div", "confirm-kicker", "Awaiting your confirmation — nothing has been created"));
  head.appendChild(el("h3", "confirm-title", proposal.title));
  card.appendChild(head);

  const rows = el("dl", "confirm-rows");
  Object.entries(proposal.preview || {}).forEach(([k, v]) => {
    if (v === null || v === undefined || v === "") return;
    const row = el("div", "confirm-row");
    row.appendChild(el("dt", null, k.replace(/_/g, " ")));
    row.appendChild(el("dd", null, String(v)));
    rows.appendChild(row);
  });
  card.appendChild(rows);
  (proposal.warnings || []).forEach((w) => card.appendChild(el("div", "confirm-warn", w)));

  const actions = el("div", "confirm-actions");
  const yes = el("button", "btn btn-primary", "Confirm and create");
  const no = el("button", "btn btn-quiet", "Decline");
  const note = el("span", "confirm-note", `expires in ${Math.round(proposal.expires_in_seconds / 60)} min`);
  actions.appendChild(yes); actions.appendChild(no); actions.appendChild(note);
  card.appendChild(actions);

  const decide = async (decision) => {
    yes.disabled = no.disabled = true;
    actions.innerHTML = "";
    const stamp = el("span", `receipt-stamp ${decision === "confirm" ? "" : "declined"}`,
      decision === "confirm" ? "Confirmed" : "Declined");
    actions.appendChild(stamp);
    await stream("/api/confirm", {
      session_id: state.sessionId, proposal_id: proposal.proposal_id, decision,
    });
  };
  yes.addEventListener("click", () => decide("confirm"));
  no.addEventListener("click", () => decide("cancel"));

  const wrap = el("div", "msg msg-agent");
  wrap.appendChild(card);
  $("#thread").appendChild(wrap);
  scroll();
}

/* ── streaming ────────────────────────────────────────────────────── */
async function stream(url, body) {
  state.busy = true;
  $("#send") && ($("#send").disabled = true);
  let msg = null;
  const traceSteps = new Map();

  try {
    const res = await fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      errorLine(detail.detail || `Request failed (${res.status})`);
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        const raw = line.slice(5).trim();
        if (raw === "[DONE]") continue;
        let ev;
        try { ev = JSON.parse(raw); } catch { continue; }

        switch (ev.type) {
          case "text":
            if (!msg) msg = newAgentMessage();
            msg.raw += ev.text;
            renderMarkdown(msg.bubble, msg.raw);
            scroll();
            break;
          case "thinking":
            if (!msg) msg = newAgentMessage();
            msg.think.hidden = false;
            msg.thinkBody.textContent += ev.text;
            break;
          case "tool_start":
            if (!msg) msg = newAgentMessage();
            chipFor(msg, ev);
            traceSteps.set(ev.id, traceStep(ev));
            break;
          case "tool_result":
            settleChip(ev);
            if (traceSteps.has(ev.id)) traceResult(traceSteps.get(ev.id), ev);
            break;
          case "proposal":
            msg = null;                       // the card gets its own block
            confirmCard(ev.proposal);
            break;
          case "action_result":
            systemLine(ev.status === "confirmed"
              ? `${ev.title} — carried out (${Object.values(ev.result || {})[0] || "done"})`
              : `${ev.title} — declined, nothing was written`);
            msg = null;
            break;
          case "error":
            errorLine(ev.message);
            msg = null;
            break;
          case "done":
            msg = null;
            break;
        }
      }
    }
  } catch (err) {
    errorLine(String(err));
  } finally {
    state.busy = false;
    if (state.principal && state.principal.context === "internal") loadSignals();
  }
}

function errorLine(text) {
  const n = el("div", "msg msg-error", text);
  $("#thread").appendChild(n);
  scroll();
}

async function send() {
  const input = $("#input");
  const text = input.value.trim();
  if (!text || state.busy) return;
  input.value = "";
  input.style.height = "auto";
  userMessage(text);
  showView("chat");
  await stream("/api/chat", { session_id: state.sessionId, message: text });
}

/* ── signals ──────────────────────────────────────────────────────── */
async function loadSignals() {
  if (!state.principal || state.principal.context !== "internal") return;
  const res = await fetch(`/api/signals?user_id=${encodeURIComponent(state.principal.user_id)}`);
  if (!res.ok) return;
  const data = await res.json();
  state.signals = data.signals;

  const crit = (data.summary.by_severity.critical || 0) + (data.summary.by_severity.high || 0);
  $("#signalCount").textContent = crit ? String(crit) : "";

  const tallies = $("#signalTallies");
  tallies.innerHTML = "";
  ["critical", "high", "medium", "low"].forEach((sev) => {
    const n = data.summary.by_severity[sev] || 0;
    if (!n) return;
    const t = el("div", `tally ${sev}`);
    t.appendChild(el("div", "tally-n", String(n)));
    t.appendChild(el("div", "tally-l", sev));
    tallies.appendChild(t);
  });

  const filters = $("#signalFilters");
  filters.innerHTML = "";
  const mkFilter = (label, type) => {
    const b = el("button", state.signalFilter === type ? "on" : "", label);
    b.addEventListener("click", () => {
      state.signalFilter = state.signalFilter === type ? null : type;
      renderSignals();
    });
    filters.appendChild(b);
  };
  mkFilter(`All (${data.summary.total})`, null);
  Object.entries(data.summary.by_type).forEach(([t, n]) =>
    mkFilter(`${t.replace(/_/g, " ")} (${n})`, t));
  renderSignals();
}

function renderSignals() {
  document.querySelectorAll("#signalFilters button").forEach((b) => b.classList.remove("on"));
  const grid = $("#signalGrid");
  grid.innerHTML = "";
  const items = state.signalFilter
    ? state.signals.filter((s) => s.type === state.signalFilter)
    : state.signals;

  items.forEach((s) => {
    const card = el("article", `signal ${s.severity}`);
    const meta = el("div", "signal-meta");
    meta.appendChild(el("span", `sev ${s.severity}`, s.severity));
    meta.appendChild(el("span", null, s.type.replace(/_/g, " ")));
    if (s.accounts.length) meta.appendChild(el("span", null, s.accounts.join(" · ")));
    card.appendChild(meta);
    card.appendChild(el("h3", null, s.title));
    card.appendChild(el("p", null, s.detail));

    if (s.evidence && s.evidence.length) {
      const ev = el("div", "evidence");
      s.evidence.slice(0, 6).forEach((e) =>
        ev.appendChild(el("span", "ref", e.ticket_id || e.order_id || "")));
      card.appendChild(ev);
    }
    if (s.recommended_action) {
      const a = el("div", "signal-action");
      a.appendChild(el("b", null, "Recommended"));
      a.appendChild(document.createTextNode(s.recommended_action));
      card.appendChild(a);
    }
    const ask = el("button", "signal-ask", "Ask the assistant about this");
    ask.addEventListener("click", () => {
      $("#input").value = `Tell me about "${s.title}" and what I should do about it.`;
      showView("chat");
      send();
    });
    card.appendChild(ask);
    grid.appendChild(card);
  });

  if (!items.length) grid.appendChild(el("p", "trace-empty", "Nothing detected in this category."));
  document.querySelectorAll("#signalFilters button").forEach((b) => {
    const label = b.textContent.split(" (")[0];
    const match = state.signalFilter
      ? label === state.signalFilter.replace(/_/g, " ")
      : label === "All";
    if (match) b.classList.add("on");
  });
}

/* ── audit ────────────────────────────────────────────────────────── */
async function loadAudit() {
  const res = await fetch("/api/audit?limit=120");
  const data = await res.json();
  const table = $("#auditTable");
  table.innerHTML = "";
  const head = el("div", "audit-row head");
  ["identity", "tool", "detail", "outcome"].forEach((h) => head.appendChild(el("div", null, h)));
  table.appendChild(head);

  if (!data.entries.length) {
    table.appendChild(el("p", "trace-empty",
      "No tool calls yet. Ask the assistant something and every call will be recorded here."));
    return;
  }
  data.entries.forEach((e) => {
    const row = el("div", "audit-row");
    row.appendChild(el("div", null, `${e.user_id} (${e.role})`));
    row.appendChild(el("div", null, e.tool));
    row.appendChild(el("div", "audit-detail", e.detail || e.args));
    row.appendChild(el("div", `audit-outcome ${e.outcome}`, e.outcome));
    table.appendChild(row);
  });
}

/* ── views ────────────────────────────────────────────────────────── */
function showView(view) {
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.view === view;
    t.classList.toggle("is-active", on);
    t.setAttribute("aria-selected", String(on));
  });
  document.querySelectorAll(".stage").forEach((s) => { s.hidden = s.dataset.view !== view; });
  if (view === "signals") loadSignals();
  if (view === "audit") loadAudit();
}

document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => showView(t.dataset.view)));
$("#refreshAudit").addEventListener("click", loadAudit);
$("#clearTrace").addEventListener("click", () => {
  $("#trace").innerHTML = "";
  $("#trace").appendChild(el("p", "trace-empty", "Cleared."));
});

$("#composer").addEventListener("submit", (e) => { e.preventDefault(); send(); });
$("#input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
$("#input").addEventListener("input", (e) => {
  e.target.style.height = "auto";
  e.target.style.height = Math.min(e.target.scrollHeight, 180) + "px";
});

boot();
