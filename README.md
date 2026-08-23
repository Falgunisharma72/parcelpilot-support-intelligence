# ParcelPilot Support Intelligence

An AI support system for ParcelPilot, a B2B logistics platform. It ships **both**
user contexts asked for in the brief — a customer-facing support chatbot and an
internal support/operations assistant — over one agent core, plus a proactive
operations view and a deterministic trust layer.

The whole thing answers from the supplied document pack and workbook. Nothing is
hard-coded to the example IDs: the PDFs are parsed into attributed clauses and
the workbook into a scoped SQLite database at startup.

```
git clone <this repo> && cd parcelpilot
make setup                       # venv + dependencies
cp .env.example .env             # paste ONE free API key (see below)
make providers                   # confirms it works, incl. tool calling
make run                         # http://localhost:8000
```

**It runs entirely on a free tier.** The model backend is pluggable and
auto-detected from whichever key is present:

| Provider | Free tier | Key |
|---|---|---|
| **Groq** — recommended | Generous, no card, very fast | `GROQ_API_KEY` — [console.groq.com/keys](https://console.groq.com/keys) |
| Google Gemini | Free from AI Studio, no card | `GEMINI_API_KEY` — [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| Cerebras | Free tier, no card | `CEREBRAS_API_KEY` |
| OpenRouter | Models suffixed `:free` | `OPENROUTER_API_KEY` |
| Mistral / Together | Free experiment tier / starting credit | `MISTRAL_API_KEY` / `TOGETHER_API_KEY` |
| Ollama | Free forever, local, no key at all | `ollama pull qwen2.5:7b` |
| Anthropic | Paid — optional, not required | `ANTHROPIC_API_KEY` |

A free 70B model is enough here *because the model does not compute anything*.
Fees, credit amounts, elapsed times, SLA clocks and contract-versus-policy
precedence are decided in code and handed over as finished verdicts — the model
picks the right tool and narrates the result. That is a much smaller ask than
"reason correctly about overlapping contracts", which is where free models fail.

`make test` runs **96 tests and needs no API key at all** — including the full
agent loop, driven by a scripted provider.

---

## What it does

| | |
|---|---|
| **Customer chat** | Answers entitlement, cancellation, credit and SLA questions for the signed-in account only. Escalates what needs a human. |
| **Internal chat** | Same core, staff scope: all accounts, internal fields, cross-account signals, and three state-changing tools. |
| **Signals** | Proactive detection across the whole support surface — breaches, open P1s, known-issue clusters, recurrences, overdue pickups, and past answers that current rules contradict. |
| **Access log** | Every tool call, the identity that made it, and whether the data layer allowed it. |

### The two example questions

> **Can Northstar cancel ORD-1001 without a cancellation fee?**
> Yes. Booked 09:00, cancellation requested 11:00 — two hours, so the SOP's
> 30-minute free window has passed and INR 250 *would* apply. Northstar's signed
> agreement waives the cancellation fee for any BOOKED shipment before pickup
> regardless of elapsed time, and the agreement outranks the SOP. The system also
> flags KI-211: this is a SwiftShip order still showing BOOKED, and SwiftShip
> pickup webhooks run up to 20 minutes late, so the carrier status is worth
> checking before cancelling.
>
> A closed ticket in the pack (TKT-450) records an agent telling Northstar the
> INR 250 fee applied. That answer was wrong, and the system says so instead of
> repeating it.

> **A pickup is three hours late through carrier fault. Do I get a credit?**
> It depends entirely on who is asking, which is why there is no single answer:
> * **Beacon Retail / Axis Labs / Northstar** — yes. The SOP threshold is 2 hours;
>   the credit is the lower of INR 500 or 10% of the shipment fee.
> * **LumenWorks** — no. Their agreement replaces the threshold with 4 hours and
>   the amount with a flat INR 300. The answer explains that the general policy
>   *would* have qualified, and why it does not apply to them.

---

## How it is put together

```
                       ┌──────────────────────────────────────────┐
  browser  ──SSE──►    │  FastAPI  ·  /api/chat  /api/confirm     │
                       └───────────────┬──────────────────────────┘
                                       │
                        ┌──────────────▼───────────────┐
                        │  Provider-neutral agent loop │   judgement, wording,
                        │  Groq / Gemini / OpenRouter  │   when to escalate
                        │  Cerebras / Ollama / Claude  │
                        └──────────────┬───────────────┘
                                       │  12 tools, filtered by permission
      ┌────────────────┬───────────────┼────────────────┬──────────────────┐
      ▼                ▼               ▼                ▼                  ▼
  Retrieval       Data gateway    Decision engines   Signals          Proposals
  BM25 + authority  SQL scoping   cancellation       8 detectors      two-phase
  ranking, conflict field          credits                            commit;
  detection         redaction      SLA + severity                     the only
                                   (all deterministic)                write path
      │                │               │                │                  │
      └────────┬───────┴───────────────┴────────────────┴──────────────────┘
               ▼
   6 PDFs → 39 attributed clauses  ·  workbook → SQLite  ·  rules.yaml (anchor-verified)
```

The split that matters: **judgement is the model's, arithmetic and precedence are
the code's.** `check_cancellation` does not hand the model policy paragraphs and
hope — it returns a decided verdict with the calculation, the rule applied, the
rule overridden, and citations. The model decides what to say and when to
escalate; it cannot quietly substitute its own number.

Full reasoning in **[ARCHITECTURE.md](ARCHITECTURE.md)**; product decisions and
what was left out in **[PRODUCT.md](PRODUCT.md)**.

---

## Tools

| Tool | Category | Notes |
|---|---|---|
| `search_policy_documents` | documents | Authority-ranked, account-scoped, superseded docs excluded |
| `find_known_issues` | documents | Matches a symptom to KI-208 / KI-211 and its workaround |
| `get_account_context` | data | Plan, contract, and the overrides that displace general rules |
| `lookup_orders` / `lookup_tickets` | data | Row- and field-scoped in SQL |
| `check_cancellation` | decision | Deterministic verdict + working |
| `check_service_credit` | decision | Real order or stated facts; per-account thresholds |
| `check_sla` | decision | Severity classification, contract targets, business-hours clock |
| `get_operational_signals` | signals | Staff only |
| `propose_escalation` | action | Confirmation-gated |
| `propose_ticket_update` | action | Staff only, confirmation-gated |
| `propose_followup_task` | action | Staff only, confirmation-gated |

Customers are offered nine of these; the internal context gets all twelve. The
dispatcher re-checks the permission regardless of what the model calls.

---

## Access control

Enforced in the data layer, not the prompt.

* **Row scoping in SQL.** A customer's query is rewritten with
  `WHERE account_id = ?` before it reaches SQLite. Other accounts' rows are never
  materialised in the process.
* **Field scoping on the way out.** Even within their own account, a customer
  never receives `assigned_to`, CSM notes, or `historical_resolution`.
* **Scope cannot be widened by argument.** A customer principal resolves to its
  own account no matter what `account_id` the model passes — so a prompt
  injection inside a ticket description has nothing to act on.
* **No enumeration oracle.** "Not yours" and "does not exist" return the identical
  message.
* **Everything is audited**, allowed or denied, and visible in the Access log tab.

`tests/test_access_control.py` attacks each of these.

## Confirmation before actions

The three action tools are proposal builders — they validate, resolve the target,
render a preview, and **write nothing**. The code that mutates data sits behind
`ProposalStore.confirm()`, reachable only from an authenticated `POST /api/confirm`.

So confirmation is a property of the architecture, not an instruction the model
could be talked out of. Proposals are single-use, TTL-bounded, and bound to the
principal *and* session that were shown the preview.
`tests/test_confirmation.py` proves each.

---

## Testing

```
make test      # 96 tests, no API key required
make verify    # every rule threshold still matches its clause in the PDFs
make providers # list free providers, and probe that tool calling works
make eval      # 15-case golden set against the live agent (needs a key)
```

The 96 tests include the agent loop itself — tool dispatch, access enforcement,
the confirmation interrupt, the step guard — driven by a scripted provider, plus
the OpenAI-compatible adapter against a mock endpoint that reproduces how free
providers actually stream (tool arguments split across deltas, missing `index`,
malformed JSON). Only `make eval` needs a real key.

`make eval` is the one that catches prompt regressions: each case is a question
with a plausible wrong answer, asserted on the tool the agent reached for, what
the answer must and must not contain, and whether anything was written.

---

## Deploying

Any Docker host. The image parses the PDFs and verifies every rule anchor at
**build** time, so a drifted registry fails the build rather than shipping.

```
make docker && make docker-run
```

* **Render** — `render.yaml` is a blueprint; New → Blueprint → pick the repo,
  paste `ANTHROPIC_API_KEY`. Free tier, no card.
* **Hugging Face Spaces** — Docker Space; the container honours port 7860.
* **Fly / Cloud Run** — `$PORT` is honoured.

---

## Notes on the data

* **All times are measured against the workbook's snapshot** (16 Aug 2026,
  11:00 IST), read from the README sheet at startup — not wall-clock `now()`.
  A system that drifts to real time gives different answers to the same question
  every day.
* **That date is a Sunday.** Every "business hours" target behaves differently
  because of it, and LumenWorks' agreement excludes weekend coverage entirely.
  Business hours are assumed to be 09:00–18:00 IST Mon–Fri — the documents never
  define them, so the assumption is pinned once and reported with every SLA answer.
* **Historical ticket resolutions are treated as context, never authority.** Both
  in the supplied pack are wrong under current rules, and the Signals view surfaces
  them as a finding in their own right.

## Layout

```
app/
  ingest/      PDFs -> attributed clauses; workbook -> SQLite
  knowledge/   retrieval + ranking; rules.yaml + anchor verification
  core/        principals & permissions, scoped gateway, business-time, proposals
  engine/      cancellation, credits, severity, SLA, signals
  agent/       tool surface, prompts, provider-neutral streaming loop
  static/      the console UI
  agent/providers/  pluggable model backends (free-tier first)
tests/         96 tests, no API key needed
evals/         golden set + runner
scripts/       LLM-assisted rules extraction (offline)
```
