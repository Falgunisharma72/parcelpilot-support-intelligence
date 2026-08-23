# Architecture note

## The problem this shape solves

The brief's hard part is not building a chatbot over documents. It is that the
source base is deliberately unreliable — a superseded policy sits next to the
current one, customer agreements silently override general rules, and two of the
historical ticket resolutions in the pack are simply wrong. A retrieval-and-
generate system over that corpus will produce fluent, well-cited, confidently
incorrect answers, because every wrong answer available has a real document
behind it.

So the central design decision is **where judgement lives**:

> The model owns judgement, wording, and when to involve a human.
> Code owns arithmetic, precedence, and access.

`check_cancellation` does not return policy paragraphs about cancellation and
hope the model reasons well. It returns a decided verdict — the fee, the
calculation, the rule applied, the rule overridden, the citations, the caveats.
The model narrates it. That one boundary is what makes the system testable
(118 tests, no API key), repeatable (same input, same answer, every run), and
auditable (every number traces to a clause).

---

## Agent design

A hand-written, **provider-neutral** agentic loop rather than an SDK helper.
Three requirements drove that:

1. The UI shows **which tool is running, as it runs** — so the loop emits an
   event before executing, not after.
2. A state-changing tool returns a *proposal* that must interrupt the turn and
   surface as a confirmation card, rather than being swallowed as another tool
   result.
3. Every tool call is audited **with the calling principal**, so the loop owns
   the principal rather than each tool function closing over it.

The loop is capped at 12 steps and degrades to a plain error that says nothing
was changed.

### The provider layer

The loop speaks a neutral conversation format, and each backend adapts it. One
adapter covers Anthropic; a second covers the OpenAI chat-completions dialect,
which Groq, Gemini, OpenRouter, Cerebras, Mistral, Together and a local Ollama
all speak — so every free option is one integration, not seven. The provider is
auto-detected from whichever key is present.

Why neutral rather than "just use the OpenAI shape everywhere": Anthropic's
thinking blocks must be replayed byte-identical on the next turn and the OpenAI
shape has nowhere to put them. So each assistant turn carries an optional
provider-native `raw` payload that the loop never looks inside.

**This is the decision that makes a free tier viable.** Because the model never
computes a fee, an elapsed time or a precedence outcome, its job reduces to
picking the right tool and narrating a finished verdict. A free 70B model does
that acceptably; the same model asked to reason unaided about overlapping
contracts would not. The deterministic core is what buys the cheap model.

Working against free endpoints needs defensiveness that a frontier model does
not, and it is contained in the adapter rather than leaking into the loop. Each
of these came from a real run, not from imagination:

* tool arguments arrive as a JSON *string* and are not always valid JSON;
* `index` on streamed tool-call deltas is sometimes absent;
* `stream_options` is rejected outright by some endpoints;
* reasoning text appears under two different field names;
* optional parameters are sent as explicit `null`, and strict providers reject
  the whole turn — so optional params declare `["string", "null"]`;
* **Gemini 3.x attaches a `thought_signature` to every tool call and 400s the
  next request if it is not echoed back.** This is the same class of problem as
  replaying Anthropic thinking blocks, which is why the neutral format already
  carried provider-native data through the round trip: supporting it needed one
  field on `ToolCall` and two lines in one adapter;
* a model without tool support fails with a 400 that has to become an actionable
  message, and a retired model id has to report the catalogue the key can reach.

`tests/test_providers.py` reproduces each against a mock endpoint, including
throttling and quota exhaustion.

### Fitting inside a free tier

Free tiers meter tokens per minute and count the *requested* `max_tokens`, so
request size is a correctness concern, not just a cost one. Two consequences:

**The model-facing payload is a projection of the UI-facing one.** The trace
panel wants every citation in full and each ranking reason; the model needs the
decision, the arithmetic and a short citation label. Sending one payload to both
pushed requests past Groq's 8,000-token ceiling. Splitting them halved request
size and also stopped the model re-deriving an answer from raw clause text when
a decided verdict was sitting beside it.

**A daily quota is not a burst limit.** Throttling clears in seconds and is
retried with backoff — but only before the first token reaches the user, since
retrying after that would duplicate visible output. Quota exhaustion fails
immediately and names the provider switch.

On Anthropic specifically: adaptive thinking, `effort: high`, and server-side
refusal fallbacks so a classifier decline routes to a comparable model rather
than dead-ending a support user. `stop_reason == "refusal"` is handled
explicitly rather than read as an answer.

**One agent, two contexts.** The customer and internal assistants share the core
and differ in three places: the principal, the tool list (9 vs 12), and the
persona section of the system prompt. Two separate agents would have meant two
places to fix every precedence bug.

**The prompt is not load-bearing for safety.** Access control lives in the data
layer and confirmation in the proposal layer, so the prompt can be tuned for
helpfulness without anyone re-auditing whether a customer can read another
customer's contract.

## Tool design

Twelve tools in five categories — documents, data, decision, signals, action.
Three principles:

* **Filtered by permission before the model sees them.** A customer session is
  never offered `get_operational_signals`. Hiding is not security, so the
  dispatcher re-checks — but not offering also stops the model wasting a turn.
* **Decision tools return verdicts, not evidence.** The distinction between
  "here are three clauses about credits" and "not eligible: 3.00 hours does not
  exceed the 4-hour threshold in *your* agreement, though it does exceed the
  general SOP's 2 hours" is the whole product.
* **No tool writes.** The three `propose_*` tools build proposals.

`check_service_credit` works from a real order *or* from stated facts, because
the brief's own example — "a pickup is three hours late because of carrier
fault" — has no order attached and still has a different right answer per
account.

## Document handling

Each PDF is parsed into **clauses** (a numbered section or a bullet within one),
and each clause carries the provenance needed to trust or distrust it: document,
`CURRENT`/`DEPRECATED`, effective date, account scope, and authority tier.

Chunking is structural rather than fixed-size, and two parser bugs found during
the build show why that matters:

* a minimum-length filter dropped `P2: 1 hour` — ten characters, and a binding
  contractual SLA target;
* KI-208 and KI-211 merged into one chunk, so a citation could no longer point
  at the specific incident.

Both were caught by the anchor verifier (below) rather than by reading output.

### Why BM25, not a vector database

The corpus is 6 one-page documents → 39 clauses. A vector store would add an
index, an embedding provider, a network hop and a similarity threshold to tune,
in exchange for nothing measurable. At this size lexical search with field
boosts is *more* accurate — real queries contain exact tokens like `PICKED_UP`,
`KI-208`, `ACCT-001`, `INR 250` — and fully deterministic, which the golden-set
eval depends on.

What actually determines quality here is **ranking by authority and scope**, not
the similarity metric:

| Tier | Source | Behaviour |
|---|---|---|
| 1 | Signed customer agreement | ×1.6, plus ×1.35 when scoped to this account |
| 2 | Current policy / SOP | ×1.25 |
| 3 | Current product documentation | ×1.0 |
| 4 | Historical tickets, internal notes | context only, never authority |
| — | Superseded | ×0.05, and filtered out entirely by default |

Account scoping is a **hard filter applied before ranking**: another customer's
contract clauses are removed from the candidate set, so no prompt can surface them.

`search()` is the seam. Swapping the candidate generator for hybrid BM25 +
vector recall leaves the authority, scoping and conflict layers untouched — the
scaling path is written down rather than pre-built.

## Structured-data handling

The workbook is loaded into **SQLite** at startup, not into dataframes. The
reason is access control: `WHERE account_id = ?` in the data layer is
enforceable and auditable, whereas filtering a Python list *after* loading
everything means the unfiltered rows existed in the agent's process, one bug
away from a leak. It also gives the action tools a real place to write and a
durable audit table.

Two things are computed, never inferred:

* **Business-hours arithmetic.** Targets come in three units — "30 minutes,
  24x7", "2 business hours", "1 business day". The snapshot is a **Sunday**, so a
  business-clock target has not started while a 24x7 one is already breached.
  The documents never define business hours; the assumption (09:00–18:00 IST,
  Mon–Fri) is pinned once and reported with every SLA result.
* **Contract coverage qualifies the whole contract.** LumenWorks' agreement
  excludes weekend and after-hours support, so *every* target in that agreement
  runs on the business clock — not just the clause the exclusion sits next to.
  Applying one clause of a contract while ignoring another is how you produce an
  answer that is contractually wrong.

## Source reliability and conflict handling

Four mechanisms, in increasing order of how much they cost to build and how much
they matter:

**1. Tiering and filtering.** Superseded documents never surface as an answer.
Asking for them explicitly returns them with a loud warning attached.

**2. Conflict detection.** When retrieved clauses speak to the same *normative*
topic from different tiers, the result carries an explicit conflict record: what
wins, what yields, and why. Restricted to decision-bearing topics — two
documents both explaining what `BOOKED` means is not a conflict, and flagging it
as one trains the reader to ignore the banner.

**3. The rules registry, verified against the documents.** Thresholds live in
`app/knowledge/rules.yaml` — but every entry carries a `clause_id` and a
verbatim `anchor`, and **startup re-parses the PDFs and asserts every anchor is
still present**. Replace a document with a version where a threshold moved, and
the app refuses to start rather than answering from a stale number. The Docker
build runs the same check, so a drifted registry fails the build.

`scripts/extract_rules.py` regenerates the registry from the PDFs with Claude,
rejecting any proposal whose anchor is not verbatim in the cited clause. So
onboarding a new document is extract → review the diff → promote, not a code
change — while the serving path still never asks a model what a number is.

**4. Uncertainty as a first-class outcome.** Verdicts carry `confidence`,
`caveats`, `assumptions`, and `needs_human` with a reason. Unknown carrier fault
returns `needs_verification`, never a guess in either direction — the SOP
explicitly forbids promising a credit on unknowns, and that rule is in code.

## Access control

* Row scoping in SQL; field scoping on the way out (`assigned_to`, CSM notes and
  `historical_resolution` never reach a customer).
* A customer principal resolves to its own account **regardless of the
  `account_id` argument the model passes** — the mitigation for a prompt
  injection inside a ticket description.
* "Not yours" and "does not exist" are the same message — otherwise the error is
  an enumeration oracle.
* Every call audited with principal and outcome, exposed at `/api/audit` and in
  the UI, because an access-control story you cannot inspect is a claim rather
  than a control.

## Confirmation before actions

Two-phase commit. Proposal builders validate and preview; `ProposalStore.confirm()`
is the sole write path and needs a proposal id that only an authenticated
`POST /api/confirm` supplies. Proposals are single-use (confirming twice does not
create two escalations), TTL-bounded (a confirmation against a stale preview is
refused), and bound to the principal *and* session that saw the preview. The
proposal id is written into the row, so every mutation points back at the
confirmation that authorised it.

## Major trade-offs

| Decision | Why | What it costs |
|---|---|---|
| Deterministic decision engines | Testable, repeatable, auditable; the difference between an answer and a defensible answer | New rule types need code, not just a document. Mitigated by the extraction script |
| BM25 over a vector store | More accurate at 39 clauses, zero infra, deterministic evals | Would need hybrid recall at ~10³ documents. `search()` is the seam |
| Rules in YAML, verified at startup | Fast, explicit, reviewable; drift fails loudly | A registry to maintain — which is the point: it is reviewable |
| SQLite | Real SQL scoping and a durable audit trail, no infra | Single-node. Swap the gateway for Postgres |
| In-process sessions and proposals | Zero dependencies for a single-instance deployment | Not horizontally scalable. Both are narrow interfaces — Redis is a one-file change |
| Keyword severity classifier | Transparent, reports its evidence, unit-testable | Brittle on unseen phrasing. The agent can override, and the timing is recomputed |
| Hand-written, provider-neutral loop | Per-tool UI events, proposal interception, per-principal auditing — and the whole agent runs on a free tier and is testable with a scripted provider | More code than an SDK tool runner, plus one adapter per dialect |
| Snapshot-anchored clock | Same question, same answer, every run | Not wired to a real clock — one constructor argument |

## What would be different at production scale

* Hybrid retrieval (BM25 + embeddings) with reranking, past a few hundred documents.
* Postgres with row-level security, so scoping is enforced by the database rather
  than by the gateway's discipline.
* Redis-backed sessions and proposals for horizontal scale.
* Real OIDC replacing the mocked principal directory — the shape is already right
  (principal resolved server-side from a session, never a client-supplied account id).
* The eval harness in CI, gating deploys on the golden set rather than on tests alone.
