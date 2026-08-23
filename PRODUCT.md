# Product note

## Which additional problem I chose

**Both** — because on this data they turn out to be the same problem seen from
two ends.

### Problem 2: Trust and reliability

Trust was the deciding constraint, so it shaped the architecture rather than
sitting on top of it. Four mechanisms, in the order they matter:

1. **The model does not do arithmetic or precedence.** Fees, credit amounts,
   elapsed times, SLA clocks and contract-versus-policy resolution are computed
   in code and returned as verdicts with their working shown. This is the
   difference between an answer that is usually right and one that is the same
   every time and can be checked.
2. **Every threshold is pinned to a verbatim quote from the document it came
   from, and startup re-verifies it.** If a policy is replaced and a number
   moves, the app refuses to start. Silent drift is the failure mode that
   destroys trust slowest and most completely, because nothing looks wrong.
3. **Conflicts are surfaced, not resolved silently.** When a contract displaces
   an SOP, the answer says so and shows both. "You'd normally be charged INR 250,
   but your agreement waives it" is a more trustworthy sentence than "no fee."
4. **Uncertainty is an outcome, not a gap to fill.** Unknown carrier fault
   returns `needs_verification`. Credits above INR 1,000 return `needs_human`.
   A breached SLA is stated plainly, as the policy requires, rather than softened.

The system also **finds guidance that was already wrong**. Both historical
resolutions in the pack are incorrect under current rules — one told Northstar a
cancellation fee applied that their contract waives, one told LumenWorks their
plan caps at 3,000 rows when the product limit is 5,000 and the failures are a
known bug. Every customer who was told one of those is still acting on it.

### Problem 1: Proactive issue detection

Which is where the second problem starts: finding those two answers is not
something a customer will ever ask for. The Signals view runs eight detectors
across the whole support surface at the snapshot and returns 13 findings:

| Detector | Finds on this data |
|---|---|
| `sla_breach` | TKT-501 (Northstar P1, 15-min contractual target, breached by 15 min), TKT-505 (Axis Labs P1, breached by 2 h) |
| `p1_open` | Both of the above — policy says escalate immediately regardless of remaining time |
| `known_issue_cluster` | TKT-502 + TKT-451 both trace to KI-208 |
| `recurring_issue` | TKT-502 is TKT-451 coming back five days after it was closed |
| `stale_guidance` | TKT-450 and TKT-451 — past answers current rules contradict |
| `overdue_pickup` | ORD-2002, 4 h 30 m overdue, carrier fault, INR 300 already owed |
| `cancellation_spike` | 4 cancellations across 3 accounts in 45 minutes |
| `awaiting_reply` | Three open tickets where the customer replied last |

Detectors are **deterministic on purpose**. An LLM-generated "here's what looks
worrying" list is unrepeatable, and you cannot page a human off something that
changes every run. The model's role is to triage and explain the findings, not
invent them — which is why each card has "Ask the assistant about this", handing
the signal to the chat with full context.

The two problems meet at TKT-505: a security ticket that reads like a question,
is P1 by policy, and has been sitting 2.5 hours past a 30-minute target. Nobody
asked about it. That is the case for a proactive view.

---

## What I would build next, in priority order

**1. Close the loop on outbound correction.** Right now the system *finds* the
two customers who were given wrong answers. It should draft the correction, route
it to the CSM, and track whether it was sent. Finding a problem you do not fix is
only half a feature, and this one has direct revenue exposure — a customer who
was wrongly told a fee applied may have paid it.

**2. Suggested replies in the agent's inbox.** The internal assistant currently
answers questions. The higher-leverage version drafts the actual customer reply
for the ticket the agent has open, pre-loaded with the verdict and citations, for
the agent to edit and send. That is where a 20-person team gets time back — it
turns the product from a reference tool into part of the workflow.

**3. Contract ingestion as a first-class flow.** Today a new customer agreement
means running the extraction script and reviewing a diff. It should be: upload
the PDF, see the extracted terms side by side with the clause each came from,
approve or correct each one. Contracts are where the money is and where the
overrides live; onboarding them should be a product surface, not a developer task.

**4. Answer-quality feedback wired to the eval set.** Every escalation is a
signal about where the agent's boundary sits, and every thumbs-down is a
candidate golden case. Without this loop the eval set only ever contains failures
someone thought of in advance.

**5. Real-time signals instead of snapshot-time.** The detectors run against a
frozen snapshot. In production they run on a schedule, hold state so an alert
fires once rather than every cycle, and push to Slack — a dashboard nobody opens
is not a proactive system.

**6. Fault attribution from carrier data.** The single most common
`needs_verification` outcome is unknown carrier fault. Every one is a human
round-trip. Pulling carrier scan events would convert a large share of "I need to
check" into decided answers.

---

## What I intentionally left out

* **A vector database.** 39 clauses. It would have added infrastructure and
  non-determinism to the eval set to solve a problem this corpus does not have.
  The scaling path is documented, and `search()` is the seam.
* **Real authentication.** The brief permits mocked auth. The *shape* is right —
  principals are resolved server-side from a session, never from a
  client-supplied account id — so swapping the directory for OIDC does not touch
  the enforcement path. Building real auth would have proved nothing about the
  interesting problem.
* **Actually executing anything.** Escalations, tasks and ticket updates write to
  local tables. Integrating a real ticketing system is plumbing.
* **Multi-turn action editing.** You can confirm or decline a proposal, not amend
  it inline. Amending means re-validating against data that may have moved, which
  is the same problem the TTL solves — worth doing properly, not quickly.
* **An LLM-based reconciliation pass over historical resolutions.** The
  stale-guidance detector uses explicit numeric checks per topic. It catches both
  cases in the pack and will miss a prose-only bad answer with no numbers in it.
  The general version needs the eval set from item 4 before it can be trusted.
* **Streaming partial verdicts.** Decision tools return atomically. Sub-second
  latency was not worth the added failure modes.

---

## The metric

**Resolved without a human, and still correct** — the share of conversations that
end in a confident answer, no escalation, and no correction or contradiction in
the following seven days.

Deflection alone is the wrong metric and actively dangerous here: a system that
confidently answers everything scores perfectly right up until the first customer
is told a fee applies that their contract waives. Accuracy alone is also wrong —
escalating everything is perfectly accurate and worth nothing.

The conjunction is what the product is for. It only improves by being right about
more things, and it is measurable from data the team already produces: escalation
rate, ticket reopens, and CSM corrections.

Two supporting metrics, both leading indicators:

* **Time-to-first-touch on P1s**, since that is what the proactive view exists to
  compress. On this snapshot two P1s were already breached and nobody had noticed.
* **Bad guidance caught before the customer acts on it** — count of stale-guidance
  findings corrected proactively. It measures the trust layer directly.
