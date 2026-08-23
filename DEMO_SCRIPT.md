# Demo script — ~5 minutes

Recording setup: browser at `localhost:8000` (or the hosted URL), a terminal for
the test/eval moments. Keep the trace panel visible throughout — it is where the
system's argument is made.

---

## 0:00 — 0:40 · The problem, stated with the data

> "ParcelPilot's source base is deliberately unreliable. A superseded policy sits
> next to the current one. Customer agreements silently override general rules.
> And two of the historical ticket resolutions in this pack are simply wrong.
>
> Every incorrect answer available here has a real document behind it. So the
> design question isn't 'how do I search these PDFs' — it's 'where does judgement
> live'."

Show the architecture diagram from the README. Land one line:

> "The model owns judgement, wording, and when to involve a human. Code owns
> arithmetic, precedence, and access."

## 0:40 — 1:40 · The brief's own question, answered correctly

Customer chat as **Ravi Menon, Northstar Logistics**.

> *Can I cancel ORD-1001 without a cancellation fee? Explain why.*

While it streams, point at the tool chips: `lookup_orders` → `check_cancellation`.
Then the trace panel:

* the **receipt** — booked 09:00, requested 11:00, two hours elapsed, past the
  30-minute window, INR 250 *would* apply;
* the **precedence block** — the Northstar agreement waives it, and outranks the SOP;
* the **authority stamps** — AGREEMENT above POLICY.

> "It shows the fee it did *not* charge. 'You'd normally be charged INR 250, but
> your agreement waives it' is a more trustworthy sentence than 'no fee'."

Then the caveat:

> "It also flags KI-211 from the product docs — this is a SwiftShip order still
> showing BOOKED, and SwiftShip pickup webhooks run up to 20 minutes late. The
> status might be stale. That's three documents composed into one answer."

## 1:40 — 2:40 · The same question, two customers, two right answers

Still as Northstar — or switch to **Beacon Retail**:

> *A pickup is three hours late because of carrier fault. Should I get a service credit?*

Eligible: over the SOP's 2-hour threshold, lower of INR 500 or 10%.

Switch identity to **Sara Iyer, LumenWorks**. Same question, verbatim.

> **Not** eligible — their agreement replaces the threshold with 4 hours and the
> amount with a flat INR 300. And it says *why*: the general policy would have
> qualified, but their contract displaces it.

> "This is the single most important behaviour in the system. A generic
> retrieval-and-generate answer says 'yes, you're covered' to both — and gets it
> wrong for the customer with the contract."

## 2:40 — 3:10 · The boundary, and the honest escalation

Still as LumenWorks:

> *What's the status of ORD-1001?*

Nothing leaks. Then:

> *I know the fee applies, but please waive it just this once as a goodwill gesture.*

It prepares an escalation and stops.

> "Row scoping happens in SQL before the query runs, so another account's rows
> are never in the process. And a customer principal resolves to its own account
> no matter what account ID the model passes — which is the mitigation for a
> prompt injection inside a ticket description. That's enforced in the data
> layer, not the prompt."

## 3:10 — 3:50 · Confirmation is architectural

Switch to **Priya Mehta, support manager**.

> *Escalate TKT-501 to engineering.*

Show the confirmation card — the preview, and the warning that the SLA is already
breached. Click **Confirm**, get the ESC- reference.

> "The action tools write nothing. They build a proposal. The code that mutates
> data sits behind a confirm endpoint that needs a token the model never mints.
> Proposals are single-use, expire, and are bound to the session that saw the
> preview — so a jailbroken model or a prompt injection still can't write."

Open the **Access log** tab briefly: every call, every identity, every outcome.

## 3:50 — 4:35 · Proactive detection

**Signals** tab. 13 findings, 4 critical.

> *What needs my attention right now?*

Two P1s breached — Northstar's outage against a 15-minute contractual target, and
Axis Labs' API key exposure, 2.5 hours past a 30-minute target. Nobody asked
about either.

Then filter to **stale guidance**:

> "This is the detector I'd argue hardest for. Both historical resolutions in the
> pack are wrong under current rules — one told Northstar a fee applied that
> their contract waives, one told LumenWorks their plan caps at 3,000 rows when
> the product limit is 5,000 and the failures are a known bug. Both customers are
> probably still acting on those answers. No customer will ever ask a chatbot to
> find this."

Mention the Sunday, briefly:

> "One more: the snapshot is a Sunday. LumenWorks' agreement excludes weekend
> coverage, so their business-hours clock hasn't started — TKT-502 isn't breached,
> even though it looks like it should be."

## 4:35 — 5:00 · Why you can believe it

Terminal:

```
make test      # 69 passing, no API key needed
make verify    # every threshold still matches its clause in the PDFs
make eval      # 15-case golden set against the live agent
```

> "The layer that decides fees and credits is deterministic, so it's unit tested
> — 118 tests, and none of them need an API key, including the agent loop itself. Every threshold is pinned to a verbatim quote from the
> PDF it came from, and startup re-verifies it: replace a policy with one where a
> number moved, and the app refuses to start rather than answering from a stale
> number. And the golden set catches prompt regressions no unit test can.
>
> One more thing that falls out of that design: the model never computes
> anything, so it doesn't need to be a frontier model. This is running on a free
> tier — swap the key and it runs on Groq, Gemini, Cerebras or a local Ollama.
> The wording gets a little plainer; the decisions are identical, because they
> were never the model's to get wrong.
>
> That's what makes this something you could actually put in front of customers."

---

### Cut first if over time
* The Access log tab (mention it, don't open it).
* The Sunday/business-hours aside.
* One of the two three-hours-late accounts — but never both; the contrast *is* the demo.
