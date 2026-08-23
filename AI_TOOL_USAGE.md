# AI tool usage

**Claude Code (Claude Opus 5)** was used throughout, and the way it was used is
worth being specific about, because "an AI wrote it" and "an AI was directed to
build this" are different claims.

### What it did

* **Read the brief and the data pack directly.** The task document and all seven
  source files were opened, parsed and analysed in-session — the assessment PDFs
  through `pdfplumber`, the workbook through `openpyxl` — rather than summarised
  by hand into a prompt.
* **Wrote the implementation**, module by module, with each layer verified before
  the next was built on top of it: ingestion → retrieval → rules → engines →
  tools → agent loop → API → UI.
* **Wrote the tests and the golden set**, then ran them.

### What that verification actually caught

Every one of these was found by running the code, not by reading it:

1. A minimum-length filter in the PDF chunker silently dropped `P2: 1 hour` —
   ten characters, and a binding contractual SLA target. Half of both customer
   agreements' support terms were missing from the index.
2. Document classification matched the word "agreement" in body text, which
   promoted Support Policy v3 to *contract* authority — precisely inverting the
   precedence rule it defines.
3. `\bfail\b` did not match "fails", so a P2 bulk-upload ticket classified as P3.
4. The severity classifier treated "existing shipments can still be viewed" as a
   workaround and downgraded a total shipment-creation outage from P1 to P2.
   Being able to *read* records is not a workaround for being unable to *create*
   them.
5. The write path lost the proposal id linking a row back to the confirmation
   that authorised it — caught by a test, not by review.
6. Conflict detection fired on descriptive topic overlap (two documents both
   explaining what `BOOKED` means), which would have trained users to ignore the
   conflict banner.
7. `app.js` guarded `$("#send")` with a truthiness check against an id that did
   not exist, so the composer silently never disabled while a turn streamed.

Running the agent against two real free tiers found what no mock would have:
Gemini 3.x rejects a follow-up request that omits the `thought_signature` from
the previous tool call; Groq's 8,000-token-per-minute ceiling counts the
requested `max_tokens`; strict providers reject explicit `null` for optional
parameters; and a *daily* quota needs different handling from a burst limit.
Three separate pinned model ids went stale during development.

Two of the bugs were in the eval harness rather than the agent: it matched
answers byte-literally, so correct answers failed on typographic punctuation
(U+202F, U+2011) and on synonyms. That is worth stating plainly, because
loosening your own assertions is also how you fake a passing eval — the
alternatives are written explicitly in `golden.yaml` for review.

The rules-anchor verifier — the startup check that every threshold still matches
a verbatim quote in its source PDF — found bugs 1 and 2 by refusing to start. It
was built as a production safeguard and paid for itself during development.

### Where the judgement was mine

The decisions that shaped the system were not delegated: putting arithmetic and
precedence in code rather than in the model; anchoring rules to verbatim document
quotes and failing startup on drift; enforcing access control in SQL and
confirmation in a two-phase commit rather than in the prompt; treating "past
answers that were wrong" as a first-class detector; and choosing BM25 over a
vector store for a 39-clause corpus.

The same applies to what was found in the data — that 16 Aug 2026 is a Sunday and
therefore every business-hours SLA answer changes; that LumenWorks' 4-hour
contract threshold makes the brief's own three-hours-late example resolve
*differently per account*; that both historical resolutions in the pack are wrong.
Those came from reading the sources against each other, and they are what the
system is built around.

### Other tools

Google Drive's MCP connector was used to pull the candidate data pack.

The running application does **not** depend on Claude. Its model backend is
pluggable and free-tier first — Groq, Gemini, Cerebras, OpenRouter, Mistral,
Together or a local Ollama — because the deterministic core means the model is
narrating verdicts rather than deriving them, so a free 70B model is sufficient.
Anthropic is one supported provider among several, not a requirement.
