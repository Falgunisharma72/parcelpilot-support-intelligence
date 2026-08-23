#!/usr/bin/env python
"""Regenerate the rules registry from the document pack, with Claude.

This is the offline half of the pattern the runtime depends on:

    PDFs  ──►  LLM extraction  ──►  reviewed rules.yaml  ──►  deterministic runtime

Onboarding a new policy document is an extraction and review step, not a code
change - but the *serving* path never asks a model what a threshold is. The
model proposes; a human reviews the diff; the runtime then applies fixed numbers
and re-verifies every anchor against the parsed PDF text on every startup.

Two properties make the review tractable:
  * the model must quote a verbatim `anchor` from the clause it used, and this
    script rejects any proposal whose anchor is not actually in that clause -
    a fabricated threshold cannot survive the check;
  * output is written to a separate file so the change is always a diff against
    the reviewed registry, never an in-place overwrite.

    python -m scripts.extract_rules                       # -> rules.proposed.yaml
    python -m scripts.extract_rules --out somewhere.yaml
    diff app/knowledge/rules.yaml app/knowledge/rules.proposed.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import anthropic
import yaml

from app.config import ANTHROPIC_API_KEY, MODEL, RULES_FILE
from app.ingest.docs import load_corpus

EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rules"],
    "properties": {
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "value", "clause_id", "anchor", "note"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Dotted path in the registry, e.g. "
                                       "'cancellation.default.BOOKED.free_window_minutes' or "
                                       "'account_overrides.ACCT-002.service_credit.failed_pickup."
                                       "threshold_hours_past_window_end'.",
                    },
                    "value": {
                        "type": "string",
                        "description": "The value as it should appear in YAML (a number, a "
                                       "boolean, or a short string).",
                    },
                    "clause_id": {"type": "string", "description": "Exact clause_id it came from."},
                    "anchor": {
                        "type": "string",
                        "description": "A verbatim substring of that clause's text, long enough "
                                       "to be unambiguous, that states this value.",
                    },
                    "note": {"type": "string", "description": "One line: what this rule governs."},
                },
            },
        }
    },
}

PROMPT = """Below is every clause parsed from ParcelPilot's document pack, with its
clause_id, source document, authority tier and text.

Extract every rule parameter a support decision could turn on: cancellation
windows and fees, service-credit thresholds, amounts and formulas, approval
thresholds, monthly caps, first-response targets per plan and per contract,
plan capabilities, and known-issue thresholds.

Rules:
- Extract only what a clause literally states. If a number is not in the text,
  do not produce a rule for it.
- Every rule must cite the clause_id it came from and quote a verbatim anchor
  from that clause's text. The anchor is checked against the source; an
  invented one is rejected.
- Do not extract anything from a DEPRECATED document.
- Where a customer agreement states a target or threshold, path it under
  account_overrides.<ACCOUNT_ID>.

CLAUSES
=======
{clauses}
"""


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RULES_FILE.parent / "rules.proposed.yaml"))
    args = parser.parse_args()

    if not ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY is not set.")
        return 2

    clauses = load_corpus()
    rendered = "\n\n".join(
        f"[{c.clause_id}] tier={c.authority_tier} doc={c.doc_id} status={c.status} "
        f"scope={c.account_scope or 'global'}\nsection: {c.section}\n{c.text}"
        for c in clauses
    )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    with client.messages.stream(
        model=MODEL,
        max_tokens=32000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high",
                       "format": {"type": "json_schema", "schema": EXTRACTION_SCHEMA}},
        messages=[{"role": "user", "content": PROMPT.format(clauses=rendered)}],
    ) as stream:
        message = stream.get_final_message()

    payload = json.loads("".join(b.text for b in message.content if b.type == "text"))
    by_id = {c.clause_id: c for c in clauses}

    accepted, rejected = [], []
    for rule in payload["rules"]:
        clause = by_id.get(rule["clause_id"])
        if clause is None:
            rejected.append((rule, "unknown clause_id"))
        elif _normalise(rule["anchor"]) not in _normalise(clause.text):
            rejected.append((rule, "anchor not present in the cited clause"))
        elif clause.status == "DEPRECATED":
            rejected.append((rule, "sourced from a superseded document"))
        else:
            accepted.append(rule)

    out = Path(args.out)
    out.write_text(yaml.safe_dump(
        {"extracted": accepted, "rejected": [{"rule": r, "reason": why} for r, why in rejected]},
        sort_keys=False, width=100))

    print(f"{len(accepted)} rules accepted, {len(rejected)} rejected -> {out}")
    for rule, why in rejected:
        print(f"  rejected {rule['path']}: {why}")
    print(f"\nReview the diff against {RULES_FILE} before promoting anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
