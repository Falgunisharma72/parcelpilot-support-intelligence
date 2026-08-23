#!/usr/bin/env python
"""Run the golden set end-to-end against the live agent.

Unit tests prove the engines are right. This proves the *agent* behaves: that it
reaches for the correct tool, respects the boundary of the session it is in, and
does not narrate its way around a verdict. It is the check that would catch a
prompt regression, which no unit test can.

    python -m evals.run_evals             # all cases
    python -m evals.run_evals --case northstar-cancel-waiver
    python -m evals.run_evals --json report.json

Exit code is non-zero if any case fails, so it can gate a deploy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from app.agent.loop import Agent, Session
from app.agent.tools import ToolRuntime
from app.config import ANTHROPIC_API_KEY, MODEL
from app.core.db import DataGateway
from app.core.principal import resolve_principal
from app.core.proposals import ProposalStore
from app.knowledge.retrieval import ClauseIndex
from app.knowledge.rules import get_rules

GOLDEN = Path(__file__).parent / "golden.yaml"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def run_case(agent: Agent, gateway: DataGateway, case: dict) -> dict:
    principal = resolve_principal(case["user"])
    session = Session(session_id=f"eval_{case['id']}", principal=principal)

    before = gateway.conn.execute("SELECT COUNT(*) FROM escalations").fetchone()[0]
    before += gateway.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    before += gateway.conn.execute("SELECT COUNT(*) FROM ticket_updates").fetchone()[0]

    text, tools_used, proposals, errors = "", [], 0, []
    started = time.time()
    for event in agent.run(session, case["ask"]):
        if event["type"] == "text":
            text += event["text"]
        elif event["type"] == "tool_start":
            tools_used.append(event["name"])
        elif event["type"] == "proposal":
            proposals += 1
        elif event["type"] == "error":
            errors.append(event["message"])
    elapsed = time.time() - started

    after = gateway.conn.execute("SELECT COUNT(*) FROM escalations").fetchone()[0]
    after += gateway.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    after += gateway.conn.execute("SELECT COUNT(*) FROM ticket_updates").fetchone()[0]

    low = text.lower()
    failures: list[str] = []

    for tool in case.get("must_call", []):
        if tool not in tools_used:
            failures.append(f"did not call {tool} (called: {', '.join(tools_used) or 'nothing'})")
    for tool in case.get("must_not_call", []):
        if tool in tools_used:
            failures.append(f"called {tool}, which it should not")
    for needle in case.get("expect", []):
        if needle.lower() not in low:
            failures.append(f"answer is missing {needle!r}")
    for needle in case.get("expect_absent", []):
        if needle.lower() in low:
            failures.append(f"answer contains {needle!r}, which it should not")
    if case.get("expect_proposal") and proposals == 0:
        failures.append("expected a confirmation card, none was produced")
    if case.get("expect_no_write") and after != before:
        failures.append("a row was written without confirmation")
    if errors:
        failures.append(f"stream error: {errors[0]}")

    return {
        "id": case["id"], "passed": not failures, "failures": failures,
        "tools": tools_used, "proposals": proposals,
        "seconds": round(elapsed, 1), "answer": text.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="Run a single case by id")
    parser.add_argument("--json", help="Write a full report to this path")
    parser.add_argument("--verbose", action="store_true", help="Print every answer")
    args = parser.parse_args()

    if not ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY is not set. The engine tests (`make test`) run without "
              "a key; this harness needs one because it exercises the live agent.")
        return 2

    cases = yaml.safe_load(GOLDEN.read_text())["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"No case named {args.case!r}")
            return 2

    gateway = DataGateway()
    runtime = ToolRuntime(gateway, get_rules(), ClauseIndex(), ProposalStore())
    agent = Agent(runtime, api_key=ANTHROPIC_API_KEY)

    print(f"\n  ParcelPilot golden set · {len(cases)} cases · {MODEL}\n")
    results = []
    for case in cases:
        result = run_case(agent, gateway, case)
        results.append(result)
        mark = f"{GREEN}PASS{RESET}" if result["passed"] else f"{RED}FAIL{RESET}"
        print(f"  {mark}  {case['id']:<30} {DIM}{result['seconds']:>5.1f}s  "
              f"{', '.join(result['tools']) or '(no tools)'}{RESET}")
        for failure in result["failures"]:
            print(f"        {RED}·{RESET} {failure}")
        if args.verbose and result["answer"]:
            for line in result["answer"].splitlines():
                print(f"        {DIM}{line}{RESET}")

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    colour = GREEN if passed == total else (YELLOW if passed > total * 0.7 else RED)
    print(f"\n  {colour}{passed}/{total} passed{RESET}   "
          f"{DIM}{sum(r['seconds'] for r in results):.0f}s total{RESET}\n")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"model": MODEL, "passed": passed, "total": total, "results": results},
            indent=2))
        print(f"  report written to {args.json}\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
