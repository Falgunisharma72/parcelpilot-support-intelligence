#!/usr/bin/env python
"""Show which model providers are usable, and prove one end to end.

Free-tier model ids change, and a stale default surfaces as an opaque 404. This
answers the three questions that actually block someone: which key do I have,
which models can it reach, and does tool calling work on the one selected -
because the agent is useless without tool calling, and not every free model has it.

    python -m scripts.check_provider          # list, then test the active one
    python -m scripts.check_provider --list   # list only
"""
from __future__ import annotations

import argparse
import sys

from app.agent.providers import (
    PRESETS, ProviderError, available, build_provider, ollama_running, resolve,
)

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"

PROBE_TOOL = [{
    "name": "get_shipment_status",
    "description": "Look up the current status of a shipment by its order id.",
    "input_schema": {
        "type": "object",
        "properties": {"order_id": {"type": "string", "description": "e.g. ORD-1001"}},
        "required": ["order_id"],
    },
}]


def show_options() -> None:
    print(f"\n  {BOLD}Model providers{RESET}   {DIM}any one is enough{RESET}\n")
    found = {p.id for p in available()}
    if ollama_running():
        found.add("ollama")
    for preset in PRESETS:
        mark = f"{GREEN}key found{RESET}" if preset.id in found else f"{DIM}not set{RESET}"
        cost = f"{GREEN}free{RESET}" if preset.id != "anthropic" else "paid"
        print(f"  {preset.label:<20} {cost:<14} {mark}")
        print(f"    {DIM}{preset.env_key} · default model {preset.default_model}{RESET}")
        print(f"    {DIM}{preset.free}{RESET}")
        print(f"    {DIM}{preset.signup}{RESET}")
        if preset.notes:
            print(f"    {DIM}{preset.notes}{RESET}")
        print()


def probe() -> int:
    try:
        provider = build_provider()
    except ProviderError as exc:
        print(f"  {RED}{exc}{RESET}\n")
        return 2

    preset = resolve()
    print(f"  Active: {BOLD}{provider.label}{RESET}"
          f"{'   ' + GREEN + 'free tier' + RESET if preset and preset.id != 'anthropic' else ''}\n")

    if hasattr(provider, "list_models"):
        models = provider.list_models()
        if models:
            print(f"  {DIM}{len(models)} models reachable, e.g. "
                  f"{', '.join(models[:5])}{RESET}\n")

    print(f"  {DIM}Probing tool calling - the agent cannot work without it…{RESET}")
    from app.agent.providers import user
    try:
        turn = None
        for event in provider.stream(
            system="You are a logistics support assistant. Use the tools available "
                   "to you rather than answering from memory.",
            messages=[user("What is the current status of order ORD-1001?")],
            tools=PROBE_TOOL,
        ):
            if event["type"] == "turn":
                turn = event["turn"]
    except ProviderError as exc:
        print(f"  {RED}FAIL{RESET}  {exc}\n")
        return 1

    if turn and turn.tool_calls:
        call = turn.tool_calls[0]
        print(f"  {GREEN}PASS{RESET}  called {call.name}({call.input})\n")
        print(f"  Ready. Run `make eval` for the golden set, or `make run` for the app.\n")
        return 0

    print(f"  {RED}FAIL{RESET}  the model answered without calling the tool.")
    print(f"  {DIM}This model probably does not support tool calling. Pick another "
          f"with PARCELPILOT_MODEL, or switch provider with PARCELPILOT_PROVIDER.{RESET}\n")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List options only")
    args = parser.parse_args()
    show_options()
    return 0 if args.list else probe()


if __name__ == "__main__":
    sys.exit(main())
