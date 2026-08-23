"""The agent loop.

Provider-neutral: it speaks the types in `agent/providers/base.py`, so the same
loop, tool surface and behaviour run on Anthropic or on any free OpenAI-
compatible endpoint (Groq, Gemini, OpenRouter, Cerebras, Mistral, Ollama).

A hand-written loop rather than an SDK helper, for three reasons that all matter
here:

  * the UI has to show *which tool is running* as it happens, so the loop emits
    an event per tool call before executing it;
  * a state-changing tool returns a proposal that must interrupt the turn and
    surface as a confirmation card, not be swallowed as another tool result;
  * every tool call is audited with the calling principal, which means the loop
    owns the principal, not the tool function.

Events are yielded as dicts and serialised to SSE by the API layer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Generator

from app.agent.prompts import system_prompt
from app.agent.providers import (
    LLMProvider, Message, ProviderError, assistant, build_provider,
    tool_results, user,
)
from app.agent.tools import (
    STATE_CHANGING, TOOL_BY_NAME, ToolRuntime, serialise_result, tools_for,
)
from app.config import MAX_AGENT_STEPS, fmt
from app.core.principal import AccessDenied, DEMO_USERS, Principal


@dataclass
class Session:
    session_id: str
    principal: Principal
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    tool_calls: int = 0


def _summarise_tool_result(name: str, result: dict) -> str:
    """One line for the UI's tool chip. The model sees the full payload."""
    if not isinstance(result, dict):
        return "done"
    if result.get("status") == "confirmation_required":
        return "prepared - awaiting confirmation"
    if result.get("found") is False:
        return "not found in this context"
    if "verdict" in result:
        v = result["verdict"]
        bits = [v.get("decision", "")]
        if v.get("amount_inr") is not None:
            bits.append(f"INR {v['amount_inr']:,.0f}")
        if v.get("needs_human"):
            bits.append("needs human")
        return " | ".join(b for b in bits if b)
    if "results" in result:
        n = len(result["results"])
        c = len(result.get("conflicts") or [])
        return f"{n} clause(s)" + (f", {c} conflict(s) flagged" if c else "")
    if "signals" in result:
        return f"{result.get('summary', {}).get('total', 0)} signal(s)"
    if "matches" in result:
        return f"{len(result['matches'])} known issue(s)"
    if "count" in result:
        return f"{result['count']} record(s)"
    if "account" in result:
        acct = result["account"]
        return f"{acct.get('account_name')} ({acct.get('plan')})"
    if "error" in result:
        return f"error: {result['error']}"
    return "done"


class Agent:
    def __init__(self, runtime: ToolRuntime, provider: LLMProvider | None = None):
        self.runtime = runtime
        self.provider = provider or build_provider()

    @property
    def label(self) -> str:
        return self.provider.label

    def run(self, session: Session, user_message: str | None = None
            ) -> Generator[dict, None, None]:
        principal = session.principal
        org = DEMO_USERS.get(principal.user_id, {}).get("org")
        system = system_prompt(principal, fmt(self.runtime.db.clock.now()), org)
        tools = tools_for(principal)

        if user_message is not None:
            session.messages.append(user(user_message))

        for step in range(MAX_AGENT_STEPS):
            turn = None
            try:
                for event in self.provider.stream(system=system,
                                                  messages=session.messages,
                                                  tools=tools):
                    if event["type"] == "turn":
                        turn = event["turn"]
                    else:
                        yield event
            except ProviderError as exc:
                yield {"type": "error", "message": str(exc)}
                return
            except Exception as exc:                          # noqa: BLE001
                yield {"type": "error",
                       "message": f"The assistant is unavailable right now "
                                  f"({type(exc).__name__}). Nothing was changed."}
                return

            if turn is None:
                yield {"type": "error",
                       "message": "The model returned nothing. Nothing was changed."}
                return

            if turn.stop == "refusal":
                yield {"type": "error",
                       "message": "This request could not be completed. Please rephrase, "
                                  "or ask for it to be escalated to the support team."}
                return

            session.messages.append(assistant(turn))

            if not turn.wants_tools:
                if turn.stop == "length":
                    yield {"type": "error",
                           "message": "The answer was cut off. Please ask for a shorter "
                                      "or more specific answer."}
                yield {"type": "done", "usage": {**turn.usage, "steps": step + 1},
                       "provider": self.provider.label}
                return

            results = []
            for call in turn.tool_calls:
                spec = TOOL_BY_NAME.get(call.name, {})
                yield {"type": "tool_start", "id": call.id, "name": call.name,
                       "category": spec.get("category", "tool"), "args": call.input,
                       "state_changing": call.name in STATE_CHANGING}
                session.tool_calls += 1

                try:
                    result = self.runtime.run(principal, session.session_id,
                                              call.name, call.input)
                    outcome, is_error = "ok", False
                except AccessDenied as exc:
                    result = {"error": str(exc),
                              "note": "Access is enforced by the data layer. Tell the user "
                                      "plainly that this is outside their access."}
                    outcome, is_error = "denied", True
                except Exception as exc:                      # noqa: BLE001
                    # Smaller models mis-shape tool arguments more often, so the
                    # error has to be actionable enough for the model to retry.
                    result = {"error": f"{type(exc).__name__}: {exc}",
                              "note": "Check the tool's required arguments and try again."}
                    outcome, is_error = "error", True

                self.runtime.db.audit(principal, session.session_id, call.name,
                                      call.input, outcome,
                                      _summarise_tool_result(call.name, result))

                yield {"type": "tool_result", "id": call.id, "name": call.name,
                       "category": spec.get("category", "tool"),
                       "summary": _summarise_tool_result(call.name, result),
                       "outcome": outcome, "payload": result}

                if isinstance(result, dict) and result.get("status") == "confirmation_required":
                    yield {"type": "proposal", "proposal": result["proposal"]}

                results.append({"id": call.id, "name": call.name,
                                "content": serialise_result(result),
                                "is_error": is_error})

            session.messages.append(tool_results(results))

        yield {"type": "error",
               "message": ("This request needed more steps than the assistant is allowed to "
                           "take. Nothing was changed - please narrow the question or ask "
                           "for it to be escalated.")}


__all__ = ["Agent", "Session"]
