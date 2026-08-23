"""The agent loop.

A hand-written loop rather than the SDK's tool runner, for three reasons that
all matter here:

  * the UI has to show *which tool is running* as it happens, so the loop needs
    to emit an event per tool call before it executes;
  * a state-changing tool returns a proposal that must interrupt the turn and
    surface as a confirmation card, not be swallowed as another tool result;
  * every tool call is audited with the calling principal, which means the loop
    owns the principal, not the tool function.

Events are yielded as dicts and serialised to SSE by the API layer.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Generator

import anthropic

from app.config import (
    EFFORT, MAX_AGENT_STEPS, MAX_TOKENS, MODEL, SHOW_THINKING, fmt,
)
from app.agent.prompts import system_prompt
from app.agent.tools import (
    STATE_CHANGING, TOOL_BY_NAME, ToolRuntime, serialise_result, tools_for,
)
from app.core.principal import AccessDenied, Principal, DEMO_USERS

# Server-side refusal fallbacks: if a safety classifier declines a turn, the API
# routes to a comparable model instead of returning a dead end to a support user.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class Session:
    session_id: str
    principal: Principal
    messages: list[dict] = field(default_factory=list)
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
        s = result.get("summary", {})
        return f"{s.get('total', 0)} signal(s)"
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
    def __init__(self, runtime: ToolRuntime, api_key: str | None = None):
        self.runtime = runtime
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    # -- API call with graceful degradation ---------------------------------
    def _stream(self, **kwargs):
        """Prefer the beta endpoint with refusal fallbacks; degrade cleanly.

        Older SDK builds do not accept `fallbacks`, and we would rather serve the
        request without the fallback than fail the turn on a parameter.
        """
        try:
            return self.client.beta.messages.stream(
                betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
        except TypeError:
            return self.client.messages.stream(**kwargs)

    # -- main loop ----------------------------------------------------------
    def run(self, session: Session, user_message: str | None = None
            ) -> Generator[dict, None, None]:
        principal = session.principal
        org = DEMO_USERS.get(principal.user_id, {}).get("org")
        system = system_prompt(principal, fmt(self.runtime.db.clock.now()), org)
        tools = tools_for(principal)

        if user_message is not None:
            session.messages.append({"role": "user", "content": user_message})

        for step in range(MAX_AGENT_STEPS):
            request: dict[str, Any] = {
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "system": system,
                "messages": session.messages,
                "tools": tools,
                "output_config": {"effort": EFFORT},
            }
            if SHOW_THINKING:
                request["thinking"] = {"type": "adaptive", "display": "summarized"}

            try:
                with self._stream(**request) as stream:
                    for event in stream:
                        if event.type == "content_block_start":
                            block = event.content_block
                            if block.type == "thinking":
                                yield {"type": "thinking_start"}
                            elif block.type == "tool_use":
                                yield {"type": "tool_pending", "name": block.name}
                        elif event.type == "content_block_delta":
                            delta = event.delta
                            if delta.type == "text_delta":
                                yield {"type": "text", "text": delta.text}
                            elif delta.type == "thinking_delta" and SHOW_THINKING:
                                yield {"type": "thinking", "text": delta.thinking}
                    response = stream.get_final_message()
            except anthropic.APIStatusError as exc:
                yield {"type": "error",
                       "message": f"The assistant is unavailable right now ({exc.status_code}). "
                                  "Please retry; nothing was changed."}
                return
            except anthropic.APIConnectionError:
                yield {"type": "error",
                       "message": "Could not reach the model. Please retry; nothing was changed."}
                return

            # A refusal is a real outcome, not an exception - handle it rather
            # than letting `content` be read as if it held an answer.
            if getattr(response, "stop_reason", None) == "refusal":
                yield {"type": "error",
                       "message": "This request could not be completed. Please rephrase, "
                                  "or ask for it to be escalated to the support team."}
                return

            session.messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                yield {"type": "done", "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "steps": step + 1,
                }}
                return

            tool_results = []
            for block in tool_uses:
                spec = TOOL_BY_NAME.get(block.name, {})
                # Tool inputs may arrive with varied JSON escaping; always take
                # the parsed dict the SDK provides rather than string-matching.
                args = dict(block.input or {})
                yield {"type": "tool_start", "id": block.id, "name": block.name,
                       "category": spec.get("category", "tool"), "args": args,
                       "state_changing": block.name in STATE_CHANGING}
                session.tool_calls += 1

                try:
                    result = self.runtime.run(principal, session.session_id, block.name, args)
                    outcome, is_error = "ok", False
                except AccessDenied as exc:
                    result = {"error": str(exc),
                              "note": "Access is enforced by the data layer. Tell the user "
                                      "plainly that this is outside their access."}
                    outcome, is_error = "denied", True
                except Exception as exc:                      # noqa: BLE001
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                    outcome, is_error = "error", True

                self.runtime.db.audit(principal, session.session_id, block.name, args,
                                      outcome, _summarise_tool_result(block.name, result))

                yield {"type": "tool_result", "id": block.id, "name": block.name,
                       "category": spec.get("category", "tool"),
                       "summary": _summarise_tool_result(block.name, result),
                       "outcome": outcome, "payload": result}

                if isinstance(result, dict) and result.get("status") == "confirmation_required":
                    yield {"type": "proposal", "proposal": result["proposal"]}

                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": serialise_result(result), "is_error": is_error,
                })

            session.messages.append({"role": "user", "content": tool_results})

        yield {"type": "error",
               "message": ("This request needed more steps than the assistant is allowed to "
                           "take. Nothing was changed - please narrow the question or ask "
                           "for it to be escalated.")}


__all__ = ["Agent", "Session"]
