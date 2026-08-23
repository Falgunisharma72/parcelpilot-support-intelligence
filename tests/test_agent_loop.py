"""The agent loop, end to end, with a scripted provider.

Because the loop is provider-neutral, a fake provider that replays a fixed
sequence of turns exercises the whole path - tool dispatch, access enforcement,
event emission, the confirmation interrupt and the step guard - with no API key
and no network. The parts that depend on a real model's judgement are covered by
the golden set in `evals/`; everything mechanical is covered here.
"""
from __future__ import annotations

import pytest

from app.agent.loop import Agent, Session
from app.agent.providers.base import ToolCall, Turn
from app.core.principal import resolve_principal


class ScriptedProvider:
    """Replays turns in order; records what it was asked."""
    id = "scripted"
    model = "scripted"
    label = "Scripted · test"
    supports_thinking = False

    def __init__(self, turns: list[Turn]):
        self.turns = list(turns)
        self.calls = 0
        self.last_tools: list[dict] = []
        self.last_messages = None

    def stream(self, *, system, messages, tools):
        self.calls += 1
        self.last_tools = tools
        self.last_messages = list(messages)
        turn = self.turns.pop(0) if self.turns else Turn(text="done.", stop="end")
        if turn.text:
            yield {"type": "text", "text": turn.text}
        yield {"type": "turn", "turn": turn}


def _agent(runtime, turns):
    provider = ScriptedProvider(turns)
    return Agent(runtime, provider=provider), provider


def _events(agent, session, message):
    return list(agent.run(session, message))


def test_multi_step_tool_chain(runtime, manager):
    """Look up an order, then decide on it - the shape almost every real answer
    takes. The second turn must see the first tool's result."""
    agent, provider = _agent(runtime, [
        Turn(tool_calls=[ToolCall("c1", "lookup_orders", {"order_id": "ORD-1001"})],
             stop="tool_use"),
        Turn(tool_calls=[ToolCall("c2", "check_cancellation", {"order_id": "ORD-1001"})],
             stop="tool_use"),
        Turn(text="No cancellation fee applies.", stop="end"),
    ])
    session = Session("s1", manager)
    events = _events(agent, session, "can Northstar cancel ORD-1001?")

    names = [e["name"] for e in events if e["type"] == "tool_start"]
    assert names == ["lookup_orders", "check_cancellation"]
    assert provider.calls == 3

    verdict = next(e for e in events if e["type"] == "tool_result"
                   and e["name"] == "check_cancellation")
    assert verdict["payload"]["verdict"]["decision"] == "cancellable_no_fee"
    assert events[-1]["type"] == "done"
    assert events[-1]["usage"]["steps"] == 3


def test_tool_events_carry_what_the_ui_needs(runtime, manager):
    agent, _ = _agent(runtime, [
        Turn(tool_calls=[ToolCall("c1", "check_sla", {"ticket_id": "TKT-505"})],
             stop="tool_use"),
        Turn(text="Breached.", stop="end"),
    ])
    events = _events(agent, Session("s1", manager), "is TKT-505 ok?")
    start = next(e for e in events if e["type"] == "tool_start")
    result = next(e for e in events if e["type"] == "tool_result")
    assert start["category"] == "decision"
    assert start["state_changing"] is False
    assert "breached" in result["summary"]
    assert result["outcome"] == "ok"


def test_access_denial_becomes_a_tool_error_not_a_crash(runtime, northstar):
    """A customer session calling an internal tool must get a clean, explainable
    error the model can relay - and it must be audited."""
    agent, _ = _agent(runtime, [
        Turn(tool_calls=[ToolCall("c1", "get_operational_signals", {})], stop="tool_use"),
        Turn(text="That is outside your access.", stop="end"),
    ])
    events = _events(agent, Session("s1", northstar), "show me all accounts")
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["outcome"] == "denied"
    assert "not permitted" in result["payload"]["error"]

    trail = runtime.db.audit_trail("s1")
    assert any(e["outcome"] == "denied" for e in trail)


def test_customer_is_never_offered_internal_tools(runtime, northstar):
    agent, provider = _agent(runtime, [Turn(text="hello", stop="end")])
    _events(agent, Session("s1", northstar), "hi")
    offered = {t["name"] for t in provider.last_tools}
    assert "get_operational_signals" not in offered
    assert "propose_ticket_update" not in offered


def test_state_changing_tool_interrupts_with_a_proposal_and_writes_nothing(runtime, manager):
    agent, _ = _agent(runtime, [
        Turn(tool_calls=[ToolCall("c1", "propose_escalation", {
            "summary": "P1 outage", "justification": "breached",
            "requested_action": "page on-call", "ticket_id": "TKT-501"})],
            stop="tool_use"),
        Turn(text="Confirm and I will raise it.", stop="end"),
    ])
    events = _events(agent, Session("s1", manager), "escalate TKT-501")

    proposal = next(e for e in events if e["type"] == "proposal")
    assert proposal["proposal"]["requires_confirmation"] is True
    start = next(e for e in events if e["type"] == "tool_start")
    assert start["state_changing"] is True
    assert runtime.db.list_actions(manager)["escalations"] == []


def test_unparseable_tool_arguments_return_an_actionable_error(runtime, manager):
    """Free models mis-shape arguments. The loop must hand back something the
    model can retry from, not fall over."""
    agent, _ = _agent(runtime, [
        Turn(tool_calls=[ToolCall("c1", "check_cancellation", {})], stop="tool_use"),
        Turn(text="I need an order id.", stop="end"),
    ])
    events = _events(agent, Session("s1", manager), "can I cancel it?")
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["outcome"] == "error"
    assert "try again" in result["payload"]["note"]


def test_unknown_tool_name_is_survivable(runtime, manager):
    agent, _ = _agent(runtime, [
        Turn(tool_calls=[ToolCall("c1", "delete_everything", {})], stop="tool_use"),
        Turn(text="That tool does not exist.", stop="end"),
    ])
    events = _events(agent, Session("s1", manager), "delete it all")
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["outcome"] == "error"


def test_refusal_is_reported_as_an_error_not_an_answer(runtime, manager):
    agent, _ = _agent(runtime, [Turn(stop="refusal")])
    events = _events(agent, Session("s1", manager), "something disallowed")
    assert events[-1]["type"] == "error"


def test_provider_failure_says_nothing_was_changed(runtime, manager):
    from app.agent.providers.base import ProviderError

    class Broken(ScriptedProvider):
        def stream(self, *, system, messages, tools):
            raise ProviderError("Groq free-tier rate limit reached. Nothing was changed.")
            yield  # pragma: no cover

    agent = Agent(runtime, provider=Broken([]))
    events = _events(agent, Session("s1", manager), "hello")
    assert events[-1]["type"] == "error"
    assert "Nothing was changed" in events[-1]["message"]


def test_runaway_loop_is_capped(runtime, manager, monkeypatch):
    monkeypatch.setattr("app.agent.loop.MAX_AGENT_STEPS", 3)
    turns = [Turn(tool_calls=[ToolCall(f"c{i}", "lookup_orders", {})], stop="tool_use")
             for i in range(10)]
    agent, _ = _agent(runtime, turns)
    events = _events(agent, Session("s1", manager), "loop forever")
    assert events[-1]["type"] == "error"
    assert "more steps" in events[-1]["message"]


def test_conversation_history_accumulates_for_the_next_turn(runtime, manager):
    agent, provider = _agent(runtime, [
        Turn(tool_calls=[ToolCall("c1", "lookup_tickets", {"ticket_id": "TKT-501"})],
             stop="tool_use"),
        Turn(text="It is a P1.", stop="end"),
    ])
    session = Session("s1", manager)
    _events(agent, session, "what is TKT-501?")
    roles = [m.role for m in session.messages]
    assert roles == ["user", "assistant", "tool_results", "assistant"]

    _events(agent, session, "and its SLA?")
    assert len(session.messages) == 6
