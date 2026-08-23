"""The confirmation gate.

The requirement is that no state-changing action happens without explicit
confirmation. These tests prove it holds as an architectural property rather
than a prompt instruction: calling the tool writes nothing, and the write path
refuses everything except a valid, unused, in-session proposal.
"""
import time

import pytest

from app.core.principal import AccessDenied, resolve_principal
from app.core.proposals import ProposalError


def _propose(runtime, principal, session="s1", **over):
    args = {"summary": "Customer reports total outage",
            "justification": "P1 with a breached first-response target",
            "requested_action": "Page the on-call engineer",
            "ticket_id": "TKT-501"}
    args.update(over)
    return runtime.run(principal, session, "propose_escalation", args)


def test_proposing_writes_nothing(runtime, manager, gateway):
    out = _propose(runtime, manager)
    assert out["status"] == "confirmation_required"
    assert gateway.list_actions(manager)["escalations"] == []


def test_confirmation_executes_exactly_once(runtime, manager, gateway):
    pid = _propose(runtime, manager)["proposal"]["proposal_id"]
    proposal = runtime.proposals.confirm(pid, manager, "s1")
    assert proposal.result["escalation_id"].startswith("ESC-")
    assert len(gateway.list_actions(manager)["escalations"]) == 1

    with pytest.raises(ProposalError):
        runtime.proposals.confirm(pid, manager, "s1")
    assert len(gateway.list_actions(manager)["escalations"]) == 1


def test_declined_proposal_can_never_be_confirmed(runtime, manager, gateway):
    pid = _propose(runtime, manager)["proposal"]["proposal_id"]
    runtime.proposals.cancel(pid, manager, "s1")
    with pytest.raises(ProposalError):
        runtime.proposals.confirm(pid, manager, "s1")
    assert gateway.list_actions(manager)["escalations"] == []


def test_a_different_session_cannot_confirm(runtime, manager):
    pid = _propose(runtime, manager, session="s1")["proposal"]["proposal_id"]
    with pytest.raises(AccessDenied):
        runtime.proposals.confirm(pid, manager, "s2")


def test_a_different_user_cannot_confirm(runtime, manager):
    """A staff-prepared escalation must not be confirmable from a customer
    session that happens to know the id."""
    pid = _propose(runtime, manager)["proposal"]["proposal_id"]
    other = resolve_principal("cust-northstar")
    with pytest.raises(AccessDenied):
        runtime.proposals.confirm(pid, other, "s1")


def test_expired_proposals_are_refused(runtime, manager, monkeypatch, gateway):
    """A confirmation given against a stale preview is not a confirmation - the
    underlying data may have moved since the user saw it."""
    pid = _propose(runtime, manager)["proposal"]["proposal_id"]
    proposal = runtime.proposals.get(pid)
    proposal.created_at = time.time() - 10_000
    with pytest.raises(ProposalError):
        runtime.proposals.confirm(pid, manager, "s1")
    assert gateway.list_actions(manager)["escalations"] == []


def test_unknown_proposal_id_is_refused(runtime, manager):
    with pytest.raises(ProposalError):
        runtime.proposals.confirm("prop_deadbeef", manager, "s1")


def test_ticket_update_preview_shows_before_and_after(runtime, manager, gateway):
    out = runtime.run(manager, "s1", "propose_ticket_update",
                      {"ticket_id": "TKT-503", "field": "status", "new_value": "closed"})
    preview = out["proposal"]["preview"]
    assert preview["current_value"] == "open"
    assert preview["new_value"] == "closed"
    assert gateway.get_ticket(manager, "TKT-503")["status"] == "open"

    runtime.proposals.confirm(out["proposal"]["proposal_id"], manager, "s1")
    assert gateway.get_ticket(manager, "TKT-503")["status"] == "closed"


def test_only_whitelisted_fields_are_updatable(runtime, manager):
    out = runtime.run(manager, "s1", "propose_ticket_update",
                      {"ticket_id": "TKT-503", "field": "description",
                       "new_value": "tampered"})
    assert "error" in out


def test_escalation_preview_warns_about_an_existing_breach(runtime, manager):
    out = _propose(runtime, manager)
    assert any("BREACHED" in w for w in out["proposal"]["warnings"])


def test_customer_cannot_prepare_a_ticket_update(runtime, northstar):
    with pytest.raises(AccessDenied):
        runtime.run(northstar, "s1", "propose_ticket_update",
                    {"ticket_id": "TKT-501", "field": "status", "new_value": "closed"})
