"""Access control is the requirement most easily claimed and least often proved.

These tests attack it from the angles that actually matter: another account's
records, internal-only fields, tools the role does not hold, and the case where
the model itself supplies a foreign account id.
"""
import pytest

from app.agent.tools import tools_for
from app.core.db import REDACTED
from app.core.principal import AccessDenied, Perm


def test_customer_sees_only_their_own_orders(gateway, lumenworks):
    ids = {o["order_id"] for o in gateway.list_orders(lumenworks)}
    assert ids == {"ORD-2001", "ORD-2002"}


def test_customer_cannot_read_another_accounts_order(gateway, lumenworks):
    assert gateway.get_order(lumenworks, "ORD-1001") is None


def test_lookup_does_not_leak_existence_of_foreign_records(runtime, lumenworks):
    """The error must not distinguish 'not yours' from 'does not exist' - that
    difference is an enumeration oracle."""
    real_but_foreign = runtime.run(lumenworks, "s", "lookup_orders", {"order_id": "ORD-1001"})
    imaginary = runtime.run(lumenworks, "s", "lookup_orders", {"order_id": "ORD-9999"})
    assert real_but_foreign["found"] is False
    assert real_but_foreign["message"] == imaginary["message"]


def test_explicit_cross_account_filter_is_rejected(gateway, lumenworks):
    with pytest.raises(AccessDenied):
        gateway.list_orders(lumenworks, account_id="ACCT-001")


def test_internal_fields_are_redacted_for_customers(gateway, northstar, agent_user):
    as_customer = gateway.get_ticket(northstar, "TKT-450")
    as_staff = gateway.get_ticket(agent_user, "TKT-450")
    assert as_customer["historical_resolution"] == REDACTED
    assert as_customer["assigned_to"] == REDACTED
    assert "INR 250" in as_staff["historical_resolution"]


def test_account_notes_are_internal(gateway, northstar, agent_user):
    assert gateway.get_account(northstar, "ACCT-001")["notes"] == REDACTED
    assert "Strategic account" in gateway.get_account(agent_user, "ACCT-001")["notes"]


def test_customer_tool_surface_excludes_internal_tools(northstar, agent_user):
    customer_tools = {t["name"] for t in tools_for(northstar)}
    staff_tools = {t["name"] for t in tools_for(agent_user)}
    assert "get_operational_signals" not in customer_tools
    assert "propose_ticket_update" not in customer_tools
    assert {"get_operational_signals", "propose_ticket_update",
            "propose_followup_task"} <= staff_tools


def test_permission_is_rechecked_at_dispatch(runtime, northstar):
    """Filtering the schema is not the control. Even if the model calls a tool it
    was never offered, dispatch must refuse it."""
    with pytest.raises(AccessDenied):
        runtime.run(northstar, "s", "get_operational_signals", {})


def test_customer_cannot_widen_scope_by_passing_another_account_id(runtime, northstar):
    """A prompt injection inside a ticket could tell the model to pass a
    different account_id. The runtime pins a customer to their own account."""
    result = runtime.run(northstar, "s", "get_account_context", {"account_id": "ACCT-002"})
    assert result["account"]["account_id"] == "ACCT-001"


def test_document_search_never_returns_another_customers_contract(runtime, lumenworks):
    result = runtime.run(lumenworks, "s", "search_policy_documents",
                         {"query": "cancellation fee waiver enterprise agreement", "k": 8})
    docs = {r["doc_id"] for r in result["results"]}
    assert not any("Northstar" in d for d in docs)


def test_deprecated_policy_is_excluded_by_default(runtime, agent_user):
    result = runtime.run(agent_user, "s", "search_policy_documents",
                         {"query": "enterprise P1 first response target", "k": 8})
    assert all(r["status"] != "DEPRECATED" for r in result["results"])


def test_deprecated_policy_carries_a_warning_when_requested(runtime, agent_user):
    result = runtime.run(agent_user, "s", "search_policy_documents",
                         {"query": "support policy v2 response targets",
                          "include_superseded": True, "k": 8})
    assert "warning" in result


def test_escalation_is_scoped_to_the_customers_own_account(runtime, northstar):
    out = runtime.run(northstar, "s", "propose_escalation", {
        "summary": "x", "justification": "y", "requested_action": "z",
        "account_id": "ACCT-002",
    })
    assert "ACCT-001" in out["proposal"]["preview"]["account"]


def test_manager_holds_approval_authority_agent_does_not(agent_user, manager):
    assert not agent_user.has(Perm.APPROVE_CREDIT)
    assert manager.has(Perm.APPROVE_CREDIT)


def test_denied_calls_are_audited(gateway, northstar):
    gateway.audit(northstar, "sess-1", "get_operational_signals", {}, "denied", "blocked")
    trail = gateway.audit_trail("sess-1")
    assert trail[0]["outcome"] == "denied"
    assert trail[0]["user_id"] == "cust-northstar"
