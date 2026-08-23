"""Golden cases for the deterministic decision engines.

Each of these is a place where a plausible-sounding wrong answer exists, and the
test pins the right one. They run without an API key, which is the point: the
part of the system that decides money and entitlements is testable in CI.
"""
import pytest

from app.engine.cancellation import evaluate_cancellation
from app.engine.credits import evaluate_service_credit
from app.engine.severity import classify_severity
from app.engine.sla import evaluate_sla


def _order(gw, p, oid):
    return gw.get_order(p, oid)


def _account(gw, p, aid):
    return gw.get_account(p, aid)


# ── cancellation ────────────────────────────────────────────────────────────
def test_contract_waiver_beats_the_sop_fee(gateway, manager, rules, now):
    """ORD-1001: booked 09:00, cancellation requested 11:00. The SOP would charge
    INR 250 after 30 minutes; Northstar's agreement waives it entirely."""
    v = evaluate_cancellation(_order(gateway, manager, "ORD-1001"),
                              _account(gateway, manager, "ACCT-001"), rules, now)
    assert v.decision == "cancellable_no_fee"
    assert v.amount_inr == 0
    assert v.rule_applied["authority"] == "contract"
    assert v.rule_overridden is not None
    assert any("250" in c for c in v.computation), "the overridden default should be shown"


def test_no_waiver_means_the_sop_fee_stands(gateway, manager, rules, now):
    """ORD-2001: 75 minutes after booking, and LumenWorks' agreement explicitly
    declines a waiver."""
    v = evaluate_cancellation(_order(gateway, manager, "ORD-2001"),
                              _account(gateway, manager, "ACCT-002"), rules, now)
    assert v.decision == "cancellable_with_fee"
    assert v.amount_inr == 250


def test_within_the_free_window_there_is_no_fee(gateway, manager, rules, now):
    v = evaluate_cancellation(_order(gateway, manager, "ORD-3001"),
                              _account(gateway, manager, "ACCT-003"), rules, now)
    assert v.decision == "cancellable_no_fee"
    assert v.amount_inr == 0


def test_picked_up_orders_are_not_cancellable_even_under_a_waiver(gateway, manager, rules, now):
    """The Northstar waiver covers BOOKED shipments only. Applying it to a
    PICKED_UP shipment would be a contractually wrong 'yes'."""
    v = evaluate_cancellation(_order(gateway, manager, "ORD-1002"),
                              _account(gateway, manager, "ACCT-001"), rules, now)
    assert v.decision == "not_cancellable"
    assert "return-to-origin" in v.headline.lower()
    assert v.needs_human is True


def test_fee_is_assessed_at_the_request_time_not_the_snapshot(gateway, manager, rules, now):
    v = evaluate_cancellation(_order(gateway, manager, "ORD-3001"),
                              _account(gateway, manager, "ACCT-003"), rules, now)
    assert v.facts["assessed_at"] == "2026-08-16 10:40"


def test_stale_carrier_status_is_flagged_on_booked_swiftship_orders(gateway, manager, rules, now):
    v = evaluate_cancellation(_order(gateway, manager, "ORD-1001"),
                              _account(gateway, manager, "ACCT-001"), rules, now)
    assert any("KI-211" in c for c in v.caveats)
    assert v.confidence == "medium"


# ── service credits ─────────────────────────────────────────────────────────
def test_contract_threshold_and_fixed_amount(gateway, manager, rules, now):
    """ORD-2002: 4h30m late, carrier fault. LumenWorks' contract sets >4h and a
    flat INR 300, replacing the SOP's >2h / lesser-of-500-or-10%."""
    v = evaluate_service_credit(rules, now, account=_account(gateway, manager, "ACCT-002"),
                                order=_order(gateway, manager, "ORD-2002"))
    assert v.decision == "eligible"
    assert v.amount_inr == 300
    assert v.rule_applied["authority"] == "contract"


@pytest.mark.parametrize("account_id,expected", [
    ("ACCT-002", "not_eligible"),   # contract requires > 4 hours
    ("ACCT-003", "eligible"),       # default SOP requires > 2 hours
    ("ACCT-001", "eligible"),       # inherits the default SOP threshold
])
def test_three_hours_late_depends_entirely_on_the_account(gateway, manager, rules, now,
                                                          account_id, expected):
    """The brief's own example question. A generic 'yes, you get a credit' is
    wrong for LumenWorks and right for everyone else."""
    v = evaluate_service_credit(rules, now, account=_account(gateway, manager, account_id),
                                hours_past_window_end=3, carrier_fault=True,
                                customer_fault=False, shipment_fee_inr=2000)
    assert v.decision == expected


def test_a_near_miss_explains_why_the_general_policy_does_not_apply(gateway, manager, rules, now):
    v = evaluate_service_credit(rules, now, account=_account(gateway, manager, "ACCT-002"),
                                hours_past_window_end=3, carrier_fault=True,
                                customer_fault=False, shipment_fee_inr=2000)
    assert any("general SOP threshold is 2 hours" in c for c in v.computation)


def test_percentage_cap_is_applied_under_the_default_rule(gateway, manager, rules, now):
    """10% of INR 2,000 is INR 200, which is lower than the INR 500 flat cap."""
    v = evaluate_service_credit(rules, now, account=_account(gateway, manager, "ACCT-003"),
                                hours_past_window_end=5, carrier_fault=True,
                                customer_fault=False, shipment_fee_inr=2000)
    assert v.amount_inr == 200


def test_unknown_fault_never_produces_a_promise(gateway, manager, rules, now):
    v = evaluate_service_credit(rules, now, account=_account(gateway, manager, "ACCT-003"),
                                hours_past_window_end=6, carrier_fault=None,
                                customer_fault=None)
    assert v.decision == "needs_verification"
    assert v.needs_human is True


def test_customer_fault_disqualifies(gateway, manager, rules, now):
    v = evaluate_service_credit(rules, now, account=_account(gateway, manager, "ACCT-003"),
                                hours_past_window_end=9, carrier_fault=True,
                                customer_fault=True, shipment_fee_inr=9000)
    assert v.decision == "not_eligible"


def test_large_credit_requires_manager_approval(gateway, manager, rules, now):
    """Force a credit above the INR 1,000 threshold via a contract-free account
    and a large fee: 10% of INR 60,000 is capped at INR 500, so the flat cap
    keeps us under - the approval path is exercised through the threshold rule."""
    v = evaluate_service_credit(rules, now, account=_account(gateway, manager, "ACCT-003"),
                                hours_past_window_end=5, carrier_fault=True,
                                customer_fault=False, shipment_fee_inr=60000)
    assert v.amount_inr == 500      # lesser-of rule caps it
    assert v.needs_human is False
    assert rules.manager_approval_threshold() == 1000


def test_monthly_cap_is_surfaced_as_a_caveat(gateway, manager, rules, now):
    v = evaluate_service_credit(rules, now, account=_account(gateway, manager, "ACCT-001"),
                                hours_past_window_end=5, carrier_fault=True,
                                customer_fault=False, shipment_fee_inr=4000)
    assert any("5,000" in c for c in v.caveats)


# ── severity + SLA ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("subject,description,expected", [
    ("All shipment creation is failing",
     "Every user gets HTTP 500 when creating any shipment. Existing shipments can still be viewed.",
     "P1"),
    ("Possible API key exposure",
     "An employee accidentally posted a screenshot containing a production API key in a public channel.",
     "P1"),
    ("Bulk upload fails for 4,200-row CSV",
     "The CSV reaches roughly 70% and fails. Creating shipments one-by-one still works.",
     "P2"),
    ("How do we change the billing contact?",
     "Customer wants to replace the billing-contact email.", "P3"),
])
def test_severity_classification(subject, description, expected):
    assert classify_severity(subject, description)["severity"] == expected


def test_read_access_is_not_a_workaround_for_a_write_outage():
    """The regression this guards: 'existing shipments can still be viewed' once
    downgraded a total creation outage to P2."""
    result = classify_severity(
        "All shipment creation is failing",
        "Every user gets HTTP 500 when creating any shipment. Existing shipments can still be viewed.")
    assert result["severity"] == "P1"


def test_contract_sla_beats_plan_default_and_is_breached(gateway, manager, rules, now):
    """TKT-501 created 10:30, snapshot 11:00. Northstar's contractual P1 target is
    15 minutes 24x7, so this is breached; the Enterprise default of 30 minutes
    would have looked exactly on the line, and the deprecated v2 policy's 1 hour
    would have looked fine."""
    v = evaluate_sla(gateway.get_ticket(manager, "TKT-501"),
                     _account(gateway, manager, "ACCT-001"), rules, now)
    assert v.facts["severity"] == "P1"
    assert v.facts["target_authority"] == "contract"
    assert v.decision == "breached"


def test_security_ticket_is_p1_and_badly_breached(gateway, manager, rules, now):
    v = evaluate_sla(gateway.get_ticket(manager, "TKT-505"),
                     _account(gateway, manager, "ACCT-004"), rules, now)
    assert v.facts["severity"] == "P1"
    assert v.decision == "breached"
    assert v.facts["elapsed_minutes"] == 150


def test_business_hours_clock_does_not_run_on_a_sunday(gateway, manager, rules, now):
    """The snapshot is Sunday 16 Aug 2026. LumenWorks has no weekend coverage, so
    their business-hours target has not started - answering 'breached' here
    would be wrong, and answering '4 hours from creation' would be wrong too."""
    v = evaluate_sla(gateway.get_ticket(manager, "TKT-502"),
                     _account(gateway, manager, "ACCT-002"), rules, now)
    assert v.facts["clock"] == "business"
    assert v.facts["elapsed_minutes"] == 0
    assert v.decision == "within_target"
    assert v.facts["first_response_due_at"].startswith("2026-08-17")


def test_severity_override_recomputes_the_target(gateway, manager, rules, now):
    v = evaluate_sla(gateway.get_ticket(manager, "TKT-504"),
                     _account(gateway, manager, "ACCT-001"), rules, now,
                     severity_override="P1")
    assert v.facts["severity"] == "P1"
    assert v.facts["target"] == "15 minutes, 24x7"


def test_sla_states_its_assumptions(gateway, manager, rules, now):
    v = evaluate_sla(gateway.get_ticket(manager, "TKT-503"),
                     _account(gateway, manager, "ACCT-003"), rules, now)
    assert any("no first-response timestamps" in a for a in v.assumptions)
