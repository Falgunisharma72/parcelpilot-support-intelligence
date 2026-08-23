"""Proactive detection (client Problem 1)."""
from app.engine.signals import detect_signals, summarise


def _signals(gateway, manager, rules, now):
    return detect_signals(gateway.list_accounts(manager), gateway.list_orders(manager),
                          gateway.list_tickets(manager), rules, now)


def test_both_p1_breaches_are_detected(gateway, manager, rules, now):
    found = _signals(gateway, manager, rules, now)
    breached = {s.evidence[0]["ticket_id"] for s in found if s.type == "sla_breach"}
    assert breached == {"TKT-501", "TKT-505"}


def test_incorrect_past_guidance_is_surfaced(gateway, manager, rules, now):
    """Both historical resolutions in the pack are wrong under current rules -
    one on a waived cancellation fee, one on a plan row limit that is actually a
    known bug. Finding these is the difference between a reactive bot and a
    product that protects customers."""
    stale = [s for s in _signals(gateway, manager, rules, now) if s.type == "stale_guidance"]
    tickets = {s.evidence[0]["ticket_id"] for s in stale}
    assert tickets == {"TKT-450", "TKT-451"}
    assert any("waives the cancellation fee" in s.detail for s in stale)
    assert any("KI-208" in s.detail for s in stale)


def test_tickets_are_clustered_onto_the_known_issue(gateway, manager, rules, now):
    clusters = [s for s in _signals(gateway, manager, rules, now)
                if s.type == "known_issue_cluster"]
    assert any(s.metrics["known_issue"] == "KI-208" for s in clusters)


def test_overdue_pickup_carries_its_credit_exposure(gateway, manager, rules, now):
    overdue = [s for s in _signals(gateway, manager, rules, now) if s.type == "overdue_pickup"]
    assert len(overdue) == 1
    assert overdue[0].evidence[0]["order_id"] == "ORD-2002"
    assert overdue[0].metrics["credit_decision"] == "eligible"
    assert overdue[0].metrics["credit_inr"] == 300


def test_recurrence_of_a_closed_issue_is_flagged(gateway, manager, rules, now):
    rec = [s for s in _signals(gateway, manager, rules, now) if s.type == "recurring_issue"]
    assert any({"TKT-502", "TKT-451"} == {e["ticket_id"] for e in s.evidence} for s in rec)


def test_cancellation_concentration_is_flagged(gateway, manager, rules, now):
    spike = [s for s in _signals(gateway, manager, rules, now) if s.type == "cancellation_spike"]
    assert spike and spike[0].metrics["count"] >= 3


def test_signals_are_ordered_by_severity(gateway, manager, rules, now):
    found = _signals(gateway, manager, rules, now)
    assert found[0].severity == "critical"
    assert summarise(found)["total"] == len(found)


def test_signals_require_staff_permission(gateway, northstar):
    import pytest
    from app.core.principal import AccessDenied
    with pytest.raises(AccessDenied):
        gateway.scan(northstar, "tickets")
