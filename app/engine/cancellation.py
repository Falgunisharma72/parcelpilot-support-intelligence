"""Cancellation eligibility and fee - computed, never inferred.

The question "can Northstar cancel ORD-1001 without a cancellation fee" has four
moving parts: the order's status, how long ago it was booked, what the SOP says,
and whether the account's signed agreement disapplies the SOP. Three of those
are lookups and one is arithmetic. None of them is a judgement call, so none of
them is left to the model.
"""
from __future__ import annotations

from datetime import datetime

from app.config import fmt, humanise_minutes, parse_ts
from app.engine.verdict import Verdict
from app.knowledge.rules import Rules


def evaluate_cancellation(order: dict, account: dict, rules: Rules,
                          now: datetime) -> Verdict:
    status = (order.get("status") or "").upper()
    account_id = account.get("account_id")
    booked_at = parse_ts(order.get("booked_at"))
    requested_at = parse_ts(order.get("cancellation_requested_at"))
    # If the customer has already asked to cancel, that request time is what the
    # fee is assessed against - not "now". Using the snapshot instead would
    # penalise a customer for how long support took to answer them.
    reference = requested_at or now
    reference_label = ("cancellation request time" if requested_at
                       else "dataset snapshot (no cancellation request recorded)")

    resolved = rules.cancellation_rule(account_id, status)
    rule = resolved["applied"] or {}
    v = Verdict(
        topic="cancellation",
        decision="unknown",
        headline="",
        facts={
            "order_id": order.get("order_id"),
            "account_id": account_id,
            "account_name": account.get("account_name"),
            "plan": account.get("plan"),
            "status": status,
            "carrier": order.get("carrier"),
            "booked_at": fmt(booked_at),
            "cancellation_requested_at": fmt(requested_at),
            "assessed_at": fmt(reference),
            "assessed_against": reference_label,
            "shipment_fee_inr": order.get("shipment_fee_inr"),
        },
    )

    applied_citation = rules.cite(resolved.get("source"))
    v.rule_applied = {
        "authority": resolved["authority"],
        "authority_tier": resolved["authority_tier"],
        "citation": applied_citation,
    }
    v.cite(applied_citation)
    if resolved.get("overrides"):
        overridden_citation = rules.cite((resolved["overrides"] or {}).get("source"))
        v.rule_overridden = {"citation": overridden_citation,
                             "rule": resolved["overrides"].get("rule")}
        v.cite(overridden_citation)

    if not rule:
        v.decision = "unknown_status"
        v.headline = f"No cancellation rule is defined for status {status!r}."
        v.confidence = "low"
        v.needs_human = True
        v.needs_human_reason = (
            f"Order status {status!r} is not covered by the current SOP; a human "
            "should confirm how to handle it."
        )
        return v

    if not rule.get("cancellable", False):
        v.decision = "not_cancellable"
        alt = rule.get("alternative")
        if status == "PICKED_UP":
            v.headline = ("This shipment has already been picked up, so it cannot be "
                          "cancelled. The return-to-origin workflow is the route to get "
                          "the parcel back.")
            v.caveats.append(
                "Return-to-origin may carry its own charges; those are not covered by "
                "the supplied document pack, so confirm before quoting a cost."
            )
            v.needs_human = True
            v.needs_human_reason = (
                "Return-to-origin is an operational workflow, not something this "
                "assistant can execute."
            )
        else:
            v.headline = f"An order in status {status} cannot be cancelled."
        v.computation.append(f"Order status is {status} -> SOP marks it non-cancellable.")
        if alt:
            v.facts["alternative"] = alt
        return v

    # --- cancellable: does a fee apply? ---
    if rule.get("waives_default_fee") or rule.get("fee_inr") == 0:
        fee = 0.0
        v.decision = "cancellable_no_fee"
        if resolved["authority"] == "contract":
            elapsed = (reference - booked_at).total_seconds() / 60 if booked_at else None
            if elapsed is not None:
                v.computation.append(
                    f"Booked {fmt(booked_at)}; cancellation assessed at {fmt(reference)} "
                    f"-> {humanise_minutes(elapsed)} elapsed."
                )
                default = (rules.raw["cancellation"]["default"].get(status) or {})
                window = default.get("free_window_minutes")
                default_fee = default.get("fee_after_window_inr")
                if window is not None and elapsed > window:
                    v.computation.append(
                        f"Under the general SOP that is past the {window}-minute free "
                        f"window and INR {default_fee:,.0f} would apply."
                    )
            v.computation.append(
                f"{account.get('account_name')}'s signed agreement waives the "
                "cancellation fee for any BOOKED shipment before pickup, regardless of "
                "elapsed time. The agreement outranks the SOP."
            )
            v.headline = (
                f"Yes - {account.get('account_name')} can cancel {order.get('order_id')} "
                "with no cancellation fee."
            )
        else:
            v.headline = f"{order.get('order_id')} can be cancelled with no fee."
    else:
        window = rule.get("free_window_minutes")
        fee_after = float(rule.get("fee_after_window_inr", 0))
        if booked_at is None:
            v.decision = "needs_verification"
            v.headline = "The booking time is missing, so the free-cancellation window cannot be checked."
            v.confidence = "low"
            v.needs_human = True
            v.needs_human_reason = "Booking timestamp is absent from the order record."
            return v
        elapsed = (reference - booked_at).total_seconds() / 60
        v.facts["minutes_since_booking"] = round(elapsed, 1)
        v.computation.append(
            f"Booked {fmt(booked_at)}; cancellation assessed at {fmt(reference)} "
            f"-> {humanise_minutes(elapsed)} elapsed."
        )
        if elapsed <= window:
            fee = 0.0
            v.decision = "cancellable_no_fee"
            v.computation.append(
                f"That is within the {window}-minute free-cancellation window -> no fee."
            )
            v.headline = (f"Yes - {order.get('order_id')} can be cancelled with no fee "
                          f"(within the {window}-minute window).")
        else:
            fee = fee_after
            v.decision = "cancellable_with_fee"
            v.computation.append(
                f"That is past the {window}-minute free window -> INR {fee_after:,.0f} "
                "cancellation fee applies."
            )
            v.headline = (f"{order.get('order_id')} can be cancelled, but an "
                          f"INR {fee_after:,.0f} cancellation fee applies.")
            if resolved.get("contract_checked"):
                v.computation.append(
                    "The account's signed agreement was checked and does not waive the "
                    "cancellation fee."
                )
                v.cite(rules.cite(resolved.get("contract_note")))

    v.amount_inr = fee
    v.facts["cancellation_fee_inr"] = fee

    # --- cross-document caveat: is the displayed status trustworthy? ---
    # Product docs record a carrier webhook delay that makes BOOKED an unreliable
    # signal for one carrier. Acting on a stale status is exactly the "confidently
    # incorrect action" the brief warns about, so it is surfaced on the verdict.
    if status == "BOOKED":
        for ki in rules.known_issues():
            if ki.get("carrier") and ki["carrier"] == order.get("carrier") and ki.get("status") != "Resolved":
                v.caveats.append(
                    f"{ki['id']}: {ki['carrier']} pickup confirmations can arrive up to "
                    f"{ki['delay_window_minutes']} minutes late, so this order may already "
                    "have been collected even though it still shows BOOKED. Verify carrier "
                    "status before cancelling."
                )
                v.cite(rules.cite(ki.get("source")))
                v.confidence = "medium"
    return v


__all__ = ["evaluate_cancellation"]
