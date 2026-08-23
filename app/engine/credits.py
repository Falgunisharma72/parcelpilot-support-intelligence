"""Failed-pickup service-credit eligibility and amount.

This is the calculation most likely to be got confidently wrong, because the
default SOP and a customer's agreement disagree on *both* the delay threshold
and the amount. LumenWorks is the trap in the supplied pack: a pickup three
hours late qualifies under the general SOP (>2h) but not under their signed
agreement (>4h). Answering "yes, you're eligible" from the SOP alone is wrong
for that customer - and it is the kind of wrong that gets promised to a customer
in writing.
"""
from __future__ import annotations

from datetime import datetime

from app.config import fmt, humanise_minutes, parse_ts
from app.engine.verdict import Verdict
from app.knowledge.rules import Rules


def _credit_amount(amount_rule: dict, shipment_fee: float | None) -> tuple[float | None, str]:
    kind = amount_rule.get("type")
    if kind == "fixed":
        value = float(amount_rule["flat_inr"])
        return value, f"Contractual fixed credit of INR {value:,.0f}."
    if kind == "lesser_of":
        flat = float(amount_rule["flat_inr"])
        pct = float(amount_rule["percent_of_shipment_fee"])
        if shipment_fee is None:
            return None, "Shipment fee is unknown, so the percentage cap cannot be applied."
        pct_value = shipment_fee * pct / 100.0
        value = min(flat, pct_value)
        return value, (
            f"Credit is the lower of INR {flat:,.0f} and {pct:.0f}% of the "
            f"INR {shipment_fee:,.0f} shipment fee (INR {pct_value:,.0f}) "
            f"-> INR {value:,.0f}."
        )
    return None, f"Unrecognised credit formula {kind!r}."


def evaluate_service_credit(
    rules: Rules,
    now: datetime,
    *,
    account: dict,
    order: dict | None = None,
    hours_past_window_end: float | None = None,
    carrier_fault: bool | None = None,
    customer_fault: bool | None = None,
    shipment_fee_inr: float | None = None,
) -> Verdict:
    """Evaluate a failed-pickup credit.

    Works from a real order when one is supplied, and from stated facts when the
    question is hypothetical ("a pickup is three hours late because of carrier
    fault") - which still resolves against *this* account's contract rather than
    a generic answer.
    """
    account_id = account.get("account_id")
    resolved = rules.service_credit_rule(account_id)
    rule = resolved["applied"]

    v = Verdict(topic="service_credit", decision="unknown", headline="")
    v.facts["account_id"] = account_id
    v.facts["account_name"] = account.get("account_name")
    v.facts["plan"] = account.get("plan")

    applied_citation = rules.cite(resolved.get("source"))
    v.rule_applied = {"authority": resolved["authority"],
                      "authority_tier": resolved["authority_tier"],
                      "citation": applied_citation}
    v.cite(applied_citation)
    if resolved.get("overrides"):
        overridden = rules.cite((resolved["overrides"] or {}).get("source"))
        v.rule_overridden = {"citation": overridden, "rule": resolved["overrides"].get("rule")}
        v.cite(overridden)

    # --- establish the facts ---
    if order:
        window_end = parse_ts(order.get("pickup_window_end"))
        picked_at = parse_ts(order.get("pickup_actual_at"))
        reference = picked_at or now
        if window_end is None:
            v.decision = "needs_verification"
            v.headline = "The scheduled pickup window is missing from this order."
            v.confidence = "low"
            v.needs_human = True
            v.needs_human_reason = "Pickup window end is absent; delay cannot be computed."
            return v
        delay_minutes = (reference - window_end).total_seconds() / 60
        hours_late = delay_minutes / 60
        carrier_fault = bool(order.get("carrier_fault")) if carrier_fault is None else carrier_fault
        customer_fault = bool(order.get("customer_fault")) if customer_fault is None else customer_fault
        shipment_fee_inr = order.get("shipment_fee_inr") if shipment_fee_inr is None else shipment_fee_inr
        v.facts.update({
            "order_id": order.get("order_id"),
            "carrier": order.get("carrier"),
            "status": order.get("status"),
            "pickup_window_end": fmt(window_end),
            "pickup_actual_at": fmt(picked_at),
            "measured_to": fmt(reference),
            "shipment_fee_inr": shipment_fee_inr,
        })
        v.computation.append(
            f"Pickup window ended {fmt(window_end)}; "
            + (f"pickup occurred {fmt(picked_at)}" if picked_at
               else f"pickup still had not happened at the snapshot ({fmt(now)})")
            + f" -> {humanise_minutes(delay_minutes)} past the window "
              f"({hours_late:.2f} hours)."
        )
        v.facts["hours_past_window_end"] = round(hours_late, 2)
    else:
        if hours_past_window_end is None:
            v.decision = "needs_verification"
            v.headline = "Not enough information to assess a service credit."
            v.confidence = "low"
            v.needs_human = True
            v.needs_human_reason = (
                "Neither an order reference nor a stated delay was provided."
            )
            return v
        hours_late = float(hours_past_window_end)
        v.facts["hours_past_window_end"] = hours_late
        v.facts["shipment_fee_inr"] = shipment_fee_inr
        v.computation.append(f"Stated delay: {hours_late:.2f} hours past the end of the pickup window.")
        v.assumptions.append(
            "Assessed from the delay as described, not from a specific order record. "
            "Quote an order ID for a binding answer."
        )
        v.confidence = "medium"

    v.facts["carrier_fault"] = carrier_fault
    v.facts["customer_fault"] = customer_fault

    threshold = float(rule["threshold_hours_past_window_end"])
    v.facts["threshold_hours"] = threshold
    authority_phrase = ("your signed agreement" if resolved["authority"] == "contract"
                        else "the current service-credit SOP")

    # --- rule 1: fault must be established. Unknown is not "no", and it is not
    # "yes" either - the SOP explicitly forbids promising a credit on unknowns. ---
    if carrier_fault is None or customer_fault is None:
        v.decision = "needs_verification"
        v.headline = "Carrier fault has not been established, so a credit cannot be confirmed yet."
        v.confidence = "low"
        v.needs_human = True
        v.needs_human_reason = (
            "The SOP forbids promising a credit while carrier fault, pickup timing or "
            "customer fault is unknown."
        )
        v.cite(rules.cite(rules.raw["service_credit"]["uncertainty_source"]))
        return v

    if customer_fault:
        v.decision = "not_eligible"
        v.headline = "No service credit: the delay is recorded as customer-caused."
        v.computation.append("Customer fault is recorded -> the eligibility test fails.")
        return v

    if not carrier_fault:
        v.decision = "not_eligible"
        v.headline = ("No service credit: the carrier has not accepted fault for this "
                      "delay.")
        v.computation.append(
            "Carrier fault is not recorded on this order -> the eligibility test fails."
        )
        v.caveats.append(
            "If the carrier subsequently accepts fault, this decision should be revisited."
        )
        return v

    # --- rule 2: the delay threshold, from whichever source has authority ---
    if hours_late <= threshold:
        v.decision = "not_eligible"
        v.computation.append(
            f"{hours_late:.2f} hours is not more than the {threshold:.0f}-hour threshold "
            f"in {authority_phrase} -> not eligible."
        )
        v.headline = (
            f"No service credit: the delay is {hours_late:.2f} hours, and "
            f"{authority_phrase} requires more than {threshold:.0f} hours."
        )
        if resolved.get("overrides"):
            default_threshold = resolved["overrides"]["rule"]["threshold_hours_past_window_end"]
            if hours_late > float(default_threshold):
                # The single most important sentence this system can produce.
                v.computation.append(
                    f"Note: the general SOP threshold is {default_threshold} hours, which "
                    f"this delay does exceed - but the signed agreement replaces that "
                    "threshold for this account."
                )
                v.caveats.append(
                    "This answer differs from the general policy because the account's "
                    "signed agreement sets a different threshold."
                )
        return v

    amount, amount_note = _credit_amount(rule["amount"], shipment_fee_inr)
    v.computation.append(
        f"{hours_late:.2f} hours is more than the {threshold:.0f}-hour threshold in "
        f"{authority_phrase}, carrier fault is accepted and there is no customer fault "
        "-> eligible."
    )
    v.computation.append(amount_note)

    if amount is None:
        v.decision = "eligible_amount_unknown"
        v.headline = "Eligible for a service credit, but the amount cannot be computed."
        v.confidence = "low"
        v.needs_human = True
        v.needs_human_reason = amount_note
        return v

    v.decision = "eligible"
    v.amount_inr = amount
    v.facts["credit_inr"] = amount
    v.headline = f"Eligible for a service credit of INR {amount:,.0f}."

    threshold_approval = rules.manager_approval_threshold()
    if amount > threshold_approval:
        v.needs_human = True
        v.needs_human_reason = (
            f"Individual credits above INR {threshold_approval:,.0f} require manager approval."
        )
        v.computation.append(
            f"INR {amount:,.0f} exceeds the INR {threshold_approval:,.0f} manager-approval "
            "threshold."
        )
        v.cite(rules.cite(rules.raw["service_credit"]["manager_approval_source"]))
    else:
        v.computation.append(
            f"INR {amount:,.0f} is within the INR {threshold_approval:,.0f} agent authority "
            "limit, so no manager approval is required."
        )

    cap = resolved.get("monthly_cap_inr")
    if cap:
        v.caveats.append(
            f"This account's agreement caps aggregate monthly service credits at "
            f"INR {cap:,.0f}. The supplied data does not include credits already issued "
            "this month, so the remaining headroom must be checked before issuing."
        )
        v.confidence = "medium"
    return v


__all__ = ["evaluate_service_credit"]
