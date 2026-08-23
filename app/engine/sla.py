"""First-response SLA targets and breach state.

Three things make this non-trivial on the supplied data, and all three are
handled here rather than left to prose:

  * targets come from a contract when one exists and from the plan matrix
    otherwise, and the deprecated v2 policy must never be consulted;
  * "2 business hours" and "30 minutes, 24x7" are different clocks - and the
    dataset snapshot (16 Aug 2026) falls on a **Sunday**, so a business-clock
    target has not even started running while a 24x7 one is already breached;
  * a contract that excludes weekend coverage qualifies every target in that
    contract, not just the one clause it appears next to.
"""
from __future__ import annotations

from datetime import datetime

from app.config import BUSINESS_HOURS_ASSUMPTION, fmt, humanise_minutes, parse_ts
from app.core.business_time import budget_minutes, deadline_from, elapsed_against
from app.engine.severity import classify_severity
from app.engine.verdict import Verdict
from app.knowledge.rules import Rules

AT_RISK_FRACTION = 0.75


def evaluate_sla(ticket: dict, account: dict, rules: Rules, now: datetime,
                 severity_override: str | None = None) -> Verdict:
    account_id = account.get("account_id")
    plan = account.get("plan")
    created_at = parse_ts(ticket.get("created_at"))

    if severity_override:
        severity = severity_override.upper()
        classification = {"severity": severity, "confidence": "high",
                          "evidence": ["severity supplied explicitly"], "note": None}
    else:
        classification = classify_severity(ticket.get("subject", ""),
                                           ticket.get("description", ""))
        severity = classification["severity"]

    target = rules.sla_target(account_id, plan, severity)
    duration = target["duration"]

    v = Verdict(topic="sla", decision="unknown", headline="")
    v.facts = {
        "ticket_id": ticket.get("ticket_id"),
        "account_id": account_id,
        "account_name": account.get("account_name"),
        "plan": plan,
        "subject": ticket.get("subject"),
        "ticket_status": ticket.get("status"),
        "severity": severity,
        "severity_confidence": classification["confidence"],
        "severity_evidence": classification["evidence"],
        "target": duration.label(),
        "target_authority": target["authority"],
        "created_at": fmt(created_at),
        "measured_at": fmt(now),
    }
    if classification.get("note"):
        v.caveats.append(classification["note"])
    v.cite(rules.cite(rules.raw["severity"]["source"]))
    v.cite(rules.cite(target.get("source")))
    v.rule_applied = {"authority": target["authority"],
                      "authority_tier": target["authority_tier"],
                      "citation": rules.cite(target.get("source"))}
    if target.get("overrides"):
        v.rule_overridden = {
            "citation": rules.cite(target["overrides"]["source"]),
            "rule": target["overrides"]["duration"].label(),
        }
        v.cite(rules.cite(target["overrides"]["source"]))
        v.computation.append(
            f"The signed agreement sets {duration.label()} for {severity}, replacing the "
            f"{plan} plan default of {target['overrides']['duration'].label()}."
        )

    if created_at is None:
        v.decision = "unknown"
        v.headline = "Ticket creation time is missing; SLA state cannot be computed."
        v.confidence = "low"
        v.needs_human = True
        return v

    deadline = deadline_from(created_at, duration)
    elapsed = elapsed_against(created_at, now, duration)
    budget = budget_minutes(duration)
    remaining = budget - elapsed

    v.facts.update({
        "first_response_due_at": fmt(deadline),
        "elapsed": humanise_minutes(elapsed),
        "elapsed_minutes": round(elapsed, 1),
        "budget_minutes": round(budget, 1),
        "remaining": humanise_minutes(remaining),
        "clock": duration.clock,
    })

    clock_note = ("24x7" if duration.clock == "calendar"
                  else "business hours only")
    v.computation.append(
        f"Ticket created {fmt(created_at)}; target is {duration.label()} ({clock_note}) "
        f"-> first response due {fmt(deadline)}."
    )
    v.computation.append(
        f"At the snapshot ({fmt(now)}), {humanise_minutes(elapsed)} of the "
        f"{humanise_minutes(budget)} target has elapsed."
    )
    if duration.clock == "business":
        v.assumptions.append(BUSINESS_HOURS_ASSUMPTION)
        if elapsed == 0:
            v.computation.append(
                f"{fmt(now)} falls outside business hours, so the business-hours clock "
                "for this target has not started running yet."
            )
    if target.get("coverage_note"):
        v.computation.append(target["coverage_note"])

    # The dataset has no first-response timestamps. Saying so is better than
    # silently assuming nobody has replied.
    v.assumptions.append(
        "The dataset records no first-response timestamps, so elapsed time is measured "
        "from ticket creation to the snapshot. If an agent has already replied, the "
        "breach state should be recomputed from that reply time."
    )

    if now > deadline:
        v.decision = "breached"
        over = elapsed - budget
        v.headline = (f"{severity} first-response target BREACHED by "
                      f"{humanise_minutes(over)} (due {fmt(deadline)}).")
        v.needs_human = True
        v.needs_human_reason = (
            "Support Policy v3 s4: a breached response target must be stated plainly and "
            "escalation recommended."
        )
        v.cite(rules.cite(rules.raw["severity"]["escalation_source"]))
    elif budget > 0 and elapsed / budget >= AT_RISK_FRACTION:
        v.decision = "at_risk"
        v.headline = (f"{severity} first response is at risk - "
                      f"{humanise_minutes(remaining)} left (due {fmt(deadline)}).")
    else:
        v.decision = "within_target"
        v.headline = (f"{severity} first response is within target - "
                      f"{humanise_minutes(remaining)} remaining (due {fmt(deadline)}).")

    if severity == "P1":
        v.caveats.append(
            "Support Policy v3 s4: P1 incidents should be escalated immediately, "
            "regardless of remaining time."
        )
        v.cite(rules.cite(rules.raw["severity"]["escalation_source"]))
    return v


__all__ = ["evaluate_sla", "AT_RISK_FRACTION"]
