"""Proactive issue detection (client Problem 1).

A chatbot only helps once somebody asks. These detectors run over the whole
support surface at the dataset snapshot and surface what deserves attention
before a customer has to chase it. Every signal carries its evidence and the
rule it was measured against, because an ops alert nobody can verify is an alert
nobody actions.

The detectors are deterministic. That is a deliberate choice for an alerting
surface: an LLM-generated "here's what looks worrying" list is unrepeatable, and
you cannot page a human off something that changes every time you run it. The
model's role is to explain and triage these, not to invent them.

Detectors implemented:
  1. sla_breach / sla_at_risk        - contract-aware, per open ticket
  2. p1_open                         - policy says escalate immediately
  3. known_issue_cluster             - several tickets tracing to one product bug
  4. recurring_issue                 - a problem that came back after being closed
  5. stale_guidance                  - past support answers that today's rules contradict
  6. overdue_pickup                  - orders past their window, with credit exposure
  7. cancellation_spike              - unusual concentration of cancellation requests
  8. awaiting_reply                  - customer has replied and nobody has come back
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

from app.config import fmt, humanise_minutes, parse_ts
from app.engine.credits import evaluate_service_credit
from app.engine.severity import classify_severity
from app.engine.sla import evaluate_sla
from app.knowledge.rules import Rules

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@dataclass
class Signal:
    signal_id: str
    type: str
    severity: str
    title: str
    detail: str
    accounts: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    recommended_action: str = ""
    suggested_tool: dict | None = None
    citations: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Stale-guidance checks
# ---------------------------------------------------------------------------
# The brief warns that historical ticket resolutions may contain incorrect
# guidance. Finding those is valuable in its own right: every one is a customer
# who was told something wrong and may still be acting on it.
#
# Each check pulls the numeric claim out of the past resolution and compares it
# with what the *current* authoritative rule says for that specific account.
# Adding a topic means adding a checker, not touching the detector.
_MONEY_RE = re.compile(r"(?:INR|Rs\.?|₹)\s*([\d,]+)", re.I)
_ROWS_RE = re.compile(r"([\d,]+)\s*rows?", re.I)


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def _check_cancellation_guidance(resolution: str, account_id: str, rules: Rules) -> dict | None:
    money = _MONEY_RE.search(resolution)
    if not money or not re.search(r"cancel", resolution, re.I):
        return None
    claimed = _num(money.group(1))
    resolved = rules.cancellation_rule(account_id, "BOOKED")
    rule = resolved["applied"] or {}
    if rule.get("waives_default_fee") or rule.get("fee_inr") == 0:
        actual = 0.0
    else:
        actual = float(rule.get("fee_after_window_inr", 0))
    if abs(claimed - actual) < 0.01:
        return None
    return {
        "claim": f"INR {claimed:,.0f} cancellation fee",
        "current_position": (
            f"INR {actual:,.0f}" +
            (" - this account's signed agreement waives the cancellation fee"
             if actual == 0 and resolved["authority"] == "contract" else "")
        ),
        "source": resolved.get("source"),
    }


def _check_bulk_upload_guidance(resolution: str, account_id: str, rules: Rules) -> dict | None:
    if not re.search(r"bulk|csv|upload|rows", resolution, re.I):
        return None
    rows = _ROWS_RE.search(resolution)
    if not rows:
        return None
    claimed = _num(rows.group(1))
    cap = rules.raw["plan_capabilities"]["bulk_upload"]
    actual = float(cap["max_rows"])
    if abs(claimed - actual) < 0.01:
        return None
    return {
        "claim": f"a supported limit of {claimed:,.0f} rows",
        "current_position": (
            f"{actual:,.0f} rows is the documented product limit; the failures above "
            "~3,000 rows are known issue KI-208, not a plan restriction"
        ),
        "source": cap.get("source"),
    }


GUIDANCE_CHECKS = (_check_cancellation_guidance, _check_bulk_upload_guidance)


# ---------------------------------------------------------------------------
def _norm_words(text: str) -> set[str]:
    stop = {"the", "a", "for", "of", "to", "is", "and", "in", "on", "with", "at"}
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) > 2 and w not in stop}


def _similarity(a: str, b: str) -> float:
    wa, wb = _norm_words(a), _norm_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def detect_signals(accounts: list[dict], orders: list[dict], tickets: list[dict],
                   rules: Rules, now: datetime) -> list[Signal]:
    by_account = {a["account_id"]: a for a in accounts}
    signals: list[Signal] = []
    open_tickets = [t for t in tickets if (t.get("status") or "").lower() == "open"]

    # --- 1 & 2: SLA state and open P1s ------------------------------------
    for t in open_tickets:
        account = by_account.get(t["account_id"])
        if not account:
            continue
        v = evaluate_sla(t, account, rules, now)
        sev = v.facts["severity"]
        ref = {"ticket_id": t["ticket_id"], "account_id": t["account_id"],
               "subject": t.get("subject"), "severity": sev,
               "target": v.facts["target"], "due": v.facts.get("first_response_due_at")}
        if v.decision == "breached":
            signals.append(Signal(
                signal_id=f"sla-breach-{t['ticket_id']}",
                type="sla_breach",
                severity="critical" if sev == "P1" else "high",
                title=f"{sev} SLA breached - {account['account_name']} ({t['ticket_id']})",
                detail=v.headline + " " + " ".join(v.computation[-2:]),
                accounts=[t["account_id"]], evidence=[ref],
                recommended_action=(
                    "State the breach to the customer and escalate. Support Policy v3 s4 "
                    "requires the breach to be stated plainly rather than hidden."),
                suggested_tool={"tool": "create_escalation",
                                "args": {"ticket_id": t["ticket_id"], "severity": sev}},
                citations=v.citations,
                metrics={"elapsed": v.facts.get("elapsed"),
                         "target": v.facts["target"],
                         "authority": v.facts["target_authority"]},
            ))
        elif v.decision == "at_risk":
            signals.append(Signal(
                signal_id=f"sla-risk-{t['ticket_id']}",
                type="sla_at_risk",
                severity="high" if sev in ("P1", "P2") else "medium",
                title=f"{sev} SLA at risk - {account['account_name']} ({t['ticket_id']})",
                detail=v.headline, accounts=[t["account_id"]], evidence=[ref],
                recommended_action="Respond before the target elapses.",
                citations=v.citations,
                metrics={"remaining": v.facts.get("remaining"), "target": v.facts["target"]},
            ))
        if sev == "P1":
            signals.append(Signal(
                signal_id=f"p1-open-{t['ticket_id']}",
                type="p1_open",
                severity="critical",
                title=f"Open P1 - {account['account_name']} ({t['ticket_id']})",
                detail=(f"{t.get('subject')}. Classified P1: "
                        f"{'; '.join(v.facts['severity_evidence']) or 'policy definition'}. "
                        "Support Policy v3 s4: P1 incidents should be escalated immediately."),
                accounts=[t["account_id"]], evidence=[ref],
                recommended_action="Escalate now, independent of remaining SLA time.",
                suggested_tool={"tool": "create_escalation",
                                "args": {"ticket_id": t["ticket_id"], "severity": "P1"}},
                citations=v.citations,
            ))

    # --- 3: known-issue clusters ------------------------------------------
    clusters: dict[str, list[dict]] = {}
    for t in tickets:
        account = by_account.get(t["account_id"]) or {}
        text = f"{t.get('subject','')} {t.get('description','')}"
        for ki in rules.match_known_issues(text, plan=account.get("plan")):
            clusters.setdefault(ki["id"], []).append(t)
    for ki_id, group in clusters.items():
        ki = next(k for k in rules.known_issues() if k["id"] == ki_id)
        if ki.get("status") == "Resolved" or len(group) < 2:
            continue
        accts = sorted({t["account_id"] for t in group})
        signals.append(Signal(
            signal_id=f"ki-cluster-{ki_id}",
            type="known_issue_cluster",
            severity="high" if len(accts) > 1 else "medium",
            title=f"{len(group)} tickets trace to {ki_id} - {ki['title']}",
            detail=(f"{ki_id} is {ki.get('status','open').lower()} since "
                    f"{ki.get('opened')}. Affected accounts: {', '.join(accts)}. "
                    f"Workaround: {ki.get('workaround') or ki.get('caution')}"),
            accounts=accts,
            evidence=[{"ticket_id": t["ticket_id"], "account_id": t["account_id"],
                       "subject": t.get("subject"), "status": t.get("status")}
                      for t in group],
            recommended_action=(
                f"Answer all {len(group)} tickets with the {ki_id} workaround rather than "
                "diagnosing each separately, and push for a product fix if the cluster grows."),
            citations=[c for c in [rules.cite(ki.get("source"))] if c],
            metrics={"tickets": len(group), "accounts": len(accts), "known_issue": ki_id},
        ))

    # --- 4: recurrence after closure --------------------------------------
    closed = [t for t in tickets if (t.get("status") or "").lower() != "open"]
    for new in open_tickets:
        for old in closed:
            score = _similarity(f"{new.get('subject')} {new.get('description')}",
                                f"{old.get('subject')} {old.get('description')}")
            if score < 0.22:
                continue
            gap = (parse_ts(new.get("created_at")) - parse_ts(old.get("created_at")))
            signals.append(Signal(
                signal_id=f"recurrence-{new['ticket_id']}-{old['ticket_id']}",
                type="recurring_issue",
                severity="medium",
                title=f"{new['ticket_id']} looks like a repeat of closed {old['ticket_id']}",
                detail=(f"'{new.get('subject')}' resembles '{old.get('subject')}', closed "
                        f"{gap.days} days earlier. A problem that comes back after being "
                        "closed usually means the earlier resolution treated a symptom."),
                accounts=sorted({new["account_id"], old["account_id"]}),
                evidence=[{"ticket_id": new["ticket_id"], "status": "open",
                           "subject": new.get("subject")},
                          {"ticket_id": old["ticket_id"], "status": old.get("status"),
                           "subject": old.get("subject"),
                           "historical_resolution": old.get("historical_resolution")}],
                recommended_action=("Review what the earlier ticket was told before replying, "
                                    "and check whether that guidance was correct."),
                metrics={"similarity": round(score, 2), "days_apart": gap.days},
            ))

    # --- 5: past guidance that current rules contradict --------------------
    for t in closed:
        resolution = t.get("historical_resolution")
        if not resolution:
            continue
        for check in GUIDANCE_CHECKS:
            finding = check(resolution, t["account_id"], rules)
            if not finding:
                continue
            account = by_account.get(t["account_id"], {})
            signals.append(Signal(
                signal_id=f"stale-guidance-{t['ticket_id']}",
                type="stale_guidance",
                severity="high",
                title=f"Past answer to {account.get('account_name', t['account_id'])} "
                      f"conflicts with current rules ({t['ticket_id']})",
                detail=(f"The closed ticket records: \"{resolution}\" That asserts "
                        f"{finding['claim']}, but the authoritative position today is "
                        f"{finding['current_position']}. The customer may still be acting "
                        "on the incorrect answer."),
                accounts=[t["account_id"]],
                evidence=[{"ticket_id": t["ticket_id"], "account_id": t["account_id"],
                           "historical_resolution": resolution,
                           "subject": t.get("subject")}],
                recommended_action=("Confirm what the customer was told, correct it proactively, "
                                    "and check whether any charge or refusal followed from it."),
                suggested_tool={"tool": "create_followup_task",
                                "args": {"ticket_id": t["ticket_id"],
                                         "title": f"Correct prior guidance on {t['ticket_id']}"}},
                citations=[c for c in [rules.cite(finding.get("source"))] if c],
                metrics={"claim": finding["claim"]},
            ))
            break

    # --- 6: overdue pickups (and the credit exposure they carry) -----------
    for o in orders:
        if o.get("pickup_actual_at") or (o.get("status") or "").upper() not in ("BOOKED", "DRAFT"):
            continue
        window_end = parse_ts(o.get("pickup_window_end"))
        if not window_end or now <= window_end:
            continue
        late_minutes = (now - window_end).total_seconds() / 60
        account = by_account.get(o["account_id"], {})
        credit = evaluate_service_credit(rules, now, account=account, order=o)
        sev = "high" if credit.decision == "eligible" else "medium"
        detail = (f"Pickup window ended {fmt(window_end)}; still not collected at the "
                  f"snapshot - {humanise_minutes(late_minutes)} overdue. ")
        detail += credit.headline
        signals.append(Signal(
            signal_id=f"overdue-pickup-{o['order_id']}",
            type="overdue_pickup",
            severity=sev,
            title=f"{o['order_id']} pickup {humanise_minutes(late_minutes)} overdue "
                  f"({account.get('account_name', o['account_id'])})",
            detail=detail, accounts=[o["account_id"]],
            evidence=[{"order_id": o["order_id"], "account_id": o["account_id"],
                       "carrier": o.get("carrier"), "status": o.get("status"),
                       "pickup_window_end": o.get("pickup_window_end"),
                       "carrier_fault": bool(o.get("carrier_fault"))}],
            recommended_action=(
                "Contact the customer before they contact you; a credit is already owed."
                if credit.decision == "eligible"
                else "Chase the carrier and confirm fault before discussing credits."),
            citations=credit.citations,
            metrics={"minutes_overdue": round(late_minutes),
                     "credit_decision": credit.decision,
                     "credit_inr": credit.amount_inr},
        ))

    # --- 7: cancellation concentration ------------------------------------
    requests = [(o, parse_ts(o.get("cancellation_requested_at"))) for o in orders
                if o.get("cancellation_requested_at")]
    recent = [(o, ts) for o, ts in requests if ts and now - ts <= timedelta(hours=24)]
    if len(recent) >= 3:
        accts = sorted({o["account_id"] for o, _ in recent})
        window = max(ts for _, ts in recent) - min(ts for _, ts in recent)
        signals.append(Signal(
            signal_id="cancellation-spike",
            type="cancellation_spike",
            severity="medium",
            title=f"{len(recent)} cancellation requests across {len(accts)} accounts "
                  f"in {humanise_minutes(window.total_seconds() / 60)}",
            detail=("Cancellation requests are unusually concentrated. Clustered across "
                    "multiple accounts, that is more often an upstream problem - a carrier "
                    "issue or a pricing/ETA change - than a coincidence."),
            accounts=accts,
            evidence=[{"order_id": o["order_id"], "account_id": o["account_id"],
                       "carrier": o.get("carrier"),
                       "cancellation_requested_at": o.get("cancellation_requested_at")}
                      for o, _ in recent],
            recommended_action=("Check for a common carrier or route before treating these as "
                                "unrelated customer decisions."),
            metrics={"count": len(recent), "accounts": len(accts),
                     "carriers": sorted({o.get("carrier") for o, _ in recent})},
        ))

    # --- 8: customer waiting on us ----------------------------------------
    for t in open_tickets:
        last = parse_ts(t.get("last_customer_message_at"))
        created = parse_ts(t.get("created_at"))
        if not last or not created or last <= created:
            continue
        waiting = (now - last).total_seconds() / 60
        if waiting < 5:
            continue
        account = by_account.get(t["account_id"], {})
        sev_class = classify_severity(t.get("subject", ""), t.get("description", ""))
        signals.append(Signal(
            signal_id=f"awaiting-reply-{t['ticket_id']}",
            type="awaiting_reply",
            severity="medium" if sev_class["severity"] != "P1" else "high",
            title=f"{t['ticket_id']} - customer replied {humanise_minutes(waiting)} ago",
            detail=(f"{account.get('account_name', t['account_id'])} sent a follow-up at "
                    f"{t.get('last_customer_message_at')} and the ticket is still open."),
            accounts=[t["account_id"]],
            evidence=[{"ticket_id": t["ticket_id"], "account_id": t["account_id"],
                       "last_customer_message_at": t.get("last_customer_message_at"),
                       "subject": t.get("subject")}],
            recommended_action="Acknowledge the follow-up.",
            metrics={"waiting": humanise_minutes(waiting)},
        ))

    signals.sort(key=lambda s: (SEVERITY_ORDER.get(s.severity, 9), s.type, s.signal_id))
    return signals


def summarise(signals: list[Signal]) -> dict:
    counts: dict[str, int] = {}
    for s in signals:
        counts[s.severity] = counts.get(s.severity, 0) + 1
    return {
        "total": len(signals),
        "by_severity": counts,
        "by_type": {t: sum(1 for s in signals if s.type == t)
                    for t in sorted({s.type for s in signals})},
        "accounts_affected": sorted({a for s in signals for a in s.accounts}),
    }


__all__ = ["Signal", "detect_signals", "summarise"]
