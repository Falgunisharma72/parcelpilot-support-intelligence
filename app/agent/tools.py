"""The agent's tool surface.

Three design rules hold across every tool here:

1. **The tool list is filtered by permission before the model ever sees it.**
   A customer session is not given `get_operational_signals` or
   `propose_ticket_update` at all. Hiding a tool is not security on its own, so
   the dispatcher re-checks the permission too - but not offering a tool the
   caller cannot use also stops the model wasting a turn discovering that.

2. **Judgement stays in the model; arithmetic and precedence stay in code.**
   `check_cancellation` does not return "here are some policy paragraphs about
   cancellation" and hope. It returns a decided verdict with its working shown.
   The model decides what to say, what to ask, and when to escalate.

3. **No tool writes.** The three action tools build proposals. The write path is
   `proposals.execute`, reachable only from an authenticated confirm request.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from app.config import BUSINESS_HOURS_ASSUMPTION, fmt
from app.core.db import DataGateway
from app.core.principal import AccessDenied, Perm, Principal
from app.core.proposals import ProposalStore, register_executor
from app.engine.cancellation import evaluate_cancellation
from app.engine.credits import evaluate_service_credit
from app.engine.signals import SIGNAL_SEVERITIES, SIGNAL_TYPES, detect_signals, summarise
from app.engine.sla import evaluate_sla
from app.knowledge.retrieval import ClauseIndex
from app.knowledge.rules import Rules

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
# `permission` is not part of the Anthropic schema - it is stripped before the
# request and used to decide which tools this principal is offered.

TOOL_SPECS: list[dict] = [
    {
        "name": "search_policy_documents",
        "permission": None,
        "category": "documents",
        "description": (
            "Search ParcelPilot's policies, SOPs, product documentation and customer "
            "agreements. Results are ranked by authority as well as relevance: a signed "
            "customer agreement outranks the general policy, which outranks product "
            "documentation. Superseded documents are excluded unless explicitly requested, "
            "and clauses belonging to another customer's agreement are never returned. "
            "Each result carries its authority tier and a citation. Use this for questions "
            "about what the rules say. For an actual cancellation, credit or SLA decision, "
            "prefer the check_* tools, which apply these rules deterministically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
                "topics": {
                    "type": "array", "items": {"type": "string"},
                    "description": ("Optional topic filter: cancellation, service_credit, "
                                    "failed_pickup, sla, bulk_upload, known_issue, security, "
                                    "plan_capability, source_precedence, approval, escalation, "
                                    "shipment_status."),
                },
                "include_superseded": {
                    "type": "boolean",
                    "description": ("Include DEPRECATED documents. Only for questions about "
                                    "what a policy *used to* say. Never as a basis for a "
                                    "current answer."),
                },
                "k": {"type": "integer", "description": "Number of clauses (default 6)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_account_context",
        "permission": None,
        "category": "data",
        "description": (
            "Return the account in context: plan, status, whether a signed agreement "
            "exists, and the contractual overrides that apply to it (support targets, "
            "cancellation terms, service-credit terms). Call this early - almost every "
            "policy answer depends on whether this account has a contract that displaces "
            "the general rules."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": ("Account to look up. Staff only; for a customer session "
                                    "this is ignored and their own account is used."),
                },
            },
        },
    },
    {
        "name": "lookup_orders",
        "permission": None,
        "category": "data",
        "description": (
            "Look up shipment orders: a single order by id, or a filtered list. Returns "
            "status, booking time, pickup window, actual pickup, fee, and recorded fault "
            "attribution. Scoped to the caller's account automatically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "e.g. ORD-1001"},
                "account_id": {"type": "string", "description": "Staff only."},
                "status": {"type": "string", "description": "DRAFT | BOOKED | PICKED_UP | DELIVERED"},
                "carrier": {"type": "string"},
            },
        },
    },
    {
        "name": "lookup_tickets",
        "permission": None,
        "category": "data",
        "description": (
            "Look up support tickets by id, by filter, or by free-text search over subject "
            "and description. Note: the historical_resolution field on closed tickets is "
            "what a past agent told the customer. It is context only and may be wrong - "
            "never treat it as policy, and check it against current rules before repeating it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string", "description": "e.g. TKT-501"},
                "account_id": {"type": "string", "description": "Staff only."},
                "status": {"type": "string", "description": "open | closed"},
                "search": {"type": "string", "description": "Free-text search."},
            },
        },
    },
    {
        "name": "check_cancellation",
        "permission": None,
        "category": "decision",
        "description": (
            "Decide whether a specific order can be cancelled and what fee, if any, applies. "
            "Resolves the account's signed agreement against the general SOP, measures the "
            "elapsed time from booking to the cancellation request, and returns a verdict "
            "with the full calculation, the rule applied, the rule overridden, and citations. "
            "Use this instead of reasoning about cancellation terms yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "check_service_credit",
        "permission": None,
        "category": "decision",
        "description": (
            "Decide failed-pickup service-credit eligibility and amount for this account. "
            "Works two ways: pass order_id for a binding answer on a real order, or pass the "
            "stated facts (hours_past_window_end, carrier_fault, customer_fault, optionally "
            "shipment_fee_inr) for a hypothetical - which is still resolved against this "
            "account's own contract, not a generic policy. Different accounts have different "
            "delay thresholds and amounts, so never answer this from memory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "account_id": {"type": "string", "description": "Staff only."},
                "hours_past_window_end": {
                    "type": "number",
                    "description": "Hours past the END of the scheduled pickup window.",
                },
                "carrier_fault": {"type": "boolean"},
                "customer_fault": {"type": "boolean"},
                "shipment_fee_inr": {"type": "number"},
            },
        },
    },
    {
        "name": "check_sla",
        "permission": None,
        "category": "decision",
        "description": (
            "Classify a ticket's severity against the current policy definitions and compute "
            "its first-response target, due time and breach state. Applies the account's "
            "contractual targets where they exist, and handles the difference between 24x7 "
            "and business-hours clocks. Pass severity_override if you have judged the "
            "severity differently from the automatic classification - the timing is then "
            "recomputed for that severity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "severity_override": {"type": "string", "description": "P1 | P2 | P3"},
            },
            "required": ["ticket_id"],
        },
    },
    {
        "name": "find_known_issues",
        "permission": None,
        "category": "documents",
        "description": (
            "Match a problem description against ParcelPilot's current known issues and "
            "their workarounds. Call this before diagnosing any reported product problem: "
            "a known issue with a documented workaround is a very different answer from a "
            "plan limitation or a one-off bug."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "The reported symptom."},
                "plan": {"type": "string", "description": "Optional plan filter."},
            },
            "required": ["description"],
        },
    },
    {
        "name": "get_operational_signals",
        "permission": Perm.VIEW_SIGNALS,
        "category": "signals",
        "description": (
            "Internal only. Run proactive detection across all accounts at the dataset "
            "snapshot. Call it with no arguments to see everything; filter only when you "
            "already know the exact type you want. Valid types are exactly: "
            "sla_breach, sla_at_risk, p1_open, known_issue_cluster, recurring_issue, "
            "stale_guidance (past support answers that current rules contradict), "
            "overdue_pickup, awaiting_reply, cancellation_spike. Anything else is an "
            "error, not an empty result. Use for 'what needs attention', triage, "
            "prioritisation, and for finding customers who were previously told something "
            "incorrect (stale_guidance)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "description": "critical | high | medium | low"},
                "types": {"type": "array", "items": {"type": "string"},
                          "description": ("Filter to specific signal types. Must be from the "
                                          "exact list in the tool description; omit to see all.")},
                "account_id": {"type": "string"},
            },
        },
    },
    {
        "name": "propose_escalation",
        "permission": Perm.CREATE_ESCALATION,
        "category": "action",
        "state_changing": True,
        "description": (
            "Prepare an escalation to the human support team for confirmation. This does "
            "NOT create it - it returns a preview that the user must explicitly confirm. "
            "Use when the request needs human judgement, an exception to policy, an action "
            "outside this system, a P1, a breached SLA, a credit above the approval "
            "threshold, or when sources conflict and the facts cannot be established."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "One line: what is being escalated."},
                "justification": {"type": "string",
                                  "description": "Why it needs a human, citing the rule or the missing fact."},
                "requested_action": {"type": "string",
                                     "description": "What you want the human to do."},
                "severity": {"type": "string", "description": "P1 | P2 | P3"},
                "ticket_id": {"type": "string"},
                "account_id": {"type": "string", "description": "Staff only."},
            },
            "required": ["summary", "justification", "requested_action"],
        },
    },
    {
        "name": "propose_ticket_update",
        "permission": Perm.UPDATE_TICKET,
        "category": "action",
        "state_changing": True,
        "description": (
            "Internal only. Prepare a change to a ticket's status or assignee for "
            "confirmation. Returns a before/after preview; nothing is written until the "
            "user confirms."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "field": {"type": "string", "description": "status | assigned_to"},
                "new_value": {"type": "string"},
                "note": {"type": "string", "description": "Why."},
            },
            "required": ["ticket_id", "field", "new_value"],
        },
    },
    {
        "name": "propose_followup_task",
        "permission": Perm.CREATE_TASK,
        "category": "action",
        "state_changing": True,
        "description": (
            "Internal only. Prepare a follow-up task for confirmation - for example to "
            "correct guidance a customer was previously given, chase a carrier, or verify "
            "a fact before promising a credit. Nothing is written until the user confirms."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "details": {"type": "string"},
                "owner": {"type": "string"},
                "due_at": {"type": "string", "description": "YYYY-MM-DD HH:MM"},
                "ticket_id": {"type": "string"},
                "account_id": {"type": "string"},
            },
            "required": ["title", "details"],
        },
    },
]

TOOL_BY_NAME = {t["name"]: t for t in TOOL_SPECS}
STATE_CHANGING = {t["name"] for t in TOOL_SPECS if t.get("state_changing")}


def tools_for(principal: Principal) -> list[dict]:
    """The Anthropic-shaped tool list this principal is allowed to see."""
    out = []
    for spec in TOOL_SPECS:
        perm = spec.get("permission")
        if perm is not None and not principal.has(perm):
            continue
        out.append({k: v for k, v in spec.items()
                    if k in ("name", "description", "input_schema")})
    return out


def tool_catalogue(principal: Principal) -> list[dict]:
    """UI-facing description of the tools available in this context."""
    return [
        {"name": s["name"], "category": s["category"],
         "state_changing": bool(s.get("state_changing"))}
        for s in TOOL_SPECS
        if s.get("permission") is None or principal.has(s["permission"])
    ]


# ---------------------------------------------------------------------------
# Executors: the only code that writes. Registered against the proposal store.
# ---------------------------------------------------------------------------
def _register_executors(gateway: DataGateway) -> None:
    def _escalate(principal: Principal, payload: dict) -> dict:
        return gateway.create_escalation(principal, **payload)

    def _update(principal: Principal, payload: dict) -> dict:
        return gateway.update_ticket(principal, **payload)

    def _task(principal: Principal, payload: dict) -> dict:
        return gateway.create_task(principal, **payload)

    register_executor("create_escalation", _escalate)
    register_executor("update_ticket", _update)
    register_executor("create_followup_task", _task)


# ---------------------------------------------------------------------------
class ToolRuntime:
    """Executes a tool call for one principal in one session."""

    def __init__(self, gateway: DataGateway, rules: Rules, index: ClauseIndex,
                 proposals: ProposalStore):
        self.db = gateway
        self.rules = rules
        self.index = index
        self.proposals = proposals
        _register_executors(gateway)

    # -- helpers ------------------------------------------------------------
    def _account_for(self, principal: Principal, account_id: str | None) -> dict:
        """Resolve the account in context.

        For a customer this ALWAYS resolves to their own account regardless of
        what the model passed - the model cannot widen its own scope by
        supplying a different id, deliberately or because a ticket description
        told it to.
        """
        target = principal.account_id if not principal.is_staff else account_id
        if not target:
            raise ValueError(
                "This is a staff session with no account in context. Specify account_id, "
                "or look up an order/ticket first to establish which account applies.")
        account = self.db.get_account(principal, target)
        if not account:
            raise AccessDenied("Account not found or not in scope.")
        return account

    def _contract_summary(self, account_id: str) -> dict:
        override = self.rules.override_for(account_id)
        if not override:
            return {"has_signed_agreement": False,
                    "note": "No customer agreement in the pack; the general policies apply."}
        return {
            "has_signed_agreement": True,
            "contract_file": override.get("contract"),
            "term": override.get("term"),
            "support_targets": override.get("first_response"),
            "coverage": override.get("coverage"),
            "cancellation": override.get("cancellation"),
            "service_credit": override.get("service_credit"),
            "note": ("These contractual terms override the general policy wherever they "
                     "conflict (Support Policy v3 s1)."),
        }

    # -- dispatch -----------------------------------------------------------
    def run(self, principal: Principal, session_id: str, name: str,
            args: dict[str, Any]) -> dict:
        spec = TOOL_BY_NAME.get(name)
        if not spec:
            raise ValueError(f"Unknown tool {name!r}")
        perm = spec.get("permission")
        if perm is not None:
            # Re-checked here even though the tool was filtered out of the
            # schema: never rely on the model only calling what it was offered.
            principal.require(perm, f"use {name}")
        handler: Callable = getattr(self, f"_t_{name}")
        return handler(principal, session_id, args)

    # -- documents ----------------------------------------------------------
    def _t_search_policy_documents(self, principal: Principal, session_id: str, args: dict) -> dict:
        query = args["query"]
        scope = principal.account_id
        if principal.is_staff:
            scope = args.get("account_id")
        hits = self.index.search(
            query,
            account_scope=scope,
            # Staff investigating a cross-account pattern still must not read a
            # customer agreement they have no account context for; contracts stay
            # scoped, and the tool says so rather than silently omitting them.
            include_other_accounts=False,
            topics=args.get("topics"),
            include_deprecated=bool(args.get("include_superseded")),
            k=int(args.get("k") or 6),
        )
        payload = {
            "query": query,
            "account_scope": scope,
            "results": [h.to_dict() for h in hits],
            "conflicts": self.index.detect_conflicts(hits),
            "precedence_rule": ("Signed customer agreement > current policy/SOP > current "
                                "product documentation > historical tickets (context only)."),
        }
        if args.get("include_superseded"):
            payload["warning"] = (
                "Superseded documents are included in these results. They record what a "
                "policy USED to say and must not be used to answer a current question.")
        return payload

    def _t_find_known_issues(self, principal: Principal, session_id: str, args: dict) -> dict:
        matches = self.rules.match_known_issues(args["description"], plan=args.get("plan"))
        return {
            "matches": [
                {**{k: v for k, v in ki.items() if k not in ("match_keywords", "source")},
                 "citation": self.rules.cite(ki.get("source"))}
                for ki in matches
            ],
            "note": ("No current known issue matches this description."
                     if not matches else
                     "A matching known issue means this is a recognised product problem "
                     "with a documented workaround, not a plan limitation."),
            "resolved_issue_caution": (
                "Resolved issues must not be used to explain new incidents unless the "
                "evidence specifically matches."),
        }

    # -- structured data ----------------------------------------------------
    def _t_get_account_context(self, principal: Principal, session_id: str, args: dict) -> dict:
        account = self._account_for(principal, args.get("account_id"))
        return {
            "account": account,
            "contract": self._contract_summary(account["account_id"]),
            "default_sla_targets": self.rules.raw["sla"]["default_first_response"].get(
                account.get("plan")),
            "snapshot": fmt(self.db.clock.now()),
        }

    def _t_lookup_orders(self, principal: Principal, session_id: str, args: dict) -> dict:
        if args.get("order_id"):
            order = self.db.get_order(principal, args["order_id"])
            if not order:
                return {"found": False,
                        "message": ("No order with that id is visible in this context. "
                                    "Check the id, or confirm it belongs to this account.")}
            return {"found": True, "order": order,
                    "account": self.db.get_account(principal, order["account_id"])}
        orders = self.db.list_orders(
            principal, account_id=args.get("account_id"),
            status=args.get("status"), carrier=args.get("carrier"))
        return {"count": len(orders), "orders": orders}

    def _t_lookup_tickets(self, principal: Principal, session_id: str, args: dict) -> dict:
        if args.get("ticket_id"):
            ticket = self.db.get_ticket(principal, args["ticket_id"])
            if not ticket:
                return {"found": False,
                        "message": "No ticket with that id is visible in this context."}
            out = {"found": True, "ticket": ticket}
            if ticket.get("historical_resolution") and principal.has(Perm.READ_INTERNAL_FIELDS):
                out["historical_resolution_warning"] = (
                    "This records what a past agent told the customer. It is context only "
                    "and may be incorrect - verify it against current rules before repeating it.")
            return out
        if args.get("search"):
            tickets = self.db.search_tickets(principal, args["search"])
        else:
            tickets = self.db.list_tickets(principal, account_id=args.get("account_id"),
                                           status=args.get("status"))
        return {"count": len(tickets), "tickets": tickets}

    # -- decisions ----------------------------------------------------------
    def _t_check_cancellation(self, principal: Principal, session_id: str, args: dict) -> dict:
        order = self.db.get_order(principal, args["order_id"])
        if not order:
            return {"found": False,
                    "message": "No order with that id is visible in this context."}
        account = self.db.get_account(principal, order["account_id"])
        verdict = evaluate_cancellation(order, account, self.rules, self.db.clock.now())
        return {"found": True, "verdict": verdict.to_dict()}

    def _t_check_service_credit(self, principal: Principal, session_id: str, args: dict) -> dict:
        order = None
        if args.get("order_id"):
            order = self.db.get_order(principal, args["order_id"])
            if not order:
                return {"found": False,
                        "message": "No order with that id is visible in this context."}
            account = self.db.get_account(principal, order["account_id"])
        else:
            account = self._account_for(principal, args.get("account_id"))
        verdict = evaluate_service_credit(
            self.rules, self.db.clock.now(), account=account, order=order,
            hours_past_window_end=args.get("hours_past_window_end"),
            carrier_fault=args.get("carrier_fault"),
            customer_fault=args.get("customer_fault"),
            shipment_fee_inr=args.get("shipment_fee_inr"),
        )
        return {"found": True, "verdict": verdict.to_dict()}

    def _t_check_sla(self, principal: Principal, session_id: str, args: dict) -> dict:
        ticket = self.db.get_ticket(principal, args["ticket_id"])
        if not ticket:
            return {"found": False,
                    "message": "No ticket with that id is visible in this context."}
        account = self.db.get_account(principal, ticket["account_id"])
        verdict = evaluate_sla(ticket, account, self.rules, self.db.clock.now(),
                               severity_override=args.get("severity_override"))
        return {"found": True, "verdict": verdict.to_dict(),
                "severity_definitions": self.rules.severity_definitions(),
                "business_hours_assumption": BUSINESS_HOURS_ASSUMPTION}

    # -- signals ------------------------------------------------------------
    def _t_get_operational_signals(self, principal: Principal, session_id: str, args: dict) -> dict:
        found = detect_signals(
            self.db.scan(principal, "accounts"), self.db.scan(principal, "orders"),
            self.db.scan(principal, "tickets"), self.rules, self.db.clock.now())
        signals = list(found)

        # An unknown filter value must be an error, never an empty result set.
        # A model that guesses `past_incorrect_guidance` instead of
        # `stale_guidance` and gets back zero signals will report "no such
        # problems exist" with total confidence - which is precisely the failure
        # this product exists to prevent. Say the filter was wrong instead.
        if args.get("types"):
            wanted = set(args["types"])
            unknown = sorted(wanted - set(SIGNAL_TYPES))
            if unknown:
                return {"error": f"Unknown signal type(s): {', '.join(unknown)}.",
                        "valid_types": sorted(SIGNAL_TYPES),
                        "note": ("No filtering was applied and nothing was searched. "
                                 "Re-run with a valid type, or with no type filter at all. "
                                 "Do not report these signals as absent.")}
            signals = [s for s in signals if s.type in wanted]
        if args.get("severity"):
            severity = str(args["severity"]).lower()
            if severity not in SIGNAL_SEVERITIES:
                return {"error": f"Unknown severity {args['severity']!r}.",
                        "valid_severities": sorted(SIGNAL_SEVERITIES),
                        "note": "No filtering was applied. Do not report signals as absent."}
            signals = [s for s in signals if s.severity == severity]
        if args.get("account_id"):
            signals = [s for s in signals if args["account_id"] in s.accounts]

        payload = {"summary": summarise(signals),
                   "signals": [s.to_dict() for s in signals[:25]],
                   "snapshot": fmt(self.db.clock.now())}
        # An empty *filtered* result is not evidence of absence, and the
        # difference matters enough to spell out in the payload.
        if not signals and found:
            payload["note"] = (
                f"No signals match this filter, but {len(found)} signals exist overall "
                f"(types present: {', '.join(sorted({s.type for s in found}))}). "
                "Re-run without the filter before telling the user nothing was found.")
        return payload

    # -- actions (proposals only) -------------------------------------------
    def _t_propose_escalation(self, principal: Principal, session_id: str, args: dict) -> dict:
        ticket = None
        if args.get("ticket_id"):
            ticket = self.db.get_ticket(principal, args["ticket_id"])
            if not ticket:
                return {"error": "That ticket is not visible in this context."}
        account_id = (ticket or {}).get("account_id") or (
            principal.account_id if not principal.is_staff else args.get("account_id"))
        if not account_id:
            return {"error": "An escalation needs an account. Provide account_id or ticket_id."}
        principal.assert_can_read_account(account_id)
        account = self.db.get_account(principal, account_id)

        severity = (args.get("severity") or "P3").upper()
        warnings: list[str] = []
        citations: list[dict] = []
        if ticket:
            sla = evaluate_sla(ticket, account, self.rules, self.db.clock.now(),
                               severity_override=args.get("severity"))
            severity = sla.facts["severity"]
            citations = sla.citations
            if sla.decision == "breached":
                warnings.append(f"SLA already breached: {sla.headline}")

        proposal = self.proposals.create(
            action="create_escalation", principal=principal, session_id=session_id,
            title=f"Escalate to the support team ({severity})",
            summary=args["summary"],
            preview={
                "account": f"{account.get('account_name')} ({account_id})",
                "ticket_id": args.get("ticket_id"),
                "severity": severity,
                "summary": args["summary"],
                "justification": args["justification"],
                "requested_action": args["requested_action"],
                "raised_by": f"{principal.display_name} ({principal.role})",
                "will_route_to": account.get("csm") or "ParcelPilot support queue",
            },
            payload={
                "account_id": account_id, "ticket_id": args.get("ticket_id"),
                "severity": severity, "summary": args["summary"],
                "justification": args["justification"],
                "requested_action": args["requested_action"],
            },
            warnings=warnings, citations=citations,
        )
        return {"status": "confirmation_required", "proposal": proposal.to_dict(),
                "instruction": ("Nothing has been created. Show the user exactly what will be "
                                "raised and ask them to confirm.")}

    def _t_propose_ticket_update(self, principal: Principal, session_id: str, args: dict) -> dict:
        ticket = self.db.get_ticket(principal, args["ticket_id"])
        if not ticket:
            return {"error": "That ticket is not visible in this context."}
        field = args["field"]
        if field not in ("status", "assigned_to"):
            return {"error": "Only 'status' and 'assigned_to' can be updated via this tool."}
        proposal = self.proposals.create(
            action="update_ticket", principal=principal, session_id=session_id,
            title=f"Update {args['ticket_id']}: {field}",
            summary=f"Change {field} from '{ticket.get(field)}' to '{args['new_value']}'",
            preview={"ticket_id": args["ticket_id"], "subject": ticket.get("subject"),
                     "field": field, "current_value": ticket.get(field),
                     "new_value": args["new_value"], "note": args.get("note")},
            payload={"ticket_id": args["ticket_id"], "field": field,
                     "new_value": args["new_value"], "note": args.get("note")},
        )
        return {"status": "confirmation_required", "proposal": proposal.to_dict(),
                "instruction": "Nothing has been changed yet. Ask the user to confirm."}

    def _t_propose_followup_task(self, principal: Principal, session_id: str, args: dict) -> dict:
        account_id = args.get("account_id") or (
            None if principal.is_staff else principal.account_id)
        if account_id:
            principal.assert_can_read_account(account_id)
        proposal = self.proposals.create(
            action="create_followup_task", principal=principal, session_id=session_id,
            title=f"Create follow-up task: {args['title']}",
            summary=args["details"][:200],
            preview={"title": args["title"], "details": args["details"],
                     "owner": args.get("owner") or principal.display_name,
                     "due_at": args.get("due_at"), "ticket_id": args.get("ticket_id"),
                     "account_id": account_id},
            payload={"title": args["title"], "details": args["details"],
                     "owner": args.get("owner") or principal.display_name,
                     "due_at": args.get("due_at"), "ticket_id": args.get("ticket_id"),
                     "account_id": account_id},
        )
        return {"status": "confirmation_required", "proposal": proposal.to_dict(),
                "instruction": "Nothing has been created yet. Ask the user to confirm."}


def serialise_result(result: Any) -> str:
    return json.dumps(result, default=str, ensure_ascii=False)


__all__ = ["ToolRuntime", "TOOL_SPECS", "TOOL_BY_NAME", "STATE_CHANGING",
           "tools_for", "tool_catalogue", "serialise_result"]
