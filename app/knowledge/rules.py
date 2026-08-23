"""Rules registry loader, with anchor verification against the live corpus.

The single most dangerous failure mode for this product is a rule that has
quietly drifted from the document it claims to come from - the system keeps
answering with total confidence from a number nobody can find in the pack any
more. So every rule in rules.yaml carries a verbatim `anchor`, and startup
re-parses the PDFs and checks each anchor still appears in the clause it cites.

Drift is a startup failure, not a runtime surprise.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import yaml

from app.config import RULES_FILE
from app.core.business_time import Duration
from app.ingest.docs import Clause, load_corpus


class RuleDriftError(RuntimeError):
    """A rule's citation no longer matches the document it came from."""


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _walk_sources(node: Any, path: str = "") -> list[tuple[str, dict]]:
    found: list[tuple[str, dict]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key.endswith("source") and isinstance(value, dict) and "clause_id" in value:
                found.append((here, value))
            else:
                found.extend(_walk_sources(value, here))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            found.extend(_walk_sources(item, f"{path}[{i}]"))
    return found


@dataclass
class Rules:
    raw: dict
    clauses_by_id: dict[str, Clause]

    # -- lookups ------------------------------------------------------------
    def override_for(self, account_id: str | None) -> dict:
        if not account_id:
            return {}
        return self.raw.get("account_overrides", {}).get(account_id, {}) or {}

    def cancellation_rule(self, account_id: str | None, status: str) -> dict:
        """Resolve the cancellation rule for an account+status, contract first.

        Returns the applicable rule plus the precedence trail, so the caller can
        explain *why* this rule won rather than just asserting the outcome.
        """
        status = (status or "").upper()
        default = self.raw["cancellation"]["default"].get(status)
        override = (self.override_for(account_id).get("cancellation") or {})
        contract_rule = override.get(status)

        if contract_rule and not override.get("inherits_default"):
            return {
                "applied": contract_rule,
                "authority": "contract",
                "authority_tier": 1,
                "source": contract_rule.get("source") or override.get("source"),
                "overrides": {"rule": default, "source": (default or {}).get("source")},
            }
        return {
            "applied": default,
            "authority": "sop",
            "authority_tier": 2,
            "source": (default or {}).get("source"),
            "overrides": None,
            "contract_checked": bool(override),
            "contract_note": (override.get("cancellation") or {}).get("source")
            if override else None,
        }

    def service_credit_rule(self, account_id: str | None) -> dict:
        default = self.raw["service_credit"]["default_failed_pickup"]
        override = self.override_for(account_id).get("service_credit") or {}
        contract_rule = override.get("failed_pickup")
        if isinstance(contract_rule, dict) and contract_rule.get("replaces_default"):
            return {
                "applied": contract_rule,
                "authority": "contract",
                "authority_tier": 1,
                "source": override.get("source"),
                "overrides": {"rule": default, "source": default.get("source")},
                "monthly_cap_inr": override.get("monthly_aggregate_cap_inr"),
            }
        return {
            "applied": default,
            "authority": "sop",
            "authority_tier": 2,
            "source": default.get("source"),
            "overrides": None,
            "monthly_cap_inr": override.get("monthly_aggregate_cap_inr"),
        }

    def sla_target(self, account_id: str | None, plan: str, severity: str) -> dict:
        severity = severity.upper()
        override = self.override_for(account_id)
        contract_targets = override.get("first_response") or {}
        coverage = override.get("coverage") or {}

        if severity in contract_targets:
            spec = dict(contract_targets[severity])
            # A contract that excludes weekend/after-hours coverage cannot have a
            # 24x7 calendar clock: the coverage clause qualifies every target in
            # that agreement. Applying one clause of a contract while ignoring
            # another is how you produce an answer that is contractually wrong.
            if coverage.get("weekend_or_after_hours") is False:
                spec["clock"] = "business"
            return {
                "duration": Duration.parse(spec),
                "authority": "contract",
                "authority_tier": 1,
                "source": override.get("first_response_source"),
                "coverage_note": coverage.get("note"),
                "overrides": {
                    "duration": Duration.parse(
                        self.raw["sla"]["default_first_response"][plan][severity]
                    ),
                    "source": self.raw["sla"]["source"],
                },
            }

        plan_targets = self.raw["sla"]["default_first_response"].get(plan)
        if not plan_targets:
            raise KeyError(f"No SLA targets defined for plan {plan!r}")
        return {
            "duration": Duration.parse(plan_targets[severity]),
            "authority": "policy",
            "authority_tier": 2,
            "source": self.raw["sla"]["source"],
            "coverage_note": None,
            "overrides": None,
        }

    def severity_definitions(self) -> dict:
        return {k: v for k, v in self.raw["severity"].items() if k.startswith("P")}

    def known_issues(self) -> list[dict]:
        return self.raw.get("known_issues", [])

    def match_known_issues(self, text: str, plan: str | None = None) -> list[dict]:
        low = _normalise(text)
        out = []
        for ki in self.known_issues():
            if any(_normalise(kw) in low for kw in ki.get("match_keywords", [])):
                plans = ki.get("applies_to_plans")
                if plans and plan and plan not in plans:
                    continue
                out.append(ki)
        return out

    def escalation_triggers(self) -> list[dict]:
        return self.raw.get("escalation_triggers", [])

    def manager_approval_threshold(self) -> float:
        return float(self.raw["service_credit"]["manager_approval_above_inr"])

    def cite(self, source: dict | None) -> dict | None:
        """Turn a rule source into a full citation with the clause text."""
        if not source:
            return None
        clause = self.clauses_by_id.get(source["clause_id"])
        if clause is None:
            return {"clause_id": source["clause_id"], "citation": source["clause_id"]}
        return {
            "clause_id": clause.clause_id,
            "citation": clause.citation(),
            "doc_id": clause.doc_id,
            "section": clause.section,
            "text": clause.text,
            "authority_tier": clause.authority_tier,
            "account_scope": clause.account_scope,
        }

    # -- integrity ----------------------------------------------------------
    def verify(self) -> list[str]:
        problems: list[str] = []
        for path, source in _walk_sources(self.raw):
            clause = self.clauses_by_id.get(source["clause_id"])
            if clause is None:
                problems.append(f"{path}: clause {source['clause_id']} no longer exists in the document pack")
                continue
            anchor = source.get("anchor")
            if anchor and _normalise(anchor) not in _normalise(clause.text):
                problems.append(
                    f"{path}: anchor {anchor!r} not found in {source['clause_id']} "
                    f"(document text changed)"
                )
        return problems


@lru_cache(maxsize=1)
def get_rules(strict: bool = True) -> Rules:
    raw = yaml.safe_load(RULES_FILE.read_text())
    clauses = {c.clause_id: c for c in load_corpus()}
    rules = Rules(raw=raw, clauses_by_id=clauses)
    problems = rules.verify()
    if problems and strict:
        raise RuleDriftError(
            "Rules registry is out of sync with the document pack:\n  - "
            + "\n  - ".join(problems)
            + "\n\nRegenerate with `python -m scripts.extract_rules` and review the diff."
        )
    return rules


__all__ = ["Rules", "get_rules", "RuleDriftError"]
