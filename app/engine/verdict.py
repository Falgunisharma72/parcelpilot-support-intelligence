"""Shared verdict shape for every deterministic decision.

A verdict is deliberately verbose. It carries not just the outcome but the
facts it used, the arithmetic it performed, the rule that applied, the rule that
was *overridden*, and anything that should stop a human from acting on it. The
model's job downstream is to narrate this - not to re-derive it, and not to
improve on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Verdict:
    topic: str
    decision: str                      # machine-readable outcome
    headline: str                      # one-line human summary
    confidence: str = "high"           # high | medium | low
    facts: dict[str, Any] = field(default_factory=dict)
    computation: list[str] = field(default_factory=list)
    rule_applied: dict | None = None
    rule_overridden: dict | None = None
    citations: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    needs_human: bool = False
    needs_human_reason: str | None = None
    amount_inr: float | None = None
    assumptions: list[str] = field(default_factory=list)

    def cite(self, citation: dict | None) -> None:
        if citation and not any(c.get("clause_id") == citation.get("clause_id")
                                for c in self.citations):
            self.citations.append(citation)

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = ["Verdict"]
