"""Severity classification against the Support Policy v3 definitions.

Deliberately transparent rather than clever: each severity has patterns lifted
from the policy's own wording, and a match reports *which* phrase in the policy
it matched. The classifier returns evidence so a human can disagree with it, and
the SLA engine accepts an explicit override so the agent (or a support user) can
say "no, this is a P2" and have the maths redone.

The traps this must survive on the supplied data:
  * "Possible API key exposure" reads like a how-to question but the policy puts
    *suspected credential exposure* squarely in P1.
  * "All shipment creation is failing" is a P1 outage, not a P2 degradation.
  * "Bulk upload fails but one-by-one works" is P2, because a workaround exists.
"""
from __future__ import annotations

import re

P1_PATTERNS = [
    (r"\b(all|every|any)\b.{0,40}\bshipments?\b.{0,30}(fail\w*|error\w*|\b500\b)",
     "complete production outage preventing all shipment creation"),
    (r"\bevery user\b", "complete production outage preventing all shipment creation"),
    (r"\b(api key|api-key|credential|secret|token)\b.{0,40}\b(expos|leak|public|screenshot|post)",
     "suspected credential exposure"),
    (r"\b(expos|leak)\w*\b.{0,40}\b(api key|credential|secret|token)\b",
     "suspected credential exposure"),
    (r"\b(security incident|breach|compromis\w+)\b", "confirmed or suspected security incident"),
    (r"\b(complete|total)\b.{0,20}\b(outage|down)\b", "complete production outage"),
]

P2_PATTERNS = [
    (r"\b(bulk upload|csv|batch)\b.{0,40}(fail\w*|error\w*)",
     "major feature materially degraded with a workaround"),
    (r"\b(unavailable|degraded|not working|broken)\b",
     "major feature unavailable or materially degraded"),
    (r"\btimeout\w*\b|\bslow\b", "materially degraded feature"),
]

P3_PATTERNS = [
    (r"\bhow (do|can|to)\b", "how-to question"),
    (r"\b(change|update|replace)\b.{0,30}\b(contact|email|address|setting)\b",
     "configuration request"),
    (r"\bstill shows\b|\bnot updated\b|\bstatus\b.{0,20}\b(stale|wrong|incorrect)\b",
     "issue with limited operational impact"),
]

# A workaround existing is what separates P1 from P2 in the policy text, so an
# explicit statement of one is strong evidence *against* P1.
# Kept deliberately narrow. An earlier, looser version treated "existing
# shipments can still be viewed" as a workaround and downgraded a total
# shipment-creation outage to P2. Being able to *read* records is not a
# workaround for being unable to *create* them - the workaround has to apply to
# the operation that is actually broken.
WORKAROUND_SIGNALS = [
    (r"\bone[- ]by[- ]one\b|\bindividually\b|\bmanually\b",
     "customer reports a working alternative path for the same operation"),
    (r"\bworkaround\b", "a workaround is described"),
    (r"\b(creating|create|booking|book)\b[^.]{0,40}\bstill works?\b",
     "the same operation still succeeds by another route"),
]


def _scan(patterns, text) -> list[str]:
    hits = []
    for pattern, why in patterns:
        if re.search(pattern, text, re.I):
            hits.append(why)
    return list(dict.fromkeys(hits))


def classify_severity(subject: str, description: str = "") -> dict:
    text = f"{subject} {description}".strip()
    p1 = _scan(P1_PATTERNS, text)
    p2 = _scan(P2_PATTERNS, text)
    p3 = _scan(P3_PATTERNS, text)
    workarounds = _scan(WORKAROUND_SIGNALS, text)

    if p1:
        # Security beats the workaround signal: an exposed credential is P1 even
        # if everything else is working perfectly.
        security = any("credential" in w or "security" in w for w in p1)
        if workarounds and not security:
            return {
                "severity": "P2",
                "confidence": "medium",
                "evidence": p1 + workarounds,
                "note": ("Outage wording matched, but the report describes a working "
                         "alternative path, which the policy treats as P2."),
            }
        return {"severity": "P1", "confidence": "high", "evidence": p1, "note": None}
    if p2:
        return {"severity": "P2", "confidence": "medium" if not workarounds else "high",
                "evidence": p2 + workarounds, "note": None}
    if p3:
        return {"severity": "P3", "confidence": "medium", "evidence": p3, "note": None}
    return {
        "severity": "P3", "confidence": "low", "evidence": [],
        "note": ("No severity signal matched; defaulted to P3. A human should confirm "
                 "before this drives an SLA commitment."),
    }


__all__ = ["classify_severity"]
