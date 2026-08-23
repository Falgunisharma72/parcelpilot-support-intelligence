"""Authority-aware clause retrieval.

Why BM25 and not a vector database
----------------------------------
The supplied corpus is six single-page documents -> 37 clauses. A vector store
here would add an index, an embedding provider, a network hop and a similarity
threshold to tune, in exchange for nothing measurable: at this size lexical
search with field boosts is both more accurate (exact tokens like "PICKED_UP",
"KI-208", "ACCT-001" and "INR 250" are what queries actually contain) and fully
deterministic, which matters because we run a golden-set eval over it.

What actually determines answer quality at this scale is not the similarity
metric, it is *ranking by authority and scope*: a signed customer agreement must
outrank a general SOP, and a DEPRECATED document must never surface as an
answer at all. That logic lives here.

The scaling path is written down rather than pre-built - see ARCHITECTURE.md.
`search()` is the seam: swap the candidate generator for hybrid BM25 + vector
recall and the authority ranking, scoping and conflict layers above it are
untouched.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.ingest.docs import (
    Clause, TIER_DEPRECATED, TIER_HISTORICAL, TIER_LABELS, load_corpus,
)

_TOKEN_RE = re.compile(r"[a-z0-9_\-]+")
_STOP = {
    "the", "a", "an", "of", "to", "for", "is", "are", "and", "or", "in", "on",
    "at", "be", "by", "it", "that", "this", "with", "as", "was", "we", "i",
    "can", "do", "does", "what", "how", "if", "my", "our", "you", "your",
}


def tokenize(text: str) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower())
    out: list[str] = []
    for t in toks:
        if t in _STOP or len(t) < 2:
            continue
        out.append(t)
        # "ord-1001" / "acct-001" / "ki-208" should also match their parts, so a
        # question about "KI 208" or "order 1001" still hits.
        if "-" in t:
            out.extend(p for p in t.split("-") if p and p not in _STOP)
    return out


@dataclass
class Hit:
    clause: Clause
    score: float
    lexical_score: float
    reasons: list[str]

    def to_dict(self) -> dict:
        c = self.clause
        return {
            "clause_id": c.clause_id,
            "citation": c.citation(),
            "doc_id": c.doc_id,
            "section": c.section,
            "text": c.text,
            "authority_tier": c.authority_tier,
            "authority": TIER_LABELS[c.authority_tier],
            "status": c.status,
            "effective_date": c.effective_date,
            "account_scope": c.account_scope,
            "topics": c.topics,
            "score": round(self.score, 3),
            "ranking_reasons": self.reasons,
        }


class ClauseIndex:
    """BM25 over clauses, with authority/scope aware re-ranking."""

    K1 = 1.4
    B = 0.72

    def __init__(self, clauses: list[Clause] | None = None):
        self.clauses = clauses if clauses is not None else load_corpus()
        self._docs: list[list[str]] = []
        for c in self.clauses:
            # Section headings carry real signal ("2. Failed-pickup service
            # credits") and are weighted by repetition rather than a separate
            # field index - simple, and enough at this corpus size.
            self._docs.append(tokenize(f"{c.section} {c.section} {c.doc_title} {c.text}"))
        self._tf = [Counter(d) for d in self._docs]
        self._len = [len(d) or 1 for d in self._docs]
        self._avg = sum(self._len) / max(1, len(self._len))
        self._df: Counter[str] = Counter()
        for d in self._docs:
            for t in set(d):
                self._df[t] += 1
        self._n = len(self._docs)

    # -- scoring ------------------------------------------------------------
    def _bm25(self, q_tokens: list[str], i: int) -> float:
        score = 0.0
        tf, dl = self._tf[i], self._len[i]
        for t in q_tokens:
            df = self._df.get(t, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self._n - df + 0.5) / (df + 0.5))
            f = tf.get(t, 0)
            if f == 0:
                continue
            score += idf * (f * (self.K1 + 1)) / (f + self.K1 * (1 - self.B + self.B * dl / self._avg))
        return score

    def search(
        self,
        query: str,
        *,
        account_scope: str | None = None,
        include_other_accounts: bool = False,
        topics: list[str] | None = None,
        doc_types: list[str] | None = None,
        include_deprecated: bool = False,
        k: int = 6,
    ) -> list[Hit]:
        """Retrieve clauses, ranked by relevance *and* authority.

        `account_scope` is a hard filter, not a hint: contract clauses belonging
        to another customer are removed before ranking, so no amount of clever
        prompting can surface them.
        """
        q = tokenize(query)
        hits: list[Hit] = []
        for i, c in enumerate(self.clauses):
            # --- hard filters (security / correctness, never negotiable) ---
            if c.account_scope and not include_other_accounts and c.account_scope != account_scope:
                continue
            if c.status == "DEPRECATED" and not include_deprecated:
                continue
            if doc_types and c.doc_type not in doc_types:
                continue
            if topics and not set(topics) & set(c.topics):
                continue

            lex = self._bm25(q, i)
            if lex <= 0 and not topics:
                continue

            reasons: list[str] = []
            score = lex

            # --- authority boost ---
            # Tier 1 (signed agreement) > tier 2 (current policy/SOP) > tier 3
            # (product docs). Multiplicative so it re-orders near-ties without
            # letting a barely-relevant contract clause beat a direct hit.
            tier_boost = {1: 1.6, 2: 1.25, 3: 1.0}.get(c.authority_tier, 0.35)
            score *= tier_boost
            if c.authority_tier == 1:
                reasons.append("signed customer agreement (highest authority)")
            elif c.authority_tier == 3:
                reasons.append("product documentation")

            if c.account_scope and c.account_scope == account_scope:
                score *= 1.35
                reasons.append(f"scoped to this account ({c.account_scope})")
            if c.status == "DEPRECATED":
                score *= 0.05
                reasons.append("DEPRECATED - not valid authority")
            if topics and set(topics) & set(c.topics):
                score *= 1.15
                reasons.append(f"topic match: {', '.join(sorted(set(topics) & set(c.topics)))}")

            hits.append(Hit(clause=c, score=score, lexical_score=lex, reasons=reasons))

        hits.sort(key=lambda h: (-h.score, h.clause.authority_tier, h.clause.clause_id))
        return hits[:k]

    # -- conflict detection --------------------------------------------------
    # Only *normative* topics can conflict. "shipment_status" and
    # "plan_capability" are descriptive - two documents both explaining what
    # BOOKED means is not a conflict, and flagging it as one trains the reader
    # to ignore the conflict banner entirely.
    CONFLICT_TOPICS = {
        "cancellation", "service_credit", "failed_pickup", "sla",
        "bulk_upload", "approval",
    }

    def detect_conflicts(self, hits: list[Hit]) -> list[dict]:
        """Flag retrieved clauses that speak to the same normative topic from
        different authority tiers.

        This does not attempt to prove semantic contradiction - at this corpus
        size that would be over-engineering with a high false-negative rate. It
        flags *potential* conflicts by topic-and-tier overlap and states which
        source wins under the precedence rule, so an answer can say "the SOP
        says X, but your agreement overrides it with Y" instead of silently
        picking one and sounding certain.
        """
        by_topic: dict[str, list[Hit]] = {}
        for h in hits:
            for t in h.clause.topics:
                if t in self.CONFLICT_TOPICS:
                    by_topic.setdefault(t, []).append(h)

        conflicts: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for topic, group in by_topic.items():
            live_tiers = {h.clause.authority_tier for h in group
                          if h.clause.authority_tier != TIER_DEPRECATED}
            has_deprecated = any(h.clause.authority_tier == TIER_DEPRECATED for h in group)
            if len(live_tiers) < 2 and not has_deprecated:
                continue
            winner = min(group, key=lambda h: h.clause.authority_tier)
            losers = [h for h in group
                      if h.clause.authority_tier > winner.clause.authority_tier
                      and h.clause.doc_id != winner.clause.doc_id]
            if not losers:
                continue
            key = (winner.clause.clause_id, "|".join(sorted(l.clause.clause_id for l in losers)))
            if key in seen:
                continue
            seen.add(key)

            if winner.clause.authority_tier == 1:
                resolution = (
                    f"The signed agreement for {winner.clause.account_scope} takes precedence "
                    "over the general policy (Support Policy v3 s1: source precedence)."
                )
            elif any(l.clause.authority_tier == TIER_DEPRECATED for l in losers):
                resolution = "A superseded document was matched and has been excluded."
            else:
                resolution = "The current policy/SOP takes precedence over product documentation."

            conflicts.append({
                "topic": topic,
                "resolution": resolution,
                "authoritative": winner.clause.citation(),
                "authoritative_tier": TIER_LABELS[winner.clause.authority_tier],
                "authoritative_text": winner.clause.text,
                "overridden": [
                    {"citation": h.clause.citation(),
                     "tier": TIER_LABELS[h.clause.authority_tier],
                     "text": h.clause.text}
                    for h in losers
                ],
            })
        return conflicts


_INDEX: ClauseIndex | None = None


def get_index() -> ClauseIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = ClauseIndex()
    return _INDEX


__all__ = ["ClauseIndex", "Hit", "get_index", "tokenize"]
