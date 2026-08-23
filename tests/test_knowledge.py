"""Retrieval, authority ranking and rules integrity."""
from app.ingest.docs import TIER_CONTRACT, TIER_DEPRECATED, TIER_POLICY


def test_every_rule_anchor_still_matches_its_document(rules):
    """The guard against silent drift: if a document is replaced and a threshold
    moves, this fails rather than the system answering from a stale number."""
    assert rules.verify() == []


def test_documents_are_tiered_correctly(index):
    tiers = {c.doc_id: c.authority_tier for c in index.clauses}
    assert tiers["05_Northstar_Logistics_Enterprise_Agreement.pdf"] == TIER_CONTRACT
    assert tiers["01_Support_Policy_v3_CURRENT.pdf"] == TIER_POLICY
    assert tiers["02_Support_Policy_v2_DEPRECATED.pdf"] == TIER_DEPRECATED


def test_policy_body_mentioning_agreements_is_not_promoted_to_contract(index):
    """Support Policy v3 says 'a signed customer agreement may override these
    defaults'. Classifying on body text would promote the policy above the
    contracts it defers to."""
    policy = [c for c in index.clauses if c.doc_id.startswith("01_")]
    assert all(c.doc_type == "policy" for c in policy)


def test_contract_clause_outranks_the_general_sop(index):
    hits = index.search("cancel booked shipment fee", account_scope="ACCT-001", k=4)
    assert hits[0].clause.authority_tier == TIER_CONTRACT


def test_short_contractual_bullets_survive_chunking(index):
    """'P2: 1 hour' is ten characters and is a binding SLA term."""
    texts = {c.text for c in index.clauses}
    assert "P2: 1 hour" in texts
    assert "P1: 2 business hours" in texts


def test_known_issues_are_separately_addressable(index):
    ki = {c.clause_id for c in index.clauses if "KI-211" in c.text}
    assert ki, "KI-211 must be its own clause, not merged into KI-208"


def test_conflicts_are_reported_with_a_resolution(index):
    hits = index.search("service credit for a late pickup", account_scope="ACCT-002", k=6)
    conflicts = index.detect_conflicts(hits)
    assert conflicts
    assert any("agreement" in c["resolution"] for c in conflicts)


def test_descriptive_overlap_is_not_reported_as_a_conflict(index):
    """Two documents both explaining what BOOKED means is not a conflict, and
    flagging it as one teaches the reader to ignore the banner."""
    hits = index.search("what does BOOKED mean", account_scope="ACCT-003", k=6)
    topics = {c["topic"] for c in index.detect_conflicts(hits)}
    assert "shipment_status" not in topics


def test_known_issue_matching_respects_plan(rules):
    assert rules.match_known_issues("bulk upload of a large csv fails", plan="Growth")
    assert not rules.match_known_issues("bulk upload of a large csv fails", plan="Standard")


def test_known_issue_carries_the_real_product_limit(runtime, lumenworks):
    """The customer's belief is usually a *wrong limit* the known issue produced
    ("our plan caps at 3,000 rows"). Returning the documented limit alongside the
    issue lets one answer correct the misconception rather than leave it standing."""
    result = runtime.run(lumenworks, "s", "find_known_issues",
                         {"description": "our 4,200-row CSV bulk upload fails at 70%",
                          "plan": "Growth"})
    assert [m["id"] for m in result["matches"]] == ["KI-208"]
    capability = result["related_plan_capabilities"][0]
    assert capability["max_rows"] == 5000
    assert "Growth" in capability["available_on"]


def test_unrelated_symptom_does_not_drag_in_capabilities(runtime, lumenworks):
    result = runtime.run(lumenworks, "s", "find_known_issues",
                         {"description": "pickup status stuck on BOOKED"})
    assert result["related_plan_capabilities"] == []
