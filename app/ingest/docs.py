"""PDF -> structured, attributed clauses.

Design note: we do not treat the document pack as an undifferentiated blob of
text. Each PDF is parsed into *clauses* (a numbered section or a bullet within
it) and each clause carries the provenance it needs to be trusted or distrusted:
which document it came from, whether that document is CURRENT or DEPRECATED,
when it took effect, whether it is scoped to a single customer account, and
which authority tier it sits in.

That metadata is what makes conflict resolution possible later. A retrieval
system that returns "INR 250 cancellation fee" without also returning "this is
a general SOP and ACCT-001 has a signed override" is how you produce a
confidently incorrect answer.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict, field
from datetime import date, datetime
from pathlib import Path

import pdfplumber

from app.config import DOCS_DIR

# --- Authority tiers --------------------------------------------------------
# Straight from Support Policy v3 s1: "use the signed customer agreement first,
# then the current support policy, then current product documentation.
# Historical tickets and internal notes are context only."
TIER_CONTRACT = 1
TIER_POLICY = 2
TIER_PRODUCT_DOC = 3
TIER_HISTORICAL = 4
TIER_DEPRECATED = 99  # never usable as authority

TIER_LABELS = {
    TIER_CONTRACT: "Signed customer agreement",
    TIER_POLICY: "Current policy / SOP",
    TIER_PRODUCT_DOC: "Current product documentation",
    TIER_HISTORICAL: "Historical context (not authoritative)",
    TIER_DEPRECATED: "Superseded document (not usable)",
}

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}


def _parse_long_date(text: str) -> date | None:
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", text)
    if not m:
        return None
    month = _MONTHS.get(m.group(2).lower())
    if not month:
        return None
    return date(int(m.group(3)), month, int(m.group(1)))


@dataclass
class Clause:
    clause_id: str
    doc_id: str
    doc_title: str
    doc_type: str          # contract | policy | sop | product_doc
    status: str            # CURRENT | DEPRECATED
    authority_tier: int
    section: str
    text: str
    page: int
    account_scope: str | None = None   # e.g. ACCT-001 for a signed agreement
    effective_date: str | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    topics: list[str] = field(default_factory=list)

    def citation(self) -> str:
        where = f"{self.doc_title} - {self.section}" if self.section else self.doc_title
        return f"{where} ({self.doc_id})"

    def to_dict(self) -> dict:
        return asdict(self)


# --- Topic tagging ----------------------------------------------------------
# A small, explicit lexicon. This is deliberately not an embedding model: with
# six one-page documents, keyword topics are more predictable, are trivially
# auditable, and let us detect *cross-document conflicts on the same topic*,
# which is the property we actually need.
TOPIC_PATTERNS = {
    "cancellation": r"\bcancel\w*",
    "service_credit": r"service credit|\bcredits?\b|\brefund\w*",
    "failed_pickup": r"\bpick-?up\b|\bcollected\b",
    "sla": r"response target|first-?response|severity|\bp[123]\b|\bsla\b|support term",
    "bulk_upload": r"bulk upload|\bcsv\b|\brows\b",
    "known_issue": r"known issue|\bki-\d+|workaround|investigating|monitoring",
    "security": r"\bsecurity\b|api key|credential|exposure",
    "plan_capability": r"\b(enterprise|growth|standard)\b|\bplans?\b|available on",
    "source_precedence": r"precedence|conflict|supersed\w*|override\w*|authority",
    "approval": r"\bapprovals?\b|\bmanager\b|\bcapped?\b",
    "escalation": r"escalat\w*",
    "shipment_status": r"\b(booked|picked_up|delivered|draft)\b|return-to-origin",
}


def tag_topics(text: str) -> list[str]:
    low = text.lower()
    return [t for t, pat in TOPIC_PATTERNS.items() if re.search(pat, low)]


# --- Parsing ----------------------------------------------------------------
_SECTION_RE = re.compile(r"^(\d+)\.\s+(.+)$")
_BULLET_RE = re.compile(r"^[●•\-\*]\s*(.+)$")
# Known issues are titled sub-blocks inside a numbered section ("KI-208 - Bulk
# Upload failures..."). Without this, two unrelated incidents merge into one
# chunk and a citation can no longer point at the specific issue.
_SUBSEC_RE = re.compile(r"^(KI-\d+)\s*[-\u2013]\s*(.+)$")


def _doc_metadata(doc_id: str, title: str, body: str) -> dict:
    status_m = re.search(r"Status:\s*([A-Z][A-Z \-]*)", body)
    status_raw = (status_m.group(1).strip() if status_m else "CURRENT")
    status = "DEPRECATED" if status_raw.startswith("DEPRECATED") else "CURRENT"

    account_m = re.search(r"Account:\s*(ACCT-\d+)", body)
    account_scope = account_m.group(1) if account_m else None

    eff_m = re.search(r"Effective:\s*([^\n]+)", body)
    updated_m = re.search(r"Updated:\s*([^\n]+)", body)
    term_m = re.search(r"Term:\s*(\d{1,2}\s+\w+\s+\d{4})", body)
    eff_src = (eff_m or updated_m or term_m)
    effective = _parse_long_date(eff_src.group(1)) if eff_src else None

    sup_m = re.search(r"Supersedes:\s*([^\n]+)", body)
    supby_m = re.search(r"Superseded by:\s*([^\n]+)", body)

    # Classify from the document *title* and its account scope only. Deliberately
    # not from the body: Support Policy v3 says "a signed customer agreement may
    # override these defaults", and matching "agreement" in the body would
    # promote the general policy to contract authority - the exact inversion the
    # precedence rule exists to prevent.
    low = f"{title} {doc_id}".lower()
    if account_scope or "agreement" in low:
        doc_type = "contract"
    elif "sop" in low:
        doc_type = "sop"
    elif "policy" in low:
        doc_type = "policy"
    else:
        doc_type = "product_doc"

    if status == "DEPRECATED":
        tier = TIER_DEPRECATED
    elif doc_type == "contract":
        tier = TIER_CONTRACT
    elif doc_type in ("policy", "sop"):
        tier = TIER_POLICY
    else:
        tier = TIER_PRODUCT_DOC

    return {
        "status": status,
        "account_scope": account_scope,
        "effective_date": effective.isoformat() if effective else None,
        "supersedes": sup_m.group(1).strip() if sup_m else None,
        "superseded_by": supby_m.group(1).strip() if supby_m else None,
        "doc_type": doc_type,
        "authority_tier": tier,
    }


def parse_pdf(path: Path) -> list[Clause]:
    with pdfplumber.open(path) as pdf:
        pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(pdf.pages)]

    body = "\n".join(t for _, t in pages)
    doc_id = path.name
    lines_all = [l.strip() for l in body.splitlines() if l.strip()]
    # Title = leading lines before the first metadata key.
    title_lines: list[str] = []
    for line in lines_all[:4]:
        if re.match(r"^(Status|Account|Effective|Term|Updated|Customer|Plan|Supersed)", line):
            break
        title_lines.append(line)
    title = " ".join(title_lines) or path.stem

    meta = _doc_metadata(doc_id, title, body)

    clauses: list[Clause] = []
    section = "Header"
    sub_section = ""
    sec_no = 0
    buf: list[str] = []
    page_of_section = 1

    def flush(page: int) -> None:
        nonlocal buf
        text = " ".join(buf).strip()
        buf = []
        # Keep short bullets. "P2: 1 hour" is ten characters and is a binding
        # contractual SLA target; a length filter tuned for prose would silently
        # delete half of a customer agreement.
        if len(text) < 4:
            return
        # Sequence within the numbered section, so ids stay stable even when a
        # section contains several titled sub-blocks.
        idx = len([c for c in clauses if c.clause_id.startswith(f"{doc_id}#s{sec_no}.")]) + 1
        cid = f"{doc_id}#s{sec_no}.{idx}"
        label = f"{section} - {sub_section}" if sub_section else section
        clauses.append(Clause(
            clause_id=cid, doc_id=doc_id, doc_title=title,
            section=label, text=text, page=page,
            topics=tag_topics(f"{label} {text}"),
            **meta,
        ))

    for page_no, page_text in pages:
        for raw in page_text.splitlines():
            line = raw.strip()
            if not line:
                continue
            sec_m = _SECTION_RE.match(line)
            if sec_m:
                flush(page_of_section)
                sec_no = int(sec_m.group(1))
                section = f"{sec_no}. {sec_m.group(2).strip()}"
                sub_section = ""
                page_of_section = page_no
                continue
            sub_m = _SUBSEC_RE.match(line)
            if sub_m:
                flush(page_of_section)
                # Keep the sub-heading label short - it goes into citations.
                head = sub_m.group(2).strip().split(".")[0][:60].strip()
                sub_section = f"{sub_m.group(1)} {head}"
                buf.append(line)
                page_of_section = page_no
                continue
            bullet_m = _BULLET_RE.match(line)
            if bullet_m:
                flush(page_of_section)
                buf.append(bullet_m.group(1).strip())
                page_of_section = page_no
                continue
            # A table row or continuation line: keep it with the current buffer.
            buf.append(line)
        flush(page_of_section)

    # Documents whose body is a table (the SLA grids) survive as one clause per
    # row-run, which keeps "Enterprise 30 minutes, 24x7 2 hours 1 business day"
    # intact and retrievable.
    return clauses


def load_corpus(docs_dir: Path = DOCS_DIR) -> list[Clause]:
    clauses: list[Clause] = []
    for path in sorted(docs_dir.glob("*.pdf")):
        clauses.extend(parse_pdf(path))
    return clauses


def corpus_fingerprint(clauses: list[Clause]) -> str:
    h = hashlib.sha256()
    for c in clauses:
        h.update(c.clause_id.encode())
        h.update(c.text.encode())
    return h.hexdigest()[:12]


if __name__ == "__main__":  # pragma: no cover - inspection helper
    cs = load_corpus()
    print(f"{len(cs)} clauses, fingerprint {corpus_fingerprint(cs)}")
    for c in cs:
        print(f"[T{c.authority_tier}] {c.doc_id} | {c.section} | scope={c.account_scope} "
              f"| {c.status} | topics={','.join(c.topics)}\n    {c.text[:160]}")
