"""Two-phase commit for state-changing actions.

The requirement is that any state-changing action needs explicit user
confirmation. The weak way to implement that is to tell the model to ask first.
This is the strong way:

  * the agent's action tools are *proposal builders*. They validate, resolve the
    target record, render a preview - and write nothing;
  * the code that actually mutates data lives behind `execute()`, which requires
    a proposal id that only a separate, authenticated `POST /api/confirm` can
    supply.

So the confirmation is a property of the architecture, not of the prompt. A
jailbroken model, a prompt injection inside a ticket description, or a bug in
the system prompt still cannot write to the database, because the write path
needs a token the model never sees the ability to mint on its own behalf.

Proposals are additionally:
  * single-use     - confirming twice does not create two escalations;
  * TTL-bounded    - a confirmation given against a stale preview is refused;
  * principal-bound - a customer session cannot confirm a proposal built in a
    staff session, and vice versa.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from app.config import PROPOSAL_TTL_SECONDS
from app.core.principal import AccessDenied, Principal


class ProposalError(RuntimeError):
    pass


@dataclass
class Proposal:
    proposal_id: str
    action: str
    session_id: str
    user_id: str
    role: str
    title: str
    summary: str
    preview: dict[str, Any]
    payload: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    status: str = "pending"          # pending | confirmed | cancelled | expired
    result: dict | None = None

    @property
    def expired(self) -> bool:
        return time.time() - self.created_at > PROPOSAL_TTL_SECONDS

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "action": self.action,
            "title": self.title,
            "summary": self.summary,
            "preview": self.preview,
            "warnings": self.warnings,
            "citations": self.citations,
            "status": self.status,
            "expires_in_seconds": max(
                0, int(PROPOSAL_TTL_SECONDS - (time.time() - self.created_at))),
            "requires_confirmation": True,
        }


# Executors are registered by the tool layer. Keeping the registry here means
# there is exactly one code path from "a human confirmed" to "the database
# changed", and it is easy to audit.
_EXECUTORS: dict[str, Callable[[Principal, dict], dict]] = {}


def register_executor(action: str, fn: Callable[[Principal, dict], dict]) -> None:
    _EXECUTORS[action] = fn


class ProposalStore:
    """In-process store. Production would use Redis or a table with a TTL;
    the interface is deliberately narrow so that swap is a one-file change."""

    def __init__(self) -> None:
        self._items: dict[str, Proposal] = {}
        self._lock = threading.Lock()

    def create(self, *, action: str, principal: Principal, session_id: str,
               title: str, summary: str, preview: dict, payload: dict,
               warnings: list[str] | None = None,
               citations: list[dict] | None = None) -> Proposal:
        if action not in _EXECUTORS:
            raise ProposalError(f"No executor registered for action {action!r}")
        proposal = Proposal(
            proposal_id=f"prop_{uuid.uuid4().hex[:12]}", action=action,
            session_id=session_id, user_id=principal.user_id, role=principal.role,
            title=title, summary=summary, preview=preview, payload=payload,
            warnings=warnings or [], citations=citations or [],
        )
        with self._lock:
            self._items[proposal.proposal_id] = proposal
        return proposal

    def get(self, proposal_id: str) -> Proposal:
        with self._lock:
            proposal = self._items.get(proposal_id)
        if proposal is None:
            raise ProposalError("No such pending action. It may have already been handled.")
        return proposal

    def _authorise(self, proposal: Proposal, principal: Principal, session_id: str) -> None:
        # Confirmation must come from the same person, in the same session, that
        # was shown the preview.
        if proposal.user_id != principal.user_id or proposal.session_id != session_id:
            raise AccessDenied("This pending action belongs to a different session.")

    def confirm(self, proposal_id: str, principal: Principal, session_id: str) -> Proposal:
        proposal = self.get(proposal_id)
        self._authorise(proposal, principal, session_id)
        if proposal.status == "confirmed":
            raise ProposalError("This action has already been carried out.")
        if proposal.status == "cancelled":
            raise ProposalError("This action was cancelled and cannot be confirmed.")
        if proposal.expired:
            proposal.status = "expired"
            raise ProposalError(
                "This pending action has expired. Ask again so the details can be "
                "re-checked against current data before anything is written.")

        executor = _EXECUTORS[proposal.action]
        # The proposal id travels into the written row, so every mutation in the
        # database points back at the confirmation that authorised it.
        result = executor(principal,
                          {**proposal.payload, "proposal_id": proposal.proposal_id})
        with self._lock:
            proposal.status = "confirmed"
            proposal.result = result
        return proposal

    def cancel(self, proposal_id: str, principal: Principal, session_id: str) -> Proposal:
        proposal = self.get(proposal_id)
        self._authorise(proposal, principal, session_id)
        if proposal.status == "pending":
            proposal.status = "cancelled"
        return proposal

    def pending_for(self, session_id: str) -> list[Proposal]:
        with self._lock:
            return [p for p in self._items.values()
                    if p.session_id == session_id and p.status == "pending" and not p.expired]


_STORE = ProposalStore()


def get_store() -> ProposalStore:
    return _STORE


__all__ = ["Proposal", "ProposalStore", "ProposalError", "get_store",
           "register_executor"]
