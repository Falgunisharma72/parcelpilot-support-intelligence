"""HTTP API + static UI host.

Streaming is server-sent events rather than a websocket: the traffic is
one-directional during a turn, SSE survives proxies and CDNs that mangle
websockets, and it degrades to a plain HTTP response if anything in the path
does not understand it.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Generator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent.loop import Agent, Session
from app.agent.tools import ToolRuntime, tool_catalogue
from app.config import ANTHROPIC_API_KEY, MODEL, fmt
from app.core.db import get_gateway
from app.core.principal import (
    DEMO_USERS, AccessDenied, Perm, resolve_principal,
)
from app.core.proposals import ProposalError, get_store
from app.engine.signals import detect_signals, summarise
from app.knowledge.retrieval import get_index
from app.knowledge.rules import get_rules

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="ParcelPilot Support Intelligence", version="1.0.0")

_gateway = get_gateway()
_rules = get_rules()          # raises on rule/document drift - fail fast, loudly
_index = get_index()
_store = get_store()
_runtime = ToolRuntime(_gateway, _rules, _index, _store)
_agent: Agent | None = None
_sessions: dict[str, Session] = {}


def agent() -> Agent:
    global _agent
    if _agent is None:
        if not ANTHROPIC_API_KEY:
            raise HTTPException(
                status_code=503,
                detail=("ANTHROPIC_API_KEY is not set. The deterministic layers (documents, "
                        "rules, decisions, signals) work without it - see /api/signals and "
                        "`make test` - but the chat agent needs a key."),
            )
        _agent = Agent(_runtime, api_key=ANTHROPIC_API_KEY)
    return _agent


# ---------------------------------------------------------------------------
class SessionRequest(BaseModel):
    user_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=4000)


class ConfirmRequest(BaseModel):
    session_id: str
    proposal_id: str
    decision: str = Field(pattern="^(confirm|cancel)$")


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str, ensure_ascii=False)}\n\n"


def _session(session_id: str) -> Session:
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Start a new chat.")
    return session


# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL,
        "agent_available": bool(ANTHROPIC_API_KEY),
        "snapshot": fmt(_gateway.clock.now()),
        "documents": len({c.doc_id for c in _index.clauses}),
        "clauses": len(_index.clauses),
        "rules_verified": not _rules.verify(),
    }


@app.get("/api/bootstrap")
def bootstrap() -> dict:
    """Everything the UI needs to render the identity switcher."""
    users = []
    for user_id, spec in DEMO_USERS.items():
        principal = resolve_principal(user_id)
        users.append({
            "user_id": user_id, "display_name": spec["display_name"],
            "role": principal.role, "role_label": principal.to_dict()["role_label"],
            "org": spec.get("org"), "account_id": principal.account_id,
            "context": principal.context,
            "tools": tool_catalogue(principal),
        })
    return {
        "users": users,
        "snapshot": fmt(_gateway.clock.now()),
        "model": MODEL,
        "agent_available": bool(ANTHROPIC_API_KEY),
        "documents": sorted({c.doc_id for c in _index.clauses}),
    }


@app.post("/api/session")
def create_session(req: SessionRequest) -> dict:
    try:
        principal = resolve_principal(req.user_id)
    except AccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    _sessions[session_id] = Session(session_id=session_id, principal=principal)
    return {
        "session_id": session_id,
        "principal": principal.to_dict(),
        "tools": tool_catalogue(principal),
        "snapshot": fmt(_gateway.clock.now()),
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    session = _session(req.session_id)
    agent_instance = agent()

    def stream() -> Generator[str, None, None]:
        try:
            for event in agent_instance.run(session, req.message):
                yield _sse(event)
        except Exception as exc:                              # noqa: BLE001
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/confirm")
def confirm(req: ConfirmRequest) -> StreamingResponse:
    """Execute or cancel a pending action, then let the agent acknowledge it.

    This is the only route into the write path. It re-resolves the principal
    from the session rather than trusting anything in the request body beyond
    the proposal id.
    """
    session = _session(req.session_id)
    principal = session.principal

    def stream() -> Generator[str, None, None]:
        try:
            if req.decision == "cancel":
                proposal = _store.cancel(req.proposal_id, principal, session.session_id)
                _gateway.audit(principal, session.session_id, proposal.action,
                               proposal.payload, "cancelled", proposal.title)
                yield _sse({"type": "action_result", "status": "cancelled",
                            "proposal_id": proposal.proposal_id, "title": proposal.title})
                follow_up = (f"[SYSTEM] The user declined the pending action "
                             f"'{proposal.title}'. Nothing was created or changed. "
                             "Acknowledge briefly and offer an alternative.")
            else:
                proposal = _store.confirm(req.proposal_id, principal, session.session_id)
                _gateway.audit(principal, session.session_id, proposal.action,
                               proposal.payload, "executed",
                               json.dumps(proposal.result, default=str))
                yield _sse({"type": "action_result", "status": "confirmed",
                            "proposal_id": proposal.proposal_id,
                            "title": proposal.title, "result": proposal.result})
                follow_up = (f"[SYSTEM] The user confirmed '{proposal.title}'. It has now been "
                             f"carried out. Result: {json.dumps(proposal.result, default=str)}. "
                             "Confirm this to the user in one or two sentences, including the "
                             "reference id and what happens next.")
        except (ProposalError, AccessDenied) as exc:
            yield _sse({"type": "error", "message": str(exc)})
            yield "data: [DONE]\n\n"
            return

        try:
            for event in agent().run(session, follow_up):
                yield _sse(event)
        except HTTPException as exc:
            yield _sse({"type": "error", "message": exc.detail})
        except Exception as exc:                              # noqa: BLE001
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/signals")
def signals(user_id: str, severity: str | None = None, type: str | None = None) -> dict:
    """Proactive operations view. Staff only, enforced by permission."""
    try:
        principal = resolve_principal(user_id)
        principal.require(Perm.VIEW_SIGNALS, "view operational signals")
    except AccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    found = detect_signals(_gateway.scan(principal, "accounts"),
                           _gateway.scan(principal, "orders"),
                           _gateway.scan(principal, "tickets"),
                           _rules, _gateway.clock.now())
    if severity:
        found = [s for s in found if s.severity == severity]
    if type:
        found = [s for s in found if s.type == type]
    return {"summary": summarise(found),
            "signals": [s.to_dict() for s in found],
            "snapshot": fmt(_gateway.clock.now())}


@app.get("/api/actions")
def actions(user_id: str) -> dict:
    try:
        principal = resolve_principal(user_id)
    except AccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return _gateway.list_actions(principal)


@app.get("/api/audit")
def audit(session_id: str | None = None, limit: int = 100) -> dict:
    """Every tool call, with the principal that made it and whether it was
    allowed. Exposed because an access-control story you cannot inspect is a
    claim, not a control."""
    return {"entries": _gateway.audit_trail(session_id, limit)}


@app.get("/api/documents")
def documents() -> dict:
    by_doc: dict[str, dict] = {}
    for clause in _index.clauses:
        entry = by_doc.setdefault(clause.doc_id, {
            "doc_id": clause.doc_id, "title": clause.doc_title,
            "doc_type": clause.doc_type, "status": clause.status,
            "authority_tier": clause.authority_tier,
            "effective_date": clause.effective_date,
            "account_scope": clause.account_scope, "clauses": 0,
        })
        entry["clauses"] += 1
    return {"documents": sorted(by_doc.values(), key=lambda d: d["doc_id"])}


# ---------------------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/dashboard")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
