"""Who is asking, and what they are allowed to do.

Access control is expressed as a principal + a permission set, and it is
enforced in the data layer (see core/db.py) - never by asking the model nicely.
The system prompt tells the agent what it *should* do; the gateway decides what
it *can* do. If the two ever disagree, the gateway wins.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Perm(str, Enum):
    READ_OWN_ACCOUNT = "read_own_account"
    READ_ALL_ACCOUNTS = "read_all_accounts"
    READ_INTERNAL_FIELDS = "read_internal_fields"   # CSM notes, assignee, historical resolutions
    VIEW_SIGNALS = "view_signals"                   # proactive ops dashboard
    CREATE_ESCALATION = "create_escalation"
    UPDATE_TICKET = "update_ticket"
    CREATE_TASK = "create_task"
    APPROVE_CREDIT = "approve_credit"               # credits above the SOP threshold


ROLE_PERMISSIONS: dict[str, set[Perm]] = {
    # Customer-facing chatbot: a customer can see their own account and raise an
    # escalation on it. Nothing else, ever.
    "customer": {Perm.READ_OWN_ACCOUNT, Perm.CREATE_ESCALATION},
    # Internal chatbot, tier-1 support.
    "support_agent": {
        Perm.READ_ALL_ACCOUNTS, Perm.READ_INTERNAL_FIELDS, Perm.VIEW_SIGNALS,
        Perm.CREATE_ESCALATION, Perm.UPDATE_TICKET, Perm.CREATE_TASK,
    },
    # Internal chatbot, support manager: same plus credit approval authority.
    "support_manager": {
        Perm.READ_ALL_ACCOUNTS, Perm.READ_INTERNAL_FIELDS, Perm.VIEW_SIGNALS,
        Perm.CREATE_ESCALATION, Perm.UPDATE_TICKET, Perm.CREATE_TASK,
        Perm.APPROVE_CREDIT,
    },
}

ROLE_LABELS = {
    "customer": "Customer",
    "support_agent": "Support agent",
    "support_manager": "Support manager",
}


class AccessDenied(PermissionError):
    """Raised when a principal reaches for something outside its scope.

    Surfaced to the agent as a tool error so it can explain the boundary to the
    user, and written to the audit log either way.
    """


@dataclass(frozen=True)
class Principal:
    user_id: str
    display_name: str
    role: str
    account_id: str | None = None          # set for customers; None for staff
    email: str | None = None
    permissions: frozenset[Perm] = field(default_factory=frozenset)

    @classmethod
    def build(cls, user_id: str, display_name: str, role: str,
              account_id: str | None = None, email: str | None = None) -> "Principal":
        if role not in ROLE_PERMISSIONS:
            raise ValueError(f"Unknown role {role!r}")
        if role == "customer" and not account_id:
            raise ValueError("A customer principal must be bound to an account")
        return cls(
            user_id=user_id, display_name=display_name, role=role,
            account_id=account_id if role == "customer" else None,
            email=email, permissions=frozenset(ROLE_PERMISSIONS[role]),
        )

    # -- checks -------------------------------------------------------------
    @property
    def is_staff(self) -> bool:
        return self.role != "customer"

    @property
    def context(self) -> str:
        return "internal" if self.is_staff else "customer"

    def has(self, perm: Perm) -> bool:
        return perm in self.permissions

    def require(self, perm: Perm, what: str = "") -> None:
        if not self.has(perm):
            raise AccessDenied(
                f"{ROLE_LABELS.get(self.role, self.role)} is not permitted to "
                f"{what or perm.value.replace('_', ' ')}."
            )

    def visible_accounts(self, all_accounts: list[str]) -> list[str]:
        if self.has(Perm.READ_ALL_ACCOUNTS):
            return list(all_accounts)
        return [self.account_id] if self.account_id else []

    def assert_can_read_account(self, account_id: str | None) -> None:
        if account_id is None:
            return
        if self.has(Perm.READ_ALL_ACCOUNTS):
            return
        if self.account_id and account_id == self.account_id:
            return
        # Deliberately does not confirm whether the record exists - an error that
        # distinguishes "not yours" from "not found" is an enumeration oracle.
        raise AccessDenied("That record does not belong to your account.")

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id, "display_name": self.display_name,
            "role": self.role, "role_label": ROLE_LABELS.get(self.role, self.role),
            "account_id": self.account_id, "context": self.context,
            "permissions": sorted(p.value for p in self.permissions),
        }


# ---------------------------------------------------------------------------
# Mocked identity directory
# ---------------------------------------------------------------------------
# The brief allows mocked auth. In production this is an OIDC/JWT claim set;
# the shape is what matters - a principal resolved *server-side* from a session,
# never a client-supplied account id that the UI could tamper with.
DEMO_USERS: dict[str, dict] = {
    "cust-northstar": {"display_name": "Ravi Menon", "role": "customer", "account_id": "ACCT-001", "email": "ravi@northstar.example", "org": "Northstar Logistics"},
    "cust-lumenworks": {"display_name": "Sara Iyer", "role": "customer", "account_id": "ACCT-002", "email": "sara@lumenworks.example", "org": "LumenWorks"},
    "cust-beacon": {"display_name": "Dev Sharma", "role": "customer", "account_id": "ACCT-003", "email": "dev@beaconretail.example", "org": "Beacon Retail"},
    "cust-axis": {"display_name": "Ana Fernandes", "role": "customer", "account_id": "ACCT-004", "email": "ana@axislabs.example", "org": "Axis Labs"},
    "staff-rohit": {"display_name": "Rohit", "role": "support_agent", "email": "rohit@parcelpilot.example", "org": "ParcelPilot Support"},
    "staff-maya": {"display_name": "Maya", "role": "support_agent", "email": "maya@parcelpilot.example", "org": "ParcelPilot Support"},
    "staff-priya": {"display_name": "Priya Mehta", "role": "support_manager", "email": "priya@parcelpilot.example", "org": "ParcelPilot Support"},
}


def resolve_principal(user_id: str) -> Principal:
    spec = DEMO_USERS.get(user_id)
    if not spec:
        raise AccessDenied("Unknown user.")
    return Principal.build(
        user_id=user_id, display_name=spec["display_name"], role=spec["role"],
        account_id=spec.get("account_id"), email=spec.get("email"),
    )


__all__ = ["Perm", "Principal", "AccessDenied", "resolve_principal", "DEMO_USERS", "ROLE_LABELS", "ROLE_PERMISSIONS"]
