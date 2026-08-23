"""Central configuration.

Everything time-related in this system is anchored to the dataset snapshot
declared in the workbook's README sheet, not to wall-clock `now()`. That is a
deliberate choice: the assessment data is a frozen snapshot, and an agent that
silently drifts to real time would produce SLA and cancellation answers that
change every day for the same input. `Clock.now()` is the single source of
truth and can be overridden per-request for what-if analysis.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
WORKBOOK = DATA_DIR / "ParcelPilot_Assessment_Data.xlsx"
RULES_FILE = ROOT / "app" / "knowledge" / "rules.yaml"
BUILD_DIR = Path(os.getenv("PARCELPILOT_BUILD_DIR", ROOT / ".build"))

TZ: tzinfo = ZoneInfo("Asia/Kolkata")

# Declared in ParcelPilot_Assessment_Data.xlsx -> README -> "Dataset snapshot".
# Parsed from the workbook at startup; this is the fallback / documented value.
DEFAULT_SNAPSHOT = datetime(2026, 8, 16, 11, 0, tzinfo=TZ)

# ---------------------------------------------------------------------------
# Business-hours assumption
# ---------------------------------------------------------------------------
# The supplied documents use "business hours" and "business days" without ever
# defining them. Rather than let the model improvise a definition per answer
# (which is exactly how you get two different SLA answers for one ticket), we
# pin one definition here, apply it deterministically, and surface it in every
# SLA tool result so a human reviewer can see the assumption we used.
BUSINESS_DAY_START_HOUR = 9
BUSINESS_DAY_END_HOUR = 18
BUSINESS_WEEKDAYS = {0, 1, 2, 3, 4}  # Mon-Fri
BUSINESS_HOURS_ASSUMPTION = (
    "Business hours are assumed to be 09:00-18:00 Asia/Kolkata, Monday-Friday. "
    "The supplied documents do not define them; this assumption is applied "
    "consistently and reported with every SLA calculation."
)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL = os.getenv("PARCELPILOT_MODEL", "claude-opus-5")
MAX_TOKENS = int(os.getenv("PARCELPILOT_MAX_TOKENS", "8000"))
EFFORT = os.getenv("PARCELPILOT_EFFORT", "high")
SHOW_THINKING = os.getenv("PARCELPILOT_SHOW_THINKING", "1") not in ("0", "false", "")
MAX_AGENT_STEPS = int(os.getenv("PARCELPILOT_MAX_AGENT_STEPS", "12"))
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Proposals (pending state-changing actions) expire; a stale confirmation is a
# confirmation against facts the user never actually saw.
PROPOSAL_TTL_SECONDS = int(os.getenv("PARCELPILOT_PROPOSAL_TTL", "900"))


@dataclass(frozen=True)
class Clock:
    """Frozen clock anchored on the dataset snapshot."""

    snapshot: datetime = DEFAULT_SNAPSHOT

    def now(self) -> datetime:
        return self.snapshot

    def iso(self) -> str:
        return self.snapshot.strftime("%Y-%m-%d %H:%M %Z")


def parse_ts(value: str | datetime | None) -> datetime | None:
    """Parse the workbook's naive 'YYYY-MM-DD HH:MM' timestamps into IST."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=TZ)
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=TZ)
        except ValueError:
            continue
    raise ValueError(f"Unparseable timestamp: {value!r}")


def fmt(dt: datetime | None) -> str | None:
    return None if dt is None else dt.strftime("%Y-%m-%d %H:%M")


def humanise_minutes(minutes: float) -> str:
    minutes = int(round(minutes))
    sign = "-" if minutes < 0 else ""
    minutes = abs(minutes)
    if minutes < 60:
        return f"{sign}{minutes} min"
    hours, rem = divmod(minutes, 60)
    if hours < 24:
        return f"{sign}{hours}h {rem:02d}m"
    days, hrs = divmod(hours, 24)
    return f"{sign}{days}d {hrs}h {rem:02d}m"


__all__ = [
    "ROOT", "DATA_DIR", "DOCS_DIR", "WORKBOOK", "RULES_FILE", "BUILD_DIR", "TZ",
    "DEFAULT_SNAPSHOT", "Clock", "parse_ts", "fmt", "humanise_minutes",
    "MODEL", "MAX_TOKENS", "EFFORT", "SHOW_THINKING", "MAX_AGENT_STEPS",
    "ANTHROPIC_API_KEY", "PROPOSAL_TTL_SECONDS", "BUSINESS_HOURS_ASSUMPTION",
    "BUSINESS_DAY_START_HOUR", "BUSINESS_DAY_END_HOUR", "BUSINESS_WEEKDAYS",
    "timedelta", "timezone",
]
