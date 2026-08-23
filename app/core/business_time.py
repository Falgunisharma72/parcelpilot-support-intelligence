"""Business-hours arithmetic for SLA targets.

The support policies express first-response targets in three different units
("30 minutes, 24x7", "2 business hours", "1 business day"). Getting an SLA
answer right therefore requires real calendar arithmetic, not a rough
multiplication. This module keeps that arithmetic in one deterministic,
unit-tested place so the language model never has to do it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta

from app.config import (
    BUSINESS_DAY_END_HOUR,
    BUSINESS_DAY_START_HOUR,
    BUSINESS_WEEKDAYS,
)

_DAY_MINUTES = (BUSINESS_DAY_END_HOUR - BUSINESS_DAY_START_HOUR) * 60


@dataclass(frozen=True)
class Duration:
    """A support-target duration, aware of whether it runs on a 24x7 clock."""

    value: float
    unit: str  # minutes | hours | days
    clock: str  # calendar (24x7) | business

    @classmethod
    def parse(cls, spec: dict) -> "Duration":
        return cls(float(spec["value"]), spec["unit"], spec.get("clock", "calendar"))

    def label(self) -> str:
        qualifier = "business " if self.clock == "business" else ""
        value = int(self.value) if float(self.value).is_integer() else self.value
        unit = self.unit if value != 1 else self.unit.rstrip("s")
        if self.clock == "calendar" and self.unit in ("minutes", "hours"):
            return f"{value} {unit}, 24x7"
        return f"{value} {qualifier}{unit}"


def _is_business_day(d: datetime) -> bool:
    return d.weekday() in BUSINESS_WEEKDAYS


def _day_open(d: datetime) -> datetime:
    return d.replace(hour=BUSINESS_DAY_START_HOUR, minute=0, second=0, microsecond=0)


def _day_close(d: datetime) -> datetime:
    return d.replace(hour=BUSINESS_DAY_END_HOUR, minute=0, second=0, microsecond=0)


def _clamp_forward(dt: datetime) -> datetime:
    """Move a moment forward to the next instant inside business hours."""
    cur = dt
    for _ in range(30):  # a fortnight of closures is more than enough
        if not _is_business_day(cur):
            cur = _day_open(cur + timedelta(days=1))
            continue
        if cur < _day_open(cur):
            return _day_open(cur)
        if cur >= _day_close(cur):
            cur = _day_open(cur + timedelta(days=1))
            continue
        return cur
    raise RuntimeError("could not find a business instant within 30 days")


def business_minutes_between(start: datetime, end: datetime) -> float:
    """Minutes of business time elapsed between two moments (never negative)."""
    if end <= start:
        return 0.0
    cur = _clamp_forward(start)
    total = 0.0
    for _ in range(400):
        if cur >= end:
            break
        close = _day_close(cur)
        segment_end = min(close, end)
        if segment_end > cur:
            total += (segment_end - cur).total_seconds() / 60
        cur = _clamp_forward(close)
    return total


def add_business_minutes(start: datetime, minutes: float) -> datetime:
    """Advance a moment by N minutes of business time."""
    cur = _clamp_forward(start)
    remaining = float(minutes)
    for _ in range(400):
        close = _day_close(cur)
        available = (close - cur).total_seconds() / 60
        if remaining <= available:
            return cur + timedelta(minutes=remaining)
        remaining -= available
        cur = _clamp_forward(close)
    raise RuntimeError("business-minute addition did not converge")


def _to_minutes(duration: Duration) -> float:
    if duration.unit == "minutes":
        return duration.value
    if duration.unit == "hours":
        return duration.value * 60
    if duration.unit == "days":
        return duration.value * (_DAY_MINUTES if duration.clock == "business" else 24 * 60)
    raise ValueError(f"unknown duration unit {duration.unit!r}")


def deadline_from(start: datetime, duration: Duration) -> datetime:
    """Compute the moment a target is due, honouring the duration's clock."""
    minutes = _to_minutes(duration)
    if duration.clock == "business":
        return add_business_minutes(start, minutes)
    return start + timedelta(minutes=minutes)


def elapsed_against(start: datetime, until: datetime, duration: Duration) -> float:
    """Elapsed time between two moments, measured on the duration's own clock."""
    if duration.clock == "business":
        return business_minutes_between(start, until)
    return max(0.0, (until - start).total_seconds() / 60)


def budget_minutes(duration: Duration) -> float:
    return _to_minutes(duration)


__all__ = [
    "Duration", "business_minutes_between", "add_business_minutes",
    "deadline_from", "elapsed_against", "budget_minutes",
]
