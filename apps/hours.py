"""Library opening hours as seen in the occupancy data.

The NTK counter resets to ~0 when the library opens (08:00 on weekdays, 10:00 on
weekends and Czech public holidays). Samples taken between midnight and opening
are the previous evening's turnstile residue, not people in the building, so the
model, the baseline and the plots only use samples from opening time onwards.
"""

import functools
from datetime import date, datetime, time, timedelta

import holidays

WEEKDAY_OPEN = time(8, 0)
WEEKEND_OPEN = time(10, 0)


@functools.lru_cache(maxsize=8)
def cz_holidays(year: int) -> holidays.HolidayBase:
    return holidays.CZ(years=year)


def is_holiday(day: date) -> bool:
    return day in cz_holidays(day.year)


def opening_time(day: date) -> time:
    """Opening time for ``day``: weekend hours on Saturdays, Sundays and holidays."""
    return WEEKEND_OPEN if day.weekday() >= 5 or is_holiday(day) else WEEKDAY_OPEN


def opens_at(day: date) -> datetime:
    return datetime.combine(day, opening_time(day))


def day_end(day: date) -> datetime:
    """Midnight that closes ``day`` (exclusive upper bound of its open window)."""
    return datetime.combine(day + timedelta(days=1), time(0, 0))


def is_open_sample(dt: datetime) -> bool:
    """True when ``dt`` falls inside its own day's open window."""
    return dt.time() >= opening_time(dt.date())


def day_type(day: date) -> str:
    """Coarse day category whose occupancy curves share a shape."""
    if day.weekday() == 6 or is_holiday(day):
        return "sun_holiday"
    if day.weekday() == 5:
        return "sat"
    if day.weekday() == 4:
        return "fri"
    return "mon_thu"
