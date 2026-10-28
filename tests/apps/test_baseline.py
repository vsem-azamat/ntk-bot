from datetime import date, datetime, timedelta

from apps.baseline import build_climatology, climatology_predict


def _day(day: date, level: int) -> list[tuple[datetime, int]]:
    """A synthetic open day: 30 samples every 20 min from 10:00, flat at ``level``."""
    start = datetime.combine(day, datetime.min.time()).replace(hour=10)
    return [(start + timedelta(minutes=20 * i), level) for i in range(30)]


def test_uses_recent_same_type_days_median():
    # Fridays 2024-03-01/08/15 at 100/200/300, plus a Monday that must be ignored.
    rows = _day(date(2024, 3, 1), 100) + _day(date(2024, 3, 8), 200) + _day(date(2024, 3, 15), 300)
    rows += _day(date(2024, 3, 18), 5000)
    clim = build_climatology(rows, date(2024, 3, 22))
    p10, p50, p90 = climatology_predict(clim, [datetime(2024, 3, 22, 12, 0)])
    assert p50[0] == 200
    assert p10[0] <= p50[0] <= p90[0]


def test_only_last_k_days_count():
    days = [date(2024, 2, 2) + timedelta(weeks=i) for i in range(4)]  # four Fridays
    rows = [r for i, d in enumerate(days) for r in _day(d, 1000 if i == 0 else 100)]
    clim = build_climatology(rows, date(2024, 3, 1), k=3)
    _, p50, _ = climatology_predict(clim, [datetime(2024, 3, 1, 12, 0)])
    assert p50[0] == 100


def test_never_looks_at_target_day_or_later():
    rows = _day(date(2024, 3, 1), 100) + _day(date(2024, 3, 8), 999)
    clim = build_climatology(rows, date(2024, 3, 8))
    _, p50, _ = climatology_predict(clim, [datetime(2024, 3, 8, 12, 0)])
    assert p50[0] == 100


def test_interpolates_between_sample_times():
    day = date(2024, 3, 1)
    rows = [(datetime(2024, 3, 1, 10, 0), 100), (datetime(2024, 3, 1, 10, 20), 200)]
    rows += [(datetime(2024, 3, 1, 10, 40 + i), 200) for i in range(20)]  # make it a usable day
    clim = build_climatology(rows, day + timedelta(weeks=1))
    _, p50, _ = climatology_predict(clim, [datetime(2024, 3, 8, 10, 10)])
    assert p50[0] == 150


def test_empty_history_predicts_zeros():
    clim = build_climatology([], date(2024, 3, 23))
    p10, p50, p90 = climatology_predict(clim, [datetime(2024, 3, 23, 14, 0)])
    assert (p10[0], p50[0], p90[0]) == (0.0, 0.0, 0.0)


def test_holiday_uses_sunday_shape_not_weekday():
    # Good Friday 2024-03-29 is a Czech holiday: it borrows Sundays, not Fridays.
    rows = _day(date(2024, 3, 22), 900) + _day(date(2024, 3, 24), 150)
    clim = build_climatology(rows, date(2024, 3, 29))
    _, p50, _ = climatology_predict(clim, [datetime(2024, 3, 29, 12, 0)])
    assert p50[0] == 150
