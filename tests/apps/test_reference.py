from datetime import date, datetime, timedelta

import numpy as np

from apps.reference import (
    MAX_GAP_MIN,
    REFERENCE_K,
    DayCurves,
    bucket_of,
    resample_day,
)


def _day(day: date, level: int, step_min: int = 20) -> list[tuple[datetime, int]]:
    """A full synthetic open day from 08:00/10:00 to 23:40, flat at ``level``."""
    start = datetime.combine(day, datetime.min.time()).replace(hour=10 if day.weekday() >= 5 else 8)
    end = datetime.combine(day + timedelta(days=1), datetime.min.time())
    out = []
    t = start
    while t < end:
        out.append((t, level))
        t += timedelta(minutes=step_min)
    return out


def test_resample_interpolates_and_blanks_outside_and_long_gaps():
    d = datetime(2024, 3, 4)
    samples = [
        (d.replace(hour=10), 100),
        (d.replace(hour=10, minute=20), 200),
        (d.replace(hour=10, minute=20) + timedelta(minutes=MAX_GAP_MIN + 20), 200),
    ]
    curve = resample_day(samples)
    assert curve[bucket_of(d.replace(hour=10, minute=10))] == 150
    assert np.isnan(curve[bucket_of(d.replace(hour=9, minute=50))])
    # Inside the over-long gap: unknown, not a straight line.
    assert np.isnan(curve[bucket_of(d.replace(hour=10, minute=50))])


def test_cadence_does_not_change_the_curve():
    a = DayCurves(_day(date(2024, 3, 4), 300, step_min=10)).curves[0]
    b = DayCurves(_day(date(2024, 3, 4), 300, step_min=20)).curves[0]
    ok = ~np.isnan(a) & ~np.isnan(b)
    assert np.allclose(a[ok], b[ok])


def test_sparse_day_is_not_a_reference():
    rows = _day(date(2024, 3, 4), 300)[:5]
    assert DayCurves(rows).days == []


def test_norm_uses_last_k_same_type_days_only_before_target():
    mondays = [date(2024, 1, 8) + timedelta(weeks=i) for i in range(REFERENCE_K + 2)]
    rows = [r for i, d in enumerate(mondays) for r in _day(d, 100 * (i + 1))]
    rows += _day(date(2024, 3, 9), 9999)  # a Saturday: other type
    curves = DayCurves(rows)
    target = mondays[-1]
    norm = curves.norm(target)
    noon = bucket_of(datetime(2024, 1, 1, 12, 0))
    used = [100 * (i + 1) for i in range(len(mondays) - 1)][-REFERENCE_K:]
    assert norm.n_days == REFERENCE_K
    assert norm.p50[noon] == np.median(used)  # the target day itself never counts


def test_norm_falls_back_to_old_days_after_a_long_gap():
    rows = _day(date(2023, 3, 6), 400)
    norm = DayCurves(rows).norm(date(2024, 3, 4))
    assert norm.n_days == 1
    assert norm.p50.max() == 400


def test_empty_history_has_an_empty_norm():
    norm = DayCurves([]).norm(date(2024, 3, 4))
    assert norm.empty
    assert not norm.p50.any()


def test_recent_ratio_compares_latest_days_with_their_norm():
    mondays = [date(2024, 3, 4) + timedelta(weeks=i) for i in range(REFERENCE_K + 1)]
    rows = [r for d in mondays[:-1] for r in _day(d, 100)] + _day(mondays[-1], 150)
    curves = DayCurves(rows)
    assert curves.recent_ratio(mondays[-1] + timedelta(days=1), 1) == 1.5
    assert DayCurves([]).recent_ratio(date(2024, 3, 4), 2) == 1.0
