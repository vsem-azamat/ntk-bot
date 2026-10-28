from datetime import date, datetime, time, timedelta

import numpy as np
import pytest

from apps.features import (
    feature_names,
    from_target,
    state_rows,
    to_target,
    training_rows,
)
from apps.level import day_state
from apps.reference import DayCurves, bucket_of
from tests.apps.test_reference import _day

GROUPS = ("calendar", "norm", "today")
MONDAYS = [date(2024, 3, 4) + timedelta(weeks=i) for i in range(4)]


def _history(last_level: int = 300):
    rows = [r for d in MONDAYS[:-1] for r in _day(d, 200)] + _day(MONDAYS[-1], last_level)
    by_day: dict[date, list] = {}
    for dt, c in rows:
        by_day.setdefault(dt.date(), []).append((dt, c))
    return rows, by_day


def test_target_roundtrip():
    y = np.array([0.0, 10.0, 500.0])
    base = np.array([5.0, 50.0, 400.0])
    assert np.allclose(from_target(to_target(y, base), base), y)
    assert to_target(np.array([400.0]), np.array([400.0]))[0] == 0.0


def test_unknown_group_is_rejected():
    with pytest.raises(ValueError):
        feature_names(("calendar", "nope"))


def test_rows_match_feature_names():
    rows, by_day = _history()
    curves = DayCurves(rows)
    state = day_state(curves, MONDAYS[-1], by_day[MONDAYS[-1]], None)
    X = state_rows(state, np.array([60, 80]), GROUPS)
    assert X.shape == (2, len(feature_names(GROUPS)))
    has_today = feature_names(GROUPS).index("has_today")
    assert (X[:, has_today] == 0).all()


def test_training_rows_only_target_samples_after_the_cut():
    rows, by_day = _history()
    curves = DayCurves(rows)
    ts = training_rows(curves, by_day, [MONDAYS[-1]], GROUPS, (time(12, 0),))
    X, z, base = ts.X, ts.z, ts.base
    after = [dt for dt, _ in by_day[MONDAYS[-1]] if dt.time() > time(12, 0)]
    assert len(z) == len(after)
    # Every target bucket lies after the last observed bucket.
    horizon = feature_names(GROUPS).index("horizon_min")
    assert (X[:, horizon] > 0).all()
    assert np.allclose(from_target(z, base), 300)


def test_training_rows_do_not_see_the_future():
    """Changing samples after the cut changes the labels, never the features."""
    rows, by_day = _history()
    curves = DayCurves(rows)
    X1 = training_rows(curves, by_day, [MONDAYS[-1]], GROUPS, (time(12, 0),)).X
    noon = datetime.combine(MONDAYS[-1], time(12, 0))
    rows2 = [(dt, c * 3 if dt > noon else c) for dt, c in rows]
    by_day2 = dict(by_day)
    by_day2[MONDAYS[-1]] = [(dt, c) for dt, c in rows2 if dt.date() == MONDAYS[-1]]
    X2 = training_rows(DayCurves(rows2), by_day2, [MONDAYS[-1]], GROUPS, (time(12, 0),)).X
    assert np.array_equal(X1, X2, equal_nan=True)


def test_norm_features_ignore_the_target_day():
    rows, _ = _history(last_level=5000)
    curves = DayCurves(rows)
    state = day_state(curves, MONDAYS[-1], [], None)
    norm = feature_names(GROUPS).index("norm")
    X = state_rows(state, np.array([bucket_of(datetime(2024, 1, 1, 12))]), GROUPS)
    assert X[0, norm] == 200
