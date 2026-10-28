from datetime import date, datetime, timedelta

import numpy as np

from apps.level import RATIO_PRIOR_BUCKETS, day_state, shape_level
from apps.reference import DayCurves, bucket_of
from tests.apps.test_reference import _day

MONDAYS = [date(2024, 3, 4) + timedelta(weeks=i) for i in range(4)]
TARGET = MONDAYS[-1]


def _curves() -> DayCurves:
    return DayCurves([r for d in MONDAYS[:-1] for r in _day(d, 200)])


def test_cold_state_is_the_norm():
    state = day_state(_curves(), TARGET, _day(TARGET, 400), cut=None)
    assert state.last_bucket is None
    assert state.ratio == 1.0
    _, p50, _ = shape_level(state)
    assert p50[bucket_of(datetime(2024, 1, 1, 15, 0))] == 200


def test_ratio_uses_only_samples_up_to_the_cut_and_is_shrunk():
    today = _day(TARGET, 400)
    cut = datetime.combine(TARGET, datetime.min.time()).replace(hour=9)
    state = day_state(_curves(), TARGET, today, cut)
    assert state.last_count == 400
    assert state.last_bucket == bucket_of(cut)
    assert state.observed_ratio == 2.0
    n = state.observed_buckets
    assert state.ratio == (RATIO_PRIOR_BUCKETS + n * 2.0) / (RATIO_PRIOR_BUCKETS + n)
    # Nothing after the cut is visible.
    assert np.isnan(state.today[bucket_of(cut) + 1 :]).all()


def test_more_observation_moves_the_ratio_towards_today():
    today = _day(TARGET, 400)
    early = day_state(
        _curves(), TARGET, today, datetime.combine(TARGET, datetime.min.time()).replace(hour=9)
    )
    late = day_state(
        _curves(), TARGET, today, datetime.combine(TARGET, datetime.min.time()).replace(hour=15)
    )
    assert 1.0 < early.ratio < late.ratio < 2.0
