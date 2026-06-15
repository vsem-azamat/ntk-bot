from datetime import datetime

from scripts.experiments.dataset import cut_times, walk_forward_days


def test_walk_forward_days_are_chronological_and_held_out():
    days = [datetime(2025, 3, d).date() for d in range(1, 21)]
    folds = walk_forward_days(days, n_folds=2, min_train=10)
    for train, test in folds:
        assert all(d < test for d in train)
        assert len(train) >= 10
    assert folds[0][1] < folds[-1][1]


def test_cut_times_includes_morning_and_intraday():
    day = datetime(2025, 3, 13).date()
    cuts = cut_times(day)
    assert cuts[0] is None  # morning / cold start
    assert datetime(2025, 3, 13, 12, 0) in cuts
