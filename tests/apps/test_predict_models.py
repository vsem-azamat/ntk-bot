from datetime import datetime, timedelta

from apps.predictModels import time_split, usable_samples


def test_usable_samples_removes_zero_counts():
    rows = [
        (datetime(2024, 3, 1, 9, 0), 42),
        (datetime(2024, 3, 1, 9, 20), 0),  # closed -> dropped
        (datetime(2024, 3, 1, 10, 10), 7),
    ]
    assert usable_samples(rows) == [
        (datetime(2024, 3, 1, 9, 0), 42),
        (datetime(2024, 3, 1, 10, 10), 7),
    ]


def test_time_split_has_no_leakage():
    base = datetime(2024, 1, 1, 9, 0)
    rows = [(base + timedelta(days=i), i) for i in range(60)]
    train, val = time_split(rows, holdout_weeks=4)

    assert train and val
    assert max(dt for dt, _ in train) < min(dt for dt, _ in val)


def test_usable_samples_drops_pre_opening_residue():
    rows = [
        (datetime(2026, 10, 27, 1, 40), 205),  # night residue -> dropped
        (datetime(2026, 10, 27, 8, 20), 49),
        (datetime(2026, 10, 31, 9, 40), 12),  # Saturday before 10:00 -> dropped
    ]
    assert usable_samples(rows) == [(datetime(2026, 10, 27, 8, 20), 49)]
