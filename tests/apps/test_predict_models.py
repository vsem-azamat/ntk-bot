from datetime import datetime, timedelta

from apps.predictModels import drop_closed_hours, time_split


def test_drop_closed_hours_removes_zero_counts():
    rows = [
        (datetime(2024, 3, 1, 9, 0), 42),
        (datetime(2024, 3, 1, 9, 20), 0),  # closed -> dropped
        (datetime(2024, 3, 1, 10, 10), 7),
    ]
    assert drop_closed_hours(rows) == [
        (datetime(2024, 3, 1, 9, 0), 42),
        (datetime(2024, 3, 1, 10, 10), 7),
    ]


def test_time_split_has_no_leakage():
    base = datetime(2024, 1, 1, 9, 0)
    rows = [(base + timedelta(days=i), i) for i in range(60)]
    train, val = time_split(rows, holdout_weeks=4)

    assert train and val
    assert max(dt for dt, _ in train) < min(dt for dt, _ in val)
