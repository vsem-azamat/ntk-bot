"""Backtest data shaping: walk-forward day folds and intraday cut times.

A 'fold' is (train_days, test_day): the model is trained on all samples from
train_days and scored on test_day at several as-of cut times. Folds always keep
train strictly before test (no leakage across the time boundary).
"""

from datetime import date, datetime, time

# Intraday moments we evaluate at. ``None`` is the cold (pre-open) start.
_INTRADAY_CUTS = (time(10, 0), time(12, 0), time(14, 0))


def walk_forward_days(
    days: list[date], n_folds: int, min_train: int
) -> list[tuple[list[date], date]]:
    """Return ``n_folds`` (train_days, test_day) folds from the tail of ``days``.

    Test days are the last ``n_folds`` days that still leave ``min_train`` earlier
    days to train on. Train is every day strictly before the test day."""
    days = sorted(days)
    folds: list[tuple[list[date], date]] = []
    candidates = [d for i, d in enumerate(days) if i >= min_train]
    for test_day in candidates[-n_folds:]:
        train = [d for d in days if d < test_day]
        folds.append((train, test_day))
    return folds


def cut_times(day: date) -> list[datetime | None]:
    """Cut times for one test day: cold start plus the intraday moments."""
    return [None] + [datetime.combine(day, t) for t in _INTRADAY_CUTS]
