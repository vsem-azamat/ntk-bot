"""Backtest data shaping: walk-forward day folds and intraday cut times.

A 'fold' is (train_days, test_day): the model is trained on all samples from
train_days and scored on test_day at several as-of cut times. Folds always keep
train strictly before test (no leakage across the time boundary).
"""

from datetime import date, datetime, time

# Intraday moments we evaluate at. ``None`` is the cold (pre-open) start.
_INTRADAY_CUTS = (time(10, 0), time(12, 0), time(14, 0))


def walk_forward_days(
    days: list[date],
    n_folds: int,
    min_train: int,
    max_train_days: int | None = None,
) -> list[tuple[list[date], date]]:
    """Return up to ``n_folds`` (train_days, test_day) folds.

    Test days are spread EVENLY across the eligible range (every day with at least
    ``min_train`` earlier days), so the backtest covers multiple seasons / exam
    periods rather than only the most recent weeks. Train is every day strictly
    before the test day, optionally capped to the most recent ``max_train_days``
    (None = use all history) to keep training time bounded."""
    days = sorted(days)
    eligible = days[min_train:]
    if not eligible:
        return []
    n = min(n_folds, len(eligible))
    if n == 1:
        idxs = [len(eligible) - 1]
    else:
        idxs = sorted({round(i * (len(eligible) - 1) / (n - 1)) for i in range(n)})
    folds: list[tuple[list[date], date]] = []
    for j in idxs:
        test_day = eligible[j]
        train = [d for d in days if d < test_day]
        if max_train_days is not None:
            train = train[-max_train_days:]
        folds.append((train, test_day))
    return folds


def cut_times(day: date) -> list[datetime | None]:
    """Cut times for one test day: cold start plus the intraday moments."""
    return [None] + [datetime.combine(day, t) for t in _INTRADAY_CUTS]
