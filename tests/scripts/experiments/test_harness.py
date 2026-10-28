from datetime import date, datetime, timedelta

import numpy as np

from scripts.experiments.harness import Band, Data, evaluate, fold_days, score, summarise
from tests.apps.test_reference import _day


def _band(values: list[float]) -> Band:
    grid = [datetime(2024, 3, 4, 8, 0) + timedelta(minutes=10 * i) for i in range(len(values))]
    v = np.array(values, float)
    return Band(grid, v - 10, v, v + 10)


def test_score_metrics_on_known_errors():
    band = _band([100, 200, 300, 200])
    targets = [(datetime(2024, 3, 4, 8, 10), 210), (datetime(2024, 3, 4, 8, 20), 330)]
    m = score(band, targets, start=datetime(2024, 3, 4, 8, 0))
    assert m["mae"] == 20
    assert m["near_mae"] == 20
    assert m["peak_err"] == 30
    assert m["peak_min"] == 0
    assert m["coverage"] == 0.5


def test_fold_days_skip_closures():
    week = date(2024, 3, 4)
    rows = _day(week, 300) + [(datetime(2024, 3, 5, 12, 0), 3)]  # Tuesday: counter stuck
    assert fold_days(Data(rows), week) == [week]


class _Spy:
    """Records how the harness drives a forecaster."""

    name = "spy"

    def __init__(self) -> None:
        self.fits: list[date] = []
        self.cuts: list[datetime] = []

    def fit(self, data, train_end):
        self.fits.append(train_end)

    def predict(self, data, day, cut):
        self.cuts.append(cut)
        return _band([0.0] * 96)


def test_evaluate_fits_before_the_week_and_scores_after_the_cut():
    week = date(2024, 3, 4)
    spy = _Spy()
    rows = evaluate(spy, Data(_day(week, 300)), weeks=(week,))
    assert spy.fits == [week]
    assert rows and {r["day"] for r in rows} == {"2024-03-04"}
    assert all(c.date() == week for c in spy.cuts)
    s = summarise(rows)
    assert "all" in s and "06:45" in s
