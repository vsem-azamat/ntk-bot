from datetime import datetime

from apps.features import ObsContext
from scripts.experiments.candidates import Candidate, fit_predict_day


def _synthetic_rows():
    # 30 days, deterministic bell-shaped day curve so a model can learn something.
    rows = []
    for day in range(1, 31):
        for step in range(0, 18 * 6):  # 10-min grid over 18h from 08:00
            minute = 8 * 60 + step * 10
            hour, mn = divmod(minute, 60)
            if hour >= 24:
                continue
            val = 300 - abs(minute - 14 * 60) // 2  # peak near 14:00
            rows.append((datetime(2025, 3, day, hour, mn), max(10, int(val))))
    return rows


def test_fit_predict_returns_monotone_band_on_grid():
    rows = _synthetic_rows()
    ctx = ObsContext.from_rows(rows)
    test_day = datetime(2025, 3, 30).date()
    train_days = [datetime(2025, 3, d).date() for d in range(1, 30)]
    cand = Candidate(name="base", groups=("base",), log_target=False)
    grid, p10, p50, p90 = fit_predict_day(
        cand, rows, ctx, train_days, test_day, cut=None, weather={}
    )
    assert len(grid) == len(p50) > 0
    for a, b, c in zip(p10, p50, p90, strict=True):
        assert a <= b <= c
