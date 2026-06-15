from datetime import datetime

from apps.features import ObsContext
from scripts.experiments.backtest import score_day
from scripts.experiments.candidates import Candidate


def _rows():
    rows = []
    for day in range(1, 16):
        for step in range(0, 18 * 6):
            minute = 8 * 60 + step * 10
            hour, mn = divmod(minute, 60)
            if hour >= 24:
                continue
            val = 300 - abs(minute - 14 * 60) // 2
            rows.append((datetime(2025, 3, day, hour, mn), max(10, int(val))))
    return rows


def test_score_day_returns_all_metric_keys():
    rows = _rows()
    ctx = ObsContext.from_rows(rows)
    cand = Candidate(name="base", groups=("base",))
    train_days = [datetime(2025, 3, d).date() for d in range(1, 15)]
    result = score_day(cand, rows, ctx, train_days, datetime(2025, 3, 15).date(), {})
    assert result
    for r in result:
        assert {"mae", "peak_value_err", "peak_time_err", "pinball", "coverage", "near_mae"} <= set(
            r
        )
