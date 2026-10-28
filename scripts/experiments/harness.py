"""Walk-forward backtest shared by every forecaster.

A fold is one calendar week. Each forecaster is fitted on samples strictly
before the week's Monday and then forecasts every open day of the week at
several cut times, seeing only that day's samples up to the cut (and every
earlier day in full, exactly as the live bot does). Scores are computed on the
real samples after the cut.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Protocol

import numpy as np

from apps.forecaster import History
from apps.hours import opens_at
from apps.reference import MIN_DAY_PEAK
from scripts.experiments.metrics import coverage, pinball_loss

# Fold weeks (Mondays) chosen to cover every regime: winter and summer exam
# periods, semester weeks, the summer break, the Christmas run-up, semester
# start, and the latest complete weeks.
FOLD_WEEKS: tuple[date, ...] = (
    date(2024, 1, 15),
    date(2024, 3, 11),
    date(2024, 5, 27),
    date(2024, 8, 5),
    date(2024, 10, 21),
    date(2024, 12, 16),
    date(2025, 1, 20),
    date(2025, 4, 7),
    date(2025, 6, 9),
    date(2025, 7, 28),
    date(2025, 9, 22),
    date(2025, 11, 24),
    date(2026, 1, 12),
    date(2026, 5, 18),
    date(2026, 8, 17),
    date(2026, 10, 12),
    date(2026, 10, 19),
)
# 06:45 is before opening on every day: the cold forecast of the morning digest.
EVAL_CUTS: tuple[time, ...] = (time(6, 45), time(10, 0), time(12, 0), time(14, 0))
# Extra cuts between the hourly training cuts and late in the day, reported
# separately: the bot is asked at any minute, not only at the headline cuts.
EXTRA_CUTS: tuple[time, ...] = (time(13, 20), time(16, 40), time(19, 0))
NEAR_HOURS = 3
BAND_LO, BAND_HI = 0.1, 0.9

Rows = list[tuple[datetime, int]]


@dataclass
class Data:
    """All usable samples. Forecasters get the full history: every query in
    ``apps`` reads only days before the target day plus the target day's samples
    up to the cut (``History.seen``); ``tests/apps/test_forecaster.py`` checks
    that perturbing anything later leaves the forecast unchanged."""

    rows: Rows
    history: History = field(init=False)

    def __post_init__(self) -> None:
        self.history = History(self.rows)

    @property
    def by_day(self) -> dict[date, Rows]:
        return self.history.by_day


@dataclass(frozen=True)
class Band:
    """Forecast on the served grid."""

    grid: list[datetime]
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray


class Forecaster(Protocol):
    name: str

    def fit(self, data: Data, train_end: date) -> None:
        """Train on samples of days strictly before ``train_end``."""
        ...

    def predict(self, data: Data, day: date, cut: datetime) -> Band:
        """Forecast ``day`` as of ``cut``: its samples up to ``cut`` and
        every earlier day."""
        ...


def _minutes(ts: list[datetime]) -> np.ndarray:
    return np.array([t.hour * 60 + t.minute for t in ts], float)


def score(band: Band, targets: Rows, start: datetime) -> dict[str, float]:
    """Metrics of ``band`` against the real samples ``targets`` (all after the
    cut). ``start`` is where the near-term window begins."""
    tx = _minutes([dt for dt, _ in targets])
    gx = _minutes(band.grid)
    y = np.array([c for _, c in targets], float)
    p10, p50, p90 = (np.interp(tx, gx, s) for s in (band.p10, band.p50, band.p90))
    err = np.abs(p50 - y)
    near = np.array([dt <= start + timedelta(hours=NEAR_HOURS) for dt, _ in targets])
    ip, ia = int(np.argmax(p50)), int(np.argmax(y))
    return {
        "mae": float(err.mean()),
        "near_mae": float(err[near].mean()) if near.any() else float("nan"),
        "peak_err": abs(float(p50[ip]) - float(y[ia])),
        "peak_min": abs(float(tx[ip] - tx[ia])),
        "pinball": (
            pinball_loss(list(y), list(p10), BAND_LO) + pinball_loss(list(y), list(p90), BAND_HI)
        )
        / 2,
        "coverage": coverage(list(p10), list(p90), list(y)),
    }


def fold_days(data: Data, week: date) -> list[date]:
    """Days of the fold week the library was open: any samples, and a peak of
    at least ``MIN_DAY_PEAK`` (closures and a stuck counter read 0-12 all day;
    nothing can forecast those and they would only add noise)."""
    days = [week + timedelta(days=i) for i in range(7)]
    return [
        d for d in days if data.by_day.get(d) and max(c for _, c in data.by_day[d]) >= MIN_DAY_PEAK
    ]


def evaluate(
    forecaster: Forecaster, data: Data, weeks: tuple[date, ...] = FOLD_WEEKS
) -> list[dict]:
    """One result row per (day, cut)."""
    out: list[dict] = []
    for week in weeks:
        forecaster.fit(data, week)
        for day in fold_days(data, week):
            samples = data.by_day[day]
            for ct in EVAL_CUTS + EXTRA_CUTS:
                cut = datetime.combine(day, ct)
                targets = [(dt, c) for dt, c in samples if dt > cut]
                if not targets:
                    continue
                band = forecaster.predict(data, day, cut)
                start = max(cut, opens_at(day))
                labels = {
                    "week": week.isoformat(),
                    "day": day.isoformat(),
                    "cut": ct.strftime("%H:%M"),
                }
                out.append({**score(band, targets, start), **labels})
    return out


METRICS = ("mae", "near_mae", "peak_err", "peak_min", "pinball", "coverage")


def summarise(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Mean of every metric per cut (each (day, cut) row weighs the same),
    plus ``all`` over the headline ``EVAL_CUTS`` and ``extra`` over the rest."""
    headline = {c.strftime("%H:%M") for c in EVAL_CUTS}
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["cut"]].append(r)
        groups["all" if r["cut"] in headline else "extra"].append(r)
    return {
        cut: {m: float(np.nanmean([r[m] for r in rs])) for m in METRICS}
        for cut, rs in sorted(groups.items())
    }
