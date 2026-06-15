"""CLI: run the backtest for every registered candidate over the prod snapshot
and write a comparison report to tmp/experiments/reports/.

Usage:
    DATA_DIR=tmp/experiments BOT_TOKEN=ci-token \\
        uv run python -m scripts.experiments.run --folds 30
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from apps.features import ObsContext
from apps.predictModels import drop_closed_hours
from bot import db
from scripts.experiments.backtest import aggregate, score_day
from scripts.experiments.candidates import Candidate
from scripts.experiments.dataset import walk_forward_days

REGISTRY = [
    Candidate(name="base_legacy", groups=("base",)),
    Candidate(name="full", groups=("base", "regime", "asof")),
    Candidate(name="full_log", groups=("base", "regime", "asof"), log_target=True),
    Candidate(name="full_wide", groups=("base", "regime", "asof"), lo_alpha=0.05, hi_alpha=0.95),
    Candidate(name="full_wider", groups=("base", "regime", "asof"), lo_alpha=0.02, hi_alpha=0.98),
]


def _by_cut(rows: list[dict]) -> dict:
    """Aggregate metrics separately for each cut label (morning/10:00/...)."""
    cuts = sorted({r["cut"] for r in rows})
    return {cut: aggregate([r for r in rows if r["cut"] == cut]) for cut in cuts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=30)
    ap.add_argument("--min-train", type=int, default=120)
    ap.add_argument("--max-train-days", type=int, default=None)
    args = ap.parse_args()

    rows = drop_closed_hours(db.fetch_occupancy())
    ctx = ObsContext.from_rows(rows)
    weather = db.fetch_weather(rows[0][0], rows[-1][0])
    days = sorted({dt.date() for dt, _ in rows})
    folds = walk_forward_days(
        days, n_folds=args.folds, min_train=args.min_train, max_train_days=args.max_train_days
    )
    print(f"{len(folds)} folds; test days {folds[0][1]} .. {folds[-1][1]}")

    report = {}
    for cand in REGISTRY:
        all_rows: list[dict] = []
        for train_days, test_day in folds:
            all_rows += score_day(cand, rows, ctx, train_days, test_day, weather)
        report[cand.name] = {"overall": aggregate(all_rows), "by_cut": _by_cut(all_rows)}
        print(cand.name, report[cand.name]["overall"])

    out = Path("tmp/experiments/reports")
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / f"report-{stamp}.json").write_text(json.dumps(report, indent=2))
    print("wrote", out / f"report-{stamp}.json")


if __name__ == "__main__":
    main()
