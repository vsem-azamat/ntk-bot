"""CLI: score forecasters on the walk-forward folds and save per-row results to
``tmp/experiments/results/<name>.json`` (see ``report.py`` for the table).

    BOT_TOKEN=ci-token uv run python -m scripts.experiments.run served catboost baseline

Samples come from the configured database, or with ``--sqlite PATH`` from a
SQLite snapshot with an ``occupancy(ts 'YYYY-MM-DD HH:MM', people)`` table.
"""

import argparse
import json
import os
import sqlite3
import time as clock
from contextlib import closing
from datetime import datetime
from pathlib import Path

import apps.forecaster as forecaster_module
from apps.forecaster import BASELINE, CATBOOST
from apps.hours import usable_samples
from bot import db
from scripts.experiments.forecasters import CatBoostOnly, Production
from scripts.experiments.harness import Data, Forecaster, evaluate, summarise

REGISTRY: dict[str, Forecaster] = {
    "served": Production(name="served"),
    "catboost": Production(name="catboost", force=CATBOOST),
    "baseline": Production(name="baseline", force=BASELINE),
    "cb_only": CatBoostOnly(name="cb_only"),
}
RESULTS = Path("tmp/experiments/results")
_SNAPSHOT_TS = "%Y-%m-%d %H:%M"
# Backtests share the machine with other work: cap CatBoost's threads.
forecaster_module.CATBOOST_PARAMS["thread_count"] = int(os.environ.get("EXPERIMENT_THREADS", "2"))
# Ablation knobs: EXPERIMENT_KNOBS='{"LEVEL_WEIGHT_POWER": 1.0}' overrides
# module constants of apps.forecaster; EXPERIMENT_TAG names the result file.
for _knob, _value in json.loads(os.environ.get("EXPERIMENT_KNOBS", "{}")).items():
    if not hasattr(forecaster_module, _knob):
        raise SystemExit(f"unknown knob {_knob}")
    setattr(forecaster_module, _knob, _value)
TAG = os.environ.get("EXPERIMENT_TAG", "")


def load(sqlite_path: str | None = None) -> Data:
    if sqlite_path is None:
        rows = db.fetch_occupancy()
    else:
        with closing(sqlite3.connect(sqlite_path)) as conn:
            raw = conn.execute("SELECT ts, people FROM occupancy ORDER BY ts").fetchall()
        rows = [(datetime.strptime(ts, _SNAPSHOT_TS), int(people)) for ts, people in raw]
    return Data(usable_samples(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", choices=sorted(REGISTRY))
    ap.add_argument("--sqlite", help="read samples from this SQLite snapshot")
    args = ap.parse_args()
    data = load(args.sqlite)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        t0 = clock.time()
        forecaster = REGISTRY[name]
        rows = evaluate(forecaster, data)
        name = f"{name}{TAG}"
        (RESULTS / f"{name}.json").write_text(json.dumps(rows))
        maes = {k: round(v["mae"], 1) for k, v in summarise(rows).items()}
        print(name, f"{clock.time() - t0:.0f}s", json.dumps(maes), flush=True)
        if isinstance(forecaster, Production):
            print("  gate:", forecaster.choices, flush=True)


if __name__ == "__main__":
    main()
