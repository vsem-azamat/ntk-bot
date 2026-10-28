"""CLI: comparison table from saved backtest results.

    uv run python -m scripts.experiments.report served baseline [--halves]

The first name is the reference: every other row also gets the paired
bootstrap 95 % interval of its MAE difference to it (resampling whole days, so
the strong correlation inside a day does not inflate the confidence).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.experiments.harness import EVAL_CUTS, summarise

RESULTS = Path("tmp/experiments/results")
BOOTSTRAP_ROUNDS = 2000
_HEADLINE = [c.strftime("%H:%M") for c in EVAL_CUTS]


def _load(name: str, half: int | None) -> list[dict]:
    rows = json.loads((RESULTS / f"{name}.json").read_text())
    if half is not None:
        weeks = sorted({r["week"] for r in rows})
        keep = set(weeks[half::2])
        rows = [r for r in rows if r["week"] in keep]
    return rows


def _day_mae(rows: list[dict]) -> dict[str, float]:
    per_day: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["cut"] in _HEADLINE:
            per_day[r["day"]].append(r["mae"])
    return {d: float(np.mean(v)) for d, v in per_day.items()}


def bootstrap_diff(rows: list[dict], ref: list[dict], seed: int = 0) -> tuple[float, float]:
    """95 % interval of mean(MAE - ref MAE) over days, resampling days."""
    a, b = _day_mae(rows), _day_mae(ref)
    days = sorted(set(a) & set(b))
    diff = np.array([a[d] - b[d] for d in days])
    rng = np.random.default_rng(seed)
    means = [diff[rng.integers(0, diff.size, diff.size)].mean() for _ in range(BOOTSTRAP_ROUNDS)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def table(names: list[str], half: int | None = None) -> str:
    head = ["model"] + [f"MAE {c}" for c in _HEADLINE] + ["MAE extra", "near@14"]
    head += ["peak", "peak min", "pinball", "cover", "cover cold", "Δ vs ref (95 %)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    ref = _load(names[0], half)
    for name in names:
        rows = _load(name, half)
        s = summarise(rows)
        a = s["all"]
        cells = [name] + [f"{s[c]['mae']:.0f}" for c in _HEADLINE]
        cells += [f"{s['extra']['mae']:.0f}" if "extra" in s else "-"]
        cells += [f"{s['14:00']['near_mae']:.0f}", f"{a['peak_err']:.0f}", f"{a['peak_min']:.0f}"]
        cells += [f"{a['pinball']:.1f}", f"{a['coverage']:.2f}", f"{s['06:45']['coverage']:.2f}"]
        if name == names[0]:
            cells.append("ref")
        else:
            lo, hi = bootstrap_diff(rows, ref)
            cells.append(f"{lo:+.1f} .. {hi:+.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--halves", action="store_true", help="also show odd / even fold weeks")
    args = ap.parse_args()
    print(table(args.names))
    if args.halves:
        for h in (0, 1):
            print(f"\nfold weeks {h}::2")
            print(table(args.names, h))


if __name__ == "__main__":
    main()
