"""A candidate = a feature-group selection + model knobs + target transform.

``fit_predict_day`` trains quantile CatBoost on the training days (with leak-free
as-of augmentation via ``build_training_matrix``) and predicts the test day's grid
as of a given cut time. This is the single function the backtest loop calls per
fold.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, time

import numpy as np
from catboost import CatBoostRegressor, Pool

from apps.features import ObsContext, build_matrix, build_training_matrix
from apps.predictModels import _grid

# As-of cut points used to AUGMENT training (so the model sees both the cold
# morning case and conditioned mid-day cases). None == cold/morning start.
_TRAIN_CUTS: tuple[time | None, ...] = (None, time(10, 0), time(12, 0), time(14, 0))


@dataclass
class Candidate:
    name: str
    groups: tuple[str, ...] = ("base", "regime", "asof")
    log_target: bool = False
    iterations: int = 300
    depth: int = 6
    learning_rate: float = 0.1
    # Lower/upper band quantiles. Widen (e.g. 0.05/0.95) to improve coverage.
    lo_alpha: float = 0.1
    hi_alpha: float = 0.9
    extra: dict = field(default_factory=dict)

    def quantiles(self) -> dict[str, float]:
        return {"p10": self.lo_alpha, "p50": 0.5, "p90": self.hi_alpha}


def _xform(y: np.ndarray, log: bool) -> np.ndarray:
    return np.log1p(y) if log else y


def _inv(y: np.ndarray, log: bool) -> np.ndarray:
    return np.expm1(y) if log else y


def fit_models(
    cand: Candidate,
    rows: list[tuple[datetime, int]],
    train_days: list[date],
    weather: dict,
) -> dict[str, CatBoostRegressor]:
    """Train the three quantile models for ``cand`` on ``train_days``.

    The trained model does NOT depend on any prediction cut time — the cut only
    affects the prediction features — so a fold trains once and reuses the models
    across all cut times. Training data is leak-free and as-of augmented via
    ``build_training_matrix``."""
    feats, y = build_training_matrix(
        rows, train_days, weather=weather, groups=cand.groups, cut_times=_TRAIN_CUTS
    )
    models: dict[str, CatBoostRegressor] = {}
    for name, alpha in cand.quantiles().items():
        loss = "RMSE" if alpha == 0.5 else f"Quantile:alpha={alpha}"
        m = CatBoostRegressor(
            loss_function=loss,
            iterations=cand.iterations,
            depth=cand.depth,
            learning_rate=cand.learning_rate,
            verbose=False,
            allow_writing_files=False,
        )
        m.fit(Pool(feats.X, _xform(y, cand.log_target), cat_features=feats.categorical_indices))
        models[name] = m
    return models


def predict_grid(
    cand: Candidate,
    models: dict[str, CatBoostRegressor],
    ctx: ObsContext,
    test_day: date,
    cut: datetime | None,
    weather: dict,
) -> tuple[list[datetime], list[float], list[float], list[float]]:
    """Predict ``test_day``'s grid as of ``cut`` using pre-trained ``models``.

    The as-of context is the history truncated to ``cut`` for the test day, so
    conditioned features reflect only what would be known at ``cut`` (morning =>
    no today-context)."""
    grid = _grid(datetime.combine(test_day, datetime.min.time()))
    pred_ctx = ctx.truncated(test_day, cut)
    gx = build_matrix(grid, ctx=pred_ctx, weather=weather, groups=cand.groups)
    out = {}
    for name in ("p10", "p50", "p90"):
        raw = _inv(np.asarray(models[name].predict(gx.X), float), cand.log_target)
        out[name] = [max(0.0, float(v)) for v in raw]
    bands = [sorted(t) for t in zip(out["p10"], out["p50"], out["p90"], strict=True)]
    return grid, [b[0] for b in bands], [b[1] for b in bands], [b[2] for b in bands]


def fit_predict_day(
    cand: Candidate,
    rows: list[tuple[datetime, int]],
    ctx: ObsContext,
    train_days: list[date],
    test_day: date,
    cut: datetime | None,
    weather: dict,
) -> tuple[list[datetime], list[float], list[float], list[float]]:
    """Convenience: train then predict one (test_day, cut). Prefer ``fit_models``
    + ``predict_grid`` in the backtest loop so a fold trains only once."""
    models = fit_models(cand, rows, train_days, weather)
    return predict_grid(cand, models, ctx, test_day, cut, weather)
