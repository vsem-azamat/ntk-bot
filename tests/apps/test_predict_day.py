import os
from datetime import datetime, timedelta

import numpy as np

from config import cnfg


async def test_predict_day_falls_back_to_baseline_and_returns_monotone_band(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cnfg, "DB_PATH", str(tmp_path / "ntk.sqlite"))
    monkeypatch.setattr(cnfg, "DATA_DIR", str(tmp_path))
    from bot import db

    db.init_db()
    base = datetime(2024, 3, 4, 9, 0)  # Monday
    for w in range(5):
        db.insert_occupancy(base + timedelta(weeks=w), 100 + w)

    from apps.predictModels import predictModels

    forecast = await predictModels.predict_day(datetime(2024, 4, 8, 12, 0))  # a Monday

    assert len(forecast.timestamps) == len(forecast.p50) > 0
    for lo, mid, hi in zip(forecast.p10, forecast.p50, forecast.p90, strict=True):
        assert lo <= mid <= hi
        assert lo >= 0


def _seed_pattern(db, days=20):
    base = datetime(2024, 1, 1, 8, 0)
    for d in range(days):
        for h in range(8, 20):  # 12 samples/day -> 240 rows (> 50 threshold)
            ts = base + timedelta(days=d, hours=h - 8)
            db.insert_occupancy(ts, 100 + (h - 8) * 20)  # learnable hour-of-day pattern


async def test_catboost_fit_save_load_predict_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cnfg, "DB_PATH", str(tmp_path / "ntk.sqlite"))
    monkeypatch.setattr(cnfg, "DATA_DIR", str(tmp_path))
    from apps.features import build_features
    from apps.predictModels import _model_path, predictModels
    from bot import db

    db.init_db()
    _seed_pattern(db)
    occ = db.fetch_occupancy()
    feats = build_features([dt for dt, _ in occ], {})
    y = np.array([c for _, c in occ], dtype=float)

    predictModels._fit_and_save(feats, y)  # trains the trio via Pool(cat_features=[5,6])

    for name in ("p10", "p50", "p90"):
        assert os.path.exists(_model_path(name)), f"{name}.cbm not written"

    preds = predictModels._predict_quantile("p50", [dt for dt, _ in occ][:5], {})
    assert len(preds) == 5
    assert all(p >= 0 for p in preds)


async def test_predict_day_catboost_branch_returns_monotone_band(tmp_path, monkeypatch):
    monkeypatch.setattr(cnfg, "DB_PATH", str(tmp_path / "ntk.sqlite"))
    monkeypatch.setattr(cnfg, "DATA_DIR", str(tmp_path))
    from apps.features import build_features
    from apps.predictModels import predictModels
    from bot import db

    db.init_db()
    _seed_pattern(db)
    occ = db.fetch_occupancy()
    predictModels._fit_and_save(build_features([dt for dt, _ in occ], {}),
                                np.array([c for _, c in occ], dtype=float))
    predictModels._write_choice("catboost")  # force the catboost branch

    import apps.weather_history as wh

    async def fake_forecast(target_day):
        return {}

    monkeypatch.setattr(wh, "forecast_weather", fake_forecast)

    forecast = await predictModels.predict_day(datetime(2024, 2, 5, 12, 0))  # Monday

    assert len(forecast.timestamps) == len(forecast.p50) > 0
    for lo, mid, hi in zip(forecast.p10, forecast.p50, forecast.p90, strict=True):
        assert lo <= mid <= hi
        assert lo >= 0
