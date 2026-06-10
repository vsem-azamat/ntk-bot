from datetime import datetime, timedelta

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
