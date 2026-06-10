from datetime import datetime

from config import cnfg


def _use_tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(cnfg, "DB_PATH", str(tmp_path / "ntk.sqlite"))


def test_weather_upsert_fetch_and_max_ts(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)
    from bot import db

    db.init_db()
    assert db.max_weather_ts() is None

    db.upsert_weather([(datetime(2024, 3, 1, 9, 0), 5.0, 0.0, 50.0, 3.0)])
    db.upsert_weather([(datetime(2024, 3, 1, 9, 0), 6.0, 0.1, 60.0, 4.0)])  # overwrite
    db.upsert_weather([(datetime(2024, 3, 1, 10, 0), 7.0, None, None, 5.0)])

    assert db.max_weather_ts() == datetime(2024, 3, 1, 10, 0)

    got = db.fetch_weather(datetime(2024, 3, 1, 9, 0), datetime(2024, 3, 1, 9, 0))
    assert got == {datetime(2024, 3, 1, 9, 0): (6.0, 0.1, 60.0, 4.0)}
