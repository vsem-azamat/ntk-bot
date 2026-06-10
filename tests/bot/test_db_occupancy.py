from datetime import datetime

from config import cnfg


def _use_tmp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "ntk.sqlite"
    monkeypatch.setattr(cnfg, "DB_PATH", str(db_path))


def test_fetch_occupancy_returns_structured_rows_oldest_first(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)
    from bot import db

    db.init_db()
    db.insert_occupancy(datetime(2024, 3, 1, 10, 0), 100)
    db.insert_occupancy(datetime(2024, 3, 1, 9, 0), 50)

    rows = db.fetch_occupancy()

    assert rows == [
        (datetime(2024, 3, 1, 9, 0), 50),
        (datetime(2024, 3, 1, 10, 0), 100),
    ]
