from datetime import datetime


def test_fetch_occupancy_returns_structured_rows_oldest_first(tmp_path, monkeypatch):
    from bot import db

    db.init_db()
    db.insert_occupancy(datetime(2024, 3, 1, 10, 0), 100)
    db.insert_occupancy(datetime(2024, 3, 1, 9, 0), 50)

    rows = db.fetch_occupancy()

    assert rows == [
        (datetime(2024, 3, 1, 9, 0), 50),
        (datetime(2024, 3, 1, 10, 0), 100),
    ]
