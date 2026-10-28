import sqlite3
from datetime import datetime
from pathlib import Path

from bot import db
from config import cnfg


def test_naive_prague_times_roundtrip_across_dst():
    summer, winter = datetime(2026, 7, 1, 23, 40), datetime(2026, 12, 1, 23, 40)
    db.insert_occupancy(summer, 10)
    db.insert_occupancy(winter, 20)
    assert db.fetch_occupancy() == [(summer, 10), (winter, 20)]
    with db._conn() as conn:
        utc = conn.execute(
            "SELECT to_char(ts AT TIME ZONE 'UTC', 'HH24:MI') FROM occupancy ORDER BY ts"
        ).fetchall()
    assert [r[0] for r in utc] == ["21:40", "22:40"]


def test_cache_picks_up_new_rows_and_same_minute_overwrite():
    db.insert_occupancy(datetime(2026, 10, 1, 9, 0), 1)
    assert db.fetch_occupancy() == [(datetime(2026, 10, 1, 9, 0), 1)]
    db.insert_occupancy(datetime(2026, 10, 1, 9, 0), 5)
    db.insert_occupancy(datetime(2026, 10, 1, 9, 20), 7)
    assert db.fetch_occupancy() == [
        (datetime(2026, 10, 1, 9, 0), 5),
        (datetime(2026, 10, 1, 9, 20), 7),
    ]


def test_outage_buffers_locally_then_flushes(monkeypatch):
    good_url = cnfg.DATABASE_URL
    db.close()
    monkeypatch.setattr(
        cnfg, "DATABASE_URL", "postgresql://nobody@127.0.0.1:1/none?connect_timeout=1"
    )
    monkeypatch.setattr(db, "_POOL_TIMEOUT_S", 1)
    db.insert_occupancy(datetime(2026, 10, 22, 10, 0), 300)  # must not raise
    assert Path(cnfg.DATA_DIR, "outbox.sqlite").exists()

    db.close()
    monkeypatch.setattr(cnfg, "DATABASE_URL", good_url)
    db.insert_occupancy(datetime(2026, 10, 22, 10, 20), 320)
    assert db.fetch_occupancy() == [
        (datetime(2026, 10, 22, 10, 0), 300),
        (datetime(2026, 10, 22, 10, 20), 320),
    ]
    assert db.flush_outbox() == 0


def _legacy(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE occupancy (ts TEXT PRIMARY KEY, people INTEGER)")
        conn.execute(
            "CREATE TABLE weather (ts TEXT PRIMARY KEY, temp REAL, precip REAL, cloud REAL, wind REAL)"
        )
        conn.executemany(
            "INSERT INTO occupancy VALUES (?, ?)",
            [("2023-09-24 12:10", 118), ("2023-09-24 12:20", 120)],
        )
        conn.execute("INSERT INTO weather VALUES ('2023-09-24 12:00', 18.5, 0, 40, 3)")


def test_legacy_import_is_one_shot_and_keeps_newer_rows(tmp_path):
    legacy = tmp_path / "ntk.sqlite"
    _legacy(legacy)
    db.insert_occupancy(datetime(2023, 9, 24, 12, 20), 999)  # already in Postgres: wins
    assert db.import_legacy_sqlite(str(legacy)) == (2, 1)
    assert db.fetch_occupancy() == [
        (datetime(2023, 9, 24, 12, 10), 118),
        (datetime(2023, 9, 24, 12, 20), 999),
    ]
    assert db.fetch_weather(datetime(2023, 9, 24), datetime(2023, 9, 25)) == {
        datetime(2023, 9, 24, 12, 0): (18.5, 0.0, 40.0, 3.0)
    }
    assert Path(f"{legacy}.imported").exists() and legacy.exists()
    assert db.import_legacy_sqlite(str(legacy)) is None


def test_scrape_failures_are_recorded():
    db.record_scrape_failure(datetime(2026, 10, 21, 17, 40), "ConnectionTimeoutError: techlib")
    with db._conn() as conn:
        assert conn.execute("SELECT count(*) FROM scrape_failures").fetchone() == (1,)
