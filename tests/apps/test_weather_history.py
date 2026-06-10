from datetime import datetime

import apps.weather_history as wh


def test_parse_hourly_payload_to_rows():
    payload = {
        "hourly": {
            "time": ["2024-03-01T09:00", "2024-03-01T10:00"],
            "temperature_2m": [5.0, 6.0],
            "precipitation": [0.0, 0.2],
            "cloudcover": [50, 60],
            "windspeed_10m": [3.0, 4.0],
        }
    }
    rows = wh._payload_to_rows(payload)
    assert rows == [
        (datetime(2024, 3, 1, 9, 0), 5.0, 0.0, 50.0, 3.0),
        (datetime(2024, 3, 1, 10, 0), 6.0, 0.2, 60.0, 4.0),
    ]


def test_payload_to_rows_tolerates_nulls():
    payload = {
        "hourly": {
            "time": ["2024-03-01T09:00"],
            "temperature_2m": [None],
            "precipitation": [None],
            "cloudcover": [None],
            "windspeed_10m": [None],
        }
    }
    rows = wh._payload_to_rows(payload)
    assert rows == [(datetime(2024, 3, 1, 9, 0), None, None, None, None)]


async def test_backfill_upserts_from_last_known_ts(tmp_path, monkeypatch):
    from config import cnfg

    monkeypatch.setattr(cnfg, "DB_PATH", str(tmp_path / "ntk.sqlite"))
    from bot import db

    db.init_db()
    db.insert_occupancy(datetime(2024, 3, 1, 9, 0), 100)  # gives backfill a start date

    async def fake_get_json(url, params):
        return {
            "hourly": {
                "time": ["2024-03-01T09:00"],
                "temperature_2m": [5.0],
                "precipitation": [0.0],
                "cloudcover": [50],
                "windspeed_10m": [3.0],
            }
        }

    monkeypatch.setattr(wh, "_get_json", fake_get_json)

    written = await wh.backfill()

    assert written == 1
    assert db.fetch_weather(datetime(2024, 3, 1, 9, 0), datetime(2024, 3, 1, 9, 0)) == {
        datetime(2024, 3, 1, 9, 0): (5.0, 0.0, 50.0, 3.0)
    }
