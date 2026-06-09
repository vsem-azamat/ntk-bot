from datetime import datetime

import pytest

from bot import db


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    path = tmp_path / "ntk.sqlite"
    monkeypatch.setattr(db.cnfg, "DB_PATH", str(path))
    db.init_db()
    return path


def test_insert_and_iter_rows(tmp_db):
    db.insert_occupancy(datetime(2024, 3, 1, 9, 30), 42)
    db.insert_occupancy(datetime(2024, 3, 1, 9, 50), 7)
    assert db.iter_rows() == ["2024-03-01 09:30 - 42", "2024-03-01 09:50 - 7"]


def test_insert_is_idempotent_upsert(tmp_db):
    db.insert_occupancy(datetime(2024, 3, 1, 9, 30), 42)
    db.insert_occupancy(datetime(2024, 3, 1, 9, 30), 99)
    assert db.iter_rows() == ["2024-03-01 09:30 - 99"]


def test_rows_are_ordered_by_time(tmp_db):
    db.insert_occupancy(datetime(2024, 3, 1, 9, 50), 7)
    db.insert_occupancy(datetime(2024, 3, 1, 9, 30), 42)
    assert db.iter_rows() == ["2024-03-01 09:30 - 42", "2024-03-01 09:50 - 7"]


def test_export_text_round_trips(tmp_db):
    db.insert_occupancy(datetime(2024, 3, 1, 9, 30), 42)
    assert db.export_text() == b"2024-03-01 09:30 - 42\n"


def test_export_text_empty_db(tmp_db):
    assert db.export_text() == b""
