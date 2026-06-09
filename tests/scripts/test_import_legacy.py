import pytest

from bot import db
from scripts.import_legacy_ntk_data import import_file


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db.cnfg, "DB_PATH", str(tmp_path / "ntk.sqlite"))
    db.init_db()
    return tmp_path


def test_import_valid_and_malformed(tmp_db):
    legacy = tmp_db / "ntk_data.txt"
    legacy.write_text(
        "2023-09-24 12:10 - 118\n"
        "2023-09-24 12:20 - 120\n"
        "\n"  # blank
        "garbage without separator\n"  # malformed
        "2023-09-24 12:30 - 127\n"
    )
    imported, skipped = import_file(str(legacy))
    assert (imported, skipped) == (3, 2)
    assert db.iter_rows() == [
        "2023-09-24 12:10 - 118",
        "2023-09-24 12:20 - 120",
        "2023-09-24 12:30 - 127",
    ]


def test_import_is_idempotent(tmp_db):
    legacy = tmp_db / "ntk_data.txt"
    legacy.write_text("2023-09-24 12:10 - 118\n")
    import_file(str(legacy))
    import_file(str(legacy))
    assert db.iter_rows() == ["2023-09-24 12:10 - 118"]


def test_import_preserves_zero_counts(tmp_db):
    legacy = tmp_db / "ntk_data.txt"
    legacy.write_text("2023-09-24 12:10 - 0\n")
    imported, skipped = import_file(str(legacy))
    assert (imported, skipped) == (1, 0)
    assert db.iter_rows() == ["2023-09-24 12:10 - 0"]
