from config import cnfg


def _use_tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(cnfg, "DB_PATH", str(tmp_path / "ntk.sqlite"))


def test_digest_state_roundtrip_and_clear(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)
    from bot import db

    db.init_db()
    assert db.get_digest_state() is None

    db.save_digest_state("2024-03-04", -100, 42, 6)
    assert db.get_digest_state() == ("2024-03-04", -100, 42, 6)

    # Only one digest is ever tracked — a new save replaces the old.
    db.save_digest_state("2024-03-04", -100, 42, 9)
    assert db.get_digest_state() == ("2024-03-04", -100, 42, 9)

    db.clear_digest_state()
    assert db.get_digest_state() is None
