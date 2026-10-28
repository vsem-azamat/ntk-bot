"""Every test gets an empty PostgreSQL database (``TEST_DATABASE_URL``) and a
temporary ``DATA_DIR`` for the local outbox and model files.

Locally: ``docker run -d --name ntk-test-pg -p 55432:5432 -e POSTGRES_PASSWORD=ntk postgres:18``
and ``export TEST_DATABASE_URL=postgresql://postgres:ntk@localhost:55432/postgres``.
"""

import os

import pytest

from bot import db
from config import cnfg

_TABLES = ("occupancy", "weather", "digest", "scrape_failures")


@pytest.fixture(autouse=True)
def _fresh_database(tmp_path, monkeypatch):
    monkeypatch.setattr(cnfg, "DATA_DIR", str(tmp_path))
    url = os.environ.get("TEST_DATABASE_URL", "")
    monkeypatch.setattr(cnfg, "DATABASE_URL", url)
    db.close()
    if url:
        db.init_db()
        with db._conn() as conn:
            conn.execute(f"TRUNCATE {', '.join(_TABLES)}")
    yield
    db.close()
