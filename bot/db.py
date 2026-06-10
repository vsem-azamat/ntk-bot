"""SQLite persistence for NTK occupancy samples.

Rows are exposed in the legacy ``"YYYY-MM-DD HH:MM - N"`` text format so the
existing parsing/ML/plotting code keeps working unchanged.
"""

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from config import cnfg

_TS_FORMAT = "%Y-%m-%d %H:%M"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(cnfg.DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create the data directory and occupancy table if they do not exist."""
    Path(cnfg.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect()) as conn, conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS occupancy ( ts TEXT PRIMARY KEY, people INTEGER NOT NULL)"
        )


def insert_occupancy(ts: datetime, people: int) -> None:
    """Insert one sample, overwriting any existing row for the same minute."""
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO occupancy (ts, people) VALUES (?, ?)"
            " ON CONFLICT(ts) DO UPDATE SET people=excluded.people",
            (ts.strftime(_TS_FORMAT), people),
        )


def iter_rows() -> list[str]:
    """Return all samples as legacy-format rows, oldest first."""
    with closing(_connect()) as conn, conn:
        rows = conn.execute("SELECT ts, people FROM occupancy ORDER BY ts").fetchall()
    return [f"{ts} - {people}" for ts, people in rows]


def fetch_occupancy() -> list[tuple[datetime, int]]:
    """Return all occupancy samples as ``(datetime, people)`` tuples, oldest first."""
    with closing(_connect()) as conn, conn:
        rows = conn.execute("SELECT ts, people FROM occupancy ORDER BY ts").fetchall()
    return [(datetime.strptime(ts, _TS_FORMAT), people) for ts, people in rows]


def export_text() -> bytes:
    """Render the whole series as a downloadable ``ntk_data.txt`` blob."""
    rows = iter_rows()
    if not rows:
        return b""
    return ("\n".join(rows) + "\n").encode("utf-8")
