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
    """Create the data directory and tables if they do not exist."""
    Path(cnfg.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect()) as conn, conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS occupancy ( ts TEXT PRIMARY KEY, people INTEGER NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS weather ("
            " ts TEXT PRIMARY KEY, temp REAL, precip REAL, cloud REAL, wind REAL)"
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


def upsert_weather(
    rows: list[tuple[datetime, float | None, float | None, float | None, float | None]],
) -> None:
    """Insert/overwrite hourly weather rows keyed by their floored-to-hour timestamp."""
    with closing(_connect()) as conn, conn:
        conn.executemany(
            "INSERT INTO weather (ts, temp, precip, cloud, wind) VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(ts) DO UPDATE SET"
            " temp=excluded.temp, precip=excluded.precip,"
            " cloud=excluded.cloud, wind=excluded.wind",
            [
                (ts.replace(minute=0, second=0, microsecond=0).strftime(_TS_FORMAT), t, p, c, w)
                for ts, t, p, c, w in rows
            ],
        )


def fetch_weather(
    start: datetime, end: datetime
) -> dict[datetime, tuple[float | None, float | None, float | None, float | None]]:
    """Return ``{hour_datetime: (temp, precip, cloud, wind)}`` within ``[start, end]``."""
    with closing(_connect()) as conn, conn:
        rows = conn.execute(
            "SELECT ts, temp, precip, cloud, wind FROM weather WHERE ts BETWEEN ? AND ?",
            (start.strftime(_TS_FORMAT), end.strftime(_TS_FORMAT)),
        ).fetchall()
    return {datetime.strptime(ts, _TS_FORMAT): (t, p, c, w) for ts, t, p, c, w in rows}


def max_weather_ts() -> datetime | None:
    """Return the latest weather timestamp, or ``None`` if the table is empty."""
    with closing(_connect()) as conn, conn:
        row = conn.execute("SELECT MAX(ts) FROM weather").fetchone()
    return datetime.strptime(row[0], _TS_FORMAT) if row and row[0] else None


def export_text() -> bytes:
    """Render the whole series as a downloadable ``ntk_data.txt`` blob."""
    rows = iter_rows()
    if not rows:
        return b""
    return ("\n".join(rows) + "\n").encode("utf-8")
