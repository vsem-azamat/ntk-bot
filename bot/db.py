"""PostgreSQL persistence for occupancy samples, hourly weather and digest state.

The app works in naive local datetimes (Europe/Prague). Columns are
``timestamptz``; every connection runs with ``timezone=Europe/Prague`` so naive
parameters are interpreted as Prague time and values read back are converted to
naive Prague datetimes.

Occupancy writes must survive a database outage: if Postgres is unreachable the
sample goes to a small local SQLite outbox (``DATA_DIR/outbox.sqlite``) and is
flushed on the next successful write or at startup.
"""

import logging
import sqlite3
import threading
from collections.abc import Generator
from contextlib import closing, contextmanager
from datetime import date, datetime
from pathlib import Path

import psycopg
from psycopg_pool import ConnectionPool

from config import cnfg

logger = logging.getLogger(__name__)

_TS_FORMAT = "%Y-%m-%d %H:%M"
_TZ = "Europe/Prague"
_POOL_TIMEOUT_S = 10

_SCHEMA = """
CREATE TABLE IF NOT EXISTS occupancy (
    ts timestamptz PRIMARY KEY,
    people integer NOT NULL
);
CREATE TABLE IF NOT EXISTS weather (
    ts timestamptz PRIMARY KEY,
    temp real,
    precip real,
    cloud real,
    wind real
);
CREATE TABLE IF NOT EXISTS digest (
    post_date date PRIMARY KEY,
    chat_id bigint NOT NULL,
    message_id bigint NOT NULL,
    last_update_hour integer NOT NULL
);
CREATE TABLE IF NOT EXISTS scrape_failures (
    ts timestamptz NOT NULL,
    error text NOT NULL
);
CREATE INDEX IF NOT EXISTS scrape_failures_ts ON scrape_failures (ts);
"""

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def _naive(ts: datetime) -> datetime:
    return ts.replace(tzinfo=None)


def _get_pool() -> ConnectionPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            if not cnfg.DATABASE_URL:
                raise RuntimeError("DATABASE_URL is not configured")
            _pool = ConnectionPool(
                cnfg.DATABASE_URL,
                min_size=1,
                max_size=4,
                timeout=_POOL_TIMEOUT_S,
                kwargs={"autocommit": True, "options": f"-c timezone={_TZ}"},
                open=True,
            )
        return _pool


_schema_ready = False


@contextmanager
def _conn() -> Generator[psycopg.Connection]:
    """Pooled connection; creates the schema on first successful use, so a
    database that was down at startup is initialised as soon as it is back."""
    global _schema_ready
    with _get_pool().connection() as conn:
        if not _schema_ready:
            conn.execute(_SCHEMA)
            _schema_ready = True
        yield conn


def close() -> None:
    """Close the connection pool (shutdown / tests switching databases)."""
    global _pool, _schema_ready
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None
        _schema_ready = False
    _cache.reset()


def init_db() -> None:
    """Create the tables if needed, then flush any buffered samples."""
    with _conn():
        pass
    flush_outbox()


# >>>>>>>>>> OCCUPANCY <<<<<<<<<< #


class _OccupancyCache:
    """In-process copy of the occupancy series, refreshed incrementally.

    The forecaster, plot and digest each read the whole series; re-downloading
    ~100k rows over the network on every /graph is wasteful. Only the latest
    minute is ever overwritten (same-minute re-insert), so re-reading from the
    last cached timestamp keeps the copy exact."""

    def __init__(self) -> None:
        self._rows: list[tuple[datetime, int]] = []
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._rows = []

    def fetch(self) -> list[tuple[datetime, int]]:
        with self._lock, _conn() as conn:
            if self._rows:
                last = self._rows[-1][0]
                fresh = conn.execute(
                    "SELECT ts, people FROM occupancy WHERE ts >= %s ORDER BY ts", (last,)
                ).fetchall()
                self._rows = [r for r in self._rows if r[0] < last]
            else:
                fresh = conn.execute("SELECT ts, people FROM occupancy ORDER BY ts").fetchall()
            self._rows.extend((_naive(ts), people) for ts, people in fresh)
            return list(self._rows)


_cache = _OccupancyCache()


def _outbox_path() -> str:
    return str(Path(cnfg.DATA_DIR) / "outbox.sqlite")


def _outbox() -> sqlite3.Connection:
    conn = sqlite3.connect(_outbox_path())
    conn.execute("CREATE TABLE IF NOT EXISTS occupancy (ts TEXT PRIMARY KEY, people INTEGER)")
    return conn


def _write_occupancy(rows: list[tuple[datetime, int]]) -> None:
    with _conn() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO occupancy (ts, people) VALUES (%s, %s)"
            " ON CONFLICT (ts) DO UPDATE SET people = excluded.people",
            rows,
        )


def flush_outbox() -> int:
    """Push buffered samples to Postgres; returns how many were flushed."""
    if not Path(_outbox_path()).exists():
        return 0
    with closing(_outbox()) as box:
        pending = box.execute("SELECT ts, people FROM occupancy ORDER BY ts").fetchall()
        if not pending:
            return 0
        _write_occupancy([(datetime.strptime(ts, _TS_FORMAT), p) for ts, p in pending])
        with box:
            box.execute("DELETE FROM occupancy")
    logger.info("Flushed %d buffered occupancy samples to Postgres", len(pending))
    return len(pending)


def insert_occupancy(ts: datetime, people: int) -> None:
    """Store one sample (overwriting the same minute). Falls back to the local
    outbox when Postgres is unreachable, so a database outage loses no data."""
    ts = ts.replace(second=0, microsecond=0)
    try:
        flush_outbox()
        _write_occupancy([(ts, people)])
    except (psycopg.OperationalError, RuntimeError) as exc:
        logger.warning("Postgres unavailable (%s); buffering sample locally", exc)
        with closing(_outbox()) as box, box:
            box.execute(
                "INSERT INTO occupancy (ts, people) VALUES (?, ?)"
                " ON CONFLICT(ts) DO UPDATE SET people=excluded.people",
                (ts.strftime(_TS_FORMAT), people),
            )


def fetch_occupancy() -> list[tuple[datetime, int]]:
    """Return all occupancy samples as ``(datetime, people)`` tuples, oldest first."""
    return _cache.fetch()


def latest_occupancy() -> tuple[datetime, int] | None:
    """Return the most recent non-zero sample, or ``None`` if there is none."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT ts, people FROM occupancy WHERE people > 0 ORDER BY ts DESC LIMIT 1"
        ).fetchone()
    return (_naive(row[0]), row[1]) if row else None


def iter_rows() -> list[str]:
    """Return all samples as legacy-format ``"YYYY-MM-DD HH:MM - N"`` rows, oldest first."""
    return [f"{ts.strftime(_TS_FORMAT)} - {people}" for ts, people in fetch_occupancy()]


def export_text() -> bytes:
    """Render the whole series as a downloadable ``ntk_data.txt`` blob."""
    rows = iter_rows()
    if not rows:
        return b""
    return ("\n".join(rows) + "\n").encode("utf-8")


def record_scrape_failure(ts: datetime, error: str) -> None:
    """Log a failed scrape so gaps in the series are explained by data, not only logs.
    Best effort: a database outage here must not mask the original failure."""
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT INTO scrape_failures (ts, error) VALUES (%s, %s)", (ts, error[:500])
            )
    except Exception:
        logger.exception("Could not record scrape failure")


# >>>>>>>>>> WEATHER <<<<<<<<<< #


def upsert_weather(
    rows: list[tuple[datetime, float | None, float | None, float | None, float | None]],
) -> None:
    """Insert/overwrite hourly weather rows keyed by their floored-to-hour timestamp."""
    with _conn() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO weather (ts, temp, precip, cloud, wind) VALUES (%s, %s, %s, %s, %s)"
            " ON CONFLICT (ts) DO UPDATE SET"
            " temp = excluded.temp, precip = excluded.precip,"
            " cloud = excluded.cloud, wind = excluded.wind",
            [
                (ts.replace(minute=0, second=0, microsecond=0), t, p, c, w)
                for ts, t, p, c, w in rows
            ],
        )


def fetch_weather(
    start: datetime, end: datetime
) -> dict[datetime, tuple[float | None, float | None, float | None, float | None]]:
    """Return ``{hour_datetime: (temp, precip, cloud, wind)}`` within ``[start, end]``."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT ts, temp, precip, cloud, wind FROM weather WHERE ts BETWEEN %s AND %s",
            (start, end),
        ).fetchall()
    return {_naive(ts): (t, p, c, w) for ts, t, p, c, w in rows}


def max_weather_ts() -> datetime | None:
    """Return the latest weather timestamp, or ``None`` if the table is empty."""
    with _conn() as conn:
        row = conn.execute("SELECT max(ts) FROM weather").fetchone()
    return _naive(row[0]) if row and row[0] else None


# >>>>>>>>>> DIGEST <<<<<<<<<< #


def get_digest_state() -> tuple[str, int, int, int] | None:
    """Return the single active digest row ``(post_date, chat_id, message_id,
    last_update_hour)`` or ``None`` if no digest is tracked."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT post_date, chat_id, message_id, last_update_hour FROM digest LIMIT 1"
        ).fetchone()
    if not row:
        return None
    post_date: date = row[0]
    return (post_date.isoformat(), row[1], row[2], row[3])


def save_digest_state(post_date: str, chat_id: int, message_id: int, last_update_hour: int) -> None:
    """Persist the active digest (only one is ever tracked at a time)."""
    with _conn() as conn, conn.transaction():
        conn.execute("DELETE FROM digest")
        conn.execute(
            "INSERT INTO digest (post_date, chat_id, message_id, last_update_hour)"
            " VALUES (%s, %s, %s, %s)",
            (post_date, chat_id, message_id, last_update_hour),
        )


def clear_digest_state() -> None:
    """Forget the active digest (after it has been deleted from the chat)."""
    with _conn() as conn:
        conn.execute("DELETE FROM digest")


# >>>>>>>>>> LEGACY IMPORT <<<<<<<<<< #


def import_legacy_sqlite(path: str) -> tuple[int, int] | None:
    """One-time copy of the pre-Postgres ``ntk.sqlite`` (occupancy + weather).

    Idempotent (``ON CONFLICT DO NOTHING``); after verifying that every legacy
    occupancy timestamp exists in Postgres it writes ``<path>.imported`` so later
    starts skip it. The SQLite file itself is left untouched as a cold copy.
    Returns ``(occupancy_rows, weather_rows)`` or ``None`` when there is nothing to do."""
    marker = Path(f"{path}.imported")
    if not Path(path).exists() or marker.exists():
        return None
    with closing(sqlite3.connect(path)) as src:
        occ = [
            (datetime.strptime(ts, _TS_FORMAT), p)
            for ts, p in src.execute("SELECT ts, people FROM occupancy")
        ]
        weather = [
            (datetime.strptime(ts, _TS_FORMAT), t, pr, c, w)
            for ts, t, pr, c, w in src.execute("SELECT ts, temp, precip, cloud, wind FROM weather")
        ]
    with _conn() as conn, conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO occupancy (ts, people) VALUES (%s, %s) ON CONFLICT (ts) DO NOTHING", occ
        )
        cur.executemany(
            "INSERT INTO weather (ts, temp, precip, cloud, wind) VALUES (%s, %s, %s, %s, %s)"
            " ON CONFLICT (ts) DO NOTHING",
            weather,
        )
        present = cur.execute(
            "SELECT count(*) FROM occupancy WHERE ts = ANY(%s)", ([ts for ts, _ in occ],)
        ).fetchone()
    if present is None or present[0] != len(occ):
        raise RuntimeError(f"Legacy import incomplete: {present} of {len(occ)} rows present")
    marker.write_text(datetime.now().isoformat(), encoding="utf-8")
    _cache.reset()
    logger.info("Imported legacy SQLite: %d occupancy, %d weather rows", len(occ), len(weather))
    return len(occ), len(weather)
