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
# The database is reached over the internet. Without these, a silently dropped
# route leaves a query blocked for ~15 min of TCP retransmits (the pool timeout
# only bounds *acquiring* a connection). With them a dead peer surfaces as an
# OperationalError within ~15 s and the sample goes to the outbox.
_CONNECT_KWARGS = {
    "autocommit": True,
    "connect_timeout": 5,
    "tcp_user_timeout": 15_000,
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 3,
    "options": f"-c timezone={_TZ} -c statement_timeout=60000",
}

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
CREATE TABLE IF NOT EXISTS settings (
    key text PRIMARY KEY,
    value text NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
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
                kwargs=dict(_CONNECT_KWARGS),
                check=ConnectionPool.check_connection,
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
    ~100k rows over the network on every /graph is wasteful. The bot only ever
    overwrites the latest minute, so re-reading from the last cached timestamp
    keeps the copy exact; a cheap row-count check catches writes made by anyone
    else (imports, restores, manual fixes) and triggers a full reload. If the
    database is unreachable, the last good copy is served."""

    def __init__(self) -> None:
        self._rows: list[tuple[datetime, int]] = []
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._rows = []

    def fetch(self) -> list[tuple[datetime, int]]:
        with self._lock:
            try:
                with _conn() as conn:
                    self._refresh(conn)
            except psycopg.Error:
                if not self._rows:
                    raise
                logger.warning("Occupancy read failed; serving the cached copy", exc_info=True)
            return list(self._rows)

    def _refresh(self, conn: psycopg.Connection) -> None:
        if self._rows:
            last = self._rows[-1][0]
            fresh = conn.execute(
                "SELECT ts, people FROM occupancy WHERE ts >= %s ORDER BY ts", (last,)
            ).fetchall()
            rows = [r for r in self._rows if r[0] < last]
            rows.extend((_naive(ts), people) for ts, people in fresh)
            total = conn.execute("SELECT count(*) FROM occupancy").fetchone()
            if total is not None and total[0] == len(rows):
                self._rows = rows
                return
        fresh = conn.execute("SELECT ts, people FROM occupancy ORDER BY ts").fetchall()
        self._rows = [(_naive(ts), people) for ts, people in fresh]


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
            box.executemany(
                "DELETE FROM occupancy WHERE ts = ? AND people = ?", [tuple(r) for r in pending]
            )
    logger.info("Flushed %d buffered occupancy samples to Postgres", len(pending))
    return len(pending)


def insert_occupancy(ts: datetime, people: int) -> bool:
    """Store one sample (overwriting the same minute). Returns True when it
    reached Postgres; on any database error it goes to the local outbox instead
    (returns False), so an outage, a read-only server or a missing table during
    a restore loses no data."""
    ts = ts.replace(second=0, microsecond=0)
    try:
        flush_outbox()
        _write_occupancy([(ts, people)])
        return True
    except (psycopg.Error, RuntimeError) as exc:
        logger.warning("Postgres write failed (%s); buffering sample locally", exc)
        buffer_occupancy(ts, people)
        return False


def buffer_occupancy(ts: datetime, people: int) -> None:
    """Put one sample into the local outbox (flushed on the next good write)."""
    with closing(_outbox()) as box, box:
        box.execute(
            "INSERT INTO occupancy (ts, people) VALUES (?, ?)"
            " ON CONFLICT(ts) DO UPDATE SET people=excluded.people",
            (ts.replace(second=0, microsecond=0).strftime(_TS_FORMAT), people),
        )


def buffered_count() -> int:
    """Number of samples waiting in the local outbox."""
    if not Path(_outbox_path()).exists():
        return 0
    with closing(_outbox()) as box:
        row = box.execute("SELECT count(*) FROM occupancy").fetchone()
    return int(row[0]) if row else 0


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


# >>>>>>>>>> SETTINGS <<<<<<<<<< #


def get_setting(key: str) -> str | None:
    """Runtime setting edited from Telegram (LLM prompt, reply probability)."""
    with _conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = %s", (key,)).fetchone()
    return row[0] if row else None


def set_setting(key: str, value: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (%s, %s)"
            " ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = now()",
            (key, value),
        )


def set_settings(values: dict[str, str]) -> None:
    """Upsert several settings atomically."""
    with _conn() as conn, conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO settings (key, value) VALUES (%s, %s)"
            " ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = now()",
            list(values.items()),
        )


# >>>>>>>>>> LEGACY IMPORT <<<<<<<<<< #


def legacy_import_pending(path: str) -> bool:
    """True while a legacy SQLite file exists that has not been imported yet."""
    return Path(path).exists() and not Path(f"{path}.imported").exists()


def import_legacy_sqlite(path: str) -> tuple[int, int] | None:
    """One-time copy of the pre-Postgres ``ntk.sqlite`` (occupancy + weather).

    Idempotent (``ON CONFLICT DO NOTHING``); after verifying that every legacy
    occupancy timestamp exists in Postgres it writes ``<path>.imported`` so later
    starts skip it. The SQLite file itself is left untouched as a cold copy.
    Returns ``(occupancy_rows, weather_rows)`` or ``None`` when there is nothing to do."""
    marker = Path(f"{path}.imported")
    if not legacy_import_pending(path):
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
        # Compare against the server-side cast so local times that collapse onto
        # one instant (e.g. inside the spring-forward gap) count once.
        stamps = [ts for ts, _ in occ]
        expected = cur.execute(
            "SELECT count(DISTINCT t) FROM unnest(%s::timestamptz[]) AS t", (stamps,)
        ).fetchone()
        present = cur.execute(
            "SELECT count(*) FROM occupancy WHERE ts = ANY(%s::timestamptz[])", (stamps,)
        ).fetchone()
    if present is None or expected is None or present[0] != expected[0]:
        raise RuntimeError(f"Legacy import incomplete: {present} of {expected} instants present")
    if expected[0] != len(occ):
        logger.warning(
            "Legacy import: %d local times map to an existing instant", len(occ) - expected[0]
        )
    marker.write_text(datetime.now().isoformat(), encoding="utf-8")
    _cache.reset()
    logger.info("Imported legacy SQLite: %d occupancy, %d weather rows", len(occ), len(weather))
    return len(occ), len(weather)
