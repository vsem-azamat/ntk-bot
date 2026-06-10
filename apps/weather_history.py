"""Open-meteo weather ingest. Owns the single definition of the model's weather
variables so archive (training) and forecast (prediction) stay consistent.
"""

import logging
from datetime import datetime, timedelta

import aiohttp

from apps.features import WeatherRow

logger = logging.getLogger(__name__)

# NTK coordinates (matches apps/weather_api.py).
_LAT = 50.1038
_LON = 14.3906
_TZ = "Europe/Berlin"

# The four variables the model consumes, in the order features.py expects them.
_HOURLY = "temperature_2m,precipitation,cloudcover,windspeed_10m"
_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


async def _get_json(url: str, params: dict) -> dict:
    """Network seam (monkeypatched in tests)."""
    async with aiohttp.ClientSession() as session, session.get(url, params=params) as resp:
        return await resp.json()


def _num(v) -> float | None:
    return None if v is None else float(v)


def _payload_to_rows(
    payload: dict,
) -> list[tuple[datetime, float | None, float | None, float | None, float | None]]:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    return [
        (
            datetime.strptime(times[i], "%Y-%m-%dT%H:%M"),
            _num(hourly["temperature_2m"][i]),
            _num(hourly["precipitation"][i]),
            _num(hourly["cloudcover"][i]),
            _num(hourly["windspeed_10m"][i]),
        )
        for i in range(len(times))
    ]


def _start_date() -> datetime | None:
    """Earliest hour we still need: hour after the last stored weather, else the
    first occupancy date. Returns None when there is no occupancy at all."""
    from bot import db

    last = db.max_weather_ts()
    if last is not None:
        return last + timedelta(hours=1)
    occupancy = db.fetch_occupancy()
    return occupancy[0][0].replace(minute=0, second=0, microsecond=0) if occupancy else None


async def backfill() -> int:
    """Fetch archive weather from the last-known hour to today and upsert it.

    Idempotent and incremental. Returns the number of rows written."""
    from bot import db

    start = _start_date()
    if start is None:
        return 0
    end = datetime.now()
    if start.date() > end.date():
        return 0

    payload = await _get_json(
        _ARCHIVE_URL,
        {
            "latitude": _LAT,
            "longitude": _LON,
            "timezone": _TZ,
            "hourly": _HOURLY,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
        },
    )
    rows = _payload_to_rows(payload)
    if rows:
        db.upsert_weather(rows)
    return len(rows)


async def forecast_weather(target_day: datetime) -> dict[datetime, WeatherRow]:
    """Return ``{hour: (temp, precip, cloud, wind)}`` forecast for ``target_day``."""
    payload = await _get_json(
        _FORECAST_URL,
        {
            "latitude": _LAT,
            "longitude": _LON,
            "timezone": _TZ,
            "hourly": _HOURLY,
            "forecast_days": 2,
        },
    )
    return {dt: (t, p, c, w) for dt, t, p, c, w in _payload_to_rows(payload)}
