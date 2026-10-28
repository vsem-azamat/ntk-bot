from datetime import date, datetime

from apps.hours import (
    FORECAST_STEP,
    day_type,
    forecast_grid,
    is_open_sample,
    opening_time,
    usable_samples,
)


def test_weekday_and_weekend_opening():
    assert opening_time(date(2026, 10, 27)).hour == 8  # Tuesday
    assert opening_time(date(2026, 10, 31)).hour == 10  # Saturday
    assert opening_time(date(2026, 10, 28)).hour == 10  # Czech holiday (Wednesday)


def test_night_residue_is_not_an_open_sample():
    assert not is_open_sample(datetime(2026, 10, 27, 1, 40))
    assert not is_open_sample(datetime(2026, 10, 31, 9, 40))
    assert is_open_sample(datetime(2026, 10, 27, 8, 0))
    assert is_open_sample(datetime(2026, 10, 27, 23, 40))


def test_day_types():
    assert day_type(date(2026, 10, 26)) == "mon_thu"
    assert day_type(date(2026, 10, 30)) == "fri"
    assert day_type(date(2026, 10, 31)) == "sat"
    assert day_type(date(2026, 11, 1)) == "sun_holiday"
    assert day_type(date(2026, 10, 28)) == "sun_holiday"


def test_usable_samples_removes_zero_counts():
    rows = [
        (datetime(2024, 3, 1, 9, 0), 42),
        (datetime(2024, 3, 1, 9, 20), 0),  # closed -> dropped
        (datetime(2024, 3, 1, 10, 10), 7),
    ]
    assert usable_samples(rows) == [
        (datetime(2024, 3, 1, 9, 0), 42),
        (datetime(2024, 3, 1, 10, 10), 7),
    ]


def test_usable_samples_drops_pre_opening_residue():
    rows = [
        (datetime(2026, 10, 27, 1, 40), 205),  # night residue -> dropped
        (datetime(2026, 10, 27, 8, 20), 49),
        (datetime(2026, 10, 31, 9, 40), 12),  # Saturday before 10:00 -> dropped
    ]
    assert usable_samples(rows) == [(datetime(2026, 10, 27, 8, 20), 49)]


def test_forecast_grid_runs_from_opening_to_midnight():
    weekday = forecast_grid(date(2024, 3, 4))
    assert weekday[0] == datetime(2024, 3, 4, 8, 0)
    assert weekday[-1] == datetime(2024, 3, 4, 23, 50)
    assert weekday[1] - weekday[0] == FORECAST_STEP
    assert forecast_grid(date(2024, 3, 9))[0] == datetime(2024, 3, 9, 10, 0)
