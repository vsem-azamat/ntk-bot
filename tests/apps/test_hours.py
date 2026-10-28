from datetime import date, datetime

from apps.hours import day_type, is_open_sample, opening_time


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
