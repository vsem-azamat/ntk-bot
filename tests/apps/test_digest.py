from datetime import date, datetime

from apps.digest import (
    DigestAction,
    DigestState,
    build_caption,
    decide_digest_action,
)
from apps.predictModels import DayForecast


def _state(d=date(2024, 3, 4), hour=6):
    return DigestState(post_date=d, chat_id=-100, message_id=42, last_update_hour=hour)


def test_post_when_in_window_and_no_state():
    assert decide_digest_action(datetime(2024, 3, 4, 6, 50), None) is DigestAction.POST


def test_no_post_before_window():
    assert decide_digest_action(datetime(2024, 3, 4, 6, 30), None) is DigestAction.NONE


def test_no_post_after_delete_time_when_no_state():
    assert decide_digest_action(datetime(2024, 3, 4, 16, 30), None) is DigestAction.NONE


def test_update_when_hour_advances():
    action = decide_digest_action(datetime(2024, 3, 4, 7, 5), _state(hour=6))
    assert action is DigestAction.UPDATE


def test_no_update_within_same_hour():
    action = decide_digest_action(datetime(2024, 3, 4, 7, 40), _state(hour=7))
    assert action is DigestAction.NONE


def test_delete_at_close_time():
    action = decide_digest_action(datetime(2024, 3, 4, 16, 0), _state(hour=15))
    assert action is DigestAction.DELETE


def test_stale_message_from_previous_day_is_deleted():
    state = _state(d=date(2024, 3, 3), hour=15)  # yesterday
    action = decide_digest_action(datetime(2024, 3, 4, 6, 50), state)
    assert action is DigestAction.DELETE


def test_build_caption_reports_current_and_peak():
    ts = [datetime(2024, 3, 4, 8, 0), datetime(2024, 3, 4, 15, 0), datetime(2024, 3, 4, 22, 0)]
    forecast = DayForecast(timestamps=ts, p10=[5, 80, 5], p50=[10, 90, 10], p90=[20, 110, 20])
    caption = build_caption(forecast, current_count=420)
    assert "Сейчас: <b>420</b> чел." in caption
    assert "<b>~90</b> около <b>15:00</b>" in caption


def test_build_caption_omits_current_when_closed():
    ts = [datetime(2024, 3, 4, 8, 0), datetime(2024, 3, 4, 15, 0)]
    forecast = DayForecast(timestamps=ts, p10=[5, 80], p50=[10, 90], p90=[20, 110])
    caption = build_caption(forecast, current_count=None)
    assert "Сейчас" not in caption
    assert "<b>~90</b> около <b>15:00</b>" in caption
