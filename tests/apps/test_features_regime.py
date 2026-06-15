from datetime import datetime

from apps.features import ObsContext, regime_features


def _history():
    # 3 prior days, rising daily peak (exam ramp).
    rows = []
    for day in (10, 11, 12):
        for hour in (9, 12, 15):
            rows.append((datetime(2025, 3, day, hour, 0), 100 * (day - 9) + hour))
    return rows


def test_trailing_level_uses_only_past_days():
    ctx = ObsContext.from_rows(_history())
    feats = regime_features(datetime(2025, 3, 13, 9, 0), ctx, trailing_days=7)
    assert feats["trailing_level"] > 0
    earlier = regime_features(datetime(2025, 3, 11, 9, 0), ctx, trailing_days=7)
    assert earlier["trailing_level"] < feats["trailing_level"]


def test_trailing_level_excludes_same_day():
    ctx = ObsContext.from_rows(_history())
    spiked = ObsContext.from_rows(_history() + [(datetime(2025, 3, 13, 9, 0), 99999)])
    base = regime_features(datetime(2025, 3, 13, 9, 0), ctx, trailing_days=7)
    same = regime_features(datetime(2025, 3, 13, 9, 0), spiked, trailing_days=7)
    assert base["trailing_level"] == same["trailing_level"]


def test_czech_holiday_flag():
    ctx = ObsContext.from_rows(_history())
    assert regime_features(datetime(2025, 1, 1, 9, 0), ctx)["is_holiday"] == 1.0
    assert regime_features(datetime(2025, 3, 13, 9, 0), ctx)["is_holiday"] == 0.0
