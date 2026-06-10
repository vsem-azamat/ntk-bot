import math
from datetime import datetime

from apps.features import CATEGORICAL_INDICES, FEATURE_NAMES, build_features


def test_feature_names_and_categorical_indices():
    assert FEATURE_NAMES == [
        "minute_sin", "minute_cos", "is_weekend",
        "yday_sin", "yday_cos", "weekday", "month",
        "temp", "precip", "cloud", "wind",
    ]
    assert CATEGORICAL_INDICES == [5, 6]


def test_cyclical_minute_encoding_wraps_around():
    feats = build_features([datetime(2024, 3, 1, 0, 0), datetime(2024, 3, 1, 23, 59)])
    x = feats.X
    assert abs(float(x[0, 0]) - float(x[1, 0])) < 0.01  # minute_sin
    assert abs(float(x[0, 1]) - float(x[1, 1])) < 0.01  # minute_cos


def test_categoricals_are_strings_and_weekend_flag():
    # 2024-03-02 is a Saturday
    feats = build_features([datetime(2024, 3, 2, 12, 0)])
    x = feats.X
    assert x[0, 5] == "5"      # weekday: Saturday == 5
    assert x[0, 6] == "3"      # month: March
    assert float(x[0, 2]) == 1.0  # is_weekend


def test_empty_timestamps_has_correct_2d_shape():
    feats = build_features([])
    assert feats.X.shape == (0, len(FEATURE_NAMES))
    assert feats.X.dtype == object


def test_cyclical_yday_encoding_wraps_around():
    x = build_features([datetime(2024, 1, 1, 12, 0), datetime(2024, 12, 31, 12, 0)]).X
    assert abs(float(x[0, 3]) - float(x[1, 3])) < 0.05  # yday_sin
    assert abs(float(x[0, 4]) - float(x[1, 4])) < 0.05  # yday_cos


def test_weather_join_fills_matching_hour_and_nans_when_missing():
    ts = datetime(2024, 3, 1, 9, 30)
    weather = {datetime(2024, 3, 1, 9, 0): (5.0, 0.0, 50.0, 3.0)}
    x = build_features([ts], weather).X
    assert [float(v) for v in x[0, 7:11]] == [5.0, 0.0, 50.0, 3.0]

    x_missing = build_features([ts], {}).X
    assert all(math.isnan(float(v)) for v in x_missing[0, 7:11])
    assert x.dtype == object
