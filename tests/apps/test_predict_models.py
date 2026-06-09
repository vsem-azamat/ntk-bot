from datetime import datetime

from apps.predictModels import (
    extract_features,
    model_filename,
    parse_row_datetime,
    predictModels,
)


def test_model_filename_has_no_parens():
    assert model_filename("RandomForestRegressor") == "model_RandomForestRegressor.pkl"


def test_parse_row_datetime():
    assert parse_row_datetime("2024-03-01 09:30 - 42") == datetime(2024, 3, 1, 9, 30)


def test_extract_features_shape_and_values():
    feats = extract_features([datetime(2024, 3, 1, 9, 30)])
    # day_of_year, weekday, total_minutes, month
    assert feats.shape == (1, 4)
    assert feats[0].tolist() == [61, 4, 570, 3]


async def test_remove_zero_values_skips_zero_and_malformed_lines():
    data = [
        "2024-03-01 09:30 - 42",
        "2024-03-01 09:50 - 0",  # zero -> dropped
        "",  # blank -> skipped, no IndexError
        "garbage line without separator",  # malformed -> skipped
        "2024-03-01 10:10 - 7",
    ]
    result = await predictModels.remove_zero_values(data)
    assert result == ["2024-03-01 09:30 - 42", "2024-03-01 10:10 - 7"]
