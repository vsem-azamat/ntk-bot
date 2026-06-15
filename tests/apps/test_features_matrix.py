from datetime import datetime

from apps.features import ObsContext, build_matrix


def _ctx():
    rows = [
        (datetime(2025, 3, 10, 9, 0), 100),
        (datetime(2025, 3, 10, 12, 0), 200),
        (datetime(2025, 3, 13, 9, 0), 120),
        (datetime(2025, 3, 13, 9, 20), 160),
    ]
    return ObsContext.from_rows(rows)


def test_base_group_matches_legacy_width():
    ts = [datetime(2025, 3, 13, 10, 0)]
    feats = build_matrix(ts, ctx=_ctx(), weather=None, groups=("base",))
    assert feats.X.shape == (1, 11)


def test_groups_extend_columns_and_names_align():
    ts = [datetime(2025, 3, 13, 10, 0)]
    feats = build_matrix(ts, ctx=_ctx(), weather=None, groups=("base", "regime", "asof"))
    assert feats.X.shape[1] == len(feats.names)
    assert "trailing_level" in feats.names
    assert "last_count" in feats.names


def test_categorical_indices_point_at_string_columns():
    ts = [datetime(2025, 3, 13, 10, 0)]
    feats = build_matrix(ts, ctx=_ctx(), weather=None, groups=("base", "regime", "asof"))
    for idx in feats.categorical_indices:
        assert isinstance(feats.X[0, idx], str)


def test_groups_without_base_have_no_categoricals():
    ts = [datetime(2025, 3, 13, 10, 0)]
    feats = build_matrix(ts, ctx=_ctx(), weather=None, groups=("regime", "asof"))
    assert feats.categorical_indices == []
    assert feats.names == [
        "trailing_level",
        "is_holiday",
        "has_today",
        "last_count",
        "peak_so_far",
        "count_delta",
        "observed_span_min",
    ]
    assert feats.X.shape == (1, 7)


def test_unknown_group_raises():
    import pytest

    with pytest.raises(ValueError):
        build_matrix([datetime(2025, 3, 13, 10, 0)], ctx=_ctx(), groups=("base", "typo"))
