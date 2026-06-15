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
