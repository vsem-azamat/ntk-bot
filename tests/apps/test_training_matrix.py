from datetime import date, datetime, time

from apps.features import ObsContext, build_training_matrix


def _rows():
    # Two prior days + one "training day" (the 12th) with a distinctive spike.
    rows = []
    for day in (10, 11):
        for hour in (9, 12, 15):
            rows.append((datetime(2025, 3, day, hour, 0), 100 + hour))
    # Training day 12: 09:00=130, 12:00=160, 14:00=900 (the spike we watch).
    rows.append((datetime(2025, 3, 12, 9, 0), 130))
    rows.append((datetime(2025, 3, 12, 12, 0), 160))
    rows.append((datetime(2025, 3, 12, 14, 0), 900))
    return rows


def test_truncated_drops_future_same_day_samples():
    ctx = ObsContext.from_rows(_rows())
    t = ctx.truncated(date(2025, 3, 12), datetime(2025, 3, 12, 12, 0))
    kept = t.samples_on(date(2025, 3, 12))
    assert [c for _, c in kept] == [130, 160]  # 14:00=900 dropped


def test_truncated_none_removes_the_whole_day():
    ctx = ObsContext.from_rows(_rows())
    t = ctx.truncated(date(2025, 3, 12), None)
    assert t.samples_on(date(2025, 3, 12)) == []
    assert t.samples_on(date(2025, 3, 11))  # other days untouched


def test_training_matrix_no_asof_self_leak():
    rows = _rows()
    train_days = [date(2025, 3, 12)]
    feats, y = build_training_matrix(
        rows,
        train_days,
        weather=None,
        groups=("base", "asof"),
        cut_times=(None, time(12, 0)),
    )
    # The 14:00 spike (900) appears as a target. For every training row whose
    # label is 900, last_count must NOT be 900 (no leak of the target itself).
    last_count_idx = feats.names.index("last_count")
    for i in range(feats.X.shape[0]):
        if y[i] == 900.0:
            assert float(feats.X[i, last_count_idx]) != 900.0


def test_training_matrix_conditioned_row_sees_prior_sample():
    rows = _rows()
    feats, y = build_training_matrix(
        rows,
        [date(2025, 3, 12)],
        weather=None,
        groups=("base", "asof"),
        cut_times=(time(12, 0),),
    )
    # With cut=12:00, the only target after the cut is 14:00 (900). Its as-of
    # last_count must be the 12:00 reading (160).
    last_count_idx = feats.names.index("last_count")
    has_today_idx = feats.names.index("has_today")
    rows_900 = [i for i in range(feats.X.shape[0]) if y[i] == 900.0]
    assert rows_900
    for i in rows_900:
        assert float(feats.X[i, has_today_idx]) == 1.0
        assert float(feats.X[i, last_count_idx]) == 160.0


def test_cold_cut_has_no_today_context():
    rows = _rows()
    feats, y = build_training_matrix(
        rows, [date(2025, 3, 12)], weather=None, groups=("base", "asof"), cut_times=(None,)
    )
    has_today_idx = feats.names.index("has_today")
    # Cold start: every row reports no today-context.
    assert all(float(feats.X[i, has_today_idx]) == 0.0 for i in range(feats.X.shape[0]))
    # All three of the day's samples are targets in the cold case.
    assert sorted(y.tolist()) == [130.0, 160.0, 900.0]


def test_names_and_columns_align():
    rows = _rows()
    feats, y = build_training_matrix(
        rows, [date(2025, 3, 12)], weather=None, groups=("base", "regime", "asof")
    )
    assert feats.X.shape[1] == len(feats.names)
    assert feats.X.shape[0] == len(y)
