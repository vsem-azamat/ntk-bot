from datetime import datetime

from apps.baseline import build_climatology, climatology_predict


def _rows():
    # Three Fridays, all 09:00, counts 100/200/300 -> p50 == 200
    return [
        (datetime(2024, 3, 1, 9, 0), 100),
        (datetime(2024, 3, 8, 9, 0), 200),
        (datetime(2024, 3, 15, 9, 0), 300),
    ]


def test_climatology_predicts_bucket_median():
    clim = build_climatology(_rows())
    p10, p50, p90 = climatology_predict(clim, [datetime(2024, 3, 22, 9, 0)])  # also a Friday
    assert p50[0] == 200
    assert p10[0] <= p50[0] <= p90[0]


def test_climatology_unknown_bucket_is_non_negative_not_crashing():
    clim = build_climatology(_rows())
    p10, p50, p90 = climatology_predict(clim, [datetime(2024, 3, 23, 14, 0)])  # unseen
    assert p10[0] >= 0 and p50[0] >= 0 and p90[0] >= 0
    assert p10[0] <= p50[0] <= p90[0]
