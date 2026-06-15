from datetime import datetime

from apps.features import ObsContext, asof_features


def _today():
    return [
        (datetime(2025, 3, 13, 9, 0), 100),
        (datetime(2025, 3, 13, 9, 20), 140),
        (datetime(2025, 3, 13, 9, 40), 200),
    ]


def test_no_samples_yet_returns_zero_flags():
    ctx = ObsContext.from_rows(_today())
    f = asof_features(datetime(2025, 3, 13, 8, 0), ctx)
    assert f["has_today"] == 0.0
    assert f["last_count"] == 0.0
    assert f["peak_so_far"] == 0.0


def test_uses_only_samples_at_or_before_cut():
    ctx = ObsContext.from_rows(_today())
    f = asof_features(datetime(2025, 3, 13, 9, 25), ctx)
    assert f["has_today"] == 1.0
    assert f["last_count"] == 140.0
    assert f["peak_so_far"] == 140.0
    assert f["slope"] > 0


def test_future_samples_never_leak():
    ctx = ObsContext.from_rows(_today())
    early = asof_features(datetime(2025, 3, 13, 9, 25), ctx)
    ctx2 = ObsContext.from_rows(_today() + [(datetime(2025, 3, 13, 10, 0), 999)])
    later = asof_features(datetime(2025, 3, 13, 9, 25), ctx2)
    assert early == later
