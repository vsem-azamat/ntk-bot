from apps.anchor import anchor_band


def test_no_observation_returns_input_unchanged():
    p10 = [10.0, 20.0]
    p50 = [15.0, 25.0]
    p90 = [20.0, 30.0]
    out = anchor_band(p10, p50, p90, anchor_index=None, anchor_value=None)
    assert out == (p10, p50, p90)


def test_p50_meets_anchor_at_cut_and_decays():
    p50 = [50.0, 60.0, 70.0, 80.0, 90.0]
    p10 = [40.0, 50.0, 60.0, 70.0, 80.0]
    p90 = [60.0, 70.0, 80.0, 90.0, 100.0]
    a10, a50, a90 = anchor_band(p10, p50, p90, anchor_index=1, anchor_value=100.0)
    assert a50[1] == 100.0
    assert a50[2] > 70.0
    assert (a50[4] - 90.0) < (a50[2] - 70.0)


def test_band_stays_monotone():
    p50 = [50.0, 60.0, 70.0]
    p10 = [40.0, 50.0, 60.0]
    p90 = [60.0, 70.0, 80.0]
    a10, a50, a90 = anchor_band(p10, p50, p90, anchor_index=0, anchor_value=80.0)
    for lo, mid, hi in zip(a10, a50, a90, strict=True):
        assert lo <= mid <= hi
