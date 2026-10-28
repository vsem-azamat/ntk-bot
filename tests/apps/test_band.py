import numpy as np

from apps.band import COLD, MIN_RESIDUALS, ResidualBand, horizon_bucket, relative_error


def test_horizon_buckets():
    assert horizon_bucket(None) == COLD
    assert horizon_bucket(30) == 1
    assert horizon_bucket(60) == 1
    assert horizon_bucket(120) == 2
    assert horizon_bucket(600) == 3


def test_band_reproduces_the_residual_quantiles():
    rng = np.random.default_rng(0)
    errors = rng.normal(0.0, 0.2, 5000)
    band = ResidualBand.fit({COLD: errors.tolist()})
    p50 = np.full(5000, 500.0)
    actual = (p50 + 30) * np.exp(errors) - 30
    lo, hi = band.apply(p50, np.full(5000, COLD))
    inside = np.mean((actual >= lo) & (actual <= hi))
    assert abs(inside - 0.8) < 0.01
    assert np.allclose(relative_error(actual, p50), errors)


def test_band_scales_with_level_and_is_monotone():
    band = ResidualBand.fit({COLD: [-0.2, 0.0, 0.2] * MIN_RESIDUALS})
    lo, hi = band.apply(np.array([0.0, 100.0, 1000.0]), np.full(3, COLD))
    assert (lo <= np.array([0.0, 100.0, 1000.0])).all()
    assert hi[2] - lo[2] > hi[1] - lo[1] > 0
    assert lo.min() >= 0


def test_thin_bucket_borrows_pooled_quantiles():
    band = ResidualBand.fit({COLD: [0.5] * MIN_RESIDUALS, 1: [-0.5]})
    pooled = band.quantiles[-1]
    assert band.quantiles[1] == pooled
    lo, hi = band.apply(np.array([100.0]), np.array([3]))  # never seen: pooled
    assert hi[0] >= 100.0 >= lo[0]


def test_uncalibrated_band_collapses_onto_median(tmp_path):
    band = ResidualBand.load(tmp_path / "missing.json")
    lo, hi = band.apply(np.array([100.0]), np.array([COLD]))
    assert lo[0] == hi[0] == 100.0


def test_save_load_roundtrip(tmp_path):
    band = ResidualBand.fit({COLD: [-0.1, 0.3] * MIN_RESIDUALS})
    band.save(tmp_path / "band.json")
    assert ResidualBand.load(tmp_path / "band.json") == band
