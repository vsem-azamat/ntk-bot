from apps.predictModels import better_than_baseline


def test_catboost_wins_when_lower_error():
    cat = {"mae": 30.0, "near_mae": 20.0}
    clim = {"mae": 40.0, "near_mae": 35.0}
    assert better_than_baseline(cat, clim) is True


def test_catboost_loses_when_worse_near_term():
    cat = {"mae": 39.0, "near_mae": 50.0}
    clim = {"mae": 40.0, "near_mae": 35.0}
    assert better_than_baseline(cat, clim) is False


def test_catboost_loses_when_worse_overall():
    cat = {"mae": 41.0, "near_mae": 10.0}
    clim = {"mae": 40.0, "near_mae": 35.0}
    assert better_than_baseline(cat, clim) is False
