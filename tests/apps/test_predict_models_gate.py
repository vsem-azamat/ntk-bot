from apps.forecaster import BASELINE, CATBOOST, better_than_baseline, choose_model


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


def test_incumbent_kept_on_marginal_loss():
    cat = {"mae": 101.6, "near_mae": 102.1}
    clim = {"mae": 98.4, "near_mae": 101.7}
    assert choose_model(cat, clim, CATBOOST) == CATBOOST
    assert (
        choose_model({"mae": 98.4, "near_mae": 90.0}, {"mae": 101.6, "near_mae": 99.0}, BASELINE)
        == BASELINE
    )


def test_switch_on_clear_win():
    assert (
        choose_model({"mae": 120.0, "near_mae": 99.0}, {"mae": 100.0, "near_mae": 90.0}, CATBOOST)
        == BASELINE
    )
    assert (
        choose_model({"mae": 80.0, "near_mae": 70.0}, {"mae": 100.0, "near_mae": 90.0}, BASELINE)
        == CATBOOST
    )


def test_incumbent_catboost_dropped_on_near_term_regression():
    cat = {"mae": 100.0, "near_mae": 150.0}
    clim = {"mae": 97.0, "near_mae": 90.0}
    assert choose_model(cat, clim, CATBOOST) == BASELINE
