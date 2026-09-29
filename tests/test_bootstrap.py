import numpy as np
import pytest

import trop
from trop import BootstrapResult, bootstrap


def _panel(N=12, T=15, treated_periods=4, effect=2.0, seed=0):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(N, 1)) + rng.normal(size=(1, T)) + 0.3 * rng.normal(size=(N, T)) + 10.0
    W = np.zeros((N, T))
    W[0, -treated_periods:] = 1.0
    Y = Y + effect * W
    return Y, W, treated_periods


def _did(Y, W, treated_units, treated_periods):
    return trop.DID_TWFE(Y, W, solver="CLARABEL")


def test_default_trop_estimator_matches_direct_call():
    Y, W, tp = _panel()
    lambdas = (0.5, 0.1, np.inf)
    res = bootstrap(Y, W, [0], tp, lambdas=lambdas, n_bootstrap=20, n_jobs=1, random_state=0)
    assert isinstance(res, BootstrapResult)
    direct = trop.TROP_TWFE_average(Y, W, [0], *lambdas, treated_periods=tp)
    assert res.estimate == pytest.approx(direct)
    assert res.n_bootstrap == 20 and res.n_failed == 0
    assert res.se > 0 and res.ci[0] <= res.ci[1]


def test_reproducible_and_independent_of_n_jobs():
    Y, W, tp = _panel()
    a = bootstrap(Y, W, [0], tp, _did, n_bootstrap=12, n_jobs=1, random_state=7)
    b = bootstrap(Y, W, [0], tp, _did, n_bootstrap=12, n_jobs=2, random_state=7)
    np.testing.assert_allclose(a.estimates, b.estimates)
    c = bootstrap(Y, W, [0], tp, _did, n_bootstrap=12, n_jobs=1, random_state=8)
    assert not np.allclose(a.estimates, c.estimates)


def test_treated_unit_not_first_row():
    Y, W, tp = _panel()
    perm = np.r_[3, 0, 1, 2, 4:Y.shape[0]]          # treated unit moves to row 1
    first = bootstrap(Y, W, [0], tp, _did, n_bootstrap=10, n_jobs=1, random_state=1)
    moved = bootstrap(Y[perm], W[perm], [1], tp, _did, n_bootstrap=10, n_jobs=1, random_state=1)
    assert moved.estimate == pytest.approx(first.estimate)


def test_relative_scale():
    Y, W, tp = _panel(effect=2.0)
    ab = bootstrap(Y, W, [0], tp, _did, n_bootstrap=10, n_jobs=1, random_state=0)
    rel = bootstrap(Y, W, [0], tp, _did, n_bootstrap=10, n_jobs=1, random_state=0, scale="relative")
    expected = ab.estimate / (Y[W > 0].mean() - ab.estimate)
    assert rel.estimate == pytest.approx(expected)


def test_stratified_resampling_with_several_treated_units():
    Y, W, tp = _panel(N=14)
    W[1, -tp:] = 1.0
    res = bootstrap(Y, W, [0, 1], tp, _did, n_bootstrap=15, n_jobs=1, random_state=0, resample="stratified")
    assert np.isfinite(res.se) and res.n_failed == 0


def test_failed_draws_are_counted_not_fatal():
    Y, W, tp = _panel()
    calls = {"n": 0}

    def flaky(Y_, W_, tu, tp_):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise RuntimeError("solver failed")
        return float(np.mean(Y_[W_ > 0]) - np.mean(Y_[W_ == 0]))

    with pytest.warns(RuntimeWarning, match="failed"):
        res = bootstrap(Y, W, [0], tp, flaky, n_bootstrap=12, n_jobs=1, prefer="threads", random_state=0)
    assert 0 < res.n_failed < 12
    assert np.isnan(res.estimates).sum() == res.n_failed


def test_all_draws_failing_raises():
    Y, W, tp = _panel()

    def ok_then_fail(Y_, W_, tu, tp_, _state={"first": True}):
        if _state["first"]:
            _state["first"] = False
            return 0.0
        raise RuntimeError("solver failed")

    with pytest.raises(RuntimeError, match="All bootstrap draws failed"):
        bootstrap(Y, W, [0], tp, ok_then_fail, n_bootstrap=5, n_jobs=1, prefer="threads", random_state=0)


@pytest.mark.parametrize("kwargs, msg", [
    (dict(), "Provide either estimator or lambdas"),
    (dict(lambdas=(0, 0, 1), estimator=_did), "lambdas applies only"),
    (dict(lambdas=(0, 0, 1), resample="all"), "resample"),
    (dict(lambdas=(0, 0, 1), scale="percent"), "scale"),
    (dict(lambdas=(0, 0, 1), n_bootstrap=1), "n_bootstrap"),
])
def test_invalid_arguments(kwargs, msg):
    Y, W, tp = _panel()
    with pytest.raises(ValueError, match=msg):
        bootstrap(Y, W, [0], tp, **kwargs)
