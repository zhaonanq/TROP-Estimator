from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
from joblib import Parallel, delayed

from .estimator import TROP_TWFE_average


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]
EstimatorFn = Callable[[np.ndarray, np.ndarray, np.ndarray, int], float]

_VALID_RESAMPLE = {"controls", "stratified"}
_VALID_SCALE = {"absolute", "relative"}


@dataclass(frozen=True)
class BootstrapResult:
    """
    Output of :func:`bootstrap`.

    Attributes
    ----------
    estimate : float
        Estimate on the original (non-resampled) panel.
    se : float
        Bootstrap standard error: standard deviation of the finite bootstrap draws.
    ci : tuple of float
        Percentile confidence interval ``(lower, upper)`` at level ``1 - alpha``.
    estimates : ndarray of shape (n_bootstrap,)
        All bootstrap draws; NaN where the estimator failed.
    n_failed : int
        Number of draws where the estimator raised or returned a non-finite value.
    alpha : float
        Significance level used for ``ci``.
    """

    estimate: float
    se: float
    ci: Tuple[float, float]
    estimates: np.ndarray
    n_failed: int
    alpha: float

    @property
    def n_bootstrap(self) -> int:
        return int(self.estimates.size)


def _to_scale(tau: float, Y: np.ndarray, W: np.ndarray, scale: str) -> float:
    """Convert an effect to the requested scale. ``relative``: tau / (mean treated outcome - tau)."""
    if scale == "absolute":
        return float(tau)
    counterfactual_mean = float(np.mean(Y[W > 0])) - tau
    return float(tau / counterfactual_mean)


def _one_draw(
    Y: np.ndarray,
    W: np.ndarray,
    treated: np.ndarray,
    control: np.ndarray,
    treated_periods: int,
    estimator: EstimatorFn,
    resample: str,
    scale: str,
    seed: np.random.SeedSequence,
) -> float:
    rng = np.random.default_rng(seed)
    control_idx = rng.choice(control, size=control.size, replace=True)
    if resample == "stratified":
        treated_idx = rng.choice(treated, size=treated.size, replace=True)
    else:
        treated_idx = treated
    rows = np.concatenate([treated_idx, control_idx])
    Y_b, W_b = Y[rows], W[rows]
    treated_b = np.arange(treated_idx.size)
    try:
        tau = float(estimator(Y_b, W_b, treated_b, treated_periods))
        value = _to_scale(tau, Y_b, W_b, scale)
    except Exception:  # a failed solve on one draw should not abort the whole bootstrap
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def bootstrap(
    Y: ArrayLike,
    W: ArrayLike,
    treated_units: Sequence[int],
    treated_periods: int,
    estimator: Optional[EstimatorFn] = None,
    *,
    lambdas: Optional[Tuple[float, float, float]] = None,
    n_bootstrap: int = 1000,
    resample: str = "controls",
    scale: str = "absolute",
    alpha: float = 0.05,
    n_jobs: int = -1,
    prefer: str = "processes",
    random_state: Optional[int] = None,
) -> BootstrapResult:
    """
    Unit-level bootstrap standard errors and confidence intervals for a panel estimator.

    Each draw resamples units (rows of ``Y``/``W``) with replacement, re-applies the
    estimator with its tuning parameters held fixed, and records the estimate. The
    standard error is the standard deviation of the draws; the confidence interval is
    the percentile interval.

    Parameters
    ----------
    Y : array_like of shape (N, T)
        Outcome matrix.
    W : array_like of shape (N, T)
        Treatment indicator matrix.
    treated_units : sequence of int
        Row indices of treated units.
    treated_periods : int
        Number of final columns treated as the post-treatment block; passed to the estimator.
    estimator : callable or None, default=None
        ``estimator(Y, W, treated_units, treated_periods) -> float``, the same signature as
        the entries of :func:`trop.default_estimator_suite`. If None, TROP with fixed
        ``lambdas`` is used.
    lambdas : tuple of float or None, default=None
        ``(lambda_unit, lambda_time, lambda_nn)`` for the default TROP estimator, typically
        selected beforehand by cross-validation on the original panel. Required when
        ``estimator`` is None; must be None otherwise.
    n_bootstrap : int, default=1000
        Number of bootstrap draws.
    resample : {"controls", "stratified"}, default="controls"
        ``"controls"``: keep the treated units fixed and resample control units with
        replacement. Suitable with a single (or few) treated unit(s).
        ``"stratified"``: resample treated and control units separately, each with
        replacement, keeping the number of each fixed.
    scale : {"absolute", "relative"}, default="absolute"
        ``"absolute"``: effects in outcome units. ``"relative"``: effects as a fraction of
        the counterfactual mean, ``tau / (mean(Y[W > 0]) - tau)``; multiply by 100 for a
        percentage change. Applied to the full-sample estimate and to every draw.
    alpha : float, default=0.05
        Significance level for the percentile confidence interval.
    n_jobs : int, default=-1
        Number of parallel jobs. ``-1`` uses all available cores.
    prefer : {"processes", "threads"}, default="processes"
        Joblib backend preference. Processes are usually much faster for the cvxpy-based
        estimators, whose problem setup holds the GIL.
    random_state : int or None, default=None
        Seed for reproducible draws. Draws are seeded individually, so results do not
        depend on ``n_jobs``.

    Returns
    -------
    BootstrapResult

    Raises
    ------
    ValueError
        If inputs are invalid.
    RuntimeError
        If every bootstrap draw fails.

    Warns
    -----
    RuntimeWarning
        If some (but not all) draws fail; they are excluded from ``se`` and ``ci``.

    Examples
    --------
    >>> lambdas = trop.TROP_cv_cycle(np.delete(Y, 0, axis=0), treated_periods,
    ...                              unit_grid, time_grid, nn_grid)
    >>> res = trop.bootstrap(Y, W, [0], treated_periods, lambdas=lambdas, random_state=0)
    >>> res.estimate, res.se, res.ci
    """
    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W, dtype=float)
    if Y.ndim != 2 or W.ndim != 2:
        raise ValueError(f"Y and W must be 2D arrays. Got Y.ndim={Y.ndim}, W.ndim={W.ndim}.")
    if Y.shape != W.shape:
        raise ValueError(f"Y and W must have the same shape. Got Y={Y.shape}, W={W.shape}.")
    N, T = Y.shape
    if not isinstance(treated_periods, (int, np.integer)) or treated_periods <= 0 or treated_periods >= T:
        raise ValueError(f"treated_periods must be an integer in [1, T-1]. Got {treated_periods}, T={T}.")
    treated_periods = int(treated_periods)

    treated = np.unique(np.asarray(treated_units, dtype=int))
    if treated.size == 0:
        raise ValueError("treated_units must contain at least one unit index.")
    if np.any(treated < 0) or np.any(treated >= N):
        raise ValueError(f"treated_units contains out-of-range indices for N={N}: {treated}")
    control = np.setdiff1d(np.arange(N), treated)
    if control.size < 2:
        raise ValueError(f"bootstrap needs at least 2 control units. Got {control.size}.")

    if resample not in _VALID_RESAMPLE:
        raise ValueError(f"resample must be one of {sorted(_VALID_RESAMPLE)}.")
    if scale not in _VALID_SCALE:
        raise ValueError(f"scale must be one of {sorted(_VALID_SCALE)}.")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2:
        raise ValueError(f"n_bootstrap must be an integer >= 2. Got {n_bootstrap}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1). Got {alpha}.")

    if estimator is None:
        if lambdas is None:
            raise ValueError("Provide either estimator or lambdas (for the default TROP estimator).")
        if len(lambdas) != 3:
            raise ValueError("lambdas must be (lambda_unit, lambda_time, lambda_nn).")
        lambda_unit, lambda_time, lambda_nn = (float(v) for v in lambdas)

        def estimator(Y_, W_, treated_units_, treated_periods_):
            return TROP_TWFE_average(Y_, W_, treated_units_, lambda_unit, lambda_time, lambda_nn,
                                     treated_periods=treated_periods_)
    elif lambdas is not None:
        raise ValueError("lambdas applies only to the default TROP estimator; pass one of estimator or lambdas.")

    # Estimate on the original panel with the same row ordering the draws use (treated first).
    rows = np.concatenate([treated, control])
    Y_o, W_o = Y[rows], W[rows]
    estimate = _to_scale(float(estimator(Y_o, W_o, np.arange(treated.size), treated_periods)), Y_o, W_o, scale)

    seeds = np.random.SeedSequence(random_state).spawn(int(n_bootstrap))
    draws = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(_one_draw)(Y, W, treated, control, treated_periods, estimator, resample, scale, seed)
        for seed in seeds
    )
    draws = np.asarray(draws, dtype=float)

    ok = draws[np.isfinite(draws)]
    n_failed = int(draws.size - ok.size)
    if ok.size == 0:
        raise RuntimeError("All bootstrap draws failed. Check the estimator and solver settings.")
    if ok.size < 2:
        raise RuntimeError(f"Only {ok.size} bootstrap draw succeeded; cannot compute a standard error.")
    if n_failed:
        warnings.warn(f"{n_failed} of {draws.size} bootstrap draws failed and were excluded.",
                      RuntimeWarning, stacklevel=2)

    se = float(np.std(ok, ddof=1))
    lower, upper = np.quantile(ok, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(estimate=float(estimate), se=se, ci=(float(lower), float(upper)),
                           estimates=draws, n_failed=n_failed, alpha=float(alpha))
