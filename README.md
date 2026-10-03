# TROP: Triply Robust Panel Estimator

`trop` is a Python package implementing the **Triply Robust Panel (TROP)** estimator for average treatment effects in panel data. The core estimator is expressed as a weighted two-way fixed effects (TWFE) objective, with an optional low-rank regression adjustment via a nuclear-norm penalty.


Reference:

> Susan Athey, Guido Imbens, Zhaonan Qu, Davide Viviano.  
> *Triply Robust Panel Estimators*  
> _Journal of Applied Econometrics_ (2026)

---

## Links

- **Documentation**: https://zhaonanq.github.io/TROP-Estimator/
- **Source code**: https://github.com/zhaonanq/TROP-Estimator
- **PyPI page**: https://pypi.org/project/trop/

---

## Installation

```
pip install trop
```

## Quick start

```python
import numpy as np
import trop

# Y: (N, T) outcomes; unit 0 is treated in the last `treated_periods` columns
W = np.zeros_like(Y)
W[0, -treated_periods:] = 1

# 1. Tune (lambda_unit, lambda_time, lambda_nn) by placebo cross-validation on the controls
lambdas = trop.TROP_cv_cycle(
    np.delete(Y, 0, axis=0), treated_periods,
    unit_grid=np.arange(0, 2, 0.2), time_grid=np.arange(0, 2, 0.2), nn_grid=np.arange(0.005, 0.2, 0.02),
)

# 2. Point estimate
tau = trop.TROP_TWFE_average(Y, W, [0], *lambdas, treated_periods=treated_periods)

# 3. Bootstrap standard error and 95% percentile interval (lambdas held fixed)
res = trop.bootstrap(Y, W, [0], treated_periods, lambdas=lambdas, n_bootstrap=1000, random_state=0)
print(res.estimate, res.se, res.ci)
```

`trop.bootstrap` also accepts any estimator with signature
`(Y, W, treated_units, treated_periods) -> float` (e.g. the entries of
`trop.default_estimator_suite`), resamples either control units only
(`resample="controls"`, the default) or treated and control units separately
(`resample="stratified"`), and can report effects relative to the counterfactual
mean (`scale="relative"`).
