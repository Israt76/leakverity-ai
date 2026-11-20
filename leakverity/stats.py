"""Statistics used for every number LeakVerity reports."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps


def nb_interval(values, n_train: int, n_test: int, level: float = 0.95, bounded: bool = False):
    """Mean and CI over repeated random splits with the Nadeau–Bengio correction.

    Repeated train/test splits share data, so the naive standard error is too small.
    Nadeau & Bengio (2003) inflate the variance by (1/J + n_test/n_train).

    bounded=True is for metrics that live in [0, 1] (AUC, accuracy, ...): the interval is
    computed on the logit scale and transformed back, so it can never leave [0, 1].
    The point estimate stays the plain mean.
    """
    v = np.asarray(values, dtype=float)
    j = len(v)
    m = float(v.mean())
    if j < 2:
        return m, float("nan"), float("nan"), 0.0
    sd = math.sqrt(float(v.var(ddof=1)))
    t = sps.t.ppf(0.5 + level / 2, j - 1)
    factor = 1.0 / j + n_test / max(n_train, 1)
    if bounded:
        eps = 0.5 / max(n_test, 1)          # continuity cap: half a case, so a perfect 1.0 is not an extreme logit
        z = np.log(np.clip(v, eps, 1 - eps) / (1 - np.clip(v, eps, 1 - eps)))
        zm = float(z.mean())
        zse = math.sqrt(factor * float(z.var(ddof=1)))
        lo = 1 / (1 + math.exp(-(zm - t * zse)))
        hi = 1 / (1 + math.exp(-(zm + t * zse)))
        return m, min(lo, m), max(hi, m), sd
    se = math.sqrt(factor * sd * sd)
    return m, m - t * se, m + t * se, sd


def bootstrap_auc(y, p, n_boot: int = 1000, seed: int = 0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], p[i]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def univariate_auc(x, y) -> float | None:
    from sklearn.metrics import roc_auc_score
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    ok = ~np.isnan(x)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2 or len(np.unique(x[ok])) < 2:
        return None
    a = roc_auc_score(y[ok], x[ok])
    return float(max(a, 1 - a))


def shapley(values: dict[frozenset, float], players: list[str]) -> dict[str, float]:
    """Exact Shapley attribution of the drop v(∅) − v(all) to each fix.

    values[S] is the metric with the fixes in S applied. A player's share is its average
    marginal drop over all orders in which fixes could be applied, so interacting leaks
    are not double-counted and the shares sum to the total change.
    """
    k = len(players)
    phi = {}
    for p in players:
        others = [q for q in players if q != p]
        total = 0.0
        for r in range(len(others) + 1):
            from itertools import combinations
            w = math.factorial(r) * math.factorial(k - r - 1) / math.factorial(k)
            for S in combinations(others, r):
                s = frozenset(S)
                total += w * (values[s] - values[s | {p}])
        phi[p] = total
    return phi


def nb_lower_one_sided(values, n_train: int, n_test: int, level: float = 0.95) -> float:
    """One-sided lower confidence bound (Nadeau–Bengio corrected) for a directional question,
    e.g. 'is the as-written result higher than the corrected one?'."""
    v = np.asarray(values, dtype=float)
    j = len(v)
    if j < 2:
        return float("nan")
    se = math.sqrt((1.0 / j + n_test / max(n_train, 1)) * float(v.var(ddof=1)))
    return float(v.mean() - sps.t.ppf(level, j - 1) * se)
