"""Verified counterfactual replay.

1. Fidelity: replay the protocol exactly as written and require it to reproduce the
   notebook's own reported number. If it does not, LeakVerity abstains.
2. Counterfactuals: re-run the experiment with every combination of the proposed
   fixes over several random splits.
3. Attribution: exact Shapley shares of the total inflation per fix, with
   Nadeau–Bengio-corrected confidence intervals across splits.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, train_test_split

from . import stats as ST
from .spec import PROBA_METRICS, Spec, Unsupported

FIX_PREPROC, FIX_SAMPLER, FIX_DEDUP, FIX_GROUP, FIX_PROXY, FIX_SELECT = (
    "preprocessing", "resampling", "duplicates", "group_overlap", "future_features", "test_selection")
FIX_PROPOSED = "proposed_features"     # suspected, not confirmed: evaluated both ways

FIX_LABELS = {
    FIX_PREPROC: "Preprocessing fitted before the split",
    FIX_SAMPLER: "Resampling before the split",
    FIX_DEDUP: "Duplicate rows across train and test",
    FIX_GROUP: "Same patient in train and test",
    FIX_PROXY: "Feature unavailable at prediction time",
    FIX_SELECT: "Model chosen using the test set",
    FIX_PROPOSED: "Proposed leak (suspected, not confirmed)",
}


@dataclass
class RunOut:
    values: list[float]                 # metric for each model candidate
    n_train: int
    n_test: int
    test_index: list[str] = field(default_factory=list)
    y_test: np.ndarray | None = None
    proba: np.ndarray | None = None     # predictions of the first candidate
    probas: list = field(default_factory=list)   # predictions of every candidate (when keep_preds)
    inner: list | None = None                    # inner-CV scores (hyper-parameter selection)

    def honest_index(self, selection: str) -> int:
        """Index of the candidate the corrected protocol would report on this split."""
        if selection == "hyper" and self.inner:
            return int(np.argmax(self.inner))
        return 0


def _metric(name):
    import sklearn.metrics as skm
    return getattr(skm, name)


def _take(X, pos):
    return X.iloc[pos] if isinstance(X, (pd.DataFrame, pd.Series)) else X[pos]


def _scores(model, X, metric):
    if metric in PROBA_METRICS:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return model.decision_function(X)
    return model.predict(X)


def run_once(spec: Spec, fixes: frozenset, seed: int, keep_preds: bool = False,
             proxy_cols: list[str] = (), candidates: list[dict] | None = None,
             split_override=None) -> RunOut:
    X = spec.design.copy()
    y = spec.y.copy()
    if FIX_PROXY in fixes and proxy_cols:
        X = X.drop(columns=[c for c in proxy_cols if c in X.columns])
    if FIX_PROPOSED in fixes and spec.proposed_cols:
        X = X.drop(columns=[c for c in spec.proposed_cols if c in X.columns])
    if FIX_DEDUP in fixes:
        if spec.record_key is not None:
            # a duplicate is the *same source record* (every column identical, IDs included),
            # never two different people who happen to share feature values
            keep = ~spec.record_key.loc[X.index].duplicated().to_numpy()
        else:
            h = pd.util.hash_pandas_object(pd.concat([X, y.rename("__y__")], axis=1), index=False)
            keep = ~h.duplicated().to_numpy()
        X, y = X[keep], y[keep]
    groups = spec.groups.loc[X.index] if (spec.groups is not None and FIX_GROUP in fixes) else None

    deferred = []
    Xc, yc = X, y
    synthetic = 0
    for s in spec.pre_steps:
        defer = (s.kind == "sampler" and FIX_SAMPLER in fixes) or (s.kind != "sampler" and FIX_PREPROC in fixes)
        if defer:
            deferred.append(s)
            continue
        est = s.build()
        if s.kind == "sampler":
            n0 = len(Xc)
            Xc, yc = est.fit_resample(Xc, yc)
            synthetic = len(Xc) - n0
        else:
            Xc = est.fit_transform(Xc, yc) if s.kind == "selector" else est.fit_transform(Xc)
    yc = np.asarray(yc)
    n = len(yc)
    test_size = spec.split.get("test_size")
    if test_size is None and spec.split.get("train_size") is None:
        test_size = {"ShuffleSplit": 0.1, "StratifiedShuffleSplit": 0.1, "GroupShuffleSplit": 0.2}.get(
            spec.split.get("via", "train_test_split"), 0.25)
    via = spec.split.get("via", "train_test_split")
    if via == "GroupShuffleSplit" and spec.groups is not None and groups is None:
        groups = spec.groups.loc[X.index]
    if split_override is not None:
        tr, te = split_override(X, yc)
    elif via in ("ShuffleSplit", "StratifiedShuffleSplit") and groups is None:
        from sklearn import model_selection as _ms
        sp = getattr(_ms, via)(n_splits=1, test_size=test_size, train_size=spec.split.get("train_size"), random_state=seed)
        tr, te = next(sp.split(np.zeros(n), yc))
    elif groups is not None:
        g = np.asarray(groups)
        if synthetic:
            g = np.concatenate([g, np.array([f"__syn{i}" for i in range(synthetic)], dtype=object)])
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, train_size=spec.split.get("train_size"), random_state=seed)
        tr, te = next(gss.split(np.zeros(n), yc, g))
    else:
        tr, te = train_test_split(np.arange(n), test_size=test_size, train_size=spec.split.get("train_size"),
                                  random_state=seed, shuffle=spec.split.get("shuffle", True),
                                  stratify=yc if spec.split.get("stratified") else None)
    Xtr, Xte, ytr, yte = _take(Xc, tr), _take(Xc, te), yc[tr], yc[te]
    metric = _metric(spec.metric)
    post = deferred + spec.post_steps
    from .spec import Step

    def fit_eval(Xa, ya, Xb, yb, params):
        for s in post:
            est = s.build()
            if s.kind == "sampler":
                Xa, ya = est.fit_resample(Xa, ya)
            else:
                Xa = est.fit_transform(Xa, ya) if s.kind == "selector" else est.fit_transform(Xa)
                Xb = est.transform(Xb)
        m = Step(spec.model.cls, spec.model.module, params, "model").build()
        m.fit(Xa, ya)
        sc = _scores(m, Xb, spec.metric)
        return float(metric(yb, sc)), sc

    cands = candidates if candidates is not None else spec.candidates
    vals, proba, probas = [], None, []
    for i, params in enumerate(cands):
        v, sc = fit_eval(Xtr, ytr, Xte, yte, params)
        vals.append(v)
        if keep_preds:
            probas.append(sc)
            if i == 0:
                proba = sc
    inner = None
    if spec.selection == "hyper" and len(cands) > 1:
        inner = []
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for params in cands:
            sc_ = [fit_eval(_take(Xtr, a), ytr[a], _take(Xtr, b), ytr[b], params)[0] for a, b in skf.split(np.zeros(len(ytr)), ytr)]
            inner.append(float(np.mean(sc_)))
    test_index = []
    if keep_preds:
        idx = np.asarray(X.index)
        test_index = [str(idx[i]) if i < len(idx) else f"__syn{i}" for i in te]
    out = RunOut(vals, len(tr), len(te), test_index, yte if keep_preds else None, proba, probas, inner)
    return out


def select_honest(spec: Spec, vals: list[float], inner: list[float] | None = None) -> float:
    if spec.selection == "seed":
        return float(np.mean(vals))
    if spec.selection == "hyper" and inner is not None:
        return float(vals[int(np.argmax(inner))])
    return float(vals[0]) if len(vals) == 1 else float(np.mean(vals))


def select_as_written(spec: Spec, vals: list[float]) -> float:
    return float(max(vals)) if spec.selection != "none" else float(vals[0])


MATERIAL = 0.005     # fair-share CI lower bound above this = material inflation


def classify(m: float, lo: float, hi: float) -> str:
    """Impact class from the 95% CI of a fix's effect.

    material    the whole CI is above the materiality threshold
    real        the CI excludes zero but could be either below or above the threshold
    small       the CI excludes zero and lies entirely below the threshold
    negligible  the whole CI lies inside ±threshold
    uncertain   anything else (the data cannot tell)
    """
    if lo > MATERIAL:
        return "material"
    if lo > 0:
        return "small" if hi <= MATERIAL else "real"
    if lo > -MATERIAL and hi < MATERIAL:
        return "negligible"
    return "uncertain"


# fixes that change which rows are in the test set: their effect is a change of
# evaluation protocol, not an isolated causal effect measured on the same rows
CHANGES_TEST_POPULATION = {"duplicates", "group_overlap"}


@dataclass
class ReplayResult:
    fidelity_ok: bool
    fidelity_value: float
    reported: float
    fixes: list[str]
    seeds: list[int]
    table: dict                      # (fixset key) -> list of per-seed (as_written, honest) values
    attribution: dict                # fix -> {mean, lo, hi, label, status}
    original: dict                   # as-written over seeds
    corrected: dict                  # fully corrected over seeds
    inflation: dict
    reference: dict                  # per-step waterfall from the reported number
    seconds: float
    runs: int


def _key(s) -> str:
    return "+".join(sorted(s)) or "none"


def replay(spec: Spec, fixes: list[str], proxy_cols: list[str], n_seeds: int = 3,
           progress=None, tol: float = 1e-3) -> ReplayResult:
    t0 = time.time()
    rs = spec.split.get("random_state")
    if not isinstance(rs, int):
        rs = 0
    # ---- fidelity --------------------------------------------------------
    ref = run_once(spec, frozenset(), rs, proxy_cols=proxy_cols)
    fid_val = select_as_written(spec, ref.values)
    fidelity_ok = abs(fid_val - spec.reported) <= tol
    if progress:
        progress("fidelity", f"Replayed protocol reproduces {fid_val:.4f} (notebook reported {spec.reported:.4f})")
    if not fidelity_ok:
        return ReplayResult(False, fid_val, spec.reported, fixes, [rs], {}, {}, {}, {}, {}, {}, time.time() - t0, 1)

    seeds = [rs] + [1000 + i for i in range(n_seeds - 1)]
    run_fixes = [f for f in fixes if f != FIX_SELECT]
    has_sel = FIX_SELECT in fixes
    subsets = [frozenset(c) for r in range(len(run_fixes) + 1) for c in itertools.combinations(run_fixes, r)]
    raw_runs: dict[tuple, RunOut] = {(frozenset(), rs): ref}
    total = len(subsets) * len(seeds)
    done = 0
    for S in subsets:
        for sd in seeds:
            if (S, sd) not in raw_runs:
                raw_runs[(S, sd)] = run_once(spec, S, sd, proxy_cols=proxy_cols)
            done += 1
            if progress:
                progress("replay", f"Counterfactual run {done}/{total}: fixes = {_key(S)}, split seed {sd}", done / total)

    # value of every subset of ALL fixes (selection fix derived from the same runs)
    def value(S: frozenset, sd: int) -> float:
        base = S - {FIX_SELECT}
        vals = raw_runs[(base, sd)].values
        r = raw_runs[(base, sd)]
        return select_honest(spec, vals, getattr(r, 'inner', None)) if FIX_SELECT in S else select_as_written(spec, vals)

    all_subsets = [frozenset(c) for r in range(len(fixes) + 1) for c in itertools.combinations(fixes, r)]
    table = {_key(S): [value(S, sd) for sd in seeds] for S in all_subsets}
    full = frozenset(fixes)
    n_tr = raw_runs[(frozenset(run_fixes), rs)].n_train
    n_te = raw_runs[(frozenset(run_fixes), rs)].n_test

    phis = {f: [] for f in fixes}
    for sd in seeds:
        vals = {S: value(S, sd) for S in all_subsets}
        ph = ST.shapley(vals, fixes) if fixes else {}
        for f in fixes:
            phis[f].append(ph[f])
    attribution = {}
    for f in fixes:
        alone = [value(frozenset(), sd) - value(frozenset({f}), sd) for sd in seeds]
        last = [value(full - {f}, sd) - value(full, sd) for sd in seeds]
        m, lo, hi, sd_ = ST.nb_interval(phis[f], n_tr, n_te)
        if len(seeds) < 2:
            lo = hi = m
        status = classify(m, lo, hi)
        la_m, la_lo, la_hi, _ = ST.nb_interval(last, n_tr, n_te)
        if len(seeds) < 2:
            la_lo = la_hi = la_m
        if status != "material" and la_lo > MATERIAL:
            status = "material"          # matters once the other leaks are removed (masked interaction)
        attribution[f] = {"label": FIX_LABELS[f], "mean": m, "lo": lo, "hi": hi, "per_seed": phis[f], "status": status,
                          "changes_test_population": f in CHANGES_TEST_POPULATION,
                          "alone": float(np.mean(alone)), "last": la_m, "last_lo": la_lo, "last_hi": la_hi}

    orig_vals = table["none"]
    corr_vals = table[_key(full)]
    bounded = spec.metric != "log_loss"
    om, olo, ohi, osd = ST.nb_interval(orig_vals, n_tr, n_te, bounded=bounded)
    cm, clo, chi, csd = ST.nb_interval(corr_vals, n_tr, n_te, bounded=bounded)
    diffs = [a - b for a, b in zip(orig_vals, corr_vals)]
    dm, dlo, dhi, _ = ST.nb_interval(diffs, n_tr, n_te)

    # scenario B: the proposed leaks turn out to be legitimate features
    legit = None
    if FIX_PROPOSED in fixes:
        lv = table[_key(full - {FIX_PROPOSED})]
        lm, llo, lhi, lsd = ST.nb_interval(lv, n_tr, n_te, bounded=bounded)
        ld = [a - b for a, b in zip(orig_vals, lv)]
        ldm, ldlo, ldhi, _ = ST.nb_interval(ld, n_tr, n_te)
        legit = {"corrected": {"mean": lm, "lo": llo, "hi": lhi, "sd": lsd, "values": lv},
                 "inflation": {"mean": ldm, "lo": ldlo, "hi": ldhi, "lo_one_sided": ST.nb_lower_one_sided(ld, n_tr, n_te),
                               "values": ld, "from_reported": spec.reported - lm}}

    # waterfall from the single reported number to the verified estimate
    luck = spec.reported - om
    waterfall = [{"label": "Reported in notebook", "value": spec.reported}]
    waterfall.append({"label": "Favourable split (luck of one seed)", "delta": -luck})
    for f in sorted(fixes, key=lambda f: -attribution[f]["mean"]):
        waterfall.append({"label": FIX_LABELS[f], "delta": -attribution[f]["mean"], "fix": f})
    waterfall.append({"label": "Verified estimate", "value": cm})

    return ReplayResult(
        True, fid_val, spec.reported, fixes, seeds, table, attribution,
        {"mean": om, "lo": olo, "hi": ohi, "sd": osd, "values": orig_vals},
        {"mean": cm, "lo": clo, "hi": chi, "sd": csd, "values": corr_vals},
        {"mean": dm, "lo": dlo, "hi": dhi, "lo_one_sided": ST.nb_lower_one_sided(diffs, n_tr, n_te),
         "values": diffs, "from_reported": spec.reported - cm},
        {"waterfall": waterfall, "n_train": n_tr, "n_test": n_te, "legit_scenario": legit},
        time.time() - t0, len(raw_runs))
