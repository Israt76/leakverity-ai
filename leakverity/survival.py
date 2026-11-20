"""Claim survival tests, run on the fully corrected protocol.

All three tests are specified in advance (fixed seeds, fixed thresholds) so that
LeakVerity cannot go fishing for a failure. A test that lacks the data it needs is
reported as Not applicable, never as a pass or a fail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import stats as ST
from .replay import run_once
from .spec import Spec

N_STABILITY = 20
STABILITY_SD_LIMIT = 0.03          # SD of the metric across re-splits (beyond normal sampling noise)
SUBGROUP_GAP = 0.05                 # subgroup metric this far below overall = concern
TEMPORAL_GAP = 0.03
MIN_CLASS = 15                      # per class per subgroup, per held-out split


def _plan(spec: Spec, i: int):
    """Which candidates to fit on re-split i, mirroring the corrected protocol.

    seed selection   -> the corrected estimate averages over seeds, so re-splits rotate through
                        the candidate seeds (each split evaluates one genuine candidate)
    hyper selection  -> every candidate is fitted and the one chosen by inner CV on the
                        training data of *that* split is evaluated
    no selection     -> the single model
    """
    if spec.selection == "seed":
        return [spec.candidates[i % len(spec.candidates)]]
    return spec.candidates


def _pick(spec: Spec, r) -> tuple[float, np.ndarray | None]:
    k = r.honest_index(spec.selection)
    return r.values[k], (r.probas[k] if r.probas else r.proba)


def selection_note(spec: Spec) -> str:
    return {"seed": f"Each re-split evaluates one of the {len(spec.candidates)} candidate seeds in turn, matching the corrected estimate's average over seeds.",
            "hyper": f"On each re-split all {len(spec.candidates)} candidates are trained and the one chosen by 3-fold CV on that split's training data is evaluated.",
            "none": "The single model from the notebook is evaluated."}[spec.selection]


def run_all(spec: Spec, fixes: list[str], proxy_cols: list[str], progress=None) -> dict:
    full = frozenset(f for f in fixes if f != "test_selection")
    runs, vals, preds = [], [], []
    for i in range(N_STABILITY):
        r = run_once(spec, full, 2000 + i, keep_preds=True, proxy_cols=proxy_cols, candidates=_plan(spec, i))
        v, p = _pick(spec, r)
        runs.append(r)
        vals.append(v)
        preds.append(p)
        if progress:
            progress("survival", f"Stability re-split {i + 1}/{N_STABILITY}", (i + 1) / N_STABILITY)
    m, lo, hi, sd = ST.nb_interval(vals, runs[0].n_train, runs[0].n_test, bounded=spec.metric != "log_loss")
    small_test = runs[0].n_test < 50
    stability = {"name": "Seed and split stability", "runs": N_STABILITY, "mean": m, "lo": lo, "hi": hi, "sd": sd,
                 "min": float(min(vals)), "max": float(max(vals)), "values": vals, "selection": selection_note(spec),
                 "status": "inconclusive" if small_test else ("pass" if sd <= STABILITY_SD_LIMIT else "concern"),
                 "rule": f"Pass if the standard deviation across {N_STABILITY} re-splits is at most {STABILITY_SD_LIMIT}."}

    # ---- subgroups -------------------------------------------------------
    if spec.subgroup is None or spec.metric != "roc_auc_score":
        subgroup = {"name": "Subgroup performance", "status": "not_applicable",
                    "reason": "No subgroup column was provided." if spec.subgroup is None else "Subgroup test currently supports AUC."}
    else:
        per = {}
        for r, pr in zip(runs, preds):
            sg = spec.subgroup.reindex(r.test_index)
            yt, pr = np.asarray(r.y_test), np.asarray(pr)
            gid = spec.groups.reindex(r.test_index) if spec.groups is not None else None
            for level in sorted(sg.dropna().unique()):
                k = (sg == level).to_numpy()
                d = per.setdefault(str(level), {"vals": [], "pos": [], "neg": [], "n": [], "ids": set()})
                d["pos"].append(int(yt[k].sum()))
                d["neg"].append(int((1 - yt[k]).sum()))
                d["n"].append(int(k.sum()))
                ids = gid[k] if gid is not None else pd.Series(np.asarray(r.test_index)[k])
                d["ids"].update(ids.dropna().astype(str).tolist())
                if yt[k].min(initial=1) != yt[k].max(initial=0):
                    d["vals"].append(roc_auc_score(yt[k], pr[k]))
        groups = []
        worst = "pass"
        for level, d in per.items():
            info = {"level": level, "n": int(round(float(np.mean(d["n"])))), "pos": int(round(float(np.mean(d["pos"])))),
                    "unique": len(d["ids"])}
            # counts are per held-out split: the 20 splits overlap, so pooled counts would overstate the evidence
            if min(np.mean(d["pos"]), np.mean(d["neg"])) < MIN_CLASS or len(d["vals"]) < 5:
                groups.append(info | {"status": "inconclusive"})
                continue
            gm, glo, ghi, _ = ST.nb_interval(d["vals"], runs[0].n_train, runs[0].n_test, bounded=True)
            status = "concern" if (gm < m - SUBGROUP_GAP and ghi < m) else "pass"
            if status == "concern":
                worst = "concern"
            groups.append(info | {"mean": gm, "lo": glo, "hi": ghi, "status": status})
        subgroup = {"name": "Subgroup performance", "column": spec.subgroup_col, "overall": m, "groups": groups,
                    "status": worst, "rule": f"Concern if a subgroup's AUC is more than {SUBGROUP_GAP} below overall and its CI excludes the overall value."}

    # ---- temporal --------------------------------------------------------
    if spec.date is None:
        temporal = {"name": "Time-based validation", "status": "not_applicable", "reason": "No date column was provided."}
    else:
        dates = pd.to_datetime(spec.date, errors="coerce")
        cutoff = dates.quantile(0.75)

        def forward_split(X, yc):
            d = dates.reindex(X.index).to_numpy()
            tr = np.where(d < np.datetime64(cutoff))[0]
            te = np.where(d >= np.datetime64(cutoff))[0]
            if spec.groups is not None:
                g = spec.groups.reindex(X.index).to_numpy()
                seen = set(g[tr].tolist())
                te = np.array([i for i in te if g[i] not in seen], dtype=int)
            return tr, te
        try:
            tr_out = run_once(spec, full, 0, proxy_cols=proxy_cols, split_override=forward_split, keep_preds=True)
            tv = float(np.mean(tr_out.values)) if spec.selection == "seed" else _pick(spec, tr_out)[0]
            yt = np.asarray(tr_out.y_test) if tr_out.y_test is not None else np.array([])
            small = tr_out.n_test < 50 or (len(yt) and min(int(yt.sum()), int((1 - yt).sum())) < MIN_CLASS)
            status = "inconclusive" if small else ("concern" if tv < m - TEMPORAL_GAP else "pass")
            temporal = {"name": "Time-based validation", "column": spec.date_col, "cutoff": str(cutoff.date()),
                        "value": tv, "random_split": m, "n_train": tr_out.n_train, "n_test": tr_out.n_test, "status": status,
                        "rule": f"Train on records before the cutoff, test on later records (new {spec.group_col or 'records'} only). "
                                f"Concern if more than {TEMPORAL_GAP} below the random-split estimate."}
        except Exception as e:  # noqa: BLE001
            temporal = {"name": "Time-based validation", "status": "not_applicable", "reason": f"Could not run: {e}"}
    return {"stability": stability, "subgroup": subgroup, "temporal": temporal,
            "limitations": ("These tests re-split the same dataset, so the 20 splits overlap and are not independent samples; "
                            "they probe robustness under the corrected protocol, not generalisation to new sites, populations or "
                            "time periods. Thresholds are fixed in advance; a pass means no problem was found, not that none exists. "
                            "Tests with too few cases are reported as inconclusive.")}
