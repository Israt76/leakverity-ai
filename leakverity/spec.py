"""Reconstruct the experiment protocol from the execution trace.

The spec is what the notebook *actually did* at the scikit-learn level: which
matrix entered preprocessing, which steps were fitted before or after the split,
how the split was made, which model candidates were trained, and which metric was
reported. Everything is read from the trace, not inferred from variable names.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .sandbox import RunResult

ALLOWED_MODULE_PREFIXES = ("sklearn.", "imblearn.", "xgboost", "lightgbm")
PROBA_METRICS = {"roc_auc_score", "average_precision_score", "log_loss"}


def _decode(v):
    if isinstance(v, dict) and "__tuple__" in v:
        return tuple(v["__tuple__"])
    if isinstance(v, dict) and "__func__" in v:
        mod, name = v["__func__"].split(":")
        if not mod.startswith(("sklearn.", "numpy", "scipy.")):
            raise Unsupported(f"function parameter from {mod} is not on the replay allowlist")
        obj = importlib.import_module(mod)
        for part in name.split("."):
            obj = getattr(obj, part)
        return obj
    return v


def literal_params(params: dict) -> dict:
    """Undo the tracer's JSON encoding (tuples and allow-listed library functions)."""
    return {k: _decode(v) for k, v in params.items()}


class Unsupported(Exception):
    """The experiment uses a construct LeakVerity cannot replay faithfully."""


@dataclass
class Step:
    cls: str
    module: str
    params: dict
    kind: str
    line: int | None = None
    name: str = ""

    def build(self):
        if not self.module.startswith(ALLOWED_MODULE_PREFIXES):
            raise Unsupported(f"{self.cls} from {self.module} is not on the replay allowlist")
        bad = {k: v for k, v in self.params.items() if isinstance(v, str) and v.startswith("<") and v.endswith(">")}
        if bad:
            raise Unsupported(f"{self.cls} has non-literal parameters ({', '.join(bad)}) that cannot be replayed")
        mod = importlib.import_module(self.module)
        return getattr(mod, self.cls)(**literal_params(self.params))

    def label(self) -> str:
        return self.cls


@dataclass
class Spec:
    design: pd.DataFrame
    y: pd.Series
    raw: pd.DataFrame | None
    pre_steps: list[Step]
    post_steps: list[Step]
    model: Step
    candidates: list[dict]
    selection: str                 # none | seed | hyper
    split: dict
    metric: str
    reported: float
    reported_line: int | None
    n_test: int
    n_train: int
    groups: pd.Series | None = None
    group_col: str | None = None
    date: pd.Series | None = None
    date_col: str | None = None
    subgroup: pd.Series | None = None
    subgroup_col: str | None = None
    record_key: pd.Series | None = None      # hash of the full source record behind each design row
    proposed_cols: list = field(default_factory=list)   # features suspected (not proven) to be unavailable at prediction time
    notes: list[str] = field(default_factory=list)
    metric_events: list[dict] = field(default_factory=list)

    @property
    def features(self) -> list[str]:
        return list(self.design.columns)


def _step(ev_or_step: dict, line=None) -> Step:
    return Step(ev_or_step["class"], ev_or_step["module"], dict(ev_or_step.get("params") or {}),
                ev_or_step.get("kind", "other"), line if line is not None else ev_or_step.get("line"),
                ev_or_step.get("name", ""))


def _module_for(cls_name: str) -> str:
    from sklearn.utils import all_estimators
    for n, c in all_estimators():
        if n == cls_name:
            return c.__module__
    return "sklearn"


def build_spec(run: RunResult, raw: pd.DataFrame | None, group_col: str | None = None,
               date_col: str | None = None, subgroup_col: str | None = None,
               claimed_value: float | None = None, claimed_metric: str | None = None) -> Spec:
    if run.design is None or run.split() is None:
        raise Unsupported("No train/test split was observed (only sklearn's train_test_split hold-out protocol is replayable in this version).")
    split = run.split()
    ev = sorted(run.events, key=lambda e: e["seq"])
    design_seq = run.summary.get("design_seq", split["seq"])

    via = split.get("via", "train_test_split")
    if via not in ("train_test_split", "GroupShuffleSplit", "ShuffleSplit", "StratifiedShuffleSplit") or (split.get("n_splits") or 1) > 1:
        raise Unsupported(f"The evaluation uses {via} with {split.get('n_splits')} folds. Cross-validation loops are detected but not yet replayed; only hold-out splits are.")
    d = run.design.copy()
    idx = d.pop("__lv_index__").astype(str)
    if "__lv_target__" not in d:
        raise Unsupported("Could not capture the target labels used for the split.")
    y = d.pop("__lv_target__")
    split_groups = d.pop("__lv_groups__") if "__lv_groups__" in d else None
    d.index = idx
    y.index = idx
    if split_groups is not None:
        split_groups.index = idx
    # restore numeric column names produced from numpy arrays
    d.columns = [str(c) for c in d.columns]

    pre, post = [], []
    for e in ev:
        if e["type"] != "fit" or e["kind"] not in ("transformer", "selector", "sampler"):
            continue
        if design_seq <= e["seq"] < split["seq"]:
            pre.append(_step(e))
        elif e["seq"] > split["seq"]:
            if e.get("rows_in_test", 0) > 0 and e.get("rows_in_train", 0) > 0 and e["kind"] != "sampler":
                # fitted on data that includes the test rows after the split
                s = _step(e)
                s.name = "fit_on_full_after_split"
                pre.append(s)
            elif e.get("rows_in_test", 0) == 0:
                post.append(_step(e))

    model_fits = [e for e in ev if e["type"] == "fit" and e["seq"] > split["seq"] and e["kind"] in ("model", "pipeline")]
    if not model_fits:
        raise Unsupported("No model was trained after the split.")
    test_hash = split.get("y_hash_test")
    metrics = [e for e in ev if e["type"] == "metric" and e["seq"] > split["seq"] and e["value"] is not None
               and (test_hash is None or e["y_hash"] == test_hash)]
    if not metrics:
        raise Unsupported("No metric was computed on the held-out test labels.")

    # the reported number: closest to the claim if one was given, else the best value if the
    # notebook selected among candidates, else the last metric computed on the test set
    names = {m["name"] for m in metrics}
    if claimed_metric:
        # the verified metric must be the claimed metric
        if claimed_metric not in names:
            pretty = lambda n: n.replace("_score", "").replace("_", " ")  # noqa: E731
            raise Unsupported(f"The claim is about {pretty(claimed_metric)}, but the notebook only computes "
                              f"{', '.join(sorted(pretty(n) for n in names))} on the test set, so the claim cannot be checked.")
        metric_name = claimed_metric
    else:
        metric_name = "roc_auc_score" if "roc_auc_score" in names else metrics[-1]["name"]
    metrics = [m for m in metrics if m["name"] == metric_name]

    # model candidates trained on the training rows before each test evaluation
    candidates, model_ev = [], None
    for m in metrics:
        prior = [f for f in model_fits if f["seq"] < m["seq"]]
        if prior:
            model_ev = prior[-1]
            candidates.append(model_ev)
    def cand_key(c):
        # pipelines: compare the parameters of their steps, not the pipeline wrapper
        return repr([st["params"] for st in c.get("steps") or []] or c["params"])
    uniq, seen = [], set()
    for c in candidates:
        if cand_key(c) not in seen:
            seen.add(cand_key(c))
            uniq.append(c)
    model_ev = uniq[0] if uniq else model_fits[-1]

    if model_ev["kind"] == "pipeline":
        steps = model_ev.get("steps") or []
        if not steps:
            raise Unsupported("Pipeline steps could not be read.")
        *trans, last = steps
        post = post + [_step(s, model_ev["line"]) for s in trans]
        model = _step(last, model_ev["line"])
        cand_params = []
        for c in uniq:
            st = c.get("steps") or []
            cand_params.append(dict(st[-1]["params"]) if st else dict(last["params"]))
    else:
        model = _step(model_ev)
        cand_params = [dict(c["params"]) for c in uniq] or [dict(model_ev["params"])]

    selection = "none"
    if len(cand_params) > 1:
        varying = {k for p in cand_params for k in p if any(q.get(k) != p.get(k) for q in cand_params)}
        selection = "seed" if varying <= {"random_state", "seed"} else "hyper"

    vals = [m["value"] for m in metrics]
    if claimed_value is not None:
        rep = min(metrics, key=lambda m: abs(m["value"] - claimed_value))
    elif selection != "none":
        rep = max(metrics, key=lambda m: m["value"])
    else:
        rep = metrics[-1]

    raw_idx = None
    groups = date = subgroup = None
    if raw is not None and run.summary.get("design_has_index"):
        try:
            raw_idx = pd.Index([int(i) for i in idx])
            if not raw_idx.isin(raw.index).all():
                raw_idx = None
        except ValueError:
            raw_idx = None
    if raw_idx is None and raw is not None and len(raw) == len(d):
        raw_idx = raw.index      # positional alignment (design built from the full frame)

    def pick(col):
        if col and raw is not None and raw_idx is not None and col in raw.columns:
            s = raw.loc[raw_idx, col]
            s.index = idx
            return s
        return None

    groups, date, subgroup = pick(group_col), pick(date_col), pick(subgroup_col)
    record_key = None
    if raw is not None and raw_idx is not None:
        record_key = pd.Series(pd.util.hash_pandas_object(raw.loc[raw_idx], index=False).to_numpy().astype(str), index=idx)
    if via == "GroupShuffleSplit":
        if split_groups is None:
            raise Unsupported("GroupShuffleSplit was used but its groups could not be captured.")
        groups = split_groups
    n_test = len(split["test_pos"])
    return Spec(design=d, y=y, raw=raw, pre_steps=pre, post_steps=post, model=model, candidates=cand_params,
                selection=selection, split=split, metric=metric_name, reported=float(rep["value"]),
                reported_line=rep.get("line"), n_test=n_test, n_train=len(split["train_pos"]), groups=groups,
                group_col=group_col if groups is not None else None, date=date,
                date_col=date_col if date is not None else None, subgroup=subgroup,
                subgroup_col=subgroup_col if subgroup is not None else None, metric_events=metrics,
                record_key=record_key)
