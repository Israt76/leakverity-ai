"""Execution tracer. Imported *inside* the sandboxed subprocess only.

It records what the experiment actually did — which data was loaded, how it was
split, what was fitted on which rows, and which metrics were computed on which
labels — without modifying the user's code. Nothing here runs in the web server.
"""
from __future__ import annotations

import builtins
import functools
import hashlib
import json
import os
import sys
import traceback

NOTEBOOK_FILENAME = "<experiment>"

EVENTS: list[dict] = []
_STATE = {"seq": 0, "est_depth": 0, "metric_depth": 0, "split_depth": 0}
_CAPTURE: dict = {"fit_inputs": {}, "split_inputs": {}, "raw_frames": []}
CONFIG: dict = {}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _next_seq() -> int:
    _STATE["seq"] += 1
    return _STATE["seq"]


def _line() -> int | None:
    f = sys._getframe(2)
    while f is not None:
        if f.f_code.co_filename == NOTEBOOK_FILENAME:
            return f.f_lineno
        f = f.f_back
    return None


def _nrows(x):
    try:
        return int(x.shape[0])
    except Exception:  # noqa: BLE001
        try:
            return len(x)
        except Exception:  # noqa: BLE001
            return None


def _ncols(x):
    try:
        return int(x.shape[1]) if len(x.shape) > 1 else 1
    except Exception:  # noqa: BLE001
        return None


def _columns(x):
    cols = getattr(x, "columns", None)
    if cols is None:
        return None
    try:
        return [str(c) for c in cols]
    except Exception:  # noqa: BLE001
        return None


def _row_hashes(x):
    import numpy as np
    import pandas as pd
    try:
        if not isinstance(x, pd.DataFrame):
            arr = np.asarray(x)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            x = pd.DataFrame(arr)
        return pd.util.hash_pandas_object(x, index=False).to_numpy()
    except Exception:  # noqa: BLE001
        return None


def _label_hash(y) -> str | None:
    import numpy as np
    try:
        a = np.asarray(y).ravel()
        return hashlib.sha1(np.ascontiguousarray(a.astype("float64")).tobytes()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return None


def _safe_params(est) -> dict:
    out = {}
    try:
        for k, v in est.get_params(deep=False).items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[k] = v
            elif isinstance(v, tuple) and all(isinstance(i, (int, float, str, bool)) for i in v):
                out[k] = {"__tuple__": list(v)}          # keep tuples distinct from lists for faithful replay
            elif isinstance(v, list) and all(isinstance(i, (int, float, str, bool)) for i in v):
                out[k] = list(v)
            elif callable(v) and getattr(v, "__module__", "").startswith(("sklearn.", "numpy", "scipy.")) and hasattr(v, "__qualname__"):
                out[k] = {"__func__": f"{v.__module__}:{v.__qualname__}"}     # e.g. SelectKBest(f_classif)
            else:
                out[k] = f"<{type(v).__name__}>"
    except Exception:  # noqa: BLE001
        pass
    return out


def _kind(est) -> str:
    try:
        from sklearn.base import is_classifier, is_regressor
        name = type(est).__name__
        if name in ("Pipeline", "ImbPipeline"):
            return "pipeline"
        if name in ("GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV"):
            return "search"
        if hasattr(est, "fit_resample"):
            return "sampler"
        if is_classifier(est) or is_regressor(est):
            return "model"
        try:
            from sklearn.feature_selection import SelectorMixin
            if isinstance(est, SelectorMixin):
                return "selector"
        except Exception:  # noqa: BLE001
            pass
        if hasattr(est, "transform"):
            return "transformer"
    except Exception:  # noqa: BLE001
        pass
    return "other"


def _pipeline_steps(est):
    steps = []
    for name, s in getattr(est, "steps", []) or []:
        if s in (None, "passthrough"):
            continue
        steps.append({"name": name, "class": type(s).__name__, "kind": _kind(s), "params": _safe_params(s),
                      "module": type(s).__module__})
    return steps


# ---------------------------------------------------------------------------
# wrappers
# ---------------------------------------------------------------------------

def _wrap_method(cls, meth: str):
    orig = getattr(cls, meth, None)
    if orig is None or getattr(orig, "_lv_wrapped", False):
        return

    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        if _STATE["est_depth"] > 0:
            return orig(self, *args, **kwargs)
        X = args[0] if args else kwargs.get("X")
        y = args[1] if len(args) > 1 else kwargs.get("y")
        seq = _next_seq()
        ev = {"type": "fit", "seq": seq, "method": meth, "class": type(self).__name__,
              "module": type(self).__module__, "kind": _kind(self), "params": _safe_params(self),
              "n_rows": _nrows(X), "n_cols": _ncols(X), "columns": _columns(X), "line": _line(),
              "obj": id(self), "has_y": y is not None, "y_hash": _label_hash(y) if y is not None else None}
        if ev["kind"] in ("pipeline", "search") or type(self).__name__ == "ImbPipeline":
            ev["steps"] = _pipeline_steps(self) or _pipeline_steps(getattr(self, "estimator", None))
            inner = getattr(self, "estimator", None)
            if inner is not None and not ev["steps"]:
                ev["inner"] = {"class": type(inner).__name__, "params": _safe_params(inner), "kind": _kind(inner)}
            if ev["kind"] == "search":
                ev["param_grid"] = repr(getattr(self, "param_grid", getattr(self, "param_distributions", None)))[:500]
        hashes = _row_hashes(X) if X is not None else None
        if hashes is not None:
            ev["_hashes"] = hashes
        _CAPTURE["fit_inputs"][seq] = (X, y)
        EVENTS.append(ev)
        _STATE["est_depth"] += 1
        try:
            return orig(self, *args, **kwargs)
        finally:
            _STATE["est_depth"] -= 1

    wrapper._lv_wrapped = True
    try:
        setattr(cls, meth, wrapper)
    except (TypeError, AttributeError):
        pass


def _wrap_predict(cls, meth: str):
    orig = getattr(cls, meth, None)
    if orig is None or getattr(orig, "_lv_wrapped", False):
        return

    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        if _STATE["est_depth"] > 0:
            return orig(self, *args, **kwargs)
        X = args[0] if args else kwargs.get("X")
        hashes = _row_hashes(X)
        EVENTS.append({"type": "predict", "seq": _next_seq(), "method": meth, "class": type(self).__name__,
                       "obj": id(self), "n_rows": _nrows(X), "line": _line(),
                       **({"_hashes": hashes} if hashes is not None else {})})
        _STATE["est_depth"] += 1
        try:
            return orig(self, *args, **kwargs)
        finally:
            _STATE["est_depth"] -= 1

    wrapper._lv_wrapped = True
    try:
        setattr(cls, meth, wrapper)
    except (TypeError, AttributeError):
        pass


METRICS = ["roc_auc_score", "accuracy_score", "f1_score", "precision_score", "recall_score",
           "average_precision_score", "balanced_accuracy_score", "log_loss", "matthews_corrcoef"]


def _wrap_metric(mod, name):
    orig = getattr(mod, name, None)
    if orig is None or getattr(orig, "_lv_wrapped", False):
        return

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        val = orig(*args, **kwargs)
        if _STATE["est_depth"] == 0 and _STATE["metric_depth"] == 0:
            y_true = args[0] if args else kwargs.get("y_true")
            try:
                v = float(val)
            except Exception:  # noqa: BLE001
                v = None
            EVENTS.append({"type": "metric", "seq": _next_seq(), "name": name, "value": v,
                           "n": _nrows(y_true), "y_hash": _label_hash(y_true), "line": _line()})
        return val

    wrapper._lv_wrapped = True
    setattr(mod, name, wrapper)


def _wrap_split(ms):
    orig = ms.train_test_split
    if getattr(orig, "_lv_wrapped", False):
        return

    @functools.wraps(orig)
    def wrapper(*arrays, **kwargs):
        import numpy as np
        _STATE["split_depth"] += 1
        try:
            return _tts(*arrays, **kwargs)
        finally:
            _STATE["split_depth"] -= 1

    def _tts(*arrays, **kwargs):
        import numpy as np
        out = orig(*arrays, **kwargs)
        n = _nrows(arrays[0]) if arrays else None
        seq = _next_seq()
        ev = {"type": "split", "seq": seq, "n": n, "n_arrays": len(arrays), "line": _line(),
              "test_size": kwargs.get("test_size"), "train_size": kwargs.get("train_size"),
              "random_state": kwargs.get("random_state") if isinstance(kwargs.get("random_state"), (int, type(None))) else "<obj>",
              "shuffle": kwargs.get("shuffle", True), "stratified": kwargs.get("stratify") is not None,
              "input_columns": _columns(arrays[0]) if arrays else None}
        try:
            pos_kwargs = dict(kwargs)
            tr_pos, te_pos = orig(np.arange(n), **pos_kwargs)
            ev["train_pos"] = [int(i) for i in tr_pos]
            ev["test_pos"] = [int(i) for i in te_pos]
            h = _row_hashes(arrays[0])
            if h is not None:
                train_set = set(h[tr_pos].tolist())
                dup_test = [int(i) for i in te_pos if h[i] in train_set]
                ev["test_rows_duplicated_in_train"] = len(dup_test)
                ev["_hashes"] = h
            ev["y_hash_test"] = _label_hash(out[3]) if len(arrays) >= 2 else None
            ev["y_hash_train"] = _label_hash(out[2]) if len(arrays) >= 2 else None
            idx = getattr(arrays[0], "index", None)
            if idx is not None:
                ev["input_index"] = [str(i) for i in idx]
        except Exception as e:  # noqa: BLE001
            ev["error"] = repr(e)
        _CAPTURE["split_inputs"][seq] = arrays[:2]
        EVENTS.append(ev)
        return out

    wrapper._lv_wrapped = True
    ms.train_test_split = wrapper


def _wrap_splitter(cls):
    orig = getattr(cls, "split", None)
    if orig is None or getattr(orig, "_lv_wrapped", False):
        return

    @functools.wraps(orig)
    def wrapper(self, X, y=None, groups=None, *a, **k):
        import numpy as np
        first = True
        for tr, te in orig(self, X, y, groups, *a, **k):
            if first and _STATE["est_depth"] == 0 and _STATE["split_depth"] == 0:
                first = False
                seq = _next_seq()
                n = _nrows(X)
                ev = {"type": "split", "seq": seq, "n": n, "n_arrays": 2 if y is not None else 1, "line": _line(),
                      "via": type(self).__name__, "n_splits": getattr(self, "n_splits", None),
                      "test_size": getattr(self, "test_size", None), "train_size": getattr(self, "train_size", None),
                      "random_state": getattr(self, "random_state", None) if isinstance(getattr(self, "random_state", None), (int, type(None))) else "<obj>",
                      "shuffle": True, "stratified": "Stratified" in type(self).__name__, "groups_provided": groups is not None,
                      "train_pos": [int(i) for i in tr], "test_pos": [int(i) for i in te],
                      "input_columns": _columns(X)}
                idx = getattr(X, "index", None)
                if idx is not None:
                    ev["input_index"] = [str(i) for i in idx]
                h = _row_hashes(X)
                if h is not None:
                    train_set = set(h[tr].tolist())
                    ev["test_rows_duplicated_in_train"] = int(sum(1 for i in te if h[i] in train_set))
                    ev["_hashes"] = h
                if y is not None:
                    ya = np.asarray(y).ravel()
                    ev["y_hash_test"] = _label_hash(ya[te])
                    ev["y_hash_train"] = _label_hash(ya[tr])
                _CAPTURE["split_inputs"][seq] = (X, y)
                _CAPTURE.setdefault("split_groups", {})[seq] = groups
                EVENTS.append(ev)
            yield tr, te

    wrapper._lv_wrapped = True
    cls.split = wrapper


def _wrap_read_csv(pd):
    orig = pd.read_csv

    @functools.wraps(orig)
    def wrapper(path, *args, **kwargs):
        target = path
        data_files = CONFIG.get("data_files", [])
        if isinstance(path, (str, os.PathLike)):
            p = os.fspath(path)
            if not os.path.exists(p):
                base = os.path.basename(p)
                match = [f for f in data_files if os.path.basename(f) == base]
                if match:
                    target = match[0]
                elif len(data_files) == 1:
                    target = data_files[0]
        df = orig(target, *args, **kwargs)
        EVENTS.append({"type": "load", "seq": _next_seq(), "path": os.path.basename(os.fspath(target)) if isinstance(target, (str, os.PathLike)) else "<buffer>",
                       "requested": os.fspath(path) if isinstance(path, (str, os.PathLike)) else "<buffer>",
                       "shape": list(df.shape), "columns": [str(c) for c in df.columns], "line": _line(),
                       "index_col": kwargs.get("index_col")})
        _CAPTURE["raw_frames"].append(df)
        return df

    pd.read_csv = wrapper


def _block_network():
    import socket

    class _NoNet(socket.socket):
        def __init__(self, *a, **k):
            raise PermissionError("Network access is disabled inside the LeakVerity sandbox.")

    socket.socket = _NoNet
    socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(PermissionError("Network disabled"))


def install(config: dict):
    CONFIG.update(config)
    os.environ.setdefault("MPLBACKEND", "Agg")
    _block_network()
    import pandas as pd
    import sklearn.metrics as skm
    import sklearn.model_selection as ms
    from sklearn.utils import all_estimators
    _wrap_read_csv(pd)
    _wrap_split(ms)
    for cname in ("GroupShuffleSplit", "ShuffleSplit", "StratifiedShuffleSplit", "KFold", "StratifiedKFold",
                  "GroupKFold", "StratifiedGroupKFold", "TimeSeriesSplit"):
        if hasattr(ms, cname):
            _wrap_splitter(getattr(ms, cname))
    for m in METRICS:
        _wrap_metric(skm, m)
    classes = [c for _, c in all_estimators()]
    from sklearn.pipeline import Pipeline
    classes.append(Pipeline)
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.utils.testing import all_estimators as imb_all
        classes.append(ImbPipeline)
        classes += [c for _, c in imb_all()]
    except Exception:  # noqa: BLE001
        pass
    for mod, cname in (("xgboost", "XGBClassifier"), ("lightgbm", "LGBMClassifier")):
        try:
            classes.append(getattr(__import__(mod), cname))
        except Exception:  # noqa: BLE001
            pass
    for c in classes:
        for meth in ("fit", "fit_transform", "fit_resample"):
            _wrap_method(c, meth)
        for meth in ("predict", "predict_proba", "decision_function"):
            _wrap_predict(c, meth)
    # notebook conveniences
    builtins.display = lambda *a, **k: print(*a)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.show = lambda *a, **k: None
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------

def dump(outdir: str, error: str | None = None):
    import numpy as np
    import pandas as pd
    os.makedirs(outdir, exist_ok=True)
    splits = [e for e in EVENTS if e["type"] == "split" and "test_pos" in e]
    primary = max(splits, key=lambda e: (e["n"] or 0, -e["seq"])) if splits else None
    summary = {"primary_split_seq": primary["seq"] if primary else None, "error": error}

    if primary is not None:
        test_h = primary.get("_hashes")
        test_set = set(test_h[primary["test_pos"]].tolist()) if test_h is not None else set()
        train_set = set(test_h[primary["train_pos"]].tolist()) if test_h is not None else set()
        for e in EVENTS:
            if e["type"] in ("fit", "predict") and "_hashes" in e and e["seq"] > primary["seq"]:
                h = e["_hashes"]
                e["rows_in_test"] = int(sum(1 for v in h.tolist() if v in test_set))
                e["rows_in_train"] = int(sum(1 for v in h.tolist() if v in train_set))
        # design matrix = input of the earliest pre-split preprocessing fit (so leaky
        # preprocessing can be replayed and corrected), otherwise the split input itself
        split_arrays = _CAPTURE["split_inputs"][primary["seq"]]
        design_seq, X = primary["seq"], split_arrays[0]
        y = split_arrays[1] if len(split_arrays) > 1 else None
        for e in sorted(EVENTS, key=lambda e: e["seq"]):
            if e["type"] == "fit" and e["seq"] < primary["seq"] and e["kind"] in ("transformer", "selector", "sampler"):
                Xi, yi = _CAPTURE["fit_inputs"][e["seq"]]
                ni = _nrows(Xi) if Xi is not None else None
                if ni and primary["n"] and ni >= 0.5 * primary["n"]:
                    design_seq, X = e["seq"], Xi
                    if yi is not None and _nrows(yi) == ni:
                        y = yi
                    elif y is not None and _nrows(y) != ni:
                        y = None
                        for e2 in sorted(EVENTS, key=lambda e: e["seq"]):
                            if e2["type"] == "fit" and e2["seq"] < primary["seq"] and e2["has_y"]:
                                Xj, yj = _CAPTURE["fit_inputs"][e2["seq"]]
                                if _nrows(yj) == ni:
                                    y = yj
                                    break
                    break
        summary["design_seq"] = design_seq
        try:
            if X is not None:
                Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
                Xdf = Xdf.copy()
                Xdf.columns = [str(c) for c in Xdf.columns]
                Xdf.insert(0, "__lv_index__", [str(i) for i in Xdf.index])
                if y is not None:
                    Xdf["__lv_target__"] = np.asarray(y).ravel()
                g = _CAPTURE.get("split_groups", {}).get(primary["seq"])
                if g is not None and _nrows(g) == len(Xdf):     # row-preserving pre-steps keep positions aligned
                    Xdf["__lv_groups__"] = [str(v) for v in np.asarray(g).ravel()]
                Xdf.to_csv(os.path.join(outdir, "design.csv"), index=False)
                summary["design_columns"] = [c for c in Xdf.columns if not c.startswith("__lv_")]
                summary["design_has_index"] = isinstance(X, pd.DataFrame)
        except Exception as ex:  # noqa: BLE001
            summary["design_error"] = repr(ex)

    for e in EVENTS:
        e.pop("_hashes", None)
    with open(os.path.join(outdir, "trace.json"), "w") as f:
        json.dump({"events": EVENTS, "summary": summary}, f, default=str)


def run(code_path: str, outdir: str, config_path: str):
    with open(config_path) as f:
        config = json.load(f)
    install(config)
    with open(code_path) as f:
        code = f.read()
    error = None
    g = {"__name__": "__main__", "__file__": "experiment.py"}
    try:
        exec(compile(code, NOTEBOOK_FILENAME, "exec"), g)
    except SystemExit:
        pass
    except BaseException:  # noqa: BLE001
        tb = traceback.format_exc()
        error = tb[-4000:]
        print(tb, file=sys.stderr)
    dump(outdir, error)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])
