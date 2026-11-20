"""Generate the corrected experiment and verify it by executing it.

The corrected script keeps the author's own data preparation (everything before
the first leaky step) and replaces the evaluation protocol with a leak-free one.
It is then executed in the sandbox; the repair is marked Verified only if its
output matches the replay engine's corrected number for the same split seed.
"""
from __future__ import annotations

import ast
import difflib
import importlib
import re

from . import replay as R
from .ingest import Source
from .spec import PROBA_METRICS, Spec


def _stmt_start(tree: ast.Module, line: int) -> int | None:
    for node in tree.body:
        end = getattr(node, "end_lineno", node.lineno)
        if node.lineno <= line <= end:
            return node.lineno
    return None


def _call_args_at(tree: ast.Module, line: int, attr_names: tuple[str, ...]) -> list[str] | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node, "lineno", -1) <= line <= getattr(node, "end_lineno", -1):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else None
            if name in attr_names:
                return [ast.unparse(a) for a in node.args]
    return None


def _df_var(tree: ast.Module) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            f = node.value.func
            if isinstance(f, ast.Attribute) and f.attr == "read_csv" and isinstance(node.targets[0], ast.Name):
                return node.targets[0].id
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Call):
                inner = f.value.func
                if isinstance(inner, ast.Attribute) and inner.attr == "read_csv" and isinstance(node.targets[0], ast.Name):
                    return node.targets[0].id
    return None


def _non_default(module: str, cls: str, params: dict) -> dict:
    from .spec import literal_params
    params = literal_params(params)
    try:
        defaults = getattr(importlib.import_module(module), cls)().get_params(deep=False)
    except Exception:  # noqa: BLE001
        return dict(params)
    return {k: v for k, v in params.items() if defaults.get(k, object()) != v}


_EXTRA_IMPORTS: set = set()


def _lit(v) -> str:
    if callable(v) and hasattr(v, "__qualname__"):
        public = v.__module__.split("._")[0]
        _EXTRA_IMPORTS.add(f"from {public} import {v.__qualname__}")
        return v.__qualname__
    return repr(v)


def _ctor(step_cls: str, params: dict) -> str:
    args = ", ".join(f"{k}={_lit(v)}" for k, v in params.items())
    return f"{step_cls}({args})"


def generate(src: Source, spec: Spec, fixes: list[str], proxy_cols: list[str], run_fixes_seed: int) -> dict:
    try:
        tree = ast.parse(src.code)
    except SyntaxError as e:
        return {"ok": False, "reason": f"Could not parse the code: {e}"}
    split_line = spec.split.get("line")
    first_lines = [s.line for s in spec.pre_steps if s.line] + [split_line]
    cut_line = min(l for l in first_lines if l)
    cut = _stmt_start(tree, cut_line)
    if cut is None:
        return {"ok": False, "reason": "The leaky step is inside a function or loop; automatic repair is not supported."}
    # variable names: design matrix and target
    design_line = spec.pre_steps[0].line if spec.pre_steps else split_line
    if spec.pre_steps:
        args = _call_args_at(tree, design_line, ("fit_transform", "fit", "fit_resample"))
        x_var = args[0] if args else None
        if args and len(args) > 1:
            y_var = args[1]          # e.g. SMOTE().fit_resample(X, y): the labels *before* resampling
        else:
            y_args = _call_args_at(tree, split_line, ("train_test_split",)) or _call_args_at(tree, split_line, ("split",))
            y_var = y_args[1] if y_args and len(y_args) > 1 else None
    else:
        args = _call_args_at(tree, split_line, ("train_test_split",)) or _call_args_at(tree, split_line, ("split",))
        x_var = args[0] if args else None
        y_var = args[1] if args and len(args) > 1 else None
    if not x_var or not y_var or not re.fullmatch(r"[A-Za-z_][\w.\[\]\"']*", x_var):
        return {"ok": False, "reason": "Could not identify the feature matrix and target variables."}
    df_var = _df_var(tree)

    _EXTRA_IMPORTS.clear()
    imports = {"import numpy as np", "import pandas as pd"}
    full = set(fixes)
    lines = ["", "# ---- LeakVerity corrected protocol ------------------------------------",
             "# Generated from the original notebook. Every change below is marked with FIX.", ""]
    lines.append(f"X_lv, y_lv = {x_var}, {y_var}")
    if R.FIX_PROXY in full and proxy_cols:
        lines.append(f"X_lv = X_lv.drop(columns={list(proxy_cols)!r})  # FIX: not available at prediction time")
    if R.FIX_PROPOSED in full and spec.proposed_cols:
        lines.append(f"X_lv = X_lv.drop(columns={list(spec.proposed_cols)!r})  # PROPOSED FIX: suspected leak, confirm before relying on it")
    if R.FIX_DEDUP in full:
        if spec.record_key is not None and df_var:
            lines.append(f"dup = {df_var}.loc[X_lv.index].duplicated().to_numpy()  # same source record (all columns, IDs included)")
        else:
            lines.append("dup = pd.util.hash_pandas_object(pd.concat([X_lv, pd.Series(np.asarray(y_lv), index=X_lv.index, name='__y__')], axis=1), index=False).duplicated().to_numpy()")
        lines.append("X_lv, y_lv = X_lv[~dup], y_lv[~dup]  # FIX: drop records exported more than once")
    deferred, kept_pre = [], []
    for s in spec.pre_steps:
        if (s.kind == "sampler" and R.FIX_SAMPLER in full) or (s.kind != "sampler" and R.FIX_PREPROC in full):
            deferred.append(s)
        else:
            kept_pre.append(s)
    for s in kept_pre:
        imports.add(f"from {s.module.split('._')[0]} import {s.cls}")
        p = _non_default(s.module, s.cls, s.params)
        if s.kind == "sampler":
            lines.append(f"X_lv, y_lv = {_ctor(s.cls, p)}.fit_resample(X_lv, y_lv)")
        else:
            lines.append(f"X_lv = {_ctor(s.cls, p)}.fit_transform(X_lv)")
    lines.append("y_arr = np.asarray(y_lv)")
    via = spec.split.get("via", "train_test_split")
    ts = spec.split.get("test_size")
    tsz = spec.split.get("train_size")
    size_args = (f"test_size={ts!r}" if ts is not None else "") + (f", train_size={tsz!r}" if tsz is not None else "")
    size_args = size_args.strip(", ")
    seed = run_fixes_seed
    if (R.FIX_GROUP in full and spec.groups is not None) or via == "GroupShuffleSplit":
        imports.add("from sklearn.model_selection import GroupShuffleSplit")
        if via == "GroupShuffleSplit":
            gexpr = _groups_expr(tree, split_line)
            if gexpr is None:
                return {"ok": False, "reason": "Could not read the groups passed to GroupShuffleSplit."}
            lines.append(f"groups = pd.Series(np.asarray({gexpr}), index={x_var}.index).loc[X_lv.index]")
        else:
            if df_var is None:
                return {"ok": False, "reason": "Could not find the data frame that holds the group column."}
            lines.append(f"groups = {df_var}.loc[X_lv.index, {spec.group_col!r}]  # FIX: keep each {spec.group_col} on one side of the split")
        lines.append(f"gss = GroupShuffleSplit(n_splits=1, {size_args or 'test_size=0.25'}, random_state={seed!r})")
        lines.append("train_idx, test_idx = next(gss.split(X_lv, y_arr, np.asarray(groups)))")
    else:
        imports.add("from sklearn.model_selection import train_test_split")
        strat = ", stratify=y_arr" if spec.split.get("stratified") else ""
        lines.append("pos = np.arange(len(y_arr))")
        lines.append(f"_, _, _, _, train_idx, test_idx = train_test_split(X_lv, y_arr, pos, {size_args + ', ' if size_args else ''}random_state={seed!r}{strat})")
    lines.append("take = lambda A, i: A.iloc[i] if hasattr(A, 'iloc') else A[i]")
    lines.append("X_train, X_test, y_train, y_test = take(X_lv, train_idx), take(X_lv, test_idx), y_arr[train_idx], y_arr[test_idx]")

    pipe_steps = deferred + spec.post_steps
    for s in pipe_steps:
        imports.add(f"from {s.module.split('._')[0]} import {s.cls}")
    imports.add(f"from {spec.model.module.split('._')[0]} import {spec.model.cls}")
    imports.add(f"from sklearn.metrics import {spec.metric}")
    if any(s.kind == "sampler" for s in pipe_steps):
        imports.add("from imblearn.pipeline import make_pipeline")
    elif pipe_steps:
        imports.add("from sklearn.pipeline import make_pipeline")

    def model_expr(params_expr: str) -> str:
        m = f"{spec.model.cls}(**{params_expr})"
        if not pipe_steps:
            return m
        parts = [_ctor(s.cls, _non_default(s.module, s.cls, s.params)) for s in pipe_steps]
        return f"make_pipeline({', '.join(parts)}, {m})"

    cands = [_non_default(spec.model.module, spec.model.cls, c) for c in spec.candidates]
    lines.append(f"CANDIDATES = {cands!r}")
    score = ("model.predict_proba(X_test)[:, 1]" if spec.metric in PROBA_METRICS else "model.predict(X_test)")
    fix_note = {"seed": "  # FIX: report the average over seeds, not the best test score",
                "hyper": "  # FIX: choose hyper-parameters by cross-validation on the training data",
                "none": ""}[spec.selection if R.FIX_SELECT in full else "none"]
    if pipe_steps and any(s in deferred for s in pipe_steps):
        lines.append("# FIX: preprocessing now lives inside the pipeline, so it is fitted on training rows only")
    if spec.selection == "hyper" and R.FIX_SELECT in full:
        imports.add("from sklearn.model_selection import StratifiedKFold")
        lines += [f"skf = StratifiedKFold(n_splits=3, shuffle=True, random_state={seed!r})", "cv = []",
                  "for params in CANDIDATES:",
                  "    fold = []",
                  "    for a, b in skf.split(np.zeros(len(y_train)), y_train):",
                  f"        model = {model_expr('params')}",
                  "        model.fit(take(X_train, a), y_train[a])",
                  f"        fold.append({spec.metric}(y_train[b], {score.replace('X_test', 'take(X_train, b)')}))",
                  "    cv.append(np.mean(fold))",
                  f"model = {model_expr('CANDIDATES[int(np.argmax(cv))]')}" + fix_note,
                  "model.fit(X_train, y_train)",
                  f"result = {spec.metric}(y_test, {score})"]
    else:
        lines += ["scores = []", "for params in CANDIDATES:" + fix_note,
                  f"    model = {model_expr('params')}", "    model.fit(X_train, y_train)",
                  f"    scores.append({spec.metric}(y_test, {score}))",
                  "result = float(np.mean(scores))" if (spec.selection != "none" and R.FIX_SELECT in full)
                  else ("result = max(scores)" if spec.selection != "none" else "result = scores[0]")]
    lines.append('print(f"LeakVerity corrected result: {result:.6f}")')

    original = src.code_lines()[:cut - 1]
    while original and not original[-1].strip():
        original.pop()
    code = "\n".join(sorted(imports | _EXTRA_IMPORTS)) + "\n\n" + "\n".join(original) + "\n" + "\n".join(lines) + "\n"
    diff = "".join(difflib.unified_diff([l + "\n" for l in src.code_lines()], [l + "\n" for l in code.split("\n")],
                                        "original", "corrected", n=2))
    return {"ok": True, "code": code, "diff": diff, "cut_line": cut, "kept_lines": cut - 1}


def _groups_expr(tree: ast.Module, line: int) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node, "lineno", -1) <= line <= getattr(node, "end_lineno", -1):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "split":
                for kw in node.keywords:
                    if kw.arg == "groups":
                        return ast.unparse(kw.value)
                if len(node.args) >= 3:
                    return ast.unparse(node.args[2])
    return None


def parse_result(stdout: str) -> float | None:
    m = re.search(r"LeakVerity corrected result: (-?\d+\.\d+)", stdout)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Structural verification of the corrected run (independent of the number match)
# ---------------------------------------------------------------------------

def verify_structure(vr, spec: Spec, fixes: list[str], proxy_cols: list[str], raw) -> list[dict]:
    """Check, from the corrected script's own execution trace, that every targeted leak is gone."""
    import pandas as pd
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    sp = vr.split()
    if sp is None:
        add("A train/test split was observed", False, "The corrected run did not perform a recognisable split.")
        return checks
    n_test = len(sp["test_pos"])
    fits = vr.fits()
    if R.FIX_PREPROC in fixes or R.FIX_SAMPLER in fixes:
        pre = [e for e in fits if e["seq"] < sp["seq"] and e["kind"] in ("transformer", "selector", "sampler")
               and (e.get("n_rows") or 0) >= 0.5 * (sp["n"] or 1)]
        add("Nothing is fitted on the full data before the split", not pre,
            "No preprocessing or resampling ran before the split." if not pre else
            f"{len(pre)} step(s) still fitted before the split: {', '.join(e['class'] for e in pre)}.")
    twins = sp.get("test_rows_duplicated_in_train") or 0
    post = [e for e in fits if e["seq"] > sp["seq"]]
    touching = [e for e in post if (e.get("rows_in_test") or 0) > twins]
    add("No step after the split is fitted on test rows", not touching,
        f"All {len(post)} fits after the split used training rows only." if not touching else
        f"{len(touching)} fit(s) saw test rows: {', '.join(e['class'] for e in touching)}.")

    idx = sp.get("input_index")
    rec = grp = None
    if raw is not None and idx and len(idx) == sp["n"]:
        try:
            ridx = pd.Index([int(i) for i in idx])
            rec = pd.util.hash_pandas_object(raw.loc[ridx], index=False).to_numpy()
            if spec.group_col and spec.group_col in raw.columns:
                grp = raw.loc[ridx, spec.group_col].to_numpy()
        except (ValueError, KeyError):
            rec = None
    if R.FIX_DEDUP in fixes:
        if rec is not None:
            shared = set(rec[sp["train_pos"]].tolist()) & set(rec[sp["test_pos"]].tolist())
            add("No source record appears in both train and test", not shared,
                "Every test record is distinct from every training record." if not shared else
                f"{len(shared)} records still appear on both sides.")
        else:
            add("No source record appears in both train and test", twins == 0,
                "Checked on feature rows (source records not traceable): " + (f"{twins} identical rows remain." if twins else "none remain."))
    if R.FIX_GROUP in fixes:
        if grp is not None:
            shared = set(grp[sp["train_pos"]].tolist()) & set(grp[sp["test_pos"]].tolist())
            add(f"No {spec.group_col} appears in both train and test", not shared,
                f"Train and test share no {spec.group_col}." if not shared else f"{len(shared)} shared values remain.")
        else:
            add(f"No {spec.group_col} appears in both train and test", sp.get("groups_provided", False),
                "The split was group-aware (GroupShuffleSplit with groups).")
    removed = (list(proxy_cols) if R.FIX_PROXY in fixes else []) + (list(spec.proposed_cols) if R.FIX_PROPOSED in fixes else [])
    proxy_cols = removed
    if removed:
        model_fits = [e for e in post if e["kind"] in ("model", "pipeline")]
        leaked = [e for e in model_fits if e.get("columns") and set(proxy_cols) & set(e["columns"])]
        width_ok = all((e.get("n_cols") or 0) == len(spec.features) - len(proxy_cols) for e in model_fits if not e.get("columns"))
        add("Removed features (confirmed and proposed) are not used", not leaked and width_ok,
            f"{', '.join(proxy_cols)} absent from all {len(model_fits)} model fits." if not leaked and width_ok
            else "A removed feature is still passed to the model.")
    if R.FIX_SELECT in fixes and spec.selection != "none":
        test_hash = sp.get("y_hash_test")
        evals = [m for m in vr.metrics() if m.get("y_hash") == test_hash]
        if spec.selection == "hyper":
            add("The test set is used once, after model selection", len(evals) == 1,
                f"{len(evals)} evaluation(s) on the test labels; selection used cross-validation on training data.")
        else:
            add("All candidate seeds are reported, not the best", len(evals) == len(spec.candidates),
                f"{len(evals)} test evaluations averaged; none selected by test score.")
    return checks
