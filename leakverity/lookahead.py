"""Static detection of look-ahead features built in pandas, before scikit-learn sees the data.

These leaks are invisible to the execution trace (they are already inside the design matrix),
but they are visible in the code. We find assignments such as

    s["vib_smooth"] = s["vibration"].rolling(15, center=True).mean()      # centred window: uses future rows
    df["next_value"] = df["value"].shift(-1)                              # negative shift: the next row
    df["x"] = df["x"].bfill()                                             # backward fill: copies later values
    df["te"] = df.groupby("city")["target"].transform("mean")             # target encoding on all rows

and report the feature they create. The effect of removing it is then *measured* by the replay
engine like any other unavailable feature; nothing is assumed.
"""
from __future__ import annotations

import ast


def _const(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return -node.operand.value
    return None


def _target_col(node) -> str | None:
    """'vib_smooth' from s["vib_smooth"] = ..., or df.assign(vib_smooth=...) keyword name."""
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    return None


def _patterns(expr, target: str | None) -> list[str]:
    found = []
    for n in ast.walk(expr):
        if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Attribute):
            continue
        name = n.func.attr
        kw = {k.arg: _const(k.value) for k in n.keywords}
        if name == "rolling" and kw.get("center") is True:
            w = _const(n.args[0]) if n.args else kw.get("window")
            found.append(f"centred rolling window{f' of {w} rows' if isinstance(w, int) else ''} (averages later rows)")
        if name == "shift":
            periods = _const(n.args[0]) if n.args else kw.get("periods")
            if isinstance(periods, (int, float)) and periods < 0:
                found.append(f"shift({periods}) (copies a later row)")
        if name in ("bfill", "backfill") or (name == "fillna" and kw.get("method") in ("bfill", "backfill")):
            found.append("backward fill (copies later values)")
        if name == "transform" and target:
            # groupby(...)[target].transform(...)
            v = n.func.value
            if isinstance(v, ast.Subscript) and isinstance(v.slice, ast.Constant) and v.slice.value == target:
                found.append(f"target encoding from '{target}' on all rows")
    return found


def find_lookahead(code: str, target: str | None, design_columns: list[str]) -> list[dict]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        cols_exprs = []
        if isinstance(node, ast.Assign):
            for t in node.targets:
                c = _target_col(t)
                if c:
                    cols_exprs.append((c, node.value))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "assign":
            cols_exprs += [(k.arg, k.value) for k in node.keywords if k.arg]
        for col, expr in cols_exprs:
            if col not in design_columns:
                continue                       # only features the model actually used
            pats = _patterns(expr, target)
            if pats:
                out.append({"column": col, "line": node.lineno, "patterns": sorted(set(pats))})
    seen, uniq = set(), []
    for f in out:
        if f["column"] not in seen:
            seen.add(f["column"])
            uniq.append(f)
    return uniq
