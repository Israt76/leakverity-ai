"""Leakage findings: each one is backed by execution evidence and mapped to a fix."""
from __future__ import annotations

import re

import numpy as np

from . import replay as R
from .ingest import Source
from .sandbox import RunResult
from .lookahead import find_lookahead
from .spec import Spec


def _fmt_n(n) -> str:
    return f"{int(n):,}"


def _code_line(src: Source, line: int | None) -> str:
    lines = src.code_lines()
    if line and 1 <= line <= len(lines):
        return lines[line - 1].strip()
    return ""


def _lines_mentioning(src: Source, token: str) -> list[int]:
    pat = re.compile(r"[\"']" + re.escape(token) + r"[\"']")
    return [i + 1 for i, l in enumerate(src.code_lines()) if pat.search(l)]


def _window_contamination(la: dict, spec: Spec) -> str | None:
    """For centred windows: how many training rows' feature values include a test row (file order)."""
    import re as _re
    m = next((_re.search(r"of (\d+) rows", p) for p in la["patterns"] if "centred" in p), None)
    split = spec.split
    if not m or len(spec.design) != split.get("n"):
        return None
    half = int(m.group(1)) // 2
    test = set(split["test_pos"])
    n = split["n"]
    hit = sum(1 for i in split["train_pos"] if any(j in test for j in range(max(0, i - half), min(n, i + half + 1)) if j != i))
    return (f"Cross-partition mixing: {hit:,} of {len(split['train_pos']):,} training rows have a `{la['column']}` value computed "
            "with at least one test row (assuming the window runs over the rows in their loaded order).")


def _time_order_evidence(src: Source, spec: Spec) -> str | None:
    """Evidence that rows are in chronological order: a date column the data is sorted by, or explicit sorting."""
    import pandas as pd
    if spec.date is not None:
        d = pd.to_datetime(spec.date, errors="coerce", format="mixed")
        if d.notna().mean() > 0.95 and (d.is_monotonic_increasing or d.is_monotonic_decreasing):
            return f"the rows are sorted by the date column `{spec.date_col}`"
        if spec.date_col and re.search(r"sort_values\(\s*\[?\s*[\"']" + re.escape(spec.date_col), src.code):
            return f"the code sorts the rows by `{spec.date_col}`"
    return None


def find(src: Source, run: RunResult, spec: Spec, audit: dict) -> list[dict]:
    out: list[dict] = []
    split = spec.split
    split_where = src.where(split.get("line"))
    n_test, n_all = len(split["test_pos"]), split["n"]

    # 1. preprocessing / resampling fitted before the split (or on data that includes the test rows)
    for st in spec.pre_steps:
        if st.kind == "sampler":
            fix, title = R.FIX_SAMPLER, f"{st.cls} resampled the data before it was split"
            expl = (f"{st.cls} created synthetic training examples from the full dataset, so synthetic points derived from "
                    "test rows end up in training. The test set then measures how well the model recognises near-copies.")
        else:
            fix = R.FIX_PREPROC
            if st.name == "fit_on_full_after_split":
                title = f"{st.cls} was fitted on data that includes the test rows"
            else:
                title = f"{st.cls} learned its parameters from all {_fmt_n(n_all)} rows before the split"
            expl = (f"{st.cls} estimates its parameters (for example means and spreads) from the data it is fitted on. "
                    "Fitting it before the split lets statistics of the test rows shape the training data.")
        out.append({"id": f"F{len(out) + 1}", "rule": "LV-" + fix.upper(), "fix": fix, "title": title,
                    "lines": [l for l in (st.line, split.get("line")) if l],
                    "evidence": [f"{st.cls}.fit ran at {src.where(st.line)}: `{_code_line(src, st.line)}`",
                                 f"The split ran afterwards at {split_where} and held out {_fmt_n(n_test)} of those rows."],
                    "explanation": expl, "source": "execution trace"})

    # 2. duplicate records across the split
    twins = split.get("test_rows_duplicated_in_train") or 0
    rec_dups, rec_known = None, False
    if spec.record_key is not None and len(spec.design) == n_all:
        keys = spec.record_key.to_numpy()
        train_keys = set(keys[split["train_pos"]].tolist())
        rec_dups = int(sum(1 for i in split["test_pos"] if keys[i] in train_keys))
        rec_known = True
    dup = rec_dups if rec_known else twins
    if dup:
        ev = [f"After the split at {split_where}, {_fmt_n(dup)} of {_fmt_n(n_test)} test rows ({dup / n_test:.1%}) are "
              + ("the same source record as a training row (every column identical, IDs included)." if rec_known
                 else "identical to a training row in every feature and the label (source records could not be traced).")]
        if rec_known and twins > dup:
            ev.append(f"A further {_fmt_n(twins - dup)} test rows match a training row on all features but come from different "
                      "records; they are not treated as duplicates.")
        out.append({"id": f"F{len(out) + 1}", "rule": "LV-DUPLICATES", "fix": R.FIX_DEDUP,
                    "title": f"{_fmt_n(dup)} test rows are copies of training records",
                    "lines": [split.get("line")], "evidence": ev,
                    "explanation": "A model can score those rows by recalling the identical training example instead of generalising. "
                                   "The fix removes records exported more than once, keeping the first copy.",
                    "source": "execution trace + source records" if rec_known else "execution trace"})

    # 3. same group (patient/user) on both sides of a random split
    if spec.groups is not None and spec.split.get("via", "train_test_split") != "GroupShuffleSplit":
        g = spec.groups
        idx = spec.design.index
        tr_g = set(g.loc[idx[split["train_pos"]]].tolist()) if len(idx) == n_all else set()
        te_rows = g.loc[idx[split["test_pos"]]] if len(idx) == n_all else None
        if te_rows is not None:
            shared = set(te_rows.tolist()) & tr_g
            rows_shared = int(te_rows.isin(shared).sum())
            if shared:
                out.append({"id": f"F{len(out) + 1}", "rule": "LV-GROUP-OVERLAP", "fix": R.FIX_GROUP,
                            "title": f"{_fmt_n(len(shared))} {spec.group_col} values appear in both train and test",
                            "lines": [split.get("line")],
                            "evidence": [f"The random split at {split_where} placed records of the same {spec.group_col} on both sides: "
                                         f"{_fmt_n(rows_shared)} of {_fmt_n(n_test)} test rows ({rows_shared / n_test:.1%}) belong to "
                                         f"a {spec.group_col} that is also in training."],
                            "explanation": (f"Records from one {spec.group_col} share stable traits and outcomes. The model can recognise the "
                                            "individual rather than learn patterns that transfer to new people."),
                            "source": "execution trace + data"})

    # 4. features unavailable at prediction time.
    # A semantic judgement (AI or name rules) is a SUSPICION, not proof of timing. Only the user's
    # confirmation makes it a definite leak (fix applied). Otherwise it becomes a PROPOSED leak: the
    # replay measures the verdict both ways (removed / kept). Strength-only flags are notes.
    for fl in audit.get("flags", []):
        col = fl["column"]
        lines = _lines_mentioning(src, col)
        ev = [f"Reason: {fl['reason']}"]
        if fl.get("auc") is not None:
            ev.append(f"On its own, `{col}` separates the two outcome classes with AUC {fl['auc']:.2f}.")
        if lines:
            ev.append(f"Selected as a model feature at {src.where(lines[0])}.")
        source = {"ai": "AI semantic audit", "user": "confirmed by the user"}.get(fl.get("source"), "rule-based semantic audit")
        if fl.get("review_kind") == "strength_only":
            out.append({"id": f"F{len(out) + 1}", "rule": "LV-NOTE-STRONG-FEATURE", "fix": None, "column": col, "needs_review": True,
                        "review_kind": "strength_only", "title": f"`{col}` is an unusually strong single predictor",
                        "lines": lines[:1], "evidence": ev, "confidence": "low",
                        "explanation": ("Strength alone is not evidence of leakage; real data has genuinely strong predictors. "
                                        "Noted for the author, not removed, and it does not affect the verdict."),
                        "source": source})
            continue
        if fl.get("source") == "user":
            out.append({"id": f"F{len(out) + 1}", "rule": "LV-FUTURE-FEATURE", "fix": R.FIX_PROXY, "column": col,
                        "title": f"`{col}` is not available when the prediction is made (confirmed)",
                        "lines": lines[:1], "evidence": ev, "confidence": "confirmed",
                        "explanation": "Confirmed by the user as recorded after the prediction point, so it is removed in the corrected protocol.",
                        "source": source})
            continue
        out.append({"id": f"F{len(out) + 1}", "rule": "LV-PROPOSED-LEAK", "fix": R.FIX_PROPOSED, "column": col, "proposed": True,
                    "title": f"`{col}` may not be available when the prediction is made (proposed)",
                    "lines": lines[:1], "evidence": ev, "confidence": fl.get("confidence"),
                    "explanation": ("If a feature is only recorded after the prediction point, the model is partly reading the answer. "
                                    "This is a suspicion from the feature's meaning, not proof of its timing, so LeakVerity evaluates the "
                                    "claim both ways (feature removed and feature kept) and reports whether the verdict depends on it. "
                                    "Confirm or reject it to settle the verdict."),
                    "source": source})

    # 4b. look-ahead features built in pandas (static, then measured by replay)
    # Centred windows, negative shifts and backward fill only leak the future if the rows are in time
    # order. Code alone cannot show that, so they are fixed automatically only with time-order evidence:
    # a date column that the data is sorted by (or that the code sorts by). Target encoding on all rows
    # leaks regardless of row order and is always fixed.
    flagged = {f.get("column") for f in out}
    order_evidence = _time_order_evidence(src, spec)
    for la in find_lookahead(src.code, audit.get("target"), list(spec.design.columns)):
        if la["column"] in flagged:
            continue
        needs_order = any("target encoding" not in p for p in la["patterns"])
        auto = (not needs_order) or order_evidence is not None
        ev = [f"Created at {src.where(la['line'])}: `{_code_line(src, la['line'])}`", "Pattern: " + "; ".join(la["patterns"]) + "."]
        if needs_order:
            ev.append(f"Time order: {order_evidence}." if order_evidence else
                      "Time order: no date column was found that the rows are sorted by, so it is unclear whether later rows are later in time.")
            mix = _window_contamination(la, spec)
            if mix:
                ev.append(mix)
        if not auto:
            out.append({"id": f"F{len(out) + 1}", "rule": "LV-LOOKAHEAD-REVIEW", "fix": R.FIX_PROPOSED, "column": la["column"],
                        "proposed": True, "title": f"`{la['column']}` mixes each row with neighbouring rows (proposed)",
                        "lines": [la["line"]], "evidence": ev,
                        "explanation": ("This feature combines each row with neighbouring rows, computed before the split. Even without "
                                        "time order, that lets test rows shape training features (cross-partition contamination), and "
                                        "if the rows are chronological it also peeks at the future. The time order could not be "
                                        "established, so LeakVerity evaluates the claim both ways (feature removed and kept) and "
                                        "reports whether the verdict depends on it."),
                        "source": "static code analysis"})
            continue
        out.append({"id": f"F{len(out) + 1}", "rule": "LV-LOOKAHEAD", "fix": R.FIX_PROXY, "column": la["column"],
                    "title": f"`{la['column']}` is built from information after each row",
                    "lines": [la["line"]], "evidence": ev,
                    "explanation": ("This feature is computed in pandas before the split, using rows that come later than the one "
                                    "being predicted" + (" (the rows are in time order)" if needs_order else "")
                                    + ", so each prediction can peek at information it would not have. LeakVerity cannot recompute "
                                    "it safely, so the measured fix removes it, which is conservative. A causal version (for "
                                    "example a trailing window) may recover some of the lost performance."),
                    "source": "static code analysis"})

    # 5. model selection on the test set
    if spec.selection != "none":
        vals = [m["value"] for m in spec.metric_events]
        line = spec.metric_events[0].get("line")
        kind = "random seeds" if spec.selection == "seed" else "hyper-parameter settings"
        out.append({"id": f"F{len(out) + 1}", "rule": "LV-TEST-SELECTION", "fix": R.FIX_SELECT,
                    "title": f"{len(spec.candidates)} models were compared on the test set and the best was reported",
                    "lines": [line] if line else [],
                    "evidence": [f"{len(spec.candidates)} {spec.model.cls} models with different {kind} were each scored on the same "
                                 f"{_fmt_n(n_test)} test labels at {src.where(line)}.",
                                 f"Scores ranged from {min(vals):.3f} to {max(vals):.3f}; the notebook reports {spec.reported:.3f}."],
                    "explanation": ("Picking the best of several test scores turns the test set into a selection tool, so the winning "
                                    "score is biased upward. An honest estimate averages the candidates or selects them on training data."),
                    "source": "execution trace"})
    return out


def fixes_for(findings: list[dict]) -> list[str]:
    order = [R.FIX_PREPROC, R.FIX_SAMPLER, R.FIX_DEDUP, R.FIX_GROUP, R.FIX_PROXY, R.FIX_PROPOSED, R.FIX_SELECT]
    present = {f["fix"] for f in findings if f.get("fix")}
    return [f for f in order if f in present]
