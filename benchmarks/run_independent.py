"""Independent evaluation: hand-written notebooks, real and synthetic data, not produced by the
generator used during development. Expected outcomes were written down before running.

    python benchmarks/independent/build_data.py && python benchmarks/run_independent.py
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leakverity.pipeline import Options, audit  # noqa: E402

IND = ROOT / "benchmarks" / "independent"
EXPECT_CONDITIONAL = {"c11_bc_centered_no_time"}     # ambiguous case: verdict must report both scenarios

# case -> (data file, expected automatic findings, acceptable verdicts, what it tests)
CASES = {
    "c01_bc_clean": ("breast_cancer.csv", set(), {"Supported"}, "real data, leak-free; strong genuine features must not be removed"),
    "c02_bc_scaler_first": ("breast_cancer.csv", {"preprocessing"}, {"Supported"}, "textbook scaler leak; module-style imports, NumPy arrays; impact should be negligible"),
    "c03_noise_selectkbest": ("noise_genes.csv", {"preprocessing"}, {"Contradicted"}, "feature selection on all data with pure-noise features; truth is chance"),
    "c04_smote_first": ("transactions.csv", {"resampling"}, {"Overstated"}, "SMOTE before the split"),
    "c05_cross_val": ("breast_cancer.csv", set(), {"Insufficient evidence"}, "cross-validation protocol: must abstain"),
    "c06_gridsearch": ("breast_cancer.csv", set(), {"Insufficient evidence"}, "GridSearchCV object: must abstain, not guess"),
    "c07_wine_tuned_on_test": ("wine.csv", {"test_selection"}, {"Supported", "Overstated"}, "k tuned on the test set inside a helper function; accuracy metric"),
    "c08_bc_exported_twice": ("breast_cancer_exported_twice.csv", {"duplicates"}, {"Supported", "Overstated"}, "records exported twice (same sample_id)"),
    "c09_sensor_centered_rolling": ("sensor.csv", {"future_features"}, {"Overstated"},
                                    "future-peeking (centred) rolling mean built in pandas before scikit-learn"),
    "c10_bc_clean_groupless_proba": ("breast_cancer.csv", set(), {"Supported"}, "leak-free with internal early stopping"),
    "c11_bc_centered_no_time": ("breast_cancer.csv", {"proposed_features"}, {"Supported"}, "AMBIGUOUS: centred window, no time order, computed before the split: proposed (not removed), evaluated both ways"),
    "c12_bc_rowwise_features": ("breast_cancer.csv", set(), {"Supported"}, "clean control: row-wise engineered features; unconditional Supported expected"),
}


def ground_truth(rows):
    sys.path.insert(0, str(IND))
    import references as REF
    for r in rows:
        f = REF.REFERENCE.get(r["case"])
        if f is None:
            r["reference"], r["gt_verdict"] = None, "Insufficient evidence"         # unsupported protocol: must abstain
        else:
            r["reference"] = float(f())
            r["gt_verdict"] = REF.expected_verdict(r["case"], r["reported"], r["reference"]) if r["reported"] is not None else "?"
        want_conditional = r["case"] in EXPECT_CONDITIONAL
        r["verdict_ok"] = r["verdict"] == r["gt_verdict"] and r.get("conditional", False) == want_conditional
        print(r["case"], "reported", r["reported"] and round(r["reported"], 3), "reference", r["reference"] and round(r["reference"], 3),
              "expected", r["gt_verdict"], "got", r["verdict"], flush=True)
    return rows


def main():
    if "--rescore" in sys.argv:
        rows = json.loads((ROOT / "benchmarks" / "results" / "independent_results.json").read_text())["rows"]
        for r in rows:
            _, expected, verdicts, what = CASES[r["case"]]
            r.update(expected=sorted(expected), tests=what, detection_ok=set(r["found"]) == expected)
        return report(ground_truth(rows))
    rows = []
    for case, (data, expected, verdicts, what) in CASES.items():
        code = (IND / "cases" / f"{case}.py").read_bytes()
        r = audit(f"{case}.py", code, data, (IND / data).read_bytes(), Options(n_seeds=3, use_ai=False))
        found = {f["fix"] for f in r.get("findings", []) if f.get("fix")}
        review = sorted(f["column"] for f in r.get("findings", []) if f.get("needs_review"))
        v = (r.get("verdict") or {}).get("verdict")
        rep = (r.get("repair") or {}).get("verification") or {}
        row = {"case": case, "tests": what, "expected": sorted(expected), "found": sorted(found), "review": review,
               "verdict": v, "acceptable": sorted(verdicts), "verdict_ok": v in verdicts,
               "detection_ok": found == expected, "repair_verified": rep.get("verified"),
               "reported": (r.get("protocol") or {}).get("reported"), "verified": ((r.get("replay") or {}).get("corrected") or {}).get("mean"),
               "reason": (r.get("verdict") or {}).get("reason", "")[:200], "seconds": r.get("seconds"),
               "conditional": bool((r.get("verdict") or {}).get("scenarios"))}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("case", "expected", "found", "review", "verdict", "verdict_ok", "repair_verified", "reported", "verified")}), flush=True)
    report(ground_truth(rows))


def report(rows):
    tp = sum(len(set(r["found"]) & set(r["expected"])) for r in rows)
    fp = sum(len(set(r["found"]) - set(r["expected"])) for r in rows)
    fn = sum(len(set(r["expected"]) - set(r["found"])) for r in rows)
    summ = {"cases": len(rows), "detection_exact": sum(r["detection_ok"] for r in rows), "tp": tp, "fp": fp, "fn": fn,
            "verdict_ok": sum(r["verdict_ok"] for r in rows),
            "repairs_verified": sum(bool(r["repair_verified"]) for r in rows if r["found"]),
            "repairs_attempted": sum(bool(r["found"]) for r in rows)}
    out = ROOT / "benchmarks" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "independent_results.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    md = ["# Independent evaluation", "",
          "Twelve hand-written notebooks, not produced by the development generator, in varied coding styles, on real public datasets "
          "bundled with scikit-learn (breast cancer, wine) and new synthetic ones. Expected *findings* were written before the first run "
          "(c09 was later re-scored more strictly). Expected *verdicts* come from an independent reference: an honest "
          "re-implementation of each model evaluated by repeated cross-validation (`benchmarks/independent/references.py`), "
          "a different method from LeakVerity's hold-out replay. Rule-based reader, 3 replay splits.", "",
          f"- Exact leak detection: **{summ['detection_exact']}/{summ['cases']}** cases (TP {tp}, FP {fp}, FN {fn})",
          f"- Verdict matches the independent reference: **{summ['verdict_ok']}/{summ['cases']}**",
          f"- Repairs verified by execution: **{summ['repairs_verified']}/{summ['repairs_attempted']}**", "",
          "| Case | What it tests | Expected leaks | Found | Reported | LeakVerity verified | Reference (CV) | Verdict | Expected verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        fmt = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
        md.append(f"| {r['case']} | {r['tests']} | {', '.join(r['expected']) or 'none'} | {', '.join(r['found']) or 'none'} | "
                  f"{fmt(r['reported'])} | {fmt(r['verified'])} | {fmt(r.get('reference'))} | {r['verdict']}{' (holds in both scenarios)' if r.get('conditional') else ''} | "
                  f"{r.get('gt_verdict')}{' (holds in both scenarios)' if r['case'] in EXPECT_CONDITIONAL else ''} |")
    md += ["", "**Notes.**",
           "- **c09** (future-peeking rolling mean built in pandas) was a miss in an earlier version. Static look-ahead detection "
           "(`leakverity/lookahead.py`: centred windows, negative shifts, backward fill, target encoding) was added after review and "
           "now catches it; removing the feature is measured as material and the verdict is Overstated.",
           "- **c11** is deliberately ambiguous, not clean: a centred window over rows with no time order, computed before the split, "
           "so test rows shape training features. LeakVerity reports the cross-partition mixing, treats the feature "
           "as a *proposed* leak instead of removing it, evaluates the claim with and without it, and reports Supported in both scenarios. **c12** is the clean control: row-wise features only, "
           "unconditional Supported.",
           "- **c03** (pure-noise features): with the default 3 replay splits on 60 test rows, the corrected CI (0.297–0.765) still "
           "contains the claimed 0.76, so LeakVerity returns *Insufficient evidence* rather than *Contradicted*. Run in Thorough "
           "mode (6 splits), the same notebook gives **Contradicted** (verified 0.502, 95% CI 0.365–0.639). Scored as a mismatch at "
           "the default setting."]
    (out / "INDEPENDENT.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
