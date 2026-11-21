"""LeakVerity benchmark with known ground truth.

Every scenario is a generated study: a dataset from a known process plus a notebook with
a planted subset of leaks (or none). Because the process is known, the *true* performance
of the honestly trained model can be measured on thousands of fresh patients.

We measure:
  detection    precision / recall per leak type, false alarms on clean notebooks
  replay       fidelity success, verified repairs
  estimation   |reported − truth| vs |verified − truth|, CI coverage of the truth
  verdicts     agreement with the ground-truth verdict, false reversals of honest claims

Usage: python benchmarks/run_benchmark.py [n_scenarios] [seed]
"""
from __future__ import annotations

import json
import pathlib
import random
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from leakverity import synth  # noqa: E402
from leakverity.pipeline import Options, audit  # noqa: E402

BASE = ["age", "sex_male", "comorbidity_index", "hba1c", "num_prior_admissions", "length_of_stay",
        "num_medications", "num_diagnoses", "emergency_admission"]
FUTURE_NAMES = ["followup_ed_visits_30d", "ed_visits_after_discharge", "post_discharge_ed_visits"]
DECOY = "discharge_hour"           # sounds post-hoc, is known at discharge: must not be flagged
LEAK_TYPES = ["preprocessing", "duplicates", "group_overlap", "future_features", "test_selection"]


def make_scenario(i: int, rng: random.Random) -> dict:
    clean = i % 4 == 0
    planted = set() if clean else {k for k in LEAK_TYPES if rng.random() < 0.5}
    if not clean and not planted:
        planted = {rng.choice(LEAK_TYPES)}
    model = rng.choice(["rf", "lr"])
    gen = dict(n_patients=rng.choice([700, 900, 1100]), seed=1000 + i, repeat_rate=rng.choice([0.5, 0.9]),
               dup_rate=0.07 if "duplicates" in planted else 0.0, future_strength=rng.choice([0.6, 1.0]))
    df = synth.generate(**gen)
    rs = np.random.default_rng(i)
    df[DECOY] = rs.integers(8, 20, len(df))
    fut = rng.choice(FUTURE_NAMES)
    df = df.rename(columns={synth.FUTURE_FEATURE: fut})
    feats = BASE + ([DECOY] if rng.random() < 0.5 else []) + ([fut] if "future_features" in planted else [])
    split_seed = rng.randint(0, 99)
    random_split = bool({"duplicates", "group_overlap"} & planted) or (not clean and rng.random() < 0.5)
    if random_split and gen["repeat_rate"] > 0:
        planted.add("group_overlap")                     # a random split of repeat patients is itself the leak
    if random_split and gen["dup_rate"] == 0:
        planted.discard("duplicates")
    if not random_split:
        planted.discard("duplicates")
    est = ("RandomForestClassifier(n_estimators=60, max_depth=8, random_state={s}, n_jobs=-1)" if model == "rf"
           else "LogisticRegression(C={s}, max_iter=500)")
    cands = list(range(5)) if model == "rf" else [0.01, 0.1, 1.0, 10.0]
    lines = ["import pandas as pd", "import numpy as np",
             "from sklearn.model_selection import train_test_split, GroupShuffleSplit",
             "from sklearn.preprocessing import StandardScaler", "from sklearn.pipeline import make_pipeline",
             "from sklearn.ensemble import RandomForestClassifier", "from sklearn.linear_model import LogisticRegression",
             "from sklearn.metrics import roc_auc_score", "",
             'df = pd.read_csv("study.csv")']
    if not random_split and gen["dup_rate"] == 0:
        lines.append('df = df.drop_duplicates("encounter_id")')
    lines += [f"features = {feats!r}", "X = df[features]", 'y = df["readmitted_30d"]']
    if "preprocessing" in planted:
        lines.append("X = StandardScaler().fit_transform(X)")
    if random_split:
        lines.append(f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state={split_seed}, stratify=y)")
    else:
        lines += [f"gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state={split_seed})",
                  'tr, te = next(gss.split(X, y, groups=df["patient_id"]))',
                  ("X_train, X_test = X[tr], X[te]" if "preprocessing" in planted else "X_train, X_test = X.iloc[tr], X.iloc[te]"),
                  "y_train, y_test = y.iloc[tr], y.iloc[te]"]
    wrap = (lambda e: e) if "preprocessing" in planted else (lambda e: f"make_pipeline(StandardScaler(), {e})")
    if "test_selection" in planted:
        lines += ["best = 0", f"for s in {cands!r}:", f"    model = {wrap(est)}", "    model.fit(X_train, y_train)",
                  "    best = max(best, roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))",
                  'print(f"Test AUC: {best:.3f}")']
    else:
        lines += [f"model = {wrap(est.replace('{s}', str(cands[0] if model == 'rf' else 1.0)))}", "model.fit(X_train, y_train)",
                  'print(f"Test AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.3f}")']
    code = "\n".join(l.replace("{s}", "s") if "for s in" not in l else l for l in lines) + "\n"

    # ---- ground truth: honest model evaluated on fresh patients ----------------
    fresh = synth.fresh_population(n_patients=5000, seed=50_000 + i, repeat_rate=gen["repeat_rate"],
                                   future_strength=gen["future_strength"])
    fresh[DECOY] = np.random.default_rng(10_000 + i).integers(8, 20, len(fresh))
    honest_feats = [f for f in feats if f != fut]
    d = df.drop_duplicates("encounter_id")
    truths = []
    for s in range(3):
        tr, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=s).split(d, groups=d.patient_id))
        sc = []
        for c in (cands if "test_selection" in planted and model == "rf" else [cands[0] if model == "rf" else 1.0]):
            m = (RandomForestClassifier(n_estimators=60, max_depth=8, random_state=c, n_jobs=-1) if model == "rf"
                 else make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=500)))
            m.fit(d.iloc[tr][honest_feats], d.iloc[tr].readmitted_30d)
            sc.append(roc_auc_score(fresh.readmitted_30d, m.predict_proba(fresh[honest_feats])[:, 1]))
        truths.append(float(np.mean(sc)))
    return {"id": f"S{i:02d}", "planted": sorted(planted), "clean": not planted, "model": model, "code": code,
            "csv": df.to_csv(index=False).encode(), "truth": float(np.mean(truths)), "features": feats,
            "future_name": fut, "decoy_used": DECOY in feats, "generator": gen}


def main(n: int = 24, seed: int = 7):
    rng = random.Random(seed)
    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for i in range(n):
        sc = make_scenario(i, rng)
        r = audit(f"{sc['id']}.py", sc["code"].encode(), "study.csv", sc["csv"],
                  Options(n_seeds=3, run_survival=False, group_col="patient_id"))
        # a proposed leak counts as detecting a feature leak (it is evaluated both ways)
        found = sorted({("future_features" if f["fix"] == "proposed_features" else f["fix"]) for f in r.get("findings", []) if f.get("fix")})
        rp = r.get("replay") or {}
        corr = rp.get("corrected") or {}
        reported = (r.get("protocol") or {}).get("reported")
        truth = sc["truth"]
        gt_verdict = "Overstated" if reported is not None and reported - truth > 0.03 else "Supported"
        row = {"id": sc["id"], "model": sc["model"], "planted": sc["planted"], "found": found,
               "flagged_decoy": any(f.get("column") == DECOY for f in r.get("findings", [])),
               "decoy_used": sc["decoy_used"], "fidelity": rp.get("fidelity_ok"),
               "repair_verified": ((r.get("repair") or {}).get("verification") or {}).get("verified"),
               "reported": reported, "verified": corr.get("mean"), "ci": [corr.get("lo"), corr.get("hi")], "truth": truth,
               "verdict": (r.get("verdict") or {}).get("verdict"), "gt_verdict": gt_verdict, "seconds": r.get("seconds"),
               "confirmed_verdict": (((r.get("verdict") or {}).get("scenarios") or {}).get("confirmed") or {}).get("verdict")}
        rows.append(row)
        print(f"{row['id']} {row['model']} planted={row['planted']} found={row['found']} rep={reported and round(reported, 3)} "
              f"ver={row['verified'] and round(row['verified'], 3)} truth={truth:.3f} verdict={row['verdict']}/{gt_verdict} "
              f"fid={row['fidelity']} fix={row['repair_verified']} {row['seconds']}s", flush=True)
    summary = summarise(rows)
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    (out_dir / "benchmark_results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    (out_dir / "BENCHMARK.md").write_text(markdown(summary, rows))
    print(json.dumps(summary, indent=1))


def summarise(rows):
    s = {"scenarios": len(rows), "clean": sum(r["clean"] if "clean" in r else not r["planted"] for r in rows)}
    per = {}
    for t in LEAK_TYPES:
        tp = sum(t in r["planted"] and t in r["found"] for r in rows)
        fn = sum(t in r["planted"] and t not in r["found"] for r in rows)
        fp = sum(t not in r["planted"] and t in r["found"] for r in rows)
        per[t] = {"tp": tp, "fn": fn, "fp": fp,
                  "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None}
    s["per_type"] = per
    TP = sum(v["tp"] for v in per.values())
    FP = sum(v["fp"] for v in per.values())
    FN = sum(v["fn"] for v in per.values())
    s["precision"] = TP / (TP + FP) if TP + FP else None
    s["recall"] = TP / (TP + FN) if TP + FN else None
    clean = [r for r in rows if not r["planted"]]
    s["clean_false_alarm_rate"] = (sum(bool(r["found"]) for r in clean) / len(clean)) if clean else None
    s["decoy_flag_rate"] = (sum(r["flagged_decoy"] for r in rows if r["decoy_used"]) / max(1, sum(r["decoy_used"] for r in rows)))
    ok = [r for r in rows if r["fidelity"]]
    s["fidelity_rate"] = len(ok) / len(rows)
    s["repair_verified_rate"] = sum(bool(r["repair_verified"]) for r in rows if r["found"]) / max(1, sum(bool(r["found"]) for r in rows))
    err_rep = [abs(r["reported"] - r["truth"]) for r in ok]
    err_ver = [abs(r["verified"] - r["truth"]) for r in ok]
    s["mean_abs_error_reported"] = float(np.mean(err_rep)) if ok else None
    s["mean_abs_error_verified"] = float(np.mean(err_ver)) if ok else None
    s["ci_coverage_of_truth"] = float(np.mean([r["ci"][0] <= r["truth"] <= r["ci"][1] for r in ok])) if ok else None
    s["verdict_agreement"] = float(np.mean([r["verdict"] == r["gt_verdict"] for r in ok])) if ok else None
    s["depends_on_review_rate"] = float(np.mean([r["verdict"] == "Depends on review" for r in ok])) if ok else None
    # when the proposed leak is confirmed (as it truly is in these generated studies)
    s["verdict_agreement_if_confirmed"] = float(np.mean([(r.get("confirmed_verdict") or r["verdict"]) == r["gt_verdict"] for r in ok])) if ok else None
    honest = [r for r in ok if r["gt_verdict"] == "Supported"]
    s["false_reversal_rate"] = (sum(r["verdict"] in ("Overstated", "Contradicted") for r in honest) / len(honest)) if honest else None
    leaky = [r for r in ok if r["gt_verdict"] == "Overstated"]
    s["overstated_caught_rate"] = (sum((r.get("confirmed_verdict") or r["verdict"]) == "Overstated" for r in leaky) / len(leaky)) if leaky else None
    return s


def markdown(s, rows):
    pct = lambda x: "—" if x is None else f"{100 * x:.0f}%"  # noqa: E731
    lines = ["# LeakVerity benchmark", "",
             f"{s['scenarios']} generated studies ({s['clean']} clean controls), rule-based feature checks, 3 replay splits each. "
             "Ground truth = the honestly trained model evaluated on 5,000 fresh synthetic patients from the same process.", "",
             "| Measure | Result |", "|---|---|",
             f"| Leak detection precision | {pct(s['precision'])} |", f"| Leak detection recall | {pct(s['recall'])} |",
             f"| False alarms on clean notebooks | {pct(s['clean_false_alarm_rate'])} |",
             f"| Decoy feature wrongly flagged | {pct(s['decoy_flag_rate'])} |",
             f"| Replays that reproduced the reported number | {pct(s['fidelity_rate'])} |",
             f"| Corrected code verified by execution | {pct(s['repair_verified_rate'])} |",
             f"| Mean error of the reported score vs truth | {s['mean_abs_error_reported']:.3f} |",
             f"| Mean error of LeakVerity's verified score vs truth | {s['mean_abs_error_verified']:.3f} |",
             f"| Truth inside LeakVerity's 95% CI | {pct(s['ci_coverage_of_truth'])} |",
             f"| Headline verdict agreement with ground truth | {pct(s['verdict_agreement'])} |",
             f"| Verdicts left as *Depends on review* (semantic proposal changes the verdict) | {pct(s['depends_on_review_rate'])} |",
             f"| Verdict agreement once the proposed leak is confirmed | {pct(s['verdict_agreement_if_confirmed'])} |",
             f"| Honest claims wrongly overturned | {pct(s['false_reversal_rate'])} |",
             f"| Overstated claims caught (once proposals are confirmed) | {pct(s['overstated_caught_rate'])} |", "",
             "## Per leak type", "", "| Leak | Precision | Recall | TP | FP | FN |", "|---|---|---|---|---|---|"]
    for t, v in s["per_type"].items():
        lines.append(f"| {t} | {pct(v['precision'])} | {pct(v['recall'])} | {v['tp']} | {v['fp']} | {v['fn']} |")
    lines += ["", "## Scenarios", "", "| ID | Model | Planted | Found | Reported | Verified (95% CI) | Truth | Verdict | Ground truth |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = "—" if r["verified"] is None else f"{r['verified']:.3f} ({r['ci'][0]:.3f}–{r['ci'][1]:.3f})"
        lines.append(f"| {r['id']} | {r['model']} | {', '.join(r['planted']) or 'none'} | {', '.join(r['found']) or 'none'} | "
                     f"{'—' if r['reported'] is None else f'{r['reported']:.3f}'} | {ci} | {r['truth']:.3f} | {r['verdict']} | {r['gt_verdict']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 24, int(sys.argv[2]) if len(sys.argv) > 2 else 7)
