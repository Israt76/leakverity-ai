"""AI reader vs rule-based reader on ambiguous feature names (known ground truth).

Prediction task: 30-day readmission, predicted at discharge. Each feature below is
generated with a known role: a post-discharge leak (derived from the outcome) or a
legitimate feature known at discharge. Several names are deliberately ambiguous: leaks
with cryptic names, and legitimate features whose names *sound* like leakage.

    python benchmarks/ai_vs_rules.py            # rule-based reader only
    ANTHROPIC_API_KEY=... python benchmarks/ai_vs_rules.py   # adds the AI reader

Writes benchmarks/results/AI_VS_RULES.md. Only feature names, example values and
univariate AUCs are sent to the AI, never the data.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leakverity import semantic, synth  # noqa: E402

# name -> (is_leak, how it is generated, what it means)
CATALOG = {
    # post-discharge information (leaks)
    "followup_ed_visits_30d": (True, "post", "ER visits in the 30 days after discharge"),
    "ed_ct_30": (True, "post", "ER visit count, 30 days after discharge (cryptic name)"),
    "readm_billing_code": (True, "post", "billing code assigned at the readmission"),
    "days_to_next_adm": (True, "post", "days until the next admission"),
    "cm_call_outcome": (True, "post", "result of the care-manager call made after discharge"),
    "snf_claims_30d": (True, "post", "nursing-facility claims in the 30 days after discharge"),
    "rx_fill_gap_post": (True, "post", "medication refill gap after discharge"),
    # legitimate, known at discharge (several sound suspicious)
    "discharge_hour": (False, "noise", "hour of discharge"),
    "followup_appt_scheduled": (False, "risk", "whether a follow-up appointment was booked before discharge"),
    "prior_readmissions": (False, "risk", "readmissions in the previous year"),
    "outcome_risk_score_admit": (False, "risk", "risk score computed at admission"),
    "lab_result_count": (False, "risk", "number of lab results during the stay"),
    "max_troponin_stay": (False, "risk", "highest troponin during the stay"),
    "disp_cd": (False, "risk", "discharge disposition code (known at discharge)"),
    "length_of_stay": (False, "base", "length of stay"),
}


def build(seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    df = synth.generate(n_patients=1500, seed=seed, dup_rate=0.0)
    rng = np.random.default_rng(seed)
    y = df["readmitted_30d"]
    # legitimate features carry real, discharge-time signal (as genuine risk factors do)
    r = df["num_prior_admissions"] + df["length_of_stay"] / 4 + df["emergency_admission"] + df["comorbidity_index"] / 2
    risk = (r - r.mean()) / r.std()
    X = pd.DataFrame(index=df.index)
    for name, (leak, kind, _) in CATALOG.items():
        if kind == "post":
            X[name] = rng.poisson(0.2 + 1.6 * y) if "code" not in name and "outcome" not in name else (y * rng.integers(1, 9, len(y)) + rng.binomial(1, 0.08, len(y)))
        elif kind == "risk":
            X[name] = np.round(risk + rng.normal(0, 0.5, len(y)), 2)
        elif kind == "base":
            X[name] = df[name]
        else:
            X[name] = rng.integers(8, 20, len(y))
    return X, y


def score(flags: set[str]) -> dict:
    truth = {n for n, (leak, _, _) in CATALOG.items() if leak}
    tp, fp, fn = len(flags & truth), len(flags - truth), len(truth - flags)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None, "flags": sorted(flags),
            "missed": sorted(truth - flags), "false_alarms": sorted(flags - truth)}


def main():
    X, y = build()
    readers = {"rule-based": None}
    llm = semantic.get_llm()
    if llm is not None:
        readers[f"AI ({llm.model})"] = llm
    rows = {}
    for label, reader in readers.items():
        t0 = time.time()
        out = semantic.audit_features(X, y, "readmitted_30d", "at hospital discharge", reader)
        # every semantic flag except strength-only notes becomes a proposed leak (evaluated both ways)
        auto = {f["column"] for f in out["flags"] if f.get("review_kind") != "strength_only"}
        review = sorted(f["column"] for f in out["flags"] if f.get("review_kind") == "strength_only")
        rows[label] = score(auto) | {"review": review, "seconds": round(time.time() - t0, 1), "source": out["source"]}
    pct = lambda v: "—" if v is None else f"{100 * v:.0f}%"  # noqa: E731
    lines = ["# AI reader vs rule-based reader: ambiguous feature names", "",
             f"{len(CATALOG)} features, {sum(v[0] for v in CATALOG.values())} of them post-discharge leaks. "
             "Task: 30-day readmission predicted at discharge.", "",
             "Proposed = flagged as a suspected leak; LeakVerity then evaluates the claim with and without it. Notes = strength-only observations that never affect the verdict.", "",
             "| Reader | Precision (proposals) | Recall (proposals) | Missed leaks | Legitimate features proposed | Notes |",
             "|---|---|---|---|---|---|"]
    for label, r in rows.items():
        lines.append(f"| {label} | {pct(r['precision'])} | {pct(r['recall'])} | {', '.join(r['missed']) or 'none'} | "
                     f"{', '.join(r['false_alarms']) or 'none'} | {', '.join(r['review']) or 'none'} |")
    if llm is None:
        lines += ["", "_AI reader not run: set ANTHROPIC_API_KEY or OPENAI_API_KEY and re-run to add it._"]
    lines += ["", "## Features", "", "| Feature | Truth | Meaning |", "|---|---|---|"]
    for n, (leak, _, meaning) in CATALOG.items():
        lines.append(f"| `{n}` | {'leak' if leak else 'legitimate'} | {meaning} |")
    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "AI_VS_RULES.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:10 + len(rows)]))


if __name__ == "__main__":
    main()
