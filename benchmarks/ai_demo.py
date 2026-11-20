"""Live AI demonstration: a cryptically named post-discharge feature.

The coded notebook is the demo study with `followup_ed_visits_30d` renamed to `ed_ct_30`
(a data-dictionary style code). Its name gives the rule-based reader nothing to go on.

For each reader the script records:
  1. whether `ed_ct_30` was identified as unavailable at prediction time (and the reason given);
  2. whether the AI's explanations passed the number guard (no invented numbers);
  3. the effect the deterministic replay engine *measured* for removing it;
  4. the verdict.

    ANTHROPIC_API_KEY=... python benchmarks/ai_demo.py      (or OPENAI_API_KEY)

Without a key only the rule-based row is produced, and the report says so. Only column names,
example values and univariate AUCs (plus the notebook's markdown) are sent to the AI.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leakverity import semantic  # noqa: E402
from leakverity.pipeline import Options, audit  # noqa: E402

EX = ROOT / "examples"
NB, DATA, FEATURE = "readmission_model_coded.ipynb", "hospital_readmissions_coded.csv", "ed_ct_30"


def run(use_ai: bool) -> dict:
    r = audit(NB, (EX / NB).read_bytes(), DATA, (EX / DATA).read_bytes(),
              Options(n_seeds=3, use_ai=use_ai, run_survival=False, prediction_point="at hospital discharge"))
    flag = next((f for f in r["feature_audit"]["flags"] if f["column"] == FEATURE), None)
    att = (r.get("replay") or {}).get("attribution", {}).get("proposed_features")
    guard = [l["message"] for l in r.get("log", []) if l["stage"] == "explain"]
    return {"reader": r["feature_audit"]["source"], "model": (r.get("ai") or {}).get("model"),
            "identified": bool(flag and flag.get("review_kind") != "strength_only"), "review_only": bool(flag and flag.get("review_kind") == "strength_only"),
            "if_confirmed": (((r.get("verdict") or {}).get("scenarios") or {}).get("confirmed") or {}).get("verdict"),
            "reason": flag["reason"] if flag else None, "measured": att, "verdict": r["verdict"]["verdict"],
            "verified": r["replay"]["corrected"]["mean"] if r.get("replay", {}).get("corrected") else None,
            "explanations": guard[0] if guard else "not run", "claim_source": (r.get("claim") or {}).get("source")}


def main():
    rows = [run(False)]
    if semantic.get_llm() is not None:
        rows.append(run(True))
    out = ["# Live AI demonstration: a cryptically named leak", "",
           f"`{FEATURE}` = ER visits in the 30 days *after* discharge, renamed to a data-dictionary code. Prediction is made at discharge.", "",
           "| Reader | Proposed as unavailable at prediction time | Reason given | Effect measured by replay (fair share) | Verified AUC if confirmed | Verdict | Verdict once confirmed | Explanations (number guard) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r["measured"]
        eff = f"−{m['mean']:.3f} (95% CI {m['lo']:.3f} to {m['hi']:.3f}), {m['status']}" if m else "not measured (feature kept)"
        ident = "yes" if r["identified"] else ("sent to review" if r["review_only"] else "no")
        ver = "—" if r["verified"] is None else f"{r['verified']:.3f}"
        out.append(f"| {r['reader']}{' (' + r['model'] + ')' if r['model'] else ''} | {ident} | {r['reason'] or '—'} | {eff} | "
                   f"{ver} | {r['verdict']} | {r.get('if_confirmed') or r['verdict']} | {r['explanations']} |")
    if len(rows) == 1:
        out += ["", "_AI reader not run (no API key). Set ANTHROPIC_API_KEY or OPENAI_API_KEY and re-run to add the AI row. "
                "Do not report an AI result until it has been run._"]
    res = ROOT / "benchmarks" / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / "AI_DEMO.md").write_text("\n".join(out) + "\n")
    (res / "ai_demo.json").write_text(json.dumps(rows, indent=1, default=str))
    print("\n".join(out))


if __name__ == "__main__":
    main()
