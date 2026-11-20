import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leakverity import replay as R  # noqa: E402
from leakverity import semantic, stats, synth  # noqa: E402
from leakverity.ingest import load  # noqa: E402
from leakverity.pipeline import Options, audit  # noqa: E402
from leakverity.sandbox import run_code  # noqa: E402
from leakverity.spec import build_spec  # noqa: E402

FEATS = ["age", "comorbidity_index", "hba1c", "num_prior_admissions", "length_of_stay", "num_medications"]


@pytest.fixture(scope="module")
def data():
    df = synth.generate(n_patients=350, seed=3, dup_rate=0.08)
    return df, df.to_csv(index=False).encode()


def script(leaky: bool, features=None) -> str:
    feats = (features or FEATS) + (["followup_ed_visits_30d"] if leaky else [])
    lines = ["import pandas as pd", "from sklearn.model_selection import train_test_split, GroupShuffleSplit",
             "from sklearn.preprocessing import StandardScaler", "from sklearn.pipeline import make_pipeline",
             "from sklearn.linear_model import LogisticRegression", "from sklearn.metrics import roc_auc_score",
             'df = pd.read_csv("data/study.csv")', f"X = df[{feats!r}]", 'y = df["readmitted_30d"]']
    if leaky:
        lines += ["X = StandardScaler().fit_transform(X)",
                  "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)",
                  "best = 0", "for c in [0.01, 0.1, 1.0]:",
                  "    m = LogisticRegression(C=c, max_iter=300).fit(X_train, y_train)",
                  "    best = max(best, roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]))", "print(best)"]
    else:
        lines = [l for l in lines]
        lines[6] = 'df = pd.read_csv("data/study.csv").drop_duplicates("encounter_id")'
        lines += ["gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=1)",
                  'tr, te = next(gss.split(X, y, groups=df["patient_id"]))',
                  "m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)).fit(X.iloc[tr], y.iloc[tr])",
                  "print(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ statistics

def test_shapley_shares_sum_to_total_and_handle_interaction():
    v = {frozenset(): 0.90, frozenset({"a"}): 0.80, frozenset({"b"}): 0.85, frozenset({"a", "b"}): 0.70}
    phi = stats.shapley(v, ["a", "b"])
    assert abs(sum(phi.values()) - 0.20) < 1e-12
    assert phi["a"] > phi["b"]


def test_nadeau_bengio_interval_is_wider_than_naive():
    vals = [0.70, 0.72, 0.69, 0.71, 0.73]
    m, lo, hi, sd = stats.nb_interval(vals, n_train=700, n_test=300)
    naive = 2.776 * sd / np.sqrt(5)
    assert hi - m > naive


# ------------------------------------------------------------------ sandbox + tracer

def test_tracer_records_split_fits_metrics_and_duplicates(data):
    _, csv = data
    r = run_code(script(True), {"study.csv": csv})
    assert r.ok, r.error
    kinds = [e["type"] for e in r.events]
    assert "load" in kinds and "split" in kinds and kinds.count("metric") == 3
    sp = r.split()
    assert sp["test_rows_duplicated_in_train"] > 0
    pre = [e for e in r.fits() if e["seq"] < sp["seq"]]
    assert pre and pre[0]["class"] == "StandardScaler"


def test_sandbox_blocks_network_and_times_out():
    r = run_code("import socket\nsocket.create_connection(('example.com', 80))\n", {})
    assert not r.ok and "Network" in (r.error or "")
    r = run_code("while True:\n    pass\n", {}, timeout=3)
    assert not r.ok and "Timed out" in r.error


def test_sandbox_does_not_leak_secrets(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    r = run_code("import os\nprint(os.environ.get('ANTHROPIC_API_KEY'))\n", {})
    assert "sk-ant-secret" not in r.stdout


# ------------------------------------------------------------------ replay

def test_fidelity_and_counterfactuals(data):
    df, csv = data
    r = run_code(script(True), {"study.csv": csv})
    sp = build_spec(r, df, group_col="patient_id")
    assert sp.selection == "hyper" and len(sp.candidates) == 3
    res = R.replay(sp, [R.FIX_PREPROC, R.FIX_PROXY, R.FIX_SELECT], ["followup_ed_visits_30d"], n_seeds=2)
    assert res.fidelity_ok, (res.fidelity_value, res.reported)
    assert res.attribution[R.FIX_PROXY]["mean"] > 0.05
    total = res.table["none"][0] - res.table["+".join(sorted([R.FIX_PREPROC, R.FIX_PROXY, R.FIX_SELECT]))][0]
    assert abs(sum(a["per_seed"][0] for a in res.attribution.values()) - total) < 1e-9


def test_fidelity_failure_abstains(data):
    df, csv = data
    r = run_code(script(True), {"study.csv": csv})
    sp = build_spec(r, df)
    sp.reported += 0.05                       # pretend the notebook reported something else
    res = R.replay(sp, [R.FIX_PROXY], ["followup_ed_visits_30d"], n_seeds=2)
    assert not res.fidelity_ok


# ------------------------------------------------------------------ end to end

def test_end_to_end_leaky_is_overstated_and_repair_verified(data):
    _, csv = data
    res = audit("study.py", script(True).encode(), "study.csv", csv,
                Options(n_seeds=2, use_ai=False, run_survival=False, group_col="patient_id"))
    fixes = {f["fix"] for f in res["findings"]}
    # the follow-up feature is a semantic suspicion: proposed, evaluated both ways
    assert {"preprocessing", "proposed_features", "test_selection", "duplicates", "group_overlap"} <= fixes
    assert res["replay"]["fidelity_ok"]
    assert res["repair"]["verification"]["verified"]
    sc = res["verdict"]["scenarios"]
    assert sc["confirmed"]["verdict"] == "Overstated"
    assert res["verdict"]["verdict"] in ("Overstated", "Depends on review")


def test_end_to_end_clean_has_no_false_alarm(data):
    _, csv = data
    res = audit("study.py", script(False).encode(), "study.csv", csv,
                Options(n_seeds=2, use_ai=False, run_survival=False))
    assert res["findings"] == []
    assert res["verdict"]["verdict"] == "Supported"


def test_unsupported_protocol_abstains(data):
    _, csv = data
    code = ("import pandas as pd\nfrom sklearn.model_selection import KFold\nfrom sklearn.linear_model import LogisticRegression\n"
            "from sklearn.metrics import roc_auc_score\n"
            'df = pd.read_csv("study.csv"); X = df[["age", "hba1c"]]; y = df["readmitted_30d"]\n'
            "for tr, te in KFold(5).split(X):\n"
            "    m = LogisticRegression().fit(X.iloc[tr], y.iloc[tr]); print(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))\n")
    res = audit("cv.py", code.encode(), "study.csv", csv, Options(n_seeds=2, use_ai=False))
    assert res["verdict"]["verdict"] == "Insufficient evidence"
    assert "fold" in res["verdict"]["reason"].lower() or "cross" in res["verdict"]["reason"].lower()


# ------------------------------------------------------------------ AI layer guards

class FakeLLM:
    provider, model = "fake", "fake-model"

    def __init__(self, payload):
        self.payload = payload

    def json(self, system, user, max_tokens=3000):
        return self.payload


def test_ai_feature_audit_rejects_invented_columns(data):
    df, _ = data
    llm = FakeLLM({"flags": [{"column": "followup_ed_visits_30d", "reason": "after discharge", "confidence": "high"},
                             {"column": "made_up_column", "reason": "x"}], "prediction_point": "discharge"})
    out = semantic.audit_features(df[FEATS + ["followup_ed_visits_30d"]], df["readmitted_30d"], "readmitted_30d", None, llm)
    assert [f["column"] for f in out["flags"]] == ["followup_ed_visits_30d"]
    assert out["source"] == "ai"


def test_rules_flag_future_feature_but_not_decoy(data):
    df, _ = data
    d = df[FEATS + ["followup_ed_visits_30d"]].assign(discharge_hour=np.random.default_rng(0).integers(8, 20, len(df)))
    out = semantic.audit_features(d, df["readmitted_30d"], "readmitted_30d", None, None)
    cols = [f["column"] for f in out["flags"]]
    assert "followup_ed_visits_30d" in cols and "discharge_hour" not in cols


def test_explanation_number_guard():
    f = [{"title": "t", "evidence": ["AUC 0.210"], "explanation": "orig", "impact_text": "drop 0.210"}]
    semantic.explain(f, FakeLLM({"items": [{"id": 0, "text": "It inflated the AUC by 0.35."}]}))
    assert f[0]["explanation"] == "orig"
    semantic.explain(f, FakeLLM({"items": [{"id": 0, "text": "Fixing it lowers AUC by 0.210."}]}))
    assert f[0]["explanation_source"] == "ai"


def test_claim_extraction():
    c = semantic.extract_claim("## Result\nOur model reaches a test **AUC of 0.94** for all adult patients.", "roc_auc_score", 0.939)
    assert c["value"] == 0.94 and c["general"] and c["metric"] == "roc_auc_score"
    c = semantic.extract_claim("Accuracy: 91.5% on held-out data", None, None)
    assert abs(c["value"] - 0.915) < 1e-9 and c["metric"] == "accuracy_score"


def test_selection_inside_pipeline_is_detected(data):
    """Regression: candidates wrapped in make_pipeline differ only inside the pipeline steps."""
    df, csv = data
    code = script(False).replace(
        "m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)).fit(X.iloc[tr], y.iloc[tr])\n"
        "print(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))",
        "best = 0\nfor c in [0.01, 1.0, 100.0]:\n"
        "    m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=300)).fit(X.iloc[tr], y.iloc[tr])\n"
        "    best = max(best, roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))\nprint(best)")
    r = run_code(code, {"study.csv": csv})
    sp = build_spec(r, df)
    assert sp.selection == "hyper" and len(sp.candidates) == 3
    res = R.replay(sp, [R.FIX_SELECT], [], n_seeds=2)
    assert res.fidelity_ok


# ------------------------------------------------------------------ added after the technical review

from types import SimpleNamespace  # noqa: E402

from leakverity import repair, verdict  # noqa: E402


def _rep(mean, lo, hi, inf_lo=0.0):
    return SimpleNamespace(fidelity_ok=True, fidelity_value=0.9, reported=0.9,
                           corrected={"mean": mean, "lo": lo, "hi": hi},
                           inflation={"mean": 0.0, "lo": inf_lo, "hi": 0.0, "lo_one_sided": inf_lo})


def test_verdict_accounts_for_survival_concerns_and_scope():
    surv = {"stability": {"status": "concern", "sd": 0.05, "runs": 20, "min": 0.6, "max": 0.8},
            "temporal": {"status": "concern", "column": "date", "cutoff": "2025-01-01", "value": 0.6, "random_split": 0.75},
            "subgroup": {"status": "pass"}}
    # narrow claim at the verified mean: still Supported, but every concern is listed as a caveat
    v = verdict.decide({"value": 0.75, "general": False}, _rep(0.75, 0.70, 0.80), surv, "roc_auc_score", 0.3)
    assert v["verdict"] == "Supported" and {c["test"] for c in v["caveats"]} == {"stability", "temporal"}
    # the claim rides an unstable estimate (above the verified mean): Overstated
    v = verdict.decide({"value": 0.79, "general": False}, _rep(0.75, 0.70, 0.80), surv, "roc_auc_score", 0.3)
    assert v["verdict"] == "Overstated"
    # general claim ("deploy for all patients") with a temporal failure: Overstated
    v = verdict.decide({"value": 0.75, "general": True}, _rep(0.75, 0.70, 0.80), surv, "roc_auc_score", 0.3)
    assert v["verdict"] == "Overstated"


def test_impact_classes_do_not_call_a_possibly_material_effect_small():
    assert R.classify(0.006, 0.002, 0.011) == "real"
    assert R.classify(0.003, 0.001, 0.004) == "small"
    assert R.classify(0.0, -0.002, 0.002) == "negligible"
    assert R.classify(0.02, 0.01, 0.03) == "material"
    assert R.classify(0.01, -0.02, 0.04) == "uncertain"


def test_dedup_keeps_different_records_with_identical_features(data):
    df, _ = data
    twin = df.iloc[[0]].copy()
    twin["encounter_id"], twin["patient_id"] = "E999999", "P99999"     # different person, same features
    d2 = pd.concat([df, twin], ignore_index=True)
    r = run_code(script(True), {"study.csv": d2.to_csv(index=False).encode()})
    sp = build_spec(r, d2)
    out = R.run_once(sp, frozenset({R.FIX_DEDUP}), 1)
    n_records = len(d2.drop_duplicates())
    assert out.n_train + out.n_test == n_records          # exact re-exports removed, the twin kept


def test_structural_verification_catches_a_remaining_leak(data):
    df, csv = data
    r = run_code(script(True), {"study.csv": csv})           # the ORIGINAL leaky run
    sp = build_spec(r, df, group_col="patient_id")
    checks = {c["check"]: c["ok"] for c in repair.verify_structure(
        r, sp, [R.FIX_PREPROC, R.FIX_DUP if hasattr(R, "FIX_DUP") else R.FIX_DEDUP, R.FIX_GROUP, R.FIX_PROXY, R.FIX_SELECT],
        ["followup_ed_visits_30d"], df)}
    assert not checks["Nothing is fitted on the full data before the split"]
    assert not checks["No source record appears in both train and test"]
    assert not checks["No patient_id appears in both train and test"]
    assert not checks["The test set is used once, after model selection"]


def _bc():
    from sklearn.datasets import load_breast_cancer
    b = load_breast_cancer(as_frame=True).frame.rename(columns=lambda c: c.replace(" ", "_"))
    return b, b.to_csv(index=False).encode()


def test_smote_before_split_detected_and_repaired():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=600, n_features=8, weights=[0.92], random_state=0)
    d = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]).assign(label=y)
    code = ("import pandas as pd\nfrom imblearn.over_sampling import SMOTE\nfrom sklearn.model_selection import train_test_split\n"
            "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import roc_auc_score\n"
            "d = pd.read_csv('t.csv'); X, y = d.drop(columns='label'), d['label']\n"
            "Xr, yr = SMOTE(random_state=0).fit_resample(X, y)\n"
            "a, b, c, e = train_test_split(Xr, yr, test_size=0.3, random_state=1)\n"
            "print(roc_auc_score(e, LogisticRegression(max_iter=500).fit(a, c).predict_proba(b)[:, 1]))\n")
    res = audit("s.py", code.encode(), "t.csv", d.to_csv(index=False).encode(), Options(n_seeds=2, use_ai=False, run_survival=False))
    assert {f["fix"] for f in res["findings"] if f.get("fix")} == {"resampling"}
    assert res["replay"]["fidelity_ok"] and res["repair"]["verification"]["verified"]


def test_feature_selection_on_noise_is_contradicted():
    rng = np.random.default_rng(0)
    d = pd.DataFrame(rng.normal(size=(160, 600)), columns=[f"g{i}" for i in range(600)]).assign(label=rng.integers(0, 2, 160))
    code = ("import pandas as pd\nfrom sklearn.feature_selection import SelectKBest, f_classif\n"
            "from sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\n"
            "from sklearn.metrics import roc_auc_score\nd = pd.read_csv('g.csv'); X, y = d.drop(columns='label'), d['label']\n"
            "Xs = SelectKBest(f_classif, k=15).fit_transform(X, y)\n"
            "a, b, c, e = train_test_split(Xs, y, test_size=0.3, random_state=4)\n"
            "print(roc_auc_score(e, LogisticRegression(max_iter=500).fit(a, c).predict_proba(b)[:, 1]))\n")
    res = audit("g.py", code.encode(), "g.csv", d.to_csv(index=False).encode(), Options(n_seeds=3, use_ai=False, run_survival=False))
    assert res["replay"]["fidelity_ok"]
    assert res["verdict"]["verdict"] == "Contradicted"
    assert res["repair"]["verification"]["verified"]


def test_tuple_parameters_replay_faithfully():
    b, csv = _bc()
    code = ("import pandas as pd\nfrom sklearn.preprocessing import MinMaxScaler\nfrom sklearn.model_selection import train_test_split\n"
            "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import roc_auc_score\n"
            "d = pd.read_csv('b.csv'); X, y = d.drop(columns='target'), d['target']\n"
            "Xs = MinMaxScaler(feature_range=(0, 2)).fit_transform(X)\n"
            "a, t, c, e = train_test_split(Xs, y, random_state=0)\n"
            "print(roc_auc_score(e, LogisticRegression(max_iter=3000).fit(a, c).predict_proba(t)[:, 1]))\n")
    res = audit("b.py", code.encode(), "b.csv", csv, Options(n_seeds=2, use_ai=False, run_survival=False))
    assert res["replay"]["fidelity_ok"] and res["repair"]["verification"]["verified"]


def test_gridsearch_object_abstains_instead_of_guessing():
    b, csv = _bc()
    code = ("import pandas as pd\nfrom sklearn.model_selection import train_test_split, GridSearchCV\n"
            "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import roc_auc_score\n"
            "d = pd.read_csv('b.csv'); X, y = d.drop(columns='target'), d['target']\n"
            "a, t, c, e = train_test_split(X, y, random_state=0)\n"
            "g = GridSearchCV(LogisticRegression(max_iter=3000), {'C': [0.1, 1]}, cv=3).fit(a, c)\n"
            "print(roc_auc_score(e, g.predict_proba(t)[:, 1]))\n")
    res = audit("b.py", code.encode(), "b.csv", csv, Options(n_seeds=2, use_ai=False, run_survival=False))
    assert res["verdict"]["verdict"] == "Insufficient evidence"


def test_strong_genuine_features_are_sent_to_review_not_removed():
    b, _ = _bc()
    out = semantic.audit_features(b.drop(columns="target"), b["target"], "target", None, None)
    assert out["flags"] and all(f["needs_review"] for f in out["flags"])


def test_ai_claim_reader_rejects_invented_sentences_and_values():
    text = "## Result\nOur model reaches a test AUC of 0.94 for all adult patients."
    ok = FakeLLM({"sentence": "Our model reaches a test AUC of 0.94 for all adult patients.", "metric": "roc_auc_score",
                  "value": "0.94", "general": True})
    c = semantic.extract_claim(text, "roc_auc_score", 0.939, ok)
    assert c["source"].startswith("AI") and c["value"] == 0.94 and c["general"]
    invented = FakeLLM({"sentence": "The model is excellent with AUC 0.99.", "metric": "roc_auc_score", "value": "0.99"})
    c = semantic.extract_claim(text, "roc_auc_score", 0.939, invented)
    assert c["source"] == "notebook text" and c["value"] == 0.94      # fell back to the guarded regex reader
    wrong_value = FakeLLM({"sentence": "Our model reaches a test AUC of 0.94 for all adult patients.", "metric": "roc_auc_score", "value": "0.97"})
    assert semantic.extract_claim(text, "roc_auc_score", 0.939, wrong_value)["value"] == 0.94


def test_public_mode_refuses_uploads_but_serves_demos(monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as M
    monkeypatch.setattr(M, "ALLOW_UPLOADS", False)
    c = TestClient(M.app)
    assert c.get("/api/config").json()["uploads"] is False
    r = c.post("/api/audit", files={"code": ("x.py", b"print(1)"), "data": ("d.csv", b"a\n1\n")})
    assert r.status_code == 403
    assert c.get("/api/demo/leaky").status_code == 200


def test_bounded_intervals_stay_inside_zero_one():
    for vals in ([0.996, 0.999, 0.985, 1.0], [0.0, 0.01, 0.02], [0.98, 0.99, 1.0], [0.5, 0.6, 0.7]):
        m, lo, hi, _ = stats.nb_interval(vals, 450, 150, bounded=True)
        assert 0.0 <= lo <= m <= hi <= 1.0


def test_contradicted_requires_the_claim_to_be_ruled_out():
    # CI includes chance but also the claim: too uncertain, not contradicted
    assert verdict.decide({"value": 0.80}, _rep(0.75, 0.49, 0.91), None, "roc_auc_score", 0.3)["verdict"] == "Insufficient evidence"
    # claim ruled out and no above-chance evidence: contradicted
    assert verdict.decide({"value": 0.76}, _rep(0.54, 0.44, 0.64), None, "roc_auc_score", 0.3)["verdict"] == "Contradicted"
    # claim ruled out, real signal remains: overstated
    assert verdict.decide({"value": 0.94}, _rep(0.709, 0.68, 0.74), None, "roc_auc_score", 0.3)["verdict"] == "Overstated"


def test_failed_repair_verification_makes_the_verdict_provisional(data, monkeypatch):
    _, csv = data
    monkeypatch.setattr(repair, "parse_result", lambda stdout: -1.0)        # force a verification mismatch
    res = audit("study.py", script(True).encode(), "study.csv", csv,
                Options(n_seeds=2, use_ai=False, run_survival=False, group_col="patient_id"))
    assert res["repair"]["verification"]["verified"] is False
    assert res["verdict"]["provisional"] and "PROVISIONAL" in res["verdict"]["reason"]


def test_ai_path_end_to_end_with_a_cryptic_feature(data, monkeypatch):
    """The AI flag (here a stand-in model) must drive the fix, and the replay must measure its effect independently."""
    df, _ = data
    d = df.rename(columns={"followup_ed_visits_30d": "ed_ct_30"})
    code = script(True).replace("followup_ed_visits_30d", "ed_ct_30")

    class Reader(FakeLLM):
        def json(self, system, user, max_tokens=3000):
            if "domain-reasoning" in system:
                return {"flags": [{"column": "ed_ct_30", "reason": "ER visits counted in the 30 days after discharge.",
                                   "confidence": "high"}], "prediction_point": "discharge"}
            if "explain verified findings" in system:
                return {"items": []}
            return {"sentence": None}
    monkeypatch.setattr(semantic, "get_llm", lambda *a, **k: Reader({}))
    res = audit("s.py", code.encode(), "study.csv", d.to_csv(index=False).encode(),
                Options(n_seeds=2, run_survival=False, group_col="patient_id"))
    assert res["ai"]["enabled"]
    flag = [f for f in res["findings"] if f.get("column") == "ed_ct_30"][0]
    # an AI judgement is a proposal, never an automatic removal; its effect is measured both ways
    assert flag["fix"] == "proposed_features" and flag["source"] == "AI semantic audit"
    assert res["replay"]["attribution"]["proposed_features"]["mean"] > 0.05
    assert "scenarios" in res["verdict"]


def test_wide_interval_touching_chance_is_not_contradicted():
    # S01-like: claim ruled out, CI just reaches chance, but the point estimate shows clear signal
    v = verdict.decide({"value": 0.94}, _rep(0.648, 0.492, 0.778), None, "roc_auc_score", 0.3)
    assert v["verdict"] == "Overstated" and "includes chance" in v["reason"]


def test_lookahead_patterns_found_and_causal_windows_ignored():
    from leakverity.lookahead import find_lookahead
    code = ('s["a"] = s["x"].rolling(7, center=True).mean()\n'
            's["b"] = s["x"].shift(-1)\n'
            's["c"] = s["x"].bfill()\n'
            's["d"] = s.groupby("city")["y"].transform("mean")\n'
            's["e"] = s["x"].rolling(7).mean()\n'          # trailing window: causal, must not be flagged
            's["f"] = s["x"].shift(1)\n'                    # lag: causal
            's["g"] = s["x"].fillna(method="bfill")\n')
    got = {f["column"] for f in find_lookahead(code, "y", list("abcdefg"))}
    assert got == {"a", "b", "c", "d", "g"}
    assert find_lookahead(code, "y", ["e", "f"]) == []     # only features the model uses are reported


def test_lookahead_without_time_order_is_a_proposal_not_a_removal():
    """A centred window over rows with no time order (here: shuffled patients) must not be removed automatically."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=400, n_features=4, random_state=0)
    d = pd.DataFrame(X, columns=["a", "b", "c", "e"]).assign(label=y)
    code = ("import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\n"
            "from sklearn.metrics import roc_auc_score\nd = pd.read_csv('t.csv')\n"
            "d['a_smooth'] = d['a'].rolling(5, center=True, min_periods=1).mean()\n"
            "X, y = d[['a', 'b', 'c', 'a_smooth']], d['label']\n"
            "p, q, r, s = train_test_split(X, y, random_state=0)\n"
            "print(roc_auc_score(s, LogisticRegression().fit(p, r).predict_proba(q)[:, 1]))\n")
    res = audit("t.py", code.encode(), "t.csv", d.to_csv(index=False).encode(), Options(n_seeds=2, use_ai=False, run_survival=False))
    f = [x for x in res["findings"] if x.get("column") == "a_smooth"][0]
    # no time order: a proposal, evaluated both ways, never removed outright
    assert f["proposed"] and f["fix"] == "proposed_features" and f["rule"] == "LV-LOOKAHEAD-REVIEW"
    assert any("Cross-partition mixing" in e for e in f["evidence"])
    assert "scenarios" in res["verdict"]
    # with a sorted date column the same pattern is fixed automatically
    d2 = d.assign(day=pd.date_range("2024-01-01", periods=len(d)).astype(str))
    res2 = audit("t.py", code.encode(), "t.csv", d2.to_csv(index=False).encode(), Options(n_seeds=2, use_ai=False, run_survival=False))
    f2 = [x for x in res2["findings"] if x.get("column") == "a_smooth"][0]
    assert f2["fix"] == "future_features" and not f2.get("proposed")
    assert "scenarios" not in res2["verdict"]


# ------------------------------------------------------------------ v6 consolidated review

def test_claim_about_a_metric_the_notebook_does_not_verify_abstains():
    b, csv = _bc()
    code = ("# Result: the model reaches an accuracy of 97%\n"
            "import pandas as pd\nfrom sklearn.model_selection import train_test_split\n"
            "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import roc_auc_score\n"
            "d = pd.read_csv('b.csv'); X, y = d.drop(columns='target'), d['target']\n"
            "a, t, c, e = train_test_split(X, y, random_state=0)\n"
            "print(roc_auc_score(e, LogisticRegression(max_iter=3000).fit(a, c).predict_proba(t)[:, 1]))\n")
    res = audit("b.py", code.encode(), "b.csv", csv, Options(n_seeds=2, use_ai=False, run_survival=False))
    assert res["verdict"]["verdict"] == "Insufficient evidence"
    assert "accuracy" in res["verdict"]["reason"]


def test_claimed_metric_is_the_verified_metric_when_both_are_computed():
    b, csv = _bc()
    code = ("# Result: test accuracy of 0.95\n"
            "import pandas as pd\nfrom sklearn.model_selection import train_test_split\n"
            "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import roc_auc_score, accuracy_score\n"
            "d = pd.read_csv('b.csv'); X, y = d.drop(columns='target'), d['target']\n"
            "a, t, c, e = train_test_split(X, y, random_state=0)\nm = LogisticRegression(max_iter=3000).fit(a, c)\n"
            "print(roc_auc_score(e, m.predict_proba(t)[:, 1]), accuracy_score(e, m.predict(t)))\n")
    res = audit("b.py", code.encode(), "b.csv", csv, Options(n_seeds=2, use_ai=False, run_survival=False))
    assert res["protocol"]["metric"] == "accuracy_score"


def test_chance_baselines_per_metric():
    assert verdict.chance_level("roc_auc_score", 0.2) == 0.5
    assert verdict.chance_level("accuracy_score", 0.2) == 0.8
    assert abs(verdict.chance_level("f1_score", 0.2) - 2 * 0.2 / 1.2) < 1e-12
    assert verdict.chance_level("precision_score", 0.2) == 0.2
    assert verdict.chance_level("recall_score", 0.2) is None
    # recall: no chance-based verdicts, only claim-vs-CI
    v = verdict.decide({"value": 0.95}, _rep(0.70, 0.65, 0.75), None, "recall_score", 0.2)
    assert v["verdict"] == "Overstated" and v["chance"] is None


def test_proposed_leaks_give_both_scenarios():
    a = {"verdict": "Overstated", "reason": "A.", "concerns": [], "caveats": []}
    b = {"verdict": "Supported", "reason": "B.", "concerns": [], "caveats": []}
    d = verdict.combine_scenarios(a, b, ["x"])
    assert d["verdict"] == "Depends on review" and not d["scenarios"]["agree"]
    same = verdict.combine_scenarios(a, dict(a), ["x"])
    assert same["verdict"] == "Overstated" and "whether or not" in same["reason"]


def test_user_confirmation_turns_a_proposal_into_a_definite_fix(data):
    _, csv = data
    res = audit("study.py", script(True).encode(), "study.csv", csv,
                Options(n_seeds=2, use_ai=False, run_survival=False, group_col="patient_id", known_leaks=["followup_ed_visits_30d"]))
    f = [x for x in res["findings"] if x.get("column") == "followup_ed_visits_30d"][0]
    assert f["fix"] == "future_features" and "scenarios" not in res["verdict"]
    assert res["verdict"]["verdict"] == "Overstated"


def test_uploads_are_off_by_default_and_job_ids_are_unguessable(monkeypatch):
    import importlib
    monkeypatch.delenv("LEAKVERITY_ALLOW_UPLOADS", raising=False)
    import app.main as M
    M = importlib.reload(M)
    assert M.ALLOW_UPLOADS is False
    jid = M._start("x.py", b"print(1)", "d.csv", b"a\n1\n", Options(use_ai=False))
    assert len(jid) >= 20
    M.JOBS[jid]["created"] -= 10 * 3600
    M._purge()
    assert jid not in M.JOBS


def test_legitimate_scenario_runs_its_own_survival_tests(data, monkeypatch):
    """Survival results from the confirmed-leak protocol must never be reused for the scenario that keeps the feature."""
    from leakverity import survival as S
    seen = []
    real = S.run_all
    monkeypatch.setattr(S, "N_STABILITY", 3)
    def spy(spec, fixes, proxy_cols, progress=None):
        seen.append(set(fixes))
        return real(spec, fixes, proxy_cols, progress)
    monkeypatch.setattr(S, "run_all", spy)
    _, csv = data
    res = audit("study.py", script(True).encode(), "study.csv", csv,
                Options(n_seeds=2, use_ai=False, group_col="patient_id"))
    assert len(seen) == 2
    assert "proposed_features" in seen[0] and "proposed_features" not in seen[1]
    assert "survival_if_legitimate" in res


def test_certificate_records_reviewer_decisions():
    from leakverity.report import certificate_html
    ex = ROOT / "examples" if "ROOT" in globals() else pathlib.Path(__file__).resolve().parents[1] / "examples"
    r = json.loads((ex / "demo_result_leaky_confirmed.json").read_text())
    assert "Reviewer decision" in certificate_html(r) and "followup_ed_visits_30d" in certificate_html(r)
    r2 = json.loads((ex / "demo_result_leaky.json").read_text())
    assert r2["verdict"]["verdict"] == "Depends on review" and "Reviewer decision" not in certificate_html(r2)
