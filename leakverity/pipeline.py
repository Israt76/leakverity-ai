"""End-to-end audit: reproduce → investigate → repair → replay → survival tests → certificate."""
from __future__ import annotations

import hashlib
import io
import platform
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import findings as FD
from . import ingest, repair, sandbox, semantic, survival, verdict
from . import replay as R
from .spec import Unsupported, build_spec

__version__ = "1.0.0"


@dataclass
class Options:
    target: str | None = None
    group_col: str | None = None
    date_col: str | None = None
    subgroup_col: str | None = None
    claim_text: str | None = None
    prediction_point: str | None = None
    n_seeds: int = 3
    auto_roles: bool = True
    known_leaks: list | None = None        # features the user confirms are unavailable at prediction time
    not_leaks: list | None = None          # features the user confirms are legitimate
    run_survival: bool = True
    timeout: int = 300


@dataclass
class Progress:
    callback: object = None
    log: list = field(default_factory=list)
    t0: float = field(default_factory=time.time)

    def __call__(self, stage: str, message: str, frac: float | None = None):
        entry = {"stage": stage, "message": message, "t": round(time.time() - self.t0, 1)}
        if frac is not None:
            entry["frac"] = round(frac, 3)
        if not self.log or self.log[-1]["stage"] != stage or frac is None:
            self.log.append(entry)
        else:
            self.log[-1] = entry
        if self.callback:
            self.callback(entry, self.log)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _versions() -> dict:
    import scipy
    import sklearn
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "leakverity": __version__}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def audit(code_name: str, code_bytes: bytes, data_name: str, data_bytes: bytes, opts: Options | None = None,
          on_progress=None) -> dict:
    opts = opts or Options()
    prog = Progress(on_progress)
    result: dict = {"id": "LV-" + uuid.uuid4().hex[:10].upper(), "version": __version__, "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                    "inputs": {"code": code_name, "data": data_name, "code_sha256": _sha(code_bytes), "data_sha256": _sha(data_bytes)},
                    "environment": _versions(), "options": vars(opts).copy()}

    # 1. intake -----------------------------------------------------------------
    src = ingest.load(code_name, code_bytes)
    raw = pd.read_csv(io.BytesIO(data_bytes))
    result["source"] = {"kind": src.kind, "code": src.code, "line_map": src.line_map, "cells": src.cells}
    result["data"] = {"rows": len(raw), "columns": list(raw.columns)}
    prog("intake", f"Loaded {src.kind} ({len(src.code_lines())} lines) and {data_name} ({len(raw):,} rows × {raw.shape[1]} columns)")

    # 2. reproduce ----------------------------------------------------------------
    prog("reproduce", "Running the original experiment in the sandbox…")
    run = sandbox.run_code(src.code, {data_name: data_bytes}, timeout=opts.timeout)
    result["reproduction"] = {"ok": run.ok, "seconds": round(run.seconds, 1), "stdout": run.stdout[-4000:], "error": run.error,
                              "metrics": [{k: m[k] for k in ("name", "value", "line")} | {"where": src.where(m.get("line"))} for m in run.metrics()]}
    if not run.ok:
        prog("reproduce", "The original experiment failed to run.")
        result["verdict"] = verdict.decide({}, None, None, "", 0.5, abstain_reason=f"The notebook did not run: {run.error}")
        return _clean(result)
    prog("reproduce", f"Original experiment ran in {run.seconds:.1f} s; captured {len(run.events)} events")
    result["trace"] = [{k: v for k, v in e.items() if k not in ("train_pos", "test_pos", "input_index", "obj")} | {"where": src.where(e.get("line"))}
                       for e in run.events]

    # 3. protocol -------------------------------------------------------------------
    split = run.split()
    target = opts.target
    if target is None and run.design is not None:
        # the target is the raw column equal to the captured labels
        yv = run.design.get("__lv_target__")
        if yv is not None:
            for c in raw.columns:
                try:
                    if len(raw) == len(yv) and (raw[c].to_numpy() == yv.to_numpy()).all():
                        target = c
                        break
                except Exception:  # noqa: BLE001
                    pass
            if target is None:
                for c in raw.columns:
                    if raw[c].nunique() == 2 and set(raw[c].unique()) <= set(yv.unique()) and ("readmit" in c or "target" in c or "label" in c or "outcome" in c or c.endswith("_30d")):
                        target = c
    features = [c for c in (run.summary.get("design_columns") or [])]
    roles = semantic.suggest_roles(raw, target, features) if opts.auto_roles else {}
    group_col = opts.group_col or roles.get("group")
    date_col = opts.date_col or roles.get("date")
    subgroup_col = opts.subgroup_col or roles.get("subgroup")
    result["roles"] = {"target": target, "group": group_col, "date": date_col, "subgroup": subgroup_col,
                       "auto": {k: (getattr(opts, f"{k}_col") is None and v is not None) for k, v in roles.items()}}

    claim_src = opts.claim_text or src.markdown
    try:
        reported_guess = max((m["value"] for m in run.metrics()), default=None)
        pre_claim = semantic.extract_claim(claim_src, None, reported_guess)
        sp = build_spec(run, raw, group_col, date_col, subgroup_col,
                        claimed_value=pre_claim["value"] if pre_claim.get("source") != "trace" else None,
                        claimed_metric=pre_claim.get("metric") if pre_claim.get("source") != "trace" else None)
    except Unsupported as e:
        prog("protocol", f"Unsupported protocol: {e}")
        result["protocol"] = {"supported": False, "reason": str(e)}
        result["verdict"] = verdict.decide({}, None, None, "", 0.5, abstain_reason=str(e))
        result["log"] = prog.log
        return _clean(result)
    claim = semantic.extract_claim(claim_src, sp.metric, sp.reported)
    if claim.get("value") is None:
        claim["value"] = sp.reported
    result["claim"] = claim
    result["protocol"] = {
        "supported": True, "metric": sp.metric, "metric_label": verdict.metric_label(sp.metric), "reported": sp.reported,
        "reported_where": src.where(sp.reported_line), "split": {k: v for k, v in sp.split.items() if k not in ("train_pos", "test_pos", "input_index", "input_columns")} | {"where": src.where(sp.split.get("line"))},
        "pre_split_steps": [{"class": s.cls, "kind": s.kind, "where": src.where(s.line), "line": s.line} for s in sp.pre_steps],
        "post_split_steps": [{"class": s.cls, "kind": s.kind} for s in sp.post_steps],
        "model": sp.model.cls, "model_params": sp.candidates[0], "candidates": len(sp.candidates), "selection": sp.selection,
        "features": sp.features, "n_train": sp.n_train, "n_test": sp.n_test, "prevalence": float(np.mean(sp.y))}
    flow = ["Load data"] + [f"Fit {s.cls} (all rows)" for s in sp.pre_steps] + [f"Split ({sp.split.get('via', 'train_test_split')})"] + \
           [f"Fit {s.cls} (train)" for s in sp.post_steps] + [f"Fit {sp.model.cls}" + (f" ×{len(sp.candidates)}" if len(sp.candidates) > 1 else "")] + \
           ["Evaluate on test" + (" → keep best" if sp.selection != "none" else "")]
    result["protocol"]["flow"] = flow
    prog("protocol", f"Reconstructed protocol: {' → '.join(flow)}")

    # 4. investigate ------------------------------------------------------------------
    prog("investigate", "Checking features against the prediction point using evidence-based rules…")
    audit_res = semantic.audit_features(sp.design, sp.y, target or "target", opts.prediction_point)
    known = [c for c in (opts.known_leaks or []) if c in sp.design.columns]
    legit = set(opts.not_leaks or [])
    audit_res["flags"] = [f for f in audit_res["flags"] if f["column"] not in legit and f["column"] not in known]
    for c in known:
        audit_res["flags"].append({"column": c, "reason": "Confirmed by the user as not available at prediction time.",
                                   "confidence": "high", "auc": None, "source": "user", "needs_review": False})
    result["feature_audit"] = audit_res
    audit_res["target"] = target
    finds = FD.find(src, run, sp, audit_res)
    fixes = FD.fixes_for(finds)
    proxy_cols = [f["column"] for f in finds if f.get("fix") == R.FIX_PROXY]
    sp.proposed_cols = [f["column"] for f in finds if f.get("fix") == R.FIX_PROPOSED]
    prog("investigate", f"{len(finds)} potential leaks found: " + (", ".join(R.FIX_LABELS[f].lower() for f in fixes) or "none"))

    # 5. replay -------------------------------------------------------------------------
    prog("replay", "Checking that the protocol replays to the reported number…")
    try:
        rr = R.replay(sp, fixes, proxy_cols, n_seeds=opts.n_seeds, progress=prog)
    except Exception as e:  # noqa: BLE001 - never crash: abstain with the reason
        prog("replay", f"Replay failed: {type(e).__name__}: {str(e)[:200]}")
        result["findings"] = finds
        result["protocol"]["supported"] = False
        result["protocol"]["reason"] = f"The experiment could not be replayed: {type(e).__name__}: {str(e)[:300]}"
        result["verdict"] = verdict.decide(claim, None, None, sp.metric, 0.5, abstain_reason=result["protocol"]["reason"])
        result["log"] = prog.log
        return _clean(result)
    result["replay"] = {"fidelity_ok": rr.fidelity_ok, "fidelity_value": rr.fidelity_value, "reported": rr.reported,
                        "fixes": rr.fixes, "seeds": rr.seeds, "table": rr.table, "attribution": rr.attribution,
                        "original": rr.original, "corrected": rr.corrected, "inflation": rr.inflation,
                        "waterfall": rr.reference.get("waterfall"), "seconds": round(rr.seconds, 1), "runs": rr.runs,
                        "n_train": rr.reference.get("n_train"), "n_test": rr.reference.get("n_test"),
                        "legit_scenario": rr.reference.get("legit_scenario"), "proposed": sp.proposed_cols}
    if not rr.fidelity_ok:
        result["findings"] = finds
        result["verdict"] = verdict.decide(claim, rr, None, sp.metric, float(np.mean(sp.y)))
        result["log"] = prog.log
        return _clean(result)
    for f in finds:
        a = rr.attribution.get(f.get("fix"))
        if a:
            f["impact"] = {k: a[k] for k in ("mean", "lo", "hi", "status", "alone", "last", "last_lo", "last_hi")}
            f["impact_text"] = (f"The {f.get('fix')} repair category has a {verdict.metric_label(sp.metric)} fair-share allocation of {a['mean']:.3f} "
                                f"(95% CI {a['lo']:.3f} to {a['hi']:.3f}); the category changes the score by {a['alone']:.3f} alone "
                                f"and {a['last']:.3f} after other categories are corrected. This is not a separate causal effect for this finding.")

    # 6. repair + verification ----------------------------------------------------------
    prog("repair", "Generating the corrected notebook…")
    seed0 = rr.seeds[0]
    rep = repair.generate(src, sp, fixes, proxy_cols, seed0) if fixes else {"ok": False, "none_needed": True,
                                                                            "reason": "No leaks were found, so the protocol needs no changes."}
    if rep.get("ok"):
        prog("repair", "Executing the corrected code to verify it…")
        vr = sandbox.run_code(rep["code"], {data_name: data_bytes}, timeout=opts.timeout)
        got = repair.parse_result(vr.stdout)
        expected = rr.table["+".join(sorted(fixes)) or "none"][0]
        checks = repair.verify_structure(vr, sp, fixes, proxy_cols, raw) if vr.ok else []
        number_ok = bool(vr.ok and got is not None and abs(got - expected) < 1e-4)
        checks.insert(0, {"check": "Corrected code reproduces the replay engine's number", "ok": number_ok,
                          "detail": f"printed {got}, replay engine {expected:.6f}" if got is not None else (vr.error or "no result printed")})
        rep["verification"] = {"ran": vr.ok, "value": got, "expected": expected, "error": vr.error, "checks": checks,
                               "verified": bool(number_ok and all(c["ok"] for c in checks))}
        n_ok = sum(c["ok"] for c in checks)
        prog("repair", f"Corrected code {'verified' if rep['verification']['verified'] else 'NOT verified'}: {n_ok}/{len(checks)} checks passed")
    result["repair"] = rep

    # 7. survival tests -----------------------------------------------------------------
    surv = None
    if opts.run_survival:
        prog("survival", "Running claim survival tests…")
        surv = survival.run_all(sp, fixes, proxy_cols, progress=prog)
        result["survival"] = surv

    # With proposed leaks, the "legitimate" scenario is a different protocol (the feature stays),
    # so its survival tests must be run on that protocol, never borrowed from the confirmed one.
    legit = (rr.reference or {}).get("legit_scenario")
    surv_legit = None
    if legit and sp.proposed_cols and opts.run_survival:
        from .replay import FIX_PROPOSED
        prog("survival", "Running survival tests for the scenario where the proposed feature is legitimate…")
        surv_legit = survival.run_all(sp, [f for f in fixes if f != FIX_PROPOSED], proxy_cols, progress=prog)
        result["survival_if_legitimate"] = surv_legit

    # 8. verdict --------------------------------------------------------------------------
    v = verdict.decide(claim, rr, surv, sp.metric, float(np.mean(sp.y)))
    if legit and sp.proposed_cols:
        from types import SimpleNamespace
        alt = SimpleNamespace(fidelity_ok=True, fidelity_value=rr.fidelity_value, reported=rr.reported,
                              corrected=legit["corrected"], inflation=legit["inflation"])
        v = verdict.combine_scenarios(v, verdict.decide(claim, alt, surv_legit, sp.metric, float(np.mean(sp.y))), sp.proposed_cols)
    top = sorted([f for f in finds if f.get("impact")], key=lambda f: -f["impact"]["mean"])
    if top and top[0]["impact"]["status"] == "material":
        v["main_driver"] = top[0]["title"]
    v["rules"] = verdict.RULES
    # A verdict is only final when every required repair was executed and passed its checks.
    if fixes:
        ver = (rep or {}).get("verification") or {}
        if not ver.get("verified"):
            failed = [c["check"] for c in ver.get("checks", []) if not c["ok"]]
            why = ("the corrected code could not be generated: " + rep.get("reason", "") if not rep.get("ok")
                   else "the corrected code did not pass verification" + (f" ({'; '.join(failed)})" if failed else
                                                                          f": {ver.get('error') or 'no result'}"))
            v["provisional"] = True
            v["provisional_reason"] = why
            v["reason"] += (f" PROVISIONAL: {why}. The numbers come from the replay engine, but the repair itself is not "
                            "verified, so treat this verdict as unconfirmed.")
    result["verdict"] = v
    result["findings"] = finds

    prog("done", f"Verdict: {v['verdict']}")
    result["log"] = prog.log
    result["seconds"] = round(time.time() - prog.t0, 1)
    return _clean(result)
