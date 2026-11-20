"""Explicit, pre-registered rules that turn verified evidence into a claim verdict."""
from __future__ import annotations

SUPPORTED, OVERSTATED, CONTRADICTED, INSUFFICIENT = "Supported", "Overstated", "Contradicted", "Insufficient evidence"
DEPENDS = "Depends on review"
TOLERANCE = 0.01

RULES = [
    ("Supported", "The claimed value is within the verified 95% CI (plus 0.01), there is no reliable paired inflation, the "
                  "corrected CI excludes chance, and no survival test fails against the claim's stated scope. Other survival "
                  "concerns are listed as caveats."),
    ("Overstated", "The claim does not survive but signal remains (the corrected CI excludes chance, or its point estimate is more "
                   "than 0.05 above chance): it is above the "
                   "verified 95% CI (plus 0.01), or the paired inflation across splits has a one-sided 95% lower bound above 0.01 "
                   "and the claim exceeds the verified mean by more than 0.02, or a survival test fails against the claim's scope "
                   "(instability while the claim exceeds the verified mean; a subgroup or time-based failure for a general claim)."),
    ("Contradicted", "The claimed value is ruled out (above the verified 95% CI plus 0.01), the corrected CI includes chance, "
                     "AND the corrected point estimate is within 0.05 of chance: the reported result does not stand and the "
                     "corrected evidence points to no meaningful signal. This is not proof that the model has none."),
    ("Insufficient evidence", "The protocol could not be replayed faithfully or verified; or the corrected estimate is too "
                              "uncertain to either confirm or rule out the claim (its CI includes both the claim and chance)."),
    ("Depends on review", "A proposed leak (a suspicion from a feature's meaning or from code that mixes rows, without proof "
                          "of timing) changes the verdict: the claim is evaluated with the feature removed and kept, and the two "
                          "verdicts differ. Confirming or rejecting the feature settles it. When both verdicts agree, that verdict "
                          "is issued and the certificate says it holds either way."),
]


def chance_level(metric: str, prevalence: float) -> float | None:
    """Performance of a model with no signal, or None when no single baseline exists.

    AUC / balanced accuracy: 0.5. Accuracy: majority-class rate. Average precision and precision:
    the positive rate (a random ranking or random labelling). F1: 2p / (1 + p), the score of
    predicting every case positive. Recall has no meaningful no-skill value (predicting everything
    positive gives 1.0), so chance-based verdicts are not issued for it.
    """
    p = prevalence
    return {"roc_auc_score": 0.5, "balanced_accuracy_score": 0.5, "accuracy_score": max(p, 1 - p),
            "average_precision_score": p, "precision_score": p, "f1_score": 2 * p / (1 + p) if p > 0 else 0.0}.get(metric)


def decide(claim: dict, replay, survival: dict | None, metric: str, prevalence: float, abstain_reason: str | None = None) -> dict:
    if abstain_reason or replay is None or not replay.fidelity_ok:
        reason = abstain_reason or (
            f"Replaying the protocol as written gave {replay.fidelity_value:.4f}, not the reported {replay.reported:.4f}, "
            "so LeakVerity cannot be sure it reconstructed the experiment correctly." if replay else "No replay was possible.")
        return {"verdict": INSUFFICIENT, "reason": reason, "concerns": []}
    cv = claim.get("value")
    if cv is None:
        return {"verdict": INSUFFICIENT, "reason": "No numerical claim was found or provided.", "concerns": []}
    c = replay.corrected
    chance = chance_level(metric, prevalence)
    has_chance = chance is not None
    if not has_chance:
        chance = float("-inf")
    concerns = []
    if survival:
        for key in ("stability", "subgroup", "temporal"):
            t = survival.get(key) or {}
            if t.get("status") == "concern":
                concerns.append(key)
    label = metric_label(metric)
    ci = f"{c['mean']:.3f} (95% CI {c['lo']:.3f}–{c['hi']:.3f})"
    no_skill_shown = has_chance and c["lo"] <= chance        # CI includes chance: above-chance not demonstrated
    excluded = cv > c["hi"] + TOLERANCE                       # the claimed value itself is ruled out
    inflated = _inflated(replay) and cv - c["mean"] > 2 * TOLERANCE
    near_chance = has_chance and c["mean"] <= chance + 0.05   # the corrected point estimate itself is close to chance
    if excluded and no_skill_shown and near_chance:
        v = CONTRADICTED
        reason = (f"The claimed {label} of {cv:.3f} is ruled out by the corrected estimate {ci}, and the corrected evidence "
                  f"does not show performance above chance ({chance:.2f}). The claim is contradicted; this does not prove the "
                  "model has no signal, only that the reported result does not stand.")
    elif excluded or (inflated and not no_skill_shown):
        v = OVERSTATED
        inf = replay.inflation
        reason = (f"The claimed {label} of {cv:.3f} is above what survives correction: {ci}. "
                  + (f"Across the replayed splits the corrected result was reliably lower: paired inflation {inf['mean']:.3f} "
                     f"(one-sided 95% lower bound {inf.get('lo_one_sided', inf['lo']):.3f}). " if _inflated(replay) else "")
                  + ("" if not has_chance else f"The model still performs better than chance ({chance:.2f})." if not no_skill_shown else
                     f"The corrected point estimate suggests some signal remains, but its CI includes chance ({chance:.2f})."))
    elif no_skill_shown and cv > chance + 0.05:
        v = INSUFFICIENT
        reason = (f"The corrected estimate {ci} is too uncertain to confirm the claimed {label} of {cv:.3f}: it neither rules the "
                  f"claim out nor shows performance above chance ({chance:.2f}). More data or more replay splits are needed."
                  + (" The as-written protocol was reliably higher than the corrected one." if inflated else ""))
    else:
        v = SUPPORTED
        reason = f"The claimed {label} of {cv:.3f} is consistent with the verified estimate {ci}."
    caveats = []
    surv = survival or {}
    for key in concerns:
        t = surv.get(key) or {}
        if key == "stability":
            text = (f"Unstable across re-splits: SD {t.get('sd', float('nan')):.3f} over {t.get('runs')} splits "
                    f"(range {t.get('min', float('nan')):.3f}–{t.get('max', float('nan')):.3f}).")
            breaks = cv - c["mean"] > TOLERANCE
        elif key == "subgroup":
            bad = [g for g in t.get("groups", []) if g.get("status") == "concern"]
            text = "Lower performance in " + ", ".join(f"{t.get('column')} = {g['level']} ({g['mean']:.3f})" for g in bad) + \
                   f" versus {t.get('overall', float('nan')):.3f} overall."
            breaks = bool(claim.get("general"))
        else:
            text = (f"On later records ({t.get('column')} after {t.get('cutoff')}) the result is {t.get('value', float('nan')):.3f} "
                    f"versus {t.get('random_split', float('nan')):.3f} on random splits.")
            breaks = bool(claim.get("general"))
        caveats.append({"test": key, "text": text, "contradicts_claim": breaks})
    if v == SUPPORTED and any(cv_["contradicts_claim"] for cv_ in caveats):
        v = OVERSTATED
        failed = [cv_ for cv_ in caveats if cv_["contradicts_claim"]]
        reason += (" However, the claim does not survive its scope: " + " ".join(c_["text"] for c_ in failed)
                   + (" The claim is stated generally, so this failure counts against it." if claim.get("general") else ""))
    elif v == SUPPORTED and caveats:
        reason += " It holds with caveats: " + " ".join(c_["text"] for c_ in caveats)
    return {"verdict": v, "reason": reason, "concerns": concerns, "caveats": caveats, "chance": chance if has_chance else None}


def _inflated(replay) -> bool:
    """Paired evidence: the as-written protocol beats the corrected one reliably across splits."""
    inf = getattr(replay, "inflation", None) or {}
    lo = inf.get("lo_one_sided", inf.get("lo"))
    return lo is not None and lo == lo and lo > TOLERANCE


def metric_label(metric: str) -> str:
    return {"roc_auc_score": "AUC", "accuracy_score": "accuracy", "f1_score": "F1", "precision_score": "precision",
            "recall_score": "recall", "average_precision_score": "average precision",
            "balanced_accuracy_score": "balanced accuracy"}.get(metric, metric)


def combine_scenarios(confirmed: dict, legit: dict, proposed: list[str]) -> dict:
    """Proposed leaks are suspicions: report the verdict under both possibilities."""
    cols = ", ".join(f"`{c}`" for c in proposed)
    if confirmed["verdict"] == legit["verdict"]:
        out = dict(confirmed)
        out["reason"] += (f" This verdict holds whether or not the proposed leak{'s' if len(proposed) > 1 else ''} ({cols}) "
                          "is confirmed.")
        out["scenarios"] = {"confirmed": confirmed, "legitimate": legit, "agree": True}
        return out
    return {"verdict": DEPENDS,
            "reason": (f"The verdict depends on whether {cols} {'is' if len(proposed) == 1 else 'are'} available at prediction time. "
                       f"If not (the proposed leak is confirmed): {confirmed['verdict']}. {confirmed['reason']} "
                       f"If it is legitimate: {legit['verdict']}. {legit['reason']} Confirm or reject it and re-run to settle the verdict."),
            "concerns": confirmed.get("concerns", []), "caveats": confirmed.get("caveats", []), "chance": confirmed.get("chance"),
            "scenarios": {"confirmed": confirmed, "legitimate": legit, "agree": False}}
