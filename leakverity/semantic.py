"""Rule-based evaluation of feature roles, prediction-time concerns and research claims.

Suspected features are reviewed rather than removed automatically. Every reported
performance value is calculated from executed experiments.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from .stats import univariate_auc

FUTURE_TERMS = re.compile(
    r"(follow[_ ]?up|post[_ ]|_post\b|after|future|next[_ ]|subsequent|outcome|result|final[_ ]|"
    r"days[_ ]to|time[_ ]to|readmi|death|died|mortality|deceased|discharge[_ ]?(status|disposition)|"
    r"\d+d\b|_\d+_?days?\b|label|target|leak)", re.I)
ID_TERMS = re.compile(r"(^id$|_id$|^id_|patient|subject|user|customer|member|mrn|encounter|account|person)", re.I)
DATE_TERMS = re.compile(r"(date|time|timestamp|_dt$|_at$|year|month|^day$|_day$)", re.I)
SUBGROUP_TERMS = re.compile(r"(age[_ ]?group|sex|gender|race|ethnic|site|hospital|region|insurance|payer|cohort)", re.I)
GENERAL_WORDS = re.compile(r"\b(all|every|any|across|general(?:ly|ize|ise|izes|ises)?|universal(?:ly)?|robust(?:ly)?|deploy\w*|"
                           r"future|prospective\w*|real[- ]world|production|clinical(?:ly)? (?:use|practice))\b", re.I)

CLAIM_RE = re.compile(
    r"(?P<metric>AUC|AUROC|ROC[- ]AUC|accuracy|F1(?:[- ]score)?|precision|recall|average precision|PR[- ]AUC|balanced accuracy)"
    r"[^0-9%\n]{0,25}?(?P<value>0?\.\d+|\d{1,3}(?:\.\d+)?\s*%)", re.I)
METRIC_MAP = {"auc": "roc_auc_score", "auroc": "roc_auc_score", "roc-auc": "roc_auc_score", "roc auc": "roc_auc_score",
              "accuracy": "accuracy_score", "f1": "f1_score", "f1-score": "f1_score", "f1 score": "f1_score",
              "precision": "precision_score", "recall": "recall_score", "average precision": "average_precision_score",
              "pr-auc": "average_precision_score", "pr auc": "average_precision_score",
              "balanced accuracy": "balanced_accuracy_score"}


# Column roles (ids, dates, subgroups) — used to configure survival tests
# ---------------------------------------------------------------------------

def _is_text(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def suggest_roles(raw: pd.DataFrame, target: str | None, features: list[str]) -> dict:
    roles = {"group": None, "date": None, "subgroup": None}
    n = len(raw)
    best = 0.0
    for c in raw.columns:
        if c == target or c in features:
            continue
        nun = raw[c].nunique(dropna=True)
        ratio = n / max(nun, 1)
        # an identifier that repeats: several records per entity, but many entities
        if ID_TERMS.search(c) and 1.15 < ratio < n / 20 and ratio > best:
            roles["group"], best = c, ratio
    for c in raw.columns:
        if roles["date"] is None and DATE_TERMS.search(c) and _is_text(raw[c]):
            parsed = pd.to_datetime(raw[c], errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.95:
                roles["date"] = c
    for c in raw.columns:
        if roles["subgroup"] is None and SUBGROUP_TERMS.search(c) and 2 <= raw[c].nunique() <= 8 and c != target:
            roles["subgroup"] = c
    return roles


RULE_STRONG_AUC = 0.80     # rule-based flags need a name match AND at least this univariate AUC

# ---------------------------------------------------------------------------
# Feature audit
# ---------------------------------------------------------------------------

def _describe(design: pd.DataFrame, y: pd.Series) -> list[dict]:
    out = []
    for c in design.columns:
        s = design[c]
        auc = univariate_auc(pd.to_numeric(s, errors="coerce"), y) if pd.api.types.is_numeric_dtype(s) else None
        ex = ", ".join(str(v) for v in s.dropna().unique()[:4])
        out.append({"column": c, "dtype": str(s.dtype), "examples": ex, "auc": auc})
    return out


def audit_features(design: pd.DataFrame, y: pd.Series, target: str,
                   prediction_point: str | None) -> dict:
    """Flag potentially unavailable predictors using transparent name and score rules."""
    desc = _describe(design, y)
    flags = []
    for d in desc:
        name = d["column"]
        if name == target:
            continue
        named = bool(FUTURE_TERMS.search(name))
        auc = d["auc"]
        # Rules cannot read meaning, so they need two independent signals before a fix is applied
        # automatically: a post-outcome-sounding name AND an unusually strong association.
        # A name match with ordinary predictive power is sent to the user for review instead,
        # because removing a legitimate predictor also lowers the score and would look like "inflation".
        if named and (auc is None or auc >= 0.6):
            strong = auc is not None and auc >= RULE_STRONG_AUC
            flags.append({"column": name, "reason": f"The name “{name}” suggests it is recorded after the prediction point or derived from the outcome"
                                                    + (f", and on its own it separates the classes with AUC {auc:.2f}." if strong else "."),
                          "confidence": "high" if strong else "low", "auc": auc, "source": "rules", "needs_review": not strong})
        elif auc is not None and auc >= 0.95:
            # strength alone is not evidence of leakage (real data has genuinely strong predictors)
            flags.append({"column": name, "reason": f"On its own this feature separates the classes almost perfectly (AUC {auc:.2f}). "
                                                    "That can mean an outcome proxy, or simply a very strong predictor.",
                          "confidence": "low", "auc": auc, "source": "rules", "needs_review": True, "review_kind": "strength_only"})
    return {"flags": flags, "source": "rules", "prediction_point": prediction_point, "columns": desc}


def extract_claim(text: str, metric_name: str | None, reported: float | None, ) -> dict:
    claim = {"text": "", "metric": metric_name, "value": reported, "general": False, "source": "trace"}
    if not text:
        return claim
    best = None
    for m in CLAIM_RE.finditer(text):
        raw_v = m.group("value").replace(" ", "")
        v = float(raw_v.rstrip("%")) / (100 if raw_v.endswith("%") else 1)
        name = METRIC_MAP.get(m.group("metric").lower().replace("–", "-"), None)
        if best is None or (metric_name and name == metric_name):
            best = (m, v, name)
    if best:
        m, v, name = best
        start = text.rfind("\n", 0, m.start()) + 1
        end = text.find("\n", m.end())
        sentence = text[start:end if end != -1 else len(text)].strip().lstrip("#").strip()
        claim.update(text=re.sub(r"\*\*|__", "", sentence), metric=name or metric_name, value=v,
                     general=bool(GENERAL_WORDS.search(sentence)), source="notebook text")
    return claim
