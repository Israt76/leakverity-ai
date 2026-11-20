"""AI layer: reasoning about meaning, never about numbers.

The model is asked questions that need domain understanding:
  * Which features would not be available at the moment the prediction is made?
  * Which columns identify a person (and so define groups that must not be split)?
  * What exactly does the author claim (metric, value, scope)?
  * How to explain each verified finding in plain language.

Guards: every column it names must exist; every flagged feature is only a
*hypothesis* until the replay measures its effect; explanations may only contain
numbers already present in the verified evidence. Without an API key, a rule-based
fallback with the same interface is used and the UI says so.
"""
from __future__ import annotations

import json
import os
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


# ---------------------------------------------------------------------------
# LLM client (optional)
# ---------------------------------------------------------------------------

class LLM:
    def __init__(self, provider: str, model: str, key: str):
        self.provider, self.model = provider, model
        if provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=key)
        else:
            import openai
            self.client = openai.OpenAI(api_key=key)

    def json(self, system: str, user: str, max_tokens: int = 3000) -> dict:
        if self.provider == "anthropic":
            msg = self.client.messages.create(model=self.model, max_tokens=max_tokens, temperature=0, system=system,
                                              messages=[{"role": "user", "content": user}])
            text = "".join(getattr(b, "text", "") for b in msg.content)
        else:
            r = self.client.chat.completions.create(model=self.model, temperature=0, max_tokens=max_tokens,
                                                    response_format={"type": "json_object"},
                                                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            text = r.choices[0].message.content or "{}"
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError("model returned no JSON")
        return json.loads(m.group(0))


def get_llm(provider=None, key=None, model=None) -> LLM | None:
    provider = (provider or os.getenv("LEAKVERITY_PROVIDER") or "").lower() or None
    if provider in ("none", "off"):
        return None
    for p in ([provider] if provider else ["anthropic", "openai"]):
        k = key or os.getenv("ANTHROPIC_API_KEY" if p == "anthropic" else "OPENAI_API_KEY")
        if k:
            try:
                return LLM(p, model or os.getenv("LEAKVERITY_MODEL") or
                           ("claude-sonnet-5-5" if p == "anthropic" else "gpt-4o-mini"), k)
            except Exception:  # noqa: BLE001
                continue
    return None


# ---------------------------------------------------------------------------
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

AUDIT_SYSTEM = """You are the domain-reasoning component of LeakVerity, a tool that audits machine-learning
experiments for data leakage. You judge only whether each feature would be available at the moment a
prediction is made. You do not compute or estimate any metric. Return one JSON object only."""

AUDIT_USER = """Prediction task: predict "{target}" {point}.
Columns used as model features (name | type | example values | univariate AUC with the target):
{cols}

For every feature that would NOT be known at prediction time (it is recorded after the prediction point,
or it is a direct consequence/encoding of the outcome), return it. Do not flag ordinary risk factors.
Return: {{"flags": [{{"column": <exact name>, "reason": <one sentence>, "confidence": "high"|"medium"|"low"}}],
          "prediction_point": <short phrase, the moment the prediction is made, as you understood it>}}"""


def _describe(design: pd.DataFrame, y: pd.Series) -> list[dict]:
    out = []
    for c in design.columns:
        s = design[c]
        auc = univariate_auc(pd.to_numeric(s, errors="coerce"), y) if pd.api.types.is_numeric_dtype(s) else None
        ex = ", ".join(str(v) for v in s.dropna().unique()[:4])
        out.append({"column": c, "dtype": str(s.dtype), "examples": ex, "auc": auc})
    return out


def audit_features(design: pd.DataFrame, y: pd.Series, target: str, prediction_point: str | None,
                   llm: LLM | None) -> dict:
    desc = _describe(design, y)
    by = {d["column"]: d for d in desc}
    flags, source, point = [], "rules", prediction_point
    if llm is not None:
        try:
            cols = "\n".join(f"- {d['column']} | {d['dtype']} | {d['examples']} | "
                             f"{'n/a' if d['auc'] is None else f'{d['auc']:.2f}'}" for d in desc)
            data = llm.json(AUDIT_SYSTEM, AUDIT_USER.format(target=target, cols=cols,
                                                            point=f"at: {prediction_point}" if prediction_point else "(prediction point not stated; infer it from the column names)"))
            for f in data.get("flags", []):
                col = f.get("column")
                if col in by:                     # guard: the column must exist
                    conf = f.get("confidence", "medium")
                    flags.append({"column": col, "reason": str(f.get("reason", ""))[:300],
                                  "confidence": conf, "auc": by[col]["auc"], "source": "ai", "needs_review": conf == "low"})
            point = point or data.get("prediction_point")
            source = "ai"
        except Exception as e:  # noqa: BLE001
            source = f"rules (AI unavailable: {str(e)[:80]})"
    if not source.startswith("ai"):
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
    return {"flags": flags, "source": source, "prediction_point": point, "columns": desc}


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

CLAIM_SYSTEM = """You are the claim-reading component of LeakVerity. From the author's text, find the single main
performance claim about the model. Copy the sentence exactly. Do not compute or round anything. Return JSON only."""

CLAIM_USER = """Author's text:
---
{text}
---
Return {{"sentence": <the exact sentence containing the claim, copied verbatim>,
         "metric": one of ["roc_auc_score","accuracy_score","f1_score","precision_score","recall_score","average_precision_score","balanced_accuracy_score"],
         "value": <the number exactly as written, without %>, "is_percent": <true if written as a percentage>,
         "general": <true if the claim asserts broad validity: deployment, all/any patients or users, robustness, generalisation, future use>,
         "population": <who the claim covers, short phrase, or null>}}
If there is no performance claim, return {{"sentence": null}}."""


def _norm_ws(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\*\*|__|`", "", t)).strip()


def ai_claim(text: str, llm: LLM) -> dict | None:
    """AI claim reading with guards: the sentence must occur in the text and the value in the sentence."""
    data = llm.json(CLAIM_SYSTEM, CLAIM_USER.format(text=text[:6000]), max_tokens=500)
    sent = data.get("sentence")
    if not sent:
        return None
    if _norm_ws(sent) not in _norm_ws(text):
        return None                                  # invented or paraphrased sentence
    raw = str(data.get("value", "")).replace("%", "").strip()
    try:
        v = float(raw)
    except ValueError:
        return None
    if raw not in sent:
        return None                                  # value not literally in the sentence
    if data.get("is_percent") or v > 1:
        v = v / 100
    metric = data.get("metric") if data.get("metric") in METRIC_MAP.values() else None
    return {"text": _norm_ws(sent), "metric": metric, "value": v, "general": bool(data.get("general")),
            "population": data.get("population"), "source": "AI reading of the notebook text"}


def extract_claim(text: str, metric_name: str | None, reported: float | None, llm: LLM | None = None) -> dict:
    claim = {"text": "", "metric": metric_name, "value": reported, "general": False, "source": "trace"}
    if not text:
        return claim
    if llm is not None:
        try:
            c = ai_claim(text, llm)
            if c is not None:
                c["metric"] = c["metric"] or metric_name
                return c
        except Exception:  # noqa: BLE001
            pass
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


# ---------------------------------------------------------------------------
# Explanations (numbers guarded)
# ---------------------------------------------------------------------------

_NUM = re.compile(r"(?<![A-Za-z])\d*\.?\d+")


def _nums(s: str) -> set[float]:
    out = set()
    for m in _NUM.finditer(s.replace(",", "")):
        try:
            out.add(round(float(m.group(0)), 4))
        except ValueError:
            pass
    return out


def numbers_grounded(candidate: str, source: str) -> bool:
    return _nums(candidate) <= (_nums(source) | {0.0, 1.0, 2.0, 95.0, 100.0})


EXPLAIN_SYSTEM = """You explain verified findings from LeakVerity to a researcher. The numbers and verdicts are
final and come from re-executed experiments. Do not add, change or estimate numbers. Do not speculate about
intent. 2-3 sentences each. Return JSON: {"items": [{"id": <id>, "text": <explanation>}]}"""


def explain(findings: list[dict], llm: LLM | None) -> dict:
    if llm is None or not findings:
        return {"rewritten": 0, "rejected": 0}
    payload = [{"id": i, "title": f["title"], "evidence": f["evidence"], "impact": f.get("impact_text", ""),
                "current": f["explanation"]} for i, f in enumerate(findings)]
    data = llm.json(EXPLAIN_SYSTEM, json.dumps(payload, default=str))
    ok = bad = 0
    for it in data.get("items", []):
        i = it.get("id")
        if not isinstance(i, int) or not 0 <= i < len(findings):
            continue
        f = findings[i]
        src = json.dumps(f, default=str)
        txt = str(it.get("text", "")).strip()
        if txt and numbers_grounded(txt, src):
            f["explanation"], f["explanation_source"] = txt, "ai"
            ok += 1
        else:
            bad += 1
    return {"rewritten": ok, "rejected": bad}
