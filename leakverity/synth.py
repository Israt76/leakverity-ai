"""Synthetic, non-identifiable hospital-readmission data with a known ground truth.

Because the generating process is known, we can draw *fresh* patients and measure a
model's true out-of-sample performance. The benchmark uses this to check whether
LeakVerity's corrected estimates are actually closer to the truth than the
original, leaky numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES_CLINICAL = ["age", "sex_male", "comorbidity_index", "hba1c", "num_prior_admissions",
                     "length_of_stay", "num_medications", "num_diagnoses", "emergency_admission"]
FUTURE_FEATURE = "followup_ed_visits_30d"


def generate(n_patients: int = 1600, seed: int = 0, repeat_rate: float = 0.6, dup_rate: float = 0.06,
             future_strength: float = 1.0, start_patient: int = 0, start_date: str = "2022-01-01",
             days: int = 1460, elderly_signal: float = 0.45) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    enc_id = 0
    start = pd.Timestamp(start_date)
    for p in range(n_patients):
        pid = start_patient + p
        risk = rng.normal()                       # latent patient frailty
        age = int(np.clip(rng.normal(64, 14), 18, 97))
        sex = int(rng.random() < 0.48)
        hba1c = round(float(np.clip(rng.normal(6.4 + 0.5 * risk, 1.0), 4.5, 13.5)), 1)
        comorb = int(np.clip(round(2.2 + 1.3 * risk + rng.normal(0, 1.4)), 0, 12))
        n_enc = 1 + rng.poisson(repeat_rate)
        prior = int(rng.poisson(max(0.2, 0.9 + 0.7 * risk)))
        first_day = rng.integers(0, days)
        for e in range(n_enc):
            day = int(min(days - 1, first_day + e * rng.integers(20, 200)))
            los = int(np.clip(round(rng.lognormal(1.3 + 0.18 * risk, 0.5)), 1, 40))
            meds = int(np.clip(round(rng.normal(12 + 2.5 * risk, 4)), 1, 45))
            diags = int(np.clip(round(rng.normal(7 + 1.2 * risk, 2.5)), 1, 16))
            emerg = int(rng.random() < 1 / (1 + np.exp(-(0.2 + 0.6 * risk))))
            drift = 0.25 * (day / days)           # mild drift over time
            signal = (1.15 * risk + 0.10 * (los - 4) + 0.22 * (prior + e) + 0.35 * emerg
                      + 0.04 * (meds - 12) + 0.06 * (comorb - 2))
            if age >= 65:
                signal *= elderly_signal          # outcome is harder to predict for older patients
            logit = -1.55 + signal + drift + rng.normal(0, 0.6)
            readmit = int(rng.random() < 1 / (1 + np.exp(-logit)))
            ed = int(rng.poisson(0.15 + future_strength * 1.9 * readmit))
            rows.append({
                "encounter_id": f"E{enc_id:06d}", "patient_id": f"P{pid:05d}",
                "admit_date": (start + pd.Timedelta(days=day)).date().isoformat(),
                "age": age, "age_group": "65+" if age >= 65 else "18-64", "sex_male": sex,
                "comorbidity_index": comorb, "hba1c": hba1c, "num_prior_admissions": prior + e,
                "length_of_stay": los, "num_medications": meds, "num_diagnoses": diags,
                "emergency_admission": emerg, FUTURE_FEATURE: ed, "readmitted_30d": readmit,
            })
            enc_id += 1
    df = pd.DataFrame(rows)
    # data-entry duplicates: the same encounter exported twice
    n_dup = int(dup_rate * len(df))
    if n_dup:
        dup = df.sample(n=n_dup, random_state=seed)
        df = pd.concat([df, dup], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)


def fresh_population(n_patients: int = 6000, seed: int = 10_000, **kw) -> pd.DataFrame:
    """Independent patients from the same process — the ground truth test set."""
    kw.setdefault("dup_rate", 0.0)
    return generate(n_patients=n_patients, seed=seed, start_patient=900_000, **kw)
