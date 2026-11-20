"""Independent reference estimates for the hand-written cases.

Each function re-implements the case's model *honestly* (no leak) and estimates its performance
with repeated cross-validation, a different method from LeakVerity's hold-out replay. The
expected verdict is then derived from the notebook's reported number against this reference:

    reported - reference <= 0.03                    -> Supported
    reported - reference  > 0.03, reference > chance + 0.05 -> Overstated
    reported - reference  > 0.03, reference <= chance + 0.05 -> Contradicted
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedGroupKFold, TimeSeriesSplit,
                                     cross_val_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

HERE = pathlib.Path(__file__).resolve().parent
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)


def _bc():
    d = pd.read_csv(HERE / "breast_cancer.csv")
    return d.drop(columns=["sample_id", "target"]), d["target"]


def c01():
    X, y = _bc()
    return cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y, cv=CV, scoring="roc_auc").mean()


def c02():
    X, y = _bc()
    return cross_val_score(make_pipeline(MinMaxScaler(), SVC(probability=True, random_state=0)), X, y, cv=CV, scoring="roc_auc").mean()


def c03():
    d = pd.read_csv(HERE / "noise_genes.csv")
    X, y = d.drop(columns="label"), d["label"]
    return cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000)), X, y, cv=CV,
                           scoring="roc_auc").mean()


def c04():
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import make_pipeline as imb_pipeline
    d = pd.read_csv(HERE / "transactions.csv")
    X, y = d.drop(columns="fraud"), d["fraud"]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
    return cross_val_score(imb_pipeline(SMOTE(random_state=0), RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1)),
                           X, y, cv=cv, scoring="roc_auc").mean()


def c07():
    d = pd.read_csv(HERE / "wine.csv")
    X, y = d.drop(columns="is_class_0"), d["is_class_0"]
    inner = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsClassifier()),
                         {"kneighborsclassifier__n_neighbors": [1, 3, 5, 7, 9, 15]}, cv=3, scoring="accuracy")
    return cross_val_score(inner, X, y, cv=CV, scoring="accuracy").mean()       # nested CV


def c08():
    d = pd.read_csv(HERE / "breast_cancer_exported_twice.csv").drop_duplicates()
    X, y = d.drop(columns=["sample_id", "target"]), d["target"]
    scores = [cross_val_score(RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=-1), X, y,
                              cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r), groups=d["sample_id"],
                              scoring="roc_auc").mean() for r in range(3)]
    return float(np.mean(scores))


def c09():
    s = pd.read_csv(HERE / "sensor.csv")
    s["vib_smooth"] = s["vibration"].rolling(15, min_periods=1).mean()          # backward-looking only
    X, y = s[["vibration", "temp", "vib_smooth"]], s["failure"]
    return cross_val_score(LogisticRegression(), X, y, cv=TimeSeriesSplit(n_splits=5), scoring="roc_auc").mean()


def c10():
    X, y = _bc()
    return cross_val_score(HistGradientBoostingClassifier(early_stopping=True, validation_fraction=0.2, random_state=0), X, y,
                           cv=CV, scoring="roc_auc").mean()


def c11():
    d = pd.read_csv(HERE / "breast_cancer.csv")
    d["radius_local"] = d["mean_radius"].rolling(3, center=True, min_periods=1).mean()
    X, y = d.drop(columns=["sample_id", "target"]), d["target"]
    return cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), X, y, cv=CV, scoring="roc_auc").mean()


def c12():
    d = pd.read_csv(HERE / "breast_cancer.csv")
    d["compactness_ratio"] = d["mean_perimeter"] ** 2 / d["mean_area"]
    d["texture_x_smoothness"] = d["mean_texture"] * d["mean_smoothness"]
    X, y = d.drop(columns=["sample_id", "target"]), d["target"]
    return cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), X, y, cv=CV, scoring="roc_auc").mean()


REFERENCE = {"c12_bc_rowwise_features": c12, "c11_bc_centered_no_time": c11, "c01_bc_clean": c01, "c02_bc_scaler_first": c02, "c03_noise_selectkbest": c03, "c04_smote_first": c04,
             "c07_wine_tuned_on_test": c07, "c08_bc_exported_twice": c08, "c09_sensor_centered_rolling": c09,
             "c10_bc_clean_groupless_proba": c10}
CHANCE = {"c07_wine_tuned_on_test": max(0.331, 1 - 0.331)}    # accuracy: majority-class rate (59/178 are class 0)


def expected_verdict(case: str, reported: float, reference: float) -> str:
    chance = CHANCE.get(case, 0.5)
    if reported - reference <= 0.03:
        return "Supported"
    return "Contradicted" if reference <= chance + 0.05 else "Overstated"
