"""Build the demo dataset and the leaky readmission notebook (examples/)."""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from leakverity.synth import generate

HERE = pathlib.Path(__file__).resolve().parent

def nb(cells):
    return {"cells": [{"cell_type": t, "metadata": {}, "source": s, **({"outputs": [], "execution_count": None} if t == "code" else {})}
                      for t, s in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}

LEAKY = [
 ("markdown", "# Predicting 30-day hospital readmission\n\nRandom forest model on 2,700 discharge encounters (synthetic, non-identifiable data)."),
 ("code", "import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import roc_auc_score"),
 ("code", "df = pd.read_csv(\"hospital_readmissions.csv\")\ndf.head()"),
 ("code", "features = [\"age\", \"sex_male\", \"comorbidity_index\", \"hba1c\", \"num_prior_admissions\",\n            \"length_of_stay\", \"num_medications\", \"num_diagnoses\", \"emergency_admission\",\n            \"followup_ed_visits_30d\"]\nX = df[features]\ny = df[\"readmitted_30d\"]"),
 ("code", "scaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)"),
 ("code", "X_train, X_test, y_train, y_test = train_test_split(\n    X_scaled, y, test_size=0.25, random_state=42, stratify=y)"),
 ("code", "best_auc, best_seed = 0, None\nfor seed in range(10):\n    model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)\n    model.fit(X_train, y_train)\n    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])\n    if auc > best_auc:\n        best_auc, best_seed = auc, seed\nprint(f\"Test AUC: {best_auc:.3f} (seed {best_seed})\")"),
 ("markdown", "## Result\n\nOur random forest predicts 30-day readmission with a test **AUC of {AUC}**, which supports deploying it at discharge for all adult patients."),
]

CLEAN = [
 ("markdown", "# Predicting 30-day hospital readmission (leak-free protocol)"),
 ("code", "import pandas as pd\nfrom sklearn.model_selection import GroupShuffleSplit\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import roc_auc_score"),
 ("code", "df = pd.read_csv(\"hospital_readmissions.csv\").drop_duplicates(\"encounter_id\")\nfeatures = [\"age\", \"sex_male\", \"comorbidity_index\", \"hba1c\", \"num_prior_admissions\",\n            \"length_of_stay\", \"num_medications\", \"num_diagnoses\", \"emergency_admission\"]\nX, y = df[features], df[\"readmitted_30d\"]"),
 ("code", "gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)\ntrain_idx, test_idx = next(gss.split(X, y, groups=df[\"patient_id\"]))\nmodel = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1))\nmodel.fit(X.iloc[train_idx], y.iloc[train_idx])\nauc = roc_auc_score(y.iloc[test_idx], model.predict_proba(X.iloc[test_idx])[:, 1])\nprint(f\"Test AUC: {auc:.3f}\")"),
 ("markdown", "## Result\n\nThe model predicts 30-day readmission with a test **AUC of {AUC}**."),
]

if __name__ == "__main__":
    df = generate()
    df.to_csv(HERE / "hospital_readmissions.csv", index=False)
    auc = sys.argv[1] if len(sys.argv) > 1 else "0.94"
    clean_auc = sys.argv[2] if len(sys.argv) > 2 else "0.71"
    (HERE / "readmission_model.ipynb").write_text(json.dumps(nb([(t, s.replace("{AUC}", auc)) for t, s in LEAKY]), indent=1))
    (HERE / "readmission_model_clean.ipynb").write_text(json.dumps(nb([(t, s.replace("{AUC}", clean_auc)) for t, s in CLEAN]), indent=1))
    print("wrote", len(df), "rows")
