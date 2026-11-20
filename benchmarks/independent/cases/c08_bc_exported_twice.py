# Breast cancer export where 15% of samples were written twice (same sample_id); random split.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv("breast_cancer_exported_twice.csv")
features = [c for c in df.columns if c not in ("sample_id", "target")]
X_train, X_test, y_train, y_test = train_test_split(df[features], df["target"], test_size=0.3, random_state=5)
rf = RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=-1).fit(X_train, y_train)
print("AUC", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))
