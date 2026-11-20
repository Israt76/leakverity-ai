# Breast cancer, leak-free pipeline. Claim: test AUC of 0.99.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

data = pd.read_csv("breast_cancer.csv")
X = data.drop(columns=["sample_id", "target"])
y = data["target"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
clf = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))])
clf.fit(X_tr, y_tr)
print("AUC", roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
