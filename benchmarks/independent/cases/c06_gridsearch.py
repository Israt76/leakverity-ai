# GridSearchCV over a pipeline, evaluated once on the test set. Legitimate, but the search object
# wraps an estimator, which this version cannot rebuild: LeakVerity should abstain, not guess.
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

df = pd.read_csv("breast_cancer.csv")
X, y = df.drop(columns=["sample_id", "target"]), df["target"]
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=3, stratify=y)
gs = GridSearchCV(make_pipeline(StandardScaler(), SVC(probability=True)), {"svc__C": [0.1, 1, 10]}, cv=3, scoring="roc_auc")
gs.fit(Xtr, ytr)
print(roc_auc_score(yte, gs.predict_proba(Xte)[:, 1]))
