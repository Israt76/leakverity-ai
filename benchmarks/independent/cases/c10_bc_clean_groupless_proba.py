# Breast cancer, leak-free, gradient boosting with a validation split used for early stopping
# inside the training data only (legitimate). Claim: AUC from a single test evaluation.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv("breast_cancer.csv")
X, y = df.drop(columns=["sample_id", "target"]), df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8, stratify=y)
hgb = HistGradientBoostingClassifier(early_stopping=True, validation_fraction=0.2, random_state=0).fit(X_train, y_train)
print("AUC", roc_auc_score(y_test, hgb.predict_proba(X_test)[:, 1]))
