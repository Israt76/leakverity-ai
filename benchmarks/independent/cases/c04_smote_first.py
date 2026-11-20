# Imbalanced fraud data, SMOTE applied to the whole dataset before splitting.
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

tx = pd.read_csv("transactions.csv")
X, y = tx.drop(columns=["fraud"]), tx["fraud"]
X_res, y_res = SMOTE(random_state=0).fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=1)
rf = RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1).fit(X_train, y_train)
print("AUC", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))
