# 2,000 random "genes", random labels. Select the 20 best genes on ALL samples, then evaluate.
# Classic selection leak: the reported AUC looks real, the truth is chance.
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

expr = pd.read_csv("noise_genes.csv")
X, y = expr.drop(columns="label"), expr["label"]
X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, random_state=4)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print("Test AUC:", round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 3))
