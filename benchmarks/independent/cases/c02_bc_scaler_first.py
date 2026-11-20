# Breast cancer, scaler fitted before the split (the textbook leak). Arrays, module-style imports.
import numpy as np
import pandas as pd
from sklearn import model_selection as ms, preprocessing as pp, svm, metrics

df = pd.read_csv("breast_cancer.csv")
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1].values
Xs = pp.MinMaxScaler().fit_transform(X)
a, b, c, d = ms.train_test_split(Xs, y, test_size=0.25, random_state=11)
m = svm.SVC(probability=True, random_state=0).fit(a, c)
print(metrics.roc_auc_score(d, m.predict_proba(b)[:, 1]))
