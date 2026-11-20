# Sensor failures. A centred rolling mean uses FUTURE days (pandas, before scikit-learn).
# Known limitation: this leak is inside the design matrix, so it is not detected or repaired.
# The time-based survival test is the expected safety net.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

s = pd.read_csv("sensor.csv")
s["vib_smooth"] = s["vibration"].rolling(15, center=True, min_periods=1).mean()
X, y = s[["vibration", "temp", "vib_smooth"]], s["failure"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
lr = LogisticRegression().fit(X_train, y_train)
print("AUC", roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]))
