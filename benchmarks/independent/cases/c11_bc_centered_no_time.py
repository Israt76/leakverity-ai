# Breast cancer: a centred rolling mean over rows that have NO time order (file order of samples),
# computed before the split. AMBIGUOUS, not clean: there is no future to peek at, but each value mixes
# neighbouring rows, so test rows shape training features (cross-partition contamination).
# Expected: no automatic removal (time order unknown), the feature sent to review, and a verdict
# that is explicitly CONDITIONAL on that review.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

df = pd.read_csv("breast_cancer.csv")
df["radius_local"] = df["mean_radius"].rolling(3, center=True, min_periods=1).mean()
X = df.drop(columns=["sample_id", "target"])
y = df["target"]
a, b, c, d = train_test_split(X, y, test_size=0.3, random_state=12, stratify=y)
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(a, c)
print("AUC", roc_auc_score(d, m.predict_proba(b)[:, 1]))
