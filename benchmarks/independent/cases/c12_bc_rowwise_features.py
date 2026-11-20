# Breast cancer, clean control: engineered features computed from each row alone (no neighbours,
# no whole-dataset statistics), scaling inside the pipeline. Must be Supported without conditions.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

df = pd.read_csv("breast_cancer.csv")
df["compactness_ratio"] = df["mean_perimeter"] ** 2 / df["mean_area"]
df["texture_x_smoothness"] = df["mean_texture"] * df["mean_smoothness"]
X = df.drop(columns=["sample_id", "target"])
y = df["target"]
a, b, c, d = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(a, c)
print("AUC", roc_auc_score(d, m.predict_proba(b)[:, 1]))
