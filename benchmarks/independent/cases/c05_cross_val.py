# Cross-validation protocol (detected, not replayed in this version): LeakVerity should abstain.
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("breast_cancer.csv")
X, y = df.drop(columns=["sample_id", "target"]), df["target"]
scores = cross_val_score(GradientBoostingClassifier(random_state=0), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
print("CV AUC", scores.mean())
