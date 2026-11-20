"""Build the datasets used by the independent cases (real bundled datasets + small synthetic ones)."""
import pathlib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, make_classification

HERE = pathlib.Path(__file__).resolve().parent

bc = load_breast_cancer(as_frame=True)
df = bc.frame.rename(columns=lambda c: c.replace(" ", "_"))
df.insert(0, "sample_id", [f"BC{i:04d}" for i in range(len(df))])
df.to_csv(HERE / "breast_cancer.csv", index=False)
dup = pd.concat([df, df.sample(frac=0.15, random_state=1)], ignore_index=True).sample(frac=1, random_state=2)
dup.to_csv(HERE / "breast_cancer_exported_twice.csv", index=False)

w = load_wine(as_frame=True).frame
w["is_class_0"] = (w.pop("target") == 0).astype(int)
w.to_csv(HERE / "wine.csv", index=False)

rng = np.random.default_rng(0)
noise = pd.DataFrame(rng.normal(size=(200, 2000)), columns=[f"g{i}" for i in range(2000)])
noise["label"] = rng.integers(0, 2, 200)
noise.to_csv(HERE / "noise_genes.csv", index=False)

X, y = make_classification(n_samples=1500, n_features=12, n_informative=4, weights=[0.94], flip_y=0.02, random_state=3)
imb = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
imb["fraud"] = y
imb.to_csv(HERE / "transactions.csv", index=False)

# daily sensor data with drift; a centred rolling mean peeks at future days
n = 900
t = pd.date_range("2024-01-01", periods=n, freq="D")
level = np.cumsum(rng.normal(0, 0.15, n))
x1 = level + rng.normal(0, 1, n)
fail = (rng.random(n) < 1 / (1 + np.exp(-(level - np.median(level))))).astype(int)
sens = pd.DataFrame({"day": t.date.astype(str), "vibration": x1, "temp": rng.normal(20, 3, n) + 0.3 * level, "failure": fail})
sens.to_csv(HERE / "sensor.csv", index=False)
print("ok")
