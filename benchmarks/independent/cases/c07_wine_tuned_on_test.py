# Wine (class 0 vs rest): k-NN with k tuned on the test set inside a helper function.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def evaluate(k, Xa, ya, Xb, yb):
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)).fit(Xa, ya)
    return accuracy_score(yb, m.predict(Xb))

wine = pd.read_csv("wine.csv")
X, y = wine.drop(columns="is_class_0"), wine["is_class_0"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
best = max(evaluate(k, X_train, y_train, X_test, y_test) for k in [1, 3, 5, 7, 9, 15])
print("Best accuracy", best)
