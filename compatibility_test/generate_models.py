import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score

out = Path("compatibility_test")
out.mkdir(exist_ok=True)

X, y = make_classification(
    n_samples=2000,
    n_features=6,
    n_informative=4,
    n_redundant=0,
    weights=[0.65, 0.35],
    class_sep=0.9,
    flip_y=0.04,
    random_state=42
)

features = [f"Feature_{i}" for i in range(1, 7)]
df = pd.DataFrame(X, columns=features)
df["Target"] = y

# Synthetic subgroup for auditing, not model training.
df["Segment"] = np.where(
    df["Feature_1"] > df["Feature_1"].median(),
    "Group_A",
    "Group_B"
)

train, test = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df["Target"]
)

models = {
    "logistic_regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42)
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ),
    "svm": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=True, random_state=42)
    )
}

test.to_csv(out / "evaluation.csv", index=False)

for name, model in models.items():
    model.fit(train[features], train["Target"])
    predictions = model.predict(test[features])

    joblib.dump(model, out / f"{name}.joblib")

    print(f"\n{name}")
    print("Accuracy:", round(
        accuracy_score(test["Target"], predictions), 4
    ))
    print("Recall:", round(
        recall_score(test["Target"], predictions), 4
    ))

print("\nAll models and evaluation CSV generated.")
