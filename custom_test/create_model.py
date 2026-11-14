
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

rng = np.random.default_rng(42)
n = 2000

# Generate synthetic employee data
df = pd.DataFrame({
    "Age": rng.integers(21, 60, n),
    "MonthlyIncome": rng.integers(20000, 150000, n),
    "YearsAtCompany": rng.integers(0, 20, n),
    "OvertimeHours": rng.integers(0, 35, n),
    "JobSatisfaction": rng.integers(1, 6, n),
    "Department": rng.choice(
        ["Engineering", "Sales", "HR"], n,
        p=[0.5, 0.35, 0.15]
    )
})

# Synthetic attrition risk generation
risk = (
    -2.0
    + 0.07 * df["OvertimeHours"]
    - 0.45 * df["JobSatisfaction"]
    - 0.025 * df["YearsAtCompany"]
    + 0.000006 * (65000 - df["MonthlyIncome"])
    + 0.8 * (df["Department"] == "Sales")
    + 0.3 * (df["Department"] == "HR")
)

probability = 1 / (1 + np.exp(-risk))
df["Attrition"] = rng.binomial(1, probability)

# Department is retained as an audit attribute,
# not included in model training.
features = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "OvertimeHours",
    "JobSatisfaction"
]

train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df["Attrition"]
)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42
)

model.fit(train_df[features], train_df["Attrition"])

predictions = model.predict(test_df[features])

output = Path("custom_test")
output.mkdir(exist_ok=True)

joblib.dump(model, output / "attrition_model.joblib")
test_df.to_csv(output / "attrition_test.csv", index=False)

print("Model and dataset generated successfully!")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")
print(f"Accuracy: {accuracy_score(test_df['Attrition'], predictions):.4f}")
print()
print(classification_report(
    test_df["Attrition"],
    predictions,
    zero_division=0
))
print("Files saved in custom_test/")

