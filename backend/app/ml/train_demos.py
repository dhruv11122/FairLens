"""Script to generate realistic demo datasets and trained scikit-learn models for FairLens."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

os.makedirs("/Users/dhruvagrawal/.gemini/antigravity/scratch/fairlens/data/demos", exist_ok=True)
os.makedirs("/Users/dhruvagrawal/.gemini/antigravity/scratch/fairlens/models/demos", exist_ok=True)

np.random.seed(42)

# =====================================================================
# Scenario 1: Loan Approval Fairness
# Protect Attributes: gender, age_group, employment_type
# Target: loan_approved (1 or 0)
# =====================================================================

def generate_loan_scenario():
    n = 2500
    genders = np.random.choice(["Male", "Female", "Non-Binary"], size=n, p=[0.48, 0.48, 0.04])
    age_groups = np.random.choice(["18-25", "26-40", "41-60", "60+"], size=n, p=[0.20, 0.45, 0.25, 0.10])
    employment = np.random.choice(["Salaried", "Self-Employed", "Gig/Contract", "Unemployed"], size=n, p=[0.55, 0.25, 0.15, 0.05])
    education = np.random.choice(["High School", "Bachelors", "Masters/PhD"], size=n, p=[0.30, 0.50, 0.20])
    
    annual_income = np.random.gamma(shape=5.0, scale=12000, size=n) # mean ~ 60k
    credit_score = np.random.normal(loc=680, scale=60, size=n).clip(450, 850)
    debt_to_income = np.random.uniform(0.1, 0.55, size=n)
    loan_amount = np.random.uniform(5000, 50000, size=n)

    # Intentionally model realistic disparate impact based on historical credit scoring bias
    # Qualification score based on genuine financial capacity
    financial_ability = (
        (credit_score - 600) / 100 * 1.5 +
        (annual_income - 40000) / 25000 * 1.2 -
        debt_to_income * 2.0
    )

    # Historical bias: slightly penalized gig workers and younger demographic in credit records
    systemic_bias = np.where(age_groups == "18-25", -0.4, 0.0) + \
                    np.where(employment == "Gig/Contract", -0.5, 0.0)

    latent_approval_prob = 1 / (1 + np.exp(-(financial_ability + systemic_bias - 0.2)))
    loan_approved = (np.random.uniform(0, 1, size=n) < latent_approval_prob).astype(int)

    df_loan = pd.DataFrame({
        "gender": genders,
        "age_group": age_groups,
        "employment_type": employment,
        "education": education,
        "annual_income": np.round(annual_income, 2),
        "credit_score": np.round(credit_score, 0).astype(int),
        "debt_to_income": np.round(debt_to_income, 3),
        "loan_amount": np.round(loan_amount, 2),
        "loan_approved": loan_approved
    })

    # Train / Test split (80/20) - audit held-out test data
    train_df, test_df = train_test_split(df_loan, test_size=0.25, random_state=42, stratify=df_loan["loan_approved"])

    feature_cols = ["education", "annual_income", "credit_score", "debt_to_income", "loan_amount", "employment_type"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["annual_income", "credit_score", "debt_to_income", "loan_amount"]),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), ["education", "employment_type"])
        ]
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42))
    ])

    clf.fit(train_df[feature_cols], train_df["loan_approved"])

    # Export test dataset and model
    test_csv_path = "/Users/dhruvagrawal/.gemini/antigravity/scratch/fairlens/data/demos/loan_evaluation_dataset.csv"
    model_path = "/Users/dhruvagrawal/.gemini/antigravity/scratch/fairlens/models/demos/loan_approval_model.joblib"
    
    test_df.to_csv(test_csv_path, index=False)
    joblib.dump(clf, model_path)
    print(f"Loan demo saved: {test_csv_path} ({len(test_df)} rows), Model: {model_path}")

# =====================================================================
# Scenario 2: Hiring / Candidate Shortlisting
# Protected Attributes: gender, ethnicity, disability_status
# Target: shortlisted (1 or 0)
# =====================================================================

def generate_hiring_scenario():
    n = 2000
    genders = np.random.choice(["Female", "Male", "Non-Binary"], size=n, p=[0.45, 0.50, 0.05])
    ethnicity = np.random.choice(["Group A", "Group B", "Group C", "Group D"], size=n, p=[0.40, 0.30, 0.20, 0.10])
    disability = np.random.choice(["No", "Yes"], size=n, p=[0.88, 0.12])
    
    years_experience = np.random.exponential(scale=5.0, size=n).clip(0, 25)
    technical_assessment = np.random.normal(loc=72, scale=12, size=n).clip(30, 100)
    interview_score = np.random.normal(loc=70, scale=14, size=n).clip(30, 100)
    degree_level = np.random.choice(["BSc", "MSc", "PhD", "Self-Taught"], size=n, p=[0.50, 0.35, 0.08, 0.07])

    latent_shortlist = (
        (technical_assessment - 70) / 10 * 1.8 +
        (interview_score - 70) / 10 * 1.5 +
        (years_experience - 4) / 4 * 0.8
    )

    # Subconscious interviewer scoring disparity simulation
    bias_term = np.where((genders == "Female") & (years_experience < 4), -0.35, 0.0) + \
                np.where(ethnicity == "Group D", -0.4, 0.0)

    prob = 1 / (1 + np.exp(-(latent_shortlist + bias_term - 0.5)))
    shortlisted = (np.random.uniform(0, 1, size=n) < prob).astype(int)

    df_hiring = pd.DataFrame({
        "gender": genders,
        "ethnicity": ethnicity,
        "disability_status": disability,
        "years_experience": np.round(years_experience, 1),
        "technical_assessment": np.round(technical_assessment, 1),
        "interview_score": np.round(interview_score, 1),
        "degree_level": degree_level,
        "shortlisted": shortlisted
    })

    train_df, test_df = train_test_split(df_hiring, test_size=0.25, random_state=42, stratify=df_hiring["shortlisted"])

    feature_cols = ["years_experience", "technical_assessment", "interview_score", "degree_level"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["years_experience", "technical_assessment", "interview_score"]),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), ["degree_level"])
        ]
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ])

    clf.fit(train_df[feature_cols], train_df["shortlisted"])

    test_csv_path = "/Users/dhruvagrawal/.gemini/antigravity/scratch/fairlens/data/demos/hiring_evaluation_dataset.csv"
    model_path = "/Users/dhruvagrawal/.gemini/antigravity/scratch/fairlens/models/demos/hiring_screening_model.joblib"
    
    test_df.to_csv(test_csv_path, index=False)
    joblib.dump(clf, model_path)
    print(f"Hiring demo saved: {test_csv_path} ({len(test_df)} rows), Model: {model_path}")

if __name__ == "__main__":
    generate_loan_scenario()
    generate_hiring_scenario()
