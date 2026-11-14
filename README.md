# FairLens — AI Model Bias & Algorithmic Fairness Auditor

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.2+-3178C6?logo=typescript)](https://www.typescriptlang.org/)
[![Fairlearn](https://img.shields.io/badge/Fairlearn-0.15+-blue)](https://fairlearn.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**FairLens** is a model-agnostic, production-style machine learning governance and algorithmic fairness auditing platform. It evaluates trained classifiers for demographic disparities, equal opportunity gaps, intersectional biases, and proxy leakage across sensitive subgroups, generating executive-ready PDF audit reports and dynamic fairness-performance tradeoff simulations.

---

## 🌟 Key Capabilities

1. **Model-Agnostic Auditing**
   - Direct execution on serialized scikit-learn classifiers and preprocessing pipelines (`.joblib`, `.pkl`).
   - Supports **Predictions-Only Mode**: audit model predictions and scores exported from PyTorch, TensorFlow, XGBoost, or proprietary external scoring APIs.

2. **Mathematical Fairness Metrics Engine**
   - **Demographic Parity**: Selection rates, Demographic Parity Difference, and 80% (Four-Fifths) rule ratio.
   - **Equal Opportunity**: True Positive Rate (TPR / Recall) equality among qualified candidates.
   - **Equalized Odds**: Joint evaluation of True Positive and False Positive error rates across cohorts.
   - **Predictive Performance**: Disaggregated accuracy, precision, F1-scores, FPR, and FNR with sample-size adequacy warnings.

3. **Compound Intersectional Fairness**
   - Automatically cross-evaluates combinations of protected attributes (e.g., `Gender × Employment Type` or `Age Group × Income Tier`).
   - Identifies masked disparities where models appear superficially fair in single-attribute evaluations while inflicting severe compounding penalty on joint cohorts.

4. **Interactive Threshold & Tradeoff Simulator**
   - Dynamically slide decision cutoffs ($\tau \in [0.05, 0.95]$) to simulate real-time adjustments to accuracy, recall, and demographic parity gaps.
   - Built on exact calculations over empirical score distributions.

5. **Explainability & Proxy Leakage Detection**
   - Global feature importance analysis for linear models, tree ensembles, and scikit-learn pipelines.
   - Automated proxy warning system detecting non-protected features exhibiting high empirical correlation with sensitive demographics.

6. **Deterministic Automated Audit Findings**
   - Grounded rule-based reasoning producing traceable, prioritized audit observations with sample counts, statistical evidence, and mitigation recommendations.

7. **Publication-Grade PDF & JSON Export**
   - One-click generation of governance audit dossiers formatted with ReportLab.
   - Machine-parsable JSON output suitable for automated CI/CD and MLOps deployment gates.

---

## 📐 Mathematical Definitions of Audited Fairness Metrics

Let $A \in \{a, a_0\}$ denote a protected attribute (with reference cohort $a_0$), $Y \in \{0, 1\}$ the ground-truth outcome, and $\hat{Y} \in \{0, 1\}$ the model's binary decision.

### 1. Demographic Parity (Statistical Parity)
The probability of receiving a favorable outcome must be independent of protected cohort membership:
$$P(\hat{Y} = 1 \mid A = a) = P(\hat{Y} = 1 \mid A = a_0)$$
- **Demographic Parity Difference:**
  $$\Delta_{\text{DP}} = \max_{a} P(\hat{Y}=1 \mid A=a) - \min_{a} P(\hat{Y}=1 \mid A=a)$$
- **Demographic Parity Ratio (Four-Fifths Rule):**
  $$\text{Ratio}_{\text{DP}} = \frac{\min_{a} P(\hat{Y}=1 \mid A=a)}{\max_{a} P(\hat{Y}=1 \mid A=a)}$$

### 2. Equal Opportunity (Hardt et al., 2016)
The classifier should exhibit equal True Positive Rates (Recall) across groups among actually qualified individuals ($Y = 1$):
$$P(\hat{Y} = 1 \mid Y = 1, A = a) = P(\hat{Y} = 1 \mid Y = 1, A = a_0)$$
$$\Delta_{\text{EO}} = \max_{a} \text{TPR}_a - \min_{a} \text{TPR}_a$$

### 3. Equalized Odds
Requires equality of both True Positive Rate (TPR) and False Positive Rate (FPR) across groups:
$$P(\hat{Y} = 1 \mid Y = y, A = a) = P(\hat{Y} = 1 \mid Y = y, A = a_0) \quad \forall y \in \{0, 1\}$$
$$\Delta_{\text{EqOdds}} = \max\left( \max_{a} \text{TPR}_a - \min_{a} \text{TPR}_a, \; \max_{a} \text{FPR}_a - \min_{a} \text{FPR}_a \right)$$

### ⚠️ The Fairness Impossibility Theorem
*(Kleinberg et al., 2016; Chouldechova, 2017)*
When the base rate of the target label differs across groups ($P(Y=1 \mid A=a) \neq P(Y=1 \mid A=a_0)$), **Demographic Parity, Equal Opportunity, and Predictive Parity (Calibration) cannot be satisfied simultaneously** by any imperfect classifier. FairLens highlights these inherent tradeoffs to guide human decision-makers rather than mandating an arbitrary compromise.

---

## 🏗️ System Architecture

```
fairlens/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── endpoints.py         # REST API for datasets, models, audits, simulator
│   │   ├── ml/
│   │   │   ├── data_processor.py    # CSV validation, summary extraction, quantile binning
│   │   │   ├── model_loader.py      # Safe inspection & inference for pipelines & classifiers
│   │   │   ├── fairness_engine.py   # Demographic parity, equal opportunity, intersectional math
│   │   │   ├── findings_engine.py   # Rule-based deterministic audit reasoning
│   │   │   ├── explainability.py    # Global feature weights & proxy correlation scanner
│   │   │   ├── threshold_simulator.py # Empirical curve generator for decision thresholding
│   │   │   └── train_demos.py       # Benchmark synthetic datasets & pretrained models
│   │   ├── reports/
│   │   │   └── pdf_generator.py     # Executive ReportLab PDF dossier generation
│   │   ├── schemas/
│   │   │   └── audit.py             # Strict Pydantic v2 domain schemas
│   │   ├── storage/
│   │   │   └── db.py                # Local SQLite metadata & audit ledger
│   │   └── main.py                  # FastAPI application entrypoint with CORS
│   └── tests/
│       ├── test_fairness_engine.py  # Precision unit tests for fairness math & edge cases
│       └── test_api_integration.py  # End-to-end integration tests on FastAPI routes
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Sidebar.tsx          # Navigation sidebar with status badge
│   │   │   ├── NewAudit.tsx         # Upload, schema mapping & 1-click demo loader
│   │   │   ├── Dashboard.tsx        # High-level overview, KPI cards & findings feed
│   │   │   ├── Performance.tsx      # Confusion matrix, accuracy, ROC-AUC, error breakdown
│   │   │   ├── FairnessAnalysis.tsx # Recharts bar comparisons & subgroup tables
│   │   │   ├── Intersectional.tsx   # Multi-attribute compound cohort matrices
│   │   │   ├── Explainability.tsx   # Feature importance chart & proxy leakage warnings
│   │   │   ├── ThresholdExplorer.tsx# Dynamic slider & tradeoff curves (Accuracy vs Parity)
│   │   │   └── Reports.tsx          # PDF report viewer & JSON exporter
│   │   ├── services/api.ts          # Fetch API client
│   │   ├── types/audit.ts           # TypeScript interfaces matching backend schemas
│   │   └── App.tsx                  # Tab navigation controller
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── data/demos/                      # Demo datasets (Loan Underwriting & Hiring Screening)
├── models/demos/                    # Demo trained pipelines (GradientBoosting & RandomForest)
└── requirements.txt
```

---

## 🚀 Quickstart & Setup Guide

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### 1. Backend Setup
```bash
# Navigate to project root
cd /Users/dhruvagrawal/.gemini/antigravity/scratch/fairlens

# Activate virtual environment
source venv/bin/activate

# (Optional: install backend dependencies if not already installed)
pip install -r requirements.txt

# Run backend test suite
PYTHONPATH=backend pytest backend/tests

# Launch FastAPI backend server (Runs on port 8000)
uvicorn app.main:app --app-dir backend --reload --port 8000
```
Verify the API is running at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Frontend Setup
In a new terminal:
```bash
cd /Users/dhruvagrawal/.gemini/antigravity/scratch/fairlens/frontend

# Launch Vite development server (Runs on port 3000)
npm run dev
```
Open your browser at: [http://localhost:3000](http://localhost:3000)

---

## 🧪 Demonstration Scenarios

FairLens comes bundled with two realistic, pre-trained benchmark scenarios:

1. **Demo 1: Loan Underwriting Fair Lending Audit**
   - **Model:** Gradient Boosting Classifier with StandardScaler and OneHotEncoder preprocessing pipeline.
   - **Protected Attributes:** `gender`, `employment_type`, `age_group`.
   - **Target:** `loan_approved` ($1$ or $0$).
   - **Disparity Revealed:** While overall model accuracy is $\sim 82\%$, self-employed and gig workers experience significantly lower selection rates and higher false-negative rates than traditional salaried employees.

2. **Demo 2: Candidate Screening & Hiring Audit**
   - **Model:** Random Forest Classifier.
   - **Protected Attributes:** `gender`, `ethnicity`, `disability_status`.
   - **Target:** `shortlisted` ($1$ or $0$).
   - **Disparity Revealed:** Subconscious interviewer historical bias reflected in evaluation scores produces compounding intersectional disparities across particular demographic groups.

You can trigger either demo with a single click from the **New Audit** screen.

---

## 🔒 Security & Trusted Model Loading

- **Pickle / Joblib Serialization Notice:** Deserializing arbitrary Python `.pkl` or `.joblib` files can execute untrusted bytecode. FairLens is architected for local internal developer audits of approved artifacts. In production enterprise environments, models should be hosted behind isolated microservice sandbox containers or evaluated via **Predictions-Only Mode**.

---

## 📄 License
This project is licensed under the MIT License.

## Lab evaluation improvement pass (November 2026)
- Findings now warn when the classifier detects no positive cases, and investigate subgroup accuracy gaps as well as parity and error-rate gaps.
- An all-zero group selection rate yields an **undefined demographic parity ratio**, displayed as `N/A` rather than misleadingly treating `0/0` as 1.
- PDF reports now include a confusion matrix and more precise metric/limitation notes.
- The UI clarifies the positive class means the class of interest and that threshold simulations do not modify saved audits.
- Added regression tests and `LAB_EVAL_GUIDE.md`, plus a deliberately synthetic predictions-only fairness example in `custom_test/`.
- Local file paths now resolve relative to the project root, with environment variable overrides for DB, uploads and reports.

**Important:** Re-run each audit to generate new findings; old historical records will not be rewritten automatically. Serialized `.joblib`/`.pkl` models are only safe to load from trusted sources.
