# FairLens lab evaluation guide (November 18)

## What FairLens does
FairLens **audits existing binary classifiers**. It does not infer discriminatory intent. It calculates classification performance, group selection rates, subgroup error rates, threshold tradeoffs, and rule-based investigation findings from an evaluation dataset.

## Quick launch
1. Install dependencies in an existing Python 3.11 virtual environment, if needed: `pip install -r requirements.txt`.
2. If needed run `cd frontend && npm install && cd ..`.
3. From project root run `./start.sh`; open `http://localhost:3000`.
4. Launch backend+frontend using a terminal that remains open.

## Recommended 5-minute live demo
1. **Built-in loan model**: click New Audit > Loan demo. Show the uploaded model, target, and sensitive attributes.
2. **Performance**: explain accuracy alone is insufficient on imbalanced labels. Use the confusion matrix.
3. **Fairness Disparities**: selection rate = P(predicted positive | group). Demographic parity difference is maximum minus minimum group selection rate. Equal opportunity compares true-positive rates conditional on actual positives.
4. **Threshold Explorer**: lower the threshold, click Recalculate; show accuracy-recall and disparity changes. Explain that this is a **temporary simulation**; it does **not** update the saved audit or exported report.
5. **Findings & PDF**: show evidence-backed warnings and export a report.

## Strong bonus demo: known synthetic disparities, no trained model upload required
On **New Audit**, choose Predictions-Only Mode and upload `custom_test/synthetic_disparity_predictions.csv`.
Configure **Actual** as target, **Predicted** as prediction column, **Score** as optional score column, **Department** as the sensitive attribute, and **1** as positive class.

Expected values (180 samples; 60 per department):

| Department | Actual positive | TP | FP | Selection rate | TPR |
|---|---:|---:|---:|---:|---:|
| Engineering | 20 | 18 | 5 | 23/60 = 38.3% | 90% |
| HR | 20 | 12 | 10 | 22/60 = 36.7% | 60% |
| Sales | 20 | 6 | 15 | 21/60 = 35.0% | 30% |

- **Demographic parity difference** = 3.33 percentage points (small).
- **Equal opportunity difference** = 60 percentage points (large).
- This proves why fairness cannot be judged by one metric alone. These records are **artificially constructed for educational validation**, not evidence about real employees.

## Another useful demo: zero-recall classifier
Our custom `attrition_model.joblib` and `attrition_test.csv` show a model with ~85.7% accuracy but 0% recall at its default threshold. The *updated* findings engine should flag severe positive-class failure **and** an accuracy gap across departments.

## Important caveats for your viva
- A gap signals something to investigate; it is not proof of illegal discrimination.
- The positive class need not be a *favorable* outcome; for fraud or attrition, positive means the event of interest.
- A group with 0 actual positives has undefined recall. When everyone is predicted negative, the demographic parity **ratio** is `0/0`, so it is displayed as `N/A`.
- Small subgroup sizes and differences in base rates affect reliability and interpretation.
- This is not a general deep-learning model uploader: `.joblib` and `.pkl` require **trusted local scikit-learn** model artifacts; arbitrary pickle loading can execute code. Predictions-only works with outputs from other systems.

## Validation on your Mac (required before evaluation)
From the project root, with venv activated:

```bash
PYTHONPATH=backend python -m pytest backend/tests -v
cd frontend && npm run build
```

Both commands must succeed on your computer before declaring the app production-ready. Then re-run both built-in demos and a **fresh audit**. Old stored audits and old PDFs will still show their old findings until re-executed.
