# Predictions-Only sensitive-attribute mapping bug fix

## Root cause
When uploading the synthetic predictions CSV, the dataset's `Actual` column could be included in the automatically suggested sensitive columns. Once chosen as the ground-truth target, it became hidden from the sensitive-attribute buttons, but remained selected in React state and was still transmitted to the backend. This caused invalid audits treating ground truth as an attribute.

## Changes
- `frontend/src/components/NewAudit.tsx`: reset mappings on a new dataset; exclude suggested target from auto-selected attributes; remove target/prediction/score when their mappings change; send only currently visible, valid selected sensitive attributes.
- `backend/app/api/endpoints.py`: reject overlapping target/prediction/score and sensitive attributes with HTTP 400; also reject empty and duplicate sensitive-attribute selections.
- `backend/tests/test_api_integration.py`: four regression cases verify invalid mappings are rejected and a Department-only Predictions-Only audit computes the expected subgroup gaps.

## How to verify
Run the updated application and upload `custom_test/synthetic_disparity_predictions.csv` in Predictions-Only Mode, mapping `Actual` as target, `Predicted` as prediction, `Score` as probability, and **only Department** as sensitive attribute. Start a NEW audit. Expected: one audited attribute; 3.3% demographic parity difference; 60.0% equal opportunity difference. Old saved audits are not retroactively corrected.

Run tests on Mac (with the project's dependencies installed):

```
PYTHONPATH=backend python -m pytest backend/tests -v
```

This patch does not modify or retrain any demonstration model.
