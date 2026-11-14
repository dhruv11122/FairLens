"""Comprehensive API Integration & Negative/Edge Case Tests for FairLens."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

# ==============================================================================
# Full Pipeline Test 1: Loan Approval Model & Dataset
# ==============================================================================

def test_full_loan_audit_workflow():
    csv_path = str(PROJECT_ROOT / "data/demos/loan_evaluation_dataset.csv")
    model_path = str(PROJECT_ROOT / "models/demos/loan_approval_model.joblib")
    
    with open(csv_path, "rb") as f:
        ds_resp = client.post("/api/datasets/upload", files={"file": ("loan.csv", f, "text/csv")})
    assert ds_resp.status_code == 200
    dataset_id = ds_resp.json()["dataset_id"]

    with open(model_path, "rb") as f:
        m_resp = client.post("/api/models/upload", files={"file": ("loan_model.joblib", f, "application/octet-stream")})
    assert m_resp.status_code == 200
    model_id = m_resp.json()["model_id"]

    audit_payload = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "target_column": "loan_approved",
        "positive_class": "1",
        "sensitive_attributes": ["gender", "employment_type", "age_group"],
        "disparity_threshold": 0.10,
        "min_subgroup_size": 15,
        "is_predictions_only": False
    }

    audit_resp = client.post("/api/audits/run", json=audit_payload)
    assert audit_resp.status_code == 200
    audit_data = audit_resp.json()
    audit_id = audit_data["audit_id"]

    assert audit_data["overall_performance"]["accuracy"] > 0.65
    assert "gender" in audit_data["attribute_fairness"]
    assert "employment_type" in audit_data["attribute_fairness"]
    assert audit_data["has_probabilities"] is True

    # PDF & JSON reports
    pdf_resp = client.get(f"/api/audits/{audit_id}/report")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"

    json_resp = client.get(f"/api/audits/{audit_id}/export-json")
    assert json_resp.status_code == 200
    assert json_resp.json()["audit_id"] == audit_id

# ==============================================================================
# Full Pipeline Test 2: Hiring / Candidate Screening Demo
# ==============================================================================

def test_full_hiring_audit_workflow():
    csv_path = str(PROJECT_ROOT / "data/demos/hiring_evaluation_dataset.csv")
    model_path = str(PROJECT_ROOT / "models/demos/hiring_screening_model.joblib")
    
    with open(csv_path, "rb") as f:
        ds_resp = client.post("/api/datasets/upload", files={"file": ("hiring.csv", f, "text/csv")})
    assert ds_resp.status_code == 200
    dataset_id = ds_resp.json()["dataset_id"]

    with open(model_path, "rb") as f:
        m_resp = client.post("/api/models/upload", files={"file": ("hiring_model.joblib", f, "application/octet-stream")})
    assert m_resp.status_code == 200
    model_id = m_resp.json()["model_id"]

    audit_payload = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "target_column": "shortlisted",
        "positive_class": "1",
        "sensitive_attributes": ["gender", "ethnicity", "disability_status"],
        "disparity_threshold": 0.10,
        "min_subgroup_size": 15,
        "is_predictions_only": False
    }

    audit_resp = client.post("/api/audits/run", json=audit_payload)
    assert audit_resp.status_code == 200
    audit_data = audit_resp.json()
    assert audit_data["intersectional_fairness"]["total_groups"] > 0
    assert len(audit_data["findings"]) > 0

# ==============================================================================
# Pipeline Test 3: Predictions-Only Mode
# ==============================================================================

def test_predictions_only_audit_workflow():
    # Build synthetic predictions-only dataset
    df = pd.DataFrame({
        "gender": ["F", "F", "M", "M", "F", "M", "F", "M"],
        "actual_y": [1, 0, 1, 0, 1, 1, 0, 0],
        "pred_y":   [1, 0, 1, 0, 0, 1, 0, 0],
        "prob_score": [0.85, 0.15, 0.90, 0.20, 0.45, 0.78, 0.30, 0.10]
    })
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    ds_resp = client.post("/api/datasets/upload", files={"file": ("predictions.csv", io.BytesIO(csv_bytes), "text/csv")})
    assert ds_resp.status_code == 200
    dataset_id = ds_resp.json()["dataset_id"]

    audit_payload = {
        "dataset_id": dataset_id,
        "target_column": "actual_y",
        "positive_class": "1",
        "sensitive_attributes": ["gender"],
        "is_predictions_only": True,
        "prediction_column": "pred_y",
        "score_column": "prob_score",
        "min_subgroup_size": 2
    }

    audit_resp = client.post("/api/audits/run", json=audit_payload)
    assert audit_resp.status_code == 200
    audit = audit_resp.json()
    assert audit["is_predictions_only"] is True
    assert audit["has_probabilities"] is True
    assert audit["overall_performance"]["accuracy"] == 0.875

# ==============================================================================
# Pipeline Test 4: Negative & Input Validation Error Checks
# ==============================================================================

def test_error_missing_target_column():
    csv_path = str(PROJECT_ROOT / "data/demos/loan_evaluation_dataset.csv")
    with open(csv_path, "rb") as f:
        ds_resp = client.post("/api/datasets/upload", files={"file": ("loan.csv", f, "text/csv")})
    dataset_id = ds_resp.json()["dataset_id"]

    audit_payload = {
        "dataset_id": dataset_id,
        "target_column": "non_existent_column",
        "positive_class": "1",
        "sensitive_attributes": ["gender"],
        "is_predictions_only": True,
        "prediction_column": "gender"
    }
    audit_resp = client.post("/api/audits/run", json=audit_payload)
    assert audit_resp.status_code == 400
    assert "Target column 'non_existent_column' not found" in audit_resp.json()["detail"]

def test_error_missing_sensitive_attribute():
    csv_path = str(PROJECT_ROOT / "data/demos/loan_evaluation_dataset.csv")
    with open(csv_path, "rb") as f:
        ds_resp = client.post("/api/datasets/upload", files={"file": ("loan.csv", f, "text/csv")})
    dataset_id = ds_resp.json()["dataset_id"]

    audit_payload = {
        "dataset_id": dataset_id,
        "target_column": "loan_approved",
        "positive_class": "1",
        "sensitive_attributes": ["invalid_protected_attr"],
        "is_predictions_only": True,
        "prediction_column": "loan_approved"
    }
    audit_resp = client.post("/api/audits/run", json=audit_payload)
    assert audit_resp.status_code == 400
    assert "Sensitive attribute 'invalid_protected_attr' not found" in audit_resp.json()["detail"]

def test_error_invalid_file_extensions():
    # Attempt to upload invalid dataset type
    ds_resp = client.post("/api/datasets/upload", files={"file": ("test.txt", b"hello", "text/plain")})
    assert ds_resp.status_code == 400

    # Attempt to upload invalid model type
    m_resp = client.post("/api/models/upload", files={"file": ("test.txt", b"hello", "text/plain")})
    assert m_resp.status_code == 400


@pytest.mark.parametrize("invalid_sensitive", ["Actual", "Predicted", "Score"])
def test_predictions_only_rejects_reserved_sensitive_columns(invalid_sensitive):
    """The ground-truth, prediction and score mappings cannot become audit groups."""
    df = pd.DataFrame({
        "Department": ["HR", "HR", "Sales", "Sales"],
        "Actual": [1, 0, 1, 0],
        "Predicted": [1, 0, 0, 1],
        "Score": [0.8, 0.2, 0.3, 0.7],
    })
    upload = client.post(
        "/api/datasets/upload",
        files={"file": ("reserved.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")}
    )
    assert upload.status_code == 200
    payload = {
        "dataset_id": upload.json()["dataset_id"],
        "target_column": "Actual",
        "positive_class": "1",
        "sensitive_attributes": ["Department", invalid_sensitive],
        "is_predictions_only": True,
        "prediction_column": "Predicted",
        "score_column": "Score",
    }
    response = client.post("/api/audits/run", json=payload)
    assert response.status_code == 400
    assert "cannot be audited" in response.json()["detail"]


def test_predictions_only_audits_only_explicit_department():
    """Regression for hidden 'Actual' selection in the Predictions-Only screen."""
    csv_path = PROJECT_ROOT / "custom_test/synthetic_disparity_predictions.csv"
    with open(csv_path, "rb") as f:
        response = client.post(
            "/api/datasets/upload",
            files={"file": ("synthetic_disparity_predictions.csv", f, "text/csv")}
        )
    assert response.status_code == 200
    payload = {
        "dataset_id": response.json()["dataset_id"],
        "target_column": "Actual",
        "positive_class": "1",
        "sensitive_attributes": ["Department"],
        "is_predictions_only": True,
        "prediction_column": "Predicted",
        "score_column": "Score",
    }
    result = client.post("/api/audits/run", json=payload)
    assert result.status_code == 200, result.text
    audit = result.json()
    assert audit["sensitive_attributes"] == ["Department"]
    assert set(audit["attribute_fairness"]) == {"Department"}
    assert abs(audit["attribute_fairness"]["Department"]["overall_demographic_parity_difference"] - 2 / 60) < 1e-3
    assert abs(audit["attribute_fairness"]["Department"]["overall_equal_opportunity_difference"] - 0.6) < 1e-3
    assert not any("'Actual'" in finding.get("title", "") for finding in audit["findings"])
