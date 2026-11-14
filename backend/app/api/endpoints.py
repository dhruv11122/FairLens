"""FastAPI Router for FairLens AI Bias & Fairness Auditing Platform."""
import os
import hashlib
import uuid
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse

from app.schemas.audit import (
    DatasetPreviewResponse, ModelInspectResponse, AuditRequest, AuditSummary,
    ThresholdSimulationRequest, ThresholdSimulationResponse
)
from app.storage.db import (
    save_model_meta, get_model_meta, save_dataset_meta, get_dataset_meta,
    save_audit, get_audit
)
from app.ml.data_processor import inspect_dataset, apply_numeric_binning
from app.ml.model_loader import inspect_model_file, run_model_inference
from app.ml.fairness_engine import (
    compute_binary_performance, compute_attribute_fairness, compute_intersectional_fairness
)
from app.ml.findings_engine import generate_audit_findings
from app.ml.threshold_simulator import generate_threshold_curve, simulate_threshold
from app.ml.explainability import compute_explainability
from app.reports.pdf_generator import generate_pdf_report

router = APIRouter(prefix="/api", tags=["FairLens Core API"])

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_UPLOAD_DIR = os.getenv("FAIRLENS_UPLOAD_DIR", str(PROJECT_ROOT / "uploads"))
REPORTS_DIR = os.getenv("FAIRLENS_REPORTS_DIR", str(PROJECT_ROOT / "reports"))
os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Cache for evaluated datasets and models in memory during session
SESSION_DATASETS: Dict[str, pd.DataFrame] = {}
SESSION_MODELS: Dict[str, Any] = {}

# ==========================================
# Model Upload & Inspection
# ==========================================

@router.post("/models/upload", response_model=ModelInspectResponse)
async def upload_model(file: UploadFile = File(...)):
    """Uploads a trained scikit-learn classifier model artifact (.joblib or .pkl)."""
    if not (file.filename.endswith(".joblib") or file.filename.endswith(".pkl") or file.filename.endswith(".pickle")):
        raise HTTPException(status_code=400, detail="Only .joblib or .pkl model artifacts are supported.")

    # Public demo mode must never deserialize arbitrary user-supplied pickle/joblib.
    # Only exact bundled, trusted demo artifacts are accepted (content-hash allowlist).
    content = await file.read()
    if os.getenv("FAIRLENS_PUBLIC_DEMO_ONLY") == "1":
        known_models = [
            PROJECT_ROOT / "models" / "demos" / "loan_approval_model.joblib",
            PROJECT_ROOT / "models" / "demos" / "hiring_screening_model.joblib",
        ]
        trusted_hashes = {
            hashlib.sha256(path.read_bytes()).digest()
            for path in known_models if path.is_file()
        }
        if hashlib.sha256(content).digest() not in trusted_hashes:
            raise HTTPException(
                status_code=403,
                detail="Public demo accepts only bundled trusted models. Use Predictions-Only mode for custom predictions or run FairLens locally for custom model uploads."
            )
    elif len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Model artifact exceeds 20 MB limit.")

    model_id = str(uuid.uuid4())[:8]
    save_path = os.path.join(BASE_UPLOAD_DIR, f"{model_id}_{file.filename}")
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        inspect_resp = inspect_model_file(save_path, model_id, file.filename)
        # Store in db
        save_model_meta(
            model_id=model_id,
            filename=file.filename,
            file_path=save_path,
            model_type=inspect_resp.model_type,
            classes=inspect_resp.classes or [],
            expected_features=inspect_resp.expected_features,
            has_proba=inspect_resp.has_predict_proba,
            pipeline_steps=inspect_resp.pipeline_steps
        )
        return inspect_resp
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted model artifact: {str(e)}")

# ==========================================
# Dataset Upload & Inspection
# ==========================================

@router.post("/datasets/upload", response_model=DatasetPreviewResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Uploads an evaluation dataset in CSV format."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV datasets are currently supported.")

    content = await file.read()
    if os.getenv("FAIRLENS_PUBLIC_DEMO_ONLY") == "1" and len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Public demo CSV upload limit is 5 MB.")
    dataset_id = str(uuid.uuid4())[:8]
    save_path = os.path.join(BASE_UPLOAD_DIR, f"{dataset_id}_{file.filename}")

    with open(save_path, "wb") as f:
        f.write(content)

    try:
        df = pd.read_csv(save_path)
        SESSION_DATASETS[dataset_id] = df
        preview = inspect_dataset(df, dataset_id, file.filename)
        save_dataset_meta(
            dataset_id=dataset_id,
            filename=file.filename,
            file_path=save_path,
            total_rows=preview.total_rows,
            total_columns=preview.total_columns,
            columns_meta=[col.model_dump() for col in preview.columns]
        )
        return preview
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV dataset: {str(e)}")

# ==========================================
# Audit Execution
# ==========================================

@router.post("/audits/run", response_model=AuditSummary)
async def run_audit(req: AuditRequest):
    """Executes an end-to-end fairness and performance audit."""
    # 1. Load dataset
    df = SESSION_DATASETS.get(req.dataset_id)
    if df is None:
        meta = get_dataset_meta(req.dataset_id)
        if not meta or not os.path.exists(meta["file_path"]):
            raise HTTPException(status_code=404, detail="Dataset not found.")
        df = pd.read_csv(meta["file_path"])
        SESSION_DATASETS[req.dataset_id] = df

    df = df.copy()

    # 2. Validate columns
    if req.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{req.target_column}' not found in dataset.")
    if not req.sensitive_attributes:
        raise HTTPException(status_code=400, detail="Select at least one sensitive attribute to audit.")
    if len(req.sensitive_attributes) != len(set(req.sensitive_attributes)):
        raise HTTPException(status_code=400, detail="Sensitive attributes must be unique.")
    reserved_columns = {req.target_column}
    if req.is_predictions_only:
        reserved_columns.update(c for c in (req.prediction_column, req.score_column) if c)
    invalid_sensitive = [col for col in req.sensitive_attributes if col in reserved_columns]
    if invalid_sensitive:
        raise HTTPException(
            status_code=400,
            detail=f"Target, prediction and score columns cannot be audited as sensitive attributes: {invalid_sensitive}"
        )
    for sens in req.sensitive_attributes:
        if sens not in df.columns:
            raise HTTPException(status_code=400, detail=f"Sensitive attribute '{sens}' not found in dataset.")

    # 3. Handle model inference vs predictions-only
    model = None
    model_name = None
    has_probabilities = False
    y_scores = None

    if req.is_predictions_only:
        if not req.prediction_column or req.prediction_column not in df.columns:
            raise HTTPException(status_code=400, detail="In predictions-only mode, 'prediction_column' must exist.")
        y_pred = df[req.prediction_column].values
        if req.score_column and req.score_column in df.columns:
            try:
                y_scores = df[req.score_column].values.astype(float)
                has_probabilities = True
            except Exception:
                y_scores = None
    else:
        if not req.model_id:
            raise HTTPException(status_code=400, detail="A valid 'model_id' is required unless using predictions-only mode.")
        m_meta = get_model_meta(req.model_id)
        if not m_meta or not os.path.exists(m_meta["file_path"]):
            raise HTTPException(status_code=404, detail="Model artifact not found.")
        
        model_name = m_meta["filename"]
        try:
            model = joblib.load(m_meta["file_path"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model from disk: {str(e)}")

        # Prepare X features: exclude target and any non-feature sensitive attributes if not in model
        if req.feature_columns:
            missing_feats = [col for col in req.feature_columns if col not in df.columns]
            if missing_feats:
                raise HTTPException(status_code=400, detail=f"Missing feature columns in dataset: {missing_feats}")
            X = df[req.feature_columns]
        elif hasattr(model, "feature_names_in_"):
            # If the model explicitly recorded its training feature names, select them in exact order
            expected_feats = list(model.feature_names_in_)
            missing_feats = [col for col in expected_feats if col not in df.columns]
            if missing_feats:
                raise HTTPException(status_code=400, detail=f"Dataset is missing features required by model: {missing_feats}")
            X = df[expected_feats]
        else:
            # Auto-infer: exclude target column and prediction/score columns
            exclude = [req.target_column]
            if req.prediction_column:
                exclude.append(req.prediction_column)
            if req.score_column:
                exclude.append(req.score_column)
            feature_cols = [c for c in df.columns if c not in exclude]
            X = df[feature_cols]

        try:
            y_pred, y_scores = run_model_inference(model, X, req.positive_class)
            has_probabilities = (y_scores is not None)
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Model inference failed on dataset features. Ensure dataset schema matches model expectations: {str(e)}"
            )

    # 4. Standardize Ground Truth Labels to Binary 0/1
    raw_y_true = df[req.target_column].values
    pos_str = str(req.positive_class).strip().lower()
    y_true_binary = np.array([1 if str(val).strip().lower() == pos_str else 0 for val in raw_y_true])
    y_pred_binary = np.array([1 if str(val).strip().lower() == pos_str or str(val) == "1" else 0 for val in y_pred])

    # 5. Apply Numeric Binning to continuous sensitive attributes if configured
    df_eval = apply_numeric_binning(df, req.binning_configs)
    df_eval["__y_true__"] = y_true_binary
    df_eval["__y_pred__"] = y_pred_binary
    if y_scores is not None:
        df_eval["__y_score__"] = y_scores

    # 6. Overall Performance Metrics
    overall_perf = compute_binary_performance(y_true_binary, y_pred_binary, y_scores)

    # 7. Subgroup Fairness Metrics per sensitive attribute
    attribute_fairness_results = {}
    for sens in req.sensitive_attributes:
        ref_grp = req.reference_groups.get(sens) if req.reference_groups else None
        res = compute_attribute_fairness(
            df=df_eval,
            attribute=sens,
            y_true_col="__y_true__",
            y_pred_col="__y_pred__",
            reference_group=ref_grp,
            min_subgroup_size=req.min_subgroup_size
        )
        attribute_fairness_results[sens] = res

    # 8. Intersectional Fairness Analysis (if >= 2 sensitive attributes)
    intersectional_result = None
    if len(req.sensitive_attributes) >= 2:
        intersectional_result = compute_intersectional_fairness(
            df=df_eval,
            attributes=req.sensitive_attributes[:2],  # Primary intersection pair
            y_true_col="__y_true__",
            y_pred_col="__y_pred__",
            min_subgroup_size=req.min_subgroup_size
        )

    # 9. Automated Findings Engine
    findings = generate_audit_findings(
        attribute_fairness=attribute_fairness_results,
        overall_performance=overall_perf,
        disparity_threshold=req.disparity_threshold,
        min_subgroup_size=req.min_subgroup_size
    )

    # 10. Explainability & Proxy Checks
    explainability = None
    if not req.is_predictions_only and model is not None:
        explainability = compute_explainability(
            model=model,
            X=X,
            sensitive_df=df[req.sensitive_attributes],
            sensitive_attrs=req.sensitive_attributes
        )
    else:
        # Check proxy correlations even in predictions-only
        feature_cols = [c for c in df.columns if c != req.target_column and c not in req.sensitive_attributes]
        explainability = compute_explainability(
            model=None,
            X=df[feature_cols] if feature_cols else pd.DataFrame(),
            sensitive_df=df[req.sensitive_attributes],
            sensitive_attrs=req.sensitive_attributes
        )

    audit_id = str(uuid.uuid4())[:8]
    summary = AuditSummary(
        audit_id=audit_id,
        created_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        model_id=req.model_id,
        model_name=model_name,
        dataset_id=req.dataset_id,
        dataset_name=get_dataset_meta(req.dataset_id)["filename"] if get_dataset_meta(req.dataset_id) else "Evaluation Dataset",
        target_column=req.target_column,
        positive_class=str(req.positive_class),
        sensitive_attributes=req.sensitive_attributes,
        overall_performance=overall_perf,
        attribute_fairness=attribute_fairness_results,
        intersectional_fairness=intersectional_result,
        findings=findings,
        explainability=explainability,
        has_probabilities=has_probabilities,
        disparity_threshold=req.disparity_threshold,
        min_subgroup_size=req.min_subgroup_size,
        is_predictions_only=req.is_predictions_only
    )

    # Persist audit summary and evaluation df in memory for simulator
    save_audit(audit_id, req.model_id, req.dataset_id, req.model_dump(), summary.model_dump())
    SESSION_DATASETS[f"audit_{audit_id}"] = df_eval

    return summary

# ==========================================
# Retrieve Audit by ID
# ==========================================

@router.get("/audits/{audit_id}", response_model=AuditSummary)
async def get_audit_by_id(audit_id: str):
    audit_data = get_audit(audit_id)
    if not audit_data:
        raise HTTPException(status_code=404, detail="Audit not found.")
    return AuditSummary(**audit_data["summary"])

# ==========================================
# Threshold Simulator Endpoint
# ==========================================

@router.post("/audits/{audit_id}/threshold-analysis", response_model=ThresholdSimulationResponse)
async def analyze_threshold(audit_id: str, req: ThresholdSimulationRequest):
    audit_data = get_audit(audit_id)
    if not audit_data:
        raise HTTPException(status_code=404, detail="Audit not found.")
    
    df_eval = SESSION_DATASETS.get(f"audit_{audit_id}")
    if df_eval is None or "__y_score__" not in df_eval.columns:
        # Check if audit summary has probabilities
        summary = audit_data.get("summary", {})
        if not summary.get("has_probabilities", False):
            raise HTTPException(status_code=400, detail="This audit does not possess prediction probability scores.")
        
        # Reload dataset from disk and re-score
        meta = get_dataset_meta(audit_data["dataset_id"])
        if not meta or not os.path.exists(meta["file_path"]):
            raise HTTPException(status_code=400, detail="Audit dataset no longer accessible for threshold simulation.")
        df_raw = pd.read_csv(meta["file_path"])
        config = audit_data["config"]
        
        # Re-derive predictions or scores
        if config.get("is_predictions_only"):
            score_col = config.get("score_column")
            if not score_col or score_col not in df_raw.columns:
                raise HTTPException(status_code=400, detail="Score column not available.")
            y_scores = df_raw[score_col].values.astype(float)
        else:
            m_meta = get_model_meta(audit_data["model_id"])
            if not m_meta or not os.path.exists(m_meta["file_path"]):
                raise HTTPException(status_code=400, detail="Model artifact no longer accessible.")
            model = joblib.load(m_meta["file_path"])
            feature_cols = config.get("feature_columns")
            if not feature_cols and hasattr(model, "feature_names_in_"):
                feature_cols = list(model.feature_names_in_)
            elif not feature_cols:
                feature_cols = [c for c in df_raw.columns if c != config.get("target_column")]
            _, y_scores = run_model_inference(model, df_raw[feature_cols], config.get("positive_class"))
        
        if y_scores is None:
            raise HTTPException(status_code=400, detail="Could not reconstruct probability scores.")
        
        raw_y_true = df_raw[config.get("target_column")].values
        pos_str = str(config.get("positive_class")).strip().lower()
        y_true_binary = np.array([1 if str(val).strip().lower() == pos_str else 0 for val in raw_y_true])
        
        df_eval = df_raw.copy()
        df_eval["__y_true__"] = y_true_binary
        df_eval["__y_score__"] = y_scores
        SESSION_DATASETS[f"audit_{audit_id}"] = df_eval

    if req.attribute not in df_eval.columns:
        raise HTTPException(status_code=400, detail=f"Attribute '{req.attribute}' not found in audit dataset.")

    y_true = df_eval["__y_true__"].values
    y_scores = df_eval["__y_score__"].values
    sensitive_series = df_eval[req.attribute].astype(str)

    res = generate_threshold_curve(
        y_true=y_true,
        y_scores=y_scores,
        sensitive_series=sensitive_series,
        current_threshold=req.threshold
    )
    return res

# ==========================================
# PDF and JSON Export
# ==========================================

@router.get("/audits/{audit_id}/report")
async def download_report_pdf(audit_id: str):
    audit_data = get_audit(audit_id)
    if not audit_data:
        raise HTTPException(status_code=404, detail="Audit not found.")
    
    summary = AuditSummary(**audit_data["summary"])
    pdf_filename = f"FairLens_Audit_{audit_id}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
    
    generate_pdf_report(summary, pdf_path)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=pdf_filename
    )

@router.get("/audits/{audit_id}/export-json")
async def export_audit_json(audit_id: str):
    audit_data = get_audit(audit_id)
    if not audit_data:
        raise HTTPException(status_code=404, detail="Audit not found.")
    return audit_data["summary"]
