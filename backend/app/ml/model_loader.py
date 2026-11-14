"""Model loading, validation, and inference wrapper."""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from app.schemas.audit import ModelInspectResponse

def inspect_model_file(file_path: str, model_id: str, filename: str) -> ModelInspectResponse:
    """Safely inspects a joblib/pickle scikit-learn model."""
    try:
        model = joblib.load(file_path)
    except Exception as e:
        raise ValueError(f"Failed to load model artifact: {str(e)}")

    model_type = type(model).__name__
    classes = getattr(model, "classes_", None)
    clean_classes = [int(c) if isinstance(c, (np.integer, np.int64)) 
                     else float(c) if isinstance(c, (np.floating, np.float64)) 
                     else str(c) for c in classes] if classes is not None else []
    
    # Check feature names if stored by scikit-learn
    expected_features = getattr(model, "feature_names_in_", None)
    clean_features = [str(f) for f in expected_features] if expected_features is not None else None
    
    has_proba = hasattr(model, "predict_proba")
    
    # Check pipeline steps
    pipeline_steps = None
    if hasattr(model, "steps"):
        pipeline_steps = [step[0] for step in model.steps]

    warning = None
    if not hasattr(model, "predict"):
        warning = "Object loaded does not provide a standard predict() method."

    return ModelInspectResponse(
        model_id=model_id,
        filename=filename,
        model_type=model_type,
        expected_features=clean_features,
        classes=clean_classes,
        has_predict_proba=has_proba,
        pipeline_steps=pipeline_steps,
        warning=warning
    )

def run_model_inference(model: Any, X: pd.DataFrame, positive_class: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Runs predict and predict_proba on input features X."""
    y_pred = model.predict(X)
    
    y_scores = None
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X)
            # Find column corresponding to positive_class
            classes = getattr(model, "classes_", None)
            if classes is not None and len(classes) == probas.shape[1]:
                # find index of positive class
                pos_idx = None
                for idx, c in enumerate(classes):
                    if str(c).strip().lower() == str(positive_class).strip().lower():
                        pos_idx = idx
                        break
                if pos_idx is not None:
                    y_scores = probas[:, pos_idx]
                else:
                    y_scores = probas[:, 1] if probas.shape[1] >= 2 else probas[:, 0]
            else:
                y_scores = probas[:, 1] if probas.shape[1] >= 2 else probas[:, 0]
        except Exception:
            y_scores = None

    return y_pred, y_scores
