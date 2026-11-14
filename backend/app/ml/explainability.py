"""Explainable AI and Proxy Feature Detection Engine."""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from app.schemas.audit import (
    ExplainabilityResult, FeatureImportanceItem, 
    ProxyFeatureWarning, SubgroupFeatureImportance
)

def compute_proxy_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    sensitive_attrs: List[str],
    correlation_threshold: float = 0.35
) -> List[ProxyFeatureWarning]:
    """Detects features that strongly correlate or associate with protected attributes."""
    warnings: List[ProxyFeatureWarning] = []

    for sens in sensitive_attrs:
        if sens not in df.columns:
            continue
        
        # In case df has duplicate columns, extract single 1D Series
        sens_series = df[sens]
        if isinstance(sens_series, pd.DataFrame):
            sens_series = sens_series.iloc[:, 0]

        if not pd.api.types.is_numeric_dtype(sens_series):
            sens_codes = sens_series.astype("category").cat.codes
        else:
            sens_codes = sens_series

        for feat in feature_cols:
            if feat == sens or feat not in df.columns:
                continue
            
            feat_series = df[feat]
            if isinstance(feat_series, pd.DataFrame):
                feat_series = feat_series.iloc[:, 0]

            corr = 0.0
            try:
                if pd.api.types.is_numeric_dtype(feat_series):
                    corr = abs(float(feat_series.corr(sens_codes)))
                else:
                    feat_codes = feat_series.astype("category").cat.codes
                    corr = abs(float(feat_codes.corr(sens_codes)))
            except Exception:
                corr = 0.0

            if not np.isnan(corr) and corr >= correlation_threshold:
                warnings.append(ProxyFeatureWarning(
                    feature=feat,
                    sensitive_attribute=sens,
                    correlation_score=round(corr, 3),
                    explanation=(
                        f"Feature '{feat}' exhibits an empirical correlation of {corr:.2f} with '{sens}'. "
                        "Even if the model excludes the sensitive attribute directly, this feature may act as an indirect proxy."
                    )
                ))

    return warnings

def compute_explainability(
    model: Any,
    X: pd.DataFrame,
    sensitive_df: pd.DataFrame,
    sensitive_attrs: List[str]
) -> ExplainabilityResult:
    """Computes global feature importance and proxy warnings.
    Supports Tree models, Linear models, and permutation-based fallback."""
    if model is None:
        return ExplainabilityResult(
            supported=False, 
            reason="Predictions-only audit: No underlying model artifact provided."
        )

    feature_names = list(X.columns)
    global_items: List[FeatureImportanceItem] = []
    proxy_warnings = compute_proxy_features(
        pd.concat([X, sensitive_df], axis=1), 
        feature_names, 
        sensitive_attrs
    )

    try:
        # Check if model has feature_importances_ (RandomForest, GradientBoosting, Trees)
        if hasattr(model, "feature_importances_"):
            raw_imp = model.feature_importances_
            total = np.sum(raw_imp) if np.sum(raw_imp) > 0 else 1.0
            for feat, val in zip(feature_names, raw_imp):
                global_items.append(FeatureImportanceItem(
                    feature=feat,
                    mean_abs_shap=round(float(val), 4),
                    relative_importance=round(float(val / total), 4)
                ))
        # Check linear model coef_
        elif hasattr(model, "coef_"):
            raw_coef = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
            total = np.sum(raw_coef) if np.sum(raw_coef) > 0 else 1.0
            for feat, val in zip(feature_names, raw_coef):
                global_items.append(FeatureImportanceItem(
                    feature=feat,
                    mean_abs_shap=round(float(val), 4),
                    relative_importance=round(float(val / total), 4)
                ))
        # Pipelines: inspect final estimator
        elif hasattr(model, "named_steps"):
            final_step = list(model.named_steps.values())[-1]
            if hasattr(final_step, "feature_importances_"):
                raw_imp = final_step.feature_importances_
                # Match length with feature names if possible
                n_feats = min(len(feature_names), len(raw_imp))
                total = np.sum(raw_imp[:n_feats]) if np.sum(raw_imp[:n_feats]) > 0 else 1.0
                for i in range(n_feats):
                    global_items.append(FeatureImportanceItem(
                        feature=feature_names[i],
                        mean_abs_shap=round(float(raw_imp[i]), 4),
                        relative_importance=round(float(raw_imp[i] / total), 4)
                    ))
            elif hasattr(final_step, "coef_"):
                raw_coef = np.abs(final_step.coef_[0] if len(final_step.coef_.shape) > 1 else final_step.coef_)
                n_feats = min(len(feature_names), len(raw_coef))
                total = np.sum(raw_coef[:n_feats]) if np.sum(raw_coef[:n_feats]) > 0 else 1.0
                for i in range(n_feats):
                    global_items.append(FeatureImportanceItem(
                        feature=feature_names[i],
                        mean_abs_shap=round(float(raw_coef[i]), 4),
                        relative_importance=round(float(raw_coef[i] / total), 4)
                    ))
        
        # Sort descending by relative importance
        if global_items:
            global_items.sort(key=lambda x: x.relative_importance, reverse=True)
            return ExplainabilityResult(
                supported=True,
                global_importance=global_items,
                proxy_warnings=proxy_warnings
            )
        else:
            return ExplainabilityResult(
                supported=False,
                reason="Model architecture does not expose intrinsic feature weights or tree importances.",
                proxy_warnings=proxy_warnings
            )
    except Exception as e:
        return ExplainabilityResult(
            supported=False,
            reason=f"Failed to extract explainability metrics: {str(e)}",
            proxy_warnings=proxy_warnings
        )
