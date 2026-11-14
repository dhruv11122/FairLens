"""Threshold Simulator and Curve Generator for Decision Tradeoff Exploration."""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from app.schemas.audit import ThresholdPoint, ThresholdSimulationResponse

def simulate_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    sensitive_series: pd.Series,
    threshold: float
) -> ThresholdPoint:
    """Calculates model metrics and fairness disparities under a specific decision cutoff."""
    y_true_b = y_true.astype(int)
    y_pred_b = (y_scores >= threshold).astype(int)
    
    total = len(y_true_b)
    pred_pos = int(np.sum(y_pred_b == 1))
    pos_samples = int(np.sum(y_true_b == 1))

    acc = float(accuracy_score(y_true_b, y_pred_b)) if total > 0 else 0.0
    prec = float(precision_score(y_true_b, y_pred_b, zero_division=0)) if pred_pos > 0 else 0.0
    rec = float(recall_score(y_true_b, y_pred_b, zero_division=0)) if pos_samples > 0 else 0.0
    f1 = float(f1_score(y_true_b, y_pred_b, zero_division=0)) if (prec + rec) > 0 else 0.0
    sel_rate = float(pred_pos / total) if total > 0 else 0.0

    # Demographic Parity Diff & Equal Opportunity Diff across sensitive subgroups
    unique_groups = sensitive_series.unique()
    group_sel_rates = []
    group_tprs = []
    group_fprs = []

    for grp in unique_groups:
        mask = (sensitive_series == grp).values
        n_grp = int(np.sum(mask))
        if n_grp == 0:
            continue
        
        y_t_grp = y_true_b[mask]
        y_p_grp = y_pred_b[mask]

        sr = float(np.sum(y_p_grp == 1) / n_grp)
        group_sel_rates.append(sr)

        n_pos_grp = int(np.sum(y_t_grp == 1))
        n_neg_grp = int(np.sum(y_t_grp == 0))

        if n_pos_grp > 0:
            tpr = float(np.sum((y_t_grp == 1) & (y_p_grp == 1)) / n_pos_grp)
            group_tprs.append(tpr)

        if n_neg_grp > 0:
            fpr = float(np.sum((y_t_grp == 0) & (y_p_grp == 1)) / n_neg_grp)
            group_fprs.append(fpr)

    dp_diff = float(max(group_sel_rates) - min(group_sel_rates)) if group_sel_rates else 0.0
    eq_opp_diff = float(max(group_tprs) - min(group_tprs)) if len(group_tprs) > 1 else None
    
    eq_odds_diff = None
    if eq_opp_diff is not None and len(group_fprs) > 1:
        fpr_diff = float(max(group_fprs) - min(group_fprs))
        eq_odds_diff = float(max(eq_opp_diff, fpr_diff))

    return ThresholdPoint(
        threshold=round(threshold, 3),
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1_score=round(f1, 4),
        selection_rate=round(sel_rate, 4),
        demographic_parity_diff=round(dp_diff, 4),
        equal_opportunity_diff=round(eq_opp_diff, 4) if eq_opp_diff is not None else None,
        equalized_odds_diff=round(eq_odds_diff, 4) if eq_odds_diff is not None else None
    )

def generate_threshold_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    sensitive_series: pd.Series,
    current_threshold: float = 0.50,
    steps: int = 25
) -> ThresholdSimulationResponse:
    """Generates the full tradeoff curve from threshold 0.05 to 0.95."""
    thresholds = np.linspace(0.05, 0.95, steps)
    curve: List[ThresholdPoint] = []

    for t in thresholds:
        curve.append(simulate_threshold(y_true, y_scores, sensitive_series, float(t)))

    current_pt = simulate_threshold(y_true, y_scores, sensitive_series, current_threshold)

    return ThresholdSimulationResponse(
        current=current_pt,
        curve=curve
    )
