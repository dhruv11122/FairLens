"""Comprehensive Algorithmic Fairness & Performance Calculation Engine."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, roc_auc_score
)
from fairlearn.metrics import (
    MetricFrame, selection_rate, 
    demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference
)
from app.schemas.audit import (
    PerformanceMetrics, ConfusionMatrix, SubgroupMetrics, 
    SubgroupDisparity, AttributeFairnessResult, IntersectionalAnalysisResult, IntersectionalCell
)

def compute_binary_performance(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_score: Optional[np.ndarray] = None
) -> PerformanceMetrics:
    """Computes overall classification metrics on binary labels (0 and 1)."""
    # Ensure binary format (0 and 1)
    y_true_b = y_true.astype(int)
    y_pred_b = y_pred.astype(int)
    
    total = len(y_true_b)
    pos_samples = int(np.sum(y_true_b == 1))
    neg_samples = int(np.sum(y_true_b == 0))

    acc = float(accuracy_score(y_true_b, y_pred_b)) if total > 0 else 0.0
    
    # Balanced accuracy handles single-class gracefully
    try:
        bal_acc = float(balanced_accuracy_score(y_true_b, y_pred_b))
    except Exception:
        bal_acc = acc

    # Precision, Recall, F1 (zero_division=np.nan or handled gracefully)
    pred_positives = np.sum(y_pred_b == 1)
    true_positives = np.sum((y_true_b == 1) & (y_pred_b == 1))
    false_positives = np.sum((y_true_b == 0) & (y_pred_b == 1))
    true_negatives = np.sum((y_true_b == 0) & (y_pred_b == 0))
    false_negatives = np.sum((y_true_b == 1) & (y_pred_b == 0))

    prec = float(true_positives / pred_positives) if pred_positives > 0 else None
    rec = float(true_positives / pos_samples) if pos_samples > 0 else None
    
    if prec is not None and rec is not None and (prec + rec) > 0:
        f1 = float(2 * (prec * rec) / (prec + rec))
    else:
        f1 = 0.0 if (prec is not None and rec is not None) else None

    fpr = float(false_positives / neg_samples) if neg_samples > 0 else None
    fnr = float(false_negatives / pos_samples) if pos_samples > 0 else None
    sel_rate = float(pred_positives / total) if total > 0 else 0.0

    roc_auc = None
    if y_score is not None and pos_samples > 0 and neg_samples > 0:
        try:
            roc_auc = float(roc_auc_score(y_true_b, y_score))
        except Exception:
            roc_auc = None

    cm = ConfusionMatrix(
        true_positive=int(true_positives),
        false_positive=int(false_positives),
        true_negative=int(true_negatives),
        false_negative=int(false_negatives)
    )

    return PerformanceMetrics(
        accuracy=round(acc, 4),
        balanced_accuracy=round(bal_acc, 4),
        precision=round(prec, 4) if prec is not None else None,
        recall=round(rec, 4) if rec is not None else None,
        f1_score=round(f1, 4) if f1 is not None else None,
        false_positive_rate=round(fpr, 4) if fpr is not None else None,
        false_negative_rate=round(fnr, 4) if fnr is not None else None,
        selection_rate=round(sel_rate, 4),
        roc_auc=round(roc_auc, 4) if roc_auc is not None else None,
        confusion_matrix=cm,
        total_samples=total,
        positive_samples=pos_samples,
        negative_samples=neg_samples
    )

def compute_attribute_fairness(
    df: pd.DataFrame,
    attribute: str,
    y_true_col: str,
    y_pred_col: str,
    reference_group: Optional[str] = None,
    min_subgroup_size: int = 15
) -> AttributeFairnessResult:
    """Computes demographic parity, equal opportunity, equalized odds, and subgroup metrics."""
    series_attr = df[attribute].astype(str)
    y_true = df[y_true_col].values.astype(int)
    y_pred = df[y_pred_col].values.astype(int)
    total_samples = len(df)

    groups = sorted(series_attr.unique().tolist())
    subgroup_metrics: Dict[str, SubgroupMetrics] = {}
    warnings: List[str] = []

    # Choose default reference group (largest subgroup if unspecified or invalid)
    group_counts = series_attr.value_counts()
    if not reference_group or reference_group not in group_counts:
        reference_group = str(group_counts.index[0])

    # Per-group metric calculation
    for grp in groups:
        mask = (series_attr == grp).values
        n_grp = int(np.sum(mask))
        sample_share = float(n_grp / total_samples) if total_samples > 0 else 0.0

        if n_grp == 0:
            continue

        y_true_grp = y_true[mask]
        y_pred_grp = y_pred[mask]
        
        n_pos = int(np.sum(y_true_grp == 1))
        n_neg = int(np.sum(y_true_grp == 0))
        n_pred_pos = int(np.sum(y_pred_grp == 1))

        tp = np.sum((y_true_grp == 1) & (y_pred_grp == 1))
        fp = np.sum((y_true_grp == 0) & (y_pred_grp == 1))
        fn = np.sum((y_true_grp == 1) & (y_pred_grp == 0))
        
        acc = float(accuracy_score(y_true_grp, y_pred_grp))
        try:
            bal_acc = float(balanced_accuracy_score(y_true_grp, y_pred_grp))
        except Exception:
            bal_acc = acc

        base_rate = float(n_pos / n_grp)
        sel_rate = float(n_pred_pos / n_grp)
        prec = float(tp / n_pred_pos) if n_pred_pos > 0 else None
        rec = float(tp / n_pos) if n_pos > 0 else None
        fpr = float(fp / n_neg) if n_neg > 0 else None
        fnr = float(fn / n_pos) if n_pos > 0 else None

        f1 = None
        if prec is not None and rec is not None and (prec + rec) > 0:
            f1 = float(2 * (prec * rec) / (prec + rec))
        elif prec is not None and rec is not None:
            f1 = 0.0

        is_low_sample = n_grp < min_subgroup_size
        if is_low_sample:
            warnings.append(f"Subgroup '{grp}' has only {n_grp} samples (< {min_subgroup_size}). Metrics have high variance.")

        subgroup_metrics[grp] = SubgroupMetrics(
            subgroup=grp,
            sample_size=n_grp,
            sample_share=round(sample_share, 4),
            base_rate=round(base_rate, 4),
            selection_rate=round(sel_rate, 4),
            accuracy=round(acc, 4),
            balanced_accuracy=round(bal_acc, 4),
            precision=round(prec, 4) if prec is not None else None,
            recall_tpr=round(rec, 4) if rec is not None else None,
            false_positive_rate=round(fpr, 4) if fpr is not None else None,
            false_negative_rate=round(fnr, 4) if fnr is not None else None,
            f1_score=round(f1, 4) if f1 is not None else None,
            sample_size_warning=is_low_sample
        )

    # Reference subgroup disparities
    subgroup_disparities: Dict[str, SubgroupDisparity] = {}
    ref_metrics = subgroup_metrics.get(reference_group)

    for grp, m in subgroup_metrics.items():
        if not ref_metrics:
            subgroup_disparities[grp] = SubgroupDisparity(subgroup=grp)
            continue

        # Demographic parity difference: sel_rate(group) - sel_rate(ref)
        dp_diff = m.selection_rate - ref_metrics.selection_rate
        # Demographic parity ratio: min(sel_rate/ref_sel, ref_sel/sel_rate)
        if ref_metrics.selection_rate > 0 and m.selection_rate > 0:
            dp_ratio = min(m.selection_rate / ref_metrics.selection_rate, ref_metrics.selection_rate / m.selection_rate)
        elif ref_metrics.selection_rate == 0 and m.selection_rate == 0:
            # The relative selection-rate ratio is 0/0, hence undefined.
            dp_ratio = None
        else:
            dp_ratio = 0.0

        # Equal opportunity: TPR(group) - TPR(ref)
        eq_opp_diff = None
        if m.recall_tpr is not None and ref_metrics.recall_tpr is not None:
            eq_opp_diff = m.recall_tpr - ref_metrics.recall_tpr

        # FPR difference: FPR(group) - FPR(ref)
        fpr_diff = None
        if m.false_positive_rate is not None and ref_metrics.false_positive_rate is not None:
            fpr_diff = m.false_positive_rate - ref_metrics.false_positive_rate

        # Equalized odds diff: max(|TPR diff|, |FPR diff|)
        eq_odds_diff = None
        if eq_opp_diff is not None and fpr_diff is not None:
            eq_odds_diff = max(abs(eq_opp_diff), abs(fpr_diff))

        acc_diff = m.accuracy - ref_metrics.accuracy
        prec_diff = (m.precision - ref_metrics.precision) if (m.precision is not None and ref_metrics.precision is not None) else None

        subgroup_disparities[grp] = SubgroupDisparity(
            subgroup=grp,
            demographic_parity_diff=round(dp_diff, 4),
            demographic_parity_ratio=round(dp_ratio, 4) if dp_ratio is not None else None,
            equal_opportunity_diff=round(eq_opp_diff, 4) if eq_opp_diff is not None else None,
            fpr_diff=round(fpr_diff, 4) if fpr_diff is not None else None,
            equalized_odds_diff=round(eq_odds_diff, 4) if eq_odds_diff is not None else None,
            accuracy_diff=round(acc_diff, 4),
            precision_diff=round(prec_diff, 4) if prec_diff is not None else None
        )

    # Use Fairlearn MetricFrame / overall differences across all groups
    sel_rates = [m.selection_rate for m in subgroup_metrics.values()]
    overall_dp_diff = float(max(sel_rates) - min(sel_rates)) if sel_rates else 0.0
    
    # overall DP ratio: min / max
    max_sr = max(sel_rates) if sel_rates else 0.0
    min_sr = min(sel_rates) if sel_rates else 0.0
    overall_dp_ratio = round(float(min_sr / max_sr), 4) if max_sr > 0 else None
    if overall_dp_ratio is None:
        warnings.append("Demographic parity ratio is undefined: all audited groups have zero positive predictions.")

    # Overall Equal Opportunity difference (max TPR - min TPR)
    tprs = [m.recall_tpr for m in subgroup_metrics.values() if m.recall_tpr is not None]
    overall_eq_opp_diff = float(max(tprs) - min(tprs)) if len(tprs) > 1 else None

    # Overall Equalized Odds difference
    fprs = [m.false_positive_rate for m in subgroup_metrics.values() if m.false_positive_rate is not None]
    if overall_eq_opp_diff is not None and len(fprs) > 1:
        overall_fpr_diff = float(max(fprs) - min(fprs))
        overall_eq_odds_diff = float(max(overall_eq_opp_diff, overall_fpr_diff))
    else:
        overall_eq_odds_diff = None

    return AttributeFairnessResult(
        attribute=attribute,
        reference_group=reference_group,
        subgroups=groups,
        subgroup_metrics=subgroup_metrics,
        subgroup_disparities=subgroup_disparities,
        overall_demographic_parity_difference=round(overall_dp_diff, 4),
        overall_demographic_parity_ratio=overall_dp_ratio,
        overall_equal_opportunity_difference=round(overall_eq_opp_diff, 4) if overall_eq_opp_diff is not None else None,
        overall_equalized_odds_difference=round(overall_eq_odds_diff, 4) if overall_eq_odds_diff is not None else None,
        warnings=warnings
    )

def compute_intersectional_fairness(
    df: pd.DataFrame,
    attributes: List[str],
    y_true_col: str,
    y_pred_col: str,
    min_subgroup_size: int = 15
) -> IntersectionalAnalysisResult:
    """Computes combined subgroup analysis across combinations of attributes (e.g. Gender x Employment)."""
    if not attributes or len(attributes) < 2:
        return IntersectionalAnalysisResult(
            attributes=attributes,
            total_groups=0,
            cells=[],
            max_selection_rate_diff=0.0,
            max_accuracy_diff=0.0
        )

    # Combine attributes into composite series
    composite = df[attributes].astype(str).agg(" × ".join, axis=1)
    y_true = df[y_true_col].values.astype(int)
    y_pred = df[y_pred_col].values.astype(int)
    total_samples = len(df)

    unique_combos = sorted(composite.unique().tolist())
    cells: List[IntersectionalCell] = []
    
    sel_rates = []
    accuracies = []
    tprs = []

    for combo in unique_combos:
        mask = (composite == combo).values
        n_grp = int(np.sum(mask))
        if n_grp == 0:
            continue

        sample_share = float(n_grp / total_samples) if total_samples > 0 else 0.0
        y_true_grp = y_true[mask]
        y_pred_grp = y_pred[mask]

        n_pos = int(np.sum(y_true_grp == 1))
        n_neg = int(np.sum(y_true_grp == 0))
        n_pred_pos = int(np.sum(y_pred_grp == 1))

        tp = np.sum((y_true_grp == 1) & (y_pred_grp == 1))
        fp = np.sum((y_true_grp == 0) & (y_pred_grp == 1))
        
        acc = float(accuracy_score(y_true_grp, y_pred_grp))
        sel_rate = float(n_pred_pos / n_grp)
        prec = float(tp / n_pred_pos) if n_pred_pos > 0 else None
        rec = float(tp / n_pos) if n_pos > 0 else None
        fpr = float(fp / n_neg) if n_neg > 0 else None

        f1 = None
        if prec is not None and rec is not None and (prec + rec) > 0:
            f1 = float(2 * (prec * rec) / (prec + rec))

        sel_rates.append(sel_rate)
        accuracies.append(acc)
        if rec is not None:
            tprs.append(rec)

        # Build group_tuple dict
        parts = combo.split(" × ")
        group_dict = {attr: parts[i] if i < len(parts) else "" for i, attr in enumerate(attributes)}

        cells.append(IntersectionalCell(
            group_tuple=group_dict,
            group_label=combo,
            sample_size=n_grp,
            sample_share=round(sample_share, 4),
            selection_rate=round(sel_rate, 4),
            accuracy=round(acc, 4),
            tpr=round(rec, 4) if rec is not None else None,
            fpr=round(fpr, 4) if fpr is not None else None,
            f1_score=round(f1, 4) if f1 is not None else None,
            is_low_sample=(n_grp < min_subgroup_size)
        ))

    max_sel_diff = float(max(sel_rates) - min(sel_rates)) if sel_rates else 0.0
    max_acc_diff = float(max(accuracies) - min(accuracies)) if accuracies else 0.0
    max_tpr_diff = float(max(tprs) - min(tprs)) if len(tprs) > 1 else None

    return IntersectionalAnalysisResult(
        attributes=attributes,
        total_groups=len(cells),
        cells=cells,
        max_selection_rate_diff=round(max_sel_diff, 4),
        max_accuracy_diff=round(max_acc_diff, 4),
        max_tpr_diff=round(max_tpr_diff, 4) if max_tpr_diff is not None else None
    )
