"""Automated, deterministic rule-based fairness findings engine."""
from typing import List, Dict
import uuid
from app.schemas.audit import AuditFinding, AttributeFairnessResult, PerformanceMetrics

def generate_audit_findings(
    attribute_fairness: Dict[str, AttributeFairnessResult],
    overall_performance: PerformanceMetrics,
    disparity_threshold: float = 0.10,
    min_subgroup_size: int = 15
) -> List[AuditFinding]:
    """Generates structured, traceable fairness findings based on statistical disparities."""
    findings: List[AuditFinding] = []

    # Model utility must be considered even when every fairness gap happens to be zero.
    if overall_performance.positive_samples > 0 and overall_performance.recall is not None:
        if overall_performance.recall == 0:
            findings.append(AuditFinding(
                id=str(uuid.uuid4())[:8], severity="high",
                title="Model fails to detect any positive cases",
                metric="Positive Class Recall", attribute="Overall model",
                affected_group="Positive class", reference_group=None,
                disparity_value=1.0,
                explanation=(f"Recall is 0.0%: none of the {overall_performance.positive_samples} "
                             "actual positive cases were identified, regardless of overall accuracy."),
                evidence=(f"Accuracy {overall_performance.accuracy:.1%}; balanced accuracy "
                          f"{overall_performance.balanced_accuracy:.1%}; "
                          f"positive predictions {overall_performance.confusion_matrix.true_positive + overall_performance.confusion_matrix.false_positive}."),
                sample_size_info=f"N={overall_performance.total_samples}; actual positives={overall_performance.positive_samples}",
                suggested_action="Investigate class imbalance, decision threshold, calibration and false-negative costs before interpreting fairness gaps."
            ))
        elif overall_performance.recall < 0.20:
            findings.append(AuditFinding(
                id=str(uuid.uuid4())[:8], severity="medium",
                title="Very low positive-class recall", metric="Positive Class Recall",
                attribute="Overall model", affected_group="Positive class", reference_group=None,
                disparity_value=round(1 - overall_performance.recall, 4),
                explanation=f"Only {overall_performance.recall:.1%} of actual positive cases are detected.",
                evidence=f"Recall={overall_performance.recall:.3f}; balanced accuracy={overall_performance.balanced_accuracy:.3f}.",
                sample_size_info=f"N={overall_performance.total_samples}; actual positives={overall_performance.positive_samples}",
                suggested_action="Evaluate alternative thresholds and class-sensitive metrics on held-out data."
            ))

    for attr, res in attribute_fairness.items():
        ref_grp = res.reference_group
        ref_metrics = res.subgroup_metrics.get(ref_grp)
        
        # Accuracy gaps can exist even when parity/TPR gaps are zero.
        reliable_groups = [(g, m) for g, m in res.subgroup_metrics.items()
                           if m.sample_size >= min_subgroup_size]
        if len(reliable_groups) >= 2:
            low_name, low = min(reliable_groups, key=lambda pair: pair[1].accuracy)
            high_name, high = max(reliable_groups, key=lambda pair: pair[1].accuracy)
            accuracy_gap = high.accuracy - low.accuracy
            if accuracy_gap > disparity_threshold:
                findings.append(AuditFinding(
                    id=str(uuid.uuid4())[:8],
                    severity="high" if accuracy_gap > 0.20 else "medium",
                    title=f"Subgroup accuracy gap in '{attr}'",
                    metric="Accuracy Difference", attribute=attr,
                    affected_group=low_name, reference_group=high_name,
                    disparity_value=round(accuracy_gap, 4),
                    explanation=(f"Accuracy is {low.accuracy:.1%} for '{low_name}' versus "
                                 f"{high.accuracy:.1%} for '{high_name}', a {accuracy_gap:.1%} gap. "
                                 "Different class prevalences may contribute to the gap."),
                    evidence=f"Accuracy difference={accuracy_gap:.3f}; base rates {low.base_rate:.1%} vs {high.base_rate:.1%}.",
                    sample_size_info=f"'{low_name}' n={low.sample_size}; '{high_name}' n={high.sample_size}",
                    suggested_action="Compare subgroup base rates, confusion matrices, balanced accuracy and error rates; investigate data quality and representation."
                ))

        # 1. Demographic Parity Check (Overall difference)
        if res.overall_demographic_parity_difference > disparity_threshold:
            # find group with lowest and highest selection rate
            sorted_groups = sorted(res.subgroup_metrics.items(), key=lambda x: x[1].selection_rate)
            min_grp, min_m = sorted_groups[0]
            max_grp, max_m = sorted_groups[-1]
            diff = res.overall_demographic_parity_difference
            
            severity = "high" if diff > 0.20 else "medium"
            findings.append(AuditFinding(
                id=str(uuid.uuid4())[:8],
                severity=severity,
                title=f"Significant Demographic Parity Disparity in '{attr}'",
                metric="Demographic Parity Difference",
                attribute=attr,
                affected_group=min_grp,
                reference_group=max_grp,
                disparity_value=round(diff, 4),
                explanation=(
                    f"Selection rate for '{min_grp}' is {min_m.selection_rate * 100:.1f}%, whereas '{max_grp}' "
                    f"is selected at {max_m.selection_rate * 100:.1f}%. This represents a {diff * 100:.1f} percentage point difference."
                ),
                evidence=(
                    f"Overall DP Difference: {diff:.3f}. Four-fifths rule check: DP Ratio is "
                    f"{res.overall_demographic_parity_ratio:.3f} (threshold: 0.80)." if res.overall_demographic_parity_ratio is not None
                    else f"Overall DP Difference: {diff:.3f}; DP ratio undefined."
                ),
                sample_size_info=f"'{min_grp}' (n={min_m.sample_size}), '{max_grp}' (n={max_m.sample_size})",
                suggested_action=(
                    "Evaluate if ground-truth qualification rates differ legitimately or if features in the model "
                    "act as disparate impact proxies. Consider threshold calibration or reweighting techniques."
                )
            ))

        # 2. Equal Opportunity Check (True Positive Rate Disparities)
        if res.overall_equal_opportunity_difference is not None and res.overall_equal_opportunity_difference > disparity_threshold:
            tpr_groups = [
                (g, m) for g, m in res.subgroup_metrics.items() 
                if m.recall_tpr is not None and m.sample_size >= min_subgroup_size
            ]
            if len(tpr_groups) >= 2:
                tpr_sorted = sorted(tpr_groups, key=lambda x: x[1].recall_tpr)
                lowest_tpr_grp, low_m = tpr_sorted[0]
                highest_tpr_grp, high_m = tpr_sorted[-1]
                tpr_diff = high_m.recall_tpr - low_m.recall_tpr

                if tpr_diff > disparity_threshold:
                    findings.append(AuditFinding(
                        id=str(uuid.uuid4())[:8],
                        severity="high" if tpr_diff > 0.15 else "medium",
                        title=f"Equal Opportunity Gap in '{attr}'",
                        metric="Equal Opportunity (TPR)",
                        attribute=attr,
                        affected_group=lowest_tpr_grp,
                        reference_group=highest_tpr_grp,
                        disparity_value=round(tpr_diff, 4),
                        explanation=(
                            f"Actual positive cases in group '{lowest_tpr_grp}' have a True Positive Rate of "
                            f"{low_m.recall_tpr * 100:.1f}%, versus {high_m.recall_tpr * 100:.1f}% "
                            f"for '{highest_tpr_grp}'. Interpret the effect according to the meaning of the positive class."
                        ),
                        evidence=(
                            f"Equal Opportunity Difference: {tpr_diff:.3f}. False Negative Rate is "
                            f"{low_m.false_negative_rate * 100:.1f}% vs {high_m.false_negative_rate * 100:.1f}%."
                        ),
                        sample_size_info=f"'{lowest_tpr_grp}' (n={low_m.sample_size}), '{highest_tpr_grp}' (n={high_m.sample_size})",
                        suggested_action=(
                            "Investigate features driving false negatives in the disadvantaged group. "
                            "Assess equalized odds post-processing or error-rate balancing."
                        )
                    ))

        # 3. False Positive Rate Disparity (Predictive Harm / Over-policing / Risk Assignment)
        fpr_groups = [
            (g, m) for g, m in res.subgroup_metrics.items() 
            if m.false_positive_rate is not None and m.sample_size >= min_subgroup_size
        ]
        if len(fpr_groups) >= 2:
            fpr_sorted = sorted(fpr_groups, key=lambda x: x[1].false_positive_rate)
            min_fpr_grp, min_fpr_m = fpr_sorted[0]
            max_fpr_grp, max_fpr_m = fpr_sorted[-1]
            fpr_diff = max_fpr_m.false_positive_rate - min_fpr_m.false_positive_rate

            if fpr_diff > disparity_threshold:
                findings.append(AuditFinding(
                    id=str(uuid.uuid4())[:8],
                    severity="medium",
                    title=f"Asymmetric False Positive Burden in '{attr}'",
                    metric="False Positive Rate Difference",
                    attribute=attr,
                    affected_group=max_fpr_grp,
                    reference_group=min_fpr_grp,
                    disparity_value=round(fpr_diff, 4),
                    explanation=(
                        f"Group '{max_fpr_grp}' exhibits a False Positive Rate of {max_fpr_m.false_positive_rate * 100:.1f}%, "
                        f"whereas '{min_fpr_grp}' has {min_fpr_m.false_positive_rate * 100:.1f}% (gap of {fpr_diff * 100:.1f}%)."
                    ),
                    evidence=f"FPR Gap: {fpr_diff:.3f}. In punitive decision settings, higher FPR causes disproportionate harm.",
                    sample_size_info=f"'{max_fpr_grp}' (n={max_fpr_m.sample_size}), '{min_fpr_grp}' (n={min_fpr_m.sample_size})",
                    suggested_action="Review decision thresholds and decision cost matrices for affected groups."
                ))

        # 4. Small sample size warnings as informative audit observations
        for grp, m in res.subgroup_metrics.items():
            if m.sample_size < min_subgroup_size:
                findings.append(AuditFinding(
                    id=str(uuid.uuid4())[:8],
                    severity="low",
                    title=f"Low Statistical Power for '{grp}' in '{attr}'",
                    metric="Sample Size Adequacy",
                    attribute=attr,
                    affected_group=grp,
                    reference_group=ref_grp,
                    disparity_value=0.0,
                    explanation=(
                        f"Subgroup '{grp}' has only {m.sample_size} samples. Metrics calculated on this subgroup "
                        "have wide confidence intervals and cannot reliably support definitive conclusions."
                    ),
                    evidence=f"Sample size {m.sample_size} is below recommended minimum of {min_subgroup_size}.",
                    sample_size_info=f"Subgroup n={m.sample_size} ({m.sample_share * 100:.1f}% of dataset)",
                    suggested_action="Collect additional evaluation samples or stratify test sets to ensure statistical power."
                ))

    # Sort findings by severity (high -> medium -> low -> info)
    severity_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    findings.sort(key=lambda x: (severity_order.get(x.severity, 4), -x.disparity_value))

    return findings
