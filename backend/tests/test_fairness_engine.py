"""Comprehensive Unit & Edge Case Tests for FairLens."""
import pytest
import numpy as np
import pandas as pd
from app.ml.fairness_engine import (
    compute_binary_performance, compute_attribute_fairness, compute_intersectional_fairness
)
from app.ml.threshold_simulator import simulate_threshold, generate_threshold_curve
from app.ml.findings_engine import generate_audit_findings
from app.ml.data_processor import apply_numeric_binning, inspect_dataset
from app.ml.explainability import compute_explainability, compute_proxy_features
from app.schemas.audit import NumericBinConfig
from fairlearn.metrics import (
    demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference, selection_rate, true_positive_rate
)

# ==============================================================================
# 1. Performance Metrics & Mathematical Accuracy
# ==============================================================================

def test_binary_performance_standard():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    y_scores = np.array([0.9, 0.4, 0.2, 0.1, 0.8, 0.6, 0.7, 0.3])

    metrics = compute_binary_performance(y_true, y_pred, y_scores)
    assert metrics.accuracy == 0.75
    assert metrics.confusion_matrix.true_positive == 3
    assert metrics.confusion_matrix.false_negative == 1
    assert metrics.confusion_matrix.false_positive == 1
    assert metrics.confusion_matrix.true_negative == 3
    assert metrics.precision == 0.75
    assert metrics.recall == 0.75
    assert metrics.roc_auc is not None and metrics.roc_auc > 0.8

def test_edge_case_zero_positives():
    """All negative labels in dataset: recall should be None, no divide by zero crash."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 1, 0, 0])
    
    metrics = compute_binary_performance(y_true, y_pred)
    assert metrics.recall is None
    assert metrics.false_positive_rate == 0.25
    assert metrics.accuracy == 0.75

def test_edge_case_zero_negatives():
    """All positive labels in dataset: FPR should be None."""
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 0, 1])
    
    metrics = compute_binary_performance(y_true, y_pred)
    assert metrics.false_positive_rate is None
    assert metrics.recall == 0.75
    assert metrics.accuracy == 0.75

def test_edge_case_zero_predicted_positives():
    """Classifier predicts negative for all samples: precision should be None."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0])
    
    metrics = compute_binary_performance(y_true, y_pred)
    assert metrics.precision is None
    assert metrics.recall == 0.0
    assert metrics.selection_rate == 0.0
    assert metrics.accuracy == 0.5

# ==============================================================================
# 2. Fairness Metrics Engine vs Fairlearn
# ==============================================================================

def test_demographic_parity_and_equal_opportunity_vs_fairlearn():
    """Verify Demographic Parity and Equal Opportunity match Fairlearn outputs identically."""
    np.random.seed(123)
    n = 400
    sensitive = np.random.choice(["Group_X", "Group_Y"], size=n, p=[0.45, 0.55])
    y_true = np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    y_pred = np.where(sensitive == "Group_X", 
                      np.random.choice([0, 1], size=n, p=[0.3, 0.7]), 
                      np.random.choice([0, 1], size=n, p=[0.8, 0.2]))

    df = pd.DataFrame({"sensitive": sensitive, "y_true": y_true, "y_pred": y_pred})

    res = compute_attribute_fairness(
        df=df,
        attribute="sensitive",
        y_true_col="y_true",
        y_pred_col="y_pred",
        reference_group="Group_X",
        min_subgroup_size=15
    )

    # Fairlearn calculations
    expected_dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
    expected_dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive)
    expected_eq_odds = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)

    assert np.isclose(res.overall_demographic_parity_difference, expected_dp_diff, atol=1e-3)
    assert np.isclose(res.overall_demographic_parity_ratio, expected_dp_ratio, atol=1e-3)
    assert np.isclose(res.overall_equalized_odds_difference, expected_eq_odds, atol=1e-3)

def test_fairness_single_group_dataset():
    """Dataset with only one subgroup should evaluate gracefully without throwing errors."""
    df = pd.DataFrame({
        "gender": ["A", "A", "A", "A"],
        "y_true": [1, 0, 1, 0],
        "y_pred": [1, 1, 0, 0]
    })
    res = compute_attribute_fairness(df, "gender", "y_true", "y_pred", min_subgroup_size=2)
    assert len(res.subgroups) == 1
    assert res.overall_demographic_parity_difference == 0.0
    assert res.overall_demographic_parity_ratio == 1.0

def test_fairness_subgroup_small_sample_warning():
    """Subgroups with sample sizes below threshold should trigger sample size warnings."""
    df = pd.DataFrame({
        "age_tier": ["Senior"] * 5 + ["Adult"] * 50,
        "y_true": [1] * 5 + [1] * 25 + [0] * 25,
        "y_pred": [1] * 5 + [1] * 20 + [0] * 30
    })
    res = compute_attribute_fairness(df, "age_tier", "y_true", "y_pred", min_subgroup_size=15)
    assert res.subgroup_metrics["Senior"].sample_size_warning is True
    assert res.subgroup_metrics["Adult"].sample_size_warning is False
    assert any("Senior" in w for w in res.warnings)

# ==============================================================================
# 3. Intersectional Fairness Analysis
# ==============================================================================

def test_intersectional_fairness_quadrant():
    df = pd.DataFrame({
        "gender": ["M", "M", "F", "F"],
        "age": ["Young", "Old", "Young", "Old"],
        "y_true": [1, 0, 1, 0],
        "y_pred": [1, 0, 0, 0]
    })
    
    res = compute_intersectional_fairness(df, ["gender", "age"], "y_true", "y_pred", min_subgroup_size=1)
    assert res.total_groups == 4
    assert res.max_selection_rate_diff == 1.0
    assert len(res.cells) == 4

def test_intersectional_empty_or_single_attribute():
    df = pd.DataFrame({"gender": ["M", "F"], "y_true": [1, 0], "y_pred": [1, 0]})
    res = compute_intersectional_fairness(df, ["gender"], "y_true", "y_pred")
    assert res.total_groups == 0
    assert len(res.cells) == 0

# ==============================================================================
# 4. Threshold Tradeoff Simulator
# ==============================================================================

def test_threshold_simulation_monotonic_selection_rate():
    """As decision cutoff increases, selection rate must monotonically decrease or stay constant."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1])
    y_scores = np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05])
    sensitive = pd.Series(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])

    sim_res = generate_threshold_curve(y_true, y_scores, sensitive, current_threshold=0.5, steps=10)
    selection_rates = [pt.selection_rate for pt in sim_res.curve]
    # Check monotonic non-increasing
    assert all(x >= y for x, y in zip(selection_rates, selection_rates[1:]))

# ==============================================================================
# 5. Data Processor & Binning
# ==============================================================================

def test_numeric_binning_quantile_and_uniform():
    df = pd.DataFrame({"income": [10000, 20000, 30000, 40000, 50000, 60000]})
    config_q = NumericBinConfig(column="income", bins=3, strategy="quantile")
    config_u = NumericBinConfig(column="income", bins=3, strategy="uniform")
    
    df_q = apply_numeric_binning(df, [config_q])
    assert df_q["income"].nunique() <= 3
    
    df_u = apply_numeric_binning(df, [config_u])
    assert df_u["income"].nunique() <= 3

def test_inspect_dataset_metadata():
    df = pd.DataFrame({
        "age": [25, 30, 45, 60],
        "gender": ["F", "M", "F", "M"],
        "loan_approved": [1, 0, 1, 1]
    })
    preview = inspect_dataset(df, "ds1", "test.csv")
    assert preview.total_rows == 4
    assert preview.total_columns == 3
    assert "loan_approved" in preview.suggested_targets
    assert "gender" in preview.suggested_sensitive

# ==============================================================================
# 6. Proxy Correlation Engine
# ==============================================================================

def test_proxy_feature_detection():
    # Construct feature that perfectly correlates with sensitive attribute
    sensitive = ["A", "A", "A", "B", "B", "B"]
    proxy_feature = [10, 10, 11, 80, 85, 82]
    independent_feature = [5, 2, 8, 3, 7, 4]
    
    df = pd.DataFrame({
        "sens": sensitive,
        "proxy": proxy_feature,
        "clean": independent_feature
    })
    
    warnings = compute_proxy_features(df, ["proxy", "clean"], ["sens"], correlation_threshold=0.5)
    assert len(warnings) == 1
    assert warnings[0].feature == "proxy"
    assert warnings[0].sensitive_attribute == "sens"
    assert warnings[0].correlation_score > 0.8

# Regression tests added after reviewing the external-model attrition audit.

def test_all_zero_selection_dp_ratio_is_undefined():
    df = pd.DataFrame({
        "department": ["A"] * 20 + ["B"] * 20,
        "actual": [0, 1] * 20,
        "predicted": [0] * 40,
    })
    result = compute_attribute_fairness(df, "department", "actual", "predicted")
    assert result.overall_demographic_parity_difference == 0.0
    assert result.overall_demographic_parity_ratio is None
    assert result.subgroup_disparities["A"].demographic_parity_ratio is None
    assert result.subgroup_disparities["B"].demographic_parity_ratio is None


def test_all_negative_prediction_warns_even_with_zero_dp_gap():
    df = pd.DataFrame({
        "department": ["A"] * 20 + ["B"] * 20,
        "actual": [1] * 10 + [0] * 10 + [1] * 5 + [0] * 15,
        "predicted": [0] * 40,
    })
    overall = compute_binary_performance(df.actual.to_numpy(), df.predicted.to_numpy())
    fairness = compute_attribute_fairness(df, "department", "actual", "predicted")
    findings = generate_audit_findings({"department": fairness}, overall, disparity_threshold=0.10)
    assert any(f.metric == "Positive Class Recall" and f.severity == "high" for f in findings)
    assert any(f.metric == "Accuracy Difference" for f in findings)


def test_no_accuracy_disparity_alert_for_tiny_group():
    df = pd.DataFrame({
        "department": ["A"] * 30 + ["B"] * 4,
        "actual": [0] * 30 + [1] * 4,
        "predicted": [0] * 34,
    })
    overall = compute_binary_performance(df.actual.to_numpy(), df.predicted.to_numpy())
    fairness = compute_attribute_fairness(df, "department", "actual", "predicted")
    findings = generate_audit_findings({"department": fairness}, overall, disparity_threshold=0.10)
    assert not any(f.metric == "Accuracy Difference" for f in findings)
    assert any(f.metric == "Sample Size Adequacy" for f in findings)
