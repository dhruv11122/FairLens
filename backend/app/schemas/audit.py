"""Pydantic schemas for FairLens: AI Model Bias & Fairness Auditor."""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field

# ==========================================
# Dataset & Model Upload Schemas
# ==========================================

class ColumnSummary(BaseModel):
    name: str
    dtype: str
    null_count: int
    unique_count: int
    sample_values: List[Any]
    is_numeric: bool

class DatasetPreviewResponse(BaseModel):
    dataset_id: str
    filename: str
    total_rows: int
    total_columns: int
    columns: List[ColumnSummary]
    head_rows: List[Dict[str, Any]]
    suggested_targets: List[str]
    suggested_sensitive: List[str]

class ModelInspectResponse(BaseModel):
    model_id: str
    filename: str
    model_type: str
    expected_features: Optional[List[str]] = None
    classes: Optional[List[Any]] = None
    has_predict_proba: bool = False
    pipeline_steps: Optional[List[str]] = None
    warning: Optional[str] = None

# ==========================================
# Audit Configuration Schemas
# ==========================================

class NumericBinConfig(BaseModel):
    column: str
    bins: int = 4
    strategy: str = "quantile"  # "quantile" or "uniform"
    labels: Optional[List[str]] = None

class AuditRequest(BaseModel):
    model_id: Optional[str] = None
    dataset_id: str
    target_column: str
    positive_class: Union[int, str, float]
    sensitive_attributes: List[str]
    feature_columns: Optional[List[str]] = None
    reference_groups: Optional[Dict[str, str]] = None  # sensitive_attr -> reference_subgroup
    binning_configs: Optional[List[NumericBinConfig]] = None
    disparity_threshold: float = 0.10  # 10% tolerance by default (e.g. 80% rule or 0.1 abs diff)
    min_subgroup_size: int = 15
    is_predictions_only: bool = False
    prediction_column: Optional[str] = None
    score_column: Optional[str] = None

# ==========================================
# Performance Metrics Schemas
# ==========================================

class ConfusionMatrix(BaseModel):
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

class PerformanceMetrics(BaseModel):
    accuracy: float
    balanced_accuracy: float
    precision: Optional[float] = None
    recall: Optional[float] = None  # True Positive Rate
    f1_score: Optional[float] = None
    false_positive_rate: Optional[float] = None
    false_negative_rate: Optional[float] = None
    selection_rate: float
    roc_auc: Optional[float] = None
    confusion_matrix: ConfusionMatrix
    total_samples: int
    positive_samples: int
    negative_samples: int

# ==========================================
# Fairness Subgroup & Disparity Schemas
# ==========================================

class SubgroupMetrics(BaseModel):
    subgroup: str
    sample_size: int
    sample_share: float
    base_rate: float  # Ground truth positive rate P(Y=1 | A=a)
    selection_rate: float  # P(Y_hat=1 | A=a)
    accuracy: float
    balanced_accuracy: float
    precision: Optional[float] = None
    recall_tpr: Optional[float] = None  # True Positive Rate
    false_positive_rate: Optional[float] = None
    false_negative_rate: Optional[float] = None
    f1_score: Optional[float] = None
    sample_size_warning: bool = False

class SubgroupDisparity(BaseModel):
    subgroup: str
    demographic_parity_diff: Optional[float] = None  # vs reference group
    demographic_parity_ratio: Optional[float] = None
    equal_opportunity_diff: Optional[float] = None  # TPR difference vs reference
    fpr_diff: Optional[float] = None
    equalized_odds_diff: Optional[float] = None  # max(|TPR diff|, |FPR diff|)
    accuracy_diff: Optional[float] = None
    precision_diff: Optional[float] = None

class AttributeFairnessResult(BaseModel):
    attribute: str
    reference_group: str
    subgroups: List[str]
    subgroup_metrics: Dict[str, SubgroupMetrics]
    subgroup_disparities: Dict[str, SubgroupDisparity]
    overall_demographic_parity_difference: float  # max - min selection rate
    overall_demographic_parity_ratio: Optional[float] = None
    overall_equal_opportunity_difference: Optional[float] = None  # max - min TPR
    overall_equalized_odds_difference: Optional[float] = None  # max(TPR diff, FPR diff)
    warnings: List[str] = []

# ==========================================
# Intersectional Fairness Schemas
# ==========================================

class IntersectionalCell(BaseModel):
    group_tuple: Dict[str, str]
    group_label: str
    sample_size: int
    sample_share: float
    selection_rate: float
    accuracy: float
    tpr: Optional[float] = None
    fpr: Optional[float] = None
    f1_score: Optional[float] = None
    is_low_sample: bool = False

class IntersectionalAnalysisResult(BaseModel):
    attributes: List[str]
    total_groups: int
    cells: List[IntersectionalCell]
    max_selection_rate_diff: float
    max_accuracy_diff: float
    max_tpr_diff: Optional[float] = None

# ==========================================
# Automated Audit Findings
# ==========================================

class AuditFinding(BaseModel):
    id: str
    severity: str  # "high", "medium", "low", "info"
    title: str
    metric: str
    attribute: str
    affected_group: str
    reference_group: Optional[str] = None
    disparity_value: float
    explanation: str
    evidence: str
    sample_size_info: str
    suggested_action: str

# ==========================================
# Explainable AI Schemas
# ==========================================

class FeatureImportanceItem(BaseModel):
    feature: str
    mean_abs_shap: float
    relative_importance: float

class SubgroupFeatureImportance(BaseModel):
    subgroup: str
    importance: List[FeatureImportanceItem]

class ProxyFeatureWarning(BaseModel):
    feature: str
    sensitive_attribute: str
    correlation_score: float
    explanation: str

class ExplainabilityResult(BaseModel):
    supported: bool
    reason: Optional[str] = None
    global_importance: List[FeatureImportanceItem] = []
    subgroup_importance: Dict[str, List[FeatureImportanceItem]] = {}
    proxy_warnings: List[ProxyFeatureWarning] = []

# ==========================================
# Threshold Simulator Schemas
# ==========================================

class ThresholdPoint(BaseModel):
    threshold: float
    accuracy: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    selection_rate: float
    demographic_parity_diff: Optional[float] = None
    equal_opportunity_diff: Optional[float] = None
    equalized_odds_diff: Optional[float] = None

class ThresholdSimulationRequest(BaseModel):
    attribute: str
    threshold: float

class ThresholdSimulationResponse(BaseModel):
    current: ThresholdPoint
    curve: List[ThresholdPoint]

# ==========================================
# Complete Audit Result Schema
# ==========================================

class AuditSummary(BaseModel):
    audit_id: str
    created_at: str
    model_id: Optional[str] = None
    model_name: Optional[str] = None
    dataset_id: str
    dataset_name: str
    target_column: str
    positive_class: str
    sensitive_attributes: List[str]
    overall_performance: PerformanceMetrics
    attribute_fairness: Dict[str, AttributeFairnessResult]
    intersectional_fairness: Optional[IntersectionalAnalysisResult] = None
    findings: List[AuditFinding]
    explainability: Optional[ExplainabilityResult] = None
    has_probabilities: bool = False
    disparity_threshold: float
    min_subgroup_size: int
    is_predictions_only: bool
