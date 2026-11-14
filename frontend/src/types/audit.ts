export interface ColumnSummary {
  name: string;
  dtype: string;
  null_count: number;
  unique_count: number;
  sample_values: any[];
  is_numeric: boolean;
}

export interface DatasetPreviewResponse {
  dataset_id: string;
  filename: string;
  total_rows: number;
  total_columns: number;
  columns: ColumnSummary[];
  head_rows: Record<string, any>[];
  suggested_targets: string[];
  suggested_sensitive: string[];
}

export interface ModelInspectResponse {
  model_id: string;
  filename: string;
  model_type: string;
  expected_features?: string[];
  classes?: any[];
  has_predict_proba: boolean;
  pipeline_steps?: string[];
  warning?: string;
}

export interface ConfusionMatrix {
  true_positive: number;
  false_positive: number;
  true_negative: number;
  false_negative: number;
}

export interface PerformanceMetrics {
  accuracy: number;
  balanced_accuracy: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  false_positive_rate?: number;
  false_negative_rate?: number;
  selection_rate: number;
  roc_auc?: number;
  confusion_matrix: ConfusionMatrix;
  total_samples: number;
  positive_samples: number;
  negative_samples: number;
}

export interface SubgroupMetrics {
  subgroup: string;
  sample_size: number;
  sample_share: number;
  base_rate: number;
  selection_rate: number;
  accuracy: number;
  balanced_accuracy: number;
  precision?: number;
  recall_tpr?: number;
  false_positive_rate?: number;
  false_negative_rate?: number;
  f1_score?: number;
  sample_size_warning: boolean;
}

export interface SubgroupDisparity {
  subgroup: string;
  demographic_parity_diff?: number;
  demographic_parity_ratio?: number;
  equal_opportunity_diff?: number;
  fpr_diff?: number;
  equalized_odds_diff?: number;
  accuracy_diff?: number;
  precision_diff?: number;
}

export interface AttributeFairnessResult {
  attribute: string;
  reference_group: string;
  subgroups: string[];
  subgroup_metrics: Record<string, SubgroupMetrics>;
  subgroup_disparities: Record<string, SubgroupDisparity>;
  overall_demographic_parity_difference: number;
  overall_demographic_parity_ratio?: number;
  overall_equal_opportunity_difference?: number;
  overall_equalized_odds_difference?: number;
  warnings: string[];
}

export interface IntersectionalCell {
  group_tuple: Record<string, string>;
  group_label: string;
  sample_size: number;
  sample_share: number;
  selection_rate: number;
  accuracy: number;
  tpr?: number;
  fpr?: number;
  f1_score?: number;
  is_low_sample: boolean;
}

export interface IntersectionalAnalysisResult {
  attributes: string[];
  total_groups: number;
  cells: IntersectionalCell[];
  max_selection_rate_diff: number;
  max_accuracy_diff: number;
  max_tpr_diff?: number;
}

export interface AuditFinding {
  id: string;
  severity: 'high' | 'medium' | 'low' | 'info';
  title: string;
  metric: string;
  attribute: string;
  affected_group: string;
  reference_group?: string;
  disparity_value: number;
  explanation: string;
  evidence: string;
  sample_size_info: string;
  suggested_action: string;
}

export interface FeatureImportanceItem {
  feature: string;
  mean_abs_shap: number;
  relative_importance: number;
}

export interface ProxyFeatureWarning {
  feature: string;
  sensitive_attribute: string;
  correlation_score: number;
  explanation: string;
}

export interface ExplainabilityResult {
  supported: boolean;
  reason?: string;
  global_importance: FeatureImportanceItem[];
  subgroup_importance: Record<string, FeatureImportanceItem[]>;
  proxy_warnings: ProxyFeatureWarning[];
}

export interface AuditSummary {
  audit_id: string;
  created_at: string;
  model_id?: string;
  model_name?: string;
  dataset_id: string;
  dataset_name: string;
  target_column: string;
  positive_class: string;
  sensitive_attributes: string[];
  overall_performance: PerformanceMetrics;
  attribute_fairness: Record<string, AttributeFairnessResult>;
  intersectional_fairness?: IntersectionalAnalysisResult;
  findings: AuditFinding[];
  explainability?: ExplainabilityResult;
  has_probabilities: boolean;
  disparity_threshold: number;
  min_subgroup_size: number;
  is_predictions_only: boolean;
}

export interface ThresholdPoint {
  threshold: number;
  accuracy: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  selection_rate: number;
  demographic_parity_diff?: number;
  equal_opportunity_diff?: number;
  equalized_odds_diff?: number;
}

export interface ThresholdSimulationResponse {
  current: ThresholdPoint;
  curve: ThresholdPoint[];
}
