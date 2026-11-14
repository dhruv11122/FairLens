import { 
  DatasetPreviewResponse, ModelInspectResponse, 
  AuditSummary, ThresholdSimulationResponse 
} from '../types/audit';

const API_BASE = window.location.port === '3000' && window.location.hostname === '127.0.0.1' 
  ? 'http://127.0.0.1:8000/api' 
  : '/api';

export const api = {
  uploadDataset: async (file: File): Promise<DatasetPreviewResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/datasets/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to upload dataset');
    }
    return res.json();
  },

  uploadModel: async (file: File): Promise<ModelInspectResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/models/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to upload model artifact');
    }
    return res.json();
  },

  runAudit: async (config: {
    model_id?: string;
    dataset_id: string;
    target_column: string;
    positive_class: string | number;
    sensitive_attributes: string[];
    feature_columns?: string[];
    reference_groups?: Record<string, string>;
    binning_configs?: any[];
    disparity_threshold?: number;
    min_subgroup_size?: number;
    is_predictions_only?: boolean;
    prediction_column?: string;
    score_column?: string;
  }): Promise<AuditSummary> => {
    const res = await fetch(`${API_BASE}/audits/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Audit execution failed');
    }
    return res.json();
  },

  getAudit: async (auditId: string): Promise<AuditSummary> => {
    const res = await fetch(`${API_BASE}/audits/${auditId}`);
    if (!res.ok) throw new Error('Audit not found');
    return res.json();
  },

  simulateThreshold: async (auditId: string, attribute: string, threshold: number): Promise<ThresholdSimulationResponse> => {
    const res = await fetch(`${API_BASE}/audits/${auditId}/threshold-analysis`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ attribute, threshold }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Threshold simulation failed');
    }
    return res.json();
  },

  getReportPdfUrl: (auditId: string) => `${API_BASE}/audits/${auditId}/report`,
  getExportJsonUrl: (auditId: string) => `${API_BASE}/audits/${auditId}/export-json`
};
