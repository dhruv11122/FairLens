import React, { useState } from 'react';
import { 
  UploadCloud, CheckCircle2, AlertCircle, FileSpreadsheet, 
  Cpu, Play, Sparkles, SlidersHorizontal
} from 'lucide-react';
import { api } from '../services/api';
import { DatasetPreviewResponse, ModelInspectResponse, AuditSummary } from '../types/audit';

interface NewAuditProps {
  onAuditComplete: (audit: AuditSummary) => void;
}

export const NewAudit: React.FC<NewAuditProps> = ({ onAuditComplete }) => {
  const [mode, setMode] = useState<'model_dataset' | 'predictions_only'>('model_dataset');
  
  // Dataset state
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [datasetPreview, setDatasetPreview] = useState<DatasetPreviewResponse | null>(null);
  const [uploadingDataset, setUploadingDataset] = useState(false);

  // Model state
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [modelInspect, setModelInspect] = useState<ModelInspectResponse | null>(null);
  const [uploadingModel, setUploadingModel] = useState(false);

  // Config selections
  const [targetColumn, setTargetColumn] = useState<string>('');
  const [positiveClass, setPositiveClass] = useState<string>('1');
  const [sensitiveAttributes, setSensitiveAttributes] = useState<string[]>([]);
  const [disparityThreshold, setDisparityThreshold] = useState<number>(0.10);
  const [minSubgroupSize, setMinSubgroupSize] = useState<number>(15);
  
  // Prediction only fields
  const [predictionCol, setPredictionCol] = useState<string>('');
  const [scoreCol, setScoreCol] = useState<string>('');

  // Never submit hidden, stale or reserved columns as sensitive attributes.
  const reservedColumns = [targetColumn, ...(mode === 'predictions_only' ? [predictionCol, scoreCol] : [])].filter(Boolean);
  const validSensitiveAttributes = sensitiveAttributes.filter(
    name => datasetPreview?.columns.some(col => col.name === name) && !reservedColumns.includes(name)
  );

  const [loadingAudit, setLoadingAudit] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Quick Demo Loader
  const loadDemoScenario = async (scenario: 'loan' | 'hiring') => {
    setErrorMsg(null);
    setLoadingAudit(true);
    try {
      let csvUrl = '';
      let modelUrl = '';
      let target = '';
      let posClass = '1';
      let sensitive: string[] = [];

      if (scenario === 'loan') {
        csvUrl = '/data/demos/loan_evaluation_dataset.csv';
        modelUrl = '/models/demos/loan_approval_model.joblib';
        target = 'loan_approved';
        posClass = '1';
        sensitive = ['gender', 'employment_type', 'age_group'];
      } else {
        csvUrl = '/data/demos/hiring_evaluation_dataset.csv';
        modelUrl = '/models/demos/hiring_screening_model.joblib';
        target = 'shortlisted';
        posClass = '1';
        sensitive = ['gender', 'ethnicity', 'disability_status'];
      }

      // Fetch files from demo path
      const csvBlob = await (await fetch(csvUrl)).blob();
      const csvFile = new File([csvBlob], scenario === 'loan' ? 'loan_evaluation_dataset.csv' : 'hiring_evaluation_dataset.csv');
      const dsPreview = await api.uploadDataset(csvFile);
      setDatasetPreview(dsPreview);

      const modelBlob = await (await fetch(modelUrl)).blob();
      const mFile = new File([modelBlob], scenario === 'loan' ? 'loan_approval_model.joblib' : 'hiring_screening_model.joblib');
      const mInspect = await api.uploadModel(mFile);
      setModelInspect(mInspect);

      // Run audit directly
      const auditRes = await api.runAudit({
        model_id: mInspect.model_id,
        dataset_id: dsPreview.dataset_id,
        target_column: target,
        positive_class: posClass,
        sensitive_attributes: sensitive,
        disparity_threshold: 0.10,
        min_subgroup_size: 15,
        is_predictions_only: false,
      });

      onAuditComplete(auditRes);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to initialize demo scenario.');
    } finally {
      setLoadingAudit(false);
    }
  };

  const handleDatasetUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.[0]) return;
    const file = e.target.files[0];
    setDatasetFile(file);
    setUploadingDataset(true);
    setErrorMsg(null);
    try {
      const preview = await api.uploadDataset(file);
      setDatasetPreview(preview);
      const suggestedTarget = preview.suggested_targets[0] || '';
      setTargetColumn(suggestedTarget);
      // Reset previous dataset mappings. Never preselect a target as an audit group.
      setPredictionCol('');
      setScoreCol('');
      setSensitiveAttributes(
        preview.suggested_sensitive.filter(name => name !== suggestedTarget).slice(0, 2)
      );
    } catch (err: any) {
      setErrorMsg(err.message || 'Dataset upload failed');
    } finally {
      setUploadingDataset(false);
    }
  };

  const handleModelUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.[0]) return;
    const file = e.target.files[0];
    setModelFile(file);
    setUploadingModel(true);
    setErrorMsg(null);
    try {
      const inspect = await api.uploadModel(file);
      setModelInspect(inspect);
    } catch (err: any) {
      setErrorMsg(err.message || 'Model upload failed');
    } finally {
      setUploadingModel(false);
    }
  };

  const handleRunAudit = async () => {
    if (!datasetPreview) {
      setErrorMsg('Please upload an evaluation dataset.');
      return;
    }
    if (mode === 'model_dataset' && !modelInspect) {
      setErrorMsg('Please upload a trained scikit-learn model artifact.');
      return;
    }
    if (!targetColumn) {
      setErrorMsg('Please select the ground truth target column.');
      return;
    }
    if (validSensitiveAttributes.length === 0) {
      setErrorMsg('Please select at least one sensitive / protected attribute.');
      return;
    }
    if (mode === 'predictions_only' && !predictionCol) {
      setErrorMsg('Please specify the predictions column for predictions-only mode.');
      return;
    }

    setLoadingAudit(true);
    setErrorMsg(null);
    try {
      const auditRes = await api.runAudit({
        model_id: mode === 'model_dataset' ? modelInspect?.model_id : undefined,
        dataset_id: datasetPreview.dataset_id,
        target_column: targetColumn,
        positive_class: positiveClass,
        sensitive_attributes: validSensitiveAttributes,
        disparity_threshold: disparityThreshold,
        min_subgroup_size: minSubgroupSize,
        is_predictions_only: mode === 'predictions_only',
        prediction_column: mode === 'predictions_only' ? predictionCol : undefined,
        score_column: mode === 'predictions_only' && scoreCol ? scoreCol : undefined,
      });

      onAuditComplete(auditRes);
    } catch (err: any) {
      setErrorMsg(err.message || 'Audit execution encountered an error.');
    } finally {
      setLoadingAudit(false);
    }
  };

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
          Configure Fairness Audit
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Upload your evaluation dataset and trained classifier, or select a pre-built demo scenario to get started instantly.
        </p>
      </div>

      {/* Demo Scenario Quick-Start Banner */}
      <div className="bg-gradient-to-r from-brand-900/40 via-surface-900 to-indigo-950/40 border border-brand-500/30 rounded-2xl p-6 shadow-xl relative overflow-hidden">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-brand-400" />
              <span className="font-semibold text-white">Instant Demonstration Scenarios</span>
              <span className="text-xs bg-brand-500/20 text-brand-300 font-medium px-2 py-0.5 rounded-full border border-brand-500/30">
                1-Click Ready
              </span>
            </div>
            <p className="text-sm text-slate-300 max-w-2xl">
              Don't have a model right now? Load authentic pre-trained benchmark classifiers and datasets to explore full audit capabilities.
            </p>
          </div>
          <div className="flex items-center gap-3 shrink-0">
            <button
              onClick={() => loadDemoScenario('loan')}
              disabled={loadingAudit}
              className="px-4 py-2 bg-brand-600 hover:bg-brand-500 disabled:opacity-50 text-white text-xs font-semibold rounded-lg shadow-md transition-all flex items-center gap-2"
            >
              <Cpu className="w-4 h-4" />
              Demo 1: Loan Underwriting
            </button>
            <button
              onClick={() => loadDemoScenario('hiring')}
              disabled={loadingAudit}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 disabled:opacity-50 text-slate-200 text-xs font-semibold rounded-lg shadow-sm transition-all flex items-center gap-2"
            >
              <Cpu className="w-4 h-4" />
              Demo 2: Candidate Screening
            </button>
          </div>
        </div>
      </div>

      {errorMsg && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl flex items-start gap-3 text-rose-300 text-sm">
          <AlertCircle className="w-5 h-5 shrink-0 mt-0.5 text-rose-400" />
          <div>
            <div className="font-semibold">Audit Configuration Notice</div>
            <div>{errorMsg}</div>
          </div>
        </div>
      )}

      {/* Mode Selector */}
      <div className="flex gap-4 border-b border-slate-800 pb-4">
        <button
          onClick={() => setMode('model_dataset')}
          className={`flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-lg transition-all ${
            mode === 'model_dataset'
              ? 'bg-slate-800 text-white border border-slate-700 shadow-sm'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Cpu className="w-4 h-4 text-brand-400" />
          Full Pipeline: Model + Dataset
        </button>
        <button
          onClick={() => setMode('predictions_only')}
          className={`flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-lg transition-all ${
            mode === 'predictions_only'
              ? 'bg-slate-800 text-white border border-slate-700 shadow-sm'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <FileSpreadsheet className="w-4 h-4 text-emerald-400" />
          Predictions-Only Mode (Model Agnostic)
        </button>
      </div>

      {/* Upload Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Dataset Upload Card */}
        <div className="bg-surface-900 border border-slate-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <FileSpreadsheet className="w-5 h-5 text-brand-400" />
              1. Evaluation Dataset (.csv)
            </h3>
            {datasetPreview && (
              <span className="text-xs bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded-full flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" /> {datasetPreview.total_rows} rows
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400">
            CSV containing ground truth labels, sensitive demographic features, and predictor inputs.
          </p>

          <label className="border-2 border-dashed border-slate-700 hover:border-brand-500 rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer transition-all bg-surface-950/40 group">
            <UploadCloud className="w-8 h-8 text-slate-500 group-hover:text-brand-400 mb-2 transition-all" />
            <span className="text-sm font-medium text-slate-300 group-hover:text-white">
              {datasetFile ? datasetFile.name : 'Click to select CSV'}
            </span>
            <span className="text-xs text-slate-400 mt-1">Comma-separated values (.csv)</span>
            <input type="file" accept=".csv" onChange={handleDatasetUpload} className="hidden" />
          </label>
        </div>

        {/* Model Upload Card (Only in full model mode) */}
        {mode === 'model_dataset' ? (
          <div className="bg-surface-900 border border-slate-800 rounded-2xl p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-white flex items-center gap-2">
                <Cpu className="w-5 h-5 text-indigo-400" />
                2. Trained Classifier Artifact
              </h3>
              {modelInspect && (
                <span className="text-xs bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 px-2 py-0.5 rounded-full flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" /> {modelInspect.model_type}
                </span>
              )}
            </div>
            <p className="text-xs text-slate-400">
              Locally trusted scikit-learn classifier or pipeline saved via Joblib (.joblib, .pkl).
            </p>

            <label className="border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer transition-all bg-surface-950/40 group">
              <Cpu className="w-8 h-8 text-slate-500 group-hover:text-indigo-400 mb-2 transition-all" />
              <span className="text-sm font-medium text-slate-300 group-hover:text-white">
                {modelFile ? modelFile.name : 'Click to select Model Artifact'}
              </span>
              <span className="text-xs text-slate-400 mt-1">.joblib or .pkl artifacts</span>
              <input type="file" accept=".joblib,.pkl,.pickle" onChange={handleModelUpload} className="hidden" />
            </label>
          </div>
        ) : (
          /* Predictions-Only Mapping Card */
          <div className="bg-surface-900 border border-slate-800 rounded-2xl p-6 space-y-4">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <FileSpreadsheet className="w-5 h-5 text-emerald-400" />
              2. Predictions Column Mapping
            </h3>
            <p className="text-xs text-slate-400">
              Provide column names inside the CSV for pre-computed model outputs.
            </p>

            <div className="space-y-3">
              <div>
                <label className="text-xs font-medium text-slate-300 block mb-1">
                  Predicted Class Column *
                </label>
                <select
                  value={predictionCol}
                  onChange={(e) => {
                    setPredictionCol(e.target.value);
                    setSensitiveAttributes(prev => prev.filter(name => name !== e.target.value));
                  }}
                  className="w-full bg-surface-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-brand-500"
                >
                  <option value="">-- Select predicted label column --</option>
                  {datasetPreview?.columns.map((c) => (
                    <option key={c.name} value={c.name}>
                      {c.name} ({c.dtype})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-xs font-medium text-slate-300 block mb-1">
                  Prediction Probability / Score Column (Optional)
                </label>
                <select
                  value={scoreCol}
                  onChange={(e) => {
                    setScoreCol(e.target.value);
                    setSensitiveAttributes(prev => prev.filter(name => name !== e.target.value));
                  }}
                  className="w-full bg-surface-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-brand-500"
                >
                  <option value="">-- Optional: Continuous probability [0, 1] --</option>
                  {datasetPreview?.columns.filter(c => c.is_numeric).map((c) => (
                    <option key={c.name} value={c.name}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Dataset Schema Configuration Section */}
      {datasetPreview && (
        <div className="bg-surface-900 border border-slate-800 rounded-2xl p-6 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <SlidersHorizontal className="w-5 h-5 text-brand-400" />
              3. Audit Mapping & Hyperparameters
            </h3>
            <span className="text-xs text-slate-400">
              {datasetPreview.total_columns} Detected Attributes
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Target Column */}
            <div>
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-1.5">
                Target Label Column (Ground Truth) *
              </label>
              <select
                value={targetColumn}
                onChange={(e) => {
                  setTargetColumn(e.target.value);
                  setSensitiveAttributes(prev => prev.filter(name => name !== e.target.value));
                }}
                className="w-full bg-surface-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-brand-500"
              >
                <option value="">-- Select target column --</option>
                {datasetPreview.columns.map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Positive Class Value */}
            <div>
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-1.5">
                Positive Class (Class of Interest) *
              </label>
              <input
                type="text"
                value={positiveClass}
                onChange={(e) => setPositiveClass(e.target.value)}
                placeholder="e.g. 1, Approved, Yes"
                className="w-full bg-surface-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-brand-500"
              />
              <span className="text-[11px] text-slate-400">
                The class treated as positive for evaluation (e.g. fraud, attrition, approved).
              </span>
            </div>

            {/* Disparity Tolerance */}
            <div>
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-1.5">
                Disparity Alert Threshold: {(disparityThreshold * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.02"
                max="0.30"
                step="0.01"
                value={disparityThreshold}
                onChange={(e) => setDisparityThreshold(parseFloat(e.target.value))}
                className="w-full accent-brand-500 cursor-pointer mt-2"
              />
              <span className="text-[11px] text-slate-400">
                Flag findings when disparity exceeds { (disparityThreshold * 100).toFixed(0) }% points.
              </span>
            </div>
          </div>

          {/* Protected / Sensitive Attributes Multi-Select */}
          <div>
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-2">
              Protected / Sensitive Attributes to Audit *
            </label>
            <div className="flex flex-wrap gap-2">
              {datasetPreview.columns.map((c) => {
                if (c.name === targetColumn || c.name === predictionCol || c.name === scoreCol) return null;
                const isSelected = validSensitiveAttributes.includes(c.name);
                return (
                  <button
                    key={c.name}
                    type="button"
                    onClick={() => {
                      if (isSelected) {
                        setSensitiveAttributes(sensitiveAttributes.filter(a => a !== c.name));
                      } else {
                        setSensitiveAttributes([...sensitiveAttributes, c.name]);
                      }
                    }}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                      isSelected
                        ? 'bg-brand-600/30 text-brand-300 border-brand-500 shadow-sm'
                        : 'bg-surface-950 text-slate-400 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    {c.name} <span className="opacity-50 text-[10px]">({c.unique_count} grps)</span>
                  </button>
                );
              })}
            </div>
            <p className="text-xs text-slate-400 mt-2">
              Select one or more categorical or demographic attributes (e.g., Gender, Age Group, Ethnicity, Employment).
            </p>
          </div>
        </div>
      )}

      {/* Action CTA */}
      <div className="flex justify-end pt-4">
        <button
          onClick={handleRunAudit}
          disabled={loadingAudit}
          className="px-6 py-3 bg-gradient-to-r from-brand-600 to-indigo-600 hover:from-brand-500 hover:to-indigo-500 disabled:opacity-50 text-white font-semibold rounded-xl shadow-lg shadow-brand-600/25 flex items-center gap-2.5 transition-all text-sm"
        >
          {loadingAudit ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
              Evaluating Fairness & Disparities...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-white" />
              Execute Complete Fairness Audit
            </>
          )}
        </button>
      </div>
    </div>
  );
};
