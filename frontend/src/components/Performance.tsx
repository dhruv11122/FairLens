import React from 'react';
import { 
  BarChart2, Target, CheckCircle, XCircle, 
  Percent, Hash, Activity 
} from 'lucide-react';
import { AuditSummary } from '../types/audit';

interface PerformanceProps {
  audit: AuditSummary;
}

export const Performance: React.FC<PerformanceProps> = ({ audit }) => {
  const perf = audit.overall_performance;
  const cm = perf.confusion_matrix;

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl lg:text-3xl font-bold text-white tracking-tight flex items-center gap-3">
          <BarChart2 className="w-7 h-7 text-brand-400" />
          Overall Model Performance
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Standard binary classification metrics computed over the complete evaluation dataset of {perf.total_samples.toLocaleString()} instances.
        </p>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <div className="bg-surface-900 border border-slate-800 rounded-xl p-4 space-y-1">
          <span className="text-[11px] font-semibold uppercase text-slate-400">Accuracy</span>
          <div className="text-2xl font-bold text-white">{(perf.accuracy * 100).toFixed(1)}%</div>
        </div>

        <div className="bg-surface-900 border border-slate-800 rounded-xl p-4 space-y-1">
          <span className="text-[11px] font-semibold uppercase text-slate-400">Balanced Accuracy</span>
          <div className="text-2xl font-bold text-white">{(perf.balanced_accuracy * 100).toFixed(1)}%</div>
        </div>

        <div className="bg-surface-900 border border-slate-800 rounded-xl p-4 space-y-1">
          <span className="text-[11px] font-semibold uppercase text-slate-400">Precision (PPV)</span>
          <div className="text-2xl font-bold text-emerald-400">
            {perf.precision !== null && perf.precision !== undefined ? `${(perf.precision * 100).toFixed(1)}%` : 'N/A'}
          </div>
        </div>

        <div className="bg-surface-900 border border-slate-800 rounded-xl p-4 space-y-1">
          <span className="text-[11px] font-semibold uppercase text-slate-400">Recall / TPR</span>
          <div className="text-2xl font-bold text-brand-400">
            {perf.recall !== null && perf.recall !== undefined ? `${(perf.recall * 100).toFixed(1)}%` : 'N/A'}
          </div>
        </div>

        <div className="bg-surface-900 border border-slate-800 rounded-xl p-4 space-y-1">
          <span className="text-[11px] font-semibold uppercase text-slate-400">F1-Score</span>
          <div className="text-2xl font-bold text-indigo-400">
            {perf.f1_score !== null && perf.f1_score !== undefined ? perf.f1_score.toFixed(3) : 'N/A'}
          </div>
        </div>

        <div className="bg-surface-900 border border-slate-800 rounded-xl p-4 space-y-1">
          <span className="text-[11px] font-semibold uppercase text-slate-400">ROC-AUC Score</span>
          <div className="text-2xl font-bold text-violet-400">
            {perf.roc_auc !== null && perf.roc_auc !== undefined ? perf.roc_auc.toFixed(3) : 'N/A'}
          </div>
        </div>
      </div>

      {/* Confusion Matrix and Rate Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Confusion Matrix Card */}
        <div className="bg-surface-900 border border-slate-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <Target className="w-5 h-5 text-brand-400" />
              Confusion Matrix
            </h3>
            <span className="text-xs text-slate-400">Total: {perf.total_samples.toLocaleString()}</span>
          </div>

          <div className="grid grid-cols-2 gap-3 pt-2">
            {/* True Positive */}
            <div className="bg-emerald-950/20 border border-emerald-800/40 rounded-xl p-4 text-center">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-emerald-400 block mb-1">
                True Positives (TP)
              </span>
              <span className="text-3xl font-extrabold text-emerald-300 font-mono">
                {cm.true_positive.toLocaleString()}
              </span>
              <p className="text-[11px] text-slate-400 mt-1">Actual Positive & Predicted Positive</p>
            </div>

            {/* False Positive */}
            <div className="bg-rose-950/20 border border-rose-800/40 rounded-xl p-4 text-center">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-rose-400 block mb-1">
                False Positives (FP)
              </span>
              <span className="text-3xl font-extrabold text-rose-300 font-mono">
                {cm.false_positive.toLocaleString()}
              </span>
              <p className="text-[11px] text-slate-400 mt-1">Type I Error (Predicted Positive)</p>
            </div>

            {/* False Negative */}
            <div className="bg-amber-950/20 border border-amber-800/40 rounded-xl p-4 text-center">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-amber-400 block mb-1">
                False Negatives (FN)
              </span>
              <span className="text-3xl font-extrabold text-amber-300 font-mono">
                {cm.false_negative.toLocaleString()}
              </span>
              <p className="text-[11px] text-slate-400 mt-1">Type II Error (Missed Opportunity)</p>
            </div>

            {/* True Negative */}
            <div className="bg-slate-800/40 border border-slate-700/40 rounded-xl p-4 text-center">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-300 block mb-1">
                True Negatives (TN)
              </span>
              <span className="text-3xl font-extrabold text-white font-mono">
                {cm.true_negative.toLocaleString()}
              </span>
              <p className="text-[11px] text-slate-400 mt-1">Actual Negative & Predicted Negative</p>
            </div>
          </div>
        </div>

        {/* Operating Error Rates */}
        <div className="bg-surface-900 border border-slate-800 rounded-2xl p-6 space-y-4">
          <h3 className="text-base font-semibold text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-indigo-400" />
            Decision Error Rates & Distribution
          </h3>

          <div className="space-y-4 pt-2">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300">Selection Rate (Predicted Favorable Rate)</span>
                <span className="font-mono text-white font-semibold">{(perf.selection_rate * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-surface-950 rounded-full h-2 overflow-hidden border border-slate-800">
                <div 
                  className="bg-brand-500 h-2 rounded-full" 
                  style={{ width: `${Math.min(100, Math.max(0, perf.selection_rate * 100))}%` }}
                ></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300">False Positive Rate (FPR = FP / Actual Negatives)</span>
                <span className="font-mono text-rose-400 font-semibold">
                  {perf.false_positive_rate !== null && perf.false_positive_rate !== undefined ? `${(perf.false_positive_rate * 100).toFixed(1)}%` : 'N/A'}
                </span>
              </div>
              <div className="w-full bg-surface-950 rounded-full h-2 overflow-hidden border border-slate-800">
                <div 
                  className="bg-rose-500 h-2 rounded-full" 
                  style={{ width: `${Math.min(100, Math.max(0, (perf.false_positive_rate || 0) * 100))}%` }}
                ></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300">False Negative Rate (FNR = FN / Actual Positives)</span>
                <span className="font-mono text-amber-400 font-semibold">
                  {perf.false_negative_rate !== null && perf.false_negative_rate !== undefined ? `${(perf.false_negative_rate * 100).toFixed(1)}%` : 'N/A'}
                </span>
              </div>
              <div className="w-full bg-surface-950 rounded-full h-2 overflow-hidden border border-slate-800">
                <div 
                  className="bg-amber-500 h-2 rounded-full" 
                  style={{ width: `${Math.min(100, Math.max(0, (perf.false_negative_rate || 0) * 100))}%` }}
                ></div>
              </div>
            </div>

            <div className="pt-2 border-t border-slate-800 text-xs text-slate-400 leading-relaxed">
              Ground truth positive prevalence: <b className="text-slate-300">{(perf.positive_samples / perf.total_samples * 100).toFixed(1)}%</b> ({perf.positive_samples} pos / {perf.negative_samples} neg).
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
