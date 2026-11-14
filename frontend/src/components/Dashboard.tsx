import React from 'react';
import { 
  ShieldCheck, AlertTriangle, Users, Layers, 
  ArrowUpRight, ArrowDownRight, CheckCircle2, Info
} from 'lucide-react';
import { AuditSummary } from '../types/audit';

interface DashboardProps {
  audit: AuditSummary;
  onNavigateTab: (tab: string) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ audit, onNavigateTab }) => {
  const perf = audit.overall_performance;
  const highSeverityFindings = audit.findings.filter(f => f.severity === 'high');
  const sensitiveList = Object.keys(audit.attribute_fairness);

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Top Banner / Audit Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-brand-500/10 text-brand-400 border border-brand-500/20">
              Audit ID: {audit.audit_id}
            </span>
            <span className="text-xs text-slate-400">• Completed {audit.created_at}</span>
          </div>
          <h1 className="text-2xl lg:text-3xl font-bold text-white tracking-tight">
            Fairness Audit Overview: {audit.dataset_name}
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            {audit.is_predictions_only 
              ? 'Predictions-Only Evaluation Mode' 
              : `Model: ${audit.model_name || audit.model_id}`} | Target: <code className="text-brand-300 bg-slate-800/80 px-1.5 py-0.5 rounded text-xs">{audit.target_column}</code>
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => onNavigateTab('reports')}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg border border-slate-700 transition-all flex items-center gap-2"
          >
            Download PDF Report
          </button>
          <button
            onClick={() => onNavigateTab('fairness')}
            className="px-4 py-2 bg-brand-600 hover:bg-brand-500 text-white text-xs font-semibold rounded-lg shadow-md shadow-brand-600/20 transition-all flex items-center gap-2"
          >
            Explore Subgroup Bias
          </button>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {/* Overall Accuracy */}
        <div className="bg-surface-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium uppercase tracking-wider">
            <span>Overall Accuracy</span>
            <ShieldCheck className="w-4 h-4 text-brand-400" />
          </div>
          <div className="text-3xl font-bold text-white tracking-tight">
            {(perf.accuracy * 100).toFixed(1)}%
          </div>
          <p className="text-[11px] text-slate-400">
            Balanced Acc: {(perf.balanced_accuracy * 100).toFixed(1)}% across {perf.total_samples.toLocaleString()} samples
          </p>
        </div>

        {/* Max Demographic Parity Gap */}
        {(() => {
          const maxDp = Math.max(...Object.values(audit.attribute_fairness).map(a => a.overall_demographic_parity_difference), 0);
          const exceeds = maxDp > audit.disparity_threshold;
          return (
            <div className="bg-surface-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-2">
              <div className="flex items-center justify-between text-slate-400 text-xs font-medium uppercase tracking-wider">
                <span>Max Demographic Gap</span>
                <Users className="w-4 h-4 text-indigo-400" />
              </div>
              <div className={`text-3xl font-bold tracking-tight ${exceeds ? 'text-amber-400' : 'text-emerald-400'}`}>
                {(maxDp * 100).toFixed(1)}%
              </div>
              <p className="text-[11px] text-slate-400">
                {exceeds ? `Exceeds ${(audit.disparity_threshold * 100).toFixed(0)}% threshold` : 'Below configured alert threshold'}
              </p>
            </div>
          );
        })()}

        {/* Max Equal Opportunity Gap */}
        {(() => {
          const eqOpps = Object.values(audit.attribute_fairness)
            .map(a => a.overall_equal_opportunity_difference)
            .filter((v): v is number => v !== null && v !== undefined);
          const maxEq = eqOpps.length > 0 ? Math.max(...eqOpps) : 0;
          return (
            <div className="bg-surface-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-2">
              <div className="flex items-center justify-between text-slate-400 text-xs font-medium uppercase tracking-wider">
                <span>Max Equal Opp Gap</span>
                <AlertTriangle className="w-4 h-4 text-rose-400" />
              </div>
              <div className={`text-3xl font-bold tracking-tight ${maxEq > 0.15 ? 'text-rose-400' : 'text-slate-100'}`}>
                {eqOpps.length > 0 ? `${(maxEq * 100).toFixed(1)}%` : 'N/A'}
              </div>
              <p className="text-[11px] text-slate-400">
                Gap in true positive rates (original audit threshold)
              </p>
            </div>
          );
        })()}

        {/* Audit Flags */}
        <div className="bg-surface-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium uppercase tracking-wider">
            <span>Fairness Findings</span>
            <AlertTriangle className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-3xl font-bold text-white tracking-tight flex items-center gap-2">
            {audit.findings.length}
            {highSeverityFindings.length > 0 && (
              <span className="text-xs bg-rose-500/20 text-rose-300 font-semibold px-2 py-0.5 rounded-full border border-rose-500/30">
                {highSeverityFindings.length} High
              </span>
            )}
          </div>
          <p className="text-[11px] text-slate-400">
            Across {sensitiveList.length} audited protected attributes
          </p>
        </div>
      </div>

      {/* Disparity Matrix Summary across Audited Attributes */}
      <div className="bg-surface-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Users className="w-5 h-5 text-brand-400" />
            Protected Attribute Disparity Overview
          </h2>
          <span className="text-xs text-slate-400">
            Threshold: {(audit.disparity_threshold * 100).toFixed(0)}% Disparity Limit
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-surface-950 text-slate-400 text-xs uppercase tracking-wider border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Protected Attribute</th>
                <th className="py-3 px-4">Reference Subgroup</th>
                <th className="py-3 px-4">Audited Groups</th>
                <th className="py-3 px-4">Demographic Parity Diff</th>
                <th className="py-3 px-4">Demographic Parity Ratio</th>
                <th className="py-3 px-4">Equal Opportunity Diff</th>
                <th className="py-3 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-200">
              {Object.entries(audit.attribute_fairness).map(([attr, res]) => {
                const dpExceeds = res.overall_demographic_parity_difference > audit.disparity_threshold;
                const eqExceeds = res.overall_equal_opportunity_difference && res.overall_equal_opportunity_difference > audit.disparity_threshold;

                return (
                  <tr key={attr} className="hover:bg-slate-800/30 transition-all">
                    <td className="py-3.5 px-4 font-semibold text-white">{attr}</td>
                    <td className="py-3.5 px-4 text-slate-400">
                      <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 text-xs">
                        {res.reference_group}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-slate-400">{res.subgroups.length} subgroups</td>
                    <td className="py-3.5 px-4 font-mono">
                      <span className={`px-2 py-0.5 rounded text-xs font-semibold ${dpExceeds ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20' : 'text-slate-300'}`}>
                        {(res.overall_demographic_parity_difference * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-3.5 px-4 font-mono text-xs">
                      {res.overall_demographic_parity_ratio !== null && res.overall_demographic_parity_ratio !== undefined ? (
                        <span className={res.overall_demographic_parity_ratio < 0.8 ? 'text-rose-400 font-semibold' : 'text-emerald-400'}>
                          {res.overall_demographic_parity_ratio.toFixed(2)} (4/5ths Rule)
                        </span>
                      ) : 'N/A'}
                    </td>
                    <td className="py-3.5 px-4 font-mono">
                      {res.overall_equal_opportunity_difference !== null && res.overall_equal_opportunity_difference !== undefined ? (
                        <span className={`px-2 py-0.5 rounded text-xs font-semibold ${eqExceeds ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' : 'text-slate-300'}`}>
                          {(res.overall_equal_opportunity_difference * 100).toFixed(1)}%
                        </span>
                      ) : (
                        <span className="text-slate-400 text-xs">N/A</span>
                      )}
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      <button
                        onClick={() => onNavigateTab('fairness')}
                        className="text-xs text-brand-400 hover:text-brand-300 font-medium"
                      >
                        Inspect →
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Top Ranked Findings Feed */}
      <div className="bg-surface-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-amber-400" />
              Automated Audit Findings & Recommendations
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Deterministic observations derived directly from computed group differences and statistical thresholds.
            </p>
          </div>
          <span className="text-xs text-slate-400">{audit.findings.length} findings recorded</span>
        </div>

        {audit.findings.length === 0 ? (
          <div className="p-8 text-center text-slate-400 bg-surface-950/40 rounded-xl border border-slate-800">
            <CheckCircle2 className="w-8 h-8 text-emerald-400 mx-auto mb-2" />
            <p className="text-sm font-semibold text-white">No configured investigation alerts</p>
            <p className="text-xs text-slate-400 mt-1">
              No evaluated rule exceeded its configured threshold. This does not establish that the model is fair or useful.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {audit.findings.slice(0, 4).map((f) => {
              const badgeBg = f.severity === 'high' 
                ? 'bg-rose-500/10 text-rose-400 border-rose-500/20' 
                : f.severity === 'medium' 
                ? 'bg-amber-500/10 text-amber-400 border-amber-500/20' 
                : 'bg-slate-800 text-slate-300 border-slate-700';

              return (
                <div key={f.id} className="p-4 bg-surface-950/60 border border-slate-800 rounded-xl space-y-2">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border ${badgeBg}`}>
                        {f.severity} Severity
                      </span>
                      <h4 className="text-sm font-semibold text-white">{f.title}</h4>
                    </div>
                    <span className="text-xs font-mono text-slate-400">{f.attribute}</span>
                  </div>

                  <p className="text-xs text-slate-300 leading-relaxed">
                    {f.explanation}
                  </p>

                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pt-2 border-t border-slate-800/80 text-[11px]">
                    <span className="text-slate-400 font-mono">
                      <b>Evidence:</b> {f.evidence} ({f.sample_size_info})
                    </span>
                    <span className="text-brand-300 font-medium">
                      Action: {f.suggested_action}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Disparity Scope Note */}
      <div className="p-4 bg-brand-950/20 border border-brand-800/40 rounded-xl flex items-start gap-3 text-xs text-slate-400 leading-relaxed">
        <Info className="w-5 h-5 text-brand-400 shrink-0 mt-0.5" />
        <div>
          <b className="text-slate-300">Methodological Note on Algorithmic Auditing:</b> Disparities in selection rate or accuracy 
          measure observable differences in classification outcomes across groups. These metrics highlight areas warranting human 
          oversight and governance, but do not on their own prove intentional misconduct or unlawful discrimination.
        </div>
      </div>
    </div>
  );
};
