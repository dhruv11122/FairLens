import React from 'react';
import { Layers, AlertCircle, Info } from 'lucide-react';
import { AuditSummary } from '../types/audit';

interface IntersectionalProps {
  audit: AuditSummary;
}

export const Intersectional: React.FC<IntersectionalProps> = ({ audit }) => {
  const inter = audit.intersectional_fairness;

  if (!inter || inter.cells.length === 0) {
    return (
      <div className="p-8 max-w-5xl mx-auto space-y-4">
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <Layers className="w-6 h-6 text-brand-400" />
          Intersectional Fairness Analysis
        </h1>
        <div className="p-6 bg-surface-900 border border-slate-800 rounded-xl text-slate-400 text-sm">
          Intersectional analysis requires at least two protected attributes configured in the audit.
          Configure an audit with multiple attributes (e.g., Gender × Employment Type) to activate this view.
        </div>
      </div>
    );
  }

  // Sort cells by selection rate to reveal compounded disparities
  const sortedCells = [...inter.cells].sort((a, b) => b.selection_rate - a.selection_rate);

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl lg:text-3xl font-bold text-white tracking-tight flex items-center gap-3">
          <Layers className="w-7 h-7 text-brand-400" />
          Intersectional Fairness Matrix
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Compound demographic evaluation across combinations: <b className="text-brand-300">{inter.attributes.join(' × ')}</b>.
          Models may appear equitable on single dimensions while compounding bias for joint subgroups.
        </p>
      </div>

      {/* Disparity Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
        <div className="bg-surface-900 border border-slate-800 rounded-2xl p-5 space-y-1">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
            Compound Subgroups Audited
          </span>
          <div className="text-3xl font-bold text-white font-mono">
            {inter.total_groups}
          </div>
          <p className="text-xs text-slate-400">Distinct intersectional cohorts</p>
        </div>

        <div className="bg-surface-900 border border-slate-800 rounded-2xl p-5 space-y-1">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
            Max Intersectional Selection Gap
          </span>
          <div className="text-3xl font-bold text-amber-400 font-mono">
            {(inter.max_selection_rate_diff * 100).toFixed(1)}%
          </div>
          <p className="text-xs text-slate-400">Highest vs lowest selection rate</p>
        </div>

        <div className="bg-surface-900 border border-slate-800 rounded-2xl p-5 space-y-1">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
            Max Accuracy Disparity
          </span>
          <div className="text-3xl font-bold text-white font-mono">
            {(inter.max_accuracy_diff * 100).toFixed(1)}%
          </div>
          <p className="text-xs text-slate-400">Intersectional performance divergence</p>
        </div>
      </div>

      {/* Intersectional Subgroup Table / Matrix */}
      <div className="bg-surface-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <h3 className="text-base font-semibold text-white">
          Intersectional Cohort Performance & Outcomes
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-surface-950 text-slate-400 text-xs uppercase tracking-wider border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Compound Group</th>
                <th className="py-3 px-4">Sample Size (n)</th>
                <th className="py-3 px-4">Selection Rate</th>
                <th className="py-3 px-4">Accuracy</th>
                <th className="py-3 px-4">Recall (TPR)</th>
                <th className="py-3 px-4">False Positive Rate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-200">
              {sortedCells.map((cell) => {
                const isDisadvantaged = cell.selection_rate < 0.25;

                return (
                  <tr key={cell.group_label} className="hover:bg-slate-800/30 transition-all">
                    <td className="py-3 px-4 font-semibold text-white flex items-center gap-2">
                      {cell.group_label}
                      {cell.is_low_sample && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                          Low Sample (n &lt; 15)
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-300">
                      {cell.sample_size} <span className="text-xs text-slate-400">({(cell.sample_share * 100).toFixed(1)}%)</span>
                    </td>
                    <td className="py-3 px-4 font-mono">
                      <span className={`px-2 py-0.5 rounded text-xs font-semibold ${isDisadvantaged ? 'bg-rose-500/10 text-rose-400' : 'text-slate-200'}`}>
                        {(cell.selection_rate * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-3 px-4 font-mono">{(cell.accuracy * 100).toFixed(1)}%</td>
                    <td className="py-3 px-4 font-mono">
                      {cell.tpr !== null && cell.tpr !== undefined ? `${(cell.tpr * 100).toFixed(1)}%` : 'N/A'}
                    </td>
                    <td className="py-3 px-4 font-mono">
                      {cell.fpr !== null && cell.fpr !== undefined ? `${(cell.fpr * 100).toFixed(1)}%` : 'N/A'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
