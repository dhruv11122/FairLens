import React, { useState } from 'react';
import { 
  Users, AlertTriangle, ArrowUpDown, Info, CheckCircle2 
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  Cell, ReferenceLine, CartesianGrid 
} from 'recharts';
import { AuditSummary } from '../types/audit';

interface FairnessAnalysisProps {
  audit: AuditSummary;
}

export const FairnessAnalysis: React.FC<FairnessAnalysisProps> = ({ audit }) => {
  const sensitiveAttrs = Object.keys(audit.attribute_fairness);
  const [selectedAttr, setSelectedAttr] = useState<string>(sensitiveAttrs[0] || '');
  const [activeMetric, setActiveMetric] = useState<'selection_rate' | 'recall_tpr' | 'accuracy' | 'false_positive_rate'>('selection_rate');

  const attrData = audit.attribute_fairness[selectedAttr];
  if (!attrData) return <div className="p-8 text-slate-400">No protected attribute selected.</div>;

  // Prepare chart data
  const chartData = Object.entries(attrData.subgroup_metrics).map(([grp, m]) => {
    const disp = attrData.subgroup_disparities[grp];
    return {
      subgroup: grp,
      selection_rate: Number((m.selection_rate * 100).toFixed(1)),
      recall_tpr: m.recall_tpr !== null && m.recall_tpr !== undefined ? Number((m.recall_tpr * 100).toFixed(1)) : 0,
      accuracy: Number((m.accuracy * 100).toFixed(1)),
      false_positive_rate: m.false_positive_rate !== null && m.false_positive_rate !== undefined ? Number((m.false_positive_rate * 100).toFixed(1)) : 0,
      sample_size: m.sample_size,
      isRef: grp === attrData.reference_group,
      dp_diff: disp?.demographic_parity_diff,
      eq_diff: disp?.equal_opportunity_diff
    };
  });

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Header & Attribute Tab Selector */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <h1 className="text-2xl lg:text-3xl font-bold text-white tracking-tight flex items-center gap-3">
            <Users className="w-7 h-7 text-brand-400" />
            Subgroup Fairness & Bias Analysis
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Evaluate demographic parity, equal opportunity, and predictive disparities across demographic subgroups.
          </p>
        </div>

        {/* Protected Attribute Selector */}
        <div className="flex items-center gap-2 bg-surface-900 border border-slate-800 p-1.5 rounded-xl">
          <span className="text-xs font-semibold text-slate-400 px-2 uppercase">Attribute:</span>
          {sensitiveAttrs.map((attr) => (
            <button
              key={attr}
              onClick={() => setSelectedAttr(attr)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                selectedAttr === attr
                  ? 'bg-brand-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {attr}
            </button>
          ))}
        </div>
      </div>

      {/* Disparity Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
        <div className="bg-surface-900 border border-slate-800 rounded-2xl p-5 space-y-1">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
            Demographic Parity Difference
          </span>
          <div className="text-3xl font-bold text-white font-mono">
            {(attrData.overall_demographic_parity_difference * 100).toFixed(1)}%
          </div>
          <p className="text-xs text-slate-400">
            Max selection rate minus min selection rate
          </p>
        </div>

        <div className="bg-surface-900 border border-slate-800 rounded-2xl p-5 space-y-1">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
            Demographic Parity Ratio (80% Rule)
          </span>
          <div className="text-3xl font-bold text-white font-mono">
            {attrData.overall_demographic_parity_ratio !== null && attrData.overall_demographic_parity_ratio !== undefined
              ? attrData.overall_demographic_parity_ratio.toFixed(2)
              : 'N/A'}
          </div>
          <p className="text-xs text-slate-400">
            Ratio min/max (0.80 benchmark for disparate impact)
          </p>
        </div>

        <div className="bg-surface-900 border border-slate-800 rounded-2xl p-5 space-y-1">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
            Equal Opportunity Difference (TPR)
          </span>
          <div className="text-3xl font-bold text-white font-mono">
            {attrData.overall_equal_opportunity_difference !== null && attrData.overall_equal_opportunity_difference !== undefined
              ? `${(attrData.overall_equal_opportunity_difference * 100).toFixed(1)}%`
              : 'N/A'}
          </div>
          <p className="text-xs text-slate-400">
            Maximum recall disparity among qualified candidates
          </p>
        </div>
      </div>

      {/* Chart Section */}
      <div className="bg-surface-900 border border-slate-800 rounded-2xl p-6 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-base font-semibold text-white">
              Subgroup Metric Comparison: {selectedAttr}
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Reference group: <b className="text-brand-300">{attrData.reference_group}</b>
            </p>
          </div>

          {/* Metric Toggle */}
          <div className="flex flex-wrap gap-2">
            {[
              { id: 'selection_rate', label: 'Selection Rate' },
              { id: 'recall_tpr', label: 'Recall (TPR)' },
              { id: 'accuracy', label: 'Accuracy' },
              { id: 'false_positive_rate', label: 'False Positive Rate' },
            ].map((m) => (
              <button
                key={m.id}
                onClick={() => setActiveMetric(m.id as any)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  activeMetric === m.id
                    ? 'bg-brand-600 text-white'
                    : 'bg-surface-950 text-slate-400 hover:text-white border border-slate-800'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        {/* Bar Chart */}
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="subgroup" stroke="#94a3b8" tick={{ fontSize: 12 }} />
              <YAxis stroke="#94a3b8" tick={{ fontSize: 12 }} domain={[0, 100]} unit="%" />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                itemStyle={{ color: '#e2e8f0', fontSize: '12px' }}
                formatter={(value: any) => [`${value}%`, 'Value']}
                labelFormatter={(label) => `Subgroup: ${label}`}
              />
              <Bar dataKey={activeMetric} radius={[6, 6, 0, 0]}>
                {chartData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.isRef ? '#6366f1' : '#3b82f6'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Subgroup Details Table */}
      <div className="bg-surface-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <h3 className="text-base font-semibold text-white">Subgroup Breakdown & Statistical Power</h3>
        
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-surface-950 text-slate-400 text-xs uppercase tracking-wider border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Subgroup</th>
                <th className="py-3 px-4">Sample Size (n)</th>
                <th className="py-3 px-4">Selection Rate</th>
                <th className="py-3 px-4">Demographic Parity Diff</th>
                <th className="py-3 px-4">Accuracy</th>
                <th className="py-3 px-4">Recall (TPR)</th>
                <th className="py-3 px-4">False Positive Rate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-200">
              {Object.entries(attrData.subgroup_metrics).map(([grp, m]) => {
                const disp = attrData.subgroup_disparities[grp];
                const isRef = grp === attrData.reference_group;

                return (
                  <tr key={grp} className="hover:bg-slate-800/30 transition-all">
                    <td className="py-3 px-4 font-semibold text-white flex items-center gap-2">
                      {grp}
                      {isRef && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-brand-500/20 text-brand-300 border border-brand-500/30">
                          Reference
                        </span>
                      )}
                      {m.sample_size_warning && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                          Low Power
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-300">
                      {m.sample_size} <span className="text-xs text-slate-400">({(m.sample_share * 100).toFixed(1)}%)</span>
                    </td>
                    <td className="py-3 px-4 font-mono">{(m.selection_rate * 100).toFixed(1)}%</td>
                    <td className="py-3 px-4 font-mono">
                      {disp?.demographic_parity_diff !== undefined && disp.demographic_parity_diff !== null ? (
                        <span className={disp.demographic_parity_diff > 0.1 ? 'text-amber-400 font-semibold' : 'text-slate-300'}>
                          {disp.demographic_parity_diff > 0 ? '+' : ''}{(disp.demographic_parity_diff * 100).toFixed(1)}%
                        </span>
                      ) : '0.0%'}
                    </td>
                    <td className="py-3 px-4 font-mono">{(m.accuracy * 100).toFixed(1)}%</td>
                    <td className="py-3 px-4 font-mono">
                      {m.recall_tpr !== null && m.recall_tpr !== undefined ? `${(m.recall_tpr * 100).toFixed(1)}%` : 'N/A'}
                    </td>
                    <td className="py-3 px-4 font-mono">
                      {m.false_positive_rate !== null && m.false_positive_rate !== undefined ? `${(m.false_positive_rate * 100).toFixed(1)}%` : 'N/A'}
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
