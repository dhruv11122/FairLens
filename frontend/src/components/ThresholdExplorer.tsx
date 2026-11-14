import React, { useState, useEffect } from 'react';
import { Sliders, RefreshCw, AlertCircle, Info } from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  CartesianGrid, ReferenceLine 
} from 'recharts';
import { api } from '../services/api';
import { AuditSummary, ThresholdSimulationResponse } from '../types/audit';

interface ThresholdExplorerProps {
  audit: AuditSummary;
}

export const ThresholdExplorer: React.FC<ThresholdExplorerProps> = ({ audit }) => {
  const sensitiveAttrs = Object.keys(audit.attribute_fairness);
  const [selectedAttr, setSelectedAttr] = useState<string>(sensitiveAttrs[0] || '');
  const [threshold, setThreshold] = useState<number>(0.50);
  const [data, setData] = useState<ThresholdSimulationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const fetchSimulation = async (attr: string, t: number) => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await api.simulateThreshold(audit.audit_id, attr, t);
      setData(res);
    } catch (err: any) {
      setErrorMsg(err.message || 'Threshold simulation failed.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (audit.has_probabilities && selectedAttr) {
      fetchSimulation(selectedAttr, threshold);
    }
  }, [audit.audit_id, selectedAttr]);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setThreshold(val);
  };

  const handleApplyThreshold = () => {
    if (selectedAttr) {
      fetchSimulation(selectedAttr, threshold);
    }
  };

  if (!audit.has_probabilities) {
    return (
      <div className="p-8 max-w-5xl mx-auto space-y-4">
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <Sliders className="w-6 h-6 text-brand-400" />
          Fairness-Performance Tradeoff Simulator
        </h1>
        <div className="p-6 bg-surface-900 border border-slate-800 rounded-xl text-slate-400 text-sm">
          Threshold simulation requires models that output continuous prediction probabilities or decision scores. 
          Provide a model with <code>predict_proba</code> support or a dataset containing prediction confidence scores to activate this feature.
        </div>
      </div>
    );
  }

  const chartCurve = data?.curve.map(pt => ({
    threshold: pt.threshold,
    Accuracy: Number((pt.accuracy * 100).toFixed(1)),
    Recall: pt.recall !== null && pt.recall !== undefined ? Number((pt.recall * 100).toFixed(1)) : 0,
    Precision: pt.precision !== null && pt.precision !== undefined ? Number((pt.precision * 100).toFixed(1)) : 0,
    'Demographic Parity Gap': Number(((pt.demographic_parity_diff || 0) * 100).toFixed(1)),
    'Equal Opportunity Gap': Number(((pt.equal_opportunity_diff || 0) * 100).toFixed(1)),
  })) || [];

  const curr = data?.current;

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <h1 className="text-2xl lg:text-3xl font-bold text-white tracking-tight flex items-center gap-3">
            <Sliders className="w-7 h-7 text-brand-400" />
            Decision Threshold & Tradeoff Simulator
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Explore decision thresholds without changing the saved audit. Overview, subgroup analysis, and exports reflect the original model predictions.
          </p>
        </div>

        {/* Attribute Selector */}
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

      {errorMsg && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-300 text-sm flex items-center gap-2">
          <AlertCircle className="w-4 h-4" />
          {errorMsg}
        </div>
      )}

      {/* Interactive Slider Card */}
      <div className="bg-surface-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-sm font-semibold text-white">Global Decision Threshold (τ):</span>
            <span className="text-lg font-extrabold text-brand-400 font-mono bg-brand-500/10 px-3 py-1 rounded-lg border border-brand-500/30">
              {threshold.toFixed(2)}
            </span>
          </div>

          <button
            onClick={handleApplyThreshold}
            disabled={loading}
            className="px-4 py-1.5 bg-brand-600 hover:bg-brand-500 text-white text-xs font-semibold rounded-lg shadow-sm flex items-center gap-1.5 transition-all"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            Recalculate
          </button>
        </div>

        <input
          type="range"
          min="0.05"
          max="0.95"
          step="0.01"
          value={threshold}
          onChange={handleSliderChange}
          onMouseUp={handleApplyThreshold}
          className="w-full accent-brand-500 cursor-pointer h-2 bg-surface-950 rounded-lg"
        />

        <div className="flex justify-between text-[11px] text-slate-400 font-mono">
          <span>0.05 (More Lenient / Higher Selection)</span>
          <span>0.50 (Standard Neutral)</span>
          <span>0.95 (More Conservative / Lower Selection)</span>
        </div>
      </div>

      {/* Recalculated Point Metrics */}
      {curr && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          <div className="bg-surface-900 border border-slate-800 rounded-xl p-4 space-y-1">
            <span className="text-[11px] font-semibold uppercase text-slate-400">Accuracy @ τ</span>
            <div className="text-2xl font-bold text-white">{(curr.accuracy * 100).toFixed(1)}%</div>
          </div>

          <div className="bg-surface-900 border border-slate-800 rounded-xl p-4 space-y-1">
            <span className="text-[11px] font-semibold uppercase text-slate-400">Precision @ τ</span>
            <div className="text-2xl font-bold text-emerald-400">
              {curr.precision !== null && curr.precision !== undefined ? `${(curr.precision * 100).toFixed(1)}%` : '0.0%'}
            </div>
          </div>

          <div className="bg-surface-900 border border-slate-800 rounded-xl p-4 space-y-1">
            <span className="text-[11px] font-semibold uppercase text-slate-400">Recall @ τ</span>
            <div className="text-2xl font-bold text-brand-400">
              {curr.recall !== null && curr.recall !== undefined ? `${(curr.recall * 100).toFixed(1)}%` : '0.0%'}
            </div>
          </div>

          <div className="bg-surface-900 border border-slate-800 rounded-xl p-4 space-y-1">
            <span className="text-[11px] font-semibold uppercase text-slate-400">Selection Rate</span>
            <div className="text-2xl font-bold text-slate-200">{(curr.selection_rate * 100).toFixed(1)}%</div>
          </div>

          <div className="bg-surface-900 border border-slate-800 rounded-xl p-4 space-y-1">
            <span className="text-[11px] font-semibold uppercase text-slate-400">Demographic Gap</span>
            <div className="text-2xl font-bold text-amber-400">
              {curr.demographic_parity_diff !== null && curr.demographic_parity_diff !== undefined 
                ? `${(curr.demographic_parity_diff * 100).toFixed(1)}%` 
                : '0.0%'}
            </div>
          </div>

          <div className="bg-surface-900 border border-slate-800 rounded-xl p-4 space-y-1">
            <span className="text-[11px] font-semibold uppercase text-slate-400">Equal Opp Gap</span>
            <div className="text-2xl font-bold text-rose-400">
              {curr.equal_opportunity_diff !== null && curr.equal_opportunity_diff !== undefined 
                ? `${(curr.equal_opportunity_diff * 100).toFixed(1)}%` 
                : '0.0%'}
            </div>
          </div>
        </div>
      )}

      {/* Tradeoff Curves Visualizer */}
      <div className="bg-surface-900 border border-slate-800 rounded-2xl p-6 space-y-6">
        <h3 className="text-base font-semibold text-white">
          Fairness vs. Accuracy Tradeoff Curve across Thresholds
        </h3>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartCurve} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="threshold" stroke="#94a3b8" tick={{ fontSize: 12 }} />
              <YAxis stroke="#94a3b8" tick={{ fontSize: 12 }} domain={[0, 100]} unit="%" />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                itemStyle={{ fontSize: '12px' }}
              />
              <ReferenceLine x={threshold} stroke="#6366f1" strokeDasharray="3 3" label={{ value: `Current τ=${threshold.toFixed(2)}`, fill: '#818cf8', fontSize: 11 }} />
              <Line type="monotone" dataKey="Accuracy" stroke="#10b981" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="Recall" stroke="#3b82f6" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="Demographic Parity Gap" stroke="#f59e0b" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="Equal Opportunity Gap" stroke="#ef4444" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-6 pt-2 text-xs">
          <span className="flex items-center gap-1.5 text-slate-300">
            <span className="w-3 h-3 rounded-full bg-emerald-500"></span> Accuracy
          </span>
          <span className="flex items-center gap-1.5 text-slate-300">
            <span className="w-3 h-3 rounded-full bg-blue-500"></span> Recall
          </span>
          <span className="flex items-center gap-1.5 text-slate-300">
            <span className="w-3 h-3 rounded-full bg-amber-500"></span> Demographic Parity Disparity
          </span>
          <span className="flex items-center gap-1.5 text-slate-300">
            <span className="w-3 h-3 rounded-full bg-rose-500"></span> Equal Opportunity Disparity
          </span>
        </div>
      </div>
    </div>
  );
};
