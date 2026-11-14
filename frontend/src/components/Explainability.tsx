import React from 'react';
import { Eye, AlertTriangle, ShieldCheck, HelpCircle } from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid 
} from 'recharts';
import { AuditSummary } from '../types/audit';

interface ExplainabilityProps {
  audit: AuditSummary;
}

export const Explainability: React.FC<ExplainabilityProps> = ({ audit }) => {
  const exp = audit.explainability;

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl lg:text-3xl font-bold text-white tracking-tight flex items-center gap-3">
          <Eye className="w-7 h-7 text-brand-400" />
          Explainable AI & Proxy Feature Detection
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Investigate which features drive decision boundaries and detect indirect proxies that correlate with protected attributes.
        </p>
      </div>

      {/* Proxy Warning Banner */}
      {exp?.proxy_warnings && exp.proxy_warnings.length > 0 && (
        <div className="bg-amber-950/20 border border-amber-800/40 rounded-2xl p-6 space-y-3">
          <div className="flex items-center gap-2 text-amber-400 font-semibold text-sm">
            <AlertTriangle className="w-5 h-5" />
            Detected {exp.proxy_warnings.length} Potential Proxy Features
          </div>
          <p className="text-xs text-slate-300">
            A model may achieve formal blindness by omitting sensitive attributes directly, but can replicate demographic 
            disparities through high correlation with proxy variables (e.g. zip code, school, occupation).
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2">
            {exp.proxy_warnings.map((pw, idx) => (
              <div key={idx} className="bg-surface-950/60 border border-amber-900/30 rounded-xl p-3.5 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-white text-xs">
                    {pw.feature} ↔ {pw.sensitive_attribute}
                  </span>
                  <span className="text-[11px] font-mono font-bold text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded">
                    Correlation: {pw.correlation_score.toFixed(2)}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  {pw.explanation}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Global Feature Importance Chart */}
      {exp?.supported && exp.global_importance.length > 0 ? (
        <div className="bg-surface-900 border border-slate-800 rounded-2xl p-6 space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-semibold text-white">Global Feature Importance</h3>
            <span className="text-xs text-slate-400">Relative Weight in Estimator</span>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                layout="vertical"
                data={exp.global_importance.slice(0, 8)}
                margin={{ top: 10, right: 30, left: 80, bottom: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                <XAxis type="number" stroke="#94a3b8" tick={{ fontSize: 12 }} unit="%" />
                <YAxis dataKey="feature" type="category" stroke="#94a3b8" tick={{ fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                  itemStyle={{ color: '#e2e8f0', fontSize: '12px' }}
                  formatter={(val: any) => [`${(Number(val) * 100).toFixed(1)}%`, 'Relative Weight']}
                />
                <Bar dataKey="relative_importance" fill="#6366f1" radius={[0, 6, 6, 0]}>
                  {exp.global_importance.slice(0, 8).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={index === 0 ? '#4f46e5' : '#6366f1'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : (
        <div className="p-6 bg-surface-900 border border-slate-800 rounded-xl text-slate-400 text-sm">
          {exp?.reason || 'Intrinsic feature importance is not available for this model type.'}
        </div>
      )}

      {/* Governance & Interpretability Note */}
      <div className="p-4 bg-slate-900/60 border border-slate-800 rounded-xl flex items-start gap-3 text-xs text-slate-400 leading-relaxed">
        <HelpCircle className="w-5 h-5 text-slate-400 shrink-0 mt-0.5" />
        <div>
          <b>Scientific Disclaimer:</b> Feature importance indicators quantify associative influence in classification logic. 
          High feature importance does not establish causal discrimination, nor does low importance guarantee harmlessness.
        </div>
      </div>
    </div>
  );
};
