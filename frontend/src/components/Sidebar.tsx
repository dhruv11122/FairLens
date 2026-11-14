import React from 'react';
import { 
  ShieldCheck, AlertTriangle, Layers, BarChart2, 
  Sliders, FileText, UploadCloud, Eye
} from 'lucide-react';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  hasAudit: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, setActiveTab, hasAudit }) => {
  const navItems = [
    { id: 'new-audit', label: 'New Audit', icon: UploadCloud, always: true },
    { id: 'dashboard', label: 'Overview', icon: ShieldCheck, requiresAudit: true },
    { id: 'performance', label: 'Model Performance', icon: BarChart2, requiresAudit: true },
    { id: 'fairness', label: 'Fairness Disparities', icon: AlertTriangle, requiresAudit: true },
    { id: 'intersectional', label: 'Intersectional Analysis', icon: Layers, requiresAudit: true },
    { id: 'explainability', label: 'Explainability & Proxies', icon: Eye, requiresAudit: true },
    { id: 'threshold', label: 'Threshold Explorer', icon: Sliders, requiresAudit: true },
    { id: 'reports', label: 'Export & Report', icon: FileText, requiresAudit: true },
  ];

  return (
    <aside className="w-64 bg-surface-900 border-r border-slate-800 flex flex-col justify-between shrink-0">
      <div>
        {/* Brand */}
        <div className="h-16 flex items-center px-6 border-b border-slate-800 gap-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-brand-600 to-indigo-400 flex items-center justify-center shadow-lg shadow-brand-500/20">
            <ShieldCheck className="w-5 h-5 text-white" />
          </div>
          <div>
            <span className="font-bold text-lg text-white tracking-tight flex items-center gap-1.5">
              FairLens<span className="text-brand-500 text-xs px-1.5 py-0.5 rounded bg-brand-500/10 border border-brand-500/20">PRO</span>
            </span>
            <p className="text-[10px] text-slate-400 font-medium uppercase tracking-wider">AI Fairness Auditor</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="p-4 space-y-1.5">
          <div className="text-[11px] font-semibold text-slate-400 px-3 mb-2 uppercase tracking-wider">Audit Workflow</div>
          {navItems.map((item) => {
            const Icon = item.icon;
            const disabled = item.requiresAudit && !hasAudit;
            const active = activeTab === item.id;

            return (
              <button
                key={item.id}
                disabled={disabled}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  active
                    ? 'bg-brand-600 text-white shadow-md shadow-brand-600/20'
                    : disabled
                    ? 'text-slate-400 cursor-not-allowed opacity-40'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                <Icon className={`w-4 h-4 ${active ? 'text-white' : 'text-slate-400'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer Info */}
      <div className="p-4 border-t border-slate-800/80 m-3 bg-surface-950/60 rounded-xl border">
        <div className="flex items-center gap-2 mb-1.5">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></div>
          <span className="text-xs font-semibold text-slate-300">Auditor Ready</span>
        </div>
        <p className="text-[11px] text-slate-400 leading-relaxed">
          Complies with Fairlearn & NIST AI Risk Management Framework foundations.
        </p>
      </div>
    </aside>
  );
};
