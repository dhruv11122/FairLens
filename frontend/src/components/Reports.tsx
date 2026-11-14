import React from 'react';
import { FileText, Download, Code, CheckCircle2, ShieldAlert } from 'lucide-react';
import { api } from '../services/api';
import { AuditSummary } from '../types/audit';

interface ReportsProps {
  audit: AuditSummary;
}

export const Reports: React.FC<ReportsProps> = ({ audit }) => {
  const pdfUrl = api.getReportPdfUrl(audit.audit_id);
  const jsonUrl = api.getExportJsonUrl(audit.audit_id);

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl lg:text-3xl font-bold text-white tracking-tight flex items-center gap-3">
          <FileText className="w-7 h-7 text-brand-400" />
          Export Governance Reports & Audit Metadata
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Download formatted executive audit reports for compliance records or export raw machine-readable JSON results.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* PDF Executive Report Card */}
        <div className="bg-surface-900 border border-slate-800 rounded-2xl p-6 space-y-4 flex flex-col justify-between">
          <div className="space-y-3">
            <div className="w-10 h-10 rounded-xl bg-brand-500/10 border border-brand-500/20 flex items-center justify-center text-brand-400">
              <FileText className="w-5 h-5" />
            </div>
            <h3 className="text-lg font-semibold text-white">Executive Audit Dossier (PDF)</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Publication-grade PDF document containing audit overview, overall classification metrics, subgroup breakdown tables, 
              ranked automated findings, and regulatory disclaimer notices.
            </p>
            <ul className="text-xs text-slate-300 space-y-1.5 pt-2">
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> NIST AI RMF & Fairlearn aligned structure
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> Executive summary with reference disparities
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> Ready for governance, risk, and compliance review
              </li>
            </ul>
          </div>

          <a
            href={pdfUrl}
            target="_blank"
            rel="noopener noreferrer"
            download={`FairLens_Audit_${audit.audit_id}.pdf`}
            className="w-full mt-4 py-3 bg-brand-600 hover:bg-brand-500 text-white font-semibold text-xs rounded-xl shadow-lg shadow-brand-600/25 flex items-center justify-center gap-2 transition-all"
          >
            <Download className="w-4 h-4" />
            Download PDF Audit Report
          </a>
        </div>

        {/* JSON Export Card */}
        <div className="bg-surface-900 border border-slate-800 rounded-2xl p-6 space-y-4 flex flex-col justify-between">
          <div className="space-y-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
              <Code className="w-5 h-5" />
            </div>
            <h3 className="text-lg font-semibold text-white">Raw Audit Metadata (JSON)</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Standard JSON object containing all computed metric frames, subgroup disparities, intersectional matrices, 
              and proxy feature associations for CI/CD pipelines and model registry integration.
            </p>
            <ul className="text-xs text-slate-300 space-y-1.5 pt-2">
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> Machine-parsable for MLOps gate checks
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> Full schema validation
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> Complete provenance and timestamps
              </li>
            </ul>
          </div>

          <a
            href={jsonUrl}
            target="_blank"
            rel="noopener noreferrer"
            download={`fairlens_audit_${audit.audit_id}.json`}
            className="w-full mt-4 py-3 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 font-semibold text-xs rounded-xl shadow-sm flex items-center justify-center gap-2 transition-all"
          >
            <Download className="w-4 h-4" />
            Export Raw JSON Result
          </a>
        </div>
      </div>

      {/* Governance & Disclaimer Banner */}
      <div className="p-5 bg-surface-900 border border-slate-800 rounded-2xl space-y-2">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-amber-400" />
          Algorithmic Fairness Impossibility Theorem & Operational Guidance
        </h4>
        <p className="text-xs text-slate-400 leading-relaxed">
          Mathematical proofs established by Kleinberg et al. (2016) and Chouldechova (2017) show that when base rates differ between groups, 
          it is mathematically impossible for a predictive model to satisfy Demographic Parity, Equal Opportunity (equal false negatives), 
          and Predictive Parity (equal precision / calibration) simultaneously (unless the classifier achieves 100% accuracy). 
          Auditors and machine learning practitioners must select fairness criteria that correspond with the ethical, operational, and regulatory 
          priorities of their specific application domain.
        </p>
      </div>
    </div>
  );
};
