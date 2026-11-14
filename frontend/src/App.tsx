import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { NewAudit } from './components/NewAudit';
import { Dashboard } from './components/Dashboard';
import { Performance } from './components/Performance';
import { FairnessAnalysis } from './components/FairnessAnalysis';
import { Intersectional } from './components/Intersectional';
import { Explainability } from './components/Explainability';
import { ThresholdExplorer } from './components/ThresholdExplorer';
import { Reports } from './components/Reports';
import { AuditSummary } from './types/audit';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('new-audit');
  const [currentAudit, setCurrentAudit] = useState<AuditSummary | null>(null);

  const handleAuditComplete = (audit: AuditSummary) => {
    setCurrentAudit(audit);
    setActiveTab('dashboard');
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-surface-950 font-sans">
      {/* Persistent Left Sidebar */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        hasAudit={currentAudit !== null}
      />

      {/* Main Content Workspace */}
      <main className="flex-1 overflow-y-auto bg-surface-950">
        {activeTab === 'new-audit' && (
          <NewAudit onAuditComplete={handleAuditComplete} />
        )}

        {currentAudit && (
          <>
            {activeTab === 'dashboard' && (
              <Dashboard audit={currentAudit} onNavigateTab={setActiveTab} />
            )}
            {activeTab === 'performance' && (
              <Performance audit={currentAudit} />
            )}
            {activeTab === 'fairness' && (
              <FairnessAnalysis audit={currentAudit} />
            )}
            {activeTab === 'intersectional' && (
              <Intersectional audit={currentAudit} />
            )}
            {activeTab === 'explainability' && (
              <Explainability audit={currentAudit} />
            )}
            {activeTab === 'threshold' && (
              <ThresholdExplorer audit={currentAudit} />
            )}
            {activeTab === 'reports' && (
              <Reports audit={currentAudit} />
            )}
          </>
        )}
      </main>
    </div>
  );
}

export default App;
