import React, { useState, useRef, useEffect } from 'react';
import { EvidenceGraph, type EvidenceGraphData } from './EvidenceGraph';
import './App.css';

// Types for our stream events
type PipelineStep = 'idle' | 'planner' | 'query_rewriting' | 'retrieval' | 'reader' | 'research_controller' | 'writer' | 'verification_and_evaluation' | 'diagnosis' | 'complete' | 'error';

interface LogEntry {
  id: string;
  time: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
}

function App() {
  const [query, setQuery] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState<PipelineStep>('idle');
  const [answer, setAnswer] = useState('');
  const [evidenceGraph, setEvidenceGraph] = useState<EvidenceGraphData | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);

  const addLog = (message: string, type: LogEntry['type'] = 'info') => {
    const now = new Date();
    const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}.${now.getMilliseconds().toString().padStart(3, '0')}`;
    setLogs(prev => [...prev, { id: Math.random().toString(36).substr(2, 9), time: timeStr, message, type }]);
  };

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isProcessing) return;

    setIsProcessing(true);
    setCurrentStep('planner');
    setAnswer('');
    setEvidenceGraph(null);
    setLogs([]);
    addLog(`Initiating query: "${query}"`);

    try {
      // Setup SSE connection
      const response = await fetch('http://localhost:8000/api/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      if (!response.body) throw new Error('ReadableStream not supported');
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      let done = false;
      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.substring(6));
                handleStreamEvent(data);
              } catch (e) {
                console.error("Failed to parse SSE line", line);
              }
            }
          }
        }
      }
    } catch (error: any) {
      addLog(`Error: ${error.message}`, 'error');
      setCurrentStep('error');
      setIsProcessing(false);
    }
  };

  const handleStreamEvent = (data: any) => {
    if (data.type === 'step_completed') {
      setCurrentStep(data.step);
      if (data.step === 'reader' && data.details.evidence_preview) {
         addLog(`[Reader] Extracted ${data.details.evidence_spans} spans of evidence.`, 'info');
      } else if (data.step === 'retrieval') {
         addLog(`[Retrieval] Retrieved ${data.details.documents_retrieved} documents. Diversity: ${data.details.document_diversity.toFixed(2)}`, 'info');
      } else if (data.step === 'diagnosis') {
         addLog(`[Diagnosis] ${data.details.category} - ${data.details.reasoning}`, data.details.category === 'satisfactory' ? 'success' : 'warning');
      } else if (data.step === 'writer') {
         addLog(`[Writer] Generated initial draft. Length: ${data.details.answer_length}`, 'info');
      } else {
         addLog(`[${data.step}] Completed phase.`, 'info');
      }
    } else if (data.type === 'retrying') {
      addLog(`[Remediation] Loop ${data.details.retry_count}. Applying directive for: ${data.details.diagnosis}`, 'warning');
      setCurrentStep('planner'); // loops back
    } else if (data.type === 'query_complete') {
      setAnswer(data.response.answer);
      setEvidenceGraph(data.response.provenance?.evidence_graph || null);
      setCurrentStep('complete');
      addLog(`[Complete] Query finalized. Faithfulness: ${data.response.metrics.faithfulness}`, 'success');
      setIsProcessing(false);
    } else if (data.type === 'query_error') {
      addLog(`[Fatal] ${data.error}`, 'error');
      setCurrentStep('error');
      setIsProcessing(false);
    }
  };

  const PipelineStepView = ({ stepId, label, current }: { stepId: PipelineStep, label: string, current: PipelineStep }) => {
    const steps: PipelineStep[] = ['idle', 'planner', 'query_rewriting', 'retrieval', 'reader', 'research_controller', 'writer', 'verification_and_evaluation', 'diagnosis', 'complete'];
    
    // Quick and dirty logic: if it's before current in array, it's complete. If it's current, it's active.
    const currentIndex = steps.indexOf(current === 'error' ? 'idle' : current);
    const thisIndex = steps.indexOf(stepId);
    
    let stateClass = '';
    if (current === stepId) stateClass = 'active';
    else if (thisIndex < currentIndex && thisIndex > 0) stateClass = 'complete';

    return (
      <div className={`pipeline-step ${stateClass}`}>
        <div className="step-icon">
          {stateClass === 'complete' ? '✓' : stateClass === 'active' ? '●' : '○'}
        </div>
        <span>{label}</span>
      </div>
    );
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Meta-RAG</h1>
      </header>

      <div className="main-content">
        <aside className="sidebar">
          <div className="section-title">Control Panel</div>
          <div className="sidebar-content">
            <form onSubmit={handleSubmit} className="query-form">
              <label htmlFor="query">Research Query</label>
              <textarea
                id="query"
                className="query-input"
                placeholder="Enter a complex multi-hop question..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                disabled={isProcessing}
              />
              <button type="submit" className="btn-primary" disabled={isProcessing || !query.trim()}>
                {isProcessing ? 'Processing...' : 'Run Analysis'}
              </button>
            </form>

            <div className="status-indicator" style={{ marginTop: '1rem' }}>
              <div className={`status-dot ${isProcessing ? 'active' : ''}`}></div>
              {isProcessing ? 'System Active' : 'System Standby'}
            </div>
            
            <div className="section-title" style={{ marginTop: 'auto', marginInline: 'calc(-1 * var(--gap-md))' }}>Diagnostics Feed</div>
            <div className="terminal-box">
              {logs.map((log) => (
                <div key={log.id} className="terminal-line" style={{ 
                  color: log.type === 'error' ? 'var(--accent-red)' : 
                         log.type === 'warning' ? 'var(--accent-orange)' : 
                         log.type === 'success' ? '#fff' : 'var(--accent-green)' 
                }}>
                  <span className="terminal-meta">[{log.time}]</span>
                  <span>{log.message}</span>
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </div>
        </aside>

        <main className="center-panel">
          <div className="section-title">Output & Provenance</div>
          <div className="answer-area">
            {!answer && !isProcessing && (
               <div className="answer-placeholder">Awaiting Input...</div>
            )}
            {!answer && isProcessing && (
               <div className="answer-placeholder" style={{ color: 'var(--accent-cyan)' }}>
                 <span className="active-pulse" style={{ padding: '0.5rem', border: '1px solid', borderRadius: '4px' }}>
                   Synthesizing...
                 </span>
               </div>
            )}
            {answer && (
               <div className="markdown-content">
                 {answer.split('\n').map((line, i) => <p key={i}>{line}</p>)}
               </div>
            )}
          </div>
        </main>

        <aside className="right-panel">
          <div className="section-title">Metacognitive Pipeline</div>
          <div className="pipeline">
            <PipelineStepView stepId="planner" label="1. Intent Planning" current={currentStep} />
            <PipelineStepView stepId="query_rewriting" label="2. Query Reformulation" current={currentStep} />
            <PipelineStepView stepId="retrieval" label="3. Hybrid Retrieval" current={currentStep} />
            <PipelineStepView stepId="reader" label="4. Evidence Extraction" current={currentStep} />
            <PipelineStepView stepId="research_controller" label="5. Controller Routing" current={currentStep} />
            <PipelineStepView stepId="writer" label="6. Initial Generation" current={currentStep} />
            <PipelineStepView stepId="verification_and_evaluation" label="7. Fact Verification" current={currentStep} />
            <PipelineStepView stepId="diagnosis" label="8. Metacognitive Diagnosis" current={currentStep} />
          </div>
          
          <div className="section-title" style={{ marginTop: '1rem' }}>Evidence Graph Topology</div>
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1rem' }}>
            <EvidenceGraph data={evidenceGraph} />
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;
