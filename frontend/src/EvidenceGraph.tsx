import React from 'react';

export interface EvidenceGraphLink {
  claim_index: number;
  claim: string;
  supporting_docs: number[];
  confidence: number;
}

export interface EvidenceGraphData {
  total_claims: number;
  supported_claims: number;
  coverage_ratio: number;
  links: EvidenceGraphLink[];
}

interface Props {
  data: EvidenceGraphData | null;
}

export const EvidenceGraph: React.FC<Props> = ({ data }) => {
  if (!data || data.total_claims === 0) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
        [Awaiting Graph Data]
      </div>
    );
  }

  const { links } = data;
  
  // Bipartite Layout: Claims on Left, Docs on Right
  const claims = links;
  
  // Extract unique docs
  const docSet = new Set<number>();
  links.forEach(l => l.supporting_docs.forEach(d => docSet.add(d)));
  const docs = Array.from(docSet).sort((a,b) => a - b);

  const graphHeight = 300;
  const graphWidth = 360;
  const padding = 20;

  const claimX = padding;
  const docX = graphWidth - padding - 40;

  const getClaimY = (index: number) => padding + (index * ((graphHeight - 2 * padding) / Math.max(1, claims.length - 1)));
  const getDocY = (index: number) => padding + (index * ((graphHeight - 2 * padding) / Math.max(1, docs.length - 1)));

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: '300px', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0 var(--gap-md)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
        <span>Claims ({data.total_claims})</span>
        <span>Documents ({docs.length})</span>
      </div>
      
      <svg width="100%" height={graphHeight} style={{ overflow: 'visible' }}>
        {/* Render Edges */}
        {claims.map((claim, cIdx) => (
          claim.supporting_docs.map(docId => {
            const dIdx = docs.indexOf(docId);
            const y1 = getClaimY(cIdx);
            const y2 = getDocY(dIdx);
            
            return (
              <path
                key={`edge-${cIdx}-${dIdx}`}
                d={`M ${claimX} ${y1} C ${claimX + 100} ${y1}, ${docX - 100} ${y2}, ${docX} ${y2}`}
                fill="none"
                stroke="var(--accent-cyan)"
                strokeWidth={1.5}
                opacity={0.3}
                className="graph-edge"
              />
            );
          })
        ))}

        {/* Render Claims (Left Nodes) */}
        {claims.map((_, cIdx) => {
          const y = getClaimY(cIdx);
          return (
            <g key={`claim-${cIdx}`}>
              <circle cx={claimX} cy={y} r={6} fill="var(--bg-base)" stroke="var(--accent-cyan)" strokeWidth={2} />
            </g>
          );
        })}

        {/* Render Docs (Right Nodes) */}
        {docs.map((docId, dIdx) => {
          const y = getDocY(dIdx);
          return (
            <g key={`doc-${dIdx}`}>
              <rect x={docX - 6} y={y - 6} width={12} height={12} fill="var(--bg-base)" stroke="var(--accent-green)" strokeWidth={2} />
              <text x={docX + 12} y={y + 4} fill="var(--text-primary)" fontSize="10" fontFamily="var(--font-mono)">
                D{docId}
              </text>
            </g>
          );
        })}
      </svg>
      
      <div style={{ display: 'flex', justifyContent: 'space-around', margin: 'var(--gap-sm) 0', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>
        <span style={{ color: 'var(--accent-cyan)' }}>Coverage: {(data.coverage_ratio * 100).toFixed(0)}%</span>
        <span style={{ color: 'var(--accent-green)' }}>Supported: {data.supported_claims} / {data.total_claims}</span>
      </div>
    </div>
  );
};
