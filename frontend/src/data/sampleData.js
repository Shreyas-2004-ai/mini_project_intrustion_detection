// ═══════════════════════════════════════════════════════════════════════════
// sampleData.js
// ─────────────
// Sample data for UI preview and demonstration ONLY.
// These values are fabricated for layout/visual testing.
// They do NOT represent actual model predictions.
//
// When the backend is connected (Flask/FastAPI calling predict_csv()),
// this data will be replaced by real inference output.
// ═══════════════════════════════════════════════════════════════════════════

export const SAMPLE_SUMMARY = {
  total_flows:   250,
  normal_flows:  187,
  attack_flows:  63,
  attack_pct:    25.2,
};

// Generate 250 sample predictions with probabilities
function genSampleFlows() {
  const flows = [];
  // Seed-like pattern for reproducible-looking demo data
  const attackIndices = new Set([2, 4, 7, 10, 14, 17, 21, 26, 29, 33, 38, 42, 44, 49, 53, 57, 61, 65, 69, 73, 77, 82, 86, 90, 94, 98, 103, 107, 111, 115, 119, 124, 128, 132, 136, 141, 145, 150, 154, 158, 163, 167, 171, 176, 180, 184, 188, 192, 197, 201, 205, 209, 214, 218, 222, 226, 231, 235, 239, 243, 247]);

  for (let i = 0; i < 250; i++) {
    const isAttack = attackIndices.has(i);
    // Generate plausible-looking probabilities
    const baseProb = isAttack ? 0.85 + Math.random() * 0.13 : 0.85 + Math.random() * 0.13;
    const probAttack = isAttack ? baseProb : (1 - baseProb);
    const probNormal = 1 - probAttack;
    flows.push({
      idx: i,
      pred: isAttack ? 1 : 0,
      probNormal: Math.round(probNormal * 10000) / 10000,
      probAttack: Math.round(probAttack * 10000) / 10000,
      confidence: Math.round(Math.max(probNormal, probAttack) * 10000) / 10000,
    });
  }
  return flows;
}

export const SAMPLE_FLOWS = genSampleFlows();

// ── Model comparison data (actual project results) ──────────────────────────
export const MODEL_COMPARISON = [
  { metric: 'Accuracy',  graphsage: 0.8817, randomForest: 0.8711 },
  { metric: 'Precision', graphsage: 0.8459, randomForest: 0.8166 },
  { metric: 'Recall',    graphsage: 0.9601, randomForest: 0.9878 },
  { metric: 'F1 Score',  graphsage: 0.8994, randomForest: 0.8941 },
];

// ── Ablation study data (actual project results) ─────────────────────────────
export const ABLATION_DATA = [
  { experiment: 'A — k-NN Graph',  accuracy: 0.8813, precision: 0.8452, recall: 0.9604, f1: 0.8991 },
  { experiment: 'B — No Edges',    accuracy: 0.8398, precision: 0.7910, recall: 0.9636, f1: 0.8688 },
  { experiment: 'C — Random',     accuracy: 0.9487, precision: 0.9318, recall: 0.9785, f1: 0.9546 },
];

// ── Graph nodes for React Flow visualization ──────────────────────────────────
// Sample graph: 20 nodes, k=5 edges, mix of normal and attack
function genGraphData() {
  const nodes = [];
  const edges = [];
  const labelMap = [0,0,1,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0];

  // Position nodes in a force-layout-like arrangement
  for (let i = 0; i < 20; i++) {
    const angle = (i / 20) * Math.PI * 2;
    const radius = 150 + (i % 3) * 40;
    const x = 250 + Math.cos(angle) * radius + (Math.random() - 0.5) * 60;
    const y = 200 + Math.sin(angle) * radius + (Math.random() - 0.5) * 60;
    const isAttack = labelMap[i] === 1;
    nodes.push({
      id: String(i),
      type: 'default',
      position: { x, y },
      data: { label: `Flow ${i}` },
      style: {
        width: 42,
        height: 42,
        borderRadius: '50%',
        background: isAttack ? 'rgba(248, 113, 113, 0.15)' : 'rgba(74, 222, 128, 0.15)',
        border: `2px solid ${isAttack ? '#f87171' : '#4ade80'}`,
        color: isAttack ? '#f87171' : '#4ade80',
        fontSize: 10,
        fontWeight: 600,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      },
    });
  }

  // k=5 edges per node (simulated)
  const edgePairs = [
    [0,1],[0,2],[0,5],[0,9],[0,10],
    [1,3],[1,6],[1,0],[1,7],[1,11],
    [2,4],[2,8],[2,12],[2,0],[2,5],
    [3,1],[3,6],[3,13],[3,7],[3,14],
    [4,2],[4,9],[4,15],[4,10],[4,16],
    [5,0],[5,2],[5,11],[5,17],[5,6],
    [6,1],[6,3],[6,18],[6,12],[6,7],
    [7,1],[7,3],[7,13],[7,8],[7,19],
    [8,2],[8,4],[8,14],[8,9],[8,15],
    [9,0],[9,4],[9,16],[9,10],[9,19],
    [10,0],[10,4],[10,17],[10,11],[10,5],
    [11,1],[11,5],[11,18],[11,12],[11,6],
    [12,2],[12,6],[12,19],[12,13],[12,7],
    [13,3],[13,7],[13,15],[13,14],[13,8],
    [14,4],[14,8],[14,16],[14,15],[14,9],
    [15,4],[15,9],[15,13],[15,17],[15,10],
    [16,4],[16,9],[16,18],[16,17],[16,11],
    [17,5],[17,10],[17,15],[17,19],[17,0],
    [18,6],[18,11],[18,16],[18,19],[18,1],
    [19,7],[19,12],[19,17],[19,18],[19,3],
  ];

  edgePairs.forEach(([src, dst], i) => {
    edges.push({
      id: `e${i}`,
      source: String(src),
      target: String(dst),
      style: { stroke: '#3a4a6e', strokeWidth: 1 },
      type: 'default',
      animated: false,
    });
  });

  return { nodes, edges };
}

export const SAMPLE_GRAPH = genGraphData();
