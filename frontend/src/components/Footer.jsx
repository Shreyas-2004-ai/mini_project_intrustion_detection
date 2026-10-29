import { Shield } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="relative border-t py-12 px-6" style={{ borderColor: 'rgba(30,58,95,0.5)' }}>
      <div className="max-w-6xl mx-auto">
        <div className="grid md:grid-cols-3 gap-8 mb-8">

          <div>
            <div className="flex items-center gap-2.5 mb-3">
              <div className="w-8 h-8 rounded-lg flex items-center justify-center"
                   style={{ background: 'rgba(59,130,246,0.12)', border: '1px solid rgba(59,130,246,0.3)' }}>
                <Shield className="w-4 h-4 text-cyber-accent-light" />
              </div>
              <span className="font-semibold text-white text-sm">GNN Intrusion Detection</span>
            </div>
            <p className="text-xs text-cyber-muted leading-relaxed">
              Graph-based network intrusion detection using GraphSAGE on UNSW-NB15 flow-similarity graphs.
            </p>
          </div>

          <div>
            <h4 className="text-xs font-semibold text-white mb-3 uppercase tracking-wider">Tech Stack</h4>
            <ul className="space-y-1.5 text-xs text-cyber-muted">
              <li>React + Vite + Tailwind CSS + GSAP</li>
              <li>PyTorch Geometric · GraphSAGE</li>
              <li>scikit-learn · Random Forest</li>
              <li>React Flow · Recharts · Flask</li>
            </ul>
          </div>

          <div>
            <h4 className="text-xs font-semibold text-white mb-3 uppercase tracking-wider">Key Results</h4>
            <ul className="space-y-1.5 text-xs text-cyber-muted">
              <li>GraphSAGE F1: <span className="font-mono text-cyber-accent-light">0.8994</span></li>
              <li>Random Forest F1: <span className="font-mono text-cyber-teal">0.8941</span></li>
              <li>Recall: <span className="font-mono text-white">0.9601</span></li>
              <li>Test set: <span className="font-mono text-white">82,332 flows</span></li>
            </ul>
          </div>
        </div>

        <div className="h-px bg-cyber-border/30 mb-5" />
        <div className="flex flex-wrap items-center justify-between gap-3 text-[11px] text-cyber-muted/50">
          <span>© 2026 GNN Intrusion Detection · Master's Mini Project · UNSW-NB15</span>
          <span>Flow-similarity graph · not IP communication topology</span>
        </div>
      </div>
    </footer>
  )
}
