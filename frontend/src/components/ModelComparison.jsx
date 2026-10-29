import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { Cpu, TreePine, GitCompare, FlaskConical } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts'
import { MODEL_COMPARISON, ABLATION_DATA } from '../data/sampleData'

gsap.registerPlugin(ScrollTrigger)

const METRICS = MODEL_COMPARISON.map(d => ({
  metric:          d.metric,
  GraphSAGE:       d.graphsage,
  'Random Forest': d.randomForest,
}))

const TOOLTIP_STYLE = {
  contentStyle: { background: '#111e30', border: '1px solid #1e3a5f', borderRadius: 10, fontSize: 12, color: '#e2e8f0' },
  cursor: { fill: 'rgba(59,130,246,0.05)' },
}

export default function ModelComparison() {
  const sectionRef = useRef(null)

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    let ctx
    try {
      ctx = gsap.context(() => {
        gsap.fromTo('.mc-heading',
          { opacity: 0, y: 24 },
          { opacity: 1, y: 0, duration: 0.6, ease: 'power3.out', immediateRender: false,
            scrollTrigger: { trigger: '.mc-heading', start: 'top 88%', once: true } })
        gsap.fromTo('.mc-card',
          { opacity: 0, y: 28 },
          { opacity: 1, y: 0, duration: 0.55, stagger: 0.1, ease: 'power3.out', immediateRender: false,
            scrollTrigger: { trigger: '.mc-cards', start: 'top 84%', once: true } })
        gsap.fromTo('.mc-chart',
          { opacity: 0, y: 28 },
          { opacity: 1, y: 0, duration: 0.6, ease: 'power3.out', immediateRender: false,
            scrollTrigger: { trigger: '.mc-chart', start: 'top 83%', once: true } })
      }, sectionRef.current)
      const t = setTimeout(() => ScrollTrigger.refresh(), 120)
      return () => { clearTimeout(t); ctx.revert() }
    } catch (e) { console.warn('[ModelComparison] GSAP error:', e); ctx?.revert() }
  }, [])

  const models = [
    {
      icon: Cpu, name: 'GraphSAGE', sub: '2 SAGEConv layers · 13,826 params · graph-aware',
      accent: '#3b82f6', key: 'graphsage',
    },
    {
      icon: TreePine, name: 'Random Forest', sub: '200 trees · same 42 features · no graph structure',
      accent: '#2dd4bf', key: 'randomForest',
    },
  ]

  return (
    <section ref={sectionRef} className="relative py-20 px-6">
      <div className="max-w-6xl mx-auto">

        {/* Heading */}
        <div className="mc-heading text-center mb-10">
          <div className="section-badge mx-auto">
            <GitCompare className="w-3 h-3" />
            Model Comparison
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white mb-2">Official Evaluation</h2>
          <p className="text-cyber-muted text-sm">
            Tested once on 82,332 held-out flows from UNSW-NB15 — not re-evaluated on uploaded data
          </p>
        </div>

        {/* Model cards */}
        <div className="mc-cards grid md:grid-cols-2 gap-5 mb-8">
          {models.map(m => (
            <div
              key={m.name}
              className="mc-card rounded-2xl p-5"
              style={{ background: 'rgba(17,30,48,0.85)', border: `1px solid ${m.accent}25` }}
            >
              <div className="flex items-center gap-3 mb-5">
                <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                     style={{ background: m.accent + '15', border: `1px solid ${m.accent}30` }}>
                  <m.icon className="w-5 h-5" style={{ color: m.accent }} />
                </div>
                <div>
                  <div className="text-base font-bold text-white">{m.name}</div>
                  <div className="text-[10px] text-cyber-muted">{m.sub}</div>
                </div>
              </div>
              <div className="grid grid-cols-4 gap-2">
                {MODEL_COMPARISON.map(d => (
                  <div key={d.metric} className="text-center p-2.5 rounded-xl"
                       style={{ background: m.accent + '08', border: `1px solid ${m.accent}18` }}>
                    <div className="text-lg font-bold font-mono" style={{ color: m.accent }}>
                      {d[m.key].toFixed(4)}
                    </div>
                    <div className="text-[9px] text-cyber-muted mt-0.5">{d.metric}</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Bar chart */}
        <div
          className="mc-chart rounded-2xl p-5 mb-6"
          style={{ background: 'rgba(17,30,48,0.85)', border: '1px solid rgba(30,58,95,0.7)' }}
        >
          <h3 className="text-sm font-semibold text-white mb-5 flex items-center gap-2">
            <GitCompare className="w-4 h-4 text-cyber-accent" />
            Metric Comparison (official test set, 82,332 flows)
          </h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={METRICS} margin={{ top: 4, right: 4, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,58,95,0.5)" />
              <XAxis dataKey="metric" stroke="#64748b" fontSize={11} axisLine={false} tickLine={false} />
              <YAxis domain={[0.7, 1.0]} stroke="#64748b" fontSize={11} axisLine={false} tickLine={false} />
              <Tooltip {...TOOLTIP_STYLE} />
              <Legend iconType="circle" iconSize={7} wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="GraphSAGE"      fill="#3b82f6" radius={[4, 4, 0, 0]} maxBarSize={36} />
              <Bar dataKey="Random Forest"  fill="#2dd4bf" radius={[4, 4, 0, 0]} maxBarSize={36} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Ablation */}
        <div
          className="mc-chart rounded-2xl p-5"
          style={{ background: 'rgba(17,30,48,0.85)', border: '1px solid rgba(30,58,95,0.7)' }}
        >
          <h3 className="text-sm font-semibold text-white mb-1 flex items-center gap-2">
            <FlaskConical className="w-4 h-4 text-amber-400" />
            Ablation Study — Graph Structure Contribution
          </h3>
          <p className="text-[11px] text-cyber-muted/70 mb-5">
            A: k-NN graph · B: no edges (node features only) · C: degree-preserving random rewiring
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={ABLATION_DATA} margin={{ top: 4, right: 4, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,58,95,0.5)" />
              <XAxis dataKey="experiment" stroke="#64748b" fontSize={10} axisLine={false} tickLine={false} />
              <YAxis domain={[0.75, 1.0]} stroke="#64748b" fontSize={11} axisLine={false} tickLine={false} />
              <Tooltip {...TOOLTIP_STYLE} />
              <Bar dataKey="f1" name="F1 Score" fill="#fbbf24" radius={[4, 4, 0, 0]} maxBarSize={50} />
            </BarChart>
          </ResponsiveContainer>
          <div className="grid md:grid-cols-3 gap-3 mt-4">
            {ABLATION_DATA.map(e => (
              <div key={e.experiment} className="px-4 py-3 rounded-xl text-xs"
                   style={{ background: 'rgba(13,21,32,0.6)', border: '1px solid rgba(30,58,95,0.5)' }}>
                <p className="font-semibold text-white mb-1">{e.experiment}</p>
                <p className="text-cyber-muted">
                  F1: <span className="font-mono text-amber-400">{e.f1}</span>
                  {' · '}Acc: <span className="font-mono text-cyber-text">{e.accuracy}</span>
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
