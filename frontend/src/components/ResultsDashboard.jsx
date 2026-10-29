import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import {
  Activity, ShieldCheck, ShieldAlert, Percent,
  Network, GitMerge, TrendingUp,
} from 'lucide-react'
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import { useAnimatedCounter } from '../hooks/useGsap'

gsap.registerPlugin(ScrollTrigger)

/* ── Individual stat card ─────────────────────────────────────────────────── */
function StatCard({ icon: Icon, value, label, color, accent, suffix = '', decimals = 0, note }) {
  const counterRef = useAnimatedCounter(value, { suffix, decimals })
  return (
    <div
      className="stat-card flex flex-col gap-2 group"
      style={{ borderColor: accent + '30' }}
      onMouseEnter={e => { e.currentTarget.style.borderColor = accent + '60' }}
      onMouseLeave={e => { e.currentTarget.style.borderColor = accent + '30' }}
    >
      <div className="flex items-center justify-between">
        <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
             style={{ background: accent + '15', border: `1px solid ${accent}30` }}>
          <Icon className="w-4 h-4" style={{ color: accent }} />
        </div>
        {note && (
          <span className="text-[9px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full"
                style={{ background: accent + '12', color: accent }}>
            {note}
          </span>
        )}
      </div>
      <div>
        <div ref={counterRef} className="text-2xl font-bold font-mono"
             style={{ color: accent }}>
          0{suffix}
        </div>
        <div className="text-xs text-cyber-muted mt-0.5">{label}</div>
      </div>
    </div>
  )
}

/* ── Custom donut label ───────────────────────────────────────────────────── */
const renderCustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, name }) => {
  if (percent < 0.05) return null
  const rad = innerRadius + (outerRadius - innerRadius) * 0.55
  const x = cx + rad * Math.cos(-midAngle * (Math.PI / 180))
  const y = cy + rad * Math.sin(-midAngle * (Math.PI / 180))
  return (
    <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central" fontSize={11} fontWeight={600}>
      {(percent * 100).toFixed(0)}%
    </text>
  )
}

/* ── Main component ───────────────────────────────────────────────────────── */
export default function ResultsDashboard({ data, isSample }) {
  const sectionRef = useRef(null)
  const { summary, graph } = data
  const numNodes = graph?.numNodes ?? 0
  const numEdges = graph?.numEdges ?? 0
  const avgDeg   = numNodes > 0 ? (numEdges / numNodes).toFixed(1) : '—'

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    let ctx
    try {
      ctx = gsap.context(() => {
        gsap.fromTo('.rdb-heading',
          { opacity: 0, y: 24 },
          { opacity: 1, y: 0, duration: 0.6, ease: 'power3.out', immediateRender: false,
            scrollTrigger: { trigger: '.rdb-heading', start: 'top 88%', once: true } })
        gsap.fromTo('.rdb-card',
          { opacity: 0, y: 28 },
          { opacity: 1, y: 0, duration: 0.5, stagger: 0.07, ease: 'power3.out', immediateRender: false,
            scrollTrigger: { trigger: '.rdb-stats', start: 'top 85%', once: true } })
        gsap.fromTo('.rdb-chart',
          { opacity: 0, y: 28 },
          { opacity: 1, y: 0, duration: 0.6, ease: 'power3.out', immediateRender: false,
            scrollTrigger: { trigger: '.rdb-chart', start: 'top 83%', once: true } })
      }, sectionRef.current)
      const t = setTimeout(() => ScrollTrigger.refresh(), 120)
      return () => { clearTimeout(t); ctx.revert() }
    } catch (e) { console.warn('[ResultsDashboard] GSAP error:', e); ctx?.revert() }
  }, [data])

  const distData = [
    { name: 'Normal',  value: summary.normal_flows,  color: '#14b8a6' },
    { name: 'Attack',  value: summary.attack_flows,   color: '#f87171' },
  ]

  const STAT_CARDS = [
    { icon: Activity,    value: summary.total_flows,   label: 'Total Flows',     accent: '#60a5fa', suffix: '', decimals: 0 },
    { icon: ShieldCheck, value: summary.normal_flows,  label: 'Normal Flows',    accent: '#2dd4bf', suffix: '', decimals: 0 },
    { icon: ShieldAlert, value: summary.attack_flows,  label: 'Detected Attacks',accent: '#f87171', suffix: '', decimals: 0, note: 'Threats' },
    { icon: Percent,     value: summary.attack_pct,    label: 'Attack Rate',     accent: '#fbbf24', suffix: '%', decimals: 1 },
    { icon: Network,     value: numNodes,               label: 'Graph Nodes',     accent: '#818cf8', suffix: '', decimals: 0 },
    { icon: GitMerge,    value: numEdges,               label: 'Graph Edges',     accent: '#38bdf8', suffix: '', decimals: 0 },
    { icon: TrendingUp,  value: parseFloat(avgDeg) || 0,label: 'Avg. Degree',    accent: '#a78bfa', suffix: '', decimals: 1 },
  ]

  return (
    <section ref={sectionRef} id="results" className="relative py-20 px-6">
      <div className="max-w-6xl mx-auto">

        {/* Heading */}
        <div className="rdb-heading text-center mb-10">
          <div className="section-badge mx-auto">
            <Activity className="w-3 h-3" />
            {isSample ? 'Sample Preview' : 'Analysis Results'}
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white mb-2">
            Detection Results
          </h2>
          <p className="text-cyber-muted text-sm">
            {isSample
              ? <span className="text-amber-400 font-medium">⚠ Sample data — not actual model predictions</span>
              : `${summary.total_flows.toLocaleString()} flows analysed by GraphSAGE`}
          </p>
        </div>

        {/* Stat cards */}
        <div className="rdb-stats grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-7 gap-3 mb-8">
          {STAT_CARDS.map((c, i) => (
            <div key={i} className="rdb-card">
              <StatCard {...c} />
            </div>
          ))}
        </div>

        {/* Charts row */}
        <div className="rdb-chart grid md:grid-cols-5 gap-6">

          {/* Donut */}
          <div
            className="md:col-span-2 rounded-2xl p-5"
            style={{ background: 'rgba(17,30,48,0.85)', border: '1px solid rgba(30,58,95,0.7)' }}
          >
            <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
              <Percent className="w-4 h-4 text-cyber-accent" />
              Flow Distribution
            </h3>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={distData}
                  cx="50%" cy="50%"
                  innerRadius={58} outerRadius={88}
                  paddingAngle={3}
                  dataKey="value"
                  labelLine={false}
                  label={renderCustomLabel}
                >
                  {distData.map((d, i) => (
                    <Cell key={i} fill={d.color} stroke="rgba(6,11,20,0.6)" strokeWidth={2} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ background: '#111e30', border: '1px solid #1e3a5f', borderRadius: 8, fontSize: 12 }}
                  formatter={(v, name) => [`${v.toLocaleString()} flows`, name]}
                />
                <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
              </PieChart>
            </ResponsiveContainer>

            {/* Legend numbers */}
            <div className="grid grid-cols-2 gap-3 mt-2">
              {distData.map(d => (
                <div key={d.name} className="text-center">
                  <div className="text-lg font-bold font-mono" style={{ color: d.color }}>
                    {d.value.toLocaleString()}
                  </div>
                  <div className="text-[10px] text-cyber-muted">{d.name}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Summary table */}
          <div
            className="md:col-span-3 rounded-2xl p-5"
            style={{ background: 'rgba(17,30,48,0.85)', border: '1px solid rgba(30,58,95,0.7)' }}
          >
            <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
              <Network className="w-4 h-4 text-cyber-accent" />
              Analysis Summary
            </h3>
            <div className="space-y-3">
              {[
                { label: 'Total flows processed',  value: summary.total_flows.toLocaleString(), accent: '#60a5fa' },
                { label: 'Normal (benign) flows',   value: `${summary.normal_flows.toLocaleString()} (${(100 - summary.attack_pct).toFixed(1)}%)`, accent: '#2dd4bf' },
                { label: 'Attack flows detected',   value: `${summary.attack_flows.toLocaleString()} (${summary.attack_pct.toFixed(1)}%)`, accent: '#f87171' },
                { label: 'Graph nodes',             value: numNodes.toLocaleString(), accent: '#818cf8' },
                { label: 'Graph edges (directed)',  value: numEdges.toLocaleString(), accent: '#38bdf8' },
                { label: 'Average out-degree',      value: avgDeg, accent: '#a78bfa' },
                { label: 'Edge construction',       value: 'k=5 cosine similarity', accent: '#fbbf24' },
              ].map(({ label, value, accent }) => (
                <div key={label} className="flex items-center justify-between py-2 border-b border-cyber-border/30">
                  <span className="text-xs text-cyber-muted">{label}</span>
                  <span className="text-xs font-semibold font-mono" style={{ color: accent }}>{value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
