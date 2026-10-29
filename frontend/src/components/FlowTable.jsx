import { useState, useMemo, useEffect, useRef } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { Search, Filter, Shield, ShieldAlert, ChevronLeft, ChevronRight, Table } from 'lucide-react'

gsap.registerPlugin(ScrollTrigger)

const PER_PAGE = 15

export default function FlowTable({ flows }) {
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('all')
  const [page,   setPage]   = useState(1)
  const sectionRef = useRef(null)

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.from('.ftable-panel', {
        opacity: 0, y: 28,
        duration: 0.65, ease: 'power3.out',
        scrollTrigger: { trigger: '.ftable-panel', start: 'top 82%' },
      })
    }, sectionRef.current)
    return () => ctx.revert()
  }, [])

  const filtered = useMemo(() => flows.filter(f => {
    if (filter === 'attack' && f.pred !== 1) return false
    if (filter === 'normal' && f.pred !== 0) return false
    if (search && !String(f.idx).includes(search.trim())) return false
    return true
  }), [flows, search, filter])

  const totalPages = Math.ceil(filtered.length / PER_PAGE)
  const cur        = Math.min(page, totalPages || 1)
  const start      = (cur - 1) * PER_PAGE
  const pageData   = filtered.slice(start, start + PER_PAGE)

  return (
    <section ref={sectionRef} className="relative py-16 px-6">
      <div className="max-w-6xl mx-auto">

        <div className="text-center mb-8">
          <div className="section-badge mx-auto">
            <Table className="w-3 h-3" />
            Flow Details
          </div>
          <h2 className="text-3xl font-extrabold text-white mb-2">Per-Flow Predictions</h2>
          <p className="text-cyber-muted text-sm">Confidence scores for every classified network flow</p>
        </div>

        <div
          className="ftable-panel rounded-2xl overflow-hidden"
          style={{ background: 'rgba(17,30,48,0.85)', border: '1px solid rgba(30,58,95,0.7)' }}
        >
          {/* Controls */}
          <div className="flex flex-wrap items-center gap-3 px-5 py-4"
               style={{ borderBottom: '1px solid rgba(30,58,95,0.5)' }}>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-cyber-muted" />
              <input
                type="text"
                placeholder="Search flow index…"
                value={search}
                onChange={e => { setSearch(e.target.value); setPage(1) }}
                className="pl-8 pr-3 py-2 rounded-lg text-sm placeholder-cyber-muted/40 focus:outline-none focus:border-cyber-accent transition-colors w-44"
                style={{ background: 'rgba(13,21,32,0.7)', border: '1px solid rgba(30,58,95,0.7)', color: '#e2e8f0' }}
              />
            </div>
            <div className="flex items-center gap-2">
              <Filter className="w-3.5 h-3.5 text-cyber-muted" />
              <select
                value={filter}
                onChange={e => { setFilter(e.target.value); setPage(1) }}
                className="py-2 px-3 rounded-lg text-sm focus:outline-none focus:border-cyber-accent cursor-pointer"
                style={{ background: 'rgba(13,21,32,0.7)', border: '1px solid rgba(30,58,95,0.7)', color: '#e2e8f0' }}
              >
                <option value="all">All flows</option>
                <option value="attack">Attacks only</option>
                <option value="normal">Normal only</option>
              </select>
            </div>
            <span className="ml-auto text-xs text-cyber-muted/70">
              {filtered.length.toLocaleString()} flows
            </span>
          </div>

          {/* Table */}
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(30,58,95,0.5)' }}>
                  {['Flow #', 'Prediction', 'P(Normal)', 'P(Attack)', 'Confidence'].map(h => (
                    <th key={h} className="text-left px-5 py-3 text-[11px] font-semibold uppercase tracking-wider text-cyber-muted/70">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {pageData.map(f => (
                  <tr
                    key={f.idx}
                    className="transition-colors"
                    style={{ borderBottom: '1px solid rgba(30,58,95,0.3)' }}
                    onMouseEnter={e => { e.currentTarget.style.background = 'rgba(30,58,95,0.15)' }}
                    onMouseLeave={e => { e.currentTarget.style.background = 'transparent' }}
                  >
                    <td className="px-5 py-3 font-mono text-xs text-cyber-muted">{f.idx}</td>
                    <td className="px-5 py-3">
                      {f.pred === 1 ? (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold"
                              style={{ background: 'rgba(248,113,113,0.1)', border: '1px solid rgba(248,113,113,0.25)', color: '#f87171' }}>
                          <ShieldAlert className="w-3 h-3" /> Attack
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold"
                              style={{ background: 'rgba(20,184,166,0.1)', border: '1px solid rgba(20,184,166,0.25)', color: '#2dd4bf' }}>
                          <Shield className="w-3 h-3" /> Normal
                        </span>
                      )}
                    </td>
                    <td className="px-5 py-3 font-mono text-xs text-cyber-text">{f.probNormal.toFixed(4)}</td>
                    <td className="px-5 py-3 font-mono text-xs text-cyber-text">{f.probAttack.toFixed(4)}</td>
                    <td className="px-5 py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(30,58,95,0.5)' }}>
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${f.confidence * 100}%`,
                              background: f.confidence >= 0.85 ? '#2dd4bf' : f.confidence >= 0.65 ? '#fbbf24' : '#f87171',
                            }}
                          />
                        </div>
                        <span className="font-mono text-[11px] text-cyber-muted">{(f.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between px-5 py-3.5"
               style={{ borderTop: '1px solid rgba(30,58,95,0.5)' }}>
            <span className="text-xs text-cyber-muted/60">
              {start + 1}–{Math.min(start + PER_PAGE, filtered.length)} of {filtered.length.toLocaleString()}
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={cur <= 1}
                className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs transition-colors disabled:opacity-30"
                style={{ background: 'rgba(13,21,32,0.6)', border: '1px solid rgba(30,58,95,0.6)', color: '#94a3b8' }}
              >
                <ChevronLeft className="w-3.5 h-3.5" /> Prev
              </button>
              <span className="px-2 text-xs text-cyber-muted">{cur} / {totalPages}</span>
              <button
                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                disabled={cur >= totalPages}
                className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs transition-colors disabled:opacity-30"
                style={{ background: 'rgba(13,21,32,0.6)', border: '1px solid rgba(30,58,95,0.6)', color: '#94a3b8' }}
              >
                Next <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
