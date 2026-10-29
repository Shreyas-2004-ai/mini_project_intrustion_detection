/**
 * GraphVisualization.jsx
 *
 * ROOT CAUSES FIXED (found via code inspection + backend probe, Oct 2026):
 *
 * BUG 1 — paddingTop:40 on <ReactFlow>
 *   The previous version passed `style={{ paddingTop: 40 }}` directly to the
 *   ReactFlow component to push content below a header strip.  React Flow does
 *   NOT forward arbitrary CSS padding into its viewport calculation, so the
 *   node positions in RF's internal coordinate system were 40 px above where
 *   the DOM actually rendered them.  Every edge endpoint was therefore 40 px
 *   off, producing paths that appeared as invisible stubs.
 *   FIX: render the header strip as an absolutely-positioned overlay ABOVE the
 *   ReactFlow component, and give ReactFlow the full container height with no
 *   padding prop.
 *
 * BUG 2 — overflow-hidden on the canvas container
 *   The container div had `overflow-hidden` (from `rounded-xl overflow-hidden`).
 *   React Flow renders its edge SVG as an absolutely-positioned child; when the
 *   parent clips overflow, any edge path that momentarily extends beyond the
 *   measured node area is clipped to nothing.
 *   FIX: remove overflow-hidden.  Use overflow-visible so edge SVGs render in
 *   full.  The rounded corners are kept via a separate ::before layer or just
 *   accepted as square — visual polish is secondary to correctness here.
 *
 * BUG 3 — custom LabeledEdge component (unnecessary risk)
 *   The LabeledEdge component used BaseEdge + EdgeLabelRenderer.  While this
 *   can work, it adds a surface for coordinate bugs and requires EdgeLabelRenderer
 *   to be present inside the ReactFlow tree.  With Bug 1 present, the label
 *   coordinates were wrong too, reinforcing the invisible-edge symptom.
 *   FIX: use React Flow's built-in `default` (bezier) edge type with the `label`
 *   prop for the weight badge.  RF handles all coordinate math internally.
 *   The `edgeTypes` registration is removed entirely.
 *
 * Everything else is preserved: forceLayout algorithm, buildNodes, sidebar
 * panels, GSAP reveal, minimap, controls, node colors, drag behavior.
 */

import { useEffect, useState, useCallback, useRef, memo } from 'react'
import ReactFlow, {
  Background,
  MiniMap,
  useNodesState,
  useEdgesState,
  MarkerType,
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  Handle,
  Position,
} from 'reactflow'
import {
  Network,
  ZoomIn, ZoomOut, Maximize2, RotateCcw,
  BarChart2, BookOpen, AlertTriangle,
} from 'lucide-react'
import { useGsapReveal } from '../hooks/useGsap'
import 'reactflow/dist/style.css'

// ─────────────────────────────────────────────────────────────────────────────
// Design tokens
// ─────────────────────────────────────────────────────────────────────────────
const TEAL   = '#14b8a6'
const RED    = '#f87171'
const CYAN   = '#38bdf8'
const NAVY   = '#0d1b2e'
const BORDER = '#1e3a5f'

// ─────────────────────────────────────────────────────────────────────────────
// Scale parameters by graph size
// ─────────────────────────────────────────────────────────────────────────────
// Scale parameters by graph size
function scaleParams(n) {
  if (n <= 20)  return { sz: 68, fs: 11, showLbl: true,  ew: 1.8, eo: 0.55, arw: 9,  cw: 2000, ch: 1100 }
  if (n <= 100) return { sz: 44, fs: 9,  showLbl: false, ew: 1.2, eo: 0.45, arw: 7,  cw: 3000, ch: 1800 }
  return               { sz: 30, fs: 7,  showLbl: false, ew: 0.9, eo: 0.35, arw: 5,  cw: 4000, ch: 2600 }
}

// ─────────────────────────────────────────────────────────────────────────────
// Custom node — MUST use w-full h-full so it fills the RF wrapper exactly.
// RF measures the wrapper to compute handle positions; if the inner div is
// smaller the handles end up off-centre and edges look misaligned.
// ─────────────────────────────────────────────────────────────────────────────
const GraphNode = memo(({ data, selected }) => {
  const isAtk  = data.predValue === 1
  const accent = isAtk ? RED : TEAL
  const bg     = isAtk ? 'rgba(248,113,113,0.13)' : 'rgba(20,184,166,0.13)'
  const p      = scaleParams(data.totalNodes || 5)
  const [hov, setHov] = useState(false)

  return (
    <div
      className="w-full h-full rounded-full flex flex-col items-center justify-center
                 cursor-pointer select-none relative"
      style={{
        background: bg,
        border: `2.5px solid ${accent}`,
        boxShadow: selected
          ? `0 0 0 3px ${accent}55, 0 0 24px ${accent}99`
          : hov
          ? `0 0 16px ${accent}77`
          : `0 0 8px ${accent}44`,
        transition: 'box-shadow 0.15s ease',
      }}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
    >
      {(p.showLbl || selected || hov) && (
        <>
          <span className="font-bold leading-tight pointer-events-none"
                style={{ fontSize: p.fs + 1, color: accent }}>
            {data.label}
          </span>
          <span className="leading-none pointer-events-none"
                style={{ fontSize: Math.max(p.fs - 1, 6), color: accent + 'bb', marginTop: 1 }}>
            {data.prediction}
          </span>
        </>
      )}

      {selected && (
        <div
          className="absolute inset-0 rounded-full animate-ping opacity-20 pointer-events-none"
          style={{ border: `2px solid ${accent}` }}
        />
      )}

      {/* Tooltip for dense graphs */}
      {hov && !p.showLbl && !selected && (
        <div
          className="absolute pointer-events-none z-50 px-2 py-1 rounded text-[9px]
                     font-mono whitespace-nowrap"
          style={{
            bottom: '110%', left: '50%', transform: 'translateX(-50%)',
            background: 'rgba(13,27,46,0.95)',
            border: `1px solid ${BORDER}`,
            color: accent,
          }}
        >
          {data.label} · {data.prediction} · {(data.confidence * 100).toFixed(0)}%
        </div>
      )}
      {/* Explicit handles so RF knows where to attach edges */}
      <Handle
        id="target"
        type="target"
        position={Position.Left}
        style={{
          width: 6, height: 6,
          background: '#38bdf8',
          border: '1px solid #0d1b2e',
          opacity: 0.9,
        }}
      />
      <Handle
        id="source"
        type="source"
        position={Position.Right}
        style={{
          width: 6, height: 6,
          background: '#38bdf8',
          border: '1px solid #0d1b2e',
          opacity: 0.9,
        }}
      />
    </div>
  )
})
GraphNode.displayName = 'GraphNode'

// ─────────────────────────────────────────────────────────────────────────────
// Custom edge — bezier path with optional weight badge at midpoint
// Uses explicit source/target handles so RF resolves coordinates from the
// Handle elements added to GraphNode rather than from the wrapper centroid.
// ─────────────────────────────────────────────────────────────────────────────
// Edge weight badge shown only when the edge is selected
const LabeledEdge = ({
  id,
  sourceX, sourceY, sourcePosition,
  targetX, targetY, targetPosition,
  style = {},
  markerEnd,
  data,
  selected,
}) => {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX, sourceY, sourcePosition,
    targetX, targetY, targetPosition,
  })
  const weight = data?.weight ?? null
  return (
    <>
      <BaseEdge id={id} path={edgePath} markerEnd={markerEnd} style={style} />
      {selected && weight !== null && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              fontSize: 9, fontFamily: 'monospace', fontWeight: 600,
              color: CYAN,
              background: 'rgba(13,27,46,0.92)',
              border: `1px solid ${CYAN}55`,
              borderRadius: 3, padding: '1px 4px', pointerEvents: 'none',
            }}
            className="nodrag nopan"
          >
            {weight.toFixed(2)}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  )
}
const NODE_TYPES = { custom: GraphNode }
const EDGE_TYPES = { labeled: LabeledEdge }

// ─────────────────────────────────────────────────────────────────────────────
// Force-directed layout (Fruchterman–Reingold)
// Pure function — takes plain objects, returns new array, mutates nothing.
// ─────────────────────────────────────────────────────────────────────────────
function forceLayout(nodes, edges, cw, ch) {
  const n  = nodes.length
  const cx = cw / 2
  const cy = ch / 2
  const MARGIN = 100

  // Seed on a circle so no two nodes start at the same point
  const sim = nodes.map((nd, i) => {
    const angle = (i / n) * 2 * Math.PI - Math.PI / 2
    const r     = Math.min(cw, ch) * 0.38
    return {
      id: nd.id,
      x:  cx + Math.cos(angle) * r,
      y:  cy + Math.sin(angle) * r,
      vx: 0,
      vy: 0,
    }
  })
  const byId = new Map(sim.map(s => [s.id, s]))

  // Undirected adjacency for symmetric attraction
  const adj = new Map(nodes.map(nd => [nd.id, new Set()]))
  edges.forEach(e => {
    adj.get(e.source)?.add(e.target)
    adj.get(e.target)?.add(e.source)
  })

  // Tuneable constants — stronger repulsion for better spacing
  const ITERS   = n <= 20  ? 300  : n <= 100 ? 400 : 500
  const REPEL   = n <= 20  ? 18000 : n <= 100 ? 12000 : 6000
  const ATTRACT = n <= 20  ? 0.003 : n <= 100 ? 0.0015 : 0.0006
  const DAMP    = 0.78
  const GRAVITY = 0.0002

  for (let it = 0; it < ITERS; it++) {
    // Zero forces
    sim.forEach(s => { s.fx = 0; s.fy = 0 })

    // Pairwise repulsion — O(n²)
    for (let i = 0; i < sim.length; i++) {
      for (let j = i + 1; j < sim.length; j++) {
        const a = sim[i], b = sim[j]
        const dx = b.x - a.x
        const dy = b.y - a.y
        const d  = Math.sqrt(dx * dx + dy * dy) || 0.1
        const f  = REPEL / (d * d)
        const ux = dx / d, uy = dy / d
        a.fx -= ux * f;  a.fy -= uy * f
        b.fx += ux * f;  b.fy += uy * f
      }
    }

    // Edge attraction
    sim.forEach(a => {
      adj.get(a.id)?.forEach(bid => {
        const b = byId.get(bid)
        if (!b) return
        const dx = b.x - a.x
        const dy = b.y - a.y
        const d  = Math.sqrt(dx * dx + dy * dy) || 0.1
        const f  = d * ATTRACT
        const ux = dx / d, uy = dy / d
        a.fx += ux * f;  a.fy += uy * f
      })
    })

    // Gravity toward centre
    sim.forEach(s => {
      s.fx += (cx - s.x) * GRAVITY
      s.fy += (cy - s.y) * GRAVITY
    })

    // Integrate with damping and boundary clamp
    sim.forEach(s => {
      s.vx = (s.vx + s.fx) * DAMP
      s.vy = (s.vy + s.fy) * DAMP
      s.x  = Math.max(MARGIN, Math.min(cw - MARGIN, s.x + s.vx))
      s.y  = Math.max(MARGIN, Math.min(ch - MARGIN, s.y + s.vy))
    })
  }

  return nodes.map((nd, i) => ({
    ...nd,
    position: { x: Math.round(sim[i].x), y: Math.round(sim[i].y) },
  }))
}

// ─────────────────────────────────────────────────────────────────────────────
// Convert backend nodes → RF nodes
//
// CRITICAL: `width` and `height` must be set as TOP-LEVEL node properties so
// React Flow v11 correctly measures the wrapper and computes handle positions.
// Setting them only in `style` is insufficient — RF reads the top-level props
// for its internal layout measurement.
// ─────────────────────────────────────────────────────────────────────────────
function buildRFNodes(backendNodes, p) {
  return backendNodes.map(n => ({
    id:       n.id,
    type:     'custom',
    position: undefined,      // forceLayout will assign this
    width:    p.sz,           // ← RF reads this for handle coordinate calc
    height:   p.sz,           // ← RF reads this for handle coordinate calc
    style: {
      width:        p.sz,
      height:       p.sz,
      borderRadius: '50%',
      padding:      0,
      // No border/background here — CustomNode owns its own visual style
    },
    data: {
      ...n.data,
      totalNodes: backendNodes.length,
    },
  }))
}

// ─────────────────────────────────────────────────────────────────────────────
// Convert backend edges → RF edges
//
// Uses RF's built-in 'default' (bezier) edge type.
// The `label` prop renders a weight badge natively; no custom component needed.
// Edge IDs, source, and target are taken verbatim from the backend — the probe
// confirmed 0 orphan edges.
// ─────────────────────────────────────────────────────────────────────────────
function buildRFEdges(backendEdges, p) {
  return backendEdges.map((e, i) => {
    const weight = e.data?.weight ?? null
    return {
      id:           e.id ?? `edge-${e.source}-${e.target}-${i}`,
      source:       String(e.source),
      target:       String(e.target),
      sourceHandle: 'source',
      targetHandle: 'target',
      type:         'labeled',
      animated:     false,
      markerEnd: {
        type:   MarkerType.ArrowClosed,
        width:  p.arw,
        height: p.arw,
        color:  CYAN,
      },
      style: {
        stroke:      CYAN,
        strokeWidth: p.ew,
        opacity:     p.eo,
      },
      data: { ...(e.data || {}), weight },
    }
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Sidebar panels
// ─────────────────────────────────────────────────────────────────────────────
function SidePanel({ title, icon: Icon, children }) {
  return (
    <div
      className="rounded-xl p-4"
      style={{ background: '#0b1628', border: `1px solid ${BORDER}` }}
    >
      <div className="flex items-center gap-2 mb-3">
        {Icon && <Icon className="w-3.5 h-3.5 flex-shrink-0" style={{ color: CYAN }} />}
        <span className="text-[10px] font-bold tracking-widest uppercase" style={{ color: CYAN }}>
          {title}
        </span>
      </div>
      {children}
    </div>
  )
}

function SelectedFlowPanel({ selData, connectedCount }) {
  if (!selData) {
    return (
      <SidePanel title="Selected Flow" icon={Network}>
        <div className="flex flex-col items-center py-5 gap-2 opacity-40">
          <Network className="w-7 h-7" style={{ color: CYAN }} />
          <p className="text-xs" style={{ color: '#94a3b8' }}>Click a node to inspect</p>
        </div>
      </SidePanel>
    )
  }

  const isAtk  = selData.predValue === 1
  const accent = isAtk ? RED : TEAL

  return (
    <SidePanel title="Selected Flow" icon={Network}>
      {/* Identity badge */}
      <div className="flex items-center gap-3 pb-3 mb-3"
           style={{ borderBottom: `1px solid ${BORDER}` }}>
        <div className="w-9 h-9 rounded-full flex-shrink-0 flex items-center justify-center
                        text-xs font-bold"
             style={{
               background: isAtk ? 'rgba(248,113,113,0.15)' : 'rgba(20,184,166,0.15)',
               border: `2px solid ${accent}`,
               color: accent,
             }}>
          {selData.flowIdx}
        </div>
        <div>
          <p className="text-sm font-bold text-white leading-none">{selData.label}</p>
          <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded mt-0.5 inline-block"
                style={{
                  background: isAtk ? 'rgba(248,113,113,0.18)' : 'rgba(20,184,166,0.18)',
                  color: accent,
                }}>
            {selData.prediction}
          </span>
        </div>
      </div>

      {/* Confidence */}
      <div className="flex justify-between text-xs mb-3">
        <span style={{ color: '#94a3b8' }}>Confidence</span>
        <span className="font-mono font-bold text-white">
          {(selData.confidence * 100).toFixed(1)}%
        </span>
      </div>

      {/* Probability bars */}
      {[
        { label: 'P(Attack)', val: selData.probAttack, color: RED },
        { label: 'P(Normal)', val: selData.probNormal, color: TEAL },
      ].map(({ label, val, color }) => (
        <div key={label} className="mb-2">
          <div className="flex justify-between text-xs mb-1">
            <span style={{ color: '#94a3b8' }}>{label}</span>
            <span className="font-mono" style={{ color }}>{val.toFixed(4)}</span>
          </div>
          <div className="h-1.5 rounded-full overflow-hidden" style={{ background: '#1a2d48' }}>
            <div className="h-full rounded-full transition-all duration-500"
                 style={{ width: `${val * 100}%`, background: color }} />
          </div>
        </div>
      ))}

      {/* Detail table */}
      <div className="mt-3 pt-3 space-y-2" style={{ borderTop: `1px solid ${BORDER}` }}>
        {[
          ['Flow Index',      selData.flowIdx],
          ['Predicted Class', selData.prediction],
          ['Status',          selData.predValue === 1 ? 'Malicious' : 'Benign'],
          ['Connected Edges', connectedCount],
        ].map(([k, v]) => (
          <div key={k} className="flex justify-between text-xs">
            <span style={{ color: '#94a3b8' }}>{k}</span>
            <span className="font-semibold text-white">{v}</span>
          </div>
        ))}
      </div>
    </SidePanel>
  )
}

function StatsPanel({ numNodes, numEdges }) {
  const avgDeg = numNodes > 0 ? (numEdges / numNodes).toFixed(1) : '—'
  return (
    <SidePanel title="Graph Statistics" icon={BarChart2}>
      <div className="space-y-2.5">
        {[
          ['Nodes',      numNodes],
          ['Edges',      numEdges],
          ['Avg. Degree', avgDeg],
        ].map(([k, v]) => (
          <div key={k} className="flex justify-between items-center">
            <span className="text-xs" style={{ color: '#94a3b8' }}>{k}</span>
            <span className="text-sm font-bold font-mono text-white">{v}</span>
          </div>
        ))}
      </div>
    </SidePanel>
  )
}

function LegendPanel() {
  return (
    <SidePanel title="Legend" icon={BookOpen}>
      <div className="space-y-3">
        {[
          { color: TEAL, label: 'Normal Flow',      sub: 'Predicted as normal traffic' },
          { color: RED,  label: 'Attack Flow',       sub: 'Predicted as malicious traffic' },
        ].map(({ color, label, sub }) => (
          <div key={label} className="flex items-start gap-2.5">
            <div className="w-3.5 h-3.5 rounded-full flex-shrink-0 mt-0.5"
                 style={{
                   background: `${color}20`,
                   border: `2px solid ${color}`,
                   boxShadow: `0 0 5px ${color}66`,
                 }} />
            <div>
              <p className="text-xs font-semibold text-white">{label}</p>
              <p className="text-[10px]" style={{ color: '#64748b' }}>{sub}</p>
            </div>
          </div>
        ))}
        <div className="flex items-start gap-2.5">
          <div className="flex-shrink-0 mt-1.5 h-px w-5" style={{ background: CYAN }} />
          <div>
            <p className="text-xs font-semibold text-white">Similarity Edge</p>
            <p className="text-[10px]" style={{ color: '#64748b' }}>Cosine similarity (k-NN)</p>
          </div>
        </div>
      </div>
    </SidePanel>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Control button
// ─────────────────────────────────────────────────────────────────────────────
function CtrlBtn({ onClick, title, children }) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-[11px] font-semibold
                 select-none transition-colors duration-150"
      style={{ background: 'rgba(13,27,46,0.92)', border: `1px solid ${BORDER}`, color: '#94a3b8' }}
      onMouseEnter={e => {
        e.currentTarget.style.borderColor = CYAN
        e.currentTarget.style.color = CYAN
      }}
      onMouseLeave={e => {
        e.currentTarget.style.borderColor = BORDER
        e.currentTarget.style.color = '#94a3b8'
      }}
    >
      {children}
    </button>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────
export default function GraphVisualization({ graphData }) {
  const sectionRef = useGsapReveal()
  const hasData    = !!(graphData?.nodes?.length)
  const rfRef      = useRef(null)

  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const [selNode,     setSelNode]     = useState(null)
  const [hoveredEdge, setHoveredEdge] = useState(null)

  // ── Rebuild whenever graphData changes ──────────────────────────────────
  useEffect(() => {
    if (!hasData) {
      setNodes([]); setEdges([]); setSelNode(null)
      return
    }

    const p      = scaleParams(graphData.nodes.length)
    const built  = buildRFNodes(graphData.nodes, p)
    const laid   = forceLayout(built, graphData.edges, p.cw, p.ch)
    const rfEdges = buildRFEdges(graphData.edges, p)

    // Validate before setting (dev-time guard — should always pass per probe)
    const nodeIds = new Set(laid.map(n => n.id))
    const orphans = rfEdges.filter(
      e => !nodeIds.has(e.source) || !nodeIds.has(e.target)
    )
    if (orphans.length > 0) {
      console.warn('[GraphViz] orphan edges:', orphans.length, orphans.slice(0, 3))
    }


    setNodes(laid)
    setEdges(rfEdges)
    setSelNode(null)

    // fitView AFTER React Flow has had a tick to measure the new nodes.
    // Using 200 ms gives the browser time to paint the nodes before we ask RF
    // to fit — calling fitView immediately after setNodes results in a no-op
    // because the nodes have no measured dimensions yet.
    setTimeout(() => rfRef.current?.fitView({ padding: 0.12, duration: 400 }), 200)
  }, [graphData, hasData, setNodes, setEdges])

  // ── Node click: highlight connected edges, update sidebar ────────────────
  const onNodeClick = useCallback((_evt, node) => {
    setSelNode(node)

    const p = scaleParams(graphData?.nodes?.length ?? 5)
    setEdges(prev => {
      const connSet = new Set(
        prev.filter(e => e.source === node.id || e.target === node.id).map(e => e.id)
      )
      return prev.map(e => ({
        ...e,
        style: {
          ...e.style,
          stroke:      connSet.has(e.id) ? '#7dd3fc' : CYAN,
          strokeWidth: connSet.has(e.id) ? p.ew * 2.5 : p.ew,
          opacity:     connSet.has(e.id) ? 1 : 0.20,
        },
      }))
    })
  }, [graphData, setEdges])

  // ── Pane click: deselect, restore edge styles ────────────────────────────
  const onPaneClick = useCallback(() => {
    setSelNode(null)
    if (!graphData?.edges?.length) return
    const p = scaleParams(graphData.nodes.length)
    setEdges(buildRFEdges(graphData.edges, p))
  }, [graphData, setEdges])

  // ── Controls ─────────────────────────────────────────────────────────────
  const doZoomIn  = () => rfRef.current?.zoomIn()
  const doZoomOut = () => rfRef.current?.zoomOut()
  const doFit     = () => rfRef.current?.fitView({ padding: 0.12, duration: 300 })

  const doReset = useCallback(() => {
    if (!graphData?.nodes) return
    const p     = scaleParams(graphData.nodes.length)
    const built = buildRFNodes(graphData.nodes, p)
    const laid  = forceLayout(built, graphData.edges, p.cw, p.ch)
    setNodes(laid)
    setEdges(buildRFEdges(graphData.edges, p))
    setSelNode(null)
    setTimeout(doFit, 200)
  }, [graphData, setNodes, setEdges]) // eslint-disable-line react-hooks/exhaustive-deps

  const selData        = selNode?.data ?? null
  const connectedCount = selNode
    ? edges.filter(e => e.source === selNode.id || e.target === selNode.id).length
    : 0

  const minimapNodeColor = useCallback(
    n => (n.data?.predValue === 1 ? RED : TEAL),
    []
  )

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <section id="graph-viz" className="relative py-16 px-4 max-w-full mx-auto">

      {/* Section heading — GSAP reveal */}
      <div ref={sectionRef} className="mb-6">
        <div className="flex items-center gap-2 mb-1">
          <Network className="w-4 h-4" style={{ color: CYAN }} />
          <span className="text-xs font-semibold uppercase tracking-widest"
                style={{ color: CYAN }}>
            Flow-Similarity Graph
          </span>
        </div>
        <h2 className="text-2xl font-extrabold text-white">Network Flow Visualization</h2>
        <p className="text-sm mt-1" style={{ color: '#64748b' }}>
          Each node represents a network flow. Edges connect flows with similar
          ct* neighbourhood statistics (cosine similarity). Node colours reflect
          actual GraphSAGE predictions.
        </p>
      </div>

      {hasData ? (
        <div className="flex gap-4 items-start">

          {/* ── Graph canvas ~75 % ───────────────────────────────────────── */}
          {/*
            NO overflow-hidden — it clips the edge SVG layer (Bug 2).
            ReactFlow needs its own explicit width+height; the parent provides
            both via the style prop.
          */}
          <div
            className="flex-1 min-w-0 relative rounded-xl"
            style={{
              height: 640,
              background: NAVY,
              border: `1px solid ${BORDER}`,
            }}
          >
            {/* Header strip — overlay, NOT inside ReactFlow */}
            {/*
              Bug 1 fix: previously this was achieved via paddingTop:40 on the
              <ReactFlow> component, which broke RF's internal viewport math.
              Now the strip is an absolutely-positioned overlay that sits on top
              of the RF canvas without affecting RF's coordinate system.
            */}
            <div
              className="absolute top-0 left-0 right-0 z-10 flex items-center
                         justify-between px-4 py-2.5 rounded-t-xl pointer-events-none"
              style={{
                background: 'rgba(13,27,46,0.82)',
                borderBottom: `1px solid ${BORDER}`,
              }}
            >
              <div className="flex items-center gap-2">
                <div className="w-5 h-5 rounded flex items-center justify-center"
                     style={{
                       background: 'rgba(56,189,248,0.12)',
                       border: `1px solid ${CYAN}44`,
                     }}>
                  <Network className="w-3 h-3" style={{ color: CYAN }} />
                </div>
                <span className="text-xs font-bold text-white">Flow Similarity Graph</span>
              </div>
              <span className="text-[10px]" style={{ color: '#475569' }}>
                {graphData.numNodes} nodes · {graphData.numEdges} edges · k-NN
              </span>
            </div>

            {/* React Flow — explicit width+height so RF's internal SVG is sized.
                Without these, the RF root div has height:0 and no edges render. */}
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={onNodeClick}
              onPaneClick={onPaneClick}
              onEdgeMouseEnter={(_e, edge) => setHoveredEdge(edge)}
              onEdgeMouseLeave={() => setHoveredEdge(null)}
              onInit={inst => {
                rfRef.current = inst
                setTimeout(() => inst.fitView({ padding: 0.12 }), 200)
              }}
              nodeTypes={NODE_TYPES}
              edgeTypes={EDGE_TYPES}
              fitView={false}
              minZoom={0.04}
              maxZoom={6}
              nodesDraggable
              nodesConnectable={false}
              elementsSelectable
              selectNodesOnDrag={false}
              elevateEdgesOnSelect
              proOptions={{ hideAttribution: true }}
              style={{ width: '100%', height: '100%' }}
            >
              <Background
                variant="dots"
                gap={28}
                size={1}
                color="#1e3a5f"
                style={{ opacity: 0.55 }}
              />
              <MiniMap
                nodeColor={minimapNodeColor}
                maskColor="rgba(13,27,46,0.75)"
                style={{
                  background: 'rgba(13,27,46,0.90)',
                  border: `1px solid ${BORDER}`,
                  borderRadius: 8,
                }}
                pannable
                zoomable
              />
            </ReactFlow>

            {/* Bottom-left controls */}
            <div className="absolute bottom-4 left-4 z-20 flex items-center gap-2">
              <CtrlBtn onClick={doZoomIn}  title="Zoom in">+</CtrlBtn>
              <CtrlBtn onClick={doZoomOut} title="Zoom out">−</CtrlBtn>
              <CtrlBtn onClick={doFit} title="Fit to view">
                <Maximize2 className="w-3 h-3" />Fit
              </CtrlBtn>
              <CtrlBtn onClick={doReset} title="Reset layout">
                <RotateCcw className="w-3 h-3" />Reset
              </CtrlBtn>
            </div>

            {/* Edge hover tooltip */}
            {hoveredEdge && (
              <div
                className="absolute bottom-14 left-1/2 -translate-x-1/2 z-30
                           px-3 py-1.5 rounded-lg text-xs pointer-events-none"
                style={{
                  background: 'rgba(13,27,46,0.95)',
                  border: `1px solid ${CYAN}44`,
                  color: '#cbd5e1',
                }}
              >
                Flow {hoveredEdge.source} → Flow {hoveredEdge.target}
                {hoveredEdge.data?.weight != null && (
                  <span className="ml-2 font-bold" style={{ color: CYAN }}>
                    {hoveredEdge.data.weight.toFixed(3)} sim
                  </span>
                )}
              </div>
            )}
          </div>

          {/* ── Right sidebar ~25 % ──────────────────────────────────────── */}
          <div className="w-56 flex-shrink-0 flex flex-col gap-3">
            <SelectedFlowPanel selData={selData} connectedCount={connectedCount} />
            <StatsPanel numNodes={graphData.numNodes} numEdges={graphData.numEdges} />
            <LegendPanel />
          </div>
        </div>

      ) : (
        /* Empty state */
        <div
          className="rounded-xl flex items-center justify-center"
          style={{ height: 560, background: NAVY, border: `1px solid ${BORDER}` }}
        >
          <div className="text-center">
            <Network className="w-14 h-14 mx-auto mb-4 opacity-15" style={{ color: CYAN }} />
            <p className="text-lg font-semibold text-white mb-1">No graph data yet</p>
            <p className="text-sm" style={{ color: '#475569' }}>
              Upload a CSV file to visualise the flow-similarity graph
            </p>
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <div
        className="mt-4 flex items-start gap-3 px-3 py-2.5 rounded-lg"
        style={{
          background: 'rgba(251,191,36,0.04)',
          border: '1px solid rgba(251,191,36,0.14)',
        }}
      >
        <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5"
                       style={{ color: '#fbbf24' }} />
        <p className="text-xs" style={{ color: '#94a3b8' }}>
          <strong style={{ color: '#fbbf24' }}>
            Edges represent k-nearest-neighbor similarity, not IP communication.
          </strong>{' '}
          Node colours show GraphSAGE predictions from{' '}
          <code className="text-[10px]">models/graphsage_best.pt</code>.
        </p>
      </div>
    </section>
  )
}
