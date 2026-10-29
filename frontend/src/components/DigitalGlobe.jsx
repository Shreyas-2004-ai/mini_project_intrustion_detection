/**
 * DigitalGlobe.jsx
 *
 * Dense, luminous digital Earth rendered on a <canvas>.
 *
 * Fix log:
 *  • BUG 1 — cursor parallax removed completely.
 *    state.parallax was added to every draw offset, causing globe to follow
 *    the mouse.  The mousemove listener and all parallax usage are deleted.
 *    offX/offY are now always 0; only the GSAP float tween moves the globe.
 *
 *  • BUG 2 — globe was clipped on the sides.
 *    The atmospheric halo drew to R*1.45 = 290 px from the centre, giving a
 *    total diameter of 580 px on a 520 px canvas.  ctx.clearRect only cleared
 *    size×size so the halo exceeded the canvas boundary and was clipped by the
 *    browser's canvas edge.
 *    Fix: CANVAS_PAINT is now SIZE+120 (640 px) while the logical SIZE stays
 *    520 px.  The canvas element stays at SIZE×SIZE; the extra paint area is
 *    centred via a negative CSS offset so the halo and orbital rings have room.
 *    R kept at 190 px, halo at R*1.42 = 270 px → diameter 540 px, fits in 640.
 *
 *  Preserved: rotation, float animation, orbital rings, travelling particles,
 *  pulsing threat nodes, depth-based brightness, GSAP cleanup.
 */
import { useEffect, useRef } from 'react'
import gsap from 'gsap'

/* ─── constants ───────────────────────────────────────────────────────────── */
const SIZE         = 520          // display size (CSS px, container)
const CANVAS_PAINT = SIZE + 120   // 640 — extra room for halo / orbitals
const R            = 190          // globe radius in logical px
const CX           = CANVAS_PAINT / 2
const CY           = CANVAS_PAINT / 2

const POINT_COUNT  = 380
const EDGE_DIST    = 0.42
const THREAT_COUNT = 7
const PARTICLE_N   = 18
const ROT_SEC      = 60           // seconds per full rotation
const FLOAT_AMP    = 7            // px vertical float

/* ─── helpers ─────────────────────────────────────────────────────────────── */
function project3d(x3, y3, z3, rotY) {
  const cosR = Math.cos(rotY), sinR = Math.sin(rotY)
  const rx =  x3 * cosR + z3 * sinR
  const rz = -x3 * sinR + z3 * cosR
  return {
    sx:    CX + rx * R,
    sy:    CY - y3 * R,
    depth: rz,
  }
}

function fibonacciSphere(n) {
  const pts = [], phi = Math.PI * (3 - Math.sqrt(5))
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / (n - 1)) * 2
    const r = Math.sqrt(Math.max(0, 1 - y * y))
    const a = phi * i
    pts.push({ x: Math.cos(a) * r, y, z: Math.sin(a) * r })
  }
  return pts
}

function orbitalRing(incl, phase, steps = 120) {
  const pts = []
  const ci = Math.cos(incl), si = Math.sin(incl)
  const cp = Math.cos(phase), sp = Math.sin(phase)
  for (let i = 0; i <= steps; i++) {
    const a  = (i / steps) * 2 * Math.PI
    const x0 = Math.cos(a), z0 = Math.sin(a)
    pts.push({
      x: x0 * cp - z0 * si * sp,
      y: z0 * ci,
      z: x0 * sp + z0 * si * cp,
    })
  }
  return pts
}

/* ─── component ───────────────────────────────────────────────────────────── */
export default function DigitalGlobe({ size = SIZE }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const dpr = Math.min(window.devicePixelRatio || 1, 2)

    // Canvas backing store = CANVAS_PAINT * dpr; CSS size stays SIZE
    canvas.width  = CANVAS_PAINT * dpr
    canvas.height = CANVAS_PAINT * dpr
    ctx.scale(dpr, dpr)

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    /* geometry */
    const rawPts = fibonacciSphere(POINT_COUNT)
    const threatIdxs = new Set()
    while (threatIdxs.size < THREAT_COUNT)
      threatIdxs.add(Math.floor(Math.random() * POINT_COUNT))
    const pts = rawPts.map((p, i) => ({ ...p, threat: threatIdxs.has(i) }))

    const edges = []
    for (let i = 0; i < pts.length; i++)
      for (let j = i + 1; j < pts.length; j++) {
        const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y, dz = pts[i].z - pts[j].z
        if (Math.sqrt(dx*dx + dy*dy + dz*dz) < EDGE_DIST) edges.push([i, j])
      }

    const orbitals = [
      orbitalRing(Math.PI * 0.18, 0),
      orbitalRing(Math.PI * 0.32, Math.PI * 0.5),
      orbitalRing(Math.PI * 0.08, Math.PI * 0.25),
    ]

    const particles = Array.from({ length: PARTICLE_N }, () => ({
      edge:   edges[Math.floor(Math.random() * edges.length)],
      t:      Math.random(),
      speed:  0.003 + Math.random() * 0.005,
      threat: Math.random() < 0.15,
    }))

    const threatPhases = pts.map((_, i) =>
      threatIdxs.has(i) ? Math.random() * Math.PI * 2 : 0
    )

    /* animation state — NO parallax field */
    const state = { rotY: 0, floatY: 0, time: 0 }

    let rotTween, floatTween, raf

    if (!reduced) {
      rotTween = gsap.to(state, {
        rotY: Math.PI * 2,
        duration: ROT_SEC,
        ease: 'none',
        repeat: -1,
      })
      floatTween = gsap.to(state, {
        floatY: FLOAT_AMP,
        duration: 5.5,
        ease: 'sine.inOut',
        repeat: -1,
        yoyo: true,
      })
    }

    /* draw loop — offX is always 0; offY is only the float */
    const draw = ts => {
      state.time = ts / 1000
      ctx.clearRect(0, 0, CANVAS_PAINT, CANVAS_PAINT)

      const rotY = state.rotY
      const oy   = state.floatY   // no parallax shift

      /* atmosphere */
      const atm = ctx.createRadialGradient(CX, CY + oy, R * 0.85, CX, CY + oy, R * 1.42)
      atm.addColorStop(0,    'rgba(30,100,255,0.0)')
      atm.addColorStop(0.6,  'rgba(30,100,255,0.10)')
      atm.addColorStop(0.82, 'rgba(56,189,248,0.18)')
      atm.addColorStop(1,    'rgba(56,189,248,0.0)')
      ctx.fillStyle = atm
      ctx.beginPath()
      ctx.arc(CX, CY + oy, R * 1.42, 0, Math.PI * 2)
      ctx.fill()

      /* sphere fill */
      const sg = ctx.createRadialGradient(
        CX - R * 0.28, CY + oy - R * 0.22, R * 0.05,
        CX, CY + oy, R
      )
      sg.addColorStop(0,    'rgba(20,60,120,0.55)')
      sg.addColorStop(0.55, 'rgba(8,25,55,0.70)')
      sg.addColorStop(1,    'rgba(4,10,26,0.80)')
      ctx.fillStyle = sg
      ctx.beginPath()
      ctx.arc(CX, CY + oy, R, 0, Math.PI * 2)
      ctx.fill()

      /* rim */
      const rim = ctx.createRadialGradient(CX, CY + oy, R * 0.88, CX, CY + oy, R)
      rim.addColorStop(0, 'rgba(56,189,248,0)')
      rim.addColorStop(1, 'rgba(56,189,248,0.22)')
      ctx.fillStyle = rim
      ctx.beginPath()
      ctx.arc(CX, CY + oy, R, 0, Math.PI * 2)
      ctx.fill()

      /* clip — sphere only */
      ctx.save()
      ctx.beginPath()
      ctx.arc(CX, CY + oy, R, 0, Math.PI * 2)
      ctx.clip()

      /* project */
      const proj = pts.map(p => project3d(p.x, p.y, p.z, rotY))
      /* shift sy by oy */
      proj.forEach(p => { p.sy += oy })

      /* edges */
      for (const [i, j] of edges) {
        const a = proj[i], b = proj[j]
        const depth = (a.depth + b.depth) / 2
        if (depth < -0.15) continue
        const alpha = ((depth + 1) / 2) * 0.22 + 0.04
        ctx.beginPath()
        ctx.moveTo(a.sx, a.sy)
        ctx.lineTo(b.sx, b.sy)
        ctx.strokeStyle = `rgba(56,189,248,${alpha.toFixed(3)})`
        ctx.lineWidth = 0.6
        ctx.stroke()
      }

      ctx.restore()

      /* orbital rings (outside sphere clip) */
      for (let ri = 0; ri < orbitals.length; ri++) {
        const ring = orbitals[ri]
        const phOff = state.time * 0.04 * (ri % 2 === 0 ? 1 : -1)
        ctx.beginPath()
        let first = true
        for (const p of ring) {
          const pp = project3d(p.x, p.y, p.z, rotY + phOff)
          pp.sy += oy
          const scale = 1.18
          const ex = CX + (pp.sx - CX) * scale
          const ey = CY + oy + (pp.sy - CY - oy) * scale
          if (first) { ctx.moveTo(ex, ey); first = false }
          else ctx.lineTo(ex, ey)
        }
        ctx.strokeStyle = 'rgba(99,179,255,0.25)'
        ctx.lineWidth = 0.8
        ctx.stroke()
      }

      /* nodes — re-clip */
      ctx.save()
      ctx.beginPath()
      ctx.arc(CX, CY + oy, R, 0, Math.PI * 2)
      ctx.clip()

      for (let i = 0; i < pts.length; i++) {
        const { sx, sy, depth } = proj[i]
        if (depth < -0.05) continue
        const b = (depth + 1) / 2

        if (pts[i].threat) {
          const pulse = 0.55 + 0.45 * Math.sin(state.time * 2.8 + threatPhases[i])
          const glow  = 5 + pulse * 8
          const tg = ctx.createRadialGradient(sx, sy, 0, sx, sy, glow * 2)
          tg.addColorStop(0, `rgba(255,80,80,${(0.4 * pulse * b).toFixed(3)})`)
          tg.addColorStop(1, 'rgba(255,80,80,0)')
          ctx.fillStyle = tg
          ctx.beginPath()
          ctx.arc(sx, sy, glow * 2, 0, Math.PI * 2)
          ctx.fill()
          ctx.beginPath()
          ctx.arc(sx, sy, 2.5 + pulse * 1.5, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(255,100,100,${(0.85 * b + 0.15).toFixed(3)})`
          ctx.fill()
        } else {
          ctx.beginPath()
          ctx.arc(sx, sy, 1.2 + b * 1.4, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(100,200,255,${(0.25 + b * 0.65).toFixed(3)})`
          ctx.fill()
          if (i % 7 === 0) {
            const nh = ctx.createRadialGradient(sx, sy, 0, sx, sy, 5)
            nh.addColorStop(0, `rgba(150,230,255,${(0.35 * b).toFixed(3)})`)
            nh.addColorStop(1, 'rgba(150,230,255,0)')
            ctx.fillStyle = nh
            ctx.beginPath()
            ctx.arc(sx, sy, 5, 0, Math.PI * 2)
            ctx.fill()
          }
        }
      }

      /* particles */
      for (const par of particles) {
        const [ei, ej] = par.edge
        par.t += par.speed
        if (par.t > 1) par.t -= 1
        const a = proj[ei], b = proj[ej]
        if (a.depth < 0 && b.depth < 0) continue
        const sx   = a.sx + (b.sx - a.sx) * par.t
        const sy   = a.sy + (b.sy - a.sy) * par.t
        const dep  = a.depth + (b.depth - a.depth) * par.t
        if (dep < 0) continue
        const col = par.threat ? '255,100,80' : '56,200,255'
        ctx.beginPath()
        ctx.arc(sx, sy, 2.2, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(${col},${((dep + 1) / 2 * 0.9).toFixed(2)})`
        ctx.fill()
        const tr = ctx.createRadialGradient(sx, sy, 0, sx, sy, 5)
        tr.addColorStop(0, `rgba(${col},0.3)`)
        tr.addColorStop(1, `rgba(${col},0)`)
        ctx.fillStyle = tr
        ctx.beginPath()
        ctx.arc(sx, sy, 5, 0, Math.PI * 2)
        ctx.fill()
      }

      ctx.restore()

      /* specular */
      const spec = ctx.createRadialGradient(
        CX - R * 0.35, CY + oy - R * 0.30, 0,
        CX - R * 0.20, CY + oy - R * 0.15, R * 0.6
      )
      spec.addColorStop(0,   'rgba(200,240,255,0.07)')
      spec.addColorStop(0.5, 'rgba(100,180,255,0.03)')
      spec.addColorStop(1,   'rgba(100,180,255,0)')
      ctx.fillStyle = spec
      ctx.beginPath()
      ctx.arc(CX, CY + oy, R, 0, Math.PI * 2)
      ctx.fill()

      raf = requestAnimationFrame(draw)
    }

    raf = requestAnimationFrame(draw)

    return () => {
      cancelAnimationFrame(raf)
      rotTween?.kill()
      floatTween?.kill()
      // No event listener to remove — parallax listener deleted
    }
  }, [size])

  /*
   * Canvas backing store is CANVAS_PAINT wide but we show it centred inside
   * a SIZE×SIZE container.  The canvas CSS size is CANVAS_PAINT so the extra
   * halo pixels are visible; the negative margin pulls it into position.
   */
  const overflow = (CANVAS_PAINT - SIZE) / 2   // 60 px each side

  return (
    <div
      className="relative select-none overflow-visible"
      style={{ width: size, height: size }}
      aria-hidden="true"
    >
      {/* ambient glow behind canvas */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(circle at 50% 50%, rgba(30,100,255,0.12) 0%, rgba(56,189,248,0.06) 45%, transparent 70%)',
          transform: 'scale(1.35)',
          zIndex: 0,
        }}
      />
      {/*
        Canvas is CANVAS_PAINT px wide (CSS).  Negative margin centres it so
        the extra 60 px bleeds equally outside the container on all sides —
        this is what lets the halo and orbitals render without clipping.
        overflow-visible on the parent div ensures those bleed regions show.
      */}
      <canvas
        ref={canvasRef}
        style={{
          width:      CANVAS_PAINT,
          height:     CANVAS_PAINT,
          display:    'block',
          position:   'absolute',
          top:        -overflow,
          left:       -overflow,
          zIndex:     1,
        }}
      />
    </div>
  )
}
