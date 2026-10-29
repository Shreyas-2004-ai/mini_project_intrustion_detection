/**
 * AnimatedBackground.jsx
 * Canvas-based animated particle + network background.
 * Respects prefers-reduced-motion. Throttles particle count on mobile.
 * Cleans up on unmount. Never blocks pointer events on content.
 */
import { useEffect, useRef } from 'react'

const isMobile = () => typeof window !== 'undefined' && window.innerWidth < 768

export default function AnimatedBackground() {
  const canvasRef = useRef(null)

  useEffect(() => {
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (prefersReduced) return

    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    const mobile = isMobile()
    const PARTICLE_COUNT = mobile ? 35 : 80
    const CONNECTION_DIST = mobile ? 100 : 160
    const PULSE_INTERVAL = 3000

    let W = 0, H = 0, raf = null, pulseTimer = null

    // ── Resize ──────────────────────────────────────────────────────────────
    const resize = () => {
      W = canvas.width  = window.innerWidth
      H = canvas.height = window.innerHeight
    }
    resize()
    window.addEventListener('resize', resize)

    // ── Particles ────────────────────────────────────────────────────────────
    class Particle {
      constructor() { this.reset(true) }
      reset(initial = false) {
        this.x  = Math.random() * W
        this.y  = initial ? Math.random() * H : (Math.random() < 0.5 ? -10 : H + 10)
        this.vx = (Math.random() - 0.5) * 0.35
        this.vy = (Math.random() - 0.5) * 0.35
        this.r  = Math.random() * 1.6 + 0.6
        // ~5% are red threat particles
        this.threat = Math.random() < 0.05
        this.alpha  = Math.random() * 0.5 + 0.2
        this.pulse  = 0
      }
      update() {
        this.x += this.vx
        this.y += this.vy
        if (this.pulse > 0) this.pulse -= 0.03
        if (this.x < -20 || this.x > W + 20 || this.y < -20 || this.y > H + 20) this.reset()
      }
      draw() {
        const glow = this.pulse > 0 ? this.pulse : 0
        const col  = this.threat ? `rgba(248,113,113,` : `rgba(59,130,246,`
        const colC = this.threat ? `rgba(248,113,113,` : `rgba(56,189,248,`
        if (glow > 0) {
          ctx.beginPath()
          ctx.arc(this.x, this.y, this.r * 3 * glow, 0, Math.PI * 2)
          ctx.fillStyle = colC + (this.alpha * glow * 0.4) + ')'
          ctx.fill()
        }
        ctx.beginPath()
        ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2)
        ctx.fillStyle = col + this.alpha + ')'
        ctx.fill()
      }
    }

    const particles = Array.from({ length: PARTICLE_COUNT }, () => new Particle())

    // ── Pulse objects ────────────────────────────────────────────────────────
    const pulses = []

    class LinePulse {
      constructor(x1, y1, x2, y2) {
        this.x1 = x1; this.y1 = y1; this.x2 = x2; this.y2 = y2
        this.t = 0
        this.threat = Math.random() < 0.08
      }
      update() { this.t += 0.018; return this.t < 1 }
      draw() {
        const px = this.x1 + (this.x2 - this.x1) * this.t
        const py = this.y1 + (this.y2 - this.y1) * this.t
        const col = this.threat ? 'rgba(248,113,113,' : 'rgba(56,189,248,'
        ctx.beginPath()
        ctx.arc(px, py, 2.5, 0, Math.PI * 2)
        ctx.fillStyle = col + (0.9 * (1 - this.t)) + ')'
        ctx.fill()
      }
    }

    // ── Periodically spawn pulses ────────────────────────────────────────────
    const spawnPulse = () => {
      if (particles.length < 2) return
      const a = particles[Math.floor(Math.random() * particles.length)]
      const b = particles[Math.floor(Math.random() * particles.length)]
      const dist = Math.hypot(a.x - b.x, a.y - b.y)
      if (dist < CONNECTION_DIST) {
        pulses.push(new LinePulse(a.x, a.y, b.x, b.y))
        a.pulse = 1
        b.pulse = 0.6
      }
      pulseTimer = setTimeout(spawnPulse, PULSE_INTERVAL + Math.random() * 1000)
    }
    pulseTimer = setTimeout(spawnPulse, 1200)

    // ── Draw loop ────────────────────────────────────────────────────────────
    const draw = () => {
      ctx.clearRect(0, 0, W, H)

      // Radial hero glow
      const grd = ctx.createRadialGradient(W * 0.5, H * 0.25, 0, W * 0.5, H * 0.25, W * 0.55)
      grd.addColorStop(0,   'rgba(59,130,246,0.055)')
      grd.addColorStop(0.5, 'rgba(139,92,246,0.025)')
      grd.addColorStop(1,   'transparent')
      ctx.fillStyle = grd
      ctx.fillRect(0, 0, W, H)

      // Connections
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x
          const dy = particles[i].y - particles[j].y
          const d  = Math.sqrt(dx * dx + dy * dy)
          if (d < CONNECTION_DIST) {
            const alpha = (1 - d / CONNECTION_DIST) * 0.12
            ctx.beginPath()
            ctx.moveTo(particles[i].x, particles[i].y)
            ctx.lineTo(particles[j].x, particles[j].y)
            ctx.strokeStyle = `rgba(59,130,246,${alpha})`
            ctx.lineWidth = 0.6
            ctx.stroke()
          }
        }
      }

      particles.forEach(p => { p.update(); p.draw() })

      // Pulses
      for (let i = pulses.length - 1; i >= 0; i--) {
        const alive = pulses[i].update()
        pulses[i].draw()
        if (!alive) pulses.splice(i, 1)
      }

      raf = requestAnimationFrame(draw)
    }
    draw()

    return () => {
      cancelAnimationFrame(raf)
      clearTimeout(pulseTimer)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="fixed inset-0 w-full h-full pointer-events-none"
      style={{ zIndex: 0, opacity: 0.85 }}
    />
  )
}
