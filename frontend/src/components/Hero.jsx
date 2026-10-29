import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import {
  Shield, ArrowRight, UploadCloud,
  Database, Network, Cpu, ChevronDown,
} from 'lucide-react'
import DigitalGlobe from './DigitalGlobe'

/* ── metric cards below the headline ──────────────────────────────────────── */
const METRICS = [
  {
    icon: Database,
    value: '257K+',
    label: 'Network Flows',
    sub:   'UNSW-NB15 dataset',
    color: '#60a5fa',
    glow:  'rgba(59,130,246,0.20)',
  },
  {
    icon: Network,
    value: '42',
    label: 'Features',
    sub:   'k=5 k-NN graph',
    color: '#2dd4bf',
    glow:  'rgba(20,184,166,0.20)',
  },
  {
    icon: Cpu,
    value: '0.8994',
    label: 'F1 Score',
    sub:   'Official test set',
    color: '#a78bfa',
    glow:  'rgba(139,92,246,0.20)',
  },
]

export default function Hero() {
  const heroRef   = useRef(null)
  const globeSize = 520

  useEffect(() => {
    const el = heroRef.current
    if (!el) return

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    /* make everything visible immediately if reduced motion */
    if (reduced) {
      el.querySelectorAll('[data-hero]').forEach(e => {
        e.style.opacity  = '1'
        e.style.transform = 'none'
      })
      return
    }

    const ctx = gsap.context(() => {
      /* badge + headline chars stagger */
      gsap.fromTo(
        '[data-hero="text"]',
        { opacity: 0, y: 38, filter: 'blur(4px)' },
        {
          opacity: 1, y: 0, filter: 'blur(0px)',
          duration: 0.80,
          stagger: 0.11,
          ease: 'power3.out',
          delay: 0.08,
        }
      )

      /* globe entrance */
      gsap.fromTo(
        '[data-hero="globe"]',
        { opacity: 0, scale: 0.82, x: 50 },
        { opacity: 1, scale: 1, x: 0, duration: 1.2, ease: 'power3.out', delay: 0.20 }
      )

      /* metric cards stagger after text */
      gsap.fromTo(
        '[data-hero="metric"]',
        { opacity: 0, y: 24 },
        { opacity: 1, y: 0, duration: 0.55, stagger: 0.10, ease: 'power3.out', delay: 0.55 }
      )

      /* floating badges on the globe */
      gsap.fromTo(
        '[data-hero="badge"]',
        { opacity: 0, scale: 0.8 },
        { opacity: 1, scale: 1, duration: 0.5, stagger: 0.15, ease: 'back.out(1.4)', delay: 1.0 }
      )

      /* scroll-cue bounce already in tailwind, just reveal it */
      gsap.fromTo('[data-hero="scroll"]',
        { opacity: 0 },
        { opacity: 1, duration: 0.5, delay: 1.4 }
      )
    }, el)

    return () => ctx.revert()
  }, [])

  return (
    <section
      ref={heroRef}
      id="home"
      className="relative min-h-screen flex items-center"
      style={{ paddingTop: 72 }}
    >
      {/* ── background layers ─────────────────────────────────────────────── */}

      {/* deep navy base — the canvas AnimatedBackground sits behind this */}
      <div className="absolute inset-0 pointer-events-none"
           style={{ background: 'linear-gradient(160deg, #04091a 0%, #060b14 50%, #05091a 100%)' }} />

      {/* subtle technical grid */}
      <div className="absolute inset-0 cyber-grid pointer-events-none" style={{ opacity: 0.055 }} />

      {/* left radial spotlight (behind headline) */}
      <div className="absolute pointer-events-none"
           style={{
             top: '-10%', left: '-8%',
             width: '70%', height: '115%',
             background: 'radial-gradient(ellipse 60% 55% at 35% 45%, rgba(59,130,246,0.13) 0%, rgba(139,92,246,0.06) 40%, transparent 70%)',
           }} />

      {/* right radial spotlight (behind globe) */}
      <div className="absolute pointer-events-none"
           style={{
             top: '-5%', right: '-12%',
             width: '65%', height: '110%',
             background: 'radial-gradient(ellipse 55% 60% at 65% 45%, rgba(30,100,220,0.16) 0%, rgba(56,189,248,0.07) 45%, transparent 72%)',
           }} />

      {/* floating ambient orbs */}
      <div className="absolute top-1/4 left-[8%] w-72 h-72 rounded-full blur-3xl pointer-events-none animate-float-slow"
           style={{ background: 'rgba(99,102,241,0.04)', animationDelay: '0s' }} />
      <div className="absolute bottom-1/4 right-[30%] w-56 h-56 rounded-full blur-3xl pointer-events-none animate-float"
           style={{ background: 'rgba(20,184,166,0.04)', animationDelay: '2.5s' }} />

      {/* ── content grid ──────────────────────────────────────────────────── */}
      <div className="relative z-10 w-full max-w-[1400px] mx-auto px-6 lg:px-14 xl:px-20 py-16">
        <div className="flex flex-col lg:flex-row items-center lg:items-center gap-10 lg:gap-6">

          {/* ── LEFT COLUMN — headline + CTAs + metrics ──────────────────── */}
          <div className="flex-1 flex flex-col items-start max-w-2xl">

            {/* Glowing badge */}
            <div
              data-hero="text"
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-[11px] font-bold tracking-widest uppercase mb-7"
              style={{
                background: 'rgba(59,130,246,0.09)',
                border: '1px solid rgba(59,130,246,0.30)',
                color: '#93c5fd',
                boxShadow: '0 0 18px rgba(59,130,246,0.15)',
              }}
            >
              <Shield className="w-3 h-3" />
              Graph Neural Network · UNSW-NB15
            </div>

            {/* Headline — oversized */}
            <h1 className="font-extrabold tracking-tight leading-[1.04] mb-5">

              {/* line 1 */}
              <span
                data-hero="text"
                className="block text-white"
                style={{ fontSize: 'clamp(2.4rem, 5vw, 4.2rem)' }}
              >
                Network
              </span>

              {/* gradient "Intrusion" */}
              <span
                data-hero="text"
                className="block"
                style={{
                  fontSize: 'clamp(2.8rem, 6vw, 5.2rem)',
                  background: 'linear-gradient(100deg, #38bdf8 0%, #60a5fa 35%, #818cf8 70%, #a78bfa 100%)',
                  WebkitBackgroundClip: 'text',
                  backgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  filter: 'drop-shadow(0 0 22px rgba(56,189,248,0.4))',
                }}
              >
                Intrusion
              </span>

              {/* line 3 */}
              <span
                data-hero="text"
                className="block text-white"
                style={{ fontSize: 'clamp(2.4rem, 5vw, 4.2rem)' }}
              >
                Detection
              </span>

              {/* sub-line gradient */}
              <span
                data-hero="text"
                className="block text-[1.35rem] sm:text-2xl font-bold mt-2"
                style={{
                  background: 'linear-gradient(90deg, #60a5fa, #2dd4bf)',
                  WebkitBackgroundClip: 'text',
                  backgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                Powered by GNNs
              </span>
            </h1>

            {/* Description */}
            <p
              data-hero="text"
              className="text-[0.95rem] leading-relaxed mb-2 max-w-[520px]"
              style={{ color: '#94a3b8' }}
            >
              Graph Neural Networks for Intelligent Network Traffic Analysis
            </p>
            <p
              data-hero="text"
              className="text-[0.85rem] leading-relaxed mb-9 max-w-[520px]"
              style={{ color: '#64748b' }}
            >
              Network flows are modelled as a flow-similarity graph using cosine similarity on{' '}
              <span style={{ color: '#94a3b8', fontWeight: 500 }}>ct_* neighbourhood features</span>.
              GraphSAGE classifies each node as{' '}
              <span style={{ color: '#2dd4bf', fontWeight: 600 }}>Normal</span> or{' '}
              <span style={{ color: '#f87171', fontWeight: 600 }}>Attack</span>.
            </p>

            {/* CTA buttons */}
            <div data-hero="text" className="flex flex-wrap gap-3 mb-12">
              <a
                href="#upload"
                className="group inline-flex items-center gap-2.5 px-7 py-3.5 rounded-xl font-semibold text-sm text-white transition-all duration-200"
                style={{
                  background: 'linear-gradient(135deg, #2563eb, #1d4ed8)',
                  border: '1px solid rgba(96,165,250,0.35)',
                  boxShadow: '0 0 24px rgba(59,130,246,0.35), inset 0 1px 0 rgba(255,255,255,0.07)',
                }}
                onMouseEnter={e => { e.currentTarget.style.boxShadow = '0 0 36px rgba(59,130,246,0.55), inset 0 1px 0 rgba(255,255,255,0.10)'; e.currentTarget.style.transform = 'translateY(-2px)' }}
                onMouseLeave={e => { e.currentTarget.style.boxShadow = '0 0 24px rgba(59,130,246,0.35), inset 0 1px 0 rgba(255,255,255,0.07)'; e.currentTarget.style.transform = '' }}
              >
                <UploadCloud className="w-4 h-4" />
                Upload CSV
                <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-0.5" />
              </a>
              <a
                href="#methodology"
                className="inline-flex items-center gap-2 px-7 py-3.5 rounded-xl font-semibold text-sm transition-all duration-200"
                style={{
                  background: 'rgba(13,21,32,0.8)',
                  border: '1px solid rgba(30,58,95,0.8)',
                  color: '#94a3b8',
                  backdropFilter: 'blur(8px)',
                }}
                onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(96,165,250,0.45)'; e.currentTarget.style.color = '#e2e8f0'; e.currentTarget.style.transform = 'translateY(-1px)' }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(30,58,95,0.8)'; e.currentTarget.style.color = '#94a3b8'; e.currentTarget.style.transform = '' }}
              >
                Explore Methodology
              </a>
            </div>

            {/* Metric cards row */}
            <div className="grid grid-cols-3 gap-3 w-full max-w-[520px]">
              {METRICS.map(m => (
                <div
                  key={m.label}
                  data-hero="metric"
                  className="flex flex-col gap-2 rounded-xl px-4 py-4 transition-all duration-200"
                  style={{
                    background: 'rgba(13,21,32,0.75)',
                    border: `1px solid ${m.color}28`,
                    backdropFilter: 'blur(8px)',
                    boxShadow: `0 0 18px ${m.glow}`,
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = m.color + '55'; e.currentTarget.style.transform = 'translateY(-2px)' }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = m.color + '28'; e.currentTarget.style.transform = '' }}
                >
                  <m.icon className="w-4 h-4 opacity-70" style={{ color: m.color }} />
                  <div className="text-xl font-bold font-mono leading-none" style={{ color: m.color }}>
                    {m.value}
                  </div>
                  <div className="text-[11px] font-semibold text-white leading-tight">{m.label}</div>
                  <div className="text-[10px] leading-tight" style={{ color: '#64748b' }}>{m.sub}</div>
                </div>
              ))}
            </div>
          </div>

          {/* ── RIGHT COLUMN — Digital Earth ─────────────────────────────── */}
          <div
            data-hero="globe"
            className="relative flex-shrink-0 flex items-center justify-center"
            style={{ width: globeSize, height: globeSize }}
          >
            {/* extra outer ambient glow ring */}
            <div
              className="absolute inset-0 rounded-full pointer-events-none"
              style={{
                background: 'radial-gradient(circle, rgba(30,100,255,0.10) 0%, rgba(56,189,248,0.04) 50%, transparent 70%)',
                transform: 'scale(1.5)',
              }}
            />

            <DigitalGlobe size={globeSize} />

            {/* Floating threat badge */}
            <div
              data-hero="badge"
              className="absolute flex items-center gap-2 px-3.5 py-2 rounded-full text-xs font-semibold"
              style={{
                top: '12%', right: '-2%',
                background: 'rgba(248,113,113,0.10)',
                border: '1px solid rgba(248,113,113,0.35)',
                color: '#f87171',
                backdropFilter: 'blur(8px)',
                boxShadow: '0 0 16px rgba(248,113,113,0.15)',
                animation: 'float 5s ease-in-out infinite',
                animationDelay: '0.4s',
              }}
            >
              <span
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{
                  background: '#f87171',
                  boxShadow: '0 0 8px rgba(248,113,113,0.9)',
                  animation: 'blink 1.1s step-end infinite',
                }}
              />
              7 threat nodes — decorative
            </div>

            {/* Floating normal badge */}
            <div
              data-hero="badge"
              className="absolute flex items-center gap-2 px-3.5 py-2 rounded-full text-xs font-semibold"
              style={{
                bottom: '15%', left: '-4%',
                background: 'rgba(20,184,166,0.09)',
                border: '1px solid rgba(20,184,166,0.30)',
                color: '#2dd4bf',
                backdropFilter: 'blur(8px)',
                animation: 'float 7s ease-in-out infinite',
                animationDelay: '1.0s',
              }}
            >
              <span
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ background: '#2dd4bf', boxShadow: '0 0 8px rgba(20,184,166,0.8)' }}
              />
              Flow similarity graph
            </div>

            {/* Orbit ring decoration */}
            <div
              className="absolute inset-0 rounded-full pointer-events-none"
              style={{
                border: '1px solid rgba(56,189,248,0.09)',
                transform: 'scale(1.18) rotate(-18deg)',
              }}
            />
            <div
              className="absolute inset-0 rounded-full pointer-events-none"
              style={{
                border: '1px solid rgba(99,130,255,0.06)',
                transform: 'scale(1.34) rotate(12deg)',
              }}
            />
          </div>
        </div>
      </div>

      {/* scroll indicator */}
      <a
        href="#methodology"
        data-hero="scroll"
        className="absolute bottom-8 left-1/2 -translate-x-1/2 z-10 flex flex-col items-center gap-1.5 transition-opacity"
        style={{ color: 'rgba(148,163,184,0.45)', opacity: 0 }}
        aria-label="Scroll to methodology"
        onMouseEnter={e => { e.currentTarget.style.color = 'rgba(148,163,184,0.85)' }}
        onMouseLeave={e => { e.currentTarget.style.color = 'rgba(148,163,184,0.45)' }}
      >
        <span className="text-[10px] font-medium tracking-widest uppercase">scroll</span>
        <ChevronDown className="w-4 h-4 animate-bounce" />
      </a>
    </section>
  )
}
