/**
 * Methodology.jsx
 *
 * Fix: GSAP's gsap.from() with { opacity:0 } sets cards invisible immediately
 * via immediateRender:true.  If ScrollTrigger never fires (section already
 * visible, reduced-motion, init error) cards stay at opacity 0 forever.
 *
 * Fix applied:
 *  • Cards start visible in CSS (no initial opacity:0 class)
 *  • Animation uses gsap.fromTo with immediateRender:false so GSAP never
 *    hides the cards before the trigger fires
 *  • Try/catch around GSAP setup so any error leaves cards fully visible
 *  • ScrollTrigger.refresh() called after a short delay to handle cases where
 *    the section is already in the viewport on page load
 */
import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import {
  Database, GitBranch, Network, Cpu, BarChart3, FlaskConical, Info,
} from 'lucide-react'

gsap.registerPlugin(ScrollTrigger)

const STEPS = [
  {
    icon: Database,
    step: '01',
    title: 'Dataset',
    accent: '#38bdf8',
    desc: 'UNSW-NB15: 175,341 training + 82,332 test flow records. 45 raw columns, 9 attack categories + normal class, zero missing values.',
    tag: 'UNSW-NB15',
  },
  {
    icon: GitBranch,
    step: '02',
    title: 'Preprocessing',
    accent: '#818cf8',
    desc: 'ID column dropped. StandardScaler on 39 numerical features, OrdinalEncoder on 3 categorical (proto, service, state). 42 model-input features total.',
    tag: '42 features',
  },
  {
    icon: Network,
    step: '03',
    title: 'Graph Construction',
    accent: '#14b8a6',
    desc: 'Flows grouped into 100-flow temporal windows. k=5 directed k-NN edges per node using cosine similarity over 8 ct_* neighbourhood features. Not IP communication topology.',
    tag: 'k=5, ct_* similarity',
  },
  {
    icon: Cpu,
    step: '04',
    title: 'GraphSAGE Training',
    accent: '#3b82f6',
    desc: 'Two SAGEConv layers (42→64→64) + Linear binary classifier. 13,826 parameters. Adam optimiser, 50 epochs, weighted CrossEntropyLoss for class imbalance.',
    tag: '13.8K params',
  },
  {
    icon: BarChart3,
    step: '05',
    title: 'Evaluation',
    accent: '#f59e0b',
    desc: 'Tested once on 82,332 held-out flows. GraphSAGE F1=0.8994, Recall=0.9601. Random Forest baseline on the same 42 features achieves F1=0.8941.',
    tag: 'F1 = 0.8994',
  },
  {
    icon: FlaskConical,
    step: '06',
    title: 'Ablation Study',
    accent: '#f87171',
    desc: 'Three conditions: (A) original k-NN graph, (B) no edges — node features only, (C) degree-preserving random rewiring. Isolates the contribution of graph structure.',
    tag: '3 conditions',
  },
]

export default function Methodology() {
  const sectionRef = useRef(null)
  const cardsRef   = useRef(null)

  useEffect(() => {
    // Guard: don't animate if reduced motion is preferred
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return

    let ctx
    try {
      ctx = gsap.context(() => {
        // Heading reveal
        gsap.fromTo(
          '.meth-heading',
          { opacity: 0, y: 28 },
          {
            opacity: 1, y: 0,
            duration: 0.7,
            ease: 'power3.out',
            immediateRender: false,   // ← never hides element before trigger
            scrollTrigger: {
              trigger: '.meth-heading',
              start: 'top 88%',
              once: true,
            },
          }
        )

        // Card stagger reveal
        gsap.fromTo(
          '.meth-card',
          { opacity: 0, y: 32 },
          {
            opacity: 1, y: 0,
            duration: 0.55,
            stagger: 0.08,
            ease: 'power3.out',
            immediateRender: false,   // ← cards stay visible until trigger fires
            scrollTrigger: {
              trigger: cardsRef.current,
              start: 'top 85%',
              once: true,
            },
          }
        )
      }, sectionRef.current)

      // Refresh ScrollTrigger after a tick so elements already in view fire correctly
      const t = setTimeout(() => ScrollTrigger.refresh(), 120)
      return () => {
        clearTimeout(t)
        ctx.revert()
      }
    } catch (err) {
      // If GSAP/ScrollTrigger throws for any reason, leave cards fully visible
      console.warn('[Methodology] GSAP init error — cards will be static:', err)
      ctx?.revert()
    }
  }, [])

  return (
    <section ref={sectionRef} id="methodology" className="relative py-24 px-6">
      <div className="max-w-6xl mx-auto">

        {/* Heading — visible by default */}
        <div className="meth-heading text-center mb-14">
          <div className="section-badge mx-auto mb-4">
            <FlaskConical className="w-3 h-3" />
            Methodology
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white mb-3">
            How It Works
          </h2>
          <p className="text-cyber-muted max-w-xl mx-auto text-sm leading-relaxed">
            A complete ML pipeline — from raw network captures to graph-based intrusion predictions.
          </p>
        </div>

        {/* Cards — visible by default; GSAP adds entrance animation on top */}
        <div
          ref={cardsRef}
          className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5"
        >
          {STEPS.map((s, i) => (
            <div
              key={i}
              className="meth-card relative rounded-2xl p-6 transition-all duration-300 hover:-translate-y-1"
              style={{
                /* opacity intentionally NOT set here — GSAP fromTo handles it
                   with immediateRender:false so the card is always visible      */
                background: 'rgba(17,30,48,0.85)',
                border: '1px solid rgba(30,58,95,0.7)',
                boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.borderColor = s.accent + '55'
                e.currentTarget.style.boxShadow   = `0 8px 32px rgba(0,0,0,0.4), 0 0 20px ${s.accent}12`
              }}
              onMouseLeave={e => {
                e.currentTarget.style.borderColor = 'rgba(30,58,95,0.7)'
                e.currentTarget.style.boxShadow   = '0 4px 20px rgba(0,0,0,0.3)'
              }}
            >
              {/* Step number */}
              <div
                className="absolute top-4 right-4 text-[11px] font-mono font-bold"
                style={{ color: s.accent + '60' }}
              >
                {s.step}
              </div>

              {/* Icon */}
              <div
                className="w-11 h-11 rounded-xl flex items-center justify-center mb-4"
                style={{ background: s.accent + '15', border: `1px solid ${s.accent}30` }}
              >
                <s.icon className="w-5 h-5" style={{ color: s.accent }} />
              </div>

              <h3 className="text-base font-bold text-white mb-2">{s.title}</h3>
              <p className="text-sm text-cyber-muted leading-relaxed mb-4">{s.desc}</p>

              <span
                className="inline-block text-[10px] font-semibold px-2.5 py-1 rounded-full"
                style={{
                  background: s.accent + '12',
                  color: s.accent,
                  border: `1px solid ${s.accent}25`,
                }}
              >
                {s.tag}
              </span>
            </div>
          ))}
        </div>

        {/* Disclaimer */}
        <div
          className="mt-8 flex items-start gap-3 px-5 py-4 rounded-xl"
          style={{ background: 'rgba(251,191,36,0.05)', border: '1px solid rgba(251,191,36,0.18)' }}
        >
          <Info className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <p className="text-xs text-cyber-muted leading-relaxed">
            <strong className="text-amber-400">Flow-similarity graph, not IP topology.</strong>{' '}
            UNSW-NB15 does not expose source/destination IPs in our representation.
            Edges connect flows with similar ct_* neighbourhood statistics via cosine
            similarity — they do not represent actual network communication paths.
          </p>
        </div>
      </div>
    </section>
  )
}
