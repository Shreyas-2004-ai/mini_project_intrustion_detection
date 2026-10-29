import { useState, useEffect, useRef } from 'react'
import { Shield, Menu, X, Upload, Network, BarChart2, BookOpen, Home } from 'lucide-react'

const NAV = [
  { href: '#home',        label: 'Home',        icon: Home },
  { href: '#methodology', label: 'Methodology',  icon: BookOpen },
  { href: '#upload',      label: 'Upload',       icon: Upload },
  { href: '#graph-viz',   label: 'Graph',        icon: Network },
  { href: '#results',     label: 'Results',      icon: BarChart2 },
]

export default function Header() {
  const [scrolled, setScrolled]   = useState(false)
  const [active,   setActive]     = useState('#home')
  const [mobileOpen, setMobileOpen] = useState(false)
  const headerRef = useRef(null)

  // Scroll shadow
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  // Active section tracking
  useEffect(() => {
    const ids = NAV.map(n => n.href.slice(1))
    const obs = new IntersectionObserver(
      entries => {
        const visible = entries.filter(e => e.isIntersecting)
        if (visible.length) {
          const top = visible.reduce((a, b) => a.boundingClientRect.top < b.boundingClientRect.top ? a : b)
          setActive('#' + top.target.id)
        }
      },
      { threshold: 0.3 }
    )
    ids.forEach(id => { const el = document.getElementById(id); if (el) obs.observe(el) })
    return () => obs.disconnect()
  }, [])

  const handleNav = (href) => {
    setActive(href)
    setMobileOpen(false)
    const el = document.querySelector(href)
    if (!el) return
    // Account for the 72px fixed header
    const top = el.getBoundingClientRect().top + window.scrollY - 72
    window.scrollTo({ top: Math.max(0, top), behavior: 'smooth' })
  }

  return (
    <>
      <header
        ref={headerRef}
        className="fixed top-0 left-0 right-0 z-50 transition-all duration-300"
        style={{
          background: scrolled
            ? 'rgba(6,11,20,0.92)'
            : 'rgba(6,11,20,0.70)',
          backdropFilter: 'blur(14px)',
          WebkitBackdropFilter: 'blur(14px)',
          borderBottom: scrolled
            ? '1px solid rgba(30,58,95,0.5)'
            : '1px solid transparent',
          boxShadow: scrolled ? '0 4px 24px rgba(0,0,0,0.4)' : 'none',
        }}
      >
        <div className="max-w-7xl mx-auto px-6 lg:px-10 h-[72px] flex items-center justify-between">

          {/* Logo */}
          <a href="#home" className="flex items-center gap-2.5 group" onClick={() => handleNav('#home')}>
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center transition-all group-hover:shadow-glow-sm"
              style={{
                background: 'linear-gradient(135deg, rgba(59,130,246,0.25), rgba(139,92,246,0.15))',
                border: '1px solid rgba(59,130,246,0.4)',
              }}
            >
              <Shield className="w-4 h-4 text-cyber-accent-light" />
            </div>
            <div className="hidden sm:block">
              <div className="text-sm font-bold text-white leading-none">GNN Intrusion</div>
              <div className="text-[10px] text-cyber-muted leading-none mt-0.5">Detection System</div>
            </div>
          </a>

          {/* Desktop nav */}
          <nav className="hidden md:flex items-center gap-1">
            {NAV.map(({ href, label }) => (
              <a
                key={href}
                href={href}
                onClick={(e) => { e.preventDefault(); handleNav(href) }}
                className="relative px-3.5 py-2 rounded-lg text-sm font-medium transition-all duration-200"
                style={{
                  color: active === href ? '#93c5fd' : '#94a3b8',
                  background: active === href ? 'rgba(59,130,246,0.1)' : 'transparent',
                }}
                onMouseEnter={e => { if (active !== href) e.currentTarget.style.color = '#e2e8f0' }}
                onMouseLeave={e => { if (active !== href) e.currentTarget.style.color = '#94a3b8' }}
              >
                {label}
                {active === href && (
                  <span
                    className="absolute bottom-1 left-1/2 -translate-x-1/2 w-1 h-1 rounded-full"
                    style={{ background: '#60a5fa' }}
                  />
                )}
              </a>
            ))}
          </nav>

          {/* Right side */}
          <div className="flex items-center gap-3">
            {/* System ready indicator */}
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full"
                 style={{ background: 'rgba(20,184,166,0.08)', border: '1px solid rgba(20,184,166,0.2)' }}>
              <span className="status-dot" />
              <span className="text-[11px] font-medium text-cyber-success">System Ready</span>
            </div>

            {/* Upload CTA */}
            <a
              href="#upload"
              onClick={(e) => { e.preventDefault(); handleNav('#upload') }}
              className="hidden md:flex btn-primary py-2 px-4 text-xs"
            >
              <Upload className="w-3.5 h-3.5" />
              Analyze
            </a>

            {/* Mobile menu button */}
            <button
              className="md:hidden w-9 h-9 rounded-lg flex items-center justify-center transition-colors"
              style={{ border: '1px solid #1e3a5f', background: 'rgba(17,30,48,0.8)' }}
              onClick={() => setMobileOpen(v => !v)}
              aria-label="Toggle menu"
            >
              {mobileOpen ? <X className="w-4 h-4 text-cyber-muted" /> : <Menu className="w-4 h-4 text-cyber-muted" />}
            </button>
          </div>
        </div>
      </header>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 md:hidden"
          onClick={() => setMobileOpen(false)}
        >
          <div
            className="absolute top-[72px] left-0 right-0 py-3 px-4"
            style={{
              background: 'rgba(6,11,20,0.97)',
              backdropFilter: 'blur(16px)',
              borderBottom: '1px solid rgba(30,58,95,0.6)',
            }}
            onClick={e => e.stopPropagation()}
          >
            {NAV.map(({ href, label, icon: Icon }) => (
              <a
                key={href}
                href={href}
                onClick={(e) => { e.preventDefault(); handleNav(href) }}
                className="flex items-center gap-3 px-4 py-3.5 rounded-xl mb-1 transition-all"
                style={{
                  background: active === href ? 'rgba(59,130,246,0.1)' : 'transparent',
                  color: active === href ? '#93c5fd' : '#94a3b8',
                  border: active === href ? '1px solid rgba(59,130,246,0.2)' : '1px solid transparent',
                }}
              >
                <Icon className="w-4 h-4" />
                <span className="text-sm font-medium">{label}</span>
              </a>
            ))}
            <div className="mt-3 pt-3 border-t border-cyber-border/50">
              <a
                href="#upload"
                onClick={(e) => { e.preventDefault(); handleNav('#upload') }}
                className="btn-primary w-full justify-center"
              >
                <Upload className="w-4 h-4" />
                Upload & Analyze
              </a>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
