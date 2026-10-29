import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

// ═══════════════════════════════════════════════════════════════════════════
// useGsapReveal — scroll-triggered reveal animation for a section
// Adds opacity 0 → 1 and translateY 30px → 0 when the element enters viewport.
// Respects prefers-reduced-motion.
// ═══════════════════════════════════════════════════════════════════════════
export function useGsapReveal(options = {}) {
  const ref = useRef(null)
  const { y = 30, stagger = 0, delay = 0 } = options

  useEffect(() => {
    const el = ref.current
    if (!el) return

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (prefersReducedMotion) {
      gsap.set(el, { opacity: 1, y: 0 })
      return
    }

    const targets = stagger > 0 ? el.children : el

    const ctx = gsap.context(() => {
      gsap.fromTo(
        targets,
        { opacity: 0, y },
        {
          opacity: 1,
          y: 0,
          duration: 0.7,
          delay,
          stagger,
          ease: 'power2.out',
          scrollTrigger: {
            trigger: el,
            start: 'top 85%',
            once: true,
          },
        }
      )
    }, el)

    return () => ctx.revert()
  }, [y, stagger, delay])

  return ref
}

// ═══════════════════════════════════════════════════════════════════════════
// useAnimatedCounter — animates a number from 0 to target on scroll trigger
// ═══════════════════════════════════════════════════════════════════════════
export function useAnimatedCounter(target, options = {}) {
  const ref = useRef(null)
  const { duration = 1.5, suffix = '', decimals = 0 } = options

  useEffect(() => {
    const el = ref.current
    if (!el) return

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (prefersReducedMotion) {
      el.textContent = target.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) + suffix
      return
    }

    const obj = { val: 0 }
    const ctx = gsap.context(() => {
      gsap.to(obj, {
        val: target,
        duration,
        ease: 'power2.out',
        scrollTrigger: {
          trigger: el,
          start: 'top 85%',
          once: true,
        },
        onUpdate: () => {
          el.textContent = obj.val.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) + suffix
        },
      })
    }, el)

    return () => ctx.revert()
  }, [target, duration, suffix, decimals])

  return ref
}
