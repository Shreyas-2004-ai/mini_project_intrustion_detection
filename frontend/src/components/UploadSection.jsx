import { useState, useCallback, useRef, useEffect } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import {
  UploadCloud, FileText, AlertCircle, CheckCircle2,
  Loader2, X, Info, ChevronRight,
} from 'lucide-react'
import { SAMPLE_SUMMARY, SAMPLE_FLOWS } from '../data/sampleData'

gsap.registerPlugin(ScrollTrigger)

function fmtBytes(b) {
  if (b < 1024) return `${b} B`
  if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / 1048576).toFixed(2)} MB`
}

export default function UploadSection({ onDataLoaded }) {
  const [isDragging, setIsDragging]   = useState(false)
  const [file,       setFile]         = useState(null)
  const [status,     setStatus]       = useState('idle')   // idle | loading | success | error
  const [errorMsg,   setErrorMsg]     = useState('')
  const [progress,   setProgress]     = useState(0)
  const fileInputRef = useRef(null)
  const sectionRef   = useRef(null)

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.from('.upload-panel', {
        opacity: 0, y: 36,
        duration: 0.7, ease: 'power3.out',
        scrollTrigger: { trigger: '.upload-panel', start: 'top 82%' },
      })
    }, sectionRef.current)
    return () => ctx.revert()
  }, [])

  const processFile = useCallback(async (f) => {
    if (!f) return

    // Validation
    if (!f.name.endsWith('.csv')) {
      setErrorMsg('Please upload a .csv file.')
      setStatus('error')
      return
    }
    if (f.size > 50 * 1024 * 1024) {
      setErrorMsg('File is too large (max 50 MB).')
      setStatus('error')
      return
    }

    setFile(f)
    setStatus('loading')
    setProgress(0)
    setErrorMsg('')

    // Simulate progress while fetching
    const ticker = setInterval(() => {
      setProgress(p => Math.min(p + Math.random() * 12, 85))
    }, 180)

    const formData = new FormData()
    formData.append('csv', f)

    try {
      const res = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        body: formData,
      })

      clearInterval(ticker)

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }))
        throw new Error(err.details || err.error || `Server error ${res.status}`)
      }

      const data = await res.json()
      setProgress(100)
      setTimeout(() => {
        setStatus('success')
        onDataLoaded(data, false)
        setTimeout(() => {
          document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' })
        }, 300)
      }, 300)

    } catch (err) {
      clearInterval(ticker)
      setErrorMsg(err.message || 'Analysis failed. Is the backend running?')
      setStatus('error')
    }
  }, [onDataLoaded])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) processFile(f)
  }, [processFile])

  const onFileChange = useCallback((e) => {
    const f = e.target.files?.[0]
    if (f) processFile(f)
  }, [processFile])

  const loadSample = () => {
    onDataLoaded({ summary: SAMPLE_SUMMARY, flows: SAMPLE_FLOWS, isSample: true }, true)
    setTimeout(() => document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' }), 100)
  }

  const reset = () => {
    setFile(null)
    setStatus('idle')
    setErrorMsg('')
    setProgress(0)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  return (
    <section ref={sectionRef} id="upload" className="relative py-24 px-6">
      <div className="max-w-3xl mx-auto">

        {/* Heading */}
        <div className="text-center mb-10">
          <div className="section-badge mx-auto">
            <UploadCloud className="w-3 h-3" />
            Upload & Predict
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white mb-3">
            Analyze Network Flows
          </h2>
          <p className="text-cyber-muted text-sm max-w-lg mx-auto leading-relaxed">
            Upload a CSV following the UNSW-NB15 schema (45 columns).
            The model classifies each flow and builds a k-NN similarity graph.
          </p>
        </div>

        {/* Upload card */}
        <div
          className="upload-panel rounded-2xl overflow-hidden"
          style={{
            background: 'rgba(17,30,48,0.85)',
            border: '1px solid rgba(30,58,95,0.7)',
            boxShadow: '0 8px 40px rgba(0,0,0,0.4)',
          }}
        >
          {/* Top bar */}
          <div
            className="flex items-center justify-between px-6 py-3.5"
            style={{ borderBottom: '1px solid rgba(30,58,95,0.5)' }}
          >
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-red-500/70" />
              <div className="w-2 h-2 rounded-full bg-yellow-500/70" />
              <div className="w-2 h-2 rounded-full bg-green-500/70" />
            </div>
            <span className="text-[11px] font-mono text-cyber-muted/60">
              POST /api/predict — multipart/form-data
            </span>
            <div className="w-16" />
          </div>

          <div className="p-6 sm:p-8">

            {/* Drop zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={onDrop}
              onClick={() => status !== 'loading' && fileInputRef.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click() }}
              aria-label="Drop zone: click or drag a CSV file here"
              className="relative rounded-xl p-8 sm:p-12 text-center transition-all duration-200 cursor-pointer"
              style={{
                border: `2px dashed ${isDragging ? 'rgba(59,130,246,0.6)' : 'rgba(30,58,95,0.7)'}`,
                background: isDragging ? 'rgba(59,130,246,0.05)' : 'rgba(13,21,32,0.5)',
                transform: isDragging ? 'scale(1.01)' : 'scale(1)',
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                className="hidden"
                onChange={onFileChange}
                aria-hidden="true"
              />

              {status === 'loading' ? (
                <div className="flex flex-col items-center gap-4">
                  <Loader2 className="w-10 h-10 text-cyber-accent animate-spin" />
                  <div className="text-sm font-medium text-cyber-muted">Analyzing flows…</div>
                  <div className="w-full max-w-xs h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(30,58,95,0.5)' }}>
                    <div
                      className="h-full rounded-full transition-all duration-300"
                      style={{
                        width: `${progress}%`,
                        background: 'linear-gradient(90deg, #3b82f6, #38bdf8)',
                      }}
                    />
                  </div>
                  <div className="text-xs text-cyber-muted/60 font-mono">{Math.round(progress)}%</div>
                </div>

              ) : status === 'success' ? (
                <div className="flex flex-col items-center gap-3">
                  <CheckCircle2 className="w-10 h-10 text-cyber-success" />
                  <div className="text-sm font-medium text-cyber-success">Analysis complete!</div>
                  <div className="flex items-center gap-2 text-xs text-cyber-muted">
                    <FileText className="w-3.5 h-3.5" />
                    {file?.name} · {file && fmtBytes(file.size)}
                  </div>
                  <button onClick={(e) => { e.stopPropagation(); reset() }}
                          className="mt-2 flex items-center gap-1.5 text-xs text-cyber-muted/60 hover:text-cyber-muted transition-colors">
                    <X className="w-3.5 h-3.5" /> Upload another file
                  </button>
                </div>

              ) : (
                <div className="flex flex-col items-center gap-3">
                  <div
                    className="w-16 h-16 rounded-2xl flex items-center justify-center mb-2"
                    style={{ background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.25)' }}
                  >
                    <UploadCloud className="w-8 h-8 text-cyber-accent" />
                  </div>
                  <div className="text-base font-semibold text-white">
                    {file ? file.name : 'Drop your CSV here'}
                  </div>
                  <div className="text-sm text-cyber-muted">
                    {file ? fmtBytes(file.size) : 'or click to browse'}
                  </div>
                  <div className="text-xs text-cyber-muted/50 mt-1">
                    UNSW-NB15 schema · 45 columns · max 50 MB
                  </div>
                </div>
              )}
            </div>

            {/* Error state */}
            {status === 'error' && (
              <div
                className="mt-4 flex items-start gap-3 px-4 py-3.5 rounded-xl"
                style={{ background: 'rgba(248,113,113,0.07)', border: '1px solid rgba(248,113,113,0.25)' }}
              >
                <AlertCircle className="w-4 h-4 text-cyber-danger flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-xs font-semibold text-cyber-danger mb-0.5">Analysis failed</p>
                  <p className="text-xs text-cyber-muted/80">{errorMsg}</p>
                </div>
                <button onClick={reset} className="ml-auto text-cyber-muted/50 hover:text-cyber-muted transition-colors">
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}

            {/* Actions */}
            <div className="mt-5 flex flex-col sm:flex-row gap-3">
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={status === 'loading'}
                className="btn-primary flex-1 justify-center disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {status === 'loading'
                  ? <><Loader2 className="w-4 h-4 animate-spin" /> Analyzing…</>
                  : <><UploadCloud className="w-4 h-4" /> Select CSV File</>
                }
              </button>
              <button
                onClick={loadSample}
                disabled={status === 'loading'}
                className="btn-secondary flex-1 justify-center disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Preview Sample Data
              </button>
            </div>

            {/* Schema hint */}
            <div
              className="mt-5 flex items-start gap-2.5 px-4 py-3 rounded-xl text-xs"
              style={{ background: 'rgba(59,130,246,0.05)', border: '1px solid rgba(59,130,246,0.15)' }}
            >
              <Info className="w-3.5 h-3.5 text-cyber-accent/70 flex-shrink-0 mt-0.5" />
              <p className="text-cyber-muted/70 leading-relaxed">
                <span className="font-medium text-cyber-muted">Required columns:</span>{' '}
                id, dur, proto, service, state, spkts, dpkts, sbytes, dbytes, …
                label and attack_cat are optional. Missing id is auto-generated.
              </p>
            </div>

            {/* Sample data warning */}
            <div className="mt-3 flex items-start gap-2 text-[11px] text-cyber-muted/50">
              <ChevronRight className="w-3 h-3 mt-0.5 flex-shrink-0" />
              Sample data uses fabricated values for layout preview — not actual model predictions.
              Real predictions appear only after uploading a valid CSV.
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
