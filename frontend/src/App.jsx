import { useState } from 'react'
import AnimatedBackground from './components/AnimatedBackground'
import Header             from './components/Header'
import Hero               from './components/Hero'
import Methodology        from './components/Methodology'
import UploadSection      from './components/UploadSection'
import ResultsDashboard   from './components/ResultsDashboard'
import FlowTable          from './components/FlowTable'
import GraphVisualization from './components/GraphVisualization'
import ModelComparison    from './components/ModelComparison'
import Footer             from './components/Footer'

export default function App() {
  const [resultData, setResultData] = useState(null)
  const [isSample,   setIsSample]   = useState(false)
  const [graphData,  setGraphData]  = useState(null)

  const handleDataLoaded = (data, sample) => {
    setResultData(data)
    setIsSample(sample)
    setGraphData(data.graph || null)
  }

  return (
    <div className="min-h-screen" style={{ background: '#060b14', color: '#e2e8f0' }}>
      {/* Fixed canvas background — always behind everything */}
      <AnimatedBackground />

      {/* Sticky header */}
      <Header />

      {/* Page content — z-10 so it sits above canvas */}
      <main className="relative" style={{ zIndex: 10 }}>
        <Hero />
        <Methodology />
        <UploadSection onDataLoaded={handleDataLoaded} />

        {resultData && (
          <>
            <ResultsDashboard data={resultData} isSample={isSample} />
            <FlowTable flows={resultData.flows} />
          </>
        )}

        {/* Graph section always visible; shows empty state until data loaded */}
        <GraphVisualization graphData={graphData} />

        <ModelComparison />
        <Footer />
      </main>
    </div>
  )
}
