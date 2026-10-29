import { useState, useEffect } from 'react'
import ReactFlow, { MarkerType } from 'reactflow'
import 'reactflow/dist/style.css'

/**
 * Minimal test component to verify React Flow edges render correctly
 * This uses hardcoded nodes and edges to isolate the rendering issue
 */
export default function TestGraphSimple() {
  const [nodes] = useState([
    {
      id: '0',
      type: 'default',
      position: { x: 250, y: 100 },
      data: { label: 'Node 0' },
    },
    {
      id: '1',
      type: 'default',
      position: { x: 100, y: 300 },
      data: { label: 'Node 1' },
    },
    {
      id: '2',
      type: 'default',
      position: { x: 400, y: 300 },
      data: { label: 'Node 2' },
    },
  ])

  const [edges] = useState([
    {
      id: 'e0-1',
      source: '0',
      target: '1',
      type: 'default',
      animated: false,
      style: {
        stroke: '#ff0000', // Bright red for visibility
        strokeWidth: 5,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: '#ff0000',
      },
    },
    {
      id: 'e0-2',
      source: '0',
      target: '2',
      type: 'default',
      animated: false,
      style: {
        stroke: '#00ff00', // Bright green
        strokeWidth: 5,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: '#00ff00',
      },
    },
    {
      id: 'e1-2',
      source: '1',
      target: '2',
      type: 'default',
      animated: false,
      style: {
        stroke: '#0000ff', // Bright blue
        strokeWidth: 5,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: '#0000ff',
      },
    },
  ])

  useEffect(() => {
    console.log('TestGraphSimple mounted with:', { nodes: nodes.length, edges: edges.length })
    console.log('Nodes:', nodes)
    console.log('Edges:', edges)
  }, [nodes, edges])

  return (
    <div className="w-full h-screen bg-gray-900 p-8">
      <div className="bg-white rounded-lg p-4 mb-4">
        <h2 className="text-xl font-bold text-gray-900">React Flow Edge Test</h2>
        <p className="text-gray-700">
          This test uses 3 nodes and 3 edges with bright colors (red, green, blue).
        </p>
        <p className="text-gray-700">
          If you can't see 3 thick colored lines connecting the nodes, there's a React Flow rendering issue.
        </p>
        <p className="text-sm text-gray-600 mt-2">
          Nodes: {nodes.length} | Edges: {edges.length}
        </p>
      </div>
      
      <div className="w-full h-[600px] border-4 border-red-500 bg-gray-800 rounded-lg">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          defaultEdgeOptions={{
            style: { strokeWidth: 5 },
          }}
        />
      </div>
    </div>
  )
}
