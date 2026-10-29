# Edge Visibility Debugging Guide

## Browser Console Checks

1. **Open Developer Tools (F12)**
2. **Go to Console tab**
3. **Look for these logs after uploading test_5_flows.csv:**

### Expected Console Output:
```
GraphVisualization: Applying layout to graph data
{
  nodeCount: 5,
  edgeCount: 20,
  sampleNode: {...},
  sampleEdge: {...}
}

Layouted nodes:
[
  {id: '0', x: 600, y: 140},
  {id: '1', x: 983, y: 385},
  ...
]

Sample processed edge:
{
  id: 'e0-3',
  source: '0',
  target: '3',
  type: 'default',
  style: { stroke: '#60a5fa', strokeWidth: 4, opacity: 0.9 },
  ...
}

Total edges to render: 20

ReactFlow initialized with 5 nodes and 20 edges
```

## DOM Inspection

1. **Go to Elements/Inspector tab**
2. **Search for:** `.react-flow__edge`
3. **You should see 20 elements with class `react-flow__edge`**

### Check Each Edge Element:

```html
<g class="react-flow__edge">
  <path 
    class="react-flow__edge-path"
    d="M... (path data)"
    style="stroke: rgb(96, 165, 250); stroke-width: 4; opacity: 0.9;"
    marker-end="url(#...)"
  />
</g>
```

## Verification Checklist

- [ ] Console shows 20 edges being processed
- [ ] Console shows ReactFlow initialized with 20 edges
- [ ] DOM has 20 `.react-flow__edge` elements
- [ ] Each edge `path` element has:
  - [ ] `d` attribute with path data (not empty)
  - [ ] `stroke` color visible (rgb(96, 165, 250) or #60a5fa)
  - [ ] `stroke-width` >= 3
  - [ ] `opacity` > 0.5
  - [ ] `marker-end` attribute (arrow)

## Common Issues

### Issue 1: Edges exist in DOM but not visible
**Symptom:** 20 edge elements in DOM, but canvas looks empty  
**Possible causes:**
- Path `d` attribute is invalid or zero-length
- Stroke color matches background
- Edges rendered behind nodes (z-index issue)
- CSS hiding edges

**Fix:** Check computed styles in DevTools

### Issue 2: No edge elements in DOM
**Symptom:** Cannot find `.react-flow__edge` in DOM  
**Possible causes:**
- Edges array is empty when passed to ReactFlow
- React Flow not rendering edges due to invalid data
- Node positions not set properly

**Fix:** Check console logs for edge count

### Issue 3: Edges only visible after interaction
**Symptom:** Edges appear after clicking or dragging  
**Possible causes:**
- React Flow needs manual trigger to render
- fitView() timing issue

**Fix:** Already added setTimeout in onInit

## Manual Test Steps

1. Upload `test_5_flows.csv`
2. Scroll to graph section
3. Open DevTools Console
4. Look for log: "Total edges to render: 20"
5. Switch to Elements tab
6. Press Ctrl+F and search: `.react-flow__edge`
7. Count results (should be 20)
8. Click first edge element
9. Check its `path` element's `style` attribute
10. Verify stroke color is visible blue
11. Try dragging a node - edges should follow

## React Flow Version Check

Current: reactflow ^11.11.4

Features used:
- ✅ useNodesState, useEdgesState
- ✅ MarkerType.ArrowClosed
- ✅ Custom node types
- ✅ defaultEdgeOptions
- ✅ fitView

All features supported in v11.11.4.

## If Edges Still Don't Show

Report in console:
1. Total edge count from logs
2. Sample edge object structure
3. DOM edge element count
4. Sample edge path `d` attribute value
5. Computed stroke color
6. Any React Flow errors

This info will help diagnose the root cause.
