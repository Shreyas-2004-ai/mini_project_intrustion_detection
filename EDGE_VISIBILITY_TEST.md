# Edge Visibility Test Report

## Test Procedure

1. **Start services** (if not running):
   ```bash
   # Terminal 1
   python backend_api.py
   
   # Terminal 2
   cd frontend && npm run dev
   ```

2. **Open browser**: http://localhost:5173

3. **Open DevTools**: Press F12

4. **Upload test file**: Upload `test_5_flows.csv`

5. **Check Console Output**:
   Look for these messages:
   ```
   GraphVisualization: Applying layout to graph data
   Layouted nodes: [...]
   Sample processed edge: {...}
   Total edges to render: 20
   ReactFlow initialized
     Nodes: 5 ['0', '1', '2', '3', '4']
     Edges: 20 ['0->3', '0->4', ...]
   fitView applied
   ```

6. **Inspect DOM**:
   - Switch to Elements tab
   - Press Ctrl+F
   - Search: `.react-flow__edge`
   - Count results (should be 20)

7. **Check Edge Styles**:
   - Click on first `.react-flow__edge` element
   - Find the `<path>` element inside
   - Check its attributes:
     - `d`: Should have path data (not empty)
     - `style` or `stroke`: Should be `#60a5fa` or `rgb(96, 165, 250)`
     - `stroke-width`: Should be `4`
     - `opacity`: Should be `0.9`

## Expected Results

### ✅ Success Indicators:
- [ ] Console shows "Total edges to render: 20"
- [ ] Console shows "ReactFlow initialized ... Edges: 20"
- [ ] DOM contains 20 `.react-flow__edge` elements
- [ ] Each edge `path` has visible stroke color
- [ ] Edges are visible on screen as cyan/blue lines
- [ ] Dragging a node causes edges to follow
- [ ] Clicking a node highlights connected edges

### ❌ Failure Indicators:
- [ ] Console shows edges: 0 or undefined
- [ ] DOM has fewer than 20 edge elements
- [ ] Edge paths have no `d` attribute or empty `d`
- [ ] Edge stroke color is black, transparent, or matches background
- [ ] Edges exist in DOM but not visible on screen

## Common Issues & Fixes

### Issue 1: Edges in DOM but not visible

**Symptoms:**
- 20 edge elements in DOM
- No visible lines on canvas
- Nodes are visible

**Root causes:**
1. **Stroke color issue**: Edges might be black or transparent
2. **Z-index issue**: Edges behind background or nodes
3. **Path data issue**: `d` attribute invalid

**Diagnosis:**
```javascript
// In browser console:
document.querySelectorAll('.react-flow__edge-path').forEach((el, i) => {
  console.log(`Edge ${i}:`, {
    d: el.getAttribute('d'),
    stroke: window.getComputedStyle(el).stroke,
    strokeWidth: window.getComputedStyle(el).strokeWidth,
    opacity: window.getComputedStyle(el).opacity
  })
})
```

**Fixes applied:**
- ✅ CSS: `.react-flow__edge-path { stroke: #60a5fa !important; stroke-width: 4 !important; }`
- ✅ CSS: `.react-flow__edge { z-index: 1 !important; }`
- ✅ Props: `elevateEdgesOnSelect={true}`
- ✅ Style: Explicit stroke color in edge definitions

### Issue 2: No edges in DOM

**Symptoms:**
- Search for `.react-flow__edge` returns 0 results
- Console shows edges being processed but not rendered

**Root causes:**
1. **Invalid edge data**: source/target don't match node IDs
2. **React Flow version issue**: v11 syntax differs from v10
3. **Node positions not set**: Edges can't be drawn without node positions

**Diagnosis:**
```javascript
// In browser console (after upload):
// Check if edges are in React Flow internal state
// This requires React DevTools extension
```

**Fixes applied:**
- ✅ Validate source/target match node IDs in backend
- ✅ Use circular layout with explicit positions
- ✅ Log node positions and edge connections
- ✅ Ensure React Flow v11 syntax

### Issue 3: Edges appear only after interaction

**Symptoms:**
- Initial load: no edges
- After clicking or dragging: edges appear

**Root causes:**
1. **Render timing**: React Flow needs time to calculate layout
2. **fitView timing**: Called before edges are ready

**Fixes applied:**
- ✅ `setTimeout(() => instance.fitView(), 100)` in onInit
- ✅ Initial positions set in circular layout
- ✅ `fitView` prop on ReactFlow component

## Manual Verification Steps

### Step 1: Visual Check
- [ ] Can you see cyan/blue lines connecting nodes?
- [ ] Do edges form connections between node pairs?
- [ ] Are there approximately 20 distinct edge lines?

### Step 2: Interaction Check
- [ ] Click and drag a node
- [ ] Do connected edges move with the node?
- [ ] Click on a node
- [ ] Do some edges become brighter/thicker?

### Step 3: Console Check
- [ ] Any error messages in console?
- [ ] Warning about invalid edge data?
- [ ] React Flow initialization message present?

### Step 4: DOM Check
- [ ] Right-click on canvas → Inspect
- [ ] Can you find `.react-flow__edges` container?
- [ ] Does it have 20 child elements?

## Reporting Issues

If edges are still not visible, report:

1. **Console output**: Copy all GraphVisualization logs
2. **DOM count**: Number of `.react-flow__edge` elements found
3. **Sample edge HTML**: Right-click edge element → Copy → Copy outerHTML
4. **Computed styles**: For one edge path element:
   ```javascript
   const path = document.querySelector('.react-flow__edge-path')
   console.log(window.getComputedStyle(path))
   ```
5. **Screenshot**: Show what you see vs what's expected
6. **Browser**: Chrome/Firefox/Edge version

## Expected Appearance

```
     Node 0 (Red)
      / | \ \
     /  |  \ \
    /   |   \ \
   /    |    \ \
  Node 1   Node 2  Node 4
(Teal)   (Red)   (Teal)
   \      |      /
    \     |     /
     \    |    /
      \ Node 3 /
         (Red)
```

Each node should have 4-5 cyan lines radiating from it (k=5 k-NN graph).
Total visual edges may appear fewer than 20 due to bidirectional edges overlapping.

## Success Criteria

The test passes if:
- ✅ 5 nodes are visible with correct colors
- ✅ At least 15 distinct edge lines are visible (some may overlap)
- ✅ Edges are cyan/blue color (#60a5fa)
- ✅ Dragging a node causes edges to follow
- ✅ Console shows 20 edges processed and rendered
- ✅ DOM contains 20 edge elements

## Current Status

**Date:** _____________  
**Tester:** _____________  
**Result:** [ ] PASS [ ] FAIL  

**Notes:**
_________________________________
_________________________________
_________________________________
