# Manual Testing Checklist for Graph Visualization

**URL:** http://localhost:5174  
**Test File:** test_5_flows.csv

---

## Pre-Test Setup

1. **Start Backend:**
   ```bash
   python backend_api.py
   ```
   ✅ Should see: "Running on http://localhost:5000"

2. **Start Frontend:**
   ```bash
   cd frontend
   npm run dev
   ```
   ✅ Should see: "Local: http://localhost:5174"

3. **Open Browser:**
   Navigate to http://localhost:5174

---

## Test Sequence

### 1. Initial Page Load ✓

**Expected:**
- Hero section with "GNN-based Network Intrusion Detection"
- Methodology section
- Upload section with drag-drop zone
- "No Graph Data" placeholder in graph section
- Footer

**Check:**
- [ ] Page loads without errors
- [ ] Navigation menu works (smooth scroll to sections)
- [ ] "System Ready" indicator shows green dot

---

### 2. CSV Upload ✓

**Action:** Click upload zone and select `test_5_flows.csv`

**Expected During Upload:**
- Spinner animation
- "Analyzing network flows…" message

**Expected After Upload:**
- Smooth scroll to results section
- Results Dashboard appears with:
  - Total Flows: 5
  - Detected Attacks: 3 (60%)
  - Confidence scores
- Flow Table with 5 rows
- Graph Visualization section populated

**Check:**
- [ ] Upload completes in 2-3 seconds
- [ ] No console errors
- [ ] Results scroll into view automatically
- [ ] Summary statistics match test results (3 attacks, 2 normal)

---

### 3. Graph Visualization - Initial View ✓

**Expected:**
- 75% graph canvas + 25% sidebar layout
- 5 nodes visible with labels F0, F1, F2, F3, F4
- 20 cyan edges connecting nodes
- Animated background particles (floating blurred circles)
- MiniMap in bottom-left corner
- Controls in left sidebar
- Custom zoom buttons in top-right

**Check:**
- [ ] All 5 nodes are visible
- [ ] All 20 edges are visible as cyan/blue lines
- [ ] Nodes have correct colors:
  - **Red nodes (Attack):** F0, F2, F3
  - **Teal nodes (Normal):** F1, F4
- [ ] Edges have arrow markers at endpoints
- [ ] Background has subtle dot pattern

---

### 4. Node Appearance ✓

**For each node, verify:**
- Circular shape (70x70px)
- Colored border (3px solid)
- Glowing shadow effect
- Label in center (F0, F1, etc.)
- Hover effect (scale up slightly)

**Color Verification:**
| Node | Prediction | Border Color | Background |
|------|------------|--------------|------------|
| F0   | Attack     | Red          | Red tint   |
| F1   | Normal     | Teal         | Teal tint  |
| F2   | Attack     | Red          | Red tint   |
| F3   | Attack     | Red          | Red tint   |
| F4   | Normal     | Teal         | Teal tint  |

**Check:**
- [ ] Node colors match predictions
- [ ] Hover effect works on all nodes
- [ ] Labels are readable

---

### 5. Edge Appearance ✓

**Expected:**
- Cyan/blue color (#60a5fa or #3b82f6)
- Width: 2.5-4px (varies by similarity)
- Opacity: 0.6-0.8
- Arrow markers at target end
- Smooth rendering (no jagged lines)

**Check:**
- [ ] All 20 edges are clearly visible
- [ ] Edges connect correct node pairs
- [ ] Edge color contrasts well with background
- [ ] No overlapping edge issues

---

### 6. Interactive Features ✓

#### A. Node Selection

**Action:** Click on Node 0 (F0 - Attack)

**Expected:**
- Node scales up (1.1x)
- Node gets brighter
- Pulse animation around node
- Connected edges highlight (thicker, brighter blue)
- Sidebar updates with node details:
  - Shield icon (red for attack)
  - "Flow 0"
  - "Attack" label in red
  - Confidence: ~70%
  - P(Attack) bar: ~70%
  - P(Normal) bar: ~30%
  - Connected Edges: 5 (k=5)

**Check:**
- [ ] Node selection visual feedback works
- [ ] Connected edges highlight correctly
- [ ] Sidebar shows correct flow details
- [ ] Probability bars animate smoothly
- [ ] Confidence percentage matches max probability

#### B. Different Node Selection

**Action:** Click on Node 1 (F1 - Normal)

**Expected:**
- Previous node deselects
- Node 1 scales up and glows
- Sidebar updates:
  - Shield icon (green/teal for normal)
  - "Flow 1"
  - "Normal" label in teal
  - Confidence: ~51%
  - P(Attack) bar: ~49%
  - P(Normal) bar: ~51%

**Check:**
- [ ] Node selection switches correctly
- [ ] Previous node returns to normal state
- [ ] Sidebar reflects new selection
- [ ] Colors match prediction (teal for normal)

#### C. Deselection

**Action:** Click on empty canvas area

**Expected:**
- Selected node returns to normal state
- All edges return to default styling
- Sidebar shows "Click a node to view details"

**Check:**
- [ ] Deselection works by clicking canvas
- [ ] All visual effects reset

---

### 7. Edge Hover ✓

**Action:** Hover over any edge

**Expected:**
- Tooltip appears at bottom center of canvas
- Shows: "Flow X → Flow Y | ZZ% similarity"
- Example: "Flow 0 → Flow 3 | 99.6% similarity"
- Tooltip has dark background with glowing border

**Check:**
- [ ] Edge hover tooltip appears
- [ ] Shows correct source and target flow IDs
- [ ] Shows similarity percentage
- [ ] Tooltip follows mouse (or stays centered)

---

### 8. Zoom and Pan Controls ✓

#### A. Custom Zoom Buttons

**Actions:**
1. Click "+" button (top-right)
2. Click "-" button (top-right)
3. Click maximize button (fit view)

**Expected:**
- Zoom in: Graph enlarges smoothly
- Zoom out: Graph shrinks smoothly
- Fit view: Graph centers and scales to fit canvas

**Check:**
- [ ] Zoom in works
- [ ] Zoom out works
- [ ] Fit view centers graph
- [ ] Buttons have hover effects

#### B. Built-in Controls

**Action:** Use React Flow controls (left sidebar)
- Zoom in button
- Zoom out button
- Fit view button
- Lock button

**Check:**
- [ ] Built-in controls work
- [ ] Controls have cybersecurity theme styling

#### C. Mouse Interactions

**Actions:**
1. Scroll wheel to zoom
2. Click and drag canvas to pan

**Check:**
- [ ] Mouse wheel zoom works
- [ ] Drag to pan works
- [ ] Smooth interactions

---

### 9. MiniMap ✓

**Expected:**
- Located in bottom-left corner
- Shows full graph overview
- Red dots for attack nodes
- Teal dots for normal nodes
- Dark background
- Clickable for navigation

**Action:** Click on MiniMap

**Expected:** Viewport moves to clicked area

**Check:**
- [ ] MiniMap displays correctly
- [ ] Node colors match main graph
- [ ] Clickable navigation works
- [ ] Viewport rectangle visible

---

### 10. Sidebar Panels ✓

#### A. Node Details Panel (Top)
- Title: "Node Details" with activity icon
- Shows selected node info or placeholder

**Check:**
- [ ] Panel styling matches cybersecurity theme
- [ ] Placeholder shows when no node selected
- [ ] Details appear on node selection

#### B. Graph Statistics Panel (Middle)
- Title: "Graph Statistics"
- Shows:
  - Nodes: 5
  - Edges: 20
  - Avg. Degree: 4.0

**Check:**
- [ ] Statistics are correct
- [ ] Numbers match actual graph

#### C. Legend Panel (Bottom)
- Title: "Legend"
- Shows:
  - Teal circle: "Normal Flow"
  - Red circle: "Attack Flow"
  - Blue line: "Similarity Edge"

**Check:**
- [ ] Legend is clear and accurate
- [ ] Colors match actual graph colors

---

### 11. Layout Quality ✓

**Expected:**
- Nodes are well-spaced (not overlapping)
- Graph uses available canvas space
- Force-directed layout creates natural clustering
- Connected nodes are close together
- No nodes off-canvas

**Check:**
- [ ] Layout looks professional
- [ ] No node overlaps
- [ ] No edge crossings (or minimal)
- [ ] Graph is centered

---

### 12. Responsive Design ✓

**Action:** Resize browser window

**Expected:**
- Layout adapts to screen size
- On desktop (>1024px): 75/25 split (graph + sidebar)
- On tablet/mobile: Stacked layout (graph on top, sidebar below)

**Check:**
- [ ] Desktop layout works (side-by-side)
- [ ] Mobile layout works (stacked)
- [ ] Graph canvas resizes appropriately

---

### 13. Performance ✓

**Expected:**
- No lag when interacting with graph
- Smooth animations (60 FPS)
- Quick upload processing (~2-3 seconds)
- No memory leaks (check DevTools)

**Check:**
- [ ] Interactions are smooth
- [ ] No visible lag or stuttering
- [ ] Upload is fast
- [ ] No console warnings/errors

---

### 14. Console Verification ✓

**Action:** Open browser DevTools (F12) → Console tab

**Expected:**
- Log: "GraphVisualization: Applying layout to graph data"
- Log: "{ nodeCount: 5, edgeCount: 20 }"
- No errors or warnings

**Check:**
- [ ] Console shows layout application
- [ ] No errors in console
- [ ] No warnings (or only minor)

---

### 15. Multiple Uploads ✓

**Action:** Upload `test_5_flows.csv` again (or different CSV)

**Expected:**
- Previous graph clears
- New graph renders
- Layout recalculates
- Node predictions may differ (model re-inference)

**Check:**
- [ ] Graph updates correctly
- [ ] No stale data from previous upload
- [ ] State resets properly

---

### 16. Edge Verification ✓

**Action:** Count visible edges manually or use MiniMap

**Expected Count:** 20 edges (5 nodes × k=5 k-NN, but undirected)

**Sample Edge Pairs from Backend:**
- F0 → F1, F0 → F2, F0 → F3, F0 → F4, F0 → F1 (5 from F0)
- F1 → F0, F1 → F2, F1 → F3, F1 → F4, F1 → F0 (5 from F1)
- ... (similarly for F2, F3, F4)

**Note:** k-NN may have some bidirectional edges, so total unique edges ≤ 25

**Check:**
- [ ] At least 15-20 visible edges
- [ ] All edges connect valid node pairs
- [ ] No duplicate edges

---

### 17. Color Accuracy ✓

**Action:** Compare node colors to backend predictions

**From test_graph_integration.py output:**
- Node 0: Attack (70.2% confidence) → Red ✓
- Node 1: Normal (50.6% confidence) → Teal ✓
- Node 2: Attack (52.2% confidence) → Red ✓
- Node 3: Attack (56.7% confidence) → Red ✓
- Node 4: Normal (54.6% confidence) → Teal ✓

**Check:**
- [ ] All node colors match predictions
- [ ] No red nodes labeled "Normal"
- [ ] No teal nodes labeled "Attack"

---

## Final Checklist

### Visual Quality
- [ ] Graph looks professional and polished
- [ ] Colors are vibrant and readable
- [ ] Typography is clear
- [ ] Icons render correctly
- [ ] Animations are smooth

### Functionality
- [ ] All interactions work as expected
- [ ] No broken features
- [ ] Performance is acceptable
- [ ] No console errors

### Data Accuracy
- [ ] Node predictions match backend
- [ ] Edge connections are valid
- [ ] Statistics are correct
- [ ] Probabilities sum to 1.0

### User Experience
- [ ] Intuitive to use
- [ ] Helpful tooltips and labels
- [ ] Clear visual feedback
- [ ] Responsive to user actions

---

## Known Issues (If Any)

**None identified in automated tests.** ✅

If you encounter any issues during manual testing, document them here:

1. Issue: _______________
   - Expected: _______________
   - Actual: _______________
   - Steps to reproduce: _______________

---

## Test Result

**Date:** _______________  
**Tester:** _______________  
**Overall Status:** ⬜ PASS  ⬜ FAIL  ⬜ PARTIAL

**Notes:**
_______________________________________________
_______________________________________________
_______________________________________________

---

**Next Steps After Manual Test:**
1. If all checks pass → Ready for demo! 🎉
2. If issues found → Document and report for fixes
3. Consider optional improvements from GRAPH_VISUALIZATION_REPORT.md
