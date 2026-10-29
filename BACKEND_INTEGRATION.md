# Backend Integration Documentation

## Overview

This document describes the Flask backend API integration that connects the React frontend to the trained GraphSAGE intrusion detection model.

## Architecture

```
Frontend (React)  ──HTTP POST──>  Backend (Flask)  ──>  Inference Pipeline
   localhost:5173                   localhost:5000        (GraphSAGE model)
                                                          
                                                          ↓
                                                          
                                    Predictions  <──  Graph Construction
                                                       & Preprocessing
```

## Files Created/Modified

### New Files

1. **`backend_api.py`** (336 lines)
   - Flask API server with CORS support
   - Endpoints:
     - `GET /api/health` — health check
     - `POST /api/predict` — CSV upload → predictions
   - CSV validation, preprocessing, and inference orchestration
   - Error handling with detailed error messages

2. **`test_api_integration.py`** (60 lines)
   - Integration test script for manual testing
   - Tests health check and prediction endpoints
   - Displays results in human-readable format

3. **`tests/test_backend_api.py`** (235 lines)
   - Comprehensive pytest test suite
   - Tests:
     - Health check endpoint
     - File upload validation (no file, wrong type, etc.)
     - CSV validation (missing columns, empty file, etc.)
     - Full inference pipeline with real predictions
     - Response format validation

4. **`test_upload.csv`** (4 lines)
   - Sample test file with 3 flows from UNSW-NB15 dataset
   - Used for integration testing

### Modified Files

1. **`frontend/src/components/UploadSection.jsx`**
   - Replaced simulated upload with real API call
   - Added error handling with user-friendly messages
   - Added state tracking for real vs. sample data
   - Sample data notice only shown for demo mode

## API Specification

### POST /api/predict

Accepts a CSV file upload and returns GraphSAGE predictions.

#### Request

- **Method:** POST
- **Content-Type:** multipart/form-data
- **Body:** Form data with key `csv` containing the uploaded file

Example using curl:
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "csv=@test_upload.csv"
```

Example using JavaScript:
```javascript
const formData = new FormData()
formData.append('csv', file)

const response = await fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  body: formData,
})

const data = await response.json()
```

#### Response (Success - 200 OK)

```json
{
  "summary": {
    "total_flows": 250,
    "normal_flows": 187,
    "attack_flows": 63,
    "attack_pct": 25.2
  },
  "flows": [
    {
      "idx": 0,
      "pred": 0,
      "probNormal": 0.8234,
      "probAttack": 0.1766,
      "confidence": 0.8234
    },
    ...
  ],
  "n_graphs": 3,
  "isSample": false
}
```

**Fields:**
- `summary.total_flows` (int) — total number of flow records processed
- `summary.normal_flows` (int) — flows predicted as normal
- `summary.attack_flows` (int) — flows predicted as attacks
- `summary.attack_pct` (float) — percentage of flows that are attacks
- `flows` (array) — per-flow predictions
  - `idx` (int) — flow index (0-based)
  - `pred` (int) — prediction (0=Normal, 1=Attack)
  - `probNormal` (float) — probability of normal (0-1)
  - `probAttack` (float) — probability of attack (0-1)
  - `confidence` (float) — max(probNormal, probAttack)
- `n_graphs` (int) — number of graphs constructed
- `isSample` (bool) — always `false` for real predictions

#### Response (Error - 4xx/5xx)

```json
{
  "error": "Error message",
  "details": "Detailed error information (optional)"
}
```

**Common Error Codes:**
- `400 Bad Request` — Invalid CSV, missing columns, validation failed
- `500 Internal Server Error` — Model loading failed, inference crashed

### GET /api/health

Simple health check endpoint.

#### Response (200 OK)

```json
{
  "status": "ok",
  "message": "GNN Intrusion Detection API is running",
  "model": "GraphSAGE"
}
```

## CSV Format Requirements

The uploaded CSV must follow the UNSW-NB15 schema:

### Required Columns
- **Categorical features:** `proto`, `service`, `state`
- **Numerical features:** ~39 additional features (dur, spkts, dpkts, sbytes, etc.)

### Optional Columns
- `id` — row identifier (auto-generated if missing)
- `label` — ground truth label (ignored during inference)
- `attack_cat` — attack category (ignored during inference)

### Example
```csv
id,dur,proto,service,state,spkts,dpkts,sbytes,dbytes,...
1,0.5,tcp,http,FIN,10,8,500,400,...
2,1.2,udp,dns,CON,5,5,250,250,...
```

**Total columns:** 45 (43 features + 2 targets)

## Running the System

### 1. Start the Backend

```bash
cd gnn_intrusion_detection
python backend_api.py
```

Output:
```
======================================================================
GNN Intrusion Detection — Backend API
======================================================================
Starting Flask server on http://localhost:5000
CORS enabled for: http://localhost:5173
Endpoints:
  GET  /api/health   — health check
  POST /api/predict  — CSV upload → predictions
======================================================================
 * Running on http://127.0.0.1:5000
```

### 2. Start the Frontend

```bash
cd frontend
npm run dev
```

Output:
```
  VITE v5.4.21  ready in 1234 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### 3. Use the Application

1. Open http://localhost:5173 in your browser
2. Upload a CSV file with UNSW-NB15 schema
3. Wait for processing (progress spinner shown)
4. View real GraphSAGE predictions in the dashboard

## Testing

### Backend Unit Tests

```bash
pytest tests/test_backend_api.py -v
```

**Results:**
```
test_health_check PASSED
test_predict_no_file PASSED
test_predict_empty_filename PASSED
test_predict_wrong_file_type PASSED
test_predict_invalid_csv_content PASSED
test_predict_missing_columns PASSED
test_predict_empty_csv PASSED
test_predict_valid_csv PASSED
test_predict_response_format PASSED

9 passed in 66.37s
```

### Integration Test

```bash
python test_api_integration.py
```

**Sample Output:**
```
======================================================================
Testing Backend API Integration
======================================================================

1. Testing /api/health endpoint...
   Status: 200
   Response: {'message': '...', 'model': 'GraphSAGE', 'status': 'ok'}

2. Testing /api/predict endpoint...
   Status: 200

   ✓ Inference successful!

   Summary:
     Total flows:   3
     Normal flows:  0
     Attack flows:  3
     Attack %:      100.0%
     Graphs created: 1
     Is sample data: False

   First 3 predictions:
     Flow 0: Attack (prob_attack=0.6893, confidence=0.6893)
     Flow 1: Attack (prob_attack=0.5077, confidence=0.5077)
     Flow 2: Attack (prob_attack=0.5178, confidence=0.5178)

   ✓ All 3 flows processed successfully

======================================================================
Integration test complete!
======================================================================
```

### Frontend Build Test

```bash
cd frontend
npm run build
```

**Result:** ✅ Built successfully in 27.45s

## Inference Pipeline

The backend reuses the existing inference pipeline without modification:

1. **CSV Loading** (`src/inference.py:load_inference_csv`)
   - Validates file existence
   - Checks for required columns
   - Adds dummy targets if absent (label, attack_cat)

2. **Preprocessing** (`src/preprocessor.py`)
   - Loads saved StandardScaler and OrdinalEncoder
   - Transforms features (never re-fitted)
   - Produces 42-dimensional feature vectors

3. **Graph Construction** (`src/inference.py:build_inference_graphs`)
   - Windows of 100 flows per graph
   - k=5 nearest neighbor edges
   - Cosine similarity on ct_* features

4. **Model Inference** (`src/inference.py:run_inference`)
   - Loads trained GraphSAGE checkpoint (`models/graphsage_best.pt`)
   - Runs on GPU if available (CUDA)
   - Returns predictions and probabilities

5. **Post-processing**
   - Formats per-flow predictions
   - Computes summary statistics
   - Returns JSON response

## Model Correctness

✅ **Verified:**
- Uses exact same preprocessing pipeline as training
- Loads the actual trained GraphSAGE checkpoint
- Preserves graph construction parameters (window=100, k=5)
- Never uses label/attack_cat as features
- Returns real model probabilities (not fabricated)
- GPU acceleration works (CUDA)

## Security & Production Notes

### Current Setup (Development)
- Flask development server
- Debug mode enabled
- CORS open to localhost:5173
- No authentication
- No rate limiting
- No input size limits (50 MB max)

### Production Recommendations
1. **Use production WSGI server:**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 backend_api:app
   ```

2. **Disable debug mode:**
   ```python
   app.run(debug=False)
   ```

3. **Restrict CORS origins:**
   ```python
   CORS(app, origins=["https://yourdomain.com"])
   ```

4. **Add authentication:**
   - API keys
   - JWT tokens
   - OAuth2

5. **Add rate limiting:**
   ```python
   from flask_limiter import Limiter
   limiter = Limiter(app, default_limits=["100 per hour"])
   ```

6. **Add monitoring:**
   - Request logging
   - Error tracking (Sentry)
   - Performance metrics (Prometheus)

## Limitations & Future Work

### Current Limitations
1. **Synchronous processing:** Backend blocks during inference (~2s for 100 flows)
2. **No batch endpoint:** Processes one CSV at a time
3. **No result caching:** Re-runs inference for identical files
4. **Limited error context:** Generic error messages for some failures

### Future Enhancements
1. **Async processing:** WebSocket for real-time progress updates
2. **Multi-class classification:** Return attack category (not just binary)
3. **Graph visualization:** Return graph structure for frontend rendering
4. **Explainability:** Add GNNExplainer results per prediction
5. **Model versioning:** Support A/B testing with multiple models
6. **Streaming inference:** Process large files in chunks

## Troubleshooting

### Backend won't start
- **Check:** Python dependencies installed (`pip list | grep -E "flask|torch"`)
- **Check:** Model checkpoint exists (`models/graphsage_best.pt`)
- **Check:** Port 5000 not in use (`netstat -an | findstr 5000`)

### Frontend can't connect to backend
- **Check:** Backend is running on http://localhost:5000
- **Check:** CORS is enabled (check backend logs for CORS errors)
- **Check:** No firewall blocking localhost connections

### Inference fails with validation error
- **Check:** CSV has required columns (proto, service, state)
- **Check:** CSV has ~45 columns (UNSW-NB15 schema)
- **Check:** CSV is not empty (at least 1 row)

### Predictions look wrong
- **Check:** CSV columns match UNSW-NB15 schema exactly
- **Check:** Feature order and types match training data
- **Check:** Model checkpoint is the trained version (not random init)

## Contact & Support

For issues, refer to:
- Backend code: `backend_api.py`
- Inference pipeline: `src/inference.py`
- Tests: `tests/test_backend_api.py`
- Frontend upload: `frontend/src/components/UploadSection.jsx`
