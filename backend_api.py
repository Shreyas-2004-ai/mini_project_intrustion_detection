"""
backend_api.py
==============
Flask API endpoint for GraphSAGE intrusion detection inference.

Provides a single endpoint:
  POST /api/predict — accepts CSV upload, returns predictions

Usage
-----
    python backend_api.py

The server runs on http://localhost:5000 with CORS enabled for
the React frontend (default: http://localhost:5173).
"""

import sys
import traceback
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.inference import predict_csv
from src.config import CATEGORICAL_COLS
from src.logger import get_logger

log = get_logger("backend_api")


# ── Graph extraction ──────────────────────────────────────────────────────────

def extract_graph_structure(graphs, predictions, probabilities):
    """
    Extract nodes and edges from PyG Data objects for frontend visualization.
    
    Parameters
    ----------
    graphs : list of PyG Data objects from inference
    predictions : np.ndarray of predictions (0=Normal, 1=Attack)
    probabilities : np.ndarray of shape (N, 2) with class probabilities
    
    Returns
    -------
    dict with 'nodes' and 'edges' lists formatted for React Flow
    """
    if not graphs or len(graphs) == 0:
        return {"nodes": [], "edges": [], "numNodes": 0, "numEdges": 0}
    
    # For visualization, we'll use the first graph (or merge multiple if needed)
    # For now, visualize the first graph only
    graph = graphs[0]
    
    # Extract edge_index (2, num_edges) tensor
    edge_index = graph.edge_index.cpu().numpy()
    edge_attr = graph.edge_attr.cpu().numpy() if hasattr(graph, 'edge_attr') and graph.edge_attr is not None else None
    
    # Number of nodes in this graph (from the Data object)
    num_nodes = graph.x.shape[0]  # Use x.shape[0] instead of num_nodes attribute
    
    # Create nodes
    nodes = []
    for i in range(num_nodes):
        pred = int(predictions[i])
        prob_normal = float(probabilities[i, 0])
        prob_attack = float(probabilities[i, 1])
        confidence = max(prob_normal, prob_attack)
        
        # Position nodes in a circular layout (frontend can adjust)
        angle = (i / num_nodes) * 2 * np.pi
        radius = 200
        x = 250 + np.cos(angle) * radius
        y = 200 + np.sin(angle) * radius
        
        # Color based on prediction
        is_attack = pred == 1
        node_color = '#f87171' if is_attack else '#4ade80'
        bg_color = 'rgba(248, 113, 113, 0.15)' if is_attack else 'rgba(74, 222, 128, 0.15)'
        
        nodes.append({
            "id": str(i),
            "type": "custom",
            "position": {"x": x, "y": y},
            "data": {
                "label": f"F{i}",
                "flowIdx": i,
                "prediction": "Attack" if is_attack else "Normal",
                "predValue": pred,
                "probNormal": round(prob_normal, 4),
                "probAttack": round(prob_attack, 4),
                "confidence": round(confidence, 4),
            },
            "style": {
                "width": 70,
                "height": 70,
                "borderRadius": "50%",
                "background": bg_color,
                "border": f"3px solid {node_color}",
                "color": node_color,
                "fontSize": 11,
                "fontWeight": 700,
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "boxShadow": f"0 0 20px {node_color}30",
            }
        })
    
    # Create edges
    edges = []
    num_edges = edge_index.shape[1]
    
    for i in range(num_edges):
        src = int(edge_index[0, i])
        dst = int(edge_index[1, i])
        
        # Get edge weight if available
        weight = 0.5
        if edge_attr is not None and i < len(edge_attr):
            weight = float(edge_attr[i, 0]) if edge_attr.ndim > 1 else float(edge_attr[i])
        
        # Calculate edge styling based on weight
        stroke_width = 1.5 + (weight * 2)  # 1.5 to 3.5 based on similarity
        opacity = 0.4 + (weight * 0.4)  # 0.4 to 0.8 based on similarity
        
        edges.append({
            "id": f"e{src}-{dst}",
            "source": str(src),
            "target": str(dst),
            "type": "default",
            "animated": False,
            "style": {
                "stroke": "#60a5fa",  # Brighter blue
                "strokeWidth": stroke_width,
                "opacity": opacity,
            },
            "data": {
                "weight": round(weight, 4),
                "similarity": round(weight * 100, 1),  # Percentage
            }
        })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "numNodes": num_nodes,
        "numEdges": num_edges,
    }


# ── Flask app setup ───────────────────────────────────────────────────────────

app = Flask(__name__)

# Enable CORS for React dev server (adjust origin if needed)
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:5173", "http://127.0.0.1:5173",
            "http://localhost:5174", "http://127.0.0.1:5174",
            "http://localhost:5175", "http://127.0.0.1:5175",
            "http://localhost:5176", "http://127.0.0.1:5176",
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
    }
})

# Upload configuration
UPLOAD_FOLDER = Path("./temp_uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max


def allowed_file(filename: str) -> bool:
    """Check if filename has .csv extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_csv_schema(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate that the uploaded CSV has the expected schema.
    
    Returns
    -------
    (is_valid, error_message)
    """
    # Check for required categorical columns (proto, service, state)
    missing_cats = [col for col in CATEGORICAL_COLS if col not in df.columns]
    if missing_cats:
        return False, f"Missing required categorical columns: {missing_cats}"
    
    # Check minimum column count (need at least the categorical + some numerical features)
    if len(df.columns) < 10:
        return False, f"CSV has only {len(df.columns)} columns. Expected ~45 for UNSW-NB15."
    
    # Check for non-empty
    if len(df) == 0:
        return False, "CSV file is empty (0 rows)."
    
    return True, ""


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "ok",
        "message": "GNN Intrusion Detection API is running",
        "model": "GraphSAGE"
    })


@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """
    POST /api/predict
    
    Accepts a CSV file upload and returns intrusion detection predictions.
    
    Request
    -------
    - Form data with key 'csv' containing the uploaded file
    
    Response
    --------
    JSON object:
    {
      "summary": {
        "total_flows": int,
        "normal_flows": int,
        "attack_flows": int,
        "attack_pct": float
      },
      "flows": [
        {
          "idx": int,
          "pred": int (0=Normal, 1=Attack),
          "probNormal": float,
          "probAttack": float,
          "confidence": float
        },
        ...
      ],
      "n_graphs": int,
      "isSample": false
    }
    
    Error Response
    --------------
    {
      "error": str,
      "details": str (optional)
    }
    """
    # Handle OPTIONS preflight for CORS
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Check if file is in request
        if 'csv' not in request.files:
            log.warning("No file part in request")
            return jsonify({"error": "No file uploaded. Expected 'csv' field."}), 400
        
        file = request.files['csv']
        
        # Check if user selected a file
        if file.filename == '':
            log.warning("Empty filename")
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file extension
        if not allowed_file(file.filename):
            log.warning("Invalid file type: %s", file.filename)
            return jsonify({
                "error": f"Invalid file type. Expected .csv, got: {file.filename}"
            }), 400
        
        # Read CSV into memory for validation
        try:
            csv_content = file.read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_content), low_memory=False)
        except Exception as e:
            log.error("Failed to parse CSV: %s", e)
            return jsonify({
                "error": "Failed to parse CSV file",
                "details": str(e)
            }), 400
        
        # Validate schema
        is_valid, error_msg = validate_csv_schema(df)
        if not is_valid:
            log.warning("CSV validation failed: %s", error_msg)
            return jsonify({
                "error": "CSV validation failed",
                "details": error_msg
            }), 400
        
        log.info("Received valid CSV: %s (%d rows, %d cols)", 
                 file.filename, len(df), len(df.columns))
        
        # Save to temporary file for inference pipeline
        filename = secure_filename(file.filename)
        temp_path = UPLOAD_FOLDER / filename
        
        # Write the content back to file (we already read it for validation)
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        
        # Run inference
        log.info("Running inference on: %s", temp_path)
        result = predict_csv(
            csv_path=temp_path,
            window_size=100,
            k=5,
            batch_size=32
        )
        
        # Extract predictions and probabilities
        predictions = result["predictions"]
        probabilities = result["probabilities"]  # shape: (N, 2)
        graphs = result.get("graphs", [])  # List of PyG Data objects
        
        # Format for frontend
        flows = []
        for i in range(len(predictions)):
            prob_normal = float(probabilities[i, 0])
            prob_attack = float(probabilities[i, 1])
            flows.append({
                "idx": i,
                "pred": int(predictions[i]),
                "probNormal": round(prob_normal, 4),
                "probAttack": round(prob_attack, 4),
                "confidence": round(max(prob_normal, prob_attack), 4),
            })
        
        # Extract graph structure for visualization
        graph_data = extract_graph_structure(graphs, predictions, probabilities)
        
        # Build response
        response = {
            "summary": result["summary"],
            "flows": flows,
            "n_graphs": result["n_graphs"],
            "graph": graph_data,
            "isSample": False  # This is real data, not sample
        }
        
        log.info("Inference complete: %d flows, %d attacks (%.1f%%)",
                 result["summary"]["total_flows"],
                 result["summary"]["attack_flows"],
                 result["summary"]["attack_pct"])
        
        # Clean up temporary file
        try:
            temp_path.unlink()
        except Exception as e:
            log.warning("Failed to delete temp file %s: %s", temp_path, e)
        
        return jsonify(response), 200
    
    except FileNotFoundError as e:
        log.error("File not found: %s", e)
        return jsonify({
            "error": "Model checkpoint not found",
            "details": str(e)
        }), 500
    
    except ValueError as e:
        log.error("Validation error: %s", e)
        return jsonify({
            "error": "Inference validation failed",
            "details": str(e)
        }), 400
    
    except Exception as e:
        log.error("Inference failed: %s", e, exc_info=True)
        return jsonify({
            "error": "Inference failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    log.info("=" * 70)
    log.info("GNN Intrusion Detection — Backend API")
    log.info("=" * 70)
    log.info("Starting Flask server on http://localhost:5000")
    log.info("CORS enabled for: http://localhost:5173")
    log.info("Endpoints:")
    log.info("  GET  /api/health   — health check")
    log.info("  POST /api/predict  — CSV upload → predictions")
    log.info("=" * 70)
    
    # Run Flask development server
    # For production, use: gunicorn -w 4 -b 0.0.0.0:5000 backend_api:app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,  # Disable in production
        threaded=True
    )
