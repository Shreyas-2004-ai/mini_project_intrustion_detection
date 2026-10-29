"""
test_backend_api.py
===================
Tests for the Flask backend API.
"""

import sys
from pathlib import Path
from io import BytesIO

import pytest
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend_api import app


@pytest.fixture
def client():
    """Flask test client fixture."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def valid_test_csv():
    """
    Create a valid test CSV with UNSW-NB15 schema.
    Uses actual test data from the dataset.
    """
    csv_path = Path(__file__).parent.parent / "data" / "raw" / "UNSW_NB15_testing-set.csv"
    if not csv_path.exists():
        pytest.skip(f"Test data not found: {csv_path}")
    
    # Load just 10 rows for quick testing
    df = pd.read_csv(csv_path, nrows=10)
    return df.to_csv(index=False).encode('utf-8')


@pytest.fixture
def invalid_csv_missing_columns():
    """CSV with missing required columns."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'dur': [0.5, 1.2, 0.8],
        'spkts': [10, 20, 15],
    })
    return df.to_csv(index=False).encode('utf-8')


@pytest.fixture
def empty_csv():
    """Empty CSV (header only)."""
    return b"id,dur,proto,service,state\n"


def test_health_check(client):
    """Test the /api/health endpoint."""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'ok'
    assert 'GraphSAGE' in data['model']


def test_predict_no_file(client):
    """Test /api/predict with no file uploaded."""
    response = client.post('/api/predict')
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    assert 'No file uploaded' in data['error']


def test_predict_empty_filename(client):
    """Test /api/predict with empty filename."""
    data = {'csv': (BytesIO(b''), '')}
    response = client.post('/api/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data


def test_predict_wrong_file_type(client):
    """Test /api/predict with non-CSV file."""
    data = {'csv': (BytesIO(b'test data'), 'test.txt')}
    response = client.post('/api/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    data = response.get_json()
    assert 'Invalid file type' in data['error']


def test_predict_invalid_csv_content(client):
    """Test /api/predict with malformed CSV."""
    invalid_csv = b"this is not, a valid, csv\nwith proper structure"
    data = {'csv': (BytesIO(invalid_csv), 'invalid.csv')}
    response = client.post('/api/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data


def test_predict_missing_columns(client, invalid_csv_missing_columns):
    """Test /api/predict with CSV missing required columns."""
    data = {'csv': (BytesIO(invalid_csv_missing_columns), 'missing_cols.csv')}
    response = client.post('/api/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    # Should mention missing columns
    assert 'validation' in data['error'].lower() or 'missing' in data['details'].lower()


def test_predict_empty_csv(client, empty_csv):
    """Test /api/predict with empty CSV (header only, no data rows)."""
    data = {'csv': (BytesIO(empty_csv), 'empty.csv')}
    response = client.post('/api/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data


def test_predict_valid_csv(client, valid_test_csv):
    """
    Test /api/predict with valid UNSW-NB15 CSV.
    
    This is the main integration test — verifies:
    1. CSV upload works
    2. Inference pipeline runs
    3. Response format is correct
    4. Predictions are returned
    """
    data = {'csv': (BytesIO(valid_test_csv), 'test.csv')}
    response = client.post('/api/predict', data=data, content_type='multipart/form-data')
    
    # Check response status
    if response.status_code != 200:
        error_data = response.get_json()
        pytest.fail(f"Inference failed: {error_data.get('error')}\n{error_data.get('details', '')}")
    
    assert response.status_code == 200
    result = response.get_json()
    
    # Verify response structure
    assert 'summary' in result
    assert 'flows' in result
    assert 'n_graphs' in result
    assert 'isSample' in result
    
    # Verify isSample is False (real predictions)
    assert result['isSample'] is False
    
    # Verify summary fields
    summary = result['summary']
    assert 'total_flows' in summary
    assert 'normal_flows' in summary
    assert 'attack_flows' in summary
    assert 'attack_pct' in summary
    
    # Verify summary math is correct
    assert summary['total_flows'] == summary['normal_flows'] + summary['attack_flows']
    assert summary['total_flows'] == 10  # We loaded 10 rows
    
    # Verify flows structure
    flows = result['flows']
    assert len(flows) == 10
    
    # Check each flow has required fields
    for flow in flows:
        assert 'idx' in flow
        assert 'pred' in flow
        assert 'probNormal' in flow
        assert 'probAttack' in flow
        assert 'confidence' in flow
        
        # Verify prediction is binary
        assert flow['pred'] in [0, 1]
        
        # Verify probabilities sum to ~1.0
        prob_sum = flow['probNormal'] + flow['probAttack']
        assert 0.99 <= prob_sum <= 1.01, f"Probabilities don't sum to 1: {prob_sum}"
        
        # Verify confidence is the max probability
        expected_confidence = max(flow['probNormal'], flow['probAttack'])
        assert abs(flow['confidence'] - expected_confidence) < 0.01


def test_predict_response_format(client, valid_test_csv):
    """Test that the response format matches what the frontend expects."""
    data = {'csv': (BytesIO(valid_test_csv), 'test.csv')}
    response = client.post('/api/predict', data=data, content_type='multipart/form-data')
    
    if response.status_code != 200:
        pytest.skip("Inference failed (likely model not trained)")
    
    result = response.get_json()
    
    # Frontend expects these exact field names
    assert all(key in result for key in ['summary', 'flows', 'isSample', 'graph'])
    assert all(key in result['summary'] for key in 
               ['total_flows', 'normal_flows', 'attack_flows', 'attack_pct'])
    
    # Frontend flow format
    flow = result['flows'][0]
    assert all(key in flow for key in ['idx', 'pred', 'probNormal', 'probAttack', 'confidence'])
    
    # Graph data format
    graph = result['graph']
    assert 'nodes' in graph
    assert 'edges' in graph
    assert 'numNodes' in graph
    assert 'numEdges' in graph
    
    # Verify graph structure
    if len(graph['nodes']) > 0:
        node = graph['nodes'][0]
        assert 'id' in node
        assert 'data' in node
        assert 'flowIdx' in node['data']
        assert 'prediction' in node['data']
        assert 'predValue' in node['data']
        assert 'confidence' in node['data']
    
    if len(graph['edges']) > 0:
        edge = graph['edges'][0]
        assert 'id' in edge
        assert 'source' in edge
        assert 'target' in edge


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
