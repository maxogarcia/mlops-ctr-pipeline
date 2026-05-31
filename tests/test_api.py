import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_placeholder():
    """Placeholder test - API tests require MLflow connection"""
    assert True

def test_prediction_request_structure():
    """Test that prediction request has correct structure"""
    from src.serve.main import PredictionRequest
    req = PredictionRequest(int_1=5, int_2=3)
    assert req.int_1 == 5
    assert req.int_2 == 3

def test_prediction_response_structure():
    """Test that prediction response has correct structure"""
    from src.serve.main import PredictionResponse
    resp = PredictionResponse(
        click_probability=0.4,
        will_click=False,
        model_version="1"
    )
    assert resp.click_probability == 0.4
    assert resp.will_click == False
