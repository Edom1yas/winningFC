#!/usr/bin/env python3
import requests
import json

def test_api():
    base_url = "http://localhost:8000"
    
    print("🧪 Testing WinningFC API")
    print("=" * 30)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✓ Health check: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ API server not running. Start with: python -m src.api.main")
        return
    
    # Test training
    print("\n📚 Training model...")
    train_data = {
        "league": "Premier League",
        "season": "2024",
        "model_type": "random_forest"
    }
    
    response = requests.post(f"{base_url}/train", json=train_data)
    if response.status_code == 200:
        print(f"✓ Training successful: {response.json()}")
    else:
        print(f"❌ Training failed: {response.status_code}")
        return
    
    # Test prediction
    print("\n🔮 Making prediction...")
    predict_data = {
        "home_team": "Arsenal",
        "away_team": "Liverpool",
        "league": "Premier League"
    }
    
    response = requests.post(f"{base_url}/predict", json=predict_data)
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Prediction: {result['predicted_result']}")
        print(f"✓ Confidence: {result['confidence']:.3f}")
        probs = result['probabilities']
        print(f"✓ Probabilities:")
        print(f"  - Home Win: {probs['home_win']:.3f}")
        print(f"  - Draw: {probs['draw']:.3f}")  
        print(f"  - Away Win: {probs['away_win']:.3f}")
    else:
        print(f"❌ Prediction failed: {response.status_code}")

if __name__ == "__main__":
    test_api()