#!/usr/bin/env python3
"""
WinningFC Demo Script
Shows the complete workflow of the soccer prediction system
"""

import sys
import os
import subprocess
import time
import requests
import json

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_main_demo():
    """Run the main demo showing data processing and model training"""
    print("🏆 WinningFC - Soccer Match Prediction Demo")
    print("=" * 50)
    
    print("\n1️⃣  Running main system demo...")
    try:
        # Run main.py to show the complete workflow
        result = subprocess.run(['python', 'main.py'], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✅ Main demo completed successfully!")
            # Show last few lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-8:]:  # Show last 8 lines
                if line.strip():
                    print(f"   {line}")
        else:
            print(f"❌ Demo failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return False
    
    return True

def test_api_endpoints():
    """Test the API endpoints if server is running"""
    print("\n2️⃣  Testing API endpoints...")
    
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
            health_data = response.json()
            print(f"   Status: {health_data['status']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("⚠️  API server not running")
        print("   To start server: ./start_server.sh")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False
    
    # Test training endpoint
    print("\n📚 Testing model training...")
    train_data = {
        "league": "Premier League",
        "season": "2024",
        "model_type": "random_forest"
    }
    
    try:
        response = requests.post(f"{base_url}/train", json=train_data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model trained successfully!")
            print(f"   Accuracy: {result['accuracy']:.3f}")
            print(f"   CV Score: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")
            print(f"   Training samples: {result['train_size']}")
        else:
            print(f"❌ Training failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False
    
    # Test prediction endpoint
    print("\n🔮 Testing match prediction...")
    predict_data = {
        "home_team": "Arsenal",
        "away_team": "Liverpool",
        "league": "Premier League"
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=predict_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            print(f"   Match: {result['home_team']} vs {result['away_team']}")
            print(f"   Prediction: {result['predicted_result']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            
            probs = result['probabilities']
            print(f"   Probabilities:")
            print(f"     - Home Win: {probs['home_win']:.3f}")
            print(f"     - Draw: {probs['draw']:.3f}")
            print(f"     - Away Win: {probs['away_win']:.3f}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False
    
    # Test other endpoints
    print("\n📊 Testing additional endpoints...")
    try:
        # Model performance
        response = requests.get(f"{base_url}/model/performance", timeout=5)
        if response.status_code == 200:
            print("✅ Model performance endpoint working")
        
        # Supported leagues
        response = requests.get(f"{base_url}/leagues", timeout=5)
        if response.status_code == 200:
            leagues = response.json()
            print(f"✅ Supported leagues: {len(leagues['leagues'])} leagues available")
            
    except Exception as e:
        print(f"⚠️  Some endpoints may not be available: {e}")
    
    return True

def show_project_structure():
    """Show the project structure and key files"""
    print("\n3️⃣  Project Structure Overview")
    print("=" * 30)
    
    structure = {
        "📁 src/": {
            "data_collection/": ["scrapers.py - Data collection and API integration"],
            "data_processing/": ["feature_engineer.py - Advanced feature engineering"],
            "models/": ["predictor.py - ML models and training pipeline"],
            "api/": ["main.py - FastAPI REST API server"]
        },
        "📄 Key Files": [
            "main.py - Main demo script",
            "start_server.sh - API server startup",
            "requirements.txt - Python dependencies",
            "README.md - Documentation"
        ]
    }
    
    for category, items in structure.items():
        print(f"\n{category}")
        if isinstance(items, dict):
            for folder, files in items.items():
                print(f"  📂 {folder}")
                for file in files:
                    print(f"    📄 {file}")
        else:
            for item in items:
                print(f"  📄 {item}")

def main():
    """Main demo function"""
    print("🎯 WinningFC Soccer Prediction System")
    print("=" * 40)
    print("Complete Data Engineering & ML Pipeline")
    print("=" * 40)
    
    # Run the main system demo
    if not run_main_demo():
        print("❌ Main demo failed, exiting...")
        return 1
    
    # Test API if available
    api_success = test_api_endpoints()
    
    # Show project structure
    show_project_structure()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 Demo Complete!")
    print("=" * 50)
    
    if api_success:
        print("✅ All systems operational")
        print("🌐 API server: http://localhost:8000")
        print("📚 API docs: http://localhost:8000/docs")
    else:
        print("✅ Core system working")
        print("⚠️  Start API server with: ./start_server.sh")
    
    print("\n🚀 Next steps:")
    print("  • Add real data sources (football-data.org, etc.)")
    print("  • Implement advanced models and ensemble methods")
    print("  • Add web dashboard for predictions")
    print("  • Deploy to cloud platform")
    
    return 0

if __name__ == "__main__":
    exit(main())