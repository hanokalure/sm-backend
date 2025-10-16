#!/usr/bin/env python3
"""
Test script to verify all 4 models work correctly
"""

import sys
from pathlib import Path
import requests
import json
import time

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_api_health():
    """Test API health endpoint"""
    try:
        response = requests.get('http://127.0.0.1:8000/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health: {data['status']}")
            print(f"   Working models: {data['working_models']}")
            return True
        else:
            print(f"❌ API Health failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Health error: {e}")
        return False

def test_model_prediction(model_name, test_message):
    """Test prediction with a specific model"""
    try:
        payload = {
            "text": test_message,
            "model": model_name
        }
        
        response = requests.post(
            'http://127.0.0.1:8000/predict', 
            json=payload, 
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {model_name}: {result['prediction']} ({result['confidence']:.1%} confidence)")
            return True
        else:
            print(f"❌ {model_name} failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ {model_name} error: {e}")
        return False

def main():
    print("🧪 Testing All 4 Models")
    print("=" * 40)
    
    # Test messages
    spam_message = "FREE! You've won £1000! Call now to claim your prize!"
    ham_message = "Hey, running 15 minutes late. Start the meeting without me."
    
    # Test API health first
    if not test_api_health():
        print("❌ API not healthy, cannot test models")
        return False
    
    print("\n📝 Testing with SPAM message:", spam_message)
    print("-" * 40)
    
    models = ['xgboost', 'svm', 'distilbert_v2', 'roberta']
    results = {}
    
    for model in models:
        results[model] = test_model_prediction(model, spam_message)
        time.sleep(0.5)  # Small delay between requests
    
    print(f"\n📝 Testing with HAM message:", ham_message)
    print("-" * 40)
    
    for model in models:
        test_model_prediction(model, ham_message)
        time.sleep(0.5)
    
    print("\n" + "=" * 40)
    print("📊 SUMMARY")
    print("=" * 40)
    
    working = [model for model, success in results.items() if success]
    broken = [model for model, success in results.items() if not success]
    
    if working:
        print(f"✅ Working models: {', '.join(working)}")
    
    if broken:
        print(f"❌ Broken models: {', '.join(broken)}")
    
    if len(working) == 4:
        print("🎉 All 4 models are working perfectly!")
        return True
    else:
        print(f"⚠️ Only {len(working)}/4 models working")
        return False

if __name__ == "__main__":
    # Wait a moment for API to start
    print("⏳ Waiting for API to start...")
    time.sleep(2)
    
    success = main()
    sys.exit(0 if success else 1)