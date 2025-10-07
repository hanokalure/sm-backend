#!/usr/bin/env python3
"""
Quick test script to verify models are loading properly
"""

import requests
import json
import sys
from config import API_CONFIG

def test_backend():
    """Test if backend is running and models are loading"""
    base_url = f"http://{API_CONFIG['host']}:{API_CONFIG['port']}"
    
    print("🧪 Testing SMS Spam Detection Backend")
    print("=" * 50)
    
    # Test health endpoint
    try:
        print("1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Backend is healthy!")
            print(f"   📊 Status: {health_data.get('status')}")
            print(f"   🤖 Models loaded: {health_data.get('models_loaded')}")
            print(f"   🔧 Working models: {health_data.get('working_models')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Cannot connect to backend: {e}")
        print(f"   💡 Make sure to start the backend: python main.py")
        return False
    
    # Test models endpoint
    try:
        print("\n2. Testing models endpoint...")
        response = requests.get(f"{base_url}/models", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            available_models = models_data.get('available_models', {})
            print(f"   ✅ Models endpoint working!")
            print(f"   📋 Available models: {len(available_models)}")
            
            for model_id, model_info in available_models.items():
                print(f"      • {model_info['name']}")
                print(f"        - Accuracy: {model_info['accuracy']}%")
                print(f"        - Focus: {model_info['training_focus']}")
                print(f"        - Dataset size: {model_info['dataset_size']}")
        else:
            print(f"   ❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Models endpoint error: {e}")
        return False
    
    # Test prediction endpoint
    try:
        print("\n3. Testing prediction endpoint...")
        test_message = "CONGRATULATIONS! You won $1000! Click here to claim NOW!"
        
        payload = {
            "text": test_message,
            "model": "xgboost"
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Prediction successful!")
            print(f"   📧 Test message: '{test_message[:50]}...'")
            print(f"   🎯 Prediction: {result.get('prediction')}")
            print(f"   📊 Confidence: {result.get('confidence'):.3f}")
            print(f"   🚨 Risk level: {result.get('risk_level')}")
            print(f"   🤖 Model used: {result.get('model_name')}")
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
        return False
    
    print("\n🎉 All tests passed! Backend is working correctly.")
    print("\n📋 Summary:")
    print(f"   • Backend URL: {base_url}")
    print(f"   • Models available: {len(available_models)}")
    print(f"   • Ready for frontend connection")
    
    return True

if __name__ == "__main__":
    try:
        success = test_backend()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Test cancelled")
        sys.exit(1)