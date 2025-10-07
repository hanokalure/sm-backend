#!/usr/bin/env python3
"""
Test script to validate that the frontend will see the correct number of models
"""

import requests
import json
from config import API_CONFIG

def test_frontend_connection():
    """Test what the frontend will see when it calls the backend"""
    base_url = f"http://{API_CONFIG['host']}:{API_CONFIG['port']}"
    
    print("🧪 Testing Frontend-Backend Connection")
    print("=" * 50)
    print(f"Backend: {base_url}")
    print(f"Frontend: C:\\sms_spam_detection\\frontend")
    print()
    
    # Test the models endpoint that the frontend calls
    try:
        print("📡 Testing /models endpoint (what frontend sees)...")
        response = requests.get(f"{base_url}/models", timeout=5)
        
        if response.status_code == 200:
            models_data = response.json()
            available_models = models_data.get('available_models', {})
            
            print(f"✅ API Response successful!")
            print(f"📊 Models returned: {len(available_models)}")
            print()
            
            for i, (model_id, model_info) in enumerate(available_models.items(), 1):
                print(f"{i}. {model_info['name']}")
                print(f"   ID: {model_id}")
                print(f"   Accuracy: {model_info['accuracy']}%")
                print(f"   Focus: {model_info['training_focus']}")
                print()
            
            # Validate the fix
            if len(available_models) == 2:
                print("🎉 SUCCESS: Frontend will now see exactly 2 models!")
                print("✅ Only working models are being returned")
                
                expected_models = ['xgboost', 'svm']
                actual_models = list(available_models.keys())
                
                if all(model in actual_models for model in expected_models):
                    print("✅ Correct models are being returned")
                else:
                    print(f"⚠️  Expected models: {expected_models}")
                    print(f"⚠️  Actual models: {actual_models}")
                
                return True
                
            else:
                print(f"❌ FAILED: Frontend will see {len(available_models)} models, expected 2")
                return False
                
        else:
            print(f"❌ API request failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend")
        print("💡 Make sure backend is running: python main.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_frontend_connection()
    
    if success:
        print()
        print("🚀 Next Steps:")
        print("1. Start your frontend:")
        print("   cd C:\\sms_spam_detection\\frontend")
        print("   npm start")
        print()
        print("2. Your frontend should now show exactly 2 models!")
        print("3. The dynamic text should say 'Choose from 2 advanced models'")
        print()
        print("✅ Problem solved!")
    else:
        print()
        print("❌ Fix not working properly. Check backend logs.")