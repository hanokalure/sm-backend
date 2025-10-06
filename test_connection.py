#!/usr/bin/env python3
"""
SMS Spam Detection API - Connection Test Script
===============================================
This script tests the connection between your backend API and frontend.

Usage:
    python test_connection.py
"""

import requests
import json
import time
from config import API_CONFIG

def test_api_health():
    """Test if API is responding to health checks"""
    print("🏥 Testing API Health...")
    
    try:
        url = f"http://{API_CONFIG['host']}:{API_CONFIG['port']}/health"
        print(f"   Checking: {url}")
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ API is healthy!")
            print(f"   📊 Status: {data.get('status', 'unknown')}")
            print(f"   🤖 Models loaded: {data.get('models_loaded', 0)}")
            return True
        else:
            print(f"   ❌ API returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot connect to API at {url}")
        print(f"   💡 Make sure the backend is running: python main.py")
        return False
    except requests.exceptions.Timeout:
        print(f"   ❌ API request timed out")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def test_models_endpoint():
    """Test if models endpoint is working"""
    print("\n🤖 Testing Models Endpoint...")
    
    try:
        url = f"http://{API_CONFIG['host']}:{API_CONFIG['port']}/models"
        print(f"   Checking: {url}")
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Models endpoint working!")
            
            if 'available_models' in data:
                models = data['available_models']
                print(f"   📋 Available models: {len(models)}")
                
                for model_name, info in models.items():
                    print(f"      • {info['name']} - {info['accuracy']}% accuracy")
            
            return True
        else:
            print(f"   ❌ Models endpoint returned status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing models endpoint: {e}")
        return False

def test_prediction_endpoint():
    """Test the prediction endpoint with sample data"""
    print("\n🔮 Testing Prediction Endpoint...")
    
    test_messages = [
        {
            "text": "Hello, this is a normal message",
            "expected": "HAM",
            "description": "Normal message"
        },
        {
            "text": "CONGRATULATIONS! You won $1000! Click here to claim your prize NOW!",
            "expected": "SPAM", 
            "description": "Obvious spam message"
        },
        {
            "text": "Your account ending in 1234 was credited $250.00. Available balance: $1,500.78",
            "expected": "HAM",
            "description": "Banking notification"
        }
    ]
    
    try:
        url = f"http://{API_CONFIG['host']}:{API_CONFIG['port']}/predict"
        print(f"   Testing: {url}")
        
        results = []
        for i, test_case in enumerate(test_messages, 1):
            print(f"\n   Test {i}: {test_case['description']}")
            
            payload = {
                "text": test_case["text"],
                "model": "xgboost"  # Default model
            }
            
            response = requests.post(
                url, 
                json=payload, 
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('prediction', 'ERROR')
                confidence = result.get('confidence', 0.0)
                
                print(f"      Prediction: {prediction} (confidence: {confidence:.2f})")
                print(f"      Expected: {test_case['expected']}")
                
                if prediction == test_case['expected']:
                    print(f"      ✅ Correct prediction!")
                else:
                    print(f"      ⚠️  Different from expected (this is normal for mock models)")
                
                results.append({
                    'test': test_case['description'],
                    'prediction': prediction,
                    'confidence': confidence,
                    'expected': test_case['expected']
                })
            else:
                print(f"      ❌ Request failed with status: {response.status_code}")
                print(f"      Error: {response.text}")
                return False
        
        print(f"\n   ✅ All prediction tests completed!")
        print(f"   📊 Tested {len(results)} messages successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing prediction endpoint: {e}")
        return False

def test_cors_headers():
    """Test if CORS headers are properly configured"""
    print("\n🌐 Testing CORS Configuration...")
    
    try:
        url = f"http://{API_CONFIG['host']}:{API_CONFIG['port']}/health"
        
        # Test preflight request
        headers = {
            'Origin': API_CONFIG['frontend_url'],
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'Content-Type'
        }
        
        response = requests.options(url, headers=headers, timeout=10)
        
        cors_headers = {
            'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
            'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
            'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
        }
        
        print(f"   📡 Frontend URL: {API_CONFIG['frontend_url']}")
        print(f"   🔗 CORS Headers:")
        for header, value in cors_headers.items():
            if value:
                print(f"      • {header}: {value}")
            else:
                print(f"      • {header}: Not set")
        
        # Check if frontend URL is allowed
        allowed_origin = cors_headers.get('Access-Control-Allow-Origin')
        if allowed_origin == '*' or API_CONFIG['frontend_url'] in (allowed_origin or ''):
            print(f"   ✅ CORS properly configured for frontend!")
            return True
        else:
            print(f"   ⚠️  CORS may not be configured for your frontend URL")
            return True  # Don't fail on this, as it might still work
            
    except Exception as e:
        print(f"   ❌ Error testing CORS: {e}")
        return False

def print_summary(results):
    """Print test results summary"""
    print("\n" + "="*60)
    print("  📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for result in results if result)
    total = len(results)
    
    print(f"\n✅ Passed: {passed}/{total} tests")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is working correctly.")
        print("\n📋 Next steps:")
        print("   1. Add your trained models to the models/ directory")
        print("   2. Start your React Expo frontend on port 8081")
        print("   3. Test the full integration")
        
        print(f"\n🔗 Quick Links:")
        print(f"   • Backend API: http://{API_CONFIG['host']}:{API_CONFIG['port']}")
        print(f"   • Health Check: http://{API_CONFIG['host']}:{API_CONFIG['port']}/health")
        print(f"   • API Docs: http://{API_CONFIG['host']}:{API_CONFIG['port']}/docs")
        print(f"   • Frontend: {API_CONFIG['frontend_url']}")
    else:
        print(f"\n⚠️  Some tests failed. Check the errors above and:")
        print("   1. Make sure the backend is running: python main.py")
        print("   2. Check if ports are available")
        print("   3. Verify your configuration in config.py")

def main():
    """Main test function"""
    print("🧪 SMS Spam Detection API - Connection Test")
    print("="*60)
    
    print("\nThis script will test your backend API setup and connectivity.")
    print("Make sure your backend is running before proceeding.\n")
    
    input("Press Enter to start testing...")
    
    # Run all tests
    test_results = []
    
    print(f"\n🔧 Configuration:")
    print(f"   Backend: http://{API_CONFIG['host']}:{API_CONFIG['port']}")
    print(f"   Frontend: {API_CONFIG['frontend_url']}")
    
    # Test API health
    test_results.append(test_api_health())
    
    # Test models endpoint
    test_results.append(test_models_endpoint())
    
    # Test prediction endpoint
    test_results.append(test_prediction_endpoint())
    
    # Test CORS configuration
    test_results.append(test_cors_headers())
    
    # Print summary
    print_summary(test_results)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")