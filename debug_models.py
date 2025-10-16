#!/usr/bin/env python3
"""
Debug script to test model loading and identify issues
"""

import pickle
import os
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_model_file(model_path, model_name):
    """Test loading a specific model file"""
    print(f"\n=== Testing {model_name} Model ===")
    print(f"Path: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ File does not exist")
        return False
        
    file_size = os.path.getsize(model_path)
    print(f"File size: {file_size / (1024*1024):.1f} MB")
    
    if file_size == 0:
        print("❌ File is empty")
        return False
        
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            
        print("✅ File loaded successfully")
        print(f"Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Dictionary keys: {list(data.keys())}")
            
            # Check if it has the expected structure
            if 'model' in data:
                print(f"Model type: {type(data['model'])}")
            if 'vectorizer' in data:
                vectorizer = data['vectorizer']
                print(f"Vectorizer type: {type(vectorizer)}")
                
                # Check if vectorizer is fitted
                if hasattr(vectorizer, 'idf_'):
                    if vectorizer.idf_ is not None:
                        print("✅ Vectorizer is fitted")
                    else:
                        print("❌ Vectorizer is not fitted (idf_ is None)")
                else:
                    print("❌ Vectorizer doesn't have idf_ attribute")
                    
            if 'preprocessor' in data:
                print(f"Preprocessor type: {type(data['preprocessor'])}")
        else:
            print(f"Data is not a dictionary, it's: {type(data)}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

def main():
    print("🔧 SMS Spam Detection - Model Debug Tool")
    print("=" * 50)
    
    # Model paths from config
    models_dir = current_dir / "models"
    model_files = {
        "XGBoost": models_dir / "xgboost_model.pkl",
        "SVM": models_dir / "svm_model.pkl", 
        "DistilBERT V2": models_dir / "distilbert_v2_model.pkl",
        "RoBERTa": models_dir / "roberta_model.pkl"
    }
    
    results = {}
    for name, path in model_files.items():
        results[name] = test_model_file(path, name)
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    working_models = [name for name, success in results.items() if success]
    broken_models = [name for name, success in results.items() if not success]
    
    if working_models:
        print(f"✅ Working models: {', '.join(working_models)}")
    
    if broken_models:
        print(f"❌ Broken models: {', '.join(broken_models)}")
        
    if not working_models:
        print("❌ No models are working properly")
        print("\n💡 Recommendations:")
        print("1. Re-train your models")
        print("2. Ensure vectorizers are fitted during training")
        print("3. Check model saving process")
    else:
        print(f"\n🎉 {len(working_models)}/{len(model_files)} models are working")

if __name__ == "__main__":
    main()