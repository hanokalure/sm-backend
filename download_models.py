#!/usr/bin/env python3
"""
SMS Spam Detection - Model Download Script
==========================================
This script helps users download the pre-trained models from external sources
since they are too large to include in the GitHub repository.

Usage:
    python download_models.py

Note: You'll need to provide your own model URLs or use your own trained models.
"""

import os
import sys
import requests
from pathlib import Path
import gdown
from config import MODEL_URLS, MODELS_DIR

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_step(step_num, title, description=""):
    """Print a formatted step"""
    print(f"\n📋 Step {step_num}: {title}")
    if description:
        print(f"   {description}")

def download_file_from_google_drive(url, destination):
    """Download file from Google Drive using gdown"""
    try:
        print(f"   📥 Downloading to: {destination}")
        
        # Extract file ID from Google Drive URL
        if "drive.google.com" in url:
            if "file/d/" in url:
                file_id = url.split("file/d/")[1].split("/")[0]
                download_url = f"https://drive.google.com/uc?id={file_id}"
            elif "folders/" in url:
                print(f"   ⚠️ Folder URLs not supported for direct download")
                print(f"   💡 Please manually download from: {url}")
                return False
            else:
                download_url = url
        else:
            download_url = url
        
        gdown.download(download_url, str(destination), quiet=False)
        
        if os.path.exists(destination) and os.path.getsize(destination) > 0:
            print(f"   ✅ Successfully downloaded: {os.path.basename(destination)}")
            return True
        else:
            print(f"   ❌ Download failed or file is empty")
            return False
            
    except Exception as e:
        print(f"   ❌ Error downloading: {e}")
        return False

def check_existing_models():
    """Check which models already exist"""
    print_step(1, "Checking Existing Models")
    
    existing_models = []
    missing_models = []
    
    expected_models = {
        "xgboost_model.pkl": "XGBoost High-Accuracy Model",
        "svm_model.pkl": "SVM Baseline Model", 
        "distilbert_v2_model.pkl": "DistilBERT Deep Classifier V2",
        "roberta_model.pkl": "RoBERTa Ultimate Spam Detector"
    }
    
    for filename, description in expected_models.items():
        filepath = MODELS_DIR / filename
        if filepath.exists() and filepath.stat().st_size > 0:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   ✅ Found: {description} ({size_mb:.1f} MB)")
            existing_models.append(filename)
        else:
            print(f"   ❌ Missing: {description}")
            missing_models.append(filename)
    
    return existing_models, missing_models

def download_models_from_urls():
    """Download models from configured URLs"""
    print_step(2, "Downloading Models from URLs")
    
    if not MODEL_URLS:
        print("   ⚠️ No model URLs configured in config.py")
        print("   💡 Please add your model URLs to MODEL_URLS in config.py")
        return False
    
    success_count = 0
    total_count = len(MODEL_URLS)
    
    for model_name, url in MODEL_URLS.items():
        print(f"\n   🔄 Downloading {model_name} model...")
        
        if model_name == "xgboost":
            destination = MODELS_DIR / "xgboost_model.pkl"
        elif model_name == "svm":
            destination = MODELS_DIR / "svm_model.pkl"
        elif model_name == "distilbert_v2":
            destination = MODELS_DIR / "distilbert_v2_model.pkl"
        elif model_name == "roberta":
            destination = MODELS_DIR / "roberta_model.pkl"
        else:
            print(f"   ⚠️ Unknown model type: {model_name}")
            continue
        
        if download_file_from_google_drive(url, destination):
            success_count += 1
        else:
            print(f"   💡 You can manually download from: {url}")
            print(f"   📁 Save as: {destination}")
    
    print(f"\n   📊 Download Summary: {success_count}/{total_count} models downloaded")
    return success_count > 0

def create_model_placeholders():
    """Create placeholder files with instructions if downloads fail"""
    print_step(3, "Creating Model Placeholders")
    
    placeholder_content = """# Model File Placeholder

This file is a placeholder for the actual trained model.

## To replace this placeholder:

1. **Download your trained model** from your storage location
2. **Rename it** to match this filename exactly
3. **Place it** in this directory (models/)
4. **Restart the API** server

## Model Format:
- XGBoost/SVM models: Pickled scikit-learn models with vectorizers
- DistilBERT/RoBERTa models: Pickled Hugging Face transformer models

## Without Real Models:
The API will work with smart rule-based fallback detection,
but accuracy will be lower than with trained models.

## File Size:
Trained models are typically 10-500MB each, too large for GitHub.
"""
    
    model_files = {
        "xgboost_model.pkl": "XGBoost High-Accuracy spam classifier",
        "svm_model.pkl": "Support Vector Machine baseline model",
        "distilbert_v2_model.pkl": "DistilBERT Deep Classifier V2 with banking focus",
        "roberta_model.pkl": "RoBERTa Ultimate transformer-based detector"
    }
    
    created_count = 0
    
    for filename, description in model_files.items():
        filepath = MODELS_DIR / filename
        
        if not filepath.exists() or filepath.stat().st_size == 0:
            placeholder_file = MODELS_DIR / f"{filename}.placeholder"
            
            try:
                with open(placeholder_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {description}\n\n")
                    f.write(placeholder_content)
                
                print(f"   📄 Created placeholder: {filename}.placeholder")
                created_count += 1
                
            except Exception as e:
                print(f"   ❌ Error creating placeholder for {filename}: {e}")
    
    if created_count > 0:
        print(f"\n   💡 Created {created_count} placeholder files")
        print(f"   📁 Check the models/ directory for .placeholder files with instructions")
    
    return created_count > 0

def print_manual_instructions():
    """Print manual download instructions"""
    print_step(4, "Manual Download Instructions")
    
    print("""
   If automatic downloads failed, you can manually download models:

   🔗 Google Drive URLs (from config.py):
   """)
    
    for model_name, url in MODEL_URLS.items():
        print(f"   • {model_name}: {url}")
    
    print("""
   📁 Save models in the following locations:
   """)
    
    model_paths = {
        "XGBoost": MODELS_DIR / "xgboost_model.pkl",
        "SVM": MODELS_DIR / "svm_model.pkl", 
        "DistilBERT V2": MODELS_DIR / "distilbert_v2_model.pkl",
        "RoBERTa": MODELS_DIR / "roberta_model.pkl"
    }
    
    for model_name, filepath in model_paths.items():
        print(f"   • {model_name}: {filepath}")
    
    print(f"""
   🔄 After downloading:
   1. Ensure files are in {MODELS_DIR}
   2. Verify filenames match exactly (case-sensitive)
   3. Restart the API: python main.py
   4. Test: python test_connection.py
   """)

def verify_downloads():
    """Verify that downloaded models are valid"""
    print_step(5, "Verifying Downloaded Models")
    
    verified_count = 0
    total_files = 0
    
    for filepath in MODELS_DIR.glob("*.pkl"):
        total_files += 1
        filename = filepath.name
        
        try:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            
            if size_mb < 0.1:  # Less than 100KB is suspicious for ML models
                print(f"   ⚠️ {filename}: Very small file ({size_mb:.1f} MB) - may be incomplete")
            elif size_mb > 1000:  # More than 1GB is unusually large
                print(f"   ⚠️ {filename}: Very large file ({size_mb:.1f} MB) - double-check this is correct")
            else:
                print(f"   ✅ {filename}: Good size ({size_mb:.1f} MB)")
                verified_count += 1
                
        except Exception as e:
            print(f"   ❌ {filename}: Error checking file - {e}")
    
    print(f"\n   📊 Verification: {verified_count}/{total_files} models appear valid")
    
    if verified_count == 0 and total_files == 0:
        print(f"   💡 No model files found. The API will use fallback rule-based detection.")
        print(f"   🎯 For best accuracy, add trained models to {MODELS_DIR}")
    
    return verified_count > 0

def main():
    """Main download function"""
    print_header("🤖 SMS Spam Detection - Model Download")
    
    print("""
This script helps you download the pre-trained models for the SMS Spam Detection API.

⚠️  Important Notes:
• Model files are large (10-500MB each) and not included in GitHub
• You need trained models for best performance
• The API will work with fallback rules if models are missing
""")
    
    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Check existing models
    existing, missing = check_existing_models()
    
    if not missing:
        print(f"\n🎉 All models are already present! No downloads needed.")
        verify_downloads()
        return True
    
    # Ask user for confirmation
    if existing:
        print(f"\n📋 Found {len(existing)} existing models, {len(missing)} missing.")
    else:
        print(f"\n📋 No models found. Need to download {len(missing)} models.")
    
    response = input(f"\nProceed with downloading missing models? [y/N]: ").strip().lower()
    
    if response not in ['y', 'yes']:
        print("\n⚠️ Download cancelled by user.")
        print("💡 You can run this script again later or manually download models.")
        return False
    
    # Try to download from URLs
    download_success = download_models_from_urls()
    
    # Create placeholders for any missing models
    create_model_placeholders()
    
    # Show manual instructions
    print_manual_instructions()
    
    # Verify what we have
    verify_success = verify_downloads()
    
    # Final summary
    print_header("📊 Download Summary")
    
    if download_success and verify_success:
        print("""
✅ SUCCESS! Models have been downloaded and verified.

🚀 Next steps:
1. Start the API: python main.py
2. Test the connection: python test_connection.py
3. Try the API at: http://localhost:8000/docs
""")
        return True
    elif verify_success:
        print("""
✅ Models are present and verified (may have been downloaded previously).

🚀 Next steps:
1. Start the API: python main.py
2. Test the connection: python test_connection.py
""")
        return True
    else:
        print("""
⚠️ No models could be automatically downloaded.

📋 Options:
1. Check the Google Drive links in config.py
2. Manually download models (see instructions above)
3. Use your own trained models
4. Run API without models (will use fallback rules)

💡 The API will still work with reduced accuracy if no models are present.
""")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during download: {e}")
        sys.exit(1)