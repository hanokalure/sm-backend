#!/usr/bin/env python3
"""
SMS Spam Detection - Local Setup Script
========================================
This script sets up the SMS Spam Detection API for local development.
It handles dependencies, model folder creation, and initial setup.

Usage:
    python setup_local.py

Requirements:
    - Python 3.8+
    - pip
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import json

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

def run_command(command, description="", check=True):
    """Run a command and handle errors"""
    print(f"   Running: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=check)
        if result.stdout:
            print(f"   ✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   Error details: {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print_step(1, "Checking Python Version")
    
    version = sys.version_info
    print(f"   Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("   ✅ Python version is compatible")
        return True
    else:
        print("   ❌ Python 3.8+ is required")
        print("   Please install Python 3.8 or higher from https://python.org")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print_step(2, "Installing Dependencies")
    
    print("   Installing required packages...")
    if run_command("pip install --upgrade pip"):
        if run_command("pip install -r requirements.txt"):
            print("   ✅ All dependencies installed successfully")
            return True
    
    print("   ❌ Failed to install dependencies")
    return False

def setup_models_directory():
    """Setup the models directory structure"""
    print_step(3, "Setting up Models Directory")
    
    current_dir = Path(__file__).parent
    models_dir = current_dir / "models"
    
    try:
        models_dir.mkdir(exist_ok=True)
        print(f"   📁 Models directory created: {models_dir}")
        
        # Create placeholder files to show expected structure
        placeholder_files = [
            "xgboost_model.pkl",
            "svm_model.pkl", 
            "distilbert_v2_model.pkl",
            "roberta_model.pkl"
        ]
        
        readme_content = """# Models Directory

This directory should contain your trained ML models:

## Expected Files:
- `xgboost_model.pkl` - XGBoost spam classifier model
- `svm_model.pkl` - Support Vector Machine model
- `distilbert_v2_model.pkl` - DistilBERT Deep Classifier V2 model
- `roberta_model.pkl` - RoBERTa Ultimate Spam Detector model

## Setup:
1. Download your trained models from Google Drive or your model storage
2. Place them in this directory with the exact names listed above
3. Restart the API server

## Note:
The API will work with mock models if these files are not present,
but for best performance, please add your actual trained models.
"""
        
        readme_file = models_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print("   ✅ Models directory structure created")
        print(f"   📖 Check {readme_file} for model placement instructions")
        return True
        
    except Exception as e:
        print(f"   ❌ Error creating models directory: {e}")
        return False

def create_environment_config():
    """Create environment configuration"""
    print_step(4, "Creating Environment Configuration")
    
    try:
        current_dir = Path(__file__).parent
        
        # Create .env file for local development
        env_content = """# SMS Spam Detection API - Local Development Configuration
# Backend API Configuration
API_HOST=127.0.0.1
API_PORT=8000
FRONTEND_URL=http://localhost:8081

# Development Settings
DEBUG=True
RELOAD=True

# Model Configuration
MODELS_DIR=./models

# Logging
LOG_LEVEL=INFO
"""
        
        env_file = current_dir / ".env"
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        print(f"   ✅ Environment file created: {env_file}")
        
        # Create run script for easy startup
        if platform.system() == "Windows":
            run_script_content = """@echo off
echo 🚀 Starting SMS Spam Detection API...
echo 📡 Frontend should be running on http://localhost:8081
echo 🌐 Backend will start on http://localhost:8000
echo.
python main.py
pause
"""
            script_file = current_dir / "run_api.bat"
        else:
            run_script_content = """#!/bin/bash
echo "🚀 Starting SMS Spam Detection API..."
echo "📡 Frontend should be running on http://localhost:8081"
echo "🌐 Backend will start on http://localhost:8000"
echo ""
python3 main.py
"""
            script_file = current_dir / "run_api.sh"
            
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(run_script_content)
            
        if platform.system() != "Windows":
            os.chmod(script_file, 0o755)
        
        print(f"   ✅ Run script created: {script_file}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error creating environment configuration: {e}")
        return False

def create_frontend_env():
    """Create .env file for frontend"""
    print_step(5, "Setting up Frontend Environment")
    
    try:
        frontend_dir = Path("C:/sms_spam_detection/frontend")
        if not frontend_dir.exists():
            print(f"   ⚠️ Frontend directory not found at {frontend_dir}")
            print("   Skipping frontend configuration")
            return True
            
        frontend_env_content = """# Frontend Environment Configuration
REACT_APP_API_URL=http://localhost:8000
NEXT_PUBLIC_API_URL=http://localhost:8000
"""
        
        frontend_env_file = frontend_dir / ".env.local"
        with open(frontend_env_file, 'w', encoding='utf-8') as f:
            f.write(frontend_env_content)
            
        print(f"   ✅ Frontend environment configured: {frontend_env_file}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error configuring frontend: {e}")
        return False

def print_completion_message():
    """Print setup completion message with instructions"""
    print_header("🎉 Setup Complete!")
    
    print("""
📋 NEXT STEPS:

1. 📁 Add Your Models:
   • Place your trained model files in the 'models/' directory
   • See models/README.md for detailed instructions

2. 🚀 Start the Backend:
   • Windows: Double-click 'run_api.bat' or run 'python main.py'
   • Mac/Linux: Run './run_api.sh' or 'python3 main.py'
   • Backend will be available at: http://localhost:8000

3. 📱 Start Your Frontend:
   • Navigate to your frontend directory: C:/sms_spam_detection/frontend
   • Run: npm start or expo start
   • Frontend should run on: http://localhost:8081

4. 🔗 Test Connection:
   • Open: http://localhost:8000/health
   • Should return: {"status": "healthy", ...}

📡 API ENDPOINTS:
   • Health Check: GET  http://localhost:8000/health
   • Models Info:  GET  http://localhost:8000/models  
   • Predict:      POST http://localhost:8000/predict
   • Batch:        POST http://localhost:8000/predict/batch

🔧 TROUBLESHOOTING:
   • Port conflicts: Change API_PORT in .env file
   • CORS issues: Check API_CONFIG in config.py
   • Model issues: Check models/README.md
   • Python issues: Ensure Python 3.8+ is installed

💡 For your friend's setup:
   1. Copy this entire folder to their computer
   2. Run: python setup_local.py
   3. Follow the same steps above
""")

def main():
    """Main setup function"""
    print_header("🚀 SMS Spam Detection API - Local Setup")
    
    print("""
This script will set up your SMS Spam Detection API for local development.
It will:
   • Check Python version compatibility
   • Install required dependencies  
   • Create models directory structure
   • Configure environment variables
   • Create startup scripts
""")
    
    input("\nPress Enter to continue...")
    
    # Run setup steps
    steps_passed = 0
    total_steps = 5
    
    if check_python_version():
        steps_passed += 1
    else:
        print("\n❌ Setup failed: Incompatible Python version")
        return False
        
    if install_dependencies():
        steps_passed += 1
    else:
        print("\n⚠️ Warning: Dependencies installation failed, continuing...")
        
    if setup_models_directory():
        steps_passed += 1
        
    if create_environment_config():
        steps_passed += 1
        
    if create_frontend_env():
        steps_passed += 1
    
    print(f"\n📊 Setup Progress: {steps_passed}/{total_steps} steps completed")
    
    if steps_passed >= 3:  # Minimum viable setup
        print_completion_message()
        return True
    else:
        print("\n❌ Setup incomplete. Please resolve the errors above and try again.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        sys.exit(1)