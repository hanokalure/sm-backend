import os
from pathlib import Path

# Get the current script directory
CURRENT_DIR = Path(__file__).parent.absolute()
MODELS_DIR = CURRENT_DIR / "models"

# Model URLs from your Google Drive
MODEL_URLS = {
    "xgboost": "https://drive.google.com/file/d/1q9LIMFBtJ-VxNaXQ9ajoQ9af-rvFZyak/view?usp=drive_link",
    "svm": "https://drive.google.com/file/d/1cObIWea7gket7gpd76jEkpsZYZ_rqRDe/view?usp=drive_link", 
    "distilbert_v2": "https://drive.google.com/drive/folders/1jGywDyU9YNgWjwkQ_68twh-55UsIE92h?usp=drive_link",
    "roberta": "https://drive.google.com/drive/folders/1ucBdYBKVuNMzR5x5KjszGrCUthKdyUDA?usp=drive_link"
}

# Local paths - Updated for Windows local development
MODEL_PATHS = {
    "xgboost": str(MODELS_DIR / "xgboost_model.pkl"),
    "svm": str(MODELS_DIR / "svm_model.pkl"),
    "distilbert_v2": str(MODELS_DIR / "distilbert_v2_model.pkl"),
    "roberta": str(MODELS_DIR / "roberta_model.pkl")
}

# API Configuration
API_CONFIG = {
    "host": "127.0.0.1",  # localhost
    "port": 8000,  # Backend port
    "frontend_url": "http://localhost:8081",  # Your frontend URL
    "reload": True,  # Enable auto-reload for development
    "debug": True
}

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)
print(f"📁 Models directory: {MODELS_DIR}")
