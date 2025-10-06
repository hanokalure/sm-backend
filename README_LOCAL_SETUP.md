# SMS Spam Detection - Local Development Setup

🚀 Complete guide to run your SMS Spam Detection API locally and connect it to your React Expo frontend.

## 📋 Overview

This setup connects:
- **Backend API** (Python FastAPI): `http://localhost:8000`
- **Frontend** (React Expo): `http://localhost:8081` 
- **Models Directory**: `./models/` (for your trained ML models)

## 🛠️ Quick Setup

### 1. Run the Setup Script

```bash
# In the backend directory
python setup_local.py
```

This will:
- ✅ Check Python version compatibility 
- ✅ Install required dependencies
- ✅ Create models directory structure
- ✅ Configure environment variables
- ✅ Create startup scripts

### 2. Add Your Models

Place your trained models in the `models/` directory:

```
models/
├── xgboost_model.pkl           # XGBoost spam classifier
├── svm_model.pkl              # Support Vector Machine model  
├── distilbert_v2_model.pkl    # DistilBERT Deep Classifier V2
├── roberta_model.pkl          # RoBERTa Ultimate Spam Detector
└── README.md                  # Model setup instructions
```

### 3. Start the Backend

**Windows:**
```bash
# Double-click the batch file
run_api.bat

# OR run manually
python main.py
```

**Mac/Linux:**
```bash
# Make executable and run
chmod +x run_api.sh
./run_api.sh

# OR run manually
python3 main.py
```

The backend will start at: `http://localhost:8000`

### 4. Start Your Frontend

```bash
# Navigate to your frontend directory
cd C:\sms_spam_detection\frontend

# Install dependencies (first time only)
npm install

# Start the development server
npm start
# OR for Expo
expo start
```

Your frontend will start at: `http://localhost:8081`

## 🧪 Test the Connection

Run the connection test script:

```bash
# Make sure backend is running first
python test_connection.py
```

This will test:
- ✅ API health check
- ✅ Models endpoint
- ✅ Prediction endpoint
- ✅ CORS configuration

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and status |
| `/health` | GET | Health check and models status |
| `/models` | GET | Available models information |
| `/predict` | POST | Single message prediction |
| `/predict/batch` | POST | Batch message predictions |

### Example API Usage

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Single Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "CONGRATULATIONS! You won $1000! Click here now!",
    "model": "xgboost"
  }'
```

**Get Models:**
```bash
curl http://localhost:8000/models
```

## 🔧 Configuration

### Backend Configuration (`config.py`)

```python
API_CONFIG = {
    "host": "127.0.0.1",               # Backend host
    "port": 8000,                      # Backend port  
    "frontend_url": "http://localhost:8081",  # Your frontend URL
    "reload": True,                    # Auto-reload for development
    "debug": True                      # Debug mode
}
```

### Frontend Configuration (`.env.local`)

```env
REACT_APP_API_URL=http://localhost:8000
NEXT_PUBLIC_API_URL=http://localhost:8000
NODE_ENV=development
REACT_APP_DEBUG=true
```

## 🚚 Deploy to Friend's Laptop

### For Your Friend:

1. **Copy the entire backend folder** to their computer

2. **Run the setup script:**
   ```bash
   python setup_local.py
   ```

3. **Copy your models** to their `models/` directory

4. **Start the backend:**
   ```bash
   python main.py
   ```

5. **Copy and setup frontend** (if needed):
   ```bash
   # Copy C:\sms_spam_detection\frontend to their computer
   # Then:
   cd frontend
   npm install
   npm start
   ```

### Quick Deploy Package

To create a deployment package:

```bash
# 1. Zip the entire backend folder
# 2. Include the models/ directory with your trained models
# 3. Include setup_local.py
# 4. Include README_LOCAL_SETUP.md (this file)

# Your friend just needs to:
# - Extract the zip
# - Run: python setup_local.py
# - Run: python main.py
```

## 🔍 Troubleshooting

### Backend Issues

**Port 8000 already in use:**
```python
# In config.py, change:
API_CONFIG = {
    "port": 8001,  # Use different port
    # ... rest of config
}
```

**Models not loading:**
```bash
# Check models directory:
ls models/

# Ensure exact filenames:
# - xgboost_model.pkl
# - svm_model.pkl  
# - distilbert_v2_model.pkl
# - roberta_model.pkl
```

**Dependencies issues:**
```bash
# Reinstall dependencies:
pip install --upgrade -r requirements.txt

# Or create virtual environment:
python -m venv spam_detection_env
# Windows:
spam_detection_env\Scripts\activate
# Mac/Linux:  
source spam_detection_env/bin/activate
pip install -r requirements.txt
```

### Frontend Issues

**Cannot connect to backend:**
1. Ensure backend is running: `http://localhost:8000/health`
2. Check frontend `.env.local` file has correct URL
3. Clear browser cache and restart frontend

**Port 8081 already in use:**
```bash
# Start frontend on different port:
PORT=8082 npm start
# Or for Expo:
expo start --port 8082

# Then update backend config.py:
API_CONFIG = {
    "frontend_url": "http://localhost:8082",
    # ... rest of config  
}
```

### CORS Issues

If you get CORS errors:
```python
# In main.py, ensure your frontend URL is in CORS origins:
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081",  # Your frontend URL
        "http://localhost:8082",  # Alternative port
        # Add more as needed
    ],
    # ... rest of CORS config
)
```

## 📊 Model Performance

| Model | Accuracy | Dataset Size | Focus |
|-------|----------|--------------|-------|
| XGBoost High-Accuracy | 98.13% | 5,885 | Comprehensive spam detection |
| DistilBERT Deep V2 | 98.59% | 4,962 | Banking/financial accuracy |
| RoBERTa Ultimate | 99.72% | 100,000 | State-of-the-art transformer |
| SVM Baseline | 95.0% | 2,940 | Fast baseline detection |

## 🔗 Quick Links

- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health  
- **Frontend:** http://localhost:8081

## 📞 Support

If you encounter issues:

1. **Check the logs** in the terminal where you ran `python main.py`
2. **Run the test script:** `python test_connection.py`
3. **Verify model files** are in `models/` directory
4. **Check ports** aren't already in use
5. **Review configuration** in `config.py` and `.env.local`

---

🎉 **You're all set!** Your SMS Spam Detection API should now be running locally and connected to your React Expo frontend.