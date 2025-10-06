# 🚀 SMS Spam Detection API

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **AI-powered SMS spam detection system with multiple ML models including XGBoost, DistilBERT, and RoBERTa. Includes FastAPI backend and React Expo frontend integration.**

## ✨ Features

- 🤖 **Multiple AI Models**: XGBoost, SVM, DistilBERT V2, RoBERTa Ultimate
- 📊 **High Accuracy**: Up to 99.72% accuracy with RoBERTa model
- 🚀 **FastAPI Backend**: RESTful API with automatic documentation
- 📱 **Frontend Ready**: CORS configured for React Expo integration
- 🔄 **Batch Processing**: Handle multiple messages simultaneously
- 🛡️ **Banking Focus**: Enhanced accuracy for financial message detection
- 📈 **Real-time Monitoring**: Health checks and performance metrics
- 🔧 **Easy Setup**: One-command installation and configuration

## 🎯 Model Performance

| Model | Accuracy | Dataset Size | Focus Area |
|-------|----------|--------------|------------|
| **RoBERTa Ultimate** | 99.72% | 100,000+ | State-of-the-art transformer |
| **DistilBERT Deep V2** | 98.59% | 4,962 | Banking/Financial messages |
| **XGBoost High-Accuracy** | 98.13% | 5,885 | Comprehensive detection |
| **SVM Baseline** | 95.0% | 2,940 | Fast lightweight detection |

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sms-spam-detector.git
cd sms-spam-detector
```

### 2. Run Setup Script

```bash
python setup_local.py
```

This automatically:
- ✅ Checks Python version compatibility
- ✅ Installs all dependencies
- ✅ Creates models directory structure
- ✅ Configures environment variables
- ✅ Creates startup scripts

### 3. Download Models

⚠️ **Important**: The trained models are not included in this repository due to GitHub's file size limitations.

**Option A: Use Your Own Models**
Place your trained model files in the `models/` directory:
```
models/
├── xgboost_model.pkl
├── svm_model.pkl
├── distilbert_v2_model.pkl
└── roberta_model.pkl
```

**Option B: Download from External Source**
```bash
# If you have models hosted elsewhere (Google Drive, etc.)
python download_models.py
```

### 4. Start the API

```bash
# Windows
python main.py

# Mac/Linux
python3 main.py
```

🎉 **API is now running at:** `http://localhost:8000`

### 5. Test the API

```bash
python test_connection.py
```

## 📡 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "CONGRATULATIONS! You won $1000! Click here now!",
       "model": "xgboost"
     }'
```

**Response:**
```json
{
  "prediction": "SPAM",
  "confidence": 0.987,
  "risk_level": "HIGH RISK",
  "model_name": "XGBoost High-Accuracy",
  "processing_time_ms": 45.2
}
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "texts": [
         "Hello, how are you?",
         "WIN FREE MONEY NOW!",
         "Your account balance is $1,234.56"
       ],
       "model": "xgboost"
     }'
```

## 📋 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and status |
| `/health` | GET | Health check and model status |
| `/models` | GET | List available models and their info |
| `/predict` | POST | Single message spam detection |
| `/predict/batch` | POST | Batch message spam detection |
| `/docs` | GET | Interactive API documentation |

## 🛠️ Installation & Setup

### Prerequisites

- **Python 3.8+** (Python 3.11+ recommended)
- **pip** package manager
- **Git** (for cloning the repository)

### Manual Installation

1. **Clone and Navigate:**
   ```bash
   git clone https://github.com/yourusername/sms-spam-detector.git
   cd sms-spam-detector
   ```

2. **Create Virtual Environment (Recommended):**
   ```bash
   python -m venv spam_detection_env
   # Windows:
   spam_detection_env\Scripts\activate
   # Mac/Linux:
   source spam_detection_env/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment:**
   ```bash
   # The setup script creates a .env file
   # Edit it if needed for your configuration
   ```

5. **Add Your Models:**
   ```bash
   # Place your trained models in models/ directory
   # See "Models" section below for details
   ```

6. **Start API:**
   ```bash
   python main.py
   ```

## 🧠 Models

### Model Files Required

The API expects the following model files in the `models/` directory:

```
models/
├── xgboost_model.pkl           # XGBoost classifier (pickled)
├── svm_model.pkl              # SVM classifier (pickled)
├── distilbert_v2_model.pkl    # DistilBERT model (pickled)
└── roberta_model.pkl          # RoBERTa model (pickled)
```

### Model Formats

- **XGBoost & SVM**: Pickled scikit-learn/XGBoost models with vectorizers
- **DistilBERT & RoBERTa**: Pickled Hugging Face transformer models with tokenizers

### Without Models

The API will work with **smart rule-based fallback** if model files are not present, but for best performance, trained models are recommended.

### Training Your Own Models

If you want to train your own models, you can:

1. Use the training scripts in the `backend/` directory
2. Prepare your dataset in CSV format with `text` and `label` columns
3. Run the training scripts to generate model files
4. Place the generated models in the `models/` directory

## 🔧 Configuration

### Backend Configuration (`config.py`)

```python
API_CONFIG = {
    "host": "127.0.0.1",               # API host
    "port": 8000,                      # API port
    "frontend_url": "http://localhost:8081",  # Frontend URL (for CORS)
    "reload": True,                    # Auto-reload in development
    "debug": True                      # Debug mode
}
```

### Environment Variables (`.env`)

```bash
# API Configuration
API_HOST=127.0.0.1
API_PORT=8000
FRONTEND_URL=http://localhost:8081

# Development Settings
DEBUG=True
LOG_LEVEL=INFO

# Model Configuration
MODELS_DIR=./models
```

## 📱 Frontend Integration

### React Expo Setup

1. **Configure Frontend Environment** (`.env.local`):
   ```bash
   REACT_APP_API_URL=http://localhost:8000
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

2. **Start Frontend:**
   ```bash
   cd your-frontend-directory
   npm install
   npm start  # or expo start
   ```

3. **Test Connection:**
   Your React Expo app should now be able to communicate with the API at `http://localhost:8000`

### JavaScript Example

```javascript
// Example API call from React/JavaScript
const predictSpam = async (message) => {
  try {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: message,
        model: 'xgboost'  // Choose your preferred model
      }),
    });
    
    const result = await response.json();
    console.log('Prediction:', result.prediction);
    console.log('Confidence:', result.confidence);
    return result;
  } catch (error) {
    console.error('API Error:', error);
  }
};
```

## 🧪 Testing

### Run Test Suite
```bash
# Test API connectivity
python test_connection.py

# Test specific endpoint
curl -X GET "http://localhost:8000/health"

# Test prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Test message", "model": "xgboost"}'
```

### Example Test Messages

```python
# HAM (Legitimate) Messages
ham_messages = [
    "Hey, are we still meeting for lunch tomorrow?",
    "Your bank account ending in 1234 was credited $500.00",
    "Thanks for the meeting today. The presentation was great!"
]

# SPAM Messages  
spam_messages = [
    "CONGRATULATIONS! You've won $50,000! Reply YES to claim!",
    "URGENT: Your account will be closed! Click here immediately!",
    "FREE iPhone! Limited time offer! Text WIN to 12345!"
]
```

## 🔍 Troubleshooting

### Common Issues

**1. Port 8000 already in use:**
```python
# In config.py, change the port:
API_CONFIG = {"port": 8001}
```

**2. Models not loading:**
```bash
# Check models directory
ls models/
# Should contain: *.pkl files

# The API will work with fallback rules if models are missing
```

**3. CORS errors from frontend:**
```python
# In main.py, update CORS origins:
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8081",
        "https://yourfrontend.com"
    ]
)
```

**4. Dependencies issues:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or create fresh environment
python -m venv fresh_env
fresh_env\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 🚚 Deployment

### Local Development
```bash
python main.py
```

### Production Deployment

#### Docker
```bash
# Build Docker image
docker build -t sms-spam-detector .

# Run container
docker run -p 8000:8000 sms-spam-detector
```

#### Cloud Deployment (Heroku, AWS, etc.)
1. Update `config.py` for production settings
2. Set environment variables
3. Deploy using your preferred cloud platform
4. Update CORS settings for your frontend domain

## 📁 Project Structure

```
sms-spam-detector/
├── main.py                    # FastAPI application
├── config.py                  # Configuration settings
├── setup_local.py             # Setup automation script
├── test_connection.py         # Connection testing
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
├── README.md                 # This file
├── backend/                  # Backend modules
│   └── distilbert_deep_classifier_v2.py
├── models/                   # Model files (not in repo)
│   ├── README.md            # Model setup instructions
│   ├── xgboost_model.pkl    # XGBoost model (add your own)
│   ├── svm_model.pkl        # SVM model (add your own)
│   ├── distilbert_v2_model.pkl # DistilBERT model (add your own)
│   └── roberta_model.pkl    # RoBERTa model (add your own)
└── run_api.bat              # Windows startup script
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Areas for Contribution

- 🧠 **New Models:** Add support for other ML models
- 📊 **Data:** Contribute training datasets
- 🌐 **Frontend:** Build web interfaces
- 📱 **Mobile:** Create mobile applications
- 📚 **Documentation:** Improve guides and examples
- 🧪 **Testing:** Add more test cases

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for transformer models
- **FastAPI** for the amazing web framework
- **scikit-learn** for machine learning tools
- **XGBoost** for gradient boosting algorithms

## 📞 Support & Contact

- **GitHub Issues:** [Report bugs or request features](https://github.com/yourusername/sms-spam-detector/issues)
- **Documentation:** Check the `README_LOCAL_SETUP.md` for detailed setup instructions

---

## 🔗 Quick Links

- 📚 **[API Documentation](http://localhost:8000/docs)** (when running locally)
- 📋 **[Local Setup Guide](README_LOCAL_SETUP.md)**
- 🧪 **[Connection Testing](test_connection.py)**

---

**Made with ❤️ for SMS spam detection**

⭐ **Star this repository if it helped you!** ⭐