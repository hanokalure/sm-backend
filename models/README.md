# Models Directory

This directory should contain your trained ML models for the SMS Spam Detection API.

⚠️ **Important**: Model files are **NOT included** in this GitHub repository due to their large size (10-500MB each).

## Expected Files

Place your trained model files here with these exact names:

```
models/
├── xgboost_model.pkl           # XGBoost spam classifier (pickled)
├── svm_model.pkl              # Support Vector Machine model (pickled)
├── distilbert_v2_model.pkl    # DistilBERT Deep Classifier V2 (pickled)
├── roberta_model.pkl          # RoBERTa Ultimate Spam Detector (pickled)
└── README.md                  # This file
```

## Model Formats

- **XGBoost & SVM**: Pickled scikit-learn/XGBoost models with vectorizers
- **DistilBERT & RoBERTa**: Pickled Hugging Face transformer models with tokenizers

## Setup Instructions

### Option 1: Use Your Own Models
1. Train your own models using the training scripts in `backend/`
2. Save them as pickle files with the exact names above
3. Place them in this directory

### Option 2: Download Pre-trained Models
If you have pre-trained models hosted elsewhere:
```bash
python download_models.py
```

### Option 3: Manual Download
1. Download your trained models from Google Drive or your model storage
2. Place them in this directory with the exact names listed above
3. Restart the API server

## Without Models

The API will work with **smart rule-based fallback** if model files are not present:
- ✅ API will start and function normally
- ⚠️ Accuracy will be lower (~78% vs 98%+)
- 💡 For production use, trained models are highly recommended

## Verification

After adding models, verify they're loaded correctly:

```bash
# Start the API
python main.py

# Test the models
python test_connection.py

# Check model status
curl http://localhost:8000/models
```

## File Sizes

Typical model file sizes:
- XGBoost: 10-50 MB
- SVM: 5-20 MB
- DistilBERT: 250-500 MB
- RoBERTa: 300-600 MB

## Security Note

- Keep model files local - don't commit to version control
- Models may contain sensitive training data
- Use secure storage for sharing models between team members

---

**Need help?** Check the main [README.md](../README.md) or run `python setup_local.py`
