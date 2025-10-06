#!/usr/bin/env python3
"""
HIGH-ACCURACY XGBOOST TRAINING
=============================
Combines all available datasets for maximum accuracy
- UCI SMS Spam Collection (5,572 messages)
- SPAM Text Messages 2017 (5,572 messages) 
- Original dataset (2,940 messages)
- Enhanced examples (custom phishing/scam samples)
Total: ~14,000+ messages
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import xgboost as xgb
import re
from pathlib import Path
import shutil
from datetime import datetime

class AdvancedSMSPreprocessor:
    """Advanced SMS preprocessor for high-accuracy spam detection"""
    
    def __init__(self):
        # Banking/Financial institutions
        self.legitimate_banks = [
            'sbi', 'hdfc', 'icici', 'axis', 'kotak', 'pnb', 'canara', 'bob', 
            'citibank', 'hsbc', 'wells fargo', 'chase', 'bank of america'
        ]
        
        # Tech companies  
        self.legitimate_tech = [
            'microsoft', 'apple', 'google', 'amazon', 'facebook', 'meta',
            'twitter', 'instagram', 'whatsapp', 'linkedin', 'paypal'
        ]
        
        # Authority terms
        self.authority_terms = [
            'irs', 'tax office', 'government', 'police', 'court', 'fbi',
            'customs', 'immigration', 'social security', 'medicare'
        ]
        
        # Scam indicators by category
        self.scam_indicators = {
            'urgency': ['urgent', 'immediately', 'expires', 'deadline', 'final notice', 'act now', 'limited time'],
            'threats': ['suspended', 'blocked', 'deactivated', 'frozen', 'cancelled', 'terminated', 'arrested'],
            'financial': ['refund', 'tax', 'owe', 'debt', 'payment failed', 'overdraft', 'credit score'],
            'prizes': ['winner', 'congratulations', 'selected', 'lottery', 'prize', 'reward', 'gift card'],
            'tech_support': ['virus', 'malware', 'infected', 'security alert', 'system error', 'update required'],
            'verification': ['verify', 'confirm', 'update details', 'reactivate', 'click here', 'login'],
            'crypto': ['bitcoin', 'crypto', 'investment', 'trading', 'profit', 'returns', 'doubled'],
            'romance': ['love', 'military', 'overseas', 'inheritance', 'funds', 'transfer', 'lonely']
        }
    
    def extract_features(self, text):
        """Extract 30 advanced numerical features"""
        if not isinstance(text, str):
            return np.zeros(30)
            
        text_lower = text.lower()
        features = []
        
        # Basic text statistics (7 features)
        features.append(len(text))  # Length
        features.append(len(text.split()))  # Word count  
        features.append(text.count('!'))  # Exclamation marks
        features.append(text.count('?'))  # Question marks
        features.append(text.count('$') + text.count('£') + text.count('€'))  # Currency
        features.append(len(re.findall(r'\d+', text)))  # Numbers
        features.append(sum(1 for c in text if c.isupper()) / max(len(text), 1))  # Uppercase ratio
        
        # Contact indicators (3 features)
        features.append(1 if re.search(r'http|www|\.com|\.net|\.org', text_lower) else 0)  # Has URL
        features.append(1 if re.search(r'\b\d{10,}\b', text) else 0)  # Has phone
        features.append(1 if re.search(r'\b\d{4,6}\b', text) else 0)  # Has short code
        
        # Scam category features (8 features)
        for category, keywords in self.scam_indicators.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            features.append(count)
        
        # Authority/legitimacy features (3 features)
        authority_count = sum(1 for term in self.authority_terms if term in text_lower)
        bank_count = sum(1 for bank in self.legitimate_banks if bank in text_lower)
        tech_count = sum(1 for tech in self.legitimate_tech if tech in text_lower)
        features.append(authority_count)
        features.append(bank_count)
        features.append(tech_count)
        
        # Advanced text analysis (9 features)
        words = text.split()
        features.append(np.mean([len(word) for word in words]) if words else 0)  # Avg word length
        features.append(len(set(words)))  # Unique words
        features.append(text.count('.'))  # Periods
        features.append(text.count(','))  # Commas
        features.append(1 if re.search(r'[A-Z]{3,}', text) else 0)  # Has caps words
        features.append(len(re.findall(r'[a-zA-Z]+', text)))  # Letter sequences
        features.append(text.count(' '))  # Spaces
        features.append(1 if any(char in text for char in '()[]{}') else 0)  # Has brackets
        features.append(text.count('-'))  # Dashes
        
        return np.array(features[:30], dtype=float)  # Ensure exactly 30 features
    
    def normalize_text(self, text):
        """Enhanced text normalization"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower().strip()
        
        # Normalize patterns  
        text = re.sub(r'\b\d{10,}\b', 'PHONE_NUMBER', text)
        text = re.sub(r'[₹£$€]\s*[\d,]+(?:\.\d{2})?', 'MONEY_AMOUNT', text)
        text = re.sub(r'\b\d{4,6}\b', 'CODE', text)
        text = re.sub(r'http\S+|www\.\S+', 'URL', text)
        
        # Clean extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


def load_all_datasets():
    """Load and combine all available datasets"""
    print("📥 Loading all datasets...")
    
    combined_data = []
    
    # 1. UCI SMS Spam Collection
    try:
        df1 = pd.read_csv('C:/006_sms/backend/data/spam.csv', encoding='latin1')
        df1 = df1[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
        df1 = df1.dropna()
        combined_data.append(df1)
        print(f"✅ UCI SMS Dataset: {len(df1):,} messages")
    except Exception as e:
        print(f"⚠️ UCI dataset error: {e}")
    
    # 2. SPAM Text Messages 2017
    try:
        df2 = pd.read_csv('C:/006_sms/backend/data/SPAM text message 20170820 - Data.csv', encoding='latin1')
        df2 = df2[['Category', 'Message']].rename(columns={'Category': 'label', 'Message': 'text'})
        df2 = df2.dropna()
        combined_data.append(df2)
        print(f"✅ SPAM 2017 Dataset: {len(df2):,} messages")
    except Exception as e:
        print(f"⚠️ SPAM 2017 dataset error: {e}")
    
    # 3. Original dataset
    try:
        df3 = pd.read_csv('C:/006_sms/data/sms_spam.csv')
        df3 = df3.dropna()
        combined_data.append(df3)
        print(f"✅ Original Dataset: {len(df3):,} messages")
    except Exception as e:
        print(f"⚠️ Original dataset error: {e}")
    
    # 4. Enhanced examples (banking, tech, authority scams)
    enhanced_data = [
        ("Dear customer, your SBI account has been temporarily suspended. Verify at sbi-secure.com immediately.", "spam"),
        ("HDFC Alert: Your card blocked. Unblock at hdfc-unlock.net", "spam"),  
        ("Microsoft Security: 5 viruses detected. Call 1-800-MS-HELP now.", "spam"),
        ("Apple Support: iPhone compromised. Contact apple-urgent.com", "spam"),
        ("IRS Notice: You owe $2500. Pay at irs-payment.gov or face arrest.", "spam"),
        ("FBI Alert: IP flagged for illegal activity. Pay fine at fbi-fine.gov", "spam"),
        ("Security Alert: Login detected. OTP: 123456. Verify at secure-login.com", "spam"),
        ("Bitcoin Giveaway: Elon Musk giving 10 BTC. Send 0.1 BTC to get 2 BTC back.", "spam"),
        ("Your SBI account credited Rs. 5000. Balance: Rs. 25430. Thank you for banking with us.", "ham"),
        ("HDFC: EMI of Rs. 15500 debited. Next due: 15th March.", "ham"),
        ("Your OTP for ICICI login is 847263. Valid for 5 minutes. Do not share.", "ham"),
        ("Microsoft Office 365 renewed. $99.99 charged. Active until March 2025.", "ham"),
        ("Google: Someone signed in from new device. If this wasn't you, secure your account.", "ham"),
        ("PayPal: You sent $25.00 to John. Transaction ID: 1A234B567C.", "ham"),
        ("Amazon: Order dispatched via UPS. Track at amazon.com/track", "ham"),
        ("Your appointment with Dr. Johnson confirmed for tomorrow 2:30 PM.", "ham"),
    ]
    
    df4 = pd.DataFrame(enhanced_data, columns=['text', 'label'])
    combined_data.append(df4)
    print(f"✅ Enhanced Examples: {len(df4):,} messages")
    
    # Combine all datasets
    if not combined_data:
        raise Exception("No datasets loaded!")
    
    final_df = pd.concat(combined_data, ignore_index=True)
    
    # Clean and standardize
    final_df = final_df.dropna()
    final_df['label'] = final_df['label'].str.lower()
    final_df = final_df[final_df['label'].isin(['ham', 'spam'])]
    
    # Remove duplicates
    final_df = final_df.drop_duplicates(subset=['text'])
    
    print(f"\n🎯 COMBINED DATASET:")
    print(f"📊 Total messages: {len(final_df):,}")
    spam_count = sum(final_df['label'] == 'spam')
    ham_count = sum(final_df['label'] == 'ham')
    print(f"📈 Spam: {spam_count:,} ({spam_count/len(final_df)*100:.1f}%)")
    print(f"📈 Ham: {ham_count:,} ({ham_count/len(final_df)*100:.1f}%)")
    
    return final_df


def train_high_accuracy_xgboost(df):
    """Train XGBoost with optimized parameters for maximum accuracy"""
    print("\n🚀 Training High-Accuracy XGBoost Model")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = AdvancedSMSPreprocessor()
    
    # Process texts and extract features
    print("🔄 Processing texts and extracting features...")
    processed_texts = [preprocessor.normalize_text(text) for text in df['text']]
    numerical_features = np.array([preprocessor.extract_features(text) for text in df['text']])
    
    # TF-IDF vectorization with optimal parameters
    print("🔄 Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Increased for better coverage
        ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
        min_df=2,
        max_df=0.85,
        stop_words='english',
        lowercase=True,
        sublinear_tf=True
    )
    
    tfidf_features = vectorizer.fit_transform(processed_texts).toarray()
    
    # Combine features
    X = np.hstack([tfidf_features, numerical_features])
    y = df['label'].map({'ham': 0, 'spam': 1}).values
    
    print(f"📊 Feature matrix: {X.shape}")
    print(f"📈 TF-IDF features: {tfidf_features.shape[1]:,}")  
    print(f"📈 Numerical features: {numerical_features.shape[1]}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Optimized XGBoost parameters for high accuracy
    print("🔄 Training with optimized parameters...")
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 8,  # Increased depth
        'learning_rate': 0.05,  # Lower learning rate for precision
        'n_estimators': 500,  # More estimators
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.5,  # More regularization
        'min_child_weight': 3,
        'gamma': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n📊 Test Results:")
    print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 F1-Score: {f1:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Skip intensive cross-validation for faster training
    print(f"\n✅ Training complete! Using test set accuracy for final metrics.")
    
    return model, vectorizer, preprocessor, accuracy, f1


def main():
    """Main training pipeline"""
    print("🔥 HIGH-ACCURACY XGBOOST TRAINING")
    print("="*80)
    print("🎯 Goal: Maximum accuracy with 14,000+ messages")
    print("📊 Datasets: UCI + SPAM2017 + Original + Enhanced")
    print("🚀 Features: 10K TF-IDF + 30 numerical features")
    print("="*80)
    
    try:
        # Backup existing model
        existing_model = "backend/models/xgboost_model.pkl"
        if os.path.exists(existing_model):
            backup_path = f"backend/models/xgboost_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            shutil.copy2(existing_model, backup_path)
            print(f"💾 Backed up existing model to: {backup_path}")
        
        # Load all datasets
        df = load_all_datasets()
        
        # Train model
        model, vectorizer, preprocessor, accuracy, f1_score = train_high_accuracy_xgboost(df)
        
        # Save new model
        model_data = {
            'model': model,
            'vectorizer': vectorizer, 
            'preprocessor': preprocessor,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'dataset_size': len(df),
            'name': 'XGBoost High-Accuracy Spam Detector',
            'training_focus': 'Comprehensive spam detection with 14K+ messages',
            'model_type': 'xgboost',
            'features': 'TF-IDF + Advanced numerical features',
            'version': '2.0_high_accuracy',
            'training_date': datetime.now().isoformat()
        }
        
        os.makedirs("backend/models", exist_ok=True)
        with open(existing_model, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n🎉 HIGH-ACCURACY TRAINING COMPLETE!")
        print("="*80)
        print(f"✅ Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"✅ F1-Score: {f1_score:.4f}")
        print(f"📊 Trained on: {len(df):,} messages")
        print(f"💾 Model saved: {existing_model}")
        print(f"🚀 Ready for high-accuracy predictions!")
        
        if accuracy > 0.92:
            print("🏆 EXCELLENT: >92% accuracy achieved!")
        elif accuracy > 0.88:
            print("✅ VERY GOOD: >88% accuracy achieved!")
        else:
            print("⚠️ Consider adding more training data")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()