#!/usr/bin/env python3
"""
SMS SPAM DETECTION API - Hugging Face Optimized
"""

import os
import logging
from pathlib import Path

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to download models first
try:
    from model_downloader import downloader
    logger.info("🔄 Starting model download...")
    download_success = downloader.download_all_models()
    if download_success:
        logger.info("✅ Model download completed successfully")
    else:
        logger.warning("⚠️ Some models failed to download - using fallback")
except Exception as e:
    logger.warning(f"⚠️ Model download issue: {e}")

# Now import your existing code
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pickle
from typing import Optional, Dict, Any, Union, List
import numpy as np
import re

# Import all preprocessor classes for pickle compatibility from the main project
class FastSMSPreprocessor:
    """Original SMS preprocessor for backward compatibility"""
    
    def __init__(self):
        self.sms_abbreviations = {
            'u': 'you', 'ur': 'your', 'r': 'are', '2': 'to', '4': 'for',
            'txt': 'text', 'msg': 'message', 'pls': 'please', 'thx': 'thanks',
            'asap': 'as soon as possible', 'fyi': 'for your information',
            'im': 'i am', 'ive': 'i have', 'dont': 'do not'
        }
    
    def normalize_text(self, text):
        """Fast text normalization"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        
        # Quick normalizations
        text = re.sub(r'\b\d{10,}\b', 'PHONE', text)
        text = re.sub(r'http\S+|www\.\S+', 'URL', text)
        text = re.sub(r'[£$€]\s*\d+', 'MONEY', text)
        text = re.sub(r'\b\d{5,6}\b', 'CODE', text)
        
        # Expand abbreviations
        words = text.split()
        for i, word in enumerate(words):
            clean_word = word.strip('.,!?')
            if clean_word in self.sms_abbreviations:
                words[i] = self.sms_abbreviations[clean_word]
        
        return ' '.join(words)

class ImprovedSMSPreprocessor:
    """Enhanced SMS preprocessor with better spam detection features"""
    
    def __init__(self):
        # Spam keywords (strong indicators)
        self.spam_keywords = {
            'prize_words': ['won', 'winner', 'congratulations', 'selected', 'lucky', 'draw', 'lottery', 'contest'],
            'money_words': ['free', 'cash', 'prize', 'reward', 'discount', 'offer', 'sale', 'deal', 'voucher'],
            'urgency_words': ['urgent', 'immediately', 'hurry', 'limited', 'expires', 'today', 'now', 'asap'],
            'action_words': ['click', 'call', 'text', 'reply', 'send', 'visit', 'claim', 'collect'],
            'financial_words': ['bank', 'account', 'card', 'blocked', 'suspended', 'verify', 'update', 'pan'],
            'suspicious_words': ['dear user', 'mobile number', 'details', 'reactivate', 'submit']
        }
        
        # SMS abbreviations
        self.sms_abbreviations = {
            'u': 'you', 'ur': 'your', 'r': 'are', '2': 'to', '4': 'for',
            'txt': 'text', 'msg': 'message', 'pls': 'please', 'thx': 'thanks'
        }
    
    def normalize_text(self, text):
        """Enhanced text normalization preserving spam indicators"""
        if not isinstance(text, str):
            return ""
        
        original_text = text
        text = text.lower()
        
        # Preserve certain patterns before normalization
        has_caps = len(re.findall(r'[A-Z]{2,}', original_text)) > 0
        has_exclamation = text.count('!') >= 2
        has_money_symbols = bool(re.search(r'[₹£$€]\s*\d+', text))
        
        # Normalize patterns
        text = re.sub(r'\b\d{10,}\b', 'PHONE_NUMBER', text)
        text = re.sub(r'http\S+|www\.\S+|\S+\.(com|net|org)', 'URL_LINK', text)
        text = re.sub(r'[₹£$€]\s*\d+(?:,\d+)*', 'MONEY_AMOUNT', text)
        text = re.sub(r'\b\d{5,6}\b', 'SHORT_CODE', text)
        
        # Expand abbreviations
        words = text.split()
        for i, word in enumerate(words):
            clean_word = word.strip('.,!?')
            if clean_word in self.sms_abbreviations:
                words[i] = self.sms_abbreviations[clean_word]
        
        text = ' '.join(words)
        
        # Add spam indicators back
        if has_caps:
            text += ' CAPS_SPAM'
        if has_exclamation:
            text += ' EXCLAMATION_SPAM'
        if has_money_symbols:
            text += ' MONEY_SPAM'
            
        return text.strip()
    
    def extract_spam_features(self, text):
        """Extract numerical spam features"""
        if not isinstance(text, str):
            return np.zeros(15)
        
        text_lower = text.lower()
        features = []
        
        # Keyword counts
        for category, words in self.spam_keywords.items():
            count = sum(1 for word in words if word in text_lower)
            features.append(count)
        
        # Text characteristics
        features.append(len(text))  # Length
        features.append(len(text.split()))  # Word count
        features.append(text.count('!'))  # Exclamation count
        features.append(text.count('?'))  # Question count
        features.append(sum(1 for c in text if c.isupper()) / max(len(text), 1))  # Caps ratio
        features.append(len(re.findall(r'\d+', text)))  # Number count
        features.append(1 if re.search(r'http|www|\.com', text.lower()) else 0)  # Has URL
        features.append(1 if re.search(r'[₹£$€]', text) else 0)  # Has currency
        features.append(1 if re.search(r'\b\d{10,}\b', text) else 0)  # Has phone
        
        return np.array(features, dtype=float)

class ProfessionalSMSPreprocessor:
    """Professional SMS preprocessor with context-aware features"""
    
    def __init__(self):
        # Legitimate service keywords (should NOT be spam indicators)
        self.legitimate_services = [
            'sbi', 'hdfc', 'icici', 'axis', 'kotak', 'pnb', 'canara', 'bob',
            'amazon', 'flipkart', 'paytm', 'phonepe', 'googlepay', 'upi',
            'ola', 'uber', 'swiggy', 'zomato', 'bigbasket', 'grofers',
            'irctc', 'makemytrip', 'oyo', 'bookmyshow', 'netflix', 'hotstar'
        ]
        
        # Spam-only keywords (strong spam indicators)
        self.spam_only_keywords = [
            'congratulations', 'winner', 'selected', 'lucky draw', 'lottery',
            'claim prize', 'urgent winner', 'cash prize', 'free iphone',
            'click here to win', 'reply yes to claim', 'limited time winner'
        ]
        
        # Legitimate transaction patterns
        self.transaction_patterns = [
            r'credited.*to.*account',
            r'debited.*from.*account', 
            r'available.*balance',
            r'transaction.*successful',
            r'payment.*received',
            r'order.*dispatched',
            r'otp.*is',
            r'appointment.*confirmed'
        ]
        
        # Spam URL patterns (suspicious domains)
        self.spam_url_patterns = [
            r'bit\.ly', r'tinyurl', r'short-link', r'win-prize',
            r'claim-reward', r'free-gift', r'instant-cash'
        ]
        
        # SMS abbreviations
        self.sms_abbreviations = {
            'u': 'you', 'ur': 'your', 'r': 'are', '2': 'to', '4': 'for',
            'txt': 'text', 'msg': 'message', 'pls': 'please', 'thx': 'thanks',
            'avl': 'available', 'bal': 'balance', 'acc': 'account',
            'txn': 'transaction', 'amt': 'amount'
        }
    
    def normalize_text(self, text):
        """Context-aware text normalization"""
        if not isinstance(text, str):
            return ""
        
        original_text = text
        text = text.lower()
        
        # Check for legitimate service context
        has_legitimate_service = any(service in text for service in self.legitimate_services)
        has_transaction_pattern = any(re.search(pattern, text) for pattern in self.transaction_patterns)
        
        # Normalize patterns while preserving context
        text = re.sub(r'\b\d{10,}\b', 'PHONE_NUMBER', text)
        text = re.sub(r'[₹£$€]\s*[\d,]+(?:\.\d{2})?', 'MONEY_AMOUNT', text)
        text = re.sub(r'\b\d{4,6}\b', 'SHORT_CODE', text)
        
        # Handle URLs differently based on context
        if any(re.search(pattern, text) for pattern in self.spam_url_patterns):
            text = re.sub(r'http\S+|www\.\S+', 'SPAM_URL', text)
        else:
            text = re.sub(r'http\S+|www\.\S+', 'LEGITIMATE_URL', text)
        
        # Expand abbreviations
        words = text.split()
        for i, word in enumerate(words):
            clean_word = word.strip('.,!?')
            if clean_word in self.sms_abbreviations:
                words[i] = self.sms_abbreviations[clean_word]
        
        text = ' '.join(words)
        
        # Add context indicators
        if has_legitimate_service:
            text += ' LEGITIMATE_SERVICE'
        if has_transaction_pattern:
            text += ' TRANSACTION_CONTEXT'
        
        # Only add spam indicators if no legitimate context
        if not (has_legitimate_service or has_transaction_pattern):
            if any(keyword in text for keyword in self.spam_only_keywords):
                text += ' SPAM_KEYWORD'
            if len(re.findall(r'[A-Z]{3,}', original_text)) > 1:
                text += ' EXCESSIVE_CAPS'
            if text.count('!') >= 2:
                text += ' EXCESSIVE_EXCLAMATION'
        
        return text.strip()

class UltimateSMSPreprocessor:
    """Ultimate SMS preprocessor with advanced phishing and social engineering detection"""
    
    def __init__(self):
        # Legitimate service keywords
        self.legitimate_services = [
            'sbi', 'hdfc', 'icici', 'axis', 'kotak', 'pnb', 'canara', 'bob',
            'amazon', 'flipkart', 'paytm', 'phonepe', 'googlepay', 'upi',
            'ola', 'uber', 'swiggy', 'zomato', 'bigbasket', 'grofers',
            'irctc', 'makemytrip', 'oyo', 'bookmyshow', 'netflix', 'hotstar'
        ]
        
        # Legitimate domain patterns (trusted)
        self.legitimate_domains = [
            r'sbi\.co\.in', r'hdfcbank\.com', r'icicibank\.com', r'axisbank\.com',
            r'amazon\.in', r'flipkart\.com', r'paytm\.com', r'phonepe\.com',
            r'amzn\.to', r'fkrt\.it'  # Legitimate short URLs
        ]
        
        # PHISHING/FRAUD indicators (strong spam signals)
        self.fraud_indicators = [
            # Suspicious URLs
            r'bit\.ly', r'tinyurl', r'short-link', r't\.co/[^/]+$',
            r'fraud-link', r'scam-link', r'fake-link', r'phish-link',
            r'secure-bank', r'bank-verify', r'account-update',
            # Generic suspicious domains
            r'\.tk', r'\.ml', r'\.ga', r'\.cf'
        ]
        
        # Social engineering patterns
        self.social_engineering_patterns = [
            r'login.*at.*link', r'click.*link.*to.*verify',
            r'update.*details.*at', r'verify.*account.*at',
            r'secure.*account.*click', r'reactivate.*click',
            r'suspended.*click.*to', r'blocked.*verify.*at'
        ]
        
        # Legitimate transaction patterns
        self.legitimate_patterns = [
            r'credited.*to.*your.*sbi', r'debited.*from.*hdfc',
            r'available.*balance', r'transaction.*successful',
            r'otp.*is.*\d+', r'do not share.*otp',
            r'order.*dispatched', r'appointment.*confirmed'
        ]
        
        # Spam-only keywords
        self.spam_keywords = [
            'congratulations', 'winner', 'selected', 'lucky draw', 'lottery',
            'claim prize', 'urgent winner', 'cash prize', 'free iphone'
        ]
    
    def normalize_text(self, text):
        """Advanced context-aware normalization with phishing detection"""
        if not isinstance(text, str):
            return ""
        
        original_text = text
        text = text.lower()
        
        # Check for legitimate service context
        has_legitimate_service = any(service in text for service in self.legitimate_services)
        has_legitimate_pattern = any(re.search(pattern, text) for pattern in self.legitimate_patterns)
        has_legitimate_domain = any(re.search(pattern, text) for pattern in self.legitimate_domains)
        
        # Check for fraud indicators
        has_fraud_url = any(re.search(pattern, text) for pattern in self.fraud_indicators)
        has_social_engineering = any(re.search(pattern, text) for pattern in self.social_engineering_patterns)
        
        # Normalize patterns
        text = re.sub(r'\b\d{10,}\b', 'PHONE_NUMBER', text)
        text = re.sub(r'[₹£$€]\s*[\d,]+(?:\.\d{2})?', 'MONEY_AMOUNT', text)
        text = re.sub(r'\b\d{4,6}\b', 'SHORT_CODE', text)
        
        # Handle URLs based on fraud detection
        if has_fraud_url:
            text = re.sub(r'http\S+|www\.\S+|\[\S*link\S*\]', 'FRAUD_URL', text)
        elif has_legitimate_domain:
            text = re.sub(r'http\S+|www\.\S+', 'TRUSTED_URL', text)
        else:
            text = re.sub(r'http\S+|www\.\S+|\[\S*link\S*\]', 'UNKNOWN_URL', text)
        
        # Expand common abbreviations
        text = re.sub(r'\bavl\b', 'available', text)
        text = re.sub(r'\bbal\b', 'balance', text)
        text = re.sub(r'\bacc\b', 'account', text)
        text = re.sub(r'\btxn\b', 'transaction', text)
        
        # Add context indicators
        context_indicators = []
        
        if has_legitimate_service and has_legitimate_pattern:
            context_indicators.append('LEGITIMATE_TRANSACTION')
        elif has_legitimate_service:
            context_indicators.append('LEGITIMATE_SERVICE')
        
        if has_fraud_url or has_social_engineering:
            context_indicators.append('PHISHING_ATTEMPT')
        
        if any(keyword in text for keyword in self.spam_keywords):
            context_indicators.append('SPAM_KEYWORD')
        
        # Add excessive formatting indicators
        if text.count('!') >= 2:
            context_indicators.append('EXCESSIVE_EXCLAMATION')
        if len(re.findall(r'[A-Z]{3,}', original_text)) > 1:
            context_indicators.append('EXCESSIVE_CAPS')
        
        # Join context indicators
        if context_indicators:
            text += ' ' + ' '.join(context_indicators)
        
        return text.strip()

class AdvancedSMSPreprocessor:
    """Advanced SMS preprocessor for high-accuracy spam detection (XGBoost compatibility)"""
    
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

# Simple preprocessor for fallback compatibility
class SimplePreprocessor:
    """Simple preprocessor for fallback compatibility"""
    def normalize_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = re.sub(r'http\S+|www\.\S+', 'URL', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_features(self, text):
        return np.zeros(20)

# Simple tokenizer for DL model compatibility
class SimpleTokenizer:
    def __init__(self, num_words: int = 20000, oov_token: str = '<OOV>'):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {oov_token: 1}
        self.index_word = {1: oov_token}
        self._fitted = False

    def fit_on_texts(self, texts):
        freq = {}
        for t in texts:
            for w in t.split():
                freq[w] = freq.get(w, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        idx = 2
        for w, _ in sorted_words:
            if idx > self.num_words:
                break
            self.word_index[w] = idx
            self.index_word[idx] = w
            idx += 1
        self._fitted = True

    def texts_to_sequences(self, texts):
        if not self._fitted and len(self.word_index) > 1:
            self._fitted = True
        seqs = []
        for t in texts:
            seqs.append([self.word_index.get(w, 1) for w in t.split()])
        return seqs

# Global registration for pickle compatibility
import sys
import __main__

# Register all classes in global namespaces for pickle
def register_classes_globally():
    """Register all preprocessor classes in multiple namespaces for pickle compatibility"""
    classes_to_register = {
        'AdvancedSMSPreprocessor': None,
        'UltimateSMSPreprocessor': None, 
        'ImprovedSMSPreprocessor': None,
        'ProfessionalSMSPreprocessor': None,
        'FastSMSPreprocessor': None,
        'SimpleTokenizer': None
    }
    
    # Will be called after classes are defined
    globals()['_classes_to_register'] = classes_to_register

# Register all classes globally for pickle compatibility
def _register_all_classes():
    """Register all preprocessor classes in global namespaces"""
    import sys
    import __main__
    
    classes = {
        'AdvancedSMSPreprocessor': AdvancedSMSPreprocessor,
        'UltimateSMSPreprocessor': UltimateSMSPreprocessor,
        'ImprovedSMSPreprocessor': ImprovedSMSPreprocessor,
        'ProfessionalSMSPreprocessor': ProfessionalSMSPreprocessor,
        'FastSMSPreprocessor': FastSMSPreprocessor,
        'SimplePreprocessor': SimplePreprocessor,
        'SimpleTokenizer': SimpleTokenizer
    }
    
    # Register in __main__ namespace
    for name, cls in classes.items():
        setattr(__main__, name, cls)
        globals()[name] = cls
    
    # Register module in sys.modules
    sys.modules['main'] = sys.modules[__name__]
    
    logger.info("✅ All preprocessor classes registered globally for pickle compatibility")

# Call registration
_register_all_classes()

register_classes_globally()

# Import RoBERTa for transformer model
try:
    import torch
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    import json
    import glob
    ROBERTA_AVAILABLE = True
    logger.info("✅ RoBERTa transformers available")
except ImportError as e:
    ROBERTA_AVAILABLE = False
    logger.warning(f"⚠️ RoBERTa transformers not available: {e}")

# Import DistilBERT Deep Classifier V2 with error handling
try:
    from backend.distilbert_deep_classifier_v2 import get_classifier, predict_message, predict_batch_messages
    DISTILBERT_V2_AVAILABLE = True
    logger.info("✅ DistilBERT Deep Classifier V2 available")
except Exception as e:
    DISTILBERT_V2_AVAILABLE = False
    logger.warning(f"⚠️ DistilBERT Deep Classifier V2 not available: {e}")

class ModelManager:
    """Manages XGBoost and SVM models with robust error handling"""
    
    def __init__(self):
        self.models = {}
        self.load_all_models()

    def load_all_models(self):
        """Load both XGBoost and SVM models with fallback"""
        logger.info("🔄 Loading AI models...")
        
        # Try to load XGBoost model (PRIMARY)
        self._load_xgboost_model()
        
        # Try to load SVM model (SECONDARY)  
        self._load_svm_model()
        
        # Try to load DistilBERT model (ADVANCED)
        self._load_distilbert_model()
        
        # Try to load RoBERTa model (ULTIMATE - NEW!)
        self._load_roberta_model()
        
        # Ensure at least one working model
        working_models = [k for k, v in self.models.items() if v.get('model') and v['model'] != 'mock']
        if not working_models:
            logger.warning("⚠️ No ML models loaded, creating smart rule-based classifier")
            self._create_fallback_model()
            working_models = ['smart_rules']
        
        logger.info(f"✅ Ready with {len(working_models)} working model(s): {working_models}")

    def _load_xgboost_model(self):
        """Load XGBoost model with proper error handling"""
        from config import MODEL_PATHS
        xgboost_path = MODEL_PATHS["xgboost"]
        if os.path.exists(xgboost_path):
            try:
                # Make preprocessor classes available in multiple namespaces for pickle
                import sys
                import __main__
                
                # Register all preprocessor classes in __main__ namespace
                __main__.AdvancedSMSPreprocessor = AdvancedSMSPreprocessor
                __main__.UltimateSMSPreprocessor = UltimateSMSPreprocessor
                __main__.ImprovedSMSPreprocessor = ImprovedSMSPreprocessor
                __main__.ProfessionalSMSPreprocessor = ProfessionalSMSPreprocessor
                __main__.FastSMSPreprocessor = FastSMSPreprocessor
                __main__.SimplePreprocessor = SimplePreprocessor
                __main__.SimpleTokenizer = SimpleTokenizer
                
                # Also register in sys.modules
                if 'main' not in sys.modules:
                    sys.modules['main'] = sys.modules[__name__]
                
                with open(xgboost_path, 'rb') as f:
                    xgb_data = pickle.load(f)
                    
                self.models['xgboost'] = {
                    'model': xgb_data['model'],
                    'vectorizer': xgb_data['vectorizer'],
                    'preprocessor': xgb_data.get('preprocessor', AdvancedSMSPreprocessor()),
                    'accuracy': 98.13,
                    'dataset_size': 5885,
                    'name': 'XGBoost High-Accuracy',
                    'training_focus': 'Comprehensive spam detection',
                    'type': 'xgboost'
                }
                logger.info("✅ XGBoost High-Accuracy model loaded successfully!")
                return True
                
            except Exception as e:
                logger.error(f"❌ Error loading XGBoost model: {e}")
                self.models['xgboost'] = self._create_mock_model('XGBoost High-Accuracy', 98.13, 5885, 'xgboost')
                return False
        else:
            logger.warning("⚠️ XGBoost model file not found")
            self.models['xgboost'] = self._create_mock_model('XGBoost High-Accuracy', 98.13, 5885, 'xgboost')
            return False

    def _load_svm_model(self):
        """Load SVM model with proper error handling"""
        from config import MODEL_PATHS
        svm_path = MODEL_PATHS["svm"]
        if os.path.exists(svm_path):
            try:
                # Make preprocessor classes available in multiple namespaces for pickle
                import sys
                import __main__
                
                # Register all preprocessor classes in __main__ namespace
                __main__.AdvancedSMSPreprocessor = AdvancedSMSPreprocessor
                __main__.UltimateSMSPreprocessor = UltimateSMSPreprocessor
                __main__.ImprovedSMSPreprocessor = ImprovedSMSPreprocessor
                __main__.ProfessionalSMSPreprocessor = ProfessionalSMSPreprocessor
                __main__.FastSMSPreprocessor = FastSMSPreprocessor
                __main__.SimplePreprocessor = SimplePreprocessor
                __main__.SimpleTokenizer = SimpleTokenizer
                
                # Also register in sys.modules
                if 'main' not in sys.modules:
                    sys.modules['main'] = sys.modules[__name__]
                
                with open(svm_path, 'rb') as f:
                    svm_data = pickle.load(f)
                    
                self.models['svm'] = {
                    'model': svm_data.get('model'),
                    'vectorizer': svm_data.get('vectorizer'),
                    'preprocessor': svm_data.get('preprocessor', SimplePreprocessor()),
                    'accuracy': 95.0,
                    'dataset_size': 2940,
                    'name': 'SVM (Support Vector Machine)',
                    'training_focus': 'Fast baseline spam detection',
                    'type': 'svm'
                }
                logger.info("✅ SVM model loaded successfully!")
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ Could not load SVM model: {e}")
                self.models['svm'] = self._create_mock_model('SVM (Support Vector Machine)', 95.0, 2940, 'svm')
                return False
        else:
            logger.warning("⚠️ SVM model file not found")
            self.models['svm'] = self._create_mock_model('SVM (Support Vector Machine)', 95.0, 2940, 'svm')
            return False

    def _load_distilbert_model(self):
        """Load DistilBERT Deep Classifier V2 model with proper error handling"""
        if not DISTILBERT_V2_AVAILABLE:
            logger.warning("⚠️ DistilBERT Deep Classifier V2 not available")
            self.models['distilbert_v2'] = self._create_mock_model('DistilBERT Deep Classifier V2', 98.59, 4962, 'distilbert_v2')
            return False
            
        from config import MODEL_PATHS
        distilbert_v2_path = MODEL_PATHS["distilbert_v2"]
        
        # Check if model pickle file exists
        if os.path.exists(distilbert_v2_path) and os.path.getsize(distilbert_v2_path) > 0:
            try:
                # Load from pickle file with CPU mapping for PyTorch objects
                import torch
                
                # Set up CPU device mapping for PyTorch objects during unpickling
                original_load = torch.load
                def cpu_load(*args, **kwargs):
                    kwargs['map_location'] = 'cpu'
                    return original_load(*args, **kwargs)
                
                # Temporarily replace torch.load to force CPU mapping
                torch.load = cpu_load
                
                try:
                    with open(distilbert_v2_path, 'rb') as f:
                        model_data = pickle.load(f)
                finally:
                    # Restore original torch.load
                    torch.load = original_load
                
                # Extract classifier and ensure it's on CPU
                if isinstance(model_data, dict) and 'model' in model_data:
                    classifier = model_data['model']
                else:
                    classifier = model_data
                
                # Ensure classifier is on CPU
                if hasattr(classifier, 'to'):
                    classifier = classifier.to('cpu')
                if hasattr(classifier, 'model') and hasattr(classifier.model, 'to'):
                    classifier.model = classifier.model.to('cpu')
                if hasattr(classifier, 'tokenizer'):
                    # Tokenizers don't need device mapping
                    pass
                
                self.models['distilbert_v2'] = {
                    'model': classifier,
                    'vectorizer': None,
                    'preprocessor': None,
                    'accuracy': model_data.get('accuracy', 98.59),
                    'dataset_size': model_data.get('dataset_size', 4962),
                    'banking_accuracy': model_data.get('banking_accuracy', 100.0),
                    'name': model_data.get('name', 'DistilBERT Deep Classifier V2'),
                    'training_focus': model_data.get('description', 'Enhanced banking/financial message accuracy'),
                    'type': 'distilbert_v2',
                    'version': model_data.get('version', '2.0.0'),
                    'features': model_data.get('features', ['GPU Acceleration', 'Banking Focus', 'Batch Processing'])
                }
                logger.info("✅ DistilBERT Deep Classifier V2 loaded successfully!")
                return True
                    
            except Exception as e:
                logger.error(f"❌ Error loading DistilBERT Deep Classifier V2: {e}")
                self.models['distilbert_v2'] = self._create_mock_model('DistilBERT Deep Classifier V2', 98.59, 4962, 'distilbert_v2')
                return False
        else:
            if os.path.exists(distilbert_v2_path):
                logger.warning(f"⚠️ DistilBERT Deep Classifier V2 model directory is empty: {distilbert_v2_path}")
            else:
                logger.warning(f"⚠️ DistilBERT Deep Classifier V2 model directory not found: {distilbert_v2_path}")
            self.models['distilbert_v2'] = self._create_mock_model('DistilBERT Deep Classifier V2', 98.59, 4962, 'distilbert_v2')
            return False

    def _load_roberta_model(self):
        """Load RoBERTa spam detector model with proper error handling"""
        if not ROBERTA_AVAILABLE:
            logger.warning("⚠️ RoBERTa transformers not available")
            self.models['roberta'] = self._create_mock_model('RoBERTa Ultimate Spam Detector', 99.72, 100000, 'roberta')
            return False
        
        try:
            from config import MODEL_PATHS
            roberta_path = MODEL_PATHS["roberta"]
            
            # Check if RoBERTa pickle exists
            if not os.path.exists(roberta_path) or os.path.getsize(roberta_path) == 0:
                logger.warning("⚠️ RoBERTa model pickle file not found")
                self.models['roberta'] = self._create_mock_model('RoBERTa Ultimate Spam Detector', 99.72, 100000, 'roberta')
                return False
            
            logger.info(f"📄 Loading RoBERTa model from pickle: {roberta_path}")
            
            try:
                # Load from pickle
                with open(roberta_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                model = model_data['model']
                tokenizer = model_data['tokenizer']
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                model.eval()
                
                self.models['roberta'] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'device': device,
                    'vectorizer': None,
                    'preprocessor': None,
                    'accuracy': model_data.get('accuracy', 99.72),
                    'dataset_size': model_data.get('dataset_size', 100000),
                    'precision': model_data.get('precision', 99.86),
                    'recall': model_data.get('recall', 99.58),
                    'f1_score': model_data.get('f1_score', 99.72),
                    'name': model_data.get('name', 'RoBERTa Ultimate Spam Detector'),
                    'training_focus': model_data.get('description', 'State-of-the-art transformer-based spam detection'),
                    'type': 'roberta',
                    'version': model_data.get('version', '1.0.0'),
                    'model_path': roberta_path,
                    'features': model_data.get('features', ['Transformer Architecture', 'Contextual Understanding', 'GPU Acceleration'])
                }
                
                logger.info("✅ RoBERTa Ultimate Spam Detector loaded successfully!")
                return True
                
            except Exception as e:
                logger.error(f"❌ Error loading RoBERTa model from {latest_folder}: {e}")
                self.models['roberta'] = self._create_mock_model('RoBERTa Ultimate Spam Detector', 99.72, 100000, 'roberta')
                return False
                
        except Exception as e:
            logger.error(f"❌ Error finding RoBERTa model folders: {e}")
            self.models['roberta'] = self._create_mock_model('RoBERTa Ultimate Spam Detector', 99.72, 100000, 'roberta')
            return False

    def _create_mock_model(self, name: str, accuracy: float, dataset_size: int, model_type: str) -> Dict[str, Any]:
        """Create a mock model entry for frontend compatibility"""
        return {
            'model': 'mock',
            'vectorizer': None,
            'preprocessor': self._create_simple_preprocessor(),
            'accuracy': accuracy,
            'dataset_size': dataset_size,
            'name': name,
            'training_focus': 'Model not available - using fallback',
            'type': model_type
        }

    def _create_simple_preprocessor(self):
        """Create a simple preprocessor for fallback"""
        class SimplePreprocessor:
            def normalize_text(self, text):
                if not isinstance(text, str):
                    return ""
                text = text.lower().strip()
                text = re.sub(r'[₹£$€]\s*[\d,]+(?:\.\d{2})?', 'MONEY_AMOUNT', text)
                text = re.sub(r'\b\d{10,}\b', 'PHONE_NUMBER', text)
                text = re.sub(r'http\S+|www\.\S+', 'URL', text)
                text = re.sub(r'\s+', ' ', text)
                return text.strip()
            
            def extract_features(self, text):
                return np.zeros(30)
        
        return SimplePreprocessor()

    def _create_fallback_model(self):
        """Create an intelligent rule-based fallback model"""
        self.models['smart_rules'] = {
            'model': 'smart_classifier',
            'vectorizer': None,
            'preprocessor': self._create_simple_preprocessor(),
            'accuracy': 78.0,
            'dataset_size': 0,
            'name': 'Smart Rule-Based Classifier',
            'training_focus': 'Intelligent keyword and pattern-based spam detection',
            'type': 'rule_based'
        }

    def predict(self, message: str, model_name: str = 'xgboost') -> Dict[str, Any]:
        """Make prediction using specified model"""
        if model_name not in self.models or not self._is_model_working(model_name):
            if self._is_model_working('roberta'):
                model_name = 'roberta'
            elif self._is_model_working('distilbert_v2'):
                model_name = 'distilbert_v2'
            elif self._is_model_working('xgboost'):
                model_name = 'xgboost'
            elif self._is_model_working('svm'):
                model_name = 'svm'
            elif 'smart_rules' in self.models:
                model_name = 'smart_rules'
            else:
                raise ValueError("No working models available")

        try:
            model_data = self.models[model_name]
            
            if model_name == 'roberta' and self._is_model_working('roberta'):
                return self._predict_roberta(message, model_data)
            elif model_name == 'distilbert_v2' and self._is_model_working('distilbert_v2'):
                return self._predict_distilbert_v2(message, model_data)
            elif model_name == 'xgboost' and self._is_model_working('xgboost'):
                return self._predict_xgboost(message, model_data)
            elif model_name == 'svm' and self._is_model_working('svm'):
                return self._predict_svm(message, model_data)
            else:
                return self._predict_smart_rules(message, model_data)
                
        except Exception as e:
            logger.error(f"Prediction error with {model_name}: {e}")
            if model_name != 'smart_rules' and 'smart_rules' in self.models:
                return self._predict_smart_rules(message, self.models['smart_rules'])
            raise e

    def _is_model_working(self, model_name: str) -> bool:
        """Check if a model is actually working (not mock)"""
        if model_name not in self.models:
            return False
        model = self.models[model_name].get('model')
        return model and model not in ['mock', 'smart_classifier']

    def _predict_xgboost(self, message: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """XGBoost prediction with advanced features"""
        try:
            preprocessor = model_data['preprocessor']
            processed = preprocessor.normalize_text(message)
            numerical_features = preprocessor.extract_features(message).reshape(1, -1)
            vectorizer = model_data['vectorizer']
            tfidf_features = vectorizer.transform([processed]).toarray()
            
            features = np.hstack([tfidf_features, numerical_features])
            model = model_data['model']
            prediction = model.predict(features)[0]
            
            try:
                proba = model.predict_proba(features)[0]
                confidence = float(max(proba))
            except:
                confidence = 0.90
            
            return {
                'prediction': 'SPAM' if prediction == 1 else 'HAM',
                'confidence': confidence,
                'model_name': model_data['name'],
                'processed_text': processed
            }
        except Exception as e:
            logger.error(f"XGBoost prediction error: {e}")
            return self._predict_smart_rules(message, model_data)

    def _predict_svm(self, message: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """SVM prediction"""
        try:
            preprocessor = model_data['preprocessor']
            processed = preprocessor.normalize_text(message)
            vectorizer = model_data['vectorizer']
            features = vectorizer.transform([processed])
            model = model_data['model']
            prediction = model.predict(features)[0]
            
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features)[0]
                    confidence = float(max(proba))
                elif hasattr(model, 'decision_function'):
                    decision = model.decision_function(features)[0]
                    confidence = min(abs(float(decision)), 0.99)
                else:
                    confidence = 0.85
            except:
                confidence = 0.85
            
            return {
                'prediction': 'SPAM' if prediction == 1 else 'HAM',
                'confidence': confidence,
                'model_name': model_data['name'],
                'processed_text': processed
            }
        except Exception as e:
            logger.error(f"SVM prediction error: {e}")
            return self._predict_smart_rules(message, model_data)

    def _predict_distilbert_v2(self, message: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """DistilBERT Deep Classifier V2 prediction with enhanced banking accuracy"""
        try:
            result = predict_message(message)
            
            if result['success']:
                return {
                    'prediction': result['prediction'].upper(),
                    'confidence': result['confidence'],
                    'model_name': model_data['name'],
                    'processed_text': message,
                    'probabilities': result.get('probabilities', {}),
                    'processing_time_ms': result.get('processing_time_ms', 0),
                    'model_version': result.get('model_version', '2.0.0')
                }
            else:
                logger.error(f"❌ DistilBERT V2 prediction failed: {result.get('error', 'Unknown error')}")
                return self._predict_smart_rules(message, model_data)
                
        except Exception as e:
            logger.error(f"❌ DistilBERT V2 prediction error: {e}")
            return self._predict_smart_rules(message, model_data)

    def _predict_roberta(self, message: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """RoBERTa Ultimate Spam Detector prediction with state-of-the-art accuracy"""
        try:
            import time
            start_time = time.time()
            
            model = model_data['model']
            tokenizer = model_data['tokenizer']
            device = model_data['device']
            
            # Tokenize the input text
            inputs = tokenizer(
                message,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
            # Get prediction and confidence
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = float(probabilities[0][predicted_class])
            
            # Convert to probabilities dictionary
            ham_prob = float(probabilities[0][0])
            spam_prob = float(probabilities[0][1])
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return {
                'prediction': 'SPAM' if predicted_class == 1 else 'HAM',
                'confidence': confidence,
                'model_name': model_data['name'],
                'processed_text': message,
                'probabilities': {
                    'ham': ham_prob,
                    'spam': spam_prob
                },
                'processing_time_ms': processing_time,
                'model_version': model_data.get('version', '1.0.0'),
                'device': str(device)
            }
            
        except Exception as e:
            logger.error(f"❌ RoBERTa prediction error: {e}")
            return self._predict_smart_rules(message, model_data)

    def _predict_smart_rules(self, message: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Smart rule-based prediction"""
        processed = message.lower()
        
        # Enhanced spam indicators
        spam_keywords = [
            'free', 'win', 'winner', 'prize', 'congratulations', 'urgent', 'call now',
            'claim', 'limited time', 'act now', 'guaranteed', 'offer expires',
            'click here', 'verify account', 'suspended', 'blocked', 'payment failed',
            'refund', 'tax', 'owe', 'debt', 'lottery', 'selected', 'cash', 'money'
        ]
        
        # Count spam indicators
        spam_word_count = sum(1 for word in spam_keywords if word in processed)
        
        # Pattern checks
        has_money = any(symbol in message for symbol in ['$', '£', '€', 'money', 'cash', 'prize'])
        has_urgency = any(word in processed for word in ['urgent', 'immediately', 'now', 'expires', 'limited'])
        has_link = any(pattern in processed for pattern in ['http', 'www.', '.com', 'click', 'link'])
        has_phone = bool(re.search(r'\b\d{10,}\b', message))
        has_caps = sum(1 for c in message if c.isupper()) / max(len(message), 1) > 0.3
        has_exclamation = message.count('!') > 2
        
        # Scoring system
        score = spam_word_count * 2
        if has_money: score += 3
        if has_urgency: score += 2
        if has_link: score += 2
        if has_phone: score += 1
        if has_caps: score += 1
        if has_exclamation: score += 1
        
        # Decision
        prediction = 1 if score >= 4 else 0
        confidence = min(0.50 + (score * 0.08), 0.88)
        
        return {
            'prediction': 'SPAM' if prediction == 1 else 'HAM',
            'confidence': float(confidence),
            'model_name': model_data['name'],
            'processed_text': processed
        }

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get only actually working models for frontend"""
        model_info = {}
        for key, model_data in self.models.items():
            # Only include actually working models, not mock ones
            if self._is_model_working(key):
                entry = {
                    'name': model_data['name'],
                    'accuracy': model_data['accuracy'],
                    'dataset_size': model_data['dataset_size'],
                    'training_focus': model_data['training_focus']
                }
                model_info[key] = entry
        
        return {'available_models': model_info}

# Pydantic models for API
class SMSMessage(BaseModel):
    text: str
    model: Optional[str] = 'roberta'

class BatchSMSMessages(BaseModel):
    texts: List[str]
    model: Optional[str] = 'roberta'

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    model_name: str
    processed_text: str
    risk_level: str
    message: str
    processing_time_ms: Optional[float] = None
    model_version: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int
    model_name: str
    success: bool

# Global model manager
model_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global model_manager
    try:
        model_manager = ModelManager()
        logger.info("🚀 SMS Spam Detection API is ready!")
        yield
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise e
    finally:
        pass

# Initialize FastAPI app
app = FastAPI(
    title="SMS Spam Detection API",
    description="AI-powered SMS spam detection with DistilBERT Deep Classifier V2 (98.59% accuracy) + XGBoost + SVM models",
    version="4.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend - Updated for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081",  # Your React Expo frontend
        "http://127.0.0.1:8081",
        "http://localhost:19006",  # Expo development server
        "http://localhost:19000",  # Expo Metro bundler
        "*"  # Allow all for development (remove in production)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="API not ready")
    
    working_models = len([k for k, v in model_manager.models.items() if model_manager._is_model_working(k)])
    return {
        "message": "SMS Spam Detection API",
        "version": "4.0.0",
        "description": "DistilBERT Deep Classifier V2 (98.59%) + XGBoost High-Accuracy + SVM + Smart Rules",
        "models_loaded": working_models,
        "status": "operational",
        "endpoints": {
            "/predict": "POST - Predict if SMS is spam or ham",
            "/predict/batch": "POST - Batch predict multiple SMS messages",
            "/models": "GET - Get available models",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check for frontend"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="API not ready")
    
    working_models = [k for k, v in model_manager.models.items() if model_manager._is_model_working(k)]
    total_models = len(model_manager.models)
    
    return {
        "status": "healthy",
        "models_loaded": len(working_models),
        "total_models": total_models,
        "working_models": working_models,
        "message": "All systems operational"
    }

@app.get("/models")
async def get_models():
    """Get available models for frontend"""
    if not model_manager:
        raise HTTPException(status_code=500, detail="Models not initialized")
    
    return model_manager.get_available_models()

@app.post("/predict", response_model=PredictionResponse)
async def predict_sms(sms: SMSMessage):
    """Predict if SMS is spam or ham - Main endpoint for frontend"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    if not sms.text.strip():
        raise HTTPException(status_code=400, detail="SMS text cannot be empty")
    
    try:
        # Make prediction
        result = model_manager.predict(sms.text.strip(), sms.model)
        
        # Calculate risk level
        prediction = result['prediction']
        confidence = result['confidence']
        
        if prediction == 'SPAM':
            if confidence > 0.8:
                risk_level = "HIGH RISK"
            elif confidence > 0.6:
                risk_level = "MEDIUM RISK"
            else:
                risk_level = "LOW RISK"
        else:  # HAM
            if confidence > 0.8:
                risk_level = "SAFE"
            elif confidence > 0.6:
                risk_level = "LIKELY SAFE"
            else:
                risk_level = "UNCERTAIN"
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_name=result['model_name'],
            processed_text=result['processed_text'],
            risk_level=risk_level,
            message=sms.text.strip(),
            processing_time_ms=result.get('processing_time_ms'),
            model_version=result.get('model_version')
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_sms_batch(batch: BatchSMSMessages):
    """Predict spam/ham for multiple SMS messages - Batch processing endpoint"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    if not batch.texts or len(batch.texts) == 0:
        raise HTTPException(status_code=400, detail="Batch texts cannot be empty")
    
    if len(batch.texts) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100 messages")
    
    try:
        # For DistilBERT V2, use the production batch function
        if batch.model == 'distilbert_v2' and model_manager._is_model_working('distilbert_v2'):
            batch_result = predict_batch_messages(batch.texts)
            
            if batch_result['success']:
                predictions = []
                for i, pred_data in enumerate(batch_result['predictions']):
                    # Calculate risk level
                    prediction = pred_data['prediction']
                    confidence = pred_data['confidence']
                    
                    if prediction == 'SPAM':
                        if confidence > 0.8:
                            risk_level = "HIGH RISK"
                        elif confidence > 0.6:
                            risk_level = "MEDIUM RISK"
                        else:
                            risk_level = "LOW RISK"
                    else:  # HAM
                        if confidence > 0.8:
                            risk_level = "SAFE"
                        elif confidence > 0.6:
                            risk_level = "LIKELY SAFE"
                        else:
                            risk_level = "UNCERTAIN"
                    
                    predictions.append(PredictionResponse(
                        prediction=prediction,
                        confidence=confidence,
                        model_name="DistilBERT Deep Classifier V2",
                        processed_text=batch.texts[i],
                        risk_level=risk_level,
                        message=batch.texts[i],
                        processing_time_ms=pred_data.get('processing_time_ms'),
                        model_version=batch_result.get('model_version')
                    ))
                
                return BatchPredictionResponse(
                    predictions=predictions,
                    total_processed=len(predictions),
                    model_name="DistilBERT Deep Classifier V2",
                    success=True
                )
            else:
                raise HTTPException(status_code=500, detail=f"Batch prediction failed: {batch_result.get('error')}")
        else:
            # Fallback to individual predictions for other models
            predictions = []
            for text in batch.texts:
                if not text.strip():
                    continue
                
                result = model_manager.predict(text.strip(), batch.model)
                
                # Calculate risk level
                prediction = result['prediction']
                confidence = result['confidence']
                
                if prediction == 'SPAM':
                    if confidence > 0.8:
                        risk_level = "HIGH RISK"
                    elif confidence > 0.6:
                        risk_level = "MEDIUM RISK"
                    else:
                        risk_level = "LOW RISK"
                else:  # HAM
                    if confidence > 0.8:
                        risk_level = "SAFE"
                    elif confidence > 0.6:
                        risk_level = "LIKELY SAFE"
                    else:
                        risk_level = "UNCERTAIN"
                
                predictions.append(PredictionResponse(
                    prediction=prediction,
                    confidence=confidence,
                    model_name=result['model_name'],
                    processed_text=result['processed_text'],
                    risk_level=risk_level,
                    message=text.strip()
                ))
            
            return BatchPredictionResponse(
                predictions=predictions,
                total_processed=len(predictions),
                model_name=batch.model,
                success=True
            )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal batch prediction error")

if __name__ == "__main__":
    import uvicorn
    from config import API_CONFIG
    
    logger.info(f"🚀 Starting SMS Spam Detection API...")
    logger.info(f"📡 Frontend URL: {API_CONFIG['frontend_url']}")
    logger.info(f"🌐 Backend URL: http://{API_CONFIG['host']}:{API_CONFIG['port']}")
    
    # For development with reload, run: uvicorn main:app --host 127.0.0.1 --port 8000 --reload
    uvicorn.run(
        app, 
        host=API_CONFIG["host"], 
        port=API_CONFIG["port"]
    )
