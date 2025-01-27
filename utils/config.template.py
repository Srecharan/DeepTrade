# utils/config.template.py
import os
from datetime import datetime

# API Keys
NEWS_API_KEY = "YOUR_NEWS_API_KEY"
ALPHA_VANTAGE_KEY = "YOUR_ALPHA_VANTAGE_KEY"
FINNHUB_KEY = "YOUR_FINNHUB_KEY"
POLYGON_KEY = "YOUR_POLYGON_KEY"
TWELVEDATA_KEY = "YOUR_TWELVEDATA_KEY"
MARKETSTACK_API = "YOUR_MARKETSTACK_API"

# Tradier Configuration
TRADIER_CONFIG = {
    'production': {
        'token': 'YOUR_PRODUCTION_TOKEN',
        'endpoint': 'https://api.tradier.com/v1'
    },
    'sandbox': {
        'token': 'YOUR_SANDBOX_TOKEN',
        'endpoint': 'https://sandbox.tradier.com/v1'
    }
}

# Use sandbox for development
USE_TRADIER_SANDBOX = True

# Reddit Configuration
REDDIT_CLIENT_ID = "YOUR_REDDIT_CLIENT_ID"
REDDIT_CLIENT_SECRET = "YOUR_REDDIT_CLIENT_SECRET"
REDDIT_USER_AGENT = "YOUR_REDDIT_USER_AGENT"

# API Rate Limits (Keep these as they are)
API_LIMITS = {
    'tradier': {
        'daily': 10000,
        'minute': 60
    },
    'twelvedata': {
        'daily': 800,
        'minute': 8
    },
    'polygon': {
        'daily': 5,
        'minute': 5
    },
    'alphavantage': {
        'daily': 500,
        'minute': 5
    },
    'finnhub': {
        'daily': 60,
        'minute': 1
    }
}

# API Endpoints (Keep these as they are)
API_ENDPOINTS = {
    'tradier': TRADIER_CONFIG['sandbox' if USE_TRADIER_SANDBOX else 'production']['endpoint'],
    'twelvedata': 'https://api.twelvedata.com/price',
    'polygon': 'https://api.polygon.io/v2/aggs/ticker/{symbol}/prev',
    'alphavantage': 'https://www.alphavantage.co/query',
    'finnhub': 'https://finnhub.io/api/v1/quote'
}