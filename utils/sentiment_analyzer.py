# utils/sentiment_analyzer.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import List, Optional
import os
import requests
from datetime import datetime, timedelta

class FinancialSentimentAnalyzer:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        try:
            if self.verbose:
                print("Loading FinBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert", cache_dir="./models/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", cache_dir="./models/finbert")
            self.model.eval()
            self.model_loaded = True
            if self.verbose:
                print("FinBERT model loaded successfully")
        except Exception as e:
            print(f"Error loading FinBERT model: {e}")
            self.model_loaded = False

    def analyze_text(self, text: str) -> float:
        """Analyze a single piece of text and return sentiment score"""
        if not self.model_loaded:
            if self.verbose:
                print("Model not loaded, returning neutral sentiment")
            return 0.0
            
        if not text or not isinstance(text, str):
            return 0.0
            
        try:
            text = text.strip()
            if len(text) < 5:
                return 0.0
                
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                scores = probabilities.numpy()[0]
                
                sentiment_score = scores[2] - scores[0]
                if self.verbose:
                    print(f"Text sentiment: {sentiment_score:.2f} (neg={scores[0]:.2f}, neu={scores[1]:.2f}, pos={scores[2]:.2f})")
                return sentiment_score
                
        except Exception as e:
            if self.verbose:
                print(f"Error analyzing text: {e}")
            return 0.0

    def analyze_multiple(self, texts: List[str]) -> float:
        """Analyze multiple texts and return average sentiment"""
        if not texts:
            return 0.0
        
        try:
            valid_texts = [text for text in texts if isinstance(text, str) and text.strip()]
            if not valid_texts:
                return 0.0
                
            if self.verbose:
                print(f"Analyzing {len(valid_texts)} texts...")
            scores = []
            for text in valid_texts:
                score = self.analyze_text(text)
                if score != 0.0:
                    scores.append(score)
                    
            if not scores:
                return 0.0
                
            avg_score = np.mean(scores)
            if self.verbose:
                print(f"Average sentiment score: {avg_score:.2f}")
            return avg_score
            
        except Exception as e:
            if self.verbose:
                print(f"Error in analyze_multiple: {e}")
            return 0.0

def get_financial_news(symbol: str, days: int = 7, api_key: Optional[str] = None) -> List[str]:
    """Fetch financial news for a given stock symbol"""
    if not api_key:
        print("No API key provided")
        return []

    url = 'https://newsapi.org/v2/everything'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    params = {
        'q': f"{symbol} stock",
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'language': 'en',
        'sortBy': 'relevancy',
        'apiKey': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            headlines = []
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                content = f"{title}. {description}" if description else title
                if content:
                    headlines.append(content)
            
            print(f"Successfully fetched {len(headlines)} news articles")
            return headlines
        else:
            print(f"Error fetching news: {response.status_code}")
            print(f"Response: {response.text}")
            return []
    except Exception as e:
        print(f"Exception while fetching news: {e}")
        return []