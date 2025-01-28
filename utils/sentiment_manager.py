import pandas as pd
import os
from datetime import datetime

class SentimentDataManager:
    def __init__(self):
        self.sentiment_db_path = 'data/sentiment/'
        os.makedirs(self.sentiment_db_path, exist_ok=True)
        
    def store_daily_sentiment(self, symbol: str, sentiment_score: float, headlines: list):
        """Store daily sentiment data for a symbol"""
        today = datetime.now().strftime('%Y-%m-%d')
        data = {
            'date': today,
            'sentiment_score': sentiment_score,
            'headlines_count': len(headlines),
            'headlines': '|'.join(headlines[:5])  # Store top 5 headlines
        }
        
        file_path = f"{self.sentiment_db_path}/{symbol}_sentiment.csv"
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Remove any existing entry for today to avoid duplicates
            df = df[df['date'] != today]
            # Append new data
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        else:
            df = pd.DataFrame([data])
            
        df['date'] = pd.to_datetime(df['date'])
        df = df.drop_duplicates(subset=['date'], keep='last')
        df.to_csv(file_path, index=False)
    
    def get_historical_sentiment(self, symbol: str) -> pd.DataFrame:
        """Get historical sentiment data for a symbol"""
        file_path = f"{self.sentiment_db_path}/{symbol}_sentiment.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            return df.drop_duplicates(subset=['date'], keep='last').set_index('date')
        return pd.DataFrame()