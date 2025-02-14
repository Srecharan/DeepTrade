# test_sentiment_integration.py
from utils.stock_manager import StockManager
import pandas as pd
import numpy as np

def test_sentiment_integration():
    stock_manager = StockManager()
    symbol = 'NVDA'
    
    try:
        data, sentiment = stock_manager.prepare_prediction_data(symbol)
        print(f"\nCurrent sentiment score: {sentiment}")
        sentiment_cols = [col for col in data.columns if 'sentiment' in col.lower()]
        print("\nSentiment columns:", sentiment_cols)
        
        print("\nSample of sentiment features (first 5 rows):")
        print(data[sentiment_cols].head())
 
        print("\nSentiment features statistics:")
        print(data[sentiment_cols].describe())

        print("\nNaN count in sentiment features:")
        print(data[sentiment_cols].isna().sum())
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    test_sentiment_integration()