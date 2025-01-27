from utils.stock_manager import StockManager
from utils.sentiment_manager import SentimentDataManager
import pandas as pd

def test_sentiment_integration():
    stock_manager = StockManager()
    sentiment_manager = SentimentDataManager()
    symbol = 'NVDA'
    
    print("Step 1: Getting current sentiment...")
    sentiment = stock_manager.analyze_sentiment(symbol)
    print(f"Current sentiment: {sentiment:.2f}")
    
    print("\nStep 2: Getting historical sentiment...")
    hist_sentiment = sentiment_manager.get_historical_sentiment(symbol)
    print("Historical sentiment records:", len(hist_sentiment) if not hist_sentiment.empty else 0)
    
    print("\nStep 3: Getting stock data with sentiment features...")
    data, current_sentiment = stock_manager.prepare_prediction_data(symbol)
    
    print("\nStep 4: Checking sentiment features...")
    sentiment_columns = [col for col in data.columns if 'sentiment' in col.lower()]
    print("Sentiment-related columns:", sentiment_columns)
    
    if sentiment_columns:
        print("\nSentiment features summary:")
        print(data[sentiment_columns].describe())

if __name__ == "__main__":
    test_sentiment_integration()