from utils.stock_manager import StockManager
from utils.sentiment_manager import SentimentDataManager
from utils.feature_importance import FeatureImportanceAnalyzer
import matplotlib.pyplot as plt
import numpy as np

def test_enhanced_features():
    # Initialize managers
    stock_manager = StockManager()
    sentiment_manager = SentimentDataManager()
    feature_analyzer = FeatureImportanceAnalyzer()
    
    # Test with NVDA
    symbols = ['TSLA', 'GME', 'NVDA', 'AMD', 'AAPL'] 
    #symbols = ['GME', 'AMD', 'MSFT'] 
    #symbols = ['JNJ', 'PG', 'KO']  # Johnson & Johnson, Procter & Gamble, Coca-Cola
    #symbols = ['PLTR', 'SNAP', 'RBLX']  # Palantir, Snap, Roblox

    for symbol in symbols:
            print(f"\n=== Analyzing {symbol} ===")
            try:
                # Get data and sentiment
                data, sentiment = stock_manager.prepare_prediction_data(symbol)
                
                # Remove any remaining NaN values
                data = data.fillna(method='ffill').fillna(method='bfill')
                
                # Prepare scaled features with sentiment
                X_scaled = stock_manager.prepare_scaled_features(data)
                
                # Analyze feature importance
                y = data['Close'].values
                min_len = min(len(X_scaled), len(y))
                X_scaled = X_scaled[:min_len]
                y = y[:min_len]
                
                importance = feature_analyzer.analyze_importance(X_scaled, y)
                print(f"\nFeature Importance for {symbol}:")
                print(importance)
                
                # Calculate total sentiment importance
                sentiment_cols = [col for col in importance['Feature'] if 'sentiment' in col.lower()]
                total_sentiment_impact = importance[importance['Feature'].isin(sentiment_cols)]['Importance'].sum()
                print(f"\nTotal sentiment impact for {symbol}: {total_sentiment_impact*100:.2f}%")
                
            except Exception as e:
                print(f"Error analyzing {symbol}: {str(e)}")

if __name__ == "__main__":
    test_enhanced_features()