# test_backtesting.py
from utils.stock_manager import StockManager
from utils.backtester import Backtester
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

def clean_features(data, features):
    """Clean and prepare feature data"""
    data = data.fillna(method='ffill')
    data = data.fillna(method='bfill')
    
    
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        for feature in missing_features:
            data[feature] = 0  
    
    return data

def test_backtesting():
    stock_manager = StockManager()
    backtester = Backtester(train_size=0.8)
    
    try:
        
        symbol = 'NVDA'
        data, sentiment = stock_manager.prepare_prediction_data(symbol)
        
        data = stock_manager.add_technical_indicators(data)
        
        if isinstance(sentiment, (int, float)):
            data['Sentiment'] = sentiment
        else:
            print("Warning: Invalid sentiment value, using neutral sentiment")
            data['Sentiment'] = 0
            
        features = [
            'RSI', 'MACD', 'SMA_20', '%K', '%D', 'ROC', 'MFI',
            'BB_upper', 'BB_lower', 'ATR', 'OBV', 'Sentiment'
        ]
        
        data = clean_features(data, features)
        

        print("\nData Quality Check:")
        print(f"Total rows: {len(data)}")
        print(f"NaN values:\n{data[features].isna().sum()}")
        print("\nFeature Statistics:")
        print(data[features].describe())
        
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )

        results = backtester.run_backtest(model, data, features, 'Close')
        
        # Print results
        print("\nBacktesting Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
            
    except Exception as e:
        print(f"Error during backtesting: {str(e)}")
        raise

if __name__ == "__main__":
    test_backtesting()