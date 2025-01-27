# test_backtesting.py
from utils.stock_manager import StockManager
from utils.backtester import Backtester
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

def clean_features(data, features):
    """Clean and prepare feature data"""
    # Forward fill NaN values first
    data = data.fillna(method='ffill')
    # Back fill any remaining NaN values
    data = data.fillna(method='bfill')
    
    # Replace infinite values with NaN and then fill them
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Ensure all features exist
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        for feature in missing_features:
            data[feature] = 0  # Default value for missing features
    
    return data

def test_backtesting():
    # Initialize
    stock_manager = StockManager()
    backtester = Backtester(train_size=0.8)
    
    try:
        # Get data for NVDA
        symbol = 'NVDA'
        data, sentiment = stock_manager.prepare_prediction_data(symbol)
        
        # Add technical indicators
        data = stock_manager.add_technical_indicators(data)
        
        # Handle sentiment data
        if isinstance(sentiment, (int, float)):
            data['Sentiment'] = sentiment
        else:
            print("Warning: Invalid sentiment value, using neutral sentiment")
            data['Sentiment'] = 0
            
        # Define features
        features = [
            'RSI', 'MACD', 'SMA_20', '%K', '%D', 'ROC', 'MFI',
            'BB_upper', 'BB_lower', 'ATR', 'OBV', 'Sentiment'
        ]
        
        # Clean and prepare features
        data = clean_features(data, features)
        
        # Verify data quality
        print("\nData Quality Check:")
        print(f"Total rows: {len(data)}")
        print(f"NaN values:\n{data[features].isna().sum()}")
        print("\nFeature Statistics:")
        print(data[features].describe())
        
        # Create and run backtest with error handling
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
        # Run backtest
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