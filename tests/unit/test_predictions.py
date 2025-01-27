from utils.prediction_system import PredictionSystem
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List
import os
from utils.reddit_sentiment import EnhancedRedditAnalyzer
from utils.sec_data_collector import SECDataCollector

def plot_predictions(symbol, predictions, history_data):
    plt.figure(figsize=(12, 6))
    
    # Plot historical prices
    plt.plot(history_data.index[-30:], history_data['Close'][-30:], 
             label='Historical Price', color='blue')
    
    # Plot prediction point
    last_date = history_data.index[-1]
    plt.scatter(last_date, predictions['risk_metrics']['current_price'], 
               color='blue', marker='o')
    
    # Plot prediction
    plt.scatter(last_date, predictions['risk_metrics']['prediction_price'], 
               color='red', marker='*', s=150, label='Prediction')
    
    # Plot confidence interval
    plt.fill_between([last_date], 
                    [predictions['confidence_interval'][0]], 
                    [predictions['confidence_interval'][1]], 
                    color='red', alpha=0.2, label='Confidence Interval')
    
    plt.title(f'{symbol} Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('visualization', exist_ok=True)
    plt.savefig(f'visualization/{symbol}_prediction.png')
    plt.close()

def test_predictions():
    prediction_system = PredictionSystem()
    symbols = ['NVDA', 'AAPL', 'MSFT']  # or ['GME', 'AMD', 'JNJ']
    
    print("\nStarting Prediction Test")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
    print("=" * 50)
    
    for symbol in symbols:
        try:
            # Get real-time price first
            current_price = prediction_system.stock_manager.get_real_time_price(symbol)
            
            # Get sentiment data
            reddit_analyzer = EnhancedRedditAnalyzer(...)
            sec_collector = SECDataCollector()
            
            # Make predictions based on real-time price
            predictions = prediction_system.predict(symbol, current_price)
            
            # Create visualization
            plot_predictions(symbol, predictions, data)
            
            # Print single summary
            print_summary(symbol, current_price, predictions, ...)
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")

if __name__ == "__main__":
    test_predictions()