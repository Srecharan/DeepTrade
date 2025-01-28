from utils.prediction_system import PredictionSystem
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List
import os
from utils.reddit_sentiment import EnhancedRedditAnalyzer
from utils.sec_data_collector import SECDataCollector

def plot_predictions(symbol, predictions, data):
    plt.figure(figsize=(12, 6))
    
    # Plot historical prices
    plt.plot(data.index[-30:], data['Close'][-30:], 
             label='Historical Price', color='blue')
    
    # Plot prediction point
    last_date = data.index[-1]
    current_price = predictions['risk_metrics']['current_price']['price'] if isinstance(predictions['risk_metrics']['current_price'], dict) else predictions['risk_metrics']['current_price']
    
    plt.scatter(last_date, current_price, 
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

def print_summary(symbol, current_price, predictions):
    print(f"\nPrediction Summary for {symbol}")
    print("-" * 40)
    print(f"Current Price: ${current_price:.2f}")
    print(f"Predicted Price: ${predictions['risk_metrics']['prediction_price']:.2f}")
    print(f"Expected Return: {predictions['risk_metrics']['prediction_return']:.2f}%")
    print(f"Confidence Score: {predictions['risk_metrics']['confidence_score']:.2%}")
    print(f"Sentiment Score: {predictions['risk_metrics']['sentiment_score']:.2f}")
    print("-" * 40)

def test_predictions():
    prediction_system = PredictionSystem()
    symbols = ['NVDA', 'AAPL', 'MSFT', 'GME', 'AMD', 'JNJ', 'META', 'GOOGL', 'AMZN']
    
    print("\nStarting Prediction Test")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
    print("=" * 50)
    
    for symbol in symbols:
        try:
            # Get historical data first
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            data = prediction_system.stock_manager.fetch_stock_data(
                symbol,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            # Make predictions
            predictions = prediction_system.predict(symbol)
            
            if predictions and not data.empty:
                # Get current price from predictions
                current_price = (predictions['risk_metrics']['current_price']['price'] 
                               if isinstance(predictions['risk_metrics']['current_price'], dict) 
                               else predictions['risk_metrics']['current_price'])
                
                # Create visualization
                plot_predictions(symbol, predictions, data)
                
                # Print prediction summary
                print_summary(symbol, current_price, predictions)
            else:
                print(f"No valid predictions or data for {symbol}")
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            import traceback
            print(traceback.format_exc())

if __name__ == "__main__":
    test_predictions()