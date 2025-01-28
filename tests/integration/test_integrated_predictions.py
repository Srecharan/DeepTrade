# test_integrated_predictions.py

from utils.prediction_system import PredictionSystem
from utils.stock_manager import StockManager
from utils.model_trainer import ModelTrainer
from utils.reddit_sentiment import EnhancedRedditAnalyzer
from utils.sec_data_collector import SECDataCollector
import pandas as pd
import numpy as np
from pytz import timezone
from typing import Dict, List
import os
from datetime import datetime, timedelta
import numpy as np
from utils.config import (
    ALPHA_VANTAGE_KEY,
    FINNHUB_KEY,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT
)

def print_price_summary(symbol: str, price_info: Dict):
    """ Price summary with verified market prices and None handling"""
    try:
        if price_info is None:
            print(f"\n{symbol} Price Details: No price data available")
            return
            
        et_tz = timezone('US/Eastern')
        current_time = datetime.now(et_tz)
        
        print(f"\n{symbol} Price Details ({current_time.strftime('%H:%M:%S ET')}):")
        
        price = price_info.get('price')
        if price is None:
            print("No current price available")
            return
            
        market_hours = price_info.get('market_hours', False)
        timestamp = price_info.get('timestamp', current_time)
        source = price_info.get('source', 'unknown')
        market_close = price_info.get('market_close')
        
        if timestamp.tzinfo is None:
            timestamp = et_tz.localize(timestamp)
        
        print(f"Current Price: ${price:.2f}")
        print(f"Timestamp: {timestamp.strftime('%H:%M:%S ET')}")
        print(f"Source: {source}")
        print(f"Market Hours: {'Yes' if market_hours else 'No'}")
        
        if market_close is not None:
            market_close_time = price_info.get('market_close_time', current_time)
            if market_close_time.tzinfo is None:
                market_close_time = et_tz.localize(market_close_time)
            print(f"Market Close: ${market_close:.2f} ({market_close_time.strftime('%H:%M:%S ET')})")
        
        time_diff = (current_time - timestamp).total_seconds()
        if time_diff > 5:
            print(f"WARNING: Price delay of {time_diff:.1f} seconds")
            
    except Exception as e:
        print(f"Error printing price summary: {str(e)}")
        
def print_timeframe_predictions(symbol: str, predictions: Dict):
    """Print predictions for each timeframe"""
    current_price = predictions['current_price']
    current_time = current_price['timestamp']
    target_time = predictions['target_timestamp']
    
    print(f"\nTimeframe: {predictions['timeframe']}")
    print(f"Current Price: ${current_price['price']:.2f}")
    print(f"Current Time: {current_time.strftime('%H:%M:%S ET')}")
    print(f"\nModel Predictions (Target: {target_time.strftime('%H:%M:%S ET')}):")
    print(f"  - LSTM: ${predictions['predictions']['lstm']:.2f}")
    print(f"  - XGBoost: ${predictions['predictions']['xgboost']:.2f}")
    print(f"  - Ensemble: ${predictions['predictions']['ensemble']:.2f}")
    print(f"\nRisk Metrics:")
    print(f"  - Expected Return: {predictions['metrics']['expected_return']:.2f}%")
    print(f"  - Confidence: {predictions['metrics']['confidence']:.2%}")
    print(f"  - Volatility: {predictions['metrics']['volatility']:.2%}")
    print(f"  - Confidence Interval: "
          f"${predictions['confidence_interval'][0]:.2f} to "
          f"${predictions['confidence_interval'][1]:.2f}")

def test_integrated_predictions():
    """Test predictions with all data sources"""
    prediction_system = PredictionSystem()
    #symbols = ['NVDA']
    symbols = ['NVDA', 'AAPL', 'MSFT', 'GME', 'AMD', 'JNJ', 'META', 'GOOGL', 'AMZN' ]
    timeframes = ['5min', '15min', '30min', '1h']
    results = {}

    print("\nStarting Integrated Prediction Test")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
    print("=" * 50)
    
    for symbol in symbols:
        try:
            print(f"\n=== Processing {symbol} ===")
            
            # Get real-time price with enhanced logging
            current_price = prediction_system.stock_manager.get_real_time_price(symbol)
            if current_price:
                print_price_summary(symbol, current_price)
            
            # Get sentiment data ONCE per symbol
            try:
    
                base_predictions = prediction_system.predict(symbol)
                sentiment_data = {
                    'sentiment_score': base_predictions['risk_metrics']['sentiment_score'],
                    'reddit_data': {'total_sentiment': base_predictions['risk_metrics']['sentiment_score']},
                    'sec_data': {'sentiment_score': base_predictions['risk_metrics']['sentiment_score']}
                }
            except Exception as e:
                print(f"Error getting sentiment data: {str(e)}")
                sentiment_data = None
            
           
            timeframe_predictions = {}
            for timeframe in timeframes:
                try:
                    predictions = prediction_system.predict_timeframe(
                        symbol, 
                        timeframe,
                        sentiment_data=sentiment_data  # Pass the pre-calculated sentiment
                    )
                    timeframe_predictions[timeframe] = predictions
                    print_timeframe_predictions(symbol, predictions)
                except Exception as e:
                    print(f"Error predicting {timeframe} for {symbol}: {str(e)}")
            
         
            results[symbol] = timeframe_predictions
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    return results

def print_summary_report(results: Dict):
    """Print a concise summary of predictions across all timeframes with timestamps"""
    print("\n" + "="*50)
    print("SUMMARY REPORT")
    print("="*50)
    
    current_time = datetime.now()
    
    for symbol, timeframes in results.items():
        print(f"\n{symbol} Predictions Summary:")
        print("-" * 30)
 
        current_price = None
        if timeframes:
            first_timeframe = next(iter(timeframes.values()))
            current_price = first_timeframe['current_price']['price']
            timestamp = first_timeframe['current_price'].get('timestamp', current_time)
            print(f"Current Price: ${current_price:.2f} ({timestamp.strftime('%H:%M:%S ET')})")
        
        print("\nPredictions by Timeframe:")
        print(f"{'Timeframe':<25} {'Expected Price':<15} {'Return':<10} {'Confidence':<10}")
        print("-" * 60)
        
        for timeframe, data in timeframes.items():
            ensemble_price = data['predictions']['ensemble']
            return_pct = data['metrics']['expected_return']
            confidence = data['metrics']['confidence']
            target_time = data['target_timestamp']
            
            timeframe_str = f"{timeframe} ({target_time.strftime('%H:%M:%S ET')})"
            print(f"{timeframe_str:<25} ${ensemble_price:<14.2f} {return_pct:<9.2f}% {confidence:<9.1%}")
            
        print("\nConfidence Intervals:")
        for timeframe, data in timeframes.items():
            ci_low, ci_high = data['confidence_interval']
            print(f"{timeframe:<10} ${ci_low:.2f} to ${ci_high:.2f}")

def main():
    print("\nStarting Integrated Testing System...")
    
    try:
        results = test_integrated_predictions()
        
        if results:
            print_summary_report(results)
        else:
            print("\nNo results generated. Check error messages above.")
            
    except Exception as e:
        print(f"\nCritical error in main execution: {str(e)}")


if __name__ == "__main__":
    main()