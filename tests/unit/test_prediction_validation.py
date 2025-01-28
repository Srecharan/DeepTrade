# test_prediction_validation.py

from utils.prediction_system import PredictionSystem
from utils.stock_manager import StockManager
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
from typing import Dict, List

class PredictionValidationSystem:
    def __init__(self):
        self.prediction_system = PredictionSystem()
        self.stock_manager = StockManager()
        self.validation_data = {}
        self.results_dir = "predictions/validation_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def store_prediction(self, symbol: str, prediction_data: Dict):
        """Store a prediction for validation"""
        timestamp = datetime.now()
        
        if symbol not in self.validation_data:
            self.validation_data[symbol] = []
            
        for timeframe, data in prediction_data.items():
            prediction_record = {
                'timestamp': timestamp.isoformat(),
                'timeframe': timeframe,
                'current_price': data['current_price']['price'],
                'predicted_price': data['predictions']['ensemble'],
                'confidence': data['metrics']['confidence'],
                'confidence_interval': data['confidence_interval'],
                'target_time': (timestamp + self._get_timeframe_delta(timeframe)).isoformat(),
                'actual_price': None,
                'validated': False
            }
            
            self.validation_data[symbol].append(prediction_record)
            
        self._save_validation_data(symbol)
            
    def _get_timeframe_delta(self, timeframe: str) -> timedelta:
        """Convert timeframe string to timedelta"""
        timeframe_map = {
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '30min': timedelta(minutes=30),
            '1h': timedelta(hours=1)
        }
        return timeframe_map.get(timeframe)
        
    def validate_predictions(self, symbol: str):
        """Validate stored predictions against actual prices"""
        if symbol not in self.validation_data:
            print(f"No predictions found for {symbol}")
            return
            
        current_time = datetime.now()
        validated_count = 0
        
        for prediction in self.validation_data[symbol]:
            if prediction['validated']:
                continue
                
            target_time = datetime.fromisoformat(prediction['target_time'])
            if current_time >= target_time:
                # Get actual price
                try:
                    actual_price_data = self.stock_manager.get_real_time_price(symbol)
                    if actual_price_data:
                        prediction['actual_price'] = actual_price_data['price']
                        prediction['validated'] = True
                        validated_count += 1
                except Exception as e:
                    print(f"Error getting actual price for {symbol}: {e}")
                    
        if validated_count > 0:
            print(f"Validated {validated_count} predictions for {symbol}")
            self._save_validation_data(symbol)
            self._calculate_and_save_metrics(symbol)
            
    def _save_validation_data(self, symbol: str):
        """Save validation data to file"""
        filepath = os.path.join(self.results_dir, f"{symbol}_test_validation_data.json")
        with open(filepath, 'w') as f:
            json.dump(self.validation_data[symbol], f, indent=4)
            
    def _calculate_and_save_metrics(self, symbol: str):
        """Calculate and save validation metrics"""
        if symbol not in self.validation_data:
            return
            
        validated_preds = [p for p in self.validation_data[symbol] if p['validated']]
        if not validated_preds:
            return
            
        metrics = {timeframe: {} for timeframe in ['5min', '15min', '30min', '1h']}
        
        for timeframe in metrics:
            timeframe_preds = [p for p in validated_preds if p['timeframe'] == timeframe]
            if timeframe_preds:
                errors = []
                direction_correct = []
                within_confidence = []
                
                for pred in timeframe_preds:
                    if pred['actual_price'] is not None:
                        error = ((pred['actual_price'] - pred['predicted_price']) / 
                                pred['current_price']) * 100
                        errors.append(abs(error))
                        
                        pred_direction = pred['predicted_price'] > pred['current_price']
                        actual_direction = pred['actual_price'] > pred['current_price']
                        direction_correct.append(pred_direction == actual_direction)
                        
                        within_ci = (pred['confidence_interval'][0] <= pred['actual_price'] <= 
                                   pred['confidence_interval'][1])
                        within_confidence.append(within_ci)
                
                if errors:
                    metrics[timeframe] = {
                        'mean_absolute_error': np.mean(errors),
                        'max_error': np.max(errors),
                        'direction_accuracy': np.mean(direction_correct) * 100,
                        'confidence_interval_accuracy': np.mean(within_confidence) * 100,
                        'sample_size': len(errors)
                    }
        
        metrics_filepath = os.path.join(self.results_dir, f"{symbol}_metrics.json")
        with open(metrics_filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        self._print_metrics_report(symbol, metrics)
    
    def _print_metrics_report(self, symbol: str, metrics: Dict):
        """Print a formatted metrics report"""
        print(f"\nValidation Metrics for {symbol}")
        print("=" * 50)
        
        for timeframe, timeframe_metrics in metrics.items():
            if timeframe_metrics:
                print(f"\n{timeframe} Timeframe (Sample size: {timeframe_metrics['sample_size']})")
                print(f"Mean Absolute Error: {timeframe_metrics['mean_absolute_error']:.2f}%")
                print(f"Maximum Error: {timeframe_metrics['max_error']:.2f}%")
                print(f"Direction Accuracy: {timeframe_metrics['direction_accuracy']:.1f}%")
                print(f"Confidence Interval Accuracy: "
                      f"{timeframe_metrics['confidence_interval_accuracy']:.1f}%")


def main():
    """Run validation system"""
    validator = PredictionValidationSystem()
    prediction_system = PredictionSystem()
    
    #symbols = ['NVDA', 'AAPL', 'MSFT', 'GME', 'AMD', 'JNJ', 'META', 'GOOGL', 'AMZN' ]
    symbols = ['NVDA']
    
    print("\nStarting Prediction Validation")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
    print("=" * 50)
    
    try:
        # First, make new predictions
        for symbol in symbols:
            print(f"\nProcessing {symbol}...")
            
            timeframe_predictions = {}
            for timeframe in ['5min', '15min', '30min', '1h']:
                try:
                    predictions = prediction_system.predict_timeframe(symbol, timeframe)
                    if predictions:
                        timeframe_predictions[timeframe] = predictions
                except Exception as e:
                    print(f"Error predicting {timeframe} for {symbol}: {e}")
            
            if timeframe_predictions:
                validator.store_prediction(symbol, timeframe_predictions)
        
        print("\nValidating previous predictions...")
        for symbol in symbols:
            validator.validate_predictions(symbol)
            
    except Exception as e:
        print(f"\nCritical error in validation: {str(e)}")

if __name__ == "__main__":
    main()