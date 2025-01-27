# utils/validation_system.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import yfinance as yf
import pytz
from typing import Dict, List
from .stock_manager import StockManager
from .prediction_system import PredictionSystem

class ValidationSystem:
    def __init__(self):
        self.prediction_system = PredictionSystem()
        self.stock_manager = StockManager()
        self.results_dir = "predictions/validation_results"
        self.et_tz = pytz.timezone('US/Eastern')
        self.utc_tz = pytz.UTC
        os.makedirs(self.results_dir, exist_ok=True)

    def validate_stored_predictions(self, symbol: str) -> Dict:
        """Make new predictions and validate stored ones"""
        try:
            # Make new predictions for each timeframe
            timeframe_predictions = {}
            for timeframe in ['5min', '15min', '30min', '1h']:
                predictions = self.prediction_system.predict_timeframe(symbol, timeframe)
                if predictions:
                    timeframe_predictions[timeframe] = predictions

            # Store new predictions
            if timeframe_predictions:
                self.store_prediction(symbol, timeframe_predictions)
                print(f"Stored {len(timeframe_predictions)} predictions for {symbol}")

            # Validate previous predictions
            return self._validate_historical_predictions(symbol)

        except Exception as e:
            print(f"Error in validation process: {str(e)}")
            return None

    def store_prediction(self, symbol: str, prediction_data: Dict):
        """Store a prediction for validation"""
        validation_file = os.path.join(self.results_dir, f"{symbol}_validation_data.json")
        timestamp = datetime.now(self.et_tz)  # Ensure timezone-aware timestamp

        # Load existing predictions
        if os.path.exists(validation_file):
            with open(validation_file, 'r') as f:
                existing_predictions = json.load(f)
        else:
            existing_predictions = []

        # Add new predictions
        for timeframe, data in prediction_data.items():
            target_time = timestamp + self._get_timeframe_delta(timeframe)
            prediction_record = {
                'timestamp': timestamp.isoformat(),
                'timeframe': timeframe,
                'current_price': data['current_price']['price'],
                'predicted_price': data['predictions']['ensemble'],
                'confidence': data['metrics']['confidence'],
                'confidence_interval': data['confidence_interval'],
                'target_time': target_time.isoformat(),
                'actual_price': None,
                'validated': False
            }
            existing_predictions.append(prediction_record)

        # Save updated predictions
        with open(validation_file, 'w') as f:
            json.dump(existing_predictions, f, indent=4)

    def _validate_historical_predictions(self, symbol: str) -> Dict:
        """Validate stored predictions using historical data"""
        validation_file = os.path.join(self.results_dir, f"{symbol}_validation_data.json")
        if not os.path.exists(validation_file):
            return None

        with open(validation_file, 'r') as f:
            predictions = json.load(f)

        # Get time range for historical data
        start_time = datetime.fromisoformat(min(p['timestamp'] for p in predictions)).astimezone(self.et_tz)
        end_time = datetime.fromisoformat(max(p['target_time'] for p in predictions)).astimezone(self.et_tz)

        try:
            # Get historical data
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(
                start=start_time.strftime('%Y-%m-%d'),
                end=(end_time + timedelta(days=1)).strftime('%Y-%m-%d'),
                interval='1m'
            )
            hist_data.index = hist_data.index.tz_convert(self.et_tz)

            # Validate predictions
            validated_count = 0
            for pred in predictions:
                if pred['validated']:
                    continue

                target_time = datetime.fromisoformat(pred['target_time']).astimezone(self.et_tz)
                now = datetime.now(self.et_tz)

                # Only validate predictions whose target time has passed
                if target_time > now:
                    continue

                # Find closest price data point
                closest_times = hist_data.index[abs(hist_data.index - target_time) <= timedelta(minutes=1)]
                if len(closest_times) > 0:
                    closest_time = min(closest_times, key=lambda x: abs(x - target_time))
                    actual_price = hist_data.loc[closest_time, 'Close']
                    
                    # Update prediction record
                    pred['actual_price'] = float(actual_price)  # Convert numpy float to Python float
                    pred['validated'] = True
                    validated_count += 1

                    # Print validation result
                    self._print_validation_result(symbol, pred, target_time, closest_time, actual_price)

            # Save updated predictions
            with open(validation_file, 'w') as f:
                json.dump(predictions, f, indent=4)

            # Calculate and save metrics
            if validated_count > 0:
                metrics = self._calculate_metrics(predictions)
                self._save_metrics(symbol, metrics)
                self._print_metrics_summary(symbol, metrics)
                return metrics

            return None

        except Exception as e:
            print(f"Error validating predictions: {str(e)}")
            return None

    def _print_validation_result(self, symbol: str, pred: Dict, target_time: datetime, 
                               actual_time: datetime, actual_price: float):
        """Print validation results for a single prediction"""
        error_pct = ((actual_price - pred['predicted_price']) / pred['current_price']) * 100
        direction_correct = ((actual_price > pred['current_price']) == 
                           (pred['predicted_price'] > pred['current_price']))
        within_ci = (pred['confidence_interval'][0] <= actual_price <= 
                    pred['confidence_interval'][1])
        
        print(f"\nValidation Results for {symbol} {pred['timeframe']}:")
        print(f"Target Time: {target_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Actual Time: {actual_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Current: ${pred['current_price']:.2f}")
        print(f"Predicted: ${pred['predicted_price']:.2f}")
        print(f"Actual: ${actual_price:.2f}")
        print(f"Error: {error_pct:.2f}%")
        print(f"Direction Correct: {direction_correct}")
        print(f"Within Confidence Interval: {within_ci}")

    def _calculate_metrics(self, predictions: List[Dict]) -> Dict:
        """Calculate validation metrics"""
        metrics = {timeframe: {} for timeframe in ['5min', '15min', '30min', '1h']}
        validated_preds = [p for p in predictions if p['validated']]
        
        for timeframe in metrics:
            timeframe_preds = [p for p in validated_preds if p['timeframe'] == timeframe]
            if timeframe_preds:
                metrics[timeframe] = self._calculate_timeframe_metrics(timeframe_preds)
                
        return metrics
    
    def _print_metrics_summary(self, symbol: str, metrics: Dict):
        """Print summary of validation metrics"""
        print(f"\nValidation Metrics Summary for {symbol}")
        print("=" * 50)
        
        for timeframe, m in metrics.items():
            if m:  # Only print if we have metrics for this timeframe
                print(f"\n{timeframe} Timeframe (n={m['sample_size']}):")
                print(f"Mean Absolute Error: {m['mean_absolute_error']:.2f}%")
                print(f"Median Error: {m['median_error']:.2f}%")
                print(f"Max Error: {m['max_error']:.2f}%")
                print(f"Direction Accuracy: {m['direction_accuracy']:.1f}%")
                print(f"Confidence Interval Coverage: {m['ci_coverage']:.1f}%")
                if 'error_std' in m:
                    print(f"Error Standard Deviation: {m['error_std']:.2f}%")


    def _calculate_timeframe_metrics(self, predictions: List[Dict]) -> Dict:
        """Calculate metrics for a specific timeframe"""
        errors = []
        direction_correct = []
        within_ci = []
        
        for pred in predictions:
            actual = pred['actual_price']
            predicted = pred['predicted_price']
            current = pred['current_price']
            
            error_pct = ((actual - predicted) / current) * 100
            errors.append(abs(error_pct))
            
            direction_correct.append(
                (actual > current) == (predicted > current)
            )
            within_ci.append(
                pred['confidence_interval'][0] <= actual <= pred['confidence_interval'][1]
            )
        
        return {
            'sample_size': len(predictions),
            'mean_absolute_error': np.mean(errors),
            'median_error': np.median(errors),
            'max_error': np.max(errors),
            'direction_accuracy': np.mean(direction_correct) * 100,
            'ci_coverage': np.mean(within_ci) * 100,
            'error_std': np.std(errors)
        }

    def _save_metrics(self, symbol: str, metrics: Dict):
        """Save metrics to file"""
        metrics_file = os.path.join(self.results_dir, f"{symbol}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)

        print("\nValidation Metrics Summary for {symbol}")
        print("=" * 50)
        for timeframe, m in metrics.items():
            if m:
                print(f"\n{timeframe} Timeframe (n={m['sample_size']}):")
                print(f"Mean Absolute Error: {m['mean_absolute_error']:.2f}%")
                print(f"Direction Accuracy: {m['direction_accuracy']:.1f}%")
                print(f"Confidence Interval Coverage: {m['ci_coverage']:.1f}%")
            
    def _get_timeframe_delta(self, timeframe: str) -> timedelta:
        """Convert timeframe string to timedelta"""
        timeframe_map = {
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '30min': timedelta(minutes=30),
            '1h': timedelta(hours=1)
        }
        return timeframe_map.get(timeframe)