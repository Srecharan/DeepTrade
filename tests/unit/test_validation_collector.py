# test_validation_collector.py
from utils.validation_system import ValidationSystem
from utils.model_improver import ModelImprover
from datetime import datetime, timedelta
import pytz
import time
import sys
import numpy as np
from typing import Optional, Dict, Tuple, List, Any
from tabulate import tabulate
import pandas as pd
import requests
from requests.exceptions import RequestException

class ValidationCollector:
    def __init__(self):
        self.validator = ValidationSystem()
        self.improver = ModelImprover()
        self.et_tz = pytz.timezone('US/Eastern')
        self.max_retries = 3
        self.retry_delay = 60  # 1 minute delay between retries
        
    def is_market_hours(self, current_time: datetime) -> bool:
        """Check if current time is during market hours in ET"""
        if current_time.tzinfo != self.et_tz:
            current_time = current_time.astimezone(self.et_tz)
        
        # Not market hours on weekends
        if current_time.weekday() >= 5:
            return False
            
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
        
        is_open = market_open <= current_time <= market_close
        return is_open

    def get_next_market_open(self, current_time: datetime) -> datetime:
        """Get next market open time"""
        if current_time.tzinfo != self.et_tz:
            current_time = current_time.astimezone(self.et_tz)
        
        next_day = current_time + timedelta(days=1)
        next_day = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
        
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        
        return next_day

    def handle_collection_error(self, symbol: str, error: Exception, error_count: dict) -> bool:
        """Handle errors during collection with proper retry logic"""
        if symbol not in error_count:
            error_count[symbol] = 0
        error_count[symbol] += 1
        
        print(f"\nError collecting data for {symbol} (attempt {error_count[symbol]}/{self.max_retries})")
        print(f"Error type: {type(error).__name__}")
        print(f"Error details: {str(error)}")
        
        if error_count[symbol] >= self.max_retries:
            print(f"Max retries reached for {symbol}. Skipping for now.")
            return False
        
        retry_time = min(self.retry_delay * error_count[symbol], 300)  # Max 5 minutes
        print(f"Waiting {retry_time} seconds before retry...")
        time.sleep(retry_time)
        return True

    def print_validation_summary(self, symbol: str, metrics: Dict):
        """Print a well-formatted validation summary for a symbol"""
        print(f"\n{'='*30} {symbol} Summary {'='*30}")
        
        table_data = []
        headers = ['Timeframe', 'Samples', 'Direction Acc.', 'MAE', 'CI Coverage']
        
        for timeframe, m in metrics.items():
            if m and m.get('sample_size', 0) > 0:
                table_data.append([
                    timeframe,
                    m['sample_size'],
                    f"{m['direction_accuracy']:.1f}%",
                    f"{m['mean_absolute_error']:.2f}%",
                    f"{m['ci_coverage']:.1f}%"
                ])
        
        if table_data:
            print(tabulate(table_data, headers=headers, tablefmt='grid'))
        else:
            print("No validation data available")

    def print_final_report(self, all_metrics: Dict[str, Dict]):
        """Print comprehensive final report"""
        print("\n" + "="*80)
        print(" "*30 + "FINAL VALIDATION REPORT")
        print("="*80 + "\n")
        
        comparison_data = []
        symbols = list(all_metrics.keys())
        
        sample_data = ["Sample Size"]
        for symbol in symbols:
            max_samples = max((m.get('sample_size', 0) for m in all_metrics[symbol].values()), default=0)
            sample_data.append(str(max_samples))
        comparison_data.append(sample_data)
        
        for timeframe in ['5min', '15min', '30min', '1h']:
            timeframe_data = [timeframe]
            
            for symbol in symbols:
                if symbol in all_metrics and timeframe in all_metrics[symbol]:
                    metrics = all_metrics[symbol][timeframe]
                    if metrics and metrics.get('sample_size', 0) > 0:
                        accuracy = f"{metrics['direction_accuracy']:.1f}%"
                    else:
                        accuracy = "N/A"
                    timeframe_data.append(accuracy)
                else:
                    timeframe_data.append("N/A")
                    
            comparison_data.append(timeframe_data)
        
        headers = ['Metric'] + symbols
        print("Performance Comparison:")
        print(tabulate(comparison_data, headers=headers, tablefmt='grid'))
        print("\n")

    def calculate_adaptive_interval(self, metrics: Dict) -> int:
        """Calculate adaptive collection interval based on performance"""
        base_interval = 300  # 5 minutes
        if not metrics:
            return base_interval
            
        accuracy_factor = 1.0
        for timeframe, m in metrics.items():
            if m and m['sample_size'] > 5:
                if m['direction_accuracy'] < 45:
                    accuracy_factor = 0.8
                elif m['direction_accuracy'] > 60:
                    accuracy_factor = 1.2
                    
        return int(base_interval * accuracy_factor)

    def run_collection(self, symbols: List[str], days: int = 1):
        """Run validation collection with enhanced reporting"""
        print(f"\nStarting Enhanced Validation Collection")
        print(f"Monitoring symbols: {', '.join(symbols)}")
        print(f"Collection period: {days} trading day{'s' if days > 1 else ''}")
        print("=" * 50)
        
        days_completed = 0
        last_trading_date = None
        all_metrics = {symbol: {} for symbol in symbols}
        error_counts = {}
        
        while days_completed < days:
            try:
                now = datetime.now(self.et_tz)
                current_date = now.date()
                
                is_market_open = self.is_market_hours(now)
                current_time = now.strftime('%H:%M:%S ET')
                
                if not is_market_open:
                    next_open = self.get_next_market_open(now)
                    wait_seconds = (next_open - now).total_seconds()
                    
                    if now.weekday() >= 5:
                        print(f"\nWeekend detected at {current_time}. Waiting for Monday.")
                    else:
                        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
                        if now > market_close:
                            # Don't increment days_completed here, wait for next market day
                            print(f"\nMarket closed for today at {current_time}")
                        else:
                            print(f"\nOutside market hours at {current_time}")
                    
                    print(f"Next collection starts at: {next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    print(f"Waiting {wait_seconds/60:.1f} minutes for next market session...")
                    
                    sleep_interval = min(wait_seconds, 300)  # 5 minutes max
                    time.sleep(sleep_interval)
                    continue
                
                if last_trading_date is None or current_date != last_trading_date:
                    last_trading_date = current_date
                    if now > market_close:  # Only increment at end of trading day
                        days_completed += 1
                        print(f"\nCompleted trading day {days_completed} of {days}")
                
                print(f"\nCollecting data at {current_time}")

                for symbol in symbols:
                    try:
                        metrics = self.validator.validate_stored_predictions(symbol)
                        if metrics:
                            all_metrics[symbol] = metrics
                            self.print_validation_summary(symbol, metrics)
                            error_counts[symbol] = 0  # Reset error count on success
                    except Exception as e:
                        if not self.handle_collection_error(symbol, e, error_counts):
                            continue
                
                wait_time = self.calculate_adaptive_interval(metrics if 'metrics' in locals() else None)
                print(f"\nWaiting {wait_time} seconds before next check...")
                time.sleep(wait_time)
                
            except KeyboardInterrupt:
                print("\nCollection stopped by user")
                break
            except Exception as e:
                print(f"\nUnexpected error in main loop: {str(e)}")
                time.sleep(60)  # Wait a minute before continuing
        
        print("\nValidation collection completed!")
        self.print_final_report(all_metrics) 
        return all_metrics
    
def main():
    try:

        symbols = ['NVDA', 'AAPL', 'MSFT', 'GME', 'AMD', 'JNJ', 'META', 'GOOGL', 'AMZN' ]
        #symbols =['NVDA']
        days = 1  # Run for 1 trading day
        
        collector = ValidationCollector()
        metrics = collector.run_collection(symbols, days)
        
    except KeyboardInterrupt:
        print("\nValidation collection stopped by user")
    except Exception as e:
        print(f"\nFatal error in validation collection: {str(e)}")
    finally:
        print("\nValidation collection process ended")

if __name__ == "__main__":
    main()