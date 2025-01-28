import json
import pandas as pd
from datetime import datetime
from typing import Dict, List
import os
from collections import defaultdict

class PerformanceAnalyzer:
    def __init__(self, results_dir: str = "predictions/validation_results"):
        self.results_dir = results_dir
        
    def analyze_daily_performance(self, symbol: str, date: str = None) -> Dict:
        """Analyze performance metrics for a specific day"""
        metrics_file = os.path.join(self.results_dir, f"{symbol}_metrics.json")
        
        if not os.path.exists(metrics_file):
            raise FileNotFoundError(f"No metrics file found for {symbol}")
            
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            
        summary = {
            'symbol': symbol,
            'date': date or datetime.now().strftime('%Y-%m-%d'),
            'timeframes': {}
        }
        
        for timeframe, data in metrics.items():
            if not data:  # Skip empty timeframes
                continue
                
            timeframe_metrics = {
                'sample_size': data['sample_size'],
                'mae': data['mean_absolute_error'],
                'direction_accuracy': data['direction_accuracy'],
                'ci_coverage': data['ci_coverage'],
                'max_error': data['max_error'],
                'error_std': data['error_std']
            }
            summary['timeframes'][timeframe] = timeframe_metrics
            
        total_samples = sum(tf['sample_size'] for tf in summary['timeframes'].values())
        weighted_mae = sum(tf['mae'] * tf['sample_size'] for tf in summary['timeframes'].values()) / total_samples
        weighted_direction = sum(tf['direction_accuracy'] * tf['sample_size'] for tf in summary['timeframes'].values()) / total_samples
        
        summary['aggregate'] = {
            'total_samples': total_samples,
            'weighted_mae': weighted_mae,
            'weighted_direction_accuracy': weighted_direction,
            'best_timeframe': max(summary['timeframes'].items(), 
                                key=lambda x: x[1]['direction_accuracy'])[0]
        }
        
        return summary
        
    def compare_performance(self, symbol: str, date1: str, date2: str) -> Dict:
        """Compare performance between two dates"""
        perf1 = self.analyze_daily_performance(symbol, date1)
        perf2 = self.analyze_daily_performance(symbol, date2)
        
        comparison = {
            'symbol': symbol,
            'dates': {'from': date1, 'to': date2},
            'timeframes': {},
            'aggregate': {}
        }
        
        for timeframe in set(perf1['timeframes'].keys()) & set(perf2['timeframes'].keys()):
            metrics1 = perf1['timeframes'][timeframe]
            metrics2 = perf2['timeframes'][timeframe]
            
            comparison['timeframes'][timeframe] = {
                'mae_change': metrics2['mae'] - metrics1['mae'],
                'direction_accuracy_change': metrics2['direction_accuracy'] - metrics1['direction_accuracy'],
                'ci_coverage_change': metrics2['ci_coverage'] - metrics1['ci_coverage'],
                'error_std_change': metrics2['error_std'] - metrics1['error_std']
            }

        comparison['aggregate'] = {
            'weighted_mae_change': perf2['aggregate']['weighted_mae'] - perf1['aggregate']['weighted_mae'],
            'weighted_direction_change': perf2['aggregate']['weighted_direction_accuracy'] - perf1['aggregate']['weighted_direction_accuracy'],
            'sample_size_change': perf2['aggregate']['total_samples'] - perf1['aggregate']['total_samples']
        }
        
        return comparison
        
    def print_daily_summary(self, symbol: str, date: str = None):
        """Print formatted daily performance summary"""
        try:
            summary = self.analyze_daily_performance(symbol, date)
            
            print(f"\nPerformance Summary for {symbol} - {summary['date']}")
            print("=" * 50)
            
            for timeframe, metrics in summary['timeframes'].items():
                print(f"\n{timeframe} Timeframe (n={metrics['sample_size']}):")
                print(f"MAE: {metrics['mae']:.2f}%")
                print(f"Direction Accuracy: {metrics['direction_accuracy']:.1f}%")
                print(f"CI Coverage: {metrics['ci_coverage']:.1f}%")
                print(f"Max Error: {metrics['max_error']:.2f}%")
                print(f"Error Std: {metrics['error_std']:.2f}%")
            
            print("\nAggregate Metrics:")
            print(f"Total Samples: {summary['aggregate']['total_samples']}")
            print(f"Weighted MAE: {summary['aggregate']['weighted_mae']:.2f}%")
            print(f"Weighted Direction Accuracy: {summary['aggregate']['weighted_direction_accuracy']:.1f}%")
            print(f"Best Performing Timeframe: {summary['aggregate']['best_timeframe']}")
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            
    def print_comparison(self, symbol: str, date1: str, date2: str):
        """Print formatted comparison between two dates"""
        try:
            comparison = self.compare_performance(symbol, date1, date2)
            
            print(f"\nPerformance Comparison for {symbol}")
            print(f"From {date1} to {date2}")
            print("=" * 50)
            
            for timeframe, changes in comparison['timeframes'].items():
                print(f"\n{timeframe} Timeframe Changes:")
                print(f"MAE: {changes['mae_change']:+.2f}%")
                print(f"Direction Accuracy: {changes['direction_accuracy_change']:+.1f}%")
                print(f"CI Coverage: {changes['ci_coverage_change']:+.1f}%")
                print(f"Error Std: {changes['error_std_change']:+.2f}%")
            
            print("\nAggregate Changes:")
            print(f"Weighted MAE: {comparison['aggregate']['weighted_mae_change']:+.2f}%")
            print(f"Weighted Direction Accuracy: {comparison['aggregate']['weighted_direction_change']:+.1f}%")
            print(f"Sample Size: {comparison['aggregate']['sample_size_change']:+d}")
            
        except Exception as e:
            print(f"Error generating comparison: {str(e)}")