import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import json
import os

class TrainingMetricsTracker:
    def __init__(self, metrics_dir: str = "models/metrics/"):
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate performance metrics"""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        return metrics
    
    def save_metrics(self, symbol: str, model_type: str, metrics: dict):
        """Save metrics to JSON file"""
        filename = f"{self.metrics_dir}/{symbol}_{model_type}_metrics.json"
        
        # Add timestamp
        metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save metrics
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def load_metrics(self, symbol: str, model_type: str) -> dict:
        """Load metrics from JSON file"""
        filename = f"{self.metrics_dir}/{symbol}_{model_type}_metrics.json"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return {}