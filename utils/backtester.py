# utils/backtester.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error

class Backtester:
    def __init__(self, train_size: float = 0.8):
        self.train_size = train_size
        self.results = {}
        
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets"""
        split_idx = int(len(data) * self.train_size)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        return train_data, test_data
    
    def evaluate_predictions(self, predictions: np.ndarray, actual: np.ndarray) -> Dict:
        """Calculate various metrics for model evaluation"""
        results = {
            'mse': mean_squared_error(actual, predictions),
            'rmse': np.sqrt(mean_squared_error(actual, predictions)),
            'mae': mean_absolute_error(actual, predictions),
            'accuracy_direction': np.mean((predictions[1:] > predictions[:-1]) == 
                                       (actual[1:] > actual[:-1]))
        }
        return results
    
    def run_backtest(self, model, data: pd.DataFrame, features: List[str],
                    target: str, lookback: int = 30) -> Dict:
        """Run backtest simulation"""
        train_data, test_data = self.split_data(data)
        
        # Train model
        X_train = train_data[features].values
        y_train = train_data[target].values
        model.fit(X_train, y_train)
        
        # Test predictions
        X_test = test_data[features].values
        y_test = test_data[target].values
        predictions = model.predict(X_test)
        
        # Calculate metrics
        results = self.evaluate_predictions(predictions, y_test)
        
        # Calculate returns
        predicted_returns = pd.Series(predictions).pct_change()
        actual_returns = pd.Series(y_test).pct_change()
        
        # Simple trading strategy
        positions = np.sign(predicted_returns)
        strategy_returns = positions * actual_returns
        
        # Add strategy metrics
        results['sharpe_ratio'] = np.sqrt(252) * (strategy_returns.mean() / strategy_returns.std())
        results['cumulative_return'] = (1 + strategy_returns).prod() - 1
        
        self.results = results
        return results