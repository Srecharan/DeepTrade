import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List
from scipy import stats
from .model_trainer import EnhancedLSTM  # Changed from LSTM to EnhancedLSTM
from .stock_manager import StockManager
import xgboost as xgb
import os
import pickle
from datetime import datetime, timedelta
import numpy as np

class PredictionSystem:
    def __init__(self, model_dir: str = "models/trained/"):
        self.model_dir = model_dir
        self.models = {}
        self.stock_manager = StockManager()
        
    def predict(self, symbol: str, window_size: int = 30, sentiment_data: Dict = None, timeframe: str = None) -> Dict:
        """Enhanced prediction with improved confidence calculation
        
        Args:
            symbol: Stock symbol
            window_size: Window size for LSTM predictions
            sentiment_data: Pre-calculated sentiment data to avoid redundant API calls
            timeframe: One of '5min', '15min', '30min', '1h', '1d'
        """
        try:
            if f'{symbol}_lstm' not in self.models:
                self.load_models(symbol)
                    
            if sentiment_data:
                print(f"Using cached sentiment for {symbol}")
                data, sentiment = self.stock_manager.prepare_prediction_data(
                    symbol, 
                    use_cached_sentiment=True,
                    sentiment_data=sentiment_data
                )
            else:
                data, sentiment = self.stock_manager.prepare_prediction_data(symbol)
            
            # Prepare features
            X = self.stock_manager.prepare_scaled_features(data)
            
            # Adapt features to match expected model input size (39)
            X = self._adapt_features(X, expected_size=39)
            
            # Get real-time price
            real_time_price = self.stock_manager.get_real_time_price(symbol)
            if real_time_price:
                current_price = real_time_price['price']
            else:
                current_price = data['Close'].iloc[-1]  # Fallback to last close
                    
            predictions = {}
            all_preds = []
            
            # LSTM prediction
            if f'{symbol}_lstm' in self.models:
                lstm_model = self.models[f'{symbol}_lstm']
                # Make sure we have enough data for the window
                if len(X) < window_size:
                    print(f"WARNING: Not enough data for window size {window_size}, using available data length {len(X)}")
                    sequence = torch.FloatTensor(X).unsqueeze(0).cuda()
                else:
                    sequence = torch.FloatTensor(X[-window_size:]).unsqueeze(0).cuda()
                
                with torch.no_grad():
                    scaled_pred = lstm_model(sequence).cpu().item()
                    lstm_price = current_price * (1 + scaled_pred/100)
                    predictions['lstm'] = lstm_price
                    all_preds.append(lstm_price)
            
            # XGBoost prediction
            if f'{symbol}_xgb' in self.models:
                xgb_model = self.models[f'{symbol}_xgb']
                # Adapt the input for XGBoost too
                xgb_input = self._adapt_features(X[-1:], expected_size=39)
                scaled_pred = xgb_model.predict(xgb_input)
                xgb_price = current_price * (1 + scaled_pred[0]/100)
                predictions['xgboost'] = xgb_price
                all_preds.append(xgb_price)
            
            # Rest of your method remains the same...
            if predictions:
                # Dynamic weights based on recent performance
                lstm_weight = 0.4
                xgb_weight = 0.6
                
                ensemble_price = (
                    lstm_weight * predictions['lstm'] + 
                    xgb_weight * predictions['xgboost']
                )
                predictions['ensemble'] = ensemble_price
                
                # Calculate metrics
                volatility = data['Close'].pct_change().std() * np.sqrt(252)
                confidence_score = self.calculate_confidence_score(all_preds, volatility)
                
                # Base confidence interval
                std_dev = np.std(all_preds)
                t_value = stats.t.ppf(0.975, df=len(all_preds)-1)
                margin = t_value * std_dev / np.sqrt(len(all_preds))
                
                predictions['confidence_interval'] = (
                    ensemble_price - margin,
                    ensemble_price + margin
                )
                
                # Risk metrics
                risk_metrics = {
                    'current_price': real_time_price if real_time_price else {
                        'price': current_price,
                        'timestamp': datetime.now(),
                        'market_hours': self.stock_manager._is_market_hours(),
                        'sources': ['historical']
                    },
                    'prediction_price': ensemble_price,
                    'volatility': volatility,
                    'prediction_return': (ensemble_price / current_price - 1) * 100,
                    'sentiment_score': sentiment,
                    'confidence_score': confidence_score
                }
                predictions['risk_metrics'] = risk_metrics
                
                # Scale predictions for timeframe if specified
                if timeframe:
                    timeframe_minutes = {
                        '5min': 5,
                        '15min': 15,
                        '30min': 30,
                        '1h': 60,
                        '1d': 1440
                    }
                    
                    if timeframe not in timeframe_minutes:
                        raise ValueError(f"Invalid timeframe: {timeframe}")
                        
                    minutes = timeframe_minutes[timeframe]
                    scaling_factor = np.sqrt(minutes / 1440)  # Scale relative to daily
                    
                    # Scale predictions
                    for model in predictions:
                        if model not in ['confidence_interval', 'risk_metrics']:
                            base_return = (predictions[model] / current_price - 1)
                            timeframe_return = base_return * scaling_factor
                            predictions[model] = current_price * (1 + timeframe_return)
                    
                    # Scale confidence interval
                    margin = margin * scaling_factor
                    predictions['confidence_interval'] = (
                        predictions['ensemble'] - margin,
                        predictions['ensemble'] + margin
                    )
                    
                    # Scale volatility
                    predictions['risk_metrics']['volatility'] *= scaling_factor
                    predictions['risk_metrics']['prediction_return'] = (
                        (predictions['ensemble'] / current_price - 1) * 100
                    )
                    
                    predictions['timeframe'] = timeframe
                    predictions['target_timestamp'] = (
                        datetime.now() + timedelta(minutes=minutes)
                    )
                
            return predictions
                
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise e

    def predict_timeframe(self, symbol: str, timeframe: str, sentiment_data: Dict = None) -> Dict:
        """Make predictions for specific timeframes
        
        Args:
            symbol: Stock symbol
            timeframe: One of '5min', '15min', '30min', '1h', '1d'
            sentiment_data: Optional pre-calculated sentiment data
        """
        # Validate timeframe
        valid_timeframes = ['5min', '15min', '30min', '1h', '1d']
        if timeframe not in valid_timeframes:
            raise ValueError(f"Invalid timeframe. Must be one of {valid_timeframes}")
        
        predictions = self.predict(
            symbol, 
            window_size=30,
            sentiment_data=sentiment_data,  
            timeframe=timeframe
        )
        
        if not predictions:
            return None
            
        current_price = predictions['risk_metrics']['current_price']
        
        minutes_map = {
            '5min': 5,
            '15min': 15,
            '30min': 30,
            '1h': 60,
            '1d': 1440
        }
        target_time = datetime.now() + timedelta(minutes=minutes_map[timeframe])
        
        return {
            'timeframe': timeframe,
            'current_price': current_price,
            'target_timestamp': target_time,
            'predictions': {
                'lstm': predictions['lstm'],
                'xgboost': predictions['xgboost'],
                'ensemble': predictions['ensemble']
            },
            'metrics': {
                'expected_return': predictions['risk_metrics']['prediction_return'],
                'confidence': predictions['risk_metrics']['confidence_score'],
                'volatility': predictions['risk_metrics']['volatility']
            },
            'confidence_interval': predictions['confidence_interval']
        }
        
    def load_models(self, symbol: str) -> None:
        """Enhanced model loading with feature verification"""
        try:
            # Load scalers first
            scaler_path = os.path.join(self.model_dir, f'{symbol}_scalers.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scalers = pickle.load(f)
                print("Successfully loaded scalers")
            
            # Get feature dimensions and print debug info
            data, _ = self.stock_manager.prepare_prediction_data(symbol)
            X = self.stock_manager.prepare_scaled_features(data)
            input_size = X.shape[1]
            print(f"DEBUG - Current data feature count: {input_size}")
            
            # Expected model input size based on model diagnostics
            expected_input_size = 39  # From model diagnostics
            
            # Load LSTM with correct input size
            lstm_path = os.path.join(self.model_dir, f'{symbol}_lstm_best.pth')
            if os.path.exists(lstm_path):
                lstm_model = EnhancedLSTM(
                    input_size=expected_input_size,  # Use the expected size from diagnostics
                    hidden_size=256,
                    num_layers=3,
                    dropout=0.3
                )
                
                state_dict = torch.load(lstm_path, weights_only=True)
                lstm_model.load_state_dict(state_dict)
                lstm_model.eval()
                lstm_model = lstm_model.cuda()  # Move to GPU
                self.models[f'{symbol}_lstm'] = lstm_model
                print("Successfully loaded LSTM model")
            
            # Load XGBoost
            xgb_path = os.path.join(self.model_dir, f'{symbol}_xgb.json')
            if os.path.exists(xgb_path):
                xgb_model = xgb.XGBRegressor()
                xgb_model.load_model(xgb_path)
                self.models[f'{symbol}_xgb'] = xgb_model
                print("Successfully loaded XGBoost model")
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise e
            
    
    def _print_prediction_summary(self, symbol: str, predictions: Dict) -> None:
            """Pretty print prediction summary"""
            risk_metrics = predictions['risk_metrics']
            print(f"\nPrediction Summary for {symbol}:")
            print(f"Current Price: ${risk_metrics['current_price']:.2f}")
            print(f"LSTM Prediction: ${predictions['lstm']:.2f}")
            print(f"XGBoost Prediction: ${predictions['xgboost']:.2f}")
            print(f"Ensemble Prediction: ${predictions['ensemble']:.2f}")
            print(f"Predicted Return: {risk_metrics['prediction_return']:.2f}%")
            print(f"Confidence Score: {risk_metrics['confidence_score']:.2%}")

    def calculate_risk_metrics(self, data: pd.DataFrame, prediction: float) -> Dict:
        """Calculate various risk metrics"""
        returns = data['Close'].pct_change().dropna()
        current_price = data['Close'].iloc[-1]
        
        volatility = returns.std() * np.sqrt(252)
        var_95 = np.percentile(returns, 5)
        recent_trend = returns.tail(5).mean()
        prediction_return = (prediction / current_price) - 1
        
        risk_free_rate = 0.04  # Assume 4% risk-free rate
        excess_return = returns.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_return / (volatility) if volatility != 0 else 0
        
        risk_metrics = {
            'volatility': volatility,
            'var_95': var_95,
            'recent_trend': recent_trend,
            'prediction_return': prediction_return,
            'sharpe_ratio': sharpe_ratio,
            'current_price': current_price,
            'prediction_price': prediction
        }
        
        return risk_metrics
    
    def calculate_prediction_confidence(self, predictions: Dict, data: pd.DataFrame) -> float:
        """Calculate confidence score based on multiple factors"""
        pred_values = np.array(list(predictions.values()))
        model_agreement = 1 - (np.std(pred_values) / np.mean(np.abs(pred_values)))
        
        volatility = data['Close'].pct_change().std() * np.sqrt(252)
        vol_score = 1 / (1 + volatility)

        sent_std = data['raw_sentiment'].tail(5).std()
        sent_score = 1 / (1 + sent_std)

        confidence = (model_agreement * 0.5 + vol_score * 0.3 + sent_score * 0.2)
        return min(max(confidence, 0), 1)  # Normalize to [0,1]
    
    # Add to prediction_system.py
    def ensemble_predict(self, predictions: Dict, data: pd.DataFrame) -> float:
        """Advanced ensemble prediction with dynamic weighting"""
        # Get recent performance metrics
        lstm_error = self.calculate_recent_error('lstm', data)
        xgb_error = self.calculate_recent_error('xgb', data)
        
        # Calculate dynamic weights
        total_error = lstm_error + xgb_error
        if total_error == 0:
            lstm_weight = 0.5
            xgb_weight = 0.5
        else:
            lstm_weight = 1 - (lstm_error / total_error)
            xgb_weight = 1 - (xgb_error / total_error)
        
        # Normalize weights
        sum_weights = lstm_weight + xgb_weight
        lstm_weight /= sum_weights
        xgb_weight /= sum_weights
        
        # Calculate weighted prediction
        ensemble_pred = (
            lstm_weight * predictions['lstm'] +
            xgb_weight * predictions['xgboost']
        )
        
        return ensemble_pred
    
    def calculate_confidence_score(self, predictions: list, volatility: float) -> float:
        """Improved confidence score calculation"""
        if len(predictions) < 2:
            return 0.0

        std_dev = np.std(predictions)
        mean_pred = np.mean(predictions)
        agreement_score = 1 - (std_dev / (mean_pred + 1e-6))  # Avoid division by zero
        
        # Adjust for volatility (higher volatility = lower confidence)
        volatility_score = 1 / (1 + volatility)
        
        # Combine scores with weights
        confidence = (0.7 * agreement_score + 0.3 * volatility_score)
        
        # Scale to reasonable range (0.5-0.95)
        confidence = 0.5 + (confidence * 0.45)
        
        return confidence

    def calculate_confidence_interval(self, current_price: float, predictions: list, 
                               volatility: float, timeframe: str) -> Tuple[float, float]:
        """Calculate dynamic confidence intervals based on multiple factors"""
        
        # 1. Base interval from prediction spread
        std_dev = np.std(predictions)
        
        # 2. Adjust for timeframe
        timeframe_factors = {
            '5min': 0.5,   
            '15min': 0.75,
            '30min': 1.0,
            '1h': 1.25    
        }
        timeframe_factor = timeframe_factors.get(timeframe, 1.0)
        
        # 3. Adjust for volatility
        volatility_factor = 1 + (volatility * 2)  # Higher volatility = wider interval
        
        # 4. Adjust for market hours
        is_market_hours = self.stock_manager._is_market_hours()
        market_factor = 1.0 if is_market_hours else 1.5  # Wider intervals outside market hours
        
        # 5. Calculate margin
        base_margin = std_dev * timeframe_factor * volatility_factor * market_factor
        
        # 6. Apply minimum margin based on current price
        min_margin = current_price * 0.001  # Minimum 0.1% margin
        margin = max(base_margin, min_margin)
        
        # Calculate mean prediction
        mean_pred = np.mean(predictions)
        
        return (mean_pred - margin, mean_pred + margin)
    
    def _adapt_features(self, X: np.ndarray, expected_size: int = 39) -> np.ndarray:
        """Adapt feature array to match the expected size of 39 for model compatibility"""
        current_size = X.shape[1]
        
        if current_size == expected_size:
            return X
            
        print(f"Adapting features from size {current_size} to {expected_size}")
        
        if current_size > expected_size:
            # If we have too many features, select only the first expected_size
            print(f"Too many features: {current_size}, truncating to {expected_size}")
            return X[:, :expected_size]
            
        elif current_size < expected_size:
            # If we have too few features, pad with zeros
            print(f"Too few features: {current_size}, padding to {expected_size}")
            padding = np.zeros((X.shape[0], expected_size - current_size))
            return np.hstack([X, padding])
    
