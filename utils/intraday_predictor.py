# utils/intraday_predictor.py
#Not using this anywhere
import numpy as np
import pandas as pd
from typing import Dict
from datetime import datetime, timedelta
import pytz

class IntradayPredictor:
    def __init__(self, base_predictor):
        self.base_predictor = base_predictor
        self.intraday_data = {}
        self.est_tz = pytz.timezone('US/Eastern')
        
    def calculate_vwap(self, data: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price"""
        return (data['price'] * data['volume']).sum() / data['volume'].sum()
        
    def calculate_price_momentum(self, data: pd.DataFrame) -> float:
        """Calculate short-term price momentum"""
        returns = data['price'].pct_change()
        return returns.mean() * np.sqrt(len(returns))
        
    def adjust_prediction(self, symbol: str, current_data: Dict) -> Dict:
        """Adjust base prediction using intraday patterns"""
        if symbol not in self.intraday_data:
            self.intraday_data[symbol] = []
        
        self.intraday_data[symbol].append(current_data)

        cutoff_time = pd.Timestamp.now() - timedelta(minutes=30)
        self.intraday_data[symbol] = [
            d for d in self.intraday_data[symbol] 
            if d['timestamp'] > cutoff_time
        ]

        base_pred = self.base_predictor.predict(symbol)
        
        if len(self.intraday_data[symbol]) < 5:  
            return base_pred
            
        df = pd.DataFrame(self.intraday_data[symbol])

        vwap = self.calculate_vwap(df)
        momentum = self.calculate_price_momentum(df)
        volatility = df['price'].std() / df['price'].mean()
        
        # Adjust prediction
        price_adjustment = 0.0
        
        current_price = df['price'].iloc[-1]
        vwap_diff = (current_price - vwap) / vwap
        price_adjustment += vwap_diff * 0.3  
        
        price_adjustment += momentum * 0.4  
        
        price_adjustment *= (1 - volatility)  
        adjusted_predictions = base_pred.copy()
        for key in ['lstm', 'xgboost', 'ensemble']:
            if key in adjusted_predictions:
                adjusted_predictions[key] *= (1 + price_adjustment)
                
        if 'risk_metrics' in adjusted_predictions:
            conf_adjustment = max(0, 1 - volatility * 2)
            adjusted_predictions['risk_metrics']['confidence_score'] *= conf_adjustment
            
        return adjusted_predictions
    
    