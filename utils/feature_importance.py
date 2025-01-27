import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import shap

class FeatureImportanceAnalyzer:
    def __init__(self):
        self.feature_names = [
            # Technical features
            'RSI', 'MACD', 'SMA_20', '%K', '%D', 'ROC', 'MFI',
            'BB_upper', 'BB_lower', 'ATR', 'OBV',
            'Signal_Line', 'BB_middle', 'Volume_MA', 'Volume_Ratio',
            
            # Sentiment features
            'raw_sentiment', 'sentiment_3d', 'sentiment_7d',
            'sentiment_momentum', 'sentiment_volume',
            'sentiment_acceleration', 'sentiment_regime',
            'sentiment_trend_strength',
            'sent_price_momentum', 'sentiment_trend'
        ]
    def analyze_importance(self, X: np.ndarray, y: np.ndarray):
        """Enhanced feature importance analysis"""
        # Add multiple analysis methods
        rf_model = RandomForestRegressor(n_estimators=200, max_depth=10)
        rf_model.fit(X, y)
        rf_importance = rf_model.feature_importances_
        
        # Add SHAP values analysis
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X)
        
        # Combine importance scores
        combined_importance = (rf_importance + np.mean(np.abs(shap_values), axis=0)) / 2
        
        feat_imp = pd.DataFrame({
            'Feature': self.feature_names[:len(combined_importance)],
            'Importance': combined_importance
        }).sort_values('Importance', ascending=False)
        
        return feat_imp