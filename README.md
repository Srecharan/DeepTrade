# DeepTrade AI: Multi-Model Stock Prediction with NLP & Automated Trading

![Python](https://img.shields.io/badge/python-v3.9-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## System Architecture

DeepTrade AI combines three powerful components to provide comprehensive stock market analysis and trading:

1. **Advanced Stock Prediction Engine**: LSTM-XGBoost ensemble for multi-timeframe price predictions
2. **Multi-Source Sentiment Analysis**: Real-time sentiment processing from news, Reddit, and SEC filings
3. **Automated Trading System**: Paper trading simulation with Tradier integration

### Technical Stack
- **Deep Learning**: PyTorch with CUDA acceleration
- **Machine Learning**: XGBoost, Scikit-learn
- **NLP**: FinBERT for financial text analysis
- **APIs**: Tradier, News API, Reddit API
- **Data Processing**: Pandas, NumPy, SciPy

## 1. Stock Prediction System

### Model Architecture
The prediction system employs an ensemble approach combining:

- **Enhanced LSTM Model**:
  - Bidirectional LSTM with attention mechanism
  - Multi-head attention for sequence processing
  - Batch normalization and residual connections
  - Dynamic dropout rates for optimal regularization

- **XGBoost Model**:
  - Gradient boosting for feature-based predictions
  - Advanced feature engineering
  - Dynamic weighting based on performance

![Stock Predictions](visualization/stock_prediction_collage.png)
*Multi-timeframe predictions across different stocks showing price action and confidence intervals*

### Training Process
The model training pipeline includes:

- **Feature Engineering**:
  - 15 technical indicators (RSI, MACD, Bollinger Bands, etc.)
  - 12 long-term trend indicators
  - 12 sentiment-based features

- **Training Strategy**:
  - Time-series cross-validation
  - Dynamic learning rate scheduling
  - Early stopping with patience
  - Model performance tracking

![Training History](visualization/stock_training_history_collage.png)
*Training convergence showing loss metrics and validation performance*

## 2. Sentiment Analysis Pipeline

### Multi-Source Integration
- **Financial News Processing**: Real-time analysis of market news (40%)
- **Reddit Sentiment**: Analysis of market-related subreddits (30%)
- **SEC Filing Analysis**: Automated processing of financial documents (30%)

### Sentiment Processing
- FinBERT model fine-tuned for financial text
- Real-time sentiment scoring and aggregation
- Advanced feature engineering:
  - Sentiment momentum indicators
  - Volume-weighted sentiment signals
  - Trend strength analysis

![Sentiment Analysis](visualization/sentiment_analysis.png)
*Sentiment analysis across different sources and stocks*

## Installation & Usage

### 1. Environment Setup
```bash
conda env create -f environment.yml
conda activate stock_pred
pip install -r requirements.txt
```

### 2. Configuration
```python
# In utils/config.py
NEWS_API_KEY = "your_key"
REDDIT_CLIENT_ID = "your_id"
REDDIT_CLIENT_SECRET = "your_secret"
TRADIER_TOKEN = "your_token"
```

### 3. Stock Selection
Define symbols in your prediction script:
```python
symbols = ['AAPL', 'NVDA', 'MSFT', 'AMD', 'GME', 'JNJ']
predictor = PredictionSystem()
predictions = predictor.predict_timeframe(symbol='AAPL', timeframe='15min')
```

## Performance Metrics

- **Price Prediction**:
  - Directional Accuracy: 82.76% (1-hour timeframe)
  - Mean Absolute Error: 0.73% (5-min predictions)
  - Confidence Scoring: 87-93%

- **Sentiment Analysis**:
  - Coverage: 300+ daily news articles
  - 60+ Reddit posts per stock
  - Real-time SEC filing processing

## License & Attribution

This project is licensed under the MIT License. Special thanks to:
- ProsusAI for the FinBERT model
- Reddit API for market sentiment data
- NewsAPI for financial news access

## Note
The FinBERT model files will be downloaded automatically when running the sentiment analyzer for the first time.