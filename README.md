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

- **Model Performance Metrics**:
  - LSTM Mean: 0.3630, Std: 0.1007
  - XGBoost Mean: 0.1408, Std: 0.0622
  - Average Training Length: 60-120 epochs
  - Convergence monitored through validation loss
  - Early stopping with 30-epoch patience
  - Cross-validation with 5 folds

The metrics represent Mean Absolute Error (MAE) on normalized returns. Model training uses a sliding window approach with dynamic batch sizes and learning rate adjustment based on validation performance.

![Training History](visualization/stock_training_history_collage.png)
*Training convergence showing loss metrics and validation performance*

## 2. Sentiment Analysis Pipeline

### Multi-Source Integration

- **Financial News Processing (40%)**
  - Real-time streaming with NewsAPI integration
  - Automated headline + description analysis
  - Relevancy-based filtering and aggregation
  - Intelligent caching with TTL management

- **Reddit Sentiment Analysis (30%)**
  - Multi-subreddit monitoring (r/wallstreetbets, r/stocks, r/investing)
  - Advanced engagement metrics (upvote ratio, comment sentiment)
  - Company name variant matching
  - Post-comment sentiment weighting system

- **SEC Filing Analysis (30%)**
  - Real-time CIK tracking and validation
  - Form-specific sentiment weighting (10-K, 10-Q, 8-K prioritization)
  - Automated filing pattern analysis
  - Temporal decay weighting for recent filings

### FinBERT Model Architecture
- **Model**: ProsusAI/finbert (fine-tuned BERT for finance)
- **Processing**:
  - Token truncation at 512 length
  - Three-class classification (positive/neutral/negative)
  - Softmax probability distribution
  - Custom sentiment score calculation

### Feature Engineering
- **Temporal Features**:
  - 3-day and 7-day moving averages
  - Sentiment momentum calculation
  - Volatility regime detection
  - Trend acceleration metrics

- **Market Integration**:
  - Volume-weighted sentiment signals
  - Price-sentiment correlation metrics
  - Trading volume impact analysis
  - Cross-source sentiment validation

### Real-time Processing
- **Data Pipeline**:
  - Asynchronous source aggregation
  - Intelligent caching system
  - Rate limit management
  - Failure recovery mechanisms

![Sentiment Analysis](visualization/sentiment_analysis.png)
*Sentiment analysis across different sources and stocks*


## 3. Automated Trading System

### Live Trading Architecture
The system employs a sophisticated paper trading implementation through Tradier's sandbox API, enabling real-time market simulation and automated execution.

- **Real-time Execution**:
  - Market hours trading (9:30 AM - 4:00 PM ET)
  - Real-time order execution and tracking
  - Live position monitoring
  - Automated P&L calculation

### Risk Management Framework
- **Position Controls**:
  - Maximum 2 concurrent positions
  - 2% capital allocation per trade
  - 1.5% stop loss implementation
  - 3% take profit targets
  - 5-180 minutes holding time limits

- **Market Risk Management**:
  - Real-time price monitoring
  - Automated stop loss/take profit
  - Dynamic position sizing
  - Market hours only trading
  - Minimum time between trades: 60 minutes

### Trading Strategy Implementation
- **Entry Logic**:
  - Multi-timeframe trend analysis
  - Volume profile validation
  - Support/Resistance levels
  - Real-time sentiment integration
  - Minimum 90% confidence threshold

- **Exit Conditions**:
  - Price-based exits (stop/target)
  - Time-based exits (max hold time)
  - Trailing stop implementation
  - Market condition filters

### Performance Summary
```
Trading Metrics (January 2025):
- Initial Capital : $100,000.00
- Final Capital  : $100,204.98
- Net P&L       : $204.98 
- Return        : 0.20%
- Total Trades  : 2
- Win Rate      : 50.0%
- Avg Hold Time : ~180 minutes
```

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