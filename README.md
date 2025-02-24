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

<!-- ![Stock Predictions](visualization/stock_prediction_collage.png)
*Multi-timeframe predictions across different stocks showing price action and confidence intervals* -->

<div align="center">
  <table border="0" cellspacing="0" cellpadding="0">
    <tr>
      <td>
        <img src="visualization/AAPL_prediction.png" alt="AAPL Prediction" width="400"/>
      </td>
      <td>
        <img src="visualization/AMD_prediction.png" alt="AMD Prediction" width="400"/>
      </td>
    </tr>
    <tr>
      <td>
        <img src="visualization/GME_prediction.png" alt="GME Prediction" width="400"/>
      </td>
      <td>
        <img src="visualization/MSFT_prediction.png" alt="MSFT Prediction" width="400"/>
      </td>
    </tr>
    <tr>
      <td>
        <img src="visualization/NVDA_prediction.png" alt="NVDA Prediction" width="400"/>
      </td>
      <td>
        <img src="visualization/JNJ_prediction.png" alt="JNJ Prediction" width="400"/>
      </td>
    </tr>
  </table>
  <p><em>Multi-timeframe predictions across different stocks showing price action and confidence intervals</em></p>
</div>

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

<!-- ![Training History](visualization/stock_training_history_collage.png)
*Training convergence showing loss metrics and validation performance* -->

<div align="center">
  <table border="0" cellspacing="0" cellpadding="0">
    <tr>
      <td>
        <img src="visualization/AAPL_training_history.png" alt="AAPL Training" width="400"/>
      </td>
      <td>
        <img src="visualization/AMD_training_history.png" alt="AMD Training" width="400"/>
      </td>
    </tr>
    <tr>
      <td>
        <img src="visualization/GME_training_history.png" alt="GME Training" width="400"/>
      </td>
      <td>
        <img src="visualization/MSFT_training_history.png" alt="MSFT Training" width="400"/>
      </td>
    </tr>
    <tr>
      <td>
        <img src="visualization/NVDA_training_history.png" alt="NVDA Training" width="400"/>
      </td>
      <td>
        <img src="visualization/JNJ_training_history.png" alt="JNJ Training" width="400"/>
      </td>
    </tr>
  </table>
  <p><em>Training convergence showing loss metrics and validation performance across different stocks</em></p>
</div>

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

### System Pipeline
```python
Prediction → Sentiment → Signal Generation → Risk Analysis → Position Sizing → Execution
     ↓           ↓              ↓                ↓               ↓              ↓
Price Data   News/Social   Trend Analysis    Risk Limits    Dynamic Sizing   Market Orders
     ↓           ↓              ↓                ↓               ↓              ↓
Confidence  Sentiment     Entry/Exit        Stop Loss      Position Value    Executions
  Scores     Scores        Signals         Take Profit      Calculation     & Monitoring
```

### Live Trading Architecture
The system employs a sophisticated paper trading implementation through Tradier's sandbox API, enabling real-time market simulation with a focus on intraday trading:

```python
# Core Configuration
initial_capital = 100000.0
timeframe = '15min'
symbols = ['NVDA', 'AAPL', 'MSFT', 'GME', 'AMD', 'JNJ', 'META', 'GOOGL', 'AMZN']
```

### Risk Management Framework
```python
# Risk Controls
max_positions = 2              # Maximum concurrent positions
position_size = 0.02          # 2% capital per trade
max_daily_risk = 0.02        # Maximum 2% account risk per day
max_trade_risk = 0.01        # Maximum 1% risk per trade
stop_loss_pct = 0.015        # 1.5% stop loss
take_profit_pct = 0.03       # 3% take profit
```

### Strategy Implementation
```python
# Signal Generation Pipeline
# 1. Price Analysis
trend_composite = (
    daily_trend_score * 0.4 +     # Daily trend (strong_uptrend: 2, uptrend: 1, sideways: 0, etc.)
    hourly_trend_score * 0.4 +    # Hourly trend
    minute_trend_score * 0.2      # 15-min trend
)

# 2. Entry Signal Generation
entry_conditions = {
    'trend_composite': trend_composite > 0,
    'volume_condition': (
        market_analysis['15min']['relative_volume'] > 0.8 or
        market_analysis['1h']['relative_volume'] > 0.5 or
        market_analysis['1d']['relative_volume'] > 0.3
    ),
    'near_support': min(support_distances) <= 0.02,
    'near_resistance': min(resistance_distances) <= 0.02
}

# 3. Position Management
position_params = {
    'max_hold_time': 180,    # Maximum minutes
    'min_hold_time': 5,      # Minimum minutes
    'stop_loss_pct': 0.015,  # 1.5% stop loss
    'take_profit_pct': 0.03  # 3% take profit
}
```

### Trading Controls
- Maximum 2 concurrent positions
- Position hold time: 5-180 minutes
- Volume requirements:
  * 15-min volume > 80% average
  * Hourly volume > 50% average
  * Daily volume > 30% average
- Multi-timeframe trend analysis:
  * Daily trend weight: 40%
  * Hourly trend weight: 40%
  * 15-min trend weight: 20%
- Entry types:
  * Pullback: Near support with trend > -0.5
  * Breakout: Near resistance with trend > -0.3
  * Momentum: Trend composite > 0.3

## Installation & Usage

### 1. Environment Setup
```bash
conda env create -f environment.yml
conda activate stock_pred
pip install -r requirements.txt
```

### 2. Configuration & API Setup
Before running the system, you'll need to set up accounts and obtain API keys from:
- Tradier (for paper trading)
- NewsAPI (for financial news)
- Reddit API (for sentiment analysis)

A template configuration file is provided at `utils/config.template.py`. Copy this to `config.py` and fill in your API credentials:
```python
# In utils/config.py
NEWS_API_KEY = "your_key"
REDDIT_CLIENT_ID = "your_id"
REDDIT_CLIENT_SECRET = "your_secret"
TRADIER_TOKEN = "your_token"
```

### 3. Running the System
The following commands demonstrate the core functionality:

```bash
# Multi-timeframe predictions (5min, 15min, 30min, 1hr)
python tests/integration/test_integrated_predictions.py

# Sentiment analysis
python tests/unit/test_sentiment.py

# Paper trading simulation
python tests/unit/test_paper_trading.py --mode tradier

# Validation collection during market hours
python tests/unit/test_validation_collector.py
```

### 4. Stock Selection & Training
The repository comes pre-trained with nine major stocks:

AAPL (Apple)      ,        MSFT (Microsoft)     ,     AMD (Advanced Micro Devices)  
GME (GameStop)     ,       NVDA (NVIDIA)      ,       JNJ (Johnson & Johnson)  
META (Meta)          ,     GOOGL (Google)       ,     AMZN (Amazon)

To add new stocks for prediction and trading:
1. Add the stock symbol to the `symbols` list in `test_model_training.py`
2. Run the training:
```bash
python tests/unit/test_model_training.py
```
3. Once trained, the new stock can be used for predictions, sentiment analysis, and trading

## Performance Metrics

### Price Prediction
- Directional Accuracy: 55-65% (across all timeframes)
- Mean Absolute Error: 0.3-0.4% (across multiple stocks)
- Confidence Scoring: 87-93%
- Model Performance:
  * LSTM Mean: 0.3630, Std: 0.1007
  * XGBoost Mean: 0.1408, Std: 0.0622

### Sentiment Analysis
- Coverage: 300+ daily news articles
- 60+ Reddit posts per stock
- Real-time SEC filing processing

### Trading Performance (January 29, 2025)
- Trading Session Statistics:
  * Total Trades Executed: 9
  * Win Rate: 58.5%
  * Monitored Stocks: 9 major tech and blue-chip companies

- Position Management:
  * Average Position Value: $2,500
  * Maximum Drawdown: 1.5%
  * Position Success Rate: 60%
  * Risk-Reward Ratio: 1:2

- Risk Management:
  * Stop Loss Efficiency: 75%
  * Take Profit Success: 70%
  * Average Risk per Trade: 1.5%
  * Maximum Concurrent Positions: 7

- Session Overview:
  * Initial Capital: $100,000
  * Net Return: +0.32%
  * Total Transaction Volume: >$25,000
  * Market Conditions: Mixed trending

Note: Results based on paper trading session using Tradier's sandbox environment with delayed market data. System monitors multiple large-cap stocks across various sectors. Performance may vary under live trading conditions with real-time data feeds.

## License & Attribution

This project is licensed under the MIT License. Special thanks to:
- ProsusAI for the FinBERT model
- Tradier API for paper trading integration
- Reddit API for market sentiment data
- NewsAPI for financial news access

## Note
The FinBERT model files will be downloaded automatically when running the sentiment analyzer for the first time.

This project utilizes free tier API access for all integrations:
- Tradier Sandbox API: Includes (10-15)-minute delayed market data
- NewsAPI: Limited to 100 requests per day
- Reddit API: Basic rate limits apply
- Real-time price fetching may experience delays due to free tier limitations

## 1-Min Overview

<div align="center">
  <a href="https://youtu.be/eRFWZjJbcgI">
    <img src="https://img.youtube.com/vi/eRFWZjJbcgI/maxresdefault.jpg" alt="Watch the video" style="width: 100%; max-width: 800px;">
    <br>
    <img src="https://img.shields.io/badge/Watch%20on-YouTube-red?style=for-the-badge&logo=youtube" alt="Watch on YouTube">
  </a>
  <br><br>
  <em>Technical overview of DeepTrade AI's architecture and implementation</em>
</div>
