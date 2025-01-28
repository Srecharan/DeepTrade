# DeepTrade AI: Multi-Model Stock Prediction with NLP & Automated Trading

![Python](https://img.shields.io/badge/python-v3.9-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview

DeepTrade AI is a cutting-edge stock prediction and automated trading system that combines deep learning, advanced NLP, and real-time sentiment analysis. The system leverages state-of-the-art models including LSTM neural networks, XGBoost, and sophisticated ensemble methods to provide accurate market predictions while incorporating sentiment signals from multiple sources.

## Key Features

### Stock Prediction Engine
- **Multi-Model Architecture**
  - LSTM Neural Networks with attention mechanism
  - XGBoost for feature-based prediction
  - Ensemble methods combining both models
  - Dynamic confidence interval calculations
- **Supported Stocks**
  - Core Tech: AAPL, NVDA, MSFT
  - Growth: AMD, GME
  - Stable: JNJ
- **Prediction Capabilities**
  - Multiple timeframes (5min to 1h)
  - Price direction and magnitude
  - Confidence scores and risk metrics

![Stock Predictions](visualization/stock_prediction_collage.png)
*Prediction results across multiple stocks*

![Training History](visualization/stock_training_history_collage.png)
*Model training convergence and validation*

### Advanced Sentiment Analysis Pipeline

The system employs a sophisticated multi-source sentiment analysis approach:

- **FinBERT Model Integration**: Leverages fine-tuned BERT architecture for financial text analysis
- **Multi-Source Data Fusion**: 
  - Financial News (40%)
  - Reddit Market Sentiment (30%)
  - SEC Filing Analysis (30%)
- **Real-time Processing**: Streaming architecture with intelligent caching
- **Feature Engineering**: Advanced NLP techniques including sentiment momentum indicators and volume-weighted signals

### Sentiment Analysis Results

![Sentiment Analysis Dashboard](visualization/sentiment_analysis.png)

Our sentiment analysis pipeline processes:
- 300+ daily news articles
- 60+ Reddit posts
- Real-time SEC filings
- Coverage of major tech stocks (AAPL, NVDA, MSFT, GOOGL, etc.)
- 87-93% prediction confidence scores

### Stock Prediction Performance

![Price Prediction Example](visualization/AAPL_prediction.png)

The LSTM-XGBoost ensemble model demonstrates strong predictive capabilities:
- High directional accuracy (82.76% for 1-hour timeframe)
- Low mean absolute error (0.73% for 5-min predictions)
- Robust confidence intervals for risk management

### Model Training Metrics

![Training History](visualization/AAPL_training_history.png)

The training process shows consistent improvement:
- Stable convergence in loss metrics
- Effective regularization preventing overfitting
- Strong validation performance

## Technical Architecture

### Model Components
```
DeepTrade/
├── models/
│   ├── finbert/          # FinBERT for sentiment analysis
│   └── trained/          # Stock-specific prediction models
│       ├── LSTM models
│       └── XGBoost models
├── data/
│   ├── sentiment/        # Multi-source sentiment data
│   └── market/          # Historical price & technical data
└── utils/              # Core prediction & trading logic
```

### Core Features
- **Sentiment Analysis**
  - FinBERT-based text processing
  - Multi-source sentiment fusion
  - Real-time sentiment updates
- **Price Prediction**
  - LSTM with attention mechanism
  - XGBoost for feature engineering
  - Ensemble prediction system
- **Automated Trading**
  - Risk management system
  - Paper trading integration
  - Multi-timeframe analysis

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/Srecharan/DeepTrade.git
cd DeepTrade
```

2. Install dependencies:
```bash
conda env create -f environment.yml
conda activate stock_pred
pip install -r requirements.txt
```

3. Configure API keys:
```python
# In utils/config.py
NEWS_API_KEY = "your_key"
REDDIT_CLIENT_ID = "your_id"
REDDIT_CLIENT_SECRET = "your_secret"
```

## Usage Examples

### Running Predictions
```python
from utils.prediction_system import PredictionSystem

predictor = PredictionSystem()
predictions = predictor.predict('AAPL', timeframe='15min')
```

### Paper Trading Simulation
```python
from utils.paper_trading import PaperTradingSimulation

simulator = PaperTradingSimulation(
    symbols=['AAPL', 'NVDA'],
    initial_capital=100000.0
)
results = simulator.run_simulation(duration_minutes=60)
```

## License & Acknowledgments

This project is licensed under the MIT License. Special thanks to:
- ProsusAI for the FinBERT model
- Reddit API for market sentiment data
- NewsAPI for financial news access

## Note
The FinBERT model files are not included in the repository due to size limitations. They will be downloaded automatically when running the sentiment analyzer for the first time.