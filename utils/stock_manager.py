import numpy as np
import pandas as pd
from transformers import pipeline
import yfinance as yf
import requests
from typing import Dict, Tuple, List, Any
from datetime import datetime, timedelta
from utils.config import NEWS_API_KEY, USE_TRADIER_SANDBOX
from utils.sentiment_analyzer import FinancialSentimentAnalyzer, get_financial_news
from utils.sentiment_manager import SentimentDataManager  # Add this too
from utils.reddit_sentiment import EnhancedRedditAnalyzer
from utils.sec_data_collector import SECDataCollector
from utils.config import TRADIER_CONFIG
from pytz import timezone
from utils.config import (
    ALPHA_VANTAGE_KEY,
    FINNHUB_KEY,
    POLYGON_KEY,
    TWELVEDATA_KEY,
    NEWS_API_KEY,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    API_LIMITS,
    API_ENDPOINTS
)

class StockManager:
    def __init__(self):
        self.et_tz = timezone('US/Eastern')
        self.models = {}
        self.sentiment_analyzer = FinancialSentimentAnalyzer(verbose=False)
        self.sentiment_manager = SentimentDataManager()
        self.cached_data = {}
        
        # Comprehensive sentiment cache
        self.sentiment_cache = {
            'news': {},
            'reddit': {},
            'sec': {},
            'combined': {}  # Final combined scores
        }

        # Initialize API keys from config
        self.alpha_vantage_key = ALPHA_VANTAGE_KEY
        self.finnhub_key = FINNHUB_KEY
        self.polygon_key = POLYGON_KEY
        self.twelvedata_key = TWELVEDATA_KEY
        self.tradier_config = TRADIER_CONFIG['sandbox' if USE_TRADIER_SANDBOX else 'production']
        self.tradier_token = self.tradier_config['token']
        self.tradier_endpoint = self.tradier_config['endpoint']
        
        # Price caching with TTL
        self.price_cache = {}
        self.cache_ttl = 60 
        
        # Define all feature sets
        self.feature_columns = {
            'technical': [
                'RSI', 'MACD', 'SMA_20', '%K', '%D', 'ROC', 'MFI',
                'BB_upper', 'BB_lower', 'ATR', 'OBV',
                'Signal_Line', 'BB_middle', 'Volume_MA', 'Volume_Ratio'
            ],
            'long_term': [
                'SMA_50', 'SMA_100', 'SMA_200', 
                'MA_Cross_50_200', 'MA_Cross_20_50',
                'ROC_20', 'ROC_60', 'ADX', 
                'ATR_20', 'BB_Width',
                'Volume_SMA_50', 'Volume_Trend'
            ],
            'sentiment': [
                'raw_sentiment', 'sentiment_3d', 'sentiment_7d',
                'sentiment_momentum', 'sentiment_volume', 
                'sentiment_acceleration', 'sentiment_regime',
                'sentiment_trend_strength', 'sent_price_momentum', 
                'sentiment_trend', 'sentiment_rsi', 'sentiment_volatility'
            ]
        }
        
    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = '1d') -> pd.DataFrame:
        """Fetch stock data with caching mechanism"""
        cache_key = f"{symbol}_{start_date}_{end_date}_{timeframe}"
        
        try:
            if cache_key not in self.cached_data:
                interval = {
                    '1min': '1m',
                    '5min': '5m',
                    '15min': '15m',
                    '30min': '30m',
                    '1h': '1h',
                    '1d': '1d'
                }.get(timeframe, '1d')
                
                data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
                
                # Fill missing volume with previous day's volume
                if 'Volume' in data.columns:
                    data['Volume'] = data['Volume'].replace(0, method='ffill')
                
                self.cached_data[cache_key] = data
                
            return self.cached_data[cache_key]
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        # Existing indicators
        data = self.add_basic_indicators(data)
        
        # Stochastic Oscillator
        high_14 = data['High'].rolling(window=14).max()
        low_14 = data['Low'].rolling(window=14).min()
        data['%K'] = (data['Close'] - low_14) / (high_14 - low_14) * 100
        data['%D'] = data['%K'].rolling(window=3).mean()
        
        # On-Balance Volume (OBV)
        data['OBV'] = (data['Volume'] * (~data['Close'].diff().le(0) * 2 - 1)).cumsum()
        
        # Price Rate of Change (ROC)
        data['ROC'] = data['Close'].pct_change(periods=12) * 100
        
        # Additional Momentum Indicators
        # Money Flow Index
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        pos_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
        neg_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
        data['MFI'] = 100 - (100 / (1 + pos_flow / neg_flow))
        
        return data

    def prepare_scaled_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare scaled features for prediction"""
        from sklearn.preprocessing import RobustScaler, MinMaxScaler

        tech_columns = [
            'RSI', 'MACD', 'SMA_20', '%K', '%D', 'ROC', 'MFI',
            'BB_upper', 'BB_lower', 'ATR', 'OBV',
            'Signal_Line', 'BB_middle', 'Volume_MA', 'Volume_Ratio'
        ]
        
        sent_columns = [
            'raw_sentiment', 'sentiment_3d', 'sentiment_7d',
            'sentiment_momentum', 'sentiment_volume', 
            'sentiment_acceleration', 'sentiment_regime',
            'sentiment_trend_strength', 
            'sent_price_momentum', 'sentiment_trend',
            'sentiment_rsi', 'sentiment_volatility'  # Adding these two
        ]

        tech_features = data[self.feature_columns['technical']].values
    
        # Long-term features
        long_term_columns = [
            'SMA_50', 'SMA_100', 'SMA_200', 
            'MA_Cross_50_200', 'MA_Cross_20_50',
            'ROC_20', 'ROC_60', 'ADX', 
            'ATR_20', 'BB_Width',
            'Volume_SMA_50', 'Volume_Trend'
        ]
        long_term_features = data[long_term_columns].values
        
        # Sentiment features
        sent_features = data[self.feature_columns['sentiment']].values
        
        # Scale features
        tech_scaler = RobustScaler()
        sent_scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Scale each feature set
        X_tech = tech_scaler.fit_transform(tech_features)
        X_long = tech_scaler.fit_transform(long_term_features)
        X_sent = sent_scaler.fit_transform(sent_features)
        
        # Print feature counts for verification
        print("\nFeature counts:")
        print(f"Technical features: {X_tech.shape[1]}")
        print(f"Long-term features: {X_long.shape[1]}")
        print(f"Sentiment features: {X_sent.shape[1]}")
        print(f"Total features: {X_tech.shape[1] + X_long.shape[1] + X_sent.shape[1]}")
        
        # Stack all features
        return np.hstack([X_tech, X_long, X_sent])
    
    def verify_all_features_present(self, data: pd.DataFrame) -> bool:
        """Verify all required features exist in data"""
        missing_tech = [col for col in self.feature_columns['technical'] 
                    if col not in data.columns]
        missing_long = [col for col in self.long_term_columns 
                    if col not in data.columns]
        missing_sent = [col for col in self.feature_columns['sentiment'] 
                    if col not in data.columns]
        
        if missing_tech or missing_long or missing_sent:
            print("Missing features:")
            if missing_tech: print("Technical:", missing_tech)
            if missing_long: print("Long-term:", missing_long)
            if missing_sent: print("Sentiment:", missing_sent)
            return False
        return True
    
    def verify_features(self, data: pd.DataFrame) -> None:
        required_features = (
            self.feature_columns['technical'] +
            self.feature_columns['long_term'] +
            self.feature_columns['sentiment']
        )
        
        print("\nVerifying features:")
        print("Expected features:", len(required_features))
        print("Current features:", len([col for col in required_features if col in data.columns]))
        
        missing = [col for col in required_features if col not in data.columns]
        if missing:
            print("WARNING: Missing features:", missing)
            
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model training"""
        data = self.add_technical_indicators(data)  # This adds all technical indicators
        data = self.prepare_sentiment_features(data)
        data = self.add_sentiment_indicators(data)
        data = self.enhance_sentiment_features(data)
        return data.dropna()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        return exp1 - exp2
    
    def get_combined_sentiment(self, symbol: str) -> Dict:
        """Get combined sentiment from all sources with caching"""
        # Check if we have cached combined sentiment
        if symbol in self.sentiment_cache['combined']:
            print(f"Using cached sentiment data for {symbol}")
            return self.sentiment_cache['combined'][symbol]
            
        sentiment_data = {}
        
        # 1. News Sentiment
        news_sentiment = self.analyze_sentiment(symbol) * 0.4
        sentiment_data['news_score'] = news_sentiment
        
        # 2. Reddit Sentiment
        try:
            reddit_analyzer = EnhancedRedditAnalyzer(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT
            )
            reddit_df = reddit_analyzer.analyze_stock_sentiment(symbol)
            reddit_sentiment = reddit_df['total_sentiment'].mean() * 0.3 if reddit_df is not None else 0
            sentiment_data['reddit_score'] = reddit_sentiment
        except Exception as e:
            print(f"Error in Reddit analysis: {e}")
            sentiment_data['reddit_score'] = 0
            
        # 3. SEC Sentiment
        try:
            sec_collector = SECDataCollector()
            sec_analysis = sec_collector.analyze_filings(symbol)
            sec_sentiment = sec_analysis['sentiment_score'] * 0.3
            sentiment_data['sec_score'] = sec_sentiment
        except Exception as e:
            print(f"Error in SEC analysis: {e}")
            sentiment_data['sec_score'] = 0
            
        # Calculate combined score
        combined_score = (
            sentiment_data['news_score'] +
            sentiment_data['reddit_score'] +
            sentiment_data['sec_score']
        )
        
        sentiment_data['combined_score'] = combined_score
        
        # Cache the results
        self.sentiment_cache['combined'][symbol] = sentiment_data
        
        return sentiment_data

   
    def analyze_sentiment(self, symbol: str, days: int = 7) -> float:
        """Analyze recent news sentiment for a stock"""
        try:
            # Check cache first
            if symbol in self.sentiment_cache:
                print(f"Using cached sentiment for {symbol}")
                return self.sentiment_cache[symbol]
                
            # If not in cache, get new sentiment
            headlines = get_financial_news(symbol, days, api_key=NEWS_API_KEY)
            
            if not headlines:
                print("No headlines found for sentiment analysis")
                return 0.0
        
            
            # Analyze headlines for sentiment
            sentiment_score = self.sentiment_analyzer.analyze_multiple(headlines)
            
            # Store the sentiment data
            if sentiment_score != 0.0:
                print(f"Got sentiment score: {sentiment_score:.2f}")
                print(f"Processed {len(headlines)} headlines")
                self.sentiment_manager.store_daily_sentiment(symbol, sentiment_score, headlines)
                
                # Print sample headlines and their individual scores for verification
                print("\nSample Headlines Analysis:")
                for headline in headlines[:3]:  # Show first 3 headlines
                    score = self.sentiment_analyzer.analyze_text(headline)
                    print(f"Headline: {headline[:100]}...")
                    print(f"Score: {score:.2f}\n")

            self.sentiment_cache[symbol] = sentiment_score
            return sentiment_score
            
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return 0.0
            
    def _get_recent_headlines(self, symbol: str, days: int) -> List[str]:
        """This method should use get_financial_news instead of implementing its own logic"""
        return get_financial_news(symbol, days, api_key=NEWS_API_KEY)
        
    def _get_recent_headlines(self, symbol: str, days: int) -> List[str]:
        url = f'https://newsapi.org/v2/everything'
        params = {
            'q': symbol,
            'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'to': datetime.now().strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'popularity',
            'apiKey': NEWS_API_KEY
        }
        
        try:
            response = requests.get(url, params=params)
            news_data = response.json()
            
            if response.status_code != 200:
                print(f"API Error: {news_data.get('message', 'Unknown error')}")
                return []
                
            return [article['title'] for article in news_data.get('articles', [])]
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []


    def update_model(self, symbol: str, model):
        """Update or add a new model for a symbol"""
        self.models[symbol] = model

    def add_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators"""
        # Bollinger Bands
        data['BB_middle'] = data['Close'].rolling(window=20).mean()
        data['BB_upper'] = data['BB_middle'] + 2 * data['Close'].rolling(window=20).std()
        data['BB_lower'] = data['BB_middle'] - 2 * data['Close'].rolling(window=20).std()
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        data['RSI'] = self.calculate_rsi(data['Close'])
        
        # ATR (Average True Range)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        data['ATR'] = true_range.rolling(14).mean()
        
        return data    
    
    # Update this method in stock_manager.py

    def prepare_sentiment_features(self, data: pd.DataFrame, symbol: str, 
                     use_cached_sentiment: bool = False, 
                     cached_sentiment: Dict = None) -> pd.DataFrame:
        data = data.copy()
        
        sentiment_data = self.get_combined_sentiment(symbol)
        combined_sentiment = sentiment_data['combined_score']
        
        # Reduce noise magnitude
        noise = np.random.normal(0, 0.005, len(data))  # Reduced from 0.01
        data['raw_sentiment'] = combined_sentiment + noise
        
        # Add smoothing
        data['raw_sentiment'] = data['raw_sentiment'].rolling(window=3, min_periods=1).mean()
        
        # Calculate sentiment momentum and other features first
        data['sentiment_3d'] = data['raw_sentiment'].rolling(window=3, min_periods=1).mean()
        data['sentiment_7d'] = data['raw_sentiment'].rolling(window=7, min_periods=1).mean()
        data['sentiment_momentum'] = data['sentiment_3d'] - data['sentiment_7d']
        
        # Additional sentiment features
        data['sentiment_volume'] = data['raw_sentiment'] * data['Volume'].pct_change().fillna(0)
        data['sentiment_acceleration'] = data['sentiment_momentum'].diff().fillna(0)
        data['sentiment_regime'] = data['raw_sentiment'].rolling(window=5).std().fillna(0)
        data['sentiment_trend_strength'] = abs(data['raw_sentiment']).rolling(window=5).mean().fillna(0)
        data['sent_price_momentum'] = data['raw_sentiment'] * data['Close'].pct_change().fillna(0)
        data['sentiment_trend'] = data['raw_sentiment'].rolling(window=10).mean().fillna(0)
        data['sentiment_rsi'] = 50 + (data['raw_sentiment'] * 50)
        data['sentiment_volatility'] = data['raw_sentiment'].rolling(window=7).std().fillna(0)
        
        # Clip sentiment values
        for col in data.columns:
            if 'sentiment' in col:
                data[col] = np.clip(data[col], -1, 1)
        
        return data

    def add_sentiment_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create indicators that combine sentiment with price action"""
        if 'raw_sentiment' in data.columns:
            data['sent_price_momentum'] = data['raw_sentiment'] * data['Close'].pct_change().fillna(0)
            data['sentiment_rsi'] = 50 + (data['raw_sentiment'] * 50)  # Scale sentiment to RSI-like range
            data['sentiment_trend'] = data['raw_sentiment']  # Start with basic trend
            
            # Fill any NaN values
            for col in ['sent_price_momentum', 'sentiment_rsi', 'sentiment_trend']:
                data[col] = data[col].fillna(method='ffill').fillna(0)
        
        return data

    def prepare_prediction_data(self, symbol: str, use_cached_sentiment: bool = False, 
                          sentiment_data: Dict = None) -> Tuple[pd.DataFrame, float]:
        """Prepare prediction data with all necessary features"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)  # 3 years of data
        
        # Fetch historical data
        data = self.fetch_stock_data(symbol, start_date.strftime('%Y-%m-%d'), 
                                end_date.strftime('%Y-%m-%d'))
        
        # Add technical indicators
        data = self.add_technical_indicators(data)
        data = self.add_long_term_indicators(data)
        
        # Add sentiment features with correct parameter names
        if use_cached_sentiment and sentiment_data:
            data = self.prepare_sentiment_features(
                data=data, 
                symbol=symbol,
                use_cached_sentiment=use_cached_sentiment,  # Fixed parameter name
                cached_sentiment=sentiment_data
            )
        else:
            data = self.prepare_sentiment_features(
                data=data, 
                symbol=symbol
            )
        
        data = self.add_sentiment_indicators(data)
        data = self.enhance_sentiment_features(data)
        
        # Get current sentiment
        current_sentiment = data['raw_sentiment'].iloc[-1] if 'raw_sentiment' in data.columns else 0.0

        return data.dropna(), current_sentiment

    def enhance_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add balanced sentiment indicators with controlled scaling"""
        # Cap the volume changes to prevent extreme values
        volume_change = data['Volume'].pct_change().clip(-1, 1)  # Limit to ±100%
        data['sentiment_volume'] = data['raw_sentiment'] * np.log1p(np.abs(volume_change)) * np.sign(volume_change)
        
        # Use rolling changes instead of point-to-point for smoother acceleration
        data['sentiment_acceleration'] = data['sentiment_momentum'].diff().rolling(3).mean()
        
        # Scale regime changes relative to the mean
        sentiment_std = data['sentiment_rsi'].rolling(5).std()
        data['sentiment_regime'] = sentiment_std / sentiment_std.rolling(20).mean().fillna(1)
        
        # Use absolute changes for trend strength
        data['sentiment_trend_strength'] = data['sentiment_trend'].abs().rolling(5).mean()
        
        return data

    def add_long_term_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add longer timeframe technical indicators"""
        # Longer-term moving averages
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_100'] = data['Close'].rolling(window=100).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # Moving average crossovers
        data['MA_Cross_50_200'] = data['SMA_50'] - data['SMA_200']
        data['MA_Cross_20_50'] = data['SMA_20'] - data['SMA_50']
        
        # Longer-term momentum
        data['ROC_20'] = data['Close'].pct_change(periods=20) * 100
        data['ROC_60'] = data['Close'].pct_change(periods=60) * 100
        
        # Trend strength indicators
        data['ADX'] = self.calculate_adx(data)
        
        # Volatility measures
        data['ATR_20'] = self.calculate_atr(data, window=20)
        data['BB_Width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
        
        # Volume trends
        data['Volume_SMA_50'] = data['Volume'].rolling(window=50).mean()
        data['Volume_Trend'] = data['Volume'] / data['Volume_SMA_50']
        
        return data

    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Calculate Plus and Minus Directional Movement
        pdm = (high - high.shift()).clip(lower=0)
        ndm = (low.shift() - low).clip(lower=0)
        
        # Smooth DM values
        pdm_smooth = pdm.rolling(period).mean()
        ndm_smooth = ndm.rolling(period).mean()
        
        # Calculate Plus and Minus Directional Indicators
        pdi = 100 * pdm_smooth / atr
        ndi = 100 * ndm_smooth / atr
        
        # Calculate Directional Movement Index
        dx = 100 * abs(pdi - ndi) / (pdi + ndi)
        
        # Calculate ADX
        adx = dx.rolling(period).mean()
        
        return adx

    def calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range for specified window"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(window).mean()
    
    def get_real_time_price(self, symbol: str) -> Dict[str, Any]:
        """Get real-time price with enhanced validation"""
        try:
            print(f"\nFetching price for {symbol}...")
            headers = {
                'Authorization': f'Bearer {self.tradier_token}',
                'Accept': 'application/json'
            }
            
            response = requests.get(
                f'{self.tradier_endpoint}/markets/quotes',
                params={'symbols': symbol, 'greeks': 'false'},
                headers=headers
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'quotes' in data and 'quote' in data['quotes']:
                    quote = data['quotes']['quote']
                    
                    # Get the best available price
                    last_price = quote.get('last')
                    if last_price is None or float(last_price) == 0:
                        last_price = quote.get('close')
                    
                    if last_price and float(last_price) > 0:
                        now = datetime.now(self.et_tz)
                        market_hours = self._is_market_hours()  # No parameter needed
                        
                        return {
                            'price': float(last_price),
                            'bid': quote.get('bid'),
                            'ask': quote.get('ask'),
                            'volume': quote.get('volume'),
                            'timestamp': now,
                            'source': 'tradier',
                            'market_hours': market_hours,
                            'market_close': quote.get('close'),
                            'market_close_time': now.replace(hour=16, minute=0, second=0, microsecond=0)
                        }
                    else:
                        print(f"Invalid price data for {symbol}")
                else:
                    print(f"Invalid quote data structure for {symbol}")
            else:
                print(f"Error response from Tradier: {response.text}")
                
            return None
                
        except Exception as e:
            print(f"Error fetching price for {symbol}: {str(e)}")
            return None

    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status from Tradier"""
        try:
            headers = {
                'Authorization': f'Bearer {self.tradier_token}',
                'Accept': 'application/json'
            }
            
            response = requests.get(
                f'{self.tradier_endpoint}/markets/clock',
                headers=headers,
                timeout=2
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'is_open': data['clock']['state'] == 'open',
                    'next_open': data['clock'].get('next_open'),
                    'next_close': data['clock'].get('next_close'),
                    'timestamp': datetime.now(self.et_tz)
                }
        except Exception as e:
            print(f"Error getting market status: {str(e)}")
            
        # Fallback to basic check
        return {'is_open': self._is_market_hours()}
    
    def _is_market_hours(self) -> bool:
        """Check if current time is during market hours"""
        if not hasattr(self, 'et_tz'):
            self.et_tz = timezone('US/Eastern')
            
        now = datetime.now(self.et_tz)
        current_time = now.time()
        
        # Market hours: 9:30 AM - 4:00 PM Eastern Time
        market_open = datetime.strptime('9:30', '%H:%M').time()
        market_close = datetime.strptime('16:00', '%H:%M').time()
        
        # Check if it's a weekday and within market hours
        return (
            now.weekday() < 5 and
            market_open <= current_time <= market_close
        )
    
    def verify_price_data(self, price_data: Dict) -> bool:
        """Verify price data is recent enough"""
        if not price_data or 'timestamp' not in price_data:
            return False
            
        now = datetime.now(self.et_tz)
        price_time = price_data['timestamp']
        
        # Convert naive timestamp to aware if needed
        if price_time.tzinfo is None:
            price_time = self.et_tz.localize(price_time)
            
        return (now - price_time).total_seconds() < 300  # 5 minutes