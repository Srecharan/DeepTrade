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
        """Fetch stock data with caching mechanism and MultiIndex handling"""
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
                print(f"Fetching data for {symbol} from {start_date} to {end_date} (Interval: {interval})")
                data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
                
                # Fix MultiIndex if needed
                if isinstance(data.columns, pd.MultiIndex):
                    print("Converting MultiIndex to single-level columns")
                    # Take the first level of the MultiIndex (Price level)
                    data.columns = [col[0] for col in data.columns]
                
                # Fill missing volume with previous day's volume
                if 'Volume' in data.columns:
                    data['Volume'] = data['Volume'].replace(0, method='ffill')
                
                self.cached_data[cache_key] = data
                
            return self.cached_data[cache_key]
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators with safety checks"""
        data = data.copy()
        
        # Check if Volume column exists, if not add it with zeros
        if 'Volume' not in data.columns:
            print("WARNING: Volume column is missing, adding default values")
            data['Volume'] = 0
        
        # Fill missing or zero volume with a small default value
        data['Volume'] = data['Volume'].replace(0, 1000)
        
        # Add Volume Moving Average with safety check
        try:
            data['Volume_MA'] = data['Volume'].rolling(window=20, min_periods=1).mean()
        except Exception as e:
            print(f"Error calculating Volume_MA: {e}")
            data['Volume_MA'] = data['Volume']
            
        # Safe Volume Ratio calculation
        try:
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA'].replace(0, 1)
        except Exception as e:
            print(f"Error calculating Volume_Ratio: {e}")
            data['Volume_Ratio'] = 1.0
            
        # SMA calculation with safety check
        try:
            data['SMA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
        except Exception as e:
            print(f"Error calculating SMA_20: {e}")
            data['SMA_20'] = data['Close']
        
        # Add basic indicators
        try:
            data = self.add_basic_indicators(data)
        except Exception as e:
            print(f"Error in add_basic_indicators: {e}")
            # Add minimal required indicators if basic indicators fail
            data['RSI'] = 50
            data['MACD'] = 0
            data['Signal_Line'] = 0
            data['BB_upper'] = data['Close'] * 1.02
            data['BB_lower'] = data['Close'] * 0.98
            data['BB_middle'] = data['Close']
            data['ATR'] = 1.0
            data['OBV'] = 0
            
        # Add remaining indicators with safety checks
        try:
            # Stochastic Oscillator
            high_14 = data['High'].rolling(window=14, min_periods=1).max()
            low_14 = data['Low'].rolling(window=14, min_periods=1).min()
            data['%K'] = (data['Close'] - low_14) / (high_14 - low_14 + 0.0001) * 100
            data['%D'] = data['%K'].rolling(window=3, min_periods=1).mean()
        except Exception as e:
            print(f"Error calculating Stochastic: {e}")
            data['%K'] = 50
            data['%D'] = 50
        
        try:
            # OBV with safety
            price_diff = data['Close'].diff().fillna(0)
            volume_sign = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
            data['OBV'] = (data['Volume'] * volume_sign).cumsum()
        except Exception as e:
            print(f"Error calculating OBV: {e}")
            data['OBV'] = 0
            
        try:
            # ROC (Rate of Change)
            data['ROC'] = data['Close'].pct_change(periods=12).fillna(0) * 100
        except Exception as e:
            print(f"Error calculating ROC: {e}")
            data['ROC'] = 0
            
        try:
            # Money Flow Index
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            money_flow = typical_price * data['Volume']
            
            # Safe logic for positive and negative flow
            price_diff = typical_price.diff().fillna(0)
            pos_flow = money_flow.where(price_diff > 0, 0).rolling(window=14, min_periods=1).sum()
            neg_flow = money_flow.where(price_diff < 0, 0).rolling(window=14, min_periods=1).sum().abs()
            
            # Safe MFI calculation avoiding division by zero
            data['MFI'] = 100 - (100 / (1 + pos_flow / neg_flow.replace(0, 1)))
        except Exception as e:
            print(f"Error calculating MFI: {e}")
            data['MFI'] = 50
        
        # Fill any remaining NaN values
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
        return data

    def prepare_scaled_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare scaled features for prediction with safety checks"""
        from sklearn.preprocessing import RobustScaler, MinMaxScaler

        # Ensure we have all required features
        all_features = (
            self.feature_columns['technical'] + 
            self.feature_columns['long_term'] + 
            self.feature_columns['sentiment']
        )
        
        # Add missing columns with default values
        for column in all_features:
            if column not in data.columns:
                print(f"Adding missing column: {column}")
                data[column] = 0.0
        
        # Select only the columns we need
        data_selected = data[all_features].copy()
        
        # Fill any NaN values
        data_selected = data_selected.fillna(0)
        
        # Scale features
        try:
            # Technical features
            tech_features = data_selected[self.feature_columns['technical']].values
            
            # Long-term features
            long_term_features = data_selected[self.feature_columns['long_term']].values
            
            # Sentiment features
            sent_features = data_selected[self.feature_columns['sentiment']].values
            
            # Scale each feature set
            tech_scaler = RobustScaler()
            sent_scaler = MinMaxScaler(feature_range=(-1, 1))
            
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
            
        except Exception as e:
            print(f"Error in scaling features: {e}")
            
            # If scaling fails, use simple normalization
            features = data_selected.values
            features_mean = np.nanmean(features, axis=0)
            features_std = np.nanstd(features, axis=0)
            features_std = np.where(features_std == 0, 1, features_std)  # Avoid division by zero
            
            # Simple normalization
            normalized_features = (features - features_mean) / features_std
            
            # Replace NaN values with zeros
            normalized_features = np.nan_to_num(normalized_features)
            
            print(f"Using fallback normalization. Feature shape: {normalized_features.shape}")
            return normalized_features
    
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
        volume_change = data['Volume'].pct_change().clip(-1, 1)  # Limit to ±100%
        data['sentiment_volume'] = data['raw_sentiment'] * np.log1p(np.abs(volume_change)) * np.sign(volume_change)
        
        data['sentiment_acceleration'] = data['sentiment_momentum'].diff().rolling(3).mean()
        sentiment_std = data['sentiment_rsi'].rolling(5).std()
        data['sentiment_regime'] = sentiment_std / sentiment_std.rolling(20).mean().fillna(1)
        data['sentiment_trend_strength'] = data['sentiment_trend'].abs().rolling(5).mean()
        
        return data

    def add_long_term_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add longer timeframe technical indicators with safety checks"""
        data = data.copy()
        
        # Longer-term moving averages with safety
        try:
            data['SMA_50'] = data['Close'].rolling(window=min(50, len(data)-1), min_periods=1).mean()
            data['SMA_100'] = data['Close'].rolling(window=min(100, len(data)-1), min_periods=1).mean()
            data['SMA_200'] = data['Close'].rolling(window=min(200, len(data)-1), min_periods=1).mean()
        except Exception as e:
            print(f"Error calculating long-term SMAs: {e}")
            data['SMA_50'] = data['Close']
            data['SMA_100'] = data['Close']
            data['SMA_200'] = data['Close']
        
        # Create SMA_20 if it doesn't exist (needed for crossover calculations)
        if 'SMA_20' not in data.columns:
            data['SMA_20'] = data['Close'].rolling(window=min(20, len(data)-1), min_periods=1).mean()
        
        # Moving average crossovers
        try:
            data['MA_Cross_50_200'] = data['SMA_50'] - data['SMA_200']
            data['MA_Cross_20_50'] = data['SMA_20'] - data['SMA_50']
        except Exception as e:
            print(f"Error calculating MA crossovers: {e}")
            data['MA_Cross_50_200'] = 0
            data['MA_Cross_20_50'] = 0
        
        # Longer-term momentum with safety
        try:
            data['ROC_20'] = data['Close'].pct_change(periods=min(20, len(data)-1)).fillna(0) * 100
            data['ROC_60'] = data['Close'].pct_change(periods=min(60, len(data)-1)).fillna(0) * 100
        except Exception as e:
            print(f"Error calculating ROC: {e}")
            data['ROC_20'] = 0
            data['ROC_60'] = 0
        
        # Trend strength indicators with safety
        try:
            data['ADX'] = self.calculate_adx(data)
        except Exception as e:
            print(f"Error calculating ADX: {e}")
            data['ADX'] = 25  # Neutral value
        
        # Volatility measures with safety
        try:
            data['ATR_20'] = self.calculate_atr(data, window=min(20, len(data)-1))
        except Exception as e:
            print(f"Error calculating ATR_20: {e}")
            data['ATR_20'] = data['Close'] * 0.02  # Default 2% volatility
        
        # Bollinger Band width with safety
        if all(col in data.columns for col in ['BB_upper', 'BB_lower', 'BB_middle']):
            try:
                data['BB_Width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
            except Exception as e:
                print(f"Error calculating BB_Width: {e}")
                data['BB_Width'] = 0.04  # Default 4% width
        else:
            print("BB columns missing, setting default BB_Width")
            data['BB_Width'] = 0.04
        
        # Volume trends with safety
        try:
            data['Volume_SMA_50'] = data['Volume'].rolling(window=min(50, len(data)-1), min_periods=1).mean()
            data['Volume_Trend'] = data['Volume'] / data['Volume_SMA_50'].replace(0, 1)
        except Exception as e:
            print(f"Error calculating Volume trends: {e}")
            data['Volume_SMA_50'] = data['Volume']
            data['Volume_Trend'] = 1.0
        
        # Fill any NaN values
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return data

    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Calculate Directional Movement
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
            
        return {'is_open': self._is_market_hours()}
    
    def _is_market_hours(self) -> bool:
        """Check if current time is during market hours"""
        if not hasattr(self, 'et_tz'):
            self.et_tz = timezone('US/Eastern')
            
        now = datetime.now(self.et_tz)
        current_time = now.time()
        
        # Market hours
        market_open = datetime.strptime('9:30', '%H:%M').time()
        market_close = datetime.strptime('16:00', '%H:%M').time()
        
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
        
        if price_time.tzinfo is None:
            price_time = self.et_tz.localize(price_time)
            
        return (now - price_time).total_seconds() < 300  # 5 minutes