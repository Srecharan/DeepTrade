from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from .prediction_system import PredictionSystem
from .stock_manager import StockManager
from pytz import timezone

class TradingStrategy:
    def __init__(self, 
                prediction_system: PredictionSystem,
                initial_capital: float = 100000.0):
        
        self.et_tz = timezone('US/Eastern') 
        # Risk Management
        self.max_positions = 2              # Maximum concurrent positions
        self.position_size = 0.02           # 2% of capital per trade
        self.max_daily_risk = 0.02          # Maximum 2% account risk per day
        self.max_trade_risk = 0.01          # Maximum 1% risk per trade
        self.max_stop_loss = 0.03           # Maximum 3% stop loss
        self.min_stop_loss = 0.01           # Minimum 1% stop loss
        
        # Entry Conditions
        self.confidence_threshold = 0.90     # Minimum 90% confidence
        self.min_expected_return = 0.005    # Minimum 0.5% expected return
        self.min_time_between_trades = 30   # Minutes between trades
        self.max_trades_per_day = 3         # Maximum trades per day
        
        # Technical Parameters
        self.required_trend_strength = 0.7   # 70% of indicators must be positive
        self.min_volume_percentile = 40     # Minimum volume requirement
        self.rsi_oversold = 30              # RSI oversold threshold
        self.rsi_overbought = 70            # RSI overbought threshold
        
        # Exit Parameters
        self.stop_loss_pct = 0.015          # Default 1.5% stop loss
        self.take_profit_pct = 0.03         # Default 3% take profit
        self.trailing_stop_pct = 0.005      # 0.5% trailing stop once in profit
        
        # Time-based Rules
        self.avoid_first_30min = True       # Avoid first 30 minutes of market
        self.avoid_last_30min = True        # Avoid last 30 minutes of market
        self.min_trade_holding = 5          # Minimum hold time (minutes)
        self.max_trade_holding = 180        # Maximum hold time (minutes)
        
        # Initialize components
        self.prediction_system = prediction_system
        self.stock_manager = StockManager()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.unrealized_capital = initial_capital
        
        # Trading state
        self.positions: Dict[str, Dict] = {}
        self.trades_history: List[Dict] = []
        self.active_orders: Dict[str, Dict] = {}
        self.daily_trade_count = 0
        self.last_trade_time = None
        
        # Add these checks
        self.min_time_between_trades = 60  # Minutes
        self.max_daily_trades = 2
        self.avoid_first_30min = True
        self.last_trade_time = None
        self.min_time_between_trades = 60  # minutes

    def evaluate_entry(self, symbol: str, timeframe: str = '15min') -> Tuple[bool, Dict]:
        try:
            print(f"\nEvaluating entry for {symbol}...")
            
            # Basic checks
            if len(self.positions) >= self.max_positions:
                return False, {"reason": "Max positions reached"}
                
            market_analysis = self._analyze_market_structure(symbol)
            price_info = self.stock_manager.get_real_time_price(symbol)
            if not price_info:
                return False, {"reason": "No price data"}
            current_price = float(price_info['price'])
            
            # Calculate trend composite
            trend_scores = {
                'strong_uptrend': 2,
                'uptrend': 1,
                'sideways': 0,
                'downtrend': -1,
                'strong_downtrend': -2
            }
            
            daily_score = trend_scores.get(market_analysis['1d']['trend'], 0)
            hourly_score = trend_scores.get(market_analysis['1h']['trend'], 0)
            minute_score = trend_scores.get(market_analysis['15min']['trend'], 0)
            
            trend_composite = (daily_score * 0.4) + (hourly_score * 0.4) + (minute_score * 0.2)
            
            # Volume condition - More lenient
            volume_condition = (
                market_analysis['15min']['relative_volume'] > 0.8 or  # 15min volume above 80% average
                market_analysis['1h']['relative_volume'] > 0.5 or     # Hourly volume above 50% average
                market_analysis['1d']['relative_volume'] > 0.3        # Daily volume above 30% average
            )
            
            # Price levels
            daily_support = market_analysis['1d']['support']
            hourly_support = market_analysis['1h']['support']
            minute_support = market_analysis['15min']['support']
            
            daily_resistance = market_analysis['1d']['resistance']
            hourly_resistance = market_analysis['1h']['resistance']
            minute_resistance = market_analysis['15min']['resistance']
            
            # Calculate distances
            support_distances = [
                (current_price - level) / current_price 
                for level in [daily_support, hourly_support, minute_support]
            ]
            resistance_distances = [
                (level - current_price) / current_price 
                for level in [daily_resistance, hourly_resistance, minute_resistance]
            ]
            
            near_support = min(support_distances) <= 0.02    # Within 2% of any support
            near_resistance = min(resistance_distances) <= 0.02  # Within 2% of any resistance
            
            # Entry types
            # Entry types
            entry_signal = None
            if volume_condition:  # Remove trend_composite requirement
                if near_support and trend_composite > -0.5:  # More lenient
                    entry_signal = 'pullback'
                elif near_resistance and trend_composite > -0.3:
                    entry_signal = 'breakout'
                elif trend_composite > 0.3:  # Lower threshold
                    entry_signal = 'momentum'
            
            # Print analysis
            print("\nEntry Conditions Analysis:")
            print(f"Trend Composite: {'✓' if trend_composite > 0 else '❌'} ({trend_composite:.2f})")
            print(f"Volume Condition: {'✓' if volume_condition else '❌'}")
            print(f"Near Support: {'✓' if near_support else '❌'}")
            print(f"Near Resistance: {'✓' if near_resistance else '❌'}")
            
            if entry_signal:
                entry_analysis = {
                    'entry_type': entry_signal,
                    'predictions': {
                        'timeframe': timeframe,
                        'entry_price': current_price,
                        'support': minute_support,
                        'resistance': minute_resistance,
                        'stop_loss': current_price * (1 - self.stop_loss_pct),
                        'take_profit': current_price * (1 + self.take_profit_pct)
                    }
                }
                return True, entry_analysis
                
            return False, {"reason": "No valid entry conditions met"}
            
        except Exception as e:
            print(f"Error in entry evaluation: {str(e)}")
            return False, {"reason": f"Error: {str(e)}"}
        
    def _check_price_trend(self, symbol: str) -> Tuple[bool, str]:
        """Check price trend and return detailed analysis"""
        try:
            price_info = self.stock_manager.get_real_time_price(symbol)
            if not price_info:
                return False, "No price data"
                
            current_price = price_info['price']
            
            # Get recent price data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            data = self.stock_manager.fetch_stock_data(
                symbol, 
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if data.empty:
                return False, "No historical data"
                
            # Calculate multiple trend indicators
            sma20 = data['Close'].rolling(window=20).mean().iloc[-1]
            sma50 = data['Close'].rolling(window=50).mean().iloc[-1]
            rsi = data['RSI'].iloc[-1] if 'RSI' in data else 50
            
            # Check different trend conditions
            above_sma20 = current_price > sma20
            sma20_above_sma50 = sma20 > sma50
            rsi_bullish = rsi > 50
            
            # Count how many trend indicators are bullish
            bullish_count = sum([above_sma20, sma20_above_sma50, rsi_bullish])
            
            # Return True if at least 2 indicators are bullish
            return bullish_count >= 2, f"Bullish indicators: {bullish_count}/3"
                
        except Exception as e:
            print(f"Error checking price trend: {str(e)}")
            return False, "Error in trend check"
            
    def enter_position(self, symbol: str, prediction_data: Dict) -> bool:
        """Enter a new position with support for historical prices"""
        try:
            # Get entry price based on entry type
            entry_type = prediction_data.get('entry_type')
            if entry_type == 'pullback':
                # Enter near support
                entry_price = prediction_data['predictions']['support'] * 1.01  # 1% above support
            elif entry_type == 'breakout':
                # Enter on resistance break
                entry_price = prediction_data['predictions']['resistance'] * 1.001  # Just above resistance
            elif entry_type == 'momentum':
                # Use current price for momentum entries
                entry_price = prediction_data['predictions']['entry_price']
            else:
                return False
                
            # Calculate position size based on risk
            risk_per_share = entry_price - prediction_data['predictions']['stop_loss']
            risk_amount = self.current_capital * self.max_trade_risk
            shares = int(risk_amount / risk_per_share)
            
            if shares == 0:
                return False
                
            # Calculate stop loss and take profit levels
            stop_loss = prediction_data['predictions']['stop_loss']
            take_profit = prediction_data['predictions']['take_profit']
            
            # Record position
            self.positions[symbol] = {
                'entry_time': datetime.now(self.et_tz),
                'entry_price': entry_price,
                'shares': shares,
                'current_value': shares * entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(),
                'prediction_data': prediction_data,
                'timeframe': prediction_data['predictions']['timeframe']
            }
            
            # Update capital
            transaction_cost = shares * entry_price * 0.001  # 0.1% transaction cost
            self.current_capital -= transaction_cost
            
            print(f"Transaction cost: ${transaction_cost:.2f}")
            print(f"\nEntered position in {symbol}:")
            print(f"Shares: {shares}")
            print(f"Entry Price: ${entry_price:.2f}")
            print(f"Position Value: ${shares * entry_price:.2f}")
            
            return True
            
        except Exception as e:
            print(f"Error entering position in {symbol}: {str(e)}")
            return False
            
    
    def evaluate_exit(self, symbol: str, current_price: Optional[float] = None, position_data: Optional[Dict] = None) -> Tuple[bool, str]:
        if not position_data and symbol not in self.positions:
            return False, "No position"
            
        position = position_data if position_data else self.positions[symbol]
        
        # Get per-share entry price
        shares = float(position.get('quantity', position.get('shares')))
        entry_price = float(position.get('cost_basis', position.get('entry_price'))) / shares  # Divide by shares!
        
        if current_price is None:
            price_info = self.stock_manager.get_real_time_price(symbol)
            if not price_info:
                print(f"Could not get current price for {symbol}")
                return False, "Could not get current price"
            current_price = float(price_info['price'])
        
        # Calculate stop loss and take profit based on per-share price
        stop_loss = entry_price * (1 - self.stop_loss_pct)
        take_profit = entry_price * (1 + self.take_profit_pct)
        
        print(f"\nExit Analysis for {symbol}:")
        print(f"Entry Price (per share): ${entry_price:.2f}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Stop Loss: ${stop_loss:.2f}")
        print(f"Take Profit: ${take_profit:.2f}")
        
        # Calculate P&L using per-share values
        entry_value = shares * entry_price
        current_value = shares * current_price
        unrealized_pl = current_value - entry_value
        unrealized_pl_pct = (unrealized_pl / entry_value) * 100
        
        print(f"Shares: {shares}")
        print(f"Unrealized P&L: ${unrealized_pl:.2f} ({unrealized_pl_pct:.2f}%)")
        
        # Check exit conditions
        if current_price <= stop_loss:
            return True, f"Stop loss triggered at {unrealized_pl_pct:.2f}%"
            
        if current_price >= take_profit:
            return True, f"Take profit triggered at {unrealized_pl_pct:.2f}%"
        
        # Check holding time
        entry_time = position.get('date_acquired', position.get('entry_time'))
        if isinstance(entry_time, str):
            entry_time_str = entry_time.replace('Z', '+00:00')
            entry_time = datetime.fromisoformat(entry_time_str).astimezone(self.et_tz)
        
        elapsed = datetime.now(self.et_tz) - entry_time
        elapsed_minutes = elapsed.total_seconds() / 60
        
        if elapsed_minutes >= self.max_trade_holding:
            return True, f"Max holding time reached ({elapsed_minutes:.1f} min)"
        
        return False, f"Holding - Within parameters (P&L: {unrealized_pl_pct:.2f}%)"

    def exit_position(self, symbol: str, exit_reason: str) -> bool:
        """Exit position with given symbol and reason"""
        if symbol not in self.positions:
            return False
            
        try:
            position = self.positions[symbol]
            exit_price = self._get_current_price(symbol)
                
            # Calculate P&L
            entry_value = position['shares'] * position['entry_price']
            exit_value = position['shares'] * exit_price
            profit_loss = exit_value - entry_value
            profit_loss_pct = (profit_loss / entry_value) * 100
            
            # Record trade
            trade_record = {
                'symbol': symbol,
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'shares': position['shares'],
                'profit_loss': profit_loss,
                'profit_loss_pct': profit_loss_pct,
                'exit_reason': exit_reason
            }
            
            self.trades_history.append(trade_record)
            
            # Update capital
            self.current_capital += exit_value
            
            # Remove position
            del self.positions[symbol]
            
            print(f"Exited position in {symbol}:")
            print(f"Exit Price: ${exit_price:.2f}")
            print(f"P&L: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
            print(f"Reason: {exit_reason}")
            
            return True
            
        except Exception as e:
            print(f"Error exiting position in {symbol}: {str(e)}")
            return False
          
            
    def get_performance_metrics(self) -> Dict:
        """Enhanced performance metrics with consistent naming"""
        # Base metrics
        metrics = {
            'total_trades': len(self.trades_history),
            'profitable_trades': len([t for t in self.trades_history if t['profit_loss'] > 0]),
            'current_capital': self.current_capital,
            'open_positions': len(self.positions)
        }
        
        # Calculate win rate
        metrics['win_rate'] = (metrics['profitable_trades'] / metrics['total_trades'] 
                            if metrics['total_trades'] > 0 else 0)
        
        # Calculate P&L components
        realized_pl = self.current_capital - self.initial_capital
        
        # Calculate unrealized P&L safely
        unrealized_pl = 0
        for symbol, pos in self.positions.items():
            current_price = self._get_current_price(symbol)
            if current_price:
                pos_value = pos['shares'] * current_price
                entry_value = pos['shares'] * pos['entry_price']
                unrealized_pl += pos_value - entry_value
        
        total_pl = realized_pl + unrealized_pl
        
        # Add P&L metrics with consistent naming
        metrics.update({
            'realized_pl': realized_pl,
            'unrealized_pl': unrealized_pl,
            'total_profit_loss': total_pl,  # Match the expected name
            'return_pct': (total_pl / self.initial_capital) * 100
        })
        
        return metrics
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price with better error handling"""
        try:
            price_info = self.stock_manager.get_real_time_price(symbol)
            if not price_info or 'price' not in price_info:
                return None
            return float(price_info['price'])
        except Exception as e:
            print(f"Error getting price for {symbol}: {str(e)}")
            return None
        

    # In TradingStrategy._find_support_resistance:
    def _find_support_resistance(self, data: pd.DataFrame, timeframe: str) -> Tuple[float, float]:
        if len(data) < 20:  # If not enough data
            return data['Low'].min(), data['High'].max()
            
        period = {'1d': 20, '1h': 48, '15min': 96}[timeframe]
        period = min(period, len(data))  # Don't use more periods than we have data
        
        highs = data['High'].rolling(window=period, min_periods=5).max()
        lows = data['Low'].rolling(window=period, min_periods=5).min()
        
        support = lows.iloc[-1]
        resistance = highs.iloc[-1]
        
        return support, resistance

    def _analyze_market_structure(self, symbol: str) -> Dict:
        """Analyze market structure across multiple timeframes"""
        print(f"\nAnalyzing market structure for {symbol}...")
        timeframes = ['1d', '1h', '15min']
        analysis = {}
        
        for tf in timeframes:
            print(f"\nProcessing {tf} timeframe...")
            try:
                # Get historical data with more data points
                lookback_days = {
                    '1d': 60,    # More data for daily trend
                    '1h': 10,    # Last 10 days for hourly
                    '15min': 5   # Last 5 days for minute data
                }[tf]
                data = self.stock_manager.fetch_stock_data(
                    symbol,
                    (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d'),
                    datetime.now().strftime('%Y-%m-%d'),
                    timeframe=tf
                )
                
                if data.empty:
                    print(f"❌ No data for {tf}")
                    continue
                
                print(f"✓ Got {len(data)} data points")
                
                # Add technical indicators
                data = self.stock_manager.add_technical_indicators(data)
                print("✓ Added technical indicators")
                
                # Get current price and indicators
                current_price = data['Close'].iloc[-1]
                
                # Volume analysis
                current_volume = data['Volume'].iloc[-1]
                avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
                relative_volume = current_volume / avg_volume if avg_volume > 0 else 0
                
                # Calculate trend
                trend = self._calculate_trend_strength(data)
                
                # Calculate support/resistance using the timeframe
                support, resistance = self._find_support_resistance(data, tf)
                
                print(f"Current price: ${current_price:.2f}")
                print(f"Trend: {trend}")
                print(f"Support: ${support:.2f}")
                print(f"Resistance: ${resistance:.2f}")
                print(f"Volume: {current_volume:,.0f} vs Avg: {avg_volume:,.0f}")
                
                analysis[tf] = {
                    'trend': trend,
                    'support': support,
                    'resistance': resistance,
                    'current_price': current_price,
                    'volume': current_volume,
                    'avg_volume': avg_volume,
                    'volume_trend': 'high' if relative_volume > 1.2 else 'normal' if relative_volume > 0.8 else 'low',
                    'relative_volume': relative_volume
                }
            except Exception as e:
                print(f"Error analyzing {tf} timeframe: {str(e)}")
                continue
        
        return analysis
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> str:
        """Calculate trend strength using multiple indicators"""
        # EMAs
        data['EMA20'] = data['Close'].ewm(span=20).mean()
        data['EMA50'] = data['Close'].ewm(span=50).mean()
        data['MA10'] = data['Close'].rolling(window=10).mean()
        current_price = data['Close'].iloc[-1]
        ma10 = data['MA10'].iloc[-1]
        ema20 = data['EMA20'].iloc[-1]
        ema50 = data['EMA50'].iloc[-1]
        
        # RSI
        rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50
        
        # Volume
        volume_sma = data['Volume'].rolling(window=20).mean()
        volume_trend = data['Volume'].iloc[-1] > volume_sma.iloc[-1]
        
        # Trend indicators
        bullish_signals = 0
        total_signals = 5

        if current_price > ma10:
            bullish_signals += 2 
        
        # Price above EMAs
        if current_price > ema20: bullish_signals += 1
        if current_price > ema50: bullish_signals += 1
        
        # EMA alignment
        if ema20 > ema50: bullish_signals += 1
        
        # RSI
        if rsi > 50: bullish_signals += 1
        
        # Volume
        if volume_trend: bullish_signals += 1
        
        # Calculate strength
        trend_strength = (bullish_signals * 0.7 + volume_trend * 0.3) / total_signals
        
        if trend_strength >= 0.8:
            return 'strong_uptrend'
        elif trend_strength >= 0.6:
            return 'uptrend'
        elif trend_strength <= 0.2:
            return 'strong_downtrend'
        elif trend_strength < 0.4:
            return 'downtrend'
        else:
            return 'sideways'

    def _validate_volume(self, market_analysis: Dict) -> bool:
        """Validate volume conditions across timeframes"""
        daily_vol = market_analysis['1d']['relative_volume']
        hourly_vol = market_analysis['1h']['relative_volume']
        
        # Primary volume check
        if daily_vol > 1.2 or hourly_vol > 1.5:
            return True
            
        # Secondary volume check
        if daily_vol > 0.8 and hourly_vol > 0.8:
            return True
            
        return False

    def _check_trend_alignment(self, analysis: Dict) -> bool:
        """Check if trends align across timeframes"""
        # Get trends
        daily_trend = analysis['1d']['trend']
        hourly_trend = analysis['1h']['trend']
        
        # Check if daily and hourly are both up or both down
        daily_bullish = 'uptrend' in daily_trend
        hourly_bullish = 'uptrend' in hourly_trend
        
        # If both timeframes agree, consider it aligned
        aligned = (daily_bullish and hourly_bullish) or (not daily_bullish and not hourly_bullish)
        
        print(f"\nTrend Alignment:")
        print(f"Daily trend: {daily_trend}")
        print(f"Hourly trend: {hourly_trend}")
        print(f"Aligned: {'✓' if aligned else '❌'}")
        
        return aligned

    def _calculate_position_size(self, current_price: float, support_level: float) -> int:
        """Calculate position size based on fixed risk"""
        # Use fixed risk amount per trade
        risk_amount = self.current_capital * self.max_trade_risk  # 1% risk per trade
        
        # Use stop loss of 1% below entry
        stop_loss_pct = 0.01  # 1% stop loss
        stop_loss_price = current_price * (1 - stop_loss_pct)
        
        # Calculate risk per share
        risk_per_share = current_price - stop_loss_price
        
        # Calculate shares based on risk
        shares = int(risk_amount / risk_per_share)
        
        # Minimum shares
        min_shares = max(1, int((self.current_capital * 0.005) / current_price))  # At least 0.5% of capital
        
        return max(min_shares, shares)
    
    def track_position(self, symbol: str, position_data: Dict) -> None:
        """Track position details and update metrics"""
        if symbol not in self.positions:
            self.positions[symbol] = {}
            
        self.positions[symbol].update({
            'last_price': position_data.get('current_price'),
            'high_price': max(position_data.get('current_price', 0), 
                            self.positions[symbol].get('high_price', 0)),
            'low_price': min(position_data.get('current_price', 0), 
                            self.positions[symbol].get('low_price', float('inf'))),
            'last_update': datetime.now(self.et_tz),
            'holding_time': (datetime.now(self.et_tz) - 
                            self.positions[symbol].get('entry_time', datetime.now(self.et_tz))).total_seconds() / 60
        })