from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
import requests
import pandas as pd
import numpy as np
from .trading_strategy import TradingStrategy
from .prediction_system import PredictionSystem
from .stock_manager import StockManager
from pytz import timezone


class PaperTradingSimulation:
    def __init__(self,
                 symbols: List[str],
                 initial_capital: float = 100000.0,
                 timeframe: str = '15min',
                 mode: str = 'tradier'):  # 'tradier' or 'realtime'
        
        self.initial_capital = initial_capital
        self.symbols = symbols
        self.timeframe = timeframe
        self.mode = mode
        self.prediction_system = PredictionSystem()
        self.trading_strategy = TradingStrategy(
            prediction_system=self.prediction_system,
            initial_capital=initial_capital
        )
        self.stock_manager = StockManager()
        self.et_tz = timezone('US/Eastern')
        
        # Tradier configuration
        if mode == 'tradier':
            self.tradier_account = "VA69917807"
            self.tradier_token = "2jacA1MpXvcFWhrg7ZQfUcf4fYww"
            self.tradier_endpoint = "https://sandbox.tradier.com/v1"
            self.tradier_headers = {
                'Authorization': f'Bearer {self.tradier_token}',
                'Accept': 'application/json'
            }
            self._verify_tradier_connection()
        
        # Performance tracking
        self.simulation_start = datetime.now()
        self.trades_log: List[Dict] = []
        self.performance_snapshots: List[Dict] = []
    
    def _verify_tradier_connection(self) -> None:
        """Verify Tradier connection and print account details"""
        try:
    
            response = requests.get(
                f'{self.tradier_endpoint}/user/profile',
                headers=self.tradier_headers
            )
            response.raise_for_status()
            profile_data = response.json()
            
            # Get account balances
            balance_response = requests.get(
                f'{self.tradier_endpoint}/accounts/{self.tradier_account}/balances',
                headers=self.tradier_headers
            )
            balance_response.raise_for_status()
            balance_data = balance_response.json()
            
            print("\nTradier Account Information:")
            print(f"Account ID: {self.tradier_account}")
            print(f"Account Type: Paper Trading")
            print(f"Total Equity: ${balance_data['balances']['total_equity']}")
            print(f"Option Level: {profile_data['profile']['account']['option_level']}")
            print("=" * 50)
            
        except Exception as e:
            raise Exception(f"Failed to connect to Tradier: {str(e)}")
    
    def _execute_tradier_trade(self, action: str, symbol: str, quantity: int) -> Dict:
        """Execute trade with improved short position handling"""
        try:
            is_short = quantity < 0
            abs_quantity = abs(quantity)
            
            if action == 'ENTER':
                side = 'sell_short' if is_short else 'buy'
            else:  # EXIT
                side = 'buy_to_cover' if is_short else 'sell'
            
            order_data = {
                'class': 'equity',
                'symbol': symbol,
                'side': side,
                'quantity': abs_quantity,  # Always use positive quantity
                'type': 'market',
                'duration': 'day'
            }
            
            print(f"\nExecuting Tradier {order_data['side'].upper()} order:")
            print(f"Symbol: {symbol}")
            print(f"Quantity: {abs_quantity}")
            
            response = requests.post(
                f'{self.tradier_endpoint}/accounts/{self.tradier_account}/orders',
                headers=self.tradier_headers,
                data=order_data
            )
            
            if response.status_code != 200:
                print(f"Tradier API error: {response.status_code}")
                print(f"Response: {response.text}")
                return {'success': False, 'error': f"API error: {response.status_code}"}
            
            order_result = response.json()
            
            if 'order' not in order_result:
                return {'success': False, 'error': 'Invalid order response'}
            
            print(f"\nTrade executed via Tradier:")
            print(f"Order ID: {order_result['order']['id']}")
            print(f"Status: {order_result['order']['status']}")
            
            return {
                'success': True,
                'order_id': order_result['order']['id'],
                'status': order_result['order']['status']
            }
                
        except Exception as e:
            print(f"Error executing Tradier trade: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {'success': False, 'error': str(e)}
    
    def _get_tradier_positions(self) -> Dict:
        """Get current positions from Tradier"""
        try:
            response = requests.get(
                f'{self.tradier_endpoint}/accounts/{self.tradier_account}/positions',
                headers=self.tradier_headers
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"Error getting Tradier positions: {str(e)}")
            return None
    
    def _get_tradier_account_balance(self) -> float:
        """Get current account balance from Tradier"""
        try:
            response = requests.get(
                f'{self.tradier_endpoint}/accounts/{self.tradier_account}/balances',
                headers=self.tradier_headers
            )
            response.raise_for_status()
            
            return float(response.json()['balances']['total_equity'])
            
        except Exception as e:
            print(f"Error getting Tradier balance: {str(e)}")
            return None
        
    def _get_check_interval(self) -> int:
        """Get interval between checks based on timeframe"""
        intervals = {
            '5min': 60,   # Check every minute
            '15min': 180,  # Check every 3 minutes
            '30min': 300,  # Check every 5 minutes
            '1h': 600     # Check every 10 minutes
        }
        return intervals.get(self.timeframe, 300)
           
    def run_simulation(self, duration_minutes: int = 60) -> Dict:
        """Run paper trading simulation with improved market hours check"""
        # Check if market is open first
        if not self._is_market_open():
            current_time = datetime.now(self.et_tz)
            market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            
            print("\n🔔 MARKET HOURS NOTICE:")
            print("The market is currently closed.")
            print(f"Market hours are 9:30 AM - 4:00 PM ET, Monday-Friday")
            print(f"Current time: {current_time.strftime('%I:%M %p ET')}")
            
            # Give more specific guidance
            if current_time.hour < 9 or (current_time.hour == 9 and current_time.minute < 30):
                print(f"Market opens in {(market_open - current_time).seconds // 60} minutes")
            elif current_time.hour >= 16:
                next_open = market_open + timedelta(days=1)
                if current_time.weekday() >= 4:  # Friday or Saturday
                    days_to_monday = (7 - current_time.weekday()) % 7
                    next_open += timedelta(days=days_to_monday)
                print(f"Market will open next at {next_open.strftime('%I:%M %p ET on %A')}")
            
            print("\nPlease run the simulation during market hours for live trading.")
            return None
            
        if self.mode == 'tradier':
            return self._run_tradier_trading(duration_minutes)
        else:
            return self._run_realtime_simulation(duration_minutes)
        
    def _run_tradier_trading(self, duration_minutes: int) -> Dict:
        """Run trading using Tradier with enhanced debugging"""
        try:
            end_time = datetime.now() + timedelta(minutes=duration_minutes)
            check_interval = self._get_check_interval()
            market_close = datetime.strptime('16:00', '%H:%M').time()
            
            print(f"\nDuration: {duration_minutes} minutes")
            print(f"Check interval: {check_interval} seconds")
            print(f"Session will end at: {min(end_time, datetime.combine(datetime.now().date(), market_close)).strftime('%H:%M:%S')}")
            print("=" * 50)
            
            while datetime.now() < end_time:
                try:
                    # Skip if market is closed
                    if not self._is_market_open():
                        print("\nMarket is closed. Ending session.")
                        break

                    current_time = datetime.now()
                    time_remaining = (end_time - current_time).total_seconds() / 60
                    
                    print(f"\n{'='*20} Trading Update {'='*20}")
                    print(f"Time: {current_time.strftime('%H:%M:%S')}")
                    print(f"Session remaining: {time_remaining:.1f} minutes")
                    print(f"Market status: {'OPEN' if self._is_market_open() else 'CLOSED'}")                   
                    
                    # Get account balance
                    account_balance = self._get_tradier_account_balance()
                    print(f"Account Balance: ${account_balance:,.2f}")
                    
                    positions_response = self._get_tradier_positions()
                    current_positions = []
                    if positions_response and isinstance(positions_response, dict):
                        if 'positions' in positions_response:
                            positions_data = positions_response['positions']
                            if isinstance(positions_data, dict) and 'position' in positions_data:
                                position_info = positions_data['position']
                                if isinstance(position_info, list):
                                    current_positions = position_info
                                elif isinstance(position_info, dict):
                                    current_positions = [position_info]
                    
      
                    current_positions = [p for p in current_positions if p['symbol'] in self.symbols]
                    
                    print(f"\nActive positions for monitored symbols: {len(current_positions)}")
                    
            
                    for position in current_positions:
                        self._process_position(position)
                    
                
                    current_symbols = [p['symbol'] for p in current_positions]
                    account_balance = self._get_tradier_account_balance()
                    
                    for symbol in self.symbols:
                        if symbol in current_symbols:
                            continue
                            
                        price_info = self.stock_manager.get_real_time_price(symbol)
                        if not price_info:
                            print(f"Could not get current price for {symbol}")
                            continue
                            
                        current_price = float(price_info['price'])
                        
                        print(f"\nAnalyzing {symbol} for entry at ${current_price:.2f}")
                        
                        should_enter, entry_data = self.trading_strategy.evaluate_entry(symbol, self.timeframe)
                        
                        if should_enter and account_balance:
                            shares = int((account_balance * self.trading_strategy.position_size) / current_price)
                            
                            if shares > 0:
                                print(f"Entry signal received for {symbol}:")
                                print(f"Shares to buy: {shares}")
                                print(f"Estimated value: ${shares * current_price:.2f}")
                                
                                success = self._execute_tradier_trade('ENTER', symbol, shares)
                                if success:
                                    self._log_trade_details('ENTER', symbol, {
                                        'entry_price': current_price,
                                        'shares': shares,
                                        'entry_type': entry_data.get('entry_type', 'market')
                                    })
                        else:
                            if should_enter:
                                print("Entry signal received but insufficient balance")
                            else:
                                print("No entry signal")
                    
                    print("\n📈 SESSION SUMMARY:")
                    print(f"Elapsed Time: {(datetime.now() - self.simulation_start).seconds // 60} minutes")
                    print(f"Open Positions: {len(current_positions)}")
                    print(f"Account Value: ${account_balance:,.2f}")
                    
                    print(f"\n{'='*20} Waiting {'='*20}")
                    print(f"Next check in {check_interval} seconds...")
                    time.sleep(check_interval)
                        
                except Exception as e:
                    print(f"Error in trading loop: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    time.sleep(5)

        
            print("\n🔔 SESSION COMPLETE")
            positions_response = self._get_tradier_positions()
            if positions_response and 'positions' in positions_response:
                if isinstance(positions_response['positions'], dict) and 'position' in positions_response['positions']:
                    current_positions = positions_response['positions']['position']
                    if isinstance(current_positions, dict):
                        current_positions = [current_positions]
                    print("\nClosing all open positions...")
                    for position in current_positions:
                        if position['symbol'] in self.symbols:
                            self._process_position(position)
       
            final_results = self.get_simulation_results()
            final_results['initial_capital'] = self.trading_strategy.initial_capital
            final_results['symbols'] = self.symbols
            final_results['timeframe'] = self.timeframe
            
            return final_results
            
        except Exception as e:
            print(f"Error in trading simulation: {str(e)}")
            return None
                
                
    def _is_market_open(self) -> bool:
        """Check if market is open using Tradier API and ET timezone"""
        try:
            current_time = datetime.now(self.et_tz)
            response = requests.get(
                f'{self.tradier_endpoint}/markets/clock',
                headers=self.tradier_headers
            )
            response.raise_for_status()
            
            return response.json()['clock']['state'] == 'open'
            
        except Exception as e:
            print(f"Error checking market status: {str(e)}")
            
            # Fallback to manual check (Pittsburgh/ET times)
            market_open = datetime.strptime('09:30', '%H:%M').time()
            market_close = datetime.strptime('16:00', '%H:%M').time()
            current_time_et = current_time.time()
            
            return (
                current_time.weekday() < 5 and  # Monday to Friday
                market_open <= current_time_et <= market_close
            )

    def _run_realtime_simulation(self, duration_minutes: int) -> Dict:
        """Run simulation with real-time market data"""
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        last_check = {symbol: datetime.now() for symbol in self.symbols}
        check_interval = self._get_check_interval()
        
        while datetime.now() < end_time:
            try:
                current_time = datetime.now()
                
                # Skip if not market hours
                if not self.stock_manager._is_market_hours():
                    print("Outside market hours. Waiting...")
                    time.sleep(60)
                    continue
                    
                # Check positions that might need to exit
                self._check_exits()
             
                for symbol in self.symbols:
                    if (current_time - last_check[symbol]).total_seconds() < check_interval:
                        continue
                        
                    last_check[symbol] = current_time
                    
                    if symbol in self.trading_strategy.positions:
                        continue
                        
                    should_enter, entry_data = self.trading_strategy.evaluate_entry(
                        symbol, 
                        self.timeframe
                    )
                    
                    if should_enter:
                        success = self.trading_strategy.enter_position(symbol, entry_data)
                        if success:
                            self._log_trade_details('ENTER', symbol, entry_data)
                            
                self._take_performance_snapshot()
                time.sleep(check_interval / 2)
                
            except Exception as e:
                print(f"Error in simulation loop: {str(e)}")
                time.sleep(5)
                
        self._close_all_positions()
        return self.get_simulation_results()
        
    def _check_exits(self) -> None:
        """Check if any positions need to exit"""
        for symbol in list(self.trading_strategy.positions.keys()):
            should_exit, reason = self.trading_strategy.evaluate_exit(symbol)
            if should_exit:
                success = self.trading_strategy.exit_position(symbol, reason)
                if success:
                    self._log_trade_details('EXIT', symbol, {'reason': reason})
                    
    def _close_all_positions(self) -> None:
        """Close all open positions"""
        for symbol in list(self.trading_strategy.positions.keys()):
            success = self.trading_strategy.exit_position(
                symbol, 
                "Simulation end"
            )
            if success:
                self._log_trade_details('EXIT', symbol, {'reason': "Simulation end"})

    def _take_performance_snapshot(self) -> None:
        """Take a snapshot of current performance metrics"""
        try:
            metrics = self.trading_strategy.get_performance_metrics()
            metrics['timestamp'] = datetime.now()
            metrics['open_positions'] = len(self.trading_strategy.positions)
            self.performance_snapshots.append(metrics)
        except Exception as e:
            print(f"Error taking performance snapshot: {str(e)}")
            
        
    def get_simulation_results(self) -> Dict:
        """Get simulation results with improved trade processing"""
        try:
            account_balance = self._get_tradier_account_balance()
            positions_response = self._get_tradier_positions()
            order_history = self._get_tradier_order_history()
                       
            all_trades = []
            
            trade_pairs = {}  
            for order in order_history:
                symbol = order['symbol']
                if symbol not in trade_pairs:
                    trade_pairs[symbol] = []
                trade_pairs[symbol].append(order)
                
            # Match entry and exit orders
            for symbol, orders in trade_pairs.items():
                orders.sort(key=lambda x: x['created_at'])  # Sort by time
                for i in range(0, len(orders) - 1, 2):  # Process pairs
                    entry = orders[i]
                    exit = orders[i + 1] if i + 1 < len(orders) else None
                    
                    if exit:
                        quantity = abs(int(entry['quantity']))
                        is_short = entry['side'] == 'sell_short'
                        entry_price = abs(float(entry['price']))
                        exit_price = abs(float(exit['price']))
                        
                        # Handle timestamps properly
                        try:
                            if isinstance(entry['created_at'], (int, str)):
                                entry_time = datetime.fromtimestamp(int(entry['created_at'])).astimezone(self.et_tz)
                            else:
                                entry_time = entry['created_at'].astimezone(self.et_tz)
                                
                            if isinstance(exit['created_at'], (int, str)):
                                exit_time = datetime.fromtimestamp(int(exit['created_at'])).astimezone(self.et_tz)
                            else:
                                exit_time = exit['created_at'].astimezone(self.et_tz)
                        except (ValueError, TypeError):
                            # If timestamp conversion fails, use current time
                            entry_time = datetime.now(self.et_tz)
                            exit_time = datetime.now(self.et_tz)
                        
                        # Calculate P&L
                        if is_short:
                            profit_loss = (entry_price - exit_price) * quantity
                        else:
                            profit_loss = (exit_price - entry_price) * quantity
                        
                        profit_loss_pct = (profit_loss / (entry_price * quantity)) * 100
                        
                        trade = {
                            'action': 'EXIT',
                            'symbol': symbol,
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'shares': quantity * (-1 if is_short else 1),
                            'profit_loss': profit_loss,
                            'profit_loss_pct': profit_loss_pct
                        }
                        all_trades.append(trade)
            
            all_trades.extend(self.trades_log)
       
            all_trades.sort(key=lambda x: x['exit_time'] if x.get('exit_time') else datetime.now(self.et_tz), reverse=True)
      
            current_positions = []
            if positions_response and 'positions' in positions_response:
                if isinstance(positions_response['positions'], dict):
                    position_data = positions_response['positions'].get('position', [])
                    if position_data:
                        if isinstance(position_data, dict):
                            current_positions = [position_data]
                        else:
                            current_positions = position_data
            
            if current_positions:
                self._update_position_prices(current_positions)

            closed_trades = [t for t in all_trades if t['action'] == 'EXIT']
            profitable_trades = len([t for t in closed_trades if t.get('profit_loss', 0) > 0])
            total_trades = len(closed_trades)
            
            total_pl = account_balance - self.initial_capital
            
            return {
                'session_duration': (datetime.now() - self.simulation_start).total_seconds() / 60,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate': profitable_trades / total_trades if total_trades > 0 else 0,
                'total_profit_loss': total_pl,
                'return_pct': (total_pl / self.initial_capital) * 100,
                'final_capital': account_balance,
                'initial_capital': self.initial_capital,
                'trades_history': all_trades,
                'open_positions': current_positions,
                'performance_history': self.performance_snapshots
            }
        except Exception as e:
            print(f"Error getting simulation results: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {
                'session_duration': 0,
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0,
                'total_profit_loss': 0,
                'return_pct': 0,
                'final_capital': self.initial_capital,
                'initial_capital': self.initial_capital,
                'trades_history': [],
                'open_positions': [],
                'performance_history': []
            }
        
    def visualize_results(self, results: Dict):
        """Create visualization of trading results"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # 1. Equity Curve
        if self.performance_snapshots:
            equity_data = [snapshot['current_capital'] for snapshot in self.performance_snapshots]
            dates = [snapshot['timestamp'] for snapshot in self.performance_snapshots]
            ax1.plot(dates, equity_data, label='Portfolio Value')
            ax1.set_title('Portfolio Value Over Time')
            ax1.grid(True)
            ax1.legend()
        
        # 2. Trade Results
        if self.trades_log:
            profits = []
            symbols = []
            for trade in self.trades_log:
                if trade['action'] == 'EXIT':
                    if 'profit_loss' in trade['data']:
                        profits.append(trade['data']['profit_loss'])
                        symbols.append(trade['symbol'])
            
            if profits:
                ax2.bar(range(len(profits)), profits, label='Trade P&L')
                ax2.set_xticks(range(len(profits)))
                ax2.set_xticklabels(symbols, rotation=45)
                ax2.set_title('Individual Trade Results')
                ax2.grid(True)
        
        # 3. Performance Metrics
        metrics = {
            'Win Rate': f"{results['win_rate']:.1%}",
            'Total P&L': f"${results['total_profit_loss']:.2f}",
            'Return': f"{results['return_pct']:.1f}%",
            'Total Trades': results['total_trades']
        }
        
        metrics_text = '\n'.join([f"{k}: {v}" for k, v in metrics.items()])
        ax3.text(0.1, 0.5, metrics_text, fontsize=12, ha='left', va='center')
        ax3.set_title('Performance Summary')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig('visualization/trading_results.png')
        plt.close()

    def _is_market_hours(self, current_time: datetime) -> bool:
        """Check if given time is during market hours"""
        if not hasattr(self, 'et_tz'):
            self.et_tz = timezone('US/Eastern')
            
        if current_time.tzinfo is None:
            current_time = self.et_tz.localize(current_time)
            
        # Market hours 9:30 AM - 4:00 PM ET
        market_open = datetime.strptime('09:30', '%H:%M').time()
        market_close = datetime.strptime('16:00', '%H:%M').time()
        current_time_et = current_time.astimezone(self.et_tz).time()
        
        # Check if it's a weekday and within market hours
        return (
            current_time.weekday() < 5 and  # Monday to Friday
            market_open <= current_time_et <= market_close
        )
    
    def _log_trade_details(self, action: str, symbol: str, data: Dict) -> None:
        try:
            current_time = datetime.now(self.et_tz)
            
            trade_log = {
                'timestamp': current_time,
                'action': action,
                'symbol': symbol,
                'entry_time': data.get('entry_time', current_time),
                'exit_time': current_time if action == 'EXIT' else None,
                'entry_price': data.get('entry_price', 0),
                'exit_price': data.get('exit_price', 0),
                'shares': data.get('shares', 0),
                'quantity': data.get('quantity', 0),  # Add this
                'profit_loss': data.get('profit_loss', 0),
                'profit_loss_pct': data.get('profit_loss_pct', 0),
                'entry_reason': data.get('entry_type', 'market'),
                'exit_reason': data.get('reason', ''),
                'holding_time': data.get('holding_time', 0)
            }
            
            self.trades_log.append(trade_log)

            if action == 'ENTER':
                print(f"\n🔵 ENTRY - {symbol}")
                print(f"Time: {trade_log['entry_time'].strftime('%H:%M:%S ET')}")
                print(f"Price: ${trade_log['entry_price']:.2f}")
                print(f"Shares: {trade_log['shares']}")
                print(f"Reason: {trade_log['entry_reason']}")
            else:
                print(f"\n🔴 EXIT - {symbol}")
                print(f"Time: {trade_log['exit_time'].strftime('%H:%M:%S ET')}")
                print(f"Price: ${trade_log['exit_price']:.2f}")
                print(f"P&L: ${trade_log['profit_loss']:.2f} ({trade_log['profit_loss_pct']:.2f}%)")
                print(f"Holding Time: {trade_log['holding_time']:.1f} minutes")
                print(f"Reason: {trade_log['exit_reason']}")
                
            self._take_performance_snapshot()
            
        except Exception as e:
            print(f"Error logging trade: {str(e)}")

    def _process_position(self, position: Dict) -> None:
        """Process position with improved P&L calculation"""
        try:
            symbol = position['symbol']
            quantity = float(position['quantity'])
            cost_basis = float(position['cost_basis']) / abs(quantity)  # Use abs for correct per-share cost
            
            if isinstance(position.get('date_acquired'), str):
                entry_time_str = position['date_acquired'].replace('Z', '+00:00')
                entry_time = datetime.fromisoformat(entry_time_str).astimezone(self.et_tz)
            else:
                entry_time = position.get('date_acquired', datetime.now(self.et_tz))
                
            current_time = datetime.now(self.et_tz)
            holding_time = (current_time - entry_time).total_seconds() / 60
            
            print(f"\n{'='*10} Position Analysis: {symbol} {'='*10}")
            print(f"Entry Time: {entry_time.strftime('%H:%M:%S ET')}")
            print(f"Holding Time: {holding_time:.1f} minutes")
            print(f"Entry Price: ${cost_basis:.2f}")
            
            price_info = self.stock_manager.get_real_time_price(symbol)
            if not price_info:
                print(f"Could not get current price for {symbol}")
                return
                
            current_price = float(price_info['price'])
            print(f"Current Price: ${current_price:.2f}")
            
            # Calculate P&L accounting for short positions
            position_value = quantity * current_price
            entry_value = quantity * cost_basis
            unrealized_pl = position_value - entry_value
            unrealized_pl_pct = (unrealized_pl / abs(entry_value)) * 100  # Use abs for correct percentage
            
            print(f"\nPosition Performance:")
            print(f"Unrealized P&L: ${unrealized_pl:.2f} ({unrealized_pl_pct:+.2f}%)")
            
            is_short = quantity < 0
            stop_loss = cost_basis * (1 + self.trading_strategy.stop_loss_pct) if is_short else cost_basis * (1 - self.trading_strategy.stop_loss_pct)
            take_profit = cost_basis * (1 - self.trading_strategy.take_profit_pct) if is_short else cost_basis * (1 + self.trading_strategy.take_profit_pct)
            
            print(f"\nExit Analysis for {symbol}:")
            print(f"Entry Price (per share): ${cost_basis:.2f}")
            print(f"Current Price: ${current_price:.2f}")
            print(f"Stop Loss: ${stop_loss:.2f}")
            print(f"Take Profit: ${take_profit:.2f}")
            print(f"Shares: {quantity}")
            print(f"Unrealized P&L: ${unrealized_pl:.2f} ({unrealized_pl_pct:.2f}%)")
            
            # Evaluate exit with corrected values
            should_exit = False
            exit_reason = "Holding - Within parameters"
            
            if is_short:
                if current_price >= stop_loss:
                    should_exit = True
                    exit_reason = f"Stop loss triggered at {unrealized_pl_pct:.2f}%"
                elif current_price <= take_profit:
                    should_exit = True
                    exit_reason = f"Take profit triggered at {unrealized_pl_pct:.2f}%"
            else:
                if current_price <= stop_loss:
                    should_exit = True
                    exit_reason = f"Stop loss triggered at {unrealized_pl_pct:.2f}%"
                elif current_price >= take_profit:
                    should_exit = True
                    exit_reason = f"Take profit triggered at {unrealized_pl_pct:.2f}%"
            
            if should_exit:
                print("\n⚠️ EXIT SIGNAL")
                print(f"Exit Reason: {exit_reason}")
                success = self._execute_tradier_trade('EXIT', symbol, int(quantity))
                if success.get('success', False):
                    self._log_trade_details('EXIT', symbol, {
                        'exit_price': current_price,
                        'entry_price': cost_basis,
                        'shares': quantity,
                        'profit_loss': unrealized_pl,
                        'profit_loss_pct': unrealized_pl_pct,
                        'reason': exit_reason,
                        'entry_time': entry_time,
                        'holding_time': holding_time
                    })
            else:
                print("\n✅ HOLDING")
                print(f"Holding Reason: {exit_reason}")
                print(f"Current P&L: {unrealized_pl_pct:+.2f}%")
                
        except Exception as e:
            print(f"Error processing position: {str(e)}")
            import traceback
            print(traceback.format_exc())    

    def _update_position_prices(self, positions: List[Dict]) -> None:
        for position in positions:
            try:
                symbol = position['symbol']
                price_info = self.stock_manager.get_real_time_price(symbol)
                if price_info:
                    position['current_price'] = float(price_info['price'])
                    # Update unrealized P&L
                    entry_price = float(position['cost_basis']) / float(position['quantity'])
                    position['unrealized_pl'] = (position['current_price'] - entry_price) * float(position['quantity'])
            except Exception as e:
                print(f"Error updating price for {symbol}: {e}")

    def _get_tradier_order_history(self) -> List[Dict]:
        """Get all order history from Tradier with proper timestamp handling"""
        try:
            response = requests.get(
                f'{self.tradier_endpoint}/accounts/{self.tradier_account}/orders',
                headers=self.tradier_headers
            )
            response.raise_for_status()          
            orders = response.json().get('orders', {}).get('order', [])
            if not isinstance(orders, list):
                orders = [orders] if orders else []
            
            if isinstance(orders, dict):
                orders = [orders]
                
            processed_orders = []
            for order in orders:
                if order.get('status') == 'filled':
                    try:
                        # Convert timestamps correctly
                        created_at = datetime.fromtimestamp(int(order.get('created_at', 0)), self.et_tz)
                        updated_at = datetime.fromtimestamp(int(order.get('updated_at', 0)), self.et_tz)
                        
                        # Get proper price and quantity
                        quantity = int(order.get('quantity', 0))
                        if order.get('side') == 'sell_short':
                            quantity = -quantity
                        
                        price = float(order.get('avg_fill_price') or order.get('price', 0))
                        
                        processed_order = {
                            'id': order.get('id'),
                            'symbol': order.get('symbol'),
                            'side': order.get('side'),
                            'quantity': quantity,
                            'price': abs(price),
                            'status': order.get('status'),
                            'created_at': created_at,
                            'updated_at': updated_at,
                        }
                        processed_orders.append(processed_order)
                    except Exception as e:
                        print(f"Error processing order: {e}")
                        continue
                        
            return processed_orders
                
        except Exception as e:
            print(f"Error getting order history: {str(e)}")
            return []