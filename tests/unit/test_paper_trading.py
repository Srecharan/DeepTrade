from utils.paper_trading import PaperTradingSimulation
from datetime import datetime
import argparse
from tabulate import tabulate
from pytz import timezone

def test_paper_trading():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='tradier',
                       choices=['tradier', 'realtime'],
                       help='Trading mode')
    args = parser.parse_args()

    symbols = ['NVDA', 'AAPL', 'MSFT', 'GME', 'AMD', 'JNJ', 'META', 'GOOGL', 'AMZN']
    
    print(f"\n{'='*20} TRADING SESSION START {'='*20}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S ET')}")
    print(f"Monitoring: {', '.join(symbols)}")
    print("=" * 70)
    
    simulation = PaperTradingSimulation(
        symbols=symbols,
        initial_capital=100000.0,
        timeframe='15min',
        mode=args.mode
    )
    
    try:
        minutes_until_close = calculate_minutes_to_market_close()
        
        if minutes_until_close <= 0:
            print("\nMarket is already closed for today!")
            print("Please run during market hours (9:30 AM - 4:00 PM ET)")
            return
            
        results = simulation.run_simulation(duration_minutes=minutes_until_close)
        
        if results:
            print_session_summary(results, simulation)  # Pass simulation instance
        else:
            print("\nNo trading activity today")
            
    except KeyboardInterrupt:
        print("\n\nTrading session interrupted by user.")
        final_results = simulation.get_simulation_results()
        print_session_summary(final_results, simulation)  

def print_session_summary(results: dict, simulation: PaperTradingSimulation):
    et_tz = timezone('US/Eastern')
    print("\n" + "="*30 + " TRADING SESSION SUMMARY " + "="*30)

    # COMPLETED TRADES (without timestamps)
    print("\n📈 COMPLETED TRADES (Most Recent 5):")
    trades = results.get('trades_history', [])
    if trades:
        trade_data = []
        headers = ["Symbol", "Entry $", "Exit $", "Shares", "P&L($)", "P&L(%)"]  # Removed time columns
        
        completed_trades = [t for t in trades if t.get('action') == 'EXIT']
        completed_trades.sort(key=lambda x: x.get('exit_time', datetime.now()), reverse=True)
        recent_trades = completed_trades[:5]
        
        for trade in recent_trades:
            trade_data.append([
                trade.get('symbol', ''),
                f"${abs(trade.get('entry_price', 0)):.2f}",
                f"${abs(trade.get('exit_price', 0)):.2f}",
                trade.get('shares', 0),
                f"${trade.get('profit_loss', 0):.2f}",
                f"{trade.get('profit_loss_pct', 0):.2f}%"
            ])
                
        if trade_data:
            print(tabulate(trade_data, headers=headers, tablefmt='grid'))
        else:
            print("No completed trades yet")
    else:
        print("No trades history available")

    # CURRENT POSITIONS (keeping timestamps)
    print("\n📊 CURRENT POSITIONS:")
    positions = results.get('open_positions', [])
    if positions:
        pos_data = []
        headers = ["Symbol", "Entry Time", "Entry $", "Current $", "Shares", "Unr. P&L($)", "P&L(%)", "Status"]
        
        for pos in positions:
            try:
                # Handle entry time (keeping this as it works correctly)
                entry_time = pos.get('date_acquired')
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                    entry_time = entry_time.astimezone(et_tz)
                
                entry_price = float(pos['cost_basis'])/float(pos['quantity'])
                price_info = simulation.stock_manager.get_real_time_price(pos['symbol'])
                current_price = float(price_info['price']) if price_info else entry_price
                unrealized_pl = (current_price - entry_price) * float(pos['quantity'])
                unrealized_pl_pct = (unrealized_pl / (entry_price * float(pos['quantity']))) * 100
                
                pos_data.append([
                    pos['symbol'],
                    entry_time.strftime('%H:%M:%S ET'),
                    f"${entry_price:.2f}",
                    f"${current_price:.2f}",
                    pos['quantity'],
                    f"${unrealized_pl:.2f}",
                    f"{unrealized_pl_pct:.2f}%",
                    "TRADING"
                ])
            except Exception as e:
                print(f"Error processing position: {e}")
                continue
                
        if pos_data:
            print(tabulate(pos_data, headers=headers, tablefmt='grid'))
        else:
            print("Error processing positions data")
    else:
        print("No open positions")

    # PERFORMANCE SUMMARY (unchanged)
    print(f"\n💰 SESSION SUMMARY:")
    summary_data = [
        ["Initial", "Final", "Total P&L", "Return", "Trades", "Win Rate"],
        [
            f"${results['initial_capital']:,.2f}",
            f"${results['final_capital']:,.2f}",
            f"${results['total_profit_loss']:,.2f}",
            f"{results['return_pct']:.2f}%",
            results.get('total_trades', 0),
            f"{results.get('win_rate', 0)*100:.1f}%"
        ]
    ]
    print(tabulate(summary_data, tablefmt='grid'))

def calculate_hold_time(trade: dict) -> float:
    try:
        et_tz = timezone('US/Eastern')
        
        if isinstance(trade.get('entry_time'), str):
            entry_time_str = trade['entry_time'].replace('Z', '+00:00')
            entry_time = datetime.fromisoformat(entry_time_str)
            if entry_time.tzinfo is None:
                entry_time = et_tz.localize(entry_time)
        else:
            entry_time = trade.get('entry_time', datetime.now(et_tz))
            if entry_time.tzinfo is None:
                entry_time = et_tz.localize(entry_time)
        
        if isinstance(trade.get('exit_time'), str):
            exit_time_str = trade['exit_time'].replace('Z', '+00:00')
            exit_time = datetime.fromisoformat(exit_time_str)
            if exit_time.tzinfo is None:
                exit_time = et_tz.localize(exit_time)
        else:
            exit_time = trade.get('exit_time', datetime.now(et_tz))
            if exit_time.tzinfo is None:
                exit_time = et_tz.localize(exit_time)
            
        return (exit_time - entry_time).total_seconds() / 60
    except Exception as e:
        print(f"Error calculating hold time: {str(e)}")
        return 0

def calculate_minutes_to_market_close() -> int:
    """Calculate minutes until market close (4 PM ET)"""
    now = datetime.now()
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    remaining_minutes = int((market_close - now).total_seconds() / 60)
    return max(1, remaining_minutes)

if __name__ == "__main__":
    test_paper_trading()