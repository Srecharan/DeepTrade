# utils/realtime_manager.py
import websocket
import json
import threading
from datetime import datetime, time
import pytz
from typing import Dict, Callable
import pandas as pd

class RealtimeDataManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws = None
        self.est_tz = pytz.timezone('US/Eastern')
        self.latest_data = {}
        self.callbacks = []
        
    def is_market_hours(self) -> bool:
        """Check if it's currently market hours (9:30 AM - 4:00 PM EST)"""
        now = datetime.now(self.est_tz)
        market_start = time(9, 30)
        market_end = time(16, 0)
        return (now.weekday() < 5 and  # Monday to Friday
                market_start <= now.time() <= market_end)
    
    def start_streaming(self, symbols: list):
        """Start websocket connection for real-time data"""
        def on_message(ws, message):
            data = json.loads(message)
            if 'data' in data:
                for tick in data['data']:
                    symbol = tick['s']
                    self.latest_data[symbol] = {
                        'price': float(tick['p']),
                        'volume': int(tick['v']),
                        'timestamp': pd.Timestamp.now()
                    }
                    # Notify callbacks
                    for callback in self.callbacks:
                        callback(symbol, self.latest_data[symbol])

        def on_error(ws, error):
            print(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            print("WebSocket connection closed")
            # Attempt to reconnect if during market hours
            if self.is_market_hours():
                self.start_streaming(symbols)

        def on_open(ws):
            print("WebSocket connection opened")
            # Subscribe to ticker updates
            subscribe_message = {
                "action": "subscribe",
                "params": f"T.{','.join(symbols)}"
            }
            ws.send(json.dumps(subscribe_message))

        if self.is_market_hours():
            websocket.enableTrace(True)
            self.ws = websocket.WebSocketApp(
                f"wss://stream.data.alpaca.markets/v2/iex",
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open,
                header={"Authorization": f"Bearer {self.api_key}"}
            )
            
            # Start WebSocket connection in a separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
        
    def register_callback(self, callback: Callable):
        """Register a callback for real-time data updates"""
        self.callbacks.append(callback)
        
    def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol"""
        if symbol in self.latest_data:
            return self.latest_data[symbol]['price']
        return None
        
    def stop_streaming(self):
        """Stop websocket connection"""
        if self.ws:
            self.ws.close()