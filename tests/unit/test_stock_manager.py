from utils.stock_manager import StockManager
import matplotlib.pyplot as plt
import numpy as np

def test_stock_predictions():
    # Initialize StockManager
    manager = StockManager()
    symbols = ['NVDA', 'AAPL', 'MSFT']
    
    for symbol in symbols:
        print(f"\n=== Analyzing {symbol} ===")
        try:
            headlines = manager._get_recent_headlines(symbol, 7)
            print(f"\nRecent Headlines for {symbol}:")
            for i, headline in enumerate(headlines[:5], 1):
                print(f"{i}. {headline}")
            data, sentiment = manager.prepare_prediction_data(symbol)
            
            print(f"Sentiment score: {sentiment:.2f}")
            print(f"Technical Indicators:")
            print(f"- RSI: {data['RSI'].iloc[-1]:.2f}")
            print(f"- MACD: {data['MACD'].iloc[-1]:.2f}")
            print(f"- SMA_20: {data['SMA_20'].iloc[-1]:.2f}")
            print(f"Data shape: {data.shape}")

            plt.plot(data.index, data['Close'], label=symbol)

            print(f"Latest RSI: {data['RSI'].iloc[-1]:.2f}")
            print(f"Latest MACD: {data['MACD'].iloc[-1]:.2f}")
            print(f"Latest SMA_20: {data['SMA_20'].iloc[-1]:.2f}")
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    plt.title('Stock Price Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_stock_predictions()