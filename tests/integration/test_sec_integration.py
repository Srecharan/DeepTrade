# test_sec_integration.py
from utils.sec_data_collector import SECDataCollector
import pandas as pd
from datetime import datetime
import time

def test_sec_integration():
    """Test SEC data collection and analysis"""
    collector = SECDataCollector()
    
    # Test with your existing stock symbols
    symbols = ['AAPL', 'MSFT', 'NVDA']
    
    results = {}
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Analyzing {symbol} SEC Data")
        print(f"{'='*50}")
        
        # Get CIK number
        cik = collector.get_cik(symbol)
        print(f"CIK Number: {cik}")
        
        # Get comprehensive SEC data
        analysis = collector.analyze_filings(symbol)
        
        # Print detailed summary
        print(f"\nFiling Analysis:")
        print(f"Total Filings: {analysis['filing_count']}")
        if analysis['latest_filing_date']:
            print(f"Latest Filing Date: {analysis['latest_filing_date']}")
        
        print(f"\nSentiment Analysis:")
        print(f"Sentiment Score: {analysis['sentiment_score']:.3f}")
        print(f"Confidence: {analysis['confidence']:.3f}")
        
        if analysis['metrics']:
            print(f"\nKey Financial Metrics:")
            for metric, value in analysis['metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"- {metric}: ${value:,.2f}")
                else:
                    print(f"- {metric}: {value}")
        
        if analysis['important_forms']:
            print(f"\nRecent Important Filings:")
            for form in analysis['important_forms'][:3]:  # Show latest 3
                print(f"- {form['date']}: {form['type']}")
        
        # Store results for later use
        results[symbol] = {
            'sec_sentiment': analysis['sentiment_score'],
            'sec_confidence': analysis['confidence'],
            'filing_count': analysis['filing_count'],
            'metrics': analysis['metrics']
        }
        
        # Respect SEC rate limits
        time.sleep(1)
    
    return results

def print_integration_suggestions(results):
    """Print suggestions for integrating SEC data into prediction system"""
    print("\nPotential Integration with Prediction System:")
    print("============================================")
    
    for symbol, data in results.items():
        print(f"\n{symbol}:")
        print(f"SEC Sentiment Score: {data['sec_sentiment']:.3f}")
        print(f"Filing Count: {data['filing_count']}")
        print(f"Confidence: {data['sec_confidence']:.3f}")
        
        print("\nSuggested Integration Points:")
        print("1. Fundamental Analysis:")
        for metric, value in data['metrics'].items():
            if isinstance(value, (int, float)):
                print(f"   - Use {metric} trend as a baseline factor")
        
        print("2. Risk Adjustment:")
        print(f"   - Adjust confidence intervals based on filing frequency")
        print(f"   - Use SEC sentiment to modify prediction weights")
        
        print("3. Long-term Indicators:")
        print("   - Track fundamental metrics for trend analysis")
        print("   - Use filing patterns for volatility prediction")

if __name__ == "__main__":
    results = test_sec_integration()
    print_integration_suggestions(results)