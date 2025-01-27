from utils.stock_manager import StockManager
from utils.sentiment_manager import SentimentDataManager
from utils.reddit_sentiment import EnhancedRedditAnalyzer
from utils.sec_data_collector import SECDataCollector
from datetime import datetime
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from utils.config import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT
)

def visualize_sentiment_analysis(df: pd.DataFrame, symbols: List[str]):
    """Create comprehensive sentiment visualization"""
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Create GridSpec for multiple plots
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # Plot 1: Multi-source Sentiment Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(symbols))
    width = 0.25
    
    ax1.bar(x - width, df['News Sentiment'], width, label='News', color='skyblue')
    ax1.bar(x, df['Reddit Sentiment'], width, label='Reddit', color='lightgreen')
    ax1.bar(x + width, df['SEC Sentiment'], width, label='SEC', color='salmon')
    
    ax1.set_ylabel('Sentiment Score')
    ax1.set_title('Sentiment Analysis by Source')
    ax1.set_xticks(x)
    ax1.set_xticklabels(symbols)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Combined Sentiment Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    sentiment_data = df[['News Sentiment', 'Reddit Sentiment', 'SEC Sentiment']].values
    sns.heatmap(sentiment_data.T, 
                annot=True, 
                cmap='RdYlGn', 
                center=0,
                yticklabels=['News', 'Reddit', 'SEC'],
                xticklabels=symbols,
                ax=ax2)
    ax2.set_title('Sentiment Heatmap')
    
    # Plot 3: Confidence vs Filing Count
    ax3 = fig.add_subplot(gs[1, 0])
    scatter = ax3.scatter(df['Filing Count'], 
                         df['Combined Score'],
                         s=df['Confidence']*200,
                         alpha=0.6,
                         c=df['Combined Score'],
                         cmap='RdYlGn')
    
    for i, symbol in enumerate(symbols):
        ax3.annotate(symbol, 
                    (df['Filing Count'].iloc[i], df['Combined Score'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    ax3.set_xlabel('Number of SEC Filings')
    ax3.set_ylabel('Combined Sentiment Score')
    ax3.set_title('Sentiment Score vs Filing Activity')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Sentiment Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    sns.boxplot(data=df[['News Sentiment', 'Reddit Sentiment', 'SEC Sentiment']], ax=ax4)
    ax4.set_title('Sentiment Distribution by Source')
    ax4.set_ylabel('Sentiment Score')
    
    plt.tight_layout()
    plt.savefig('visualization/sentiment_analysis.png')
    print("\nSentiment visualization saved as 'sentiment_analysis.png'")

def analyze_sentiment_impact():
    """Analyze and explain sentiment impact on predictions"""
    print("\n" + "="*50)
    print("SENTIMENT IMPACT ANALYSIS")
    print("="*50)
    
    print("\n1. Sentiment Integration Pipeline:")
    print("   - FinBERT Model: Fine-tuned BERT for financial text")
    print("   - Multi-source fusion: News (40%), Reddit (30%), SEC (30%)")
    print("   - Real-time sentiment streaming and caching")
    
    print("\n2. Feature Engineering:")
    print("   - Raw sentiment smoothing with 3-day moving average")
    print("   - Sentiment momentum indicators")
    print("   - Volume-weighted sentiment signals")
    print("   - Trend strength measurement")
    
    print("\n3. Trading Impact Metrics:")
    print("   - Sentiment-price correlation analysis")
    print("   - Sentiment regime detection")
    print("   - Entry/exit signal modification")
    
def test_comprehensive_sentiment():
    # Initialize all components
    stock_manager = StockManager()
    sentiment_manager = SentimentDataManager()
    sec_collector = SECDataCollector()
    reddit_analyzer = EnhancedRedditAnalyzer(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    
    # Symbols to analyze
    symbols = ['NVDA', 'AAPL', 'MSFT', 'GME', 'AMD', 'JNJ', 'META', 'GOOGL', 'AMZN']
    
    # Store results for each source
    results = []
    
    print("\n" + "="*50)
    print("COMPREHENSIVE SENTIMENT ANALYSIS")
    print("="*50)
    
    for symbol in symbols:
        try:
            print(f"\n{'-'*20} Analyzing {symbol} {'-'*20}")
            
            # 1. News Sentiment
            print("\n1. News Sentiment Analysis:")
            news_sentiment = stock_manager.analyze_sentiment(symbol)
            
            # 2. Reddit Sentiment
            print("\n2. Reddit Sentiment Analysis:")
            reddit_df = reddit_analyzer.analyze_stock_sentiment(symbol)
            reddit_sentiment = reddit_df['total_sentiment'].mean() if reddit_df is not None else 0
            
            if reddit_df is not None:
                print("\nTop Reddit Posts:")
                top_posts = reddit_df.nlargest(3, 'engagement')
                for _, post in top_posts.iterrows():
                    print(f"- {post['title'][:100]}...")
                    print(f"  Sentiment: {post['total_sentiment']:.2f}")
                    print(f"  Engagement: {post['engagement']}")
            
            # 3. SEC Sentiment
            print("\n3. SEC Filings Analysis:")
            sec_analysis = sec_collector.analyze_filings(symbol)
            
            if sec_analysis['important_forms']:
                print("\nRecent Important Filings:")
                for form in sec_analysis['important_forms'][:3]:
                    print(f"- {form['date']}: {form['type']}")
            
            # 4. Combined Sentiment
            combined = stock_manager.get_combined_sentiment(symbol)
            
            # Store all results
            results.append({
                'Symbol': symbol,
                'News Sentiment': news_sentiment,
                'Reddit Sentiment': reddit_sentiment,
                'SEC Sentiment': sec_analysis['sentiment_score'],
                'Combined Score': combined['combined_score'],
                'Filing Count': sec_analysis['filing_count'],
                'Confidence': sec_analysis['confidence']
            })
            
            # 5. Store in Sentiment Manager
            sentiment_manager.store_daily_sentiment(
                symbol,
                combined['combined_score'],
                []  # Headlines stored separately
            )
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
            continue
    
    # Print summary table
    if results:
        df = pd.DataFrame(results)
        df = df.round(3)
        
        # Print summary table
        print("\n" + "="*50)
        print("SENTIMENT ANALYSIS SUMMARY")
        print("="*50)
        
        print("\n" + tabulate(df, headers='keys', tablefmt='grid', showindex=False))
        
        # Additional Statistics
        print("\nSentiment Statistics:")
        print("-" * 30)
        print(f"Most Positive: {df.loc[df['Combined Score'].idxmax(), 'Symbol']} ({df['Combined Score'].max():.3f})")
        print(f"Most Negative: {df.loc[df['Combined Score'].idxmin(), 'Symbol']} ({df['Combined Score'].min():.3f})")
        print(f"Average Sentiment: {df['Combined Score'].mean():.3f}")
        print(f"Sentiment Volatility: {df['Combined Score'].std():.3f}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df.to_csv(f'data/sentiment/sentiment_analysis_{timestamp}.csv', index=False)
        print(f"\nResults saved to: sentiment_analysis_{timestamp}.csv")
        
        return df
    return None

if __name__ == "__main__":
    # Run the main analysis
    results_df = test_comprehensive_sentiment()
    
    # Only proceed with visualization if we have results
    if results_df is not None:
        symbols = ['NVDA', 'AAPL', 'MSFT', 'GME', 'AMD', 'JNJ', 'META', 'GOOGL', 'AMZN']
        visualize_sentiment_analysis(results_df, symbols)
        analyze_sentiment_impact()