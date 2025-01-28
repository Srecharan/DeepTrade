# enhanced_reddit_sentiment.py
import praw
import pandas as pd
from datetime import datetime, timedelta
import time
from textblob import TextBlob

class EnhancedRedditAnalyzer:
    def __init__(self, client_id, client_secret, user_agent):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.company_names = {
            'AAPL': ['Apple', 'AAPL', 'iPhone'],
            'MSFT': ['Microsoft', 'MSFT', 'Azure'],
            'NVDA': ['NVIDIA', 'NVDA', 'GeForce']
        }
    
    def get_text_sentiment(self, text):
        """Analyze text sentiment using TextBlob"""
        try:
            return TextBlob(text).sentiment.polarity
        except:
            return 0.0

    def analyze_post(self, post, symbol):
        """Detailed analysis of a single post"""
        try:
            full_text = f"{post.title} {post.selftext}"

            post_sentiment = self.get_text_sentiment(full_text)
            post.comments.replace_more(limit=0)
            top_comments = list(post.comments)[:5]

            comment_sentiments = []
            for comment in top_comments:
                sentiment = self.get_text_sentiment(comment.body)
                comment_sentiments.append(sentiment)
            
            avg_comment_sentiment = sum(comment_sentiments) / len(comment_sentiments) if comment_sentiments else 0
            
            engagement = post.score + post.num_comments
            upvote_ratio = post.upvote_ratio if hasattr(post, 'upvote_ratio') else 0.5
            
            # Combined sentiment score
            total_sentiment = (
                post_sentiment * 0.4 +          # Post content weight
                avg_comment_sentiment * 0.3 +   # Comments weight
                (upvote_ratio - 0.5) * 0.3      # Community reception weight
            )
            
            return {
                'title': post.title,
                'created_utc': datetime.fromtimestamp(post.created_utc),
                'score': post.score,
                'comments': post.num_comments,
                'engagement': engagement,
                'upvote_ratio': upvote_ratio,
                'post_sentiment': post_sentiment,
                'comment_sentiment': avg_comment_sentiment,
                'total_sentiment': total_sentiment
            }
        except Exception as e:
            print(f"Error analyzing post: {str(e)}")
            return None

    def analyze_stock_sentiment(self, symbol, days_back=3):
        """Analyze sentiment for a stock across multiple subreddits"""
        search_terms = self.company_names.get(symbol, [symbol])
        subreddits = ['wallstreetbets', 'stocks', 'investing']
        sentiment_data = []
        
        print(f"\nAnalyzing sentiment for {symbol} using terms: {search_terms}")
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                print(f"\nSearching r/{subreddit_name}...")
                
                for term in search_terms:
                    for post in subreddit.search(term, time_filter='week', sort='hot', limit=5):
                        analysis = self.analyze_post(post, symbol)
                        if analysis:
                            analysis['subreddit'] = subreddit_name
                            analysis['search_term'] = term
                            sentiment_data.append(analysis)
                            
                            print(f"Found post: {post.title[:100]}...")
                            print(f"Sentiment: {analysis['total_sentiment']:.2f}, "
                                  f"Engagement: {analysis['engagement']}")
                
                time.sleep(2)  
                
            except Exception as e:
                print(f"Error in subreddit {subreddit_name}: {str(e)}")
                continue
        
        return pd.DataFrame(sentiment_data) if sentiment_data else None

def main():
    """Test the enhanced Reddit sentiment analyzer"""
    CLIENT_ID = "iBwT-MV4tfjKRqsWgNH9tg"
    CLIENT_SECRET = "CHvuOU6N_xNmKt7n9yICtVQBz6LKlQ"
    USER_AGENT = "script:stock_sentiment:v1.0 (by /u/srecharan)"
    
    analyzer = EnhancedRedditAnalyzer(CLIENT_ID, CLIENT_SECRET, USER_AGENT)
    
    # Test with stocks
    for symbol in ['AAPL', 'MSFT', 'NVDA']:
        print(f"\n{'='*50}")
        print(f"Analyzing {symbol}")
        print(f"{'='*50}")
        
        df = analyzer.analyze_stock_sentiment(symbol)
        
        if df is not None and not df.empty:
            print(f"\nSummary for {symbol}:")
            print(f"Total posts analyzed: {len(df)}")
            print(f"Average sentiment: {df['total_sentiment'].mean():.3f}")
            print(f"Total engagement: {df['engagement'].sum():,}")
            
            # Show most positive and negative posts
            most_positive = df.loc[df['total_sentiment'].idxmax()]
            print(f"\nMost positive post:")
            print(f"Title: {most_positive['title']}")
            print(f"Sentiment: {most_positive['total_sentiment']:.3f}")
            print(f"Engagement: {most_positive['engagement']:,}")
            
            most_negative = df.loc[df['total_sentiment'].idxmin()]
            print(f"\nMost negative post:")
            print(f"Title: {most_negative['title']}")
            print(f"Sentiment: {most_negative['total_sentiment']:.3f}")
            print(f"Engagement: {most_negative['engagement']:,}")
        else:
            print(f"No data found for {symbol}")
        
        time.sleep(3)

if __name__ == "__main__":
    main()