# utils/reddit_sentiment.py
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
        # Company name mappings
        self.company_names = {
            'AAPL': ['Apple', 'AAPL', 'iPhone'],
            'MSFT': ['Microsoft', 'MSFT', 'Azure'],
            'NVDA': ['NVIDIA', 'NVDA', 'GeForce'],
            'META': ['Meta', 'Facebook', 'FB', 'Instagram', 'WhatsApp'],
            'GOOGL': ['Google', 'Alphabet', 'GOOGL', 'Chrome', 'Android'],
            'AMZN': ['Amazon', 'AMZN', 'AWS', 'Prime']
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
            # Get post content
            full_text = f"{post.title} {post.selftext}"
            
            # Analyze post sentiment
            post_sentiment = self.get_text_sentiment(full_text)
            
            # Get top comments
            post.comments.replace_more(limit=0)
            top_comments = list(post.comments)[:5]
            
            # Analyze comment sentiment
            comment_sentiments = []
            for comment in top_comments:
                sentiment = self.get_text_sentiment(comment.body)
                comment_sentiments.append(sentiment)
            
            avg_comment_sentiment = sum(comment_sentiments) / len(comment_sentiments) if comment_sentiments else 0
            
            # Calculate engagement score
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
        total_posts = 0
        
        print(f"\nAnalyzing sentiment for {symbol}...")
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                posts_analyzed = 0
                
                for term in search_terms:
                    # Search for posts
                    for post in subreddit.search(term, time_filter='week', sort='hot', limit=5):
                        analysis = self.analyze_post(post, symbol)
                        if analysis:
                            analysis['subreddit'] = subreddit_name
                            analysis['search_term'] = term
                            sentiment_data.append(analysis)
                            posts_analyzed += 1
                    
                    time.sleep(2)  # Respect rate limits
                
                if posts_analyzed > 0:
                    total_posts += posts_analyzed
                
            except Exception as e:
                print(f"Error in subreddit {subreddit_name}: {str(e)}")
                continue
        
        if sentiment_data:
            df = pd.DataFrame(sentiment_data)
            avg_sentiment = df['total_sentiment'].mean()
            print(f"Analyzed {total_posts} posts across {len(subreddits)} subreddits")
            print(f"Average Sentiment: {avg_sentiment:.3f}")
            return df
        return None