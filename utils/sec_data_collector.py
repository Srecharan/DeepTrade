# utils/sec_data_collector.py
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Union
import numpy as np

class SECDataCollector:
    def __init__(self):
        # Known CIK numbers for major companies
        self.cik_lookup = {
            'AAPL': '0000320193',
            'MSFT': '0000789019',
            'NVDA': '0001045810',
            'GOOGL': '0001652044',
            'META': '0001326801',
            'AMZN': '0001018724',
            'TSLA': '0001318605',
            'GME': '0001326380',
            'AMD': '0000002488',
            'JNJ': '0000200406'
        }
        
        self.headers = {
            'User-Agent': 'Stock Prediction Research Project (srecharan@gmail.com)',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        }
        
        self.base_url = "https://data.sec.gov"
    
    def get_cik(self, symbol: str) -> Optional[str]:
        if symbol in self.cik_lookup:
            return self.cik_lookup[symbol]
            
        try:
            time.sleep(0.2)  # Rate limiting
            url = f"{self.base_url}/files/company_tickers.json"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                companies = response.json()
                for entry in companies.values():
                    if entry['ticker'] == symbol:
                        cik = str(entry['cik_str']).zfill(10)
                        self.cik_lookup[symbol] = cik  # Cache for future use
                        return cik
            return None
            
        except Exception as e:
            print(f"Error in CIK lookup for {symbol}: {e}")
            return None

    def get_company_facts(self, symbol: str) -> Optional[Dict]:
        """Get company facts from SEC"""
        cik = self.get_cik(symbol)
        if not cik:
            print(f"No CIK found for {symbol}")
            return None
            
        try:
            time.sleep(0.2)  
            url = f"{self.base_url}/api/xbrl/companyfacts/CIK{cik}.json"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error {response.status_code} getting company facts for {symbol}")
                return None
                
        except Exception as e:
            print(f"Error fetching company facts for {symbol}: {e}")
            return None

    def get_recent_filings(self, symbol: str, days_back: int = 90) -> List[Dict]:
        """Get recent SEC filings"""
        cik = self.get_cik(symbol)
        if not cik:
            return []
            
        try:
            time.sleep(0.2)  
            url = f"{self.base_url}/submissions/CIK{cik}.json"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                print(f"Error {response.status_code} getting filings for {symbol}")
                return []
                
            data = response.json()
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            filings = []
            recent = data.get('filings', {}).get('recent', {})
            if not recent:
                return []
                
            for i, date in enumerate(recent.get('filingDate', [])):
                if date >= cutoff_date:
                    filings.append({
                        'date': date,
                        'form': recent['form'][i],
                        'description': recent.get('primaryDocDescription', [''])[i],
                        'file_num': recent.get('fileNumber', [''])[i]
                    })
            
            return filings
            
        except Exception as e:
            print(f"Error fetching filings for {symbol}: {e}")
            return []

    def analyze_filings(self, symbol: str) -> Dict:
        """Analyze recent filings and financial data"""
        filings = self.get_recent_filings(symbol)
        facts = self.get_company_facts(symbol)
        
        analysis = {
            'filing_count': len(filings),
            'sentiment_score': 0.0,
            'confidence': 0.0,
            'metrics': {},
            'latest_filing_date': None,
            'important_forms': []
        }
        
        if filings:
            analysis['latest_filing_date'] = max(f['date'] for f in filings)
            
            # Track important form types
            form_counts = {}
            for filing in filings:
                form_type = filing['form']
                form_counts[form_type] = form_counts.get(form_type, 0) + 1
                
                if form_type in ['10-K', '10-Q', '8-K']:
                    analysis['important_forms'].append({
                        'type': form_type,
                        'date': filing['date']
                    })
            
            # Calculate basic sentiment based on filing patterns
            analysis['sentiment_score'] = self._calculate_filing_sentiment(filings)
            analysis['confidence'] = min(len(filings) / 10, 1.0)  # More filings = more confidence
        
        if facts:
            analysis['metrics'] = self._extract_key_metrics(facts)
        
        return analysis

    def _calculate_filing_sentiment(self, filings: List[Dict]) -> float:
        """Calculate sentiment score from filing patterns"""
        sentiment = 0.0
        
        # More recent filings get higher weight
        for i, filing in enumerate(sorted(filings, key=lambda x: x['date'], reverse=True)):
            weight = 1.0 / (i + 1)  # Decreasing weights for older filings
            
            # Different forms have different sentiment impacts
            if filing['form'] == '8-K':  # Current reports - could be good or bad news
                sentiment += weight * 0.1
            elif filing['form'] in ['10-Q', '10-K']:  # Regular reporting is good
                sentiment += weight * 0.2
            elif filing['form'] == '4':  # Insider trading forms
                sentiment += weight * 0.05
        
        return np.clip(sentiment, -1, 1)

    def _extract_key_metrics(self, facts: Dict) -> Dict:
        """Extract key financial metrics from company facts"""
        metrics = {}
        
        try:
            us_gaap = facts.get('facts', {}).get('us-gaap', {})
            
            # Key metrics to extract
            metric_keys = [
                'Assets',
                'Liabilities',
                'StockholdersEquity',
                'Revenues',
                'NetIncomeLoss'
            ]
            
            for key in metric_keys:
                if key in us_gaap:
                    values = us_gaap[key].get('units', {}).get('USD', [])
                    if values:
                        # Get most recent value
                        recent = max(values, key=lambda x: x['end'])
                        metrics[key] = recent['val']
        
        except Exception as e:
            print(f"Error extracting metrics: {e}")
        
        return metrics

def main():
    """Test SEC data collection"""
    collector = SECDataCollector()
    symbols = ['AAPL', 'MSFT', 'NVDA']
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Analyzing {symbol}")
        print(f"{'='*50}")
        
        # Get CIK
        cik = collector.get_cik(symbol)
        print(f"CIK: {cik}")
        
        # Get and analyze filings
        analysis = collector.analyze_filings(symbol)
        
        print(f"\nFiling Analysis:")
        print(f"Total Filings: {analysis['filing_count']}")
        print(f"Latest Filing: {analysis.get('latest_filing_date', 'None')}")
        print(f"Sentiment Score: {analysis['sentiment_score']:.3f}")
        print(f"Confidence: {analysis['confidence']:.3f}")
        
        if analysis['metrics']:
            print("\nKey Metrics:")
            for metric, value in analysis['metrics'].items():
                print(f"{metric}: ${value:,.2f}")
        
        if analysis['important_forms']:
            print("\nImportant Recent Filings:")
            for form in analysis['important_forms'][:3]:
                print(f"- {form['date']}: {form['type']}")
        
        time.sleep(0.5)  # Respect SEC rate limits

if __name__ == "__main__":
    main()