"""
Kafka streaming pipeline for financial data ingestion
"""

import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import logging

logger = logging.getLogger(__name__)

class KafkaStreamingPipeline:
    def __init__(self, config_path: str = "config/kafka_config.json", simulation_mode: bool = False):
        self.config_path = config_path
        self.simulation_mode = simulation_mode
        self.config = self._load_config()
        self.producer = None
        self.consumers = {}
        self.event_count = 0
        
        if not simulation_mode:
            self._initialize_kafka()
    
    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "bootstrap_servers": ["localhost:9092"],
            "topics": {
                "financial-news": {"partitions": 3, "replication_factor": 1},
                "social-sentiment": {"partitions": 3, "replication_factor": 1},
                "regulatory-filings": {"partitions": 2, "replication_factor": 1},
                "market-data": {"partitions": 4, "replication_factor": 1}
            },
            "producer": {
                "batch_size": 16384,
                "linger_ms": 10,
                "compression_type": "snappy",
                "acks": "all"
            },
            "consumer": {
                "group_id": "deeptrade-consumer-group",
                "auto_offset_reset": "latest",
                "enable_auto_commit": True
            },
            "serialization": {
                "format": "avro",
                "schema_registry_url": "http://localhost:8081"
            }
        }
    
    def _initialize_kafka(self):
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.config["bootstrap_servers"],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                **self.config["producer"]
            )
            logger.info("Kafka producer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    def produce_event(self, topic: str, event_data: Dict[str, Any]) -> bool:
        if self.simulation_mode:
            return self.simulate_event_production(topic, event_data)
        
        try:
            future = self.producer.send(topic, event_data)
            record_metadata = future.get(timeout=10)
            self.event_count += 1
            
            logger.debug(f"Event sent to {topic}: partition {record_metadata.partition}, offset {record_metadata.offset}")
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to send event to {topic}: {e}")
            return False
    
    def simulate_event_production(self, topic: str, event_data: Dict[str, Any]) -> bool:
        self.event_count += 1
        logger.debug(f"Simulated event production to {topic}")
        return True
    
    def create_consumer(self, topics: List[str], group_id: str = None) -> KafkaConsumer:
        if self.simulation_mode:
            logger.info(f"Simulated consumer created for topics: {topics}")
            return None
        
        consumer_config = self.config["consumer"].copy()
        if group_id:
            consumer_config["group_id"] = group_id
        
        consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=self.config["bootstrap_servers"],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            **consumer_config
        )
        
        self.consumers[group_id or consumer_config["group_id"]] = consumer
        return consumer
    
    def process_financial_news(self, source: str = "NewsAPI"):
        """Process financial news events from various sources"""
        news_events = []
        
        if source == "NewsAPI":
            news_events = self._fetch_news_api_events()
        elif source == "Bloomberg":
            news_events = self._fetch_bloomberg_events()
        elif source == "Reuters":
            news_events = self._fetch_reuters_events()
        
        for event in news_events:
            enriched_event = self._enrich_news_event(event)
            self.produce_event("financial-news", enriched_event)
        
        return len(news_events)
    
    def process_social_sentiment(self, source: str = "Reddit"):
        """Process social sentiment data from Reddit, Twitter, etc."""
        sentiment_events = []
        
        if source == "Reddit":
            sentiment_events = self._fetch_reddit_sentiment()
        elif source == "Twitter":
            sentiment_events = self._fetch_twitter_sentiment()
        elif source == "StockTwits":
            sentiment_events = self._fetch_stocktwits_sentiment()
        
        for event in sentiment_events:
            enriched_event = self._enrich_sentiment_event(event)
            self.produce_event("social-sentiment", enriched_event)
        
        return len(sentiment_events)
    
    def process_regulatory_filings(self):
        """Process SEC EDGAR filings"""
        filings = self._fetch_sec_filings()
        
        for filing in filings:
            enriched_filing = self._enrich_filing_event(filing)
            self.produce_event("regulatory-filings", enriched_filing)
        
        return len(filings)
    
    def process_market_data(self, source: str = "YahooFinance"):
        """Process real-time market data"""
        market_events = []
        
        if source == "YahooFinance":
            market_events = self._fetch_yahoo_market_data()
        elif source == "AlphaVantage":
            market_events = self._fetch_alpha_vantage_data()
        
        for event in market_events:
            self.produce_event("market-data", event)
        
        return len(market_events)
    
    def _fetch_news_api_events(self) -> List[Dict[str, Any]]:
        if self.simulation_mode:
            return [
                {
                    "symbol": "AAPL",
                    "headline": "Apple reports strong Q4 earnings",
                    "content": "Apple Inc. exceeded analyst expectations...",
                    "source": "NewsAPI",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        # Real NewsAPI integration would go here
        return []
    
    def _fetch_reddit_sentiment(self) -> List[Dict[str, Any]]:
        if self.simulation_mode:
            return [
                {
                    "symbol": "TSLA",
                    "text": "Tesla looking bullish after Shanghai factory news",
                    "sentiment": "positive",
                    "score": 0.8,
                    "source": "Reddit",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        # Real Reddit API integration would go here
        return []
    
    def _fetch_sec_filings(self) -> List[Dict[str, Any]]:
        if self.simulation_mode:
            return [
                {
                    "symbol": "GOOGL",
                    "filing_type": "10-K",
                    "filing_date": "2024-12-26",
                    "url": "https://sec.gov/filing/example",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        # Real SEC EDGAR integration would go here
        return []
    
    def _fetch_yahoo_market_data(self) -> List[Dict[str, Any]]:
        if self.simulation_mode:
            return [
                {
                    "symbol": "AMZN",
                    "price": 185.50,
                    "volume": 1000000,
                    "change": 2.5,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        # Real Yahoo Finance integration would go here
        return []
    
    def _enrich_news_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        event["event_type"] = "financial_news"
        event["processing_timestamp"] = datetime.now().isoformat()
        return event
    
    def _enrich_sentiment_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        event["event_type"] = "social_sentiment"
        event["processing_timestamp"] = datetime.now().isoformat()
        return event
    
    def _enrich_filing_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        event["event_type"] = "regulatory_filing"
        event["processing_timestamp"] = datetime.now().isoformat()
        return event
    
    def start_streaming(self, target_events_per_day: int = 9000):
        """Start the streaming pipeline"""
        events_per_second = target_events_per_day / (24 * 60 * 60)  # ~0.1 events/sec for 9K/day
        
        logger.info(f"Starting streaming pipeline: {target_events_per_day} events/day target")
        
        def streaming_worker():
            while True:
                # Process different data sources
                self.process_financial_news("NewsAPI")
                self.process_social_sentiment("Reddit")
                self.process_regulatory_filings()
                self.process_market_data("YahooFinance")
                
                time.sleep(1.0 / events_per_second)
        
        streaming_thread = threading.Thread(target=streaming_worker, daemon=True)
        streaming_thread.start()
        
        return streaming_thread
    
    def get_throughput_stats(self) -> Dict[str, Any]:
        return {
            "total_events": self.event_count,
            "events_per_minute": self.event_count,  # Simplified calculation
            "topics_active": len(self.config["topics"]),
            "simulation_mode": self.simulation_mode
        }
    
    def validate_configuration(self) -> bool:
        required_keys = ["bootstrap_servers", "topics", "producer", "consumer"]
        return all(key in self.config for key in required_keys)
    
    def get_topics_simulation(self) -> List[str]:
        return list(self.config["topics"].keys())
    
    def close(self):
        if self.producer:
            self.producer.close()
        
        for consumer in self.consumers.values():
            if consumer:
                consumer.close()
        
        logger.info("Kafka streaming pipeline closed") 