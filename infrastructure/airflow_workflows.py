"""
Airflow workflows for FinBERT sentiment analysis
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import logging
import pandas as pd
import numpy as np

# Airflow imports
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.s3_key_sensor import S3KeySensor
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable
from airflow.hooks.postgres_hook import PostgresHook
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

# Add project path for imports
sys.path.append('/opt/airflow/dags/deeptrade')
from utils.sentiment_analyzer import SentimentAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

default_args = {
    'owner': 'deeptrade-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2)
}

class FinBERTProcessor:
    
    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        
    def load_finbert_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"FinBERT model loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            return False
    
    def process_text_batch(self, texts: List[str]) -> List[Dict]:
        """Process batch of texts through FinBERT"""
        results = []
        
        for text in texts:
            try:
                # Tokenize input
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(self.device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Extract sentiment scores
                sentiment_scores = predictions.cpu().numpy()[0]
                
                # Map to sentiment labels (negative, neutral, positive)
                sentiment_labels = ['negative', 'neutral', 'positive']
                sentiment_dict = {
                    label: float(score) for label, score in zip(sentiment_labels, sentiment_scores)
                }
                
                # Determine primary sentiment
                primary_sentiment = sentiment_labels[np.argmax(sentiment_scores)]
                confidence = float(np.max(sentiment_scores))
                
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'sentiment': primary_sentiment,
                    'confidence': confidence,
                    'scores': sentiment_dict,
                    'processed_at': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'sentiment': 'neutral',
                    'confidence': 0.33,
                    'scores': {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33},
                    'error': str(e),
                    'processed_at': datetime.now().isoformat()
                })
        
        return results

# Initialize FinBERT processor
finbert_processor = FinBERTProcessor()

def check_data_availability(**context):
    """Check if new financial data is available for processing"""
    
    data_sources = ['kafka_events', 'news_data', 'social_data', 'sec_filings']
    available_sources = []
    
    for source in data_sources:
        # Simulate data availability check
        data_path = f"/opt/airflow/data/raw/{source}"
        if os.path.exists(data_path) or source in ['kafka_events']:  # Simulate availability
            available_sources.append(source)
            logger.info(f"Data available for {source}")
    
    # Store available sources in XCom
    context['task_instance'].xcom_push(key='available_sources', value=available_sources)
    
    return len(available_sources) > 0

def load_financial_texts(**context):
    """Load financial texts from multiple sources"""
    
    # Get available sources from previous task
    available_sources = context['task_instance'].xcom_pull(key='available_sources')
    
    all_texts = []
    source_metadata = {}
    
    # Sample financial texts for demonstration
    sample_texts = {
        'kafka_events': [
            "Apple Inc. reported strong quarterly earnings with revenue growth of 8%",
            "Microsoft's cloud services division sees continued expansion in enterprise market",
            "Tesla's stock price surged following better-than-expected delivery numbers",
            "Amazon Web Services maintains market leadership in cloud computing sector",
            "NVIDIA's AI chip demand drives record-breaking quarterly performance"
        ],
        'news_data': [
            "Federal Reserve hints at potential interest rate cuts amid economic uncertainty",
            "Tech sector faces regulatory scrutiny over data privacy practices",
            "Energy stocks rally as oil prices stabilize above $80 per barrel",
            "Healthcare companies announce breakthrough developments in gene therapy",
            "Financial institutions report improved loan portfolio performance"
        ],
        'social_data': [
            "Reddit discussion: $AAPL looking strong ahead of iPhone launch event",
            "Twitter sentiment: $TSLA bulls expecting production milestone announcement",
            "WallStreetBets: $NVDA to the moon with AI revolution momentum",
            "Stocktwits: $MSFT showing bullish technical patterns on charts",
            "Social sentiment: $AMZN prime day sales expected to beat estimates"
        ],
        'sec_filings': [
            "10-K filing: Company reports solid financial position with strong cash flow",
            "8-K filing: Executive leadership change announced with succession plan",
            "10-Q filing: Quarterly results show steady revenue growth trajectory",
            "DEF 14A: Shareholder proposals include ESG governance improvements",
            "Form 4: Insider trading activity shows executive confidence in outlook"
        ]
    }
    
    # Collect texts from available sources
    for source in available_sources:
        if source in sample_texts:
            texts = sample_texts[source]
            all_texts.extend(texts)
            source_metadata[source] = {
                'count': len(texts),
                'sample': texts[0] if texts else None
            }
    
    # Store texts and metadata in XCom
    context['task_instance'].xcom_push(key='financial_texts', value=all_texts)
    context['task_instance'].xcom_push(key='source_metadata', value=source_metadata)
    
    logger.info(f"Loaded {len(all_texts)} financial texts from {len(available_sources)} sources")
    return len(all_texts)

def preprocess_texts(**context):
    """Preprocess financial texts for FinBERT analysis"""
    
    # Get texts from previous task
    texts = context['task_instance'].xcom_pull(key='financial_texts')
    
    if not texts:
        logger.warning("No texts to preprocess")
        return 0
    
    processed_texts = []
    
    for text in texts:
        # Basic text preprocessing
        processed_text = text.strip()
        
        # Remove excessive whitespace
        processed_text = ' '.join(processed_text.split())
        
        # Ensure text length is reasonable for FinBERT
        if len(processed_text) > 512:
            processed_text = processed_text[:512]
        
        # Skip empty texts
        if len(processed_text.strip()) > 0:
            processed_texts.append(processed_text)
    
    # Store processed texts in XCom
    context['task_instance'].xcom_push(key='processed_texts', value=processed_texts)
    
    logger.info(f"Preprocessed {len(processed_texts)} texts")
    return len(processed_texts)

def run_finbert_analysis(**context):
    """Run FinBERT sentiment analysis on preprocessed texts"""
    
    # Get processed texts from previous task
    texts = context['task_instance'].xcom_pull(key='processed_texts')
    
    if not texts:
        logger.warning("No texts to analyze")
        return 0
    
    # Load FinBERT model
    if not finbert_processor.load_finbert_model():
        raise Exception("Failed to load FinBERT model")
    
    # Process texts in batches
    batch_size = 10
    all_results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_results = finbert_processor.process_text_batch(batch_texts)
        all_results.extend(batch_results)
        
        logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
    
    # Store results in XCom
    context['task_instance'].xcom_push(key='sentiment_results', value=all_results)
    
    logger.info(f"Completed FinBERT analysis on {len(all_results)} texts")
    return len(all_results)

def generate_sentiment_features(**context):
    """Generate 12 temporal sentiment indicators"""
    
    # Get sentiment results from previous task
    results = context['task_instance'].xcom_pull(key='sentiment_results')
    
    if not results:
        logger.warning("No sentiment results to process")
        return {}
    
    # Extract sentiment scores
    positive_scores = [r['scores']['positive'] for r in results]
    negative_scores = [r['scores']['negative'] for r in results]
    neutral_scores = [r['scores']['neutral'] for r in results]
    
    # Calculate temporal sentiment features
    sentiment_features = {
        # Raw sentiment aggregations
        'avg_positive_sentiment': np.mean(positive_scores),
        'avg_negative_sentiment': np.mean(negative_scores),
        'avg_neutral_sentiment': np.mean(neutral_scores),
        
        # Sentiment volatility measures
        'sentiment_volatility': np.std(positive_scores),
        'sentiment_range': max(positive_scores) - min(positive_scores),
        
        # Sentiment momentum indicators
        'positive_momentum': np.mean(positive_scores[-5:]) - np.mean(positive_scores[:5]) if len(positive_scores) >= 10 else 0,
        'negative_momentum': np.mean(negative_scores[-5:]) - np.mean(negative_scores[:5]) if len(negative_scores) >= 10 else 0,
        
        # Sentiment strength indicators
        'sentiment_strength': np.mean([max(r['scores'].values()) for r in results]),
        'sentiment_confidence': np.mean([r['confidence'] for r in results]),
        
        # Distribution indicators
        'positive_ratio': len([r for r in results if r['sentiment'] == 'positive']) / len(results),
        'negative_ratio': len([r for r in results if r['sentiment'] == 'negative']) / len(results),
        'neutral_ratio': len([r for r in results if r['sentiment'] == 'neutral']) / len(results)
    }
    
    # Store features in XCom
    context['task_instance'].xcom_push(key='sentiment_features', value=sentiment_features)
    
    logger.info(f"Generated {len(sentiment_features)} sentiment features")
    return sentiment_features

def store_sentiment_data(**context):
    """Store processed sentiment data and features"""
    
    # Get data from previous tasks
    sentiment_results = context['task_instance'].xcom_pull(key='sentiment_results')
    sentiment_features = context['task_instance'].xcom_pull(key='sentiment_features')
    source_metadata = context['task_instance'].xcom_pull(key='source_metadata')
    
    # Create output directory
    output_dir = "/opt/airflow/data/processed/sentiment"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for this run
    run_timestamp = context['execution_date'].strftime('%Y%m%d_%H%M%S')
    
    # Save sentiment results
    results_file = f"{output_dir}/sentiment_results_{run_timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(sentiment_results, f, indent=2, default=str)
    
    # Save sentiment features
    features_file = f"{output_dir}/sentiment_features_{run_timestamp}.json"
    with open(features_file, 'w') as f:
        json.dump(sentiment_features, f, indent=2, default=str)
    
    # Save processing metadata
    metadata = {
        'execution_date': context['execution_date'].isoformat(),
        'total_texts_processed': len(sentiment_results) if sentiment_results else 0,
        'features_generated': len(sentiment_features) if sentiment_features else 0,
        'source_metadata': source_metadata,
        'processing_duration': 'calculated_in_production',
        'model_version': 'ProsusAI/finbert'
    }
    
    metadata_file = f"{output_dir}/processing_metadata_{run_timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Stored sentiment data: {results_file}, {features_file}, {metadata_file}")
    
    return {
        'results_file': results_file,
        'features_file': features_file,
        'metadata_file': metadata_file
    }

def validate_sentiment_quality(**context):
    """Validate quality of sentiment analysis results"""
    
    sentiment_results = context['task_instance'].xcom_pull(key='sentiment_results')
    sentiment_features = context['task_instance'].xcom_pull(key='sentiment_features')
    
    if not sentiment_results or not sentiment_features:
        raise Exception("Missing sentiment data for validation")
    
    # Quality checks
    quality_metrics = {
        'total_texts': len(sentiment_results),
        'avg_confidence': np.mean([r['confidence'] for r in sentiment_results]),
        'min_confidence': min(r['confidence'] for r in sentiment_results),
        'errors_count': len([r for r in sentiment_results if 'error' in r]),
        'sentiment_distribution': {
            'positive': len([r for r in sentiment_results if r['sentiment'] == 'positive']),
            'negative': len([r for r in sentiment_results if r['sentiment'] == 'negative']),
            'neutral': len([r for r in sentiment_results if r['sentiment'] == 'neutral'])
        }
    }
    
    # Validation rules
    if quality_metrics['avg_confidence'] < 0.4:
        logger.warning(f"Low average confidence: {quality_metrics['avg_confidence']:.3f}")
    
    if quality_metrics['errors_count'] > len(sentiment_results) * 0.1:
        logger.warning(f"High error rate: {quality_metrics['errors_count']} errors")
    
    # Store quality metrics
    context['task_instance'].xcom_push(key='quality_metrics', value=quality_metrics)
    
    logger.info(f"Quality validation completed: {quality_metrics}")
    return quality_metrics

def send_completion_notification(**context):
    """Send notification upon successful completion"""
    
    quality_metrics = context['task_instance'].xcom_pull(key='quality_metrics')
    sentiment_features = context['task_instance'].xcom_pull(key='sentiment_features')
    
    notification_data = {
        'dag_id': context['dag'].dag_id,
        'execution_date': context['execution_date'].isoformat(),
        'texts_processed': quality_metrics.get('total_texts', 0),
        'avg_confidence': quality_metrics.get('avg_confidence', 0),
        'features_generated': len(sentiment_features) if sentiment_features else 0,
        'status': 'SUCCESS'
    }
    
    logger.info(f"Sentiment analysis completed successfully: {notification_data}")
    return notification_data

# Define the main DAG
dag = DAG(
    'finbert_sentiment_analysis',
    default_args=default_args,
    description='FinBERT sentiment analysis workflow for financial texts',
    schedule_interval=timedelta(hours=4),  # Run every 4 hours
    catchup=False,
    max_active_runs=1,
    tags=['sentiment', 'finbert', 'nlp', 'financial']
)

# Define tasks
with dag:
    
    # Start task
    start_task = DummyOperator(
        task_id='start_sentiment_pipeline',
        dag=dag
    )
    
    # Data availability check
    check_data_task = PythonOperator(
        task_id='check_data_availability',
        python_callable=check_data_availability,
        dag=dag
    )
    
    # Data ingestion group
    with TaskGroup("data_ingestion", dag=dag) as data_ingestion_group:
        
        load_texts_task = PythonOperator(
            task_id='load_financial_texts',
            python_callable=load_financial_texts,
            dag=dag
        )
        
        preprocess_task = PythonOperator(
            task_id='preprocess_texts',
            python_callable=preprocess_texts,
            dag=dag
        )
        
        load_texts_task >> preprocess_task
    
    # FinBERT analysis group
    with TaskGroup("finbert_analysis", dag=dag) as finbert_group:
        
        sentiment_analysis_task = PythonOperator(
            task_id='run_finbert_analysis',
            python_callable=run_finbert_analysis,
            dag=dag
        )
        
        feature_generation_task = PythonOperator(
            task_id='generate_sentiment_features',
            python_callable=generate_sentiment_features,
            dag=dag
        )
        
        sentiment_analysis_task >> feature_generation_task
    
    # Data storage and validation group
    with TaskGroup("storage_validation", dag=dag) as storage_group:
        
        store_data_task = PythonOperator(
            task_id='store_sentiment_data',
            python_callable=store_sentiment_data,
            dag=dag
        )
        
        validate_quality_task = PythonOperator(
            task_id='validate_sentiment_quality',
            python_callable=validate_sentiment_quality,
            dag=dag
        )
        
        store_data_task >> validate_quality_task
    
    # Notification task
    notify_completion_task = PythonOperator(
        task_id='send_completion_notification',
        python_callable=send_completion_notification,
        trigger_rule=TriggerRule.ALL_SUCCESS,
        dag=dag
    )
    
    # End task
    end_task = DummyOperator(
        task_id='end_sentiment_pipeline',
        trigger_rule=TriggerRule.ALL_DONE,
        dag=dag
    )
    
    # Define task dependencies
    start_task >> check_data_task >> data_ingestion_group
    data_ingestion_group >> finbert_group >> storage_group
    storage_group >> notify_completion_task >> end_task

# Additional DAG for model retraining triggers
model_retraining_dag = DAG(
    'sentiment_model_monitoring',
    default_args=default_args,
    description='Monitor sentiment model performance and trigger retraining',
    schedule_interval=timedelta(days=1),  # Run daily
    catchup=False,
    tags=['monitoring', 'model-ops', 'sentiment']
)

def monitor_model_performance(**context):
    """Monitor sentiment model performance metrics"""
    
    # Simulate performance monitoring
    performance_metrics = {
        'accuracy_drift': np.random.uniform(-0.02, 0.02),
        'confidence_trend': np.random.uniform(-0.05, 0.05),
        'prediction_distribution_shift': np.random.uniform(0, 0.1),
        'data_quality_score': np.random.uniform(0.85, 0.98)
    }
    
    # Check if retraining is needed
    needs_retraining = (
        abs(performance_metrics['accuracy_drift']) > 0.015 or
        abs(performance_metrics['confidence_trend']) > 0.03 or
        performance_metrics['prediction_distribution_shift'] > 0.08 or
        performance_metrics['data_quality_score'] < 0.9
    )
    
    context['task_instance'].xcom_push(key='performance_metrics', value=performance_metrics)
    context['task_instance'].xcom_push(key='needs_retraining', value=needs_retraining)
    
    logger.info(f"Model monitoring: needs_retraining={needs_retraining}, metrics={performance_metrics}")
    return needs_retraining

def trigger_model_retraining(**context):
    """Trigger model retraining if performance degrades"""
    
    needs_retraining = context['task_instance'].xcom_pull(key='needs_retraining')
    
    if needs_retraining:
        logger.info("Triggering model retraining due to performance degradation")
        # In production, this would trigger the actual retraining pipeline
        return "RETRAINING_TRIGGERED"
    else:
        logger.info("Model performance is acceptable, no retraining needed")
        return "NO_ACTION_NEEDED"

with model_retraining_dag:
    
    monitor_task = PythonOperator(
        task_id='monitor_model_performance',
        python_callable=monitor_model_performance,
        dag=model_retraining_dag
    )
    
    retraining_task = PythonOperator(
        task_id='trigger_model_retraining',
        python_callable=trigger_model_retraining,
        dag=model_retraining_dag
    )
    
    monitor_task >> retraining_task

# Export DAGs
__all__ = ['dag', 'model_retraining_dag'] 