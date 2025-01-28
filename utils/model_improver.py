# utils/model_improver.py
from datetime import datetime, timedelta
import json
import os
import numpy as np
from typing import Dict, List

class ModelImprover:
    def __init__(self):
        self.results_dir = "predictions/validation_results"
        
    def analyze_results(self, symbol: str, min_samples: int = 5) -> Dict:
        """Analyze validation results and generate recommendations"""
        metrics = self._load_metrics(symbol)
        if not metrics:
            return None
            
        analysis = {}
        for timeframe, m in metrics.items():
            if m['sample_size'] < min_samples:
                analysis[timeframe] = {
                    'status': 'insufficient_data',
                    'sample_size': m['sample_size']
                }
                continue
                
            timeframe_analysis = {
                'metrics': m,
                'issues': self._identify_issues(m),
                'improvements': self._suggest_improvements(m)
            }
            analysis[timeframe] = timeframe_analysis
            
        recommendations = self._generate_recommendations(analysis)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'analysis': analysis,
            'recommendations': recommendations
        }
        
        self._save_report(symbol, report)
        return report
        
    def _identify_issues(self, metrics: Dict) -> List[Dict]:
        """Identify issues in the predictions"""
        issues = []
        
        if metrics['mean_absolute_error'] > 1.0:
            issues.append({
                'type': 'high_error',
                'severity': 'high',
                'details': f"High mean absolute error: {metrics['mean_absolute_error']:.2f}%"
            })
            
        if metrics['direction_accuracy'] < 60:
            issues.append({
                'type': 'poor_direction',
                'severity': 'high',
                'details': f"Low direction accuracy: {metrics['direction_accuracy']:.1f}%"
            })
            
        if metrics['ci_coverage'] < 90:
            issues.append({
                'type': 'confidence_interval',
                'severity': 'medium',
                'details': f"Low confidence interval coverage: {metrics['ci_coverage']:.1f}%"
            })
            
        return issues
        
    def _suggest_improvements(self, metrics: Dict) -> List[Dict]:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        if metrics['mean_absolute_error'] > 1.0:
            suggestions.extend([
                {
                    'area': 'model_training',
                    'suggestion': 'Increase training data window',
                    'implementation': 'Modify training_window in model_trainer.py'
                },
                {
                    'area': 'feature_engineering',
                    'suggestion': 'Review and remove weak features',
                    'implementation': 'Use feature_importance analysis in model_trainer.py'
                }
            ])
            
        if metrics['direction_accuracy'] < 60:
            suggestions.extend([
                {
                    'area': 'features',
                    'suggestion': 'Add momentum indicators',
                    'implementation': 'Add RSI, MACD variations to technical_indicators'
                },
                {
                    'area': 'features',
                    'suggestion': 'Include shorter-term indicators',
                    'implementation': 'Add 1-min and 5-min technical indicators'
                }
            ])
            
        if metrics['ci_coverage'] < 90:
            suggestions.extend([
                {
                    'area': 'confidence',
                    'suggestion': 'Adjust confidence calculation',
                    'implementation': 'Modify calculate_confidence_score in prediction_system.py'
                },
                {
                    'area': 'volatility',
                    'suggestion': 'Include market volatility in confidence estimation',
                    'implementation': 'Add VIX-based adjustments to confidence calculation'
                }
            ])
            
        return suggestions
        
    def _generate_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate overall improvement recommendations"""
        recommendations = []
        
        error_issues = []
        direction_issues = []
        confidence_issues = []
        
        for timeframe, data in analysis.items():
            if isinstance(data, dict) and 'issues' in data:
                for issue in data['issues']:
                    if issue['type'] == 'high_error':
                        error_issues.append(timeframe)
                    elif issue['type'] == 'poor_direction':
                        direction_issues.append(timeframe)
                    elif issue['type'] == 'confidence_interval':
                        confidence_issues.append(timeframe)
        
        if error_issues:
            recommendations.append({
                'priority': 'High',
                'area': 'Model Accuracy',
                'affected_timeframes': error_issues,
                'suggestion': 'Improve prediction accuracy',
                'implementation': [
                    'Review and optimize feature engineering',
                    'Increase training data window',
                    'Consider ensemble weight adjustments',
                    'Add market regime detection'
                ]
            })
            
        if direction_issues:
            recommendations.append({
                'priority': 'High',
                'area': 'Direction Prediction',
                'affected_timeframes': direction_issues,
                'suggestion': 'Enhance directional accuracy',
                'implementation': [
                    'Add trend strength indicators',
                    'Include market sentiment analysis',
                    'Implement momentum-based features'
                ]
            })
            
        if confidence_issues:
            recommendations.append({
                'priority': 'Medium',
                'area': 'Confidence Estimation',
                'affected_timeframes': confidence_issues,
                'suggestion': 'Improve confidence intervals',
                'implementation': [
                    'Incorporate market volatility into CI calculation',
                    'Add dynamic adjustments based on time of day',
                    'Implement market regime detection',
                    'Consider VIX-based scaling factors'
                ]
            })
            
        return recommendations
    
    def _load_metrics(self, symbol: str) -> Dict:
        """Load metrics from file"""
        metrics_file = os.path.join(self.results_dir, f"{symbol}_metrics.json")
        if not os.path.exists(metrics_file):
            return None
            
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    def _save_report(self, symbol: str, report: Dict):
        """Save improvement report to file"""
        report_file = os.path.join(self.results_dir, f"{symbol}_improvement_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
    
    def print_report_summary(self, symbol: str):
        """Print a human-readable summary of the improvement report"""
        report_file = os.path.join(self.results_dir, f"{symbol}_improvement_report.json")
        if not os.path.exists(report_file):
            print(f"No improvement report found for {symbol}")
            return
            
        with open(report_file, 'r') as f:
            report = json.load(f)
            
        print(f"\nImprovement Report Summary for {symbol}")
        print("=" * 50)
        print(f"Analysis timestamp: {datetime.fromisoformat(report['timestamp']).strftime('%Y-%m-%d %H:%M:%S ET')}")
        
        for timeframe, analysis in report['analysis'].items():
            print(f"\n{timeframe} Timeframe:")
            if 'status' in analysis:
                print(f"Status: {analysis['status']}")
                print(f"Samples: {analysis['sample_size']}")
                continue
                
            print("Issues Found:")
            for issue in analysis['issues']:
                print(f"- {issue['details']} (Severity: {issue['severity']})")
                
            print("\nSuggested Improvements:")
            for improvement in analysis['improvements']:
                print(f"- {improvement['suggestion']}")
                print(f"  Implementation: {improvement['implementation']}")
        
        print("\nOverall Recommendations:")
        for rec in report['recommendations']:
            print(f"\nPriority: {rec['priority']}")
            print(f"Area: {rec['area']}")
            print(f"Affected Timeframes: {', '.join(rec['affected_timeframes'])}")
            print("Implementation Steps:")
            for step in rec['implementation']:
                print(f"- {step}")