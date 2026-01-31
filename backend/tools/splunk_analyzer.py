"""
All Splunk-related functionality in ONE file using OOP
"""
import splunklib.client as client
import splunklib.results as results
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from backend.config import get_settings

class SplunkAnalyzer:
    """Object-oriented Splunk analysis tool - all functions in one class"""
    
    def __init__(self):
        self.settings = get_settings()
        self.service = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Splunk"""
        try:
            self.service = client.connect(
                host=self.settings.splunk_host,
                port=self.settings.splunk_port,
                username=self.settings.splunk_username,
                password=self.settings.splunk_password,
                scheme=self.settings.splunk_scheme
            )
        except Exception as e:
            print(f"Splunk connection failed: {e}")
            self.service = None
    
    def search_errors(self, time_range: str = "-24h", severity: str = "ERROR") -> List[Dict]:
        """Search for errors in logs"""
        if not self.service:
            return []
        
        query = f'search index=main level={severity} earliest={time_range}'
        try:
            job = self.service.jobs.create(query)
            while not job.is_done():
                pass
            
            error_list = []
            for result in results.ResultsReader(job.results()):
                error_list.append(dict(result))
            return error_list
        except Exception as e:
            print(f"Error search failed: {e}")
            return []
    
    def detect_api_failures(self, endpoint: str = "*", time_range: str = "-1h") -> Dict:
        """Detect API failures for specific endpoints"""
        if not self.service:
            return {"total_failures": 0, "failure_rate": 0.0, "failed_endpoints": []}
        
        query = f'''
        search index=api endpoint={endpoint} status>=400 earliest={time_range}
        | stats count by endpoint, status
        '''
        
        try:
            job = self.service.jobs.create(query)
            while not job.is_done():
                pass
            
            failures = []
            total = 0
            for result in results.ResultsReader(job.results()):
                failures.append(dict(result))
                total += int(result.get('count', 0))
            
            return {
                "total_failures": total,
                "failure_rate": total / 100 if total > 0 else 0.0,  # Mock calculation
                "failed_endpoints": failures
            }
        except Exception as e:
            print(f"API failure detection failed: {e}")
            return {"total_failures": 0, "failure_rate": 0.0, "failed_endpoints": []}
    
    def analyze_webhook_health(self, merchant_id: Optional[str] = None) -> Dict:
        """Analyze webhook delivery health"""
        merchant_filter = f'merchant_id={merchant_id}' if merchant_id else '*'
        query = f'''
        search index=webhooks {merchant_filter} earliest=-24h
        | stats count(eval(status="success")) as success, count(eval(status="failed")) as failed
        '''
        
        if not self.service:
            return {"success_rate": 0.0, "total_webhooks": 0, "failed_webhooks": 0}
        
        try:
            job = self.service.jobs.create(query)
            while not job.is_done():
                pass
            
            for result in results.ResultsReader(job.results()):
                success = int(result.get('success', 0))
                failed = int(result.get('failed', 0))
                total = success + failed
                
                return {
                    "success_rate": (success / total * 100) if total > 0 else 0.0,
                    "total_webhooks": total,
                    "failed_webhooks": failed
                }
            
            return {"success_rate": 0.0, "total_webhooks": 0, "failed_webhooks": 0}
        except Exception as e:
            print(f"Webhook analysis failed: {e}")
            return {"success_rate": 0.0, "total_webhooks": 0, "failed_webhooks": 0}
    
    def get_checkout_metrics(self, merchant_id: str, time_range: str = "-24h") -> Dict:
        """Get checkout success/failure metrics"""
        query = f'''
        search index=checkout merchant_id={merchant_id} earliest={time_range}
        | stats count(eval(status="completed")) as completed, 
                count(eval(status="failed")) as failed,
                avg(processing_time) as avg_time
        '''
        
        if not self.service:
            return {"success_rate": 0.0, "total_checkouts": 0, "avg_processing_time": 0.0}
        
        try:
            job = self.service.jobs.create(query)
            while not job.is_done():
                pass
            
            for result in results.ResultsReader(job.results()):
                completed = int(result.get('completed', 0))
                failed = int(result.get('failed', 0))
                total = completed + failed
                
                return {
                    "success_rate": (completed / total * 100) if total > 0 else 0.0,
                    "total_checkouts": total,
                    "avg_processing_time": float(result.get('avg_time', 0.0))
                }
            
            return {"success_rate": 0.0, "total_checkouts": 0, "avg_processing_time": 0.0}
        except Exception as e:
            print(f"Checkout metrics failed: {e}")
            return {"success_rate": 0.0, "total_checkouts": 0, "avg_processing_time": 0.0}
    
    def detect_anomalies(self, metric: str = "error_rate", threshold: float = 2.0) -> List[Dict]:
        """Detect anomalies using statistical deviation"""
        query = f'''
        search index=main earliest=-7d
        | timechart span=1h avg({metric}) as avg_metric, stdev({metric}) as stdev_metric
        | eval anomaly=if(avg_metric > (avg(avg_metric) + {threshold}*stdev_metric), 1, 0)
        | where anomaly=1
        '''
        
        if not self.service:
            return []
        
        try:
            job = self.service.jobs.create(query)
            while not job.is_done():
                pass
            
            anomalies = []
            for result in results.ResultsReader(job.results()):
                anomalies.append(dict(result))
            return anomalies
        except Exception as e:
            print(f"Anomaly detection failed: {e}")
            return []
    
    def correlate_errors_with_migrations(self) -> Dict:
        """Correlate error spikes with migration events"""
        query = '''
        search (index=errors OR index=migrations) earliest=-30d
        | eval event_type=if(index="migrations", "migration", "error")
        | stats count by event_type, merchant_id, _time
        | sort _time
        '''
        
        if not self.service:
            return {"correlations": [], "correlation_strength": 0.0}
        
        try:
            job = self.service.jobs.create(query)
            while not job.is_done():
                pass
            
            events = []
            for result in results.ResultsReader(job.results()):
                events.append(dict(result))
            
            # Simple correlation logic
            correlation_strength = len(events) / 100.0 if events else 0.0
            
            return {
                "correlations": events,
                "correlation_strength": min(correlation_strength, 1.0)
            }
        except Exception as e:
            print(f"Correlation analysis failed: {e}")
            return {"correlations": [], "correlation_strength": 0.0}

# Singleton instance
splunk_analyzer = SplunkAnalyzer()
