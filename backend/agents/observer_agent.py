"""Observer Agent - Signal Ingestion"""
from typing import List, Dict
from backend.models.ticket import Ticket, TicketPriority
from backend.models.merchant import Merchant, MerchantActivity
from backend.tools.splunk_analyzer import splunk_analyzer
import asyncio

class ObserverAgent:
    """Observes all signals across the system"""
    
    def __init__(self):
        self.splunk = splunk_analyzer
        self.signals = []
    
    async def observe_tickets(self, tickets: List[Ticket]) -> Dict:
        """Monitor incoming tickets"""
        high_priority = [t for t in tickets if t.priority in [TicketPriority.HIGH, TicketPriority.CRITICAL]]
        
        return {
            "signal_type": "tickets",
            "total_tickets": len(tickets),
            "high_priority_count": len(high_priority),
            "tickets": tickets,
            "requires_immediate_attention": len(high_priority) > 5
        }
    
    async def observe_api_errors(self) -> Dict:
        """Monitor API failures"""
        api_data = self.splunk.detect_api_failures(time_range="-1h")
        
        return {
            "signal_type": "api_errors",
            "total_failures": api_data["total_failures"],
            "failure_rate": api_data["failure_rate"],
            "critical": api_data["failure_rate"] > 10.0
        }
    
    async def observe_webhooks(self) -> Dict:
        """Monitor webhook health"""
        webhook_data = self.splunk.analyze_webhook_health()
        
        return {
            "signal_type": "webhooks",
            "success_rate": webhook_data["success_rate"],
            "failed_count": webhook_data["failed_webhooks"],
            "critical": webhook_data["success_rate"] < 80.0
        }
    
    async def observe_merchant_activity(self, merchant: Merchant) -> Dict:
        """Monitor individual merchant health"""
        checkout_metrics = self.splunk.get_checkout_metrics(merchant.id)
        
        return {
            "signal_type": "merchant_activity",
            "merchant_id": merchant.id,
            "checkout_success_rate": checkout_metrics["success_rate"],
            "health_degraded": checkout_metrics["success_rate"] < 90.0
        }
    
    async def observe_all(self, tickets: List[Ticket], merchants: List[Merchant]) -> List[Dict]:
        """Parallel observation of all signals"""
        tasks = [
            self.observe_tickets(tickets),
            self.observe_api_errors(),
            self.observe_webhooks()
        ]
        
        # Add merchant observations
        for merchant in merchants[:10]:  # Limit to 10 for demo
            tasks.append(self.observe_merchant_activity(merchant))
        
        signals = await asyncio.gather(*tasks)
        self.signals = signals
        return signals

observer_agent = ObserverAgent()
