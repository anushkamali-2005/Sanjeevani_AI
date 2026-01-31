"""Reasoning Agent - Root Cause Analysis"""
from typing import List, Dict
from backend.services.llm_service import llm_service
from backend.models.ticket import TicketPattern

class ReasoningAgent:
    """Performs root cause analysis on observed signals"""
    
    def __init__(self):
        self.llm = llm_service
    
    async def detect_patterns(self, signals: List[Dict]) -> List[TicketPattern]:
        """Detect patterns across signals"""
        # Filter ticket signals
        ticket_signals = [s for s in signals if s.get("signal_type") == "tickets"]
        
        if not ticket_signals:
            return []
        
        patterns = []
        for signal in ticket_signals:
            if signal.get("high_priority_count", 0) > 3:
                # Use LLM to analyze pattern
                analysis = self.llm.analyze_ticket_pattern(
                    [t.dict() for t in signal.get("tickets", [])]
                )
                
                patterns.append(TicketPattern(
                    pattern_id=f"pattern_{len(patterns)}",
                    affected_tickets=[t.id for t in signal.get("tickets", [])],
                    common_category="migration_issue",
                    confidence_score=analysis.get("confidence", 0.5),
                    root_cause_hypothesis=analysis.get("root_cause", "Unknown"),
                    affected_merchant_count=analysis.get("affected_merchants", 0),
                    first_occurrence=signal.get("tickets", [{}])[0].created_at if signal.get("tickets") else None,
                    last_occurrence=signal.get("tickets", [{}])[-1].created_at if signal.get("tickets") else None
                ))
        
        return patterns
    
    async def analyze_root_cause(self, patterns: List[TicketPattern], signals: List[Dict]) -> Dict:
        """Deep root cause analysis"""
        context = {
            "patterns": [p.dict() for p in patterns],
            "signals": signals,
            "critical_signals": [s for s in signals if s.get("critical", False)]
        }
        
        # Use LLM for deep analysis
        root_cause_analysis = self.llm.generate_root_cause_analysis(context)
        
        # Calculate overall confidence
        confidence = sum(p.confidence_score for p in patterns) / len(patterns) if patterns else 0.0
        
        return {
            "root_cause": root_cause_analysis,
            "confidence": confidence,
            "patterns": patterns,
            "assumptions": [
                "Tickets are correctly categorized",
                "API errors correlate with migration stage",
                "Webhook failures indicate config issues"
            ],
            "uncertainties": [
                "Cannot confirm if merchant followed migration guide",
                "Platform bug vs merchant config unclear in some cases"
            ]
        }

reasoning_agent = ReasoningAgent()
