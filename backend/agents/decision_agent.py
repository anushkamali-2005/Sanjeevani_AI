"""Decision Agent - Action Recommendation"""
from typing import List, Dict
from backend.models.action import AgentAction, ActionType, RiskLevel, ActionStatus, AgentDecision
from backend.services.llm_service import llm_service
from datetime import datetime
import uuid

class DecisionAgent:
    """Decides on actions based on reasoning"""
    
    def __init__(self):
        self.llm = llm_service
        self.confidence_threshold_auto = 0.90
        self.confidence_threshold_recommend = 0.70
    
    async def decide_actions(self, root_cause_analysis: Dict) -> AgentDecision:
        """Decide on appropriate actions"""
        confidence = root_cause_analysis.get("confidence", 0.0)
        root_cause = root_cause_analysis.get("root_cause", "Unknown")
        
        # Get LLM recommendations
        recommended_actions = self.llm.recommend_actions(
            root_cause=root_cause,
            severity="high" if confidence > 0.8 else "medium"
        )
        
        # Convert to AgentAction objects
        actions = []
        for rec in recommended_actions:
            action = AgentAction(
                id=str(uuid.uuid4()),
                action_type=ActionType(rec.get("action_type", "send_communication")),
                confidence=confidence,
                risk_level=RiskLevel(rec.get("risk_level", "low")),
                requires_approval=self._requires_approval(confidence, rec.get("risk_level")),
                reasoning=rec.get("reasoning", ""),
                target="affected_merchants",
                parameters={"message": "Proactive support communication"},
                impact_assessment=rec.get("impact", "")
            )
            actions.append(action)
        
        decision = AgentDecision(
            decision_id=str(uuid.uuid4()),
            ticket_ids=[p.pattern_id for p in root_cause_analysis.get("patterns", [])],
            root_cause=root_cause,
            confidence=confidence,
            proposed_actions=actions,
            assumptions=root_cause_analysis.get("assumptions", []),
            uncertainty_factors=root_cause_analysis.get("uncertainties", [])
        )
        
        return decision
    
    def _requires_approval(self, confidence: float, risk_level: str) -> bool:
        """Determine if action requires human approval"""
        if risk_level in ["high", "critical"]:
            return True
        if confidence < self.confidence_threshold_auto:
            return True
        return False
    
    async def explain_decision(self, decision: AgentDecision) -> str:
        """Generate human-readable explanation"""
        explanation = self.llm.explain_decision(decision.dict())
        return explanation

decision_agent = DecisionAgent()
