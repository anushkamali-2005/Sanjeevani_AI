"""LangGraph Orchestrator - Main Agent Workflow"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Annotated
from backend.agents.observer_agent import observer_agent
from backend.agents.reasoning_agent import reasoning_agent
from backend.agents.decision_agent import decision_agent
from backend.models.ticket import Ticket
from backend.models.merchant import Merchant
from backend.models.action import AgentDecision
import operator

# Define state
class AgentState(TypedDict):
    tickets: List[Ticket]
    merchants: List[Merchant]
    signals: List[Dict]
    patterns: List
    root_cause_analysis: Dict
    decision: AgentDecision
    explanation: str
    actions_taken: List[str]

class AgentOrchestrator:
    """LangGraph-based agent orchestration"""
    
    def __init__(self):
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("observe", self._observe_node)
        workflow.add_node("aggregate", self._aggregate_node)
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("decide", self._decide_node)
        workflow.add_node("execute", self._execute_node)
        
        # Add edges with conditional routing
        workflow.set_entry_point("observe")
        workflow.add_edge("observe", "aggregate")
        workflow.add_conditional_edges(
            "aggregate",
            self._route_by_priority,
            {
                "immediate": "reason",
                "batch": "reason"
            }
        )
        workflow.add_edge("reason", "decide")
        workflow.add_conditional_edges(
            "decide",
            self._route_by_confidence,
            {
                "auto": "execute",
                "approve": "execute",
                "escalate": END
            }
        )
        workflow.add_edge("execute", END)
        
        return workflow.compile()
    
    async def _observe_node(self, state: AgentState) -> AgentState:
        """Observer phase - parallel signal collection"""
        signals = await observer_agent.observe_all(
            tickets=state.get("tickets", []),
            merchants=state.get("merchants", [])
        )
        state["signals"] = signals
        return state
    
    async def _aggregate_node(self, state: AgentState) -> AgentState:
        """Aggregate signals"""
        # Simple aggregation logic
        state["aggregated"] = True
        return state
    
    async def _reason_node(self, state: AgentState) -> AgentState:
        """Reasoning phase"""
        patterns = await reasoning_agent.detect_patterns(state["signals"])
        root_cause = await reasoning_agent.analyze_root_cause(patterns, state["signals"])
        
        state["patterns"] = patterns
        state["root_cause_analysis"] = root_cause
        return state
    
    async def _decide_node(self, state: AgentState) -> AgentState:
        """Decision phase"""
        decision = await decision_agent.decide_actions(state["root_cause_analysis"])
        explanation = await decision_agent.explain_decision(decision)
        
        state["decision"] = decision
        state["explanation"] = explanation
        return state
    
    async def _execute_node(self, state: AgentState) -> AgentState:
        """Execution phase"""
        # Mock execution
        state["actions_taken"] = [a.action_type for a in state["decision"].proposed_actions]
        return state
    
    def _route_by_priority(self, state: AgentState) -> str:
        """Route based on signal priority"""
        critical_signals = [s for s in state["signals"] if s.get("critical", False)]
        return "immediate" if critical_signals else "batch"
    
    def _route_by_confidence(self, state: AgentState) -> str:
        """Route based on confidence level"""
        confidence = state["decision"].confidence
        
        if confidence >= 0.90:
            return "auto"
        elif confidence >= 0.70:
            return "approve"
        else:
            return "escalate"
    
    async def run(self, tickets: List[Ticket], merchants: List[Merchant]) -> Dict:
        """Run the complete agent workflow"""
        initial_state = AgentState(
            tickets=tickets,
            merchants=merchants,
            signals=[],
            patterns=[],
            root_cause_analysis={},
            decision=None,
            explanation="",
            actions_taken=[]
        )
        
        result = await self.graph.ainvoke(initial_state)
        return result

orchestrator = AgentOrchestrator()
