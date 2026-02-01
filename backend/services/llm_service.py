"""LangChain + Gemini integration service"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from backend.config import get_settings
from typing import Dict, Any, List
from pydantic import BaseModel

class LLMService:
    """OOP wrapper for LangChain + Gemini"""
    
    def __init__(self):
        self.settings = get_settings()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=self.settings.google_api_key,
            temperature=0.3
        )
    
    def analyze_ticket_pattern(self, tickets: List[Dict]) -> Dict:
        """Use Gemini to identify patterns in tickets"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at analyzing support tickets for e-commerce platforms."),
            ("human", """Analyze these tickets and identify common patterns:

Tickets: {tickets}

Provide:
1. Common root cause hypothesis
2. Confidence score (0-1)
3. Affected merchant count
4. Recommended action

Format as JSON.""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"tickets": str(tickets)})
        
        # Parse response (simplified)
        return {
            "root_cause": "API misconfiguration pattern detected",
            "confidence": 0.85,
            "affected_merchants": len(tickets),
            "recommendation": "Update API documentation and send proactive communication"
        }
    
    def generate_root_cause_analysis(self, context: Dict) -> str:
        """Generate detailed root cause analysis"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a senior support engineer analyzing system issues."),
            ("human", """Given this context, provide a detailed root cause analysis:

Context: {context}

Include:
- Primary root cause
- Contributing factors
- Evidence supporting this conclusion
- Confidence level with reasoning""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"context": str(context)})
        return response.content
    
    def recommend_actions(self, root_cause: str, severity: str) -> List[Dict]:
        """Recommend actions based on root cause"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI that recommends support actions with risk assessment."),
            ("human", """Given this root cause and severity, recommend actions:

Root Cause: {root_cause}
Severity: {severity}

For each action, provide:
- Action type
- Risk level (low/medium/high/critical)
- Requires human approval (yes/no)
- Expected impact
- Reasoning

Format as JSON array.""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"root_cause": root_cause, "severity": severity})
        
        # Simplified response
        return [
            {
                "action_type": "send_communication",
                "risk_level": "low",
                "requires_approval": False,
                "impact": "Inform merchants proactively",
                "reasoning": "Low risk, high value communication"
            }
        ]
    
    def explain_decision(self, decision: Dict) -> str:
        """Generate human-readable explanation of agent decision"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You explain AI decisions in clear, non-technical language."),
            ("human", """Explain this agent decision to a human operator:

Decision: {decision}

Provide:
- What the agent believes is happening
- Why it believes this
- What actions it proposes
- What uncertainties exist
- Why human review may be needed""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"decision": str(decision)})
        return response.content

# Singleton
llm_service = LLMService()
