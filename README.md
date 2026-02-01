# SupportPilot 🚀

## Self-Healing Autonomous Support Agent for E-commerce Migration

SupportPilot is an **autonomous multi-agent system** built with **LangGraph** and **Google Gemini AI** that automatically detects, analyzes, and resolves support issues during e-commerce platform migrations. Unlike simple LLM-based chatbots, SupportPilot employs a **graph-based agentic workflow** where multiple specialized agents collaborate autonomously to handle complex support scenarios.

---

## 🎯 Key Features

- **Autonomous Decision Making**: Agents independently decide when to escalate, auto-resolve, or request approval
- **Multi-Agent Collaboration**: 5 specialized agents working in a directed graph workflow
- **Real-time Pattern Detection**: AI-powered analysis of tickets, API errors, and webhook failures
- **Self-Healing Actions**: Automatic remediation of common migration issues
- **Conditional Routing**: Dynamic workflow paths based on confidence and priority levels

---

## 🏗️ System Architecture

### LangGraph Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LANGGRAPH ORCHESTRATOR                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌──────────┐     ┌───────────┐     ┌──────────┐     ┌──────────┐        │
│    │ OBSERVE  │────▶│ AGGREGATE │────▶│  REASON  │────▶│  DECIDE  │        │
│    │  Agent   │     │   Node    │     │  Agent   │     │  Agent   │        │
│    └──────────┘     └───────────┘     └──────────┘     └──────────┘        │
│         │                │                  │               │               │
│         ▼                ▼                  ▼               ▼               │
│   ┌──────────┐    ┌───────────┐     ┌──────────┐    ┌───────────┐          │
│   │ Tickets  │    │ Priority  │     │ Pattern  │    │ Confidence│          │
│   │ Webhooks │    │ Routing   │     │ Detection│    │ Routing   │          │
│   │ API Logs │    │           │     │ Root     │    │           │          │
│   └──────────┘    └───────────┘     │ Cause    │    │ ≥90%: Auto│          │
│                        │            │ Analysis │    │ ≥70%: Approve        │
│                        ▼            └──────────┘    │ <70%: Escalate       │
│                  ┌───────────┐                      └───────────┘          │
│                  │ Immediate │                            │                │
│                  │ vs Batch  │                            ▼                │
│                  └───────────┘                      ┌──────────┐           │
│                                                     │ EXECUTE  │           │
│                                                     │  Agent   │           │
│                                                     └──────────┘           │
│                                                           │                │
│                                                           ▼                │
│                                                     ┌──────────┐           │
│                                                     │   END    │           │
│                                                     └──────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent Descriptions

| Agent | Role | Autonomous Capabilities |
|-------|------|------------------------|
| **Observer Agent** | Collects signals from tickets, APIs, webhooks | Parallel data ingestion, anomaly flagging |
| **Aggregator Node** | Prioritizes and routes signals | Critical path detection, batch optimization |
| **Reasoning Agent** | Pattern detection and root cause analysis | LLM-powered hypothesis generation |
| **Decision Agent** | Proposes actions with confidence scoring | Risk assessment, approval routing |
| **Executor Agent** | Executes approved remediation actions | Self-healing operations |

---

## 🔄 Autonomous Workflow Details

### 1. Observation Phase
```python
# Observer Agent collects signals from multiple sources in parallel
signals = await observer_agent.observe_all(
    tickets=state["tickets"],
    merchants=state["merchants"]
)
# Sources: Support tickets, API error logs, Webhook delivery status, Checkout metrics
```

### 2. Conditional Routing (Priority-Based)
```python
def _route_by_priority(self, state: AgentState) -> str:
    """Autonomous routing based on signal criticality"""
    critical_signals = [s for s in state["signals"] if s.get("critical", False)]
    return "immediate" if critical_signals else "batch"
```

### 3. Reasoning Phase (AI-Powered)
```python
# Pattern detection using Gemini AI
patterns = await reasoning_agent.detect_patterns(state["signals"])
# Root cause analysis with structured output
root_cause = await reasoning_agent.analyze_root_cause(patterns, signals)
```

### 4. Confidence-Based Decision Routing
```python
def _route_by_confidence(self, state: AgentState) -> str:
    """Autonomous decision routing"""
    confidence = state["decision"].confidence
    
    if confidence >= 0.90:
        return "auto"      # Execute without human approval
    elif confidence >= 0.70:
        return "approve"   # Request approval before execution
    else:
        return "escalate"  # Escalate to human operator
```

### 5. Self-Healing Execution
```python
# Autonomous action execution for high-confidence decisions
state["actions_taken"] = [
    action.action_type for action in state["decision"].proposed_actions
]
```

---

## 📁 Project Structure

```
SupportPilot/
├── backend/
│   ├── agents/
│   │   ├── orchestrator.py      # LangGraph workflow orchestration
│   │   ├── observer_agent.py    # Signal collection agent
│   │   ├── reasoning_agent.py   # Pattern detection & root cause analysis
│   │   └── decision_agent.py    # Action recommendation agent
│   ├── models/
│   │   ├── ticket.py            # Pydantic ticket model
│   │   ├── merchant.py          # Pydantic merchant model
│   │   └── action.py            # Agent decision & action models
│   ├── services/
│   │   └── llm_service.py       # Gemini AI integration via LangChain
│   ├── tools/
│   │   └── splunk_analyzer.py   # Log analysis tool (with mock fallback)
│   └── main.py                  # FastAPI application
├── frontend/
│   └── index.html               # Real-time dashboard UI
├── data/
│   ├── sample_tickets.json      # Test data
│   └── sample_merchants.json    # Test data
├── requirements.txt
└── .env.example
```

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Agent Framework** | LangGraph | Directed graph-based agent orchestration |
| **LLM Integration** | LangChain + Gemini 2.5 Flash | AI reasoning and analysis |
| **Backend** | FastAPI | Async REST API |
| **Data Models** | Pydantic | Type-safe data validation |
| **Log Analysis** | Splunk SDK (optional) | Enterprise log ingestion |
| **Frontend** | Vanilla JS | Real-time dashboard |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/anushkamali-2005/SupportPilot.git
cd SupportPilot
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Add your GOOGLE_API_KEY to .env
```

### 3. Run the Server
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8001
```

### 4. Open Dashboard
Navigate to: http://localhost:8001/index.html

---

## 📊 Dashboard Features

- **Real-time Agent Calls**: See which agents are being invoked with their arguments
- **Workflow Visualization**: Animated pipeline showing current execution step
- **Dynamic Metrics**: Tickets, patterns, resolutions update per scenario
- **Error Distribution Charts**: Visual breakdown of error types
- **Risk Assessment Gauge**: Real-time severity indicator
- **Streaming Text**: All analysis results stream character-by-character
- **6 Test Scenarios**: Checkout failures, webhook issues, API errors, etc.

---

## 🤖 How Autonomous Agents Work

### State Management
The system uses a `TypedDict` state that flows through the entire graph:

```python
class AgentState(TypedDict):
    tickets: List[Ticket]           # Input tickets
    merchants: List[Merchant]       # Merchant context
    signals: List[Dict]             # Collected signals
    patterns: List                  # Detected patterns
    root_cause_analysis: Dict       # AI analysis results
    decision: AgentDecision         # Proposed actions
    explanation: str                # Human-readable explanation
    actions_taken: List[str]        # Executed actions
```

### Autonomous Behaviors

1. **Self-Triage**: Observer agent autonomously categorizes signal severity
2. **Dynamic Routing**: Workflow adapts based on priority (immediate vs batch)
3. **Confidence-Gated Execution**: High-confidence actions auto-execute
4. **Graceful Escalation**: Low-confidence cases route to humans
5. **Tool Integration**: Agents can invoke external tools (Splunk, APIs)

---

## 📈 Example Scenarios

| Scenario | Tickets | Patterns | Auto-Resolution Rate |
|----------|---------|----------|---------------------|
| Mixed Migration Issues | 8 | 3 | 94% |
| Checkout Flow Failures | 23 | 5 | 87% |
| Webhook Delivery Issues | 15 | 4 | 91% |
| API Misconfiguration | 34 | 6 | 78% |
| Migration Rollback | 19 | 4 | 82% |
| Critical Multi-Merchant Outage | 67 | 9 | 95% |

---

## 🔐 Environment Variables

```env
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional Splunk Integration
SPLUNK_HOST=localhost
SPLUNK_PORT=8089
SPLUNK_USERNAME=admin
SPLUNK_PASSWORD=changeme
```

---

## 🧪 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/agent/run` | POST | Execute full agent workflow |
| `/api/agent/status` | GET | Get agent health status |
| `/api/tickets` | GET | List all tickets |
| `/api/merchants` | GET | List all merchants |

---

## 📚 References

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Google Gemini API](https://ai.google.dev/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

## 👥 Team

Built for NMIMS Hackathon 2026

---

## 📄 License

MIT License
