# Self-Healing Support Agent for Headless E-commerce Migration

## Overview
Agentic AI system that observes, reasons, decides, and acts on support issues during platform migration using LangGraph orchestration and Gemini AI.

## Tech Stack
- **Agent Framework**: LangGraph (orchestration backbone)
- **LLM**: LangChain + Gemini API
- **Backend**: FastAPI
- **Analysis**: Splunk (optional)
- **Frontend**: HTML/CSS/Vanilla JavaScript
- **Database**: SQLite

## Features
- ✅ Parallel signal observation across tickets, APIs, webhooks, and merchant activity
- ✅ LLM-powered root cause analysis using Gemini
- ✅ Confidence-based action routing (auto-execute, recommend, escalate)
- ✅ Human-in-the-loop for high-risk actions
- ✅ Explainable AI decisions
- ✅ Real-time dashboard with workflow visualization

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Gemini API key
# GOOGLE_API_KEY=your_actual_gemini_api_key
```

### 3. Run the Application
```bash
# Start the FastAPI backend
python -m backend.main

# Or using uvicorn directly
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access Dashboard
Open your browser to: `http://localhost:8000/static/index.html`

## Architecture

### Agent Workflow
```
START → OBSERVE (parallel) → AGGREGATE → REASON → DECIDE (conditional) → EXECUTE → END
```

### Key Components

#### Observer Agent
- Monitors tickets, API errors, webhooks, and merchant activity in parallel
- Flags critical signals for immediate attention

#### Reasoning Agent
- Detects patterns across signals using LLM
- Performs root cause analysis with confidence scoring

#### Decision Agent
- Recommends actions based on confidence levels
- Routes to auto-execute (>90%), human approval (70-90%), or escalate (<70%)

#### Orchestrator (LangGraph)
- Coordinates multi-agent workflow
- Implements conditional routing and parallel execution

## API Endpoints

- `GET /` - Health check
- `POST /api/agent/run` - Run complete agent workflow
- `GET /api/tickets` - Get all tickets
- `GET /api/merchants` - Get all merchants
- `GET /api/agent/status` - Get agent status

## Project Structure
```
self-healing-support-agent/
├── backend/
│   ├── agents/          # Observer, Reasoning, Decision, Orchestrator
│   ├── models/          # Pydantic data models
│   ├── services/        # LLM service (LangChain + Gemini)
│   ├── tools/           # Splunk analyzer
│   ├── config.py        # Configuration management
│   └── main.py          # FastAPI application
├── frontend/
│   ├── css/             # Styling
│   ├── js/              # Dashboard logic
│   └── index.html       # Main dashboard
├── data/                # Sample data files
├── requirements.txt
└── README.md
```

## Configuration

All configuration is managed through environment variables in `.env`:

- `GOOGLE_API_KEY` - Your Gemini API key (required)
- `SPLUNK_HOST` - Splunk host (optional, defaults to localhost)
- `CONFIDENCE_THRESHOLD_AUTO` - Auto-execute threshold (default: 0.90)
- `CONFIDENCE_THRESHOLD_RECOMMEND` - Recommendation threshold (default: 0.70)

## Development

### Running Tests
```bash
pytest tests/
```

### Adding New Agents
1. Create agent file in `backend/agents/`
2. Implement async methods for agent logic
3. Add node to orchestrator workflow in `backend/agents/orchestrator.py`

## License
MIT

## Support
For issues or questions, please open an issue on GitHub.
