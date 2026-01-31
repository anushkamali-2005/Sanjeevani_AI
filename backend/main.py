"""FastAPI Application Entry Point"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.agents.orchestrator import orchestrator
from backend.models.ticket import Ticket
from backend.models.merchant import Merchant
from backend.models.action import AgentDecision
from typing import List
import json
from pathlib import Path

app = FastAPI(title="Self-Healing Support Agent", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Load sample data
def load_sample_data():
    data_dir = Path("data")
    with open(data_dir / "sample_tickets.json") as f:
        tickets = [Ticket(**t) for t in json.load(f)]
    with open(data_dir / "sample_merchants.json") as f:
        merchants = [Merchant(**m) for m in json.load(f)]
    return tickets, merchants

@app.get("/")
async def root():
    return {"message": "Self-Healing Support Agent API", "status": "online"}

@app.post("/api/agent/run", response_model=dict)
async def run_agent(background_tasks: BackgroundTasks):
    """Run the complete agent workflow"""
    try:
        tickets, merchants = load_sample_data()
        result = await orchestrator.run(tickets, merchants)
        
        return {
            "status": "success",
            "decision": result["decision"].dict() if result.get("decision") else None,
            "explanation": result.get("explanation", ""),
            "actions_taken": result.get("actions_taken", []),
            "confidence": result.get("decision").confidence if result.get("decision") else 0.0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tickets", response_model=List[Ticket])
async def get_tickets():
    """Get all tickets"""
    tickets, _ = load_sample_data()
    return tickets

@app.get("/api/merchants", response_model=List[Merchant])
async def get_merchants():
    """Get all merchants"""
    _, merchants = load_sample_data()
    return merchants

@app.get("/api/agent/status")
async def agent_status():
    """Get agent health status"""
    return {
        "status": "healthy",
        "observer": "active",
        "reasoning": "active",
        "decision": "active",
        "executor": "active"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
