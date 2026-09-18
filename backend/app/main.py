"""
Darukaa.Earth Biodiversity Intelligence Chatbot — FastAPI Application
Main entry point with /chat, /analyze, /stats, and /health endpoints.
Serves the frontend as static files.
Includes startup health check for the hybrid RAG pipeline.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
import logging
import traceback
from dotenv import load_dotenv
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
logger = logging.getLogger("darukaa.main")

from backend.app.schemas import (
    ChatRequest, ChatResponse, AnalyzeRequest, AnalyzeResponse,
    Recommendation, MetricImpact, SlotState, SourceChunk, SourceItem, Confidence,
    RetrievalTrace
)
from backend.app.conversation import process_message, process_structured_input
from backend.app.retrieval import get_collection_stats, run_startup_health_check, get_health
from backend.app.reasoning import get_graph_stats

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


# --- Lifespan: startup health check ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup health check for the hybrid RAG retrieval pipeline."""
    run_startup_health_check()
    yield


# App
app = FastAPI(
    title="Darukaa.Earth Biodiversity Intelligence API",
    description=(
        "An evidence-based biodiversity advisory system combining a hybrid RAG pipeline "
        "(dense embeddings + BM25 sparse search + Reciprocal Rank Fusion + cross-encoder reranking) "
        "over a vector knowledge base with a structured intervention relationship graph for multi-metric "
        "reasoning. Supports both conversational chat and structured JSON input."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS (allow all origins for hackathon demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- API Endpoints ---

@app.get("/health", tags=["System"])
@app.get("/api/health", tags=["System"])
async def health_check():
    """Health check endpoint with retrieval pipeline status."""
    pipeline_health = get_health()
    return {
        "status": "healthy",
        "service": "Darukaa.Earth Biodiversity Intelligence",
        "pipeline": {
            "chromadb": "ok" if pipeline_health.get("chromadb_ok") else "degraded",
            "chromadb_chunks": pipeline_health.get("chromadb_count", 0),
            "bm25": "ok" if pipeline_health.get("bm25_ok") else "degraded",
            "bm25_documents": pipeline_health.get("bm25_count", 0),
            "reranker": "ok" if pipeline_health.get("reranker_ok") else "unavailable",
            "reranker_model": pipeline_health.get("reranker_model", ""),
        }
    }


@app.get("/stats", tags=["System"])
@app.get("/api/stats", tags=["System"])
async def get_stats():
    """Return knowledge base and system statistics."""
    kb_stats = get_collection_stats()
    graph_stats = get_graph_stats()
    return {
        "knowledge_base": kb_stats,
        "relationship_graph": graph_stats,
        "tracked_variables": 9,
        "system": "Darukaa.Earth Biodiversity Intelligence v2.0 — Hybrid RAG"
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Multi-turn conversational endpoint. Send a message and optionally
    a session_id to continue an existing conversation.
    Response includes retrieval_trace showing hybrid RAG pipeline stages.
    """
    print(f"[API /chat] Incoming request: session_id={request.session_id!r}, message={request.message!r}")
    try:
        response = process_message(request.session_id, request.message)
        return response
    except Exception as e:
        print(f"[API /chat] Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Structured Analysis"])
@app.post("/api/analyze", response_model=AnalyzeResponse, tags=["Structured Analysis"])
async def analyze(request: AnalyzeRequest):
    """
    Structured JSON analysis endpoint. Submit environmental variables directly
    (bypassing conversational slot-filling) to receive an immediate recommendation.
    Response includes retrieval_trace showing hybrid RAG pipeline stages.
    """
    variables = {}
    if request.soil_organic_carbon_pct is not None:
        variables["soil_organic_carbon_pct"] = request.soil_organic_carbon_pct
    if request.soil_ph is not None:
        variables["soil_ph"] = request.soil_ph
    if request.soil_moisture is not None:
        variables["soil_moisture"] = request.soil_moisture
    if request.land_use_type is not None:
        variables["land_use_type"] = request.land_use_type
    if request.rainfall_pattern is not None:
        variables["rainfall_pattern"] = request.rainfall_pattern
    if request.region_climate_zone is not None:
        variables["region_climate_zone"] = request.region_climate_zone
    if request.species_richness_observation is not None:
        variables["species_richness_observation"] = request.species_richness_observation
    if request.pollution_or_deforestation_pressure is not None:
        variables["pollution_or_deforestation_pressure"] = request.pollution_or_deforestation_pressure
    if request.geo_coordinates is not None:
        variables["geo_coordinates"] = request.geo_coordinates

    env_count = sum(1 for k, v in variables.items() if v is not None and k != "geo_coordinates")
    if env_count < 2:
        raise HTTPException(
            status_code=422,
            detail="At least 2 environmental variables are required for analysis. "
                   "Provide land_use_type, rainfall_pattern, region_climate_zone, "
                   "soil_organic_carbon_pct, or other variables."
        )

    try:
        rec_data, evidence_chunks, ranked, retrieval_trace_dict = process_structured_input(variables)
    except Exception as e:
        logger.error(f"[Analyze] Error during process_structured_input: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Additional analysis is temporarily unavailable.",
                "error_code": "ANALYSIS_SERVICE_ERROR",
                "detail": str(e)
            }
        )

    if rec_data is None or not ranked:
        raise HTTPException(
            status_code=404,
            detail="No matching intervention found for the given environmental variables. "
                   "Try providing more specific or additional variables."
        )

    try:
        def _dict_to_rec(d: dict) -> Recommendation:
            metrics = [
                MetricImpact(
                    metric=m.get("metric", ""),
                    direction=m.get("direction", "increase"),
                    magnitude=m.get("magnitude", "")
                )
                for m in d.get("metrics_impacted", [])
            ]
            return Recommendation(
                recommendation=d.get("recommendation", ""),
                mechanism=d.get("mechanism", ""),
                metrics_impacted=metrics,
                time_horizon=d.get("time_horizon", "medium_term"),
                confidence=d.get("confidence", "medium"),
                source=d.get("source", ""),
                connected_variables=d.get("connected_variables", []),
                matched_variables=d.get("matched_variables", []),
                overlap_count=d.get("overlap_count", 0),
                co_benefits=d.get("co_benefits", []),
                trade_offs=d.get("trade_offs", [])
            )

        primary_dict = rec_data.get("primary_recommendation", rec_data)
        primary_rec = _dict_to_rec(primary_dict)

        alt_dict = rec_data.get("alternative_recommendation")
        alt_rec = _dict_to_rec(alt_dict) if alt_dict else None
        why_alt = rec_data.get("why_alternative")

        slots_state = SlotState(
            soil_organic_carbon_pct=variables.get("soil_organic_carbon_pct"),
            soil_ph=variables.get("soil_ph"),
            soil_moisture=variables.get("soil_moisture"),
            land_use_type=variables.get("land_use_type"),
            rainfall_pattern=variables.get("rainfall_pattern"),
            region_climate_zone=variables.get("region_climate_zone"),
            species_richness_observation=variables.get("species_richness_observation"),
            pollution_or_deforestation_pressure=variables.get("pollution_or_deforestation_pressure"),
        )

        raw_sources = rec_data.get("sources", [])
        if raw_sources:
            sources_list = [SourceItem(**s) if isinstance(s, dict) else s for s in raw_sources]
        else:
            sources_list = [
                SourceItem(
                    type="document",
                    id=chunk.get("id", ""),
                    title=chunk.get("document_title") or chunk.get("source", ""),
                    page=chunk.get("page_number"),
                    excerpt=chunk.get("text", "")[:220] + "..." if len(chunk.get("text", "")) > 220 else chunk.get("text", ""),
                    file=chunk.get("source_file"),
                    source=chunk.get("source", ""),
                    score=chunk.get("score")
                )
                for chunk in evidence_chunks[:5]
            ]

        top_score = ranked[0].get("score", ranked[0].get("overlap_count", 0))
        top_matched = ranked[0].get("matched_variables", [])

        reasoning_trace = (
            f"Analysed {env_count} environmental variables across the relationship graph. "
            f"Matched {len(ranked)} candidate interventions satisfying multi-variable preconditions. "
            f"Primary match '{primary_rec.recommendation}' scored {top_score} precondition overlaps "
            f"grounded in user variables: {', '.join(top_matched)}. "
            f"Retrieved {len(evidence_chunks)} evidence chunks via hybrid RAG pipeline "
            f"(dense + BM25 + RRF + cross-encoder reranking)."
        )

        # Build retrieval trace from dict
        ret_trace = RetrievalTrace(**retrieval_trace_dict) if retrieval_trace_dict else None

        return AnalyzeResponse(
            recommendation=primary_rec,
            alternative_recommendation=alt_rec,
            why_alternative=why_alt,
            matched_variables=top_matched,
            overlap_count=top_score,
            slots_used=slots_state,
            sources=sources_list,
            sources_used=sources_list,
            reasoning_trace=reasoning_trace,
            retrieval_trace=ret_trace,
        )
    except Exception as e:
        logger.error(f"[Analyze] Unexpected error assembling response: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Additional analysis is temporarily unavailable.",
                "error_code": "ANALYSIS_SERVICE_ERROR",
                "detail": str(e)
            }
        )


# --- Static File Serving (Frontend) ---

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", tags=["Frontend"])
    async def serve_frontend():
        """Serve the main frontend page."""
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Frontend not found. Access /docs for API documentation."}
else:
    @app.get("/", tags=["Frontend"])
    async def root():
        return {"message": "Darukaa.Earth API is running. Access /docs for interactive API documentation."}
