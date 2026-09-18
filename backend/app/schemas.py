"""
Pydantic schemas for Darukaa.Earth Biodiversity Intelligence Chatbot.
Defines strict input/output models for API endpoints and internal data flow.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# --- Enums ---

class Direction(str, Enum):
    increase = "increase"
    decrease = "decrease"


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class TimeHorizon(str, Enum):
    short_term = "short_term"
    short_to_medium_term = "short_to_medium_term"
    medium_term = "medium_term"
    medium_to_long_term = "medium_to_long_term"
    long_term = "long_term"


# --- Slot State ---

class SlotState(BaseModel):
    """Tracks known/unknown variable values for a conversation session."""
    soil_organic_carbon_pct: Optional[float] = None
    soil_ph: Optional[float] = None
    soil_moisture: Optional[str] = None
    land_use_type: Optional[str] = None
    rainfall_pattern: Optional[str] = None
    region_climate_zone: Optional[str] = None
    species_richness_observation: Optional[str] = None
    pollution_or_deforestation_pressure: Optional[str] = None
    geo_lat: Optional[float] = None
    geo_lon: Optional[float] = None
    state: Optional[str] = None
    country: Optional[str] = "India"


# --- Recommendation Output ---

class MetricImpact(BaseModel):
    metric: str
    direction: Direction
    magnitude: str


class Recommendation(BaseModel):
    """The strict output schema for every recommendation — never freeform prose."""
    recommendation: str
    mechanism: str
    metrics_impacted: list[MetricImpact]
    time_horizon: str
    confidence: Confidence
    source: str
    connected_variables: list[str]
    matched_variables: list[str] = Field(default_factory=list, description="Explicit user variables that triggered this match")
    overlap_count: int = Field(default=0, description="Number of precondition overlaps")
    co_benefits: list[str] = Field(default_factory=list, description="Ecological and economic co-benefits")
    trade_offs: list[str] = Field(default_factory=list, description="Management trade-offs and operational caveats")


# --- API Request/Response Models ---

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class RetrievalTrace(BaseModel):
    """Traces the hybrid RAG pipeline stages for inspectability."""
    query_used: str = ""
    dense_candidates: int = 0
    bm25_candidates: int = 0
    rrf_candidates: int = 0
    reranker_ran: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    final_chunk_ids: list[str] = Field(default_factory=list)
    fallback_used: Optional[str] = None
    cache_hit: bool = False
    latency_ms: float = 0.0


class SourceItem(BaseModel):
    """
    Structured source attribution item supporting:
    - "document": Peer-reviewed reports (IPCC, IPBES) with exact page number & excerpt
    - "web" / "web_search": Live web search results with real URL and retrieval timestamp
    - "structured_data": Real numeric CSV datasets with filename and column detail
    """
    type: str = "document"  # "document", "structured_data", "web_search"
    source_type: Optional[str] = "document"  # "document", "web", "structured_data"
    id: Optional[str] = None  # Clean tag: e.g. "DOC_01", "WEB_01"
    tag: Optional[str] = None  # Canonical tag identifier (e.g. DOC_01, WEB_01)
    title: Optional[str] = None
    page: Optional[int] = None
    page_number: Optional[int] = None
    file: Optional[str] = None
    file_name: Optional[str] = None
    chunk_id: Optional[str] = None
    excerpt: Optional[str] = None
    detail: Optional[str] = None
    url: Optional[str] = None
    source_url: Optional[str] = None
    retrieved_at: Optional[str] = None
    relevance_label: Optional[str] = None  # "Highly Relevant", "Relevant", "Moderately Relevant"
    is_cited: Optional[bool] = False
    verified: Optional[bool] = True

    # Backward-compatible fields
    text: Optional[str] = None
    source: Optional[str] = None
    score: Optional[float] = None


# Alias for backwards compatibility
SourceChunk = SourceItem


class ChatResponse(BaseModel):
    session_id: str
    response_text: str
    answer: Optional[str] = None  # Clean modern API alias for response_text
    recommendation: Optional[Recommendation] = None
    alternative_recommendation: Optional[Recommendation] = None
    why_alternative: Optional[str] = None
    slots: SlotState
    missing_slots: list[str] = []
    sources: list[SourceItem] = Field(default_factory=list, description="Structured sources array (documents, CSVs, web)")
    sources_used: list[SourceItem] = Field(default_factory=list, description="Alias for backward compatibility")
    source_status: Optional[str] = "verified"  # "internal_verified", "web_verified", "unverified_fallback", "llm_knowledge"
    verified: Optional[bool] = True
    slot_updated: Optional[str] = None
    retrieval_trace: Optional[RetrievalTrace] = None


class AnalyzeRequest(BaseModel):
    """Direct structured JSON input — bypasses conversational slot-filling."""
    soil_organic_carbon_pct: Optional[float] = Field(None, description="Soil organic carbon percentage (e.g. 0.3)")
    soil_ph: Optional[float] = Field(None, description="Soil pH value (e.g. 6.5)")
    soil_moisture: Optional[str] = Field(None, description="Soil moisture condition: low, moderate, high")
    land_use_type: Optional[str] = Field(None, description="Land use type: monoculture_wheat, pastoral, mixed_farming, etc.")
    rainfall_pattern: Optional[str] = Field(None, description="Rainfall pattern: low, moderate, high")
    region_climate_zone: Optional[str] = Field(None, description="Climate zone: arid, semi_arid, tropical, temperate, etc.")
    species_richness_observation: Optional[str] = Field(None, description="Qualitative observation of species richness: low, moderate, high")
    pollution_or_deforestation_pressure: Optional[str] = Field(None, description="Pollution/deforestation pressure: none, low, moderate, high")
    geo_coordinates: Optional[dict] = Field(None, description='Geo coordinates: {"lat": 26.9, "lon": 75.8}')


class AnalyzeResponse(BaseModel):
    recommendation: Recommendation
    alternative_recommendation: Optional[Recommendation] = None
    why_alternative: Optional[str] = None
    matched_variables: list[str] = Field(default_factory=list)
    overlap_count: int = Field(default=0)
    slots_used: SlotState
    sources: list[SourceItem] = Field(default_factory=list)
    sources_used: list[SourceItem] = Field(default_factory=list)
    reasoning_trace: Optional[str] = None
    retrieval_trace: Optional[RetrievalTrace] = None
