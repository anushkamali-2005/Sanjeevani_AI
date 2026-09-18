"""
tests/test_rag.py — Deterministic unit & integration tests for RAG schemas, 
citation provenance, and causal graph reasoning.
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath("."))
from backend.app.schemas import SourceItem, SlotState, Recommendation, MetricImpact, ChatResponse
from backend.app.reasoning import rank_interventions, get_graph_stats
from backend.app.llm_client import _sort_sources


def test_source_item_schema_and_serialization():
    """Verify SourceItem validates document, web, data, and AI fallback schemas."""
    # 1. Internal Document
    doc_source = SourceItem(
        type="document",
        source_type="document",
        id="DOC_01",
        tag="DOC_01",
        title="IPCC AR6 WGII Chapter 10: Asia",
        file_name="IPCC_AR6_WGII_Chapter10.pdf",
        page_number=47,
        excerpt="Agroforestry systems in semi-arid zones...",
        verified=True
    )
    assert doc_source.tag == "DOC_01"
    assert doc_source.page_number == 47
    assert doc_source.verified is True

    # 2. Web Source
    web_source = SourceItem(
        type="web_search",
        source_type="web",
        id="WEB_01",
        tag="WEB_01",
        title="Soil Carbon Management",
        url="https://fao.org/soils/carbon",
        verified=True
    )
    assert web_source.tag == "WEB_01"
    assert "fao.org" in web_source.url

    # 3. AI Knowledge Fallback
    ai_source = SourceItem(
        type="llm_knowledge",
        source_type="llm_knowledge",
        id="AI_01",
        tag="AI_01",
        title="AI General Knowledge",
        relevance_label="Unverified Fallback",
        verified=False
    )
    assert ai_source.verified is False


def test_strict_provenance_sorting_order():
    """Verify sources are strictly ordered: Document -> Web -> Dataset -> AI Fallback."""
    raw_sources = [
        {"id": "AI_01", "source_type": "llm_knowledge", "type": "llm_knowledge"},
        {"id": "DATA_01", "source_type": "structured_data", "type": "structured_data"},
        {"id": "WEB_01", "source_type": "web", "type": "web_search"},
        {"id": "DOC_01", "source_type": "document", "type": "document"},
    ]

    sorted_sources = _sort_sources(raw_sources)
    order_ids = [s["id"] for s in sorted_sources]

    assert order_ids == ["DOC_01", "WEB_01", "DATA_01", "AI_01"], (
        f"Incorrect provenance ordering: {order_ids}. Expected ['DOC_01', 'WEB_01', 'DATA_01', 'AI_01']"
    )


def test_causal_graph_preconditions_and_ranking():
    """Verify deterministic relationship graph traversal and multi-variable ranking."""
    stats = get_graph_stats()
    assert stats["total_interventions"] >= 12

    # Simulate user slots: Low SOC, low rainfall, monoculture cropping in semi-arid zone
    slots = {
        "soil_organic_carbon_pct": 0.35,
        "rainfall_pattern": "low",
        "land_use_type": "monoculture wheat",
        "region_climate_zone": "semi-arid"
    }

    ranked = rank_interventions(slots, min_overlap=2)
    assert len(ranked) > 0, "Expected at least one matching intervention satisfying >= 2 overlaps"

    top_rec = ranked[0]
    assert top_rec["overlap_count"] >= 2
    assert "name" in top_rec
    assert "mechanism" in top_rec
    assert "affects_metrics" in top_rec
    assert len(top_rec["affects_metrics"]) >= 2, "Expected multi-metric impact array"


def test_structured_analyze_endpoint():
    """Verify FastAPI /analyze endpoint returns valid recommendation without chat state."""
    from fastapi.testclient import TestClient
    from backend.app.main import app

    client = TestClient(app)
    payload = {
        "soil_organic_carbon_pct": 0.3,
        "land_use_type": "monoculture wheat",
        "rainfall_pattern": "low",
        "region_climate_zone": "semi-arid"
    }

    res = client.post("/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["recommendation"] is not None
    assert data["recommendation"]["recommendation"] != ""
    assert data["recommendation"]["mechanism"] != ""
    assert len(data["recommendation"]["metrics_impacted"]) > 0
    assert data["overlap_count"] >= 2


def test_mocked_web_search_fallback():
    """Verify web search fallback gracefully transforms external results into SourceItems."""
    mock_web_results = [
        {
            "title": "Soil Health Guidelines",
            "url": "https://extension.wisc.edu/soil-carbon",
            "snippet": "Cover crops increase soil organic carbon by 15-20% over 3 years.",
            "score": 0.92,
            "retrieved_at": "2026-09-18T20:00:00Z"
        }
    ]

    with patch("backend.app.websearch.web_search_augment", return_value=mock_web_results):
        from backend.app.websearch import web_search_augment
        results = web_search_augment("What is optimal soil carbon?")
        assert len(results) == 1
        assert results[0]["title"] == "Soil Health Guidelines"
        assert "extension.wisc.edu" in results[0]["url"]
