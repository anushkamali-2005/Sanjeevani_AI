"""
Conversation state machine for Darukaa.Earth Biodiversity Intelligence Chatbot.
Manages per-session slot filling, contradiction/correction handling,
provisional recommendations, and multi-metric reasoning handoff.
"""

import uuid
from typing import Optional
import logging
logger = logging.getLogger("darukaa.conversation")

from backend.app.schemas import (
    SlotState, ChatResponse, SourceItem, SourceChunk, Recommendation, MetricImpact, Confidence, RetrievalTrace
)
from backend.app.llm_client import (
    extract_slots, synthesize_recommendation, generate_clarifying_question,
    synthesize_conversational_answer
)
from backend.app.reasoning import rank_interventions, categorize_variables
from backend.app.retrieval import retrieve, hybrid_retrieve, build_retrieval_query
from backend.app.geo_lookup import lookup_climate
from backend.app.csv_knowledge import get_structured_grounding_context
from backend.app.websearch import should_trigger_web_search, web_search_augment


# In-memory session store
_sessions: dict[str, dict] = {}

# Mandatory slots for high-confidence output
MANDATORY_SLOTS = ["soil_organic_carbon_pct", "land_use_type", "rainfall_pattern", "region_climate_zone"]

# Priority order for clarifying questions
SLOT_PRIORITY = ["land_use_type", "rainfall_pattern", "region_climate_zone", "soil_organic_carbon_pct", "soil_ph", "soil_moisture"]

# All trackable slot names
ALL_SLOTS = [
    "soil_organic_carbon_pct", "soil_ph", "soil_moisture",
    "land_use_type", "rainfall_pattern", "region_climate_zone",
    "species_richness_observation", "pollution_or_deforestation_pressure",
    "geo_lat", "geo_lon", "state", "country"
]


def _get_or_create_session(session_id: str | None) -> tuple[str, dict]:
    """Get an existing session or create a new one."""
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]

    new_id = session_id or str(uuid.uuid4())[:8]
    _sessions[new_id] = {
        "slots": {},
        "history": [],
        "last_recommendation": None,
    }
    return new_id, _sessions[new_id]


def _slots_to_state(slots: dict) -> SlotState:
    """Convert raw slots dict to SlotState pydantic model."""
    return SlotState(
        soil_organic_carbon_pct=slots.get("soil_organic_carbon_pct"),
        soil_ph=slots.get("soil_ph"),
        soil_moisture=slots.get("soil_moisture"),
        land_use_type=slots.get("land_use_type"),
        rainfall_pattern=slots.get("rainfall_pattern"),
        region_climate_zone=slots.get("region_climate_zone"),
        species_richness_observation=slots.get("species_richness_observation"),
        pollution_or_deforestation_pressure=slots.get("pollution_or_deforestation_pressure"),
        geo_lat=slots.get("geo_lat"),
        geo_lon=slots.get("geo_lon"),
        state=slots.get("state"),
        country=slots.get("country", "India"),
    )


def _count_known_variables(slots: dict) -> int:
    """Count how many environmental variables are known (non-None)."""
    env_vars = [
        "soil_organic_carbon_pct", "soil_ph", "soil_moisture",
        "land_use_type", "rainfall_pattern", "region_climate_zone",
        "species_richness_observation", "pollution_or_deforestation_pressure"
    ]
    count = sum(1 for v in env_vars if slots.get(v) is not None)
    if slots.get("state") is not None and slots.get("region_climate_zone") is None:
        count += 1
    return count


def _get_missing_mandatory(slots: dict) -> list[str]:
    """Return mandatory slots that are still empty."""
    return [s for s in MANDATORY_SLOTS if slots.get(s) is None]


def _get_next_question_slot(slots: dict) -> str | None:
    """Return the highest-priority missing slot to ask about."""
    for slot in SLOT_PRIORITY:
        if slots.get(slot) is None:
            return slot
    return None


def _build_topic_filter(slots: dict) -> list[str]:
    """Build topic tag filters based on known variables."""
    filters = []
    land_use = slots.get("land_use_type", "")
    if land_use:
        if "monoculture" in str(land_use).lower():
            filters.extend(["monoculture", "biodiversity"])
        if "pastoral" in str(land_use).lower() or "grazing" in str(land_use).lower():
            filters.extend(["grazing", "biodiversity"])
        if "forest" in str(land_use).lower():
            filters.extend(["reforestation", "biodiversity"])

    if slots.get("soil_organic_carbon_pct") is not None:
        filters.extend(["soil", "carbon"])

    rainfall = slots.get("rainfall_pattern", "")
    if rainfall and "low" in str(rainfall).lower():
        filters.extend(["semi_arid", "water_harvesting"])

    if not filters:
        filters = ["biodiversity", "soil"]

    return list(set(filters))


def _is_greeting_or_help(message: str) -> bool:
    """Detect if the message is a generic greeting or help query without ecological data."""
    import re
    msg = message.strip().lower()
    cleaned = re.sub(r"[^\w\s]", "", msg).strip()
    greetings = [
        "hello", "hi", "hey", "hlo", "hllo", "helo", "help", "start",
        "what is this", "what can you do", "who are you",
        "good morning", "good afternoon", "good evening", "namaste", "howdy", "sup"
    ]
    return any(cleaned == g or cleaned.startswith(g + " ") or cleaned.endswith(" " + g) for g in greetings)


def _is_requesting_early_recommendation(message: str) -> bool:
    """Detect if the user is explicitly requesting a recommendation, intervention, or environmental assessment."""
    msg = message.lower()
    triggers = [
        "recommend", "what should i do", "give me advice", "suggest", "what can i plant",
        "what intervention", "action plan", "solution", "soil carbon", "carbon stock",
        "how to improve", "tell me about", "how can i", "best practice", "advice"
    ]
    return any(t in msg for t in triggers)


def _build_rec_models(synthesis_dict: dict) -> tuple[Optional[Recommendation], Optional[Recommendation], Optional[str]]:
    """Helper to convert synthesized JSON into Pydantic Recommendation objects."""
    if not synthesis_dict:
        return None, None, None

    primary_data = synthesis_dict.get("primary_recommendation")
    if not primary_data:
        # Check if root is already the single recommendation
        if "recommendation" in synthesis_dict:
            primary_data = synthesis_dict
        else:
            return None, None, None

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

    primary_rec = _dict_to_rec(primary_data)

    alt_data = synthesis_dict.get("alternative_recommendation")
    alt_rec = _dict_to_rec(alt_data) if alt_data else None
    why_alt = synthesis_dict.get("why_alternative")

    return primary_rec, alt_rec, why_alt


def process_message(session_id: str | None, message: str) -> ChatResponse:
    """
    Process a user chat message through the conversation state machine.

    Handles:
    - Multi-slot extraction from single turn
    - Contradiction & value correction tracking
    - Helpful routing for greetings and ambiguous prompts
    - Provisional low-confidence recommendations for early explicit requests (1-2 variables)
    - Full multi-metric graph reasoning & vector retrieval once >=3 variables are known
    """
    sid, session = _get_or_create_session(session_id)
    session["history"].append({"role": "user", "content": message})
    user_turn_count = sum(1 for turn in session["history"] if turn.get("role") == "user")

    # Step 1: Extract slot values from message
    extracted = extract_slots(message, current_slots=session["slots"])

    # Step 2: Track slot changes & contradictions
    updated_slots_log = []
    for key, value in extracted.items():
        if key in ALL_SLOTS and value is not None:
            old_value = session["slots"].get(key)
            if old_value is not None and str(old_value).lower() != str(value).lower():
                updated_slots_log.append((key, old_value, value))
            session["slots"][key] = value

    # Step 3: Check for real extracted slots
    has_real_slots = any(v is not None for k, v in extracted.items() if k in ALL_SLOTS)

    # Step 4: Auto-fill from geo coordinates if supplied
    if session["slots"].get("geo_lat") and session["slots"].get("geo_lon"):
        geo_info = lookup_climate(session["slots"]["geo_lat"], session["slots"]["geo_lon"])
        if session["slots"].get("region_climate_zone") is None:
            session["slots"]["region_climate_zone"] = geo_info.get("region_climate_zone")
        if session["slots"].get("rainfall_pattern") is None:
            session["slots"]["rainfall_pattern"] = geo_info.get("rainfall_pattern")

    known_count = _count_known_variables(session["slots"])
    missing_mandatory = _get_missing_mandatory(session["slots"])
    slots_state = _slots_to_state(session["slots"])

    # Format slot update string if any
    slot_update_desc = None
    if updated_slots_log:
        slot_update_desc = ", ".join([f"{k.replace('_', ' ')} from '{old}' to '{new}'" for k, old, new in updated_slots_log])

    # Determine if message is an informational query / question rather than pure parameter input
    msg_lower = message.strip().lower()
    is_question = (
        "?" in message or
        msg_lower.startswith(("what", "why", "how", "can", "is", "are", "tell me", "explain", "does", "which", "when", "who")) or
        any(q in msg_lower for q in ["what is", "how does", "what are", "why is", "tell me about", "can you", "difference between", "latest target", "percentage for"])
    )

    # Case A: Informational Question or No environmental slots known (General Knowledge Q&A or Greetings)
    if is_question or (known_count == 0 and not has_real_slots):
        if _is_greeting_or_help(message) and user_turn_count <= 1:
            response_text = (
                "Welcome to **Darukaa.Earth**, your evidence-based Biodiversity Intelligence Advisory Portal.\n\n"
                "To generate scientifically grounded recommendations, I evaluate 9 key environmental variables:\n"
                "• **Land use / crop type** (e.g. monoculture wheat, mixed farming, pastoral)\n"
                "• **Rainfall pattern** (e.g. low/rainfed, seasonal, high monsoon)\n"
                "• **Climate zone** (e.g. semi-arid, arid, tropical, temperate)\n"
                "• **Soil conditions** (organic carbon %, pH, moisture)\n\n"
                "To get started, simply describe your land conditions or ask any biodiversity/agricultural question!"
            )
            session["history"].append({"role": "assistant", "content": response_text})
            return ChatResponse(
                session_id=sid,
                response_text=response_text,
                answer=response_text,
                recommendation=None,
                slots=slots_state,
                missing_slots=missing_mandatory,
                sources=[],
                sources_used=[],
                slot_updated=slot_update_desc,
                retrieval_trace=None
            )
        elif _is_greeting_or_help(message):
            response_text = (
                "Hello again! I am ready to assist you.\n\n"
                "You can ask me questions about soil health, biodiversity, and agroecological management, "
                "or describe your site conditions (crop type, soil carbon, rainfall) for personalized intervention recommendations."
            )
            session["history"].append({"role": "assistant", "content": response_text})
            return ChatResponse(
                session_id=sid,
                response_text=response_text,
                answer=response_text,
                recommendation=None,
                slots=slots_state,
                missing_slots=missing_mandatory,
                sources=[],
                sources_used=[],
                slot_updated=slot_update_desc,
                retrieval_trace=None
            )
        else:
            # ===============================================================
            # CONVERSATIONAL GROUNDED RAG PIPELINE
            # Architecture: Internal ChromaDB -> Threshold Check -> Web Search Fallback -> Grounded LLM
            # ===============================================================
            retrieval_query = build_retrieval_query(message, session["slots"])

            # 1. Search Internal ChromaDB + BM25 with strict relevance thresholding
            evidence_chunks, retrieval_trace_dict = hybrid_retrieve(
                retrieval_query, topic_filter=None, k=6, slots=session["slots"]
            )
            retrieval_trace = RetrievalTrace(**retrieval_trace_dict)
            is_internal_sufficient = retrieval_trace_dict.get("is_sufficient", False)

            # 2. Check if live web search should be triggered (fallback or time-sensitive)
            retrieval_scores = [c.get("score", 0.0) for c in evidence_chunks if c.get("score") is not None]
            max_score = max(retrieval_scores) if retrieval_scores else 0.0
            trigger_web = should_trigger_web_search(
                message,
                internal_is_sufficient=is_internal_sufficient,
                max_retrieval_score=max_score
            )

            web_results = []
            if trigger_web:
                web_results = web_search_augment(message, max_results=3)

            # 3. Structured CSV Grounding Context
            grounding_text, structured_sources = get_structured_grounding_context(session["slots"], message)

            # 4. Detailed Server-Side Debugging Log
            print(f"\n[RETRIEVAL]\nquery = {message}", flush=True)
            print(f"[INTERNAL]\nresults = {len(evidence_chunks)}\nscores = {retrieval_scores}\naccepted_results = {len(evidence_chunks)}", flush=True)
            print(f"[WEB FALLBACK]\ntriggered = {trigger_web}", flush=True)
            if trigger_web:
                print(f"[WEB SEARCH]\nquery = {message}\nresults_count = {len(web_results)}", flush=True)
                print(f"[WEB SOURCES]", flush=True)
                for i, ws in enumerate(web_results):
                    print(f"source {i+1} = {ws.get('title')} + {ws.get('url') or ws.get('source_url')}", flush=True)
            print(f"[LLM CONTEXT]\nnumber of internal sources = {len(evidence_chunks)}\nnumber of web sources = {len(web_results)}", flush=True)

            # 5. Synthesize Grounded Conversational Answer with Citations ([DOC_01], [WEB_01])
            qa_result = synthesize_conversational_answer(
                user_query=message,
                internal_chunks=evidence_chunks,
                web_results=web_results,
                structured_grounding=structured_sources
            )

            response_text = qa_result["answer"]
            source_status = qa_result.get("source_status", "verified")
            is_verified = qa_result.get("verified", True)

            print(f"[FINAL]\nanswer generated = {bool(response_text)}\nsource_status = {source_status}\nverified = {is_verified}\n", flush=True)

            # Convert to SourceItem models with complete provenance metadata
            source_items = []
            for s in qa_result.get("sources", []):
                source_items.append(SourceItem(
                    type=s.get("type", "document"),
                    source_type=s.get("source_type", "document"),
                    id=s.get("id") or s.get("tag"),
                    tag=s.get("tag") or s.get("id"),
                    title=s.get("title"),
                    page=s.get("page"),
                    page_number=s.get("page_number"),
                    file=s.get("file"),
                    file_name=s.get("file_name"),
                    chunk_id=s.get("chunk_id"),
                    excerpt=s.get("excerpt"),
                    detail=s.get("detail"),
                    url=s.get("url"),
                    source_url=s.get("source_url"),
                    retrieved_at=s.get("retrieved_at"),
                    relevance_label=s.get("relevance_label"),
                    is_cited=s.get("is_cited", False),
                    verified=s.get("verified", True),
                    source=s.get("title") or s.get("file") or s.get("source"),
                    score=s.get("score")
                ))

            # Guarantee strict provenance priority: Documents first, Web second, Datasets third, AI fallback last
            source_order = {"document": 0, "web": 1, "web_search": 1, "structured_data": 2, "llm_knowledge": 3}
            source_items.sort(key=lambda s: source_order.get(s.source_type or s.type, 99))

            session["history"].append({"role": "assistant", "content": response_text})

            return ChatResponse(
                session_id=sid,
                response_text=response_text,
                answer=response_text,
                recommendation=None,
                slots=slots_state,
                missing_slots=missing_mandatory,
                sources=source_items,
                sources_used=source_items,
                source_status=source_status,
                verified=is_verified,
                slot_updated=slot_update_desc,
                retrieval_trace=retrieval_trace,
            )

    # Case B: Early recommendation request (<3 variables known) OR insufficient variables (<3)
    is_early_req = _is_requesting_early_recommendation(message)

    if known_count < 3 and not is_early_req:
        # Standard clarifying question loop
        next_slot = _get_next_question_slot(session["slots"])
        question = generate_clarifying_question(next_slot) if next_slot else "Could you tell me more about your environmental conditions?"

        ack_parts = []
        if updated_slots_log:
            ack_parts.append(f"Got it! Updating {slot_update_desc}.\n")
        elif extracted:
            ack_parts.append("Thank you! I've recorded the following environmental parameters:")
            for k, v in extracted.items():
                if k in ALL_SLOTS and v is not None:
                    ack_parts.append(f"  • **{k.replace('_', ' ').title()}**: {v}")
            ack_parts.append(f"\n({known_count}/4 mandatory variables identified so far)")

        ack_parts.append(f"\n{question}")
        response_text = "\n".join(ack_parts)
        session["history"].append({"role": "assistant", "content": response_text})

        return ChatResponse(
            session_id=sid,
            response_text=response_text,
            answer=response_text,
            recommendation=None,
            slots=slots_state,
            missing_slots=missing_mandatory,
            sources_used=[],
            slot_updated=slot_update_desc,
        )

    # Case C: Provisional recommendation (1-2 variables + explicit request) OR Full recommendation (>=3 variables)
    min_overlap = 2 if known_count >= 3 else 1
    ranked = rank_interventions(session["slots"], min_overlap=min_overlap)

    if not ranked:
        ranked = rank_interventions(session["slots"], min_overlap=1)

    if not ranked:
        next_slot = _get_next_question_slot(session["slots"])
        question = generate_clarifying_question(next_slot) if next_slot else "Could you provide your land use type or soil parameters?"
        response_text = (
            "I've recorded your inputs, but couldn't find a sufficiently verified intervention in the relationship graph for this exact combination.\n\n"
            f"{question}"
        )
        session["history"].append({"role": "assistant", "content": response_text})
        return ChatResponse(
            session_id=sid,
            response_text=response_text,
            answer=response_text,
            recommendation=None,
            slots=slots_state,
            missing_slots=missing_mandatory,
            sources_used=[],
            slot_updated=slot_update_desc,
            retrieval_trace=None
        )

    # Retrieve supporting evidence — use hybrid pipeline with slot-enriched query
    top_intervention = ranked[0]
    retrieval_query = build_retrieval_query(message, session["slots"])
    # Append top intervention name for better targeting
    retrieval_query += f" intervention: {top_intervention['name']}"
    topic_filter = _build_topic_filter(session["slots"])
    evidence_chunks, retrieval_trace_dict = hybrid_retrieve(
        retrieval_query, topic_filter=topic_filter, k=6, slots=session["slots"]
    )
    retrieval_trace = RetrievalTrace(**retrieval_trace_dict)

    # Grounding context from CSV datasets
    grounding_text, structured_sources = get_structured_grounding_context(session["slots"], message)

    # Optional web search augmentation if scores are low or time-sensitive
    retrieval_scores = [c.get("score", 0.0) for c in evidence_chunks if c.get("score") is not None]
    max_score = max(retrieval_scores) if retrieval_scores else 0.0
    web_grounding = []
    if should_trigger_web_search(message, max_score):
        web_grounding = web_search_augment(message, max_results=2)

    # Synthesize recommendation
    synthesis_result = synthesize_recommendation(
        user_variables=session["slots"],
        top_interventions=ranked[:2],
        evidence_chunks=evidence_chunks,
        structured_grounding=structured_sources,
        web_grounding=web_grounding
    )

    primary_rec, alt_rec, why_alt = _build_rec_models(synthesis_result)

    # Adjust confidence if provisional
    if known_count < 3 and primary_rec:
        primary_rec.confidence = Confidence.low

    # Assemble structured sources array
    structured_sources_list = []
    raw_sources = synthesis_result.get("sources", [])
    if raw_sources:
        for s in raw_sources:
            if isinstance(s, dict):
                structured_sources_list.append(SourceItem(
                    type=s.get("type", "document"),
                    title=s.get("title") or s.get("source"),
                    page=s.get("page"),
                    excerpt=s.get("excerpt"),
                    file=s.get("file"),
                    detail=s.get("detail"),
                    url=s.get("url"),
                    retrieved_at=s.get("retrieved_at"),
                    id=s.get("id"),
                    source=s.get("source") or s.get("title") or s.get("file"),
                    score=s.get("score")
                ))

    # Always ensure structured CSV sources are present in the response
    existing_files = {s.file for s in structured_sources_list if s.type == "structured_data" and s.file}
    for s in (structured_sources or []):
        if s.get("file") not in existing_files:
            structured_sources_list.append(SourceItem(
                type="structured_data",
                file=s.get("file"),
                detail=s.get("detail"),
                title=s.get("file"),
                source=f"Dataset: {s.get('file')}"
            ))

    if not structured_sources_list:
        for chunk in evidence_chunks[:5]:
            structured_sources_list.append(SourceItem(
                type="document",
                id=chunk.get("id", ""),
                title=chunk.get("document_title") or chunk.get("source", ""),
                page=chunk.get("page_number"),
                excerpt=chunk.get("text", "")[:220] + "..." if len(chunk.get("text", "")) > 220 else chunk.get("text", ""),
                file=chunk.get("source_file"),
                source=chunk.get("source", ""),
                score=chunk.get("score")
            ))
        for s in structured_sources:
            structured_sources_list.append(SourceItem(
                type="structured_data",
                file=s.get("file"),
                detail=s.get("detail"),
                title=s.get("file"),
                source=f"Dataset: {s.get('file')}"
            ))

    # Formulate contextual response text
    response_parts = []
    if updated_slots_log:
        response_parts.append(f"Got it, updating {slot_update_desc}. I've regenerated your recommendation based on this new information.\n")

    if known_count < 3:
        missing_names = ", ".join([s.replace('_', ' ') for s in missing_mandatory[:2]])
        response_parts.append(
            f"**Provisional Recommendation (Low Confidence — {known_count}/4 mandatory variables known):**\n"
            f"Based on your inputs, here is an initial indication. To sharpen this and provide high confidence, please also share: **{missing_names}**."
        )
    else:
        response_parts.append(
            f"Based on your full environmental context ({known_count} variables analysed across the causal graph), here is your evidence-based recommendation:"
        )

    response_text = "\n".join(response_parts)
    session["history"].append({"role": "assistant", "content": response_text})
    session["last_recommendation"] = synthesis_result

    return ChatResponse(
        session_id=sid,
        response_text=response_text,
        recommendation=primary_rec,
        alternative_recommendation=alt_rec,
        why_alternative=why_alt,
        slots=slots_state,
        missing_slots=missing_mandatory,
        sources=structured_sources_list,
        sources_used=structured_sources_list,
        slot_updated=slot_update_desc,
        retrieval_trace=retrieval_trace,
    )


def process_structured_input(variables: dict) -> tuple[dict, list[dict], list[dict], dict]:
    """
    Process a structured JSON input (from /analyze endpoint).
    Bypasses conversational slot-filling entirely.
    Returns (rec_data, evidence_chunks, ranked_interventions, retrieval_trace_dict).
    """
    geo = variables.get("geo_coordinates")
    if geo and isinstance(geo, dict):
        lat = geo.get("lat")
        lon = geo.get("lon")
        if lat is not None and lon is not None:
            geo_info = lookup_climate(lat, lon)
            if variables.get("region_climate_zone") is None:
                variables["region_climate_zone"] = geo_info.get("region_climate_zone")
            if variables.get("rainfall_pattern") is None:
                variables["rainfall_pattern"] = geo_info.get("rainfall_pattern")

    ranked = rank_interventions(variables, min_overlap=2)
    if not ranked:
        ranked = rank_interventions(variables, min_overlap=1)

    if not ranked:
        return None, [], [], {}

    top_intervention = ranked[0]
    retrieval_query = build_retrieval_query(
        f"intervention: {top_intervention['name']}", variables
    )
    evidence_chunks, retrieval_trace_dict = hybrid_retrieve(
        retrieval_query, k=6, slots=variables
    )

    grounding_text, structured_sources = get_structured_grounding_context(variables)
    rec_data = synthesize_recommendation(
        user_variables=variables,
        top_interventions=ranked[:2],
        evidence_chunks=evidence_chunks,
        structured_grounding=structured_sources
    )

    return rec_data, evidence_chunks, ranked, retrieval_trace_dict
